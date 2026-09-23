"""Application-owned LRU pool of expert records (research, off by default).

Every cached expert row the production path reads is copied out of the file
cache, about 880 MB/token and about 37 ms/token in the 2026-09-23 read replay.
This pool keeps whole expert records in one anonymous memory region laid out
like the slab pack (a 4 KiB header, then fixed-stride records), wrapped once
as a Metal buffer. A decode layer addresses every routed expert in place
through the packed-slab kernels (bit 31 plus the slot), so a hit costs no read
and no copy. A miss reads its nine rows into a free or least recently used
slot through ``F_NOCACHE`` descriptors, so the file cache does not hold a
second copy.

A fixed pool cannot follow free memory the way the file cache does. In the
replay a 5 GB pool tied production and a 6 GB pool read 15 to 24 ms/token
faster; 3 and 4 GB lost. ``FLASHNEXT_EXPERT_POOL_GB`` sets the size (0, the
default, disables it). ``FLASHNEXT_EXPERT_POOL_LOCK=1`` mlocks the region so
macOS cannot compress it.

A slot is only overwritten after its last reader finished: layer L fills its
misses after layer L's router sync, which depends on layer L-1's MoE output,
and every slot this layer routes to is moved to the most recent end before a
victim is chosen.
"""
from __future__ import annotations

import fcntl
import mmap
import os
from collections import OrderedDict

import numpy as np

from .slab_pack import get_slab_layout, libc_mlock

SIZE_FLAG = "FLASHNEXT_EXPERT_POOL_GB"
LOCK_FLAG = "FLASHNEXT_EXPERT_POOL_LOCK"
HEADER = 4096
SLOT_BIT = 0x80000000
_PARTS = (
    ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
    ("up_proj", "weight"), ("up_proj", "scales"), ("up_proj", "biases"),
    ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
)


def configured_gb() -> float:
    try:
        return max(0.0, float(os.environ.get(SIZE_FLAG, "0") or 0))
    except ValueError as error:
        raise ValueError(f"{SIZE_FLAG} must be a number of GB") from error


def enabled() -> bool:
    return configured_gb() > 0


class ExpertPool:
    def __init__(self, store, size_bytes: int, group_size: int = 32,
                 lock: bool = False):
        layout = get_slab_layout(group_size)
        self.store = store
        self.stride = layout.record_stride
        self.offsets = {key: layout.offset(*key) for key in _PARTS}
        self.slots = (int(size_bytes) - HEADER) // self.stride
        if self.slots < 16:
            raise ValueError("expert pool is too small for one decode layer")
        length = -(-(HEADER + self.slots * self.stride) // mmap.PAGESIZE) * mmap.PAGESIZE
        self._mm = mmap.mmap(-1, length)
        self.array = np.frombuffer(self._mm, dtype=np.uint8)
        self.locked = bool(lock) and libc_mlock(
            self.array.__array_interface__["data"][0], length
        )
        import mlx.core as mx

        # MLX shape dimensions are 32-bit, so a byte view of more than 2 GiB
        # cannot be wrapped. The kernels address the pool through a char
        # pointer, so a uint32 view of the same memory serves (up to 8 GiB).
        self.buffer_mx = mx.from_dlpack(
            np.frombuffer(self._mm, dtype=np.uint32), copy=False
        )
        self._view = memoryview(self._mm)
        self._slot_of: "OrderedDict[tuple, int]" = OrderedDict()
        self._free = list(range(self.slots - 1, -1, -1))
        self._fds: dict = {}
        self.hits = 0
        self.misses = 0

    def assign(self, layer: int, experts):
        """Return ({expert: slot}, [(expert, slot) to fill]) for one layer."""
        slot_of = self._slot_of
        slots, misses = {}, []
        for expert in experts:
            key = (layer, expert)
            slot = slot_of.get(key)
            if slot is not None:
                slot_of.move_to_end(key)
                slots[expert] = slot
        self.hits += len(slots)
        for expert in experts:
            if expert in slots:
                continue
            if self._free:
                slot = self._free.pop()
            else:
                _old, slot = slot_of.popitem(last=False)
            slot_of[(layer, expert)] = slot
            slots[expert] = slot
            misses.append((expert, slot))
        self.misses += len(misses)
        return slots, misses

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(os.path.join(self.store.dir, shard), os.O_RDONLY)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            try:
                fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
            except OSError:
                pass
            self._fds[shard] = fd
        return fd

    def _fill_part(self, name, targets) -> None:
        ref = self.store.refs[name]
        fd = self._fd(ref.shard)
        view = self._view
        size = ref.row_bytes
        for expert, base in targets:
            if os.preadv(fd, [view[base:base + size]], ref.start + expert * size) != size:
                raise OSError(f"short expert pool read for {name} row {expert}")

    def fill(self, prefix: str, misses, submit, chunk: int = 2):
        """Queue the nine row reads of every miss; returns the futures."""
        futures = []
        for projection, part in _PARTS:
            name = f"{prefix}.{projection}.{part}"
            offset = HEADER + self.offsets[(projection, part)]
            for start in range(0, len(misses), chunk):
                targets = [
                    (expert, offset + slot * self.stride)
                    for expert, slot in misses[start:start + chunk]
                ]
                futures.append(submit(self._fill_part, name, targets))
        return futures

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def for_store(store, group_size: int = 32):
    """The process's pool for this store, created on first use."""
    pool = getattr(store, "_flashnext_expert_pool", None)
    if pool is None and enabled():
        pool = ExpertPool(
            store, int(configured_gb() * 1e9), group_size,
            lock=os.environ.get(LOCK_FLAG, "0") == "1",
        )
        store._flashnext_expert_pool = pool
    return pool
