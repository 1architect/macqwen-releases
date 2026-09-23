"""Random-access reader for safetensors shards.

The checkpoint is larger than memory, so nothing is loaded whole. Each shard
file is memory-mapped once. Callers ask for the rows they need and get back a
small mx.array. Pages stay clean and file-backed, so macOS can drop them
without writing to swap.
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import queue
import sys
import json
import fcntl
import mmap
import os
import struct
import threading
import time
from typing import Dict, Sequence, Tuple

import mlx.core as mx
import numpy as np

# safetensors dtype -> (numpy read dtype, mlx target dtype)
# BF16 has no numpy equivalent, so it travels as uint16 and is reinterpreted.
_DTYPES: Dict[str, Tuple[np.dtype, object]] = {
    "F32": (np.dtype("<f4"), mx.float32),
    "F16": (np.dtype("<f2"), mx.float16),
    "BF16": (np.dtype("<u2"), mx.bfloat16),
    "U32": (np.dtype("<u4"), mx.uint32),
    "I32": (np.dtype("<i4"), mx.int32),
    "I64": (np.dtype("<i8"), mx.int64),
    "U64": (np.dtype("<u8"), mx.uint64),
    "F64": (np.dtype("<f8"), mx.float32),
    "BOOL": (np.dtype("?"), mx.bool_),
    "U16": (np.dtype("<u2"), mx.uint16),
    "U8": (np.dtype("u1"), mx.uint8),
    "I8": (np.dtype("i1"), mx.int8),
}

_READ_PROFILE = threading.local()
_PHYSICAL_MISS_TRACE = os.environ.get("FLASHNEXT_PHYSICAL_MISS_TRACE") == "1"


def _trace_read(name: str, row: int, requested_bytes: int, reader, *args):
    """Run one read and attribute its process-wide physical delta."""
    from .diskio import disk_bytes_read

    before = disk_bytes_read()
    value = reader(*args)
    after = disk_bytes_read()
    from .physical_miss import record_trace_read

    delta = after - before if before >= 0 and after >= before else 0
    record_trace_read(name, row, delta, requested_bytes)
    return value


def begin_read_profile() -> None:
    """Collect positioned-read intervals for one worker task."""
    _READ_PROFILE.stats = {"pread_intervals": [], "pread_calls": 0, "pread_bytes": 0}


def finish_read_profile() -> dict:
    stats = getattr(_READ_PROFILE, "stats", None) or {
        "pread_intervals": [], "pread_calls": 0, "pread_bytes": 0,
    }
    _READ_PROFILE.stats = None
    return stats


def _profiled_pread(fd: int, size: int, offset: int) -> bytes:
    stats = getattr(_READ_PROFILE, "stats", None)
    if stats is None:
        return os.pread(fd, size, offset)
    began = time.perf_counter()
    data = os.pread(fd, size, offset)
    ended = time.perf_counter()
    stats["pread_intervals"].append((began, ended))
    stats["pread_calls"] += 1
    stats["pread_bytes"] += len(data)
    return data


def _profiled_preadv(fd: int, buffers, offset: int) -> int:
    stats = getattr(_READ_PROFILE, "stats", None)
    if stats is None:
        return os.preadv(fd, buffers, offset)
    began = time.perf_counter()
    count = os.preadv(fd, buffers, offset)
    ended = time.perf_counter()
    stats["pread_intervals"].append((began, ended))
    stats["pread_calls"] += 1
    stats["pread_bytes"] += count
    return count


class TensorRef:
    __slots__ = ("shard", "dtype", "shape", "start", "row_bytes")

    def __init__(self, shard: str, dtype: str, shape: Sequence[int], start: int):
        self.shard = shard
        self.dtype = dtype
        self.shape = tuple(shape)
        self.start = start
        trailing = int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1
        self.row_bytes = trailing * _DTYPES[dtype][0].itemsize


_ADVICE = {
    "normal": mmap.MADV_NORMAL,
    "random": mmap.MADV_RANDOM,
    "sequential": mmap.MADV_SEQUENTIAL,
}
_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
    return _LIBC


class SafeTensorStore:
    """Memory-mapped view over every shard of a safetensors checkpoint."""

    def __init__(self, model_dir: str):
        self.dir = os.path.expanduser(model_dir)
        self._flashnext_env = {
            key: os.environ[key]
            for key in (
                "FLASHNEXT_METAL_RUNTIME", "FLASHNEXT_METAL_G64",
                "FLASHNEXT_SLAB_G64", "FLASHNEXT_SLAB_GLOBAL",
                "FLASHNEXT_SLAB_PACK", "FLASHNEXT_SLAB_POLICY",
                "FLASHNEXT_FUSED_SHARED", "FLASHNEXT_FUSED_SHARED_PARTS",
                "FLASHNEXT_FUSED_UP_SWIGLU", "FLASHNEXT_STREAM_PACK",
                "FLASHNEXT_PREAD_CHUNK", "FLASHNEXT_IO_WORKERS",
                "FLASHNEXT_READ",
            )
            if key in os.environ
        }
        self.refs: Dict[str, TensorRef] = {}
        self._maps: Dict[str, mmap.mmap] = {}
        self._fds: Dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._map_lock = threading.Lock()
        self._views: Dict[str, np.ndarray] = {}
        self._shared_views: Dict[str, np.ndarray] = {}
        self._read_mode = os.environ.get("FLASHNEXT_READ", "pread")
        # Rows per positioned read. Chunk 1 gives every expert its own read and
        # its own allocation, which the main thread then concatenates. Chunk 2
        # halves the read count, lets one worker fill a contiguous run, and
        # pairs with the shared read buffer in `expert_cache` to remove the
        # concatenate entirely. The pair measured +6.3% on the production
        # harness against a 4.4% band. Chunk 8 alone was 8% worse, because
        # three reads per layer cannot keep the NVMe queue busy.
        self._pread_chunk = int(os.environ.get("FLASHNEXT_PREAD_CHUNK", "2"))
        # Kernel read-ahead on the shard descriptors. Off measured 1.3% faster,
        # flat across miss rates, inside the band; the default stays on.
        self._rdahead = os.environ.get("FLASHNEXT_RDAHEAD", "1") != "0"
        self._dlpack = os.environ.get("FLASHNEXT_DLPACK", "1") == "1"
        self._mmap_advice = os.environ.get("FLASHNEXT_MMAP_ADVICE", "random")
        self._pinned_rows: set = set()
        # Residency tracking. Mapping resident rows instead of reading them was
        # measured and rejected, but knowing which rows are resident is what
        # tells cache-aware routing whether a cheaper expert was available.
        # This tracks rows the process read in a bounded LRU at dictionary
        # cost; `bench_residency.py` measured it at 97.6% precise.
        self._track_residency = os.environ.get("FLASHNEXT_TRACK_RESIDENT") == "1"
        self._resident_cap = int(
            os.environ.get("FLASHNEXT_RESIDENT_ROWS", "12000")
        )
        self._resident_lru: "OrderedDict[tuple, None]" = OrderedDict()
        self._pinned = []
        # Locked expert working set. Other processes can shrink the page cache
        # the expert stream depends on; a 3 GB competing process measured
        # about 7% slower decode through 13% more physical reads. With a
        # budget, the most recently read expert rows stay mlocked in the
        # checkpoint's own file pages, so no row is held twice. Locking runs on
        # one background thread; readers only enqueue. Off at 0.
        self._lock_budget = int(
            float(os.environ.get("FLASHNEXT_EXPERT_LOCK_GB", "0")) * 1e9
        )
        self._locked: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._locked_bytes = 0
        self._locked_mutex = threading.Lock()
        self._lock_queue = None
        self._lock_thread = None
        self._lock_failed = False

        index_path = os.path.join(self.dir, "model.safetensors.index.json")
        with open(index_path) as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))

        for shard in shards:
            self.add_shard(shard)

    def add_shard(self, shard: str) -> None:
        """Register an extra safetensors shard beside the indexed checkpoint."""
        path = os.path.join(self.dir, shard)
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            self.refs[name] = TensorRef(
                shard,
                meta["dtype"],
                meta["shape"],
                data_start + meta["data_offsets"][0],
            )

    def _view(self, name: str) -> np.ndarray:
        """A numpy view of one tensor, mapped once and reused.

        Fancy-indexing this view gathers rows in C. Reading them one at a time
        through np.frombuffer costs a Python round trip per row, which measured
        250 MB/s against an SSD that does 3 GB/s.
        """
        view = self._views.get(name)
        if view is None:
            ref = self.refs[name]
            read_dtype, _ = _DTYPES[ref.dtype]
            view = np.memmap(
                os.path.join(self.dir, ref.shard),
                dtype=read_dtype,
                mode="r",
                offset=ref.start,
                shape=ref.shape,
            )
            # Without this the kernel reads ahead around every scattered row.
            # np.memmap opens its own mapping, so self._maps advice misses it.
            self._advise_view(view)
            self._views[name] = view
        return view

    def _advise_view(self, view: np.ndarray) -> None:
        try:
            view._mmap.madvise(_ADVICE[self._mmap_advice])
        except (AttributeError, OSError):
            pass

    def set_mmap_advice(self, value: str) -> None:
        self._mmap_advice = value
        for view in self._views.values():
            self._advise_view(view)
        for handle in self._maps.values():
            try:
                handle.madvise(_ADVICE[value])
            except (AttributeError, OSError):
                pass

    def _map(self, shard: str) -> mmap.mmap:
        handle = self._maps.get(shard)
        if handle is None:
            with self._map_lock:
                handle = self._maps.get(shard)
                if handle is None:
                    file = open(os.path.join(self.dir, shard), "rb")
                    try:
                        handle = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
                    finally:
                        # mmap keeps its own descriptor on this platform. The
                        # Python file object is not needed after mapping.
                        file.close()
                    handle.madvise(_ADVICE[self._mmap_advice])
                    self._maps[shard] = handle
        return handle

    def _shared_view(self, name: str) -> np.ndarray:
        view = self._shared_views.get(name)
        if view is None:
            ref = self.refs[name]
            read_dtype, _ = _DTYPES[ref.dtype]
            view = np.ndarray(
                ref.shape,
                dtype=read_dtype,
                buffer=self._map(ref.shard),
                offset=ref.start,
            )
            self._shared_views[name] = view
        return view

    def _mapped(self, name: str, mode: str) -> np.ndarray:
        """The mapped view a non-pread mode gathers from."""
        if mode == "shared_mmap":
            return self._shared_view(name)
        if mode == "mmap":
            return self._view(name)
        raise ValueError(f"unknown FlashNext read mode: {mode!r}")

    def shape(self, name: str) -> Tuple[int, ...]:
        return self.refs[name].shape

    def rows_np(
        self, name: str, indices: Sequence[int], read_mode: str | None = None
    ) -> np.ndarray:
        """Gather rows as numpy. Safe to call from worker threads: the reads
        and page faults release the GIL, so concurrent calls keep the NVMe
        queue busy instead of idling between serial reads."""
        rows = indices if isinstance(indices, list) else list(indices)
        mode = read_mode or self._read_mode
        if mode in ("pread", "preadv"):
            out = self.empty_rows(name, len(rows))
            self._pread_into(name, rows, out, mode)
            return out
        return np.ascontiguousarray(self._mapped(name, mode)[rows])

    def _row_span(self, name: str, row: int) -> tuple[int, int]:
        """Page-aligned address and length of one row in the shared map."""
        ref = self.refs[name]
        view = self._shared_view(name)
        address = int(view.__array_interface__["data"][0]) + int(row) * ref.row_bytes
        page = mmap.PAGESIZE
        start = address - address % page
        end = (address + ref.row_bytes + page - 1) // page * page
        return start, end - start

    def pin_rows(self, name: str, rows: Sequence[int]) -> int:
        """Keep selected file-backed rows resident without copying them."""
        libc = _libc()
        total = 0
        for row in rows:
            start, length = self._row_span(name, row)
            if libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(length)) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            self._pinned.append((start, length))
            self._pinned_rows.add((name, int(row)))
            total += length
        return total

    def pin_size(self, name: str, rows: Sequence[int]) -> int:
        """Return the bytes `pin_rows` would lock without changing residency."""
        ref = self.refs[name]
        # The mapping base is page-aligned, so its address does not affect
        # the page span. Avoid opening a shard while routing candidates.
        base = ref.start
        page = mmap.PAGESIZE
        total = 0
        for row in rows:
            address = base + int(row) * ref.row_bytes
            start = address - address % page
            end = (address + ref.row_bytes + page - 1) // page * page
            total += end - start
        return total

    def unpin_all(self) -> None:
        with self._locked_mutex:
            # A row pinned while locked loses its lock with the pin below;
            # forget it so its next read locks it again.
            for key in self._pinned_rows:
                span = self._locked.pop(key, None)
                if span is not None:
                    self._locked_bytes -= span[1]
            self._pinned_rows.clear()
        libc = _libc()
        for address, length in self._pinned:
            libc.munlock(ctypes.c_void_p(address), ctypes.c_size_t(length))
        self._pinned.clear()

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
                    if not self._rdahead:
                        # F_RDAHEAD is 45 on Darwin and older Pythons do not
                        # export it. Refuse rather than report a comparison
                        # whose setting silently did not apply.
                        command = getattr(fcntl, "F_RDAHEAD", 45)
                        if fcntl.fcntl(fd, command, 0) != 0:
                            raise OSError(
                                "F_RDAHEAD could not be cleared on "
                                f"{shard}; refusing to run the arm"
                            )
                    self._fds[shard] = fd
        return fd

    def _mark_read(self, name: str, rows) -> None:
        """Record rows just pulled in, newest last, evicting the oldest."""
        table = self._resident_lru
        for row in rows:
            key = (name, int(row))
            if key in table:
                table.move_to_end(key)
            else:
                table[key] = None
        while len(table) > self._resident_cap:
            table.popitem(last=False)

    def believed_resident(self, name: str, row: int) -> bool:
        """Whether this row is expected to be in memory without a read."""
        key = (name, int(row))
        if key in self._pinned_rows:
            return True
        if not self._track_residency:
            return False
        if key in self._resident_lru:
            self._resident_lru.move_to_end(key)
            return True
        return False

    def set_residency_tracking(self, enabled: bool) -> None:
        """Enable or disable tracking without reloading the model store."""
        self._track_residency = bool(enabled)

    def _pread_into(
        self, name: str, rows: Sequence[int], out: np.ndarray, read_mode: str
    ) -> None:
        """Read rows with positioned I/O into ``out[slot]``, in row order."""
        ref = self.refs[name]
        dtype, _ = _DTYPES[ref.dtype]
        fd = self._fd(ref.shard)
        for slot, row in enumerate(rows):
            offset = ref.start + row * ref.row_bytes
            if read_mode == "preadv":
                args = (fd, [memoryview(out[slot]).cast("B")], offset)
                read = (
                    _trace_read(name, row, ref.row_bytes, _profiled_preadv, *args)
                    if _PHYSICAL_MISS_TRACE
                    else _profiled_preadv(*args)
                )
            else:
                args = (fd, ref.row_bytes, offset)
                data = (
                    _trace_read(name, row, ref.row_bytes, _profiled_pread, *args)
                    if _PHYSICAL_MISS_TRACE
                    else _profiled_pread(*args)
                )
                read = len(data)
                if read == ref.row_bytes:
                    out[slot] = np.frombuffer(data, dtype=dtype).reshape(ref.shape[1:])
            if read != ref.row_bytes:
                raise OSError(f"short pread for {name} row {row}")
        if self._track_residency:
            self._mark_read(name, rows)
        if self._lock_budget > 0 and ".switch_mlp." in name:
            self._enqueue_lock(name, rows)

    def _enqueue_lock(self, name: str, rows) -> None:
        if self._lock_failed:
            return
        if self._lock_queue is None:
            with self._map_lock:
                if self._lock_queue is None:
                    self._lock_queue = queue.Queue()
                    self._lock_thread = threading.Thread(
                        target=self._lock_worker, name="flashnext-lock",
                        daemon=True,
                    )
                    self._lock_thread.start()
        self._lock_queue.put((name, tuple(int(row) for row in rows)))

    def _lock_worker(self) -> None:
        libc = _libc()
        while True:
            item = self._lock_queue.get()
            try:
                if item is None:
                    return
                name, rows = item
                for row in rows:
                    self._lock_row(libc, name, row)
            finally:
                self._lock_queue.task_done()

    def _lock_row(self, libc, name: str, row: int) -> None:
        """Lock one row, then unlock the least recently read over budget."""
        if self._lock_failed:
            return
        key = (name, row)
        with self._locked_mutex:
            if key in self._locked:
                self._locked.move_to_end(key)
                return
            if key in self._pinned_rows:
                # Unlocking it later would drop the routing pin as well.
                return
            start, length = self._row_span(name, row)
            if libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(length)) != 0:
                error = ctypes.get_errno()
                self._lock_failed = True
                print(
                    f"flashnext: expert locking stopped: {os.strerror(error)}",
                    file=sys.stderr,
                )
                return
            self._locked[key] = (start, length)
            self._locked_bytes += length
            while self._locked_bytes > self._lock_budget and self._locked:
                old_key, (old_start, old_length) = self._locked.popitem(last=False)
                if old_key not in self._pinned_rows:
                    libc.munlock(
                        ctypes.c_void_p(old_start), ctypes.c_size_t(old_length)
                    )
                self._locked_bytes -= old_length

    def drain_expert_locks(self) -> None:
        """Wait until every queued row has been locked. For tests and probes."""
        if self._lock_queue is not None:
            self._lock_queue.join()

    def expert_lock_stats(self) -> dict:
        return {
            "budget_bytes": self._lock_budget,
            "locked_bytes": self._locked_bytes,
            "locked_rows": len(self._locked),
            "failed": self._lock_failed,
        }

    def _unlock_experts(self) -> None:
        if self._lock_queue is not None:
            self._lock_queue.put(None)
            self._lock_thread.join(timeout=5)
            self._lock_queue = None
        libc = _libc()
        with self._locked_mutex:
            for start, length in self._locked.values():
                libc.munlock(ctypes.c_void_p(start), ctypes.c_size_t(length))
            self._locked.clear()
            self._locked_bytes = 0

    def empty_rows(self, name: str, count: int) -> np.ndarray:
        """Allocate one destination for a whole layer's gather."""
        ref = self.refs[name]
        dtype, _ = _DTYPES[ref.dtype]
        return np.empty((count, *ref.shape[1:]), dtype=dtype)

    def expert_record_view(
        self,
        name: str,
        buffer: np.ndarray,
        count: int,
        offset: int,
        record_stride: int,
    ) -> np.ndarray:
        """View one tensor row inside each expert-major destination record."""
        ref = self.refs[name]
        dtype, _ = _DTYPES[ref.dtype]
        if buffer.dtype != np.uint8 or not buffer.flags.c_contiguous:
            raise ValueError("expert record buffer must be contiguous uint8")
        if offset < 0 or record_stride < ref.row_bytes:
            raise ValueError("expert record layout cannot contain the tensor row")
        tail_strides = []
        stride = dtype.itemsize
        for dimension in reversed(ref.shape[1:]):
            tail_strides.append(stride)
            stride *= dimension
        tail_strides.reverse()
        return np.ndarray(
            shape=(count, *ref.shape[1:]),
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=(record_stride, *tail_strides),
        )

    def rows_into(
        self,
        name: str,
        indices: Sequence[int],
        out: np.ndarray,
        read_mode: str | None = None,
    ) -> None:
        """Gather rows straight into a caller-owned slice.

        Same reads as `rows_np`, but every chunk lands in its final position,
        so the caller never concatenates. Safe from worker threads: each call
        owns a disjoint slice.
        """
        rows = indices if isinstance(indices, list) else list(indices)
        mode = read_mode or self._read_mode
        if mode in ("pread", "preadv"):
            self._pread_into(name, rows, out, mode)
            return
        out[:] = self._mapped(name, mode)[rows]

    def to_mx(self, name: str, out: np.ndarray) -> mx.array:
        """Wrap a numpy gather as mx. Call from the main thread only."""
        _, target = _DTYPES[self.refs[name].dtype]
        array = mx.from_dlpack(out, copy=False) if self._dlpack else mx.array(out)
        if target is mx.bfloat16:
            array = array.view(mx.bfloat16)
        return array

    def rows(self, name: str, indices: Sequence[int]) -> mx.array:
        """Return the requested rows of `name`, stacked along axis 0."""
        ref = self.refs[name]
        _, target = _DTYPES[ref.dtype]
        array = mx.array(self.rows_np(name, indices))
        # uint16 -> bfloat16 is a reinterpret, not a conversion.
        if target is mx.bfloat16:
            array = array.view(mx.bfloat16)
        return array

    def whole_np(self, name: str) -> np.ndarray:
        """The full tensor as numpy. One sequential pass, no gather."""
        return np.asarray(self._view(name))

    def close(self) -> None:
        self._unlock_experts()
        self.unpin_all()
        self._shared_views.clear()
        if hasattr(self, "_slab_pack") and self._slab_pack is not None:
            try:
                self._slab_pack.close()
            except Exception:
                pass
            self._slab_pack = None
        for handle in self._maps.values():
            handle.close()
        for fd in self._fds.values():
            os.close(fd)
        self._views.clear()
        self._maps.clear()
        self._fds.clear()
