"""Expert weights that live on disk and visit memory only when routed to.

A dense SwitchLinear holds every expert resident: 512 experts x 48 layers is
45 GB. Only `top_k` experts run per token, so this class keeps a bounded LRU
of expert rows and reads the rest from the checkpoint on demand.

The gather still runs against a contiguous tensor. Rather than maintaining one
big cache buffer and paying a full copy on every miss, the needed rows are
stacked per call. A stack of ten rows is 4 MB, which is far cheaper than
rewriting a 20 MB cache buffer.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort

from .store import SafeTensorStore


_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("FLASHNEXT_IO_WORKERS", "16")),
    thread_name_prefix="flashnext-io",
)
_PARTS = ("weight", "scales", "biases")

# Reading whole tensors instead of gathering rows was measured and rejected.
# The arithmetic looked right (a scattered gather runs ~900 MB/s, a sequential
# one ~2.5 GB/s, crossing near 185 experts), but on a 93-token prefill it ran
# 208 s against 124.6 s for the gather. Two reasons the model missed: a prefill
# routes about 400 distinct experts, not 512, so bulk reads 28% more bytes than
# it needs; and 943 MB per layer evicts the page cache that later layers want.
# Set FLASHNEXT_BULK to a threshold to experiment.
_BULK_ABOVE = int(os.environ.get("FLASHNEXT_BULK", 1 << 30))

# Rows per read. A gather's throughput collapses once its output buffer gets
# large: measured at 16 workers, 1027 MB/s for 10 rows, 1205 MB/s for 96, and
# 484 MB/s for 290 (a 237 MB buffer). Decode routes 10 and is unaffected;
# prefill routes hundreds and is split into chunks.
_CHUNK = int(os.environ.get("FLASHNEXT_CHUNK", 96))

# Expert sets never repeat exactly between tokens (measured 0%), but they
# overlap 35.7%. Touching the previous token's rows for the layers ahead warms
# the page cache while the GPU works. Results are discarded: this only makes
# the real read faster, so it cannot change what the model computes.
_LAST: Dict[int, List[int]] = {}
_WARM_ON = os.environ.get("FLASHNEXT_WARM") == "1"
_WARM = ThreadPoolExecutor(max_workers=4, thread_name_prefix="flashnext-warm")
def _touch(store, prefix: str, experts: List[int]) -> None:
    try:
        for part in _PARTS:
            store.rows_np(f"{prefix}.{part}", experts)
    except Exception:
        pass


class ExpertLRU:
    """Per-projection reader, with an optional and unhelpful row cache.

    Caching routed experts was tried three ways and lost every time. Decode at
    capacity 0 runs 1390 ms/token; at 16 it runs 2824, at 96 it runs 4257 with
    a 76.5% hit rate. The reasons, in the order they were found:

      1. Merging hits with `mx.stack` leaves a lazy node per call for the next
         `mx.eval` to materialize, which cost more than re-reading.
      2. Merging with `np.stack` instead still loses: cached rows are views
         into the buffer they were read in, so one cached row pins the whole
         chunk and the process starts paging.
      3. Routed sets never repeat between tokens (0% over 480 samples, 35.7%
         overlap), so no whole-result cache can hit at all.

    Decode is I/O bound at the drive's limit: a token needs 1475 MB and the
    gather runs at 1068 MB/s, which is 1381 ms against 1390 ms measured. The
    48 host syncs cost 8 ms in total. Reading fewer bytes is the only lever
    left, and that means a lower-bit checkpoint, not a code change.
    """

    __slots__ = ("store", "prefix", "capacity", "_rows", "hits", "misses", "_missing")

    def __init__(self, store: SafeTensorStore, prefix: str, capacity: int):
        self.store = store
        self.prefix = prefix
        self.capacity = capacity
        self._rows: "OrderedDict[int, Tuple[mx.array, mx.array, mx.array]]" = (
            OrderedDict()
        )
        self.hits = 0
        self.misses = 0
        self._missing = []

    def submit(self, experts: List[int], bulk: bool = False):
        """Queue reads so a whole layer flies at once, chunked to stay fast."""
        if bulk:
            return [
                [_POOL.submit(self.store.whole_np, f"{self.prefix}.{part}")]
                for part in _PARTS
            ]
        mode = self.store._read_mode
        if mode == "hybrid":
            mode = (
                "shared_mmap"
                if len(experts) <= self.store._hybrid_cutoff
                else "pread"
            )
        pending = []
        for part in _PARTS:
            part_mode = mode
            if mode == "mixed":
                part_mode = "pread" if part == "weight" else "shared_mmap"
            chunk = (
                self.store._pread_chunk
                if part_mode in ("pread", "preadv")
                else _CHUNK
            )
            pieces = [
                experts[i : i + chunk] for i in range(0, len(experts), chunk)
            ]
            pending.append([
                _POOL.submit(
                    self.store.rows_np,
                    f"{self.prefix}.{part}",
                    piece,
                    part_mode,
                )
                for piece in pieces
            ])
        return pending

    def to_mx(self, raw):
        out = []
        for part, chunks in zip(_PARTS, raw):
            block = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            out.append(self.store.to_mx(f"{self.prefix}.{part}", block))
        return tuple(out)

    def _read(self, experts: List[int]):
        return tuple(
            self.store.to_mx(f"{self.prefix}.{p}", self.store.rows_np(f"{self.prefix}.{p}", experts))
            for p in _PARTS
        )

    def fetch_np(self, experts: List[int], fresh):
        """Merge cached rows with freshly read ones, in numpy.

        The first version of this cache merged with mx.stack and was slower
        than no cache at all: every stack left a lazy node for the next
        mx.eval to materialize, and that dominated decode. np.stack copies in
        C and leaves nothing behind, so the 76% hit rate finally pays.
        """
        # Build the answer before evicting anything: when the routed set is
        # larger than the cache, inserting first drops rows this call needs.
        current = {}
        if fresh is not None:
            for slot, expert in enumerate(self._missing):
                current[expert] = tuple(part[slot] for part in fresh)

        picked = [current.get(e) or self._rows[e] for e in experts]
        out = []
        for index in range(3):
            rows = [row[index] for row in picked]
            out.append(np.stack(rows) if len(rows) > 1 else rows[0][None])

        for expert in experts:
            if expert in self._rows:
                self._rows.move_to_end(expert)
        for expert, row in current.items():
            self._rows[expert] = row
            if len(self._rows) > self.capacity:
                self._rows.popitem(last=False)
        return tuple(
            self.store.to_mx(f"{self.prefix}.{part}", block)
            for part, block in zip(_PARTS, out)
        )

    def plan_missing(self, experts: List[int]):
        """Which experts this call must read. Records them for fetch_np."""
        self._missing = [e for e in experts if e not in self._rows]
        self.hits += len(experts) - len(self._missing)
        self.misses += len(self._missing)
        return self._missing

    def clear(self) -> None:
        self._rows.clear()


def build_plan(indices):
    """Resolve routed experts to cache slots. Costs one host sync; share it."""
    flat = indices.reshape(-1)
    mx.eval(flat)
    routed = flat.tolist()
    order: Dict[int, int] = {}
    for expert in routed:
        if expert not in order:
            order[expert] = len(order)
    local = mx.array([order[e] for e in routed], dtype=mx.uint32).reshape(indices.shape)
    return list(order), local


class StreamingSwitchLinear(nn.Module):
    """Drop-in replacement for QuantizedSwitchLinear backed by the checkpoint."""

    def __init__(
        self,
        store: SafeTensorStore,
        prefix: str,
        group_size: int,
        bits: int,
        mode: str,
        capacity: int,
        slab=None,
    ):
        super().__init__()
        self.slab = slab
        self.cache = ExpertLRU(store, prefix, capacity)
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        shape = store.shape(f"{prefix}.weight")
        self.num_experts = shape[0]
        self._out_dims = shape[1]

    @property
    def output_dims(self) -> int:
        return self._out_dims

    def __call__(self, x, indices, sorted_indices=False, plan=None, weights=None):
        # `plan` carries the expert list and remapped indices resolved once per
        # layer. Resolving them here instead would force one host sync per
        # projection, tripling the stalls per token.
        if plan is None:
            plan = build_plan(indices)
        experts, local = plan

        if weights is None:
            weights = self.cache.fetch(experts)
        weight, scales, biases = weights

        return mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            rhs_indices=local,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class ResidentSlab:
    """Experts kept in unified memory, indexed by gather_qmm without a copy.

    Earlier caches merged cached and freshly-read *weights* into one array, and
    that merge cost more than re-reading. This one never merges weights: the
    slab is passed to gather_qmm whole and the routed slots index into it, so a
    hit costs 0.59 ms against 9.3 ms for a cold read of the same experts. Hits
    and misses run as two separate GLU passes and are summed at the output,
    which is 51 KB rather than the 30 MB of weights.
    """

    __slots__ = ("store", "prefix", "capacity", "slot", "parts", "_pending")

    def __init__(self, store, prefix: str, capacity: int):
        self.store = store
        self.prefix = prefix
        self.capacity = capacity
        self.slot: Dict[int, int] = {}
        self.parts = None
        self._pending: List[int] = []

    def admit(self, experts: List[int]) -> None:
        """Fill the slab on first sight; never evict, so no row is rewritten."""
        if self.parts is not None or len(self.slot) >= self.capacity:
            return
        for expert in experts:
            if expert not in self.slot and len(self._pending) < self.capacity:
                self.slot[expert] = len(self._pending)
                self._pending.append(expert)
        if len(self._pending) >= self.capacity:
            self.build()

    def build(self) -> None:
        if self.parts is not None or not self._pending:
            return
        self.parts = tuple(
            self.store.rows(f"{self.prefix}.{part}", self._pending) for part in _PARTS
        )
        mx.eval(self.parts)
        self._pending = []

    def ready(self) -> bool:
        return self.parts is not None


class StreamingSwitchGLU(nn.Module):
    """SwitchGLU whose three projections stream from the checkpoint."""

    def __init__(
        self,
        store: SafeTensorStore,
        prefix: str,
        group_size: int,
        bits: int,
        mode: str,
        capacity: int,
        activation,
        layer_id: int = -1,
        next_prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.next_prefix = next_prefix
        slab_size = int(os.environ.get("FLASHNEXT_SLAB", 0))
        make = lambda name: StreamingSwitchLinear(
            store, f"{prefix}.{name}", group_size, bits, mode, capacity,
            slab=ResidentSlab(store, f"{prefix}.{name}", slab_size) if slab_size else None,
        )
        self.hits = 0
        self.misses = 0
        self.gate_proj = make("gate_proj")
        self.up_proj = make("up_proj")
        self.down_proj = make("down_proj")
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        flat = indices.reshape(-1)
        mx.eval(flat)
        routed = flat.tolist()

        slabs = [p.slab for p in (self.gate_proj, self.up_proj, self.down_proj)]
        use_slab = slabs[0] is not None and slabs[0].ready()
        if slabs[0] is not None and not use_slab:
            for sl in slabs:
                sl.admit(list(dict.fromkeys(routed)))
            use_slab = slabs[0].ready()

        if not use_slab:
            return self._one_pass(x, indices, routed, None)

        hit = [e for e in routed if e in slabs[0].slot]
        miss = [e for e in routed if e not in slabs[0].slot]
        self.hits += len(hit)
        self.misses += len(miss)
        if not miss:
            return self._one_pass(x, indices, routed, slabs)
        if not hit:
            return self._one_pass(x, indices, routed, None)

        # Sum the two groups at the output. Accumulate in float32: three
        # bfloat16 adds drift by one ULP against the dense path.
        out = self._one_pass(x, indices, routed, slabs, mask=hit).astype(mx.float32)
        out = out + self._one_pass(x, indices, routed, None, mask=miss).astype(
            mx.float32
        )
        return out.astype(mx.bfloat16)

    def _one_pass(self, x, indices, routed, slabs, mask=None):
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        if slabs is not None:
            local = mx.array(
                [slabs[0].slot.get(e, 0) for e in routed], dtype=mx.uint32
            ).reshape(indices.shape)
            weights = [sl.parts for sl in slabs]
        else:
            wanted = list(dict.fromkeys(e for e in routed if mask is None or e in set(mask)))
            if self.gate_proj.cache.store._sort_reads:
                wanted.sort()
            if not wanted:
                wanted = [routed[0]]
            order = {e: i for i, e in enumerate(wanted)}
            local = mx.array(
                [order.get(e, 0) for e in routed], dtype=mx.uint32
            ).reshape(indices.shape)
            pending = [p.cache.submit(wanted, False) for p in projections]
            weights = [
                p.cache.to_mx([[f.result() for f in part] for part in fs])
                for p, fs in zip(projections, pending)
            ]

        do_sort = indices.size >= 64
        inv = None
        xs = x
        if do_sort:
            xs, local, inv = _gather_sort(x, local)

        g = projections[0](xs, local, plan=(None, local), weights=weights[0],
                           sorted_indices=do_sort)
        u = projections[1](xs, local, plan=(None, local), weights=weights[1],
                           sorted_indices=do_sort)
        o = projections[2](self.activation(u, g), local, plan=(None, local),
                           weights=weights[2], sorted_indices=do_sort)
        if do_sort:
            o = _scatter_unsort(o, inv, indices.shape)
        o = o.squeeze(-2)

        if mask is not None:
            keep = set(mask)
            m = mx.array(
                [1.0 if e in keep else 0.0 for e in routed], dtype=mx.bfloat16
            ).reshape(*indices.shape, 1)
            o = o * m
        return o

    def stats(self):
        hits = sum(p.cache.hits for p in (self.gate_proj, self.up_proj, self.down_proj))
        misses = sum(
            p.cache.misses for p in (self.gate_proj, self.up_proj, self.down_proj)
        )
        return hits, misses
