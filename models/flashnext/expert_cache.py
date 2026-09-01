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
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort

from . import hostwindow
from .store import SafeTensorStore


_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("FLASHNEXT_IO_WORKERS", "16")),
    thread_name_prefix="flashnext-io",
)


def _submit_read(*args):
    """Hand one read to the pool, counted while `hostwindow` is on.

    The counter is what lets a host window claim the drive was idle. Without
    it the claim is an argument about the code rather than a measurement.
    """
    future = _POOL.submit(*args)
    return hostwindow.track(future) if hostwindow.ENABLED else future


_PARTS = ("weight", "scales", "biases")

# Optional wall-clock split of a decode token. Off by default: the counters
# add two time.perf_counter calls per layer. Set FLASHNEXT_PROFILE_IO=1 and
# read the totals with profile_totals(). This measures how long the main
# thread blocks on expert reads, which no derived rate can tell you.
_PROFILE = os.environ.get("FLASHNEXT_PROFILE_IO") == "1"
_PREFILL_PROGRESS = None
_TIMERS = {
    "io_wait": 0.0,
    "router_sync": 0.0,
    "ngram_wait": 0.0,
    "to_mx": 0.0,
    "moe_issue": 0.0,
    "score_sync": 0.0,
    "topk_python": 0.0,
    "shared_expert": 0.0,
    "io_calls": 0,
}


def set_prefill_progress(callback) -> None:
    """Observe completed streamed MoE layers during one prompt prefill."""
    global _PREFILL_PROGRESS
    _PREFILL_PROGRESS = callback


def profile_totals() -> dict:
    return dict(_TIMERS)


def reset_profile() -> None:
    for key in _TIMERS:
        _TIMERS[key] = 0.0
    _TIMERS["io_calls"] = 0

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
# Start expert reads as soon as the routed set is known on the host, instead
# of waiting for switch_mlp to run its own host sync. The bytes read are
# identical; only the moment the reads are issued changes. Set to 0 to A/B.
# Measured 9.9% slower: see the handoff. Default off. "1" submits, "2"
# computes the routed list without submitting, to separate the cost of the
# extra eval from memory-controller contention.
# Read every chunk straight into one destination per part, so the main thread
# never concatenates the pieces. With FLASHNEXT_PREAD_CHUNK=1 the old path
# concatenated one array per expert, copying the whole layer again.
# A list so a benchmark can flip it on a live backend. Read through
# `shared_buffer()`; the module constant could not be changed after import and
# a comparison that cannot change its setting measures the same thing twice.
_SHARED_BUFFER = [os.environ.get("FLASHNEXT_SHARED_READ_BUFFER", "0") != "0"]


def shared_buffer() -> bool:
    return _SHARED_BUFFER[0]


def set_shared_buffer(enabled: bool) -> None:
    _SHARED_BUFFER[0] = bool(enabled)
_EARLY_SUBMIT_MODE = os.environ.get("FLASHNEXT_EARLY_SUBMIT", "0")
_EARLY_SUBMIT = _EARLY_SUBMIT_MODE != "0"
_WARM = ThreadPoolExecutor(max_workers=4, thread_name_prefix="flashnext-warm")
def _touch(store, prefix: str, experts: List[int]) -> None:
    try:
        for part in _PARTS:
            store.rows_np(f"{prefix}.{part}", experts)
    except Exception:
        pass


def warm_layer(switch_mlp, layer_id) -> None:
    """Warm this layer's likely experts while the GPU runs the router.

    A layer blocks on `mx.eval(scores)` for its attention and GDN work with
    the drive idle. Expert sets overlap 35.7% between consecutive tokens, so
    reading the previous token's set for this layer during that window turns
    part of the next read into a page-cache hit. Results are discarded, so
    this cannot change what the model computes.
    """
    if not _WARM_ON or layer_id is None:
        return
    experts = _LAST.get(layer_id)
    if not experts:
        return
    for projection in (
        switch_mlp.gate_proj,
        switch_mlp.up_proj,
        switch_mlp.down_proj,
    ):
        if projection.slab is not None:
            return
        _WARM.submit(
            _touch, projection.cache.store, projection.cache.prefix, experts
        )


def record_layer(layer_id, experts) -> None:
    """Remember this layer's routed set as the next token's prediction."""
    if _WARM_ON and layer_id is not None:
        _LAST[layer_id] = list(experts)


class _SharedRead:
    """A destination plus the futures filling its disjoint slices."""

    __slots__ = ("buffer", "futures")

    def __init__(self, buffer, futures):
        self.buffer = buffer
        self.futures = futures

    def wait(self):
        for future in self.futures:
            future.result()
        return self


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
                [_submit_read(self.store.whole_np, f"{self.prefix}.{part}")]
                for part in _PARTS
            ]
        mode = self.store._read_mode
        if mode == "hybrid":
            mode = (
                "shared_mmap"
                if len(experts) <= self.store._hybrid_cutoff
                else "pread"
            )
        if _SHARED_BUFFER[0]:
            return self._submit_shared(experts, mode)
        pending = []
        for part in _PARTS:
            part_mode = mode
            if mode == "mixed":
                part_mode = "pread" if part == "weight" else "shared_mmap"
            chunk = (
                self.store._pread_chunk
                if part_mode in ("pread", "preadv", "resident")
                else _CHUNK
            )
            pieces = [
                experts[i : i + chunk] for i in range(0, len(experts), chunk)
            ]
            pending.append([
                _submit_read(
                    self.store.rows_np,
                    f"{self.prefix}.{part}",
                    piece,
                    part_mode,
                )
                for piece in pieces
            ])
        return pending

    def _submit_shared(self, experts: List[int], mode: str):
        """One destination per part; each chunk writes its own slice."""
        pending = []
        for part in _PARTS:
            part_mode = mode
            if mode == "mixed":
                part_mode = "pread" if part == "weight" else "shared_mmap"
            chunk = (
                self.store._pread_chunk
                if part_mode in ("pread", "preadv", "resident")
                else _CHUNK
            )
            name = f"{self.prefix}.{part}"
            buffer = self.store.empty_rows(name, len(experts))
            futures = [
                _submit_read(
                    self.store.rows_into,
                    name,
                    experts[start : start + chunk],
                    buffer[start : start + chunk],
                    part_mode,
                )
                for start in range(0, len(experts), chunk)
            ]
            pending.append(_SharedRead(buffer, futures))
        return pending

    def to_mx(self, raw):
        out = []
        for part, chunks in zip(_PARTS, raw):
            if isinstance(chunks, _SharedRead):
                block = chunks.buffer
            else:
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


def _await_read(pending):
    """Resolve one projection's pending reads into what `to_mx` expects."""
    return [
        item.wait() if isinstance(item, _SharedRead)
        else [f.result() for f in item]
        for item in pending
    ]


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

    def prefetch(self, wanted) -> None:
        """Issue this layer's expert reads before the next host sync.

        `wanted` must equal the list `_one_pass` would build, in the same
        order, or the reads are discarded and re-issued normally.
        """
        if not _EARLY_SUBMIT or not wanted:
            return
        if _EARLY_SUBMIT_MODE == "2":
            return
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        if projections[0].slab is not None:
            return
        wanted = list(wanted)
        if self.gate_proj.cache.store._sort_reads:
            wanted.sort()
        self._prefetch = (
            wanted,
            [p.cache.submit(wanted, False) for p in projections],
        )

    def __call__(self, x, indices, allow_sort=True) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        flat = indices.reshape(-1)
        if _PROFILE:
            began = time.perf_counter()
            mx.eval(flat)
            _TIMERS["router_sync"] += time.perf_counter() - began
        else:
            mx.eval(flat)
        with hostwindow.window("route_tolist"):
            routed = flat.tolist()
        observer = _PREFILL_PROGRESS
        if observer is not None and self.layer_id >= 0:
            observer(self.layer_id)

        slabs = [p.slab for p in (self.gate_proj, self.up_proj, self.down_proj)]
        use_slab = slabs[0] is not None and slabs[0].ready()
        if slabs[0] is not None and not use_slab:
            for sl in slabs:
                sl.admit(list(dict.fromkeys(routed)))
            use_slab = slabs[0].ready()

        if not use_slab:
            return self._one_pass(x, indices, routed, None, allow_sort=allow_sort)

        hit = [e for e in routed if e in slabs[0].slot]
        miss = [e for e in routed if e not in slabs[0].slot]
        self.hits += len(hit)
        self.misses += len(miss)
        if not miss:
            return self._one_pass(x, indices, routed, slabs, allow_sort=allow_sort)
        if not hit:
            return self._one_pass(x, indices, routed, None, allow_sort=allow_sort)

        # Sum the two groups at the output. Accumulate in float32: three
        # bfloat16 adds drift by one ULP against the dense path.
        out = self._one_pass(
            x, indices, routed, slabs, mask=hit, allow_sort=allow_sort
        ).astype(mx.float32)
        out = out + self._one_pass(
            x, indices, routed, None, mask=miss, allow_sort=allow_sort
        ).astype(
            mx.float32
        )
        return out.astype(mx.bfloat16)

    def _one_pass(self, x, indices, routed, slabs, mask=None, allow_sort=True):
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        if slabs is not None:
            local = mx.array(
                [slabs[0].slot.get(e, 0) for e in routed], dtype=mx.uint32
            ).reshape(indices.shape)
            weights = [sl.parts for sl in slabs]
        else:
            with hostwindow.window("plan_host"):
                wanted = list(
                    dict.fromkeys(
                        e for e in routed if mask is None or e in set(mask)
                    )
                )
                if self.gate_proj.cache.store._sort_reads:
                    wanted.sort()
                if not wanted:
                    wanted = [routed[0]]
                order = {e: i for i, e in enumerate(wanted)}
                local = mx.array(
                    [order.get(e, 0) for e in routed], dtype=mx.uint32
                ).reshape(indices.shape)
            prefetched = getattr(self, "_prefetch", None)
            self._prefetch = None
            if prefetched is not None and prefetched[0] == wanted:
                pending = prefetched[1]
            else:
                pending = [p.cache.submit(wanted, False) for p in projections]
            if _PROFILE:
                began = time.perf_counter()
                with hostwindow.window("io_await"):
                    raw = [_await_read(fs) for fs in pending]
                _TIMERS["io_wait"] += time.perf_counter() - began
                _TIMERS["io_calls"] += 1
                began = time.perf_counter()
                with hostwindow.window("to_mx_host"):
                    weights = [
                        p.cache.to_mx(chunks)
                        for p, chunks in zip(projections, raw)
                    ]
                _TIMERS["to_mx"] += time.perf_counter() - began
            elif hostwindow.ENABLED:
                # Split the wait from the conversion so each lands in its own
                # window. Production keeps the interleaved form below, so this
                # reordering never reaches a normal run.
                with hostwindow.window("io_await"):
                    raw = [_await_read(fs) for fs in pending]
                with hostwindow.window("to_mx_host"):
                    weights = [
                        p.cache.to_mx(chunks)
                        for p, chunks in zip(projections, raw)
                    ]
            else:
                weights = [
                    p.cache.to_mx(_await_read(fs))
                    for p, fs in zip(projections, pending)
                ]

        issue_began = time.perf_counter() if _PROFILE else 0.0
        with hostwindow.window("moe_issue_host"):
            return self._issue(
                x, indices, projections, local, weights, mask, routed,
                allow_sort, issue_began,
            )

    def _issue(
        self, x, indices, projections, local, weights, mask, routed,
        allow_sort, issue_began,
    ):
        do_sort = indices.size >= 64 and allow_sort
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
        if _PROFILE:
            _TIMERS["moe_issue"] += time.perf_counter() - issue_began
        return o

    def stats(self):
        hits = sum(p.cache.hits for p in (self.gate_proj, self.up_proj, self.down_proj))
        misses = sum(
            p.cache.misses for p in (self.gate_proj, self.up_proj, self.down_proj)
        )
        return hits, misses
