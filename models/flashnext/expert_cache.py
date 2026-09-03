"""Expert weights that live on disk and visit memory only when routed to.

A dense SwitchLinear holds every expert resident: 512 experts x 48 layers is
45 GB. Only `top_k` experts run per token, so this class reads routed rows
from the checkpoint on demand.

The gather still runs against a contiguous tensor. Rather than maintaining one
big cache buffer and paying a full copy on every miss, the needed rows are
stacked per call. A stack of ten rows is 4 MB, which is far cheaper than
rewriting a 20 MB cache buffer.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

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
    "score_sync_bytes": 0.0,
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
# never concatenates the pieces. At FLASHNEXT_PREAD_CHUNK=1 the old path gave
# every expert its own allocation and then copied the whole layer again, which
# cost 35.9 ms per token.
#
# This was measured once before and returned as a tie: the buffer saved 35 ms
# of copy and the GPU drain grew 37 ms. The unmeasured explanation in the
# research log was the write pattern, since 16 workers scatter across one
# buffer where the concatenate was a single sequential copy. Chunk size and
# this switch had never been tested together. They now are. At chunk 2 each
# worker writes one contiguous run and most of the NVMe queue depth survives:
#
#   12 arms, clean boot, 40 tokens
#   concat chunk 1   2.67 gen median   467.7 MB/token
#   buffer chunk 2   2.83 gen median   457.7 MB/token
#   +6.3% gen median against a 4.4% band, so it stands
#   ahead in 10 of 12 pairs, sign test p = 0.019, fewer bytes in 10 of 12
#
# Token IDs are identical across every arm: the same bytes land in a different
# destination layout, so nothing the model computes changes. Prefill is
# unaffected, measured A/B/B/A at 512 and 2048 tokens.
#
# The pair was measured on the pread family only, which is what
# `exact-quality`, `cache-aware`, `standard` and `fused-quality` run. `fast`
# and `fast-quality` read through `shared_mmap`, where the chunk is _CHUNK and
# the copy is a numpy slice assignment rather than a concatenate, so none of
# the evidence above applies to them. Defaulting on for every mode would have
# changed two profiles nobody has measured. It stays off there until someone
# does. Setting the variable to 1 or 0 forces it either way for every mode.
#
# A list so a benchmark can flip it on a live backend. The module constant
# could not be changed after import, and a comparison that cannot change its
# setting measures the same thing twice.
_PREAD_MODES = ("pread", "preadv", "resident")
_SHARED_BUFFER = [os.environ.get("FLASHNEXT_SHARED_READ_BUFFER")]


# `empty_rows` allocates a new numpy block for every part of every layer, so a
# token creates 432 host allocations and hands each one to the GPU as a new
# Metal-visible buffer. This ring reuses a few destinations instead. Depth has
# to exceed one: a layer's MoE output is not evaluated until the next layer's
# router sync, so the GPU may still be reading the previous destination. A list,
# for the same reason the shared-buffer switch is one.
_ARENA = [int(os.environ.get("FLASHNEXT_BUFFER_ARENA", "0"))]
_ARENA_WIDTH = 16
# One pool for the whole model, not one per layer. Every layer's gate_proj
# weight has the same shape, so the same block serves all 48 of them and the
# GPU sees a handful of buffers per token instead of 432. Keyed per layer the
# ring costs about 9 GB and the machine swaps; keyed by shape it costs about
# 150 MB. Allocation happens on the main thread inside `_submit_shared`, so no
# lock is needed.
_ARENA_POOL: dict = {}


def buffer_arena() -> int:
    """Ring depth, or 0 for one fresh allocation per part per layer."""
    return _ARENA[0]


def set_buffer_arena(depth) -> None:
    _ARENA[0] = int(depth)


def shared_buffer(mode: str = "pread") -> bool:
    """Whether this read mode fills one destination per part."""
    forced = _SHARED_BUFFER[0]
    if forced is not None:
        return forced != "0"
    return mode in _PREAD_MODES


def set_shared_buffer(enabled) -> None:
    """Force the switch, or pass None to return to the per-mode default."""
    _SHARED_BUFFER[0] = None if enabled is None else ("1" if enabled else "0")
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
    """Per-projection reader for routed expert rows.

    The old row-level LRU merged cached and fresh rows in numpy. That path was
    never used by the runtime and made `capacity` appear to control caching.
    The active path submits all routed rows as one read and converts them once.
    """

    __slots__ = ("store", "prefix", "capacity")

    def __init__(self, store: SafeTensorStore, prefix: str, capacity: int):
        self.store = store
        self.prefix = prefix
        self.capacity = capacity


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
        if shared_buffer(mode):
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
            depth = _ARENA[0]
            buffer = (
                self._reused_rows(name, len(experts), depth) if depth
                else self.store.empty_rows(name, len(experts))
            )
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

    def _reused_rows(self, name: str, count: int, depth: int):
        """Hand out one destination from a ring that lives for the whole run.

        The ring is built at the router's top-k width, so a layer that routes
        fewer experts takes a prefix of the same block rather than a new
        allocation. A prefix of a C-contiguous block is still contiguous, so
        `to_mx` wraps it without a copy exactly as before.

        Depth must exceed one. A layer's MoE output is not evaluated until the
        next layer's router sync, so the GPU may still be reading the previous
        destination when the following layer asks for one.
        """
        # Key on the projection and part, not on the shape. `gate_proj` and
        # `up_proj` share a shape, so a shape key advances the ring twice per
        # layer and wraps before the GPU has read the earlier block. That is
        # not a crash; it silently changes the output.
        key = name.rsplit(".", 2)[-2:]
        key = (key[0], key[1], depth)
        ring = _ARENA_POOL.get(key)
        if ring is None:
            ring = [[self.store.empty_rows(name, _ARENA_WIDTH)
                     for _ in range(depth)], 0]
            _ARENA_POOL[key] = ring
        if count > _ARENA_WIDTH:
            # Wider than the ring was built for. Allocate rather than truncate
            # the gather; a silent short read would change the output.
            return self.store.empty_rows(name, count)
        buffers, index = ring
        ring[1] = (index + 1) % depth
        return buffers[index][:count]

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

    def fetch(self, experts: List[int]):
        """Read routed rows synchronously for a standalone projection call."""
        return self._read(experts)


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
        self._metal_executors = {}

    @property
    def metal_combines_scores(self) -> bool:
        """Report the opt-in decode path to the patched MoE block."""
        return (
            os.environ.get("FLASHNEXT_METAL_RUNTIME") == "1"
            and self.gate_proj.slab is None
            and self.layer_id != 0
        )

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

    def __call__(self, x, indices, allow_sort=True, scores=None) -> mx.array:
        flat_input = x.reshape(-1, x.shape[-1])
        x = mx.expand_dims(x, (-2, -3))
        # `_moe_call` leaves the routed list here when one-sync is on. It is
        # the same list this method would fetch, built from values already on
        # the host, so taking it removes a Metal round trip per layer.
        handed = getattr(self, "_routed_host", None)
        self._routed_host = None
        if handed is not None:
            routed, _shape = handed
        else:
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
            return self._one_pass(
                x, indices, routed, None, allow_sort=allow_sort,
                flat_input=flat_input, scores=scores,
            )

        hit = [e for e in routed if e in slabs[0].slot]
        miss = [e for e in routed if e not in slabs[0].slot]
        self.hits += len(hit)
        self.misses += len(miss)
        if not miss:
            return self._one_pass(
                x, indices, routed, slabs, allow_sort=allow_sort,
                flat_input=flat_input, scores=scores,
            )
        if not hit:
            return self._one_pass(
                x, indices, routed, None, allow_sort=allow_sort,
                flat_input=flat_input, scores=scores,
            )

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

    def _one_pass(
        self, x, indices, routed, slabs, mask=None, allow_sort=True,
        flat_input=None, scores=None,
    ):
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
            if (
                self.metal_combines_scores
                and scores is not None
                and mask is None
                and flat_input is not None
                and flat_input.shape[0] <= 8
                and self.gate_proj.group_size == 32
                and self.gate_proj.bits == 4
            ):
                from .metal_runtime import MetalMoEExecutor

                slots = indices.shape[-1]
                local = local.reshape(flat_input.shape[0], slots)
                routed_scores = scores.reshape(flat_input.shape[0], slots)
                expert_count = weights[0][0].shape[0]
                key = (expert_count, flat_input.shape[-1], slots)
                executor = self._metal_executors.get(key)
                if executor is None:
                    executor = MetalMoEExecutor(
                        expert_count, flat_input.shape[-1], slots
                    )
                    self._metal_executors[key] = executor
                projection_packs = {
                    "gate_proj": weights[0],
                    "up_proj": weights[1],
                    "down_proj": weights[2],
                }
                output = executor.execute(
                    flat_input, local, projection_packs,
                    scores=routed_scores,
                )
                output = output.reshape(
                    *indices.shape[:-1], output.shape[-1]
                )
                if (
                    os.environ.get("FLASHNEXT_METAL_VERIFY") == "1"
                    and self.layer_id == 1
                ):
                    custom_gate, custom_up, custom_down = executor.execute(
                        flat_input, local, projection_packs, return_all=True
                    )
                    reference_gate_raw = projections[0](
                        x, local.reshape(indices.shape),
                        plan=(None, local.reshape(indices.shape)),
                        weights=weights[0], sorted_indices=False,
                    )
                    reference_up_raw = projections[1](
                        x, local.reshape(indices.shape),
                        plan=(None, local.reshape(indices.shape)),
                        weights=weights[1], sorted_indices=False,
                    )
                    reference_down = projections[2](
                        self.activation(reference_up_raw, reference_gate_raw),
                        local.reshape(indices.shape),
                        plan=(None, local.reshape(indices.shape)),
                        weights=weights[2], sorted_indices=False,
                    ).squeeze(-2)
                    reference_gate = reference_gate_raw.squeeze(-2)
                    reference_up = reference_up_raw.squeeze(-2)
                    reference = self._issue(
                        x, indices, projections, local.reshape(indices.shape),
                        weights, None, routed, allow_sort, 0.0,
                    )
                    reference = (reference * scores[..., None]).sum(axis=-2)
                    mx.eval(
                        output, reference, custom_gate, custom_up, custom_down,
                        reference_gate, reference_up, reference_down,
                    )
                    component_errors = []
                    for actual_part, expected_part in (
                        (custom_gate, reference_gate),
                        (custom_up, reference_up),
                        (custom_down, reference_down),
                    ):
                        expected_part = expected_part.reshape(actual_part.shape)
                        component_errors.append(mx.max(mx.abs(
                            actual_part.astype(mx.float32)
                            - expected_part.astype(mx.float32)
                        )).item())
                    error = mx.max(mx.abs(
                        output.astype(mx.float32) - reference.astype(mx.float32)
                    )).item()
                    print(
                        f"metal layer {self.layer_id}: slots={slots} "
                        f"parts={component_errors} max_abs={error}"
                    )
                if _PROFILE:
                    _TIMERS["moe_issue"] += time.perf_counter() - issue_began
                return output
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
