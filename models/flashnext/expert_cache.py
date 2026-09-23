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

import json
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort

from . import hostwindow
from .slab_pack import (
    DOWN_BIASES_OFFSET,
    DOWN_SCALES_OFFSET,
    DOWN_WEIGHT_OFFSET,
    GATE_BIASES_OFFSET,
    GATE_SCALES_OFFSET,
    GATE_WEIGHT_OFFSET,
    RECORD_STRIDE,
    validate_slab_allocation,
    UP_BIASES_OFFSET,
    UP_SCALES_OFFSET,
    UP_WEIGHT_OFFSET,
)
from .store import SafeTensorStore, begin_read_profile, finish_read_profile
from .physical_miss import (
    allocate_physical_miss_hybrid_slots,
    last_hybrid_summary,
    load_profile,
)


_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("FLASHNEXT_IO_WORKERS", "16")),
    thread_name_prefix="flashnext-io",
)


# This state is diagnostic only. It does not control submission or worker
# selection. Set FLASHNEXT_PROFILE_SCORE_SYNC=1 to collect score-sync data.
# Pool-state tracking belongs only to the explicit score-sync diagnostic.
_SCORE_SYNC_PROFILE = os.environ.get("FLASHNEXT_PROFILE_SCORE_SYNC") == "1"
_POOL_STATE_LOCK = threading.Lock()
_POOL_STATE = {"queued": 0, "running": 0, "completed": 0}


def _pool_submit() -> None:
    if not _SCORE_SYNC_PROFILE:
        return
    with _POOL_STATE_LOCK:
        _POOL_STATE["queued"] += 1


def _pool_started() -> None:
    with _POOL_STATE_LOCK:
        _POOL_STATE["queued"] -= 1
        _POOL_STATE["running"] += 1


def _pool_finished() -> None:
    with _POOL_STATE_LOCK:
        _POOL_STATE["running"] -= 1
        _POOL_STATE["completed"] += 1


def read_pool_state() -> dict:
    """Return the diagnostic read-pool state without changing scheduling."""
    with _POOL_STATE_LOCK:
        return dict(_POOL_STATE)


def _tracked_read_call(function, *args):
    _pool_started()
    try:
        return function(*args)
    finally:
        _pool_finished()


def _physical_bytes_read() -> int:
    """Return physical bytes, or -1 when the platform counter is unavailable."""
    from models.flashnext.diskio import disk_bytes_read

    return disk_bytes_read()


# Thread QoS for the read workers and the calling thread. macOS schedules
# threads with an unspecified or default QoS class on either core type and may
# wake them late; decode measured about 80 ms/token between submitting a read
# and a worker starting it. `user-interactive` asks for the performance cores
# and the shortest wake latency. The pool threads apply the requested class
# lazily on their next task. Off ("default") leaves submission unchanged.
_QOS_CLASSES = {
    "user-interactive": 0x21,
    "user-initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
}
_QOS = [None]          # requested class value, or None when never enabled
_QOS_ROUTE = [False]   # route tasks through _qos_call (enabled once, or dirty)
_QOS_LOCAL = threading.local()
try:
    import ctypes as _ctypes

    _LIBPTHREAD = _ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    _LIBPTHREAD.pthread_set_qos_class_self_np.argtypes = [
        _ctypes.c_uint, _ctypes.c_int,
    ]
    _LIBPTHREAD.pthread_set_qos_class_self_np.restype = _ctypes.c_int
except (OSError, AttributeError):  # pragma: no cover - macOS only
    _LIBPTHREAD = None


def _apply_thread_qos(value: int) -> None:
    if getattr(_QOS_LOCAL, "value", None) == value or _LIBPTHREAD is None:
        return
    if _LIBPTHREAD.pthread_set_qos_class_self_np(value, 0) != 0:
        raise OSError(f"pthread_set_qos_class_self_np({value:#x}) failed")
    _QOS_LOCAL.value = value


def _qos_call(function, *args):
    value = _QOS[0]
    if value is not None:
        _apply_thread_qos(value)
    return function(*args)


def set_io_qos(name: str) -> None:
    """Request a QoS class for read workers and the calling thread."""
    if name not in _QOS_CLASSES:
        raise ValueError(f"unknown QoS class {name!r}")
    if name == "default" and _QOS[0] is None:
        return
    value = _QOS_CLASSES[name]
    _QOS[0] = value
    _QOS_ROUTE[0] = True
    _apply_thread_qos(value)


def io_qos() -> str:
    value = _QOS[0]
    for name, known in _QOS_CLASSES.items():
        if known == value:
            return name
    return "default"


def _submit_read(*args):
    """Hand one read to the pool, counted while `hostwindow` is on.

    The counter is what lets a host window claim the drive was idle. Without
    it the claim is an argument about the code rather than a measurement.
    """
    if _QOS_ROUTE[0]:
        args = (_qos_call, *args)
    track_pool = _SCORE_SYNC_PROFILE
    if track_pool:
        _pool_submit()
    try:
        if _PROFILE:
            future = _POOL.submit(
                _profiled_read_call,
                time.perf_counter(),
                track_pool,
                *args,
            )
        elif track_pool:
            future = _POOL.submit(_tracked_read_call, *args)
        else:
            future = _POOL.submit(*args)
    except BaseException:
        if track_pool:
            with _POOL_STATE_LOCK:
                _POOL_STATE["queued"] -= 1
        raise
    return hostwindow.track(future) if hostwindow.ENABLED else future


_PARTS = ("weight", "scales", "biases")
_STREAM_RECORD_PARTS = (
    ("gate_proj", "weight", GATE_WEIGHT_OFFSET),
    ("gate_proj", "scales", GATE_SCALES_OFFSET),
    ("gate_proj", "biases", GATE_BIASES_OFFSET),
    ("up_proj", "weight", UP_WEIGHT_OFFSET),
    ("up_proj", "scales", UP_SCALES_OFFSET),
    ("up_proj", "biases", UP_BIASES_OFFSET),
    ("down_proj", "weight", DOWN_WEIGHT_OFFSET),
    ("down_proj", "scales", DOWN_SCALES_OFFSET),
    ("down_proj", "biases", DOWN_BIASES_OFFSET),
)

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
    "score_sync_physical_bytes": 0,
    "score_sync_calls": 0,
    "score_sync_threshold_active": 0,
    "score_sync_threshold_inactive": 0,
    "score_sync_pool_queued_before": 0,
    "score_sync_pool_queued_after": 0,
    "score_sync_pool_running_before": 0,
    "score_sync_pool_running_after": 0,
    "score_sync_pool_completed_before": 0,
    "score_sync_pool_completed_after": 0,
    "io_calls": 0,
    "io_wait_packed": 0.0,
    "io_calls_packed": 0,
    "read_tasks": 0,
    "pread_calls": 0,
    "pread_bytes": 0,
    "queue_delay_sum": 0.0,
    "task_service_sum": 0.0,
    "pread_service_sum": 0.0,
    "critical_queue": 0.0,
    "critical_pread": 0.0,
    "critical_task_overhead": 0.0,
    "completion_overhead": 0.0,
    "layer_completion_sum": 0.0,
    "layer_completion_count": 0,
}


# Diagnostic route trace: callback(layer_id, routed_expert_ids) for every
# streamed MoE call, prefill included. None disables it.
_ROUTE_TRACE = [None]


def set_route_trace(callback=None) -> None:
    _ROUTE_TRACE[0] = callback


def set_prefill_progress(callback) -> None:
    """Observe completed streamed MoE layers during one prompt prefill."""
    global _PREFILL_PROGRESS
    _PREFILL_PROGRESS = callback


def profile_totals() -> dict:
    return dict(_TIMERS)


def reset_profile() -> None:
    # Keep each counter's type: count fields stay int, times stay float.
    for key, value in _TIMERS.items():
        _TIMERS[key] = 0 if isinstance(value, int) else 0.0
    with _POOL_STATE_LOCK:
        if _POOL_STATE["queued"] == 0 and _POOL_STATE["running"] == 0:
            _POOL_STATE["completed"] = 0


def set_profile(enabled: bool) -> None:
    """Enable or disable I/O timing for an already imported runtime."""
    global _PROFILE
    _PROFILE = bool(enabled)


def profile_enabled() -> bool:
    return bool(_PROFILE)


def set_score_sync_profile(enabled: bool) -> None:
    """Enable score-sync attribution without enabling other I/O timers."""
    global _SCORE_SYNC_PROFILE
    _SCORE_SYNC_PROFILE = bool(enabled)


def score_sync_profile_enabled() -> bool:
    return bool(_SCORE_SYNC_PROFILE or _PROFILE)


def score_sync_begin(threshold_active: bool):
    """Start one opt-in score-sync attribution span.

    A false threshold records an inactive path and returns no timing handle.
    """
    if not (_SCORE_SYNC_PROFILE or _PROFILE):
        return None
    if not threshold_active:
        _TIMERS["score_sync_threshold_inactive"] += 1
        return None
    _TIMERS["score_sync_threshold_active"] += 1
    return (
        time.perf_counter(),
        _physical_bytes_read(),
        read_pool_state() if _SCORE_SYNC_PROFILE else None,
    )


def score_sync_end(handle) -> None:
    """Finish one score-sync span and add its wall, byte, and pool totals."""
    if handle is None:
        return
    began, bytes_before, pool_before = handle
    ended = time.perf_counter()
    bytes_after = _physical_bytes_read()
    pool_after = read_pool_state() if pool_before is not None else None
    elapsed = ended - began
    physical = (
        bytes_after - bytes_before
        if bytes_before >= 0 and bytes_after >= bytes_before
        else 0
    )
    _TIMERS["score_sync"] += elapsed
    _TIMERS["score_sync_bytes"] += physical
    _TIMERS["score_sync_physical_bytes"] += physical
    _TIMERS["score_sync_calls"] += 1
    if pool_before is not None and pool_after is not None:
        for state in ("queued", "running", "completed"):
            _TIMERS[f"score_sync_pool_{state}_before"] += pool_before[state]
            _TIMERS[f"score_sync_pool_{state}_after"] += pool_after[state]


def score_sync_totals() -> dict:
    """Return score-sync attribution in stable, human-readable fields."""
    totals = profile_totals()
    active = totals["score_sync_threshold_active"]
    return {
        "wall_seconds": totals["score_sync"],
        "physical_bytes": totals["score_sync_physical_bytes"],
        "calls": totals["score_sync_calls"],
        "threshold_active_calls": active,
        "threshold_inactive_calls": totals["score_sync_threshold_inactive"],
        "threshold_path_active": bool(active),
        "pool": {
            state: {
                "before": totals[f"score_sync_pool_{state}_before"],
                "after": totals[f"score_sync_pool_{state}_after"],
            }
            for state in ("queued", "running", "completed")
        },
    }


class _ProfiledRead:
    __slots__ = (
        "value", "submitted", "started", "ended", "pread_intervals",
        "pread_calls", "pread_bytes",
    )

    def __init__(self, value, submitted, started, ended, stats):
        self.value = value
        self.submitted = submitted
        self.started = started
        self.ended = ended
        self.pread_intervals = stats["pread_intervals"]
        self.pread_calls = stats["pread_calls"]
        self.pread_bytes = stats["pread_bytes"]


def _profiled_read_call(submitted, track_pool, function, *args):
    started = time.perf_counter()
    if track_pool:
        _pool_started()
    begin_read_profile()
    try:
        value = function(*args)
    finally:
        stats = finish_read_profile()
        if track_pool:
            _pool_finished()
    ended = time.perf_counter()
    return _ProfiledRead(value, submitted, started, ended, stats)


def _resolve_future(future, timings):
    value = future.result()
    if isinstance(value, _ProfiledRead):
        if timings is not None:
            timings.append(value)
        return value.value
    return value


def _overlap(start, end, lower, upper):
    return max(0.0, min(end, upper) - max(start, lower))


def _record_read_timing(timings, wait_started, wait_ended) -> None:
    if not timings:
        return
    for timing in timings:
        _TIMERS["read_tasks"] += 1
        _TIMERS["pread_calls"] += timing.pread_calls
        _TIMERS["pread_bytes"] += timing.pread_bytes
        _TIMERS["queue_delay_sum"] += max(
            0.0, timing.started - timing.submitted
        )
        _TIMERS["task_service_sum"] += max(0.0, timing.ended - timing.started)
        _TIMERS["pread_service_sum"] += sum(
            max(0.0, ended - started)
            for started, ended in timing.pread_intervals
        )

    critical = max(timings, key=lambda timing: timing.ended)
    queue = _overlap(
        wait_started, critical.started, wait_started, wait_ended
    )
    service = _overlap(
        critical.started, critical.ended, wait_started, wait_ended
    )
    pread = sum(
        _overlap(started, ended, wait_started, wait_ended)
        for started, ended in critical.pread_intervals
    )
    wait = max(0.0, wait_ended - wait_started)
    _TIMERS["layer_completion_sum"] += wait
    _TIMERS["layer_completion_count"] += 1
    _TIMERS["critical_queue"] += queue
    _TIMERS["critical_pread"] += pread
    _TIMERS["critical_task_overhead"] += max(0.0, service - pread)
    _TIMERS["completion_overhead"] += max(0.0, wait - queue - service)


def set_metal_runtime(enabled: bool) -> None:
    os.environ["FLASHNEXT_METAL_RUNTIME"] = "1" if enabled else "0"


def metal_runtime() -> bool:
    return os.environ.get("FLASHNEXT_METAL_RUNTIME") == "1"


def set_metal_g64(enabled: bool) -> None:
    """Set the default Q4/G64 custom Metal executor variant."""
    os.environ["FLASHNEXT_METAL_G64"] = "1" if enabled else "0"


def metal_g64() -> bool:
    return os.environ.get("FLASHNEXT_METAL_G64", "1") == "1"

# Rows per read. A gather's throughput collapses once its output buffer gets
# large: measured at 16 workers, 1027 MB/s for 10 rows, 1205 MB/s for 96, and
# 484 MB/s for 290 (a 237 MB buffer). Decode routes 10 and is unaffected;
# prefill routes hundreds and is split into chunks. Applies to shared_mmap.
_CHUNK = int(os.environ.get("FLASHNEXT_CHUNK", 96))

# Read every chunk straight into one destination per part, so the main thread
# never concatenates the pieces. At FLASHNEXT_PREAD_CHUNK=1 the old path gave
# every expert its own allocation and then copied the whole layer again, which
# cost 35.9 ms per token. With chunk 2 each worker writes one contiguous run and
# most of the NVMe queue depth survives:
#
#   12 arms, clean boot, 40 tokens
#   concat chunk 1   2.67 gen median   467.7 MB/token
#   buffer chunk 2   2.83 gen median   457.7 MB/token
#   +6.3% gen median against a 4.4% band, ahead in 10 of 12 pairs
#
# Token IDs are identical: the same bytes land in a different destination
# layout. The pair was measured on the pread family only. `fast` and
# `fast-quality` read through `shared_mmap`, where none of that evidence
# applies, so they keep separate chunks unless the switch forces it.
#
# A list so a benchmark can flip it on a live backend.
_PREAD_MODES = ("pread", "preadv")
_SHARED_BUFFER = [os.environ.get("FLASHNEXT_SHARED_READ_BUFFER")]


def shared_buffer(mode: str = "pread") -> bool:
    """Whether this read mode fills one destination per part."""
    forced = _SHARED_BUFFER[0]
    if forced is not None:
        return forced != "0"
    return mode in _PREAD_MODES


def set_shared_buffer(enabled) -> None:
    """Force the switch, or pass None to return to the per-mode default."""
    _SHARED_BUFFER[0] = None if enabled is None else ("1" if enabled else "0")


def stream_pack_enabled() -> bool:
    """Return whether cold experts use one expert-major destination buffer."""
    return os.environ.get("FLASHNEXT_STREAM_PACK", "0") == "1"


class _StreamedPackRead:
    """One expert-major buffer filled by nine coalesced worker tasks."""

    __slots__ = ("buffer", "futures")

    def __init__(self, buffer, futures):
        self.buffer = buffer
        self.futures = futures

    def wait(self, timings=None):
        # Stream-pack layers wait here, not in `_await_projection_tasks`, so
        # keep-warm has to cover this wait as well. Without it the packed
        # layers idled the GPU during reads and its clock fell (mean state
        # 13.1 to 13.8 against 15.0, bundle-slab run of 2026-09-23).
        if _KEEPWARM[0] and _KEEPWARM_STREAM_PACK:
            _keep_gpu_warm_until(self.futures)
        for future in self.futures:
            _resolve_future(future, timings)
        return self

    def to_mx(self):
        return mx.from_dlpack(self.buffer, copy=False)


class _SharedRead:
    """A destination plus the futures filling its disjoint slices."""

    __slots__ = ("buffer", "futures")

    def __init__(self, buffer, futures):
        self.buffer = buffer
        self.futures = futures

    def wait(self, timings=None):
        for future in self.futures:
            _resolve_future(future, timings)
        return self


def submit_projection_tasks(projections, experts):
    """Queue one layer's reads for gate, up and down."""
    return [projection.cache.submit(experts) for projection in projections]


class ExpertLRU:
    """Per-projection reader for routed expert rows.

    The name is historical: nothing is cached here. All routed rows of a
    layer are submitted as one read and converted once.
    """

    __slots__ = ("store", "prefix")

    def __init__(self, store: SafeTensorStore, prefix: str):
        self.store = store
        self.prefix = prefix

    def submit(self, experts: List[int]):
        """Queue reads so a whole layer flies at once, chunked to stay fast."""
        mode = self.store._read_mode
        if shared_buffer(mode):
            return self._submit_shared(experts, mode)
        chunk = self.store._pread_chunk if mode in _PREAD_MODES else _CHUNK
        return [
            [
                _submit_read(
                    self.store.rows_np, f"{self.prefix}.{part}",
                    experts[start : start + chunk], mode,
                )
                for start in range(0, len(experts), chunk)
            ]
            for part in _PARTS
        ]

    def _submit_shared(self, experts: List[int], mode: str):
        """One destination per part; each chunk writes its own slice."""
        chunk = self.store._pread_chunk if mode in _PREAD_MODES else _CHUNK
        pending = []
        for part in _PARTS:
            name = f"{self.prefix}.{part}"
            buffer = self.store.empty_rows(name, len(experts))
            futures = [
                _submit_read(
                    self.store.rows_into, name, experts[start : start + chunk],
                    buffer[start : start + chunk], mode,
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

    def fetch(self, experts: List[int]):
        """Read routed rows synchronously for a standalone projection call."""
        return tuple(
            self.store.to_mx(
                f"{self.prefix}.{part}",
                self.store.rows_np(f"{self.prefix}.{part}", experts),
            )
            for part in _PARTS
        )


def _await_read(pending, timings=None):
    """Resolve one projection's pending reads into what `to_mx` expects."""
    return [
        item.wait(timings) if isinstance(item, _SharedRead)
        else [_resolve_future(future, timings) for future in item]
        for item in pending
    ]


# GPU keep-warm. The GPU governor drops to its lowest performance states when
# decode leaves the GPU idle during expert reads: at two or more cold experts
# per layer the miss sweep measured 96% of active time in P15 fall to 0%, and
# every command buffer ran about twice as long. With this on, the main thread
# keeps short ALU-only spin kernels queued on a separate GPU stream while it
# waits for reads, so the GPU never looks idle to the governor. The spins touch
# no memory and change no model value.
_KEEPWARM = [os.environ.get("FLASHNEXT_GPU_KEEPWARM", "0") == "1"]
_KEEPWARM_ITERS = [int(os.environ.get("FLASHNEXT_GPU_KEEPWARM_ITERS", "60000"))]
_KEEPWARM_PERIOD = [float(os.environ.get("FLASHNEXT_GPU_KEEPWARM_PERIOD_MS", "0.5")) / 1000]
_KEEPWARM_STATE: dict = {}
# Rollback for the stream-pack keep-warm fix of 2026-09-23. Read once at
# import so the decode path pays one branch.
_KEEPWARM_STREAM_PACK = os.environ.get("FLASHNEXT_GPU_KEEPWARM_STREAM_PACK", "1") == "1"
_KEEPWARM_SOURCE = """
    uint lane = thread_position_in_grid.x;
    float value = float(lane);
    int loops = iterations[0];
    for (int i = 0; i < loops; ++i) {
        value = fma(value, 0.5f, 0.25f);
    }
    out[lane] = value;
"""


def set_gpu_keepwarm(enabled: bool) -> None:
    _KEEPWARM[0] = bool(enabled)


def gpu_keepwarm() -> bool:
    return _KEEPWARM[0]


def _keepwarm_spin():
    state = _KEEPWARM_STATE
    if not state:
        state["kernel"] = mx.fast.metal_kernel(
            name="flashnext_gpu_keepwarm",
            input_names=["iterations"],
            output_names=["out"],
            source=_KEEPWARM_SOURCE,
        )
        state["stream"] = mx.new_stream(mx.gpu)
        state["iterations"] = mx.array([_KEEPWARM_ITERS[0]], dtype=mx.int32)
    result = state["kernel"](
        inputs=[state["iterations"]],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(32,)], output_dtypes=[mx.float32],
        stream=state["stream"],
    )[0]
    mx.async_eval(result)
    return result


def _pending_futures(pending) -> list:
    futures = []
    for projection in pending:
        for item in projection:
            if isinstance(item, _SharedRead):
                futures.extend(item.futures)
            elif isinstance(item, list):
                futures.extend(item)
            else:
                futures.append(item)
    return futures


class _Latch:
    """Set an event once every read of a layer has completed.

    One event per layer replaces polling the whole future list with
    ``concurrent.futures.wait`` every period, which built a waiter over
    every pending future on each pass.
    """

    __slots__ = ("remaining", "lock", "event")

    def __init__(self, count: int):
        self.remaining = count
        self.lock = threading.Lock()
        self.event = threading.Event()

    def count_down(self, _future) -> None:
        with self.lock:
            self.remaining -= 1
            finished = self.remaining == 0
        if finished:
            self.event.set()


def _keep_gpu_warm_until_done(pending) -> None:
    """Submit one short spin per period until every read of this layer is done.

    A spin lasts about one period at the top clock, longer at a low clock, so
    only a few spins can queue past the end of a wait. They run on their own
    stream, one threadgroup wide, beside the layer's real work.
    """
    _keep_gpu_warm_until(_pending_futures(pending))


def _keep_gpu_warm_until(futures) -> None:
    """Spin until every future in ``futures`` has completed."""
    futures = [future for future in futures if not future.done()]
    if not futures:
        return
    latch = _Latch(len(futures))
    for future in futures:
        future.add_done_callback(latch.count_down)
    period = _KEEPWARM_PERIOD[0]
    while not latch.event.is_set():
        _keepwarm_spin()
        latch.event.wait(period)


def _await_projection_tasks(pending, timings=None):
    """Resolve the gate, up and down reads of one layer."""
    if _KEEPWARM[0]:
        _keep_gpu_warm_until_done(pending)
    return [_await_read(futures, timings) for futures in pending]


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
    ):
        super().__init__()
        self.cache = ExpertLRU(store, prefix)
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


def _compatible_pin_profile(store=None, expected_group_size=None, layer_ids=None):
    """Return pin history only when it belongs to this store's layout."""
    data = _load_pin_profile_or_none(_slab_profile_path(store, expected_group_size))
    if data is None or store is None:
        return data
    from .routing import pin_profile_compatible

    if layer_ids is None and expected_group_size is not None:
        # Global slab allocation consumes only the layers represented by the
        # saved ranking.  A mixed export is safe when those selected layers
        # match, even if unrelated layers use another group size.
        ranked = data.get("ranked_counts") or data.get("ranked_scores") or data.get("layers")
        if isinstance(ranked, dict):
            layer_ids = tuple(ranked)

    compatible, _reason = pin_profile_compatible(
        store, data, expected_group_size=expected_group_size,
        layer_ids=layer_ids,
    )
    return data if compatible else None


def _pin_profile_cache_identity(store):
    if store is None:
        return None
    from .routing import checkpoint_identity_for_store

    return checkpoint_identity_for_store(store)


def _pin_profile_path() -> str:
    return os.path.expanduser(
        os.environ.get("FLASHNEXT_PIN_CACHE", "~/.cache/flashnext/pins.json")
    )


def _load_pin_profile(pin_file: str | None = None) -> dict | None:
    """Load the pin profile, distinguishing absence from corruption."""
    pin_file = pin_file or _pin_profile_path()
    if not os.path.isfile(pin_file):
        return None
    try:
        with open(pin_file, "r") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid FlashNext pin profile {pin_file}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"invalid FlashNext pin profile {pin_file}: expected an object")
    normalized = dict(data)
    for field in ("layers", "ranked_scores", "ranked_counts", "cumulative_counts"):
        value = data.get(field, {})
        if not isinstance(value, dict):
            raise ValueError(
                f"invalid FlashNext pin profile {pin_file}: {field} must be an object"
            )
        normalized[field] = value
    for field in ("layers", "ranked_scores", "ranked_counts", "cumulative_counts"):
        clean = {}
        for layer, values in normalized[field].items():
            try:
                layer_id = int(layer)
                if layer_id < 0 or str(layer_id) in clean:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid FlashNext pin profile {pin_file}: {field}/{layer}"
                ) from error
            clean[str(layer_id)] = values
        normalized[field] = clean
    for layer, values in normalized["layers"].items():
        if not isinstance(values, list):
            raise ValueError(f"invalid FlashNext pin profile {pin_file}: layers/{layer}")
        seen = set()
        for expert in values:
            try:
                expert_id = int(expert)
                if expert_id < 0 or expert_id in seen:
                    raise ValueError
                seen.add(expert_id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid FlashNext pin profile {pin_file}: layers/{layer}"
                ) from error
    for field in ("ranked_scores", "ranked_counts", "cumulative_counts"):
        for layer, values in normalized[field].items():
            if not isinstance(values, list):
                raise ValueError(f"invalid FlashNext pin profile {pin_file}: {field}/{layer}")
            seen = set()
            clean = []
            for pair in values:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        f"invalid FlashNext pin profile {pin_file}: {field}/{layer}"
                    )
                try:
                    expert_id = int(pair[0])
                    score = float(pair[1])
                    if expert_id < 0 or expert_id in seen or not math.isfinite(score):
                        raise ValueError
                    seen.add(expert_id)
                    clean.append((expert_id, score))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid FlashNext pin profile {pin_file}: {field}/{layer}"
                    ) from error
            normalized[field][layer] = clean
    return normalized


_WARNED_PROFILES: set = set()


def _load_pin_profile_or_none(pin_file: str | None = None) -> dict | None:
    """Load a pin profile for the runtime, treating a corrupt file as absent.

    The profile only chooses slab contents. A damaged file must fall back to
    streaming rather than stop the model from loading; diagnostics that need
    the error call ``_load_pin_profile`` directly.
    """
    try:
        return _load_pin_profile(pin_file)
    except (RuntimeError, ValueError) as error:
        message = str(error)
        if message not in _WARNED_PROFILES:
            _WARNED_PROFILES.add(message)
            print(f"flashnext: ignoring pin profile: {message}", file=sys.stderr)
        return None


def _slab_profile_path(store=None, expected_group_size=None) -> str:
    """Snapshot compatible live history once for a checkpoint's slab."""
    live = _pin_profile_path()
    if os.environ.get("FLASHNEXT_SLAB_PROFILE") != "frozen" or store is None:
        return live
    identity = _pin_profile_cache_identity(store)
    if not identity:
        return live
    directory = os.path.dirname(live) or "."
    frozen = os.path.join(directory, f"slab-frozen-{identity}.json")
    if os.path.exists(frozen):
        _mark_frozen_used(frozen)
        return frozen
    profile = _load_pin_profile_or_none(live)
    if profile is None:
        return frozen
    from .routing import pin_profile_compatible

    compatible, _reason = pin_profile_compatible(
        store, profile, expected_group_size=expected_group_size,
    )
    if not compatible:
        return frozen
    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, prefix=".slab-frozen-",
    ) as handle:
        json.dump(profile, handle)
        handle.flush()
        try:
            os.link(handle.name, frozen)
        except FileExistsError:
            pass
    _mark_frozen_used(frozen)
    return frozen


_FROZEN_MARKED: set = set()


def _mark_frozen_used(path: str) -> None:
    """Touch a frozen snapshot once per process and prune stale ones.

    The checkpoint identity includes shard metadata, so an identity change
    leaves the old snapshot behind. Age-based pruning matches the slab packs.
    """
    if path in _FROZEN_MARKED:
        return
    _FROZEN_MARKED.add(path)
    from pathlib import Path

    from .slab_pack import mark_used_and_prune

    mark_used_and_prune(Path(path), "slab-frozen-*.json")


def _g64_pin_profile_compatible(store, profile: dict | None) -> tuple[bool, str | None]:
    """Require checkpoint-specific, Q4/G64 history before packed selection.

    Older pin files contain expert IDs only. Those IDs can describe a different
    checkpoint and must not select residents for a remapped G64 layout.
    """
    if profile is None:
        return False, "Q4/G64 slab pack needs checkpoint-specific pin history"
    quantization = profile.get("quantization") or {}
    if not isinstance(quantization, dict):
        return False, "pin history has invalid Q4/G64 provenance"
    group_size = quantization.get(
        "group_size", profile.get("group_size", profile.get("quantization_group_size"))
    )
    try:
        if int(group_size) != 64:
            return False, "pin history does not declare Q4/G64"
    except (TypeError, ValueError):
        return False, "pin history has no Q4/G64 provenance"
    recorded = profile.get("checkpoint_identity", profile.get("model_identity"))
    if not recorded:
        return False, "pin history has no checkpoint identity"
    try:
        from .routing import pin_profile_compatible

        compatible, reason = pin_profile_compatible(
            store, profile, expected_group_size=64
        )
    except (ImportError, OSError, TypeError, ValueError):
        compatible, reason = False, "cannot verify Q4/G64 checkpoint identity"
    if not compatible:
        if reason == "pin history has no checkpoint identity":
            return False, reason
        if reason == "pin history belongs to another checkpoint":
            return False, reason
        if reason == "pin history has incompatible quantization":
            return False, "pin history does not declare Q4/G64"
        return False, reason or "cannot verify Q4/G64 checkpoint identity"
    return True, None


def _pin_profile_signature(store=None, expected_group_size=None) -> tuple:
    """Return a cheap cache key that changes when the pin profile changes."""
    path = _slab_profile_path(store, expected_group_size)
    try:
        stat = os.stat(path)
    except OSError:
        return (path, None)
    return (
        path, stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns,
    )


_GLOBAL_SLAB_CACHE: Dict[Any, Dict[int, List[int]]] = {}


def get_physical_miss_slab_allocation(
    total_slots: int,
    min_slots: int = 4,
    max_slots: int = 6,
    num_layers: int = 12,
    store=None,
    expected_group_size: int | None = None,
) -> Dict[int, List[int]]:
    """Read evidence for the guarded physical-miss hybrid probe."""
    profile_path = os.environ.get(
        "FLASHNEXT_PHYSICAL_MISS_PROFILE", "~/.cache/flashnext/physical-misses.json"
    )
    try:
        profile = load_profile(profile_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "physical-miss slab policy needs a valid measured profile at "
            f"{os.path.expanduser(profile_path)}: {error}"
        ) from error
    # Import lazily to keep the module's existing model boundary unchanged.
    canonical = get_skew_slab_allocation(
        total_slots,
        min_slots=min_slots,
        max_slots=max_slots,
        num_layers=num_layers,
        store=store,
        expected_group_size=expected_group_size,
    )
    if not canonical:
        raise RuntimeError("physical-miss hybrid needs a canonical skew allocation")
    allocation = allocate_physical_miss_hybrid_slots(
        profile,
        canonical,
        total_slots,
        min_slots=min_slots,
        max_slots=max_slots,
        num_layers=num_layers,
        min_samples=int(os.environ.get("FLASHNEXT_PHYSICAL_MISS_MIN_SAMPLES", "1")),
        require_core_calibration=True,
    )
    return allocation


def get_global_slab_allocation(
    total_slots: int,
    min_slots: int = 1,
    store=None,
    expected_group_size: int | None = None,
) -> Dict[int, List[int]]:
    """Allocate a global slot budget across layers.

    If min_slots <= 1, distributes slots purely by descending individual candidate scores.
    If min_slots > 1 (e.g. 4), concentrates the budget into the highest-utility layers with
    at least min_slots each, avoiding layer I/O bursts.
    """
    if total_slots <= 0:
        return {}
    min_slots = int(os.environ.get("FLASHNEXT_SLAB_MIN_SLOTS", str(min_slots)))
    cache_key = (
        "global", _pin_profile_signature(store, expected_group_size), total_slots, min_slots,
        expected_group_size, _pin_profile_cache_identity(store),
    )
    if cache_key in _GLOBAL_SLAB_CACHE:
        return _GLOBAL_SLAB_CACHE[cache_key]
    data = _compatible_pin_profile(store, expected_group_size)
    if data is None:
        return {}
    ranked = data["ranked_scores"]
    if ranked:
        if min_slots <= 1:
            all_candidates = []
            for l_str, pairs in ranked.items():
                l_idx = int(l_str)
                for exp, score in pairs:
                    all_candidates.append((float(score), l_idx, int(exp)))
            if all_candidates:
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                allocation: Dict[int, List[int]] = {}
                for _, l_idx, exp in all_candidates[:total_slots]:
                    allocation.setdefault(l_idx, []).append(exp)
                _GLOBAL_SLAB_CACHE[cache_key] = allocation
                return allocation

        # Score each layer by the sum of its top min_slots experts
        layer_scores = []
        for l_str, pairs in ranked.items():
            l_idx = int(l_str)
            s = sum(float(score) for _, score in pairs[:min_slots])
            layer_scores.append((s, l_idx))
        layer_scores.sort(reverse=True)

        k = max(1, total_slots // max(1, min_slots))
        selected = [l_idx for _, l_idx in layer_scores[:k]]

        allocation: Dict[int, List[int]] = {}
        for l_idx in selected:
            pairs = ranked.get(str(l_idx), [])
            allocation[l_idx] = [int(exp) for exp, _ in pairs[:min_slots]]

        # Remainder slots distributed greedily by next highest score
        rem = total_slots - sum(len(v) for v in allocation.values())
        for _ in range(rem):
            best_l, best_s, best_e = -1, -1.0, -1
            for l_idx in selected:
                curr_cnt = len(allocation[l_idx])
                pairs = ranked.get(str(l_idx), [])
                if curr_cnt < len(pairs):
                    cand_e, cand_s = pairs[curr_cnt]
                    if float(cand_s) > best_s:
                        best_s = float(cand_s)
                        best_l = l_idx
                        best_e = int(cand_e)
            if best_l >= 0:
                allocation[best_l].append(best_e)

        _GLOBAL_SLAB_CACHE[cache_key] = allocation
        return allocation
    # Fallback to layers if ranked_scores is empty
    layers = data["layers"]
    allocation = {}
    slots_left = total_slots
    for l_str, experts in layers.items():
        if slots_left <= 0:
            break
        l_idx = int(l_str)
        take = [int(e) for e in experts[:slots_left]]
        if take:
            allocation[l_idx] = take
            slots_left -= len(take)
    _GLOBAL_SLAB_CACHE[cache_key] = allocation
    return allocation


def get_skew_slab_allocation(
    total_slots: int,
    min_slots: int = 4,
    max_slots: int = 6,
    num_layers: int = 12,
    store=None,
    expected_group_size: int | None = None,
) -> Dict[int, List[int]]:
    """Allocate a global slot budget across layers with skew awareness.

    Pins a base topology of `num_layers` (default: 12) with `min_slots` (default: 4) each,
    then greedily distributes remaining slots to the layers with highest marginal hit
    utility, capped at `max_slots` per layer.
    """
    if total_slots <= 0:
        return {}
    min_slots = int(os.environ.get("FLASHNEXT_SLAB_MIN_SLOTS", str(min_slots)))
    max_slots = int(os.environ.get("FLASHNEXT_SLAB_MAX_SLOTS", str(max_slots)))
    num_layers = int(os.environ.get("FLASHNEXT_SLAB_NUM_LAYERS", str(num_layers)))
    cache_key = (
        "skew", _pin_profile_signature(store, expected_group_size), total_slots, min_slots, max_slots,
        num_layers, expected_group_size, _pin_profile_cache_identity(store),
        os.environ.get("FLASHNEXT_SLAB_COUNTS", "turn"),
    )
    if cache_key in _GLOBAL_SLAB_CACHE:
        return _GLOBAL_SLAB_CACHE[cache_key]

    data = _compatible_pin_profile(store, expected_group_size)
    if data is None:
        return {}
    # Prefer ranked_counts if present, fallback to ranked_scores. The opt-in
    # cumulative history replaces both when it exists.
    from .routing import slab_counts_mode

    ranked = (
        data["cumulative_counts"] if slab_counts_mode() == "cumulative" else {}
    ) or data["ranked_counts"] or data["ranked_scores"]
    if not ranked:
        return get_global_slab_allocation(
            total_slots,
            min_slots=min_slots,
            store=store,
            expected_group_size=expected_group_size,
        )

    # Score each layer by the sum of its top min_slots candidates
    layer_scores = []
    for l_str, pairs in ranked.items():
        l_idx = int(l_str)
        s = sum(float(score) for _, score in pairs[:min_slots])
        layer_scores.append((s, l_idx))
    layer_scores.sort(reverse=True)

    k = min(num_layers, max(1, total_slots // max(1, min_slots)))
    selected = [l_idx for _, l_idx in layer_scores[:k]]

    allocation: Dict[int, List[int]] = {}
    for l_idx in selected:
        pairs = ranked.get(str(l_idx), [])
        allocation[l_idx] = [int(exp) for exp, _ in pairs[:min_slots]]

    # Distribute remaining slots greedily to highest marginal candidate
    rem = total_slots - sum(len(v) for v in allocation.values())
    for _ in range(rem):
        best_l, best_s, best_e = -1, -1.0, -1
        for l_idx in selected:
            curr_cnt = len(allocation[l_idx])
            if curr_cnt >= max_slots:
                continue
            pairs = ranked.get(str(l_idx), [])
            if curr_cnt < len(pairs):
                cand_e, cand_s = pairs[curr_cnt]
                if float(cand_s) > best_s:
                    best_s = float(cand_s)
                    best_l = l_idx
                    best_e = int(cand_e)
        if best_l >= 0:
            allocation[best_l].append(best_e)

    _GLOBAL_SLAB_CACHE[cache_key] = allocation
    return allocation


def _slab_allocation(store, budget: int, group_size: int) -> Dict[int, List[int]]:
    """The global slab allocation for the configured policy."""
    min_slots = int(os.environ.get("FLASHNEXT_SLAB_MIN_SLOTS", 4))
    policy = os.environ.get("FLASHNEXT_SLAB_POLICY", "skew")
    if policy == "uniform":
        return get_global_slab_allocation(
            budget, min_slots=min_slots, store=store,
            expected_group_size=group_size,
        )
    if policy == "physical-miss-hybrid":
        allocation = get_physical_miss_slab_allocation(
            budget, min_slots=min_slots, store=store,
            expected_group_size=group_size,
        )
        store._slab_alloc_provenance = last_hybrid_summary()
        return allocation
    if policy == "physical-miss":
        raise ValueError(
            "physical-miss is historical and unavailable; "
            "use physical-miss-hybrid for the guarded probe"
        )
    return get_skew_slab_allocation(
        budget, min_slots=min_slots, store=store,
        expected_group_size=group_size,
    )


# Layer id of the MTP expert block. It is not a backbone layer, so it must not
# alias one: slab allocation, packed slots and route traces key on layer ids.
MTP_LAYER_ID = -1


class StreamingSwitchGLU(nn.Module):
    """SwitchGLU whose three projections stream from the checkpoint.

    Decode rows on an eligible layer run through the custom Metal executor,
    which reads packed slab experts in place and streamed experts from fresh
    buffers. Prefill, layer 0, the MTP block and other layouts run MLX
    ``gather_qmm`` on the streamed rows.
    """

    def __init__(
        self,
        store: SafeTensorStore,
        prefix: str,
        group_size: int,
        bits: int,
        mode: str,
        activation,
        layer_id: int = -1,
        next_prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.next_prefix = next_prefix
        g64 = group_size == 64
        slab_g64 = os.environ.get("FLASHNEXT_SLAB_G64") == "1"
        if g64 and metal_g64():
            from .metal_runtime import G64_RUNTIME_READY

            if not G64_RUNTIME_READY:
                raise RuntimeError(
                    "Q4/G64 custom runtime is not ready; full-model digest gate failed"
                )
        if g64 and slab_g64 and stream_pack_enabled():
            raise ValueError(
                "Q4/G64 slab streaming packs are unavailable until their layout is supported"
            )
        self.gate_proj = StreamingSwitchLinear(store, f"{prefix}.gate_proj", group_size, bits, mode)
        self.up_proj = StreamingSwitchLinear(store, f"{prefix}.up_proj", group_size, bits, mode)
        self.down_proj = StreamingSwitchLinear(store, f"{prefix}.down_proj", group_size, bits, mode)
        self.activation = activation
        self.hits = 0
        self.misses = 0
        object.__setattr__(self, "_routed_host", None)
        self._metal_executors = {}
        self._metal_runtime_capable = self._compute_metal_runtime_capable()
        self._slab_pack_capable = self._metal_runtime_capable and validate_slab_allocation(
            store, {layer_id: ()}, layout=group_size
        )
        self.slab_pack = None
        self.slab_expert_to_slot = {}
        self._slab_pack_disabled_reason = None
        self._attach_slab_pack(store, group_size, slab_g64)
        # Research sidecar with each expert's scale and bias rows in one
        # record (FLASHNEXT_SMALL_SIDECAR, off). Stream-pack path only.
        from .small_sidecar import for_store as _small_sidecar_for_store

        sidecar = _small_sidecar_for_store(store)
        self._small_sidecar = (
            sidecar if sidecar is not None and layer_id in sidecar.layers else None
        )

    def _attach_slab_pack(self, store, group_size: int, slab_g64: bool) -> None:
        """Map this layer's packed experts, when a pack is configured."""
        budget = int(os.environ.get("FLASHNEXT_SLAB_GLOBAL", 0))
        if os.environ.get("FLASHNEXT_SLAB_PACK") != "1" or budget <= 0:
            return
        if group_size == 64:
            if not slab_g64:
                self._slab_pack_disabled_reason = (
                    "Q4/G64 slab pack requires FLASHNEXT_SLAB_G64=1"
                )
                return
            history_ok, reason = _g64_pin_profile_compatible(
                store, _load_pin_profile_or_none(_slab_profile_path(store, 64))
            )
            if not history_ok:
                self._slab_pack_disabled_reason = reason
                return
        if not self._slab_pack_capable:
            return
        alloc = getattr(store, "_slab_alloc", None)
        if alloc is None:
            alloc = _slab_allocation(store, budget, group_size)
            store._slab_alloc = alloc
        # Missing history is expected on first launch. Keep reference
        # streaming active until a valid allocation becomes available.
        if not alloc:
            return
        if not validate_slab_allocation(store, alloc, layout=group_size):
            store._slab_pack_disabled = True
            self._slab_pack_disabled_reason = "slab allocation does not match checkpoint layout"
            return
        from .slab_pack import get_or_create_slab_pack

        pack = getattr(store, "_slab_pack", None)
        if pack is None:
            pack = get_or_create_slab_pack(store, alloc, layout=group_size)
            store._slab_pack = pack
        self.slab_pack = pack
        self.slab_expert_to_slot = {
            expert: pack.layer_expert_to_slot[(self.layer_id, expert)]
            for expert in alloc.get(self.layer_id, [])
            if (self.layer_id, expert) in pack.layer_expert_to_slot
        }

    @property
    def metal_combines_scores(self) -> bool:
        """Report the custom decode path to the patched MoE block."""
        return (
            os.environ.get("FLASHNEXT_METAL_RUNTIME") == "1"
            # Layer 0 stays on the reference path (historical guard). The
            # MTP block uses MTP_LAYER_ID and stays there too.
            and self.layer_id > 0
            and self._metal_runtime_capable
        )

    @property
    def metal_runtime_capable(self) -> bool:
        """Whether this layer matches the custom executor's Q4 contract."""
        return self._metal_runtime_capable

    def _compute_metal_runtime_capable(self) -> bool:
        """Check the generic custom executor contract once during setup."""
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        group_sizes = {projection.group_size for projection in projections}
        if len(group_sizes) != 1:
            return False
        supported_group = next(iter(group_sizes))
        if supported_group == 64 and not metal_g64():
            return False
        if supported_group not in (32, 64):
            return False
        if any(
            projection.bits != 4 or projection.mode != "affine"
            for projection in projections
        ):
            return False
        gate = self.gate_proj
        up = self.up_proj
        down = self.down_proj
        if gate.num_experts != up.num_experts or gate.num_experts != down.num_experts:
            return False
        try:
            gate_shape = gate.cache.store.shape(f"{gate.cache.prefix}.weight")
            up_shape = up.cache.store.shape(f"{up.cache.prefix}.weight")
            down_shape = down.cache.store.shape(f"{down.cache.prefix}.weight")
            hidden = gate_shape[2] * 8
            inter = gate.output_dims
            down_input = down_shape[2] * 8
        except (IndexError, KeyError, TypeError):
            return False
        if (
            len(gate_shape) != 3
            or len(up_shape) != 3
            or len(down_shape) != 3
            or up_shape[2] * 8 != hidden
            or inter != up.output_dims
            or down_input != inter
            or down.output_dims != hidden
        ):
            return False
        store = gate.cache.store
        if getattr(store, "refs", None):
            for projection, shape in (
                (gate, gate_shape), (up, up_shape), (down, down_shape)
            ):
                metadata_shape = (
                    shape[0], shape[1], shape[2] * 8 // supported_group
                )
                for part in ("scales", "biases"):
                    try:
                        if tuple(store.shape(f"{projection.cache.prefix}.{part}")) != metadata_shape:
                            return False
                    except (IndexError, KeyError, TypeError):
                        return False
        return hidden % supported_group == 0 and inter % supported_group == 0

    def _submit_stream_pack(self, wanted) -> _StreamedPackRead:
        """Read all cold components into one slab-compatible destination."""
        store = self.gate_proj.cache.store
        mode = store._read_mode
        buffer = np.empty(len(wanted) * RECORD_STRIDE, dtype=np.uint8)
        switch_prefix = self.gate_proj.cache.prefix.rsplit(".", 1)[0]
        configured_chunk = int(os.environ.get("FLASHNEXT_STREAM_PACK_CHUNK", "0"))
        chunk = configured_chunk if configured_chunk > 0 else len(wanted)
        futures = []
        sidecar = self._small_sidecar
        parts = _STREAM_RECORD_PARTS
        if sidecar is not None:
            parts = tuple(entry for entry in parts if entry[1] == "weight")
            for start in range(0, len(wanted), chunk):
                futures.append(_submit_read(
                    sidecar.read_into, self.layer_id,
                    wanted[start : start + chunk], buffer, start,
                ))
        for projection, part, offset in parts:
            name = f"{switch_prefix}.{projection}.{part}"
            for start in range(0, len(wanted), chunk):
                piece = wanted[start : start + chunk]
                destination = store.expert_record_view(
                    name,
                    buffer,
                    len(piece),
                    offset + start * RECORD_STRIDE,
                    RECORD_STRIDE,
                )
                futures.append(
                    _submit_read(store.rows_into, name, piece, destination, mode)
                )
        return _StreamedPackRead(buffer, futures)

    def _get_dummy_streamed_weights(self, hidden_size: int):
        dummy = getattr(self, "_cached_dummy_weights", None)
        if dummy is None:
            group_size = self.gate_proj.group_size
            inter = self.gate_proj.output_dims
            gw = mx.zeros((1, inter, hidden_size // 8), dtype=mx.uint32)
            gs = mx.zeros((1, inter, hidden_size // group_size), dtype=mx.bfloat16)
            gb = mx.zeros((1, inter, hidden_size // group_size), dtype=mx.bfloat16)
            uw = mx.zeros((1, inter, hidden_size // 8), dtype=mx.uint32)
            us = mx.zeros((1, inter, hidden_size // group_size), dtype=mx.bfloat16)
            ub = mx.zeros((1, inter, hidden_size // group_size), dtype=mx.bfloat16)
            dw = mx.zeros((1, hidden_size, inter // 8), dtype=mx.uint32)
            ds = mx.zeros((1, hidden_size, inter // group_size), dtype=mx.bfloat16)
            db = mx.zeros((1, hidden_size, inter // group_size), dtype=mx.bfloat16)
            dummy = [
                (gw, gs, gb),
                (uw, us, ub),
                (dw, ds, db),
            ]
            self._cached_dummy_weights = dummy
        return dummy

    def _executor(self, key, expert_count: int, hidden_size: int, slots: int):
        """The Metal executor for one route shape, built on first use."""
        executor = self._metal_executors.get(key)
        if executor is None:
            from .metal_runtime import MetalMoEExecutor

            executor = MetalMoEExecutor(
                expert_count, hidden_size, slots,
                group_size=self.gate_proj.group_size,
            )
            self._metal_executors[key] = executor
        return executor

    def _read_weights(self, projections, wanted):
        """Read ``wanted`` for every projection and wrap the rows for MLX."""
        pending = submit_projection_tasks(projections, wanted)
        if _PROFILE:
            timings = []
            began = time.perf_counter()
            with hostwindow.window("io_await"):
                raw = _await_projection_tasks(pending, timings)
            ended = time.perf_counter()
            _TIMERS["io_wait"] += ended - began
            _TIMERS["io_calls"] += 1
            _record_read_timing(timings, began, ended)
            began = time.perf_counter()
            with hostwindow.window("to_mx_host"):
                weights = [
                    projection.cache.to_mx(chunks)
                    for projection, chunks in zip(projections, raw)
                ]
            _TIMERS["to_mx"] += time.perf_counter() - began
            return weights
        with hostwindow.window("io_await"):
            raw = _await_projection_tasks(pending)
        with hostwindow.window("to_mx_host"):
            return [
                projection.cache.to_mx(chunks)
                for projection, chunks in zip(projections, raw)
            ]

    def __call__(
        self, x, indices, allow_sort=True, scores=None, shared_y=None,
        shared=None, shared_gate=None,
    ) -> mx.array:
        self._last_fused_shared = False
        flat_input = x.reshape(-1, x.shape[-1])
        x = mx.expand_dims(x, (-2, -3))
        handed = getattr(self, "_routed_host", None)
        routed = None
        if handed is not None:
            # Built by the MoE block from values it already synced.
            object.__setattr__(self, "_routed_host", None)
            if handed[1] is indices:
                routed = handed[0]
        if routed is None:
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
        if _ROUTE_TRACE[0] is not None:
            _ROUTE_TRACE[0](self.layer_id, routed)

        custom = (
            scores is not None
            and flat_input.shape[0] <= 8
            and self.metal_combines_scores
        )
        if custom and self.slab_expert_to_slot:
            return self._packed_pass(
                indices, routed, flat_input, scores, shared_y, shared, shared_gate,
            )
        return self._one_pass(
            x, indices, routed, allow_sort, flat_input,
            scores if custom else None, shared_y, shared, shared_gate,
        )

    def _packed_pass(
        self, indices, routed, flat_input, scores, shared_y, shared, shared_gate,
    ):
        """Custom Metal decode with packed slab hits addressed in place."""
        expert_to_slot = self.slab_expert_to_slot
        miss = [e for e in routed if e not in expert_to_slot]
        self.hits += len(routed) - len(miss)
        self.misses += len(miss)
        wanted = list(dict.fromkeys(miss))
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        hidden_size = flat_input.shape[-1]
        streamed_record = None
        if not wanted:
            weights = self._get_dummy_streamed_weights(hidden_size)
        elif stream_pack_enabled():
            pending = self._submit_stream_pack(wanted)
            timings = [] if _PROFILE else None
            began = time.perf_counter()
            pending.wait(timings)
            ended = time.perf_counter()
            if _PROFILE:
                _TIMERS["io_wait"] += ended - began
                _TIMERS["io_wait_packed"] += ended - began
                _TIMERS["io_calls"] += 1
                _TIMERS["io_calls_packed"] += 1
                _record_read_timing(timings, began, ended)
            began = time.perf_counter()
            streamed_record = pending.to_mx()
            weights = self._get_dummy_streamed_weights(hidden_size)
            if _PROFILE:
                _TIMERS["to_mx"] += time.perf_counter() - began
        else:
            weights = self._read_weights(projections, wanted)
        miss_order = {e: i for i, e in enumerate(wanted)}
        encoded_routes = [
            0x80000000 | expert_to_slot[e] if e in expert_to_slot else miss_order[e]
            for e in routed
        ]

        slots = indices.shape[-1]
        tokens = flat_input.shape[0]
        local = mx.array(encoded_routes, dtype=mx.uint32).reshape(tokens, slots)
        routed_scores = scores.reshape(tokens, slots)
        executor = self._executor(
            ("slab_pack", hidden_size, slots),
            max(len(expert_to_slot) + len(wanted), slots), hidden_size, slots,
        )
        output = executor.execute(
            flat_input, local,
            {"gate_proj": weights[0], "up_proj": weights[1], "down_proj": weights[2]},
            scores=routed_scores,
            slab_pack=self.slab_pack.buffer_mx,
            stream_pack=streamed_record,
            shared_y=shared_y,
            shared=shared,
            shared_gate=shared_gate,
        )
        if shared_y is not None or (shared is not None and shared_gate is not None):
            self._last_fused_shared = True
        output = output.reshape(*indices.shape[:-1], output.shape[-1])
        return output if self.gate_proj.group_size == 64 else output.astype(mx.bfloat16)

    def _one_pass(
        self, x, indices, routed, allow_sort, flat_input, scores, shared_y,
        shared, shared_gate,
    ):
        """Read every routed expert, then run the executor or gather_qmm.

        ``scores`` is set only when the custom Metal executor applies.
        """
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        with hostwindow.window("plan_host"):
            wanted = list(dict.fromkeys(routed))
            order = {e: i for i, e in enumerate(wanted)}
            local = mx.array(
                [order[e] for e in routed], dtype=mx.uint32
            ).reshape(indices.shape)
        weights = self._read_weights(projections, wanted)

        issue_began = time.perf_counter() if _PROFILE else 0.0
        with hostwindow.window("moe_issue_host"):
            if scores is None:
                return self._issue(
                    x, indices, projections, local, weights, allow_sort,
                    issue_began,
                )
            slots = indices.shape[-1]
            tokens = flat_input.shape[0]
            hidden_size = flat_input.shape[-1]
            # Adaptive top-k pads dropped slots with the first expert (score
            # 0), so a row can name fewer distinct experts than it has slots.
            expert_count = max(weights[0][0].shape[0], slots)
            executor = self._executor(
                (expert_count, hidden_size, slots), expert_count, hidden_size, slots,
            )
            output = executor.execute(
                flat_input, local.reshape(tokens, slots),
                {"gate_proj": weights[0], "up_proj": weights[1], "down_proj": weights[2]},
                scores=scores.reshape(tokens, slots),
                shared_y=shared_y,
                shared=shared,
                shared_gate=shared_gate,
            )
            if shared_y is not None or (shared is not None and shared_gate is not None):
                self._last_fused_shared = True
            if _PROFILE:
                _TIMERS["moe_issue"] += time.perf_counter() - issue_began
            return output.reshape(*indices.shape[:-1], output.shape[-1])

    def _issue(
        self, x, indices, projections, local, weights, allow_sort, issue_began,
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
        if _PROFILE:
            _TIMERS["moe_issue"] += time.perf_counter() - issue_began
        return o


if os.environ.get("FLASHNEXT_IO_QOS", "default") != "default":
    set_io_qos(os.environ["FLASHNEXT_IO_QOS"])
