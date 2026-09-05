#!/usr/bin/env python3
"""Rigorous benchmark for Issue #45: Metal barrier and fence cost under true unbuffered SSD DMA.

Methodological controls:
1. True unbuffered I/O using fcntl F_NOCACHE (48) on Darwin.
2. Real large model shard (>4 GB) with advancing offsets across arms to prevent storage cache hits.
3. Thread synchronization latch after a completed read, plus completion timing
   around the native GPU call. This records an observational interval only.
4. Physical byte verification using proc_pid_rusage (ri_diskio_bytesread).
5. Multi-arm interleaved rotation across scheduler strategies (serial, barrier, fence).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Iterable

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DEFAULT_STRATEGIES = ("serial", "barrier", "fence")
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    resolution_band_pct: float
    samples: int


def summarize(samples: Iterable[float]) -> TimingSummary:
    values = tuple(float(v) for v in samples)
    if not values:
        raise ValueError("samples cannot be empty")
    med = statistics.median(values)
    dev = max(abs(v - med) for v in values)
    band = 0.0 if med == 0 else 2.0 * dev / abs(med) * 100.0
    return TimingSummary(
        median_ms=round(med, 3),
        minimum_ms=round(min(values), 3),
        maximum_ms=round(max(values), 3),
        resolution_band_pct=round(band, 1),
        samples=len(values),
    )


def interleaved_order(values: tuple[str, ...], arms: int) -> list[str]:
    order = []
    n = len(values)
    for arm in range(arms):
        offset = arm % n
        order.extend(values[offset:] + values[:offset])
    return order


class SynchronizedDmaWorker:
    def __init__(self, fd: int, offset: int, target_bytes: int, chunk_bytes: int = 1024 * 1024):
        self.fd = fd
        self.offset = offset
        self.target_bytes = target_bytes
        self.chunk_bytes = chunk_bytes
        self.read_completed_latch = threading.Event()
        self.gpu_done = threading.Event()
        self.read_after_gpu = False
        self.read_completions: list[float] = []
        self.bytes_read = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def wait_read_completed(self, timeout: float = 5.0) -> bool:
        return self.read_completed_latch.wait(timeout)

    def join(self, timeout: float = 10.0):
        self._thread.join(timeout)

    def _run(self):
        try:
            total = 0
            first_chunk = min(self.chunk_bytes, self.target_bytes)
            data = os.pread(self.fd, first_chunk, self.offset)
            total += len(data)
            self.read_completions.append(time.perf_counter())
            # The latch means one read completed. It does not prove DMA stayed
            # active after the syscall returned.
            self.read_completed_latch.set()

            while total < self.target_bytes:
                n = min(self.chunk_bytes, self.target_bytes - total)
                data = os.pread(self.fd, n, self.offset + total)
                if not data:
                    break
                total += len(data)
                self.read_completions.append(time.perf_counter())
                if self.gpu_done.is_set():
                    # This flag has one narrow meaning: a read completed after
                    # the GPU call returned. It is not a continuous-overlap
                    # or physical-DMA proof.
                    self.read_after_gpu = True

            self.bytes_read = total
        except BaseException as exc:
            self.error = exc
            self.read_completed_latch.set()


def run_benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=os.path.expanduser(
            "~/models/Qwen3.8-Flash-Next-MLX-oQ4/model-00001-of-00022.safetensors"
        ),
        help="Path to large (>1GB) model file for unbuffered reads",
    )
    parser.add_argument(
        "--mb-levels",
        default="0,16,32,64",
        help="Comma-separated physical background DMA MB levels",
    )
    parser.add_argument("--arms", type=int, default=5, help="Rounds of interleaved arms per strategy")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause seconds between arms")
    args = parser.parse_args()

    import mlx.core as mx
    from models.flashnext.diskio import ReadMeter
    from models.flashnext.metal_native import init_native_moe, run_native_moe

    init_native_moe()

    hidden_size = 2560
    inter_size = 640
    slots = 8
    expert_count = 16

    def make_pack(seed, out_w, in_w):
        v = ((mx.arange(expert_count * out_w * in_w, dtype=mx.float32) + seed * 3) % 17 - 8) / 64
        return mx.quantize(
            v.reshape(expert_count, out_w, in_w).astype(mx.bfloat16),
            group_size=32,
            bits=4,
        )

    gate_pack = make_pack(0, inter_size, hidden_size)
    up_pack = make_pack(1, inter_size, hidden_size)
    down_pack = make_pack(2, hidden_size, inter_size)

    x = np.ones((1, hidden_size), dtype=np.float32)
    routes = np.arange(slots, dtype=np.uint32).reshape(1, slots) % expert_count
    scores = np.ones((1, slots), dtype=np.float32) / slots

    # Pre-warm GPU kernels
    for strat in DEFAULT_STRATEGIES:
        run_native_moe(x, routes, scores, gate_pack, up_pack, down_pack, strategy=strat, expert_count=expert_count)

    levels = [float(x.strip()) for x in args.mb_levels.split(",") if x.strip()]
    file_path = Path(args.file).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    fd = os.open(str(file_path), os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)

    print("=== Issue #45: True Unbuffered SSD DMA vs Metal Scheduler Contention ===")
    print(f"File: {file_path.name} ({file_size / 1e9:.2f} GB) with F_NOCACHE")
    print(f"Strategies: {', '.join(DEFAULT_STRATEGIES)}")
    print(f"Arms per strategy: {args.arms}")
    print(f"System load average: {os.getloadavg()}\n")

    current_offset = 0
    all_summaries = {}

    try:
        for mb in levels:
            bytes_target = int(mb * 1024 * 1024)
            print(f"--- Running Level: {mb:.0f} MB DMA ---")
            schedule = interleaved_order(DEFAULT_STRATEGIES, args.arms)

            samples_gpu: dict[str, list[float]] = {s: [] for s in DEFAULT_STRATEGIES}
            samples_host: dict[str, list[float]] = {s: [] for s in DEFAULT_STRATEGIES}
            samples_phys_mb: dict[str, list[float]] = {s: [] for s in DEFAULT_STRATEGIES}
            overlaps: dict[str, int] = {s: 0 for s in DEFAULT_STRATEGIES}
            completion_during_native_call: dict[str, int] = {
                s: 0 for s in DEFAULT_STRATEGIES
            }
            read_spans_ms: dict[str, list[float]] = {s: [] for s in DEFAULT_STRATEGIES}

            meter = ReadMeter()

            for strategy in schedule:
                meter.reset()
                worker = None
                if bytes_target > 0:
                    if current_offset + bytes_target > file_size:
                        current_offset = 0
                    worker = SynchronizedDmaWorker(fd, current_offset, bytes_target)
                    current_offset += bytes_target
                    worker.start()
                    if not worker.wait_read_completed():
                        raise RuntimeError("Worker failed to signal completed read")

                call_start = time.perf_counter()
                native_error = None
                try:
                    out, gpu_ms = run_native_moe(
                        x, routes, scores, gate_pack, up_pack, down_pack,
                        strategy=strategy, expert_count=expert_count,
                    )
                except BaseException:
                    native_error = sys.exc_info()
                call_end = time.perf_counter()
                host_ms = (call_end - call_start) * 1000.0

                if worker is not None:
                    # Always finish the worker before the next arm or FD close,
                    # including when the native call raises.
                    worker.gpu_done.set()
                    worker.join()
                    if worker.is_alive():
                        raise RuntimeError("DMA worker did not join")
                    if worker.error:
                        raise RuntimeError(f"Worker failed: {worker.error}")
                    if worker.bytes_read != bytes_target:
                        raise RuntimeError(
                            f"DMA worker read {worker.bytes_read} bytes, "
                            f"expected {bytes_target}"
                        )
                    if worker.read_after_gpu:
                        overlaps[strategy] += 1
                    completions = worker.read_completions
                    if any(
                        call_start <= completed <= call_end
                        for completed in completions
                    ):
                        completion_during_native_call[strategy] += 1
                    if len(completions) > 1:
                        read_spans_ms[strategy].append(
                            (completions[-1] - completions[0]) * 1000.0
                        )

                if native_error is not None:
                    raise native_error[1].with_traceback(native_error[2])

                phys_bytes = meter.bytes_since()
                phys_mb = (phys_bytes / 1e6) if phys_bytes >= 0 else 0.0

                samples_gpu[strategy].append(gpu_ms)
                samples_host[strategy].append(host_ms)
                samples_phys_mb[strategy].append(phys_mb)

                if args.pause > 0:
                    time.sleep(args.pause)

            if bytes_target > 0:
                physical_values = [
                    value for values in samples_phys_mb.values() for value in values
                ]
                if not physical_values or any(value <= 0 for value in physical_values):
                    raise RuntimeError(
                        "physical read telemetry is unavailable; refusing DMA strategy comparison"
                    )
                physical_median = statistics.median(physical_values)
                expected_mb = bytes_target / 1e6
                if physical_median <= 0 or any(
                    not 0.5 <= value / expected_mb <= 2.0
                    for value in physical_values
                ):
                    raise RuntimeError(
                        "physical read volume is not reasonable for the requested "
                        "DMA level; refusing unsupported comparison"
                    )
                if any(
                    abs(value - physical_median) / physical_median > 0.10
                    for value in physical_values
                ):
                    raise RuntimeError(
                        "physical read volumes differ by more than 10%; "
                        "refusing unsupported DMA strategy comparison"
                    )

            print(f"Level {mb:.0f} MB Results (n={len(schedule)//len(DEFAULT_STRATEGIES)} per arm):")
            print(f"{'Strategy':<10} | {'GPU med (ms)':<12} | {'Host med (ms)':<13} | {'Phys MB Read':<12} | {'Post-native read':<16} | {'Read interval':<13} | {'GPU Band':<8}")
            print("-" * 105)
            level_res = {}
            for strat in DEFAULT_STRATEGIES:
                g_sum = summarize(samples_gpu[strat])
                h_sum = summarize(samples_host[strat])
                avg_phys = statistics.mean(samples_phys_mb[strat])
                post_gpu_pct = (overlaps[strat] / len(samples_gpu[strat]) * 100) if bytes_target > 0 else 0.0
                during_native_call_pct = (
                    completion_during_native_call[strat]
                    / len(samples_gpu[strat]) * 100
                    if bytes_target > 0 else 0.0
                )
                read_span = (
                    statistics.median(read_spans_ms[strat])
                    if read_spans_ms[strat] else 0.0
                )
                print(
                    f"{strat:<10} | {g_sum.median_ms:12.3f} | {h_sum.median_ms:13.3f} | "
                    f"{avg_phys:12.1f} | {post_gpu_pct:6.0f}% post | "
                    f"{during_native_call_pct:6.0f}% call/{read_span:5.1f} ms | "
                    f"{g_sum.resolution_band_pct:7.1f}%"
                )
                level_res[strat] = {
                    "gpu_summary": g_sum,
                    "host_summary": h_sum,
                    "phys_mb": avg_phys,
                    "post_gpu_read_pct": post_gpu_pct,
                    "read_completion_during_native_call_pct": during_native_call_pct,
                    "read_completion_span_ms": read_span,
                }
            all_summaries[mb] = level_res
            print()

    finally:
        os.close(fd)

    print("=== Summary Across DMA Levels (observational GPU/read metrics) ===")
    header = f"{'DMA Level':<10} | {'Serial (ms)':<12} | {'Barrier (ms)':<12} | {'Fence (ms)':<12} | {'Barrier Delta':<14} | {'Fence Delta':<12}"
    print(header)
    print("-" * len(header))
    for mb in levels:
        res = all_summaries[mb]
        s_gpu = res["serial"]["gpu_summary"].median_ms
        b_gpu = res["barrier"]["gpu_summary"].median_ms
        f_gpu = res["fence"]["gpu_summary"].median_ms
        b_diff = b_gpu - s_gpu
        f_diff = f_gpu - s_gpu
        print(
            f"{mb:5.0f} MB   | {s_gpu:12.3f} | {b_gpu:12.3f} | {f_gpu:12.3f} | "
            f"{b_diff:+13.3f} ms | {f_diff:+11.3f} ms"
        )
    print(
        "Observation limits: the latch marks a completed read. Post-native-call read "
        "completion does not prove continuous physical DMA or full overlap."
    )


if __name__ == "__main__":
    run_benchmark()
