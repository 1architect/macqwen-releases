#!/usr/bin/env python3
"""Measure the explicit native Metal dependency-chain scheduler.

The native bridge owns its command queue and encoders. This harness sweeps the
explicit ``serial``, ``barrier``, and ``fence`` strategies with interleaved
arms, checks every output against the same expected array, and reports a
median plus a per-arm resolution band. It does not open model files. Optional
background I/O needs an explicit regular file path and a bounded byte count.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Callable, Iterable

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DEFAULT_STRATEGIES = ("serial", "barrier", "fence")
VALID_STRATEGIES = frozenset(DEFAULT_STRATEGIES)
MAX_WIDTH = 16_384
MAX_STEPS = 64
MAX_BACKGROUND_BYTES = 256 * 1024 * 1024


def parse_strategies(value: str) -> tuple[str, ...]:
    """Parse a non-empty list of unique native scheduler strategies."""
    result = []
    for item in value.split(","):
        text = item.strip()
        if text not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {', '.join(DEFAULT_STRATEGIES)}: {text!r}"
            )
        if text not in result:
            result.append(text)
    if not result:
        raise ValueError("at least one scheduler strategy is required")
    return tuple(result)


def interleaved_order(values: Iterable[str], arms: int, rounds: int = 1) -> list[str]:
    """Return a rotated schedule so each arm sees every position."""
    values = tuple(values)
    if not values:
        raise ValueError("at least one arm is required")
    if arms < 1 or rounds < 1:
        raise ValueError("arms and rounds must be positive")
    order = []
    for round_index in range(rounds):
        for arm in range(arms):
            offset = (round_index * arms + arm) % len(values)
            order.extend(values[offset:] + values[:offset])
    return order


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    resolution_band_pct: float
    samples: int


def summarize(samples: Iterable[float]) -> TimingSummary:
    """Summarize samples using twice the largest median deviation as a band."""
    values = tuple(float(value) for value in samples)
    if not values:
        raise ValueError("at least one timing sample is required")
    median = statistics.median(values)
    deviation = max(abs(value - median) for value in values)
    band = 0.0 if median == 0 else 2.0 * deviation / abs(median) * 100.0
    return TimingSummary(
        median_ms=median,
        minimum_ms=min(values),
        maximum_ms=max(values),
        resolution_band_pct=band,
        samples=len(values),
    )


def verify_output(input_values: np.ndarray, output: np.ndarray, steps: int,
                  reference: np.ndarray | None = None) -> None:
    """Reject a scheduler result that changes input or fails the exact chain."""
    expected = input_values + np.float32(steps)
    # The native kernel adds one in float32 at every step. Repeated addition
    # can differ from one final ``+ steps`` by a few ULPs at 48 steps.
    if not np.allclose(output, expected, rtol=1e-6, atol=1e-5):
        raise RuntimeError("native scheduler output differs from input + steps")
    if reference is not None and not np.array_equal(output, reference):
        raise RuntimeError("native scheduler output differs between strategy arms")


def bounded_file_read(path: str, byte_count: int, chunk_bytes: int = 1 << 20) -> int:
    """Read at most ``byte_count`` bytes from a user-selected regular file."""
    if byte_count < 0 or byte_count > MAX_BACKGROUND_BYTES:
        raise ValueError(
            f"background read must be between 0 and {MAX_BACKGROUND_BYTES} bytes"
        )
    if chunk_bytes < 1:
        raise ValueError("background read chunk must be positive")
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ValueError(f"background path is not a regular file: {file_path}")
    fd = os.open(file_path, os.O_RDONLY)
    total = 0
    try:
        size = os.fstat(fd).st_size
        target = min(byte_count, size)
        while total < target:
            data = os.pread(fd, min(chunk_bytes, target - total), total)
            if not data:
                break
            total += len(data)
    finally:
        os.close(fd)
    return total


def _timed_call(run: Callable[[], np.ndarray], input_values: np.ndarray,
                steps: int, reference: np.ndarray | None) -> float:
    before = input_values.copy()
    began = time.perf_counter()
    output = run()
    elapsed = (time.perf_counter() - began) * 1000.0
    if not np.array_equal(input_values, before):
        raise RuntimeError("native scheduler changed its input buffer")
    verify_output(input_values, output, steps, reference)
    return elapsed


def _background_thread(path: str | None, byte_count: int, chunk_bytes: int):
    if not path or byte_count == 0:
        return None, []
    errors: list[BaseException] = []

    def read():
        try:
            bounded_file_read(path, byte_count, chunk_bytes)
        except BaseException as exc:  # report the worker error after join
            errors.append(exc)

    worker = threading.Thread(target=read, name="flashnext-background-read")
    worker.start()
    return worker, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                        help="comma-separated native strategies")
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--arms", type=int, default=3,
                        help="interleaved repetitions per strategy")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--background-file", default=None,
                        help="optional regular file for bounded background reads")
    parser.add_argument("--background-mb", type=float, default=0.0,
                        help="bytes to read per arm, capped at 256 MB; default 0")
    parser.add_argument("--background-chunk-kb", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        strategies = parse_strategies(args.strategies)
    except ValueError as exc:
        parser.error(str(exc))
    if args.arms < 3:
        parser.error("--arms must be at least 3 for interleaved measurements")
    if not 1 <= args.width <= MAX_WIDTH:
        parser.error(f"--width must be between 1 and {MAX_WIDTH}")
    if not 1 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be between 1 and {MAX_STEPS}")
    background_bytes = int(args.background_mb * 1024 * 1024)
    if background_bytes and not args.background_file:
        parser.error("--background-mb requires --background-file")
    if background_bytes > MAX_BACKGROUND_BYTES:
        parser.error("--background-mb exceeds the 256 MB safety cap")
    if args.background_chunk_kb < 1:
        parser.error("--background-chunk-kb must be positive")

    try:
        from models.flashnext.metal_native import probe_native, run_dependency_chain
    except ImportError as exc:
        print(f"native scheduler unavailable: {exc}")
        return 2
    status = probe_native()
    if not status.available:
        print(f"native scheduler unavailable: {status.reason}")
        return 2

    input_values = np.linspace(-1.0, 1.0, args.width, dtype=np.float32)
    # Compile and cache the native pipeline before any timed sample. Warm each
    # encoder strategy once so the first schedule position is not penalized.
    reference = None
    for strategy in strategies:
        warm = run_dependency_chain(
            input_values, steps=args.steps, strategy=strategy
        )
        verify_output(input_values, warm, args.steps, reference)
        if reference is None:
            reference = warm
    samples: dict[str, list[float]] = {strategy: [] for strategy in strategies}
    schedule = interleaved_order(strategies, args.arms, args.rounds)
    for strategy in schedule:
        worker, errors = _background_thread(
            args.background_file, background_bytes, args.background_chunk_kb * 1024
        )
        try:
            elapsed = _timed_call(
                lambda selected=strategy: run_dependency_chain(
                    input_values, steps=args.steps, strategy=selected
                ),
                input_values, args.steps, reference,
            )
        finally:
            if worker is not None:
                worker.join()
        if errors:
            raise RuntimeError(f"background read failed: {errors[0]}")
        samples[strategy].append(elapsed)

    print(f"width={args.width}, steps={args.steps}, samples={len(schedule)}")
    if args.background_file:
        print(f"background={Path(args.background_file).expanduser()} "
              f"bytes_per_arm={background_bytes}")
    print("\nstrategy  median_ms  min_ms  max_ms  resolution_band  samples")
    for strategy in strategies:
        result = summarize(samples[strategy])
        print(f"{strategy:8s}  {result.median_ms:9.3f}  "
              f"{result.minimum_ms:6.3f}  {result.maximum_ms:6.3f}  "
              f"{result.resolution_band_pct:15.1f}%  {result.samples:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
