"""Can a tiny spinning kernel keep the GPU clock up across I/O gaps?

Decode alternates short GPU bursts with SSD waits. The miss sweep showed the
GPU falling from P15 to P1-P2 once two or more experts per layer come from
the drive, and every command buffer then ran about twice as long. This
checkpoint-free probe reproduces the pattern: a fixed burst of matmuls, then
a sleep that stands in for the reads. It compares:

  continuous  bursts back to back, no gap
  gap         bursts separated by the sleep
  gap+spin    the same, with a one-threadgroup ALU loop running on a second
              stream through the gap; no memory traffic

It reports burst time and GPU performance-state residency for each.

    python -m models.flashnext.tests.bench.bench_gpu_keepwarm
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import mlx.core as mx

from macqwen.results import output_path
from models.flashnext.tests.bench.gpu_pstates import GpuStates, summarize

_SPIN_SOURCE = """
    uint lane = thread_position_in_grid.x;
    float value = float(lane);
    int loops = iterations[0];
    for (int i = 0; i < loops; ++i) {
        value = fma(value, 1.0000001f, 0.5f);
    }
    out[lane] = value;
"""


def spin_kernel():
    return mx.fast.metal_kernel(
        name="flashnext_keepwarm_spin",
        input_names=["iterations"],
        output_names=["out"],
        source=_SPIN_SOURCE,
    )


SPIN_GROUPS = [1]


def spin(kernel, iterations: int, stream):
    lanes = 32 * SPIN_GROUPS[0]
    return kernel(
        inputs=[mx.array([iterations], dtype=mx.int32)],
        grid=(lanes, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(lanes,)], output_dtypes=[mx.float32],
        stream=stream,
    )[0]


def calibrate_spin(kernel, stream, target_ms: float) -> int:
    """Loop count whose spin lasts about ``target_ms`` at the current clock."""
    iterations = 100_000
    for _ in range(6):
        began = time.perf_counter()
        mx.eval(spin(kernel, iterations, stream))
        elapsed = (time.perf_counter() - began) * 1000
        iterations = max(1000, int(iterations * target_ms / max(elapsed, 0.01)))
    return iterations


def run(mode: str, bursts: int, gap_ms: float, burst_ops: int, size: int,
        kernel, spin_stream, iterations: int, meter) -> dict:
    a = mx.random.normal((size, size)).astype(mx.bfloat16)
    mx.eval(a)
    times = []
    before = meter.sample()
    for _ in range(bursts):
        began = time.perf_counter()
        x = a
        for _ in range(burst_ops):
            x = (x @ a) * 0.01
        mx.eval(x)
        times.append((time.perf_counter() - began) * 1000)
        if mode == "continuous":
            continue
        if mode == "gap+spin":
            # Covers the gap on its own stream; the next burst does not wait
            # for it.
            mx.async_eval(spin(kernel, iterations, spin_stream))
        time.sleep(gap_ms / 1000)
    after = meter.sample()
    states = meter.delta(before, after).get("GPUPH", {})
    meter.release(before)
    meter.release(after)
    summary = summarize(states)
    return {
        "mode": mode,
        "burst_ms_median": st.median(times[3:]),
        "burst_ms_p90": sorted(times[3:])[int(0.9 * (len(times) - 3))],
        "mean_active_state": summary["mean_active_state"],
        "active_residency": {k: round(v, 3) for k, v in summary["active_residency"].items() if v >= 0.01},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bursts", type=int, default=60)
    parser.add_argument("--gap-ms", type=float, default=6.0)
    parser.add_argument("--burst-ops", type=int, default=6)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--spin-groups", type=int, default=1,
                        help="threadgroups of 32 lanes in the spin kernel")
    parser.add_argument("--modes", default="continuous,gap,gap+spin")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    SPIN_GROUPS[0] = max(1, args.spin_groups)
    args.json = output_path("flashnext", "bench_gpu_keepwarm", "keepwarm.json", args.json)

    meter = GpuStates()
    kernel = spin_kernel()
    spin_stream = mx.new_stream(mx.gpu)
    iterations = calibrate_spin(kernel, spin_stream, args.gap_ms)
    print(f"spin iterations for ~{args.gap_ms} ms: {iterations}", flush=True)

    rows = []
    order = tuple(m for m in args.modes.split(",") if m)
    for round_index in range(args.rounds):
        for mode in (order if round_index % 2 == 0 else tuple(reversed(order))):
            row = run(mode, args.bursts, args.gap_ms, args.burst_ops, args.size,
                      kernel, spin_stream, iterations, meter)
            row["round"] = round_index
            rows.append(row)
            print(
                f"{mode:<11} burst {row['burst_ms_median']:.2f} ms (p90 "
                f"{row['burst_ms_p90']:.2f}) | mean state "
                f"{row['mean_active_state']:.2f} | {row['active_residency']}",
                flush=True,
            )
            time.sleep(1.0)
    args.json.write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "json"},
                                     "spin_iterations": iterations, "rows": rows}, indent=1))
    print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
