#!/usr/bin/env python3
"""Diagnose FlashNext read-pool width and task grouping.

This diagnostic keeps the production 60-slot skew pack, Frontier 8A, the
current Up-QMV/SwiGLU setting, reads, destinations, requested bytes, and
greedy token trajectory fixed. It runs one fresh child per condition because
the pool width is captured when ``expert_cache`` imports.

The worker sweep is the premise test. The task-topology comparison is allowed
only when the sweep shows material queue residence. It compares the existing
projection-major tasks with one task per expert, where each task reads the
three projections sequentially into the same projection-major destinations.
The separate production mode compares 16 and 8 workers with profiling off.
It preserves a private copy of the same pin profile for every 32-token arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb, sqrt
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import tempfile

from models.flashnext.settings.launch import CHAT_ENV


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "models" / "flashnext" / "bench_slab_production.py"
DEFAULT_WORKERS = (8, 16, 24, 32)
QUEUE_THRESHOLD_MS = 5.0


def _child_environment(
    workers: int, topology: str, profile_io: bool = True,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(CHAT_ENV)
    environment.update({
        "FLASHNEXT_IO_WORKERS": str(workers),
        "FLASHNEXT_IO_TASK_TOPOLOGY": topology,
        "FLASHNEXT_PROFILE_IO": "1" if profile_io else "0",
        "FLASHNEXT_PROFILE_BOUNDARIES": "0",
        "FLASHNEXT_PROFILE_SCORE_SYNC": "0",
        "FLASHNEXT_STREAM_PACK": "0",
        "FLASHNEXT_FUSED_SHARED_PARTS": "0",
        "FLASHNEXT_READ": "pread",
        "FLASHNEXT_PREAD_CHUNK": "2",
        "FLASHNEXT_SHARED_READ_BUFFER": "1",
    })
    return environment


def _run_child(
    workers: int, topology: str, tokens: int, pairs: int,
    *, profile_io: bool = True, pin_cache: Path | None = None,
) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="flashnext-io-arm-", delete=False
    ) as handle:
        result_path = Path(handle.name)
    command = [
        str(sys.executable), str(BENCHMARK),
        "--arms", "slabpack60_skew_f10_up",
        "--tokens", str(tokens),
        "--pairs", str(pairs),
        "--pause", "0",
        "--json", str(result_path),
    ]
    if profile_io:
        command.append("--profile-io")
    environment = _child_environment(workers, topology, profile_io)
    if pin_cache is not None:
        environment["FLASHNEXT_PIN_CACHE"] = str(pin_cache)
    label = f"workers={workers} topology={topology}"
    print(f"Arm process: {label}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    returncode = process.wait()
    try:
        payload = json.loads(result_path.read_text()) if result_path.is_file() else {}
    finally:
        result_path.unlink(missing_ok=True)
    if returncode != 0:
        raise RuntimeError(f"{label} exited with status {returncode}")
    conditions = payload.get("conditions", [])
    if len(conditions) != 1:
        raise RuntimeError(f"{label} did not produce one condition")
    condition = conditions[0]
    arms = condition.get("arms", [])
    if not arms:
        raise RuntimeError(f"{label} did not produce an arm")
    return {
        "workers": workers,
        "topology": topology,
        "condition": condition.get("condition", ""),
        "arms": arms,
        "digest": next(
            (arm.get("digest", "") for arm in arms if arm.get("digest")), ""
        ),
    }


def production_effect(records: list[dict], tokens: int = 32) -> dict:
    """Compare matched worker arms only after control and digest checks."""
    by_width = {width: [] for width in (16, 8)}
    for record in records:
        width = record["workers"]
        if width not in by_width or record["topology"] != "projection":
            raise ValueError("unexpected worker or topology condition")
        if len(record["arms"]) != 1:
            raise ValueError("each process must contain exactly one arm")
        arm = record["arms"][0]
        if arm.get("tokens") != tokens or arm.get("io_workers") != width:
            raise ValueError("token count or effective worker count mismatch")
        if arm.get("profile_io") is not False:
            raise ValueError("production comparison requires profiling off")
        if arm.get("io_topology") != "projection":
            raise ValueError("effective topology mismatch")
        if arm.get("allocated_slots") != 60 or not arm.get("mlock_ok"):
            raise ValueError("production comparison requires 60 locked slots")
        if arm.get("gen_rate", 0) <= 0 or arm.get("phys_mb_tok", -1) < 0:
            raise ValueError("generation or physical-read metrics unavailable")
        if not all(key in arm.get("vm_counters", {})
                   for key in ("swapin", "swapout", "pageout", "compress", "decompress")):
            raise ValueError("VM counters unavailable")
        by_width[width].append(arm)
    control, candidate = by_width[16], by_width[8]
    if len(control) < 3 or len(control) != len(candidate):
        raise ValueError("at least three complete worker pairs are required")
    arms = control + candidate
    for field in ("digest", "allocation_digest"):
        values = {arm.get(field) for arm in arms}
        if len(values) != 1 or not next(iter(values)) or "none" in values:
            raise ValueError(f"{field} mismatch or missing evidence")
    effects = [
        (new["gen_rate"] / old["gen_rate"] - 1) * 100
        for old, new in zip(control, candidate)
    ]
    physical_deltas = [
        old["phys_mb_tok"] - new["phys_mb_tok"]
        for old, new in zip(control, candidate)
    ]
    band = 2 * st.stdev(effects) / sqrt(len(effects))
    wins = sum(effect > 0 for effect in effects)
    non_ties = sum(effect != 0 for effect in effects)
    vm_warnings = [
        {"arm": index + 1, "workers": width, "vm": arm.get("vm_counters", {})}
        for width, rows in by_width.items() for index, arm in enumerate(rows)
        if any(arm.get("vm_counters", {}).get(key, 0) > max(256, tokens * 8)
               for key in ("swapin", "swapout", "pageout"))
    ]
    mean = st.mean(effects)
    return {
        "pairs": len(effects), "mean_percent": mean,
        "median_percent": st.median(effects), "two_se_percent": band,
        "wins": wins,
        "sign_p": sum(comb(non_ties, k) for k in range(wins, non_ties + 1))
        / 2 ** non_ties,
        "physical_reduction_median_mb_token": st.median(physical_deltas),
        "paired_physical_reductions_mb_token": physical_deltas,
        "paired_effects_percent": effects,
        "vm_warnings": vm_warnings,
        "resolved_gain": mean > band and band <= 10 and not vm_warnings,
        "digest": arms[0]["digest"],
        "allocation_digest": arms[0]["allocation_digest"],
    }


def production_compare(args) -> int:
    """Resolve the diagnostic's eight-worker candidate without profiling."""
    if args.tokens != 32 or args.rounds < 3:
        raise ValueError("production comparison requires 32 tokens and at least three pairs")
    source = Path(os.environ.get(
        "FLASHNEXT_PIN_CACHE", "~/.cache/flashnext/pins.json"
    )).expanduser()
    initial_pins = source.read_bytes()
    records = []
    with tempfile.TemporaryDirectory(prefix="flashnext-worker-pins-") as folder:
        for round_index in range(args.rounds):
            order = (16, 8) if round_index % 2 == 0 else (8, 16)
            for width in order:
                pins = Path(folder) / f"pins-{round_index}-{width}.json"
                pins.write_bytes(initial_pins)
                record = _run_child(
                    width, "projection", 32, 1, profile_io=False, pin_cache=pins,
                )
                record["round"] = round_index + 1
                records.append(record)
    payload = {
        "mode": "production", "tokens": 32, "rounds": args.rounds,
        "profile_io": False,
        "initial_pin_digest": hashlib.sha256(initial_pins).hexdigest(),
        "records": records,
    }
    # Retain raw evidence even if the cross-process checks fail.
    _write_result(args.json, payload)
    effect = production_effect(records)
    payload["effect"] = effect
    payload["summaries"] = []
    for width in (16, 8):
        summary = _summary([record for record in records if record["workers"] == width])
        # Unprofiled timing fields are unavailable, not measured zeros.
        payload["summaries"].append({
            key: summary[key] for key in (
                "workers", "topology", "arms", "gen_median", "tail_median",
                "physical_mb_token_median", "active_mb_median", "digests",
            )
        })
    for summary in payload["summaries"]:
        print(json.dumps(summary, sort_keys=True), flush=True)
    print(
        f"8 vs 16 workers: mean {effect['mean_percent']:+.2f}%, "
        f"median {effect['median_percent']:+.2f}%, "
        f"paired two-SE band: {effect['two_se_percent']:.2f}%, "
        f"wins {effect['wins']}/{effect['pairs']}, p={effect['sign_p']:.3f}",
        flush=True,
    )
    print("Short-arm evidence only. Defaults remain a user decision.", flush=True)
    _write_result(args.json, payload)
    return 0


def _arm_metrics(record: dict) -> dict:
    breakdown = record.get("io_breakdown_ms_tok", {})
    return {
        "gen_rate": float(record.get("gen_rate", 0.0)),
        "tail_rate": float(record.get("tail_rate", 0.0)),
        "physical_mb_token": float(record.get("phys_mb_tok", 0.0)),
        "active_mb": float(record.get("active_mb", 0.0)),
        "queue_ms_token": float(breakdown.get("critical_queue", 0.0)),
        "pread_ms_token": float(breakdown.get("critical_pread", 0.0)),
        "task_overhead_ms_token": float(
            breakdown.get("critical_task_overhead", 0.0)
        ),
        "completion_ms_token": float(
            breakdown.get("completion_overhead", 0.0)
        ),
        "io_wait_ms_token": float(record.get("io_wait_ms_tok", 0.0)),
        "layer_completion_ms_layer": float(
            breakdown.get(
                "layer_completion", breakdown.get("completion_overhead", 0.0)
            )
        ),
        "digest": record.get("digest", ""),
    }


def _summary(records: list[dict]) -> dict:
    arms = [arm for record in records for arm in record["arms"]]
    metrics = {
        key: [values[key] for values in map(_arm_metrics, arms)]
        for key in (
            "gen_rate", "tail_rate", "physical_mb_token", "active_mb",
            "queue_ms_token", "pread_ms_token", "task_overhead_ms_token",
            "completion_ms_token", "io_wait_ms_token",
            "layer_completion_ms_layer",
        )
    }
    return {
        "workers": records[0]["workers"],
        "topology": records[0]["topology"],
        "arms": len(arms),
        "gen_median": st.median(metrics["gen_rate"]),
        "tail_median": st.median(metrics["tail_rate"]),
        "physical_mb_token_median": st.median(metrics["physical_mb_token"]),
        "active_mb_median": st.median(metrics["active_mb"]),
        "queue_ms_token_median": st.median(metrics["queue_ms_token"]),
        "pread_ms_token_median": st.median(metrics["pread_ms_token"]),
        "task_overhead_ms_token_median": st.median(
            metrics["task_overhead_ms_token"]
        ),
        "completion_ms_token_median": st.median(
            metrics["completion_ms_token"]
        ),
        "io_wait_ms_token_median": st.median(metrics["io_wait_ms_token"]),
        "layer_completion_ms_layer_median": st.median(
            metrics["layer_completion_ms_layer"]
        ),
        "digests": sorted({arm.get("digest", "") for arm in arms}),
    }


def _write_result(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}", flush=True)


def section17_control(args) -> int:
    records = [
        _run_child(16, "projection", args.tokens, 1)
        for _round in range(args.rounds)
    ]
    summary = _summary(records)
    material = summary["queue_ms_token_median"] >= args.queue_threshold
    payload = {
        "mode": "control",
        "tokens": args.tokens,
        "rounds": args.rounds,
        "records": records,
        "summary": summary,
        "premise": {
            "queue_threshold_ms_token": args.queue_threshold,
            "worker_sweep_eligible": material,
        },
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(
        "PREMISE GATE: "
        + ("queue residence is material; worker sweep is eligible"
           if material else "queue residence is not material; worker sweep remains blocked"),
        flush=True,
    )
    _write_result(args.json, payload)
    return 0


def worker_sweep(args) -> int:
    if args.control_json is None or not args.control_json.is_file():
        raise SystemExit(
            "worker sweep requires --control-json from the Section 17 control"
        )
    control = json.loads(args.control_json.read_text())
    if not control.get("premise", {}).get("worker_sweep_eligible", False):
        print("PREMISE GATE: blocked; no worker sweep was run", flush=True)
        return 0
    records = []
    for round_index in range(args.rounds):
        workers = (
            DEFAULT_WORKERS
            if round_index % 2 == 0
            else tuple(reversed(DEFAULT_WORKERS))
        )
        for width in workers:
            records.append(_run_child(width, "projection", args.tokens, 1))
    summaries = [
        _summary([record for record in records if record["workers"] == width])
        for width in DEFAULT_WORKERS
    ]
    print("WORKER SUMMARY", flush=True)
    for row in summaries:
        print(json.dumps(row, sort_keys=True), flush=True)
    queue_values = [row["queue_ms_token_median"] for row in summaries]
    material = (
        max(queue_values) - min(queue_values) >= args.queue_threshold
        or max(queue_values) >= args.queue_threshold
    )
    selected = min(
        summaries,
        key=lambda row: (
            row["io_wait_ms_token_median"],
            -row["gen_median"],
            row["workers"],
        ),
    )["workers"]
    print(
        "PREMISE GATE: "
        + ("queue residence is material; topology comparison is eligible"
           if material
           else "queue residence is not material; topology comparison remains blocked"),
        flush=True,
    )
    payload = {
        "mode": "workers",
        "tokens": args.tokens,
        "rounds": args.rounds,
        "records": records,
        "summaries": summaries,
        "premise": {
            "queue_threshold_ms_token": args.queue_threshold,
            "topology_eligible": material,
        },
        "selected_workers": selected,
    }
    _write_result(args.json, payload)
    return 0


def topology_compare(args) -> int:
    if args.worker_json is None or not args.worker_json.is_file():
        raise SystemExit(
            "topology comparison requires --worker-json from the worker sweep"
        )
    worker_payload = json.loads(args.worker_json.read_text())
    premise = worker_payload.get("premise", {})
    if not premise.get("topology_eligible", False):
        print("PREMISE GATE: blocked; no task-topology benchmark was run", flush=True)
        return 0
    workers = int(worker_payload.get("selected_workers", args.workers))
    records = []
    for round_index in range(args.rounds):
        topologies = (
            ("projection", "expert")
            if round_index % 2 == 0
            else ("expert", "projection")
        )
        for topology in topologies:
            records.append(_run_child(workers, topology, args.tokens, 1))
    summaries = [_summary([record]) for record in records]
    print("TOPOLOGY SUMMARY", flush=True)
    for row in summaries:
        print(json.dumps(row, sort_keys=True), flush=True)
    digests = sorted({digest for row in summaries for digest in row["digests"]})
    if len(digests) > 1:
        print(
            "DIGEST GATE: mismatch; no topology interpretation is allowed",
            flush=True,
        )
    by_topology = {
        topology: [row for row in summaries if row["topology"] == topology]
        for topology in ("projection", "expert")
    }
    projection = [row["gen_median"] for row in by_topology["projection"]]
    expert = [row["gen_median"] for row in by_topology["expert"]]
    if projection and expert:
        projection_rate = st.median(projection)
        expert_rate = st.median(expert)
        change = (expert_rate / projection_rate - 1.0) * 100.0
        print(
            f"INTERPRETATION: expert-task grouping changes generation by {change:+.1f}% "
            "against the projection-task control; inspect queue, pread, layer "
            "completion, physical bytes, RAM, and digest before judging it.",
            flush=True,
        )
    _write_result(args.json, {
        "mode": "topology",
        "workers": workers,
        "tokens": args.tokens,
        "rounds": args.rounds,
        "records": records,
        "summaries": summaries,
        "digests": digests,
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("control", "workers", "topology", "production"), default="control"
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--queue-threshold", type=float, default=QUEUE_THRESHOLD_MS
    )
    parser.add_argument("--worker-json", type=Path)
    parser.add_argument("--control-json", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.tokens <= 0 or args.rounds <= 0 or args.workers <= 0:
        parser.error("tokens, rounds, and workers must be positive")
    if args.mode == "production":
        return production_compare(args)
    if args.mode == "control":
        return section17_control(args)
    if args.mode == "workers":
        return worker_sweep(args)
    return topology_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
