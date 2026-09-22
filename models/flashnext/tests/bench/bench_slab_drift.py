#!/usr/bin/env python3
"""Measure normal-chat slab drift with fresh, paired Vontra launches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

from macqwen.measurement import MeasurementRun, source_hash, validate_path
from macqwen.results import ROOT, output_path


def _prompt(question: str) -> str:
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


PROMPTS = (
    _prompt("Explique a fotossintese em duas frases."),
    _prompt("Escreva uma função Python que remova duplicatas de uma lista mantendo a ordem."),
    _prompt("Explique em duas frases por que a Lua tem fases."),
)
LIVE_PINS = Path("~/.cache/flashnext/pins.json").expanduser()
PACK_DIR = LIVE_PINS.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def child(args) -> None:
    # Importing this module creates an MLX device; keep it out of the parent.
    from models.flashnext.tests.bench import bench_slab_production as slab

    if args.condition == "frozen":
        slab.FROZEN_ARM_PINS = args.arm_pins
        slab.install_frozen_pins(args.frozen_pins)
    else:
        os.environ["FLASHNEXT_PIN_CACHE"] = str(args.rolling_pins)
    row = slab.run_arm(
        0, 0, args.tokens, 60, slab_pack=True, policy="skew",
        fuse_shared=True, fuse_shared_parts=False, stream_pack=0,
        fuse_up_swiglu=True, resident_experts=8,
        require_existing_pack=False, prompt=PROMPTS[args.prompt_index],
    )
    path = output_path("flashnext", "bench_slab_drift", args.json.name, args.json)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(
        f"  -> {args.condition}: {row['gen_rate']:.2f} tok/s, "
        f"{row['phys_mb_tok']:.1f} MB/tok, {row['hit_pct']:.1f}% hits, "
        f"alloc {row['allocation_digest']}, digest {row['digest']}",
        flush=True,
    )


def measure(args) -> None:
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.tests.bench.bench_production import benchmark_provenance

    destination = output_path("flashnext", "bench_slab_drift", "slab-drift.json", args.json)
    record = validate_path(
        args.record or destination.with_name("slab-drift-arms.jsonl"), ROOT, "flashnext"
    )
    checkpoint = resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL"))
    provenance = benchmark_provenance(checkpoint)
    if not LIVE_PINS.is_file():
        raise RuntimeError(f"normal-chat pin history is missing: {LIVE_PINS}")
    profile = json.loads(LIVE_PINS.read_text())
    if profile.get("checkpoint_identity") != provenance["checkpoint_identity"]:
        raise RuntimeError("normal-chat pin history belongs to another checkpoint")
    if int(profile.get("quantization", {}).get("group_size", 0)) != 32:
        raise RuntimeError("normal-chat pin history is not Q4/G32")

    frozen = destination.with_name("frozen-pins.json")
    rolling = destination.with_name("rolling-pins.json")
    starting_profile = LIVE_PINS.read_bytes()
    frozen.write_bytes(starting_profile)
    rolling.write_bytes(starting_profile)
    source_pins_digest = digest(frozen)
    schedule = [
        (round_index, condition)
        for round_index in range(args.pairs)
        for condition in (
            ("rolling", "frozen") if round_index % 2 == 0
            else ("frozen", "rolling")
        )
    ]
    run = MeasurementRun(record, runtime="flashnext", experiment="slab-drift", metadata={
        **provenance,
        "harness_sha256": source_hash(Path(__file__)),
        "slab_harness_sha256": source_hash(Path(__file__).with_name("bench_slab_production.py")),
        "prompt": list(PROMPTS), "sampling": "greedy", "tokens": args.tokens,
        "schedule": schedule, "resident_experts": 8, "slab_slots": 60,
        "initial_pin_sha256_prefix": source_pins_digest,
        "pin_paths": {"source": str(LIVE_PINS), "frozen": str(frozen), "rolling": str(rolling)},
        "environment": {
            key: os.environ.get(key) for key in (
                "FLASHNEXT_METAL_RUNTIME", "FLASHNEXT_SLAB_GLOBAL", "FLASHNEXT_SLAB_PACK",
                "FLASHNEXT_PREAD_CHUNK", "FLASHNEXT_IO_WORKERS", "FLASHNEXT_GPU_KEEPWARM",
                "FLASHNEXT_SLAB_COUNTS",
            )
        },
    })
    run.start()
    rows = {"rolling": [], "frozen": []}
    try:
        for index, (round_index, condition) in enumerate(schedule, 1):
            arm_id = f"round-{round_index + 1}-{condition}"
            source = rolling if condition == "rolling" else frozen
            before = digest(source)
            packs_before = set(PACK_DIR.glob("slab-pack-slots*.bin"))
            arm_json = destination.with_name(f"arm-{index:02d}.json")
            arm_pins = destination.with_name(f"arm-{index:02d}-pins.json")
            command = [
                sys.executable, "-m", "models.flashnext.tests.bench.bench_slab_drift",
                "--child", "--condition", condition,
                "--prompt-index", str(round_index % len(PROMPTS)),
                "--tokens", str(args.tokens), "--json", str(arm_json),
                "--frozen-pins", str(frozen), "--rolling-pins", str(rolling),
                "--arm-pins", str(arm_pins),
            ]
            print(
                f"Arm {index:2d}/{len(schedule)}: Running {condition} "
                f"(prompt {round_index + 1})...", flush=True,
            )
            began = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True)
            launch_s = time.perf_counter() - began
            if completed.stdout:
                print(completed.stdout, end="", flush=True)
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr, flush=True)
            after = digest(rolling if condition == "rolling" else arm_pins) \
                if completed.returncode == 0 else None
            row = json.loads(arm_json.read_text()) if arm_json.is_file() else {}
            row.update({
                "launch_s": round(launch_s, 2), "pin_digest_before": before,
                "pin_digest_after": after, "prompt_index": round_index % len(PROMPTS),
                "new_pack": bool(row.get("pack_path")) and Path(row["pack_path"]) not in packs_before,
            })
            # Persist the raw arm before checking status, digest, or pack path.
            run.arm(
                arm_id=arm_id, condition=condition, round_index=round_index,
                command=command, status="raw" if completed.returncode == 0 else "failed",
                metrics=row, token_digest=row.get("digest"),
                returncode=completed.returncode,
            )
            if completed.returncode or not row.get("mlock_ok") or row.get("tokens") != args.tokens:
                raise RuntimeError(f"{arm_id} failed or did not complete {args.tokens} tokens")
            rows[condition].append(row)
            if len(rows["rolling"]) == len(rows["frozen"]):
                left, right = rows["rolling"][-1], rows["frozen"][-1]
                if left["digest"] != right["digest"]:
                    raise RuntimeError(f"round {round_index + 1} token digests differ")
                run.validation(arm_id, "passed", exact_digest=left["digest"])
            if index < len(schedule):
                time.sleep(args.pause)

        effects = [
            (frozen_arm["gen_rate"] / rolling_arm["gen_rate"] - 1) * 100
            for rolling_arm, frozen_arm in zip(rows["rolling"], rows["frozen"])
        ]
        band = 2 * st.stdev(effects) / len(effects) ** 0.5 if len(effects) > 1 else None
        summary = {
            "paired_frozen_vs_rolling_pct": effects,
            "mean_pct": st.mean(effects), "two_se_band_pct": band,
            "rolling_allocations": [row["allocation_digest"] for row in rows["rolling"]],
            "frozen_allocations": [row["allocation_digest"] for row in rows["frozen"]],
            "rolling_pin_digests": [row["pin_digest_after"] for row in rows["rolling"]],
            "rows": rows,
        }
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        run.finish("completed", **{key: value for key, value in summary.items() if key != "rows"})
        print(
            f"Frozen vs rolling: {summary['mean_pct']:+.1f}% mean "
            f"(two-SE band ±{band:.1f}%). Wrote {destination}", flush=True,
        )
    except BaseException as error:
        run.failure(str(error), error_type=type(error).__name__)
        run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "completed_with_failures")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--pause", type=float, default=2.5)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--condition", choices=("rolling", "frozen"))
    parser.add_argument("--prompt-index", type=int)
    parser.add_argument("--frozen-pins", type=Path)
    parser.add_argument("--rolling-pins", type=Path)
    parser.add_argument("--arm-pins", type=Path)
    args = parser.parse_args()
    if args.tokens < 1 or args.pairs < 3 or args.pause < 0:
        parser.error("tokens must be positive, pairs at least 3, and pause non-negative")
    if args.child:
        if (
            args.condition is None or args.prompt_index is None
            or not 0 <= args.prompt_index < len(PROMPTS)
            or any(path is None for path in (
                args.json, args.frozen_pins, args.rolling_pins, args.arm_pins
            ))
        ):
            parser.error("child mode needs its condition, prompt and profile paths")
        child(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
