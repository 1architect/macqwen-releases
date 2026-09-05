#!/usr/bin/env python3
"""Compare real chat.sh startup paths over a 256-token generation.

Each child process starts through ``chat.sh``. The child emits one JSON record
after every 32 generated tokens. The normal path inherits the caller's
environment. The canonical path adds the current 60-slot Frontier 8A controls.
This keeps startup effects visible and does not alter the established
32-token production benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from models.flashnext.settings.launch import CHAT_ENV


ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "chat.sh"
PROMPT = (
    "Explique a fotossintese em detalhes, em portugues, com pelo menos 800 "
    "palavras. Desenvolva a explicacao de forma continua e nao encerre antes "
    "de cobrir as etapas, estruturas celulares, reagentes, produtos e fatores "
    "que alteram a taxa do processo."
)

def run_path(name: str, phase: str, run_label: str, args) -> list[dict]:
    environment = os.environ.copy()
    if name in {
        "canonical", "slabpack48", "slabpack60",
        "historical", "current", "cache-aware", "physical-miss-hybrid",
        "core48-calibration",
    }:
        environment.update(CHAT_ENV)
    if name == "slabpack48":
        environment["FLASHNEXT_SLAB_GLOBAL"] = "48"
    elif name == "core48-calibration":
        environment["FLASHNEXT_SLAB_GLOBAL"] = "48"
    elif name == "slabpack60":
        environment["FLASHNEXT_SLAB_GLOBAL"] = "60"
    elif name == "historical":
        environment["FLASHNEXT_FUSED_UP_SWIGLU"] = "0"
    elif name == "current":
        environment["FLASHNEXT_FUSED_UP_SWIGLU"] = "1"
    elif name == "cache-aware":
        environment.update({
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
        })
    if name == "physical-miss-hybrid":
        environment["FLASHNEXT_SLAB_POLICY"] = "physical-miss-hybrid"
        environment["FLASHNEXT_PHYSICAL_MISS_PROFILE"] = str(
            args.physical_miss_profile
        )
    environment["FLASHNEXT_PIN_CACHE"] = str(args.pin_profiles[run_label])
    if args.trace_profile:
        if len(args.paths) * len(args.phases) != 1 or args.rounds != 1:
            raise ValueError("physical-miss tracing requires one startup path")
        environment.update({
            "FLASHNEXT_PHYSICAL_MISS_TRACE": "1",
            "FLASHNEXT_PHYSICAL_MISS_PROFILE": str(args.trace_profile),
            "FLASHNEXT_IO_WORKERS": "1",
            "FLASHNEXT_PREAD_CHUNK": "1",
        })
    command = [
        str(CHAT), "--model", "flashnext", "--profile", "plain",
        "--exact-quality", "--no-think" if phase == "answer" else "--think",
        "--benchmark-product-json",
        "--benchmark-prompt", args.prompt, "--benchmark-window", str(args.window),
        "--benchmark-tokens", str(args.tokens), "--benchmark-label", run_label,
        "--benchmark-product-sampling",
        "configured" if phase == "natural" else "greedy",
    ]
    print(f"START {name}: {' '.join(command)}", flush=True)
    records = []
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line:
                print(line, flush=True)
            continue
        records.append(record)
        print(json.dumps(record), flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{run_label} chat.sh exited with status {returncode}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--paths", nargs="+", default=["normal", "canonical"],
                        choices=(
                            "normal", "canonical", "slabpack48", "slabpack60",
                            "historical", "current", "cache-aware",
                            "physical-miss-hybrid", "core48-calibration",
                        ))
    parser.add_argument(
        "--phase", dest="phases", nargs="+",
        choices=("natural", "thinking", "answer"),
    )
    parser.add_argument(
        "--trace-profile", type=Path,
        help="write serialized physical-miss evidence for one startup path",
    )
    parser.add_argument(
        "--physical-miss-profile", type=Path,
        default=Path("~/.cache/flashnext/physical-misses.json").expanduser(),
        help="measured evidence used by the physical-miss startup path",
    )
    args = parser.parse_args()
    args.default_phase = args.phases is None
    args.phases = args.phases or ["thinking"]
    if args.tokens <= 0 or args.window <= 0 or args.tokens % args.window:
        parser.error("tokens must be a positive multiple of window")
    if args.rounds <= 0:
        parser.error("rounds must be positive")
    source_pins = Path(
        os.environ.get("FLASHNEXT_PIN_CACHE", "~/.cache/flashnext/pins.json")
    ).expanduser()
    if not source_pins.is_file():
        raise SystemExit(f"missing routing profile: {source_pins}")
    if "physical-miss-hybrid" in args.paths:
        from models.flashnext.physical_miss import load_profile
        from models.flashnext.slab_topology import (
            canonical_skew_allocation_from_pins,
            simulate_topologies,
        )

        profile = load_profile(args.physical_miss_profile)
        canonical = canonical_skew_allocation_from_pins(source_pins)
        simulation = simulate_topologies(profile, canonical)
        gate = simulation["canonical-core-hybrid"]["offline_ceiling"]
        print(json.dumps({
            "type": "premise", "path": "physical-miss-hybrid",
            "offline_ceiling": gate,
        }), flush=True)
        if not gate["passes"]:
            print(
                "physical-miss-hybrid blocked: predicted saving "
                f"{gate['predicted_mb_per_token']:.2f} MB/token is below "
                f"{gate['minimum_mb_per_token']:.2f} MB/token",
                flush=True,
            )
            return 0
    with tempfile.TemporaryDirectory(prefix="flashnext-product-long-") as directory:
        args.pin_profiles = {}
        conditions = [
            (name if args.default_phase else f"{name}:{phase}", name, phase)
            for phase in args.phases for name in args.paths
        ]
        schedule = []
        for round_index in range(args.rounds):
            ordered = conditions if round_index % 2 == 0 else list(reversed(conditions))
            for condition, name, phase in ordered:
                schedule.append((f"{condition}:r{round_index + 1}", name, phase))
        for run_label, _name, _phase in schedule:
            target = Path(directory) / f"pins-{run_label.replace(':', '-')}.json"
            shutil.copyfile(source_pins, target)
            args.pin_profiles[run_label] = target
        all_records = {
            run_label: run_path(name, phase, run_label, args)
            for run_label, name, phase in schedule
        }
    completes = {
        name: next((row for row in rows if row.get("type") == "complete"), {})
        for name, rows in all_records.items()
    }
    print("SUMMARY", flush=True)
    for name, rows in all_records.items():
        windows = [row for row in rows if row.get("type") == "window"]
        rates = [row.get("generation_tps", 0.0) for row in windows]
        print(json.dumps({
            "path": name,
            "windows": len(windows),
            "generation_tps": rates,
            "digest": completes[name].get("digest", ""),
            "generated_tokens": completes[name].get("generated_tokens", 0),
        }), flush=True)
    incomplete = [
        name for name, complete in completes.items()
        if complete.get("generated_tokens") != args.tokens
    ]
    if incomplete:
        print(
            f"INCOMPLETE: expected {args.tokens} tokens from "
            f"{', '.join(incomplete)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
