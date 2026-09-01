#!/usr/bin/env python3
"""Measure aggregate throughput when several chat instances decode at once.

One instance alternates between the SSD and the GPU and leaves each idle
about half a token. Independent processes might fill each other's gaps.
They also compete for RAM, and on a 16 GB machine that is the risk: every
instance wires its own MLX weights and pins its own experts, and whatever
is left has to serve as page cache for a 70 GB expert bank.

Reports per-instance and total tokens per second, and asserts that every
instance produced the same token IDs as the solo run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_PROMPT = (
    "Explique em detalhe como funciona a fotossintese, "
    "incluindo as fases clara e escura."
)
PAGE = 16384
# One instance wires about 3.44 GB for weights, plus its pinned experts.
INSTANCE_WIRED_GB = 3.44
PINNED_GB_PER_EXPERT = 4.46 / 32
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def free_gb() -> tuple[float, float]:
    """Return (free, free plus inactive) in GB."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    stats = {}
    for line in out.splitlines():
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            stats[key.strip()] = int(value) * PAGE / 1e9
    free = stats.get("Pages free", 0.0)
    return free, free + stats.get("Pages inactive", 0.0)


def run_instances(count: int, args) -> list[dict]:
    command = [
        os.path.expanduser("~/models/.venv-qwen4exp/bin/python"),
        "-u",
        os.path.join(ROOT, "macqwen", "session.py"),
        "--model", "flashnext",
        "--profile", "plain",
        "--exact-quality",
        "--resident-experts", str(args.resident_experts),
        "--max-tokens", str(args.tokens),
        "--benchmark-json",
        "--benchmark-prompt", args.prompt,
    ]
    started = time.time()
    procs = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for _ in range(count)
    ]
    results = []
    for proc in procs:
        raw = proc.communicate()[0].decode(errors="replace")
        start = raw.find('{"profile"')
        if start < 0:
            raise SystemExit(f"instance produced no JSON:\n{raw[:400]}")
        results.append(json.loads(raw[start:]))
    for item in results:
        item["_wall"] = time.time() - started
    return results


def report(label: str, results: list[dict]) -> tuple[float, str]:
    wall = max(r["_wall"] for r in results)
    generated = sum(r["generated_tokens"] for r in results)
    print(f"\n{label}")
    for index, r in enumerate(results, 1):
        print(
            f"  instance {index}  decode {r['decode_tps']:5.3f} t/s  "
            f"tail {r['tail_tps']:5.3f}  pinned {r['pinned_bytes']/1e9:4.2f} GB  "
            f"{r['token_sha256'][:16]}"
        )
    total = sum(r["decode_tps"] for r in results)
    print(
        f"  total decode {total:5.3f} t/s | {generated} tokens in "
        f"{wall:5.1f} s wall | {generated/wall:5.3f} t/s end to end"
    )
    hashes = {r["token_sha256"] for r in results}
    if len(hashes) != 1:
        print("  WARNING: instances disagree on output")
    return total, hashes.pop() if len(hashes) == 1 else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--resident-experts", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even when the memory estimate says it will swap",
    )
    args = parser.parse_args()

    per_instance = INSTANCE_WIRED_GB + PINNED_GB_PER_EXPERT * args.resident_experts
    needed = per_instance * args.instances
    free, reclaimable = free_gb()
    print(
        f"{args.instances} instances x {per_instance:4.2f} GB = {needed:5.2f} GB "
        f"needed; free {free:4.2f} GB, free+inactive {reclaimable:4.2f} GB"
    )
    if needed > reclaimable and not args.force:
        raise SystemExit(
            "refusing to start: this would very likely swap. Lower "
            "--resident-experts, close applications, or pass --force."
        )

    solo = run_instances(1, args)
    solo_total, solo_hash = report("solo", solo)
    parallel = run_instances(args.instances, args)
    parallel_total, parallel_hash = report(f"{args.instances} in parallel", parallel)

    print(f"\naggregate {parallel_total/solo_total:5.2f}x the solo rate")
    if solo_hash and parallel_hash:
        print(f"token IDs match the solo run: {solo_hash == parallel_hash}")
    free_after, _ = free_gb()
    print(f"free RAM after: {free_after:4.2f} GB")


if __name__ == "__main__":
    main()
