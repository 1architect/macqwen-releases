#!/usr/bin/env python3
"""What does the target lose when a draft model runs in a second process?

Speculative decoding was rejected four times here. Every rejection has the
same sentence attached: the draft and the verifier run sequentially, so their
combined cost exceeds one exact decode. The anchored depth-2 external draft
reached 92 percent acceptance and tied the control at 1.747 against 1.749
tok/s. It paid for exactly what it saved.

That is a scheduling result, not a quality result. Two independent processes
reached 1.52x aggregate on this machine, at a measured exchange rate of about
31 percent of the GPU for 52 percent more work. If the draft runs beside the
target instead of in front of it, its cost moves into the window that
exchange rate prices.

This benchmark measures the one number that decides whether the plumbing is
worth building: the fraction of its solo rate the target keeps while a draft
process runs. Nothing here drafts or verifies. It measures contention only.

    ./models/flashnext/bench_draft_contention.py --arms 3

Read `docs/flashnext/handoff.md` before changing the arm count or the order.
Arms alternate so that drift reaches both conditions equally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PYTHON = os.path.expanduser("~/models/.venv-qwen4exp/bin/python")
PAGE = 16384
DEFAULT_PROMPT = (
    "Explique em detalhe como funciona a fotossintese, "
    "incluindo as fases clara e escura."
)


def free_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "Pages free":
            return int(value.strip().rstrip(".")) * PAGE / 1e9
    return 0.0


def start_draft(args):
    """Start the draft process and wait until its weights are loaded."""
    proc = subprocess.Popen(
        [PYTHON, "-u", os.path.join(ROOT, "models", "flashnext", "draft_worker.py"),
         "--model", args.draft_model,
         "--block", str(args.draft_block),
         "--duty", str(args.duty)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    line = proc.stdout.readline().decode(errors="replace")
    if '"ready"' not in line:
        proc.kill()
        raise SystemExit(f"draft worker did not start: {line[:200]}")
    return proc


def stop_draft(proc) -> dict:
    proc.terminate()
    rest = proc.communicate(timeout=60)[0].decode(errors="replace")
    for line in reversed(rest.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {}


def run_target(args) -> dict:
    """One target arm, in its own process, exactly as the chat runs it."""
    command = [
        PYTHON, "-u", os.path.join(ROOT, "macqwen", "session.py"),
        "--model", "flashnext",
        "--profile", "plain",
        "--exact-quality",
        "--resident-experts", str(args.resident_experts),
        "--max-tokens", str(args.tokens),
        "--benchmark-json",
        "--benchmark-prompt", args.prompt,
    ]
    raw = subprocess.run(
        command, capture_output=True,
    ).stdout.decode(errors="replace")
    start = raw.find('{"profile"')
    if start < 0:
        raise SystemExit(f"target produced no JSON:\n{raw[-600:]}")
    return json.loads(raw[start:])


def band(a: list[float], b: list[float], base: float) -> float:
    """Two standard errors of the difference between the medians, percent."""
    if len(a) < 2 or len(b) < 2 or not base:
        return float("inf")
    spread = math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))
    return 2 * spread / base * 100


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    wins = sum(1 for d in diffs if d < 0)
    total = len(diffs)
    tail = sum(math.comb(total, k) for k in range(wins, total + 1)) / 2 ** total
    return wins, total, tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=int, default=3,
                        help="arms per condition; the rules ask for three")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--resident-experts", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--draft-model", default="~/models/Qwen3.5-0.8B-MLX-4bit")
    parser.add_argument("--draft-block", type=int, default=2,
                        help="tokens per draft generation")
    parser.add_argument("--duty", type=float, default=1.0,
                        help="draft duty cycle; 1.0 is the worst case")
    args = parser.parse_args()

    model = os.path.expanduser(args.draft_model)
    if not os.path.isdir(model):
        raise SystemExit(f"draft model not found: {model}")

    print(f"free RAM before {free_gb():4.2f} GB, draft duty {args.duty:.2f}, "
          f"block {args.draft_block}, {args.arms} arms per condition",
          flush=True)
    print("\n  arm  condition   gen     tail    MB/tok   free GB", flush=True)

    solo, beside, drafts = [], [], []
    ids = set()
    for index in range(args.arms * 2):
        # Alternate, starting solo, so drift reaches both conditions equally.
        contended = index % 2 == 1
        draft = start_draft(args) if contended else None
        began = time.time()
        result = run_target(args)
        if draft is not None:
            drafts.append(stop_draft(draft))
        ids.add(result["token_sha256"])
        (beside if contended else solo).append(result)
        print(
            f"  {index + 1:>3}  {'beside' if contended else 'solo':<10} "
            f"{result['decode_tps']:5.3f}  {result['tail_tps']:5.3f}  "
            f"{result.get('mb_per_token', -1):7.1f}  {free_gb():6.2f}"
            f"   ({time.time() - began:4.1f}s)",
            flush=True,
        )

    solo_rates = [r["decode_tps"] for r in solo]
    beside_rates = [r["decode_tps"] for r in beside]
    solo_median = st.median(solo_rates)
    beside_median = st.median(beside_rates)
    retention = beside_median / solo_median if solo_median else 0.0
    resolution = band(solo_rates, beside_rates, solo_median)

    print(f"\n  solo    median {solo_median:5.3f} t/s  "
          f"range {min(solo_rates):.3f}-{max(solo_rates):.3f}")
    print(f"  beside  median {beside_median:5.3f} t/s  "
          f"range {min(beside_rates):.3f}-{max(beside_rates):.3f}")
    print(f"  the target keeps {retention * 100:4.1f} percent of its solo rate")
    print(f"  this run resolves differences above {resolution:.1f} percent")

    pairs = list(zip(solo_rates, beside_rates))
    diffs = [(b - a) / a * 100 for a, b in pairs]
    if len(diffs) >= 3:
        wins, total, tail = sign_test(diffs)
        print(f"  paired: mean {st.mean(diffs):+.1f} percent, "
              f"target slower in {wins} of {total} pairs, sign test p = {tail:.3f}")

    if drafts:
        rates = [d.get("draft_tps", 0.0) for d in drafts if d]
        if rates:
            print(f"  draft produced {st.median(rates):5.1f} tok/s "
                  f"while the target ran")

    if len(ids) == 1:
        print(f"  target token IDs identical across every arm: {ids.pop()[:16]}")
    else:
        print(f"  WARNING: target output changed across arms ({len(ids)} hashes)")

    # A benchmark must prove its own premise before it reports. One arm per
    # condition cannot resolve anything, and a cold machine reading four times
    # the baseline volume is not measuring the same system.
    refusals = []
    if args.arms < 3:
        refusals.append(
            f"only {args.arms} arm(s) per condition; the rules ask for three"
        )
    if not math.isfinite(resolution) or resolution > 20.0:
        refusals.append(
            f"the band is {resolution:.1f} percent, wider than any plausible effect"
        )
    volumes = [r.get("mb_per_token", -1) for r in solo + beside]
    volumes = [v for v in volumes if v > 0]
    if volumes and st.median(volumes) > 600:
        refusals.append(
            f"{st.median(volumes):.0f} MB/token against a 390 MB baseline, so "
            "the page cache was cold and this is not the production system"
        )
    if refusals:
        print("\n  NO RESULT. This run cannot support a projection:")
        for reason in refusals:
            print(f"    - {reason}")
        print(
            "  Close unrelated applications, wait for the VM quiescence gate, "
            "and re-run with --arms 3 or more."
        )
        return

    # The decision this benchmark exists to inform.
    print("\n  What this means for off-process speculation, at 92 percent")
    print("  acceptance on a depth-2 anchored draft and a verify block that")
    print("  costs about 1.15 times one decode:")
    committed = 1.92
    block_cost = 1.15
    if retention:
        projected = committed / (block_cost / retention)
        print(f"    rate = {committed:.2f} / ({block_cost:.2f} / {retention:.3f})"
              f" = {projected:.2f} x solo")
        print(f"    on a 2.713 tok/s baseline that is {2.713 * projected:.2f} tok/s")
    print("  Those two constants are assumptions, not measurements from this")
    print("  run. Only the retention figure above is measured here.")


if __name__ == "__main__":
    main()
