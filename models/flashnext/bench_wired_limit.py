#!/usr/bin/env python3
"""Rigorous benchmark for Issue #43: Pre-load Metal Wired Memory Limit.

Compares FLASHNEXT_WIRED_GB=0 (baseline) vs FLASHNEXT_WIRED_GB=2 (2 GB wired),
with the limit set strictly BEFORE model instantiation.

Methodological controls:
1. Fresh backend instantiation per arm with explicit wired limit verification.
2. Interleaved reversed pairs ([w0, w2], [w2, w0]...) to eliminate thermal drift and cache bias.
3. Live physical disk telemetry via proc_pid_rusage (ReadMeter).
4. Deterministic token digest checking across conditions.
5. Paired statistical sign test and resolution band reporting.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from math import comb
import os
import statistics as st
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx

PROMPT = (
    "<|im_start|>user\nExplique a fotossintese em duas frases."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def run_one_arm(wired_gb_val: float, tokens: int) -> dict:
    os.environ["FLASHNEXT_WIRED_GB"] = str(wired_gb_val)
    os.environ["FLASHNEXT_METAL_RUNTIME"] = "1"
    os.environ["FLASHNEXT_SLAB"] = "0"

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext.diskio import ReadMeter, free_memory_mb
    from models.flashnext.loader import wired_gb, set_wired_gb

    # Strictly apply before any backend/model tensor allocations
    set_wired_gb(wired_gb_val)
    if wired_gb() != wired_gb_val:
        raise RuntimeError(f"wired_gb mismatch: expected {wired_gb_val}, got {wired_gb()}")

    meter = ReadMeter()
    free_before = free_memory_mb()
    backend = FlashNextBackend()

    backend.reset()
    backend.append_text(PROMPT)
    meter.reset()

    began = time.perf_counter()
    _text, stats = backend.generate(max_tokens=tokens)
    wall = time.perf_counter() - began
    phys_bytes = meter.bytes_since()

    ids = tuple(backend.tape[-stats.tokens:]) if stats.tokens else ()
    digest = hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest()[:16] if ids else "none"
    tail_rate = (stats.tail_tokens / stats.tail_seconds) if stats.tail_seconds else 0.0
    phys_mb_tok = (phys_bytes / stats.tokens / 1e6) if stats.tokens and phys_bytes > 0 else 0.0

    # Cleanup backend to ensure next arm starts with a fresh instance
    backend.store.close()
    del backend
    gc.collect()
    mx.metal.clear_cache()

    return {
        "wired_gb": wired_gb_val,
        "tokens": stats.tokens,
        "gen_rate": round(stats.rate, 3),
        "tail_rate": round(tail_rate, 3),
        "phys_mb_tok": round(phys_mb_tok, 1),
        "free_mb_before": round(free_before, 0),
        "digest": digest,
        "wall_s": round(wall, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=24, help="Tokens per arm")
    parser.add_argument("--pairs", type=int, default=4, help="Number of reversed pairs (total arms = pairs * 2)")
    parser.add_argument("--pause", type=float, default=3.0, help="Pause seconds between arms")
    args = parser.parse_args()

    print("=== Issue #43: Pre-Load Metal Wired Limit Comparison ===")
    print(f"Tokens per arm: {args.tokens}")
    print(f"Pairs: {args.pairs} (Total arms: {args.pairs * 2})")
    print(f"System load average: {os.getloadavg()}\n")

    # Interleaved reversed pair sequence
    cond_order = []
    for i in range(args.pairs):
        if i % 2 == 0:
            cond_order.extend([(0.0, "wired0"), (2.0, "wired2")])
        else:
            cond_order.extend([(2.0, "wired2"), (0.0, "wired0")])

    results = {"wired0": [], "wired2": []}
    pair_matches = []

    for idx, (w_gb, label) in enumerate(cond_order, 1):
        print(f"Arm {idx:2d}/{len(cond_order)}: Running {label} (wired={w_gb:.1f} GB)...", flush=True)
        res = run_one_arm(w_gb, args.tokens)
        results[label].append(res)
        print(
            f"  -> Gen: {res['gen_rate']:.2f} t/s | Tail: {res['tail_rate']:.2f} t/s | "
            f"Phys: {res['phys_mb_tok']:.1f} MB/tok | Free RAM: {res['free_mb_before']:.0f} MB | "
            f"Digest: {res['digest']}",
            flush=True,
        )
        if idx < len(cond_order):
            time.sleep(args.pause)

    # Paired evaluation
    w0_arms = results["wired0"]
    w2_arms = results["wired2"]

    w0_rates = [a["gen_rate"] for a in w0_arms]
    w2_rates = [a["gen_rate"] for a in w2_arms]
    w0_tails = [a["tail_rate"] for a in w0_arms]
    w2_tails = [a["tail_rate"] for a in w2_arms]
    w0_phys = [a["phys_mb_tok"] for a in w0_arms if a["phys_mb_tok"] > 0]
    w2_phys = [a["phys_mb_tok"] for a in w2_arms if a["phys_mb_tok"] > 0]

    w0_digest = w0_arms[0]["digest"] if w0_arms else "none"
    w2_digest = w2_arms[0]["digest"] if w2_arms else "none"

    print("\n=== Summary Results ===")
    print(f"{'Condition':<10} | {'Gen med':<8} | {'Range':<14} | {'Tail med':<9} | {'Phys MB/tok':<12} | {'Digest':<16}")
    print("-" * 75)
    print(
        f"{'wired0':<10} | {st.median(w0_rates):8.2f} | "
        f"{min(w0_rates):.2f}..{max(w0_rates):.2f}     | {st.median(w0_tails):9.2f} | "
        f"{st.median(w0_phys):12.1f} | {w0_digest:<16}"
    )
    print(
        f"{'wired2':<10} | {st.median(w2_rates):8.2f} | "
        f"{min(w2_rates):.2f}..{max(w2_rates):.2f}     | {st.median(w2_tails):9.2f} | "
        f"{st.median(w2_phys):12.1f} | {w2_digest:<16}"
    )

    pairs = list(zip(w0_rates, w2_rates))
    diffs_pct = [(y - x) / x * 100 for x, y in pairs]
    w2_wins = sum(1 for d in diffs_pct if d > 0)
    total_pairs = len(pairs)
    p_val = sum(comb(total_pairs, k) for k in range(w2_wins, total_pairs + 1)) / (2 ** total_pairs)

    med_diff_pct = st.median(diffs_pct)
    mean_diff_pct = st.mean(diffs_pct)

    # Resolution band
    span = max(w0_rates) - min(w0_rates)
    band = (span / st.median(w0_rates) * 100) if st.median(w0_rates) else 0.0

    print(f"\nPaired analysis over {total_pairs} pairs:")
    print(f"  Mean paired diff: {mean_diff_pct:+.1f}%")
    print(f"  Median paired diff: {med_diff_pct:+.1f}%")
    print(f"  wired2 ahead in {w2_wins} of {total_pairs} pairs, sign test p = {p_val:.3f}")
    print(f"  Resolution band: {band:.1f}%")
    if abs(med_diff_pct) < band:
        print("  -> Result is UNRESOLVED within the resolution band.")
    elif med_diff_pct > 0:
        print("  -> wired2 demonstrates a resolved improvement.")
    else:
        print("  -> wired2 demonstrates a resolved regression.")


if __name__ == "__main__":
    main()
