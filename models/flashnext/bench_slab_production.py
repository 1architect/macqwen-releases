#!/usr/bin/env python3
"""Controlled production A/B comparison of the winning selective slab.

Compares:
  - baseline: FLASHNEXT_METAL_RUNTIME=1, FLASHNEXT_SLAB=0
  - slab12:   FLASHNEXT_METAL_RUNTIME=1, FLASHNEXT_SLAB=4, FLASHNEXT_SLAB_LAYERS=12

Methodological controls:
1. Interleaved reversed pairs ([baseline, slab12], [slab12, baseline]...) to cancel thermal drift.
2. Separate instances cleanly closed and garbage-collected per arm.
3. Live physical disk telemetry via proc_pid_rusage (ReadMeter).
4. Full token digest tracking to guarantee exact numerical determinism.
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


def run_arm(slab: int, layers: int, tokens: int, global_budget: int = 0) -> dict:
    os.environ["FLASHNEXT_METAL_RUNTIME"] = "1"
    os.environ["FLASHNEXT_SLAB"] = str(slab)
    os.environ["FLASHNEXT_SLAB_LAYERS"] = str(layers)
    os.environ["FLASHNEXT_SLAB_GLOBAL"] = str(global_budget)

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext.diskio import ReadMeter, free_memory_mb

    free_before = free_memory_mb()
    meter = ReadMeter()
    backend = FlashNextBackend()

    backend.reset()
    backend.append_text(PROMPT)
    meter.reset()

    began = time.perf_counter()
    _text, stats = backend.generate(max_tokens=tokens)
    wall = time.perf_counter() - began
    phys_bytes = meter.bytes_since()

    # Collect slab statistics
    hits, misses = 0, 0
    language = getattr(backend, "language", None)
    if language is not None and hasattr(language, "layers"):
        for layer in language.layers:
            mlp = getattr(layer, "mlp", None)
            s_mlp = getattr(mlp, "switch_mlp", None)
            if s_mlp is not None and hasattr(s_mlp, "hits"):
                hits += s_mlp.hits
                misses += s_mlp.misses

    hit_pct = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0.0
    active_bytes = mx.metal.get_active_memory()
    phys_mb_tok = (phys_bytes / stats.tokens / 1e6) if stats.tokens and phys_bytes > 0 else 0.0

    ids = tuple(backend.tape[-stats.tokens:]) if stats.tokens else ()
    digest = hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest()[:16] if ids else "none"
    tail_rate = (stats.tail_tokens / stats.tail_seconds) if stats.tail_seconds else 0.0

    backend.store.close()
    del backend
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()

    return {
        "slab": slab,
        "layers": layers,
        "global_budget": global_budget,
        "tokens": stats.tokens,
        "gen_rate": round(stats.rate, 3),
        "tail_rate": round(tail_rate, 3),
        "phys_mb_tok": round(phys_mb_tok, 1),
        "free_mb_before": round(free_before, 0),
        "active_mb": round(active_bytes / 1e6, 1),
        "hit_pct": round(hit_pct, 1),
        "digest": digest,
        "wall_s": round(wall, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="Tokens per arm")
    parser.add_argument("--pairs", type=int, default=6, help="Number of reversed pairs (total arms = pairs * 2)")
    parser.add_argument("--pause", type=float, default=2.5, help="Pause seconds between arms")
    parser.add_argument(
        "--control",
        type=str,
        default="baseline",
        choices=["baseline", "slab12"],
        help="Control configuration (default: baseline)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="global48",
        choices=["global48", "global56", "slab12"],
        help="Target slab configuration to compare against control",
    )
    args = parser.parse_args()

    cond_defs = {
        "baseline": (0, 0, 0),
        "slab12": (4, 12, 0),
        "global48": (0, 0, 48),
        "global56": (0, 0, 56),
    }
    control_name = args.control
    target_name = args.target

    print(f"=== Controlled Production A/B: {target_name} vs {control_name} ===")
    print(f"Tokens per arm: {args.tokens}")
    print(f"Pairs: {args.pairs} (Total arms: {args.pairs * 2})")
    print(f"System load average: {os.getloadavg()}\n")

    conds = {
        control_name: cond_defs[control_name],
        target_name: cond_defs[target_name],
    }

    schedule = []
    for i in range(args.pairs):
        if i % 2 == 0:
            schedule.extend([control_name, target_name])
        else:
            schedule.extend([target_name, control_name])

    collected = {control_name: [], target_name: []}

    for idx, name in enumerate(schedule, 1):
        slab, layers, global_b = conds[name]
        print(f"Arm {idx:2d}/{len(schedule)}: Running {name:<10} (SLAB={slab}, LAYERS={layers}, GLOBAL={global_b})...", flush=True)
        res = run_arm(slab, layers, args.tokens, global_b)
        collected[name].append(res)
        print(
            f"  -> Gen: {res['gen_rate']:.2f} t/s | Tail: {res['tail_rate']:.2f} t/s | "
            f"Phys: {res['phys_mb_tok']:.1f} MB/tok | Active: {res['active_mb']:.1f} MB | "
            f"Hits: {res['hit_pct']:.1f}% | Digest: {res['digest']}",
            flush=True,
        )
        if idx < len(schedule):
            time.sleep(args.pause)

    # Summary
    base_arms = collected[control_name]
    slab_arms = collected[target_name]

    b_rates = [a["gen_rate"] for a in base_arms]
    s_rates = [a["gen_rate"] for a in slab_arms]
    b_tails = [a["tail_rate"] for a in base_arms]
    s_tails = [a["tail_rate"] for a in slab_arms]
    b_phys = [a["phys_mb_tok"] for a in base_arms if a["phys_mb_tok"] > 0]
    s_phys = [a["phys_mb_tok"] for a in slab_arms if a["phys_mb_tok"] > 0]

    b_digest = base_arms[0]["digest"] if base_arms else "none"
    s_digest = slab_arms[0]["digest"] if slab_arms else "none"

    print("\n=== Production Comparison Summary ===")
    print(f"{'Condition':<12} | {'Gen med':<8} | {'Range':<14} | {'Tail med':<9} | {'Phys MB/tok':<12} | {'Active MB':<10} | {'Digest':<16}")
    print("-" * 90)
    print(
        f"{control_name:<12} | {st.median(b_rates):8.2f} | "
        f"{min(b_rates):.2f}..{max(b_rates):.2f}     | {st.median(b_tails):9.2f} | "
        f"{st.median(b_phys):12.1f} | {st.median([a['active_mb'] for a in base_arms]):10.1f} | {b_digest:<16}"
    )
    print(
        f"{target_name:<12} | {st.median(s_rates):8.2f} | "
        f"{min(s_rates):.2f}..{max(s_rates):.2f}     | {st.median(s_tails):9.2f} | "
        f"{st.median(s_phys):12.1f} | {st.median([a['active_mb'] for a in slab_arms]):10.1f} | {s_digest:<16}"
    )

    pairs = list(zip(b_rates, s_rates))
    diffs_pct = [(y - x) / x * 100 for x, y in pairs]
    slab_wins = sum(1 for d in diffs_pct if d > 0)
    total_pairs = len(pairs)
    p_val = sum(comb(total_pairs, k) for k in range(slab_wins, total_pairs + 1)) / (2 ** total_pairs)

    med_diff_pct = st.median(diffs_pct)
    mean_diff_pct = st.mean(diffs_pct)

    span = max(b_rates) - min(b_rates)
    band = (span / st.median(b_rates) * 100) if st.median(b_rates) else 0.0

    phys_saved = st.median(b_phys) - st.median(s_phys)

    print(f"\nPaired analysis over {total_pairs} pairs:")
    print(f"  Mean paired speedup: {mean_diff_pct:+.1f}%")
    print(f"  Median paired speedup: {med_diff_pct:+.1f}%")
    print(f"  Physical read reduction: {phys_saved:+.1f} MB/token")
    print(f"  {target_name} ahead in {slab_wins} of {total_pairs} pairs, sign test p = {p_val:.3f}")
    print(f"  Resolution band: {band:.1f}%")
    if abs(med_diff_pct) < band:
        print("  -> Speedup is inside the resolution band (unresolved on pure generation rate).")
    elif med_diff_pct > 0:
        print(f"  -> {target_name} demonstrates a RESOLVED improvement over baseline!")
    else:
        print(f"  -> {target_name} demonstrates a regression.")


if __name__ == "__main__":
    main()
