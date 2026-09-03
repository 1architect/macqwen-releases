#!/usr/bin/env python3
"""Unified resident slab sweep measuring physical MB/tok, hits, and resident memory.

Evaluates:
  SLAB=0 (baseline)
  SLAB=1, 2, 4 (all 48 layers)
  SLAB=4 across 8, 12, 16 selective layers

Calculates:
  Physical MB saved per token / Added Resident MB.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx

PROMPT = (
    "<|im_start|>user\nExplique a fotossintese em duas frases."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

DEFAULT_CONFIGS = [
    (0, 0, "SLAB=0 (baseline)"),
    (1, 48, "SLAB=1 (48 layers)"),
    (2, 48, "SLAB=2 (48 layers)"),
    (4, 48, "SLAB=4 (48 layers)"),
    (4, 8, "SLAB=4 (8 layers)"),
    (4, 12, "SLAB=4 (12 layers)"),
    (4, 16, "SLAB=4 (16 layers)"),
]


def run_arm(slab: int, layers: int, tokens: int) -> dict:
    os.environ["FLASHNEXT_METAL_RUNTIME"] = "1"
    os.environ["FLASHNEXT_SLAB"] = str(slab)
    os.environ["FLASHNEXT_SLAB_LAYERS"] = str(layers)

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext.diskio import ReadMeter

    backend = FlashNextBackend()
    meter = ReadMeter()

    # Reset and prepare prompt
    backend.reset()
    backend.append_text(PROMPT)
    meter.reset()

    # Generate
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

    # Cleanup cleanly to free memory before next condition
    backend.store.close()
    del backend
    gc.collect()
    mx.metal.clear_cache()

    return {
        "slab": slab,
        "layers": layers,
        "tokens": stats.tokens,
        "gen_rate": round(stats.rate, 3),
        "tail_rate": round(tail_rate, 3),
        "phys_mb_tok": round(phys_mb_tok, 1),
        "active_mb": round(active_bytes / 1e6, 1),
        "hits": hits,
        "misses": misses,
        "hit_pct": round(hit_pct, 1),
        "digest": digest,
        "wall_s": round(wall, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32, help="tokens per arm")
    parser.add_argument("--pause", type=int, default=4, help="seconds to pause between arms to cool machine")
    parser.add_argument("--json", default="", help="output json path")
    args = parser.parse_args()

    print("=== Unified Resident Slab Sweep ===")
    print(f"Tokens per arm: {args.tokens}")
    print(f"System load average: {os.getloadavg()}\n")

    results = []
    baseline_phys = None
    baseline_active = None

    for slab, layers, label in DEFAULT_CONFIGS:
        print(f"Running {label:<22} (SLAB={slab}, LAYERS={layers})...", flush=True)
        res = run_arm(slab, layers, args.tokens)
        res["label"] = label

        if baseline_phys is None:
            baseline_phys = res["phys_mb_tok"]
            baseline_active = res["active_mb"]
            res["phys_saved_mb_tok"] = 0.0
            res["added_active_mb"] = 0.0
            res["efficiency"] = 0.0
        else:
            saved = baseline_phys - res["phys_mb_tok"]
            added = res["active_mb"] - baseline_active
            res["phys_saved_mb_tok"] = round(saved, 1)
            res["added_active_mb"] = round(added, 1)
            res["efficiency"] = round(saved / added, 4) if added > 0 else 0.0

        results.append(res)
        print(
            f"  -> Gen: {res['gen_rate']} t/s | Tail: {res['tail_rate']} t/s | "
            f"Phys: {res['phys_mb_tok']} MB/tok (Saved: {res.get('phys_saved_mb_tok', 0):+.1f}) | "
            f"Hits: {res['hit_pct']}% | Active: {res['active_mb']} MB (+{res.get('added_active_mb', 0):.1f} MB) | "
            f"Digest: {res['digest']}",
            flush=True,
        )

        time.sleep(args.pause)

    print("\n=== Sweep Summary Table ===")
    header = (
        f"{'Configuration':<24} | {'Hit %':<6} | {'Phys MB/tok':<11} | {'Saved MB/t':<10} | "
        f"{'Active MB':<9} | {'Added MB':<8} | {'Saved/Added':<11} | {'Gen t/s':<7} | {'Digest':<16}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['label']:<24} | {r['hit_pct']:5.1f}% | {r['phys_mb_tok']:10.1f}  | "
            f"{r.get('phys_saved_mb_tok', 0):+9.1f}  | {r['active_mb']:8.1f}  | "
            f"{r.get('added_active_mb', 0):+7.1f}  | {r.get('efficiency', 0):10.4f}  | "
            f"{r['gen_rate']:6.2f}  | {r['digest']:<16}"
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {args.json}")


if __name__ == "__main__":
    main()
