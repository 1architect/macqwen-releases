#!/usr/bin/env python3
"""Test exact recovery of fast-routing tail experts held in RAM."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx
from transformers import AutoTokenizer

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import (
    FAST_LAYERS,
    mean_keeps,
    reset_keep_stats,
    set_layer_thresholds,
    set_renorm_blend,
    set_resident_experts,
    set_route_observer,
    set_threshold,
)
from models.flashnext.expert_cache import profile_totals, reset_profile
from models.flashnext.loader import load_streaming


MODEL = str(resolve_flashnext())
PROMPT = (
    "Explique em cerca de 200 palavras como a fotossíntese transforma luz "
    "solar em energia química."
)
PARTS = ("weight", "scales", "biases")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def fast_keep(scores, threshold):
    total = sum(scores)
    accumulated = 0.0
    for position, score in enumerate(scores):
        accumulated += score / total
        if accumulated >= threshold:
            return position + 1
    return len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--blend", type=float, default=0.1)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    model, _, store = load_streaming(
        MODEL, expert_capacity=0, verbose=True, keep_vision=False, use_mtp=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    language = model.language_model
    store._read_mode = os.environ.get("FLASHNEXT_READ", "pread")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]

    set_threshold(0.85)
    set_layer_thresholds({})
    set_renorm_blend(1.0)
    cache = language.make_cache()
    logits = language(ids, cache=cache).logits
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)

    candidates = {layer: Counter() for layer in range(48)}

    def collect(layer, experts, scores, keeps):
        threshold = 0.40 if layer in FAST_LAYERS else 0.20
        for expert_row, score_row, normal_keep in zip(experts, scores, keeps):
            if args.exact:
                mass = sum(score_row)
                for expert, score in zip(
                    expert_row[:normal_keep], score_row[:normal_keep]
                ):
                    candidates[layer][expert] += score / mass
                continue
            keep = fast_keep(score_row, threshold)
            mass = sum(score_row)
            for expert, score in zip(expert_row[keep:], score_row[keep:]):
                candidates[layer][expert] += score / mass

    set_route_observer(collect)
    values = []
    decode_began = time.perf_counter()
    for _ in range(args.warmup):
        values.append(int(token.item()))
        logits = language(token[None], cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
    set_route_observer(None)

    pool = [len(v) for v in candidates.values()]
    resident = {
        layer: {expert for expert, _ in values.most_common(args.hot)}
        for layer, values in candidates.items()
    }
    got = [len(v) for v in resident.values()]
    print(
        f"candidates/layer: min {min(pool)} mean {sum(pool)/len(pool):.1f} max {max(pool)}"
        f"   pinned/layer: mean {sum(got)/len(got):.1f} of {args.hot} asked",
        flush=True,
    )
    pinned = 0
    pin_started = time.perf_counter()
    for layer_number, experts in resident.items():
        block = language.model.layers[layer_number].mlp.switch_mlp
        prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
        for projection in PROJECTIONS:
            for part in PARTS:
                pinned += store.pin_rows(
                    f"{prefix}.{projection}.{part}", sorted(experts)
                )
    pin_elapsed = time.perf_counter() - pin_started

    if args.exact:
        store._read_mode = os.environ.get("FLASHNEXT_READ", "pread")
        set_threshold(0.85)
        set_layer_thresholds({})
        set_renorm_blend(1.0)
    else:
        set_resident_experts(resident)
        store._read_mode = "shared_mmap"
        set_threshold(0.20)
        set_layer_thresholds({layer: 0.40 for layer in FAST_LAYERS})
        set_renorm_blend(args.blend)
    reset_keep_stats()
    reset_profile()

    began = time.perf_counter()
    for _ in range(args.tokens):
        values.append(int(token.item()))
        logits = language(token[None], cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
    elapsed = time.perf_counter() - began
    complete_elapsed = time.perf_counter() - decode_began
    set_resident_experts(None)
    store.unpin_all()

    print(f"pinned: {pinned / 1e9:.2f} GB", flush=True)
    print(f"pin time: {pin_elapsed:.2f}s", flush=True)
    print(f"experts: {mean_keeps():.2f}", flush=True)
    print(
        "complete decode rate: "
        f"{(args.warmup + args.tokens) / complete_elapsed:.2f} tok/s",
        flush=True,
    )
    print(f"pinned tail rate: {args.tokens / elapsed:.2f} tok/s", flush=True)
    totals = profile_totals()
    if totals.get("io_calls"):
        per = args.tokens
        for key in sorted(totals):
            if key == "io_calls":
                print(f"prof {key}: {totals[key]}", flush=True)
            else:
                print(f"prof {key}: {totals[key] / per * 1000:.2f} ms/token", flush=True)
    # A rate is only comparable against a run that produced the same tokens.
    # Print a digest so two runs can be compared without diffing prose.
    digest = hashlib.sha256(
        ",".join(str(value) for value in values).encode()
    ).hexdigest()[:16]
    print(f"tokens: {len(values)}  sha256: {digest}", flush=True)
    print(tokenizer.decode(values), flush=True)


if __name__ == "__main__":
    main()
