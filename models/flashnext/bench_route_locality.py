#!/usr/bin/env python3
"""Measure how much a reply's expert routing moves around, by prompt style.

Throughput on this machine depends on what the model generates: a focused
technical explanation runs about 23% faster than open-ended output at an
identical expert count per layer. This probe tests why. It records the routed
expert set for every layer on every generated token, then reports:

  reuse      fraction of a token's experts that the previous token also used
  distinct   experts touched across the whole reply, per layer

High reuse and few distinct experts mean a narrow working set that residency
can hold. Low reuse and many distinct experts mean the reply keeps walking
into new territory, which no pinning policy can fix.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx

DEFAULT_MODEL = "~/models/Qwen3.8-Flash-Next-MLX-oQ4"
# 3.0 tok/s needs about 75.3% coverage; 32 experts give 62.7 to 69.6%
SIZES = (8, 16, 32, 48, 64, 96, 128)
PROMPTS = {
    "focused": (
        "Explique em detalhe como funciona a fotossintese, "
        "incluindo as fases clara e escura."
    ),
    "open-ended": (
        "this is an energy efficiency test on a notebook just keep talking"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=120)
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")
    path = os.path.expanduser(args.model)

    from transformers import AutoTokenizer
    from models.flashnext.adaptive_topk import set_route_observer
    from models.flashnext.loader import load_streaming

    model, _, _ = load_streaming(path, expert_capacity=0, verbose=False,
                                 keep_vision=False, use_mtp=False)
    tokenizer = AutoTokenizer.from_pretrained(path)
    language = model.language_model

    for label, prompt in PROMPTS.items():
        history: dict[int, list[set]] = {}

        def observe(layer, experts, scores, keeps):
            for row, keep in zip(experts, keeps):
                history.setdefault(layer, []).append(set(row[:keep]))

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = mx.array(tokenizer(text)["input_ids"])[None]
        language._position_ids = None
        language._rope_deltas = None
        cache = language.make_cache()
        out = language(ids, cache=cache)
        token = mx.argmax(out.logits[:, -1, :], axis=-1)
        mx.eval(token)
        out = None
        mx.clear_cache()

        set_route_observer(observe)
        try:
            for _ in range(args.tokens):
                step = language(token[None], cache=cache)
                token = mx.argmax(step.logits[:, -1, :], axis=-1)
                mx.eval(token)
        finally:
            set_route_observer(None)

        from collections import Counter

        reuse, distinct, per_token, coverage = [], [], [], []
        for layer, seq in history.items():
            # the prefill contributes many rows; keep only the decode tail
            seq = seq[-args.tokens:]
            if len(seq) < 2:
                continue
            for previous, current in zip(seq, seq[1:]):
                if current:
                    reuse.append(len(previous & current) / len(current))
            distinct.append(len(set().union(*seq)))
            per_token.append(st.mean(len(s) for s in seq))
            # what share of routed slots do the N most-used experts hold?
            counts = Counter()
            for s in seq:
                counts.update(s)
            total = sum(counts.values())
            ranked = [n for _, n in counts.most_common()]
            row = {}
            for size in SIZES:
                row[size] = sum(ranked[:size]) / total if total else 0.0
            coverage.append(row)
        curve = " ".join(
            f"{size}:{st.mean(r[size] for r in coverage):5.1%}" for size in SIZES
        )
        print(f"{label:<12} reuse {st.mean(reuse):5.1%} | distinct/layer "
              f"{st.mean(distinct):5.1f}", flush=True)
        print(f"{'':<12} coverage by pinned experts/layer: {curve}", flush=True)
        gb = {size: size * 48 * 3.07 / 1024 for size in SIZES}
        print(f"{'':<12} pinned GB:                        "
              + " ".join(f"{size}:{gb[size]:5.2f}" for size in SIZES), flush=True)


if __name__ == "__main__":
    main()
