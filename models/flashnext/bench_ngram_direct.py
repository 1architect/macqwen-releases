#!/usr/bin/env python3
"""Compare legacy all-shard n-gram reads with direct shard dispatch."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx
from transformers import AutoTokenizer

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import set_renorm_blend, set_threshold
from models.flashnext.loader import load_streaming


MODEL = str(resolve_flashnext())
PROMPT = "Explain photosynthesis in six detailed sentences."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--order",
        nargs="+",
        choices=("legacy", "direct"),
        default=("legacy", "direct", "direct", "legacy"),
    )
    args = parser.parse_args()

    model, _, store = load_streaming(
        MODEL, expert_capacity=0, verbose=True, keep_vision=False, use_mtp=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    language = model.language_model
    table = language.model.layers[1].ple.ple_embedding.ngram_embedding
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]
    store._read_mode = "pread"
    store.set_mmap_advice("random")
    set_threshold(0.85)
    set_renorm_blend(1.0)

    for name in args.order:
        table.direct = name == "direct"
        cache = language.make_cache()
        prefill_began = time.perf_counter()
        logits = language(ids, cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        prefill_elapsed = time.perf_counter() - prefill_began
        values = []
        began = time.perf_counter()
        for _ in range(args.tokens):
            values.append(int(token.item()))
            logits = language(token[None], cache=cache).logits
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
        elapsed = time.perf_counter() - began
        raw = b"".join(
            value.to_bytes(8, "little", signed=True) for value in values
        )
        print(
            f"{name}: {args.tokens / elapsed:.3f} tok/s, "
            f"prefill {ids.size / prefill_elapsed:.3f} tok/s, "
            f"sha256 {hashlib.sha256(raw).hexdigest()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
