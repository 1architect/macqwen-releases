#!/usr/bin/env python3
"""Compare the retained adaptive-routing profiles without reloading."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("FLASHNEXT_RENORM", "0")

import mlx.core as mx
from transformers import AutoTokenizer

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import (
    mean_keeps,
    reset_keep_stats,
    set_layer_thresholds,
    set_renorm_blend,
    set_threshold,
)
from models.flashnext.loader import load_streaming


MODEL = str(resolve_flashnext())
PROMPTS = (
    "2 + 2 =",
    "The chemical symbol for gold is",
    "The capital of France is",
)
QUALITY_PROMPTS = PROMPTS + (
    "7 * 8 =",
    "The capital of Japan is",
    "Water freezes at",
    "The largest planet in the Solar System is",
    "The author of 1984 was",
    "Complete this Python expression: len([1, 2, 3]) =",
    "Translate 'hello' to Portuguese:",
)
PROFILES = {
    "pread": (0.85, {}, 1.0, "pread"),
    "shared-mmap": (0.85, {}, 1.0, "shared_mmap"),
    "hybrid": (0.85, {}, 1.0, "hybrid"),
    "mixed": (0.85, {}, 1.0, "mixed"),
}


def generate(language, tokenizer, prompt, count, chat):
    text = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if chat
        else prompt
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]
    cache = language.make_cache()
    token = mx.argmax(language(ids, cache=cache).logits[:, -1, :], axis=-1)
    mx.eval(token)
    reset_keep_stats()
    values = []
    began = time.time()
    for _ in range(count):
        values.append(int(token.item()))
        token = mx.argmax(
            language(token[None], cache=cache).logits[:, -1, :], axis=-1
        )
        mx.eval(token)
    elapsed = time.time() - began
    return tokenizer.decode(values), count / elapsed, mean_keeps()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES)
    parser.add_argument("--prompt")
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    model, _, store = load_streaming(
        MODEL, expert_capacity=0, verbose=True, keep_vision=False, use_mtp=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    language = model.language_model
    store.set_mmap_advice("random")
    prompts = (args.prompt,) if args.prompt else (
        QUALITY_PROMPTS if args.quality else PROMPTS
    )
    selected = tuple(args.profiles or PROFILES)
    orders = (selected, tuple(reversed(selected)))[: args.rounds]

    for round_number, order in enumerate(orders, 1):
        print(f"\nROUND {round_number}", flush=True)
        for name in order:
            threshold, layers, blend, read_mode = PROFILES[name]
            set_threshold(threshold)
            set_layer_thresholds(layers)
            set_renorm_blend(blend)
            store._read_mode = read_mode
            rates = []
            print(f"\n{name}", flush=True)
            for prompt in prompts:
                text, rate, keeps = generate(
                    language,
                    tokenizer,
                    prompt,
                    args.tokens,
                    args.chat,
                )
                rates.append(rate)
                print(
                    f"{rate:.2f} tok/s | {keeps:.2f} experts | "
                    f"{prompt!r} -> {text!r}",
                    flush=True,
                )
            print(f"MEAN {sum(rates) / len(rates):.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
