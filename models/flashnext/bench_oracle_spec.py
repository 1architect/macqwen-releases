#!/usr/bin/env python3
"""Measure the zero-cost-draft upper bound for exact speculative decoding."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("FLASHNEXT_RENORM", "1")
os.environ.setdefault("FLASHNEXT_DLPACK", "1")
os.environ.setdefault("FLASHNEXT_OVERLAP", "1")

import mlx.core as mx
from transformers import AutoTokenizer

from models.flashnext.adaptive_topk import set_renorm_blend, set_threshold
from models.flashnext.loader import load_streaming


MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-oQ4")
PROMPT = "Explain photosynthesis in six detailed sentences."


def input_ids(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return mx.array(tokenizer(text)["input_ids"])[None]


def greedy_tokens(language, ids, count):
    cache = language.make_cache()
    logits = language(ids, cache=cache).logits
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    values = []
    for _ in range(count):
        values.append(int(token.item()))
        logits = language(token[None], cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
    return values


def verify(language, ids, values, block_size, exact_verifier=False, argmax_only=False):
    cache = language.make_cache()
    logits = language(ids, cache=cache).logits
    mx.eval(logits)
    results = []
    began = time.perf_counter()
    for offset in range(0, len(values), block_size):
        block = mx.array(values[offset : offset + block_size])[None]
        if exact_verifier:
            out = language(
                block,
                cache=cache,
                speculative_verify=True,
                return_hidden=argmax_only,
                skip_logits=argmax_only,
            )
            if argmax_only:
                result = language.speculative_argmax_from_hidden(
                    out.hidden_states[-1]
                )
            else:
                result = out.logits
        else:
            result = language(block, cache=cache).logits
        mx.eval(result)
        if exact_verifier:
            results.append(mx.argmax(result, axis=-1) if not argmax_only else result)
    elapsed = time.perf_counter() - began
    if exact_verifier:
        observed = []
        for result in results:
            observed.extend(int(value) for value in result[0].tolist())
        if observed[:-1] != values[1:]:
            mismatch = next(
                index
                for index, pair in enumerate(zip(observed[:-1], values[1:]))
                if pair[0] != pair[1]
            )
            raise AssertionError(
                f"verifier mismatch at {mismatch + 1}: "
                f"{observed[mismatch]} != {values[mismatch + 1]}"
            )
    return len(values) / elapsed, elapsed


def verify_one_block_then_tail(language, ids, values, block_size):
    """Verify one exact block, then finish with normal singleton decode."""
    block_size = min(int(block_size), len(values))
    if block_size < 1:
        raise ValueError("one-block-tail needs at least one token per block")

    cache = language.make_cache()
    logits = language(ids, cache=cache).logits
    mx.eval(logits)
    first = int(mx.argmax(logits[:, -1, :], axis=-1).item())

    block = mx.array(values[:block_size], dtype=mx.uint32)[None]
    block_started = time.perf_counter()
    out = language(
        block,
        cache=cache,
        speculative_verify=True,
        return_hidden=True,
        skip_logits=True,
    )
    target_ids = language.speculative_argmax_from_hidden(
        out.hidden_states[-1]
    )[0].astype(mx.uint32)
    mx.eval(target_ids)
    block_elapsed = time.perf_counter() - block_started

    targets = [int(value) for value in target_ids.tolist()]
    expected_targets = values[1 : min(len(values), block_size + 1)]
    if targets[: len(expected_targets)] != expected_targets:
        mismatch = next(
            index
            for index, pair in enumerate(
                zip(targets[: len(expected_targets)], expected_targets)
            )
            if pair[0] != pair[1]
        )
        raise AssertionError(
            f"block mismatch at {mismatch + 1}: "
            f"{targets[mismatch]} != {expected_targets[mismatch]}"
        )

    observed = [first, *targets[: block_size - 1]]
    tail_started = time.perf_counter()
    if len(observed) < len(values):
        token = mx.array([targets[-1]], dtype=mx.uint32)
        while len(observed) < len(values):
            observed.append(int(token.item()))
            step = language(token.reshape(1, 1), cache=cache).logits
            token = mx.argmax(step[:, -1, :], axis=-1).astype(mx.uint32)
            mx.eval(token)
    tail_elapsed = time.perf_counter() - tail_started

    if observed != values:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(observed, values))
                if pair[0] != pair[1]
            ),
            min(len(observed), len(values)),
        )
        actual = observed[mismatch] if mismatch < len(observed) else "missing"
        expected = values[mismatch] if mismatch < len(values) else "missing"
        raise AssertionError(
            f"whole mismatch at {mismatch}: {actual} != {expected}"
        )

    tail_tokens = len(values) - block_size
    whole_elapsed = block_elapsed + tail_elapsed
    block_rate = block_size / block_elapsed
    tail_rate = tail_tokens / tail_elapsed if tail_tokens else 0.0
    whole_rate = len(values) / whole_elapsed
    print(
        f"one block: {block_size} tok em {block_elapsed:.2f}s = "
        f"{block_rate:.2f} tok/s",
        flush=True,
    )
    print(
        f"exact tail: {tail_tokens} tok em {tail_elapsed:.2f}s = "
        f"{tail_rate:.2f} tok/s",
        flush=True,
    )
    print(
        f"whole: {len(values)} tok em {whole_elapsed:.2f}s = "
        f"{whole_rate:.2f} tok/s",
        flush=True,
    )
    print("tokens: identical", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--blocks", type=int, nargs="+", default=(8, 4, 2, 1))
    parser.add_argument(
        "--sort-order",
        nargs="+",
        choices=("off", "on"),
        help="A/B sequence for physically sorted expert reads",
    )
    parser.add_argument("--exact-verifier", action="store_true")
    parser.add_argument("--argmax-only", action="store_true")
    parser.add_argument(
        "--one-block-tail",
        action="store_true",
        help="verify one exact block, then use normal exact singleton decode",
    )
    args = parser.parse_args()

    model, _, store = load_streaming(
        MODEL, expert_capacity=0, verbose=True, keep_vision=False, use_mtp=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    language = model.language_model
    if args.exact_verifier or args.one_block_tail:
        from mlx_vlm.models.qwen3_5 import language as qwen35_language

        from models.flashnext.qwen4_verifier import Qwen4ExactSpeculativeVerifier

        qwen35_language._EXACT_SPECULATIVE_VERIFIER = Qwen4ExactSpeculativeVerifier()
    store._read_mode = "hybrid"
    store.set_mmap_advice("random")
    set_threshold(0.85)
    set_renorm_blend(1.0)

    ids = input_ids(tokenizer)
    values = greedy_tokens(language, ids, args.tokens)
    print(f"oracle: {tokenizer.decode(values)!r}", flush=True)
    if args.one_block_tail:
        verify_one_block_then_tail(
            language,
            ids,
            values,
            args.blocks[0],
        )
        return
    modes = args.sort_order
    runs = (
        [(args.blocks[0], value) for value in modes]
        if modes
        else [(size, None) for size in args.blocks]
    )
    for size, mode in runs:
        sort_mode = mode if args.sort_order else None
        if sort_mode is not None:
            store._sort_reads = sort_mode == "on"
        rate, elapsed = verify(
            language,
            ids,
            values,
            size,
            exact_verifier=args.exact_verifier,
            argmax_only=args.argmax_only,
        )
        label = f"block {size}"
        if sort_mode is not None:
            label += f", sort {sort_mode}"
        print(
            f"{label}: {args.tokens} tok em {elapsed:.2f}s = {rate:.2f} tok/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
