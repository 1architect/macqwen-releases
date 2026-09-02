#!/usr/bin/env python3
"""Does timing one layer repeatedly measure what 48 different layers cost?

`bench_layer_split.py` times a component by calling one instance many times in
a row. That is how every microbenchmark here works, and it has an assumption
buried in it: that a layer costs the same whether its weights were touched a
microsecond ago or a whole token ago.

The model does the opposite. It walks 48 distinct layers once each, so by the
time layer 0 runs again the other 47 have been through memory, along with the
token's expert reads. If locality is worth anything, the repeated-one-layer arm
is optimistic and every component figure derived from it is too low.

That would explain a gap this project has seen twice from different
directions: `score_sync` is 142.91 ms while its named components add to about
57, and whole layers measure 262 ms while the parts add to 176. Both leave
about 86 ms.

Two arms, same call count, same chaining, same dependency link:

  repeat    one layer, N times
  distinct  N different layers, once each

Nothing else differs. If the arms match, locality is not the explanation and
the missing time is somewhere else. If distinct is much slower, the component
table in `bench_layer_split.py` is measuring a machine the model never runs on.

This benchmark changes no model behavior and needs no quality gate.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx

from models.flashnext.diskio import disk_bytes_read, free_memory_mb
from models.flashnext.loader import load_streaming

PROMPT = "Explique a fotossintese em duas frases."


def link(x0, reps):
    began = time.perf_counter()
    x, outs = x0, []
    for _ in range(reps):
        y = x * 1.0
        outs.append(y)
        x = x0 + mx.sum(y).astype(x0.dtype) * 0.0
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0 / reps


def run(call_for, x0, reps):
    """Chain `reps` calls, each waiting on the one before it."""
    began = time.perf_counter()
    x, outs = x0, []
    for i in range(reps):
        y = call_for(i)(x)
        outs.append(y)
        x = x0 + mx.sum(y).astype(x0.dtype) * 0.0
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--reps", type=int, default=36,
                        help="calls per arm; 36 is the linear-layer count")
    parser.add_argument("--arms", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")
    from macqwen.checkpoints import resolve_flashnext
    from transformers import AutoTokenizer

    path = str(resolve_flashnext(args.model))
    model, _, _ = load_streaming(
        path, expert_capacity=0, verbose=False, keep_vision=False, use_mtp=False
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    language = model.language_model

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]

    language._position_ids = None
    language._rope_deltas = None
    cache = language.make_cache()
    out = language(ids, cache=cache)
    token = mx.argmax(out.logits[:, -1, :], axis=-1)
    mx.eval(token)
    out = None
    mx.clear_cache()
    for _ in range(4):
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        mx.eval(token)

    linear = [l for l in language.model.layers if getattr(l, "is_linear", False)]
    print(f"model        {path}")
    print(f"free memory  {free_memory_mb():.0f} MB")
    print(f"linear layers {len(linear)}, reps per arm {args.reps}\n")

    # The attention branch is the part that owns per-layer weights, so compare
    # it alone. The MoE would drag streamed reads into the comparison.
    attn = [l.linear_attn for l in linear]
    x0 = mx.random.normal((1, 1, language.model.args.hidden_size)).astype(
        mx.bfloat16
    ) * 0.02
    mx.eval(x0)

    def one(_i):
        return lambda x: attn[0](x, mask=None, cache=None)

    def many(i):
        layer = attn[i % len(attn)]
        return lambda x: layer(x, mask=None, cache=None)

    base = link(x0, args.reps)
    before = disk_bytes_read()
    repeat = statistics.median(
        [run(one, x0, args.reps) for _ in range(args.arms)]
    ) / args.reps - base
    distinct = statistics.median(
        [run(many, x0, args.reps) for _ in range(args.arms)]
    ) / args.reps - base
    read = disk_bytes_read() - before

    print(f"  {'arm':28s} {'ms/call':>9} {'ms/token':>10}")
    print(f"  {'repeat, one layer':28s} {repeat:9.4f} {repeat * 36:10.2f}")
    print(f"  {'distinct, 36 layers':28s} {distinct:9.4f} {distinct * 36:10.2f}")
    ratio = distinct / repeat if repeat > 0 else 0.0
    print(f"  {'ratio':28s} {ratio:9.2f}x")
    print(f"\n  physical read during timing  {read/1e6:.2f} MB")

    print("\nreading:")
    if ratio > 1.3:
        print("  Distinct layers cost more. Every component figure in")
        print("  bench_layer_split.py came from the repeat arm, so each is")
        print("  optimistic by roughly this ratio, and the unattributed part")
        print("  of score_sync is that error rather than a missing stage.")
    else:
        print("  The arms match, so locality across layers is not the")
        print("  explanation. The unattributed part of score_sync is")
        print("  somewhere the component capture does not reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
