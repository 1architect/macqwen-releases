#!/usr/bin/env python3
"""Price the elementwise glue between a decode layer's named components.

`score_sync` is 142.91 ms per token and the named GPU components inside it add
to about 57. `bench_layer_split.py` puts the remainder in the layer glue by
subtracting parts from whole layers, but those whole-layer arms carry streamed
reads and moved 0.5 ms between runs, so a subtraction from them is not
evidence. This measures the glue directly.

The glue is every line in `Qwen4ExpDecoderLayer.__call__` and the tail of
`_moe_call` that has no timer of its own:

    hidden_states + self.ple(...)                        once per token
    branch[..., None, :] * injection_weights[..., None]  twice per layer
    hyper_input + injection.reshape(...)                 twice per layer
    (expert_values * scores[..., None]).sum(axis=-2)     once per layer
    shared_gate * shared, then y + shared_y              once per layer

None of it is a matmul. All of it is elementwise work on hidden_size by
hc_count arrays, and it runs 48 or 96 times per token, which is why it can be
large without any single line looking expensive.

Shapes come from the checkpoint config, not from guesses. Timing serialises
each repetition against the previous one, because the layer chain forbids the
GPU from overlapping them, and prices the dependency link separately so it can
be subtracted.

This benchmark changes no model behavior and needs no quality gate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx

from models.flashnext.diskio import free_memory_mb


def chained_serial(build, x0, reps):
    """`reps` calls where each waits for the previous.

    `x + sum(y) * 0` is exactly `x` for finite y, so the dependency is real
    while the values are untouched.
    """
    began = time.perf_counter()
    x = x0
    outs = []
    for _ in range(reps):
        y = build(x)
        outs.append(y)
        x = x0 + mx.sum(y).astype(x0.dtype) * 0.0
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0


def link_cost(x0, reps):
    began = time.perf_counter()
    x = x0
    outs = []
    for _ in range(reps):
        y = x * 1.0
        outs.append(y)
        x = x0 + mx.sum(y).astype(x0.dtype) * 0.0
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0 / reps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--reps", type=int, default=64)
    parser.add_argument("--arms", type=int, default=5)
    parser.add_argument("--experts", type=int, default=8,
                        help="routed slots per layer; decode keeps about 8")
    args = parser.parse_args()

    from macqwen.checkpoints import resolve_flashnext

    path = str(resolve_flashnext(args.model))
    config = json.load(open(os.path.join(path, "config.json")))
    text = config.get("text_config", config)
    hidden = text.get("hidden_size", config.get("hidden_size"))
    hc = text.get("hc_count", config.get("hc_count"))
    layers = text.get("num_hidden_layers", config.get("num_hidden_layers"))
    slots = args.experts

    print(f"model        {path}")
    print(f"shapes       hidden={hidden} hc_count={hc} layers={layers} "
          f"slots={slots}")
    print(f"free memory  {free_memory_mb():.0f} MB")

    dt = mx.bfloat16
    # Exactly the arrays the decoder layer holds at decode.
    branch = mx.random.normal((1, 1, hidden)).astype(dt)
    weights = mx.random.normal((1, 1, hc)).astype(dt)
    hyper_in = mx.random.normal((1, 1, hc, hidden)).astype(dt)
    values = mx.random.normal((1, 1, slots, hidden)).astype(dt)
    scores = mx.random.uniform(shape=(1, 1, slots)).astype(dt)
    gate = mx.random.uniform(shape=(1, 1, 1)).astype(dt)
    shared = mx.random.normal((1, 1, hidden)).astype(dt)
    resid = mx.random.normal((1, 1, hidden)).astype(dt)
    mx.eval(branch, weights, hyper_in, values, scores, gate, shared, resid)

    def injection(b):
        # branch[..., None, :] * injection_weights[..., None], then the add
        inj = b[..., None, :] * weights[..., None]
        return (hyper_in + inj.reshape(*hyper_in.shape)).reshape(1, 1, -1)[
            ..., :hidden
        ]

    def moe_combine(s):
        y = (values * s[..., None]).sum(axis=-2)
        return y + gate * shared

    def ple_add(h):
        return h + resid

    # --- the ops that drain at router_sync -------------------------------
    # `mx.eval(flat)` in StreamingSwitchGLU drains everything queued since
    # score_sync: the keep mask, the renormalisation, and the shared expert.
    # The mask is the interesting one. With exact-quality pinning, `active` is
    # built by handing MLX a Python list of bools, 48 times per token.
    k = text.get("num_experts_per_tok", config.get("num_experts_per_tok", 10))
    width = slots
    inds10 = mx.arange(k, dtype=mx.uint32).reshape(1, 1, k)
    scores10 = mx.random.uniform(shape=(1, 1, k)).astype(dt)
    topk_mass = scores10.sum(axis=-1, keepdims=True)
    keep_array = mx.array([[[width]]], dtype=mx.int32)
    rows = [[True] * width]
    mx.eval(inds10, scores10, topk_mass, keep_array)

    def keep_mask(sc):
        # slice, arange compare, and the two wheres
        i = inds10[..., :width]
        s2 = sc[..., :width]
        active = mx.arange(width) < keep_array
        i = mx.where(active, i, i[..., :1])
        return mx.where(active, s2, 0)

    def mask_from_list(sc):
        # the pinned-resident path: an mx.array built from a Python list
        active = mx.array([r[:width] for r in rows], dtype=mx.bool_).reshape(
            1, 1, width
        )
        return mx.where(active, sc[..., :width], 0)

    def renorm(sc):
        selected = sc[..., :width].sum(axis=-1, keepdims=True)
        normalizer = topk_mass + 1.0 * (selected - topk_mass)
        return sc[..., :width] / normalizer

    work = [
        ("injection combine", injection, branch, 2 * layers),
        ("MoE output combine", moe_combine, scores, layers),
        ("PLE residual add", ple_add, resid, 1),
        ("keep mask", keep_mask, scores10, layers),
        ("mask from Python list", mask_from_list, scores10, layers),
        ("renormalisation", renorm, scores10, layers),
    ]

    print(f"\ntiming       {args.reps} reps chained, median of {args.arms} arms\n")
    print(f"  {'glue op':24s} {'ms/call':>9} {'count':>6} {'ms/token':>9}")
    total = 0.0
    for name, build, seed, count in work:
        link = link_cost(seed, args.reps)
        runs = [chained_serial(build, seed, args.reps) for _ in range(args.arms)]
        per = max(0.0, statistics.median(runs) / args.reps - link)
        total += per * count
        print(f"  {name:24s} {per:9.4f} {count:6d} {per * count:9.2f}")
    print(f"  {'TOTAL':24s} {'':9} {'':6} {total:9.2f}")

    print("\nreading:")
    print("  The first three rows drain at score_sync, the last three at")
    print("  router_sync. router_sync measured 21.75 to 22.81 ms per token,")
    print("  of which the shared expert graph is 2.03 to 2.56. Compare the")
    print("  mask and renormalisation rows against what is left of it.")
    print("  score_sync is not a fixed quantity: it grew from 142.91 to")
    print("  187.37 ms when the shared read buffer removed 33 ms of host")
    print("  copy, so subtracting components from it measures scheduling,")
    print("  not a missing stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
