#!/usr/bin/env python3
"""Split the GPU half of a decode token across the layer's components.

`score_sync` is 142.91 ms per token, but it is a measurement bucket, not a
stage: it is the main thread blocked while MLX drains its one stream. Three
items inside it are already priced. GatedDeltaNet costs about 57 ms, the expert
`gather_qmm` 13 to 16 ms, and 96 eval round trips about 11 ms. That leaves
roughly 60 ms in QSA attention, the hyper-connections, PLE, the router and the
norms which has never been measured individually in the running model. It is
the largest unexamined block in a token.

Method follows the rule this project had to learn twice. One `mx.eval` per
stage charges each one a full round trip, so summing across stages inflates
the total; the earlier GDN figures were withdrawn for exactly that. Here every
component is timed as a chain of repetitions under a single eval, over
distinct inputs so nothing is folded away, and the single-eval round trip is
reported beside the results rather than subtracted from them.

Inputs are captured from the real model during a real decode step, so every
shape, dtype and weight is the one production uses. Nothing is reconstructed
by hand.

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

DEFAULT_PROMPT = "Explique a fotossintese em duas frases."


class Capture:
    """Record one real call's arguments, then get out of the way.

    Reconstructing a component's inputs by hand is how a benchmark ends up
    measuring a different kernel than the model runs. The gather benchmark hit
    this: a hand-built activation shape measured ten times slower than the same
    projection inside the block.
    """

    def __init__(self):
        self.calls = {}
        self._targets = []      # (name, instance)
        self._patched = {}      # class -> original __call__

    def wrap(self, name, obj):
        """Register one instance. Matching is by identity, not by class.

        Both hyper-connections are the same class, so a class-keyed capture
        would record one of them twice and miss the other.
        """
        self._targets.append((name, obj))
        cls = type(obj)
        if cls in self._patched:
            return
        original = cls.__call__
        self._patched[cls] = original
        owner = self

        def capturing(self_, *args, _o=original, **kwargs):
            for nm, inst in owner._targets:
                if inst is self_ and nm not in owner.calls:
                    owner.calls[nm] = (self_, args, dict(kwargs))
                    break
            return _o(self_, *args, **kwargs)

        cls.__call__ = capturing

    def restore(self):
        for cls, original in self._patched.items():
            cls.__call__ = original
        self._patched = {}


def find_components(language):
    """One representative instance of each component, and how many run."""
    layers = language.model.layers
    found = {}
    counts = {}
    # The whole layer, timed the same way, is the control the component rows
    # need. If the parts do not add up to it, the parts are not the map.
    for layer in layers:
        if getattr(layer, "is_linear", False):
            found.setdefault("LAYER_linear", layer)
            counts["LAYER_linear"] = counts.get("LAYER_linear", 0) + 1
        else:
            found.setdefault("LAYER_attn", layer)
            counts["LAYER_attn"] = counts.get("LAYER_attn", 0) + 1
    for layer in layers:
        if "ple" in layer:
            found.setdefault("ple", layer.ple)
            counts["ple"] = counts.get("ple", 0) + 1
        if getattr(layer, "is_linear", False):
            found.setdefault("linear_attn", layer.linear_attn)
            counts["linear_attn"] = counts.get("linear_attn", 0) + 1
        else:
            found.setdefault("self_attn", layer.self_attn)
            counts["self_attn"] = counts.get("self_attn", 0) + 1
        found.setdefault("attn_hyper_connection", layer.attn_hyper_connection)
        counts["attn_hyper_connection"] = counts.get("attn_hyper_connection", 0) + 1
        found.setdefault("mlp_hyper_connection", layer.mlp_hyper_connection)
        counts["mlp_hyper_connection"] = counts.get("mlp_hyper_connection", 0) + 1
        found.setdefault("router_gate", layer.mlp.gate)
        counts["router_gate"] = counts.get("router_gate", 0) + 1
        found.setdefault("shared_expert", layer.mlp.shared_expert)
        counts["shared_expert"] = counts.get("shared_expert", 0) + 1
    return found, counts


def spread(x, reps):
    """`reps` distinct copies of one captured array, evaluated up front.

    Repeating one input lets the graph collapse every rep into a single
    computation, and the benchmark then measures nothing.
    """
    if not isinstance(x, mx.array):
        return [x] * reps
    if x.dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64, mx.bool_):
        # An index tensor cannot be perturbed without changing what it selects.
        return [x] * reps
    out = [x + mx.array(i * 1e-6, x.dtype) for i in range(reps)]
    mx.eval(out)
    return out


def first_array(y):
    while isinstance(y, (tuple, list)):
        y = y[0]
    return y


def chained_parallel(fn, inputs):
    """len(inputs) INDEPENDENT calls under one eval.

    This is what the gather benchmark used, and for a bandwidth-bound kernel
    it is right. For the small latency-bound kernels in a decode layer it is
    not: independent copies pipeline on the GPU, so the per-call figure comes
    out far below what the same component costs when it has to wait for the
    one before it. Kept only to show that gap.
    """
    began = time.perf_counter()
    outs = [fn(x) for x in inputs]
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0


def chained_serial(fn, x0, reps):
    """`reps` calls where each one waits for the previous.

    In the model a component runs once per layer and the next layer cannot
    start until it finishes. Timing independent copies measures a machine that
    is allowed to overlap them, which the dependency chain forbids. Feeding a
    zeroed scalar reduction of the previous output back into the next input
    rebuilds that dependency without changing any value: `x + sum(y) * 0` is
    exactly `x` for finite y.
    """
    began = time.perf_counter()
    x = x0
    outs = []
    for _ in range(reps):
        y = fn(x)
        arr = first_array(y)
        outs.append(arr)
        x = x0 + mx.sum(arr).astype(x0.dtype) * 0.0
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0


def link_cost(x0, reps):
    """Price the dependency link on its own, so it can be subtracted."""
    began = time.perf_counter()
    x = x0
    outs = []
    for _ in range(reps):
        arr = x * 1.0
        outs.append(arr)
        x = x0 + mx.sum(arr).astype(x0.dtype) * 0.0
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0 / reps


def sync_floor(reps=64):
    probe = mx.zeros((1,))
    mx.eval(probe)
    began = time.perf_counter()
    for _ in range(reps):
        mx.eval(probe + 0.0)
    return (time.perf_counter() - began) * 1000.0 / reps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warm", type=int, default=6,
                        help="decode tokens before capturing, so the cache and "
                             "the hidden states are the ones production sees")
    parser.add_argument("--reps", type=int, default=32)
    parser.add_argument("--arms", type=int, default=3)
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
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]

    print(f"model        {path}")
    print(f"free memory  {free_memory_mb():.0f} MB")

    components, counts = find_components(language)
    print("components   " + ", ".join(
        f"{name}x{counts[name]}" for name in sorted(counts)
    ))

    # Warm the cache and the hidden states, then capture one real call each.
    language._position_ids = None
    language._rope_deltas = None
    cache = language.make_cache()
    out = language(ids, cache=cache)
    token = mx.argmax(out.logits[:, -1, :], axis=-1)
    mx.eval(token)
    out = None
    mx.clear_cache()
    for _ in range(args.warm):
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        mx.eval(token)

    capture = Capture()
    for name, obj in components.items():
        capture.wrap(name, obj)
    step = language(token[None], cache=cache)
    token = mx.argmax(step.logits[:, -1, :], axis=-1)
    mx.eval(token)
    capture.restore()

    missing = [n for n in components if n not in capture.calls]
    if missing:
        print(f"REFUSED: no call captured for {', '.join(missing)}. "
              f"The wrapper did not sit on the class the model runs.",
              file=sys.stderr)
        return 1

    floor = sync_floor()
    print(f"sync floor   {floor:.3f} ms per eval round trip")
    print(f"timing       {args.reps} reps under one eval, "
          f"median of {args.arms} arms\n")

    before = disk_bytes_read()
    results = {}
    parallel = {}
    reads = {}
    stateful = {}
    for name, (owner, call_args, call_kwargs) in capture.calls.items():
        if not call_args or not isinstance(call_args[0], mx.array):
            continue
        rest = call_args[1:]
        x0 = call_args[0]
        inputs = spread(x0, args.reps)
        original = type(owner).__call__
        call = (lambda x, o=owner, r=rest, k=call_kwargs, f=original:
                f(o, x, *r, **k))
        ser, par = [], []
        read_before = disk_bytes_read()
        for _ in range(args.arms):
            ser.append(chained_serial(call, x0, args.reps))
            par.append(chained_parallel(call, inputs))
        reads[name] = (disk_bytes_read() - read_before) / (
            2 * args.arms * args.reps
        )
        link = link_cost(x0, args.reps)
        results[name] = max(
            0.0, statistics.median(ser) / args.reps - link
        )
        parallel[name] = statistics.median(par) / args.reps
        stateful[name] = any(
            k in call_kwargs and call_kwargs[k] is not None for k in ("cache",)
        )
    after = disk_bytes_read()

    print(f"  {'component':24s} {'serial':>8} {'indep':>8} {'count':>6} "
          f"{'ms/token':>9} {'MB/call':>8}")
    total = 0.0
    for name in sorted(results, key=lambda n: -results[n] * counts[n]):
        per_token = results[name] * counts[name]
        if not name.startswith("LAYER_"):
            total += per_token
        mark = " *" if stateful.get(name) else ""
        print(f"  {name:24s} {results[name]:8.4f} {parallel[name]:8.4f} "
              f"{counts[name]:6d} {per_token:9.2f} "
              f"{reads.get(name, 0)/1e6:8.2f}{mark}")
    print(f"  {'PARTS TOTAL':24s} {'':8} {'':8} {'':6} {total:9.2f}")
    whole = sum(
        results[n] * counts[n] for n in results if n.startswith("LAYER_")
    )
    if whole:
        print(f"  {'WHOLE LAYERS':24s} {'':8} {'':8} {'':6} {whole:9.2f}")
        print(f"  {'parts miss':24s} {'':8} {'':8} {'':6} "
              f"{whole - total:9.2f}")
    print("\n  serial: each call waits for the previous, as the layer chain")
    print("  forces. indep: independent copies, which the GPU overlaps. The")
    print("  gap between the two columns is pipelining the model cannot use.")

    if any(stateful.values()):
        print("\n  * carries a cache. Chained reps advance it, so the KV-backed")
        print("    components grow their context across the chain and their")
        print("    figure is an upper bound.")

    read = after - before
    print(f"\npremise:")
    print(f"  physical read during timing   {read/1e6:.2f} MB")
    if read > 64 << 20:
        print("  NOTE: the MoE component streams experts, so its arm includes")
        print("  drive time. Read its row as compute plus I/O, not compute.")

    print("\nreading:")
    print("  The GPU half of a token measured 172.4 ms: score_sync 142.91,")
    print("  router_sync 22.81, final_eval 6.70. Compare the attributed total")
    print("  above against it. A large residual means components fuse in situ")
    print("  or the capture missed a stage; a small one means this split is")
    print("  the map of where the GPU time goes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
