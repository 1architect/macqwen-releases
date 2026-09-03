#!/usr/bin/env python3
"""How many times a token blocks on `mx.eval`, and what each block costs.

A stack sample of 30 seconds of decode put 46% of the main thread inside
`IOSurfaceSharedEvent waitUntilSignaledValue`, reached through
`mlx::core::eval -> array::wait() -> Event::wait()`, while the IOKit GPU
utilisation counter said the shaders were busy about 8% of the time. Both
readings can hold at once if the cost is command-buffer round trip rather than
compute: a token issues many tiny batch-1 graphs, and each one pays submit and
signal latency with almost nothing for the shaders to do when it arrives.

If that is the shape of it, the lever is fewer evals per token, not faster
kernels, and the first thing to know is how many there are and how the block
time is distributed across them. A hundred evals at 0.2 ms is a different
problem from ten at 2 ms.

The IOKit value is a relative comparison signal only. It is not converted to
milliseconds here. Use Metal System Trace for absolute GPU time.

`mx.eval` is wrapped for the duration of a decode. That is a Python-level
count, so it sees what the model asks for and not any eval MLX runs inside
C++. The `profile_totals` timers bracket two of these per layer by name;
this counts all of them.

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

from models.flashnext.diskio import free_memory_mb
from models.flashnext.gpustat import GPUMeter
from models.flashnext.loader import load_streaming

PROMPT = "Explique a fotossintese em duas frases."


class EvalCounter:
    """Wrap `mx.eval`, recording one duration per call."""

    def __init__(self, split=False):
        self.durations: list[float] = []
        self.submits: list[float] = []
        self.waits: list[float] = []
        self.split = split
        self._original = None

    def install(self):
        self._original = mx.eval
        durations, submits, waits = self.durations, self.submits, self.waits
        split = self.split
        original = self._original

        def counted(*args, **kwargs):
            began = time.perf_counter()
            try:
                if split:
                    mid = time.perf_counter()
                    try:
                        mx.async_eval(*args)
                        mid = time.perf_counter()
                        submits.append(mid - began)
                    except Exception:
                        submits.append(0.0)
                    out = original(*args, **kwargs)
                    waits.append(time.perf_counter() - mid)
                    return out
                return original(*args, **kwargs)
            finally:
                durations.append(time.perf_counter() - began)

        mx.eval = counted

    def restore(self):
        if self._original is not None:
            mx.eval = self._original
            self._original = None

    def reset(self):
        self.durations = []
        # the closure holds the list, so re-install to rebind it
        if self._original is not None:
            self.restore()
            self.install()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--warm", type=int, default=4)
    parser.add_argument("--cache-limit-mb", type=int, default=-1,
                        help="MLX buffer cache limit. A token wraps 48 arrays "
                             "of about 24 MB, so 1.1 GB of allocation churn. "
                             "If MLX returns those to the OS and re-allocates, "
                             "the cost lands inside mx.eval as "
                             "MetalAllocator::make_buffer, which a stack "
                             "sample of a real decode did catch. Raising the "
                             "limit trades page cache for fewer allocations, "
                             "so watch MB/token as well as the clock.")
    parser.add_argument("--dummy", type=int, default=0,
                        help="side of a square matmul added to every layer's "
                             "graph. The result is multiplied by zero before "
                             "it reaches the output, so tokens stay identical "
                             "while the GPU is given known extra work. If eval "
                             "block time grows by what the matmul costs, the "
                             "eval block includes more device work. The IOKit "
                             "value remains a relative signal only. If it barely "
                             "moves, use Metal trace before attributing the "
                             "difference to GPU idle time.")
    parser.add_argument("--dummy-ops", type=int, default=0,
                        help="count of trivial cancelled elementwise ops added "
                             "to every layer. The matmul sweep showed cost per "
                             "GFLOP falling as the matmul grew, which is per-op "
                             "latency rather than throughput. These ops carry "
                             "almost no work, so the slope prices a graph node "
                             "on its own.")
    parser.add_argument("--resident-mb", type=int, default=0,
                        help="megabytes of resident weights swept once per "
                             "token, standing in for a small model running in "
                             "the idle window. The matmul sweep reused a 2 MB "
                             "array, so it measured arithmetic with no memory "
                             "traffic. A real supervisor model streams its "
                             "weights, and every previous attempt to add "
                             "traffic to this machine lost to contention. "
                             "Watch MB/token: if the page cache gives ground, "
                             "the expert reads pay for it.")
    parser.add_argument("--split-async", action="store_true",
                        help="split every eval into async_eval then eval. "
                             "async_eval does the CPU-side scheduling and "
                             "returns; the eval that follows only waits. The "
                             "fixed 86 ms survives every test for GPU work, "
                             "graph nodes, page faults and eval count, so the "
                             "remaining question is which side of the "
                             "submission boundary it sits on.")
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

    if args.cache_limit_mb >= 0:
        mx.set_cache_limit(args.cache_limit_mb * 1024 * 1024)
    print(f"model        {path}")
    print(f"free memory  {free_memory_mb():.0f} MB")
    print(f"cache limit  "
          + ("default" if args.cache_limit_mb < 0
             else f"{args.cache_limit_mb} MB"))

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

    # Known GPU work, added to every layer, cancelled before it can change a
    # value. `y + sum(D @ D) * 0` is exactly `y` for finite D.
    if args.dummy:
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer

        side = args.dummy
        pad = mx.random.normal((side, side)).astype(mx.bfloat16) * 0.01
        mx.eval(pad)
        flops = 2 * side ** 3
        original_layer = Qwen4ExpDecoderLayer.__call__

        def with_dummy(self_, hidden_states, *rest, _o=original_layer, **kw):
            out = _o(self_, hidden_states, *rest, **kw)
            return out + mx.sum(pad @ pad).astype(out.dtype) * 0.0

        Qwen4ExpDecoderLayer.__call__ = with_dummy
        print(f"dummy        {side}x{side} matmul per layer, "
              f"{flops/1e6:.0f} MFLOP, {flops*48/1e9:.1f} GFLOP per token")

    if args.dummy_ops:
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer

        n_ops = args.dummy_ops
        seed = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        mx.eval(seed)
        original_ops = Qwen4ExpDecoderLayer.__call__

        def with_ops(self_, hidden_states, *rest, _o=original_ops, **kw):
            out = _o(self_, hidden_states, *rest, **kw)
            acc = seed
            for _ in range(n_ops):
                acc = acc + seed          # one graph node, no real work
            return out + mx.sum(acc).astype(out.dtype) * 0.0

        Qwen4ExpDecoderLayer.__call__ = with_ops
        print(f"dummy ops    {n_ops} elementwise adds per layer, "
              f"{n_ops * 48} per token")

    sidecar = None
    if args.resident_mb:
        # bf16 matvec against a resident bank, so the traffic is a real read of
        # every byte rather than a cache-resident square matmul.
        rows = args.resident_mb * 1024 * 1024 // (2 * 2560)
        sidecar = mx.random.normal((rows, 2560)).astype(mx.bfloat16) * 0.01
        probe = mx.zeros((2560,), dtype=mx.bfloat16)
        mx.eval(sidecar, probe)
        print(f"resident     {args.resident_mb} MB bank, {rows:,} x 2560, "
              f"swept once per token")

        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpModel
        original_model = Qwen4ExpModel.__call__

        def with_sidecar(self_, *a, _o=original_model, **k):
            out = _o(self_, *a, **k)
            side = mx.sum(sidecar @ probe)
            if isinstance(out, tuple):
                return out
            return out + side.astype(out.dtype) * 0.0

        Qwen4ExpModel.__call__ = with_sidecar

    counter = EvalCounter(split=args.split_async)
    meter = GPUMeter(interval=0.004)
    counter.install()
    if meter.available():
        meter.start()
    from models.flashnext.diskio import disk_bytes_read
    read_before = disk_bytes_read()
    began = time.time()
    for _ in range(args.tokens):
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        mx.eval(token)
    elapsed = time.time() - began
    read_mb = (disk_bytes_read() - read_before) / 1e6 / args.tokens
    gpu = meter.stop() if meter.available() else {"samples": 0}
    counter.restore()

    durations = counter.durations
    token_ms = elapsed / args.tokens * 1000
    per_token = len(durations) / args.tokens
    total_ms = sum(durations) / args.tokens * 1000
    ordered = sorted(durations, reverse=True)

    print(f"\n  read                     {read_mb:8.1f} MB/token")
    print(f"  mlx cache                {mx.get_cache_memory()/1e6:8.1f} MB")
    print(f"  token                    {token_ms:8.1f} ms")
    print(f"  evals per token          {per_token:8.1f}")
    print(f"  blocked in eval          {total_ms:8.1f} ms  "
          f"{total_ms/token_ms*100:5.1f}% of the token")
    print(f"  mean per eval            {statistics.mean(durations)*1000:8.3f} ms")
    print(f"  median per eval          {statistics.median(durations)*1000:8.3f} ms")
    if gpu.get("samples"):
        relative = gpu["relative_busy_fraction"]
        print(f"  IOKit relative signal       {relative*100:5.1f}%  "
              "comparison only; not GPU ms")

    if counter.submits and counter.waits:
        sub = sum(counter.submits) / args.tokens * 1000
        wait = sum(counter.waits) / args.tokens * 1000
        print(f"\n  async_eval, CPU side     {sub:8.1f} ms  "
              f"{sub/token_ms*100:5.1f}% of the token")
        print(f"  eval after it, waiting   {wait:8.1f} ms  "
              f"{wait/token_ms*100:5.1f}%")

    # Where the block time sits: a few large evals or many small ones.
    print(f"\n  {'bucket':>16}  {'evals/token':>11} {'ms/token':>9} {'share':>6}")
    edges = [(0, .0005), (.0005, .001), (.001, .002), (.002, .005),
             (.005, .02), (.02, 1e9)]
    for low, high in edges:
        sel = [d for d in durations if low <= d < high]
        if not sel:
            continue
        ms = sum(sel) / args.tokens * 1000
        label = f"{low*1000:.2g}-{high*1000:.2g} ms" if high < 1e9 else ">20 ms"
        print(f"  {label:>16}  {len(sel)/args.tokens:11.1f} {ms:9.1f} "
              f"{ms/total_ms*100:5.1f}%")

    print(f"\n  top 5 single evals per token, ms: "
          + ", ".join(f"{d*1000:.1f}" for d in ordered[:5]))

    print("\nreading:")
    print("  If the block time is spread evenly across many small evals, the")
    print("  cost is per-eval round trip and the lever is issuing fewer, larger")
    print("  graphs. If a handful of large evals dominate, the cost is the work")
    print("  inside them and the eval count is not the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
