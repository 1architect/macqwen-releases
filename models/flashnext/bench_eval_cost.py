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

    def __init__(self):
        self.durations: list[float] = []
        self._original = None

    def install(self):
        self._original = mx.eval
        durations = self.durations

        def counted(*args, **kwargs):
            began = time.perf_counter()
            try:
                return self._original(*args, **kwargs)
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

    print(f"model        {path}")
    print(f"free memory  {free_memory_mb():.0f} MB")

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

    counter = EvalCounter()
    meter = GPUMeter(interval=0.004)
    counter.install()
    if meter.available():
        meter.start()
    began = time.time()
    for _ in range(args.tokens):
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        mx.eval(token)
    elapsed = time.time() - began
    gpu = meter.stop() if meter.available() else {"samples": 0}
    counter.restore()

    durations = counter.durations
    token_ms = elapsed / args.tokens * 1000
    per_token = len(durations) / args.tokens
    total_ms = sum(durations) / args.tokens * 1000
    ordered = sorted(durations, reverse=True)

    print(f"\n  token                    {token_ms:8.1f} ms")
    print(f"  evals per token          {per_token:8.1f}")
    print(f"  blocked in eval          {total_ms:8.1f} ms  "
          f"{total_ms/token_ms*100:5.1f}% of the token")
    print(f"  mean per eval            {statistics.mean(durations)*1000:8.3f} ms")
    print(f"  median per eval          {statistics.median(durations)*1000:8.3f} ms")
    if gpu.get("samples"):
        busy = gpu["busy_fraction"] * token_ms
        print(f"  GPU busy, IOKit          {busy:8.1f} ms  "
              f"{gpu['busy_fraction']*100:5.1f}%")

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
