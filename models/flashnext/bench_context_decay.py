#!/usr/bin/env python3
"""Decode rate against context length.

Every published rate for this runtime comes from `bench_production.py`, which
uses one fixed prompt of about 20 tokens. Real chat turns run at 4,500 to
5,600 and the terminal reports 2.0 to 2.2 there against 2.83 on the harness.
The research log records that gap and names the harness prompt as the reason,
but nothing has measured the curve between the two.

Attention arithmetic cannot explain it. Thirty-six of the 48 layers use a
linear-attention recurrence whose state does not grow with context, and the
twelve full-attention layers hold a few megabytes at 5,000 tokens. So either
the cost is elsewhere, or it is the expert working set losing page cache to
the growing state.

Prefill is paid but not timed. Only the decode that follows is measured, with
physical bytes, so a rate change can be read against a byte change.

Lengths run round-robin. An ascending sweep lets each length warm the next,
which is how an earlier batching study moved its own crossover point.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[128, 1024, 4096])
    parser.add_argument("--model")
    parser.add_argument("--tokens", type=int, default=24,
                        help="decode tokens timed at each context length")
    parser.add_argument("--windows", type=int, default=3,
                        help="split the decode so prefill aftermath separates "
                             "from steady state")
    parser.add_argument("--rounds", type=int, default=4,
                        help="passes over the lengths; the first is discarded")
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")

    from models.flashnext.adaptive_topk import set_route_observer
    from models.flashnext.diskio import disk_bytes_read, free_memory_mb
    from models.flashnext.prefill import prefill_language
    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend(model_path=args.model)
    language = backend.language

    def one(length):
        # A repeated token would route one expert set and flatter the result.
        ids = mx.array(
            [[(index * 7919) % 150000 + 1000 for index in range(length)]],
            dtype=mx.uint32,
        )
        backend.reset()
        cache = language.make_cache()
        language._position_ids = None
        language._rope_deltas = None
        _hidden, token = prefill_language(language, ids, cache)

        # Count kept experts per layer during decode only. A longer context
        # could route more widely, which would raise physical bytes without
        # any page-cache effect. This separates the two.
        kept_total = [0, 0]

        def observe(_layer, _expert_rows, _score_rows, keeps):
            kept_total[0] += sum(keeps)
            kept_total[1] += len(keeps)

        # Report the decode in windows. A wide prefill leaves the page cache
        # holding the prompt's experts, so the first tokens after it are the
        # aftermath of the prefill rather than the cost of the context. If the
        # last window recovers toward the short-context rate, that is what the
        # headline number was measuring.
        set_route_observer(observe)
        width = max(1, args.tokens // args.windows)
        windows = []
        before = disk_bytes_read()
        began = time.perf_counter()
        mark, mark_bytes = began, before
        for index in range(args.tokens):
            logits = language(token[None], cache=cache).logits
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            if (index + 1) % width == 0:
                now, now_bytes = time.perf_counter(), disk_bytes_read()
                windows.append((width / (now - mark),
                                (now_bytes - mark_bytes) / width))
                mark, mark_bytes = now, now_bytes
        elapsed = time.perf_counter() - began
        physical = disk_bytes_read() - before
        set_route_observer(None)
        experts = kept_total[0] / kept_total[1] if kept_total[1] else 0.0
        free = free_memory_mb()
        cache = None
        mx.clear_cache()
        return args.tokens / elapsed, physical / args.tokens, free, experts, windows

    collected = {length: [] for length in args.lengths}
    for round_index in range(args.rounds):
        for length in args.lengths:
            collected[length].append(one(length))
            rate, per_token, free, experts, windows = collected[length][-1]
            shape = "  ".join(f"{r:4.2f}/{b/1e6:.0f}MB" for r, b in windows)
            print(f"  round {round_index + 1} ctx {length:>5}  "
                  f"{rate:5.2f} tok/s  {per_token/1e6:6.1f} MB/token  "
                  f"{experts:4.2f} experts/layer  free {free:.0f} MB",
                  flush=True)
            print(f"      windows  {shape}", flush=True)

    print()
    print(f"  medians over {args.rounds - 1} rounds, the first discarded")
    print(f"  {'context':>8}{'tok/s':>9}{'last win':>9}{'MB/token':>10}"
          f"{'experts':>9}{'vs shortest':>13}")
    base = None
    for length in args.lengths:
        kept = collected[length][1:] or collected[length]
        rate = st.median(r for r, _b, _f, _e, _w in kept)
        per_token = st.median(b for _r, b, _f, _e, _w in kept)
        experts = st.median(e for _r, _b, _f, e, _w in kept)
        last = st.median(w[-1][0] for _r, _b, _f, _e, w in kept if w)
        if base is None:
            base = rate
        print(f"  {length:>8}{rate:>9.2f}{last:>9.2f}{per_token/1e6:>10.1f}"
              f"{experts:>9.2f}{(rate / base - 1) * 100:>12.1f}%", flush=True)

    print()
    print("  If the rate falls while bytes per token hold, the cost is not the")
    print("  expert stream and something in the runtime scales with context.")
    print("  If bytes per token rise with it, the growing cache is taking page")
    print("  cache away from the expert bank, which is a RAM capacity result.")


if __name__ == "__main__":
    main()
