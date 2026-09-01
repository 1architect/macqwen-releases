#!/usr/bin/env python3
"""Why prefill is fast, measured across prompt lengths.

Prefill is not fast because it is a different code path. It is fast because a
layer reads each distinct expert once and uses it for every token in the
prompt. At 93 tokens it routes about 290 distinct experts per layer and runs
0.77 tok/s, slower than decode. Near 5,000 tokens it approaches 62 tok/s,
because distinct experts per layer saturates at 512 while tokens keep growing.

Decode reads about 8 experts per layer to produce one token, so its bytes per
token cannot amortise at all. That ratio, reads divided by tokens served, is
the whole difference, and this measures it directly.

Reported per prompt length: tokens per second, physical bytes read, bytes per
token, the rate the drive actually sustained, and distinct experts per layer.
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
                        default=[128, 512, 1024, 2048])
    parser.add_argument("--model")
    parser.add_argument("--rounds", type=int, default=3,
                        help="passes over the lengths; the first is discarded")
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")

    from models.flashnext.adaptive_topk import set_route_observer
    from models.flashnext.diskio import disk_bytes_read
    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend(model_path=args.model)
    language = backend.language

    def one(length):
        """Prefill once, returning rate, physical bytes and experts per layer."""
        distinct: dict[int, set] = {}

        def observe(layer, expert_rows, score_rows, keeps):
            seen = distinct.setdefault(layer, set())
            for experts, keep in zip(expert_rows, keeps):
                seen.update(experts[:keep])

        # A repeated token would route one expert set and flatter the result.
        ids = mx.array(
            [[(index * 7919) % 150000 + 1000 for index in range(length)]],
            dtype=mx.uint32,
        )
        backend.reset()
        cache = language.make_cache()
        language._position_ids = None
        language._rope_deltas = None
        set_route_observer(observe)
        before = disk_bytes_read()
        began = time.perf_counter()
        from models.flashnext.prefill import prefill_language

        _hidden, token = prefill_language(language, ids, cache)
        mx.eval(token)
        elapsed = time.perf_counter() - began
        physical = disk_bytes_read() - before
        set_route_observer(None)
        cache = None
        mx.clear_cache()
        per_layer = (
            sum(len(v) for v in distinct.values()) / len(distinct)
            if distinct else 0.0
        )
        return length / elapsed, physical, per_layer

    # Round-robin, so every length meets a comparable cache state. Running
    # them in ascending order lets each warm the next, which made 4 tokens
    # read less than 2 and flattered every larger batch.
    collected = {length: [] for length in args.lengths}
    for round_index in range(args.rounds):
        for length in args.lengths:
            collected[length].append(one(length))
        print(f"  round {round_index + 1} of {args.rounds} done", flush=True)

    print()
    print(f"  medians over {args.rounds - 1} rounds, the first discarded")
    print(f"  {'tokens':>7}{'tok/s':>9}{'read GB':>10}{'MB/token':>10}"
          f"{'drive':>10}{'experts/layer':>15}", flush=True)
    rows = []
    for length in args.lengths:
        kept = collected[length][1:] or collected[length]
        rate = st.median(r for r, _p, _e in kept)
        physical = st.median(p for _r, p, _e in kept)
        per_layer = st.median(e for _r, _p, e in kept)
        mb_token = physical / length / 1e6 if physical > 0 else -1.0
        drive = physical / (length / rate) / 1e9 if rate else -1.0
        rows.append((length, rate, physical, mb_token, drive, per_layer))
        print(f"  {length:>7}{rate:>9.2f}{physical/1e9:>10.2f}{mb_token:>10.1f}"
              f"{drive:>9.2f}G{per_layer:>15.0f}", flush=True)

    print()
    print("  Decode, for comparison: 2.71 tok/s, 390 MB/token, about 8")
    print("  experts per layer for one token.")
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print()
        print(f"  From {first[0]} to {last[0]} tokens the rate moved "
              f"{first[1]:.2f} -> {last[1]:.2f} tok/s while bytes per token "
              f"moved {first[3]:.0f} -> {last[3]:.0f} MB.")
        print("  If the drive rate barely moves while bytes per token fall,")
        print("  prefill is winning on amortisation, not on throughput, and")
        print("  there is nothing in the path for decode to copy.")


if __name__ == "__main__":
    main()
