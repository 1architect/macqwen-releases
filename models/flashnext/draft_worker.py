#!/usr/bin/env python3
"""Generate continuously with a small draft model, to load the machine.

This is not a speculative decoder. It is the load half of
`bench_draft_contention.py`: a second process doing the work a drafter would
do, so we can measure what the target loses to it.

Every in-process attempt to overlap reads with compute lost on this machine,
because the overlapped work fed the GPU work the main thread was waiting for.
Two independent processes reached 1.52x aggregate. A drafter is independent
work by the same definition, so the question is only what it costs the target.

Prints one JSON line on exit. Stops on SIGTERM.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

_STOP = [False]


def _halt(signum, frame) -> None:
    _STOP[0] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/Qwen3.5-0.8B-MLX-4bit")
    parser.add_argument("--block", type=int, default=2,
                        help="tokens per generation, as a drafter would emit")
    parser.add_argument("--duty", type=float, default=1.0,
                        help="fraction of the time to generate, 0 to 1")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this long, 0 waits for SIGTERM")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _halt)
    signal.signal(signal.SIGINT, _halt)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, tokenizer = load(os.path.expanduser(args.model))
    prompt = mx.array(tokenizer.encode("Explique a fotossintese."))

    # Report readiness so the parent starts its timing after the load.
    print(json.dumps({"ready": True}), flush=True)

    began = time.time()
    tokens = 0
    blocks = 0
    while not _STOP[0]:
        if args.seconds and time.time() - began >= args.seconds:
            break
        block_began = time.time()
        for index, _ in enumerate(generate_step(prompt, model)):
            tokens += 1
            if index + 1 >= args.block:
                break
        blocks += 1
        if args.duty < 1.0:
            worked = time.time() - block_began
            idle = worked * (1.0 / max(args.duty, 0.01) - 1.0)
            end = time.time() + idle
            while time.time() < end and not _STOP[0]:
                time.sleep(min(0.01, max(0.0, end - time.time())))

    elapsed = time.time() - began
    print(json.dumps({
        "draft_tokens": tokens,
        "draft_blocks": blocks,
        "draft_seconds": round(elapsed, 3),
        "draft_tps": round(tokens / elapsed, 3) if elapsed else 0.0,
    }), flush=True)


if __name__ == "__main__":
    main()
