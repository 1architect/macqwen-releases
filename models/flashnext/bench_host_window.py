#!/usr/bin/env python3
"""How much host time runs with the drive and the GPU both idle.

Path 2 asks whether small host bookkeeping can move off the serial chain. That
only pays if the work currently runs in a window where neither device is doing
anything, because such a window is dead time on both units and removing it
cannot be absorbed the way the read-path savings were.

This measures the size of that window before anything is moved. It changes no
model behavior and asserts identical token IDs across passes, so a perturbed
runtime cannot report a result.

Read `hostwindow.py` for how each device is judged idle. The short version:
the drive is measured with a counter around every pool submission, and the GPU
is idle by construction because MLX drains its one stream at `mx.eval` and
every window here sits between two evals.

    FLASHNEXT_HOST_WINDOW=1 python models/flashnext/bench_host_window.py \\
        --tokens 24 --passes 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx

from models.flashnext import hostwindow
from models.flashnext.gpustat import GPUMeter
from models.flashnext.diskio import disk_bytes_read, free_memory_mb
from models.flashnext.expert_cache import profile_totals, reset_profile
from models.flashnext.loader import load_streaming

DEFAULT_PROMPT = "Explique a fotossintese em duas frases."


def decode_pass(language, ids, count):
    """One greedy decode of `count` tokens, timed and instrumented."""
    language._position_ids = None
    language._rope_deltas = None
    cache = language.make_cache()
    output = language(ids, cache=cache)
    token = mx.argmax(output.logits[:, -1, :], axis=-1)
    mx.eval(token)
    output = None
    mx.clear_cache()

    produced = []
    reset_profile()
    hostwindow.reset()
    # Every other number here is the main thread blocked. This one is the GPU
    # itself, read from IOKit, so the "GPU running" row can be checked instead
    # of assumed.
    meter = GPUMeter(interval=0.004)
    meter.start() if meter.available() else None
    read_before = disk_bytes_read()
    began = time.time()
    drain = 0.0
    for _ in range(count):
        produced.append(int(token.item()))
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        # The head and everything the layer loop left unevaluated land here.
        # It is GPU time like any other eval block, so it is counted with the
        # per-layer syncs rather than lost to the residual.
        drain_began = time.perf_counter()
        mx.eval(token)
        drain += time.perf_counter() - drain_began
    elapsed = time.time() - began
    gpu = meter.stop() if meter.available() else {"samples": 0}
    read_bytes = disk_bytes_read() - read_before
    timers = profile_totals()
    timers["final_eval"] = drain
    timers["gpu_busy_fraction"] = gpu.get("busy_fraction", -1.0)
    timers["gpu_samples"] = gpu.get("samples", 0)
    return (
        produced,
        elapsed / count,
        read_bytes / count,
        timers,
        hostwindow.totals(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--hot", type=int, default=32,
                        help="resident experts per layer, as exact-quality uses")
    args = parser.parse_args()

    if not hostwindow.ENABLED:
        print(
            "REFUSED: set FLASHNEXT_HOST_WINDOW=1. Without it the windows are "
            "not recorded and this would report an empty table.",
            file=sys.stderr,
        )
        return 2

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
    print(f"tokens       {args.tokens} per pass, {args.passes} passes\n")

    reference = None
    kept = None
    for index in range(1, args.passes + 1):
        produced, seconds, read_bytes, timers, windows = decode_pass(
            language, ids, args.tokens
        )
        if reference is None:
            reference = produced
        elif produced != reference:
            print("REFUSED: token IDs changed between passes", file=sys.stderr)
            return 1
        # `io_wait` only fills in under FLASHNEXT_PROFILE_IO. Printing a bare
        # 0.0 without it would read as "no wait", which is the opposite of
        # true. The io_await window below carries that number either way.
        wait = timers["io_wait"] / args.tokens * 1000
        wait_text = f"{wait:6.1f} ms" if wait > 0 else "off"
        print(
            f"pass {index}  {seconds*1000:7.1f} ms/token  "
            f"{1/seconds:5.2f} tok/s  {read_bytes/1e6:6.1f} MB/token  "
            f"io_wait {wait_text}"
        )
        kept = (windows, args.tokens)
        last_timers = timers

    print("\nprofile timers, last pass, ms per token:")
    for key in sorted(last_timers):
        if key == "io_calls":
            continue
        print(f"  {key:22s} {last_timers[key]/args.tokens*1000:8.2f}")

    print("\nhost windows, last pass, per token:")
    print(hostwindow.report(kept[1]))

    exclusive = sum(slot[0] for slot in kept[0].values())
    await_window = kept[0].get("io_await", [0.0, 0.0, 0, 0])
    movable = sum(
        slot[0]
        for name, slot in kept[0].items()
        if name not in ("io_await", "to_mx_host")
    )

    print("\ncontrol:")
    print(
        f"  io_await exclusive share       {await_window[3]}/{await_window[2]}"
        "  (must be 0; it blocks on the drive by definition)"
    )
    if await_window[2] and await_window[3]:
        print("  REFUSED: the read counter did not see reads in flight")
        return 1

    windows, tokens = kept
    if last_timers.get("score_sync", 0.0) or last_timers.get("router_sync", 0.0):
        # Every eval blocks the main thread until MLX drains its one stream,
        # so the time spent inside an eval is the GPU actually running. The
        # drive's busy time is the io_await window, which the read counter
        # backs. What is left over belongs to neither device.
        # `ngram_wait` brackets `_direct_rows` in ngram.py, which reads n-gram
        # rows off the drive. It is storage time, not GPU time. Counting it
        # with the eval blocks overstated the GPU by 6.3 ms per token.
        gpu = (
            last_timers["score_sync"]
            + last_timers["router_sync"]
            + last_timers["final_eval"]
        )
        # Two independent sources for the same quantity. The window is backed
        # by the read counter; the timer brackets the same call. They should
        # agree, and the gap is printed so a silent disagreement cannot pass.
        drive_window = windows.get("io_await", [0.0, 0.0, 0, 0])[1]
        drive = (
            last_timers.get("io_wait", 0.0) or drive_window
        ) + last_timers.get("ngram_wait", 0.0)
        idle = sum(slot[0] for slot in windows.values())
        token_ms = seconds * 1000
        accounted = (gpu + drive + idle) / tokens * 1000
        print("\ndevice duty, last pass, per token:")
        print(f"  {'state':26s} {'ms':>8}  {'share':>6}")
        for label, value in (
            ("GPU running", gpu),
            ("drive reading", drive),
            ("neither, host only", idle),
        ):
            ms = value / tokens * 1000
            print(f"  {label:26s} {ms:8.1f}  {ms/token_ms*100:5.1f}%")
        print(f"  {'unaccounted':26s} {token_ms-accounted:8.1f}  "
              f"{(token_ms-accounted)/token_ms*100:5.1f}%")
        frac = last_timers.get("gpu_busy_fraction", -1.0)
        if frac >= 0:
            measured = frac * token_ms
            print(f"  {'':26s} {'':8}  {'':6}")
            print(f"  {'GPU busy, IOKit counter':26s} {measured:8.1f}  "
                  f"{frac*100:5.1f}%   over "
                  f"{int(last_timers.get('gpu_samples', 0))} samples")
            # `gpu` is a total in seconds; every other figure here is
            # milliseconds per token.
            gpu_ms = gpu / tokens * 1000
            print(f"  {'eval block time claims':26s} {gpu_ms:8.1f}  "
                  f"{gpu_ms/token_ms*100:5.1f}%")
            if measured < gpu_ms * 0.7:
                print("  The GPU is idle for much of what the eval blocks on,")
                print("  so eval block time is not GPU time and the unattributed")
                print("  part of score_sync is the host waiting, not kernels.")
            elif measured > gpu_ms * 1.3:
                print("  The GPU is busier than the eval blocks account for,")
                print("  so work is running outside the timed evals.")
            else:
                print("  The two agree, so eval block time is GPU time and the")
                print("  component table is missing real kernels.")
        print(f"  {'token':26s} {token_ms:8.1f}")
        print(
            f"  drive: expert timer "
            f"{last_timers.get('io_wait',0.0)/tokens*1000:.1f} ms against "
            f"counter window {drive_window/tokens*1000:.1f} ms, plus "
            f"{last_timers.get('ngram_wait',0.0)/tokens*1000:.1f} ms of n-gram"
        )
        print("  The two devices never overlap here: the drive window and the")
        print("  eval windows are disjoint by construction, so these add.")

    print("\nreading:")
    print(
        f"  exclusive host time            "
        f"{exclusive/kept[1]*1000:.2f} ms/token"
    )
    print(
        f"  of that, low-bandwidth work    "
        f"{movable/kept[1]*1000:.2f} ms/token"
    )
    print("  to_mx_host is excluded above: it copies the whole layer, so it")
    print("  is bulk data movement rather than bookkeeping.")
    print("  Path 2 wants 15 to 20 ms in the low-bandwidth row to be worth a")
    print("  move. Below that, leave the schedule alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
