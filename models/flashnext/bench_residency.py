#!/usr/bin/env python3
"""Is the residency tracker right often enough to be worth using?

The `resident` read mode maps rows it believes the page cache holds. On a
resident row the map costs 0.99 ms against pread's 3.26, saving 2.27. On a
cold row it costs 21.89 against 13.62, losing 8.27. The gate therefore has to
be right about 78% of the time before it breaks even:

    break-even accuracy = 8.27 / (8.27 + 2.27) = 78.5%

`mincore` gives ground truth but costs 7.2 us per row, more than it saves, so
it cannot gate the real read path. It can measure the tracker offline, which
is what this does. Run it before trusting FLASHNEXT_TRACK_RESIDENT.
"""
from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

MAP_SAVE = 2.27      # ms per 72-row layer, mapped vs pread, resident
MAP_COST = 8.27      # ms per 72-row layer, mapped vs pread, cold
BREAK_EVEN = MAP_COST / (MAP_COST + MAP_SAVE)


def truly_resident(store, name: str, row: int) -> bool:
    """Ask the kernel whether every page of this row is cached."""
    ref = store.refs[name]
    view = store._shared_view(name)
    base = int(view.__array_interface__["data"][0])
    page = mmap.PAGESIZE
    address = base + int(row) * ref.row_bytes
    start = address - address % page
    length = address + ref.row_bytes - start
    pages = (length + page - 1) // page
    buffer = ctypes.create_string_buffer(pages)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(length), buffer) != 0:
        return False
    return all(byte & 1 for byte in buffer.raw[:pages])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--cap", type=int, default=12000)
    args = parser.parse_args()

    # The gate only runs on the `resident` read path. Set this before the
    # backend is imported: routing reads the default at import time.
    os.environ["FLASHNEXT_READ"] = "resident"
    os.environ["FLASHNEXT_TRACK_RESIDENT"] = "1"
    os.environ["FLASHNEXT_RESIDENT_ROWS"] = str(args.cap)
    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")

    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend(model_path=args.model)
    store = backend.store
    prompt = ("<|im_start|>user\nExplique a fotossintese em duas frases."
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    # Record what the gate decided and what was actually true, per row.
    checked = {"true_hit": 0, "false_hit": 0, "true_miss": 0, "false_miss": 0}
    original = type(store).believed_resident

    def observed(self, name, row):
        believed = original(self, name, row)
        actual = truly_resident(self, name, row)
        if believed and actual:
            checked["true_hit"] += 1
        elif believed and not actual:
            checked["false_hit"] += 1
        elif not believed and actual:
            checked["false_miss"] += 1
        else:
            checked["true_miss"] += 1
        return believed

    backend.reset()
    backend.append_text(prompt)
    backend.generate(max_tokens=8)          # warm the tracker first

    type(store).believed_resident = observed
    try:
        backend.reset()
        backend.append_text(prompt)
        backend.generate(max_tokens=args.tokens)
    finally:
        type(store).believed_resident = original

    believed_resident = checked["true_hit"] + checked["false_hit"]
    total = sum(checked.values())
    if not total:
        print("  the gate was never consulted.")
        print(f"  read mode is {store._read_mode!r}; it must be 'resident'.")
        print("  nothing was measured.")
        return
    if not believed_resident:
        print(f"  the gate ran {total} times and never claimed a row.")
        print(f"  tracking={store._track_residency} cap={store._resident_cap}")
        print(f"  {checked['false_miss']} of those rows were in fact cached.")
        return
    precision = checked["true_hit"] / believed_resident
    print(f"  rows gated              {total}")
    print(f"  claimed resident        {believed_resident}"
          f"  ({believed_resident / total:.1%} of reads)")
    print(f"    correct               {checked['true_hit']}")
    print(f"    wrong, actually cold  {checked['false_hit']}")
    print(f"  missed a cached row     {checked['false_miss']}")
    print()
    print(f"  precision               {precision:.1%}")
    print(f"  break-even              {BREAK_EVEN:.1%}")
    saved = checked["true_hit"] * MAP_SAVE - checked["false_hit"] * MAP_COST
    print(f"  modelled net            {saved / 72:+.1f} ms, from isolated"
          f" read timings")
    print()
    if precision <= BREAK_EVEN:
        print("  the gate is too inaccurate to use: a wrong guess costs 3.6x")
        print("  what a right one saves.")
        return
    print("  The gate clears its accuracy bar. That is all this measures.")
    print("  It does not mean the chat gets faster: `resident` mode already")
    print("  measured neutral in production, where the memory controller took")
    print("  back more than the read path saved. The modelled net above comes")
    print("  from isolated read timings and has no weight until an end-to-end")
    print("  run agrees with it:")
    print()
    print("    bench_production.py --compare track-resident")


if __name__ == "__main__":
    main()
