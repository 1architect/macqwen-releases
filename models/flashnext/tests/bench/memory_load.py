#!/usr/bin/env python3
"""Hold a fixed amount of active anonymous memory until stopped.

A controlled stand-in for another application competing for RAM. It writes
every page once, then rewrites one byte per page every few seconds so the
pages stay active rather than being compressed early. Stop it with SIGTERM or
Ctrl-C.

    python -m models.flashnext.tests.bench.memory_load --gb 3
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

PAGE = 16384


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb", type=float, default=3.0)
    parser.add_argument("--period", type=float, default=5.0)
    args = parser.parse_args()
    size = int(args.gb * 1e9)
    block = bytearray(size)
    for offset in range(0, size, PAGE):
        block[offset] = 1
    print(f"holding {size / 1e9:.2f} GB", flush=True)

    running = [True]

    def stop(*_args):
        running[0] = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    value = 1
    while running[0]:
        value = (value + 1) & 0xFF
        for offset in range(0, size, PAGE):
            block[offset] = value
        time.sleep(args.period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
