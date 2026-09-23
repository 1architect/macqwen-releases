#!/usr/bin/env python3
"""Does a no-copy Metal buffer over a mapped shard read only what a kernel touches?

Maps a large page-aligned region of one checkpoint shard read-only, wraps it
as an MLX array without a copy (``mx.from_dlpack(..., copy=False)``, which
uses ``newBufferWithBytesNoCopy``), then runs a gather that touches a few
expert-sized rows. The process physical-read counter shows whether Metal made
the whole region resident at commit or only faulted the touched pages.

It also times the gather against the normal path (``pread`` into a fresh
buffer, then wrap) for the same rows, cold and warm. No model is loaded.

    python -m models.flashnext.tests.bench.probe_zero_copy_shard --region-gb 1
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from macqwen.results import output_path  # noqa: E402
from models.flashnext.diskio import disk_bytes_read  # noqa: E402

PAGE = 16384
ROW = 819_200  # one Q4/G32 expert projection weight row


_LIBC = __import__("ctypes").CDLL("/usr/lib/libSystem.B.dylib")


def resident_fraction(view: np.ndarray) -> float:
    """Share of the mapped pages the kernel reports resident (mincore)."""
    import ctypes

    length = view.nbytes
    pages = (length + PAGE - 1) // PAGE
    vector = (ctypes.c_char * pages)()
    _LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    if _LIBC.mincore(ctypes.c_void_p(view.ctypes.data), length, vector) != 0:
        return -1.0
    return sum(1 for byte in bytes(vector) if byte & 1) / pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default="")
    parser.add_argument("--region-gb", type=float, default=1.0)
    parser.add_argument("--rows", type=int, default=24)
    args = parser.parse_args()
    target = output_path("flashnext", "zero-copy-probe", "zero-copy.json")

    if args.shard:
        shard = args.shard
    else:
        from macqwen.checkpoints import resolve_flashnext

        directory = Path(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
        shards = sorted(directory.glob("model-*.safetensors"))
        shard = str(shards[len(shards) // 2])
    size = os.path.getsize(shard)
    region = min(size - PAGE, int(args.region_gb * 1e9)) // PAGE * PAGE
    offset = ((size - region) // 2) // PAGE * PAGE
    generator = random.Random(7)
    rows = sorted(generator.sample(range(region // ROW - 1), args.rows))
    result = {"shard": shard, "region_bytes": region, "rows": len(rows)}

    fd = os.open(shard, os.O_RDONLY)
    mapping = mmap.mmap(fd, offset + region, prot=mmap.PROT_READ)
    host = np.frombuffer(mapping, dtype=np.uint8, count=region, offset=offset)
    host32 = host.view(np.uint32)

    result["resident_before"] = resident_fraction(host)
    before = disk_bytes_read()
    began = time.perf_counter()
    wrapped = mx.from_dlpack(host32, copy=False)
    wrap_s = time.perf_counter() - began
    wrap_read = disk_bytes_read() - before

    words = ROW // 4
    index = mx.array(
        np.concatenate([np.arange(r * words, (r + 1) * words, dtype=np.uint32) for r in rows])
    )
    before = disk_bytes_read()
    began = time.perf_counter()
    total = mx.take(wrapped, index).astype(mx.uint64).sum()
    mx.eval(total)
    gather_cold_s = time.perf_counter() - began
    gather_cold_read = disk_bytes_read() - before
    result["resident_after_gather"] = resident_fraction(host)

    began = time.perf_counter()
    total_warm = mx.take(wrapped, index).astype(mx.uint64).sum()
    mx.eval(total_warm)
    gather_warm_s = time.perf_counter() - began

    expected = int(sum(int(host32[r * words:(r + 1) * words].astype(np.uint64).sum()) for r in rows))
    result.update({
        "wrap_seconds": wrap_s, "wrap_physical_bytes": wrap_read,
        "gather_cold_seconds": gather_cold_s,
        "gather_cold_physical_bytes": gather_cold_read,
        "touched_bytes": len(rows) * ROW,
        "gather_warm_seconds": gather_warm_s,
        "value_matches_host": int(total.item()) == expected,
    })
    del wrapped, index, total, total_warm, host32, host
    mapping.close()
    os.close(fd)

    # Normal path for the same rows: pread into fresh buffers, then wrap.
    # These rows are now cached, so this times the warm copy path.
    fd = os.open(shard, os.O_RDONLY)
    before = disk_bytes_read()
    began = time.perf_counter()
    buffer = np.empty(len(rows) * ROW, dtype=np.uint8)
    for position, r in enumerate(rows):
        chunk = os.pread(fd, ROW, offset + r * ROW)
        buffer[position * ROW:(position + 1) * ROW] = np.frombuffer(chunk, dtype=np.uint8)
    array = mx.from_dlpack(buffer.view(np.uint32), copy=False)
    total = array.astype(mx.uint64).sum()
    mx.eval(total)
    result.update({
        "pread_seconds": time.perf_counter() - began,
        "pread_physical_bytes": disk_bytes_read() - before,
        "pread_value_matches": int(total.item()) == expected,
    })
    os.close(fd)

    target.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
