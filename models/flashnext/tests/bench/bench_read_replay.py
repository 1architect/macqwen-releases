#!/usr/bin/env python3
"""Replay recorded decode routes through different expert read engines.

No model is loaded. The expert reads of real chat turns
(``bench_route_trace``) are replayed layer by layer from the checkpoint
shards, with a fixed sleep per layer standing in for the GPU phase and an
optional anonymous ballast standing in for the model's own memory. Only the
read path changes between engines, so the replay prices read mechanics that a
full-model run cannot separate.

Engines:

- ``production``: today's decode read path. Slab-pack layers read cold
  experts into one expert-major record through ``_StreamedPackRead``; other
  layers read per projection into shared buffers and wrap them for MLX. The
  current frozen slab and each turn's recorded pin set apply; keep-warm and
  QoS follow the chat environment.
- ``pool-direct`` / ``pool-buffered``: one application-owned LRU pool of
  expert records sized by ``--pool-gb``. A hit costs nothing; a miss reads its
  nine rows into a slot, with ``F_NOCACHE`` descriptors (direct) or the
  store's normal descriptors (buffered). Slab experts stay outside the pool.

The ``production`` engine must reproduce the model's read wait before any
pool number is trusted.

    python -m models.flashnext.tests.bench.bench_read_replay --engine production
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402

TRACE = ROOT / "results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
PAGE = 16384


def evict_expert_pages(store) -> int:
    """Invalidate the cached pages of every routed expert tensor."""
    import ctypes
    import mmap

    libc = ctypes.CDLL(None, use_errno=True)
    ms_invalidate = 0x2
    total = 0
    for name, ref in store.refs.items():
        if ".switch_mlp." not in name or "mtp" in name:
            continue
        base = store._shared_view(name).__array_interface__["data"][0]
        start = base - base % mmap.PAGESIZE
        end = base + ref.shape[0] * ref.row_bytes
        length = -(-(end - start) // mmap.PAGESIZE) * mmap.PAGESIZE
        if libc.msync(ctypes.c_void_p(start), ctypes.c_size_t(length), ms_invalidate):
            raise OSError(ctypes.get_errno(), f"msync failed for {name}")
        total += length
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True,
                        choices=("production", "production-native",
                                 "pool-direct", "pool-buffered"))
    parser.add_argument("--tokens", type=int, default=128, help="decode tokens per turn")
    parser.add_argument("--gap-ms", type=float, default=2.6,
                        help="sleep per layer standing in for the GPU phase")
    parser.add_argument("--pool-gb", type=float, default=4.0)
    parser.add_argument("--ballast-gb", type=float, default=3.4)
    parser.add_argument("--evict", action="store_true",
                        help="drop every expert tensor's clean pages from the "
                             "file cache before the replay (msync MS_INVALIDATE), "
                             "so each engine starts from the same cold state")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "read-replay", f"{args.engine}.json", args.json)

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    import mlx.core as mx
    import numpy as np

    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext import expert_cache as ec
    from models.flashnext.diskio import disk_bytes_read, free_memory_mb
    from models.flashnext.slab_pack import RECORD_STRIDE
    from models.flashnext.store import SafeTensorStore

    trace = json.load(gzip.open(TRACE))
    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    store = SafeTensorStore(checkpoint)
    slab = ec._slab_allocation(store, int(os.environ["FLASHNEXT_SLAB_GLOBAL"]), 32)
    slab = {int(layer): set(experts) for layer, experts in slab.items()}
    prefix = {
        layer: f"language_model.model.layers.{layer}.mlp.switch_mlp"
        for layer in range(48)
    }
    offsets = dict(((p, part), off) for p, part, off in ec._STREAM_RECORD_PARTS)
    gap = args.gap_ms / 1000

    evicted = None
    if args.evict:
        evicted = evict_expert_pages(store)

    ballast = None
    if args.ballast_gb > 0:
        ballast = subprocess.Popen(
            [sys.executable, "-m", "models.flashnext.tests.bench.memory_load",
             "--gb", str(args.ballast_gb)],
            cwd=ROOT, stdout=subprocess.PIPE, text=True,
        )
        ballast.stdout.readline()

    # ---- engines ---------------------------------------------------------
    lrus = {
        (layer, projection): ec.ExpertLRU(store, f"{prefix[layer]}.{projection}")
        for layer in range(48) for projection in PROJECTIONS
    }
    chunk = int(os.environ.get("FLASHNEXT_STREAM_PACK_CHUNK", "2"))

    def production_layer(layer, experts):
        wanted = [e for e in experts if e not in slab.get(layer, ())]
        if not wanted:
            return 0
        if layer in slab:
            buffer = np.empty(len(wanted) * RECORD_STRIDE, dtype=np.uint8)
            futures = []
            for projection, part, offset in ec._STREAM_RECORD_PARTS:
                name = f"{prefix[layer]}.{projection}.{part}"
                for start in range(0, len(wanted), chunk):
                    piece = wanted[start:start + chunk]
                    view = store.expert_record_view(
                        name, buffer, len(piece), offset + start * RECORD_STRIDE,
                        RECORD_STRIDE,
                    )
                    futures.append(ec._submit_read(
                        store.rows_into, name, piece, view, store._read_mode,
                    ))
            ec._StreamedPackRead(buffer, futures).wait()
            mx.from_dlpack(buffer, copy=False)
        else:
            projections = [lrus[(layer, p)] for p in PROJECTIONS]
            pending = [p.submit(wanted) for p in projections]
            raw = ec._await_projection_tasks(pending)
            for projection, chunks in zip(projections, raw):
                projection.to_mx(chunks)
        return len(wanted)

    from models.flashnext.tests.bench import native_read

    def native_layer(layer, experts):
        """Production destinations, one native call per layer."""
        wanted = [e for e in experts if e not in slab.get(layer, ())]
        if not wanted:
            return 0
        batch = native_read.Batch()
        raw = None
        if layer in slab:
            buffer = np.empty(len(wanted) * RECORD_STRIDE, dtype=np.uint8)
            base = native_read.address(buffer)
            for projection, part, offset in ec._STREAM_RECORD_PARTS:
                ref = store.refs[f"{prefix[layer]}.{projection}.{part}"]
                batch.add(store._fd(ref.shard), ref.start, ref.row_bytes, wanted,
                          base + offset, RECORD_STRIDE)
        else:
            raw = []
            for projection in PROJECTIONS:
                parts = []
                for part in PARTS:
                    name = f"{prefix[layer]}.{projection}.{part}"
                    ref = store.refs[name]
                    destination = store.empty_rows(name, len(wanted))
                    batch.add(store._fd(ref.shard), ref.start, ref.row_bytes, wanted,
                              native_read.address(destination), ref.row_bytes)
                    parts.append(ec._SharedRead(destination, []))
                raw.append(parts)
        future = ec._submit_read(batch.run)
        if ec._KEEPWARM[0]:
            ec._keep_gpu_warm_until([future])
        ec._resolve_future(future, None)
        if raw is None:
            mx.from_dlpack(buffer, copy=False)
        else:
            for projection, chunks in zip(PROJECTIONS, raw):
                lrus[(layer, projection)].to_mx(chunks)
        return len(wanted)

    pool = None
    if args.engine.startswith("pool"):
        stride = -(-RECORD_STRIDE // PAGE) * PAGE
        slots = int(args.pool_gb * 1e9) // stride
        pool = np.empty(slots * stride, dtype=np.uint8)
        pool[::PAGE] = 0  # make every page real
        slot_of: "OrderedDict[tuple, int]" = OrderedDict()
        free_slots = list(range(slots))
        direct_fds = {}

        def fd_for(shard):
            if args.engine == "pool-buffered":
                return store._fd(shard)
            fd = direct_fds.get(shard)
            if fd is None:
                fd = os.open(os.path.join(store.dir, shard), os.O_RDONLY)
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
                direct_fds[shard] = fd
            return fd

        view = memoryview(pool)

        def fill(name, ref_start, row_bytes, shard, targets):
            fd = fd_for(shard)
            for expert, base in targets:
                read = os.preadv(fd, [view[base:base + row_bytes]],
                                 ref_start + expert * row_bytes)
                if read != row_bytes:
                    raise OSError(f"short read {name} {expert}")

        def pool_layer(layer, experts):
            wanted = [e for e in experts if e not in slab.get(layer, ())]
            misses = []
            for expert in wanted:
                key = (layer, expert)
                if key in slot_of:
                    slot_of.move_to_end(key)
                    continue
                if free_slots:
                    slot = free_slots.pop()
                else:
                    _old, slot = slot_of.popitem(last=False)
                slot_of[key] = slot
                misses.append((expert, slot))
            if not misses:
                return 0
            futures = []
            for projection in PROJECTIONS:
                for part in PARTS:
                    name = f"{prefix[layer]}.{projection}.{part}"
                    ref = store.refs[name]
                    offset = offsets[(projection, part)]
                    for start in range(0, len(misses), chunk):
                        targets = [
                            (expert, slot * stride + offset)
                            for expert, slot in misses[start:start + chunk]
                        ]
                        futures.append(ec._submit_read(
                            fill, name, ref.start, ref.row_bytes, ref.shard, targets,
                        ))
            if ec._KEEPWARM[0]:
                ec._keep_gpu_warm_until(futures)
            for future in futures:
                ec._resolve_future(future, None)
            return len(misses)

    run_layer = {
        "production": production_layer, "production-native": native_layer,
    }.get(args.engine, pool_layer if pool is not None else None)

    # ---- replay ----------------------------------------------------------
    events_by_turn = {}
    for turn, phase, token, layer, experts in trace["events"]:
        if phase == "decode" and 1 <= token <= args.tokens:
            events_by_turn.setdefault(turn, []).append((token, layer, experts))
    evidence = {
        "engine": args.engine, "tokens_per_turn": args.tokens, "gap_ms": args.gap_ms,
        "pool_gb": args.pool_gb if pool is not None else None,
        "ballast_gb": args.ballast_gb, "slab_layers": sorted(slab),
        "evicted_expert_bytes": evicted,
        "trace": str(TRACE.relative_to(ROOT)), "turns": [],
        "free_mb_before": free_memory_mb(),
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith("FLASHNEXT_")},
    }
    try:
        for turn, events in sorted(events_by_turn.items()):
            info = trace["turns"][turn]
            pinned_names = []
            if args.engine.startswith("production"):
                for layer, experts in (info.get("pinned") or {}).items():
                    for projection in PROJECTIONS:
                        for part in PARTS:
                            name = f"{prefix[int(layer)]}.{projection}.{part}"
                            store.pin_rows(name, list(experts))
                            pinned_names.append(name)
            per_token, read_ms, bytes_before = {}, {}, disk_bytes_read()
            token_bytes = {}
            last_token, token_began = None, None
            streamed = 0
            for token, layer, experts in events:
                if token != last_token:
                    now_bytes = disk_bytes_read()
                    if last_token is not None:
                        token_bytes[last_token] = now_bytes - token_began
                    token_began, last_token = now_bytes, token
                began = time.perf_counter()
                streamed += run_layer(layer, experts)
                read_ms[token] = read_ms.get(token, 0.0) + (time.perf_counter() - began) * 1000
                time.sleep(gap)
            token_bytes[last_token] = disk_bytes_read() - token_began
            store.unpin_all()
            tokens = sorted(read_ms)
            steady = tokens[8:]  # the production pins apply after token 9
            result = {
                "prompt": info["prompt"], "tokens": len(tokens),
                "read_ms_per_token": round(statistics.mean(read_ms[t] for t in steady), 1),
                "mb_per_token": round(statistics.mean(token_bytes[t] for t in steady) / 1e6, 1),
                "streamed_experts_per_layer": round(streamed / (len(tokens) * 48), 2),
                "model_mb_per_token": round(statistics.mean(
                    info["physical_bytes_per_token"][t] for t in steady
                    if t < len(info["physical_bytes_per_token"])) / 1e6, 1),
            }
            evidence["turns"].append(result)
            target.write_text(json.dumps(evidence, indent=1) + "\n")
            print(f"{args.engine:13s} {result['prompt']:14s} read {result['read_ms_per_token']:6.1f} ms/token  "
                  f"{result['mb_per_token']:6.1f} MB/token (model {result['model_mb_per_token']})  "
                  f"misses/layer {result['streamed_experts_per_layer']}", flush=True)
    finally:
        if ballast is not None:
            ballast.send_signal(signal.SIGTERM)
            ballast.wait()
    evidence["status"] = "completed"
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
