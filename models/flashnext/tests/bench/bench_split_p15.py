#!/usr/bin/env python3
"""Split one decode token into its waits with the chat defaults at P15.

Every earlier cost split ran before GPU keep-warm, with the GPU clock
collapsed during read waits. This one loads the normal-chat backend (chat
environment, checkpoint pin policy, frozen slab, keep-warm on) and decodes the
long-states prompt twice in one process: an unprofiled arm for the reference
rate, then a profiled arm (`FLASHNEXT_PROFILE_IO`) for the split. Both arms
must produce the same tokens. The profiled arm's timers are attribution only;
its rate is not a throughput result.

    python -m models.flashnext.tests.bench.bench_split_p15 --tokens 128
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402
from models.flashnext.tests.bench.bench_long_states import PROMPT, Clocks  # noqa: E402

# Timers the runtime keeps in `expert_cache._TIMERS`, in seconds. The rest of
# the token is host work plus the final token sync in the decode loop.
BUCKETS = (
    "io_wait", "to_mx", "moe_issue", "router_sync", "score_sync",
    "topk_python", "shared_expert", "ngram_wait",
)


def decode(backend, tokens, clocks, meter):
    backend.reset()
    backend.append_text(PROMPT)
    state = {}

    def on_prefilled():
        from models.flashnext.expert_cache import reset_profile

        reset_profile()
        meter.reset()
        clocks.mark()
        state["began"] = time.perf_counter()

    _text, stats = backend.generate(tokens, on_prefilled=on_prefilled)
    seconds = time.perf_counter() - state["began"]
    read = meter.bytes_since()
    gpu = (clocks.mark() or {}).get("GPUPH", {})
    ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
    return {
        "tokens": stats.tokens,
        "gen_rate": round(stats.rate, 3),
        "ms_per_token": round(seconds / max(1, stats.tokens) * 1000, 2),
        "mb_per_token": round(read / 1e6 / max(1, stats.tokens), 1),
        "gpu_mean_state": gpu.get("mean_active_state"),
        "gpu_active_share": gpu.get("active_share"),
        "digest": hashlib.sha256(repr(ids).encode()).hexdigest()[:16],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "split-p15", "split-p15.json", args.json)

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_PROFILE_IO"] = "0"
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-split-p15-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from macqwen.backends.flashnext import FlashNextBackend
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext import expert_cache
    from models.flashnext.checkpoint_policy import resolve_resident_experts
    from models.flashnext.diskio import ReadMeter

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32
    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    clocks, meter = Clocks(), ReadMeter()
    evidence = {
        "checkpoint": checkpoint, "resident_experts": resident,
        "tokens": args.tokens, "status": "running",
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("FLASHNEXT_")
        },
    }

    def save():
        target.write_text(json.dumps(evidence, indent=1) + "\n")

    save()
    reference = decode(backend, args.tokens, clocks, meter)
    evidence["reference"] = reference
    save()
    print(f"reference  {reference}", flush=True)

    expert_cache.set_profile(True)
    try:
        profiled = decode(backend, args.tokens, clocks, meter)
    finally:
        expert_cache.set_profile(False)
    totals = expert_cache.profile_totals()
    count = max(1, profiled["tokens"])
    split = {key: round(totals[key] / count * 1000, 2) for key in BUCKETS}
    split["remainder"] = round(profiled["ms_per_token"] - sum(split.values()), 2)
    profiled["split_ms_per_token"] = split
    profiled["io_calls_per_token"] = round(totals["io_calls"] / count, 2)
    # Subset of io_wait: the stream-pack (slab) layers only.
    profiled["io_wait_packed_ms_per_token"] = round(
        totals.get("io_wait_packed", 0.0) / count * 1000, 2
    )
    profiled["io_calls_packed_per_token"] = round(
        totals.get("io_calls_packed", 0) / count, 2
    )
    profiled["pread_calls_per_token"] = round(totals["pread_calls"] / count, 1)
    evidence["profiled"] = profiled
    evidence["status"] = (
        "completed" if profiled["digest"] == reference["digest"] else "digest_mismatch"
    )
    save()
    print(f"profiled   {profiled}", flush=True)
    for key, value in split.items():
        print(f"  {key:14s} {value:8.2f} ms/token", flush=True)
    print(f"  io_wait_packed {profiled['io_wait_packed_ms_per_token']:8.2f} ms/token over "
          f"{profiled['io_calls_packed_per_token']} layers, "
          f"{profiled['pread_calls_per_token']} preads/token", flush=True)
    print(f"wrote {target}")
    return 0 if evidence["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
