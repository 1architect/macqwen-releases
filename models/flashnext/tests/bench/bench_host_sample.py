#!/usr/bin/env python3
"""Where the main thread spends a decode token, by source line, at P15.

A background thread samples the main thread's Python stack every
``--interval-ms`` during decode with the normal-chat backend (chat defaults,
checkpoint pin policy, keep-warm on). Each sample is charged to the innermost
frame inside this repository or MLX-VLM, so blocking calls (``mx.eval``,
future waits, keep-warm latches) and pure Python graph construction show up
as separate lines. ``sys.setswitchinterval`` is lowered so the sampler also
gets the GIL while the main thread runs Python. The sampler adds some work of
its own; the shares, not the absolute rate, are the result.

    python -m models.flashnext.tests.bench.bench_host_sample --tokens 96
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402
from models.flashnext.tests.bench.bench_long_states import PROMPT  # noqa: E402

KEEP = (str(ROOT), "mlx_vlm")


def site(frame):
    """Innermost frame in this repository or mlx_vlm, plus its caller chain."""
    chain = []
    while frame is not None:
        name = frame.f_code.co_filename
        if any(part in name for part in KEEP) and "bench_host_sample" not in name:
            short = name.split("site-packages/")[-1].replace(str(ROOT) + "/", "")
            chain.append(f"{short}:{frame.f_lineno} {frame.f_code.co_name}")
        frame = frame.f_back
    return chain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--interval-ms", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=45)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "host-sample", "host-sample.json", args.json)

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-host-sample-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from macqwen.backends.flashnext import FlashNextBackend
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32
    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)

    main_id = threading.get_ident()
    leaf, inclusive = Counter(), Counter()
    state = {"on": False, "stop": False, "samples": 0}

    def sampler():
        period = args.interval_ms / 1000
        while not state["stop"]:
            if state["on"]:
                frame = sys._current_frames().get(main_id)
                chain = site(frame) if frame is not None else []
                state["samples"] += 1
                leaf[chain[0] if chain else "(outside repository)"] += 1
                for entry in set(chain):
                    inclusive[entry] += 1
            time.sleep(period)

    sys.setswitchinterval(0.0002)
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    backend.reset()
    backend.append_text(PROMPT)
    timing = {}

    def on_prefilled():
        timing["began"] = time.perf_counter()
        state["on"] = True

    _text, stats = backend.generate(args.tokens, on_prefilled=on_prefilled)
    state["on"] = False
    seconds = time.perf_counter() - timing["began"]
    state["stop"] = True
    thread.join()
    ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
    ms_token = seconds / max(1, stats.tokens) * 1000
    total = max(1, state["samples"])

    def table(counter):
        return [
            {"site": key, "share": round(count / total, 4),
             "ms_per_token": round(count / total * ms_token, 2)}
            for key, count in counter.most_common(args.top)
        ]

    evidence = {
        "checkpoint": checkpoint, "tokens": stats.tokens,
        "ms_per_token": round(ms_token, 2), "samples": state["samples"],
        "digest": hashlib.sha256(repr(ids).encode()).hexdigest()[:16],
        "leaf": table(leaf), "inclusive": table(inclusive),
    }
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"{stats.tokens} tokens, {ms_token:.1f} ms/token, {state['samples']} samples, "
          f"digest {evidence['digest']}")
    print("-- leaf (where the main thread is) --")
    for row in evidence["leaf"]:
        print(f"{row['ms_per_token']:7.2f} ms  {row['share']*100:5.1f}%  {row['site']}")
    print("-- inclusive --")
    for row in evidence["inclusive"][:30]:
        print(f"{row['ms_per_token']:7.2f} ms  {row['share']*100:5.1f}%  {row['site']}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
