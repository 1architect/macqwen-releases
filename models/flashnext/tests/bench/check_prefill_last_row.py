#!/usr/bin/env python3
"""Gate for FLASHNEXT_PREFILL_LAST_ROW: identical final logits, then timing.

For each prompt length, prefills the same prompt twice on fresh caches:
once through the full-logit path (every position projected) and once through
the last-row path (only the final hidden row projected). It compares the
final-row logits with ``mx.array_equal`` and reports prefill time and MLX peak
memory for both. Order alternates between lengths.

    python -m models.flashnext.tests.bench.check_prefill_last_row --lengths 128 512 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 1024])
    args = parser.parse_args()
    target = output_path("flashnext", "prefill-last-row", "prefill-last-row.json")

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    import tempfile

    os.environ["FLASHNEXT_PIN_CACHE"] = os.path.join(
        tempfile.mkdtemp(prefix="flashnext-last-row-"), "pins.json")

    import mlx.core as mx

    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend()
    language = backend.language
    source = (ROOT / "docs" / "flashnext" / "research.md").read_text()
    all_ids = backend.tokenizer.encode(source)
    results = []

    def full(ids):
        cache = language.make_cache()
        mx.reset_peak_memory()
        began = time.perf_counter()
        logits = language(ids, cache=cache).logits[:, -1, :]
        mx.eval(logits)
        seconds = time.perf_counter() - began
        return logits, seconds, mx.get_peak_memory()

    def last_row(ids):
        cache = language.make_cache()
        mx.reset_peak_memory()
        began = time.perf_counter()
        output = language(ids, cache=cache, return_hidden=True, skip_logits=True)
        hidden = output.hidden_states[-1][:, -1:]
        logits = language.speculative_logits_from_hidden(hidden).reshape(1, -1)
        mx.eval(logits)
        seconds = time.perf_counter() - began
        return logits, seconds, mx.get_peak_memory()

    for position, length in enumerate(args.lengths):
        ids = mx.array(all_ids[:length])[None]
        order = (full, last_row) if position % 2 == 0 else (last_row, full)
        outputs = {}
        for function in order:
            language._position_ids = None
            language._rope_deltas = None
            outputs[function.__name__] = function(ids)
            mx.clear_cache()
        full_logits, full_s, full_peak = outputs["full"]
        row_logits, row_s, row_peak = outputs["last_row"]
        full_logits = full_logits.reshape(row_logits.shape).astype(row_logits.dtype)
        equal = bool(mx.array_equal(full_logits, row_logits).item())
        difference = float(mx.max(mx.abs(
            full_logits.astype(mx.float32) - row_logits.astype(mx.float32))).item())
        result = {
            "tokens": length, "equal": equal, "max_abs_difference": difference,
            "argmax_equal": int(mx.argmax(full_logits).item()) == int(mx.argmax(row_logits).item()),
            "full_seconds": round(full_s, 3), "last_row_seconds": round(row_s, 3),
            "full_peak_gb": round(full_peak / 1e9, 3), "last_row_peak_gb": round(row_peak / 1e9, 3),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        target.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
