#!/usr/bin/env python3
"""Prefill rate, physical reads and I/O wait for long prompts, paired arms.

One loaded normal-chat backend prefills the same long prompt under each
condition in reversed rounds, then decodes a few greedy tokens so a changed
runtime cannot pass with different output. A 2,048-token prefill reads tens
of gigabytes, far more than the page cache holds, so each arm starts close to
cold whatever ran before it.

Conditions flip live switches only:

* ``baseline``: current reads.
* ``ngram-parallel``: n-gram rows of large lookups read on the I/O pool
  (FLASHNEXT_NGRAM_PARALLEL_MIN=64).

Read coalescing and the prefill MoE pipeline were measured on 2026-09-22 and
removed; the research log keeps their results.

    python -m models.flashnext.tests.bench.bench_prefill_io --tokens 2048 \
        --conditions baseline ngram-parallel --rounds 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402

CONDITIONS = ("baseline", "ngram-parallel")


def apply_condition(backend, name: str) -> None:
    from models.flashnext import ngram

    del backend
    ngram.set_parallel_min_rows(64 if name == "ngram-parallel" else 0)


def build_prompt(tokenizer, tokens: int) -> str:
    source = (ROOT / "docs" / "flashnext" / "research.md").read_text()
    ids = tokenizer.encode(source)[: max(16, tokens - 40)]
    body = tokenizer.decode(ids)
    return ("<|im_start|>user\nResuma o texto a seguir em uma frase.\n\n" + body
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["baseline", "ngram-parallel"])
    parser.add_argument("--profile-io", action="store_true")
    args = parser.parse_args()
    target = output_path("flashnext", "prefill-io", "prefill-io.json")

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_GPU_KEEPWARM"] = "0"
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-prefill-io-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext import expert_cache
    from models.flashnext.diskio import ReadMeter, free_memory_mb

    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    prompt = build_prompt(backend.tokenizer, args.tokens)
    meter = ReadMeter()
    evidence = {"checkpoint": checkpoint, "tokens": args.tokens,
                "conditions": args.conditions, "arms": [], "status": "running"}
    if args.profile_io:
        expert_cache.set_profile(True)
    first_digest = None
    for round_index in range(args.rounds):
        order = args.conditions if round_index % 2 == 0 else list(reversed(args.conditions))
        for name in order:
            apply_condition(backend, name)
            backend.reset()
            backend.append_text(prompt)
            if args.profile_io:
                expert_cache.reset_profile()
            marks = {}
            free = free_memory_mb()
            import mlx.core as mx

            mx.reset_peak_memory()
            meter.reset()
            began = time.perf_counter()

            def on_prefilled():
                marks["prefill_s"] = time.perf_counter() - began
                marks["prefill_bytes"] = meter.bytes_since()
                marks["peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
                if args.profile_io:
                    marks["profile"] = {
                        key: round(value, 4) if isinstance(value, float) else value
                        for key, value in expert_cache.profile_totals().items()
                    }

            _text, stats = backend.generate(args.decode, on_prefilled=on_prefilled)
            ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
            digest = hashlib.sha256(repr(ids).encode()).hexdigest()[:16]
            arm = {
                "round": round_index + 1, "condition": name,
                "prompt_tokens": stats.prompt_tokens,
                "prefill_s": round(marks["prefill_s"], 3),
                "prefill_tok_s": round(stats.prompt_tokens / marks["prefill_s"], 2),
                "prefill_gb": round(marks["prefill_bytes"] / 1e9, 3),
                "drive_gb_s": round(marks["prefill_bytes"] / 1e9 / marks["prefill_s"], 3),
                "free_mb_before": free, "digest": digest,
                "mlx_peak_gb": marks["peak_gb"],
                "profile": marks.get("profile"),
            }
            evidence["arms"].append(arm)
            target.write_text(json.dumps(evidence, indent=1) + "\n")
            print(f"round {round_index + 1} {name:18s} {arm['prompt_tokens']} tok "
                  f"{arm['prefill_s']:7.2f} s  {arm['prefill_tok_s']:6.2f} tok/s  "
                  f"{arm['prefill_gb']:6.2f} GB  {arm['drive_gb_s']:.2f} GB/s  "
                  f"peak {arm['mlx_peak_gb']} GB  "
                  f"digest {digest}", flush=True)
            if first_digest is None:
                first_digest = digest
            elif digest != first_digest:
                evidence["status"] = "digest_mismatch"
                target.write_text(json.dumps(evidence, indent=1) + "\n")
                raise SystemExit(f"{name} produced different tokens")
    summary = {}
    for name in args.conditions:
        rates = [a["prefill_tok_s"] for a in evidence["arms"] if a["condition"] == name]
        summary[name] = {"median_tok_s": statistics.median(rates), "arms": rates}
    evidence["summary"] = summary
    evidence["status"] = "completed"
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
