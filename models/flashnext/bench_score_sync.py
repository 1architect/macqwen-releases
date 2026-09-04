#!/usr/bin/env python3
"""Attribute score-sync time on the frozen 60-slot Frontier 8A control.

This is a single-control diagnostic. It does not compare arms or change model
outputs. The default run decodes 24 tokens and prints one attribution row per
token. Set ``--passes`` to repeat the same control and verify token IDs.

Example::

    .venv/bin/python models/flashnext/bench_score_sync.py --model PATH
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx


PROMPT = "Explique a fotossintese em duas frases."

# This is slabpack60_skew with Frontier 8A. Every value is explicit so a shell
# environment cannot silently turn this diagnostic into another experiment.
FROZEN_CONTROL_ENV = {
    "FLASHNEXT_METAL_RUNTIME": "1",
    "FLASHNEXT_SLAB": "0",
    "FLASHNEXT_SLAB_LAYERS": "0",
    "FLASHNEXT_SLAB_GLOBAL": "60",
    "FLASHNEXT_SLAB_PACK": "1",
    "FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING": "1",
    "FLASHNEXT_SLAB_POLICY": "skew",
    "FLASHNEXT_SLAB_MIN_SLOTS": "4",
    "FLASHNEXT_SLAB_MAX_SLOTS": "6",
    "FLASHNEXT_SLAB_NUM_LAYERS": "12",
    "FLASHNEXT_FUSED_SHARED": "1",
    "FLASHNEXT_FUSED_SHARED_PARTS": "0",
    "FLASHNEXT_FUSED_UP_SWIGLU": "0",
    "FLASHNEXT_STREAM_PACK": "0",
    "FLASHNEXT_STREAM_PACK_CHUNK": "0",
    "FLASHNEXT_EARLY_SUBMIT": "0",
    "FLASHNEXT_WARM": "0",
    "FLASHNEXT_TOPK_THRESHOLD": "0.85",
    "FLASHNEXT_PROFILE_IO": "0",
    "FLASHNEXT_PROFILE_SCORE_SYNC": "1",
}


def configure_frozen_control() -> dict[str, str]:
    """Apply and return the exact control environment."""
    for key, value in FROZEN_CONTROL_ENV.items():
        os.environ[key] = value
    return dict(FROZEN_CONTROL_ENV)


def _delta(before: dict, after: dict) -> dict:
    """Return one token's score-sync counter delta."""
    pool = {
        state: {
            edge: after["pool"][state][edge] - before["pool"][state][edge]
            for edge in ("before", "after")
        }
        for state in ("queued", "running", "completed")
    }
    return {
        "wall_seconds": after["wall_seconds"] - before["wall_seconds"],
        "physical_bytes": after["physical_bytes"] - before["physical_bytes"],
        "calls": after["calls"] - before["calls"],
        "threshold_active_calls": (
            after["threshold_active_calls"]
            - before["threshold_active_calls"]
        ),
        "threshold_inactive_calls": (
            after["threshold_inactive_calls"]
            - before["threshold_inactive_calls"]
        ),
        "pool": pool,
    }


def format_attribution(token_number: int, sample: dict) -> str:
    """Format one token's wall, byte, threshold, and pool attribution."""
    active = sample["threshold_active_calls"]
    inactive = sample["threshold_inactive_calls"]
    threshold = "active" if active else "inactive"
    pool = sample["pool"]
    pool_text = " ".join(
        f"{state[0]}={pool[state]['before']}/{pool[state]['after']}"
        for state in ("queued", "running", "completed")
    )
    return (
        f"token {token_number:03d}  threshold={threshold} "
        f"active={active} inactive={inactive} calls={sample['calls']}  "
        f"score_sync={sample['wall_seconds'] * 1000:.2f} ms  "
        f"physical={sample['physical_bytes'] / 1e6:.3f} MB  "
        f"pool {pool_text}"
    )


def decode_pass(language, ids, count: int) -> list[int]:
    """Decode one pass and print score-sync attribution per token."""
    from models.flashnext.expert_cache import (
        profile_totals,
        reset_profile,
        score_sync_totals,
        set_profile,
        set_score_sync_profile,
    )

    set_profile(False)
    set_score_sync_profile(True)
    language._position_ids = None
    language._rope_deltas = None
    cache = language.make_cache()
    output = language(ids, cache=cache)
    token = mx.argmax(output.logits[:, -1, :], axis=-1)
    mx.eval(token)
    output = None
    mx.clear_cache()
    reset_profile()

    produced = []
    for number in range(1, count + 1):
        produced.append(int(token.item()))
        before = score_sync_totals()
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        mx.eval(token)
        after = score_sync_totals()
        print(format_attribution(number, _delta(before, after)), flush=True)
    # Keep the normal totals available to callers and make the profile explicit
    # in the final report without enabling the broad I/O profiler.
    totals = profile_totals()
    print(
        f"score_sync total={totals['score_sync'] * 1000:.2f} ms  "
        f"physical={totals['score_sync_physical_bytes'] / 1e6:.3f} MB",
        flush=True,
    )
    return produced


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--passes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.tokens < 1 or args.passes < 1:
        parser.error("--tokens and --passes must be at least 1")

    configure_frozen_control()
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.loader import load_streaming
    from transformers import AutoTokenizer

    model_path = str(resolve_flashnext(str(args.model) if args.model else None))
    model, _, _ = load_streaming(
        model_path,
        expert_capacity=0,
        verbose=False,
        keep_vision=False,
        use_mtp=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]
    language = model.language_model

    print(f"model {model_path}")
    print("control slabpack60_skew Frontier 8A, stream packs disabled")
    reference = None
    for index in range(1, args.passes + 1):
        print(f"\npass {index}/{args.passes}")
        produced = decode_pass(language, ids, args.tokens)
        if reference is None:
            reference = produced
        elif produced != reference:
            print("REFUSED: token IDs changed between passes", file=sys.stderr)
            return 1
    if args.passes > 1:
        print("token IDs identical across passes: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
