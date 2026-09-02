#!/usr/bin/env python3
"""Capture a few decode tokens as a Metal .gputrace.

The Metal System Trace template gives one interval per command buffer and MLX
leaves its encoders unlabelled, so about 127 ms per token of GPU span time has
never been attributed to a kernel. A GPU capture is the instrument that can:
Xcode's GPU pipeline profiler reports per-dispatch timing inside each command
buffer.

Capture only a couple of steady-state tokens. A token issues roughly 250
command buffers, so the file grows quickly, and the load and prefill are not
what needs explaining.

Run with capture enabled, which Metal reads when it creates the device:

    MTL_CAPTURE_ENABLED=1 ~/models/.venv-qwen4exp/bin/python \\
      models/flashnext/capture_dispatches.py --out /tmp/decode.gputrace
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx
from transformers import AutoTokenizer

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import set_threshold
from models.flashnext.loader import load_streaming

PROMPT = "Explique a fotossintese em duas frases."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/decode.gputrace")
    parser.add_argument("--warm", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=2)
    args = parser.parse_args()

    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise SystemExit(
            "set MTL_CAPTURE_ENABLED=1 before starting python; Metal reads it "
            "when the device is created and a later setenv does nothing"
        )
    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")

    model_dir = str(resolve_flashnext())
    model, _, _ = load_streaming(
        model_dir, expert_capacity=0, verbose=False, keep_vision=False,
        use_mtp=False,
    )
    language = model.language_model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    set_threshold(0.85)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="np")["input_ids"]
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(language)
    token = mx.argmax(language(mx.array(ids), cache=cache).logits[:, -1, :], axis=-1)
    mx.eval(token)

    def step():
        nonlocal token
        logits = language(token[None], cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

    for _ in range(args.warm):
        step()
    print(f"warm done, capturing {args.tokens} tokens", flush=True)

    if os.path.exists(args.out):
        shutil.rmtree(args.out, ignore_errors=True)
    mx.metal.start_capture(args.out)
    began = time.perf_counter()
    for _ in range(args.tokens):
        step()
    elapsed = time.perf_counter() - began
    mx.metal.stop_capture()

    size = subprocess.run(["du", "-sh", args.out], capture_output=True,
                          text=True).stdout.split()[0]
    print(f"captured {args.tokens} tokens in {elapsed * 1000 / args.tokens:.1f} "
          f"ms/token", flush=True)
    print(f"trace: {args.out}  ({size})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
