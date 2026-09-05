#!/usr/bin/env python3
"""Bracket the decode rate between an all-RAM and an all-disk expert gather.

Every earlier probe changed how bytes move and the token time did not follow.
This one changes only where the bytes come from. Both arms route the same
number of experts per layer, issue the same number of reads and run the same
GPU work, so the difference between them is the disk and nothing else.

`ram` pins one fixed expert set and reuses it every token, so every read is
served from memory. `disk` picks a fresh random set per layer per token out of
512, so almost every read reaches the drive. Production sits between them.

Output is not the model's real reply. This measures time, not text.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import set_threshold
from models.flashnext.expert_cache import profile_totals, reset_profile
from models.flashnext.loader import load_streaming

MODEL = str(resolve_flashnext())
PROMPT = "Explique a fotossintese em duas frases."
PARTS = ("weight", "scales", "biases")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


_PICK = {"mode": "ram", "fixed": [], "rng": None, "fixed_route": False,
         "miss": 0.0}


def patch(language) -> None:
    """Replace the routed expert list on the class, keeping every shape.

    Assigning `__call__` on an instance does nothing: Python resolves special
    methods on the type. Patch the shared class once instead.
    """
    blocks = [
        layer.mlp.switch_mlp
        for layer in language.model.layers
        if getattr(layer.mlp, "switch_mlp", None) is not None
    ]
    cls = type(blocks[0])
    for block in blocks:
        if type(block) is not cls:
            raise RuntimeError("switch_mlp classes differ across layers")
    original = cls.__call__

    def call(self, x, inds, *args, _o=original, **kwargs):
        # The one-sync path caches the original route on this object. The
        # synthetic route below must not consume that stale host list.
        self._routed_host = None
        width = inds.shape[-1]
        if _PICK["mode"] == "ram":
            fixed = _PICK["fixed"]
            miss = _PICK["miss"]
            if miss:
                # Route part of the layer outside the pinned pool. The ram and
                # disk arms are the two ends of one line and nothing sampled
                # between them, so a GPU cost that rises the moment the drive
                # is touched cannot be told from one that rises with bytes.
                # Everything else is held: same width, same dispatch count,
                # same code. Only the share of reads that reaches the drive
                # changes. Pool and cold indices come from disjoint ranges, so
                # the routed set stays distinct.
                cold = int(round(width * miss))
                hot = width - cold
                vals = []
                if hot:
                    picked = _PICK["rng"].choice(
                        len(fixed), size=min(hot, len(fixed)), replace=False
                    ).tolist()
                    vals.extend(fixed[index] for index in picked)
                if cold:
                    pool = len(fixed)
                    vals.extend(
                        (_PICK["rng"].choice(512 - pool, size=cold,
                                             replace=False) + pool).tolist()
                    )
                return _o(
                    self, x,
                    mx.broadcast_to(mx.array(vals, dtype=inds.dtype), inds.shape),
                    *args, **kwargs,
                )
            if len(fixed) > width and not _PICK["fixed_route"]:
                # A pool wider than the route separates two causes that the
                # plain ram arm confounds. Reusing eight experts every token
                # keeps the whole working set in cache, so "no drive" also
                # means "hot memory". Drawing eight of a pinned pool keeps the
                # drive at zero and makes the working set large again.
                vals = _PICK["rng"].choice(
                    len(fixed), size=width, replace=False
                ).tolist()
                vals = [fixed[i] for i in vals]
            else:
                vals = (fixed * (width // len(fixed) + 1))[:width]
        else:
            vals = _PICK["rng"].choice(512, size=width, replace=False).tolist()
        return _o(
            self, x,
            mx.broadcast_to(mx.array(vals, dtype=inds.dtype), inds.shape),
            *args, **kwargs,
        )

    cls.__call__ = call
    return len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ram", "disk"), required=True)
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--experts", type=int, default=8,
                        help="routed experts per layer")
    parser.add_argument("--miss", type=float, default=0.0,
                        help="ram mode: share of each layer's route drawn from "
                             "outside the pinned pool, so it reaches the drive. "
                             "0 is the all-resident arm, 1 is all-cold. Samples "
                             "the line between them")
    parser.add_argument("--route-fixed", action="store_true",
                        help="pin the pool but route the same experts every "
                             "token. Pairs with --pool to hold pinned bytes "
                             "and memory pressure constant while the touched "
                             "working set changes, which is the only clean "
                             "A/B for locality at zero drive")
    parser.add_argument("--pool", type=int, default=0,
                        help="ram mode: pin this many experts per layer and "
                             "draw the route from them; 0 pins exactly the "
                             "routed set, which keeps the working set tiny")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")
    model, _, _ = load_streaming(
        MODEL, expert_capacity=0, verbose=False, keep_vision=False, use_mtp=False
    )
    language = model.language_model
    store = language.model.layers[0].mlp.switch_mlp.gate_proj.cache.store
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    set_threshold(0.85)

    fixed = list(range(max(args.pool, args.experts)))
    pinned = 0
    if args.mode == "ram":
        for layer in language.model.layers:
            block = getattr(layer.mlp, "switch_mlp", None)
            if block is None:
                continue
            prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
            for projection in PROJECTIONS:
                for part in PARTS:
                    pinned += store.pin_rows(f"{prefix}.{projection}.{part}", fixed)
    _PICK["miss"] = args.miss
    _PICK["fixed_route"] = args.route_fixed
    _PICK["mode"] = args.mode
    _PICK["fixed"] = fixed
    _PICK["rng"] = np.random.default_rng(args.seed)
    patched = patch(language)
    print(f"patched layers: {patched}", flush=True)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="np")["input_ids"]
    cache = None
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(language)
    token = mx.array(ids)
    logits = language(token, cache=cache).logits
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    from models.flashnext.diskio import disk_bytes_read, vm_counters, vm_delta

    vm_before = vm_counters()
    before = disk_bytes_read()
    reset_profile()
    # The buffer ring reuses destinations the GPU may still be reading, so the
    # token digest is the acceptance test, not a nicety.
    produced = []
    began = time.perf_counter()
    for _ in range(args.tokens):
        logits = language(token[None], cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        produced.append(int(token.item()))
    elapsed = time.perf_counter() - began
    physical = disk_bytes_read() - before
    vm_after = vm_counters()
    from models.flashnext.diskio import free_memory_mb

    free = free_memory_mb()
    store.unpin_all()

    from models.flashnext.expert_cache import buffer_arena

    digest = hashlib.sha256(
        ",".join(str(value) for value in produced).encode()
    ).hexdigest()[:16]
    print(f"arena: {buffer_arena()}", flush=True)
    print(f"digest: {digest}", flush=True)
    print(f"free memory: {free} MB", flush=True)
    print(f"mode: {args.mode}", flush=True)
    touched = args.experts if args.route_fixed else len(fixed)
    print(f"pool: {len(fixed)} pinned/layer, route {args.experts}, "
          f"touched {touched}/layer", flush=True)
    print(f"pinned: {pinned / 1e9:.2f} GB", flush=True)
    print(f"rate: {args.tokens / elapsed:.2f} tok/s", flush=True)
    print(f"ms/token: {elapsed / args.tokens * 1000:.1f}", flush=True)
    print(f"miss: {args.miss}", flush=True)
    print(f"MB/token: {physical / args.tokens / 1e6:.1f}", flush=True)
    print(f"rdahead: {int(store._rdahead)}", flush=True)
    print(vm_delta(vm_before, vm_after, args.tokens), flush=True)
    totals = profile_totals()
    if totals.get("io_calls"):
        for key in sorted(totals):
            if key == "io_calls":
                print(f"prof {key}: {totals[key]}", flush=True)
            else:
                print(f"prof {key}: {totals[key] / args.tokens * 1000:.2f} ms/token", flush=True)


if __name__ == "__main__":
    main()
