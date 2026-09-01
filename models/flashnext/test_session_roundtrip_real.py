#!/usr/bin/env python3
"""One-token parity test for a saved FlashNext cache on the real model."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx

from models.flashnext.loader import load_streaming
from models.flashnext.sessions import SessionStore
from macqwen.checkpoints import resolve_flashnext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    model_path = resolve_flashnext(args.model)
    os.environ["FLASHNEXT_TOPK_THRESHOLD"] = "0.85"

    from transformers import AutoTokenizer

    started = time.time()
    model, _, _ = load_streaming(
        str(model_path), verbose=False, keep_vision=False, use_mtp=False
    )
    model_seconds = time.time() - started
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    language = model.language_model
    cache = language.make_cache()

    prompt_ids = tokenizer(
        "session cache probe", add_special_tokens=False
    )["input_ids"]
    first = language(mx.array(prompt_ids)[None], cache=cache)
    mx.eval(first.logits)

    # Cross QSA's 2,048-token budget without paying for a full long prefill.
    # The recurrent state is synthetic, but the sparse QSA continuation is real.
    cached_tokens = 2052
    for entry in cache:
        if type(entry).__name__ != "QSAKVCache":
            continue
        source = int(entry.offset)
        repeats = math.ceil(cached_tokens / source)
        entry.keys = mx.tile(
            entry.keys[..., :source, :], (1, 1, repeats, 1)
        )[..., :cached_tokens, :]
        entry.values = mx.tile(
            entry.values[..., :source, :], (1, 1, repeats, 1)
        )[..., :cached_tokens, :]
        entry.index_keys = mx.tile(
            entry.index_keys[:, :source], (1, repeats, 1)
        )[:, :cached_tokens]
        entry.index_position_ids = mx.arange(
            cached_tokens, dtype=mx.int32
        )[None]
        entry.offset = cached_tokens
    context_ids = (prompt_ids * math.ceil(cached_tokens / len(prompt_ids)))[
        :cached_tokens
    ]

    profile = {
        "mode": "standard",
        "threshold": 0.85,
        "think": False,
        "stop_ids": [],
        "prompt_protocol": {"test": "synthetic-sparse-qsa-v1"},
        "renorm": 1.0,
        "tail_warmup": None,
        "tail_experts": None,
        "resident_experts": None,
        "mtp_depth": 0,
    }
    with tempfile.TemporaryDirectory(prefix="flashnext-real-session-") as directory:
        sessions = SessionStore(directory, model_path, profile, language)
        operation_started = time.time()
        summary = sessions.save("probe", cache, context_ids, False)
        save_seconds = time.time() - operation_started

        suffix_ids = tokenizer(" next", add_special_tokens=False)["input_ids"][:1]
        qsa_before = [
            (
                mx.contiguous(entry.index_keys),
                mx.contiguous(entry.index_position_ids),
            )
            for entry in cache
            if type(entry).__name__ == "QSAKVCache"
        ]

        sparse_probe = sessions.load("probe")
        fa_index = language.model.fa_idx
        attention = language.model.layers[fa_index].self_attn
        hidden = language.model.embed_tokens(mx.array(suffix_ids)[None])
        sparse_mask = attention.indexer(
            hidden,
            sparse_probe.cache[fa_index],
            mx.array([[cached_tokens]], dtype=mx.int32),
        )
        if sparse_mask is None:
            raise SystemExit("sparse QSA path did not activate")
        mx.eval(sparse_mask)

        baseline = language(mx.array(suffix_ids)[None], cache=cache).logits[:, -1, :]
        mx.eval(baseline)
        expected = int(mx.argmax(baseline, axis=-1).item())

        operation_started = time.time()
        loaded = sessions.load("probe")
        load_seconds = time.time() - operation_started
        qsa_after = [
            (entry.index_keys, entry.index_position_ids)
            for entry in loaded.cache
            if type(entry).__name__ == "QSAKVCache"
        ]
        auxiliary_diff = 0.0
        for before, after in zip(qsa_before, qsa_after):
            auxiliary_diff = max(
                auxiliary_diff,
                float(mx.max(mx.abs(before[0] - after[0])).item()),
                float(mx.max(mx.abs(before[1] - after[1])).item()),
            )
        language._position_ids = loaded.position_ids
        language._rope_deltas = loaded.rope_deltas
        restored = language(
            mx.array(suffix_ids)[None], cache=loaded.cache
        ).logits[:, -1, :]
        mx.eval(restored)
        actual = int(mx.argmax(restored, axis=-1).item())
        max_diff = float(mx.max(mx.abs(baseline - restored)).item())

        arrays = sum(type(entry).__name__ == "ArraysCache" for entry in loaded.cache)
        qsa = sum(type(entry).__name__ == "QSAKVCache" for entry in loaded.cache)
        print(
            f"tokens={cached_tokens} size={summary.size_bytes / 1e6:.1f}MB "
            f"arrays={arrays} qsa={qsa} sparse={int(sparse_mask.sum().item())} "
            f"token={actual} max_diff={max_diff:.9g} aux_diff={auxiliary_diff:.9g} "
            f"model={model_seconds:.2f}s save={save_seconds:.2f}s "
            f"load={load_seconds:.2f}s elapsed={time.time() - started:.1f}s"
        )
        if arrays != 36 or qsa != 12:
            raise SystemExit("unexpected cache layout")
        if actual != expected or max_diff != 0.0 or auxiliary_diff != 0.0:
            raise SystemExit("restored cache changed the next-token logits")


if __name__ == "__main__":
    main()
