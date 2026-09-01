#!/usr/bin/env python3
"""Split a decode token into physical disk reads and everything else.

The complete chat cannot show where a token spends its time. This probe runs
the same greedy reply several times in one process and reports time and
physical disk bytes per token. Identical token IDs are asserted, so a changed
runtime cannot pass by producing different work.

macOS note: `getrusage` `ru_inblock` returns 0 on Darwin. Physical bytes come
from `proc_pid_rusage` with `RUSAGE_INFO_V4` instead.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx

from models.flashnext.expert_cache import profile_totals, reset_profile
from models.flashnext.loader import load_streaming


DEFAULT_PROMPT = "Explique a fotossintese em duas frases."
LAYER_TIMERS: dict[str, float] = {}


FORCE_EVAL = [False]


def wrap_layer_classes(language) -> None:
    """Time the non-MoE layer families on the loaded model.

    Wrap after loading. `qsa_chunk` and the loader replace `__call__` on these
    classes, so wrapping the imported class first is silently overwritten.
    """
    layers = language.model.layers
    targets = {}
    for layer in layers:
        targets.setdefault("decoder_layer", type(layer))
        if "ple" in layer:
            targets.setdefault("ple", type(layer.ple))
        if getattr(layer, "is_linear", False):
            targets.setdefault("gated_delta", type(layer.linear_attn))
        else:
            targets.setdefault("qsa_attention", type(layer.self_attn))
            targets.setdefault("moe_block", type(layer.mlp))
        targets.setdefault("hyper_connection", type(layer.attn_hyper_connection))
    for label, cls in targets.items():
        LAYER_TIMERS[label] = 0.0
        original = cls.__call__

        def timed(self, *args, _label=label, _original=original, **kwargs):
            began = time.perf_counter()
            try:
                result = _original(self, *args, **kwargs)
                if FORCE_EVAL[0] and isinstance(result, mx.array):
                    mx.eval(result)
                return result
            finally:
                LAYER_TIMERS[_label] += time.perf_counter() - began

        cls.__call__ = timed
# Established gather rate for this checkpoint and drive, in MB/s. Attributing
# time to disk needs a reference measured elsewhere. Dividing the observed
# bytes by the observed rate is circular and always returns 100%.
REFERENCE_GATHER_MBS = 1070.0
_RUSAGE_INFO_V4 = 4
_BYTESREAD_OFFSET = 16 + 16 * 8  # after ri_uuid[16] and 16 uint64 fields
_LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.dylib")
_RUSAGE_BUFFER = (ctypes.c_uint8 * 512)()


def disk_bytes_read() -> int:
    """Return physical bytes this process has read, or -1 when unavailable."""
    if _LIBSYSTEM.proc_pid_rusage(
        os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(_RUSAGE_BUFFER)
    ) != 0:
        return -1
    raw = bytes(_RUSAGE_BUFFER[_BYTESREAD_OFFSET : _BYTESREAD_OFFSET + 8])
    return int.from_bytes(raw, "little")


def decode_pass(language, ids, count):
    language._position_ids = None
    language._rope_deltas = None
    cache = language.make_cache()
    output = language(ids, cache=cache)
    token = mx.argmax(output.logits[:, -1, :], axis=-1)
    mx.eval(token)
    output = None
    mx.clear_cache()

    produced = []
    for key in LAYER_TIMERS:
        LAYER_TIMERS[key] = 0.0
    reset_profile()
    read_before = disk_bytes_read()
    began = time.time()
    drain = 0.0
    for _ in range(count):
        produced.append(int(token.item()))
        step = language(token[None], cache=cache)
        token = mx.argmax(step.logits[:, -1, :], axis=-1)
        drain_began = time.perf_counter()
        mx.eval(token)
        drain += time.perf_counter() - drain_began
    elapsed = time.time() - began
    read_bytes = disk_bytes_read() - read_before
    memory = (mx.get_active_memory(), mx.get_cache_memory())
    timers = profile_totals()
    timers["final_eval"] = drain
    return produced, elapsed / count, read_bytes / count, memory, timers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--profile-layers", action="store_true")
    parser.add_argument(
        "--eval-layers",
        action="store_true",
        help="force a sync inside each wrapped layer to attribute GPU time; "
        "this perturbs the total but splits the score-sync drain",
    )
    parser.add_argument(
        "--gather-rate",
        type=float,
        default=REFERENCE_GATHER_MBS,
        help="reference gather rate in MB/s used to attribute time to disk",
    )
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")
    from macqwen.checkpoints import resolve_flashnext

    path = str(resolve_flashnext(args.model))


    from transformers import AutoTokenizer

    model, _, _ = load_streaming(
        path, expert_capacity=0, verbose=False, keep_vision=False, use_mtp=False
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    language = model.language_model
    if args.profile_layers:
        wrap_layer_classes(language)
    FORCE_EVAL[0] = args.eval_layers
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = mx.array(tokenizer(text)["input_ids"])[None]

    reference = None
    results = []
    for index in range(1, args.passes + 1):
        produced, seconds, read_bytes, memory, timers = decode_pass(
            language, ids, args.tokens
        )
        layers = {k: v / args.tokens * 1000 for k, v in LAYER_TIMERS.items()}
        if reference is None:
            reference = produced
        elif produced != reference:
            raise SystemExit("token IDs changed between passes; result invalid")
        results.append((seconds, read_bytes, timers))
        print(
            f"pass {index}  {1 / seconds:5.2f} tok/s  {seconds * 1000:7.1f} "
            f"ms/token  physical read {read_bytes / 1e6:8.1f} MB/token  "
            f"mlx active {memory[0] / 1e9:5.2f} GB  cache {memory[1] / 1e9:5.2f} GB",
            flush=True,
        )
        if timers["io_calls"]:
            io_ms = timers["io_wait"] / args.tokens * 1000
            sync_ms = timers["router_sync"] / args.tokens * 1000
            per = lambda key: timers[key] / args.tokens * 1000
            ngram_ms, tomx_ms = per("ngram_wait"), per("to_mx")
            issue_ms, eval_ms = per("moe_issue"), per("final_eval")
            score_ms, topk_ms = per("score_sync"), per("topk_python")
            shared_ms = per("shared_expert")
            other_ms = (
                seconds * 1000
                - io_ms - sync_ms - ngram_ms - tomx_ms - issue_ms - eval_ms
                - score_ms - topk_ms - shared_ms
            )
            print(
                f"          expert reads {io_ms:6.1f}  ngram {ngram_ms:5.1f}  "
                f"to_mx {tomx_ms:6.1f}  moe issue {issue_ms:6.1f}  "
                f"router sync {sync_ms:5.1f}  final eval {eval_ms:6.1f}  "
                f"rest {other_ms:6.1f} ms  |  read rate "
                f"{read_bytes / 1e6 / (timers['io_wait'] / args.tokens):5.0f} MB/s",
                flush=True,
            )
            # only meaningful with FLASHNEXT_PROFILE_IO=1, and only defined
            # inside the block above that computes them
            print(
                f"          score sync {score_ms:6.1f}  "
                f"topk python {topk_ms:5.1f}  "
                f"shared expert {shared_ms:5.1f} ms",
                flush=True,
            )
        if layers:
            inside = ", ".join(f"{k} {v:.1f} ms" for k, v in layers.items())
            print(f"          inside rest: {inside}", flush=True)

    best_seconds, best_bytes, best_timers = min(results)
    observed = best_bytes / best_seconds / 1e6
    disk_seconds = best_bytes / 1e6 / args.gather_rate
    print(f"\nidentical token IDs across {args.passes} passes: yes")
    print(f"observed gather rate   {observed:7.0f} MB/s")
    print(f"reference gather rate  {args.gather_rate:7.0f} MB/s")
    print(
        f"physical reads explain {disk_seconds / best_seconds:.0%} "
        f"of a decode token; {(best_seconds - disk_seconds) * 1000:.0f} ms "
        "is left for compute, dispatch, and host syncs"
    )


if __name__ == "__main__":
    main()
