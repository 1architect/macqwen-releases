#!/usr/bin/env python3
"""bench_decode.py - decode speed versus context length.

Measures prefill and decode speed at a ladder of context lengths, using the
ContextVM V0 engine. One process per KV configuration.

    python3 bench_decode.py --kv-bits 4
    python3 bench_decode.py                 # fp16 KV, fused attention kernel
"""

import argparse, re, subprocess, sys, time
from pathlib import Path

import mlx.core as mx

from models.qwen27b.frankenstein_engine import (E2, MACBAT, FrankensteinEngine, Printer,
                                 host_mem, patch_lm_head_last_token)

FILLER_SYSTEM = "You are a code analyst. Answer briefly."
FIRST = "Read the Swift source that follows in later messages. Reply with 'ready'."


def page_stats():
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        def n(label):
            m = re.search(rf"{label}:\s+(\d+)", vm)
            return int(m.group(1)) if m else 0
        return n("Pageins"), n("Pageouts"), n(r"Swapins"), n(r"Swapouts")
    except Exception:
        return (0, 0, 0, 0)


def gather_source(root, want_chars):
    """Concatenate real Swift source until the character budget is met."""
    out, total = [], 0
    for p in sorted(Path(root).rglob("*.swift")):
        if any(x in p.parts for x in (".build", "DerivedData", ".git")):
            continue
        try:
            t = p.read_text(errors="replace")
        except Exception:
            continue
        out.append(f"// FILE: {p.name}\n{t}")
        total += len(t)
        if total >= want_chars:
            break
    text = "\n\n".join(out)
    while len(text) < want_chars:
        text += text
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=E2)
    p.add_argument("--root", default=MACBAT)
    p.add_argument("--rungs", default="1024,2048,4096,8192,16384",
                   help="target logical context sizes")
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--prefill-step-size", type=int, default=1024)
    p.add_argument("--kv-bits", type=int, default=None)
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=1024)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--mlx-cache-limit-gb", type=float, default=None,
                   help="cap the MLX buffer cache so freed memory returns to the OS")
    p.add_argument("--lm-head-last", action="store_true")
    p.add_argument("--log", default=None)
    a = p.parse_args()
    tag = f"kv{a.kv_bits or 'fp16'}_p{a.prefill_step_size}_c{a.mlx_cache_limit_gb}"
    out = Printer(a.log or f"bench_decode_{tag}_{int(time.time())}.log")

    if a.lm_head_last:
        patch_lm_head_last_token()
    if a.mlx_cache_limit_gb is not None:
        mx.set_cache_limit(int(a.mlx_cache_limit_gb * 1e9))
    free, swap = host_mem()
    out(f"CONFIG: kv_bits={a.kv_bits} start={a.quantized_kv_start} "
        f"prefill={a.prefill_step_size} rep_penalty={a.repetition_penalty} "
        f"mlx_cache_limit={a.mlx_cache_limit_gb}")
    out(f"HOST  : free {free:.2f} GB swap {swap:.2f} GB")
    t0 = time.perf_counter()
    eng = FrankensteinEngine(a.model, prefill_step_size=a.prefill_step_size,
                             kv_bits=a.kv_bits, kv_group_size=a.kv_group_size,
                             quantized_kv_start=a.quantized_kv_start,
                             repetition_penalty=a.repetition_penalty,
                             loop_guard=False)
    out(f"LOAD  : {time.perf_counter()-t0:.1f}s  mlx active {mx.get_active_memory()/1e9:.2f} GB")

    source = gather_source(a.root, 400_000)
    eng.open_conversation(FILLER_SYSTEM, FIRST, reasoning_effort="low")
    eng.generate(max_tokens=16, echo=False)

    rungs = [int(x) for x in a.rungs.split(",")]
    rows = []
    cursor = 0
    for target in rungs:
        need = target - len(eng.tape)
        if need > 0:
            # about 3.6 characters per token for Swift source
            chunk = source[cursor:cursor + int(need * 3.6)]
            cursor += len(chunk)
            eng.append_user(chunk + "\n\nSummarize the code above in exactly 100 words.")
        pi0, po0, si0, so0 = page_stats()
        _, st = eng.generate(max_tokens=a.decode_tokens, echo=False)
        pi1, po1, si1, so1 = page_stats()
        rows.append((st, pi1 - pi0, so1 - so0))
        out(f"ctx {st.logical_tokens:>6} | prefill {st.new_prompt_tokens:>6} tok @ "
            f"{st.prompt_tps:>6.1f} t/s | decode {st.gen_tokens:>3} @ {st.gen_tps:>5.2f} t/s"
            f" | kv {st.cache_gb:>5.2f} GB (attn {st.attn_cache_gb:>5.2f})"
            f" | mlx act {st.active_gb:>5.2f} pool {st.pool_gb:>4.2f} peak {st.peak_gb:>5.2f}"
            f" | free {st.host_free_gb:>5.2f} swap {st.swap_gb:>5.2f}"
            f" | pagein {pi1-pi0:>6} swapout {so1-so0:>5}")
        if not eng.check_invariant():
            out("!! invariant broken")
            break
        if st.host_free_gb < 0.3:
            out("!! host memory low, stopping ladder")
            break

    out(f"\n{'='*96}\nSUMMARY {tag}\n{'='*96}")
    out(f"{'ctx':>7} {'decode t/s':>11} {'prefill t/s':>12} {'attn kv GB':>11} {'pagein':>8}")
    for st, pi, so in rows:
        out(f"{st.logical_tokens:>7} {st.gen_tps:>11.2f} {st.prompt_tps:>12.1f} "
            f"{st.attn_cache_gb:>11.3f} {pi:>8}")
    if rows:
        first, last = rows[0][0], rows[-1][0]
        out(f"\ndecode {first.gen_tps:.2f} t/s @ {first.logical_tokens} tok"
            f"  ->  {last.gen_tps:.2f} t/s @ {last.logical_tokens} tok"
            f"  ({last.gen_tps/max(first.gen_tps,1e-9)*100:.0f}% retained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
