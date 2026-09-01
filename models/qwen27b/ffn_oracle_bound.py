#!/usr/bin/env python3
"""ffn_oracle_bound.py - the ceiling on any FFN sparsification.

Block selection can never beat picking the best neurons directly. So mask each
FFN to its top-k neurons by |activation| in every layer, with perfect hindsight,
and measure the damage on real text. This is the best case for any router,
any permutation, any bundling scheme. If the oracle already breaks the model at
a keep fraction, no MoE-ification of this dense model can reach it.
"""
import os, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL = os.environ.get("MODEL", "/Users/gioma/.lmstudio/models/gioma/"
                                "Qwen3.8-27B-Apple-MLX-V3.1-Compact")
KEEPS = [1.0, 0.50, 0.25, 0.125, 0.0625]
STATE = {"keep": 1.0}
ENGINE_SOURCE = Path(__file__).with_name("frankenstein_engine.py")


def flush(*a): print(*a, flush=True)


class Oracle(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        h = nn.silu(self.inner.gate_proj(x)) * self.inner.up_proj(x)
        f = STATE["keep"]
        if f < 1.0:
            I = h.shape[-1]
            k = max(1, int(I * f))
            a = mx.abs(h)
            thr = mx.sort(a, axis=-1)[..., -k:-k + 1]
            h = mx.where(a >= thr, h, mx.zeros_like(h))
        return self.inner.down_proj(h)


def main():
    t0 = time.time()
    model, tok = load(MODEL)
    layers = model.language_model.model.layers
    for l in layers:
        l.mlp = Oracle(l.mlp)

    src = ENGINE_SOURCE.read_text()
    text = ("You are a careful engineer. Read this file and describe the design.\n\n"
            + src[:9000])
    ids = tok.encode(text)[:768]
    x = mx.array([ids])
    flush(f"loaded in {time.time()-t0:.0f}s, {len(ids)} tokens, {len(layers)} layers\n")

    base_lp = None
    base_top = None
    flush(f"{'keep':>7}{'GB/tok bf16':>13}{'tok/s @2.8GB/s':>16}"
          f"{'NLL':>9}{'ppl':>9}{'top1 match':>12}{'KL':>9}")
    for f in KEEPS:
        STATE["keep"] = f
        logits = model(x)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        nll = float(-mx.mean(lp[:-1][mx.arange(len(ids) - 1), tgt]))
        top = mx.argmax(logits, axis=-1)
        if base_lp is None:
            base_lp, base_top = lp, top
            match, kl = 1.0, 0.0
        else:
            match = float(mx.mean((top == base_top).astype(mx.float32)))
            kl = float(mx.mean(mx.sum(mx.exp(base_lp) * (base_lp - lp), axis=-1)))
        gbt = 34.23 * f
        flush(f"{f*100:>6.2f}%{gbt:>13.2f}{2.8/gbt:>16.2f}"
              f"{nll:>9.4f}{np.exp(nll):>9.2f}{match*100:>11.1f}%{kl:>9.4f}")
        mx.clear_cache()
    flush(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
