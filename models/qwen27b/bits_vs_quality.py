#!/usr/bin/env python3
"""bits_vs_quality.py - does spending the spare RAM actually help?

The plan is to raise the body from 2.71 bits/param to about 3.95, using RAM
that currently sits empty. That is an expectation, not a measurement. This
measures it.

Layers already in the near-lossless intermediate store get requantised at a
range of widths and dropped into the live model. Everything else stays as it
ships. So the difference in loss is caused only by the extra bits.

Held-out text only: none of these files were used for calibration.
"""
import glob, json, time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

REF = str(Path.home() / "models/Qwen3.8-27B-Apple-MLX-V3.1-Compact")
STORE = Path.home() / ".frankenstein" / "v4" / "q8"
# Do not delete the files named here, and do not edit them. They are the
# held-out corpus. Perplexity is comparable across builds only while the
# corpus stays byte-identical, so bench_decode.py and profile_prefill.py
# stay in this repo even though nothing imports them.
HELD_OUT = [
    "macqwen/tools/repo_token_cache.py",
    "models/qwen27b/speculative_prefill.py",
    "models/qwen27b/bench_decode.py",
]


def flush(*a):
    print(*a, flush=True)


def load_store():
    """layer index -> {module path: bf16 weight}"""
    out = {}
    for f in sorted(STORE.glob("*.safetensors")):
        raw = mx.load(str(f))
        for k in raw:
            if not k.endswith(".weight.q"):
                continue
            base = k[: -len(".q")]
            body = base[len("model.language_model.layers."):]
            li = int(body.split(".")[0])
            path = body[len(str(li)) + 1: -len(".weight")]
            W = mx.dequantize(raw[base + ".q"], raw[base + ".s"], raw[base + ".b"],
                              group_size=64, bits=8)
            out.setdefault(li, {})[path] = W
        del raw
    return out


def resolve(layer, path):
    node = layer
    parts = path.split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    return node, parts[-1]


def qbytes(mod):
    n = mod.scales.size * mod.group_size
    return n * mod.bits / 8 + mod.scales.size * 4


def nll(model, tok, texts, cap=512):
    tot, cnt = 0.0, 0
    for t in texts:
        ids = tok.encode(t)[:cap]
        if len(ids) < 64:
            continue
        x = mx.array([ids])
        lg = model(x)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        tot += float(mx.sum(-lp[:-1][mx.arange(len(ids) - 1), tgt]))
        cnt += len(ids) - 1
        mx.eval(lg)
        del lg, lp
        mx.clear_cache()
    return tot / cnt


def main():
    t0 = time.time()
    flush("loading intermediate store")
    store = load_store()
    layers_avail = sorted(store)
    flush(f"{len(layers_avail)} layers available: {layers_avail}")

    model, tok = load(REF)
    root = Path(__file__).resolve().parents[2]
    texts = [(root / f).read_text()[:8000] for f in HELD_OUT]
    L = model.language_model.model.layers

    orig = {}
    base_bytes = 0
    for li in layers_avail:
        for path in store[li]:
            node, attr = resolve(L[li], path)
            mod = getattr(node, attr)
            orig[(li, path)] = mod
            base_bytes += qbytes(mod)

    b = nll(model, tok, texts)
    flush(f"\nbaseline (ships as-is): NLL {b:.4f}  ppl {np.exp(b):.3f}   "
          f"those layers hold {base_bytes/1e9:.2f} GB "
          f"= {base_bytes*8/sum(m.scales.size*m.group_size for m in orig.values()):.2f} bits/param")
    flush(f"loaded in {time.time()-t0:.0f}s\n")

    flush(f"{'width':>8}{'GB (those layers)':>19}{'delta GB':>10}"
          f"{'NLL':>9}{'ppl':>9}{'ppl gain':>10}")
    flush(f"{'ships':>8}{base_bytes/1e9:>19.2f}{0.0:>10.2f}{b:>9.4f}"
          f"{np.exp(b):>9.3f}{0.0:>10.2%}")
    for bits, group in ((4, 64), (5, 64), (6, 64), (8, 64)):
        nb = 0
        for li in layers_avail:
            for path, W in store[li].items():
                node, attr = resolve(L[li], path)
                out_f, in_f = W.shape
                ql = nn.QuantizedLinear(in_f, out_f, bias=False,
                                        group_size=group, bits=bits)
                w, s, bi = mx.quantize(W, group_size=group, bits=bits)
                ql.weight, ql.scales, ql.biases = w, s, bi
                setattr(node, attr, ql)
                nb += qbytes(ql)
        mx.eval(model.parameters())
        v = nll(model, tok, texts)
        flush(f"{bits:>7}b{nb/1e9:>19.2f}{(nb-base_bytes)/1e9:>10.2f}{v:>9.4f}"
              f"{np.exp(v):>9.3f}{(np.exp(b)-np.exp(v))/np.exp(b):>10.2%}")
        for (li, path), mod in orig.items():
            node, attr = resolve(L[li], path)
            setattr(node, attr, mod)
        mx.clear_cache()

    frac = len(layers_avail) / len(L)
    flush(f"\nthese are {len(layers_avail)} of {len(L)} layers ({frac:.0%} of the body).")
    flush(f"scale the delta GB by {1/frac:.1f} for the whole model.")
    flush(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
