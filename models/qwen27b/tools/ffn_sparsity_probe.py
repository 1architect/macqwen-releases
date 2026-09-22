#!/usr/bin/env python3
"""ffn_sparsity_probe.py - can the dense FFN be read as a MoE?

A SwiGLU FFN is an exact sum of `intermediate_size` rank-1 terms:

    y = sum_i  down[:, i] * ( silu(gate[i]@x) * up[i]@x )

So cutting the FFN into contiguous neuron blocks is exact. The only loss comes
from *skipping* blocks. This measures that loss on real tokens: for each block
count N and keep fraction, it reports the relative error of the layer output.

The result decides the whole streaming design. Bytes read per token are
proportional to the keep fraction.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL = os.environ.get(
    "MODEL", str(Path.home() / "models/Qwen3.8-27B-Apple-MLX-V3.1-Compact")
)
NBLOCKS = [32, 64, 128]
KEEP    = [0.50, 0.25, 0.125, 0.0625, 0.03125]
CAP = {}          # layer index -> summary rows
HID = {}          # layer index -> block scores (T, N) for the largest N
ENGINE_SOURCE = Path(__file__).with_name("frankenstein_engine.py")


def flush(*a):
    print(*a, flush=True)


class Probe(nn.Module):
    def __init__(self, inner, idx):
        super().__init__()
        self.inner = inner
        self.idx = idx

    def __call__(self, x):
        h = nn.silu(self.inner.gate_proj(x)) * self.inner.up_proj(x)
        if self.idx in CAP:
            analyse(self.idx, h, self.inner.down_proj)
        return self.inner.down_proj(h)


def analyse(idx, h, down):
    h = h.astype(mx.float32)
    if h.ndim == 3:
        h = h[0]
    T, I = h.shape
    y = down(h.astype(mx.bfloat16)).astype(mx.float32)
    ny = mx.sqrt(mx.sum(y * y, axis=-1)) + 1e-9

    rows = []
    for N in NBLOCKS:
        B = I // N
        hb = h.reshape(T, N, B)
        score = mx.sqrt(mx.sum(hb * hb, axis=-1))            # (T, N)
        order = mx.argsort(-score, axis=-1)
        if N == NBLOCKS[-1]:
            HID[idx] = np.array(order[:, : max(1, N // 8)])
        for f in KEEP:
            k = max(1, int(round(N * f)))
            keep = order[:, :k]
            mask = mx.zeros((T, N))
            mask[mx.arange(T)[:, None], keep] = 1.0
            hm = (hb * mask[:, :, None]).reshape(T, I)
            ym = down(hm.astype(mx.bfloat16)).astype(mx.float32)
            err = mx.sqrt(mx.sum((ym - y) ** 2, axis=-1)) / ny
            rows.append((N, f, k, float(mx.mean(err)), float(mx.max(err))))
    # neuron-level: fraction of neurons holding 99% of the energy
    e = mx.sort(h * h, axis=-1)[:, ::-1]
    c = mx.cumsum(e, axis=-1)
    frac = mx.mean(mx.sum((c < 0.99 * c[:, -1:]).astype(mx.float32), axis=-1)) / I
    CAP[idx] = {"rows": rows, "n99": float(frac)}
    flush(f"  layer {idx:>2} done  neurons for 99% energy: {float(frac)*100:.1f}%")


def main():
    t0 = time.time()
    flush(f"loading {MODEL}")
    model, tok = load(MODEL)
    layers = model.language_model.model.layers
    L = len(layers)
    probe_idx = [1, 7, 15, 23, 31, 39, 47, 55, 62]
    for i in probe_idx:
        CAP[i] = {}
        layers[i].mlp = Probe(layers[i].mlp, i)

    src = ENGINE_SOURCE.read_text()[:6000]
    text = ("Read this Python source and explain the memory model.\n\n"
            + src + "\n\nExplain the cache strategy in detail.")
    ids = tok.encode(text)[:512]
    flush(f"model loaded in {time.time()-t0:.0f}s, {L} layers, {len(ids)} tokens")

    x = mx.array([ids])
    _ = model(x)
    mx.eval(_)
    flush(f"forward done in {time.time()-t0:.0f}s")

    flush("\n=== relative output error of the FFN when blocks are skipped ===")
    flush(f"{'blocks':>7}{'keep':>8}{'active':>8}{'bytes/tok':>11}{'tok/s':>8}   mean err   max err   (per layer mean)")
    agg = {}
    for i, d in CAP.items():
        for (N, f, k, m, mx_) in d["rows"]:
            agg.setdefault((N, f, k), []).append(m)
    FFN_GB = 34.23
    SSD = 2.8
    for (N, f, k), v in sorted(agg.items()):
        b = FFN_GB * f
        flush(f"{N:>7}{k:>5}/{N:<2}{f*100:>7.2f}%{b:>9.2f}GB{SSD/b:>8.2f}   {np.mean(v):>8.4f}  "
              f"{np.max(v):>8.4f}")
    flush("\nneurons holding 99% of activation energy, by layer:")
    for i, d in sorted(CAP.items()):
        flush(f"  layer {i:>2}: {d['n99']*100:5.1f}%")

    flush("\n=== block reuse between consecutive tokens (cache locality) ===")
    for i, o in sorted(HID.items()):
        ov = [len(set(o[t]) & set(o[t-1])) / o.shape[1] for t in range(1, o.shape[0])]
        flush(f"  layer {i:>2}: top-{o.shape[1]} of {NBLOCKS[-1]} blocks, "
              f"consecutive-token overlap {np.mean(ov)*100:5.1f}%")
    flush(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
