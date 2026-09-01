#!/usr/bin/env python3
"""sensitivity.py - calibrate the bit allocator against real loss.

The allocator minimises a proxy: the squared error a tensor injects into its
own output, weighted by activation power. That proxy is blind to what happens
downstream. Two tensors can distort their own output equally and cost the model
very different amounts, because one feeds a softmax and the other sums into a
residual stream that washes it out.

Observed consequence: the MLP was allocated about 2.5 bits, and the model
started inventing library method names. Factual recall lives in the MLP, and
the proxy never measured recall.

So measure it. For each tensor family at each depth, quantise only that group
harder and read the loss change on held-out text. The ratio of measured loss to
predicted distortion is the correction the allocator was missing.

    factor(group) = delta_NLL_measured / delta_distortion_predicted

Groups the model actually depends on get a factor above one and win bits back.
"""
import json, time, glob
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL = Path.home() / "models/Qwen3.8-27B-Apple-MLX-V4-ends-b"
STORE = Path.home() / ".frankenstein" / "v4" / "q8"
WORK = Path.home() / ".frankenstein" / "v4"
# Perturbing "down to 2 bits" is a no-op for the tensors already at 2 bits,
# which is most of the MLP. So inject calibrated noise instead and requantise at
# each tensor's CURRENT width: the perturbation is uniform across groups, it is
# defined for every tensor, and resident memory does not move.
NOISE = 0.05            # relative Frobenius magnitude of the injected error
BUCKETS = 4
HELD_OUT = [
    "macqwen/tools/repo_token_cache.py",
    "models/qwen27b/speculative_prefill.py",
]


def flush(*a):
    print(*a, flush=True)


def resolve(layer, path):
    node = layer
    parts = path.split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    return node, parts[-1]


def main():
    t0 = time.time()
    flush("indexing the intermediate store")
    index = {}                      # (layer, path) -> (file, base, quantised?)
    for f in sorted(STORE.glob("*.safetensors")):
        raw = mx.load(str(f))
        for k in raw:
            if k.endswith(".weight.q"):
                base, quant = k[:-2], True
            elif k.endswith(".weight") and raw[k].ndim == 2:
                base, quant = k, False
            else:
                continue
            body = base[len("model.language_model.layers."):]
            if not body[0].isdigit():
                continue
            li = int(body.split(".")[0])
            path = body[len(str(li)) + 1: -len(".weight")]
            index[(li, path)] = (str(f), base, quant)
        del raw
        mx.clear_cache()
    flush(f"{len(index)} tensors indexed")

    groups = {}
    for (li, path) in index:
        fam = path.split(".")[-1]
        groups.setdefault((fam, li * BUCKETS // 64), []).append((li, path))

    model, tok = load(str(MODEL))
    L = model.language_model.model.layers
    root = Path(__file__).resolve().parents[2]
    texts = [(root / f).read_text()[:7000] for f in HELD_OUT if (root / f).exists()]

    def nll():
        tot = cnt = 0.0
        for t in texts:
            ids = tok.encode(t)[:384]
            lg = model(mx.array([ids]))[0].astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            tot += float(mx.sum(-lp[:-1][mx.arange(len(ids) - 1), mx.array(ids[1:])]))
            cnt += len(ids) - 1
            del lg, lp
            mx.clear_cache()
        return tot / cnt

    plan = json.loads((WORK / "plan.json").read_text())
    calib = {}
    cpath = Path.home() / ".frankenstein" / "calibration.npz"
    if cpath.exists():
        for k, v in np.load(cpath).items():
            kk = k
            for pre in ("language_model.model.", "model.language_model."):
                if kk.startswith(pre):
                    kk = kk[len(pre):]
            calib[kk] = v

    base = nll()
    flush(f"\nbaseline NLL {base:.4f}   ({time.time()-t0:.0f}s)\n")
    flush(f"{'family':<16}{'depth':>7}{'n':>4}{'dNLL':>9}{'predicted':>12}{'factor':>9}")

    out = {}
    for g in sorted(groups, key=lambda x: (x[0], x[1])):
        fam, b = g
        saved = {}
        p = 0.0
        for (li, path) in groups[g]:
            node, attr = resolve(L[li], path)
            old = getattr(node, attr)
            bits = getattr(old, "bits", 4)
            gs = getattr(old, "group_size", 64)
            # Perturb the model's OWN weights. Noise needs no pristine source,
            # so this touches no files: reloading a 1.8 GB shard per tensor was
            # costing minutes per group and measuring nothing extra.
            W = mx.dequantize(old.weight, old.scales, old.biases,
                              group_size=gs, bits=bits).astype(mx.float32)
            scale = NOISE * float(mx.sqrt(mx.mean(W * W)))
            Wp = W + mx.random.normal(W.shape) * scale
            a = calib.get(f"layers.{li}.{path}")
            av = mx.array(np.asarray(a, dtype=np.float32)) if a is not None \
                else mx.ones((W.shape[1],))
            if av.shape[0] != W.shape[1]:
                av = mx.ones((W.shape[1],))
            E = (W - Wp) * av
            p += float(mx.mean(mx.sum(E * E, axis=-1)))
            saved[(li, path)] = old
            o, i = W.shape
            ql = nn.QuantizedLinear(i, o, bias=False, group_size=gs, bits=bits)
            q, sc, bi = mx.quantize(Wp.astype(mx.bfloat16), group_size=gs, bits=bits)
            ql.weight, ql.scales, ql.biases = q, sc, bi
            setattr(node, attr, ql)
            del W, Wp, E
            mx.clear_cache()
        mx.eval(model.parameters())
        d = nll() - base
        for (li, path), mod in saved.items():
            node, attr = resolve(L[li], path)
            setattr(node, attr, mod)
        mx.clear_cache()
        factor = d / p if p > 1e-12 else 1.0
        out[f"{fam}|{b}"] = {"dnll": d, "predicted": p, "factor": factor}
        flush(f"{fam:<16}{b:>7}{len(groups[g]):>4}{d:>9.4f}{p:>12.4g}{factor:>9.4g}")

    (WORK / "sensitivity.json").write_text(json.dumps(out, indent=1))
    flush(f"\nwrote {WORK / 'sensitivity.json'}   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
