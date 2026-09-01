#!/usr/bin/env python3
"""bit_allocator.py - spend the RAM budget where it buys the most quality.

The v3 recipe assigns bit widths by hand, one rule per tensor family. That
ignores two things the hardware cares about: how much a tensor actually
distorts its own output when quantized, and how many bytes that costs.

This measures both, then solves the allocation.

    calibrate   record per-input-channel activation power for every linear
    plan        score every (tensor, bit width) pair and fill the budget

The score for one tensor is the expected squared output error it injects:

    score(t, b) = mean_rows  || (W - Q_b(W)) * a ||^2 ,  a_c = sqrt(E[x_c^2])

Errors from different tensors add, so the allocation is a knapsack. Each
tensor's options are ordered in bits, so the greedy marginal-gain-per-byte
solution is exact under a Lagrange multiplier.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

MODEL = str(Path.home() / "models/Qwen3.8-27B-Apple-MLX-V3.1-Compact")
CALIB = Path.home() / ".frankenstein" / "calibration.npz"
REPO_ROOT = Path(__file__).resolve().parents[2]


def flush(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- calibrate

class Watch(nn.Module):
    """Records the mean square of each input channel, then delegates."""

    def __init__(self, inner, name, store):
        super().__init__()
        self.inner = inner
        self._name = name
        self._store = store

    def __call__(self, x):
        f = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        s = mx.sum(f * f, axis=0)
        n = f.shape[0]
        prev = self._store.get(self._name)
        self._store[self._name] = (s, n) if prev is None else (prev[0] + s, prev[1] + n)
        return self.inner(x)


def walk(module, prefix=""):
    """Yield (parent, attribute, full name) for every linear-like child."""
    for k, v in module.items() if hasattr(module, "items") else []:
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, nn.Module):
            if hasattr(v, "weight") and v.weight.ndim == 2 and "norm" not in k:
                yield module, k, name
            else:
                yield from walk(v, name)
        elif isinstance(v, (list, tuple)):
            for i, sub in enumerate(v):
                if isinstance(sub, nn.Module):
                    yield from walk(sub, f"{name}.{i}")


def calibrate(args):
    from mlx_lm import load
    import mlx_lm.models.qwen3_5 as q5

    def last_only(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        return self.lm_head(out[:, -1:, :])
    q5.TextModel.__call__ = last_only          # skip the 248320-wide logits

    t0 = time.time()
    model, tok = load(args.model)
    flush(f"loaded in {time.time()-t0:.0f}s")

    store = {}
    targets = [t for t in walk(model)]
    hooked = 0
    for parent, key, name in targets:
        if "embed_tokens" in name or name.startswith("visual"):
            continue      # a lookup, and the vision tower this build never runs
        setattr(parent, key, Watch(getattr(parent, key), name, store))
        hooked += 1
    flush(f"watching {hooked} linear tensors")

    texts = []
    if not args.only_corpus:
        for p in (
            REPO_ROOT / "macqwen" / "chat.py",
            Path(__file__).with_name("paged_kv.py"),
            Path(__file__).with_name("frankenstein_engine.py"),
        ):
            texts.append(p.read_text()[:6000])
        texts.append((REPO_ROOT / "docs" / "RESEARCH-LOG.md").read_text()[:6000])
        texts.append("Explain step by step how to design a paged attention kernel. "
                     "Then write the Metal shader.\n" * 12)
    for corpus in args.corpus:
        path = Path(corpus).expanduser()
        texts.append(path.read_text(encoding="utf-8")[: args.corpus_chars])

    total = 0
    for i, t in enumerate(texts):
        ids = tok.encode(t)[: args.tokens]
        if len(ids) < 32:
            continue
        x = mx.array([ids])
        for j in range(0, len(ids), 256):
            out = model(x[:, j:j + 256])
            # Evaluate the output and the accumulators in ONE call. Two calls
            # make MLX free the intermediates after the first, then recompute
            # the whole forward pass to satisfy the second.
            mx.eval(out, *[v[0] for v in store.values()])
        total += len(ids)
        flush(f"  sample {i}: {len(ids)} tokens (total {total})")

    out = {}
    for name, (s, n) in store.items():
        out[name] = np.array(mx.sqrt(s / n))
    target = Path(args.out).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **out)
    flush(f"\nwrote {len(out)} activation profiles to {target}")
    flush(f"{total} calibration tokens, {time.time()-t0:.0f}s")

    rms = {k: float(v.mean()) for k, v in out.items()}
    hot = sorted(rms.items(), key=lambda kv: -kv[1])[:8]
    flush("\nhighest mean input activation (most sensitive to weight error):")
    for k, v in hot:
        flush(f"  {v:10.3f}  {k}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--model", default=MODEL)
    c.add_argument("--tokens", type=int, default=1024)
    c.add_argument("--corpus", action="append", default=[],
                   help="additional task-aware calibration text")
    c.add_argument("--corpus-chars", type=int, default=24000)
    c.add_argument("--out", default=str(CALIB))
    c.add_argument("--only-corpus", action="store_true",
                   help="exclude the built-in Python and Metal samples")
    c.set_defaults(fn=calibrate)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
