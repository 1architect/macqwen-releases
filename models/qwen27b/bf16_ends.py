#!/usr/bin/env python3
"""bf16_ends.py - run the two ends of the model at exact BF16, from SSD.

The embedding and the output head are 2.54 GB each at BF16, 9.4% of the model.
Quantizing them costs RAM and quality for no speed gain, because neither one
is bandwidth-bound the way the body is.

    embedding   a row lookup. One row per token, 10 KB. SSD latency is 0.3 ms.
    lm_head     a full matmul, but only a few thousand rows can ever win.
                A small resident head picks the candidates, then the exact
                BF16 rows for those candidates come off the SSD.

So both ends run exact, and the RAM they used goes back to the body.

    extract   pull the two tensors out of the BF16 shards into raw row-major
    test      measure shortlist recall and logit fidelity against exact BF16
"""
import argparse, json, os, struct, time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

SRC = Path.home() / "bf16-src"
OUT = Path.home() / ".frankenstein" / "bf16-ends"
MODEL = "/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-V3.1-Compact"
ENGINE_SOURCE = Path(__file__).with_name("frankenstein_engine.py")


def flush(*a):
    print(*a, flush=True)


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def extract_tensor(shard, name, dest):
    """Copy one tensor's raw bytes out of a safetensors shard."""
    head, base = st_header(shard)
    if name not in head:
        raise KeyError(f"{name} not in {shard.name}: {list(head)[:4]}...")
    info = head[name]
    a, b = info["data_offsets"]
    shape, dtype = info["shape"], info["dtype"]
    if dtype != "BF16":
        raise ValueError(f"{name} is {dtype}, expected BF16")
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(shard, "rb") as f, open(dest, "wb") as o:
        f.seek(base + a)
        left = b - a
        while left:
            chunk = f.read(min(1 << 24, left))
            o.write(chunk)
            left -= len(chunk)
    flush(f"  {name}\n    {shape} BF16 -> {dest}  "
          f"{(b-a)/1e9:.3f} GB in {time.time()-t0:.1f}s")
    return shape


def extract(args):
    meta = {}
    meta["embed_shape"] = extract_tensor(
        SRC / "model-00003-of-00018.safetensors",
        "model.language_model.embed_tokens.weight", OUT / "embed.bf16")
    meta["head_shape"] = extract_tensor(
        SRC / "model-00018-of-00018.safetensors",
        "lm_head.weight", OUT / "head.bf16")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    flush(f"\nwrote {OUT}/meta.json")


# ------------------------------------------------------------------ modules

class SSDEmbedding(nn.Module):
    """Exact BF16 embedding held on SSD. Hot rows stay in the page cache."""

    def __init__(self, path, shape):
        super().__init__()
        self.mm = np.memmap(path, dtype=np.uint16, mode="r",
                            shape=(shape[0], shape[1]))
        self.dim = shape[1]

    def __call__(self, ids):
        idx = np.array(ids, copy=False).reshape(-1)
        rows = np.asarray(self.mm[idx])
        out = mx.array(rows).view(mx.bfloat16)
        return out.reshape(*ids.shape, self.dim)

    def as_linear(self, x):
        raise NotImplementedError("this model does not tie embeddings")


class ShortlistHead(nn.Module):
    """Exact BF16 logits for the only tokens that can win.

    `small` is a cheap resident head. It ranks the vocabulary, the top `k`
    rows come off the SSD at BF16, and those get exact logits. Everything
    else is -inf, which every sampler downstream already handles.
    """

    def __init__(self, small, path, shape, k=2048):
        super().__init__()
        self.small = small
        self.mm = np.memmap(path, dtype=np.uint16, mode="r",
                            shape=(shape[0], shape[1]))
        self.k = k
        self.V = shape[0]
        self.reads = 0
        self.bytes = 0

    def __call__(self, x):
        approx = self.small(x)
        B, T, V = approx.shape
        flat = approx.reshape(-1, V)
        idx = mx.argpartition(-flat, self.k, axis=-1)[:, :self.k]
        mx.eval(idx)
        out = mx.full((flat.shape[0], V), -mx.inf, dtype=mx.float32)
        xs = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        for r in range(flat.shape[0]):
            ids = np.array(idx[r])
            rows = mx.array(np.asarray(self.mm[ids])).view(mx.bfloat16)
            self.reads += 1
            self.bytes += ids.size * self.mm.shape[1] * 2
            exact = xs[r] @ rows.astype(mx.float32).T
            out[r, mx.array(ids)] = exact
        return out.reshape(B, T, V)


def attach_ssd_ends(model, k=1024, verbose=True):
    """Move both ends of a loaded V4 model onto the SSD at exact BF16.

    The build ships a 2-bit embedding purely so stock mlx_lm can load it. That
    copy is dropped here and the RAM goes back. The 2-bit head stays, but only
    to rank the vocabulary: the logits that matter come from exact BF16 rows.
    """
    meta = json.loads((OUT / "meta.json").read_text())
    mx.eval(model.parameters())
    before = mx.get_active_memory() / 1e9
    lm = model.language_model
    lm.model.embed_tokens = SSDEmbedding(OUT / "embed.bf16", meta["embed_shape"])
    lm.lm_head = ShortlistHead(lm.lm_head, OUT / "head.bf16", meta["head_shape"], k)
    mx.clear_cache()
    after = mx.get_active_memory() / 1e9
    if verbose:
        flush(f"BF16 ends: resident {before:.2f} -> {after:.2f} GB "
              f"(freed {before-after:.2f} GB), shortlist k={k}")
    return before - after


def load_v4(path, k=1024, verbose=True):
    """Load a V4 build with both ends on the SSD at exact BF16."""
    from mlx_lm import load as _load
    import mlx_lm.models.qwen3_5 as q5

    def last_only(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        return self.lm_head(out[:, -1:, :])
    q5.TextModel.__call__ = last_only     # the shortlist ranks one position

    model, tok = _load(path)
    attach_ssd_ends(model, k=k, verbose=verbose)
    return model, tok


def load_v4_lean(path, k=1024, verbose=True):
    """Load a V4 build without ever materialising the embedding it discards.

    `load_v4` uses stock mlx_lm, which builds `embed_tokens`, quantises it to
    2 bits, loads 0.397 GB into it, and then throws it away when the SSD
    embedding takes over. The discarded copy is only transient, but transient
    is what peak memory is made of, and peak decides whether a build fits.

    So replicate the load and intervene: drop the embedding from the
    quantisation plan and from the weight list, and put SSDEmbedding in place
    before `load_weights` runs.
    """
    from pathlib import Path as _Path
    import mlx.nn as _nn
    from mlx_lm.utils import load_tokenizer
    import mlx_lm.utils as _u
    # The helper is private in some builds and public in others.
    get_model_classes = (getattr(_u, "get_model_classes", None)
                         or getattr(_u, "_get_classes"))
    import mlx_lm.models.qwen3_5 as q5

    def last_only(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        return self.lm_head(out[:, -1:, :])
    q5.TextModel.__call__ = last_only

    path = _Path(path)
    config = json.loads((path / "config.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    EMB = "language_model.model.embed_tokens"
    config.get("quantization", {}).pop(EMB, None)
    config.get("quantization_config", {}).pop(EMB, None)

    weights = {}
    for wf in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(wf)))

    model_class, model_args_class = get_model_classes(config=config)
    model = model_class(model_args_class.from_dict(config))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # The swap happens here, before quantisation sizes anything and before a
    # single embedding byte is read.
    model.language_model.model.embed_tokens = SSDEmbedding(
        OUT / "embed.bf16", meta["embed_shape"])
    weights = {w: v for w, v in weights.items() if "embed_tokens" not in w}

    quant = config.get("quantization")
    if quant:
        def class_predicate(p, m):
            if p in quant:
                return quant[p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights
        _nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                     mode=quant.get("mode", "affine"),
                     class_predicate=class_predicate)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    resident = mx.get_active_memory() / 1e9

    model.language_model.lm_head = ShortlistHead(
        model.language_model.lm_head, OUT / "head.bf16", meta["head_shape"], k)
    mx.clear_cache()
    if verbose:
        flush(f"BF16 ends: resident {resident:.2f} GB, embedding never loaded, "
              f"shortlist k={k}")
    return model, load_tokenizer(path)


# -------------------------------------------------------------------- test

def test(args):
    from mlx_lm import load
    meta = json.loads((OUT / "meta.json").read_text())
    t0 = time.time()
    model, tok = load(args.model)
    flush(f"loaded in {time.time()-t0:.0f}s")

    lm = model.language_model
    text = ENGINE_SOURCE.read_text()[:8000]
    ids = tok.encode(text)[:args.tokens]
    x = mx.array([ids])

    # hidden states, once
    h = lm.model(x)
    mx.eval(h)
    h = h.astype(mx.float32)

    exact_w = np.memmap(OUT / "head.bf16", dtype=np.uint16, mode="r",
                        shape=tuple(meta["head_shape"]))
    flush("\ncomputing exact BF16 logits as ground truth (chunked)")
    V, H = meta["head_shape"]
    exact = mx.zeros((h.shape[1], V), dtype=mx.float32)
    step = 16384
    hh = h[0]
    for s in range(0, V, step):
        e = min(s + step, V)
        W = mx.array(np.asarray(exact_w[s:e])).view(mx.bfloat16).astype(mx.float32)
        exact[:, s:e] = hh @ W.T
        mx.eval(exact)
    ex_top = mx.argmax(exact, axis=-1)
    ex_lp = exact - mx.logsumexp(exact, axis=-1, keepdims=True)

    cur = lm.lm_head(h)[0].astype(mx.float32)
    cur_lp = cur - mx.logsumexp(cur, axis=-1, keepdims=True)
    cur_match = float(mx.mean((mx.argmax(cur, axis=-1) == ex_top).astype(mx.float32)))
    cur_kl = float(mx.mean(mx.sum(mx.exp(cur_lp) * (cur_lp - ex_lp), axis=-1)))
    flush(f"\nbaseline, the resident 4-bit head this model ships with:")
    flush(f"  top-1 agreement with exact BF16 {cur_match*100:5.1f}%   "
          f"KL {cur_kl:.5f}   RAM 0.70 GB")

    def approx_head(bits):
        """Logits from a head quantised to `bits`, built in row blocks."""
        a = mx.zeros((hh.shape[0], V), dtype=mx.float32)
        for s0 in range(0, V, step):
            e0 = min(s0 + step, V)
            W = mx.array(np.asarray(exact_w[s0:e0])).view(mx.bfloat16)
            q, sc, bi = mx.quantize(W, group_size=64, bits=bits)
            D = mx.dequantize(q, sc, bi, group_size=64, bits=bits).astype(mx.float32)
            a[:, s0:e0] = hh @ D.T
            mx.eval(a)
        return a

    def head_bytes(bits):
        return (V * H * bits / 8 + V * H / 64 * 4) / 1e9

    flush(f"\nselector head -> shortlist -> exact BF16 rows off SSD")
    flush(f"{'selector':>10}{'RAM':>7}{'k':>7}{'recall':>9}{'top-1':>8}"
          f"{'KL':>10}{'mass':>8}{'MB/tok':>8}")
    for bits in (2, 3, 4):
        ap = approx_head(bits)
        for k in (256, 1024):
            idx = mx.argpartition(-ap, k, axis=-1)[:, :k]
            mx.eval(idx)
            rows = mx.arange(ap.shape[0])[:, None]
            hit = float(mx.mean(mx.sum((idx == ex_top[:, None]).astype(mx.float32), axis=-1)))
            sl = mx.full(exact.shape, -mx.inf, dtype=mx.float32)
            sl[rows, idx] = exact[rows, idx]
            sl_lp = sl - mx.logsumexp(sl, axis=-1, keepdims=True)
            m = float(mx.mean((mx.argmax(sl, axis=-1) == ex_top).astype(mx.float32)))
            # forward KL is infinite by construction, so report the reverse:
            # how much the shortlist distribution disagrees with the exact one
            kl = float(mx.mean(mx.sum(mx.where(mx.isinf(sl_lp), 0.0,
                       mx.exp(sl_lp) * (sl_lp - ex_lp)), axis=-1)))
            mass = float(mx.mean(mx.sum(mx.exp(ex_lp[rows, idx]), axis=-1)))
            flush(f"{bits:>8}b{head_bytes(bits):>6.2f}G{k:>7}{hit*100:>8.2f}%"
                  f"{m*100:>7.1f}%{kl:>10.5f}{mass*100:>7.2f}%{k*H*2/1e6:>8.1f}")
            del sl, sl_lp
            mx.clear_cache()
        del ap
        mx.clear_cache()
    flush(f"\n{time.time()-t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.set_defaults(fn=extract)
    t = sub.add_parser("test")
    t.add_argument("--model", default=MODEL)
    t.add_argument("--tokens", type=int, default=256)
    t.set_defaults(fn=test)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
