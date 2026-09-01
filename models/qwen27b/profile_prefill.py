#!/usr/bin/env python3
"""profile_prefill.py - where does prefill time actually go?

Times one GDN layer and one full-attention layer directly, then projects to
the 48/16 layer split, and compares against a measured whole-model prefill.
"""
import argparse, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import wired_limit
from mlx_lm.models.cache import make_prompt_cache
from models.qwen27b.paged_kv import require_free_memory

E2 = ("/Users/gioma/.lmstudio/models/gioma/"
      "Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1")


def timeit(fn, n=5):
    for _ in range(2):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=E2)
    p.add_argument("--chunk", type=int, default=256)
    a = p.parse_args()
    require_free_memory(8.0)
    print(f"loading...", flush=True)
    model, tok = load(a.model)
    lm = model.language_model.model
    layers = lm.layers
    H = lm.args.hidden_size if hasattr(lm, "args") else 5120
    T = a.chunk
    n_lin = sum(1 for l in layers if l.is_linear)
    n_att = len(layers) - n_lin
    print(f"layers: {n_lin} GDN + {n_att} attention, chunk {T}\n", flush=True)

    with wired_limit(model):
        x = mx.random.normal((1, T, H)).astype(mx.bfloat16)
        cache = make_prompt_cache(model)

        gdn_i = next(i for i, l in enumerate(layers) if l.is_linear)
        att_i = next(i for i, l in enumerate(layers) if not l.is_linear)

        t_gdn = timeit(lambda: layers[gdn_i](x, mask=None, cache=cache[gdn_i]))
        t_att = timeit(lambda: layers[att_i](x, mask="causal", cache=cache[att_i]))

        # MLP alone, to separate it from the mixing op
        mlp = layers[gdn_i].mlp
        t_mlp = timeit(lambda: mlp(x))

        proj = n_lin * t_gdn + n_att * t_att
        print(f"{'per layer':<22}{'ms':>9}{'x count':>10}{'total ms':>11}")
        print(f"{'GDN layer (incl MLP)':<22}{t_gdn*1e3:9.2f}{n_lin:>10}{n_lin*t_gdn*1e3:11.1f}")
        print(f"{'attention layer':<22}{t_att*1e3:9.2f}{n_att:>10}{n_att*t_att*1e3:11.1f}")
        print(f"{'  (MLP alone)':<22}{t_mlp*1e3:9.2f}{64:>10}{64*t_mlp*1e3:11.1f}")
        print(f"\nprojected chunk time : {proj*1e3:.0f} ms  "
              f"-> {T/proj:.1f} tok/s")

        cache2 = make_prompt_cache(model)
        t_full = timeit(lambda: model(mx.zeros((1, T), dtype=mx.int32), cache=cache2), n=3)
        print(f"measured whole model : {t_full*1e3:.0f} ms  -> {T/t_full:.1f} tok/s")
        share_gdn = n_lin * t_gdn / proj * 100
        share_att = n_att * t_att / proj * 100
        print(f"\nGDN share {share_gdn:.0f}%   attention share {share_att:.0f}%")
        print(f"MLP is {64*t_mlp/proj*100:.0f}% of projected time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
