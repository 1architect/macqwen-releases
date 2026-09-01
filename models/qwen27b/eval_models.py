#!/usr/bin/env python3
"""eval_models.py - head to head on held-out text.

Each model loads in its own process, because two of these do not fit in 16 GB
at once. Reports loss, perplexity, resident size and decode rate.

The held-out files were never used for calibration.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Do not delete the files named here, and do not edit them. They are the
# held-out corpus. Perplexity is comparable across builds only while the
# corpus stays byte-identical, so bench_decode.py and profile_prefill.py
# stay in this repo even though nothing imports them.
HELD_OUT = [
    "macqwen/tools/repo_token_cache.py",
    "models/qwen27b/speculative_prefill.py",
    "models/qwen27b/profile_prefill.py",
    "models/qwen27b/repo_context_image.py",
    "docs/server.md",
    "docs/testing.md",
]

CHILD = r'''
import json, sys, time
import numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
path, files = sys.argv[1], json.loads(sys.argv[2])
t0 = time.time()
model, tok = load(path)
mx.eval(model.parameters())
res = mx.get_active_memory() / 1e9
load_s = time.time() - t0
tot = cnt = 0.0
for f in files:
    text = open(f).read()[:9000]
    ids = tok.encode(text)[:512]
    if len(ids) < 64:
        continue
    lg = model(mx.array([ids]))[0].astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    tot += float(mx.sum(-lp[:-1][mx.arange(len(ids)-1), mx.array(ids[1:])]))
    cnt += len(ids) - 1
    del lg, lp
    mx.clear_cache()
nll = tot / cnt
c = make_prompt_cache(model)
ids = tok.encode(open(files[0]).read()[:4000])[:512]
o = model(mx.array([ids]), cache=c); mx.eval(o)
y = mx.argmax(o[:, -1:], axis=-1); mx.eval(y)
t1 = time.time(); n = 24
for _ in range(n):
    o = model(y, cache=c); y = mx.argmax(o[:, -1:], axis=-1); mx.eval(y)
dec = n / (time.time() - t1)
print(json.dumps({"nll": nll, "ppl": float(np.exp(nll)), "resident_gb": res,
                  "decode_tps": dec, "load_s": load_s, "tokens": int(cnt)}))
'''


def run(path):
    files = [str(ROOT / f) for f in HELD_OUT if (ROOT / f).exists()]
    p = subprocess.run([sys.executable, "-c", CHILD, path, json.dumps(files)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip().splitlines()[-1] if p.stderr else "failed"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    models = sys.argv[1:]
    rows = []
    for m in models:
        name = Path(m).name
        print(f"evaluating {name}", flush=True)
        r = run(m)
        r["name"] = name
        rows.append(r)
        print(f"  {r}", flush=True)

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\nno model evaluated successfully")
        return 1
    base = ok[0]
    print(f"\n{'model':<42}{'GB':>7}{'NLL':>9}{'ppl':>9}{'vs base':>10}{'tok/s':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<42}  FAILED: {r['error'][:60]}")
            continue
        d = (base["ppl"] - r["ppl"]) / base["ppl"] * 100
        print(f"{r['name']:<42}{r['resident_gb']:>7.2f}{r['nll']:>9.4f}"
              f"{r['ppl']:>9.3f}{d:>9.2f}%{r['decode_tps']:>8.2f}")
    print(f"\nheld out: {', '.join(HELD_OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
