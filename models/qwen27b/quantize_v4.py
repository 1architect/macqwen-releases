#!/usr/bin/env python3
"""quantize_v4.py - fill the RAM budget with the bits that buy the most.

The v3 recipe picks a bit width per tensor family by hand. This picks it per
tensor by measurement, then solves for the assignment that minimises total
output distortion under a fixed byte budget.

For weight W, activation profile a, and quantiser Q:

    score = mean_rows || (W - Q(W)) * a ||^2

That is the expected squared error this tensor injects into its own output.
Errors from separate tensors add, so the assignment is a knapsack. Options are
ordered by size, so greedy marginal-gain-per-byte is exact.

Three stages, because the BF16 source is 54 GB and the disk is 57 GB free:

    score   consume each shard as it lands: measure, store an 8-bit copy that
            is near lossless, delete the shard
    plan    solve the allocation against a byte budget
    build   requantise the 8-bit store down to the assigned widths

Stage `score` can run while the download is still going.
"""
import argparse, json, os, re, struct, time, heapq, shutil
from pathlib import Path

import numpy as np
import mlx.core as mx

SRC = Path.home() / "bf16-src"
WORK = Path.home() / ".frankenstein" / "v4"
CALIB = Path.home() / ".frankenstein" / "calibration.npz"
# Source of the tokenizer and the non-quantisation half of config.json. Any
# built model serves; V3.1 was deleted, so do not depend on one path existing.
def _reference():
    """Any built model of this family will do: only the tokenizer and the
    non-quantisation half of config.json are copied. Naming specific builds
    was fragile, because they get deleted to make room."""
    root = Path.home() / "models"
    cands = [d for d in sorted(root.glob("*"))
             if (d / "config.json").exists() and (d / "tokenizer.json").exists()]
    for d in cands:                       # prefer a Qwen3.8 build
        if "Qwen3.8-27B" in d.name:
            return d
    if cands:
        return cands[0]
    raise SystemExit(f"no reference model with a tokenizer under {root}")


REF = _reference()

# (bits, group_size) pairs MLX affine supports and that are worth considering
# Every width at every group size MLX supports. Group size was previously
# offered only at 4 bits, which was an oversight: the scale and bias pair costs
# 4 bytes per group whatever the width, so the overhead is proportionally
# WORST at low widths. A 2-bit tensor pays 1.00 bits/weight of bookkeeping at
# group 32 and 0.25 at group 128. Those tensors are most of the model.
OPTIONS = [(b, g) for b in (2, 3, 4, 5, 6, 8) for g in (32, 64, 128)]
STORE_BITS, STORE_GROUP = 8, 64      # intermediate for the tough tensors

# Requantising out of the intermediate store costs error, and the cost grows as
# the target approaches the store's own precision. Measured on a real tensor:
#
#     target   2b      3b      4b      5b      6b      8b
#     penalty  0.02%   0.06%   0.32%   1.7%    8.2%    27%
#
# So the store is only safe for targets at or below 4 bits. Families the
# measurement wants above 4 bits keep their true BF16 in the store instead.
FRAGILE = ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")
STORE_MAX_BITS = 4                   # cap for anything held at 8 bits


def is_fragile(name):
    return any(f in name for f in FRAGILE)

SHARDS = ["00001", "00002", "00004", "00005", "00006", "00007", "00008",
          "00009", "00010", "00011", "00012", "00013", "00014", "00015",
          "00016", "00017"]


def flush(*a):
    print(*a, flush=True)


def canon(name):
    """One key shape for both the safetensors names and the runtime names."""
    n = name
    for p in ("model.language_model.", "language_model.model.",
              "language_model.", "model."):
        if n.startswith(p):
            n = n[len(p):]
            break
    if n.endswith(".weight"):
        n = n[: -len(".weight")]
    return n


def nbytes(out, inp, bits, group):
    return out * inp * bits / 8 + (out * inp / group) * 4


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_bf16(path, info, base):
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    arr = np.frombuffer(raw, dtype=np.uint16).reshape(info["shape"])
    return mx.array(arr).view(mx.bfloat16)


# ------------------------------------------------------------------- score

def score_shard(shard, calib, scores, store_dir):
    head, base = st_header(shard)
    saved = {}
    for name, info in head.items():
        if name == "__metadata__":
            continue
        if name.startswith("model.visual") or "mtp" in name:
            continue
        key = canon(name)
        shape = info["shape"]
        if len(shape) != 2 or shape[0] < 64:
            # norms, biases, conv kernels: keep exact, they are tiny
            saved[name] = ("raw", read_bf16(shard, info, base))
            continue
        W = read_bf16(shard, info, base)
        out, inp = W.shape
        a = calib.get(key)
        if a is None:
            a = np.ones(inp, dtype=np.float32)
        av = mx.array(np.asarray(a, dtype=np.float32))
        if av.shape[0] != inp:
            av = mx.ones((inp,))
        Wf = W.astype(mx.float32)
        row = {}
        for bits, group in OPTIONS:
            if inp % group:
                continue
            q, s, b = mx.quantize(W, group_size=group, bits=bits)
            D = mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.float32)
            E = (Wf - D) * av
            row[f"{bits}g{group}"] = float(mx.mean(mx.sum(E * E, axis=-1)))
        scores[key] = {"shape": [out, inp], "scores": row,
                       "fragile": is_fragile(name)}
        if is_fragile(name):
            saved[name] = ("raw", W)          # true BF16, no round trip
        else:
            q, s, b = mx.quantize(W, group_size=STORE_GROUP, bits=STORE_BITS)
            saved[name] = ("q8", (q, s, b))
            mx.eval(q, s, b)
    dest = store_dir / (shard.stem + ".q8.safetensors")
    flat = {}
    for name, (kind, v) in saved.items():
        if kind == "raw":
            flat[name] = v
        else:
            q, s, b = v
            flat[name + ".q"] = q
            flat[name + ".s"] = s
            flat[name + ".b"] = b
    mx.save_safetensors(str(dest), flat)
    return len(scores)


def cmd_score(args):
    WORK.mkdir(parents=True, exist_ok=True)
    store = WORK / "q8"
    store.mkdir(exist_ok=True)
    calib = {canon(k): v for k, v in np.load(CALIB).items()} if CALIB.exists() else {}
    flush(f"{len(calib)} activation profiles loaded")

    spath = WORK / "scores.json"
    scores = json.loads(spath.read_text()) if spath.exists() else {}
    done = set(json.loads((WORK / "done.json").read_text())) if (WORK / "done.json").exists() else set()

    for s in SHARDS:
        if s in done:
            continue
        f = SRC / f"model-{s}-of-00018.safetensors"
        flag = Path(str(f) + ".done")
        waited = 0
        while not flag.exists():
            if waited == 0:
                flush(f"waiting for shard {s}")
            time.sleep(20)
            waited += 20
        t0 = time.time()
        n = score_shard(f, calib, scores, store)
        spath.write_text(json.dumps(scores))
        done.add(s)
        (WORK / "done.json").write_text(json.dumps(sorted(done)))
        if not args.keep:
            f.unlink(missing_ok=True)
        free = shutil.disk_usage("/System/Volumes/Data").free / 1e9
        flush(f"shard {s}: {n} tensors scored, {time.time()-t0:.0f}s, "
              f"{free:.1f} GB free")
    flush(f"\nscored {len(scores)} tensors -> {spath}")


# -------------------------------------------------------------------- plan

def cmd_plan(args):
    scores = json.loads((WORK / "scores.json").read_text())
    flush(f"{len(scores)} tensors, budget {args.budget:.2f} GB")

    floors = {}
    for spec in getattr(args, "floor", []) or []:
        fam, _, bits = spec.partition(":")
        floors[fam] = int(bits)
    depth_floors = []
    for spec in getattr(args, "floor_layers", []) or []:
        rng, _, bits = spec.partition(":")
        lo, _, hi = rng.partition("-")
        depth_floors.append((int(lo), int(hi), int(bits)))
    if depth_floors:
        flush("depth floors: " + ", ".join(f"{l}-{h}>={b}b"
                                           for l, h, b in depth_floors))
    if floors:
        flush("floors: " + ", ".join(f"{k}>={v}b" for k, v in floors.items()))

    opts = {}
    for key, d in scores.items():
        out, inp = d["shape"]
        fam = key.split(".")[-1]
        floor = floors.get(fam, 0)
        m = re.match(r"layers\.(\d+)\.", key)
        if m:
            li = int(m.group(1))
            for lo, hi, b in depth_floors:
                if lo <= li <= hi:
                    floor = max(floor, b)
        cand = []
        capped = not d.get("fragile", False)
        for tag, sc in d["scores"].items():
            bits, group = tag.split("g")
            bits, group = int(bits), int(group)
            # An explicit floor overrides the store cap. Requantising out of
            # the 8-bit store costs 0.3% at 4 bits and 1.7% at 5, which is
            # small next to the effect a floor is there to test.
            limit = max(STORE_MAX_BITS, floor)
            if capped and bits > limit:
                continue
            if bits < floor:
                continue         # family floor
            cand.append((nbytes(out, inp, bits, group), sc, bits, group))
        cand.sort()                      # by size
        keep = []
        for c in cand:                   # drop options that cost more and score worse
            while keep and keep[-1][1] <= c[1]:
                keep.pop()
            keep.append(c)
        opts[key] = keep

    cur = {k: 0 for k in opts}
    total = sum(opts[k][0][0] for k in opts)
    heap = []
    for k in opts:
        if len(opts[k]) > 1:
            b0, s0, _, _ = opts[k][0]
            b1, s1, _, _ = opts[k][1]
            heapq.heappush(heap, (-(s0 - s1) / (b1 - b0), k))

    budget = args.budget * 1e9
    while heap and total < budget:
        gain, k = heapq.heappop(heap)
        i = cur[k]
        b0 = opts[k][i][0]
        b1 = opts[k][i + 1][0]
        if total - b0 + b1 > budget:
            continue
        total += b1 - b0
        cur[k] = i + 1
        if i + 2 < len(opts[k]):
            s1 = opts[k][i + 1][1]
            s2 = opts[k][i + 2][1]
            b2 = opts[k][i + 2][0]
            heapq.heappush(heap, (-(s1 - s2) / (b2 - b1), k))

    plan = {k: {"bits": opts[k][cur[k]][2], "group_size": opts[k][cur[k]][3]}
            for k in opts}
    (WORK / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True))

    params = sum(scores[k]["shape"][0] * scores[k]["shape"][1] for k in scores)
    dist = sum(opts[k][cur[k]][1] for k in opts)
    base = sum(opts[k][0][1] for k in opts)
    flush(f"\nallocated {total/1e9:.2f} GB over {params/1e9:.2f}B params "
          f"= {total*8/params:.2f} bits/param")
    flush(f"distortion {dist:.4g} (all-minimum would be {base:.4g}, "
          f"{base/max(dist,1e-30):.1f}x worse)\n")

    fam = {}
    for k, v in plan.items():
        f = k.split(".")[-1] if "." in k else k
        fam.setdefault(f, []).append(v["bits"])
    flush(f"{'tensor family':<24}{'n':>4}{'bits: min  mean  max':>24}")
    for f, bs in sorted(fam.items()):
        flush(f"{f:<24}{len(bs):>4}{min(bs):>10}{np.mean(bs):>6.1f}{max(bs):>6}")




# ------------------------------------------------------------------- build

def runtime_name(src):
    """safetensors name in the BF16 source -> name the MLX runtime expects."""
    if src.startswith("lm_head."):
        return "language_model." + src
    if src.startswith("model.language_model."):
        return "language_model.model." + src[len("model.language_model."):]
    return src


def cmd_build(args):
    plan_path = Path(args.plan) if args.plan else WORK / "plan.json"
    plan = json.loads(plan_path.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = WORK / "q8"
    files = sorted(store.glob("*.q8.safetensors"))
    if not files:
        raise SystemExit(f"no intermediate shards in {store}; run `score` first")

    qcfg = {"group_size": 64, "bits": 4}
    index = {}
    buckets = {}

    def bucket_for(rt):
        if ".layers." in rt:
            return f"model-layer-{int(rt.split('.layers.')[1].split('.')[0]):02d}"
        if "embed_tokens" in rt:
            return "model-embed"
        if "lm_head" in rt:
            return "model-head"
        return "model-final"

    t0 = time.time()
    for f in files:
        raw = mx.load(str(f))
        names = sorted({k.rsplit(".", 1)[0] for k in raw if k.endswith((".q", ".s", ".b"))})
        # fragile tensors were stored as true BF16, so they are plain entries
        names += [k for k in raw
                  if k.endswith(".weight") and is_fragile(k) and raw[k].ndim == 2]
        for src in sorted(set(names)):
            key = canon(src)
            rt = runtime_name(src)
            if "embed_tokens" in rt and args.drop_embed:
                continue
            if src + ".q" in raw:
                W = mx.dequantize(raw[src + ".q"], raw[src + ".s"], raw[src + ".b"],
                                  group_size=STORE_GROUP, bits=STORE_BITS)
            else:
                W = raw[src]                  # true BF16, never round tripped
            if "lm_head" in rt:
                bits, group = args.head_bits, 64
            else:
                spec = plan.get(key)
                if spec is None:
                    bits, group = 8, 64
                else:
                    bits, group = spec["bits"], spec["group_size"]
            q, sc, bi = mx.quantize(W, group_size=group, bits=bits)
            mx.eval(q, sc, bi)
            base = rt[: -len(".weight")] if rt.endswith(".weight") else rt
            qcfg[base] = {"bits": bits, "group_size": group, "mode": "affine"}
            b = bucket_for(rt)
            d = buckets.setdefault(b, {})
            d[base + ".weight"] = q
            d[base + ".scales"] = sc
            d[base + ".biases"] = bi
        for k, v in raw.items():
            if k.endswith((".q", ".s", ".b")) or k in names:
                continue
            rt = runtime_name(k)
            if "embed_tokens" in rt and args.drop_embed:
                continue
            buckets.setdefault(bucket_for(rt), {})[rt] = v
        del raw
        mx.clear_cache()
        flush(f"  {f.name} -> {len(buckets)} buckets, {time.time()-t0:.0f}s")

    # embed and lm_head live outside the shard stream: they were pulled to raw
    # BF16 by bf16_ends.py. Quantise them here so the build is a drop-in model.
    ends = Path.home() / ".frankenstein" / "bf16-ends"
    if (ends / "meta.json").exists():
        em = json.loads((ends / "meta.json").read_text())
        for fname, shape_key, rt, bits in (
                ("embed.bf16", "embed_shape",
                 "language_model.model.embed_tokens", args.embed_bits),
                ("head.bf16", "head_shape",
                 "language_model.lm_head", args.head_bits)):
            if args.drop_embed and "embed_tokens" in rt:
                continue
            shape = tuple(em[shape_key])
            mm = np.memmap(ends / fname, dtype=np.uint16, mode="r", shape=shape)
            qs, ss, bs = [], [], []
            for s0 in range(0, shape[0], 16384):     # row blocks, to stay small
                W = mx.array(np.asarray(mm[s0:s0 + 16384])).view(mx.bfloat16)
                q, sc, bi = mx.quantize(W, group_size=64, bits=bits)
                mx.eval(q, sc, bi)
                qs.append(q); ss.append(sc); bs.append(bi)
            d = buckets.setdefault(bucket_for(rt + ".weight"), {})
            d[rt + ".weight"] = mx.concatenate(qs, axis=0)
            d[rt + ".scales"] = mx.concatenate(ss, axis=0)
            d[rt + ".biases"] = mx.concatenate(bs, axis=0)
            qcfg[rt] = {"bits": bits, "group_size": 64, "mode": "affine"}
            flush(f"  {rt} at {bits} bits")
            del qs, ss, bs
            mx.clear_cache()

    total = 0
    for b, d in sorted(buckets.items()):
        path = out / f"{b}.safetensors"
        mx.save_safetensors(str(path), d)
        total += path.stat().st_size
        for k in d:
            index[k] = f"{b}.safetensors"
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total},
                    "weight_map": dict(sorted(index.items()))}, indent=2))

    cfg = json.loads((REF / "config.json").read_text())
    cfg["quantization"] = qcfg
    cfg["quantization_config"] = qcfg
    generation = args.generation.lower()
    cfg[f"{generation}_method"] = args.method_note
    cfg[f"{generation}_budget_gb"] = args.budget_note
    cfg[f"{generation}_plan"] = plan_path.name
    if args.constraints_note:
        cfg[f"{generation}_constraints"] = args.constraints_note
    if args.drop_embed:
        cfg[f"{generation}_embed"] = "external BF16, see bf16_ends.py"
        cfg["external_embedding"] = "bf16-ends/embed.bf16"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json"):
        src_f = REF / name
        if src_f.exists():
            shutil.copy2(src_f, out / name)

    flush(f"\nwrote {out}  {total/1e9:.2f} GB in {time.time()-t0:.0f}s")


# ----------------------------------------------------------------- rescore

def cmd_rescore(args):
    """Re-measure every option from the intermediate store.

    The BF16 shards are gone, so scores come from the q8 store instead: true
    BF16 for the fragile families, 8 bits for the rest. Requantising out of an
    8-bit source costs 0.02% at 2 bits and 0.32% at 4, and non-fragile tensors
    are capped at 4 bits anyway, so the numbers stand.
    """
    calibration = Path(args.calibration) if args.calibration else CALIB
    calib = ({canon(k): v for k, v in np.load(calibration).items()}
             if calibration.exists() else {})
    flush(f"{len(calib)} activation profiles")
    store = WORK / "q8"
    files = sorted(store.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no intermediate store in {store}")

    scores = {}
    t0 = time.time()
    for f in files:
        raw = mx.load(str(f))
        names = {k.rsplit(".", 1)[0] for k in raw if k.endswith(".q")}
        names |= {k for k in raw if k.endswith(".weight") and is_fragile(k)
                  and raw[k].ndim == 2}
        for src in sorted(names):
            if src + ".q" in raw:
                W = mx.dequantize(raw[src + ".q"], raw[src + ".s"], raw[src + ".b"],
                                  group_size=STORE_GROUP, bits=STORE_BITS)
            else:
                W = raw[src]
            if W.ndim != 2 or W.shape[0] < 64:
                continue
            out, inp = W.shape
            key = canon(src)
            a = calib.get(key)
            av = mx.array(np.asarray(a, dtype=np.float32)) if a is not None \
                else mx.ones((inp,))
            if av.shape[0] != inp:
                av = mx.ones((inp,))
            Wf = W.astype(mx.float32)
            row = {}
            for bits, group in OPTIONS:
                if inp % group:
                    continue
                q, sc, bi = mx.quantize(W, group_size=group, bits=bits)
                D = mx.dequantize(q, sc, bi, group_size=group,
                                  bits=bits).astype(mx.float32)
                E = (Wf - D) * av
                row[f"{bits}g{group}"] = float(mx.mean(mx.sum(E * E, axis=-1)))
            scores[key] = {"shape": [out, inp], "scores": row,
                           "fragile": is_fragile(src)}
            del W, Wf
            mx.clear_cache()
        del raw
        mx.clear_cache()
        flush(f"  {f.name}  {len(scores)} tensors  {time.time()-t0:.0f}s")

    output = Path(args.out) if args.out else WORK / "scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores))
    flush(f"\nrescored {len(scores)} tensors, {len(OPTIONS)} options each "
          f"-> {output}  ({time.time()-t0:.0f}s)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--keep", action="store_true", help="do not delete shards")
    s.set_defaults(fn=cmd_score)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(Path.home() / "models/"
                                        "Qwen3.8-27B-Apple-MLX-V4"))
    b.add_argument("--embed-bits", type=int, default=4)
    b.add_argument("--head-bits", type=int, default=4,
                   help="resident shortlist head; exact logits come from SSD")
    b.add_argument("--drop-embed", action="store_true",
                   help="omit embed_tokens; the runtime reads BF16 from SSD")
    b.add_argument("--budget-note", type=float, default=11.0)
    b.add_argument("--plan", help="JSON allocation plan; defaults to the V4 plan")
    b.add_argument("--generation", default="v4",
                   help="metadata prefix written to config.json")
    b.add_argument("--method-note",
                   default="measured_activation_weighted_bit_allocation")
    b.add_argument("--constraints-note")
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("rescore")
    r.add_argument("--calibration", help="activation profile NPZ")
    r.add_argument("--out", help="output score JSON")
    r.set_defaults(fn=cmd_rescore)
    q = sub.add_parser("plan")
    q.add_argument("--floor-layers", action="append", default=[],
                   metavar="LO-HI:BITS",
                   help="minimum bits for a depth range, e.g. 0-31:3. The proxy "
                        "weights error by activation power, which grows with "
                        "depth, so it protects late layers and starves early "
                        "ones. Error propagation runs the other way.")
    q.add_argument("--floor", action="append", default=[],
                   metavar="FAMILY:BITS",
                   help="minimum bits for a tensor family, e.g. o_proj:5. The "
                        "allocator's proxy scores each tensor by the error in "
                        "its own output, which cannot see that corrupting what "
                        "attention retrieved changes WHICH token gets copied.")
    q.add_argument("--budget", type=float, default=11.0, help="GB for the body")
    q.set_defaults(fn=cmd_plan)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
