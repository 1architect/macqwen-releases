#!/usr/bin/env python3
"""Price the routed-expert gather inside the real model's shapes.

The bandwidth record puts GDN at 67 GB/s and a chained Q4 matvec at 105 GB/s.
It never priced `gather_qmm`, which is where the expert weights are actually
consumed. A token gathers about 1,152 MB of expert data across 48 layers. At
105 GB/s that costs about 11 ms. At 20 GB/s it costs about 58 ms. The GPU half
of a token is 152 to 186 ms, so the answer either closes that block or opens a
30 to 45 ms target.

Every array is read once and evaluated before timing, so the drive is out of
the measurement. The benchmark asserts that: it reads the process disk counter
around the timed section and refuses to report if the drive served anything.

Timing follows the rule the research log had to learn twice. One `mx.eval` per
operation charges each one a full round trip, and summing across stages then
inflates the total. Reps are chained under a single eval, over distinct inputs
so nothing is folded away. The single-op round trip is reported beside it as a
separate number, never subtracted.

This benchmark changes no model behavior. It needs no quality gate.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx

from mlx_vlm.models.switch_layers import SwiGLU, _gather_sort, _scatter_unsort

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.diskio import disk_bytes_read, free_memory_mb
from models.flashnext.store import SafeTensorStore

PARTS = ("weight", "scales", "biases")
# The loader hands StreamingSwitchGLU the SwitchGLU it replaces, so the model
# runs this exact activation. `_one_pass` calls it as activation(up, gate).
_ACTIVATION = SwiGLU()
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_experts(store, prefix, rows):
    """Read one layer's expert rows and make them resident.

    Returns the same three arrays `_one_pass` hands to `gather_qmm`, built by
    the same store call, so the dtypes and the layout are the model's own.
    """
    packs = {}
    for projection in PROJECTIONS:
        parts = []
        for part in PARTS:
            name = f"{prefix}.{projection}.{part}"
            block = store.rows_np(name, rows, "pread")
            parts.append(store.to_mx(name, block))
        packs[projection] = tuple(parts)
    mx.eval([array for pack in packs.values() for array in pack])
    return packs


def load_experts_scattered(store, prefix, rows, chunk):
    """Build the same arrays the way the shared read buffer does.

    One destination per part, filled by pool workers each writing a disjoint
    contiguous run of `chunk` rows. The concatenate path above produces the
    same bytes through one sequential copy on the main thread.

    The research log flagged this difference as the unmeasured explanation for
    why the shared buffer's 35 ms saving came back as GPU drain: the GPU may
    read memory in a different coherency state. Same bytes, same kernel, only
    the write pattern differs, so a throughput gap here is that effect and
    nothing else.
    """
    from models.flashnext.expert_cache import _POOL

    packs = {}
    for projection in PROJECTIONS:
        parts = []
        for part in PARTS:
            name = f"{prefix}.{projection}.{part}"
            buffer = store.empty_rows(name, len(rows))
            futures = [
                _POOL.submit(
                    store.rows_into, name,
                    rows[start:start + chunk],
                    buffer[start:start + chunk], "pread",
                )
                for start in range(0, len(rows), chunk)
            ]
            for future in futures:
                future.result()
            parts.append(store.to_mx(name, buffer))
        packs[projection] = tuple(parts)
    mx.eval([array for pack in packs.values() for array in pack])
    return packs


def expert_bytes(store, prefix, width):
    """Bytes one gather touches per projection, from the checkpoint header."""
    per = {}
    for projection in PROJECTIONS:
        total = 0
        for part in PARTS:
            total += store.refs[f"{prefix}.{projection}.{part}"].row_bytes
        per[projection] = total * width
    return per


def make_inputs(reps, shape, dtype):
    """Distinct inputs, one per rep, evaluated before timing.

    Repeating one input lets the graph collapse the reps into a single
    computation and the benchmark then measures nothing.
    """
    xs = [mx.random.normal(shape).astype(dtype) * 0.02 for _ in range(reps)]
    mx.eval(xs)
    return xs


def slots(width):
    local = mx.arange(width, dtype=mx.uint32).reshape(1, 1, width)
    mx.eval(local)
    return local


def one_projection(x, pack, local, group_size, bits, sort, expand):
    """One `gather_qmm` on the shapes that projection sees in the model.

    `gate_proj` and `up_proj` take the layer input broadcast across slots, so
    it arrives expanded to (1, 1, 1, 1, hidden). `down_proj` takes the
    activation output, which is already per-slot at (1, 1, width, moe_inter),
    and is not expanded again. Feeding one shape to all three fails.
    """
    weight, scales, biases = pack
    return mx.gather_qmm(
        mx.expand_dims(x, (-2, -3)) if expand else x,
        weight,
        scales,
        biases,
        rhs_indices=local,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
        sorted_indices=sort,
    )


def moe_block(x, packs, local, group_size, bits, sort, indices_shape=None):
    """The compute half of `_one_pass`, with the reads already done."""
    xs = mx.expand_dims(x, (-2, -3))
    idx = local
    inv = None
    if sort:
        xs, idx, inv = _gather_sort(xs, local)
    common = dict(
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
        sorted_indices=sort,
    )
    gate = mx.gather_qmm(xs, *packs["gate_proj"], rhs_indices=idx, **common)
    up = mx.gather_qmm(xs, *packs["up_proj"], rhs_indices=idx, **common)
    out = mx.gather_qmm(
        _ACTIVATION(up, gate), *packs["down_proj"], rhs_indices=idx, **common
    )
    if sort:
        out = _scatter_unsort(out, inv, indices_shape)
    return out.squeeze(-2)


def chained(builder, xs):
    """Time `len(xs)` independent operations under one eval.

    Returns milliseconds for the whole chain. Divide by the rep count for a
    per-call figure that carries no sync of its own.
    """
    began = time.perf_counter()
    outs = [builder(x) for x in xs]
    mx.eval(outs)
    return (time.perf_counter() - began) * 1000.0


def sync_floor(reps):
    """One `mx.eval` round trip, measured on an array that costs nothing."""
    probe = mx.zeros((1,))
    mx.eval(probe)
    began = time.perf_counter()
    for _ in range(reps):
        mx.eval(probe + 0.0)
    return (time.perf_counter() - began) * 1000.0 / reps


def report(label, ms, reps, byte_count, layers):
    per = ms / reps
    rate = byte_count / (per / 1000.0) / 1e9 if per > 0 else 0.0
    print(
        f"  {label:26s} {per:7.3f} ms  {rate:7.1f} GB/s  "
        f"{per * layers:7.1f} ms/token"
    )
    return per


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--width", type=int, default=8,
                        help="routed slots per layer; decode keeps about 8")
    parser.add_argument("--prefill-width", type=int, default=64,
                        help="slot count that turns the sort path on")
    parser.add_argument("--reps", type=int, default=64)
    parser.add_argument("--arms", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--write-pattern", choices=("concat", "scatter", "both"),
                        default="both",
                        help="how the weight arrays were written before the "
                             "gather reads them")
    parser.add_argument("--chunk", type=int, default=2,
                        help="rows per worker in the scatter arm")
    args = parser.parse_args()

    model = args.model or str(resolve_flashnext())
    store = SafeTensorStore(model)
    prefix = f"language_model.model.layers.{args.layer}.mlp.switch_mlp"
    if f"{prefix}.gate_proj.weight" not in store.refs:
        print(f"no switch_mlp at layer {args.layer} in {model}", file=sys.stderr)
        return 2

    shape = store.shape(f"{prefix}.gate_proj.weight")
    hidden = store.shape(f"{prefix}.down_proj.weight")[1]
    moe_inter = shape[1]
    width = min(args.width, shape[0])
    prefill = min(args.prefill_width, shape[0])
    print(f"model        {model}")
    print(f"layer {args.layer}   experts={shape[0]} hidden={hidden} "
          f"moe_inter={moe_inter} g={args.group_size} bits={args.bits}")
    print(f"free memory  {free_memory_mb():.0f} MB")

    rows = list(range(max(width, prefill)))
    packs = load_experts(store, prefix, rows)
    narrow = {
        name: tuple(part[:width] for part in pack)
        for name, pack in packs.items()
    }
    mx.eval([array for pack in narrow.values() for array in pack])

    per_projection = expert_bytes(store, prefix, width)
    block_bytes = sum(per_projection.values())
    wide_bytes = sum(expert_bytes(store, prefix, prefill).values())
    print(f"gather       {width} slots, {block_bytes/1e6:.1f} MB per layer, "
          f"{block_bytes*args.layers/1e6:.0f} MB per token")

    xs = make_inputs(args.reps, (1, 1, hidden), mx.bfloat16)
    wide_xs = make_inputs(args.reps, (1, 1, hidden), mx.bfloat16)
    local = slots(width)
    wide_local = slots(prefill)

    # `down_proj` consumes the activation output. Guessing its shape gets a
    # different kernel: a hand-built (1, 1, width, moe_inter) measured ten
    # times slower than the same projection inside the block. Take the shape
    # from a real gate output instead.
    probe = one_projection(
        xs[0], narrow["gate_proj"], local, args.group_size, args.bits, False, True
    )
    mx.eval(probe)
    mids = make_inputs(args.reps, probe.shape, mx.bfloat16)
    print(f"shapes       gate/up in {tuple(xs[0].shape)} out {tuple(probe.shape)}")

    reference = moe_block(xs[0], narrow, local, args.group_size, args.bits, False)
    mx.eval(reference)

    floor = sync_floor(64)
    print(f"sync floor   {floor:.3f} ms per eval round trip")

    before = disk_bytes_read()
    results = {}
    for arm in range(args.arms):
        for projection in PROJECTIONS:
            pack = narrow[projection]
            expand = projection != "down_proj"
            ms = chained(
                lambda x, p=pack, e=expand: one_projection(
                    x, p, local, args.group_size, args.bits, False, e
                ),
                xs if expand else mids,
            )
            results.setdefault(projection, []).append(ms)
        ms = chained(
            lambda x: moe_block(
                x, narrow, local, args.group_size, args.bits, False
            ),
            xs,
        )
        results.setdefault("block", []).append(ms)
        ms = chained(
            lambda x: moe_block(
                x, packs, wide_local, args.group_size, args.bits, True,
                wide_local.shape,
            ),
            wide_xs,
        )
        results.setdefault("block_sorted", []).append(ms)
    after = disk_bytes_read()

    check = moe_block(xs[0], narrow, local, args.group_size, args.bits, False)
    mx.eval(check)
    exact = bool(mx.array_equal(check, reference))

    print(f"\nchained, {args.reps} reps under one eval, "
          f"median of {args.arms} arms, {width} slots:")
    per_call = {}
    for projection in PROJECTIONS:
        per_call[projection] = report(
            projection,
            statistics.median(results[projection]),
            args.reps,
            per_projection[projection],
            args.layers,
        )
    block = report(
        "three projections",
        statistics.median(results["block"]),
        args.reps,
        block_bytes,
        args.layers,
    )
    # Prefill runs one pass for the whole prompt, so the per-layer cost here
    # is amortised over every prompt token. The ms/token column is left at one
    # layer to stop anyone reading it as a decode figure.
    print(f"\nsorted path, {prefill} slots (prefill shape), per layer:")
    report(
        "block with sort",
        statistics.median(results["block_sorted"]),
        args.reps,
        wide_bytes,
        1,
    )

    separate = sum(per_call.values())
    print(
        f"\nsum of separate projections {separate:.3f} ms against "
        f"{block:.3f} ms measured as one block"
    )

    read = after - before
    print(f"\npremise:")
    print(f"  physical read during timing   {read/1e6:.2f} MB")
    print(f"  outputs bit-identical         {exact}")
    ok = True
    if read > 8 << 20:
        print("  REFUSED: the drive served reads, so this timed I/O too")
        ok = False
    if not exact:
        print("  REFUSED: the timed path changed the output")
        ok = False
    if not ok:
        return 1

    # --- write pattern: does the GPU care how the weights were written? ----
    if args.write_pattern == "both":
        print(f"\nwrite pattern, {width} slots, same bytes, same kernel:")
        arms = {}
        for label, builder in (
            ("concat, main thread", lambda: load_experts(store, prefix, rows)),
            ("scatter, %d workers" % ((len(rows) + args.chunk - 1) // args.chunk),
             lambda: load_experts_scattered(store, prefix, rows, args.chunk)),
        ):
            built = builder()
            pack = {
                n: tuple(part[:width] for part in pk) for n, pk in built.items()
            }
            mx.eval([a for pk in pack.values() for a in pk])
            runs = []
            for _ in range(args.arms):
                runs.append(chained(
                    lambda x, pk=pack: moe_block(
                        x, pk, local, args.group_size, args.bits, False
                    ),
                    xs,
                ))
            per = statistics.median(runs) / args.reps
            arms[label] = per
            rate = block_bytes / (per / 1000.0) / 1e9
            print(f"  {label:26s} {per:7.3f} ms  {rate:7.1f} GB/s  "
                  f"{per * args.layers:7.1f} ms/token")
            # the two paths must produce identical weights
            first = pack["gate_proj"][0]
            if label.startswith("concat"):
                reference_weights = first
            else:
                mx.eval(first, reference_weights)
                same = bool(mx.array_equal(first, reference_weights))
                print(f"  {'bytes identical':26s} {same}")
                if not same:
                    print("  REFUSED: the two write paths produced different "
                          "bytes, so this compares nothing")
                    return 1
        names = list(arms)
        delta = (arms[names[1]] - arms[names[0]]) / arms[names[0]] * 100.0
        gap = (arms[names[1]] - arms[names[0]]) * args.layers
        print(f"  {'scatter vs concat':26s} {delta:+6.1f}%  "
              f"{gap:+6.1f} ms/token")
        print("  A gap here is the coherency effect the research log named and")
        print("  never measured. No gap means the shared buffer is innocent and")
        print("  score_sync's remainder is scheduling, not work.")

    print(f"\nreading:")
    print(f"  gather at {block_bytes/ (block/1000.0) / 1e9:.1f} GB/s puts the "
          f"expert matmuls at {block*args.layers:.0f} ms per token")
    print("  the GPU half of a token measured 152 to 186 ms")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
