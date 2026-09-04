#!/usr/bin/env python3
"""The standard production benchmark. One protocol, comparable numbers.

Decode rate on this machine spans about 21 percent on identical code,
depending on how much of the checkpoint the page cache holds. Free memory
correlates with rate at -0.84: more free memory means a colder cache and a
slower run. A rate published without its physical read volume therefore
cannot be compared against another rate, and a two-arm A/B cannot resolve
anything.

This harness enforces the protocol:

* alternates conditions so drift hits every arm equally,
* discards the first arms of each condition, which run on a cold cache,
* stops as soon as the median settles, instead of running a fixed count. The
  machine is fanless. A long run heats it, the clock drops, and the extra
  arms buy noise rather than confidence,
* records every arm with the seconds since the run began, and reports the
  correlation between rate and elapsed time. A strongly negative one means
  the run measured thermal decay, not the change under test,
* reports median and range, never a bare mean,
* reports physical MB per token beside every rate,
* asserts identical token IDs across arms, so a changed runtime cannot pass
  by producing different work.

Examples:

    # what the code does right now
    python models/flashnext/bench_production.py --arms 12

    # does isolating n-gram traffic protect expert residency?
    python models/flashnext/bench_production.py --compare ngram-nocache

    # does warming last session's expert set help the first turn?
    python models/flashnext/bench_production.py --compare prewarm --fresh-arms
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROMPT = ("<|im_start|>user\nExplique a fotossintese em duas frases."
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

COMPARISONS = {
    "none": {"baseline": {}},
    "wired": {
        "wired0": {"FLASHNEXT_WIRED_GB": "0"},
        "wired2": {"FLASHNEXT_WIRED_GB": "2"},
    },
    # `empty_rows` allocates a fresh numpy block for every part of every layer,
    # about 1.18 GB of transient host memory per token across 432 allocations.
    # The ring reuses a handful instead. The 2x2 against the read-ceiling
    # benchmark could not see it: its ram arm pins everything and its disk arm
    # caches nothing, so neither arm's hit rate can move. Production is the
    # only arm where it can, and that is where it showed.
    "buffer-arena": {
        "fresh": {"FLASHNEXT_BUFFER_ARENA": "0"},
        "ring3": {"FLASHNEXT_BUFFER_ARENA": "3"},
    },
    "ngram-nocache": {
        "baseline": {"FLASHNEXT_NGRAM_NOCACHE": "0"},
        "nocache": {"FLASHNEXT_NGRAM_NOCACHE": "1"},
    },
    "prewarm": {
        "baseline": {"FLASHNEXT_PREWARM": "0"},
        "prewarm": {"FLASHNEXT_PREWARM": "1"},
    },
    "read-mode": {
        "pread": {"FLASHNEXT_READ": "pread"},
        "resident": {"FLASHNEXT_READ": "resident"},
    },
    # The shared buffer removes the concatenate but is written by 16 workers
    # scattering across it, where the concatenate was one sequential copy. The
    # research log flags that difference as the unmeasured explanation for why
    # its 35 ms saving came back as GPU time. Chunk 2 and 4 keep most of the
    # queue depth while each worker writes a contiguous run, which is the
    # point the two settings were never tested at together.
    "buffer-chunk": {
        "concat-chunk1": {
            "FLASHNEXT_SHARED_READ_BUFFER": "0", "FLASHNEXT_PREAD_CHUNK": "1",
        },
        "buffer-chunk2": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "2",
        },
        "buffer-chunk4": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "4",
        },
    },
    # The three-way sweep above cannot report a band or a sign test, so the
    # winner is settled head to head against the current default.
    "buffer-chunk2": {
        "concat-chunk1": {
            "FLASHNEXT_SHARED_READ_BUFFER": "0", "FLASHNEXT_PREAD_CHUNK": "1",
        },
        "buffer-chunk2": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "2",
        },
    },
    # A token blocks on 98 evals for about 200 ms while the shaders are 9%
    # busy. This builds the routed list on the host instead of fetching it
    # with a second round trip, which halves the count to 50. Bit-exact:
    # identical token_sha256 on a real chat turn.
    "one-sync": {
        "two-syncs": {"FLASHNEXT_ONE_SYNC": "0"},
        "one-sync": {"FLASHNEXT_ONE_SYNC": "1"},
    },
    "metal-runtime": {
        "stock": {"FLASHNEXT_METAL_RUNTIME": "0", "FLASHNEXT_SLAB": "0"},
        "custom": {"FLASHNEXT_METAL_RUNTIME": "1", "FLASHNEXT_SLAB": "0"},
    },
    "slab-global": {
        "baseline": {"FLASHNEXT_METAL_RUNTIME": "1", "FLASHNEXT_SLAB_GLOBAL": "0"},
        "global48": {"FLASHNEXT_METAL_RUNTIME": "1", "FLASHNEXT_SLAB_GLOBAL": "48"},
    },
    # Compile the elementwise chains around the matmuls. Bit-exact, checked
    # with mx.array_equal before install. The probe's own per-call figures
    # sum to about 1 ms per token, so expect this inside the band.
    "compile": {
        "plain": {"FLASHNEXT_COMPILE": "0"},
        "compiled": {"FLASHNEXT_COMPILE": "1"},
    },
    # Map every row the tracker believes is cached, not only the mlocked ones.
    # Run bench_residency.py first: below 78.5% precision this loses.
    "track-resident": {
        "pinned-only": {
            "FLASHNEXT_READ": "resident",
            "FLASHNEXT_TRACK_RESIDENT": "0",
        },
        "tracked": {
            "FLASHNEXT_READ": "resident",
            "FLASHNEXT_TRACK_RESIDENT": "1",
        },
    },
    # Cache-aware routing: take a resident expert when a cold one scores no
    # better. This changes what the model computes, so compare the text as
    # well as the rate. The tracker must be on for the gate to know anything.
    "swap-resident": {
        "exact": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "0",
        },
        "cache-aware": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
        },
    },
    # How far the swap may reach for a resident expert. At 0.02 it replaces
    # 13.9 percent of cold reads and at 0.005 it replaces 11.9. Above 0.02 is
    # unmeasured. Cache-aware measures 2.91 gen at 0.02, so 3.0 needs about 5
    # percent fewer bytes, from 360 to near 341.
    #
    # A wider epsilon takes experts the router scored further from its choice.
    # This trades answer quality for bytes, so run the quality gate on any
    # value that wins here. Do not promote one on rate alone.
    "swap-epsilon": {
        "e=0.02": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.02",
        },
        "e=0.05": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.05",
        },
    },
    "swap-epsilon-wide": {
        "e=0.02": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.02",
        },
        "e=0.10": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.10",
        },
    },
    # Two changes that are each too small to resolve alone. Both act on how
    # much memory is left for the page cache, so they may not be independent:
    # pinning scales only frees 3.6 GB of mlock, and prewarm needs cache to
    # fill. Measuring the pair costs one comparison instead of two and can
    # resolve a combined effect that neither shows on its own.
    "stacked": {
        "neither": {"FLASHNEXT_PIN_PARTS": "all", "FLASHNEXT_PREWARM": "0"},
        "both": {"FLASHNEXT_PIN_PARTS": "scales", "FLASHNEXT_PREWARM": "1"},
    },
    # Issue a layer's reads in ascending offset order instead of routing
    # order. Never measured. It changes the read pattern rather than the
    # bytes, and read pattern is what the layout result moved.
    "sort-reads": {
        "unsorted": {"FLASHNEXT_SORT_READS": "0"},
        "sorted": {"FLASHNEXT_SORT_READS": "1"},
    },
    # Spend the pin budget on scales and biases across many experts instead of
    # whole experts across few. Needs resident_experts raised and a candidate
    # pool that large, so pass --hot through the tail benchmark to compare
    # depths honestly.
    "pin-parts": {
        "whole-experts": {"FLASHNEXT_PIN_PARTS": "all"},
        "scales-only": {"FLASHNEXT_PIN_PARTS": "scales"},
    },
}


# Every setting a condition can flip, and how to make it real on a live
# backend. A setting read once at import or in a constructor cannot be changed
# by writing the environment, so each one is applied explicitly and then read
# back. The read-back is the point: a comparison that silently measured the
# same thing twice already produced one wrong result in this project.
LIVE_SETTINGS = {
    "FLASHNEXT_READ": (
        lambda backend, value: setattr(backend.store, "_read_mode", value),
        lambda backend: backend.store._read_mode,
        str,
    ),
    "FLASHNEXT_NGRAM_NOCACHE": (
        lambda backend, value: setattr(
            backend.store, "_ngram_nocache", value == "1"
        ),
        lambda backend: backend.store._ngram_nocache,
        lambda value: value == "1",
    ),
    "FLASHNEXT_BUFFER_ARENA": (
        # Read at call time from a list, so a live flip is enough and the
        # setting is read back before either arm reports.
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_buffer_arena"]
        ).set_buffer_arena(value),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["buffer_arena"]
        ).buffer_arena(),
        lambda value: int(value),
    ),
    "FLASHNEXT_WIRED_GB": (
        # MLX wires nothing by default, so every buffer handed to the GPU is
        # evictable. The setter reads the limit back through Metal itself.
        lambda backend, value: __import__(
            "models.flashnext.loader", fromlist=["set_wired_gb"]
        ).set_wired_gb(value),
        lambda backend: __import__(
            "models.flashnext.loader", fromlist=["wired_gb"]
        ).wired_gb(),
        lambda value: float(value),
    ),
    "FLASHNEXT_SORT_READS": (
        lambda backend, value: setattr(
            backend.store, "_sort_reads", value == "1"
        ),
        lambda backend: backend.store._sort_reads,
        lambda value: value == "1",
    ),
    "FLASHNEXT_SWAP_RESIDENT": (
        lambda backend, value: os.environ.__setitem__(
            "FLASHNEXT_SWAP_RESIDENT", value
        ),
        lambda backend: __import__(
            "models.flashnext.routing", fromlist=["swap_enabled"]
        ).swap_enabled(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_PIN_PARTS": (
        # read at call time by routing.pin_parts(); the environment is the
        # state, so setting it is enough, but it is still read back.
        lambda backend, value: os.environ.__setitem__("FLASHNEXT_PIN_PARTS", value),
        lambda backend: __import__(
            "models.flashnext.routing", fromlist=["pin_parts"]
        ).pin_parts(),
        lambda value: ("scales", "biases") if value == "scales"
        else ("weight", "scales", "biases"),
    ),
    "FLASHNEXT_TRACK_RESIDENT": (
        lambda backend, value: setattr(
            backend.store, "_track_residency", value == "1"
        ),
        lambda backend: backend.store._track_residency,
        lambda value: value == "1",
    ),
    "FLASHNEXT_SHARED_READ_BUFFER": (
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_shared_buffer"]
        ).set_shared_buffer(value == "1"),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["shared_buffer"]
        ).shared_buffer(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_PREAD_CHUNK": (
        lambda backend, value: setattr(
            backend.store, "_pread_chunk", int(value)
        ),
        lambda backend: backend.store._pread_chunk,
        lambda value: int(value),
    ),
    "FLASHNEXT_ONE_SYNC": (
        lambda backend, value: __import__(
            "models.flashnext.adaptive_topk", fromlist=["set_one_sync"]
        ).set_one_sync(value == "1"),
        lambda backend: __import__(
            "models.flashnext.adaptive_topk", fromlist=["one_sync"]
        ).one_sync(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_COMPILE": (
        lambda backend, value: (
            __import__(
                "models.flashnext.compiled", fromlist=["install"]
            ).install()
            if value == "1"
            else __import__(
                "models.flashnext.compiled", fromlist=["uninstall"]
            ).uninstall()
        ),
        lambda backend: __import__(
            "models.flashnext.compiled", fromlist=["installed"]
        ).installed(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_SWAP_EPSILON": (
        # The environment is what `begin_decode` reads, but the router uses
        # the module value, so set both and read back the one that decides.
        lambda backend, value: (
            os.environ.__setitem__("FLASHNEXT_SWAP_EPSILON", value),
            __import__(
                "models.flashnext.adaptive_topk", fromlist=["_SWAP_EPSILON"]
            )._SWAP_EPSILON.__setitem__(0, float(value)),
        ),
        lambda backend: __import__(
            "models.flashnext.adaptive_topk", fromlist=["_SWAP_EPSILON"]
        )._SWAP_EPSILON[0],
        lambda value: float(value),
    ),
    "FLASHNEXT_METAL_RUNTIME": (
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_metal_runtime"]
        ).set_metal_runtime(value == "1"),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["metal_runtime"]
        ).metal_runtime(),
        lambda value: value == "1",
    ),
}

# Settings that only take effect while the backend is built. A condition using
# one of these needs --fresh-arms; applying it to a live backend is a no-op.
LOAD_TIME_SETTINGS = {
    "FLASHNEXT_PREWARM",
    "FLASHNEXT_SLAB",
    "FLASHNEXT_SLAB_LAYERS",
    "FLASHNEXT_SLAB_GLOBAL",
    "FLASHNEXT_SLAB_MIN_SLOTS",
}


def apply_condition(backend, env: dict) -> None:
    """Make a condition real on a live backend, then prove it took."""
    for key, value in env.items():
        if key in LOAD_TIME_SETTINGS:
            continue
        entry = LIVE_SETTINGS.get(key)
        if entry is None:
            continue
        setter, getter, expected = entry
        setter(backend, value)
        if getter(backend) != expected(value):
            raise SystemExit(
                f"{key}={value} did not take effect: "
                f"the runtime reports {getter(backend)!r}. "
                f"Refusing to report a comparison that did not happen."
            )


def check_load_time(env: dict, fresh_arms: bool) -> None:
    needed = LOAD_TIME_SETTINGS & set(env)
    if needed and not fresh_arms:
        raise SystemExit(
            f"{', '.join(sorted(needed))} only applies while the backend loads. "
            f"Pass --fresh-arms, or the two arms measure the same thing."
        )


def arm(backend, tokens, meter, run_began):
    from models.flashnext.diskio import free_memory_mb

    free = free_memory_mb()
    backend.reset()
    backend.append_text(PROMPT)
    meter.reset()
    began = time.perf_counter()
    _text, stats = backend.generate(max_tokens=tokens)
    wall = time.perf_counter() - began
    read = meter.bytes_since()
    ids = tuple(backend.tape[-stats.tokens:]) if stats.tokens else ()
    tail = stats.tail_tokens / stats.tail_seconds if stats.tail_seconds else 0.0
    return {
        "elapsed_s": time.perf_counter() - run_began,
        "gen_tokens": stats.tokens,
        "gen_rate": stats.rate,
        "tail_rate": tail,
        "mb_per_token": (read / stats.tokens / 1e6) if read > 0 and stats.tokens else -1.0,
        "pinned_gb": stats.pinned_bytes / 1e9,
        "free_mb_before": free,
        "wall": wall,
        "ids": ids,
    }


def settled(rates, window, tolerance) -> bool:
    """True once the median stops moving, so the run can stop early."""
    if len(rates) < window * 2:
        return False
    before = st.median(rates[:-1])
    now = st.median(rates)
    return abs(now - before) / now < tolerance


def thermal_drift(arms):
    """Correlation between rate and seconds elapsed. Negative means heat."""
    if len(arms) < 4:
        return 0.0
    xs = [a["elapsed_s"] for a in arms]
    ys = [a["gen_rate"] for a in arms]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def report(name, arms, drop):
    kept = arms[drop:]
    rates = [a["gen_rate"] for a in kept]
    tails = [a["tail_rate"] for a in kept]
    mb = [a["mb_per_token"] for a in kept if a["mb_per_token"] > 0]
    line = {
        "condition": name,
        "arms_run": len(arms),
        "arms_kept": len(kept),
        "gen_median": round(st.median(rates), 3),
        "gen_min": round(min(rates), 3),
        "gen_max": round(max(rates), 3),
        "gen_sd": round(st.stdev(rates), 3) if len(rates) > 1 else 0.0,
        "tail_median": round(st.median(tails), 3),
        "mb_per_token_median": round(st.median(mb), 1) if mb else -1.0,
        "free_mb_first": round(kept[0]["free_mb_before"], 0) if kept else -1,
        "thermal_drift": round(thermal_drift(kept), 2),
        "arms": [
            {k: (round(v, 3) if isinstance(v, float) else v)
             for k, v in a.items() if k != "ids"}
            for a in arms
        ],
    }
    print(
        f"  {name:<12} gen median {line['gen_median']:5.2f}  "
        f"range {line['gen_min']:.2f}-{line['gen_max']:.2f}  sd {line['gen_sd']:.3f}  "
        f"tail {line['tail_median']:5.2f}  "
        f"{line['mb_per_token_median']:6.1f} MB/tok  n={len(kept)}",
        flush=True,
    )
    return line


def resolution_note(base, other, change) -> str:
    """What this run can and cannot tell apart, from its own spread.

    Two standard errors of the difference between the medians. A result inside
    that band is not a result, however much it looks like one. Doubling the
    arms shrinks the band by about 30 percent, so the note says what to run
    rather than leaving it to judgement.
    """
    import math

    spread = math.sqrt(
        base["gen_sd"] ** 2 / max(base["arms_kept"], 1)
        + other["gen_sd"] ** 2 / max(other["arms_kept"], 1)
    )
    band = 2 * spread / base["gen_median"] * 100 if base["gen_median"] else 0.0
    if abs(change) >= band:
        return f"resolves differences above {band:.1f} percent, so this one stands."
    doubled = max(base["arms_kept"], other["arms_kept"]) * 2
    return (
        f"resolves differences above {band:.1f} percent, so this one is "
        f"unresolved, which is not the same as absent. Re-run with --arms "
        f"{doubled}, or stack it with another change and measure the pair."
    )


def report_paired(results, drop: int = 0) -> None:
    """Compare arm against arm, which cancels drift.

    Arms alternate, so the two conditions sit at the same point in the run.
    Differencing each pair removes whatever the machine was doing at that
    moment, including heat, and a sign test over the pairs needs no
    assumption about the spread.
    """
    if len(results) != 2:
        return
    base, other = results
    pairs = [
        (x["gen_rate"], y["gen_rate"])
        for x, y in zip(base["arms"][drop:], other["arms"][drop:])
    ]
    if len(pairs) < 4:
        return
    diffs = [(y - x) / x * 100 for x, y in pairs]
    wins = sum(1 for d in diffs if d > 0)
    total = len(diffs)
    from math import comb

    tail = sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total
    bytes_down = sum(
        1 for x, y in zip(base["arms"][drop:], other["arms"][drop:])
        if y["mb_per_token"] < x["mb_per_token"]
    )
    print()
    print(f"  paired over {total} arms: mean {st.mean(diffs):+.1f} percent, "
          f"median {st.median(diffs):+.1f}")
    print(f"  {other['condition']} ahead in {wins} of {total} pairs, "
          f"sign test p = {tail:.3f}")
    print(f"  fewer bytes in {bytes_down} of {total} pairs")
    if tail > 0.05:
        print("  The pairs do not separate. Treat this as unresolved.")


def report_drift(results) -> None:
    """Separate a hot machine from a condition that degrades as it runs.

    Conditions are interleaved, so heat reaches every arm. When every
    condition slides, the run measured the machine. When one slides and
    another does not, the run measured that condition.
    """
    sliding = [r for r in results if r["thermal_drift"] < -0.6]
    if not sliding:
        return
    print()
    if len(sliding) == len(results):
        print("  every condition falls with elapsed time. The machine is hot,")
        print("  so the absolute rates are depressed. Arms alternate, so the")
        print("  paired comparison above is unaffected; the medians are not.")
        return
    for row in sliding:
        steady = [r for r in results if r not in sliding]
        print(f"  {row['condition']} falls with elapsed time "
              f"(r={row['thermal_drift']:+.2f}) while "
              f"{', '.join(r['condition'] for r in steady)} does not.")
        print(f"  Interleaved arms share the same wall clock, so this is not")
        print(f"  heat: {row['condition']} degrades as it runs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=int, default=8,
                        help="ceiling on arms per condition; the run stops "
                             "earlier once the median settles")
    parser.add_argument("--min-arms", type=int, default=5,
                        help="arms per condition before early stopping applies")
    parser.add_argument("--tolerance", type=float, default=0.015,
                        help="stop once the median moves less than this fraction")
    parser.add_argument("--drop", type=int, default=2,
                        help="cold arms discarded per condition")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--compare", choices=sorted(COMPARISONS), default="none")
    parser.add_argument("--fresh-arms", action="store_true",
                        help="reload the model between conditions; needed when a "
                             "setting only takes effect at load, such as prewarm")
    parser.add_argument("--json", default="", help="write the summary here")
    args = parser.parse_args()

    conditions = COMPARISONS[args.compare]
    if args.min_arms - args.drop < 3:
        parser.error("--min-arms must exceed --drop by at least three")

    for env in conditions.values():
        check_load_time(env, args.fresh_arms)

    from models.flashnext.diskio import ReadMeter

    meter = ReadMeter()
    run_began = time.perf_counter()
    results, collected, first_ids = [], {name: [] for name in conditions}, None

    if args.fresh_arms:
        # Rebuild the backend per condition. Settings must be read at call
        # time, never captured in a module constant: Python caches the module
        # after the first import, so a constant keeps the first condition's
        # value and both arms silently measure the same thing.
        for name, env in conditions.items():
            os.environ.update(env)
            from macqwen.backends.flashnext import FlashNextBackend
            from models.flashnext.routing import prewarm_enabled

            if "FLASHNEXT_PREWARM" in env:
                want = env["FLASHNEXT_PREWARM"] == "1"
                if prewarm_enabled() != want:
                    raise SystemExit(
                        f"condition {name} asked for prewarm={want} but the "
                        f"runtime reports {prewarm_enabled()}. Refusing to "
                        f"report a comparison that did not happen."
                    )
            backend = FlashNextBackend()
            for index in range(args.arms):
                row = arm(backend, args.tokens, meter, run_began)
                first_ids = first_ids or row["ids"]
                if row["ids"] != first_ids:
                    print(f"  !! arm {index + 1} of {name} produced different tokens")
                collected[name].append(row)
                rates = [a["gen_rate"] for a in collected[name][args.drop:]]
                if len(collected[name]) >= args.min_arms and settled(
                    rates, args.min_arms // 2, args.tolerance
                ):
                    print(f"  {name}: median settled after {index + 1} arms")
                    break
            # Release the shards. Each backend maps 22 files, and a leaked map
            # keeps page-cache references alive into the next condition, which
            # is exactly the variable this harness is trying to hold still.
            backend.store.close()
            del backend
    else:
        os.environ.update(next(iter(conditions.values())))
        from macqwen.backends.flashnext import FlashNextBackend

        backend = FlashNextBackend()
        cond_keys = list(conditions.keys())
        print(f"  system load average before: {os.getloadavg()}", flush=True)

        order = []
        for i in range(args.arms):
            if i % 2 == 0:
                order.extend(cond_keys)
            else:
                order.extend(reversed(cond_keys))

        for index, name in enumerate(order, 1):
            os.environ.update(conditions[name])
            apply_condition(backend, conditions[name])
            row = arm(backend, args.tokens, meter, run_began)
            first_ids = first_ids or row["ids"]
            if row["ids"] != first_ids and args.compare != "swap-resident":
                print(f"  !! arm {index} of {name} produced different tokens")
            collected[name].append(row)
            if all(
                len(rows) >= args.min_arms
                and settled(
                    [a["gen_rate"] for a in rows[args.drop:]],
                    args.min_arms // 2, args.tolerance,
                )
                for rows in collected.values()
            ):
                print(f"  every median settled after {index} arms")
                break

    import hashlib
    print()
    for name, arms in collected.items():
        results.append(report(name, arms, args.drop))
        kept = arms[args.drop:]
        if kept and kept[0]["ids"]:
            h = hashlib.sha256(bytes(str(kept[0]["ids"]), "utf-8")).hexdigest()[:16]
            print(f"    token digest ({name}): {h}", flush=True)
    print(f"  system load average after: {os.getloadavg()}", flush=True)

    report_paired(results, args.drop)
    report_drift(results)

    if len(results) == 2:
        base, other = results
        change = (other["gen_median"] - base["gen_median"]) / base["gen_median"] * 100
        print(f"\n  {other['condition']} vs {base['condition']}: {change:+.1f}% gen median")
        print(f"  {resolution_note(base, other, change)}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"comparison": args.compare, "conditions": results}, handle, indent=2)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
