#!/usr/bin/env python3
"""Is there a cheaper expert the router would barely miss?

Adaptive top-k computes 10 experts per layer and keeps about 8, so two are
already scored and discarded. If a kept expert has to come off the drive while
a discarded one is already in memory and scores almost as well, taking the
resident one turns a cold read into a cached one for a bounded loss of routed
mass.

This measures whether that situation arises. It changes nothing: the observer
records the router's real decisions and the swap is only counted, never made.
Build the mechanism only if the opportunity is here, and not before.

The loss is reported as the routed mass given up, the same quantity adaptive
top-k already trades when it drops experts 9 and 10.
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROMPT = ("<|im_start|>user\nExplique a fotossintese em duas frases."
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def swaps_for(chosen, spare, epsilon):
    """Pair each cold pick with the best unused resident spare.

    A spare serves one pick. The cheapest cold pick is matched first, because
    giving up the least routed mass is the point.
    """
    used, taken = set(), []
    for cold_expert, cold_score in sorted(chosen, key=lambda pair: pair[1]):
        best = None
        for spare_expert, spare_score in spare:
            if spare_expert in used:
                continue
            if cold_score - spare_score <= epsilon:
                if best is None or spare_score > best[1]:
                    best = (spare_expert, spare_score)
        if best is not None:
            used.add(best[0])
            taken.append(cold_score - best[1])
    return taken


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.05])
    args = parser.parse_args()

    # Track residency without acting on it: the read path stays `pread`.
    os.environ["FLASHNEXT_TRACK_RESIDENT"] = "1"
    os.environ.setdefault("FLASHNEXT_READ", "pread")
    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext.adaptive_topk import set_route_observer

    backend = FlashNextBackend()
    store = backend.store
    prefixes: dict[int, str] = {}
    for index, layer in enumerate(backend.language.model.layers):
        block = getattr(layer.mlp, "switch_mlp", None)
        if block is not None:
            prefixes[index] = block.gate_proj.cache.prefix.rsplit(".", 1)[0]

    def resident(layer: int, expert: int) -> bool:
        """Every tensor of this expert already in memory."""
        prefix = prefixes.get(layer)
        if prefix is None:
            return False
        return all(
            store.believed_resident(f"{prefix}.{projection}.weight", expert)
            for projection in PROJECTIONS
        )

    stats = {
        "decisions": 0,
        "kept": 0,
        "kept_cold": 0,
        "swappable": {e: 0 for e in args.epsilons},
        "mass_lost": {e: [] for e in args.epsilons},
    }

    def observe(layer, expert_rows, score_rows, keeps):
        for experts, scores, keep in zip(expert_rows, score_rows, keeps):
            stats["decisions"] += 1
            stats["kept"] += keep
            chosen = list(zip(experts[:keep], scores[:keep]))
            dropped = list(zip(experts[keep:], scores[keep:]))
            spare = [(e, s) for e, s in dropped if resident(layer, e)]
            cold = [(e, s) for e, s in chosen if not resident(layer, e)]
            stats["kept_cold"] += len(cold)
            if not spare or not cold:
                continue
            # Pair the cheapest cold pick with the best resident spare.
            for epsilon in args.epsilons:
                taken = swaps_for(cold, spare, epsilon)
                stats["swappable"][epsilon] += len(taken)
                stats["mass_lost"][epsilon].extend(taken)

    backend.reset()
    backend.append_text(PROMPT)
    backend.generate(max_tokens=args.warmup)      # let the tracker fill

    # `generate` installs the routing profile's own observer and clears it
    # afterwards, so setting one here is discarded. Chain instead: whatever
    # the runtime installs still runs, and this one runs after it.
    import models.flashnext.adaptive_topk as topk

    original_setter = topk.set_route_observer

    def chained(observer, max_rows=None):
        if observer is None:
            # Preserve the runtime's row cap while chaining after its reset.
            original_setter(observe, max_rows)
            return

        def both(layer, expert_rows, score_rows, keeps):
            observer(layer, expert_rows, score_rows, keeps)
            observe(layer, expert_rows, score_rows, keeps)

        original_setter(both, max_rows)

    topk.set_route_observer = chained
    try:
        original_setter(observe)
        backend.reset()
        backend.append_text(PROMPT)
        backend.generate(max_tokens=args.tokens)
    finally:
        topk.set_route_observer = original_setter
        original_setter(None)

    kept = stats["kept"]
    cold = stats["kept_cold"]
    if not kept:
        raise SystemExit(
            "the observer saw no routing decisions. The routing profile "
            "installs its own observer inside generate(); this run failed to "
            "chain onto it, so nothing was measured."
        )
    print(f"  routing decisions      {stats['decisions']:,}")
    print(f"  experts kept           {kept:,}")
    print(f"  of those, cold         {cold:,}  ({cold / kept:.1%})")
    print()
    print(f"  {'epsilon':>8}{'swaps':>10}{'of cold':>10}{'cold bytes':>13}"
          f"{'mass given up':>16}")
    for epsilon in args.epsilons:
        swaps = stats["swappable"][epsilon]
        losses = stats["mass_lost"][epsilon]
        share = swaps / cold if cold else 0.0
        mass = st.mean(losses) if losses else 0.0
        print(f"  {epsilon:>8.3f}{swaps:>10,}{share:>9.1%}{-share:>12.1%}"
              f"{mass:>15.4f}")
    print()
    print("  `cold bytes` is the change in physical reads if every counted")
    print("  swap were taken. `mass given up` is the mean routed weight lost")
    print("  per swap; adaptive top-k already discards experts 9 and 10.")
    print()
    print("  This measures opportunity only. Nothing was swapped, and the")
    print("  tokens are the model's real ones.")


if __name__ == "__main__":
    main()
