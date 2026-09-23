#!/usr/bin/env python3
"""Offline expert-cache simulator over a recorded route trace.

Replays the reads recorded by ``bench_route_trace`` through several cache
policies at several capacities and reports the physical MB per decode token
each would read. It needs no model. The unit is one expert record (all nine
tensors of one expert in one layer); the page cache works on pages, so the
absolute numbers are an approximation and the comparison between policies is
the result.

Always resident, never counted as misses and never occupying the simulated
cache: the slab-pack experts. The pinned experts of each turn become resident
at the token where the runtime pins them (``--pin-token``) and occupy cache
capacity from then on, because they are locked page-cache pages.

Policies:

* ``lru``: least recently used. The macOS page cache is closer to a
  two-queue clock, so ``slru`` is also reported.
* ``slru``: segmented LRU; a second hit promotes to a protected segment
  (``--protected`` share of capacity).
* ``lfu``: least frequently used with exponential ageing per decode token.
* ``lru-admit``: LRU that does not admit an expert until it has been seen
  ``--admit`` times, the cache-admission idea (B2).
* ``opt``: Belady's optimal replacement with bypass, the ceiling for any
  policy that does not change routing.

    python -m models.flashnext.tests.bench.cache_sim results/flashnext/<run>/route-trace.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import sys
from collections import OrderedDict, defaultdict


def load(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def accesses(trace):
    """Flatten events to (turn, phase, token, key) with key = (layer, expert)."""
    slab = {int(layer): set(experts) for layer, experts in trace["slab"].items()}
    rows = []
    for turn, phase, token, layer, experts in trace["events"]:
        resident = slab.get(layer, ())
        for expert in experts:
            if expert in resident:
                continue
            rows.append((turn, phase, token, (layer, expert)))
    return rows


def pinned_schedule(trace, pin_token):
    """(turn, token) -> set of keys pinned from that point."""
    schedule = {}
    for turn, info in enumerate(trace["turns"]):
        keys = {
            (int(layer), int(expert))
            for layer, experts in info.get("pinned", {}).items()
            for expert in experts
        }
        schedule[turn] = keys
    return schedule


class Stats:
    def __init__(self):
        self.decode_misses = 0
        self.prefill_misses = 0
        self.decode_tokens = set()

    def miss(self, turn, phase, token):
        if phase == "decode":
            self.decode_misses += 1
        else:
            self.prefill_misses += 1

    def token(self, turn, phase, token):
        if phase == "decode":
            self.decode_tokens.add((turn, token))


def _arc_replace(t1, t2, b1, b2, p, in_b2):
    if t1 and (len(t1) > p or (in_b2 and len(t1) == p)):
        victim, _ = t1.popitem(last=False)
        b1[victim] = True
    elif t2:
        victim, _ = t2.popitem(last=False)
        b2[victim] = True
    elif t1:
        victim, _ = t1.popitem(last=False)
        b1[victim] = True


def simulate(rows, capacity, policy, pins, pin_token, options):
    stats = Stats()
    cache = OrderedDict()           # lru, lru-admit, probation for slru
    protected = OrderedDict()       # slru protected segment
    frequency = defaultdict(float)  # lfu scores and admission counts
    seen = defaultdict(int)
    locked = set()
    heap = []                       # lfu lazy heap (score, stamp, key)
    stamp = 0
    last_token = None
    decay = options.decay
    protected_capacity = int(capacity * options.protected)

    # Belady: next use index per row.
    next_use = None
    if policy == "opt":
        next_use = [math.inf] * len(rows)
        last_seen = {}
        for index in range(len(rows) - 1, -1, -1):
            key = rows[index][3]
            next_use[index] = last_seen.get(key, math.inf)
            last_seen[key] = index
        opt_next = {}
        opt_heap = []

    last_access, penultimate = {}, {}
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    arc = {"p": 0}
    current_turn = None
    for index, (turn, phase, token, key) in enumerate(rows):
        if turn != current_turn:
            current_turn = turn
            locked = set()
        if phase == "decode" and token == pin_token and not locked and pins.get(turn):
            locked = set(pins[turn])
            for pinned in locked:
                cache.pop(pinned, None)
                protected.pop(pinned, None)
                t1.pop(pinned, None)
                t2.pop(pinned, None)
        stats.token(turn, phase, token)
        if policy == "lfu" and phase == "decode" and (turn, token) != last_token:
            last_token = (turn, token)
            frequency_scale = decay
            # Ageing multiplies every score; apply lazily through a global
            # scale instead of touching every entry.
            options.scale = getattr(options, "scale", 1.0) / frequency_scale
        if key in locked:
            continue
        room = capacity - len(locked)
        if room <= 0:
            stats.miss(turn, phase, token)
            continue

        if policy == "lru":
            if key in cache:
                cache.move_to_end(key)
                continue
            stats.miss(turn, phase, token)
            cache[key] = True
            while len(cache) > room:
                cache.popitem(last=False)
        elif policy == "lru-admit":
            seen[key] += 1
            if key in cache:
                cache.move_to_end(key)
                continue
            stats.miss(turn, phase, token)
            if seen[key] >= options.admit:
                cache[key] = True
                while len(cache) > room:
                    cache.popitem(last=False)
        elif policy == "slru":
            if key in protected:
                protected.move_to_end(key)
                continue
            if key in cache:
                del cache[key]
                protected[key] = True
                while len(protected) > protected_capacity:
                    demoted, _ = protected.popitem(last=False)
                    cache[demoted] = True
                while len(cache) + len(protected) > room and cache:
                    cache.popitem(last=False)
                continue
            stats.miss(turn, phase, token)
            cache[key] = True
            while len(cache) + len(protected) > room and cache:
                cache.popitem(last=False)
        elif policy == "lfu":
            scale = getattr(options, "scale", 1.0)
            frequency[key] += scale
            stamp += 1
            if key in cache:
                heapq.heappush(heap, (frequency[key], stamp, key))
                continue
            stats.miss(turn, phase, token)
            cache[key] = True
            heapq.heappush(heap, (frequency[key], stamp, key))
            while len(cache) > room:
                score, _, victim = heapq.heappop(heap)
                if victim in cache and score == frequency[victim]:
                    del cache[victim]
        elif policy == "lru2":
            # LRU-K with K=2: evict the entry whose second-most-recent access
            # is oldest; entries seen once rank oldest of all.
            stamp += 1
            previous = last_access.get(key)
            last_access[key] = stamp
            penultimate[key] = previous if previous is not None else -1e18 + stamp
            if key in cache:
                heapq.heappush(heap, (penultimate[key], stamp, key))
                continue
            stats.miss(turn, phase, token)
            cache[key] = True
            heapq.heappush(heap, (penultimate[key], stamp, key))
            while len(cache) > room:
                score, _, victim = heapq.heappop(heap)
                if victim in cache and score == penultimate[victim]:
                    del cache[victim]
        elif policy == "arc":
            if key in t1 or key in t2:
                t1.pop(key, None)
                t2[key] = True
                t2.move_to_end(key)
                continue
            stats.miss(turn, phase, token)
            c = room
            if key in b1:
                arc["p"] = min(c, arc["p"] + max(len(b2) // max(1, len(b1)), 1))
                del b1[key]
                _arc_replace(t1, t2, b1, b2, arc["p"], key in b2)
                t2[key] = True
            elif key in b2:
                arc["p"] = max(0, arc["p"] - max(len(b1) // max(1, len(b2)), 1))
                del b2[key]
                _arc_replace(t1, t2, b1, b2, arc["p"], True)
                t2[key] = True
            else:
                if len(t1) + len(b1) >= c:
                    if len(t1) < c:
                        b1.popitem(last=False)
                        _arc_replace(t1, t2, b1, b2, arc["p"], False)
                    else:
                        t1.popitem(last=False)
                elif len(t1) + len(t2) + len(b1) + len(b2) >= c:
                    if len(t1) + len(t2) + len(b1) + len(b2) >= 2 * c and b2:
                        b2.popitem(last=False)
                    _arc_replace(t1, t2, b1, b2, arc["p"], False)
                t1[key] = True
        elif policy == "opt":
            upcoming = next_use[index]
            if key in cache:
                opt_next[key] = upcoming
                heapq.heappush(opt_heap, (-upcoming, key))
                continue
            stats.miss(turn, phase, token)
            if upcoming == math.inf:
                continue
            if len(cache) >= room:
                # Evict the entry used farthest in the future, or bypass.
                while opt_heap:
                    negative, victim = opt_heap[0]
                    if victim in cache and opt_next.get(victim) == -negative:
                        break
                    heapq.heappop(opt_heap)
                if opt_heap and -opt_heap[0][0] > upcoming:
                    _, victim = heapq.heappop(opt_heap)
                    del cache[victim]
                    opt_next.pop(victim, None)
                else:
                    continue
            cache[key] = True
            opt_next[key] = upcoming
            heapq.heappush(opt_heap, (-upcoming, key))
        else:
            raise ValueError(policy)
    options.scale = 1.0
    tokens = max(1, len(stats.decode_tokens))
    return stats.decode_misses / tokens, stats.prefill_misses


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--gb", nargs="+", type=float, default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--policies", nargs="+",
                        default=["lru", "slru", "lfu", "lru-admit", "opt"])
    parser.add_argument("--pin-token", type=int, default=9)
    parser.add_argument("--protected", type=float, default=0.8)
    parser.add_argument("--decay", type=float, default=0.995)
    parser.add_argument("--admit", type=int, default=2)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))
    from macqwen.results import output_path

    target = output_path("flashnext", "cache-sim", "cache-sim.json", args.json)

    trace = load(args.trace)
    record = int(trace["record_bytes"])
    rows = accesses(trace)
    pins = pinned_schedule(trace, args.pin_token)
    measured = []
    for info in trace["turns"]:
        values = [b for b in info["physical_bytes_per_token"] if b >= 0]
        if values:
            measured.append(sum(values) / len(values) / 1e6)
    print(f"record {record / 1e6:.3f} MB, {len(rows)} non-slab reads, "
          f"measured decode {', '.join(f'{m:.0f}' for m in measured)} MB/token")
    table = {}
    header = "GB   " + "".join(f"{p:>12s}" for p in args.policies)
    print(header)
    for gb in args.gb:
        capacity = int(gb * 1e9 // record)
        row = {}
        for policy in args.policies:
            misses, _prefill = simulate(rows, capacity, policy, pins, args.pin_token, args)
            row[policy] = round(misses * record / 1e6, 1)
        table[gb] = row
        print(f"{gb:<5g}" + "".join(f"{row[p]:12.1f}" for p in args.policies), flush=True)
    if True:
        with open(target, "w") as handle:
            json.dump({"measured_mb_per_token": measured, "mb_per_token": table,
                       "record_bytes": record, "trace": args.trace}, handle, indent=1)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
