"""Offline cache simulator on synthetic traces. No model load."""
from __future__ import annotations

import random
import types
import unittest

from models.flashnext.tests.bench import cache_sim


def options(**values):
    base = dict(protected=0.8, decay=0.995, admit=2)
    base.update(values)
    return types.SimpleNamespace(**base)


def rows_from(sequence):
    return [(0, "decode", token, (0, expert)) for token, expert in enumerate(sequence)]


class CacheSimTests(unittest.TestCase):
    def test_lru_counts_cyclic_misses(self):
        rows = rows_from([1, 2, 3, 1, 2, 3])
        misses, _ = cache_sim.simulate(rows, 2, "lru", {}, 9, options())
        self.assertEqual(misses * 6, 6)
        misses, _ = cache_sim.simulate(rows, 3, "lru", {}, 9, options())
        self.assertEqual(misses * 6, 3)

    def test_opt_never_worse_than_other_policies(self):
        generator = random.Random(4)
        sequence = [int(generator.paretovariate(1.2)) % 40 for _ in range(3000)]
        rows = rows_from(sequence)
        opt, _ = cache_sim.simulate(rows, 12, "opt", {}, 9, options())
        for policy in ("lru", "slru", "lfu", "lru-admit", "lru2", "arc"):
            other, _ = cache_sim.simulate(rows, 12, policy, {}, 9, options())
            self.assertLessEqual(opt, other + 1e-12, policy)

    def test_locked_keys_are_hits(self):
        rows = rows_from([5, 5, 5, 5])
        pins = {0: {(0, 5)}}
        misses, _ = cache_sim.simulate(rows, 1, "lru", pins, 0, options())
        self.assertEqual(misses, 0)


if __name__ == "__main__":
    unittest.main()
