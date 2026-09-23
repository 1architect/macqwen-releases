"""Coalesced large-gather reads must fill the same slots as rows_into."""
from __future__ import annotations

import random
import tempfile
import unittest

import numpy as np

from models.flashnext.store import SafeTensorStore
from models.flashnext.tests.unit.test_store_read_modes import write_checkpoint


class CoalescedReadTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.expected = write_checkpoint(self._dir.name)
        self.store = SafeTensorStore(self._dir.name)
        self.name = "block.experts.weight"

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def check(self, experts, gap):
        self.store._coalesce_gap = gap
        out = self.store.empty_rows(self.name, len(experts))
        pairs = sorted((expert, slot) for slot, expert in enumerate(experts))
        self.store.rows_into_slots(self.name, pairs, out)
        np.testing.assert_array_equal(out, self.expected[experts])

    def test_adjacent_and_scattered_rows_keep_their_slots(self):
        for gap in (0, 1, 3):
            self.check([5, 0, 1, 2, 9, 7, 11], gap)
            self.check([3], gap)
            self.check(list(range(12))[::-1], gap)

    def test_random_orders(self):
        generator = random.Random(3)
        for _ in range(50):
            experts = generator.sample(range(12), generator.randint(1, 12))
            self.check(experts, generator.randint(0, 4))


if __name__ == "__main__":
    unittest.main()
