"""Cache-aware routing must trade the least mass it can, and only when asked.

This is the first change here that alters what the model computes, so its
boundaries matter more than its speed: off by default, never taking an expert
the router scored materially higher, and never using one spare twice.
"""
from __future__ import annotations

import unittest

from models.flashnext.adaptive_topk import (
    _SWAP_EPSILON,
    _SWAP_RESIDENT,
    set_swap_resident,
    swap_row,
)


class SwapRowTests(unittest.TestCase):
    def resident(self, *experts):
        held = set(experts)
        return lambda expert: expert in held

    def test_a_cold_pick_takes_a_close_resident_spare(self):
        experts, weights = swap_row(
            [1, 7, 9], [0.5, 0.021, 0.019], 2, self.resident(1, 9), 0.02
        )
        self.assertEqual(experts, [1, 9, 7])
        self.assertAlmostEqual(weights[1], 0.019)

    def test_a_distant_spare_is_left_alone(self):
        experts, weights = swap_row(
            [1, 7, 9], [0.5, 0.021, 0.001], 2, self.resident(1, 9), 0.005
        )
        self.assertEqual(experts, [1, 7, 9])

    def test_nothing_moves_when_every_pick_is_resident(self):
        before = [1, 7, 9]
        experts, _weights = swap_row(
            before, [0.5, 0.021, 0.019], 2, self.resident(1, 7, 9), 0.02
        )
        self.assertIs(experts, before)

    def test_nothing_moves_when_no_spare_is_resident(self):
        before = [1, 7, 9]
        experts, _weights = swap_row(
            before, [0.5, 0.021, 0.019], 2, self.resident(1), 0.02
        )
        self.assertIs(experts, before)

    def test_one_spare_serves_one_pick(self):
        experts, _weights = swap_row(
            [7, 8, 9], [0.021, 0.020, 0.019], 2, self.resident(9), 0.02
        )
        self.assertEqual(experts.count(9), 1)
        self.assertEqual(sorted(experts), [7, 8, 9])

    def test_it_keeps_the_strongest_spare(self):
        experts, weights = swap_row(
            [7, 9, 10], [0.030, 0.029, 0.011], 1, self.resident(9, 10), 0.05
        )
        self.assertEqual(experts[0], 9)
        self.assertAlmostEqual(weights[0], 0.029)

    def test_the_expert_set_is_preserved(self):
        experts, weights = swap_row(
            [1, 7, 9, 10], [0.5, 0.021, 0.020, 0.019], 2,
            self.resident(9, 10), 0.02,
        )
        self.assertEqual(sorted(experts), [1, 7, 9, 10])
        self.assertEqual(sorted(round(w, 6) for w in weights),
                         sorted(round(w, 6) for w in [0.5, 0.021, 0.020, 0.019]))

    def test_a_weight_follows_its_expert(self):
        experts, weights = swap_row(
            [1, 7, 9], [0.5, 0.021, 0.019], 2, self.resident(1, 9), 0.02
        )
        pairs = dict(zip(experts, weights))
        self.assertAlmostEqual(pairs[9], 0.019)
        self.assertAlmostEqual(pairs[7], 0.021)


class SwitchTests(unittest.TestCase):
    def tearDown(self):
        set_swap_resident(None)

    def test_it_is_off_by_default(self):
        self.assertIsNone(_SWAP_RESIDENT[0])

    def test_setting_and_clearing(self):
        set_swap_resident(lambda layer, expert: True, 0.01)
        self.assertIsNotNone(_SWAP_RESIDENT[0])
        self.assertAlmostEqual(_SWAP_EPSILON[0], 0.01)
        set_swap_resident(None)
        self.assertIsNone(_SWAP_RESIDENT[0])

    def test_the_environment_default_is_off(self):
        import os

        from models.flashnext.routing import swap_enabled

        saved = os.environ.pop("FLASHNEXT_SWAP_RESIDENT", None)
        try:
            self.assertFalse(swap_enabled())
        finally:
            if saved is not None:
                os.environ["FLASHNEXT_SWAP_RESIDENT"] = saved


if __name__ == "__main__":
    unittest.main()
