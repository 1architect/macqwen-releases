"""Counting a routing opportunity must not overstate it.

The analysis decides whether cache-aware routing is worth building. If it
counts a swap that could not be taken, the mechanism gets built on a number
that was never there.
"""
from __future__ import annotations

import unittest

from models.flashnext.bench_route_swap import swaps_for


class SwapCountTests(unittest.TestCase):
    def test_a_close_resident_spare_is_a_swap(self):
        taken = swaps_for([(7, 0.021)], [(9, 0.019)], 0.005)
        self.assertEqual(len(taken), 1)
        self.assertAlmostEqual(taken[0], 0.002)

    def test_a_spare_below_the_epsilon_is_not(self):
        self.assertEqual(swaps_for([(7, 0.021)], [(9, 0.001)], 0.005), [])

    def test_one_spare_serves_one_pick(self):
        taken = swaps_for([(7, 0.021), (8, 0.020)], [(9, 0.019)], 0.02)
        self.assertEqual(len(taken), 1)

    def test_two_spares_serve_two_picks(self):
        taken = swaps_for(
            [(7, 0.021), (8, 0.020)], [(9, 0.019), (10, 0.018)], 0.02
        )
        self.assertEqual(len(taken), 2)

    def test_it_gives_up_the_least_mass_it_can(self):
        # both spares qualify; the stronger one must be chosen
        taken = swaps_for([(7, 0.030)], [(9, 0.029), (10, 0.011)], 0.05)
        self.assertAlmostEqual(taken[0], 0.001)

    def test_the_cheapest_pick_is_matched_first(self):
        # one spare, two candidates: the smaller loss must win
        taken = swaps_for([(7, 0.100), (8, 0.020)], [(9, 0.019)], 0.2)
        self.assertEqual(len(taken), 1)
        self.assertAlmostEqual(taken[0], 0.001)

    def test_nothing_resident_means_nothing_to_count(self):
        self.assertEqual(swaps_for([(7, 0.021)], [], 0.05), [])

    def test_no_cold_pick_means_nothing_to_count(self):
        self.assertEqual(swaps_for([], [(9, 0.019)], 0.05), [])


if __name__ == "__main__":
    unittest.main()
