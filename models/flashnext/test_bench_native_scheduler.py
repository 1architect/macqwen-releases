"""Checkpoint-free tests for the native scheduler benchmark helpers."""
from __future__ import annotations

import unittest

import numpy as np

from models.flashnext.bench_native_scheduler import (
    interleaved_order,
    parse_strategies,
    summarize,
    verify_output,
)


class NativeSchedulerParserTests(unittest.TestCase):
    def test_parse_default_strategies(self):
        self.assertEqual(parse_strategies("serial,barrier,fence"),
                         ("serial", "barrier", "fence"))

    def test_parse_strategies_deduplicates_values(self):
        self.assertEqual(parse_strategies("fence, barrier, fence"),
                         ("fence", "barrier"))

    def test_parse_strategies_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            parse_strategies("serial,unknown")


class NativeSchedulerStatsTests(unittest.TestCase):
    def test_interleaving_gives_each_arm_each_position(self):
        order = interleaved_order(("serial", "barrier", "fence"), arms=3, rounds=1)
        self.assertEqual(len(order), 9)
        self.assertEqual(order[:3], ["serial", "barrier", "fence"])
        self.assertEqual(order[3:6], ["barrier", "fence", "serial"])
        self.assertEqual(order[6:9], ["fence", "serial", "barrier"])

    def test_summary_reports_median_and_resolution_band(self):
        result = summarize((10.0, 11.0, 9.0))
        self.assertEqual(result.median_ms, 10.0)
        self.assertEqual(result.minimum_ms, 9.0)
        self.assertEqual(result.maximum_ms, 11.0)
        self.assertEqual(result.samples, 3)
        self.assertAlmostEqual(result.resolution_band_pct, 20.0)

    def test_verify_output_requires_exact_chain_result(self):
        inputs = np.array([0.0, 1.0], dtype=np.float32)
        verify_output(inputs, np.array([3.0, 4.0], dtype=np.float32), 3)
        with self.assertRaises(RuntimeError):
            verify_output(inputs, np.array([3.0, 5.0], dtype=np.float32), 3)

    def test_verify_output_requires_exact_cross_strategy_result(self):
        inputs = np.array([0.0, 1.0], dtype=np.float32)
        reference = np.array([3.0, 4.0], dtype=np.float32)
        close_but_different = reference.copy()
        close_but_different[0] = np.nextafter(
            close_but_different[0], np.float32(4.0)
        )
        with self.assertRaisesRegex(RuntimeError, "between strategy arms"):
            verify_output(inputs, close_but_different, 3, reference)


if __name__ == "__main__":
    unittest.main()
