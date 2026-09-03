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


class NativeMoEExecutionTests(unittest.TestCase):
    def test_native_moe_strategies_match_and_record_gpu_time(self):
        try:
            import mlx.core as mx
        except ImportError:
            raise unittest.SkipTest("MLX required to generate test packs")

        from models.flashnext.metal_native import probe_native, run_native_moe, init_native_moe

        status = probe_native()
        if not status.available:
            raise unittest.SkipTest(status.reason)

        init_native_moe()

        hidden_size = 64
        inter_size = 32
        slots = 2
        expert_count = 2

        def make_pack(seed, out_w, in_w):
            v = ((mx.arange(expert_count * out_w * in_w, dtype=mx.float32) + seed) % 9 - 4) / 16
            return mx.quantize(
                v.reshape(expert_count, out_w, in_w).astype(mx.bfloat16),
                group_size=32,
                bits=4,
            )

        gate_pack = make_pack(0, inter_size, hidden_size)
        up_pack = make_pack(1, inter_size, hidden_size)
        down_pack = make_pack(2, hidden_size, inter_size)

        x = np.ones((1, hidden_size), dtype=np.float32)
        routes = np.array([[0, 1]], dtype=np.uint32)
        scores = np.array([[0.5, 0.5]], dtype=np.float32)

        results = {}
        for strategy in ("serial", "barrier", "fence"):
            out, gpu_ms = run_native_moe(
                x, routes, scores, gate_pack, up_pack, down_pack,
                strategy=strategy, expert_count=expert_count,
            )
            self.assertEqual(out.shape, (1, hidden_size))
            self.assertGreater(gpu_ms, 0.0)
            results[strategy] = out

        np.testing.assert_allclose(results["barrier"], results["serial"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(results["fence"], results["serial"], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

