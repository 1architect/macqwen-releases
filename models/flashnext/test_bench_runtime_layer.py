"""Checkpoint-free tests for the one-layer benchmark adapter."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from models.flashnext.bench_runtime_layer import (
    _run_custom,
    _runner_from_target,
    comparison_stats,
    compare_outputs,
    custom_uses_native_path,
    expected_cold_bytes,
    make_cell,
    parse_misses,
    worst_validation,
)


class BenchRuntimeParsingTests(unittest.TestCase):
    def test_parse_default_miss_sweep(self):
        self.assertEqual(parse_misses("0,.25,.5,1.0"), (0.0, 0.25, 0.5, 1.0))

    def test_parse_misses_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_misses("0,1.1")

    def test_make_cell_uses_local_order_and_disjoint_cold_rows(self):
        cell = make_cell(0.25, 8)
        self.assertEqual(cell.hot, (0, 1, 2, 3, 4, 5))
        self.assertEqual(cell.cold, (8, 9))
        self.assertEqual(cell.route, cell.hot + cell.cold)
        self.assertEqual(len(set(cell.hot) & set(cell.cold)), 0)

    def test_make_cell_accepts_a_distant_cold_pool(self):
        cell = make_cell(0.25, 8, cold_base=504)
        self.assertEqual(cell.hot, (0, 1, 2, 3, 4, 5))
        self.assertEqual(cell.cold, (504, 505))
        self.assertEqual(cell.route, cell.hot + cell.cold)

    def test_expected_cold_bytes_counts_all_projection_parts(self):
        prefix = "layer.moe"
        refs = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                refs[f"{prefix}.{projection}.{part}"] = SimpleNamespace(
                    row_bytes=10
                )
        store = SimpleNamespace(refs=refs)
        self.assertEqual(expected_cold_bytes(store, prefix, 2), 180)


class BenchRuntimeComparisonTests(unittest.TestCase):
    def test_comparison_stats_reports_gain_and_resolution(self):
        gain, band = comparison_stats((1.0, 1.0, 1.0), (0.9, 0.9, 0.9))
        self.assertAlmostEqual(gain, 10.0)
        self.assertEqual(band, 0.0)

    def test_compare_outputs_reports_exact_and_tolerance(self):
        reference = np.array([[[1.0, 2.0]]], dtype=np.float32)
        exact = compare_outputs(reference, reference.copy(), 1e-4, 1e-4)
        self.assertTrue(exact["exact"])
        self.assertTrue(exact["within_tolerance"])

        close = compare_outputs(reference, np.array([[1.001, 2.0]]), 2e-3, 1e-3)
        self.assertTrue(close["within_tolerance"])
        self.assertFalse(close["exact"])

        wrong = compare_outputs(reference, np.zeros((1, 3)), 1e-4, 1e-4)
        self.assertFalse(wrong["shape_equal"])
        self.assertFalse(wrong["within_tolerance"])

    def test_worst_validation_does_not_hide_an_early_failure(self):
        passing = compare_outputs([1.0], [1.0], 1e-3, 1e-3)
        failing = compare_outputs([1.0], [1.1], 1e-3, 1e-3)
        self.assertIs(worst_validation((passing, failing)), failing)


class BenchRuntimeAdapterTests(unittest.TestCase):
    def test_reference_fallback_is_not_reported_as_native_custom_timing(self):
        self.assertTrue(custom_uses_native_path(SimpleNamespace()))
        self.assertTrue(custom_uses_native_path(
            SimpleNamespace(last_path="custom-metal")
        ))
        self.assertFalse(custom_uses_native_path(
            SimpleNamespace(last_path="reference")
        ))

    def test_metal_executor_gets_local_count_flat_input_and_one_weighted_output(self):
        class MetalMoEExecutor:
            instances = []

            def __init__(self, expert_count, hidden_size, top_k):
                self.init_args = (expert_count, hidden_size, top_k)
                self.calls = []
                type(self).instances.append(self)

            def execute(self, x, routes, projections, *, scores=None):
                self.calls.append((x, routes, projections))
                # One projected output per local route slot.
                return np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32)

        cell = make_cell(0.25, 2)
        packs = {"gate_proj": (1, 2, 3), "up_proj": (4, 5, 6), "down_proj": (7, 8, 9)}
        args = SimpleNamespace(
            group_size=32, bits=4, mode="affine", activation=None,
            width=2, hidden_size=2,
        )
        context = {
            "x": np.zeros((1, 1, 2), dtype=np.float32),
            "packs": packs,
            "indices": np.array([[[0, 1]]], dtype=np.int32),
            "scores": np.array([[[0.25, 0.75]]], dtype=np.float32),
            "routes": np.array([[0, 1]], dtype=np.int32),
            "expert_count": 2,
            "hidden_size": 2,
            "top_k": 2,
        }
        runner = _runner_from_target(MetalMoEExecutor, context)
        self.assertEqual(runner.init_args, (2, 2, 2))

        output = _run_custom(runner, context)
        np.testing.assert_allclose(output, np.array([[5.0, 7.0]], dtype=np.float32))
        seen_x, seen_routes, seen_projections = runner.calls[0]
        self.assertEqual(seen_x.shape, (1, 2))
        np.testing.assert_array_equal(seen_routes, np.array([[0, 1]], dtype=np.int32))
        self.assertIs(seen_projections, packs)


if __name__ == "__main__":
    unittest.main()
