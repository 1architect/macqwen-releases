from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.bonsai2.qmm_metadata import install, new_counters


class QmmMetadataTests(unittest.TestCase):
    def test_prepares_once_and_preserves_fp32_and_fp16_qmm_results(self):
        class Packed:
            def __init__(self, embedding=False):
                self.weight = mx.array(
                    np.arange(64 * 8, dtype=np.uint32).reshape(64, 8)
                )
                self.scales = mx.array(
                    np.linspace(0.2, 0.8, 64, dtype=np.float16)[:, None]
                )
                self.biases = mx.array(
                    np.linspace(-0.2, 0.2, 64, dtype=np.float16)[:, None]
                )
                self.embedding = embedding
                self.block = 0
                self.signs = None

            def __call__(self, x):
                return mx.quantized_matmul(
                    x, self.weight, self.scales, self.biases,
                    transpose=True, group_size=128, bits=2,
                )

        packed = Packed()
        embedding = Packed(embedding=True)
        model = types.SimpleNamespace(
            named_modules=lambda: [("projection", packed), ("embedding", embedding)]
        )
        runtime = types.ModuleType("runtime")
        runtime.Packed = Packed
        counters = new_counters()
        x32 = mx.array(np.arange(2 * 128, dtype=np.float32).reshape(2, 128) / 17)
        x16 = x32.astype(mx.float16)
        before32 = packed(x32)
        before16 = packed(x16)
        mx.eval(before32, before16)

        with patch.dict(sys.modules, {"runtime": runtime}):
            stats = install(model, True, counters)
            self.assertEqual(stats["prepared_modules"], 1)
            self.assertEqual(str(packed.scales.dtype), "mlx.core.float32")
            after32 = packed(x32)
            after16 = packed(x16)
            untouched = embedding(x32)
            mx.eval(after32, after16, untouched)
            np.testing.assert_array_equal(np.asarray(after32), np.asarray(before32))
            np.testing.assert_array_equal(np.asarray(after16), np.asarray(before16))
            self.assertEqual(counters["prepared_calls"], 2)
            self.assertEqual(counters["fp32_calls"], 1)
            self.assertEqual(counters["lower_precision_calls"], 1)
            self.assertEqual(counters["unsupported_calls"], 0)
            install(model, False, counters)

        self.assertEqual(str(packed.scales.dtype), "mlx.core.float32")

    def test_admission_plans_persistent_and_temporary_bytes_before_cast(self):
        class Packed:
            def __init__(self):
                self.weight = mx.zeros((64, 8), dtype=mx.uint32)
                self.scales = mx.ones((64, 1), dtype=mx.float16)
                self.biases = mx.zeros((64, 1), dtype=mx.float16)
                self.embedding = False
                self.block = 0
                self.signs = None

            def __call__(self, x):
                return x

        packed = Packed()
        model = types.SimpleNamespace(named_modules=lambda: [("projection", packed)])
        runtime = types.ModuleType("runtime")
        runtime.Packed = Packed
        counters = new_counters()
        events = []
        original_scales = packed.scales

        def admit(stats):
            events.append(("admission", str(packed.scales.dtype), packed.scales is original_scales))
            self.assertEqual(stats["projected_persistent_growth_bytes"], 256)
            self.assertEqual(stats["temporary_original_replacement_bytes"], 768)
            self.assertEqual(stats["current_group_temporary_coexistence_bytes"], 768)
            return False

        with patch.dict(sys.modules, {"runtime": runtime}), patch(
            "mlx.core.eval", side_effect=lambda *_values: events.append(("eval",))
        ):
            stats = install(model, True, counters, pressure_check=admit)
        try:
            self.assertEqual(events[0], ("admission", "mlx.core.float16", True))
            self.assertEqual(events[1], ("eval",))
            self.assertEqual(stats["projected_persistent_bytes"], 512)
            self.assertEqual(stats["prepared_bytes"], 512)
            self.assertEqual(stats["additional_bytes"], 256)
        finally:
            install(model, False, counters)

    def test_zero_persistent_budget_rejects_without_touching_modules(self):
        class Packed:
            def __init__(self):
                self.weight = mx.zeros((64, 8), dtype=mx.uint32)
                self.scales = mx.ones((64, 1), dtype=mx.float16)
                self.biases = mx.zeros((64, 1), dtype=mx.float16)
                self.embedding = False
                self.block = 0
                self.signs = None

            def __call__(self, x):
                return x

        packed = Packed()
        model = types.SimpleNamespace(named_modules=lambda: [("projection", packed)])
        runtime = types.ModuleType("runtime")
        runtime.Packed = Packed
        counters = new_counters()
        original_scales = packed.scales
        original_biases = packed.biases

        with patch.dict(sys.modules, {"runtime": runtime}), patch(
            "mlx.core.eval"
        ) as evaluate:
            stats = install(model, True, counters, max_additional_bytes=0)
        try:
            evaluate.assert_not_called()
            self.assertIs(packed.scales, original_scales)
            self.assertIs(packed.biases, original_biases)
            self.assertEqual(str(packed.scales.dtype), "mlx.core.float16")
            self.assertFalse(hasattr(packed, "_bonsai_qmm_prepared"))
            self.assertEqual(stats["admission_status"], "rejected")
            self.assertEqual(stats["prepared_modules"], 0)
            self.assertEqual(counters["groups_rejected"], 1)
            self.assertEqual(counters["rejection_reasons"], {"persistent_budget": 1})
            self.assertEqual(stats["failure"]["reason"], "persistent_budget")
        finally:
            install(model, False, counters)

    def test_temporary_budget_rejects_before_group_cast(self):
        class Packed:
            def __init__(self):
                self.weight = mx.zeros((64, 8), dtype=mx.uint32)
                self.scales = mx.ones((64, 1), dtype=mx.float16)
                self.biases = mx.zeros((64, 1), dtype=mx.float16)
                self.embedding = False
                self.block = 0
                self.signs = None

            def __call__(self, x):
                return x

        packed = Packed()
        model = types.SimpleNamespace(named_modules=lambda: [("projection", packed)])
        runtime = types.ModuleType("runtime")
        runtime.Packed = Packed
        counters = new_counters()
        original_scales, original_biases = packed.scales, packed.biases

        with patch.dict(sys.modules, {"runtime": runtime}), patch(
            "mlx.core.eval"
        ) as evaluate:
            stats = install(model, True, counters, max_temporary_bytes=767)
        try:
            evaluate.assert_not_called()
            self.assertIs(packed.scales, original_scales)
            self.assertIs(packed.biases, original_biases)
            self.assertEqual(stats["admission_status"], "rejected")
            self.assertEqual(stats["failure"]["reason"], "temporary_budget")
            self.assertEqual(counters["rejection_reasons"], {"temporary_budget": 1})
        finally:
            install(model, False, counters)

    def test_rejected_group_is_reported_and_left_unmodified(self):
        class Packed:
            def __init__(self):
                self.weight = mx.zeros((64, 8), dtype=mx.uint32)
                self.scales = mx.ones((64, 1), dtype=mx.float16)
                self.biases = mx.zeros((64, 1), dtype=mx.float16)
                self.embedding = False
                self.block = 0
                self.signs = None

            def __call__(self, x):
                return x

        first, second, third = Packed(), Packed(), Packed()
        model = types.SimpleNamespace(
            named_modules=lambda: [
                ("first", first), ("second", second), ("third", third)
            ]
        )
        runtime = types.ModuleType("runtime")
        runtime.Packed = Packed
        counters = new_counters()
        second_scales = second.scales
        third_scales = third.scales

        def reject_second(stats):
            return stats["current_group_index"] == 1

        with patch.dict(sys.modules, {"runtime": runtime}):
            stats = install(
                model,
                True,
                counters,
                group_size=1,
                pressure_check=reject_second,
            )
        try:
            self.assertEqual(str(first.scales.dtype), "mlx.core.float32")
            self.assertIs(second.scales, second_scales)
            self.assertIs(third.scales, third_scales)
            self.assertEqual(stats["admission_status"], "partial")
            self.assertEqual(counters["admission_calls"], 2)
            self.assertEqual(counters["groups_admitted"], 1)
            self.assertEqual(counters["groups_rejected"], 2)
            self.assertEqual(
                counters["admission_failures"][0]["reason"], "pressure_callback"
            )
            self.assertEqual(stats["rejected_groups"][0]["names"], ["second"])
            self.assertEqual(stats["rejected_groups"][1]["names"], ["third"])
        finally:
            install(model, False, counters)

    def test_backend_pressure_callback_records_real_memory_admission(self):
        from models.bonsai2.backend import _qmm_metadata_pressure_check

        stats = {
            "current_group_source_bytes": 100,
            "current_group_temporary_allocation_bytes": 400,
            "current_group_temporary_coexistence_bytes": 500,
            "projected_persistent_growth_bytes": 450,
        }
        with patch("models.bonsai2.backend._physical_memory_bytes", return_value=1000), patch(
            "models.bonsai2.backend._QMM_PRESSURE_HEADROOM_BYTES", 100
        ), patch("mlx.core.get_active_memory", return_value=400), patch(
            "mlx.core.get_cache_memory", return_value=100
        ):
            self.assertTrue(_qmm_metadata_pressure_check(stats))
        self.assertEqual(stats["pressure"]["current_bytes"], 500)
        self.assertEqual(stats["pressure"]["projected_peak_bytes"], 950)
        self.assertEqual(stats["pressure_rejection_reason"], "physical_memory_headroom")


if __name__ == "__main__":
    unittest.main()
