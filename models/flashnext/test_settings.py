from __future__ import annotations

import unittest

from macqwen.backends.flashnext import FlashNextBackend
from macqwen.model_settings import FLASHNEXT_DEFAULTS


class SettingsTests(unittest.TestCase):
    def make_backend(self):
        backend = FlashNextBackend.__new__(FlashNextBackend)
        backend.routing_profile = "exact-quality"
        backend.swap_epsilon = 0.02
        backend.threshold = 0.85
        backend.resident_experts = 32
        backend.pin_budget_gb = 6.0
        backend.tail_experts = 6
        backend.tail_warmup = 8
        backend.fusion_block = 23
        backend.fusion_min_margin = 1.0
        backend.fusion_min_block = 20
        backend.fusion_margin_tokens = 8
        backend.fusion_max_prompt = 512
        backend.fusion_model = "/models/some-draft-model"
        backend.tape = []
        backend._rebuild_routing = lambda: None
        return backend

    def test_lists_modes_and_advanced_values(self):
        text = self.make_backend().configure("")
        self.assertIn("exact-quality", text)
        self.assertIn("fused-quality", text)
        self.assertIn("cache-aware", text)
        self.assertIn("swap-epsilon", text)
        self.assertIn("resident-experts", text)
        self.assertIn("pinned-experts", text)
        self.assertIn("pin-budget-gb", text)
        self.assertIn("fusion-min-margin", text)

    def test_selects_experimental_fused_mode_with_warning(self):
        backend = self.make_backend()
        result = backend.configure("routing fused-quality")
        self.assertEqual(backend.routing_profile, "fused-quality")
        self.assertIn("experimental", result)

    def test_changes_numeric_settings(self):
        backend = self.make_backend()
        backend.configure("threshold 1.0")
        backend.configure("resident-experts 48")
        backend.configure("pin-budget-gb 5.5")
        backend.configure("swap-epsilon 0.005")
        self.assertEqual(backend.threshold, 1.0)
        self.assertEqual(backend.resident_experts, 48)
        self.assertEqual(backend.pin_budget_gb, 5.5)
        self.assertEqual(backend.swap_epsilon, 0.005)

    def test_selects_cache_aware_mode_with_quality_warning(self):
        backend = self.make_backend()
        result = backend.configure("routing cache-aware")
        self.assertEqual(backend.routing_profile, "cache-aware")
        self.assertIn("exact-quality gave better answers", result)

    def test_pinned_experts_is_a_resident_experts_alias(self):
        backend = self.make_backend()
        backend.configure("pinned-experts 24")
        self.assertEqual(backend.resident_experts, 24)

    def test_rejects_invalid_values_without_mutation(self):
        backend = self.make_backend()
        with self.assertRaises(ValueError):
            backend.configure("fusion-min-block 24")
        self.assertEqual(backend.fusion_min_block, 20)
        with self.assertRaises(ValueError):
            backend.configure("threshold 2")
        self.assertEqual(backend.threshold, 0.85)

    def test_restores_defaults(self):
        backend = self.make_backend()
        backend.routing_profile = "fast"
        backend.swap_epsilon = 0.5
        backend.threshold = 0.2
        backend.configure("defaults")
        self.assertEqual(backend.routing_profile, FLASHNEXT_DEFAULTS["routing"])
        self.assertEqual(backend.threshold, FLASHNEXT_DEFAULTS["threshold"])
        self.assertEqual(
            backend.swap_epsilon, FLASHNEXT_DEFAULTS["swap_epsilon"]
        )
        self.assertEqual(
            backend.resident_experts,
            FLASHNEXT_DEFAULTS["resident_experts"],
        )
        self.assertEqual(
            backend.pin_budget_gb,
            FLASHNEXT_DEFAULTS["pin_budget_gb"],
        )


if __name__ == "__main__":
    unittest.main()
