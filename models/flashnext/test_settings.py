from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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

    def settings_backend(self, profile):
        backend = self.make_backend()
        backend.routing_profile = profile
        effective = 0.2 if profile == "fast" else backend.threshold
        backend.routing = SimpleNamespace(
            mode=profile,
            threshold=effective,
            quality=profile in (
                "fast-quality", "exact-quality", "cache-aware", "fused-quality"
            ),
            cache_aware=profile == "cache-aware",
        )
        return backend

    @staticmethod
    def settings_line(text, name):
        return next(
            line for line in text.splitlines() if f"  {name} " in line
        )

    def test_irrelevant_settings_are_dimmed_and_active_settings_are_normal(self):
        cases = (
            ("exact-quality", "resident-experts", "swap-epsilon"),
            ("cache-aware", "swap-epsilon", "tail-experts"),
            ("fast-quality", "tail-experts", "fusion-block"),
            ("fused-quality", "fusion-block", "tail-experts"),
        )
        for profile, active, inactive in cases:
            with self.subTest(profile=profile):
                text = self.settings_backend(profile)._settings_text()
                active_line = self.settings_line(text, active)
                inactive_line = self.settings_line(text, inactive)
                self.assertNotIn("\033[2m", active_line)
                self.assertNotIn("\033[0m", active_line)
                self.assertTrue(inactive_line.startswith("\033[2m"))
                self.assertTrue(inactive_line.endswith("\033[0m"))

    def test_fast_displays_effective_and_configured_thresholds(self):
        backend = self.settings_backend("fast")
        backend.threshold = 0.85
        line = self.settings_line(backend._settings_text(), "threshold")
        self.assertTrue(line.startswith("\033[2m"))
        self.assertTrue(line.endswith("\033[0m"))
        self.assertIn("0.2", line)
        self.assertIn("0.85", line)

    def test_fast_quality_displays_threshold_transition(self):
        backend = self.settings_backend("fast-quality")
        backend.threshold = 0.85
        line = self.settings_line(backend._settings_text(), "threshold")
        self.assertIn("0.85", line)
        self.assertIn("0.2", line)
        self.assertIn("warmup", line)
        self.assertIn("tail threshold 0.2", line)

    def test_global_swap_override_is_active_and_displayed(self):
        backend = self.settings_backend("exact-quality")
        with patch.dict(os.environ, {
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.37",
        }):
            line = self.settings_line(backend._settings_text(), "swap-epsilon")
        self.assertNotIn("\033[2m", line)
        self.assertNotIn("\033[0m", line)
        self.assertIn("0.37", line)

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
