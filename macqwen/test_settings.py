"""What `/settings` shows about the next turn."""
from __future__ import annotations

import unittest


class DecodingInSettingsTests(unittest.TestCase):
    """`/settings` is the "what will the next turn do" view, so the decoding
    controls belong in it. The dedicated commands stay the writers."""

    def backend(self):
        from macqwen.backends.flashnext import FlashNextBackend
        from macqwen.sampling import Sampling

        backend = FlashNextBackend.__new__(FlashNextBackend)
        for key, value in dict(
            routing_profile="exact-quality", swap_epsilon=0.02, threshold=0.85,
            resident_experts=32, pin_budget_gb=6.0, tail_experts=0,
            tail_warmup=8, fusion_block=23, fusion_min_margin=1.0,
            fusion_min_block=4, fusion_margin_tokens=8, fusion_max_prompt=512,
            fusion_model="", model_path="", session_dir="~/x",
            thinking_enabled=True, sampling=Sampling(),
            reasoning_effort="high", think_budget=4096, answer_budget=4096,
        ).items():
            setattr(backend, key, value)
        return backend

    def test_the_decoding_controls_are_shown(self):
        text = self.backend()._settings_text()
        for label in ("sampling", "effort", "thinking", "token-budget"):
            self.assertIn(label, text)
        self.assertIn("top-p 0.95", text)
        self.assertIn("high", text)
        self.assertIn("4096 answer + 4096 reasoning", text)

    def test_greedy_is_named_not_shown_as_a_temperature(self):
        from macqwen.sampling import Sampling

        backend = self.backend()
        backend.sampling = Sampling.greedy_settings()
        self.assertIn("sampling            greedy", backend._settings_text())

    def test_a_shared_budget_says_so(self):
        backend = self.backend()
        backend.think_budget = 0
        self.assertIn("reasoning shares it", backend._settings_text())

    def test_the_writers_are_named(self):
        text = self.backend()._settings_text()
        self.assertIn("/sampling", text)
        self.assertIn("/effort", text)
