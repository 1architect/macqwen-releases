"""What `/settings` shows about the next turn."""
from __future__ import annotations

import unittest
from unittest.mock import patch


class DecodingInSettingsTests(unittest.TestCase):
    """Shared decoding controls belong to Session, backend settings to registry."""

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

    def test_backend_settings_do_not_duplicate_shared_controls(self):
        text = self.backend()._settings_text()
        for label in ("sampling", "effort", "thinking", "token-budget"):
            self.assertNotIn(label, text)
        self.assertIn("routing", text)
        self.assertIn("threshold", text)

    def test_shared_status_owns_decoding_controls(self):
        from macqwen.session import Session
        session = Session.__new__(Session)
        session.backend = self.backend()
        session.backend.tape = []
        session.profile = "plain"
        session.preferences = {
            "model": "flashnext", "thinking_enabled": True,
            "show_thinking": False, "animate": True, "effort": "high",
            "temperature": 1.0, "max_tokens": -1, "think_budget": 4096,
        }
        session.tools = None
        with patch("macqwen.session.rss_gb", return_value=0.0):
            text = session.status()
        self.assertIn("effort=high", text)
        self.assertIn("think-tokens=4096", text)

    def test_model_settings_renders_shared_then_backend(self):
        class FakeBackend:
            def configure(self, argument):
                return "backend settings\n  public-value  1"

        session = object.__new__(__import__("macqwen.session", fromlist=["Session"]).Session)
        session.backend = FakeBackend()
        session.profile = "plain"
        session.preferences = {
            "thinking_enabled": False, "effort": "xhigh", "temperature": 1.0,
            "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "max_tokens": -1, "think_budget": -1,
        }
        text = session.model_settings("")
        self.assertLess(text.index("Chat settings"), text.index("backend settings"))
        self.assertIn("reasoning shares it", text)
