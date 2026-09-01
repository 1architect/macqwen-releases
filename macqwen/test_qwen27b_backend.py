from __future__ import annotations

from types import SimpleNamespace
import unittest

from macqwen.backends.frankenstein import FrankensteinBackend


class FakeEngine:
    tape = [1, 2]
    pending = [3]
    cache = []
    cache_tokens = 2

    def generate(self, **kwargs):
        kwargs["progress"](1, 1)
        kwargs["on_token"](1, SimpleNamespace(text="answer"))
        stats = SimpleNamespace(
            finish="stop",
            gen_tokens=2,
            gen_tps=4.0,
            new_prompt_tokens=3,
            prompt_tps=6.0,
            host_free_gb=5.0,
            swap_gb=1.0,
        )
        return "answer", stats

    def check_invariant(self):
        return True


class BackendTests(unittest.TestCase):
    def test_settings_list_startup_values_and_reject_live_changes(self):
        backend = FrankensteinBackend.__new__(FrankensteinBackend)
        backend._startup_settings = {"prefill-step-size": 256, "kv-bits": 4}
        self.assertIn("prefill-step-size", backend.configure(""))
        with self.assertRaises(ValueError):
            backend.configure("prefill-step-size 512")

    def setUp(self):
        self.backend = FrankensteinBackend.__new__(FrankensteinBackend)
        self.backend.engine = FakeEngine()

    def test_generate_adapts_stats_and_streams(self):
        pieces = []
        prefills = []
        progress = []
        text, stats = self.backend.generate(
            20,
            out=pieces.append,
            on_prefilled=lambda: prefills.append(True),
            on_prefill_progress=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(text, "answer")
        self.assertEqual(pieces, ["answer"])
        self.assertEqual(prefills, [True])
        self.assertEqual(progress, [(1, 1)])
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(stats.rate, 4.0)
        self.assertEqual(stats.prompt_tokens, 3)
        self.assertEqual(stats.prompt_rate, 6.0)

    def test_session_names_cannot_escape_the_session_directory(self):
        from pathlib import Path

        self.backend._session_dir = Path("/sessions")
        for name in ("../outside", "two words", "", "/absolute"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.backend._path(name)


if __name__ == "__main__":
    unittest.main()
