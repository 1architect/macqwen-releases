"""Session save and load run on the generation thread (2026-09-23 bug)."""
from __future__ import annotations

import threading
import unittest

import mlx.core as mx

from macqwen import session as chat_session


class ModelThreadTests(unittest.TestCase):
    def test_lazy_array_from_generation_stream_evaluates_there(self):
        def make_lazy():
            stream = mx.new_stream(mx.gpu)
            return mx.add(mx.ones((4,)), 1, stream=stream)

        lazy = chat_session._GENERATION_EXECUTOR.submit(make_lazy).result()
        with self.assertRaises(RuntimeError):
            mx.eval(mx.add(lazy, 0))
        lazy = chat_session._GENERATION_EXECUTOR.submit(make_lazy).result()
        value = chat_session._on_model_thread(lambda: mx.sum(lazy).item())
        self.assertEqual(value, 8.0)

    def test_save_and_load_call_the_backend_on_the_generation_thread(self):
        seen = {}

        class Backend:
            tape, pending = [1], []

            def save_session(self, name):
                seen["save"] = threading.current_thread().name
                return "saved"

            def load_session(self, name):
                seen["load"] = threading.current_thread().name
                return "loaded"

        chat = object.__new__(chat_session.Session)
        chat.backend = Backend()
        self.assertEqual(chat.save_session("x"), "saved")
        self.assertEqual(chat.load_session("x"), "loaded")
        self.assertTrue(seen["save"].startswith("macqwen-generation"))
        self.assertTrue(seen["load"].startswith("macqwen-generation"))


if __name__ == "__main__":
    unittest.main()
