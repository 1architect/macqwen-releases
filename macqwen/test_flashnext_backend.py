from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

from macqwen.backends.base import GenerationCancelled
from macqwen.backends.flashnext import FlashNextBackend
from macqwen.sampling import Sampling


class _Routing:
    pinned_bytes = 0
    pinned_signature = ""

    def reset(self):
        pass

    def begin_decode(self):
        pass

    def after_token(self, _count, _limit):
        return False

    def finish_decode(self):
        pass


class _Tokenizer:
    def decode(self, _ids):
        return ""


class FlashNextBackendTests(unittest.TestCase):
    def backend(self):
        backend = FlashNextBackend.__new__(FlashNextBackend)
        backend.tape = []
        backend.pending = []
        backend.turn_closed = True
        backend._replay_needed = False
        backend._decoder = None
        backend.routing_profile = "exact-quality"
        backend.routing = _Routing()
        backend.language = SimpleNamespace(
            model=SimpleNamespace(layers=[]),
            _position_ids=None,
            _rope_deltas=None,
            make_cache=lambda: [SimpleNamespace(offset=0)],
        )
        backend.cache = [SimpleNamespace(offset=0)]
        backend.tokenizer = _Tokenizer()
        backend.stops = set()
        backend.thinking_enabled = False
        backend._interactive_budgets = None
        backend.sampling = Sampling.greedy_settings()
        return backend

    def test_manual_cancellation_keeps_cache_and_prefills_only_new_turn(self):
        backend = self.backend()
        backend.pending = [10]
        prompt_lengths = []
        checks = [0]

        def prefill(_language, ids, cache, sampler):
            del sampler
            prompt_lengths.append(int(ids.shape[1]))
            cache[0].offset += int(ids.shape[1])
            return None, mx.array([65], dtype=mx.uint32)

        def should_cancel():
            checks[0] += 1
            return checks[0] == 2

        with patch(
            "models.flashnext.prefill.prefill_language", prefill
        ), patch(
            "models.flashnext.expert_cache.set_prefill_progress", lambda _callback: None
        ):
            with self.assertRaises(GenerationCancelled):
                backend.generate(2, should_cancel=should_cancel)

            self.assertEqual(backend.tape, [10])
            self.assertFalse(backend._replay_needed)
            self.assertFalse(backend.turn_closed)
            self.assertTrue(backend.check_invariant())

            backend.pending = [20, 21]
            backend.generate(0, should_cancel=lambda: False)

        self.assertEqual(prompt_lengths, [1, 2])


if __name__ == "__main__":
    unittest.main()
