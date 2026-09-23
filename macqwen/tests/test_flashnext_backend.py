from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

from macqwen.backends.base import GenerationCancelled
from macqwen.backends.flashnext import (
    FlashNextBackend,
    UNLIMITED_GENERATION_HORIZON,
)
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

    def _observe(self, *_args):
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
        backend._mtp_active = False
        backend._mtp_blocked = False
        backend.mtp_depth = 3
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

    def test_loaded_session_clears_pending_replay(self):
        backend = self.backend()
        backend.tape = [1, 2, 3]
        backend._replay_needed = True
        loaded = SimpleNamespace(
            cache=[SimpleNamespace(offset=5)], token_ids=[4, 5, 6, 7, 8],
            turn_closed=True, thinking=False, position_ids=None,
            rope_deltas=None,
        )
        sessions = SimpleNamespace(
            profile={"mode": "exact-quality"},
            saved_profile=lambda _name: {"mode": "exact-quality"},
            load=lambda _name: loaded,
        )
        backend._sessions = lambda: sessions
        backend._fused_pending = False

        backend.load_session("saved")

        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())


class _ScriptedLanguage:
    """Yield scripted token IDs, then a stop ID, through greedy argmax."""

    def __init__(self, script, stop):
        self.model = SimpleNamespace(layers=[])
        self._position_ids = None
        self._rope_deltas = None
        self.script = list(script)
        self.stop = stop
        self.position = 0

    def make_cache(self):
        return [SimpleNamespace(offset=0)]

    def __call__(self, _token, cache=None):
        if self.position < len(self.script):
            value = self.script[self.position]
        else:
            value = self.stop
        self.position += 1
        row = [0.0] * 16
        row[value] = 1.0
        return SimpleNamespace(logits=mx.array([[row]]))


class UnlimitedGenerationTests(unittest.TestCase):
    def backend(self, script, stop=9, budgets=None, thinking=False):
        backend = FlashNextBackend.__new__(FlashNextBackend)
        backend.tape = []
        backend.pending = [10]
        backend.turn_closed = True
        backend._replay_needed = False
        backend._decoder = None
        backend.routing_profile = "exact-quality"
        backend.routing = _Routing()
        backend.language = _ScriptedLanguage(script, stop)
        backend.cache = [SimpleNamespace(offset=0)]
        backend.tokenizer = _Tokenizer()
        backend.stops = {stop}
        backend.thinking_enabled = thinking
        backend._interactive_budgets = budgets
        backend.sampling = Sampling.greedy_settings()
        backend._mtp_active = False
        backend._mtp_blocked = False
        backend.mtp_depth = 3
        return backend

    def prefill(self, first):
        def run(_language, _ids, _cache, sampler):
            del sampler
            return None, mx.array([first], dtype=mx.uint32)

        return patch(
            "models.flashnext.prefill.prefill_language", run
        ), patch(
            "models.flashnext.expert_cache.set_prefill_progress",
            lambda _callback: None,
        )

    def test_unlimited_answer_stops_naturally(self):
        backend = self.backend([6], budgets=(-1, -1))
        first, second = self.prefill(5)
        with first, second:
            _text, stats = backend.generate(
                -1, should_cancel=lambda: False)
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(backend.tape, [10, 5, 6])
        self.assertEqual(stats.finish, "stop")

    def test_unlimited_without_budgets_stops_naturally(self):
        backend = self.backend([6])
        first, second = self.prefill(5)
        with first, second:
            _text, stats = backend.generate(
                -1, should_cancel=lambda: False)
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(stats.finish, "stop")

    def test_finite_limit_reports_length(self):
        backend = self.backend([6, 7, 8, 9], stop=15)
        first, second = self.prefill(5)
        with first, second:
            _text, stats = backend.generate(
                3, should_cancel=lambda: False)
        self.assertEqual(stats.tokens, 3)
        self.assertEqual(stats.finish, "length")

    def test_unlimited_reasoning_never_force_closes(self):
        # thinking on with think -1 must not call encode at all:
        # _Tokenizer has no encode, so any close attempt raises.
        backend = self.backend([6], budgets=(50, -1), thinking=True)
        first, second = self.prefill(5)
        with first, second:
            _text, stats = backend.generate(
                -1, should_cancel=lambda: False)
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(stats.finish, "stop")

    def test_finite_reasoning_force_closes_at_boundary(self):
        backend = self.backend([6, 7, 8], budgets=(50, 2),
                               thinking=True)
        backend.tokenizer = SimpleNamespace(decode=lambda _ids: "",
                                            encode=lambda _text, **_kw: [99])
        first, second = self.prefill(5)
        with first, second:
            _text, stats = backend.generate(
                5, should_cancel=lambda: False)
        self.assertEqual(backend.tape[:3], [10, 5, 99])

    def test_mtp_decoder_receives_positive_horizon(self):
        backend = self.backend([6])
        backend._mtp_active = True
        backend.language.mtp = SimpleNamespace()
        seen = {}

        class FakeMTP:
            def __init__(self, _language, depth=3):
                del depth
                self.target_cache = None

            def append(self, _ids):
                pass

            def set_route_observer(self, _observer):
                pass

            def generate(self, max_tokens, _stops):
                seen["horizon"] = max_tokens
                yield 7

        with patch(
            "models.flashnext.speculative.MTPGreedy", FakeMTP
        ), patch.object(
            FlashNextBackend, "_is_native_mtp_decoder",
            lambda _self, decoder: isinstance(decoder, FakeMTP),
        ):
            first, second = self.prefill(5)
            with first, second:
                _text, stats = backend.generate(
                    -1, should_cancel=lambda: False)
        self.assertEqual(seen["horizon"], UNLIMITED_GENERATION_HORIZON)
        self.assertEqual(stats.tokens, 1)
        self.assertEqual(stats.finish, "stop")


if __name__ == "__main__":
    unittest.main()
