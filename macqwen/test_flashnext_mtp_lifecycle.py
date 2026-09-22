"""Native MTP decoder lifecycle: stale prevention without a model load."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

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

    def _observe(self, *_args):
        pass

    def session_profile(self, _stops):
        return {}


class _Tokenizer:
    def decode(self, _ids):
        return ""


class FakeMTPDecoder:
    """Stand-in for a live native MTP decoder with separate draft state."""

    def __init__(self, target_cache):
        self.target_cache = target_cache
        self.appended = []
        self.generated = 0

    def append(self, ids):
        self.appended.append([int(v) for v in ids.reshape(-1).tolist()])

    def set_route_observer(self, _observer):
        pass

    def generate(self, max_tokens, _stops):
        self.generated += 1
        yield 7


class MTPDecoderLifecycleTests(unittest.TestCase):
    def backend(self, *, mtp=True, greedy=True, blocked=False):
        backend = FlashNextBackend.__new__(FlashNextBackend)
        backend.tape = []
        backend.pending = []
        backend.turn_closed = True
        backend._replay_needed = False
        backend._decoder = None
        backend.routing_profile = "exact-quality"
        backend.routing = _Routing()
        language = SimpleNamespace(
            model=SimpleNamespace(layers=[]),
            _position_ids=None,
            _rope_deltas=None,
            make_cache=lambda: [SimpleNamespace(offset=0)],
        )
        if mtp:
            language.mtp = SimpleNamespace()
        backend.language = language
        backend.cache = [SimpleNamespace(offset=0)]
        backend.tokenizer = _Tokenizer()
        backend.stops = set()
        backend.thinking_enabled = False
        backend._interactive_budgets = None
        backend.sampling = (
            Sampling.greedy_settings()
            if greedy
            else Sampling(temperature=0.5, top_p=0.9, top_k=20)
        )
        backend._mtp_active = bool(mtp)
        backend._mtp_blocked = bool(blocked)
        backend.mtp_depth = 3
        return backend

    def test_fresh_conversation_starts_mtp_when_eligible(self):
        backend = self.backend()
        backend.pending = [10]
        created = []

        class Factory:
            def __init__(self, language, depth):
                created.append(depth)
                self.decoder = FakeMTPDecoder(backend.cache)

            def __getattr__(self, name):
                return getattr(self.decoder, name)

        with patch(
            "models.flashnext.speculative.MTPGreedy", Factory
        ), patch.object(
            FlashNextBackend, "_is_native_mtp_decoder",
            lambda self, decoder: True,
        ), patch(
            "models.flashnext.expert_cache.set_prefill_progress",
            lambda _callback: None,
        ):
            _text, stats = backend.generate(1, should_cancel=lambda: False)
        self.assertEqual(created, [3])
        self.assertEqual(stats.tokens, 1)
        self.assertIsNotNone(backend._decoder)
        self.assertFalse(backend._mtp_blocked)

    def test_second_turn_and_tool_append_reuse_live_decoder(self):
        for first_tape in ([], [10, 7]):
            backend = self.backend()
            backend.tape = list(first_tape)
            live = FakeMTPDecoder(backend.cache)
            backend._decoder = live
            backend.pending = [20]
            with patch.object(
                FlashNextBackend, "_is_native_mtp_decoder",
                lambda self, decoder: decoder is live,
            ), patch(
                "models.flashnext.expert_cache.set_prefill_progress",
                lambda _callback: None,
            ):
                backend.generate(1, should_cancel=lambda: False)
            self.assertIs(backend._decoder, live)
            self.assertFalse(backend._mtp_blocked)
            self.assertEqual(live.generated, 1)

    def test_non_greedy_turn_drops_decoder_and_blocks(self):
        backend = self.backend(greedy=False)
        live_cache = [SimpleNamespace(offset=5)]
        live = FakeMTPDecoder(live_cache)
        backend._decoder = live
        backend.pending = [10]

        def prefill(_language, ids, cache, sampler):
            del sampler
            return None, mx.array([65], dtype=mx.uint32)

        # SimpleNamespace cannot take __call__; use a tiny stub instead.
        class FakeLanguage:
            def __init__(self):
                self.model = SimpleNamespace(layers=[])
                self._position_ids = None
                self._rope_deltas = None
                self.mtp = SimpleNamespace()

            def make_cache(self):
                return [SimpleNamespace(offset=0)]

            def __call__(self, _token, cache=None):
                return SimpleNamespace(logits=mx.zeros((1, 1, 16)))

        backend.language = FakeLanguage()
        with patch.object(
            FlashNextBackend, "_is_native_mtp_decoder",
            lambda self, decoder: decoder is live,
        ), patch(
            "models.flashnext.prefill.prefill_language", prefill
        ), patch(
            "models.flashnext.expert_cache.set_prefill_progress",
            lambda _callback: None,
        ):
            _text, stats = backend.generate(1, should_cancel=lambda: False)
        self.assertIsNone(backend._decoder)
        self.assertIs(backend.cache, live_cache)
        self.assertTrue(backend._mtp_blocked)
        self.assertEqual(stats.tokens, 1)

    def test_return_to_greedy_stays_blocked(self):
        backend = self.backend(blocked=True)
        backend.pending = [10]

        def prefill(_language, ids, cache, sampler):
            del sampler
            return None, mx.array([65], dtype=mx.uint32)

        class FakeLanguage:
            def __init__(self):
                self.model = SimpleNamespace(layers=[])
                self._position_ids = None
                self._rope_deltas = None
                self.mtp = SimpleNamespace()

            def make_cache(self):
                return [SimpleNamespace(offset=0)]

            def __call__(self, _token, cache=None):
                return SimpleNamespace(logits=mx.zeros((1, 1, 16)))

        backend.language = FakeLanguage()
        with patch(
            "models.flashnext.speculative.MTPGreedy"
        ) as factory, patch(
            "models.flashnext.prefill.prefill_language", prefill
        ), patch(
            "models.flashnext.expert_cache.set_prefill_progress",
            lambda _callback: None,
        ):
            backend.generate(1, should_cancel=lambda: False)
        factory.assert_not_called()
        self.assertTrue(backend._mtp_blocked)

    def test_replay_needed_keeps_mtp_available(self):
        # Cancellation with replay rebuilds target and MTP histories
        # together from the full tape, so no block is needed.
        backend = self.backend()
        backend._decoder = FakeMTPDecoder(backend.cache)
        backend.tape = [10, 7]
        backend._mark_replay_needed()
        self.assertIsNone(backend._decoder)
        self.assertFalse(backend._mtp_blocked)
        self.assertTrue(backend._replay_needed)

    def test_session_load_blocks_and_reset_clears(self):
        backend = self.backend()
        backend.model_path = "/tmp/fake"
        backend.session_dir = "/tmp/fake-sessions"
        loaded = SimpleNamespace(
            cache=[SimpleNamespace(offset=4)],
            token_ids=[1, 2, 3, 4],
            turn_closed=False,
            thinking=False,
            position_ids=None,
            rope_deltas=None,
        )

        class FakeStore:
            def __init__(self, *_args):
                self.profile = {}

            def saved_profile(self, _name):
                return {}

            def load(self, _name):
                return loaded

        with patch("models.flashnext.sessions.SessionStore", FakeStore):
            message = backend.load_session("demo")
        self.assertTrue(backend._mtp_blocked)
        self.assertIsNone(backend._decoder)
        self.assertIn("MTP disabled until reset", message)
        backend.reset()
        self.assertFalse(backend._mtp_blocked)

    def test_eligibility_gates(self):
        self.assertTrue(self.backend()._native_mtp_eligible())
        self.assertFalse(self.backend(greedy=False)._native_mtp_eligible())
        self.assertFalse(self.backend(mtp=False)._native_mtp_eligible())
        self.assertFalse(self.backend(blocked=True)._native_mtp_eligible())
        fused = self.backend()
        fused.routing_profile = "fused-quality"
        self.assertFalse(fused._native_mtp_eligible())


if __name__ == "__main__":
    unittest.main()
