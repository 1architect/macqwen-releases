"""Native MTP decoder lifecycle: stale prevention without a model load."""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

from macqwen.backends.flashnext import FlashNextBackend, _resolve_startup_mtp
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
        backend.threshold = 0.85
        backend.resident_experts = 32
        backend.pin_budget_gb = 6.0
        backend.tail_experts = 6
        backend.tail_warmup = 8
        backend.swap_epsilon = 0.02
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


class FakeLanguage:
    """Minimal target language: greedy-friendly zero logits."""

    def __init__(self, with_mtp=True):
        self.model = SimpleNamespace(layers=[])
        self._position_ids = None
        self._rope_deltas = None
        if with_mtp:
            self.mtp = SimpleNamespace()

    def make_cache(self):
        return [SimpleNamespace(offset=0)]

    def __call__(self, _token, cache=None):
        return SimpleNamespace(logits=mx.zeros((1, 1, 16)))


def drive_target_turn(backend, max_tokens=1):
    """Run one target-only turn with stubbed prefill."""

    def prefill(_language, _ids, _cache, sampler):
        del sampler
        return None, mx.array([65], dtype=mx.uint32)

    with patch(
        "models.flashnext.prefill.prefill_language", prefill
    ), patch(
        "models.flashnext.expert_cache.set_prefill_progress",
        lambda _callback: None,
    ):
        return backend.generate(max_tokens, should_cancel=lambda: False)


class MTPHistoryInvariantTests(MTPDecoderLifecycleTests):
    def test_fresh_sampled_first_turn_blocks(self):
        backend = self.backend(greedy=False)
        backend.language = FakeLanguage()
        backend.pending = [10]
        drive_target_turn(backend)
        self.assertIsNone(backend._decoder)
        self.assertTrue(backend._mtp_blocked)
        backend.sampling = Sampling.greedy_settings()
        backend.pending = [11]
        with patch(
            "models.flashnext.speculative.MTPGreedy"
        ) as factory:
            drive_target_turn(backend)
        factory.assert_not_called()
        self.assertTrue(backend._mtp_blocked)

    def test_fresh_separate_budget_first_turn_blocks(self):
        backend = self.backend()
        backend.language = FakeLanguage()
        backend.thinking_enabled = True
        backend._interactive_budgets = (10, 5)
        backend.tokenizer = SimpleNamespace(decode=lambda _ids: "",
                                            encode=lambda _text, **_kw: [99])
        backend.pending = [10]
        drive_target_turn(backend)
        self.assertTrue(backend._mtp_blocked)
        backend._interactive_budgets = None
        backend.pending = [11]
        with patch(
            "models.flashnext.speculative.MTPGreedy"
        ) as factory:
            drive_target_turn(backend)
        factory.assert_not_called()
        self.assertTrue(backend._mtp_blocked)

    def test_zero_token_prefill_blocks(self):
        backend = self.backend(greedy=False)
        backend.language = FakeLanguage()
        backend.pending = [10]
        _text, stats = drive_target_turn(backend, max_tokens=0)
        self.assertEqual(stats.tokens, 0)
        self.assertTrue(backend._mtp_blocked)

    def test_full_replay_eligible_feeds_complete_history(self):
        backend = self.backend()
        backend.tape = [1, 2]
        backend.pending = [3]
        backend._replay_needed = True
        created = []

        def factory(language, depth=3):
            created.append(depth)
            return FakeMTPDecoder(backend.cache)

        with patch(
            "models.flashnext.speculative.MTPGreedy", factory
        ), patch.object(
            FlashNextBackend, "_is_native_mtp_decoder",
            lambda self, _decoder: True,
        ), patch(
            "models.flashnext.expert_cache.set_prefill_progress",
            lambda _callback: None,
        ):
            backend.generate(1, should_cancel=lambda: False)
        self.assertEqual(created, [3])
        live = backend._decoder
        self.assertEqual(live.appended, [[1, 2, 3]])
        self.assertFalse(backend._mtp_blocked)
        self.assertFalse(backend._replay_needed)

    def test_full_replay_ineligible_blocks(self):
        backend = self.backend(greedy=False)
        backend.language = FakeLanguage()
        backend.tape = [1, 2]
        backend.pending = [3]
        backend._replay_needed = True
        drive_target_turn(backend)
        self.assertTrue(backend._mtp_blocked)
        backend.sampling = Sampling.greedy_settings()
        backend.pending = [4]
        with patch(
            "models.flashnext.speculative.MTPGreedy"
        ) as factory:
            drive_target_turn(backend)
        factory.assert_not_called()

    def test_mtp_depth_updates_live_decoder_without_dropping(self):
        backend = self.backend()
        backend.language = FakeLanguage()
        backend.store = SimpleNamespace(
            set_residency_tracking=lambda _value: None
        )
        live = FakeMTPDecoder(backend.cache)
        live.depth = 3
        backend._decoder = live
        with patch.object(
            FlashNextBackend, "_is_native_mtp_decoder",
            lambda self, decoder: decoder is live,
        ), patch(
            "models.flashnext.routing.RoutingProfile",
            lambda *args, **kwargs: SimpleNamespace(),
        ):
            result = backend.configure("mtp-depth 5")
        self.assertIn("mtp-depth", result)
        self.assertEqual(backend.mtp_depth, 5)
        self.assertIs(backend._decoder, live)
        self.assertEqual(live.depth, 5)
        self.assertFalse(backend._mtp_blocked)

    def test_rebuild_drops_mtp_when_fused(self):
        backend = self.backend()
        backend.language = FakeLanguage()
        backend.store = SimpleNamespace(
            set_residency_tracking=lambda _value: None
        )
        live_cache = [SimpleNamespace(offset=5)]
        live = FakeMTPDecoder(live_cache)
        backend._decoder = live
        backend.routing_profile = "fused-quality"
        with patch.object(
            FlashNextBackend, "_is_native_mtp_decoder",
            lambda self, decoder: decoder is live,
        ), patch(
            "models.flashnext.routing.RoutingProfile",
            lambda *args, **kwargs: SimpleNamespace(),
        ):
            backend._rebuild_routing()
        self.assertIsNone(backend._decoder)
        self.assertIs(backend.cache, live_cache)
        self.assertTrue(backend._mtp_blocked)


class MTPStartupWiringTests(unittest.TestCase):
    def test_resolution_defaults_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_NATIVE_MTP", None)
            os.environ.pop("FLASHNEXT_MTP_DEPTH", None)
            self.assertEqual(_resolve_startup_mtp(None, None), ("off", 3))
    def test_environment_on(self):
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "on",
                                     "FLASHNEXT_MTP_DEPTH": "4"}):
            self.assertEqual(_resolve_startup_mtp(None, None), ("on", 4))

    def test_environment_off(self):
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "off"}):
            os.environ.pop("FLASHNEXT_MTP_DEPTH", None)
            self.assertEqual(_resolve_startup_mtp(None, None), ("off", 3))

    def test_explicit_beats_environment(self):
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "on"}):
            self.assertEqual(_resolve_startup_mtp("off", None)[0], "off")
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "off"}):
            self.assertEqual(_resolve_startup_mtp("on", None)[0], "on")

    def test_invalid_value_fails(self):
        with self.assertRaises(ValueError):
            _resolve_startup_mtp("maybe", None)

    def constructed(self, with_mtp_module=True, **kwargs):
        calls = {}
        language = SimpleNamespace(
            model=SimpleNamespace(layers=[]),
            make_cache=lambda: [],
            _position_ids=None,
            _rope_deltas=None,
        )
        if with_mtp_module:
            language.mtp = SimpleNamespace()
        model = SimpleNamespace(language_model=language)
        store = SimpleNamespace(set_residency_tracking=lambda _v: None)

        def fake_load(_path, **load_kwargs):
            calls.update(load_kwargs)
            return model, {}, store

        class FakeTokenizer:
            eos_token_id = 1

            @classmethod
            def from_pretrained(cls, _path):
                return cls()

            def convert_tokens_to_ids(self, _token):
                return 2

            def get_vocab(self):
                return {}

        with patch("macqwen.backends.flashnext.resolve_flashnext",
                   lambda _p: "/tmp/x"), patch(
            "models.flashnext.loader.load_streaming", fake_load), patch(
            "models.flashnext.qsa_chunk.apply", lambda: None), patch(
            "macqwen.backends.flashnext._load_transformers_tokenizer",
            lambda: FakeTokenizer), patch(
            "models.flashnext.routing.RoutingProfile",
            lambda *args, **kwargs: SimpleNamespace()), patch(
            "models.flashnext.routing.prewarm_enabled", lambda: False):
            backend = FlashNextBackend(**kwargs)
        return backend, calls

    def test_normal_path_env_on_reaches_loader(self):
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "on"}):
            backend, calls = self.constructed()
        self.assertEqual(backend.native_mtp, "on")
        self.assertTrue(calls.get("use_mtp"))
        self.assertTrue(backend._mtp_active)

    def test_normal_path_default_is_target_only(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_NATIVE_MTP", None)
            backend, calls = self.constructed()
        self.assertEqual(backend.native_mtp, "off")
        self.assertFalse(calls.get("use_mtp"))
        self.assertFalse(backend._mtp_active)

    def test_explicit_off_beats_env_on(self):
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "on"}):
            backend, calls = self.constructed(native_mtp="off")
        self.assertEqual(backend.native_mtp, "off")
        self.assertFalse(calls.get("use_mtp"))

    def test_cli_values_omits_mtp(self):
        from models.flashnext.settings import get_registry

        values = get_registry().cli_values(SimpleNamespace(), "flashnext")
        self.assertNotIn("native_mtp", values)
        self.assertNotIn("mtp_depth", values)

    def test_display_reflects_constructed_state(self):
        from models.flashnext.settings import get_registry

        registry = get_registry()
        native = registry.get("flashnext", "native-mtp")
        constructed = SimpleNamespace(native_mtp="on", _mtp_active=True,
                                      mtp_depth=3)
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "off"}):
            self.assertEqual(native.value(constructed), "on")
            self.assertTrue(native.is_active(constructed))
        target_only = SimpleNamespace(native_mtp="off", _mtp_active=False,
                                      mtp_depth=3)
        with patch.dict(os.environ, {"FLASHNEXT_NATIVE_MTP": "on"}):
            self.assertEqual(native.value(target_only), "off")
            self.assertFalse(native.is_active(target_only))


if __name__ == "__main__":
    unittest.main()
