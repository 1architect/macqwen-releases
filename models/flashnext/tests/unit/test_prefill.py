from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import mlx.core as mx

from models.flashnext.prefill import prefill_language, prefill_target
from models.flashnext.expert_cache import (
    StreamingSwitchGLU,
    set_prefill_progress,
)


class _Cache:
    def __init__(self):
        self.state = mx.array([0], dtype=mx.int32)


class _Output:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class _Language:
    def __init__(self):
        self.calls = []
        self.argmax_shapes = []
        self.args = SimpleNamespace(hidden_size=1, hc_count=1)

    def __call__(
        self,
        ids,
        cache,
        skip_logits=False,
        return_hidden=False,
        capture_layer_ids=None,
    ):
        self.calls.append(
            (int(ids.shape[1]), skip_logits, return_hidden, capture_layer_ids)
        )
        cache[0].state = cache[0].state + ids.sum()
        logits = (
            None
            if skip_logits
            else (mx.arange(32) == ids[..., None]).astype(mx.float32)
        )
        hidden = [ids[..., None]] if return_hidden else None
        return _Output(logits, hidden)

    @staticmethod
    def speculative_logits_from_hidden(hidden):
        return hidden

    def speculative_argmax_from_hidden(self, hidden):
        self.argmax_shapes.append(tuple(hidden.shape))
        return hidden[..., 0]


class PrefillTests(unittest.TestCase):
    def test_streamed_layer_reports_progress_after_its_input_is_ready(self):
        block = StreamingSwitchGLU.__new__(StreamingSwitchGLU)
        object.__setattr__(block, "layer_id", 7)
        for name in ("gate_proj", "up_proj", "down_proj"):
            object.__setattr__(block, name, SimpleNamespace(slab=None))
        object.__setattr__(block, "_one_pass", lambda *_args, **_kwargs: "output")
        seen = []
        set_prefill_progress(seen.append)
        try:
            output = block(mx.array([[1.0]]), mx.array([0]))
        finally:
            set_prefill_progress(None)
        self.assertEqual(output, "output")
        self.assertEqual(seen, [7])

    def test_short_prompt_keeps_original_single_call(self):
        language = _Language()
        cache = [_Cache()]
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        logits, token = prefill_language(language, ids, cache)

        self.assertEqual(language.calls, [(3, False, False, None)])
        self.assertIsNone(logits)
        self.assertEqual(int(token.item()), 3)

    def test_large_logits_are_released_before_decode(self):
        language = _Language()
        cache = [_Cache()]
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        with (
            patch("models.flashnext.prefill.PREFILL_RELEASE_BYTES", 1),
            patch("models.flashnext.prefill.mx.clear_cache") as clear_cache,
        ):
            _, token = prefill_language(language, ids, cache)

        self.assertEqual(int(token.item()), 3)
        clear_cache.assert_called_once_with()

    def test_large_prefill_keeps_one_full_moe_call(self):
        language = _Language()
        cache = [_Cache()]
        ids = mx.arange(1, 11, dtype=mx.int32)[None]

        with patch("models.flashnext.prefill.QSA_CHUNK_THRESHOLD", 4):
            logits, token = prefill_language(language, ids, cache)

        self.assertEqual(language.calls, [(10, True, True, None)])
        self.assertEqual(language.argmax_shapes, [(1, 1, 1)])
        self.assertIsNone(logits)
        self.assertEqual(int(token.item()), 10)
        self.assertEqual(int(cache[0].state.item()), 55)

    def test_target_contract_can_request_logits_and_hidden_states(self):
        language = _Language()
        cache = [_Cache()]
        ids = mx.arange(1, 11, dtype=mx.int32)[None]

        with (
            patch("models.flashnext.prefill.QSA_CHUNK_THRESHOLD", 4),
            patch("models.flashnext.prefill.mx.clear_cache") as clear_cache,
        ):
            logits, hidden, token = prefill_target(
                language, ids, cache, want_logits=True, want_hidden=True
            )

        self.assertEqual(language.calls, [(10, False, True, [])])
        self.assertEqual(tuple(logits.shape), (1, 10, 32))
        self.assertEqual(tuple(hidden[0].shape), (1, 10, 1))
        self.assertEqual(int(token.item()), 10)
        clear_cache.assert_called_once_with()

    def test_fast_draft_target_prefill_uses_shared_contract(self):
        from models.flashnext.speculative import FastDraftGreedy

        language = _Language()
        cache = [_Cache()]
        decoder = FastDraftGreedy.__new__(FastDraftGreedy)
        decoder.language = language
        decoder.target_cache = cache
        decoder.external_draft = False
        decoder.draft_disabled = True
        decoder._exact_profile = lambda: None
        ids = mx.arange(1, 11, dtype=mx.int32)[None]

        with (
            patch("models.flashnext.speculative.QSA_CHUNK_THRESHOLD", 4),
            patch("models.flashnext.prefill.QSA_CHUNK_THRESHOLD", 4),
            patch(
                "models.flashnext.speculative.prefill_target",
                wraps=prefill_target,
            ) as shared,
        ):
            decoder.append(ids)

        shared.assert_called_once_with(language, ids, cache)
        self.assertEqual(language.calls, [(10, True, True, None)])
        self.assertEqual(int(decoder.next_main.item()), 10)

    def test_mtp_target_capture_uses_shared_contract(self):
        from models.flashnext.speculative import MTPGreedy

        language = _Language()
        cache = [_Cache()]
        decoder = MTPGreedy.__new__(MTPGreedy)
        decoder.language = language
        decoder.target_cache = cache
        ids = mx.arange(1, 11, dtype=mx.int32)[None]

        with (
            patch("models.flashnext.prefill.QSA_CHUNK_THRESHOLD", 4),
            patch(
                "models.flashnext.speculative.prefill_target",
                wraps=prefill_target,
            ) as shared,
        ):
            logits, hidden = decoder._target_capture(ids)

        shared.assert_called_once_with(
            language, ids, cache, want_logits=True, want_hidden=True
        )
        self.assertEqual(language.calls, [(10, False, True, [])])
        self.assertEqual(tuple(logits.shape), (1, 10, 32))
        self.assertEqual(tuple(hidden.shape), (1, 10, 1))


if __name__ == "__main__":
    unittest.main()
