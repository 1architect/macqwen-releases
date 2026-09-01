from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import mlx.core as mx

from models.flashnext.prefill import prefill_language
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

    def __call__(
        self,
        ids,
        cache,
        skip_logits=False,
        return_hidden=False,
    ):
        self.calls.append((int(ids.shape[1]), skip_logits, return_hidden))
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

        self.assertEqual(language.calls, [(3, False, False)])
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

        self.assertEqual(language.calls, [(10, True, True)])
        self.assertEqual(language.argmax_shapes, [(1, 1, 1)])
        self.assertIsNone(logits)
        self.assertEqual(int(token.item()), 10)
        self.assertEqual(int(cache[0].state.item()), 55)


if __name__ == "__main__":
    unittest.main()
