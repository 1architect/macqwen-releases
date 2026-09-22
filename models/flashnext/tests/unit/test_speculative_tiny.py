#!/usr/bin/env python3
"""Small exactness test for Qwen4 block verification and cache rollback."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx_vlm.models.qwen3_5 import language as base_language
from mlx_vlm.models.qwen4_exp.config import ModelConfig, VisionConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel, TextConfig

from models.flashnext.adaptive_topk import (
    apply as patch_topk,
    set_renorm_blend,
    set_threshold,
)
from models.flashnext.patch_rmsnorm import apply as patch_norm
from models.flashnext.prefill import prefill_target
from models.flashnext.qwen4_verifier import Qwen4ExactSpeculativeVerifier
from models.flashnext.speculative import (
    FastDraftGreedy,
    snapshot_cache,
)


def tiny_language() -> LanguageModel:
    text = TextConfig(
        model_type="qwen4_exp",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=2,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=1,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=2,
    )
    config = ModelConfig(
        text_config=text,
        vision_config=VisionConfig(),
        model_type="qwen4_exp",
        vocab_size=64,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = LanguageModel(text, config)
    model.eval()
    return model


class SpeculativeTinyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_norm()
        patch_topk()
        cls.language = tiny_language()
        base_language._EXACT_SPECULATIVE_VERIFIER = Qwen4ExactSpeculativeVerifier()
        set_threshold(1.0)
        set_renorm_blend(1.0)

    def prefill(self):
        cache = self.language.make_cache()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        mx.eval(self.language(prompt, cache=cache).logits)
        return cache

    def test_block_and_rollback_match_single_tokens(self):
        values = [5, 9, 13, 17, 21]
        reference_cache = self.prefill()
        reference_logits = []
        for value in values:
            token = mx.array([[value]], dtype=mx.uint32)
            output = self.language(token, cache=reference_cache)
            mx.eval(output.logits)
            reference_logits.append(output.logits)
        reference_logits = mx.concatenate(reference_logits, axis=1)

        verify_cache = self.prefill()
        snapshot = snapshot_cache(verify_cache)
        output = self.language(
            mx.array([values], dtype=mx.uint32),
            cache=verify_cache,
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
        )
        block_logits = self.language.speculative_logits_from_hidden(
            output.hidden_states[-1]
        )
        mx.eval(block_logits)
        self.assertEqual(
            float(mx.max(mx.abs(reference_logits - block_logits)).item()),
            0.0,
        )

        for keep_count in (1, 2):
            verify_cache = self.prefill()
            snapshot = snapshot_cache(verify_cache)
            output = self.language(
                mx.array([values], dtype=mx.uint32),
                cache=verify_cache,
                speculative_verify=True,
                return_hidden=True,
                skip_logits=True,
            )
            decoder = FastDraftGreedy(self.language, object(), depth=2)
            decoder.target_cache = verify_cache
            decoder._rollback_verified(
                snapshot, output, len(values), keep_count
            )
            probe = mx.array([[29]], dtype=mx.uint32)
            actual = self.language(probe, cache=verify_cache).logits

            expected_cache = self.prefill()
            for value in values[:keep_count]:
                token = mx.array([[value]], dtype=mx.uint32)
                mx.eval(self.language(token, cache=expected_cache).logits)
            expected = self.language(probe, cache=expected_cache).logits
            mx.eval(actual, expected)
            self.assertEqual(
                float(mx.max(mx.abs(expected - actual)).item()), 0.0
            )

    def assert_external_draft_exact(
        self,
        fallback_on_reject,
        release_draft_before_verify=False,
        draft_min_margin=None,
        draft_min_block=1,
        draft_margin_tokens=None,
    ):
        target = tiny_language()
        draft = tiny_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        cache = target.make_cache()
        output = target(prompt, cache=cache)
        token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
        expected = []
        for _ in range(16):
            expected.append(int(token.item()))
            output = target(token.reshape(1, 1), cache=cache)
            token = mx.argmax(
                output.logits[:, -1, :], axis=-1
            ).astype(mx.uint32)
            mx.eval(token)

        target._position_ids = None
        target._rope_deltas = None
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            depth=2,
            draft_language=draft,
            fallback_on_reject=fallback_on_reject,
            release_draft_before_verify=release_draft_before_verify,
            draft_min_margin=draft_min_margin,
            draft_min_block=draft_min_block,
            draft_margin_tokens=draft_margin_tokens,
        )
        decoder.append(prompt)
        actual = list(decoder.generate(16, set()))
        self.assertEqual(actual, expected)
        return decoder

    def test_external_draft_anchor_preserves_greedy_tokens(self):
        self.assert_external_draft_exact(False)

    def test_external_draft_fallback_preserves_greedy_tokens(self):
        self.assert_external_draft_exact(True)

    def test_transient_external_draft_preserves_greedy_tokens(self):
        decoder = self.assert_external_draft_exact(False, True)
        self.assertTrue(decoder.draft_disabled)
        self.assertIsNone(decoder.draft_language)
        self.assertIsNone(decoder.draft_cache)
        self.assertIsNone(decoder.draft_next)

    def test_transient_draft_releases_on_initial_stop(self):
        target = tiny_language()
        draft = tiny_language()
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            depth=2,
            draft_language=draft,
            release_draft_before_verify=True,
        )
        decoder.append(mx.array([[3, 7, 11]], dtype=mx.uint32))
        stop = int(decoder.next_main.item())

        self.assertEqual(list(decoder.generate(16, {stop})), [])
        self.assertTrue(decoder.draft_disabled)
        self.assertIsNone(decoder.draft_language)
        self.assertIsNone(decoder.draft_cache)
        self.assertIsNone(decoder.draft_next)

    def test_speculative_stop_keeps_cache_for_tool_append(self):
        target = tiny_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)

        decoder = FastDraftGreedy(target, SimpleNamespace(_read_mode="pread"), depth=2)
        decoder.append(prompt)
        stop = int(decoder.next_main.item())
        self.assertEqual(list(decoder.generate(16, {stop})), [])

        decoder.append(mx.array([[4]], dtype=mx.uint32))
        actual = int(decoder.next_main.item())

        expected_language = tiny_language()
        expected_cache = expected_language.make_cache()
        expected = prefill_target(
            expected_language,
            mx.concatenate([prompt, mx.array([[4]], dtype=mx.uint32)], axis=1),
            expected_cache,
        )
        expected_next = int(expected[2].item())
        self.assertEqual(actual, expected_next)

    def test_transient_release_is_idempotent(self):
        target = tiny_language()
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            draft_language=tiny_language(),
            release_draft_before_verify=True,
        )

        decoder._release_transient_draft()
        elapsed = decoder.stats.release_seconds
        decoder._release_transient_draft()

        self.assertEqual(decoder.stats.release_seconds, elapsed)

    def test_transient_draft_releases_after_generation_error(self):
        decoder = FastDraftGreedy(
            tiny_language(),
            SimpleNamespace(_read_mode="pread"),
            draft_language=tiny_language(),
            release_draft_before_verify=True,
        )

        with patch.object(
            decoder,
            "_generate_external",
            side_effect=RuntimeError("draft failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                list(decoder.generate(16, set()))

        self.assertTrue(decoder.draft_disabled)
        self.assertIsNone(decoder.draft_language)
        self.assertIsNone(decoder.draft_cache)
        self.assertIsNone(decoder.draft_next)

    def test_confidence_abort_preserves_greedy_tokens(self):
        self.assert_external_draft_exact(False, True, 1e9, 2)

    def test_long_prefill_disables_external_draft_and_stays_exact(self):
        target = tiny_language()
        draft = tiny_language()
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        reference_cache = target.make_cache()
        reference = target(prompt, cache=reference_cache)
        expected = int(mx.argmax(reference.logits[:, -1, :], axis=-1).item())

        target._position_ids = None
        target._rope_deltas = None
        decoder = FastDraftGreedy(
            target,
            SimpleNamespace(_read_mode="pread"),
            depth=2,
            draft_language=draft,
            release_draft_before_verify=True,
        )
        with patch("models.flashnext.speculative.QSA_CHUNK_THRESHOLD", 2):
            decoder.append(prompt)

        self.assertTrue(decoder.draft_disabled)
        self.assertEqual(int(decoder.next_main.item()), expected)

    def test_exact_verifier_exposes_pre_mixer_residual(self):
        values = [5, 9, 13]
        plain = self.language(
            mx.array([values], dtype=mx.uint32),
            cache=self.prefill(),
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
        )
        captured = self.language(
            mx.array([values], dtype=mx.uint32),
            cache=self.prefill(),
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
            capture_layer_ids=[],
        )
        # Without the request the sink holds only the final mixed hidden.
        self.assertEqual(len(plain.hidden_states), 1)
        # With [] the sink holds pre-final HC residual first, final last.
        self.assertEqual(len(captured.hidden_states), 2)
        hc_hidden, final_hidden = captured.hidden_states
        self.assertEqual(hc_hidden.shape[-1], 64 * 2)
        self.assertEqual(final_hidden.shape[-1], 64)
        mx.eval(hc_hidden, final_hidden)
        # The final mixed hidden is unchanged by the extra capture.
        self.assertEqual(
            float(
                mx.max(mx.abs(plain.hidden_states[-1] - final_hidden)).item()
            ),
            0.0,
        )

    def test_mtp_multi_cycle_matches_sequential_target(self):
        from models.flashnext.mtp import attach
        from models.flashnext.speculative import MTPGreedy

        language = tiny_language()
        attach(language)
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        cache = language.make_cache()
        output = language(prompt, cache=cache)
        token = mx.argmax(output.logits[:, -1, :], axis=-1).astype(mx.uint32)
        expected = []
        for _ in range(24):
            expected.append(int(token.item()))
            output = language(token.reshape(1, 1), cache=cache)
            token = mx.argmax(
                output.logits[:, -1, :], axis=-1
            ).astype(mx.uint32)
            mx.eval(token)

        language._position_ids = None
        language._rope_deltas = None
        decoder = MTPGreedy(language, depth=3)
        decoder.append(prompt)
        actual = list(decoder.generate(24, set()))
        self.assertEqual(actual, expected)
        # The draft chain must actually execute, not sit idle.
        self.assertGreater(decoder.stats.drafted, 0)
        self.assertGreater(decoder.stats.cycles, 1)

    def test_target_replay_replays_single_tokens(self):
        # Recurrent replay must be singleton-equivalent: a batched
        # forward predicts the same tokens but can leave different
        # recurrent states, which diverges later Vontra runs. Lock the
        # one-call-per-token shape here, not just predictions.
        from models.flashnext.speculative import (
            MTPGreedy,
            restore_cache,
            snapshot_cache,
        )

        calls = []

        class FakeLanguage:
            def make_cache(self):
                return [SimpleNamespace(
                    state=mx.zeros((2, 2)), meta_state=None)]

            def __call__(self, ids, cache=None):
                calls.append(tuple(int(v) for v in ids.shape))
                return SimpleNamespace(logits=mx.zeros((1, 1, 4)))

        language = FakeLanguage()
        decoder = MTPGreedy.__new__(MTPGreedy)
        decoder.language = language
        decoder.target_cache = language.make_cache()
        decoder.stats = SimpleNamespace(replayed=0)
        snapshot = snapshot_cache(decoder.target_cache)
        decoder._target_replay(
            snapshot, mx.array([[9, 10, 11]], dtype=mx.uint32))
        self.assertEqual(calls, [(1, 1), (1, 1), (1, 1)])
        self.assertEqual(decoder.stats.replayed, 3)

    def test_target_replay_states_match_singleton(self):
        from models.flashnext.mtp import attach
        from models.flashnext.speculative import (
            MTPGreedy,
            restore_cache,
            snapshot_cache,
        )

        def states(cache):
            return snapshot_cache(cache)

        def state_max_abs(left, right):
            peak = 0.0

            def walk(a, b):
                nonlocal peak
                if isinstance(a, (list, tuple)):
                    for x, y in zip(a, b):
                        walk(x, y)
                    return
                if a is None or b is None:
                    return
                peak = max(peak, float(
                    mx.max(mx.abs(a.astype(mx.float32)
                                  - b.astype(mx.float32))).item()))

            for (a_state, _), (b_state, _) in zip(left, right):
                walk(a_state, b_state)
            return peak

        language = tiny_language()
        attach(language)
        prompt = mx.array([[3, 7, 11]], dtype=mx.uint32)
        cache = language.make_cache()
        mx.eval(language(prompt, cache=cache).logits)
        block = mx.array([[5, 9, 13]], dtype=mx.uint32)
        snapshot = snapshot_cache(cache)

        replayed = language.make_cache()
        restore_cache(replayed, snapshot)
        decoder = MTPGreedy.__new__(MTPGreedy)
        decoder.language = language
        decoder.target_cache = replayed
        decoder.stats = SimpleNamespace(replayed=0)
        decoder._target_replay(snapshot_cache(replayed), block)

        sequential = language.make_cache()
        restore_cache(sequential, snapshot)
        for position in range(block.shape[1]):
            mx.eval(language(block[:, position:position + 1],
                             cache=sequential).logits)

        self.assertEqual(
            state_max_abs(states(replayed), states(sequential)), 0.0)

if __name__ == "__main__":
    unittest.main()
