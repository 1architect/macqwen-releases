from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx

from models.flashnext import qsa_chunk
from models.flashnext.qsa_chunk import _chunk_mask, _dense_mask_bytes
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding
from mlx_vlm.models.qwen4_exp.language import (
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpAttention,
    Qwen4ExpQSAIndexer,
)


class QSAChunkTests(unittest.TestCase):
    def test_followup_context_routes_around_dense_mask_budget(self):
        # REAP's 1,978-token follow-up with 8,815 cached tokens and 512
        # selected blocks requests 10,930,459,648 bytes upstream.
        self.assertEqual(
            _dense_mask_bytes(1, 1978, 8815, 512),
            10930459648,
        )

    def test_dispatch_uses_cached_prefix_size(self):
        from models.flashnext import qsa_chunk

        attention = SimpleNamespace(
            indexer=SimpleNamespace(block_topk=512, compress_ratio=4)
        )
        cache = SimpleNamespace(offset=8815)
        x = mx.zeros((1, 1978, 1), dtype=mx.bfloat16)
        with patch.object(
            qsa_chunk, "_prepare_indexer", side_effect=RuntimeError("chunked")
        ), patch.object(qsa_chunk, "_ORIGINAL_CALL", Mock()) as original:
            with self.assertRaisesRegex(RuntimeError, "chunked"):
                qsa_chunk._chunked_call(attention, x, cache=cache)
        original.assert_not_called()

    def test_dispatch_keeps_small_decode_on_original_path(self):
        from models.flashnext import qsa_chunk

        attention = SimpleNamespace(
            indexer=SimpleNamespace(block_topk=512, compress_ratio=4)
        )
        cache = SimpleNamespace(offset=8815)
        x = mx.zeros((1, 1, 1), dtype=mx.bfloat16)
        original = object()
        with patch.object(qsa_chunk, "_ORIGINAL_CALL", return_value=original):
            self.assertIs(
                qsa_chunk._chunked_call(attention, x, cache=cache), original
            )

    def test_chunk_mask_matches_dense_reference(self):
        indexer = SimpleNamespace(
            block_topk=2,
            compress_ratio=4,
            head_dim=3,
        )
        query = mx.arange(1 * 2 * 16 * 3).reshape(1, 2, 16, 3) / 100
        pooled = mx.arange(1 * 1 * 8 * 3).reshape(1, 1, 8, 3) / 50
        chunks = [
            _chunk_mask(indexer, query, pooled, 32, 0, 8, start, end)
            for start, end in ((0, 5), (5, 11), (11, 16))
        ]
        actual = mx.concatenate(chunks, axis=2)

        scores = query @ pooled.transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1)
        scores = scores / (indexer.head_dim**0.5)
        query_ends = mx.arange(16) + 1
        complete_counts = query_ends // indexer.compress_ratio
        valid = mx.arange(8)[None, None, :] < complete_counts[None, :, None]
        scores = mx.where(valid, scores, -mx.inf)
        blocks = mx.argpartition(scores, kth=-2, axis=-1)[..., -2:]
        token_indices = mx.arange(32)
        token_blocks = token_indices // indexer.compress_ratio
        selected = mx.any(
            token_blocks[None, None, None, :] == blocks[..., None],
            axis=2,
        )
        tail_start = complete_counts * indexer.compress_ratio
        tail = (token_indices[None, None, :] >= tail_start[None, :, None]) & (
            token_indices[None, None, :] < query_ends[None, :, None]
        )
        causal = token_indices[None, None, :] < query_ends[None, :, None]
        reference = mx.where(
            (complete_counts > indexer.block_topk)[None, :, None],
            selected | tail,
            causal,
        )[:, None]
        mx.eval(actual, reference)

        self.assertTrue(bool(mx.array_equal(actual, reference).item()))

    def test_chunk_mask_matches_cached_prefix_and_partial_tail(self):
        indexer = SimpleNamespace(
            block_topk=2,
            compress_ratio=4,
            head_dim=3,
        )
        past_len = 8
        key_len = 40
        max_complete_blocks = key_len // indexer.compress_ratio
        query_len = 16
        query = mx.arange(1 * 2 * query_len * 3).reshape(
            1, 2, query_len, 3
        ) / 100
        pooled = mx.arange(1 * 1 * max_complete_blocks * 3).reshape(
            1, 1, max_complete_blocks, 3
        ) / 50
        chunks = [
            _chunk_mask(
                indexer, query, pooled, key_len, past_len,
                max_complete_blocks, start, end
            )
            for start, end in ((0, 5), (5, 11), (11, query_len))
        ]
        actual = mx.concatenate(chunks, axis=2)

        scores = query @ pooled.transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1)
        scores = scores / (indexer.head_dim**0.5)
        query_ends = past_len + mx.arange(query_len) + 1
        complete_counts = query_ends // indexer.compress_ratio
        valid = mx.arange(max_complete_blocks)[None, None, :] < complete_counts[None, :, None]
        scores = mx.where(valid, scores, -mx.inf)
        blocks = mx.argpartition(scores, kth=-2, axis=-1)[..., -2:]
        token_indices = mx.arange(key_len)
        token_blocks = token_indices // indexer.compress_ratio
        selected = mx.any(
            token_blocks[None, None, None, :] == blocks[..., None], axis=2
        )
        tail_start = complete_counts * indexer.compress_ratio
        tail = (token_indices[None, None, :] >= tail_start[None, :, None]) & (
            token_indices[None, None, :] < query_ends[None, :, None]
        )
        causal = token_indices[None, None, :] < query_ends[None, :, None]
        reference = mx.where(
            (complete_counts > indexer.block_topk)[None, :, None],
            selected | tail,
            causal,
        )[:, None]
        mx.eval(actual, reference)
        self.assertTrue(bool(mx.array_equal(actual, reference).item()))


class QSAIndexerTests(unittest.TestCase):
    def setUp(self):
        original = Qwen4ExpQSAIndexer.__call__
        if original is qsa_chunk._indexer_call:
            original = qsa_chunk._ORIGINAL_INDEXER_CALL
        self.original = original
        for name, value in (
            ("QSA_CACHE_POOLED_KEYS", True),
            ("QSA_SCATTER_DECODE", False),
            ("_ORIGINAL_INDEXER_CALL", original),
        ):
            self.enterContext(patch.object(qsa_chunk, name, value))
        self.enterContext(patch.object(Qwen4ExpQSAIndexer, "__call__", qsa_chunk._indexer_call))
        config = SimpleNamespace(
            hidden_size=16, indexer_n_heads=2, indexer_kv_heads=1,
            indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
            rms_norm_eps=1e-6,
        )
        mx.random.seed(81)
        self.indexer = Qwen4ExpQSAIndexer(
            config, Qwen3_5RotaryEmbedding(8, mrope_section=[2, 1, 1])
        )

    def assertArrayEqual(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.array_equal(actual, expected).item())

    def step(self, caches, length, positions=None, dtype=mx.float32, batch=1):
        reference, candidate = caches
        x = mx.random.normal((batch, length, 16)).astype(dtype)
        upstream_rope = self.indexer._apply_rope
        rotated = []

        def capture(values, ids):
            result = upstream_rope(values, ids)
            rotated.append(result)
            return result

        with patch.object(self.indexer, "_apply_rope", side_effect=capture):
            expected = self.original(self.indexer, x, reference, positions)
        actual = self.indexer(x, candidate, positions)
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertArrayEqual(actual, expected)
            if qsa_chunk.QSA_CACHE_POOLED_KEYS and candidate is not None:
                self.assertArrayEqual(candidate._flashnext_pooled_keys[-1], rotated[-1])
        if candidate is not None:
            self.assertArrayEqual(candidate.index_keys, reference.index_keys)
            self.assertArrayEqual(candidate.index_position_ids, reference.index_position_ids)
            kv = mx.tile(x[:, None, :, :8], (1, 1, 1, 4))
            for cache in caches:
                cache.update_and_fetch(kv, kv)
        return actual

    def test_boundaries_tails_and_growing_decode_all_arms(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            self.indexer.set_dtype(dtype)
            for caching, scatter in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(dtype=dtype, caching=caching, scatter=scatter):
                    with patch.object(qsa_chunk, "QSA_CACHE_POOLED_KEYS", caching), patch.object(
                        qsa_chunk, "QSA_SCATTER_DECODE", scatter
                    ):
                        caches = QSAKVCache(), QSAKVCache()
                        for length in (3, 4, 4, 1, 1, 1, 1, 1, 5, 1):
                            self.step(caches, length, dtype=dtype)
                        if not caching:
                            self.assertFalse(hasattr(caches[1], "_flashnext_pooled_keys"))

    def test_only_newly_completed_blocks_are_transformed(self):
        caches = QSAKVCache(), QSAKVCache()
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 13)
            self.assertEqual(pool.call_args.args[-2:], (0, 3))
            pool.reset_mock()
            self.step(caches, 1)
            self.step(caches, 1)
            pool.assert_not_called()
            self.step(caches, 1)
            self.assertEqual(pool.call_count, 1)
            self.assertEqual(pool.call_args.args[-2:], (3, 4))
            pool.reset_mock()
            self.step(caches, 9)
            self.assertEqual(pool.call_args.args[-2:], (4, 6))

    def test_trim_and_regrow_rebuild(self):
        for trim in (1, 4, 8, 16):
            with self.subTest(trim=trim):
                caches = QSAKVCache(), QSAKVCache()
                self.step(caches, 16)
                for cache in caches:
                    self.assertEqual(cache.trim(trim), trim)
                with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
                    self.step(caches, trim)
                    self.assertEqual(pool.call_args.args[-2:], (0, 4))
                self.step(caches, 1)

    def test_same_length_state_restore_rebuilds(self):
        caches = QSAKVCache(), QSAKVCache()
        donors = QSAKVCache(), QSAKVCache()
        self.step(caches, 16)
        self.step(donors, 16)
        for cache, donor in zip(caches, donors):
            self.assertEqual(len(donor.state), 4)
            cache.state = donor.state
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 1)
            self.assertEqual(pool.call_args.args[-2:], (0, 4))
        restored = QSAKVCache(), QSAKVCache()
        for cache, donor in zip(restored, caches):
            cache.state = donor.state
            self.assertFalse(hasattr(cache, "_flashnext_pooled_keys"))
        self.step(restored, 3)

    def test_positions_text_mrope_and_rank_transitions(self):
        for mode in ("text", "mrope", "mixed"):
            caches = QSAKVCache(), QSAKVCache()
            for step, length in enumerate((13, 1, 2, 3, 1)):
                start = caches[0].offset
                positions = mx.broadcast_to(
                    (mx.arange(start, start + length) * 3 + 7)[None], (2, length)
                ) + mx.array([[0], [41]])
                if mode == "mrope" or (mode == "mixed" and step % 2):
                    positions = mx.stack((positions, positions * 2, positions + 11))
                with self.subTest(mode=mode, step=step):
                    self.step(caches, length, positions, batch=2)

    def test_position_replacement_rebuilds(self):
        caches = QSAKVCache(), QSAKVCache()
        self.step(caches, 16)
        for cache in caches:
            cache.index_position_ids = cache.index_position_ids + 21
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 1)
            self.assertEqual(pool.call_args.args[-2:], (0, 4))

    def test_batch_filter_and_extract(self):
        for multimodal in (False, True):
            caches = QSAKVCache(), QSAKVCache()
            positions = mx.arange(32).reshape(2, 16)
            if multimodal:
                positions = mx.stack((positions, positions + 7, positions * 2))
            self.step(caches, 16, positions, batch=2)
            for cache in caches:
                cache.filter(mx.array([1, 0]))
            self.step(caches, 1, batch=2)
            extracted = tuple(cache.extract(1) for cache in caches)
            self.assertFalse(hasattr(extracted[1], "_flashnext_pooled_keys"))
            self.step(extracted, 3)
            for cache in caches:
                cache.filter(mx.array([1]))
            self.step(caches, 3)

    def test_quantized_cache_and_conversion(self):
        caches = QSAKVCache(), QSAKVCache()
        self.step(caches, 13)
        caches = tuple(cache.to_quantized(group_size=32, bits=4) for cache in caches)
        self.assertIsInstance(caches[1], QSAQuantizedKVCache)
        self.step(caches, 3)
        saved = tuple(cache.state for cache in caches)
        self.step(caches, 4)
        for cache, state in zip(caches, saved):
            cache.state = state
        self.step(caches, 1)
        for cache in caches:
            cache.trim(2)
        self.step(caches, 5)

    def test_cache_identity_and_indexer_identity(self):
        caches = QSAKVCache(), QSAKVCache()
        self.step(caches, 16)
        other = QSAKVCache(), QSAKVCache()
        self.step(other, 16)
        self.step(caches, 1)
        caches[1]._flashnext_pooled_keys = other[1]._flashnext_pooled_keys
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 1)
            self.assertEqual(pool.call_args.args[-2:], (0, 4))
        saved = caches[1]._flashnext_pooled_keys
        caches[1]._flashnext_pooled_keys = (object(), *saved[1:])
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 1)
            self.assertEqual(pool.call_args.args[-2:], (0, 4))

    def test_disable_then_reenable_rebuilds(self):
        caches = QSAKVCache(), QSAKVCache()
        self.step(caches, 16)
        with patch.object(qsa_chunk, "QSA_CACHE_POOLED_KEYS", False):
            self.step(caches, 4)
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 1)
            self.assertEqual(pool.call_args.args[-2:], (0, 5))

    def test_raw_key_replacement_rebuilds(self):
        caches = QSAKVCache(), QSAKVCache()
        self.step(caches, 16)
        for cache in caches:
            cache.index_keys = cache.index_keys + 1
        with patch.object(qsa_chunk, "_pool_keys", wraps=qsa_chunk._pool_keys) as pool:
            self.step(caches, 1)
            self.assertEqual(pool.call_args.args[-2:], (0, 4))

    def test_scatter_matches_tied_scores_and_partial_tails(self):
        indexer = SimpleNamespace(block_topk=2, compress_ratio=4, head_dim=8)
        for batch in (1, 2):
            for key_len in range(12, 21):
                blocks = key_len // 4
                query = mx.zeros((batch, 2, 1, 8))
                pooled = mx.zeros((batch, 1, blocks, 8))
                args = (indexer, query, pooled, key_len, key_len - 1, blocks, 0, 1)
                self.assertArrayEqual(
                    qsa_chunk._chunk_mask(*args, scatter=True),
                    qsa_chunk._chunk_mask(*args, scatter=False),
                )

    def test_without_cache(self):
        for scatter in (False, True):
            with patch.object(qsa_chunk, "QSA_SCATTER_DECODE", scatter):
                self.step((None, None), 1)
                self.step((None, None), 16)

    def test_cache_and_scatter_dispatch_independently(self):
        for caching, scatter in ((True, False), (False, True), (True, True)):
            caches = QSAKVCache(), QSAKVCache()
            with patch.object(qsa_chunk, "QSA_CACHE_POOLED_KEYS", caching), patch.object(
                qsa_chunk, "QSA_SCATTER_DECODE", scatter
            ), patch.object(qsa_chunk, "_chunk_mask", wraps=qsa_chunk._chunk_mask) as mask:
                self.step(caches, 13)
                mask.reset_mock()
                self.step(caches, 1)
                mask.assert_called_once()
                self.assertEqual(mask.call_args.kwargs["scatter"], scatter)
                self.assertEqual(hasattr(caches[1], "_flashnext_pooled_keys"), caching)

    def test_non_scalar_offsets_keep_upstream(self):
        cache = SimpleNamespace(offset=mx.array([3, 5]))
        with patch.object(qsa_chunk, "_ORIGINAL_INDEXER_CALL", return_value="upstream") as original:
            x = mx.zeros((2, 1, 16))
            self.assertEqual(self.indexer(x, cache, None), "upstream")
            original.assert_called_once_with(self.indexer, x, cache, None)

    def test_disabled_calls_only_upstream(self):
        with patch.object(qsa_chunk, "QSA_CACHE_POOLED_KEYS", False), patch.object(
            qsa_chunk, "QSA_SCATTER_DECODE", False
        ), patch.object(qsa_chunk, "_ORIGINAL_INDEXER_CALL", return_value="upstream") as original, patch.object(
            qsa_chunk, "_prepare_indexer", side_effect=AssertionError("prepared")
        ):
            self.assertEqual(qsa_chunk._indexer_call(None, None, None, None), "upstream")
            original.assert_called_once_with(None, None, None, None)

    def test_scatter_only_keeps_prefill_upstream(self):
        with patch.object(qsa_chunk, "QSA_CACHE_POOLED_KEYS", False), patch.object(
            qsa_chunk, "QSA_SCATTER_DECODE", True
        ), patch.object(qsa_chunk, "_ORIGINAL_INDEXER_CALL", return_value="upstream") as original:
            x = mx.zeros((1, 16, 16))
            self.assertEqual(self.indexer(x, None, None), "upstream")
            original.assert_called_once()


class QSAInstallTests(unittest.TestCase):
    def test_apply_is_idempotent_and_restores_patches(self):
        attention_call = Qwen4ExpAttention.__call__
        indexer_call = Qwen4ExpQSAIndexer.__call__
        with patch.object(Qwen4ExpAttention, "__call__", attention_call), patch.object(
            Qwen4ExpQSAIndexer, "__call__", indexer_call
        ), patch.object(Qwen4ExpAttention, "_flashnext_chunked_qsa", False, create=True), patch.object(
            qsa_chunk, "_ORIGINAL_CALL", None
        ), patch.object(qsa_chunk, "_ORIGINAL_INDEXER_CALL", None):
            self.assertTrue(qsa_chunk.apply())
            self.assertIs(Qwen4ExpQSAIndexer.__call__, qsa_chunk._indexer_call)
            self.assertIs(Qwen4ExpAttention.__call__, qsa_chunk._chunked_call)
            self.assertIs(qsa_chunk._ORIGINAL_INDEXER_CALL, indexer_call)
            self.assertFalse(qsa_chunk.apply())
        self.assertIs(Qwen4ExpQSAIndexer.__call__, indexer_call)
        self.assertIs(Qwen4ExpAttention.__call__, attention_call)


if __name__ == "__main__":
    unittest.main()
