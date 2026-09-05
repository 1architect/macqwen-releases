from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx

from models.flashnext.qsa_chunk import _chunk_mask, _dense_mask_bytes


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


if __name__ == "__main__":
    unittest.main()
