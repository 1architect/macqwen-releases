from __future__ import annotations

from types import SimpleNamespace
import unittest

import mlx.core as mx

from models.flashnext.qsa_chunk import _chunk_mask


class QSAChunkTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
