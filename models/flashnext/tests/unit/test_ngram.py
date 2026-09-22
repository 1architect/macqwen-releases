from __future__ import annotations

import unittest

import mlx.core as mx

from models.flashnext.ngram import StreamingShardedEmbedding


class FakeShard:
    def __init__(self, base):
        self.base = base
        self.calls = []

    def _rows(self, rows):
        self.calls.append(tuple(int(row) for row in rows))
        return mx.array(
            [[self.base + int(row), self.base + int(row) + 1] for row in rows]
        )

    def __call__(self, indices):
        mx.eval(indices)
        return self._rows(indices.tolist())


class StreamingShardedEmbeddingTest(unittest.TestCase):
    def test_direct_matches_legacy_and_skips_unused_shards(self):
        shards = [FakeShard(0), FakeShard(100), FakeShard(200)]
        table = StreamingShardedEmbedding(shards, (3, 2, 4), 2)
        indices = mx.array([[0, 2, 5, 8]], dtype=mx.int64)

        table.direct = False
        expected = table(indices)
        mx.eval(expected)
        for shard in shards:
            shard.calls.clear()

        table.direct = True
        actual = table(indices)
        mx.eval(actual)

        self.assertEqual(expected.tolist(), actual.tolist())
        self.assertEqual(shards[0].calls, [(0, 2)])
        self.assertEqual(shards[1].calls, [])
        self.assertEqual(shards[2].calls, [(0, 3)])


if __name__ == "__main__":
    unittest.main()


class FakeStore:
    """Rows are deterministic, so a wrong cache hit is visible in the values."""

    def __init__(self):
        self.reads = 0

    def rows(self, name, indices):
        self.reads += 1
        return mx.array([[float(row), float(row) + 0.5] for row in indices])


class CachedRowTests(unittest.TestCase):
    """A capacity smaller than the request must not lose this call's rows."""

    def table(self, capacity):
        from models.flashnext.ngram import StreamingQuantizedEmbedding

        table = StreamingQuantizedEmbedding.__new__(StreamingQuantizedEmbedding)
        table.store = FakeStore()
        table.prefix = "t"
        table.dims = 2
        table.mode = "affine"
        table.group_size, table.bits = 2, 4
        table.capacity = capacity
        from collections import OrderedDict

        table._cache = OrderedDict()
        return table

    def dequantized(self, table, rows):
        # bypass real dequantisation: the store already returns final values
        import models.flashnext.ngram as ngram

        original = ngram.mx.dequantize
        ngram.mx.dequantize = lambda w, s, b, **kw: w
        try:
            return table._rows(rows)
        finally:
            ngram.mx.dequantize = original

    def test_more_missing_rows_than_capacity_still_returns_them(self):
        table = self.table(capacity=2)
        out = self.dequantized(table, [0, 1, 2, 3])
        self.assertEqual(out.shape, (4, 2))
        self.assertEqual([int(v) for v in out[:, 0].tolist()], [0, 1, 2, 3])

    def test_a_cached_row_is_reused_without_a_second_read(self):
        table = self.table(capacity=8)
        self.dequantized(table, [5, 6])
        before = table.store.reads
        out = self.dequantized(table, [5, 6])
        self.assertEqual(table.store.reads, before)
        self.assertEqual([int(v) for v in out[:, 0].tolist()], [5, 6])

    def test_a_reused_row_survives_later_eviction(self):
        table = self.table(capacity=2)
        self.dequantized(table, [1, 2])
        out = self.dequantized(table, [1, 3, 4])
        self.assertEqual([int(v) for v in out[:, 0].tolist()], [1, 3, 4])
