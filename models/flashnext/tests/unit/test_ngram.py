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

