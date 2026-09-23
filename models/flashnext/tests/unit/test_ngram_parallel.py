"""Parallel n-gram row reads must return the serial path's exact values."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn

from models.flashnext import ngram
from models.flashnext.store import SafeTensorStore


class ParallelNgramTests(unittest.TestCase):
    def test_parallel_matches_serial(self):
        mx.random.seed(9)
        sizes = (40, 25, 33)
        tensors, weight_map = {}, {}
        shard_file = "model-00001-of-00001.safetensors"
        for index, rows in enumerate(sizes):
            source = nn.Embedding(rows, 128)
            source.weight = (mx.random.normal((rows, 128)) * 0.1).astype(mx.bfloat16)
            table = nn.QuantizedEmbedding.from_embedding(source, group_size=32, bits=4)
            for part in ("weight", "scales", "biases"):
                key = f"ngram.shards.{index}.{part}"
                tensors[key] = getattr(table, part)
                weight_map[key] = shard_file
        with tempfile.TemporaryDirectory() as directory:
            mx.save_safetensors(os.path.join(directory, shard_file), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": weight_map}, out)
            store = SafeTensorStore(directory)
            try:
                shards = [
                    ngram.StreamingQuantizedEmbedding(store, f"ngram.shards.{i}", 128)
                    for i in range(len(sizes))
                ]
                table = ngram.StreamingShardedEmbedding(shards, sizes, 128)
                ids = mx.array([[(7 * k) % sum(sizes) for k in range(90)]])
                ngram.set_parallel_min_rows(0)
                serial = table(ids)
                ngram.set_parallel_min_rows(8)
                parallel = table(ids)
                self.assertTrue(mx.array_equal(serial, parallel).item())
            finally:
                ngram.set_parallel_min_rows(0)
                store.close()


if __name__ == "__main__":
    unittest.main()
