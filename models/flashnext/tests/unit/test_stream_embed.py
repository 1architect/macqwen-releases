"""A streamed input embedding must return QuantizedEmbedding's exact rows."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn

from models.flashnext.ngram import StreamingQuantizedEmbedding
from models.flashnext.store import SafeTensorStore

PREFIX = "language_model.model.embed_tokens"


class StreamEmbedTests(unittest.TestCase):
    def test_rows_match_quantized_embedding(self):
        mx.random.seed(5)
        table = nn.QuantizedEmbedding(300, 256, group_size=32, bits=4)
        source = nn.Embedding(300, 256)
        source.weight = (mx.random.normal((300, 256)) * 0.1).astype(mx.bfloat16)
        table = nn.QuantizedEmbedding.from_embedding(source, group_size=32, bits=4)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), {
                f"{PREFIX}.weight": table.weight,
                f"{PREFIX}.scales": table.scales,
                f"{PREFIX}.biases": table.biases,
            })
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {
                    f"{PREFIX}.{part}": shard for part in ("weight", "scales", "biases")
                }}, out)
            store = SafeTensorStore(directory)
            try:
                streamed = StreamingQuantizedEmbedding(store, PREFIX, 256)
                ids = mx.array([[3, 299, 0, 3, 150]])
                self.assertTrue(mx.array_equal(table(ids), streamed(ids)).item())
                self.assertEqual(table(ids).dtype, streamed(ids).dtype)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
