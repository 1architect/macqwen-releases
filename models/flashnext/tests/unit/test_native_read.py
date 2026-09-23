"""Batched native reads must land the same bytes as the store's own reads."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from models.flashnext.tests.bench import native_read
from models.flashnext.store import SafeTensorStore


class NativeReadTests(unittest.TestCase):
    def test_batch_matches_store_rows(self):
        mx.random.seed(2)
        tensors = {
            "t.weight": mx.random.randint(0, 2**31, (64, 3000)).astype(mx.uint32),
            "t.scales": mx.random.randint(0, 65535, (64, 777)).astype(mx.uint16),
        }
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            try:
                experts = [5, 63, 0, 17, 17, 40]
                batch = native_read.Batch()
                outputs = {}
                for name in tensors:
                    ref = store.refs[name]
                    out = store.empty_rows(name, len(experts))
                    batch.add(store._fd(ref.shard), ref.start, ref.row_bytes, experts,
                              native_read.address(out), ref.row_bytes)
                    outputs[name] = out
                # A strided destination, as in the expert-major stream record.
                ref = store.refs["t.scales"]
                stride = ref.row_bytes + 4096
                record = np.zeros(len(experts) * stride, dtype=np.uint8)
                batch.add(store._fd(ref.shard), ref.start, ref.row_bytes, experts,
                          native_read.address(record) + 100, stride)
                batch.run()
                for name, out in outputs.items():
                    self.assertTrue(np.array_equal(out, store.rows_np(name, experts)), name)
                expected = store.rows_np("t.scales", experts).view(np.uint8)
                for slot in range(len(experts)):
                    start = slot * stride + 100
                    self.assertTrue(np.array_equal(
                        record[start:start + ref.row_bytes], expected[slot]))
            finally:
                store.close()

    def test_failed_read_raises(self):
        batch = native_read.Batch()
        buffer = np.zeros(16, dtype=np.uint8)
        batch.add(-1, 0, 16, [0], native_read.address(buffer), 16)
        with self.assertRaises(OSError):
            batch.run()


if __name__ == "__main__":
    unittest.main()
