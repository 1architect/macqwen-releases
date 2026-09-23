"""Expert pool: LRU slot assignment and byte-exact record fills."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_pool
from models.flashnext.slab_pack import get_slab_layout
from models.flashnext.store import SafeTensorStore

PREFIX = "language_model.model.layers.4.mlp.switch_mlp"
ROWS = {"weight": 819_200, "scales": 102_400, "biases": 102_400}


class _Store:
    dir = "/nonexistent"
    refs = {}


class AssignTests(unittest.TestCase):
    def make(self, slots):
        stride = get_slab_layout(32).record_stride
        return expert_pool.ExpertPool(_Store(), expert_pool.HEADER + slots * stride)

    def test_hits_misses_and_lru_eviction(self):
        pool = self.make(16)
        slots, misses = pool.assign(1, list(range(10)))
        self.assertEqual(len(misses), 10)
        self.assertEqual(len(set(slots.values())), 10)
        slots2, misses2 = pool.assign(1, [3, 4, 20])
        self.assertEqual(slots2[3], slots[3])
        self.assertEqual([e for e, _ in misses2], [20])
        # Fill the pool, then force evictions: the oldest keys go first and
        # keys routed in the same call survive.
        pool.assign(2, list(range(5)))           # 16 slots now used
        slots3, misses3 = pool.assign(1, [3, 4, 30, 31])
        self.assertEqual(slots3[3], slots[3])
        self.assertEqual(slots3[4], slots[4])
        evicted = {slots[0], slots[1]}
        self.assertEqual({slot for _, slot in misses3}, evicted)
        self.assertEqual(pool.hits, 2 + 2)
        self.assertEqual(pool.misses, 10 + 1 + 5 + 2)

    def test_too_small_pool_is_refused(self):
        with self.assertRaises(ValueError):
            self.make(4)


class FillTests(unittest.TestCase):
    def test_fill_places_checkpoint_rows_at_record_offsets(self):
        mx.random.seed(8)
        experts = 5
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part, size in ROWS.items():
                tensors[f"{PREFIX}.{projection}.{part}"] = mx.random.randint(
                    0, 255, (experts, size)
                ).astype(mx.uint8)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            layout = get_slab_layout(32)
            pool = expert_pool.ExpertPool(
                store, expert_pool.HEADER + 16 * layout.record_stride)
            workers = ThreadPoolExecutor(4)
            try:
                slots, misses = pool.assign(4, [3, 0, 4])
                futures = pool.fill(PREFIX, misses, workers.submit, chunk=2)
                for future in futures:
                    future.result()
                for expert, slot in slots.items():
                    base = expert_pool.HEADER + slot * layout.record_stride
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for part, size in ROWS.items():
                            start = base + layout.offset(projection, part)
                            expected = store.rows_np(
                                f"{PREFIX}.{projection}.{part}", [expert]
                            ).tobytes()
                            self.assertEqual(
                                bytes(pool.array[start:start + size]), expected,
                                f"expert {expert} {projection}.{part}",
                            )
            finally:
                workers.shutdown()
                pool.close()
                store.close()


class ConfigTests(unittest.TestCase):
    def test_off_by_default_and_parsed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(expert_pool.SIZE_FLAG, None)
            self.assertFalse(expert_pool.enabled())
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "6"}):
            self.assertEqual(expert_pool.configured_gb(), 6.0)
            self.assertTrue(expert_pool.enabled())


if __name__ == "__main__":
    unittest.main()
