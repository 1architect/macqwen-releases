from __future__ import annotations

import unittest

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from models.bonsai2.cache import SUPPORTED_CACHE_STEPS, set_kv_cache_step


class CacheTests(unittest.TestCase):
    def test_step_is_instance_local_and_skips_other_cache_types(self):
        ordinary = KVCache()
        sliding = RotatingKVCache(max_size=32)

        set_kv_cache_step([ordinary, sliding], 1024)

        self.assertEqual(ordinary.step, 1024)
        self.assertEqual(KVCache.step, 256)
        self.assertEqual(sliding.step, 256)

    def test_only_supported_steps_are_accepted(self):
        for step in SUPPORTED_CACHE_STEPS:
            cache = KVCache()
            set_kv_cache_step(cache, step)
            self.assertEqual(cache.step, step)
        with self.assertRaisesRegex(ValueError, "unsupported Bonsai-2 KV cache step"):
            set_kv_cache_step(KVCache(), 512)

    def test_real_mlx_values_survive_cache_boundaries(self):
        # Keep this tiny while still crossing both growth policies' edges.
        chunks = (255, 1, 1, 766, 1, 1)
        expected_length = sum(chunks)
        for step in SUPPORTED_CACHE_STEPS:
            with self.subTest(step=step):
                cache = KVCache()
                set_kv_cache_step(cache, step)
                expected_keys = []
                expected_values = []
                offset = 0

                for length in chunks:
                    keys = mx.arange(offset * 2, (offset + length) * 2)
                    keys = keys.reshape(1, 1, length, 2)
                    values = keys + 10000
                    actual_keys, actual_values = cache.update_and_fetch(
                        keys, values
                    )
                    expected_keys.append(keys)
                    expected_values.append(values)
                    offset += length
                    mx.eval(actual_keys, actual_values)
                    mx.synchronize()
                    self.assertEqual(cache.offset, offset)
                    if offset in (255, 256, 257, 1023, 1024, 1025):
                        self.assertEqual(actual_keys.shape[2], offset)
                        self.assertEqual(actual_values.shape[2], offset)

                expected_keys = mx.concatenate(expected_keys, axis=2)
                expected_values = mx.concatenate(expected_values, axis=2)
                actual_keys, actual_values = cache.state
                mx.eval(expected_keys, expected_values, actual_keys, actual_values)
                mx.synchronize()
                self.assertEqual(cache.offset, expected_length)
                self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys)))
                self.assertTrue(bool(mx.array_equal(actual_values, expected_values)))


if __name__ == "__main__":
    unittest.main()
