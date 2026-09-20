from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.bonsai2.q4_attention_kernel import (
    affine_q4_reference,
    causal_visible_tokens,
    kv_head_for_query,
    page_ranges,
    unpack_q4_codes,
    unsupported_reason,
)


class Q4AttentionKernelTests(unittest.TestCase):
    def test_page_boundaries_include_partial_pages(self):
        self.assertEqual(page_ranges(1), ((0, 1),))
        self.assertEqual(page_ranges(31), ((0, 31),))
        self.assertEqual(page_ranges(32), ((0, 32),))
        self.assertEqual(page_ranges(33), ((0, 32), (32, 33)))

    def test_gqa_maps_six_query_heads_to_each_kv_head(self):
        self.assertEqual(
            [kv_head_for_query(head) for head in range(24)],
            [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6,
        )

    def test_causal_visibility_uses_absolute_query_positions(self):
        self.assertEqual(causal_visible_tokens(0, 0, 33), 1)
        self.assertEqual(causal_visible_tokens(31, 0, 33), 32)
        self.assertEqual(causal_visible_tokens(31, 1, 33), 33)
        self.assertEqual(causal_visible_tokens(100, 0, 33), 33)

    def test_extracts_and_affine_corrects_q4_codes(self):
        packed = mx.array([[0x76543210] * 8], dtype=mx.uint32)
        codes = unpack_q4_codes(packed, head_dim=64)
        self.assertEqual(codes.shape, (1, 64))
        self.assertEqual(codes[0, :8].tolist(), list(range(8)))
        scales = mx.array([[2.0]], dtype=mx.float16)
        biases = mx.array([[-1.0]], dtype=mx.float16)
        corrected = affine_q4_reference(packed, scales, biases, group_size=64)
        mx.eval(corrected)
        np.testing.assert_array_equal(
            np.asarray(corrected)[0, :8], np.arange(8) * 2 - 1
        )

    def test_rejects_unsupported_cache_mask_and_geometry(self):
        queries = mx.zeros((1, 24, 1, 256), dtype=mx.float16)
        packed = mx.zeros((1, 4, 1, 32), dtype=mx.uint32)
        metadata = mx.zeros((1, 4, 1, 4), dtype=mx.float16)
        keys = (packed, metadata, metadata)
        values = (packed, metadata, metadata)
        cache = SimpleNamespace(bits=4, group_size=64, offset=1)
        self.assertIsNone(unsupported_reason(queries, keys, values, cache, "causal"))
        self.assertIn(
            "mask", unsupported_reason(queries, keys, values, cache, mx.ones((1, 1)))
        )
        multi_row = mx.zeros((1, 24, 2, 256), dtype=mx.float16)
        self.assertIn(
            "unmasked", unsupported_reason(multi_row, keys, values, cache, None)
        )
        cache.bits = 8
        self.assertIn("Q4", unsupported_reason(queries, keys, values, cache, "causal"))

    def test_accepts_bonsai_fp32_query_dtype(self):
        import models.bonsai2.q4_attention_kernel as kernel_module

        queries = mx.zeros((1, 24, 1, 256), dtype=mx.float32)
        packed = mx.zeros((1, 4, 1, 32), dtype=mx.uint32)
        metadata = mx.zeros((1, 4, 1, 4), dtype=mx.float16)
        cache = SimpleNamespace(bits=4, group_size=64, offset=1)
        captured = {}

        def fake_kernel(**kwargs):
            captured.update(kwargs)
            return [mx.zeros(queries.shape, dtype=mx.float32)]

        with patch.object(kernel_module, "_get_kernel", return_value=fake_kernel):
            output = kernel_module.fused_q4_attention(
                queries, (packed, metadata, metadata),
                (packed, metadata, metadata), cache, 1.0, "causal",
            )
        self.assertEqual(output.dtype, mx.float32)
        self.assertEqual(captured["template"], [("T", mx.float32)])


if __name__ == "__main__":
    unittest.main()
