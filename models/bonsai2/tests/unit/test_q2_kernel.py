from __future__ import annotations

import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.bonsai2.q2_kernel import (
    Q2MPPInfeasible,
    Q2_PREFILL_BATCH_SIZES,
    Q2_PROJECTION_GEOMETRIES,
    q2_mpp_matmul,
    probe_q2_mpp,
    probe_q2_projection_geometries,
    q2_affine_reference,
    q2_dequantized_reference,
    q2_rounding_diagnostics,
    unpack_q2_codes,
)


class Q2KernelTests(unittest.TestCase):
    def test_extracts_low_to_high_codes_from_packed_words(self):
        packed = mx.array([[0xE4E4E4E4]], dtype=mx.uint32)
        self.assertEqual(
            unpack_q2_codes(packed).tolist(),
            [[0, 1, 2, 3] * 4],
        )

    def test_affine_reference_matches_stock_quantized_matmul(self):
        rows, input_size, batch = 8, 256, 3
        codes = np.arange(rows * input_size, dtype=np.uint32).reshape(
            rows, input_size
        ) % 4
        packed = np.zeros((rows, input_size // 16), dtype=np.uint32)
        for index in range(input_size):
            packed[:, index // 16] |= codes[:, index] << (2 * (index % 16))
        weight = mx.array(packed)
        scales = mx.array(
            np.linspace(0.2, 0.8, rows * 2, dtype=np.float16).reshape(rows, 2)
        )
        biases = mx.array(
            np.linspace(-0.2, 0.2, rows * 2, dtype=np.float16).reshape(rows, 2)
        )
        x = mx.array(
            np.arange(batch * input_size, dtype=np.float16).reshape(
                batch, input_size
            )
            / 17
        )
        reference = q2_affine_reference(x, weight, scales, biases)
        stock = mx.quantized_matmul(
            x, weight, scales, biases, transpose=True,
            group_size=128, bits=2,
        )
        mx.eval(reference, stock)
        np.testing.assert_array_equal(np.asarray(reference), np.asarray(stock))

    def test_dequantized_reference_promotes_metadata_for_fp32_and_fp16(self):
        weight = mx.array([[3] * 8], dtype=mx.uint32)
        scales = mx.array([[0.1]], dtype=mx.float16)
        biases = mx.array([[0.2]], dtype=mx.float16)
        for dtype in (mx.float32, mx.float16):
            with self.subTest(dtype=str(dtype)):
                x = mx.concatenate(
                    [mx.ones((1, 1), dtype=dtype),
                     mx.zeros((1, 127), dtype=dtype)],
                    axis=1,
                )
                reference = q2_dequantized_reference(
                    x, weight, scales, biases
                )
                stock = mx.quantized_matmul(
                    x, weight, scales, biases, transpose=True,
                    group_size=128, bits=2,
                )
                mx.eval(reference, stock)
                np.testing.assert_array_equal(
                    np.asarray(reference), np.asarray(stock)
                )

    def test_probe_records_unsupported_geometry_without_a_fallback(self):
        x = mx.zeros((31, 128), dtype=mx.float16)
        weight = mx.zeros((8, 8), dtype=mx.uint32)
        metadata = mx.zeros((8, 1), dtype=mx.float16)
        record = probe_q2_mpp(x, weight, metadata, metadata)
        self.assertFalse(record.feasible)
        self.assertFalse(record.exact)
        self.assertIn("32-row", record.reason)
        self.assertTrue(issubclass(Q2MPPInfeasible, RuntimeError))

    def test_projection_probe_requires_the_named_geometry_set(self):
        with self.assertRaisesRegex(ValueError, "missing Q2 projection"):
            probe_q2_projection_geometries({}, batch_sizes=(32,))
        self.assertEqual(
            Q2_PREFILL_BATCH_SIZES, (32, 128, 256, 512)
        )
        self.assertEqual(
            set(Q2_PROJECTION_GEOMETRIES),
            {"gate_proj", "up_proj", "down_proj"},
        )

    def test_mpp_grid_counts_one_threadgroup_per_output_tile(self):
        captured = {}

        def fake_kernel(**kwargs):
            captured.update(kwargs)
            return [mx.zeros((32, 128), dtype=mx.float16)]

        x = mx.zeros((32, 128), dtype=mx.float16)
        weight = mx.zeros((128, 8), dtype=mx.uint32)
        metadata = mx.zeros((128, 1), dtype=mx.float16)
        with patch(
            "models.bonsai2.q2_kernel._get_mpp_kernel",
            return_value=fake_kernel,
        ):
            q2_mpp_matmul(x, weight, metadata, metadata)
        self.assertEqual(captured["grid"], (256, 1, 1))
        self.assertEqual(captured["threadgroup"], (256, 1, 1))

    def test_partial_prompt_rows_are_padded_for_one_temporary_tile(self):
        captured = {}

        def fake_kernel(**kwargs):
            captured.update(kwargs)
            return [mx.zeros((32, 128), dtype=mx.float16)]

        x = mx.ones((31, 128), dtype=mx.float16)
        weight = mx.zeros((128, 8), dtype=mx.uint32)
        metadata = mx.zeros((128, 1), dtype=mx.float16)
        with patch(
            "models.bonsai2.q2_kernel._get_mpp_kernel",
            return_value=fake_kernel,
        ):
            result = q2_mpp_matmul(x, weight, metadata, metadata)
        self.assertEqual(tuple(captured["inputs"][3].shape), (32, 128))
        self.assertEqual(tuple(result.shape), (31, 128))

    def test_rounding_diagnostic_separates_affine_and_mlxs_dequantization(self):
        rows, input_size = 128, 256
        rng = np.random.default_rng(11)
        weight = mx.array(
            rng.integers(0, 2**32, (rows, input_size // 16), dtype=np.uint32)
        )
        scales = mx.array(rng.normal(0.1, 0.05, (rows, 2)).astype(np.float16))
        biases = mx.array(rng.normal(0.0, 0.05, (rows, 2)).astype(np.float16))
        x = mx.array(rng.normal(0.0, 1.0, (32, input_size)).astype(np.float16))
        diagnostic = q2_rounding_diagnostics(x, weight, scales, biases)
        self.assertGreater(diagnostic["affine_vs_stock_max_abs"], 0.0)
        self.assertGreaterEqual(
            diagnostic["affine_vs_dequantized_max_abs"],
            diagnostic["dequantized_vs_stock_max_abs"],
        )
        # The diagnostic dequantization is a real MLX reference intermediate,
        # not a claim that a dense fallback belongs in the runtime.
        mx.eval(q2_dequantized_reference(x, weight, scales, biases))


if __name__ == "__main__":
    unittest.main()
