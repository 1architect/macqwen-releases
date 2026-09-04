"""Checkpoint-free tests for the bfloat16 SwiGLU contract harness."""
from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from models.flashnext.swiglu_contract import (
    DEFAULT_CANDIDATES,
    bfloat16_bits,
    bfloat16_from_bits,
    compare_swiglu,
    exhaustive_gate_patterns,
    metal_swiglu_available,
)


class TestSwiGLUContract(unittest.TestCase):
    def test_bfloat16_bit_round_trip(self):
        raw = np.array([0x0000, 0x3F80, 0x8000, 0x7F80, 0xFFFF], dtype=np.uint16)
        np.testing.assert_array_equal(bfloat16_bits(bfloat16_from_bits(raw)), raw)

    def test_reference_candidate_matches_on_representative_values(self):
        gate = bfloat16_from_bits([0x0000, 0x3F80, 0xBF80, 0x4120, 0xC120])
        up = bfloat16_from_bits([0x3F80, 0x3FA0, 0x3F00, 0x4020, 0xC000])
        reports = compare_swiglu(gate, up, candidates=("bf16_silu_mul",))
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].exact)
        self.assertEqual(reports[0].total, 5)

    def test_default_candidates_are_cross_platform(self):
        self.assertTrue(DEFAULT_CANDIDATES)
        self.assertTrue(all(not name.startswith("metal_") for name in DEFAULT_CANDIDATES))

    def test_exhaustive_scan_uses_bounded_batches_and_reports_mismatches(self):
        report = exhaustive_gate_patterns(
            up_bits=0x3F9E,
            batch_size=1024,
            candidates=("bf16_silu_mul", "fp32_silu_mul"),
            max_mismatches=3,
        )
        self.assertEqual(report.total, 65536)
        self.assertEqual(report.batch_size, 1024)
        self.assertEqual(report.up_bits, 0x3F9E)
        self.assertTrue(report.candidate("bf16_silu_mul").exact)
        fp32 = report.candidate("fp32_silu_mul")
        self.assertGreater(fp32.mismatches, 0)
        self.assertLessEqual(len(fp32.examples), 3)
        self.assertIn("mismatches", report.format())

    def test_invalid_scan_arguments_fail_early(self):
        with self.assertRaises(ValueError):
            exhaustive_gate_patterns(batch_size=0)
        with self.assertRaises(ValueError):
            exhaustive_gate_patterns(up_bits=0x10000)
        with self.assertRaises(ValueError):
            compare_swiglu(mx.zeros((2,)), mx.zeros((3,)))
        with self.assertRaises(ValueError):
            compare_swiglu(
                mx.zeros((2,)), mx.zeros((2,)), candidates=("missing",)
            )

    def test_metal_candidate_exhaustive_contract(self):
        available, reason = metal_swiglu_available()
        if not available:
            raise unittest.SkipTest(reason)
        for up_bits in (0x3F80, 0x3F9E):
            with self.subTest(up_bits=hex(up_bits)):
                report = exhaustive_gate_patterns(
                    up_bits=up_bits,
                    batch_size=4096,
                    candidates=("metal_swiglu", "metal_mlx_header_bf16"),
                    max_mismatches=4,
                )
                fp32 = report.candidate("metal_swiglu")
                header = report.candidate("metal_mlx_header_bf16")
                self.assertGreater(fp32.mismatches, 0)
                self.assertEqual(header.mismatches, 0)
                self.assertEqual(len(fp32.examples), 4)


if __name__ == "__main__":
    unittest.main()
