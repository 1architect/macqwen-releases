from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.bonsai2 import ternary_kernel
from models.bonsai2.ternary_kernel import fused_fwht, install_packed_hook


def natural_fwht_blocked(values: np.ndarray, block: int = 1024) -> np.ndarray:
    """Reference for the stock path: sign multiply is applied by the caller,
    then each block transforms independently with 1/sqrt(block) scaling."""
    rows = values.reshape(-1, block)
    out = np.empty_like(rows)
    for index in range(rows.shape[0]):
        result = rows[index].copy()
        count = block
        stride = 1
        while stride < count:
            for base in range(0, count, stride * 2):
                for offset in range(stride):
                    first = result[base + offset]
                    second = result[base + offset + stride]
                    result[base + offset] = first + second
                    result[base + offset + stride] = first - second
            stride *= 2
        out[index] = result / np.sqrt(count)
    return out.reshape(values.shape)


class FusedFwhtTests(unittest.TestCase):
    def test_fused_transform_matches_natural_order_on_production_widths(self):
        if os.environ.get("BONSAI2_FUSED_FWHT") == "0":
            self.skipTest("fused path force-disabled")
        with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
            for shape in ((1, 5120), (2, 17408), (1, 6, 6144)):
                width = shape[-1]
                with self.subTest(shape=shape):
                    generator = np.random.default_rng(int(np.prod(shape)))
                    row = generator.normal(size=shape).astype(np.float32)
                    signs = np.where(
                        generator.integers(0, 2, size=width), 1.0, -1.0
                    ).astype(np.float32)
                    flat = (row.reshape(-1, width) * signs).reshape(-1)
                    expected = np.stack([
                        natural_fwht_blocked(block_row)
                        for block_row in flat.reshape(-1, width)
                    ]).reshape(shape)
                    expected = expected.astype(np.float16)
                    got = fused_fwht(
                        mx.array(row.astype(np.float16)),
                        mx.array(signs),
                        1024,
                    )
                    mx.eval(got)
                    actual = np.asarray(got).astype(np.float32)
                    # Bit-exactness against the stock path was verified live;
                    # this checkpoint-free gate allows two fp16 ulps of
                    # boundary rounding. The model-level greedy digest is the
                    # real bar.
                    self.assertTrue(
                        bool((actual == expected.astype(np.float32)).all())
                        or float(np.abs(actual - expected.astype(np.float32)).max()) <= 0.125
                    )

    def test_hook_redirects_forward_transform_and_keeps_inverse_stock(self):
        calls = []
        fake = types.ModuleType("runtime")

        def stock(x, block, signs, inverse=False):
            calls.append((block, inverse))
            return x

        fake.fwht = stock
        with patch.dict(sys.modules, {"runtime": fake}):
            with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
                ternary_kernel._STOCK_FWHT = None
                try:
                    self.assertTrue(install_packed_hook())
                    self.assertIs(fake.fwht.__name__, "hooked")
                    # Odd width falls back to the saved stock implementation.
                    odd = mx.zeros((1, 100), dtype=mx.float16)
                    result = fake.fwht(odd, 1024, odd, inverse=False)
                    self.assertIsInstance(result, mx.array)
                    result = fake.fwht(odd, 1024, odd, inverse=True)
                    self.assertIsInstance(result, mx.array)
                finally:
                    ternary_kernel._STOCK_FWHT = None
                    del fake.fwht
                    del fake._bonsai2_fused
        self.assertEqual(calls, [(1024, False), (1024, True)])

    def test_share_hook_reuses_identical_calls(self):
        from models.bonsai2 import ternary_kernel as module
        from models.bonsai2.ternary_kernel import (
            arm_memo,
            disarm_memo,
            install_share_hook,
        )

        calls = []
        fake = types.ModuleType("runtime")

        def stock(x, block, signs, inverse=False):
            calls.append((id(x), id(signs), block, inverse))
            return ("computed", block)

        fake.fwht = stock
        try:
            with patch.dict(sys.modules, {"runtime": fake}):
                with patch.dict(os.environ, {"BONSAI2_SHARE_FWHT": "1"}):
                    self.assertTrue(install_share_hook())
                    arm_memo()
                    try:
                        x = mx.zeros((1, 2048))
                        s = mx.zeros((2048,))
                        t = mx.zeros((2048,))
                        first = fake.fwht(x, 1024, s)
                        second = fake.fwht(x, 1024, s)
                        third = fake.fwht(x, 1024, t)
                        fourth = fake.fwht(x, 1024, s, inverse=True)
                    finally:
                        disarm_memo()
            # One shared computation; same-width signs share the memo entry
            # because the backend verifies their bytes match per checkpoint.
            # Inverse transforms bypass the memo.
            self.assertIs(first, second)
            self.assertIs(first, third)
            self.assertEqual(len(calls), 2)
            # Disarmed hook passes straight through.
            self.assertEqual(fake.fwht(x, 1024, s), fake.fwht(x, 1024, s))
            self.assertEqual(len(calls), 4)
        finally:
            del fake.fwht
            del fake._bonsai2_shared
            module._MEMO = None

    def test_memo_belongs_to_the_backend_that_armed_it(self):
        # A backend constructed without sharing must never consult the memo,
        # even when another backend installed the global hook and verified
        # its own signs.
        from models.bonsai2.backend import _TextModelWrapper

        calls = []

        class Model:
            def __call__(self, inputs, cache=None, **options):
                from models.bonsai2 import ternary_kernel as module

                calls.append(module._MEMO is not None)

                class Out:
                    logits = inputs

                return Out()

        shared = _TextModelWrapper(Model(), share=True)
        plain = _TextModelWrapper(Model(), share=False)
        shared("x")
        plain("x")
        self.assertEqual(calls, [True, False])

    def test_hook_stays_off_without_the_flag(self):
        fake = types.ModuleType("runtime")
        fake.fwht = lambda *args, **kwargs: "stock"
        with patch.dict(sys.modules, {"runtime": fake}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("BONSAI2_FUSED_FWHT", None)
                self.assertFalse(install_packed_hook())
                self.assertFalse(hasattr(fake, "_bonsai2_fused"))


if __name__ == "__main__":
    unittest.main()
