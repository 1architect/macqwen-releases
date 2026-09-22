from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
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
                    # Both arms must consume the same fp16-rounded input
                    # bytes; the production packed path does not receive the
                    # original unrounded fp32 sample.
                    row = generator.normal(size=shape).astype(np.float16)
                    signs = np.where(
                        generator.integers(0, 2, size=width), 1.0, -1.0
                    ).astype(np.float32)
                    flat = (row.astype(np.float32).reshape(-1, width) * signs).reshape(-1)
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
                    np.testing.assert_array_max_ulp(
                        actual, expected.astype(np.float32), maxulp=2
                    )

    def test_oversized_block_and_foreign_dtype_fall_back_to_stock(self):
        from models.bonsai2 import ternary_kernel as module

        seen = []

        def stock_fn(x, block, signs, inverse=False):
            seen.append(block)
            return mx.zeros((1, 2048), dtype=mx.float16)

        module._STOCK_FWHT = stock_fn
        try:
            with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
                wide = mx.zeros((1, 4096), dtype=mx.float16)
                signs = mx.zeros((1, 4096), dtype=mx.float32)
                fused_fwht(wide, signs, 2048)
                bfloat16 = mx.zeros((1, 2048), dtype=mx.bfloat16)
                fused_fwht(bfloat16, signs, 1024)
        finally:
            module._STOCK_FWHT = None
        # Both fell back instead of compiling an unsupported dispatch.
        self.assertEqual(seen, [2048, 1024])

    def test_fp32_kernel_preserves_type_and_matches_reference(self):
        with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
            generator = np.random.default_rng(20260920)
            shape = (2, 2048)
            row = generator.normal(size=shape).astype(np.float32)
            signs = np.where(
                generator.integers(0, 2, size=shape[-1]), 1.0, -1.0
            ).astype(np.float32)
            expected = np.stack([
                natural_fwht_blocked(values, 1024)
                for values in (row * signs).reshape(-1, shape[-1])
            ]).reshape(shape)
            counters = ternary_kernel.new_fwht_counters()
            counters["requested"] = True
            ternary_kernel.set_fwht_counters(counters)
            try:
                got = fused_fwht(mx.array(row), mx.array(signs), 1024)
                mx.eval(got)
            finally:
                ternary_kernel.set_fwht_counters(None)
            self.assertEqual(str(got.dtype), "mlx.core.float32")
            np.testing.assert_allclose(
                np.asarray(got), expected, rtol=2e-5, atol=2e-5
            )
            self.assertEqual(counters["selected"], 1)
            self.assertEqual(counters["fallbacks"], 0)
            self.assertEqual(counters["input_dtypes"].get("float32"), 1)

    def test_hook_redirects_forward_transform_and_keeps_inverse_stock(self):
        import tempfile

        calls = []
        fake = types.ModuleType("runtime")

        def stock(x, block, signs, inverse=False):
            calls.append((block, inverse))
            return x

        fake.fwht = stock
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "runtime").mkdir()
            fake.__file__ = str(checkpoint / "runtime" / "runtime.py")
            with patch.dict(sys.modules, {"runtime": fake}):
                with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
                    from models.bonsai2.ternary_kernel import (
                        restore_runtime_hooks,
                    )

                    ternary_kernel._STOCK_FWHT = None
                    try:
                        self.assertTrue(install_packed_hook(checkpoint))
                        self.assertIs(fake.fwht.__name__, "hooked")
                        # Odd width falls back to the saved stock implementation.
                        odd = mx.zeros((1, 100), dtype=mx.float16)
                        result = fake.fwht(odd, 1024, odd, inverse=False)
                        self.assertIsInstance(result, mx.array)
                        result = fake.fwht(odd, 1024, odd, inverse=True)
                        self.assertIsInstance(result, mx.array)
                    finally:
                        # Restore through the hook so no stale original
                        # survives keyed by this module id. A deleted
                        # entry would let a later recycled id restore
                        # another test's stock implementation.
                        self.assertTrue(restore_runtime_hooks(checkpoint))
                        ternary_kernel._STOCK_FWHT = None
                        self.assertIs(fake.fwht, stock)
                        self.assertFalse(hasattr(fake, "_bonsai2_fused"))
        self.assertEqual(calls, [(1024, False), (1024, True)])

    def test_share_hook_reuses_identical_calls(self):
        from models.bonsai2 import ternary_kernel as module
        from models.bonsai2.ternary_kernel import (
            arm_memo,
            disarm_memo,
            install_share_hook,
        )

        import tempfile

        calls = []
        fake = types.ModuleType("runtime")

        def stock(x, block, signs, inverse=False):
            calls.append((id(x), id(signs), block, inverse))
            return ("computed", block)

        fake.fwht = stock
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        checkpoint = Path(tmp.name)
        (checkpoint / "runtime").mkdir()
        fake.__file__ = str(checkpoint / "runtime" / "runtime.py")
        from models.bonsai2.ternary_kernel import restore_runtime_hooks

        try:
            with patch.dict(sys.modules, {"runtime": fake}):
                with patch.dict(os.environ, {"BONSAI2_SHARE_FWHT": "1"}):
                    try:
                        self.assertTrue(install_share_hook(checkpoint))
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
                        # One shared computation; same-width signs share
                        # the memo entry because the backend verifies
                        # their bytes match per checkpoint.
                        # Inverse transforms bypass the memo.
                        self.assertIs(first, second)
                        self.assertIs(first, third)
                        self.assertEqual(len(calls), 2)
                        # Disarmed hook passes straight through.
                        self.assertEqual(
                            fake.fwht(x, 1024, s), fake.fwht(x, 1024, s)
                        )
                        self.assertEqual(len(calls), 4)
                    finally:
                        # Restore through the hook so no stale original
                        # survives keyed by this module id. A deleted
                        # entry would let a later recycled id restore
                        # another test's stock implementation.
                        self.assertTrue(restore_runtime_hooks(checkpoint))
                        self.assertIs(fake.fwht, stock)
                        self.assertFalse(hasattr(fake, "_bonsai2_shared"))
        finally:
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

    def test_foreign_runtime_module_refuses_patching(self):
        import tempfile

        from models.bonsai2.ternary_kernel import restore_runtime_hooks

        fake = types.ModuleType("runtime")
        fake.fwht = lambda *args, **kwargs: "stock"
        with tempfile.TemporaryDirectory() as first:
            with tempfile.TemporaryDirectory() as second:
                (Path(first) / "runtime").mkdir()
                fake.__file__ = str(Path(first) / "runtime" / "runtime.py")
                with patch.dict(sys.modules, {"runtime": fake}):
                    with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
                        with self.assertRaisesRegex(RuntimeError, "foreign runtime"):
                            install_packed_hook(Path(second))
                self.assertEqual(fake.fwht.__name__, "<lambda>")

    def test_restore_returns_a_patched_module_to_stock(self):
        import tempfile

        from models.bonsai2.ternary_kernel import restore_runtime_hooks

        fake = types.ModuleType("runtime")

        def stock(x, block, signs, inverse=False):
            return "stock"

        fake.fwht = stock
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "runtime").mkdir()
            fake.__file__ = str(checkpoint / "runtime" / "runtime.py")
            with patch.dict(sys.modules, {"runtime": fake}):
                with patch.dict(os.environ, {"BONSAI2_FUSED_FWHT": "1"}):
                    self.assertTrue(install_packed_hook(checkpoint))
                    self.assertIs(fake.fwht.__name__, "hooked")
                    self.assertTrue(restore_runtime_hooks(checkpoint))
                    self.assertIs(fake.fwht, stock)
                    self.assertFalse(hasattr(fake, "_bonsai2_fused"))


class HookCompositionTests(unittest.TestCase):
    """The 2x2 matrix over share_fwht and fused_fwht.

    Composition is canonical: stock restores first, fused installs
    innermost, share outermost. Every cell asserts the exact active
    layering so construction order across backends cannot silently drop
    the share layer or leak flags through the process environment.
    """

    def setUp(self):
        from models.bonsai2 import ternary_kernel as module

        self.module = module
        self.saved = (
            module._INSTALLED,
            module._STOCK_FWHT,
            module._STOCK_OWNER,
            dict(module._ORIGINALS),
            module._MEMO,
        )
        module._INSTALLED = None
        module._STOCK_FWHT = None
        module._STOCK_OWNER = None
        module._ORIGINALS.clear()
        module._MEMO = None

    def tearDown(self):
        installed, stock, owner, originals, memo = self.saved
        self.module._INSTALLED = installed
        self.module._STOCK_FWHT = stock
        self.module._STOCK_OWNER = owner
        self.module._ORIGINALS.clear()
        self.module._ORIGINALS.update(originals)
        self.module._MEMO = memo

    def test_matrix_cells_install_exact_layers(self):
        import tempfile

        from models.bonsai2.ternary_kernel import (
            apply_runtime_hooks,
            arm_memo,
            disarm_memo,
        )

        cells = [
            # (fused, share, outermost name, fused flag, shared flag)
            (False, False, "stock", False, False),
            (True, False, "hooked", True, False),
            (False, True, "shared", False, True),
            (True, True, "shared", True, True),
        ]
        for fused, share, name, want_fused, want_shared in cells:
            with self.subTest(fused=fused, share=share):
                calls = []
                fake = types.ModuleType("runtime")

                def stock(x, block, signs, inverse=False):
                    calls.append((block, inverse))
                    return x

                fake.fwht = stock
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory)
                    (checkpoint / "runtime").mkdir()
                    fake.__file__ = str(checkpoint / "runtime" / "runtime.py")
                    self.module._INSTALLED = None
                    self.module._ORIGINALS.clear()
                    with patch.dict(sys.modules, {"runtime": fake}):
                        with patch.dict(os.environ, {}, clear=True):
                            apply_runtime_hooks(
                                checkpoint, fused=fused, share=share
                            )
                            self.assertIs(fake.fwht.__name__, name)
                            self.assertEqual(
                                hasattr(fake, "_bonsai2_fused"), want_fused
                            )
                            self.assertEqual(
                                hasattr(fake, "_bonsai2_shared"), want_shared
                            )
                            self.assertEqual(
                                os.environ.get("BONSAI2_FUSED_FWHT"),
                                "1" if fused else None,
                            )
                            self.assertEqual(
                                os.environ.get("BONSAI2_SHARE_FWHT"),
                                "1" if share else None,
                            )
                            if share:
                                arm_memo()
                                try:
                                    # Keep this composition test on the
                                    # unsupported path; the real fp32 kernel
                                    # is covered by the numerical test above.
                                    x = mx.zeros((1, 2048), dtype=mx.bfloat16)
                                    s = mx.zeros((2048,))
                                    first = fake.fwht(x, 1024, s)
                                    second = fake.fwht(x, 1024, s)
                                finally:
                                    disarm_memo()
                                # Share outermost: one computation for two
                                # identical armed calls; the count (not
                                # identity, the stock echoes its input)
                                # proves the memo hit.
                                self.assertEqual(len(calls), 1)
                            # Repeat construction is a verified no-op.
                            current = fake.fwht
                            apply_runtime_hooks(
                                checkpoint, fused=fused, share=share
                            )
                            self.assertIs(fake.fwht, current)

    def test_switching_composition_restores_stock_first(self):
        import tempfile

        from models.bonsai2.ternary_kernel import (
            apply_runtime_hooks,
            arm_memo,
            disarm_memo,
        )

        calls = []
        fake = types.ModuleType("runtime")

        def stock(x, block, signs, inverse=False):
            calls.append((block, inverse))
            return x

        fake.fwht = stock
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "runtime").mkdir()
            fake.__file__ = str(checkpoint / "runtime" / "runtime.py")
            with patch.dict(sys.modules, {"runtime": fake}):
                with patch.dict(os.environ, {}, clear=True):
                    apply_runtime_hooks(checkpoint, fused=False, share=True)
                    self.assertIs(fake.fwht.__name__, "shared")
                    # Fused after share must not strand the share layer:
                    # the composition rebuilds from stock with share off.
                    apply_runtime_hooks(checkpoint, fused=True, share=False)
                    self.assertIs(fake.fwht.__name__, "hooked")
                    self.assertFalse(hasattr(fake, "_bonsai2_shared"))
                    arm_memo()
                    try:
                        x = mx.zeros((1, 2048), dtype=mx.bfloat16)
                        s = mx.zeros((2048,))
                        fake.fwht(x, 1024, s)
                        fake.fwht(x, 1024, s)
                    finally:
                        disarm_memo()
                    # No memo layer: both calls reach the fallback stock.
                    self.assertEqual(len(calls), 2)
                    # Back to stock when both flags are off.
                    apply_runtime_hooks(checkpoint, fused=False, share=False)
                    self.assertIs(fake.fwht, stock)

    def test_older_hook_keeps_its_own_stock(self):
        import tempfile

        from models.bonsai2.ternary_kernel import apply_runtime_hooks

        calls_a, calls_b = [], []
        fake_a = types.ModuleType("runtime")
        fake_b = types.ModuleType("runtime")

        def stock_a(x, block, signs, inverse=False):
            calls_a.append(inverse)
            return x

        def stock_b(x, block, signs, inverse=False):
            calls_b.append(inverse)
            return x

        fake_a.fwht = stock_a
        fake_b.fwht = stock_b
        with tempfile.TemporaryDirectory() as first:
            with tempfile.TemporaryDirectory() as second:
                checkpoint_a = Path(first)
                checkpoint_b = Path(second)
                (checkpoint_a / "runtime").mkdir()
                (checkpoint_b / "runtime").mkdir()
                fake_a.__file__ = str(checkpoint_a / "runtime" / "runtime.py")
                fake_b.__file__ = str(checkpoint_b / "runtime" / "runtime.py")
                with patch.dict(os.environ, {}, clear=True):
                    with patch.dict(sys.modules, {"runtime": fake_a}):
                        apply_runtime_hooks(checkpoint_a, fused=True)
                        hooked_a = fake_a.fwht
                    with patch.dict(sys.modules, {"runtime": fake_b}):
                        apply_runtime_hooks(checkpoint_b, fused=True)
                    # A later install replaced the global stock, but the
                    # older hook captured its own module's implementation.
                    hooked_a("x", 1024, "s", inverse=True)
                    self.assertEqual(calls_a, [True])
                    self.assertEqual(calls_b, [])

    def test_restore_refuses_a_foreign_runtime_module(self):
        import tempfile

        from models.bonsai2.ternary_kernel import restore_runtime_hooks

        fake = types.ModuleType("runtime")
        fake.fwht = lambda *args, **kwargs: "stock"
        with tempfile.TemporaryDirectory() as first:
            with tempfile.TemporaryDirectory() as second:
                (Path(first) / "runtime").mkdir()
                fake.__file__ = str(Path(first) / "runtime" / "runtime.py")
                with patch.dict(sys.modules, {"runtime": fake}):
                    with self.assertRaisesRegex(RuntimeError, "foreign runtime"):
                        restore_runtime_hooks(Path(second))


if __name__ == "__main__":
    unittest.main()
