from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.bonsai2.gemv_kernel import gemv_enabled, gemv_q2, install_gemv_hook


class GemvKernelTests(unittest.TestCase):
    def test_stays_off_without_the_flag(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BONSAI2_GEMV", None)
            self.assertFalse(gemv_enabled())
            x = mx.zeros((1, 512), dtype=mx.float16)
            w = mx.zeros((8, 32), dtype=mx.uint32)
            self.assertIsNone(gemv_q2(x, w, w, w))

    def test_rejects_non_decode_shapes(self):
        with patch.dict(os.environ, {"BONSAI2_GEMV": "1"}):
            x = mx.zeros((4, 512), dtype=mx.float16)
            w = mx.zeros((8, 32), dtype=mx.uint32)
            self.assertIsNone(gemv_q2(x, w, w, w))
            odd = mx.zeros((1, 100), dtype=mx.float16)
            wide = mx.zeros((7, 100), dtype=mx.uint32)
            self.assertIsNone(gemv_q2(odd, wide, wide, wide))

    def test_matches_stock_on_production_shapes(self):
        checkpoint = Path(
            os.environ.get("MACQWEN_MODEL_ROOT", "~/models")
        ).expanduser() / "Ternary-Bonsai-2-27B-mlx-2bit"
        if not (checkpoint / "model.safetensors").is_file():
            self.skipTest("needs the local Bonsai-2 checkpoint")
        import sys

        sys.path.insert(0, str(checkpoint / "runtime"))
        w = mx.load(str(checkpoint / "model.safetensors"))
        with patch.dict(os.environ, {"BONSAI2_GEMV": "1"}):
            mx.random.seed(0)
            for key, width in (
                ("language_model.model.layers.0.mlp.gate_proj", 5120),
                ("language_model.model.layers.0.mlp.down_proj", 17408),
            ):
                with self.subTest(key=key):
                    arrays = [w[key + "." + s] for s in ("weight", "scales", "biases")]
                    x = mx.random.normal((1, width)).astype(mx.float16)
                    mx.eval(x)
                    ref = mx.quantized_matmul(
                        x, *arrays, transpose=True, group_size=128, bits=2
                    )
                    got = gemv_q2(x, *arrays)
                    mx.eval(ref, got)
                    self.assertTrue(
                        bool(
                            (
                                np.asarray(ref).astype(np.float32)
                                == np.asarray(got).astype(np.float32)
                            ).all()
                        )
                    )

    def test_hook_routes_decode_calls_and_keeps_the_fallback(self):
        checkpoint = Path(
            os.environ.get("MACQWEN_MODEL_ROOT", "~/models")
        ).expanduser() / "Ternary-Bonsai-2-27B-mlx-2bit"
        if not (checkpoint / "runtime" / "runtime.py").is_file():
            self.skipTest("needs the local Bonsai-2 checkpoint runtime")
        import sys

        sys.path.insert(0, str(checkpoint / "runtime"))
        import runtime
        from models.bonsai2.gemv_kernel import uninstall_gemv_hook

        self.assertTrue(hasattr(runtime, "Packed"))
        with patch.dict(os.environ, {"BONSAI2_GEMV": "1"}):
            try:
                self.assertTrue(install_gemv_hook())
                self.assertTrue(runtime.Packed._bonsai2_gemv)
            finally:
                self.assertTrue(uninstall_gemv_hook())
                self.assertFalse(hasattr(runtime.Packed, "_bonsai2_gemv"))


if __name__ == "__main__":
    unittest.main()
