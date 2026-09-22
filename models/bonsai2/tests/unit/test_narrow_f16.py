from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.bonsai2.gemv_kernel import (
    NARROW_GATEUP_WEIGHT_SHAPE,
    install_narrow_hook,
    new_narrow_counters,
    uninstall_narrow_hook,
)
from models.bonsai2.qmm_metadata import install as install_qmm, new_counters


class _FakePacked:
    def __init__(self, rows, width, block=512):
        self.weight = mx.array(
            np.arange(rows * (width // 16), dtype=np.uint32).reshape(
                rows, width // 16
            )
        )
        self.scales = mx.array(
            np.linspace(0.2, 0.8, rows * (width // 128), dtype=np.float16).reshape(
                rows, width // 128
            )
        )
        self.biases = mx.array(
            np.linspace(-0.2, 0.2, rows * (width // 128), dtype=np.float16).reshape(
                rows, width // 128
            )
        )
        self.embedding = False
        self.block = block
        self.signs = mx.ones((width,), dtype=mx.float32)

    def __call__(self, x):
        return ("wrapped", x.shape)


def _fake_packed(rows, width, block=512):
    return _FakePacked(rows, width, block=block)


class NarrowExclusionTests(unittest.TestCase):
    def test_gateup_shape_excluded_from_preparation(self):
        gate = _fake_packed(17408, 5120)
        other = _fake_packed(64, 512)
        model = types.SimpleNamespace(
            named_modules=lambda: [("mlp.gate_proj", gate), ("other", other)]
        )
        runtime = types.ModuleType("runtime")
        runtime.Packed = type(gate)
        counters = new_counters()
        with patch.dict(sys.modules, {"runtime": runtime}):
            stats = install_qmm(
                model, True, counters,
                exclude_weight_shapes=(NARROW_GATEUP_WEIGHT_SHAPE,),
            )
            self.assertEqual(stats["prepared_modules"], 1)
            self.assertEqual(
                [e["name"] for e in stats["excluded_module_manifest"]],
                ["mlp.gate_proj"],
            )
            self.assertEqual(str(gate.scales.dtype), "mlx.core.float16")
            self.assertEqual(str(other.scales.dtype), "mlx.core.float32")
            install_qmm(model, False, counters)


class NarrowHookTests(unittest.TestCase):
    def _runtime(self, packed_type):
        runtime = types.ModuleType("runtime")
        runtime.Packed = packed_type
        runtime.fwht = lambda x, block, signs: x
        return runtime

    def test_routes_gate_shape_and_falls_back(self):
        gate = _fake_packed(17408, 5120)
        down = _fake_packed(5120, 17408)
        runtime = self._runtime(type(gate))
        counters = new_narrow_counters()
        with patch.dict(sys.modules, {"runtime": runtime}):
            self.assertTrue(install_narrow_hook(counters))
            try:
                # Production activations are FP32; the narrow tail needs them.
                x = mx.array(np.arange(5120, dtype=np.float32).reshape(1, 1, 5120))
                mx.eval(x)
                out = runtime.Packed.__call__(gate, x)
                self.assertEqual(tuple(out.shape), (1, 1, 17408))
                self.assertEqual(str(out.dtype), "mlx.core.float32")
                self.assertEqual(counters["narrow_calls"], 1)
                self.assertEqual(counters["narrow_fallbacks"], 0)
                # fp16 activation falls back: narrow tail is FP32-input only
                xf = mx.array(np.arange(5120, dtype=np.float16).reshape(1, 1, 5120))
                mx.eval(xf)
                tag, _shape = runtime.Packed.__call__(gate, xf)
                self.assertEqual(tag, "wrapped")
                # down shape falls back to the wrapped path
                xd = mx.array(np.arange(17408, dtype=np.float32).reshape(1, 1, 17408))
                mx.eval(xd)
                tag, _shape = runtime.Packed.__call__(down, xd)
                self.assertEqual(tag, "wrapped")
                reasons = counters["fallback_reasons"]
                self.assertEqual(reasons.get("shape_or_dtype"), 1)
                self.assertEqual(reasons.get("kernel_none"), 1)
            finally:
                self.assertTrue(uninstall_narrow_hook())

    def test_multitoken_falls_back(self):
        gate = _fake_packed(17408, 5120)
        runtime = self._runtime(type(gate))
        counters = new_narrow_counters()
        with patch.dict(sys.modules, {"runtime": runtime}):
            self.assertTrue(install_narrow_hook(counters))
            try:
                x = mx.zeros((1, 4, 5120), dtype=mx.float16)
                mx.eval(x)
                tag, _shape = runtime.Packed.__call__(gate, x)
                self.assertEqual(tag, "wrapped")
                self.assertEqual(counters["narrow_calls"], 0)
                self.assertEqual(counters["fallback_reasons"].get("kernel_none"), 1)
            finally:
                self.assertTrue(uninstall_narrow_hook())


if __name__ == "__main__":
    unittest.main()
