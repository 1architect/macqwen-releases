"""Compiled hyper-connection glue must match the plain chain bit for bit."""
from __future__ import annotations

import types
import unittest

import mlx.core as mx
import mlx.nn as nn

from models.flashnext import compile_glue, patch_rmsnorm


class CompileGlueTests(unittest.TestCase):
    def setUp(self):
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual

        patch_rmsnorm.apply()
        config = types.SimpleNamespace(
            hc_count=4, hidden_size=256, rms_norm_eps=1e-6, hc_lowrank=32,
        )
        mx.random.seed(3)
        self.hc = Qwen4ExpGatedResidual(config)
        self.hc.update(nn.utils.tree_map(
            lambda p: (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16),
            self.hc.parameters(),
        ))
        nn.quantize(self.hc, group_size=32, bits=4,
                    class_predicate=lambda _p, m: isinstance(m, nn.Linear))
        self.x = (mx.random.normal((1, 1, 1024)) * 2).astype(mx.bfloat16)

    def test_inject_then_hc_matches(self):
        branch = (mx.random.normal((1, 1, 256))).astype(mx.bfloat16)
        weights = mx.random.uniform(shape=(1, 1, 4)).astype(mx.bfloat16)
        plain_hidden = compile_glue._inject(self.x, branch, weights)
        compiled_hidden = compile_glue._compiled_inject(self.x, branch, weights)
        self.assertTrue(mx.array_equal(plain_hidden, compiled_hidden).item())


if __name__ == "__main__":
    unittest.main()
