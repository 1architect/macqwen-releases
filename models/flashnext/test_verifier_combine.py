"""Contract tests: verifier streamed-MoE combine mirrors production."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from models.flashnext.adaptive_topk import (
    apply as patch_topk,
    set_renorm_blend,
    set_threshold,
)
from models.flashnext.qwen4_verifier import Qwen4ExactSpeculativeVerifier


class FakeSwitch:
    def __init__(self, combines=True, fused_shared=False):
        self.calls = []
        self.metal_combines_scores = combines
        self._last_fused_shared = fused_shared

    def __call__(self, hidden, indices, **kwargs):
        self.calls.append((tuple(indices.shape), sorted(kwargs)))
        batch, length, width = hidden.shape
        return mx.zeros((batch, length, width))


def fake_moe(combines=True, fused_shared=False):
    return SimpleNamespace(
        gate=nn.Linear(8, 6, bias=False),
        top_k=2,
        switch_mlp=FakeSwitch(combines, fused_shared),
        shared_expert=nn.Linear(8, 8, bias=False),
        shared_expert_gate=nn.Linear(8, 8, bias=False),
    )


class VerifierCombineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_topk()
        set_threshold(1.0)
        set_renorm_blend(1.0)
        cls.verifier = Qwen4ExactSpeculativeVerifier()

    def hidden(self):
        return mx.zeros((1, 2, 8))

    def test_production_combine_receives_scores_and_shared_y(self):
        moe = fake_moe(combines=True)
        with patch.dict(os.environ, {"FLASHNEXT_FUSED_SHARED": "1",
                                     "FLASHNEXT_FUSED_SHARED_PARTS": "0"}):
            out = self.verifier._adaptive_moe(moe, self.hidden())
        mx.eval(out)
        self.assertEqual(len(moe.switch_mlp.calls), 1)
        _shape, kwargs = moe.switch_mlp.calls[0]
        self.assertIn("scores", kwargs)
        self.assertIn("shared_y", kwargs)
        # Switch returned zeros and reported no fused shared, so the
        # result is exactly one shared addition, not a second reduction.
        self.assertEqual(tuple(out.shape), (1, 2, 8))

    def test_fused_shared_true_skips_duplicate_add(self):
        moe = fake_moe(combines=True, fused_shared=True)
        with patch.dict(os.environ, {"FLASHNEXT_FUSED_SHARED": "1",
                                     "FLASHNEXT_FUSED_SHARED_PARTS": "0"}):
            out = self.verifier._adaptive_moe(moe, self.hidden())
        mx.eval(out)
        self.assertEqual(float(mx.max(mx.abs(out)).item()), 0.0)

    def test_fused_shared_parts_passes_shared_pair(self):
        moe = fake_moe(combines=True)
        with patch.dict(os.environ, {"FLASHNEXT_FUSED_SHARED": "1",
                                     "FLASHNEXT_FUSED_SHARED_PARTS": "1"}):
            self.verifier._adaptive_moe(moe, self.hidden())
        _shape, kwargs = moe.switch_mlp.calls[0]
        self.assertIn("shared", kwargs)
        self.assertIn("shared_gate", kwargs)

    def test_generic_fallback_unchanged(self):
        moe = fake_moe(combines=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_FUSED_SHARED", None)
            os.environ.pop("FLASHNEXT_FUSED_SHARED_PARTS", None)
            out = self.verifier._adaptive_moe(moe, self.hidden())
        mx.eval(out)
        _shape, kwargs = moe.switch_mlp.calls[0]
        self.assertEqual(kwargs, [])
        self.assertEqual(tuple(out.shape), (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
