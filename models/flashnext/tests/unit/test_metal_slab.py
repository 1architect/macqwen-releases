"""Test unified single-pass resident slab execution in custom Metal runtime."""
from __future__ import annotations

import unittest
import os
import numpy as np
import mlx.core as mx

from models.flashnext.metal_runtime import MetalMoEExecutor, probe_capabilities

class TestMetalUnifiedSlab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("GITHUB_ACTIONS") == "true":
            raise unittest.SkipTest("Metal JIT integration is unavailable on GitHub Actions")

    def test_slab_route_encoding(self):
        slab_idx = 5
        encoded = 0x80000000 | slab_idx
        self.assertTrue(bool(encoded & 0x80000000))
        self.assertEqual(encoded & 0x7FFFFFFF, 5)

        streamed_idx = 3
        self.assertFalse(bool(streamed_idx & 0x80000000))
        self.assertEqual(streamed_idx & 0x7FFFFFFF, 3)

    def test_global_slab_allocation(self):
        import json, tempfile, os
        from models.flashnext.expert_cache import get_global_slab_allocation, _GLOBAL_SLAB_CACHE

        mock_scores = {
            "0": [[10, 0.5], [11, 0.1]],
            "5": [[405, 1.96], [406, 0.8]],
            "40": [[118, 1.79]],
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump({"ranked_scores": mock_scores}, f)
            temp_path = f.name

        orig = os.environ.get("FLASHNEXT_PIN_CACHE")
        os.environ["FLASHNEXT_PIN_CACHE"] = temp_path
        _GLOBAL_SLAB_CACHE.clear()
        try:
            # Request budget of 3 slots: top 3 global scores are (5, 405) [1.96], (40, 118) [1.79], (5, 406) [0.8]
            alloc = get_global_slab_allocation(3)
            self.assertEqual(alloc.get(5), [405, 406])
            self.assertEqual(alloc.get(40), [118])
            self.assertNotIn(0, alloc)
        finally:
            if orig is not None:
                os.environ["FLASHNEXT_PIN_CACHE"] = orig
            else:
                os.environ.pop("FLASHNEXT_PIN_CACHE", None)
            _GLOBAL_SLAB_CACHE.clear()
            os.remove(temp_path)

    def test_scratchless_fused_down_bfloat16(self):
        caps = probe_capabilities()
        if not caps["available"]:
            raise unittest.SkipTest(caps["reason"])
        hidden = 2560
        inter = 640
        tokens = 1
        slots = 8

        rng = np.random.default_rng(42)
        w = mx.array(rng.integers(0, 15, size=(16, hidden, inter // 8), dtype=np.uint32))
        s = mx.array(rng.normal(0.1, 0.01, size=(16, hidden, inter // 32)).astype(np.float16)).astype(mx.bfloat16)
        b = mx.array(rng.normal(0.0, 0.01, size=(16, hidden, inter // 32)).astype(np.float16)).astype(mx.bfloat16)
        down = (w, s, b)

        x_act = mx.array(rng.normal(0.0, 1.0, size=(tokens, slots, inter)).astype(np.float32)).astype(mx.bfloat16)
        routes = mx.array([[0, 2, 4, 6, 8, 10, 12, 14]], dtype=mx.uint32)
        scores = mx.array([[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=mx.float32)

        executor = MetalMoEExecutor(expert_count=16, hidden_size=hidden, top_k=slots)
        from models.flashnext.metal_runtime import _as_projection
        down_proj = _as_projection(down)
        out = executor._metal_fused_down_combine(x_act, routes, scores, down_proj, hidden)
        mx.eval(out)
        self.assertEqual(out.shape, (tokens, hidden))
        self.assertEqual(out.dtype, mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
