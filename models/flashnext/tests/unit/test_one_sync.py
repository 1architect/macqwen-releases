"""One-sync hand-off: the host route must equal the device route exactly."""
from __future__ import annotations

import unittest

import mlx.core as mx

from models.flashnext import adaptive_topk


class _Switch:
    """Stands in for StreamingSwitchGLU and records what it was handed."""

    metal_combines_scores = False

    def __init__(self):
        self._routed_host = None
        self.seen = []

    def __call__(self, x, inds):
        handed = self._routed_host
        self._routed_host = None
        self.seen.append((handed, inds, inds.reshape(-1).tolist()))
        return mx.zeros((*inds.shape, x.shape[-1]), dtype=x.dtype)


class _Block:
    top_k = 10
    _flashnext_layer_id = 7

    def __init__(self, experts, hidden):
        self.router = mx.random.normal((hidden, experts))
        self.switch_mlp = _Switch()

    def gate(self, x):
        return x @ self.router

    def shared_expert(self, x):
        return x

    def shared_expert_gate(self, x):
        return x[..., :1]


class OneSyncTests(unittest.TestCase):
    def setUp(self):
        self.threshold = adaptive_topk._THRESHOLD[0]
        self.one_sync = adaptive_topk._ONE_SYNC[0]
        adaptive_topk.set_threshold(0.85)

    def tearDown(self):
        adaptive_topk.set_threshold(self.threshold)
        adaptive_topk.set_one_sync(self.one_sync)

    def test_host_route_matches_device_route(self):
        mx.random.seed(11)
        block = _Block(64, 32)
        adaptive_topk.set_one_sync(True)
        for rows in (1, 3, 8):
            for _ in range(20):
                x = mx.random.normal((1, rows, 32)) * 3
                adaptive_topk._moe_call(block, x)
                handed, inds, device = block.switch_mlp.seen[-1]
                self.assertIsNotNone(handed, f"no hand-off for {rows} rows")
                self.assertIs(handed[1], inds)
                self.assertEqual(handed[0], device)

    def test_off_and_prefill_rows_hand_nothing(self):
        mx.random.seed(12)
        block = _Block(64, 32)
        adaptive_topk.set_one_sync(False)
        adaptive_topk._moe_call(block, mx.random.normal((1, 1, 32)))
        self.assertIsNone(block.switch_mlp.seen[-1][0])
        adaptive_topk.set_one_sync(True)
        adaptive_topk._moe_call(block, mx.random.normal((1, 16, 32)))
        self.assertIsNone(block.switch_mlp.seen[-1][0])


if __name__ == "__main__":
    unittest.main()
