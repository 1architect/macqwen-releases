"""Regression tests for checkpoint-specific FlashNext runtime compatibility."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from models.flashnext import expert_cache
from models.flashnext import slab_pack


class _ShapeStore:
    def __init__(self):
        self.calls = []

    def shape(self, name):
        return (3, 2, 2) if name.endswith(".weight") else (3, 2, 1)

    def rows(self, name, experts):
        self.calls.append((name, list(experts)))
        return np.asarray(experts, dtype=np.float32)


class CheckpointRuntimeCompatibilityTests(unittest.TestCase):
    def test_reap_g64_uses_reference_path_with_explicit_opt_out(self):
        prefix = "language_model.model.layers.1.mlp.switch_mlp"
        shapes = {
            f"{prefix}.gate_proj.weight": (288, 640, 320),
            f"{prefix}.gate_proj.scales": (288, 640, 40),
            f"{prefix}.gate_proj.biases": (288, 640, 40),
            f"{prefix}.up_proj.weight": (288, 640, 320),
            f"{prefix}.up_proj.scales": (288, 640, 40),
            f"{prefix}.up_proj.biases": (288, 640, 40),
            f"{prefix}.down_proj.weight": (288, 2560, 80),
            f"{prefix}.down_proj.scales": (288, 2560, 10),
            f"{prefix}.down_proj.biases": (288, 2560, 10),
        }
        store = SimpleNamespace(
            refs={},
            shape=lambda name: shapes[name],
        )

        with mock.patch.dict(
            expert_cache.os.environ,
            {
                "FLASHNEXT_METAL_RUNTIME": "1",
                "FLASHNEXT_METAL_G64": "0",
                "FLASHNEXT_SLAB": "0",
                "FLASHNEXT_SLAB_GLOBAL": "60",
                "FLASHNEXT_SLAB_PACK": "1",
                "FLASHNEXT_SLAB_POLICY": "skew",
            },
            clear=False,
        ):
            with mock.patch.object(slab_pack, "get_or_create_slab_pack") as pack:
                switch_mlp = expert_cache.StreamingSwitchGLU(
                    store, prefix, 64, 4, "affine",
                    activation=lambda value: value,
                    layer_id=1,
                )

        self.assertFalse(switch_mlp.metal_runtime_capable)
        self.assertFalse(switch_mlp.metal_combines_scores)
        self.assertFalse(pack.called)
        self.assertIsNone(switch_mlp.slab_pack)
        self.assertEqual(switch_mlp.gate_proj.group_size, 64)
        self.assertEqual(switch_mlp.gate_proj.num_experts, 288)


if __name__ == "__main__":
    unittest.main()
