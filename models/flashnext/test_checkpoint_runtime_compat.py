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
    def test_resident_slab_drops_stale_preload_ids(self):
        store = _ShapeStore()
        slab = expert_cache.ResidentSlab(
            store, "layer.switch_mlp.gate_proj", 3,
            initial_experts=[4, 2, 2, -1, 0],
        )

        self.assertEqual(slab.slot, {0: 1, 2: 0})
        self.assertEqual(store.calls, [
            ("layer.switch_mlp.gate_proj.weight", [2, 0]),
            ("layer.switch_mlp.gate_proj.scales", [2, 0]),
            ("layer.switch_mlp.gate_proj.biases", [2, 0]),
        ])

    def test_warm_history_drops_ids_from_wider_checkpoint(self):
        store = SimpleNamespace()
        projection = SimpleNamespace(
            slab=None,
            cache=SimpleNamespace(store=store, prefix="layer.switch_mlp.gate_proj"),
        )
        switch_mlp = SimpleNamespace(
            gate_proj=SimpleNamespace(
                num_experts=3,
                slab=None,
                cache=SimpleNamespace(
                    store=store, prefix="layer.switch_mlp.gate_proj"
                ),
            ),
            up_proj=projection,
            down_proj=projection,
        )

        with mock.patch.object(expert_cache, "_WARM_ON", True), \
             mock.patch.object(expert_cache, "_LAST", {7: [287, 2, 512, 0]}), \
             mock.patch.object(expert_cache._WARM, "submit") as submit:
            expert_cache.warm_layer(switch_mlp, 7)

        self.assertEqual(submit.call_count, 3)
        self.assertEqual(submit.call_args_list[0].args[3], [2, 0])

    def test_reap_g64_uses_reference_path_with_default_metal_setting(self):
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
                "FLASHNEXT_SLAB": "0",
                "FLASHNEXT_SLAB_GLOBAL": "60",
                "FLASHNEXT_SLAB_PACK": "1",
                "FLASHNEXT_SLAB_POLICY": "skew",
            },
            clear=False,
        ):
            with mock.patch.object(slab_pack, "get_or_create_slab_pack") as pack:
                switch_mlp = expert_cache.StreamingSwitchGLU(
                    store, prefix, 64, 4, "affine", 0,
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
