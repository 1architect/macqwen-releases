"""Checkpoint-free tests for the opt-in Q4/G64 integration boundary."""
from __future__ import annotations

from types import SimpleNamespace
import json
import tempfile
from collections import Counter
import unittest
from unittest import mock

from models.flashnext import expert_cache
from models.flashnext import routing


class _G64Store:
    refs = {}

    def shape(self, name):
        if name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
            return (288, 640, 320)
        if name.endswith("down_proj.weight"):
            return (288, 2560, 80)
        if name.endswith("gate_proj.scales") or name.endswith("up_proj.scales"):
            return (288, 640, 40)
        if name.endswith("down_proj.scales"):
            return (288, 2560, 10)
        if name.endswith("gate_proj.biases") or name.endswith("up_proj.biases"):
            return (288, 640, 40)
        if name.endswith("down_proj.biases"):
            return (288, 2560, 10)
        raise KeyError(name)


class G64IntegrationTests(unittest.TestCase):
    def _make(self, *, g64_ready=True):
        with mock.patch("models.flashnext.metal_runtime.G64_RUNTIME_READY", g64_ready):
            return expert_cache.StreamingSwitchGLU(
                _G64Store(),
                "language_model.model.layers.1.mlp.switch_mlp",
                64, 4, "affine", 0,
                activation=lambda value: value,
                layer_id=1,
            )

    def test_g64_reference_requires_explicit_opt_out(self):
        with mock.patch.dict(
            expert_cache.os.environ,
            {
                "FLASHNEXT_METAL_RUNTIME": "1",
                "FLASHNEXT_METAL_G64": "0",
                "FLASHNEXT_SLAB_PACK": "1",
                "FLASHNEXT_SLAB_GLOBAL": "60",
            }, clear=False,
        ):
            switch = self._make()
        self.assertFalse(switch.metal_runtime_capable)
        self.assertFalse(switch.metal_combines_scores)
        self.assertIsNone(switch.slab_pack)

    def test_g64_custom_runtime_is_the_default(self):
        with mock.patch.dict(
            expert_cache.os.environ,
            {
                "FLASHNEXT_METAL_RUNTIME": "1",
                "FLASHNEXT_SLAB_PACK": "0",
                "FLASHNEXT_SLAB_GLOBAL": "0",
            }, clear=False,
        ):
            expert_cache.os.environ.pop("FLASHNEXT_METAL_G64", None)
            switch = self._make()
            self.assertTrue(switch.metal_runtime_capable)
            self.assertTrue(switch.metal_combines_scores)

    def test_g64_custom_runtime_rejects_unready_opt_in(self):
        with mock.patch.dict(
            expert_cache.os.environ,
            {
                "FLASHNEXT_METAL_RUNTIME": "1",
                "FLASHNEXT_METAL_G64": "1",
                "FLASHNEXT_SLAB_PACK": "0",
                "FLASHNEXT_SLAB_GLOBAL": "0",
            }, clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "full-model digest gate failed"):
                self._make(g64_ready=False)

    def test_g64_pack_fails_closed_without_checkpoint_history(self):
        with mock.patch.dict(
            expert_cache.os.environ,
            {
                "FLASHNEXT_METAL_RUNTIME": "1",
                "FLASHNEXT_METAL_G64": "1",
                "FLASHNEXT_SLAB_G64": "1",
                "FLASHNEXT_SLAB_PACK": "1",
                "FLASHNEXT_SLAB_GLOBAL": "60",
                "FLASHNEXT_PIN_CACHE": "/nonexistent/g64-pins.json",
            }, clear=False,
        ), mock.patch.object(expert_cache, "get_skew_slab_allocation", return_value={}), mock.patch(
            "models.flashnext.slab_pack.get_or_create_slab_pack"
        ) as make_pack:
            switch = self._make()
        self.assertTrue(switch.metal_runtime_capable)
        self.assertIsNone(switch.slab_pack)
        self.assertIn("checkpoint-specific", switch._slab_pack_disabled_reason)
        make_pack.assert_not_called()

    def test_g64_does_not_fall_back_to_resident_slab_for_old_history(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(
                {"group_size": 32, "layers": {"1": [3, 7]},
                 "ranked_scores": {}, "ranked_counts": {}}, handle
            )
            handle.flush()
            with mock.patch.dict(
                expert_cache.os.environ,
                {
                    "FLASHNEXT_METAL_RUNTIME": "1",
                    "FLASHNEXT_METAL_G64": "1",
                    "FLASHNEXT_SLAB_G64": "1",
                    "FLASHNEXT_SLAB_PACK": "1",
                    "FLASHNEXT_SLAB_GLOBAL": "60",
                    "FLASHNEXT_PIN_CACHE": handle.name,
                }, clear=False,
            ):
                switch = self._make()
        self.assertIsNone(switch.gate_proj.slab)
        self.assertIn("does not declare Q4/G64", switch._slab_pack_disabled_reason)

    def test_fresh_pin_profile_records_g64_checkpoint_provenance(self):
        store = SimpleNamespace(
            _read_mode="pread",
            _flashnext_requested_read_mode=None,
            dir="/checkpoint",
            unpin_all=lambda: None,
            shape=lambda name: (
                (288, 2560, 80)
                if name.endswith("down_proj.weight")
                else (288, 640, 320)
                if name.endswith("weight")
                else (288, 2560, 10)
                if name.endswith("down_proj.scales")
                else (288, 640, 40)
            ),
        )
        profile = routing.RoutingProfile("standard", store, None)
        profile.pinned = {1: {2}}
        profile.candidates = {1: Counter({2: 0.9})}
        profile.pinned_signature = "new"
        profile._checkpoint_identity = "g64-id"
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                routing.os.environ,
                {
                    "FLASHNEXT_PIN_CACHE": f"{directory}/pins.json",
                    "FLASHNEXT_METAL_G64": "1",
                }, clear=False
        ):
            profile.save_pins()
            with open(f"{directory}/pins.json") as handle:
                saved = json.load(handle)
        self.assertEqual(saved["group_size"], 64)
        self.assertEqual(saved["checkpoint_identity"], "g64-id")


if __name__ == "__main__":
    unittest.main()
