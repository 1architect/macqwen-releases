"""Checkpoint and budget guards for persisted FlashNext pin profiles."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from models.flashnext import expert_cache, routing


class _ProfileStore:
    """Small store fake that records every row-consuming operation."""

    def __init__(self, identity="checkpoint-a", group_size=64, layouts=None):
        self.dir = "/checkpoint"
        self._flashnext_checkpoint_identity = identity
        self._layouts = dict(layouts or {0: group_size})
        # This models an export whose global config may disagree with its
        # tensor shapes.  The pin metadata must ignore this aggregate value.
        self.global_group_size = group_size
        self._read_mode = "pread"
        self.pins = []
        self.refs = {
            f"language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.weight": object()
            for layer in self._layouts
        }

    def unpin_all(self):
        self.pins.clear()

    def shape(self, name):
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        group_size = self._layouts[layer]
        if ".down_proj." in name:
            if name.endswith(".weight"):
                return (4, 64, 2)
            if name.endswith(".scales") or name.endswith(".biases"):
                return (4, 64, 128 // group_size)
        if name.endswith(".weight"):
            return (4, 2, 8)
        if name.endswith(".scales") or name.endswith(".biases"):
            return (4, 2, 64 // group_size)
        raise KeyError(name)

    def pin_size(self, _name, rows):
        return 4 * len(rows)

    def pin_rows(self, name, rows):
        self.pins.append((name, tuple(rows)))
        return self.pin_size(name, rows)


def _language():
    prefix = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            switch_mlp=SimpleNamespace(
                gate_proj=SimpleNamespace(cache=SimpleNamespace(prefix=prefix))
            )
        )
    )
    return SimpleNamespace(model=SimpleNamespace(layers=[layer]))


def _profile(identity="checkpoint-a", group_size=64, layers=None):
    return {
        "checkpoint_identity": identity,
        "quantization": {"group_size": group_size},
        "group_size": group_size,
        "layers": {"0": [0, 1] if layers is None else layers},
        "ranked_scores": {},
        "ranked_counts": {},
    }


class PinProfileTests(unittest.TestCase):
    def test_reference_profile_collects_identity_without_g64(self):
        store = _ProfileStore()
        with mock.patch.dict(routing.os.environ, {"FLASHNEXT_METAL_G64": "0"}), \
             mock.patch("models.flashnext.slab_pack.checkpoint_identity", return_value="from-disk") as identity:
            store._flashnext_checkpoint_identity = None
            profile = routing.RoutingProfile("standard", store, _language())
        identity.assert_called_once_with(store.dir)
        self.assertEqual(profile._checkpoint_identity, "from-disk")
        self.assertEqual(store._flashnext_checkpoint_identity, "from-disk")

    def test_save_requires_checkpoint_and_quantization_provenance(self):
        store = _ProfileStore()
        profile = routing.RoutingProfile("standard", store, _language())
        profile.pinned = {0: {0}}
        profile.candidates = {0: Counter({0: 1.0})}
        profile.pinned_signature = "saved"
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            routing.os.environ, {"FLASHNEXT_PIN_CACHE": f"{directory}/pins.json"}
        ):
            profile.save_pins()
            with open(f"{directory}/pins.json") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["checkpoint_identity"], "checkpoint-a")
        self.assertEqual(payload["quantization"]["group_size"], 64)
        self.assertEqual(
            payload["quantization"]["layouts"],
            {"0": {"group_size": 64, "bits": 4}},
        )

    def test_layout_metadata_uses_tensor_shapes_for_mixed_export(self):
        # The global setting says G64, but layer 0's actual scales are G32.
        store = _ProfileStore(
            group_size=64, layouts={0: 32, 1: 64}
        )
        self.assertEqual(
            routing.quantization_for_store(store),
            {
                "bits": 4,
                "layouts": {
                    "0": {"group_size": 32, "bits": 4},
                    "1": {"group_size": 64, "bits": 4},
                },
            },
        )

    def test_flat_profile_is_rejected_for_mixed_layout(self):
        store = _ProfileStore(
            group_size=64, layouts={0: 32, 1: 64}
        )
        self.assertEqual(
            routing.pin_profile_compatible(store, _profile(group_size=64)),
            (False, "pin history has incompatible quantization"),
        )

    def test_profile_check_covers_only_the_requested_mixed_layout(self):
        store = _ProfileStore(group_size=64, layouts={0: 32, 1: 64})
        payload = _profile()
        payload["layers"] = {"0": [0], "1": [0]}
        payload["quantization"] = {
            "bits": 4,
            "layouts": {
                "0": {"group_size": 32, "bits": 4},
                "1": {"group_size": 64, "bits": 4},
            },
        }
        payload.pop("group_size")
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            json.dump(payload, handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                accepted = expert_cache._compatible_pin_profile(
                    store, 64, layer_ids=(1,)
                )
                self.assertEqual(accepted["layers"]["1"], [0])
                self.assertIsNone(
                    expert_cache._compatible_pin_profile(
                        store, 64, layer_ids=(0,)
                    )
                )

    def test_save_skips_an_unverifiable_profile(self):
        store = _ProfileStore(identity=None)
        profile = routing.RoutingProfile("standard", store, _language())
        profile.pinned = {0: {0}}
        profile.pinned_signature = "unverified"
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            routing.os.environ, {"FLASHNEXT_PIN_CACHE": f"{directory}/pins.json"}
        ):
            profile.save_pins()
            self.assertFalse((Path(directory) / "pins.json").exists())

    def test_prewarm_rejects_a_different_checkpoint_before_pinning(self):
        store = _ProfileStore(identity="checkpoint-b")
        profile = routing.RoutingProfile("standard", store, _language())
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            json.dump(_profile(identity="checkpoint-a"), handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                self.assertEqual(profile.prewarm(), 0)
        self.assertEqual(store.pins, [])

    def test_prewarm_rejects_a_different_quantization_before_pinning(self):
        store = _ProfileStore(group_size=32)
        profile = routing.RoutingProfile("standard", store, _language())
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            json.dump(_profile(group_size=64), handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                self.assertEqual(profile.prewarm(), 0)
        self.assertEqual(store.pins, [])

    def test_prewarm_rejects_unmatched_quantization_fields(self):
        store = _ProfileStore()
        profile = routing.RoutingProfile("standard", store, _language())
        payload = _profile()
        payload["quantization"]["bits"] = 8
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            json.dump(payload, handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                self.assertEqual(profile.prewarm(), 0)
        self.assertEqual(store.pins, [])

    def test_prewarm_plans_rows_within_budget_before_consuming_them(self):
        store = _ProfileStore()
        # Nine projection-part rows cost 36 bytes per expert. Only one of the
        # two saved experts fits in this 50-byte budget.
        profile = routing.RoutingProfile(
            "standard", store, _language(), pin_budget_gb=50e-9
        )
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            json.dump(_profile(), handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                self.assertEqual(profile.prewarm(), 36)
        self.assertEqual(len(store.pins), 9)
        self.assertEqual(profile.pinned, {0: {0}})
        self.assertEqual(profile.pinned_bytes, 36)

    def test_prewarm_does_not_consume_rows_when_one_expert_exceeds_budget(self):
        store = _ProfileStore()
        profile = routing.RoutingProfile(
            "standard", store, _language(), pin_budget_gb=20e-9
        )
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            json.dump(_profile(layers=[0]), handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                self.assertEqual(profile.prewarm(), 0)
        self.assertEqual(store.pins, [])

    def test_slab_selection_rejects_stale_history_for_the_current_store(self):
        store = _ProfileStore(identity="checkpoint-b")
        with tempfile.NamedTemporaryFile(mode="w") as handle:
            payload = _profile(identity="checkpoint-a")
            payload["ranked_scores"] = {"0": [[0, 1.0]]}
            json.dump(payload, handle)
            handle.flush()
            with mock.patch.dict(
                routing.os.environ, {"FLASHNEXT_PIN_CACHE": handle.name}
            ):
                self.assertIsNone(
                    expert_cache._compatible_pin_profile(
                        store, 64, layer_ids=(0,)
                    )
                )


if __name__ == "__main__":
    unittest.main()
