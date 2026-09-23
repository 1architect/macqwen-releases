"""Regressions for the 2026-09-23 runtime refactor. No checkpoint needed."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from models.flashnext import expert_cache, routing, sessions, slab_pack


class CorruptPinProfileTests(unittest.TestCase):
    def test_truncated_profile_falls_back_to_streaming(self):
        with tempfile.TemporaryDirectory() as root:
            pins = Path(root) / "pins.json"
            pins.write_text('{"layers": {"0": [1, 2')
            expert_cache._GLOBAL_SLAB_CACHE.clear()
            with patch.dict(os.environ, {"FLASHNEXT_PIN_CACHE": str(pins)}):
                self.assertEqual(expert_cache.get_skew_slab_allocation(60), {})
                with self.assertRaises(RuntimeError):
                    expert_cache._load_pin_profile(str(pins))
            expert_cache._GLOBAL_SLAB_CACHE.clear()


class SavePinsTests(unittest.TestCase):
    def test_write_is_complete_and_leaves_no_temporary(self):
        profile = routing.RoutingProfile.__new__(routing.RoutingProfile)
        profile.mode = "exact-quality"
        profile.resident_experts = 8
        profile.pinned = {0: {3, 5}}
        profile.pinned_signature = "new"
        profile._saved_signature = ""
        profile.candidates = {0: Counter({3: 1.0, 5: 0.5})}
        profile.route_counts = {0: Counter({3: 2, 5: 1})}
        profile.store = object()
        profile._checkpoint_identity = "abc"
        with tempfile.TemporaryDirectory() as root:
            pins = Path(root) / "pins.json"
            with patch.dict(os.environ, {"FLASHNEXT_PIN_CACHE": str(pins)}), patch.object(
                routing, "quantization_for_store",
                lambda _store: {"group_size": 32, "bits": 4},
            ):
                profile.save_pins()
            self.assertEqual(json.loads(pins.read_text())["checkpoint_identity"], "abc")
            self.assertEqual(sorted(p.name for p in Path(root).iterdir()), ["pins.json"])


class MtpLayerIdTests(unittest.TestCase):
    def test_mtp_block_never_uses_backbone_layer_zero(self):
        self.assertNotEqual(expert_cache.MTP_LAYER_ID, 0)
        block = expert_cache.StreamingSwitchGLU.__new__(expert_cache.StreamingSwitchGLU)
        object.__setattr__(block, "_metal_runtime_capable", True)
        with patch.dict(os.environ, {"FLASHNEXT_METAL_RUNTIME": "1"}):
            for layer_id, expected in ((expert_cache.MTP_LAYER_ID, False), (0, False), (1, True)):
                object.__setattr__(block, "layer_id", layer_id)
                self.assertEqual(block.metal_combines_scores, expected)


class RoutingLayerCountTests(unittest.TestCase):
    def test_counters_follow_the_model(self):
        profile = routing.RoutingProfile.__new__(routing.RoutingProfile)
        profile.language = SimpleNamespace(model=SimpleNamespace(layers=[0, 1, 2]))
        self.assertEqual(profile._layer_count(), 3)
        profile.language = None
        self.assertEqual(profile._layer_count(), 48)


class SessionFingerprintTests(unittest.TestCase):
    def test_opt_in_compiled_graphs_invalidate_sessions(self):
        self.assertIn("compile_glue.py", sessions._ENGINE_FILES)
        self.assertIn("compiled.py", sessions._ENGINE_FILES)


class RegistryDefaultTests(unittest.TestCase):
    def test_declared_defaults_match_chat_launch(self):
        from models.flashnext.settings import get_registry
        from models.flashnext.settings.launch import CHAT_ENV

        registry = get_registry()
        checked = 0
        for setting in registry.for_backend("flashnext"):
            if setting.env_key not in CHAT_ENV:
                continue
            launch = CHAT_ENV[setting.env_key]
            default = setting.default
            if setting.name == "gpu-keepwarm":
                launch = "on" if launch == "1" else "off"
            self.assertEqual(str(default), launch, setting.name)
            checked += 1
        self.assertGreater(checked, 5)


class FrozenSnapshotPruneTests(unittest.TestCase):
    def test_prunes_only_stale_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            current = root / "slab-frozen-current.json"
            stale = root / "slab-frozen-stale.json"
            pack = root / "slab-pack-slots60-old.bin"
            for path in (current, stale, pack):
                path.write_text("{}")
            old = time.time() - 30 * 86400
            for path in (current, stale, pack):
                os.utime(path, (old, old))
            with patch.dict(os.environ, {"FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS": "14"}):
                removed = slab_pack.mark_used_and_prune(current, "slab-frozen-*.json")
            self.assertEqual(removed, 1)
            self.assertTrue(current.exists())
            self.assertFalse(stale.exists())
            self.assertTrue(pack.exists())


class CheckpointIdentityTests(unittest.TestCase):
    def test_identity_ignores_metadata_that_changes_across_boots(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001.safetensors"}})
            )
            (root / "config.json").write_text("{}")
            shard = root / "model-00001.safetensors"
            shard.write_bytes(b"x" * 64)
            before = slab_pack.checkpoint_identity(root)
            os.chmod(shard, 0o600)          # moves ctime only
            self.assertEqual(slab_pack.checkpoint_identity(root), before)
            real_stat = os.stat
            with patch("pathlib.Path.stat", lambda path, **kw: _moved_device(real_stat(path))):
                self.assertEqual(slab_pack.checkpoint_identity(root), before)
            os.utime(shard, (1, 1))         # a content-style change moves mtime
            self.assertNotEqual(slab_pack.checkpoint_identity(root), before)


def _moved_device(result):
    """The same file after a reboot: new device number and inode."""
    return SimpleNamespace(
        st_mode=result.st_mode, st_size=result.st_size,
        st_mtime_ns=result.st_mtime_ns, st_ctime_ns=result.st_ctime_ns + 1,
        st_dev=result.st_dev + 1, st_ino=result.st_ino + 1,
    )


if __name__ == "__main__":
    unittest.main()
