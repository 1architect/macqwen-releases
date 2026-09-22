"""Checkpoint-policy resolution. No model load; identity only."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from models.flashnext.checkpoint_policy import (
    VONTRA_4BIT_MTP_IDENTITY,
    content_identity,
    policy_for_identity,
    resolve_resident_experts,
)

VONTRA_DIR = "/Users/gioma/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"


class CheckpointPolicyTests(unittest.TestCase):
    def test_known_identity_carries_override(self):
        self.assertEqual(
            policy_for_identity(VONTRA_4BIT_MTP_IDENTITY),
            {"resident_experts": 8})

    def test_unknown_identity_carries_no_override(self):
        self.assertEqual(policy_for_identity("0" * 64), {})
        self.assertEqual(policy_for_identity(None), {})
        self.assertEqual(policy_for_identity(""), {})

    def test_vontra_checkpoint_resolves_to_8(self):
        if not os.path.isdir(VONTRA_DIR):
            self.skipTest("Vontra checkpoint not present")
        self.assertEqual(resolve_resident_experts(None, VONTRA_DIR), 8)

    def test_unknown_checkpoint_resolves_to_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                resolve_resident_experts(None, directory))

    def test_no_basename_dependence(self):
        # A directory named exactly like Vontra but with different
        # content must not receive the Vontra policy.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Qwen3.8-Flash-Next-MLX-4bit-MTP"
            path.mkdir()
            (path / "config.json").write_text(json.dumps(
                {"model_type": "qwen4_exp"}))
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w": "s.safetensors"}}))
            (path / "s.safetensors").write_bytes(b"")
            self.assertIsNone(resolve_resident_experts(None, path))

    def test_explicit_value_always_wins(self):
        self.assertEqual(
            resolve_resident_experts(32, VONTRA_DIR), 32)
        self.assertEqual(resolve_resident_experts(0, VONTRA_DIR), 0)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                resolve_resident_experts(16, directory), 16)

    def test_build_backend_applies_vontra_policy_when_implicit(self):
        from macqwen.session import build_backend

        seen = {}

        class FakeBackend:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self._setting_sources = {}

        args = SimpleNamespace(model_path=VONTRA_DIR, session_dir=None)
        prefs = {"thinking_enabled": False, "effort": "medium",
                 "profile": "plain"}
        with unittest.mock.patch(
            "macqwen.backends.flashnext.FlashNextBackend", FakeBackend), \
            unittest.mock.patch.object(
                sys, "argv", ["chat.sh", "--model", "flashnext"]):
            if not os.path.isdir(VONTRA_DIR):
                self.skipTest("Vontra checkpoint not present")
            backend = build_backend("flashnext", args, prefs)
        self.assertEqual(seen["resident_experts"], 8)
        self.assertEqual(
            backend._setting_sources["resident-experts"], "checkpoint policy")

    def test_build_backend_keeps_explicit_value(self):
        from macqwen.session import build_backend

        seen = {}

        class FakeBackend:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self._setting_sources = {}

        args = SimpleNamespace(model_path=VONTRA_DIR, session_dir=None,
                               resident_experts=32)
        prefs = {"thinking_enabled": False, "effort": "medium",
                 "profile": "plain"}
        with unittest.mock.patch(
            "macqwen.backends.flashnext.FlashNextBackend", FakeBackend), \
            unittest.mock.patch.object(
                sys, "argv",
                ["chat.sh", "--model", "flashnext", "--resident-experts", "32"]):
            backend = build_backend("flashnext", args, prefs)
        self.assertEqual(seen["resident_experts"], 32)
        self.assertEqual(
            backend._setting_sources.get("resident-experts"), "CLI")

    def test_abbreviated_flag_counts_as_explicit(self):
        # argparse expands `--pinned 32`; sys.argv never shows the full name.
        from macqwen.session import build_backend

        seen = {}

        class FakeBackend:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self._setting_sources = {}

        args = SimpleNamespace(model_path=VONTRA_DIR, session_dir=None,
                               resident_experts=32)
        prefs = {"thinking_enabled": False, "effort": "medium",
                 "profile": "plain"}
        with unittest.mock.patch(
            "macqwen.backends.flashnext.FlashNextBackend", FakeBackend), \
            unittest.mock.patch.object(
                sys, "argv", ["chat.sh", "--pinned", "32"]):
            backend = build_backend("flashnext", args, prefs)
        self.assertEqual(seen["resident_experts"], 32)
        self.assertEqual(
            backend._setting_sources.get("resident-experts"), "CLI")

    def test_identity_ignores_path_case_and_metadata(self):
        if not os.path.isdir(VONTRA_DIR):
            self.skipTest("Vontra checkpoint not present")
        upper = VONTRA_DIR.replace("/models/", "/Models/")
        if not os.path.isdir(upper):
            self.skipTest("filesystem is case-sensitive")
        self.assertEqual(content_identity(VONTRA_DIR), content_identity(upper))
        self.assertEqual(resolve_resident_experts(None, upper), 8)

    def test_content_identity_ignores_location(self):
        with tempfile.TemporaryDirectory() as root:
            copies = []
            for name in ("a", "b"):
                path = Path(root) / name
                path.mkdir()
                header = json.dumps({"w": {
                    "dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
                (path / "s.safetensors").write_bytes(
                    struct.pack("<Q", len(header)) + header + b"x")
                (path / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"w": "s.safetensors"}}))
                copies.append(path)
            self.assertEqual(
                content_identity(copies[0]), content_identity(copies[1]))
            os.chmod(copies[1] / "s.safetensors", 0o600)
            self.assertEqual(
                content_identity(copies[0]), content_identity(copies[1]))
            (copies[1] / "s.safetensors").write_bytes(
                (copies[0] / "s.safetensors").read_bytes()[:-1] + b"yy")
            self.assertNotEqual(
                content_identity(copies[0]), content_identity(copies[1]))

    def test_build_backend_unknown_checkpoint_keeps_default(self):
        from macqwen.session import build_backend

        seen = {}

        class FakeBackend:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self._setting_sources = {}

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(model_path=directory, session_dir=None,
                                   resident_experts=None)
            prefs = {"thinking_enabled": False, "effort": "medium",
                     "profile": "plain"}
            with unittest.mock.patch(
                "macqwen.backends.flashnext.FlashNextBackend", FakeBackend), \
                unittest.mock.patch.object(
                    sys, "argv", ["chat.sh", "--model", "flashnext"]):
                build_backend("flashnext", args, prefs)
        # No policy and no flag: the backend's generic default applies.
        self.assertNotIn("resident_experts", seen)

    def test_settings_report_effective_value_and_source(self):
        from models.flashnext.settings import get_registry

        registry = get_registry()
        setting = registry.get("flashnext", "resident-experts")
        backend = SimpleNamespace(
            resident_experts=8,
            _setting_sources={"resident-experts": "checkpoint policy"},
        )
        self.assertEqual(setting.value(backend), 8)
        self.assertEqual(
            setting.source_for(backend), "checkpoint policy")
        generic = SimpleNamespace(
            resident_experts=32, _setting_sources={})
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_RESIDENT_EXPERTS", None)
            self.assertEqual(setting.value(generic), 32)


if __name__ == "__main__":
    unittest.main()
