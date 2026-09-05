from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macqwen.checkpoints import installed_flashnext, resolve_flashnext, resolve_qwen27b


def flashnext(root: Path, name: str, model_type: str = "qwen4_exp") -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"}
    }))
    (path / "model-00001-of-00001.safetensors").touch()
    return path


class CheckpointTests(unittest.TestCase):
    def test_auto_selects_the_only_complete_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = flashnext(root, "custom")
            flashnext(root, "wrong", "other")
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                self.assertEqual(installed_flashnext(), [expected])
                self.assertEqual(resolve_flashnext(), expected.resolve())

    def test_partial_download_is_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = flashnext(root, "partial")
            (path / "model-00001-of-00001.safetensors").unlink()
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                self.assertEqual(installed_flashnext(), [])

    def test_multiple_checkpoints_require_an_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flashnext(root, "first")
            flashnext(root, "second")
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                with self.assertRaisesRegex(ValueError, "choose a Flash-Next checkpoint"):
                    resolve_flashnext()

    def test_stale_saved_checkpoint_falls_back_to_the_only_complete_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = flashnext(root, "reap288")
            stale = root / "deleted-oq4"
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                with self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                    resolve_flashnext(stale)
                self.assertEqual(
                    resolve_flashnext(stale, allow_stale_fallback=True),
                    expected.resolve(),
                )

    def test_stale_fallback_requires_one_complete_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "deleted-oq4"
            flashnext(root, "first")
            flashnext(root, "second")
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                with self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                    resolve_flashnext(stale, allow_stale_fallback=True)

    def test_existing_incomplete_checkpoint_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            damaged = flashnext(root, "damaged")
            (damaged / "model-00001-of-00001.safetensors").unlink()
            flashnext(root, "reap288")
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                with self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                    resolve_flashnext(damaged, allow_stale_fallback=True)

    def test_invalid_environment_checkpoint_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flashnext(root, "reap288")
            with patch.dict(
                os.environ,
                {
                    "MACQWEN_MODEL_ROOT": str(root),
                    "MACQWEN_FLASHNEXT_MODEL": str(root / "deleted-oq4"),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                    resolve_flashnext()

    def test_qwen27b_discovery_uses_config_instead_of_a_local_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "any-name"
            expected.mkdir()
            (expected / "config.json").write_text(json.dumps({"vocab_size": 248320}))
            with patch.dict(os.environ, {"MACQWEN_MODEL_ROOT": str(root)}, clear=False):
                self.assertEqual(resolve_qwen27b(), expected.resolve())


if __name__ == "__main__":
    unittest.main()
