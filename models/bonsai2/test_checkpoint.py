from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.bonsai2.checkpoint import installed, resolve_bonsai2, runtime_available


def checkpoint(root: Path, name: str = "Ternary-Bonsai-2-27B-mlx-2bit") -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "prism_hadamard_qwen35",
    }))
    for filename in (
        "tokenizer.json", "tokenizer_config.json",
        "model-00001-of-00001.safetensors",
    ):
        (path / filename).touch()
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"}
    }))
    return path


class CheckpointTests(unittest.TestCase):
    def test_alias_and_discovery_require_all_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = checkpoint(root)
            environment = {
                "MACQWEN_MODEL_ROOT": str(root),
                "MACQWEN_BONSAI2_MODEL": "",
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(installed(), [expected])
                self.assertEqual(resolve_bonsai2("b2"), expected.resolve())
                (expected / "tokenizer.json").unlink()
                self.assertEqual(installed(), [])

    def test_runtime_requires_both_entry_points_and_packed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(runtime_available(root))
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "requirements.txt").write_text("mlx")
            (runtime / "unrelated.py").write_text("x = 1\n")
            self.assertFalse(runtime_available(root))
            (runtime / "runtime.py").write_text("VALUE = 1\n")
            (runtime / "vision_artifact.py").write_text("VALUE = 2\n")
            self.assertFalse(runtime_available(root))
            (runtime / "runtime.py").write_text("class Packed:\n    pass\n")
            self.assertTrue(runtime_available(root))

    def test_single_safetensors_without_index_is_compatible(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Ternary-Bonsai-2-27B-mlx-2bit"
            path.mkdir()
            (path / "config.json").write_text(json.dumps({
                "model_type": "prism_hadamard_qwen35",
            }))
            (path / "tokenizer.json").touch()
            (path / "tokenizer_config.json").touch()
            (path / "model.safetensors").touch()
            environment = {
                "MACQWEN_MODEL_ROOT": str(root),
                "MACQWEN_BONSAI2_MODEL": "",
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(installed(), [path])


if __name__ == "__main__":
    unittest.main()
