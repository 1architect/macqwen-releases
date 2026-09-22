from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.k2_horizon.checkpoint import installed, resolve_k2_horizon


def checkpoint(root: Path, name: str = "K2-Horizon-7B-MLX-8bit") -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "k2_horizon",
        "model_file": "model.py",
    }))
    for filename in (
        "model.py", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "model-00001-of-00001.safetensors",
    ):
        (path / filename).touch()
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"}
    }))
    return path


class CheckpointTests(unittest.TestCase):
    def test_alias_and_discovery_require_all_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = checkpoint(root)
            environment = {
                "MACQWEN_MODEL_ROOT": str(root),
                "MACQWEN_K2_HORIZON_MODEL": "",
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(installed(), [expected])
                self.assertEqual(resolve_k2_horizon("k2"), expected.resolve())
                (expected / "model.py").unlink()
                self.assertEqual(installed(), [])


if __name__ == "__main__":
    unittest.main()
