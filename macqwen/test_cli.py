from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from macqwen import cli


class LauncherTests(unittest.TestCase):
    def test_no_argument_launch_always_selects_flashnext(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            preferences_file = Path(root, "preferences.json")
            preferences_file.write_text(json.dumps({"model": "qwen27b"}))
            with patch.dict(
                os.environ,
                {"MACQWEN_FLASHNEXT_PYTHON": str(python)},
                clear=False,
            ):
                command, _ = cli.command([
                    "--preferences-file", str(preferences_file),
                ])
        model_index = command.index("--model") + 1
        self.assertEqual(command[model_index], "flashnext")

    def test_flashnext_uses_its_environment_and_forwards_profile(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False):
                command, _ = cli.command(["--model", "flashnext", "--profile", "agent"])
        self.assertEqual(command[2], str(python))
        self.assertIn("flashnext", command)
        self.assertIn("agent", command)

    def test_slash_server_alias_starts_server_mode(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False
            ):
                command, _ = cli.command(["/server"])
        self.assertIn("--server", command)
        self.assertEqual(command[command.index("--model") + 1], "flashnext")

    def test_qwen27b_uses_the_given_checkpoint_and_safe_kernel_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            model = Path(root, "model")
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"vocab_size": 248320}))
            environment = {
                "MACQWEN_QWEN27B_PYTHON": str(python),
                "MACQWEN_MODEL": str(model),
            }
            with patch.dict(os.environ, environment, clear=False):
                command, child_env = cli.command(["--model", "qwen27b"])
        self.assertIn(str(model), command)
        self.assertIn("--bf16-ends", command)
        self.assertEqual(child_env["MLX_QMM_BK"], "32")


if __name__ == "__main__":
    unittest.main()
