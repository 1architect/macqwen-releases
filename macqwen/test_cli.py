from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import json

from macqwen import cli


class LauncherTests(unittest.TestCase):
    def test_flashnext_child_environment_has_backend_chat_preset(self):
        from models.flashnext.settings.launch import CHAT_ENV

        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False):
                _command, child_env = cli.command(["--model", "flashnext"])
        for key, value in CHAT_ENV.items():
            self.assertEqual(child_env[key], value)

    def test_explicit_flashnext_environment_override_wins(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ,
                {
                    "MACQWEN_FLASHNEXT_PYTHON": str(python),
                    "FLASHNEXT_SLAB_GLOBAL": "56",
                },
                clear=False,
            ):
                _command, child_env = cli.command(["--model", "flashnext"])
        self.assertEqual(child_env["FLASHNEXT_SLAB_GLOBAL"], "56")

    def test_warns_when_branch_does_not_include_known_main(self):
        stale = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        branch = subprocess.CompletedProcess(
            [], 0, stdout="codex/research\n", stderr=""
        )
        with patch("macqwen.cli.subprocess.run", side_effect=(stale, branch)):
            warning = cli.branch_sync_warning(Path("/repo"))
        self.assertIn("codex/research", warning)
        self.assertIn("origin/main", warning)

    def test_skips_warning_when_branch_includes_known_main(self):
        current = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("macqwen.cli.subprocess.run", return_value=current) as run:
            warning = cli.branch_sync_warning(Path("/repo"))
        self.assertEqual(warning, "")
        run.assert_called_once()

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

    def test_flashnext_forwards_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False
            ):
                command, _ = cli.command(["--checkpoint", "oq4"])
        index = command.index("--model-path")
        self.assertEqual(command[index + 1], "oq4")

    def test_flashnext_forwards_benchmark_token_limits_together(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False
            ):
                command, _ = cli.command([
                    "--model", "flashnext", "--profile", "plain",
                    "--max-tokens", "32", "--think-budget", "4096",
                    "--benchmark-json", "--benchmark-prompt", "hello",
                ])
        self.assertEqual(command[command.index("--max-tokens") + 1], "32")
        self.assertEqual(command[command.index("--think-budget") + 1], "4096")
        self.assertIn("--benchmark-json", command)

    def test_setup_installs_the_pinned_extra_in_a_local_environment(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, ".venv")
            with patch("macqwen.cli.subprocess.check_call") as check_call:
                cli.setup_environment(["--venv", str(target)])
        commands = check_call.call_args_list
        self.assertEqual(commands[0].args[0][-2:], ["venv", str(target.resolve())])
        self.assertIn("[flashnext]", commands[-1].args[0][-1])

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
        selected = Path(command[command.index("--model-path") + 1])
        self.assertEqual(selected, model.resolve())
        self.assertIn("--bf16-ends", command)
        self.assertEqual(child_env["MLX_QMM_BK"], "32")


if __name__ == "__main__":
    unittest.main()
