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
    def test_default_launch_uses_the_only_installed_checkpoint(self):
        checkpoint = Path("/models/only")
        with patch("macqwen.cli.installed_checkpoints", return_value=[
            ("bonsai2", checkpoint),
        ]):
            self.assertEqual(
                cli._default_checkpoint_args(["--profile", "plain"]),
                [
                    "--profile", "plain", "--model", "bonsai2",
                    "--checkpoint", str(checkpoint),
                ],
            )

    def test_default_launch_asks_when_multiple_checkpoints_exist(self):
        choices = [
            ("flashnext", Path("/models/oq4")),
            ("bonsai2", Path("/models/bonsai2")),
        ]
        with patch("macqwen.cli.installed_checkpoints", return_value=choices), \
                patch.object(cli.sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="2"):
            selected = cli._default_checkpoint_args([])
        self.assertEqual(
            selected,
            ["--model", "bonsai2", "--checkpoint", "/models/bonsai2"],
        )

    def test_flashnext_child_environment_has_backend_chat_preset(self):
        from models.flashnext.settings.launch import CHAT_ENV

        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False), \
                    patch("macqwen.cli._supports_python", return_value=True):
                _command, child_env = cli.command(["--model", "flashnext"])
        for key, value in CHAT_ENV.items():
            self.assertEqual(child_env[key], value)

    def test_bonsai_uses_the_shared_runtime_environment(self):
        shared = cli.MANAGED_PYTHON
        with patch.dict(os.environ, {
            "MACQWEN_PYTHON": "",
            "MACQWEN_BONSAI2_PYTHON": "",
            "VIRTUAL_ENV": "/unrelated/venv",
        }, clear=False), \
                patch("macqwen.cli._supports_python", side_effect=lambda path, model: path == shared), \
                patch("macqwen.cli.setup_environment"):
            self.assertEqual(cli._interpreter("bonsai2"), shared)

    def test_first_launch_prepares_the_managed_environment_and_retries(self):
        with patch("macqwen.cli._supports_python", side_effect=(False, True)), \
                patch("macqwen.cli.setup_environment") as setup:
            self.assertEqual(cli._interpreter("flashnext"), cli.MANAGED_PYTHON)
        setup.assert_called_once_with(["--venv", str(cli.MANAGED_ENV)])

    def test_repeated_launch_reuses_the_managed_environment(self):
        with patch("macqwen.cli._supports_python", return_value=True), \
                patch("macqwen.cli.setup_environment") as setup:
            self.assertEqual(cli._interpreter("flashnext"), cli.MANAGED_PYTHON)
            self.assertEqual(cli._interpreter("bonsai2"), cli.MANAGED_PYTHON)
        setup.assert_not_called()

    def test_explicit_override_is_rejected_when_its_runtime_is_wrong(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_PYTHON": str(python)}, clear=False
            ), patch("macqwen.cli._supports_python", return_value=False), \
                    patch("macqwen.cli._python_runtime_error", return_value="mlx 0.31"):
                with self.assertRaisesRegex(SystemExit, "MACQWEN_PYTHON.*mlx 0.31"):
                    cli._interpreter("flashnext")

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
            ), patch("macqwen.cli._supports_python", return_value=True):
                _command, child_env = cli.command(["--model", "flashnext"])
        self.assertEqual(child_env["FLASHNEXT_SLAB_GLOBAL"], "56")

    def test_explicit_metal_runtime_opt_out_wins(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ,
                {
                    "MACQWEN_FLASHNEXT_PYTHON": str(python),
                    "FLASHNEXT_METAL_RUNTIME": "0",
                },
                clear=False,
            ), patch("macqwen.cli._supports_python", return_value=True):
                _command, child_env = cli.command(["--model", "flashnext"])
        self.assertEqual(child_env["FLASHNEXT_METAL_RUNTIME"], "0")

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
            ), patch("macqwen.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--preferences-file", str(preferences_file),
                ])
        model_index = command.index("--model") + 1
        self.assertEqual(command[model_index], "flashnext")

    def test_flashnext_uses_its_environment_and_forwards_profile(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False), \
                    patch("macqwen.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--model", "flashnext", "--profile", "agent",
                    "--seed", "17",
                ])
        self.assertEqual(command[2], str(python))
        self.assertIn("flashnext", command)
        self.assertIn("agent", command)
        self.assertEqual(command[-2:], ["--seed", "17"])

    def test_flashnext_forwards_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False
            ), patch("macqwen.cli._supports_python", return_value=True):
                command, _ = cli.command(["--checkpoint", "oq4"])
        index = command.index("--model-path")
        self.assertEqual(command[index + 1], "oq4")

    def test_flashnext_forwards_benchmark_token_limits_together(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False
            ), patch("macqwen.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--model", "flashnext", "--profile", "plain",
                    "--max-tokens", "32", "--think-budget", "4096",
                    "--benchmark-json", "--benchmark-prompt", "hello",
                ])
        self.assertEqual(command[command.index("--max-tokens") + 1], "32")
        self.assertEqual(command[command.index("--think-budget") + 1], "4096")
        self.assertIn("--benchmark-json", command)

    def test_setup_installs_the_shared_runtime_in_a_local_environment(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, ".venv")
            with patch("macqwen.cli.subprocess.check_call") as check_call:
                cli.setup_environment(["--venv", str(target)])
        commands = check_call.call_args_list
        self.assertEqual(commands[0].args[0][-2:], ["venv", str(target.resolve())])
        self.assertEqual(commands[-1].args[0][-1], str(cli.ROOT))

    def test_slash_server_alias_starts_server_mode(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python)}, clear=False
            ), patch("macqwen.cli._supports_python", return_value=True):
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
            for filename in (
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "model-00001-of-00001.safetensors",
            ):
                (model / filename).touch()
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"weight": "model-00001-of-00001.safetensors"}
            }))
            assets = model / "bf16-ends"
            assets.mkdir()
            (assets / "embed.bf16").write_bytes(b"\0\0")
            (assets / "head.bf16").write_bytes(b"\0\0")
            (assets / "meta.json").write_text(json.dumps({
                "embed_shape": [1, 1], "head_shape": [1, 1],
            }))
            environment = {
                "MACQWEN_QWEN27B_PYTHON": str(python),
                "MACQWEN_MODEL": str(model),
            }
            with patch.dict(os.environ, environment, clear=False), \
                    patch("macqwen.cli._supports_python", return_value=True):
                command, child_env = cli.command(["--model", "qwen27b"])
        selected = Path(command[command.index("--model-path") + 1])
        self.assertEqual(selected, model.resolve())
        self.assertIn("--bf16-ends", command)
        self.assertEqual(child_env["MLX_QMM_BK"], "32")


if __name__ == "__main__":
    unittest.main()
