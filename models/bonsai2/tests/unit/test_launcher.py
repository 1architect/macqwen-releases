from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from macqwen import cli


class LauncherTests(unittest.TestCase):
    def test_model_owns_its_python_environment_and_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ,
                {"MACQWEN_BONSAI2_PYTHON": str(python)},
                clear=False,
            ), patch("macqwen.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--model", "bonsai2", "--checkpoint", "b2",
                ])
        self.assertEqual(command[2], str(python))
        self.assertEqual(command[command.index("--model") + 1], "bonsai2")
        self.assertEqual(command[command.index("--model-path") + 1], "b2")

    def test_chat_construction_caps_the_allocator_cache(self):
        from macqwen import preferences
        from macqwen.session import build_backend

        args = SimpleNamespace(
            model_path="/models/b2",
            prefill_step_size=512,
            session_dir=None,
        )
        with patch(
            "models.bonsai2.backend.BonsaiBackend"
        ) as backend_class:
            build_backend("bonsai2", args, dict(preferences.DEFAULTS))
        _, options = backend_class.call_args
        self.assertEqual(options["allocator_cache_mb"], 256.0)


if __name__ == "__main__":
    unittest.main()
