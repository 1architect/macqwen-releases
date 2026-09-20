from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macqwen import cli


class LauncherTests(unittest.TestCase):
    def test_model_owns_its_python_environment_and_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ,
                {"MACQWEN_K2_HORIZON_PYTHON": str(python)},
                clear=False,
            ), patch("macqwen.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--model", "k2-horizon", "--checkpoint", "k2",
                ])
        self.assertEqual(command[2], str(python))
        self.assertEqual(command[command.index("--model") + 1], "k2-horizon")
        self.assertEqual(command[command.index("--model-path") + 1], "k2")


if __name__ == "__main__":
    unittest.main()
