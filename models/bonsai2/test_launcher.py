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
                {"MACQWEN_BONSAI2_PYTHON": str(python)},
                clear=False,
            ):
                command, _ = cli.command([
                    "--model", "bonsai2", "--checkpoint", "b2",
                ])
        self.assertEqual(command[2], str(python))
        self.assertEqual(command[command.index("--model") + 1], "bonsai2")
        self.assertEqual(command[command.index("--model-path") + 1], "b2")


if __name__ == "__main__":
    unittest.main()
