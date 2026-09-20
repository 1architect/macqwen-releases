from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from macqwen.testsuite.catalog import build_catalog, runtime_directories
from macqwen.testsuite.terminal import choose_model


class TestSuiteDiscoveryTests(unittest.TestCase):
    def test_discovers_runtime_providers(self):
        runtimes = runtime_directories()
        self.assertIn("flashnext", runtimes)
        self.assertIn("bonsai2", runtimes)
        self.assertIn("k2_horizon", runtimes)
        self.assertIn("qwen27b", runtimes)
        for runtime in runtimes:
            self.assertTrue(build_catalog(runtime))

    def test_explicit_checkpoint_skips_installed_checkpoint_prompt(self):
        checkpoint = Path("/tmp/test-checkpoint")
        with patch("macqwen.testsuite.terminal.installed_checkpoints", return_value=[]):
            self.assertEqual(choose_model("qwen27b", str(checkpoint)),
                             ("qwen27b", checkpoint.resolve()))


if __name__ == "__main__":
    unittest.main()
