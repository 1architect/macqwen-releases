from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macqwen.reasoning_policy import (
    effective_reasoning_effort,
    effective_think_budget,
    has_reap_prune,
    reasoning_effort_status,
    think_budget_status,
)


def checkpoint(root: Path, reap: bool) -> Path:
    root.mkdir()
    config = {"model_type": "qwen4_exp"}
    if reap:
        config["reap_prune"] = {
            "source_experts": 512, "kept_experts": 288, "ranking": "reap"
        }
    (root / "config.json").write_text(json.dumps(config))
    return root


class ReasoningPolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("MACQWEN_ALLOW_REAP_XHIGH", None)
        self.addCleanup(self.env.stop)

    def test_detects_reap_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(has_reap_prune(checkpoint(Path(directory) / "reap", True)))
            self.assertFalse(has_reap_prune(checkpoint(Path(directory) / "plain", False)))

    def test_keeps_reap_xhigh_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "reap", True)
            self.assertEqual(effective_reasoning_effort("xhigh", path), "xhigh")

    def test_caps_reap_xhigh_think_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "reap", True)
            self.assertEqual(effective_think_budget(32768, "xhigh", path), 4096)
            self.assertEqual(effective_think_budget(1024, "xhigh", path), 1024)

    def test_caps_unlimited_reap_xhigh_think_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "reap", True)
            self.assertEqual(effective_think_budget(None, "xhigh", path), 4096)
            self.assertIn(
                "requested unlimited",
                think_budget_status(None, "xhigh", path),
            )

    def test_keeps_lower_reap_efforts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "reap", True)
            for level in ("low", "medium", "high"):
                with self.subTest(level=level):
                    self.assertEqual(effective_reasoning_effort(level, path), level)

    def test_override_allows_reap_xhigh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "reap", True)
            with patch.dict(os.environ, {"MACQWEN_ALLOW_REAP_XHIGH": "1"}):
                self.assertEqual(effective_reasoning_effort("xhigh", path), "xhigh")
                self.assertEqual(effective_think_budget(32768, "xhigh", path), 32768)

    def test_non_reap_xhigh_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "plain", False)
            self.assertEqual(effective_reasoning_effort("xhigh", path), "xhigh")
            self.assertEqual(effective_think_budget(32768, "xhigh", path), 32768)

    def test_status_explains_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "reap", True)
            self.assertEqual(reasoning_effort_status("xhigh", path), "effort=xhigh")


if __name__ == "__main__":
    unittest.main()
