"""Check comparison validity without loading a checkpoint."""
from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from models.flashnext import bench_chat_parity as bench

from models.flashnext.bench_chat_parity import (
    CONDITIONS, SETTINGS_CONDITIONS, condition_settings, summarize, token_digest, vm_warnings,
)


class ChatParityTests(unittest.TestCase):
    def test_runtime_edit_changes_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "models/flashnext"
            runtime.mkdir(parents=True)
            (root / "chat.sh").write_text("launcher")
            (runtime / "bench_chat_parity.py").write_text("driver")
            source = runtime / "expert_cache.py"
            source.write_text("VERSION = 1")
            fingerprint = bench.source_fingerprint(root)
            bench.require_unchanged_source(fingerprint, root)
            (runtime / "test_ignored.py").write_text("test only")
            self.assertEqual(bench.source_fingerprint(root), fingerprint)
            source.write_text("VERSION = 2")
            with self.assertRaisesRegex(RuntimeError, "Runtime source changed"):
                bench.require_unchanged_source(fingerprint, root)

    def test_failed_child_preserves_status_and_terminal_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = root / "pins.json"
            pins.write_text("{}")
            result_path = root / "result.json"
            argv = ["bench", "--mode", "pins", "--rounds", "3", "--prompt", "request",
                    "--json", str(result_path)]
            with patch("sys.argv", argv), \
                 patch.dict("os.environ", {"FLASHNEXT_PIN_CACHE": str(pins)}), \
                 patch.object(bench, "source_fingerprint", return_value="fixed"), \
                 patch.object(bench, "capture_arm", return_value=(1, "", "NameError: missing")), \
                 patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "pins32 exited"):
                    bench.main()
            result = json.loads(result_path.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["planned_rounds"], 3)
            self.assertEqual(result["records"], [])
            self.assertEqual(result["child_failure"]["stderr_tail"], "NameError: missing")

    @staticmethod
    def records():
        records = []
        for index in range(3):
            order = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
            for name in order:
                records.append({
                    "condition": name, "round": index + 1, "tokens": 32,
                    "prompt_digest": "raw-prompt" if name == "raw" else "chat-prompt",
                    "digest": "raw-output" if name == "raw" else "chat-output",
                    "python": "/python", "checkpoint": "/oq4", "effort": "xhigh",
                    "allocation_digest": "pack", "allocated_slots": 60,
                    "mlock_ok": True, "io_workers": 16, "profile_io": False,
                    "render_tty": True,
                    "thinking": False, "sampling": "greedy",
                    "vm_counters": {"swapin": 0, "swapout": 0, "pageout": 0},
                    "gen_rate": 3.0 if name == "raw" else 2.0,
                    "tail_rate": 2.0, "physical_mb_token": 300.0,
                    "active_mb": 3600.0, "decode_wall_seconds": 16.0,
                    "turn_wall_seconds": 20.0, "callback_seconds": 0.0,
                })
        return records

    def test_raw_workload_difference_is_not_a_rendering_regression(self):
        result = summarize(self.records())
        self.assertEqual(result["rendering_mean_percent"], 0.0)
        self.assertEqual(result["conditions"]["raw"]["gen_rate"], 3.0)
        self.assertEqual(result["conditions"]["rendered"]["gen_rate"], 2.0)

    def test_rendered_pair_requires_identical_prompt_and_output(self):
        for field in ("prompt_digest", "digest"):
            records = self.records()
            for row in records:
                if row["condition"] == "rendered":
                    row[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(records)

    def test_missing_or_changed_controls_refuse_interpretation(self):
        for field, value in (
            ("tokens", 31), ("python", "/different"), ("allocation_digest", "different"),
            ("io_workers", 8), ("profile_io", True), ("render_tty", False),
        ):
            records = self.records()
            records[2][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(records)

    def test_incomplete_round_refuses_interpretation(self):
        with self.assertRaises(ValueError):
            summarize(self.records()[:-1])

    def test_token_digest_preserves_order(self):
        self.assertNotEqual(token_digest([1, 2]), token_digest([2, 1]))

    def test_settings_hold_horizon_and_allow_different_trajectories(self):
        records = []
        for index in range(3):
            for name in SETTINGS_CONDITIONS:
                thinking, sampled = condition_settings(name)
                row = dict(self.records()[0])
                row.update(condition=name, round=index + 1, thinking=thinking,
                           sampling="configured" if sampled else "greedy",
                           sampling_settings={"temperature": 1.0 if sampled else 0.0},
                           seed=42, prompt_digest=f"prompt-{thinking}", digest=name)
                records.append(row)
        self.assertEqual(len(summarize(records, "settings")["conditions"]), 4)
        records[-1]["thinking"] = False
        with self.assertRaises(ValueError):
            summarize(records, "settings")

    def test_workload_comparison_requires_different_prompts(self):
        records = []
        for index in range(3):
            for name in ("reference", "everyday"):
                row = dict(self.records()[0])
                row.update(condition=name, round=index + 1, prompt_digest=name, digest=name)
                records.append(row)
        self.assertEqual(summarize(records, "workload")["workload_mean_percent"], 0.0)
        for row in records:
            row["prompt_digest"] = "same"
        with self.assertRaises(ValueError):
            summarize(records, "workload")

    def test_active_swap_invalidates_attribution_without_dropping_arms(self):
        records = self.records()
        records[0]["vm_counters"]["swapout"] = 44204
        result = summarize(records)
        self.assertEqual(len(result["vm_warnings"]), 1)
        self.assertIn("contamination", result["attribution_status"])
        self.assertEqual(len(result["conditions"]), 3)

    def test_missing_vm_counters_are_not_quiet_counters(self):
        records = self.records()
        del records[0]["vm_counters"]
        self.assertEqual(len(vm_warnings(records)), 1)

    def test_pin_comparison_requires_real_memory_reduction_and_identical_tokens(self):
        records = []
        for index in range(3):
            for count in (32, 8):
                row = dict(self.records()[0])
                row.update(condition=f"pins{count}", round=index + 1,
                           resident_experts=count, pinned_mb=count * 150.0,
                           profile_pins=False)
                records.append(row)
        result = summarize(records, "pins")
        self.assertEqual(result["pin_mean_percent"], 0.0)
        self.assertLess(result["pinned_mb_medians"]["pins8"], result["pinned_mb_medians"]["pins32"])
        for row in records:
            if row["condition"] == "pins8":
                row["digest"] = "changed-output"
        with self.assertRaises(ValueError):
            summarize(records, "pins")


if __name__ == "__main__":
    unittest.main()
