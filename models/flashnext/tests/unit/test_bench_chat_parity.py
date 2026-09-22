"""Check comparison validity without loading a checkpoint."""
from __future__ import annotations

import unittest
import json
import struct
from pathlib import Path
import tempfile
from unittest.mock import patch

from macqwen import results as macqwen_results

from models.flashnext.tests.bench import bench_chat_parity as bench

from models.flashnext.tests.bench.bench_chat_parity import (
    CONDITIONS, SETTINGS_CONDITIONS, checkpoint_runtime_capability, condition_settings,
    summarize, token_digest, vm_warnings,
)


class ChatParityTests(unittest.TestCase):
    def test_runtime_edit_changes_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "models/flashnext"
            runtime.mkdir(parents=True)
            (root / "chat.sh").write_text("launcher")
            bench_folder = runtime / "tests/bench"
            bench_folder.mkdir(parents=True)
            (bench_folder / "bench_chat_parity.py").write_text("driver")
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
            self.enterContext(patch.object(macqwen_results, "RESULTS_ROOT", root))
            pins = root / "pins.json"
            pins.write_text("{}")
            result_path = root / "result.json"
            argv = ["bench", "--mode", "pins", "--rounds", "3", "--prompt", "request",
                    "--json", str(result_path)]
            with patch("sys.argv", argv), \
                 patch.dict("os.environ", {"FLASHNEXT_PIN_CACHE": str(pins)}), \
                 patch.object(bench, "source_fingerprint", return_value="fixed"), \
                 patch("macqwen.checkpoints.resolve_flashnext", return_value=Path("/reap")), \
                 patch.object(bench, "checkpoint_runtime_capability", return_value={"group_size": 64}), \
                 patch.object(bench, "capture_arm", return_value=(1, "raw output", "NameError: missing")), \
                 patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "pins32 exited"):
                    bench.main()
            result = json.loads(result_path.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["planned_rounds"], 3)
            self.assertEqual(result["records"], [])
            self.assertEqual(result["child_failure"]["stderr_tail"], "NameError: missing")
            self.assertEqual(result["child_failure"]["stdout"], "raw output")
            self.assertTrue(result["benchmark_source_fingerprint"])

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

    @staticmethod
    def checkpoint(directory: Path, group_size: int, expert_count: int = 288) -> Path:
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({
            "model_type": "qwen4_exp",
            "text_config": {"num_hidden_layers": 48, "num_experts": expert_count,
                            "hidden_size": 2560, "moe_intermediate_size": 640},
            "quantization": {"bits": 4, "group_size": group_size},
        }))
        prefix = "language_model.model.layers.{}.mlp.switch_mlp"
        names = {
            f"{prefix.format(layer)}.{projection}.{part}": "part.safetensors"
            for layer in range(48)
            for projection in ("gate_proj", "up_proj", "down_proj")
            for part in ("weight", "scales", "biases")
        }
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"language_model.model.embed_tokens.weight": "part.safetensors", **names},
        }))
        tensors = {}
        for layer in range(48):
          layer_prefix = prefix.format(layer)
          for projection, width, groups, packed in (
                ("gate_proj", 640, 80, 320), ("up_proj", 640, 80, 320),
                ("down_proj", 2560, 20, 80)):
            for part in ("weight", "scales", "biases"):
                tensors[f"{layer_prefix}.{projection}.{part}"] = {
                    "dtype": "U32" if part == "weight" else "BF16",
                    "shape": [expert_count, width, packed if part == "weight" else groups * 32 // group_size],
                    "data_offsets": [0, 0],
                }
        header = json.dumps(tensors).encode()
        (directory / "part.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
        return directory

    def test_q4g64_reference_fallback_is_validated_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory) / "reap", 64)
            records = self.records()
            for row in records:
                row.update(checkpoint=str(checkpoint), allocation_digest=None,
                           allocated_slots=0, mlock_ok=False)
            self.assertEqual(checkpoint_runtime_capability(checkpoint)["slab_mode"], "reference")
            self.assertEqual(summarize(records, expected_runtime="checkpoint")["conditions"]["raw"]["gen_rate"], 3.0)

    def test_q4g32_missing_slab_is_rejected_even_with_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory) / "oq4", 32, 512)
            records = self.records()
            for row in records:
                row.update(checkpoint=str(checkpoint), allocation_digest=None,
                           allocated_slots=0, mlock_ok=False)
            with self.assertRaises(ValueError):
                summarize(records, expected_runtime="checkpoint")

    def test_q4g32_512_expert_checkpoint_accepts_real_slab_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory) / "oq4", 32, 512)
            records = self.records()
            for row in records:
                row.update(checkpoint=str(checkpoint), allocation_digest="pack",
                           allocated_slots=60, mlock_ok=True)
            result = summarize(records, expected_runtime="checkpoint")
            self.assertEqual(result["runtime_validation"]["slab_mode"], "packed")

    def test_reference_fallback_rejects_mixed_slab_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory) / "reap", 64)
            records = self.records()
            for row in records:
                row.update(checkpoint=str(checkpoint), allocation_digest=None,
                           allocated_slots=0, mlock_ok=False)
            records[0]["allocated_slots"] = 60
            with self.assertRaises(ValueError):
                summarize(records, expected_runtime="checkpoint")

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

    def pin_records(self, effects=(0.0, 0.0, 0.0), group_size=64):
        records = []
        for index, effect in enumerate(effects):
            order = (32, 8) if index % 2 == 0 else (8, 32)
            for count in order:
                row = dict(self.records()[0])
                row.update(condition=f"pins{count}", round=index + 1,
                           resident_experts=count, pinned_mb=count * 100.0,
                           profile_pins=False, gen_rate=2.0 + (effect if count == 8 else 0.0))
                if group_size == 64:
                    row.update(allocation_digest=None, allocated_slots=0, mlock_ok=False,
                               slab_objects=0, executor_count=0,
                               environment=bench.condition_environment(f"pins{count}", {"group_size": 64}))
                records.append(row)
        return records

    def test_pin_controls_disable_all_slabs_only_for_g64(self):
        original = dict(bench.CHAT_ENV)
        for condition in bench.PIN_CONDITIONS:
            controls = bench.condition_environment(condition, {"group_size": 64})
            for key in ("FLASHNEXT_SLAB", "FLASHNEXT_SLAB_GLOBAL", "FLASHNEXT_SLAB_PACK",
                        "FLASHNEXT_SLAB_G64", "FLASHNEXT_STREAM_PACK", "FLASHNEXT_METAL_G64"):
                self.assertEqual(controls[key], "0")
            self.assertEqual(bench.condition_environment(condition, {"group_size": 32}), original)
        self.assertEqual(bench.condition_environment("rendered", {"group_size": 64}), original)
        self.assertEqual(bench.CHAT_ENV, original)

    def test_reap_pin_startup_and_exactness_gates(self):
        capability = {"group_size": 64, "slab_mode": "reference", "allocated_slots": 0, "mlock_ok": False}
        with patch.object(bench, "_runtime_capability", return_value=capability):
            result = summarize(self.pin_records(), "pins", expected_runtime="checkpoint")
            self.assertNotIn("60-slot", result["scope"])
            for field, value in (("allocated_slots", 60), ("slab_objects", 1),
                                 ("executor_count", 1), ("resident_experts", 16),
                                 ("digest", "changed"), ("tokens", 31)):
                records = self.pin_records()
                records[0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    summarize(records, "pins", expected_runtime="checkpoint")
            for key in ("FLASHNEXT_METAL_G64", "FLASHNEXT_SLAB", "FLASHNEXT_SLAB_GLOBAL",
                        "FLASHNEXT_SLAB_PACK", "FLASHNEXT_SLAB_G64", "FLASHNEXT_STREAM_PACK"):
                records = self.pin_records()
                records[0]["environment"][key] = "1"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "runtime controls"):
                    summarize(records, "pins", expected_runtime="checkpoint")

    def test_pin_sign_test_is_two_sided_and_excludes_ties(self):
        for effects, counts, p_value in (
            ((1.0,) * 6, (6, 0, 0), 0.03125),
            ((-1.0,) * 6, (0, 6, 0), 0.03125),
            ((0.0,) * 6, (0, 0, 6), 1.0),
            ((1.0, 1.0, 0.0), (2, 0, 1), 0.5),
            ((1.0, -1.0, 0.0), (1, 1, 1), 1.0),
        ):
            with self.subTest(effects=effects):
                result = summarize(self.pin_records(effects, group_size=32), "pins")
                self.assertEqual(tuple(result[key] for key in ("pin_wins", "pin_losses", "pin_ties")), counts)
                self.assertEqual(result["pin_sign_p"], p_value)
                self.assertEqual(result["pin_non_tied_pairs"], counts[0] + counts[1])

    def test_pin_main_preserves_raw_records_when_validation_fails(self):
        for failure in ("tokens", "source"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.enterContext(patch.object(macqwen_results, "RESULTS_ROOT", root))
                pins = root / "pins.json"
                pins.write_text("{}")
                result_path = root / "result.json"
                rows = self.pin_records()
                if failure == "tokens":
                    for row in rows:
                        if row["condition"] == "pins8":
                            row["digest"] = "different"
                captured = []

                def capture(command, environment):
                    captured.append((command, environment))
                    row = dict(rows[len(captured) - 1], type="chat-parity")
                    return 0, json.dumps(row), ""

                argv = ["bench", "--mode", "pins", "--rounds", "3", "--prompt", "request",
                        "--json", str(result_path)]
                with patch("sys.argv", argv), \
                     patch.dict("os.environ", {"FLASHNEXT_PIN_CACHE": str(pins), "FLASHNEXT_SLAB_PACK": "1"}), \
                     patch.object(bench, "source_fingerprint", return_value="fixed"), \
                     patch.object(bench, "require_unchanged_harness",
                                  side_effect=[None, RuntimeError("harness changed")] if failure == "source" else None), \
                     patch("macqwen.checkpoints.resolve_flashnext", return_value=Path("/oq4")), \
                     patch.object(bench, "checkpoint_runtime_capability", return_value={
                         "group_size": 64, "slab_mode": "reference", "allocated_slots": 0, "mlock_ok": False,
                     }), \
                     patch.object(bench, "capture_arm", side_effect=capture), \
                     patch("builtins.print"):
                    with self.assertRaises((ValueError, RuntimeError)):
                        bench.main()
                saved = json.loads(result_path.read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(len(saved["records"]), len(captured))
                self.assertEqual(len(captured), 1 if failure == "source" else 6)
                self.assertTrue(saved["last_child"]["stdout"])
                for command, environment in captured:
                    self.assertIn("--exact-quality", command)
                    self.assertEqual(environment["FLASHNEXT_SLAB_PACK"], "0")
                    self.assertEqual(environment["FLASHNEXT_SLAB_GLOBAL"], "0")
                    self.assertEqual(environment["FLASHNEXT_METAL_G64"], "0")
                if failure == "tokens":
                    self.assertEqual([row["condition"] for row in saved["records"]],
                                     ["pins32", "pins8", "pins8", "pins32", "pins32", "pins8"])

    def test_harness_fingerprint_covers_pin_driver_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "models/flashnext/tests/bench"
            folder.mkdir(parents=True)
            driver = folder / "bench_chat_parity.py"
            statistics_source = folder / "bench_production.py"
            driver.write_text("driver")
            statistics_source.write_text("statistics")
            fingerprint = bench.benchmark_harness_fingerprint(root)
            driver.write_text("changed driver")
            self.assertNotEqual(bench.benchmark_harness_fingerprint(root), fingerprint)
            driver.write_text("driver")
            statistics_source.write_text("changed statistics")
            self.assertNotEqual(bench.benchmark_harness_fingerprint(root), fingerprint)

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
