"""The harness must refuse to report a comparison that did not happen.

A condition whose setting never took effect measures the same thing twice and
reports the difference as a result. That produced one wrong published number
already, so the guard is tested rather than trusted.
"""
from __future__ import annotations

from types import SimpleNamespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.flashnext import bench_production as bench
from models.flashnext.bench_production import (
    arm,
    benchmark_provenance,
    COMPARISONS,
    LIVE_SETTINGS,
    LOAD_TIME_SETTINGS,
    apply_condition,
    check_load_time,
    check_metal_runtime_checkpoint,
    effective_chat_environment,
    inspect_metal_runtime,
    report_drift,
    runtime_source_fingerprints,
    write_evidence,
)


def backend():
    return SimpleNamespace(
        store=SimpleNamespace(
            _read_mode="pread", _ngram_nocache=False, _track_residency=False
        )
    )


class ConditionTests(unittest.TestCase):
    def test_every_condition_setting_is_applicable_or_load_time(self):
        known = set(LIVE_SETTINGS) | LOAD_TIME_SETTINGS | {"FLASHNEXT_PIN_PARTS"}
        for name, conditions in COMPARISONS.items():
            for label, env in conditions.items():
                for key in env:
                    with self.subTest(comparison=name, condition=label, key=key):
                        self.assertIn(key, known, f"{key} would silently do nothing")

    def test_a_live_setting_actually_changes_the_store(self):
        target = backend()
        apply_condition(target, {"FLASHNEXT_TRACK_RESIDENT": "1"})
        self.assertTrue(target.store._track_residency)
        apply_condition(target, {"FLASHNEXT_READ": "resident"})
        self.assertEqual(target.store._read_mode, "resident")

    def test_it_refuses_a_setting_that_does_not_take(self):
        class Stubborn:
            _read_mode = "pread"

            def __setattr__(self, key, value):
                pass

        with self.assertRaises(SystemExit):
            apply_condition(
                SimpleNamespace(store=Stubborn()), {"FLASHNEXT_READ": "resident"}
            )

    def test_a_load_time_setting_needs_fresh_arms(self):
        with self.assertRaises(SystemExit):
            check_load_time({"FLASHNEXT_PREWARM": "1"}, fresh_arms=False)
        check_load_time({"FLASHNEXT_PREWARM": "1"}, fresh_arms=True)

    def test_prewarm_is_declared_load_time(self):
        self.assertIn("FLASHNEXT_PREWARM", LOAD_TIME_SETTINGS)
        self.assertNotIn("FLASHNEXT_PREWARM", LIVE_SETTINGS)

    def test_metal_runtime_arms_are_named_as_flashnext_paths(self):
        comparison = COMPARISONS["metal-runtime"]
        self.assertEqual(set(comparison), {"mlx-reference", "custom-runtime"})
        self.assertNotIn("stock", comparison)
        self.assertNotIn("custom", comparison)
        self.assertNotEqual(
            comparison["mlx-reference"]["FLASHNEXT_METAL_RUNTIME"],
            comparison["custom-runtime"]["FLASHNEXT_METAL_RUNTIME"],
        )
        self.assertIn("FLASHNEXT_METAL_RUNTIME", LOAD_TIME_SETTINGS)

    def test_effective_chat_environment_applies_defaults_and_keeps_overrides(self):
        environment = effective_chat_environment({
            "FLASHNEXT_METAL_G64": "1",
            "UNRELATED": "keep-out",
        })
        self.assertEqual(environment["FLASHNEXT_METAL_G64"], "1")
        self.assertEqual(environment["FLASHNEXT_METAL_RUNTIME"], "1")
        self.assertEqual(environment["FLASHNEXT_READ"], "pread")
        self.assertNotIn("UNRELATED", environment)

    def test_q4_g64_is_rejected_for_generic_runtime_comparison(self):
        with patch(
            "models.flashnext.bench_chat_parity.checkpoint_runtime_capability",
            return_value={"group_size": 64},
        ):
            with self.assertRaises(SystemExit):
                check_metal_runtime_checkpoint("/checkpoint")

    @staticmethod
    def _runtime_backend(path="custom-metal", capable=True, executors=True):
        layers = []
        for index in range(48):
            projection = SimpleNamespace(group_size=32)
            active = {}
            if executors and index != 0:
                active = {"one": SimpleNamespace(last_path=path)}
            switch = SimpleNamespace(
                metal_runtime_capable=capable,
                gate_proj=projection,
                up_proj=projection,
                down_proj=projection,
                _metal_executors=active,
            )
            layers.append(SimpleNamespace(mlp=SimpleNamespace(switch_mlp=switch)))
        return SimpleNamespace(
            language=SimpleNamespace(model=SimpleNamespace(layers=layers))
        )

    def test_runtime_probe_accepts_reference_and_custom_paths(self):
        reference = self._runtime_backend(executors=False)
        custom = self._runtime_backend()
        self.assertEqual(inspect_metal_runtime(reference, False)["paths"], [])
        self.assertEqual(
            inspect_metal_runtime(custom, True)["executor_layers"],
            list(range(1, 48)),
        )

    def test_runtime_probe_rejects_wrong_path(self):
        with self.assertRaises(SystemExit):
            inspect_metal_runtime(self._runtime_backend(path="reference"), True)

    def test_evidence_writer_preserves_failure_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run.json"
            payload = {
                "status": "failed",
                "failure": {"type": "token_mismatch"},
                "raw_arms": {"custom-runtime": [{"gen_rate": 1.0}]},
            }
            write_evidence(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)

    def test_drift_report_does_not_claim_a_cause(self):
        results = [
            {"condition": "a", "elapsed_rate_correlation": -0.8},
            {"condition": "b", "elapsed_rate_correlation": 0.1},
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report_drift(results)
        text = output.getvalue().lower()
        self.assertIn("not a causal conclusion", text)
        self.assertNotIn("heat", text)

    def test_arm_publishes_measurements_before_runtime_validation(self):
        class Meter:
            def reset(self):
                pass

            def bytes_since(self):
                return 4096

        class Backend:
            tape = [101, 102]

            def reset(self):
                pass

            def append_text(self, _prompt):
                pass

            def generate(self, max_tokens, on_prefilled):
                self.tape.extend([201, 202])
                on_prefilled()
                return "answer", SimpleNamespace(
                    tokens=2, rate=4.0, tail_tokens=1, tail_seconds=0.5,
                    pinned_bytes=1024,
                )

        def probe(_backend, _enabled, phase="after"):
            if phase == "after":
                raise SystemExit("post-generation path validation failed")
            return {"capable_layers": 0, "paths": []}

        published = []
        with patch(
            "models.flashnext.diskio.free_memory_mb", return_value=1000,
        ), patch(
            "models.flashnext.bench_production.apply_condition",
        ), patch(
            "models.flashnext.bench_production.inspect_metal_runtime",
            side_effect=probe,
        ):
            with self.assertRaises(SystemExit):
                arm(
                    Backend(), 2, Meter(), 0.0,
                    condition={"FLASHNEXT_METAL_RUNTIME": "1"},
                    validate_metal_runtime=True,
                    on_raw_result=published.append,
                    provenance={
                        "checkpoint_identity": "checkpoint-a",
                        "runtime_source_fingerprint": "runtime-a",
                    },
                )

        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["ids"], (201, 202))
        self.assertEqual(published[0]["gen_tokens"], 2)
        self.assertEqual(published[0]["mb_per_token"], 4096 / 2 / 1e6)
        self.assertEqual(published[0]["checkpoint_identity"], "checkpoint-a")

    def test_benchmark_provenance_is_checkpoint_and_runtime_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model.safetensors"}})
            )
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            with patch(
                "models.flashnext.bench_chat_parity.source_fingerprint",
                return_value="runtime-a",
            ), patch(
                "models.flashnext.slab_pack.checkpoint_identity",
                return_value="checkpoint-a",
            ), patch(
                "models.flashnext.bench_production.benchmark_harness_fingerprint",
                return_value="benchmark-a",
            ):
                provenance = benchmark_provenance(checkpoint)
        self.assertEqual(provenance["checkpoint_identity"], "checkpoint-a")
        self.assertEqual(provenance["runtime_source_fingerprint"], "runtime-a")
        self.assertEqual(provenance["source_fingerprint"], "runtime-a")
        self.assertEqual(provenance["benchmark_source_fingerprint"], "benchmark-a")
        self.assertEqual(
            provenance["source_fingerprints"],
            {"runtime": "runtime-a", "benchmark": "benchmark-a"},
        )

    def test_runtime_source_fingerprints_include_the_production_harness(self):
        with patch(
            "models.flashnext.bench_chat_parity.source_fingerprint",
            return_value="runtime-a",
        ), patch(
            "models.flashnext.bench_production.benchmark_harness_fingerprint",
            return_value="benchmark-a",
        ):
            self.assertEqual(
                runtime_source_fingerprints(),
                {"runtime": "runtime-a", "benchmark": "benchmark-a"},
            )

    def test_terminal_wrapper_preserves_specific_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            payload = {
                "status": "running",
                "failure": {
                    "type": "token_mismatch",
                    "message": "candidate diverged",
                },
                "source_fingerprints": {
                    "runtime": "runtime-a", "benchmark": "benchmark-a",
                },
            }
            previous = bench._ACTIVE_EVIDENCE
            bench._ACTIVE_EVIDENCE = {"path": path, "payload": payload}
            try:
                with patch.object(
                    bench,
                    "runtime_source_fingerprints",
                    return_value=payload["source_fingerprints"],
                ):
                    bench._record_terminal_failure(SystemExit("wrapper exit"))
            finally:
                bench._ACTIVE_EVIDENCE = previous

            saved = json.loads(path.read_text())
        self.assertEqual(saved["failure"]["type"], "token_mismatch")
        self.assertEqual(saved["failure"]["message"], "candidate diverged")
        self.assertEqual(saved["terminal_failure"]["type"], "SystemExit")


if __name__ == "__main__":
    unittest.main()
