from __future__ import annotations

from pathlib import Path
import pstats
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models.bonsai2 import bench


class BenchTests(unittest.TestCase):
    def setUp(self):
        # Comparison lifecycle tests use synthetic child rows; give them a
        # comparable identity so they exercise accounting rather than model
        # discovery.  Provenance-specific tests call the real helpers below.
        self._provenance_patch = patch.object(
            bench,
            "provenance_manifest",
            return_value={"checkpoint": {"identity": "fixture"},
                          "runtime_source_fingerprints": {"bench": "fixture"},
                          "dependency_info": {"mlx.core": {"source_sha256": "fixture"}},
                          "harness_fingerprint": "fixture"},
        )
        self._provenance_patch.start()
        self.addCleanup(self._provenance_patch.stop)

    def test_prefill_comparison_overrides_child_constructor_step(self):
        self.assertEqual(bench.COMPARISONS["prefill"], {
            "prefill-512": {"prefill_step_size": 512},
            "prefill-256": {"prefill_step_size": 256},
        })
        constructor, effective = bench._constructor_options(
            {"prefill_step_size": 256, "cache_step": 1024}, 512
        )
        self.assertEqual(effective, 256)
        self.assertEqual(constructor, {})

    def test_q4_attention_comparison_pins_precision_allocator_and_chunk(self):
        arms = bench.COMPARISONS["q4-attention"]
        self.assertEqual(
            {tuple(arm["quantized_kv"]) for arm in arms.values()}, {(4, 64)}
        )
        self.assertEqual(
            {arm["allocator_cache_mb"] for arm in arms.values()}, {256}
        )
        self.assertEqual(
            {arm["prefill_step_size"] for arm in arms.values()}, {512}
        )
        self.assertEqual(
            {arm["q4_attention_tiling"] for arm in arms.values()}, {False, True}
        )

    def test_fused_q4_attention_comparison_keeps_stock_control(self):
        arms = bench.COMPARISONS["q4-attention-fused"]
        self.assertEqual(
            {arm["fused_q4_attention"] for arm in arms.values()}, {False, True}
        )
        self.assertEqual(
            {arm["q4_attention_tiling"] for arm in arms.values()}, {False}
        )
        self.assertEqual(
            {tuple(arm["quantized_kv"]) for arm in arms.values()}, {(4, 64)}
        )
        self.assertEqual(
            {arm["trace_memory"] for arm in arms.values()}, {False}
        )

    def test_context_fixtures_are_numbered_and_scaled(self):
        for name, count in (("context-1k", 32), ("context-2k", 128),
                            ("context-8k", 384), ("context-16k", 768)):
            _system, prompt, _tool, _repeat = bench.FIXTURES[name]
            self.assertEqual(prompt.count("Record "), count)
            self.assertNotIn("Context sentence", prompt)
            self.assertIn("at least 300 words", prompt)

    def test_runnable_prefill_catalog_is_bounded(self):
        from models.bonsai2.tests.catalog import build_catalog

        catalog = build_catalog()
        bounded = (
            "baseline-short", "q2-prefill-mpp-1k",
            "q4-attention-fused-1k", "prepared-qmm-metadata-1k",
        )
        for test_id in bounded:
            spec = catalog[test_id]
            command = spec.script(SimpleNamespace(
                python="python", checkpoint="checkpoint"
            ), Path("result.jsonl"))
            self.assertIn("context-1k", command)
            self.assertEqual(command[command.index("--rounds") + 1], "2")
        self.assertFalse(catalog["q4-attention-fused-8k"].runnable)

    def test_question_only_fixture_matches_flashnext_reference(self):
        system, question, tool, repeat = bench.FIXTURES["question-only"]
        self.assertEqual(system, "")
        self.assertEqual(question, "Explique a fotossintese em duas frases.")
        self.assertIsNone(tool)
        self.assertIsNone(repeat)

    def test_append_jsonl_is_durable_and_readable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run.jsonl"
            bench.append_jsonl(path, {"type": "raw", "tokens": [1, 2]})
            bench.append_jsonl(path, {"type": "failure", "error": "kept"})
            self.assertEqual(bench.read_jsonl(path)[1]["error"], "kept")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_cache_metrics_handles_empty_and_all_offsets(self):
        self.assertEqual(bench.cache_metrics(SimpleNamespace(cache=[]))["count"], 0)
        dtype = SimpleNamespace(itemsize="mlx.float16")
        caches = [SimpleNamespace(
            offset=index, keys=SimpleNamespace(shape=(2, 4, 256, 8), dtype=dtype),
            values=SimpleNamespace(shape=(2, 4, 256, 8), dtype=dtype),
            nbytes=2 * 4 * 256 * 8 * 2 * 2,
        ) for index in range(36)]
        metrics = bench.cache_metrics(SimpleNamespace(cache=caches))
        self.assertEqual(metrics["offsets"], list(range(36)))
        self.assertEqual(metrics["capacities"], [256] * 36)
        self.assertEqual(metrics["count"], 36)
        nbytes = 2 * 4 * 256 * 8 * 2 * 2
        self.assertEqual(metrics["capacity_bytes"], nbytes * 36)
        self.assertEqual(metrics["payload_bytes"], sum(nbytes * i // 256 for i in range(36)))

    def test_windows_mark_product_tail_and_intervals(self):
        arrivals = [{"token": i, "at_s": 0.05 + i / 10} for i in range(64)]
        windows = bench._windows(arrivals, 32)
        self.assertEqual([(item["start_token"], item["is_product_tail"]) for item in windows],
                         [(1, False), (33, True)])
        self.assertAlmostEqual(windows[0]["interval_median_s"], 0.1)
        self.assertAlmostEqual(windows[0]["boundary_elapsed_s"], 3.15)
        self.assertAlmostEqual(windows[0]["rate_tps"], 32 / 3.15)

    def test_windows_keep_end_to_end_and_decode_relative_first_token_times(self):
        windows = bench._windows([
            {"token": 1, "at_s": 10.0, "decode_at_s": 5.0},
            {"token": 2, "at_s": 10.2, "decode_at_s": 5.2},
        ], 2)
        self.assertEqual(windows[0]["first_token_latency_s"], 10.0)
        self.assertEqual(windows[0]["decode_relative_first_token_latency_s"], 5.0)
        self.assertAlmostEqual(windows[0]["rate_tps"], 2 / 5.2)

    def test_resource_policy_reports_deadline_and_vm_pressure(self):
        policy = {
            "max_prefill_seconds": 2,
            "max_generation_seconds": 2,
            "max_swap_pages": 3,
            "max_pageout_pages": 3,
        }
        deadline = bench._resource_failure(
            10.0, {"vm_counters": {"swapin": 1}},
            {"vm_counters": {"swapin": 2}}, "prefill", policy, now=13.0,
        )
        self.assertEqual(deadline["reason"], "deadline")
        pressure = bench._resource_failure(
            10.0, {"vm_counters": {"swapin": 1}},
            {"vm_counters": {"swapin": 5}}, "prefill", policy, now=11.0,
        )
        self.assertEqual(pressure["reason"], "swap_pressure")

    def test_generation_deadline_aborts_on_token_boundary(self):
        backend = Mock(pending=[1, 2])
        backend.open_conversation.return_value = 2
        backend.check_invariant.return_value = True
        mx = SimpleNamespace(random=Mock(), synchronize=Mock())

        def generate(**kwargs):
            kwargs["on_prefilled"]()
            kwargs["on_decode_token"](1, "")

        backend.generate.side_effect = generate
        policy = {
            "max_prefill_seconds": 60,
            "max_generation_seconds": 0,
            "max_swap_pages": 2**63 - 1,
            "max_pageout_pages": 2**63 - 1,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "deadline.jsonl"
            with (
                patch("models.bonsai2.backend.BonsaiBackend", return_value=backend),
                patch.dict(sys.modules, {"mlx": SimpleNamespace(core=mx), "mlx.core": mx}),
                patch.object(bench, "snapshot", return_value={}),
                patch.object(bench, "_disk", return_value=0),
                patch.object(bench, "_vm", return_value={}),
            ):
                result = bench.child_arm(
                    checkpoint=str(Path(directory) / "missing"),
                    arm_id="deadline-arm", condition="control", options={},
                    record_path=path, fixture="context-1k", horizon=1, window=1,
                    thinking=False, effort="medium", sampling="greedy",
                    prefill_step_size=512, round_index=0, resource_limits=policy,
                )
            arm = next(row for row in bench.read_jsonl(path) if row["type"] == "arm")
        self.assertEqual(result, 1)
        self.assertEqual(arm["error"]["type"], "ResourceAbort")
        self.assertEqual(arm["resource_abort"]["phase"], "generation")
        self.assertEqual(arm["tokens"], [1])

    def test_silent_child_timeout_is_reported_and_cleaned_up(self):
        result = bench._stream_child(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            dict(__import__("os").environ), "silent-arm", timeout_seconds=0.05,
        )
        self.assertTrue(result.timed_out)
        self.assertIn("BonsaiParentTimeout", result.stderr)
        self.assertNotEqual(result.returncode, 0)

    def test_provenance_distinguishes_changed_and_unknown_identity(self):
        result = bench.validate_provenance(
            {"source": "a", "dependency": {"version": None}},
            {"source": "b", "dependency": {"version": None}},
        )
        self.assertEqual(result["status"], "changed")
        self.assertIn("before.source", result["changed"])
        self.assertIn("before.dependency.version", result["unknown"])

    def test_real_manifest_matches_itself_without_model_inference(self):
        manifest = bench.provenance_manifest("b2")
        result = bench.validate_provenance(manifest, manifest, manifest)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["changed"], [])
        self.assertEqual(result["unknown"], [])

    def test_provenance_failure_increments_validation_count_and_preserves_status(self):
        def runner(command, **_kwargs):
            args = {
                command[index]: command[index + 1]
                for index in range(len(command) - 1)
                if command[index].startswith("--")
                and not command[index + 1].startswith("--")
            }
            bench.append_jsonl(args["--record"], {
                "type": "arm", "schema": 1, "arm_id": args["--arm-id"],
                "condition": args["--condition"], "round": int(args["--round"]),
                "status": "raw", "tokens": [1],
                "token_digest": bench._digest([1]),
                "stats": {"rate_tps": 1}, "provenance_validation": {
                    "status": "changed", "changed": ["during.source"],
                    "unknown": [],
                },
            })
            return SimpleNamespace(returncode=0, stderr="")

        with TemporaryDirectory() as directory:
            record = Path(directory) / "provenance.jsonl"
            summary = bench.run_comparison(
                checkpoint="b2", comparison="baseline", record_path=record,
                rounds=2, runner=runner,
            )
            validations = [
                row for row in bench.read_jsonl(record)
                if row.get("type") == "validation"
            ]
        self.assertEqual(summary["validation_failures"], 1)
        self.assertEqual(summary["status"], "completed_with_failures")
        provenance = next(row for row in validations if row.get("check") == "provenance")
        self.assertEqual(provenance["status"], "failed")
        self.assertEqual(provenance["provenance_status"], "changed")
        self.assertNotEqual(provenance["status"], provenance["provenance_status"])

    def test_provenance_change_before_first_arm_counts_and_stops(self):
        base = bench.provenance_manifest("b2")
        manifest = {
            **base,
            "runtime_source_fingerprints": {
                **base["runtime_source_fingerprints"],
                "models/bonsai2/bench.py": "original",
            },
        }
        changed = {
            **manifest,
            "runtime_source_fingerprints": {
                **manifest["runtime_source_fingerprints"],
                "models/bonsai2/bench.py": "changed",
            },
        }
        calls = []

        def provenance(_checkpoint):
            calls.append(True)
            return manifest if len(calls) == 1 else changed

        with TemporaryDirectory() as directory, patch.object(
            bench, "provenance_manifest", side_effect=provenance
        ):
            summary = bench.run_comparison(
                checkpoint="b2", comparison="baseline",
                record_path=Path(directory) / "before.jsonl", rounds=2,
                runner=lambda *_args, **_kwargs: SimpleNamespace(
                    returncode=0, stderr=""
                ),
            )
        self.assertEqual(summary["validation_failures"], 1)
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(summary["stop_reason"]["type"], "provenance_changed_before_arm")

    def test_checkpoint_optional_absence_is_known_but_required_absence_is_unknown(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}", encoding="utf-8")
            info = bench.checkpoint_info(str(path))
        self.assertFalse(info["files"]["model.py"]["present"])
        self.assertFalse(info["files"]["model.safetensors.index.json"]["present"])
        result = bench.validate_provenance(info, info)
        self.assertEqual(result["status"], "unknown")
        self.assertNotIn("before.files.model.py.present", result["unknown"])
        self.assertIn("before.files.tokenizer.json", result["unknown"])
        changed = {
            **info,
            "files": {
                **info["files"],
                "config.json": {
                    **info["files"]["config.json"], "present": False,
                },
            },
        }
        changed["identity"] = info["identity"]
        result = bench.validate_provenance(info, changed)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("before.files.config.json.present", result["unknown"])

    def test_unreadable_optional_identity_is_unknown(self):
        with patch.object(Path, "is_file", side_effect=OSError("permission denied")):
            identity = bench._checkpoint_file_identity(Path("model.py"), False)
        self.assertEqual(identity["state"], "unknown")
        self.assertEqual(identity["reason"], "unreadable_path")

    def test_namespace_package_root_and_binary_module_origin_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = root / "namespace_pkg"
            namespace.mkdir()
            binary = root / "core.cpython-312-darwin.so"
            binary.write_bytes(b"mlx binary")
            original = bench.importlib.util.find_spec

            def find_spec(name):
                if name == "namespace_pkg":
                    return SimpleNamespace(
                        origin=None, submodule_search_locations=[str(namespace)]
                    )
                if name == "mlx.core":
                    return SimpleNamespace(
                        origin=str(binary), submodule_search_locations=None
                    )
                return original(name)

            with patch.object(bench.importlib.util, "find_spec", side_effect=find_spec):
                self.assertEqual(bench._module_root("namespace_pkg"), namespace)
                self.assertEqual(bench._module_origin("mlx.core"), binary)
                info = bench.dependency_info()
            self.assertEqual(info["mlx.core"]["source_sha256"], bench.sha256(binary))

    def test_production_profile_pins_allocator_without_hiding_callback_cap(self):
        constructor, configured = bench._constructor_options(
            {"prefill_step_size": 512}, 512, "interactive-production"
        )
        self.assertEqual(constructor["allocator_cache_mb"], 256.0)
        self.assertEqual(configured, 512)
        self.assertEqual(bench.operating_profile("baseline"), "interactive-production")
        self.assertEqual(
            bench.operating_profile("prepared-qmm-metadata"),
            "interactive-production",
        )

    def test_ordering_is_reverse_interleaved(self):
        self.assertEqual(
            bench.ordered_conditions(["a", "b", "c"], 3),
            [(0, "a"), (0, "b"), (0, "c"),
             (1, "c"), (1, "b"), (1, "a"),
             (2, "a"), (2, "b"), (2, "c")],
        )
        # Two rounds screen; promotion still needs three arms per condition.
        self.assertEqual(
            bench.ordered_conditions(["a", "b"], 2),
            [(0, "a"), (0, "b"), (1, "b"), (1, "a")],
        )
        with self.assertRaises(ValueError):
            bench.ordered_conditions(["a", "b"], 1)

    def test_paired_stats_labels_each_direction(self):
        control = [{"round": i, "status": "raw", "stats": {"rate_tps": 10}}
                   for i in range(3)]
        candidate = [{"round": 0, "status": "raw", "stats": {"rate_tps": 11}},
                     {"round": 1, "status": "raw", "stats": {"rate_tps": 9}},
                     {"round": 2, "status": "raw", "stats": {"rate_tps": 10}}]
        result = bench.paired_stats(control, candidate)
        self.assertEqual([pair["direction"] for pair in result["pairs"]],
                         ["improvement", "regression", "tie"])
        self.assertEqual(result["direction"], "tie")
        self.assertEqual(result["ties"], 1)

    def test_profile_option_stays_out_of_constructor(self):
        self.assertIn("profile", bench.COMPARISONS)
        self.assertEqual(bench.COMPARISONS["profile"]["cprofile"], {"profile": True})
        constructor, _effective = bench._constructor_options({"profile": True}, 512)
        self.assertEqual(constructor, {})

    def test_generation_profile_splits_phases_and_writes_pstats(self):
        profiler = bench.GenerationProfile()
        self.addCleanup(profiler.stop)

        def prefill_work():
            return sum(range(100))

        def decode_work():
            return sum(range(200))

        profiler.start()
        prefill_work()
        profiler.start_decode()
        decode_work()
        profiler.stop()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        record = profiler.save(str(Path(directory.name) / "probe.jsonl"), "probe-arm")
        self.assertEqual(sorted(record["phases"]), ["decode", "prefill"])
        self.assertTrue(record["scope"].startswith("backend.generate"))
        self.assertIn("GPU waits", record["clock"])
        self.assertGreater(record["phases"]["decode"]["total_s"], 0.0)
        for phase in ("prefill", "decode"):
            path = Path(record["phases"][phase]["pstats_path"])
            self.assertTrue(path.exists(), path)
            stats = pstats.Stats(str(path))
            self.assertGreater(stats.total_calls, 0)
            for function in (prefill_work, decode_work):
                code = function.__code__
                key = (code.co_filename, code.co_firstlineno, code.co_name)
                self.assertEqual(key in stats.stats, function.__name__ == f"{phase}_work")
        self.assertIn("bench.py", record["phases"]["decode"]["top30"])

    def test_generation_profile_saves_never_started_and_empty_decode(self):
        for started in (False, True):
            with self.subTest(started=started), TemporaryDirectory() as directory:
                profiler = bench.GenerationProfile()
                self.addCleanup(profiler.stop)
                if started:
                    profiler.start()
                    sum(range(10))
                record = profiler.save(str(Path(directory) / "probe.jsonl"), "probe-arm")
                self.assertEqual(set(record["phases"]), {"prefill", "decode"})
                for phase, result in record["phases"].items():
                    if started and phase == "prefill":
                        self.assertEqual(result["status"], "captured")
                        self.assertGreater(pstats.Stats(result["pstats_path"]).total_calls, 0)
                    else:
                        self.assertEqual(result["status"], "empty")
                        self.assertEqual(result["total_calls"], 0)
                        self.assertEqual(result["total_s"], 0.0)

    def test_generation_profile_repeated_saves_have_unique_names(self):
        profiler = bench.GenerationProfile()
        self.addCleanup(profiler.stop)
        profiler.start()
        sum(range(10))
        profiler.start_decode()
        sum(range(20))
        profiler.stop()
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "probe.jsonl")
            records = [profiler.save(path, "probe-arm") for _ in range(2)]
            paths = [result["pstats_path"] for record in records
                     for result in record["phases"].values()]
            self.assertEqual(len(set(paths)), 4)
            for saved in paths:
                self.assertGreater(pstats.Stats(saved).total_calls, 0)

    def test_child_arm_saves_profile_and_retains_generation_error(self):
        for prefilled in (False, True):
            with self.subTest(prefilled=prefilled), TemporaryDirectory() as directory:
                error = RuntimeError(f"generation failed: prefilled={prefilled}")

                def generate(*, on_prefilled, **_kwargs):
                    if prefilled:
                        on_prefilled()
                    raise error

                profiler = bench.GenerationProfile()
                self.addCleanup(profiler.stop)
                backend = Mock(pending=[1, 2])
                backend.open_conversation.return_value = 2
                backend.generate.side_effect = generate
                mx = SimpleNamespace(random=Mock(), synchronize=Mock())
                path = str(Path(directory) / "failed.jsonl")
                with (
                    patch("models.bonsai2.backend.BonsaiBackend", return_value=backend) as constructor,
                    patch.dict(sys.modules, {"mlx": SimpleNamespace(core=mx), "mlx.core": mx}),
                    patch.object(bench, "GenerationProfile", return_value=profiler),
                    patch.object(bench, "snapshot", return_value={}),
                    patch.object(bench, "_disk", return_value=0),
                ):
                    result = bench.child_arm(
                        checkpoint=str(Path(directory) / "missing"), arm_id="failed-arm",
                        condition="cprofile", options={"profile": True}, record_path=path,
                        fixture="context-2k", horizon=32, window=32, thinking=False,
                        effort="medium", sampling="greedy", prefill_step_size=512, round_index=0,
                    )
                self.assertEqual(result, 1)
                constructor.assert_called_once_with(str(Path(directory) / "missing"), prefill_step_size=512)
                backend.generate.assert_called_once()
                rows = bench.read_jsonl(path)
                self.assertEqual([row["type"] for row in rows], ["arm", "profile", "cleanup"])
                arm, profile, cleanup = rows
                self.assertEqual(arm["status"], "failed")
                self.assertEqual(arm["error"], {"type": "RuntimeError", "message": str(error)})
                self.assertEqual(profile["arm_id"], arm["arm_id"])
                self.assertEqual(profile["status"], "captured")
                self.assertEqual(profile["phases"]["prefill"]["status"], "captured")
                self.assertEqual(profile["phases"]["decode"]["status"], "captured" if prefilled else "empty")
                self.assertEqual("prefill" in arm["snapshots"], prefilled)
                for phase in profile["phases"].values():
                    if phase["status"] == "captured":
                        self.assertGreater(pstats.Stats(phase["pstats_path"]).total_calls, 0)
                self.assertEqual(cleanup["status"], "complete")

    def test_run_comparison_keeps_raw_rows_and_records_digest_failures(self):
        calls = []

        def fake_runner(command, **_kwargs):
            calls.append(command)
            args = {command[index]: command[index + 1] for index in range(len(command) - 1)
                    if command[index].startswith("--") and not command[index + 1].startswith("--")}
            round_index = int(args["--round"])
            candidate = args["--condition"] == "allocator-256"
            tokens = [2] if candidate else [1]
            bench.append_jsonl(args["--record"], {
                "type": "arm", "schema": 1, "arm_id": args["--arm-id"],
                "condition": args["--condition"], "round": round_index, "status": "raw",
                "tokens": tokens, "token_digest": bench._digest(tokens),
                "stats": {"rate_tps": 11 if candidate else 10},
            })
            return SimpleNamespace(returncode=0, stderr="")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            summary = bench.run_comparison(checkpoint=str(Path(directory) / "missing"),
                                           comparison="allocator", record_path=path,
                                           rounds=3, seed=123, runner=fake_runner)
            rows = bench.read_jsonl(path)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1:4], ["-m", "models.bonsai2.bench", "--child"])
        self.assertEqual(summary["metadata"]["seed"], 123)
        self.assertTrue(all(call[call.index("--seed") + 1] == "123" for call in calls))
        self.assertEqual([call[call.index("--condition") + 1] for call in calls],
                         ["control", "allocator-256"])
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(summary["validation_failures"], 1)
        self.assertEqual(sum(row.get("type") == "arm" for row in rows), 2)
        self.assertTrue(all("tokens" in row for row in rows if row.get("type") == "arm"))
        self.assertTrue(any(row.get("type") == "validation" and row.get("status") == "failed"
                            for row in rows))

    def test_run_comparison_stops_after_keyboard_interrupt_and_keeps_partial_arm(self):
        calls = []

        def runner(command, **_kwargs):
            args = {command[index]: command[index + 1]
                    for index in range(len(command) - 1)
                    if command[index].startswith("--")
                    and not command[index + 1].startswith("--")}
            calls.append(args["--arm-id"])
            bench.append_jsonl(args["--record"], {
                "type": "arm", "arm_id": args["--arm-id"],
                "condition": args["--condition"], "round": int(args["--round"]),
                "status": "raw", "tokens": [1],
                "token_digest": bench._digest([1]),
                "stats": {"rate_tps": 10},
            })
            if len(calls) == 3:
                raise KeyboardInterrupt
            return SimpleNamespace(returncode=0, stderr="")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.jsonl"
            summary = bench.run_comparison(
                checkpoint=str(Path(directory) / "missing"),
                comparison="allocator", record_path=path, rounds=3,
                runner=runner,
            )
            rows = bench.read_jsonl(path)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(calls, [
            "round-1-control", "round-1-allocator-256", "round-2-allocator-256",
        ])
        arms = [row for row in rows if row.get("type") == "arm"]
        self.assertEqual(len(arms), 3)
        self.assertEqual(arms[-1]["tokens"], [1])
        interrupted_arm = next(
            row for values in summary["conditions"].values() for row in values
            if row["arm_id"] == "round-2-allocator-256"
        )
        self.assertEqual(interrupted_arm["child_returncode"], 130)
        self.assertTrue(interrupted_arm["interrupted"])
        self.assertTrue(any(
            row.get("type") == "validation"
            and row.get("arm_id") == "round-2-allocator-256"
            and row.get("status") == "interrupted"
            for row in rows
        ))

    def test_run_comparison_stops_on_subprocess_cancellation_code(self):
        calls = []

        def runner(command, **_kwargs):
            args = {command[index]: command[index + 1]
                    for index in range(len(command) - 1)
                    if command[index].startswith("--")
                    and not command[index + 1].startswith("--")}
            calls.append(args["--arm-id"])
            bench.append_jsonl(args["--record"], {
                "type": "arm", "arm_id": args["--arm-id"],
                "condition": args["--condition"], "round": int(args["--round"]),
                "status": "raw", "tokens": [1],
                "token_digest": bench._digest([1]),
                "stats": {"rate_tps": 10},
            })
            return SimpleNamespace(returncode=-2 if len(calls) == 3 else 0, stderr="")

        with TemporaryDirectory() as directory:
            summary = bench.run_comparison(
                checkpoint=str(Path(directory) / "missing"),
                comparison="allocator", record_path=Path(directory) / "signal.jsonl",
                rounds=3, runner=runner,
            )
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(len(calls), 3)
        self.assertEqual(summary["interruption"]["type"], "subprocess_termination")

    def test_run_comparison_preserves_child_failure(self):
        def failed_runner(command, **_kwargs):
            return SimpleNamespace(returncode=7, stderr="model failed")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "failed.jsonl"
            summary = bench.run_comparison(checkpoint=str(Path(directory) / "missing"),
                                           comparison="baseline", record_path=path,
                                           rounds=3, runner=failed_runner)
            rows = bench.read_jsonl(path)
        failed = [row for row in rows if row.get("type") == "arm"]
        self.assertEqual(summary["failed_arms"], 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error"]["returncode"], 7)
        self.assertIn("model failed", failed[0]["error"]["message"])
        self.assertEqual(summary["stop_reason"]["type"], "child_failure")

    def test_raw_arm_with_nonzero_child_exit_counts_as_failed(self):
        def raw_but_failed_runner(command, **_kwargs):
            record = command[command.index("--record") + 1]
            arm = command[command.index("--arm-id") + 1]
            condition = command[command.index("--condition") + 1]
            round_index = int(command[command.index("--round") + 1])
            bench.append_jsonl(record, {"type": "arm", "arm_id": arm,
                                        "condition": condition, "round": round_index,
                                        "status": "raw", "tokens": [1],
                                        "token_digest": bench._digest([1]),
                                        "stats": {"rate_tps": 1}})
            return SimpleNamespace(returncode=2, stderr="post-run validation failed")

        with TemporaryDirectory() as directory:
            summary = bench.run_comparison(checkpoint=str(Path(directory) / "missing"),
                                           comparison="baseline",
                                           record_path=Path(directory) / "raw-failed.jsonl",
                                           rounds=3, runner=raw_but_failed_runner)
        self.assertEqual(summary["failed_arms"], 1)
        self.assertEqual(summary["stop_reason"]["type"], "child_failure")

    def test_child_env_drops_the_ambient_kv_toggle(self):
        seen = []

        def recording_runner(command, **kwargs):
            seen.append(kwargs.get("env", {}))
            return SimpleNamespace(returncode=7, stderr="no model")

        with TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"MACQWEN_BONSAI2_KV": "8"}):
                bench.run_comparison(checkpoint=str(Path(directory) / "missing"),
                                     comparison="baseline",
                                     record_path=Path(directory) / "env.jsonl",
                                     rounds=2, runner=recording_runner)
        self.assertTrue(seen)
        for env in seen:
            self.assertNotIn("MACQWEN_BONSAI2_KV", env)

    def test_paired_stats_exclude_arms_that_did_not_run_clean(self):
        def mixed_runner(command, **_kwargs):
            args = {command[index]: command[index + 1] for index in range(len(command) - 1)
                    if command[index].startswith("--") and not command[index + 1].startswith("--")}
            record, arm, condition = args["--record"], args["--arm-id"], args["--condition"]
            round_index = int(args["--round"])
            if condition == "control":
                bench.append_jsonl(record, {"type": "arm", "arm_id": arm,
                                            "condition": condition, "round": round_index,
                                            "status": "raw", "tokens": [1],
                                            "token_digest": bench._digest([1]),
                                            "stats": {"rate_tps": 10}})
                return SimpleNamespace(returncode=0, stderr="")
            bench.append_jsonl(record, {"type": "arm", "arm_id": arm,
                                        "condition": condition, "round": round_index,
                                        "status": "raw", "tokens": [1],
                                        "token_digest": bench._digest([1]),
                                        "stats": {"rate_tps": 12}})
            return SimpleNamespace(returncode=2, stderr="post-run validation failed")

        with TemporaryDirectory() as directory:
            summary = bench.run_comparison(checkpoint=str(Path(directory) / "missing"),
                                           comparison="allocator",
                                           record_path=Path(directory) / "mixed.jsonl",
                                           rounds=2, runner=mixed_runner)
        self.assertEqual(summary["failed_arms"], 1)
        self.assertEqual(summary["paired"]["allocator-256"]["pair_count"], 0)
        self.assertIsNone(summary["paired"]["allocator-256"]["mean_delta_pct"])

    def test_main_reports_failure_through_its_exit_code(self):
        with patch.object(bench, "run_comparison",
                          return_value={"status": "completed"}) as run:
            self.assertEqual(
                bench.main(["--compare", "baseline", "--jsonl", "x.jsonl"]), 0
            )
        run.assert_called_once()
        with patch.object(bench, "run_comparison",
                          return_value={"status": "completed_with_failures"}):
            self.assertEqual(
                bench.main(["--compare", "baseline", "--jsonl", "x.jsonl"]), 1
            )
        with patch.object(bench, "run_comparison",
                          return_value={"status": "interrupted"}):
            self.assertEqual(
                bench.main(["--compare", "baseline", "--jsonl", "x.jsonl"]), 130
            )

    def test_attention_path_validation_excludes_insufficient_fused_arm(self):
        def counts(*, calls, attempts, selected, fallbacks, stock):
            values = {
                name: {"total": value, "multi_row": value, "single_row": 0}
                for name, value in (
                    ("attention_calls", calls), ("fused_attempts", attempts),
                    ("fused_selected", selected), ("fused_fallbacks", fallbacks),
                    ("stock_selected", stock), ("tiled_selected", 0),
                )
            }
            values["fallback_reasons"] = {}
            return values

        def runner(command, **_kwargs):
            args = {command[index]: command[index + 1]
                    for index in range(len(command) - 1)
                    if command[index].startswith("--")
                    and not command[index + 1].startswith("--")}
            candidate = args["--condition"] == "q4-fused"
            selected = 1 if candidate and int(args["--round"]) == 0 else 4 if candidate else 0
            attempts = 4 if candidate else 0
            stock = 0 if candidate else 4
            bench.append_jsonl(args["--record"], {
                "type": "arm", "arm_id": args["--arm-id"],
                "condition": args["--condition"], "round": int(args["--round"]),
                "status": "raw", "tokens": [1],
                "token_digest": bench._digest([1]),
                "stats": {"rate_tps": 10},
                "attention_counters": counts(
                    calls=4, attempts=attempts, selected=selected,
                    fallbacks=4 - selected if candidate else 0, stock=stock,
                ),
            })
            return SimpleNamespace(returncode=0, stderr="")

        with TemporaryDirectory() as directory:
            summary = bench.run_comparison(
                checkpoint=str(Path(directory) / "missing"),
                comparison="q4-attention-fused",
                record_path=Path(directory) / "paths.jsonl", rounds=2,
                runner=runner,
            )
        self.assertEqual(summary["execution_path_failures"], 1)
        self.assertEqual(summary["paired"]["q4-fused"]["pair_count"], 0)
        self.assertIsNone(summary["paired"]["q4-fused"]["mean_delta_pct"])
        self.assertEqual(summary["stop_reason"]["reason"], "insufficient_fused_calls")

    def test_q2_path_validation_rejects_stock_fallbacks(self):
        stock = {
            "q2_candidate_calls": 0, "q2_selected": 0,
            "q2_fallbacks": 0, "fallback_reasons": {},
        }
        candidate = {
            "q2_candidate_calls": 4, "q2_selected": 3,
            "q2_fallbacks": 1, "fallback_reasons": {"mpp_infeasible": 1},
        }
        base = {"status": "raw", "child_returncode": 0}
        self.assertTrue(
            bench._execution_path_validation(
                "q2-prefill-mpp", {**base, "condition": "q2-stock", "q2_counters": stock}
            )["passed"]
        )
        rejected = bench._execution_path_validation(
            "q2-prefill-mpp", {**base, "condition": "q2-mpp", "q2_counters": candidate}
        )
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["reason"], "q2_candidate_not_selected")

    def test_q2_path_validation_requires_independent_projection_coverage(self):
        counters = {
            "q2_candidate_calls": 4,
            "q2_eligible_calls": 4,
            "q2_attempts": 4,
            "q2_selected": 4,
            "q2_fallbacks": 0,
            "q2_eligible_geometries": {"17408x5120": 4},
            "q2_phase_calls": {"unknown": 0, "prefill": 4, "decode": 0},
            "q2_phase_selected": {"unknown": 0, "prefill": 4, "decode": 0},
            "q2_phase_fallbacks": {"unknown": 0, "prefill": 0, "decode": 0},
            "q2_policy_excluded_calls": 2,
            "q2_policy_exclusion_reasons": {"single_row_decode": 2},
            "q2_phase_policy_excluded": {"unknown": 0, "prefill": 0, "decode": 2},
        }
        result = bench._execution_path_validation(
            "q2-prefill-mpp", {
                "status": "raw", "child_returncode": 0,
                "condition": "q2-mpp", "q2_counters": counters,
            }
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "q2_projection_coverage_incomplete")
        self.assertEqual(result["policy_exclusion_reasons"], {"single_row_decode": 2})

        counters["q2_eligible_geometries"]["5120x17408"] = 0
        counters["q2_policy_exclusion_reasons"] = {}
        result = bench._execution_path_validation(
            "q2-prefill-mpp", {
                "status": "raw", "child_returncode": 0,
                "condition": "q2-mpp", "q2_counters": counters,
            }
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "q2_exclusion_accounting_invalid")

    def test_oracle_path_validation_requires_multiple_verification_blocks(self):
        base = {"status": "raw", "child_returncode": 0}
        stock = bench._execution_path_validation(
            "exact-speculative-oracle-4",
            {**base, "condition": "spec-stock", "speculative_stats": {
                "selected": 0, "verification_blocks": 0, "fallbacks": 0,
            }},
        )
        self.assertTrue(stock["passed"])
        candidate = bench._execution_path_validation(
            "exact-speculative-oracle-4",
            {**base, "condition": "spec-oracle-4", "speculative_stats": {
                "oracle": True, "selected": 1, "verification_blocks": 1,
                "fallbacks": 0,
            }},
        )
        self.assertFalse(candidate["passed"])
        self.assertEqual(candidate["reason"], "oracle_verifier_not_selected")

    def test_prefill_pair_is_reported_as_diagnostic(self):
        matched = {"status": "matched"}
        control = [{"round": 0, "status": "raw", "provenance_validation": matched,
                    "stats": {"prefill_seconds": 10}}]
        candidate = [{"round": 0, "status": "raw", "provenance_validation": matched,
                     "stats": {"prefill_seconds": 8}}]
        result = bench.paired_prefill_stats(control, candidate)
        self.assertEqual(result["mean_delta_pct"], 20.0)
        self.assertFalse(result["resolved"])

    def test_prefill_pair_excludes_unknown_provenance(self):
        control = [{"round": 0, "status": "raw",
                    "provenance_validation": {"status": "unknown"},
                    "stats": {"prefill_seconds": 10}}]
        candidate = [{"round": 0, "status": "raw",
                     "provenance_validation": {"status": "matched"},
                     "stats": {"prefill_seconds": 8}}]
        result = bench.paired_prefill_stats(control, candidate)
        self.assertEqual(result["pair_count"], 0)
        self.assertIsNone(result["mean_delta_pct"])

    def test_qmm_coverage_uses_checkpoint_structure_not_selected_subset(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                '{"modules": [{"path": "projection.a", "embedding": false}, '
                '{"path": "projection.b", "embedding": false}, '
                '{"path": "model.embed_tokens", "embedding": true}]}',
                encoding="utf-8",
            )
            base = {
                "status": "raw", "child_returncode": 0,
                "condition": "qmm-prepared", "checkpoint": str(checkpoint),
                "qmm_metadata": {
                    "eligible_module_manifest": [
                        {"name": "projection.a"}, {"name": "projection.b"}
                    ],
                    "prepared_module_manifest": ["projection.a", "projection.b"],
                },
                "qmm_metadata_counters": {
                    "prepared_calls": 2, "fp32_calls": 2,
                    "lower_precision_calls": 0, "unsupported_calls": 0,
                    "phase_calls": {"unknown": 0, "prefill": 2, "decode": 0},
                    "phase_prepared_calls": {"unknown": 0, "prefill": 2, "decode": 0},
                    "phase_fp32_calls": {"unknown": 0, "prefill": 2, "decode": 0},
                    "phase_lower_precision_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "phase_unsupported_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "module_calls": {"projection.a": 1, "projection.b": 1},
                },
            }
            self.assertTrue(bench._execution_path_validation(
                "prepared-qmm-metadata", base
            )["passed"])
            subset = {
                **base,
                "qmm_metadata": {
                    "eligible_module_manifest": [{"name": "projection.a"}],
                    "prepared_module_manifest": ["projection.a"],
                },
                "qmm_metadata_counters": {
                    **base["qmm_metadata_counters"],
                    "prepared_calls": 1, "fp32_calls": 1,
                    "phase_calls": {"unknown": 0, "prefill": 1, "decode": 0},
                    "phase_prepared_calls": {"unknown": 0, "prefill": 1, "decode": 0},
                    "phase_fp32_calls": {"unknown": 0, "prefill": 1, "decode": 0},
                    "module_calls": {"projection.a": 1},
                },
            }
            rejected = bench._execution_path_validation(
                "prepared-qmm-metadata", subset
            )
            self.assertFalse(rejected["passed"])
            self.assertIn("projection.b", rejected["missing_modules"])
            stock = {
                **base,
                "condition": "qmm-stock",
                "qmm_metadata": {
                    "eligible_module_manifest": [],
                    "prepared_module_manifest": [],
                },
                "qmm_metadata_counters": {
                    "prepared_calls": 0, "fp32_calls": 0,
                    "lower_precision_calls": 0, "unsupported_calls": 0,
                    "phase_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "phase_prepared_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "phase_fp32_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "phase_lower_precision_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "phase_unsupported_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "module_calls": {},
                },
            }
            self.assertTrue(bench._execution_path_validation(
                "prepared-qmm-metadata", stock
            )["passed"])

    def test_qmm_coverage_rejects_duplicate_and_unexpected_projection_names(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                '{"modules": [{"path": "projection.a", "embedding": false}]}',
                encoding="utf-8",
            )
            row = {
                "status": "raw", "child_returncode": 0,
                "condition": "qmm-prepared", "checkpoint": str(checkpoint),
                "qmm_metadata": {
                    "eligible_module_manifest": [
                        {"name": "projection.a"}, {"name": "projection.a"}
                    ],
                    "prepared_module_manifest": ["projection.a", "unexpected"],
                },
                "qmm_metadata_counters": {
                    "prepared_calls": 2, "fp32_calls": 2,
                    "lower_precision_calls": 0, "unsupported_calls": 0,
                    "phase_calls": {"unknown": 0, "prefill": 2, "decode": 0},
                    "phase_prepared_calls": {"unknown": 0, "prefill": 2, "decode": 0},
                    "phase_fp32_calls": {"unknown": 0, "prefill": 2, "decode": 0},
                    "phase_lower_precision_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "phase_unsupported_calls": {"unknown": 0, "prefill": 0, "decode": 0},
                    "module_calls": {"projection.a": 1, "unexpected": 1},
                },
            }
            result = bench._execution_path_validation(
                "prepared-qmm-metadata", row
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["manifest_duplicates"], ["projection.a"])
        self.assertEqual(result["unexpected_prepared"], ["unexpected"])

    def test_cached_tool_fixture_generates_before_appending_results(self):
        backend = Mock(pending=[1, 2])
        backend.open_conversation.return_value = 2
        cache_at_append = {}

        def generate(*_args, **_kwargs):
            backend.tape.extend(backend.pending)
            backend.pending = []
            return "t", SimpleNamespace(tokens=1)

        def append_tool_results(results, enable_thinking=True):
            cache_at_append.update({
                "pending": list(backend.pending),
                "tape": list(backend.tape),
                "enable_thinking": enable_thinking,
                "results": list(results),
            })
            backend.pending.extend([3])
            return 3

        backend.append_tool_results.side_effect = append_tool_results
        backend.generate.side_effect = generate
        backend.check_invariant.return_value = True
        backend.quantized_kv = None
        backend.attention_counters = {"fused_selected": {"total": 2}}
        backend.tape = []
        mx = SimpleNamespace(random=Mock(), synchronize=Mock())
        with TemporaryDirectory() as directory:
            with (
                patch("models.bonsai2.backend.BonsaiBackend", return_value=backend),
                patch.dict(sys.modules, {"mlx": SimpleNamespace(core=mx), "mlx.core": mx}),
                patch.object(bench, "snapshot", return_value={}),
                patch.object(bench, "_disk", return_value=0),
            ):
                bench.child_arm(
                    checkpoint=str(Path(directory) / "missing"), arm_id="fixture-arm",
                    condition="control", options={}, record_path=str(Path(directory) / "f.jsonl"),
                    fixture="cached-tool-result", horizon=4, window=4, thinking=False,
                    effort="medium", sampling="greedy", prefill_step_size=512, round_index=0,
                )
            arm = next(row for row in bench.read_jsonl(Path(directory) / "f.jsonl")
                       if row.get("type") == "arm")
        order = [str(call) for call in backend.mock_calls]
        setup_generate = next(
            index for index, name in enumerate(order) if name.startswith("call.generate")
        )
        append = next(
            index for index, name in enumerate(order) if name.startswith("call.append_tool_results")
        )
        # The cache must hold a generated turn before tool results frame it.
        self.assertLess(setup_generate, append)
        backend.open_conversation.assert_called_once_with(
            "Use tool results as context.", bench._ANALYSIS_REQUEST,
            tools=None, enable_thinking=False, reasoning_effort="medium",
        )
        backend.append_tool_results.assert_called_once_with(
            ['{"setting":"example","value":42}'], enable_thinking=False,
        )
        self.assertEqual(cache_at_append["pending"], [])
        self.assertEqual(cache_at_append["tape"], [1, 2])
        self.assertEqual(arm["attention_counters"], {"fused_selected": {"total": 2}})

    def test_digest_reference_comes_from_a_clean_control_arm(self):
        def runner(command, **_kwargs):
            args = {
                command[index]: command[index + 1]
                for index in range(len(command) - 1)
                if command[index].startswith("--")
                and not command[index + 1].startswith("--")
            }
            round_index = int(args["--round"])
            condition = args["--condition"]
            tokens = [999] if condition == "control" and round_index == 0 else [1]
            bench.append_jsonl(args["--record"], {
                "type": "arm", "arm_id": args["--arm-id"],
                "condition": condition, "round": round_index,
                "status": "raw", "tokens": tokens,
                "token_digest": bench._digest(tokens),
                "stats": {"rate_tps": 10},
            })
            return SimpleNamespace(
                returncode=3 if condition == "control" and round_index == 0 else 0,
                stderr="failed control" if round_index == 0 and condition == "control" else "",
            )

        with TemporaryDirectory() as directory:
            summary = bench.run_comparison(
                checkpoint=str(Path(directory) / "missing"),
                comparison="allocator", record_path=Path(directory) / "digest.jsonl",
                rounds=2, runner=runner,
            )
        self.assertEqual(summary["failed_arms"], 1)
        self.assertEqual(summary["validation_failures"], 0)
        self.assertIsNone(summary["expected_greedy_digest"])
        self.assertEqual(summary["paired"]["allocator-256"]["pair_count"], 0)
        self.assertEqual(summary["stop_reason"]["type"], "child_failure")


if __name__ == "__main__":
    unittest.main()
