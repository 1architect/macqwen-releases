from __future__ import annotations

from pathlib import Path
import pstats
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models.k2_horizon import bench


class BenchTests(unittest.TestCase):
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

    def test_context_fixtures_are_numbered_and_scaled(self):
        for name, count in (("context-2k", 128), ("context-8k", 384), ("context-16k", 768)):
            _system, prompt, _tool, _repeat = bench.FIXTURES[name]
            self.assertEqual(prompt.count("Record "), count)
            self.assertNotIn("Context sentence", prompt)
            self.assertIn("at least 300 words", prompt)

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

    def test_ordering_is_reverse_interleaved(self):
        self.assertEqual(
            bench.ordered_conditions(["a", "b", "c"], 3),
            [(0, "a"), (0, "b"), (0, "c"),
             (1, "c"), (1, "b"), (1, "a"),
             (2, "a"), (2, "b"), (2, "c")],
        )
        with self.assertRaises(ValueError):
            bench.ordered_conditions(["a", "b"], 2)

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
                    patch("models.k2_horizon.backend.K2HorizonBackend", return_value=backend) as constructor,
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
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[0][1:4], ["-m", "models.k2_horizon.bench", "--child"])
        self.assertEqual(summary["metadata"]["seed"], 123)
        self.assertTrue(all(call[call.index("--seed") + 1] == "123" for call in calls))
        self.assertEqual([call[call.index("--condition") + 1] for call in calls],
                         ["control", "allocator-256", "allocator-256", "control", "control", "allocator-256"])
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(summary["validation_failures"], 3)
        self.assertEqual(sum(row.get("type") == "arm" for row in rows), 6)
        self.assertTrue(all("tokens" in row for row in rows if row.get("type") == "arm"))
        self.assertTrue(any(row.get("type") == "validation" and row.get("status") == "failed"
                            for row in rows))

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
        self.assertEqual(summary["failed_arms"], 3)
        self.assertEqual(len(failed), 3)
        self.assertEqual(failed[0]["error"]["returncode"], 7)
        self.assertIn("model failed", failed[0]["error"]["message"])

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
        self.assertEqual(summary["failed_arms"], 3)


if __name__ == "__main__":
    unittest.main()
