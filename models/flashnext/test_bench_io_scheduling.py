"""Pure-Python checks for the I/O scheduling diagnostic harness.

These tests validate command construction, fixed controls, and interpretation
without importing MLX or starting a model process.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile

from models.flashnext import bench_io_scheduling as bench


class IOSchedulingHarnessTests(unittest.TestCase):
    @staticmethod
    def production_records():
        return [
            {"workers": width, "topology": "projection", "arms": [{
                "tokens": 32, "io_workers": width, "io_topology": "projection",
                "profile_io": False, "allocated_slots": 60, "mlock_ok": True,
                "gen_rate": 2.0 if width == 16 else rate,
                "tail_rate": 2.0, "phys_mb_tok": 300.0, "active_mb": 3600.0,
                "digest": "same", "allocation_digest": "same-pack",
                "vm_counters": {key: 0 for key in (
                    "swapin", "swapout", "pageout", "compress", "decompress",
                )},
            }]}
            for rate in (2.1, 2.2, 2.3) for width in (16, 8)
        ]

    def test_production_effect_uses_paired_percentages(self):
        result = bench.production_effect(self.production_records())
        self.assertAlmostEqual(result["mean_percent"], 10.0)
        self.assertAlmostEqual(result["two_se_percent"], 10 / 3 ** 0.5)
        self.assertEqual(result["sign_p"], 0.125)
        self.assertTrue(result["resolved_gain"])

    def test_production_refuses_changed_work_or_missing_metrics(self):
        for field, value in (
            ("digest", "different"), ("allocation_digest", "different"),
            ("profile_io", True), ("io_workers", 24),
            ("tokens", 31), ("vm_counters", {}),
        ):
            with self.subTest(field=field):
                records = self.production_records()
                records[1]["arms"][0][field] = value
                with self.assertRaises(ValueError):
                    bench.production_effect(records)

    def test_vm_movement_prevents_resolved_gain(self):
        records = self.production_records()
        records[1]["arms"][0]["vm_counters"]["swapout"] = 257
        result = bench.production_effect(records)
        self.assertFalse(result["resolved_gain"])
        self.assertEqual(len(result["vm_warnings"]), 1)

    def test_production_reverses_pairs_and_isolates_pin_writes(self):
        seen = []
        fixtures = self.production_records()
        by_width = {
            width: iter([row for row in fixtures if row["workers"] == width])
            for width in (16, 8)
        }

        def child(width, topology, tokens, pairs, *, profile_io, pin_cache):
            self.assertFalse(profile_io)
            self.assertEqual(pin_cache.read_bytes(), b"initial pins")
            pin_cache.write_bytes(b"child observations")
            seen.append(width)
            return next(by_width[width])

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "pins.json"
            source.write_bytes(b"initial pins")
            with patch.dict("os.environ", {"FLASHNEXT_PIN_CACHE": str(source)}), \
                    patch.object(bench, "_run_child", side_effect=child), \
                    patch("builtins.print"):
                result = bench.production_compare(SimpleNamespace(tokens=32, rounds=3, json=None))
            self.assertEqual(result, 0)
            self.assertEqual(source.read_bytes(), b"initial pins")
        self.assertEqual(seen, [16, 8, 8, 16, 16, 8])

    def test_production_environment_disables_profiling(self):
        with patch.dict("os.environ", {"FLASHNEXT_PROFILE_IO": "1"}):
            environment = bench._child_environment(8, "projection", profile_io=False)
        self.assertEqual(environment["FLASHNEXT_PROFILE_IO"], "0")
        self.assertEqual(environment["FLASHNEXT_PROFILE_SCORE_SYNC"], "0")

    def test_environment_keeps_canonical_controls_and_changes_only_diagnostic(self):
        environment = bench._child_environment(24, "expert")
        self.assertEqual(environment["FLASHNEXT_IO_WORKERS"], "24")
        self.assertEqual(environment["FLASHNEXT_IO_TASK_TOPOLOGY"], "expert")
        self.assertEqual(environment["FLASHNEXT_SLAB_GLOBAL"], "60")
        self.assertEqual(environment["FLASHNEXT_SLAB_POLICY"], "skew")
        self.assertEqual(environment["FLASHNEXT_PREAD_CHUNK"], "2")
        self.assertEqual(environment["FLASHNEXT_READ"], "pread")
        self.assertEqual(environment["FLASHNEXT_STREAM_PACK"], "0")

    def test_summary_reports_all_section_17_metrics(self):
        arm = {
            "gen_rate": 2.0,
            "tail_rate": 1.9,
            "phys_mb_tok": 500.0,
            "active_mb": 3600.0,
            "io_wait_ms_tok": 200.0,
            "digest": "same",
            "io_breakdown_ms_tok": {
                "critical_queue": 120.0,
                "critical_pread": 70.0,
                "critical_task_overhead": 5.0,
                "completion_overhead": 5.0,
            },
        }
        result = bench._summary([{"workers": 16, "topology": "projection", "arms": [arm]}])
        self.assertEqual(result["workers"], 16)
        self.assertEqual(result["queue_ms_token_median"], 120.0)
        self.assertEqual(result["pread_ms_token_median"], 70.0)
        self.assertEqual(result["task_overhead_ms_token_median"], 5.0)
        self.assertEqual(result["completion_ms_token_median"], 5.0)
        self.assertEqual(result["io_wait_ms_token_median"], 200.0)
        self.assertEqual(result["physical_mb_token_median"], 500.0)

    def test_topology_is_not_eligible_without_material_queue_residence(self):
        values = [
            {"workers": width, "topology": "projection", "arms": [{
                "gen_rate": 2.0, "tail_rate": 2.0, "phys_mb_tok": 500.0,
                "active_mb": 3600.0, "io_wait_ms_tok": 10.0, "digest": "same",
                "io_breakdown_ms_tok": {"critical_queue": 1.0, "critical_pread": 8.0,
                                         "critical_task_overhead": 0.1,
                                         "completion_overhead": 0.1},
            }]} for width in (8, 16, 24, 32)
        ]
        queue = [bench._summary([value])["queue_ms_token_median"] for value in values]
        self.assertFalse(
            max(queue) - min(queue) >= bench.QUEUE_THRESHOLD_MS
            or max(queue) >= bench.QUEUE_THRESHOLD_MS
        )


if __name__ == "__main__":
    unittest.main()
