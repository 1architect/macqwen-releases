"""Checkpoint-free tests for Frontier 10 boundary diagnostics."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from models.flashnext.boundary_profiler import (
    BOUNDARY_LABELS,
    BoundaryProfiler,
    profile_enabled,
    selected_boundaries,
)
from models.flashnext.metal_runtime import MetalMoEExecutor


class BoundaryProfilerTests(unittest.TestCase):
    def test_profiler_is_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            profiler = BoundaryProfiler()
        completed = []
        value = profiler.measure(
            "gate_qmv", lambda: completed.append("issued") or 7,
            lambda result: completed.append(result),
        )
        self.assertEqual(value, 7)
        self.assertEqual(completed, ["issued"])
        self.assertFalse(profiler.snapshot()["enabled"])
        self.assertEqual(profiler.snapshot()["events"], [])

    def test_enabled_profile_records_issue_and_completion_with_labels(self):
        ticks = iter((1.0, 1.125, 2.125, 3.0, 3.250, 4.0))
        profiler = BoundaryProfiler(enabled=True, clock=lambda: next(ticks))
        completed = []
        profiler.measure(
            "gate_qmv", lambda: "gate", lambda result: completed.append(result)
        )
        profiler.measure(
            "fused_down", lambda: "down", lambda result: completed.append(result)
        )

        snapshot = profiler.snapshot()
        self.assertEqual(completed, ["gate", "down"])
        self.assertEqual([item["label"] for item in snapshot["events"]],
                         ["gate_qmv", "fused_down"])
        self.assertAlmostEqual(snapshot["events"][0]["issue_ms"], 125.0)
        self.assertAlmostEqual(snapshot["events"][0]["completion_ms"], 1000.0)
        self.assertEqual(snapshot["totals"]["fused_down"]["count"], 1)

    def test_selected_profile_forces_only_one_boundary_completion(self):
        profiler = BoundaryProfiler(enabled=True, boundary="up_qmv")
        completed = []
        profiler.measure(
            "gate_qmv", lambda: "gate", lambda result: completed.append(result)
        )
        profiler.measure(
            "up_qmv", lambda: "up", lambda result: completed.append(result)
        )

        self.assertEqual(completed, ["up"])
        snapshot = profiler.snapshot()
        self.assertEqual(snapshot["selected"], ["up_qmv"])
        self.assertEqual(snapshot["totals"]["gate_qmv"]["count"], 0)
        self.assertEqual(snapshot["totals"]["up_qmv"]["count"], 1)

    def test_selected_boundary_environment_picks_one_label(self):
        with mock.patch.dict(
            os.environ,
            {"FLASHNEXT_PROFILE_BOUNDARY": "swiglu"},
            clear=True,
        ):
            self.assertEqual(selected_boundaries(), ("swiglu",))
            profiler = BoundaryProfiler()
        self.assertEqual(profiler.selected, ("swiglu",))
        self.assertTrue(profiler.selected_for("swiglu"))
        self.assertFalse(profiler.selected_for("gate_qmv"))

    def test_unknown_label_is_rejected(self):
        profiler = BoundaryProfiler(enabled=True)
        with self.assertRaisesRegex(ValueError, "unknown boundary label"):
            profiler.record("router", 1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "unknown boundary label"):
            BoundaryProfiler(enabled=True, boundary="router")

    def test_executor_boundary_flag_is_opt_in_without_importing_mlx(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            disabled = MetalMoEExecutor(2, 32, 1, backend="reference")
        self.assertFalse(disabled.boundary_profile["enabled"])
        self.assertEqual(tuple(disabled.boundary_profile["totals"]),
                         BOUNDARY_LABELS)

        with mock.patch.dict(os.environ, {"FLASHNEXT_PROFILE_BOUNDARIES": "1"},
                             clear=True):
            enabled = MetalMoEExecutor(2, 32, 1, backend="reference")
            self.assertTrue(profile_enabled())
        self.assertTrue(enabled.boundary_profile["enabled"])

    def test_reset_keeps_profile_enabled_and_removes_events(self):
        profiler = BoundaryProfiler(enabled=True)
        profiler.record("swiglu", 1.0, 2.0)
        profiler.reset()
        self.assertTrue(profiler.snapshot()["enabled"])
        self.assertEqual(profiler.snapshot()["events"], [])


if __name__ == "__main__":
    unittest.main()
