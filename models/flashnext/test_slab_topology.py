"""Pure-Python tests for physical-evidence slab topology analysis.

These tests are intentionally not executed by the implementation agent.
"""
from __future__ import annotations

import unittest

from models.flashnext.physical_miss import (
    HISTORICAL_POLICY,
    HYBRID_POLICY,
    add_observation,
    allocate_physical_miss_hybrid_slots,
    empty_profile,
    hybrid_allocation_summary,
)
from models.flashnext.slab_topology import (
    canonical_core_calibration,
    offline_ceiling_gate,
    physical_miss_score,
    simulate_topologies,
    window_evidence,
)


def canonical_allocation():
    return {layer: [0, 1, 2, 3, 4] for layer in range(12)}


def profile_with_rows(rows):
    profile = empty_profile("unit")
    profile["tokens"] = 256
    for layer, expert, physical, requested in rows:
        add_observation(
            profile, layer, expert,
            physical_miss_bytes=physical,
            requested_bytes=requested,
            samples=8,
        )
    return profile


class TestPhysicalMissHybrid(unittest.TestCase):
    def test_preserves_canonical_48_slot_core(self):
        profile = profile_with_rows([(layer, 9, 1000, 1000) for layer in range(12)])
        allocation = allocate_physical_miss_hybrid_slots(
            profile, canonical_allocation(), min_samples=2
        )
        core = {(layer, expert) for layer in range(12) for expert in range(4)}
        selected = {(layer, expert) for layer, values in allocation.items() for expert in values}
        self.assertTrue(core <= selected)
        self.assertEqual(len(selected & core), 48)

    def test_only_twelve_extensions_are_eligible(self):
        rows = [(layer, 9, 1000, 1000) for layer in range(12)]
        allocation = allocate_physical_miss_hybrid_slots(profile_with_rows(rows), canonical_allocation())
        self.assertEqual(sum(len(values) for values in allocation.values()), 60)
        self.assertEqual(sum(len(values) - 4 for values in allocation.values()), 12)
        self.assertEqual(set(allocation), set(range(12)))
        self.assertTrue(all(len(values) <= 6 for values in allocation.values()))

    def test_missing_evidence_keeps_canonical_allocation(self):
        canonical = canonical_allocation()
        allocation = allocate_physical_miss_hybrid_slots(empty_profile("unit"), canonical)
        self.assertEqual(allocation, canonical)

    def test_stale_evidence_keeps_canonical_allocation(self):
        profile = profile_with_rows([(0, 4, 100, 100), (0, 9, 1000, 1000)])
        profile["created_at"] = "2020-01-01T00:00:00+00:00"
        canonical = canonical_allocation()
        allocation = allocate_physical_miss_hybrid_slots(profile, canonical)
        self.assertEqual(allocation, canonical)

    def test_runtime_calibration_requirement_falls_back_without_core48_metadata(self):
        profile = profile_with_rows([(0, 4, 100, 100), (0, 9, 1000, 1000)])
        canonical = canonical_allocation()
        allocation = allocate_physical_miss_hybrid_slots(
            profile, canonical, require_core_calibration=True
        )
        self.assertEqual(allocation, canonical)

    def test_negative_net_replacement_is_rejected(self):
        rows = [(0, 4, 1000, 1000), (0, 9, 1050, 1050)]
        allocation = allocate_physical_miss_hybrid_slots(profile_with_rows(rows), canonical_allocation())
        self.assertEqual(allocation[0][4], 4)

    def test_positive_net_replacement_changes_extension_only(self):
        rows = [(0, 4, 100, 100), (0, 9, 1000, 1000)]
        canonical = canonical_allocation()
        allocation = allocate_physical_miss_hybrid_slots(profile_with_rows(rows), canonical)
        self.assertEqual(allocation[0][:4], canonical[0][:4])
        self.assertEqual(allocation[0][4], 9)

    def test_provenance_reports_cost_and_net_value(self):
        profile = profile_with_rows([(0, 4, 100, 100), (0, 9, 1000, 1000)])
        canonical = canonical_allocation()
        allocation = allocate_physical_miss_hybrid_slots(profile, canonical)
        summary = hybrid_allocation_summary(profile, canonical, allocation)
        self.assertEqual(summary["policy"], HYBRID_POLICY)
        self.assertEqual(summary["preserved_core_slots"], 48)
        self.assertEqual(summary["changed_extension_slots"], 1)
        self.assertGreater(summary["net_value_bytes"], 0)

    def test_historical_full_replacement_is_not_hybrid_policy(self):
        self.assertNotEqual(HISTORICAL_POLICY, HYBRID_POLICY)


class TestSlabTopologySimulator(unittest.TestCase):
    def test_objective_uses_physical_miss_bytes(self):
        profile = profile_with_rows([(0, 9, 1000, 1000), (0, 10, 1, 1000000)])
        allocation = {0: [9]}
        self.assertEqual(physical_miss_score(profile, allocation), 1000)

    def test_core_calibration_exposes_all_twelve_layers(self):
        plan = canonical_core_calibration(canonical_allocation())
        self.assertEqual(plan["slots"], 48)
        self.assertEqual(len(plan["layers"]), 12)
        self.assertTrue(plan["equal_residency"])
        self.assertEqual(set(plan["extension_candidates"]), set(range(12)))

    def test_topology_candidates_have_required_names(self):
        profile = profile_with_rows([
            (layer, expert, 1000, 1000)
            for layer in range(12) for expert in range(10)
        ])
        result = simulate_topologies(profile, canonical_allocation())
        self.assertEqual(
            set(result), {"current", "depth6", "depth8", "depth10", "canonical-core-hybrid"}
        )
        for row in result.values():
            self.assertEqual(row["objective"], "measured physical-miss bytes")

    def test_offline_ceiling_gate_requires_twenty_mb_per_token(self):
        self.assertTrue(offline_ceiling_gate(20_000_000, 1)["passes"])
        self.assertFalse(offline_ceiling_gate(19_999_999, 1)["passes"])

    def test_window_evidence_preserves_phase(self):
        rows = window_evidence([
            {"type": "window", "window": 1, "phase": "thinking"},
            {"type": "window", "window": 2, "phase": "answer"},
        ])
        self.assertEqual([row["phase"] for row in rows], ["thinking", "answer"])


if __name__ == "__main__":
    unittest.main()
