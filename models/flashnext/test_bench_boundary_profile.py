"""Checkpoint-free tests for benchmark boundary aggregation."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from models.flashnext.bench_slab_production import (
    aggregate_boundary_profiles,
    collect_boundary_profiles,
)


class BoundaryAggregationTests(unittest.TestCase):
    def test_aggregation_sums_executors_and_divides_by_token_count(self):
        snapshots = [
            {
                "enabled": True,
                "selected": ["up_qmv"],
                "totals": {
                    "up_qmv": {
                        "count": 2,
                        "issue_ms": 8.0,
                        "completion_ms": 12.0,
                        "total_ms": 20.0,
                    }
                },
            },
            {
                "enabled": True,
                "selected": ["up_qmv"],
                "totals": {
                    "up_qmv": {
                        "count": 1,
                        "issue_ms": 4.0,
                        "completion_ms": 6.0,
                        "total_ms": 10.0,
                    }
                },
            },
        ]

        result = aggregate_boundary_profiles(snapshots, tokens=2)

        self.assertEqual(result["executor_count"], 2)
        self.assertEqual(result["selected"], ["up_qmv"])
        self.assertEqual(result["totals"]["up_qmv"]["count"], 3)
        self.assertEqual(
            result["per_token"]["up_qmv"],
            {"issue_ms": 6.0, "completion_ms": 9.0, "total_ms": 15.0},
        )

    def test_collection_walks_every_layer_and_deduplicates_executor_objects(self):
        first = SimpleNamespace(boundary_profile={"selected": ["gate_qmv"]})
        second = SimpleNamespace(boundary_profile={"selected": ["gate_qmv"]})
        layers = [
            SimpleNamespace(mlp=SimpleNamespace(
                switch_mlp=SimpleNamespace(_metal_executors={"a": first})
            )),
            SimpleNamespace(mlp=SimpleNamespace(
                switch_mlp=SimpleNamespace(_metal_executors={"b": second, "copy": first})
            )),
        ]
        backend = SimpleNamespace(language=SimpleNamespace(layers=layers))

        profiles = collect_boundary_profiles(backend)

        self.assertEqual(profiles, [first.boundary_profile, second.boundary_profile])

    def test_missing_language_or_executors_returns_empty_collection(self):
        self.assertEqual(collect_boundary_profiles(SimpleNamespace()), [])
        backend = SimpleNamespace(language=SimpleNamespace(layers=[SimpleNamespace()]))
        self.assertEqual(collect_boundary_profiles(backend), [])


if __name__ == "__main__":
    unittest.main()
