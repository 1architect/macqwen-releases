from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from models.flashnext.expert_cache import (
    ExpertLRU,
    _ProfiledRead,
    _record_read_timing,
    profile_totals,
    reset_profile,
)


class FakeStore:
    def __init__(self):
        self.calls = []

    def rows_np(self, name, experts):
        self.calls.append((name, list(experts)))
        base = {"weight": 0, "scales": 100, "biases": 200}[name.rsplit(".", 1)[-1]]
        return np.asarray([[base + expert, base + expert + 0.5] for expert in experts])

    @staticmethod
    def to_mx(_name, block):
        return mx.array(block)


class ExpertReaderTests(unittest.TestCase):
    def test_fetch_reads_each_part_in_route_order(self):
        store = FakeStore()
        reader = ExpertLRU(store, "block.experts", capacity=32)

        weight, scales, biases = reader.fetch([2, 0, 2])

        self.assertEqual(
            store.calls,
            [
                ("block.experts.weight", [2, 0, 2]),
                ("block.experts.scales", [2, 0, 2]),
                ("block.experts.biases", [2, 0, 2]),
            ],
        )
        self.assertEqual(weight.tolist(), [[2, 2.5], [0, 0.5], [2, 2.5]])
        self.assertEqual(scales.tolist(), [[102, 102.5], [100, 100.5], [102, 102.5]])
        self.assertEqual(biases.tolist(), [[202, 202.5], [200, 200.5], [202, 202.5]])

    def test_fetch_does_not_retain_rows_between_calls(self):
        store = FakeStore()
        reader = ExpertLRU(store, "block.experts", capacity=32)

        reader.fetch([1])
        reader.fetch([1])

        self.assertEqual(len(store.calls), 6)


class ReadProfileTests(unittest.TestCase):
    def test_completion_wait_has_distinct_service_components(self):
        reset_profile()
        timing = _ProfiledRead(
            value=None,
            submitted=0.0,
            started=2.0,
            ended=10.0,
            stats={
                "pread_intervals": [(3.0, 7.0)],
                "pread_calls": 1,
                "pread_bytes": 4096,
            },
        )

        _record_read_timing([timing], wait_started=1.0, wait_ended=11.0)
        profile = profile_totals()

        self.assertEqual(profile["critical_queue"], 1.0)
        self.assertEqual(profile["critical_pread"], 4.0)
        self.assertEqual(profile["critical_task_overhead"], 4.0)
        self.assertEqual(profile["completion_overhead"], 1.0)
        self.assertEqual(profile["pread_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()
