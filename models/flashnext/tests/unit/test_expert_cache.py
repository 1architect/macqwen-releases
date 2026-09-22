from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from models.flashnext.expert_cache import (
    ExpertLRU,
    _ProfiledRead,
    _record_read_timing,
    profile_totals,
    reset_profile,
)
from models.flashnext import expert_cache


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
    def test_unprofiled_read_submits_without_slab_state(self):
        function = lambda value: value
        with patch.object(expert_cache, "_POOL") as pool, \
             patch.object(expert_cache, "_PROFILE", False), \
             patch.object(expert_cache, "_SCORE_SYNC_PROFILE", False), \
             patch.object(expert_cache.hostwindow, "ENABLED", False):
            future = expert_cache._submit_read(function, "row")
            pool.submit.assert_called_once_with(function, "row")
            self.assertIs(future, pool.submit.return_value)

    def test_expert_task_reads_all_nine_components(self):
        calls = []
        allocations = []

        class Store:
            _read_mode = "pread"
            _pread_chunk = 2
            _hybrid_cutoff = 2

            @staticmethod
            def empty_rows(name, count):
                buffer = np.empty((count, 1), dtype=np.uint8)
                allocations.append((name, buffer.shape, id(buffer)))
                return buffer

            @staticmethod
            def rows_into(name, rows, destination, mode):
                calls.append((name, tuple(rows), mode, destination.shape[0]))

        class Future:
            def result(self):
                return None

        def submit(function, *args):
            function(*args)
            return Future()

        projections = [
            SimpleNamespace(
                cache=ExpertLRU(Store(), name, capacity=32)
            )
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        with patch.object(expert_cache, "_submit_read", side_effect=submit), \
             patch.object(expert_cache, "_SHARED_BUFFER", [None]), \
             patch.object(expert_cache, "_ARENA", [0]):
            control = [projection.cache.submit([7, 9], False) for projection in projections]
            control_allocations = [
                (name, shape) for name, shape, _identity in allocations
            ]
            control_calls = sorted(calls)
            allocations.clear()
            calls.clear()

            pending = expert_cache._submit_expert_group(projections, [7, 9])
            pending.wait()
            grouped_allocations = [
                (name, shape) for name, shape, _identity in allocations
            ]
            grouped_calls = sorted(calls)

        self.assertEqual(len(control), 3)
        self.assertEqual(control_allocations, grouped_allocations)
        self.assertEqual(
            [shape[0] for _name, shape in grouped_allocations], [2] * 9
        )
        self.assertEqual(len(control_calls), 9)
        self.assertEqual(len(grouped_calls), 18)
        for calls_for_name in (control_calls, grouped_calls):
            grouped_rows = {}
            for name, rows, _mode, size in calls_for_name:
                grouped_rows.setdefault(name, []).extend(rows)
                self.assertLessEqual(size, 2)
            self.assertEqual(
                {name: sorted(rows) for name, rows in grouped_rows.items()},
                {
                    f"{projection}.{part}": [7, 9]
                    for projection in ("gate_proj", "up_proj", "down_proj")
                    for part in ("weight", "scales", "biases")
                },
            )

        self.assertEqual(
            {name for name, _rows, _mode, _size in grouped_calls},
            {
                f"{projection}.{part}"
                for projection in ("gate_proj", "up_proj", "down_proj")
                for part in ("weight", "scales", "biases")
            },
        )
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
