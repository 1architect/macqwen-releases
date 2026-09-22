"""GPU keep-warm loop: spins while reads run, stops when they finish."""
from __future__ import annotations

from concurrent.futures import Future
import threading
import time
import unittest
from unittest import mock

from models.flashnext import expert_cache
from models.flashnext.tests.bench.gpu_pstates import summarize


class KeepWarmTests(unittest.TestCase):
    def test_off_by_default_and_toggles(self):
        self.assertFalse(expert_cache.gpu_keepwarm())
        expert_cache.set_gpu_keepwarm(True)
        try:
            self.assertTrue(expert_cache.gpu_keepwarm())
        finally:
            expert_cache.set_gpu_keepwarm(False)

    def test_finished_reads_submit_no_spin(self):
        done = Future()
        done.set_result(None)
        with mock.patch.object(expert_cache, "_keepwarm_spin") as spin:
            expert_cache._keep_gpu_warm_until_done([[done]])
        spin.assert_not_called()

    def test_spins_until_every_read_is_done(self):
        pending = Future()
        timer = threading.Timer(0.01, pending.set_result, args=(None,))
        timer.start()
        with mock.patch.object(expert_cache, "_keepwarm_spin") as spin, \
                mock.patch.object(expert_cache, "_KEEPWARM_PERIOD", [0.001]):
            expert_cache._keep_gpu_warm_until_done([[pending]])
        timer.join()
        self.assertTrue(pending.done())
        self.assertGreaterEqual(spin.call_count, 2)

    def test_collects_futures_from_every_pending_shape(self):
        futures = [Future() for _ in range(3)]
        shared = expert_cache._SharedRead(None, [futures[0]])
        found = expert_cache._pending_futures([[shared, [futures[1]]], [futures[2]]])
        self.assertEqual(set(map(id, found)), set(map(id, futures)))


class StateSummaryTests(unittest.TestCase):
    def test_mean_active_state_ignores_off(self):
        summary = summarize({"OFF": 1000, "P1": 10, "P15": 30})
        self.assertAlmostEqual(summary["active_share"], 40 / 1040)
        self.assertAlmostEqual(summary["mean_active_state"], (1 * 10 + 15 * 30) / 40)


if __name__ == "__main__":
    unittest.main()
