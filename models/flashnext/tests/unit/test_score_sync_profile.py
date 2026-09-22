from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from models.flashnext import adaptive_topk, expert_cache


class ScoreSyncProfileTests(unittest.TestCase):
    def setUp(self):
        self.old_profile = expert_cache.profile_enabled()
        self.old_score_profile = expert_cache.score_sync_profile_enabled()
        expert_cache.set_profile(False)
        expert_cache.set_score_sync_profile(False)
        expert_cache.reset_profile()

    def tearDown(self):
        expert_cache.set_profile(self.old_profile)
        expert_cache.set_score_sync_profile(self.old_score_profile)
        expert_cache.reset_profile()

    def test_profile_is_default_off_and_has_no_span(self):
        self.assertFalse(expert_cache.score_sync_profile_enabled())
        self.assertIsNone(expert_cache.score_sync_begin(True))
        totals = expert_cache.score_sync_totals()
        self.assertEqual(totals["wall_seconds"], 0.0)
        self.assertFalse(totals["threshold_path_active"])

    def test_pool_state_tracks_queued_running_and_completed(self):
        expert_cache.set_score_sync_profile(True)

        expert_cache._pool_submit()
        self.assertEqual(expert_cache.read_pool_state()["queued"], 1)
        expert_cache._pool_started()
        self.assertEqual(expert_cache.read_pool_state()["running"], 1)
        expert_cache._pool_finished()
        self.assertEqual(
            expert_cache.read_pool_state(),
            {"queued": 0, "running": 0, "completed": 1},
        )

    def test_span_records_wall_bytes_and_pool_snapshots(self):
        expert_cache.set_score_sync_profile(True)
        expert_cache._pool_submit()

        with patch.object(
            expert_cache, "_physical_bytes_read", side_effect=[100, 356]
        ), patch.object(
            expert_cache.time, "perf_counter", side_effect=[2.0, 2.25]
        ):
            handle = expert_cache.score_sync_begin(True)
            expert_cache._pool_started()
            expert_cache._pool_finished()
            expert_cache.score_sync_end(handle)

        totals = expert_cache.profile_totals()
        self.assertEqual(totals["score_sync"], 0.25)
        self.assertEqual(totals["score_sync_bytes"], 256)
        self.assertEqual(totals["score_sync_physical_bytes"], 256)
        self.assertEqual(totals["score_sync_calls"], 1)
        self.assertEqual(totals["score_sync_threshold_active"], 1)
        self.assertEqual(totals["score_sync_pool_queued_before"], 1)
        self.assertEqual(totals["score_sync_pool_queued_after"], 0)
        self.assertEqual(totals["score_sync_pool_running_before"], 0)
        self.assertEqual(totals["score_sync_pool_running_after"], 0)
        self.assertEqual(totals["score_sync_pool_completed_before"], 0)
        self.assertEqual(totals["score_sync_pool_completed_after"], 1)

    def test_inactive_threshold_is_reported(self):
        expert_cache.set_score_sync_profile(True)
        self.assertIsNone(expert_cache.score_sync_begin(False))
        totals = expert_cache.score_sync_totals()
        self.assertEqual(totals["threshold_inactive_calls"], 1)
        self.assertEqual(totals["threshold_active_calls"], 0)
        self.assertFalse(totals["threshold_path_active"])

    def test_adaptive_path_keeps_eval_direct_when_profile_is_disabled(self):
        source = inspect.getsource(adaptive_topk._moe_call)
        self.assertIn("if score_profile_enabled:", source)
        self.assertIn("elif needed:\n            mx.eval(scores, inds)", source)
        self.assertNotIn("score_sync_begin(True)\n        try:", source)


if __name__ == "__main__":
    unittest.main()
