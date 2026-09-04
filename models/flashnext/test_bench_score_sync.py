from __future__ import annotations

import unittest
from unittest.mock import patch

from models.flashnext.bench_score_sync import (
    FROZEN_CONTROL_ENV,
    _delta,
    configure_frozen_control,
    format_attribution,
)


def profile(wall, physical, active, inactive, calls, queue, running, completed):
    return {
        "wall_seconds": wall,
        "physical_bytes": physical,
        "threshold_active_calls": active,
        "threshold_inactive_calls": inactive,
        "calls": calls,
        "pool": {
            "queued": {"before": queue[0], "after": queue[1]},
            "running": {"before": running[0], "after": running[1]},
            "completed": {"before": completed[0], "after": completed[1]},
        },
    }


class ScoreSyncBenchmarkTests(unittest.TestCase):
    def test_control_forces_60_slot_8a_and_disables_other_frontiers(self):
        with patch.dict("os.environ", {"FLASHNEXT_FUSED_SHARED_PARTS": "1"}):
            applied = configure_frozen_control()
        self.assertEqual(applied["FLASHNEXT_SLAB_GLOBAL"], "60")
        self.assertEqual(applied["FLASHNEXT_SLAB_POLICY"], "skew")
        self.assertEqual(applied["FLASHNEXT_FUSED_SHARED"], "1")
        self.assertEqual(applied["FLASHNEXT_FUSED_SHARED_PARTS"], "0")
        self.assertEqual(applied["FLASHNEXT_STREAM_PACK"], "0")
        self.assertEqual(applied["FLASHNEXT_PROFILE_SCORE_SYNC"], "1")
        self.assertEqual(set(applied), set(FROZEN_CONTROL_ENV))

    def test_delta_reports_one_token(self):
        before = profile(1.0, 100, 10, 0, 10, (4, 3), (2, 1), (8, 9))
        after = profile(1.25, 356, 14, 0, 14, (9, 0), (1, 0), (9, 18))
        sample = _delta(before, after)
        self.assertEqual(sample["wall_seconds"], 0.25)
        self.assertEqual(sample["physical_bytes"], 256)
        self.assertEqual(sample["threshold_active_calls"], 4)
        self.assertEqual(sample["pool"]["queued"], {"before": 5, "after": -3})

    def test_format_includes_required_attribution(self):
        sample = _delta(
            profile(0.0, 0, 0, 0, 0, (0, 0), (0, 0), (0, 0)),
            profile(0.25, 256000, 4, 0, 4, (5, 2), (2, 1), (1, 8)),
        )
        line = format_attribution(3, sample)
        self.assertIn("token 003", line)
        self.assertIn("threshold=active", line)
        self.assertIn("score_sync=250.00 ms", line)
        self.assertIn("physical=0.256 MB", line)
        self.assertIn("q=5/2", line)
        self.assertIn("r=2/1", line)
        self.assertIn("c=1/8", line)


if __name__ == "__main__":
    unittest.main()
