"""Read-worker QoS switch. No model load."""
from __future__ import annotations

import threading
import unittest

from models.flashnext import expert_cache


class IoQosTests(unittest.TestCase):
    def tearDown(self):
        expert_cache._QOS[0] = None
        expert_cache._QOS_ROUTE[0] = False

    def test_default_leaves_submission_direct(self):
        expert_cache._QOS[0] = None
        expert_cache._QOS_ROUTE[0] = False
        expert_cache.set_io_qos("default")
        self.assertFalse(expert_cache._QOS_ROUTE[0])
        self.assertEqual(expert_cache._submit_read(lambda: 7).result(), 7)

    def test_enabled_routes_tasks_and_applies_class_on_worker(self):
        expert_cache.set_io_qos("user-interactive")
        self.assertEqual(expert_cache.io_qos(), "user-interactive")
        seen = {}

        def probe():
            seen["value"] = getattr(expert_cache._QOS_LOCAL, "value", None)
            seen["thread"] = threading.current_thread().name
            return 3

        self.assertEqual(expert_cache._submit_read(probe).result(), 3)
        self.assertEqual(seen["value"], 0x21)
        self.assertTrue(seen["thread"].startswith("flashnext-io"))

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            expert_cache.set_io_qos("fastest")


if __name__ == "__main__":
    unittest.main()
