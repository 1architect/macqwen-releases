"""Tests for the loaded-object slab report. No model load."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from models.flashnext.tests.bench.slab_status import summarize_slab


def fake_store(**overrides):
    values = {
        "_slab_alloc": {15: [1, 2, 3]},
        "_slab_pack": SimpleNamespace(
            allocation_digest="abc123", size=123456),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_switch(slots=None, reason=None):
    return SimpleNamespace(
        slab_expert_to_slot=dict(slots or {}),
        _slab_pack_disabled_reason=reason,
    )


class SlabStatusTests(unittest.TestCase):
    def test_active_pack_reported(self):
        report = summarize_slab(
            fake_store(), [fake_switch(), fake_switch({1: 0, 2: 1})])
        self.assertTrue(report["slab_pack_created"])
        self.assertEqual(report["slab_pack_digest"], "abc123")
        self.assertEqual(report["slab_pack_bytes"], 123456)
        self.assertEqual(report["active_slab_layers"], 1)
        self.assertEqual(report["total_packed_slots"], 2)
        self.assertEqual(report["per_layer_slots"], {1: 2})
        self.assertEqual(report["disabled_reasons"], {})

    def test_absent_pack_reported(self):
        report = summarize_slab(
            fake_store(_slab_alloc=None, _slab_pack=None),
            [fake_switch(reason="no history")])
        self.assertFalse(report["slab_pack_created"])
        self.assertIsNone(report["slab_alloc_layers"])
        self.assertEqual(report["active_slab_layers"], 0)
        self.assertEqual(report["total_packed_slots"], 0)
        self.assertEqual(
            report["disabled_reasons"], {0: "no history"})


if __name__ == "__main__":
    unittest.main()
