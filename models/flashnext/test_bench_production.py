"""The harness must refuse to report a comparison that did not happen.

A condition whose setting never took effect measures the same thing twice and
reports the difference as a result. That produced one wrong published number
already, so the guard is tested rather than trusted.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from models.flashnext.bench_production import (
    COMPARISONS,
    LIVE_SETTINGS,
    LOAD_TIME_SETTINGS,
    apply_condition,
    check_load_time,
)


def backend():
    return SimpleNamespace(
        store=SimpleNamespace(
            _read_mode="pread", _ngram_nocache=False, _track_residency=False
        )
    )


class ConditionTests(unittest.TestCase):
    def test_every_condition_setting_is_applicable_or_load_time(self):
        known = set(LIVE_SETTINGS) | LOAD_TIME_SETTINGS | {"FLASHNEXT_PIN_PARTS"}
        for name, conditions in COMPARISONS.items():
            for label, env in conditions.items():
                for key in env:
                    with self.subTest(comparison=name, condition=label, key=key):
                        self.assertIn(key, known, f"{key} would silently do nothing")

    def test_a_live_setting_actually_changes_the_store(self):
        target = backend()
        apply_condition(target, {"FLASHNEXT_TRACK_RESIDENT": "1"})
        self.assertTrue(target.store._track_residency)
        apply_condition(target, {"FLASHNEXT_READ": "resident"})
        self.assertEqual(target.store._read_mode, "resident")

    def test_it_refuses_a_setting_that_does_not_take(self):
        class Stubborn:
            _read_mode = "pread"

            def __setattr__(self, key, value):
                pass

        with self.assertRaises(SystemExit):
            apply_condition(
                SimpleNamespace(store=Stubborn()), {"FLASHNEXT_READ": "resident"}
            )

    def test_a_load_time_setting_needs_fresh_arms(self):
        with self.assertRaises(SystemExit):
            check_load_time({"FLASHNEXT_PREWARM": "1"}, fresh_arms=False)
        check_load_time({"FLASHNEXT_PREWARM": "1"}, fresh_arms=True)

    def test_prewarm_is_declared_load_time(self):
        self.assertIn("FLASHNEXT_PREWARM", LOAD_TIME_SETTINGS)
        self.assertNotIn("FLASHNEXT_PREWARM", LIVE_SETTINGS)


if __name__ == "__main__":
    unittest.main()
