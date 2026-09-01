from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from macqwen import preferences


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "preferences.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file_gives_defaults(self):
        self.assertEqual(preferences.load(self.path), preferences.DEFAULTS)

    def test_round_trip(self):
        values = dict(preferences.DEFAULTS)
        values.update(thinking_enabled=True, max_tokens=512, effort="xhigh")
        preferences.save(values, self.path)
        self.assertEqual(preferences.load(self.path), values)

    def test_invalid_values_fall_back_per_key(self):
        self.path.write_text(json.dumps({
            "thinking_enabled": "yes",     # wrong type
            "effort": "extreme",           # not a choice
            "max_tokens": True,            # bool is not a token limit
            "show_thinking": True,         # valid, must survive
        }))
        values = preferences.load(self.path)
        self.assertFalse(values["thinking_enabled"])
        self.assertEqual(values["effort"], "medium")
        self.assertEqual(values["max_tokens"], -1)
        self.assertTrue(values["show_thinking"])

    def test_generation_limit_reserves_answer_capacity(self):
        values = dict(
            preferences.DEFAULTS,
            thinking_enabled=True,
            max_tokens=120,
            think_budget=512,
        )
        self.assertEqual(preferences.answer_limit(values), 120)
        self.assertEqual(preferences.think_limit(values), 512)
        self.assertEqual(preferences.generation_limit(values), 632)

    def test_unused_zero_thinking_budget_migrates_to_the_new_default(self):
        self.path.write_text(json.dumps({"think_budget": 0}))
        values = preferences.load(self.path)
        self.assertEqual(
            values["think_budget"], preferences.DEFAULT_THINK_TOKENS
        )

    def test_explicit_path_does_not_import_legacy_files(self):
        # a missing file at an explicit path must not pull in the user's
        # real ~/.cache/flashnext or ~/.frankenstein settings
        self.assertEqual(preferences.load(self.path), preferences.DEFAULTS)

    def test_legacy_name_is_migrated(self):
        self.path.write_text(json.dumps({"show_think": True}))
        self.assertTrue(preferences.load(self.path)["show_thinking"])

    def test_unknown_keys_are_not_written_back(self):
        preferences.save({"thinking_enabled": True, "junk": 1}, self.path)
        self.assertNotIn("junk", json.loads(self.path.read_text()))

    def test_file_is_private(self):
        preferences.save(dict(preferences.DEFAULTS), self.path)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_no_temporary_file_is_left_behind(self):
        preferences.save(dict(preferences.DEFAULTS), self.path)
        leftovers = [p.name for p in self.path.parent.iterdir()
                     if p.name.startswith(".preferences.")]
        self.assertEqual(leftovers, [])

    def test_every_default_passes_its_own_validator(self):
        for name, (default, valid) in preferences.SCHEMA.items():
            self.assertTrue(valid(default), f"{name} default fails validation")


if __name__ == "__main__":
    unittest.main()
