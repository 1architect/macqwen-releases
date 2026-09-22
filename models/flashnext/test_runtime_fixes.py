"""Regressions for the 2026-09-22 runtime review. No checkpoint needed."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import mlx.core as mx

from models.flashnext import expert_cache, routing, sessions, slab_pack
from models.flashnext.prefill import prefill_language
from models.flashnext.test_prefill import _Cache, _Language


class CanonicalPathTests(unittest.TestCase):
    def test_other_case_spelling_resolves_to_disk_spelling(self):
        with tempfile.TemporaryDirectory() as root:
            real = Path(root) / "Models"
            real.mkdir()
            other = Path(root) / "models"
            if not other.exists():
                self.skipTest("filesystem is case-sensitive")
            self.assertEqual(
                slab_pack.canonical_path(other), slab_pack.canonical_path(real))
            self.assertEqual(slab_pack.canonical_path(other).name, "Models")

    def test_missing_path_falls_back_to_resolve(self):
        missing = Path(tempfile.gettempdir()) / "flashnext-no-such-dir-xyz"
        self.assertEqual(slab_pack.canonical_path(missing), missing.resolve())


class SlabPackPruneTests(unittest.TestCase):
    def test_prunes_only_long_unused_packs(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            current = root / "slab-pack-slots60-current.bin"
            fresh = root / "slab-pack-slots60-fresh.bin"
            stale = root / "slab-pack-slots60-stale.bin"
            other = root / "unrelated.bin"
            for path in (current, fresh, stale, other):
                path.write_bytes(b"x")
            old = time.time() - 30 * 86400
            for path in (current, stale, other):
                os.utime(path, (old, old))
            with patch.dict(os.environ, {"FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS": "14"}):
                removed = slab_pack._mark_used_and_prune(current)
            self.assertEqual(removed, 1)
            self.assertTrue(current.exists())
            self.assertGreater(current.stat().st_mtime, old)
            self.assertTrue(fresh.exists())
            self.assertTrue(other.exists())
            self.assertFalse(stale.exists())

    def test_zero_age_disables_pruning(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            current = root / "slab-pack-slots60-a.bin"
            stale = root / "slab-pack-slots60-b.bin"
            for path in (current, stale):
                path.write_bytes(b"x")
            old = time.time() - 365 * 86400
            os.utime(stale, (old, old))
            with patch.dict(os.environ, {"FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS": "0"}):
                self.assertEqual(slab_pack._mark_used_and_prune(current), 0)
            self.assertTrue(stale.exists())


class CumulativeCountsTests(unittest.TestCase):
    def test_merge_decays_history_and_adds_the_turn(self):
        previous = {"3": [(7, 10.0), (9, 1.0)]}
        turn = {3: Counter({9: 4, 11: 2}), 5: Counter({1: 3})}
        merged = routing.merge_cumulative_counts(previous, turn, 0.5)
        self.assertEqual(merged["3"], [(7, 5.0), (9, 4.5), (11, 2.0)])
        self.assertEqual(merged["5"], [(1, 3.0)])

    def test_mode_defaults_to_turn(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_SLAB_COUNTS", None)
            self.assertEqual(routing.slab_counts_mode(), "turn")
        with patch.dict(os.environ, {"FLASHNEXT_SLAB_COUNTS": "bogus"}):
            self.assertEqual(routing.slab_counts_mode(), "turn")

    def test_skew_allocation_uses_cumulative_history_only_when_enabled(self):
        profile = {
            "layers": {},
            "ranked_scores": {},
            "ranked_counts": {"4": [[1, 9], [2, 8], [3, 7], [4, 6]]},
            "cumulative_counts": {"4": [[10, 9.0], [11, 8.0], [12, 7.0], [13, 6.0]]},
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pins.json"
            path.write_text(json.dumps(profile))
            base = {"FLASHNEXT_PIN_CACHE": str(path)}
            for mode, expected in (("turn", [1, 2, 3, 4]),
                                   ("cumulative", [10, 11, 12, 13])):
                env = dict(base, FLASHNEXT_SLAB_COUNTS=mode)
                with patch.dict(os.environ, env):
                    expert_cache._GLOBAL_SLAB_CACHE.clear()
                    allocation = expert_cache.get_skew_slab_allocation(
                        4, min_slots=4, max_slots=6, num_layers=1)
                self.assertEqual(allocation, {4: expected}, mode)
            expert_cache._GLOBAL_SLAB_CACHE.clear()


class SessionFingerprintTests(unittest.TestCase):
    def test_metal_executor_invalidates_sessions(self):
        self.assertIn("metal_runtime.py", sessions._ENGINE_FILES)


class ProfileResetTests(unittest.TestCase):
    def test_reset_keeps_counter_types(self):
        expert_cache._TIMERS["read_tasks"] += 3
        expert_cache._TIMERS["io_wait"] += 0.5
        expert_cache.reset_profile()
        self.assertIsInstance(expert_cache._TIMERS["read_tasks"], int)
        self.assertIsInstance(expert_cache._TIMERS["io_calls"], int)
        self.assertIsInstance(expert_cache._TIMERS["io_wait"], float)
        self.assertEqual(expert_cache._TIMERS["read_tasks"], 0)


class LastRowPrefillTests(unittest.TestCase):
    def test_flag_projects_only_the_last_row_of_a_short_prompt(self):
        language = _Language()
        cache = [_Cache()]
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        with (
            patch("models.flashnext.prefill.PREFILL_LAST_ROW", True),
            patch("models.flashnext.prefill.mx.clear_cache") as clear_cache,
        ):
            _, token = prefill_language(language, ids, cache)
        self.assertEqual(language.calls, [(3, True, True, None)])
        self.assertEqual(language.argmax_shapes, [(1, 1, 1)])
        self.assertEqual(int(token.item()), 3)
        # A short prompt keeps the allocator cache, as the full path does.
        clear_cache.assert_not_called()

    def test_default_keeps_full_logits(self):
        language = _Language()
        prefill_language(language, mx.array([[1, 2]], dtype=mx.int32), [_Cache()])
        self.assertEqual(language.calls, [(2, False, False, None)])


if __name__ == "__main__":
    unittest.main()
