#!/usr/bin/env python3
"""Small structural tests for the FlashNext session format."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx

from models.flashnext.sessions import SessionError, SessionStore


class ArraysCache:
    def __init__(self, size: int):
        self.cache = [None] * size
        self._left_padding = None
        self._left_padding_advance = 0
        self._lengths = None
        self._lengths_advance = 0


class QSAKVCache:
    step = 4

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.index_keys = None
        self.index_position_ids = None


class FakeLanguage:
    def __init__(self):
        self._position_ids = None
        self._rope_deltas = None

    def make_cache(self):
        return [ArraysCache(4), QSAKVCache()]


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flashnext-session-")
        root = Path(self.temporary.name)
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text("{}")
        (self.model / "model-00001.safetensors").write_bytes(b"fake-model")
        self.language = FakeLanguage()
        self.store = SessionStore(
            root / "sessions", self.model, {"mode": "test"}, self.language
        )

    def tearDown(self):
        self.temporary.cleanup()

    def one_token_cache(self):
        cache = self.language.make_cache()
        cache[0].cache[0] = mx.array([[[1.0]]])
        qsa = cache[1]
        qsa.offset = 1
        qsa.keys = mx.ones((1, 1, 1, 2))
        qsa.values = mx.ones((1, 1, 1, 2))
        qsa.index_keys = mx.ones((1, 1, 2))
        qsa.index_position_ids = mx.array([[0]], dtype=mx.int32)
        self.language._position_ids = mx.array([[0]], dtype=mx.int32)
        self.language._rope_deltas = mx.array([[0]], dtype=mx.int32)
        return cache

    def test_round_trip_all_cache_state(self):
        cache = self.language.make_cache()
        cache[0].cache = [
            mx.arange(6).reshape(1, 2, 3),
            mx.ones((1, 2, 2)),
            None,
            mx.array([[7, 8]], dtype=mx.int32),
        ]
        cache[0]._left_padding = mx.array([2], dtype=mx.int32)
        cache[0]._left_padding_advance = 1
        cache[0]._lengths = mx.array([9], dtype=mx.int32)
        cache[0]._lengths_advance = 3

        qsa = cache[1]
        qsa.offset = 3
        qsa.keys = mx.arange(20, dtype=mx.float32).reshape(1, 1, 5, 4)
        qsa.values = mx.arange(15, dtype=mx.float32).reshape(1, 1, 5, 3)
        qsa.index_keys = mx.arange(18, dtype=mx.float32).reshape(1, 3, 6)
        qsa.index_position_ids = mx.arange(9, dtype=mx.int32).reshape(3, 1, 3)
        self.language._position_ids = mx.arange(9, dtype=mx.int32).reshape(3, 1, 3)
        self.language._rope_deltas = mx.array([[0]], dtype=mx.int32)

        summary = self.store.save("round-trip", cache, [10, 11, 12], False)
        loaded = self.store.load("round-trip")

        self.assertGreater(summary.size_bytes, 0)
        self.assertEqual(loaded.token_ids, [10, 11, 12])
        self.assertFalse(loaded.first_turn)
        self.assertEqual(loaded.cache[0].cache[0].tolist(), cache[0].cache[0].tolist())
        self.assertIsNone(loaded.cache[0].cache[2])
        self.assertEqual(loaded.cache[0]._left_padding.tolist(), [2])
        self.assertEqual(loaded.cache[0]._left_padding_advance, 1)
        self.assertEqual(loaded.cache[0]._lengths_advance, 3)
        self.assertEqual(loaded.cache[1].offset, 3)
        self.assertEqual(loaded.cache[1].keys.shape[2], 4)
        self.assertEqual(
            loaded.cache[1].keys[..., :3, :].tolist(),
            cache[1].keys[..., :3, :].tolist(),
        )
        self.assertEqual(
            loaded.cache[1].index_keys.tolist(), cache[1].index_keys.tolist()
        )
        self.assertEqual(
            loaded.cache[1].index_position_ids.tolist(),
            cache[1].index_position_ids.tolist(),
        )
        self.assertEqual(loaded.position_ids.tolist(), self.language._position_ids.tolist())
        self.assertEqual(loaded.rope_deltas.tolist(), [[0]])

    def test_thinking_mode_round_trip(self):
        summary = self.store.save(
            "thinker", self.one_token_cache(), [10], False, True
        )
        self.assertTrue(summary.thinking)
        self.assertTrue(self.store.load("thinker").thinking)
        self.assertTrue(self.store.list()[0].thinking)

        self.store.save("direct", self.one_token_cache(), [12], False)
        self.assertFalse(self.store.load("direct").thinking)

    def test_thinking_does_not_gate_compatibility(self):
        """Toggling /thinking must never lock a saved conversation out."""
        self.store.save("either", self.language.make_cache(), [], True, True)
        self.assertNotIn("think", json.dumps(self.store.profile))
        self.assertTrue(self.store.load("either").thinking)

    def test_saved_profile_can_select_a_compatible_loader(self):
        self.store.save("profile", self.language.make_cache(), [], True)
        other = SessionStore(
            self.store.directory, self.model, {"mode": "other"}, self.language
        )
        self.assertEqual(other.saved_profile("profile"), {"mode": "test"})
        compatible = SessionStore(
            self.store.directory,
            self.model,
            other.saved_profile("profile"),
            self.language,
        )
        self.assertEqual(compatible.load("profile").token_ids, [])

    def test_empty_session_and_commands(self):
        self.store.save("empty", self.language.make_cache(), [], True)
        loaded = self.store.load("empty")
        self.assertTrue(loaded.first_turn)
        self.assertEqual(loaded.token_ids, [])
        self.assertEqual([item.name for item in self.store.list()], ["empty"])
        self.assertTrue(self.store.delete("empty"))
        self.assertFalse(self.store.delete("empty"))

    def test_rejects_bad_name_and_profile(self):
        with self.assertRaises(SessionError):
            self.store.save("../escape", self.language.make_cache(), [], True)
        self.store.save("valid", self.language.make_cache(), [], True)
        other = SessionStore(
            Path(self.temporary.name) / "sessions",
            self.model,
            {"mode": "other"},
            self.language,
        )
        with self.assertRaisesRegex(SessionError, "another model or profile"):
            other.save("valid", self.language.make_cache(), [], True)
        self.assertTrue(self.store.load("valid").first_turn)
        with self.assertRaisesRegex(SessionError, "profile"):
            other.load("valid")

    def test_detects_payload_corruption_and_truncation(self):
        sessions = Path(self.temporary.name) / "sessions"
        self.store.save("corrupt", self.one_token_cache(), [7], False)
        corrupt = sessions / "corrupt.safetensors"
        with corrupt.open("r+b") as handle:
            handle.seek(-1, 2)
            byte = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([byte[0] ^ 1]))
        with self.assertRaisesRegex(SessionError, "checksum"):
            self.store.load("corrupt")

        self.store.save("truncated", self.one_token_cache(), [8], False)
        truncated = sessions / "truncated.safetensors"
        with truncated.open("r+b") as handle:
            handle.truncate(truncated.stat().st_size - 1)
        summaries = {item.name: item for item in self.store.list()}
        self.assertFalse(summaries["truncated"].valid)

    def test_malformed_header_does_not_escape(self):
        sessions = Path(self.temporary.name) / "sessions"
        sessions.mkdir(exist_ok=True)
        raw = b"[]"
        (sessions / "malformed.safetensors").write_bytes(
            len(raw).to_bytes(8, "little") + raw
        )
        summaries = {item.name: item for item in self.store.list()}
        self.assertFalse(summaries["malformed"].valid)
        with self.assertRaises(SessionError):
            self.store.load("malformed")


if __name__ == "__main__":
    unittest.main()


class EngineFingerprintCoverageTests(unittest.TestCase):
    """Every local file that changes the cache must move the fingerprint.

    A session restored against changed code is silently wrong, and the
    checksum cannot catch it: the payload is intact, the code is not.
    """

    def test_it_covers_each_file_that_shapes_the_cache(self):
        import inspect

        from models.flashnext import sessions

        source = inspect.getsource(sessions._engine_fingerprint)
        for name in (
            "adaptive_topk.py",
            "expert_cache.py",
            "loader.py",
            "ngram.py",
            "patch_rmsnorm.py",
            "prefill.py",
            "qsa_chunk.py",
            "store.py",
        ):
            with self.subTest(name=name):
                self.assertIn(name, source)

    def test_the_listed_files_all_exist(self):
        import inspect
        import re
        from pathlib import Path

        from models.flashnext import sessions

        source = inspect.getsource(sessions._engine_fingerprint)
        listed = re.findall(r'"([a-z_]+\.py)"', source)
        self.assertTrue(listed)
        local = Path(sessions.__file__).resolve().parent
        for name in listed:
            with self.subTest(name=name):
                self.assertTrue((local / name).is_file(), f"{name} is gone")
