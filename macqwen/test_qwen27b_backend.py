from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from macqwen.backends.base import CANCELLABLE_PREFILL_STEP_SIZE
from macqwen.backends.frankenstein import FrankensteinBackend


class FakeEngine:
    prefill_step_size = 512
    tape = [1, 2]
    pending = [3]
    cache = []
    cache_tokens = 2

    def __init__(self):
        self.prefill_step_sizes = []

    def generate(self, **kwargs):
        self.prefill_step_sizes.append(kwargs["prefill_step_size"])
        kwargs["progress"](1, 1)
        kwargs["on_token"](1, SimpleNamespace(text="answer"))
        stats = SimpleNamespace(
            finish="stop",
            gen_tokens=2,
            gen_tps=4.0,
            new_prompt_tokens=3,
            prompt_tps=6.0,
            host_free_gb=5.0,
            swap_gb=1.0,
        )
        return "answer", stats

    def check_invariant(self):
        return True


class BackendTests(unittest.TestCase):
    def test_settings_list_startup_values_and_reject_live_changes(self):
        backend = FrankensteinBackend.__new__(FrankensteinBackend)
        backend._startup_settings = {"prefill-step-size": 256, "kv-bits": 4}
        self.assertIn("prefill-step-size", backend.configure(""))
        with self.assertRaises(ValueError):
            backend.configure("prefill-step-size 512")

    def setUp(self):
        self.backend = FrankensteinBackend.__new__(FrankensteinBackend)
        self.backend.engine = FakeEngine()

    def test_generate_adapts_stats_and_streams(self):
        pieces = []
        prefills = []
        progress = []
        text, stats = self.backend.generate(
            20,
            out=pieces.append,
            on_prefilled=lambda: prefills.append(True),
            on_prefill_progress=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(text, "answer")
        self.assertEqual(pieces, ["answer"])
        self.assertEqual(prefills, [True])
        self.assertEqual(progress, [(1, 1)])
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(stats.rate, 4.0)
        self.assertEqual(stats.prompt_tokens, 3)
        self.assertEqual(stats.prompt_rate, 6.0)
        self.assertEqual(self.backend.engine.prefill_step_sizes, [512])

    def test_cancellable_generation_uses_short_prefill_steps(self):
        self.backend.generate(20, should_cancel=lambda: False)
        self.assertEqual(
            self.backend.engine.prefill_step_sizes,
            [CANCELLABLE_PREFILL_STEP_SIZE],
        )

    def test_session_names_cannot_escape_the_session_directory(self):
        from pathlib import Path

        self.backend._session_dir = Path("/sessions")
        for name in ("../outside", "two words", "", "/absolute"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.backend._path(name)


class SessionEngine:
    """Minimal live conversation for session round-trip tests."""

    def __init__(self, cache, tape, model_path):
        self.cache = cache
        self.tape = list(tape)
        self.pending = []
        self.turn_closed = True
        self.stats = []
        self.path = Path(model_path)


def _session_backend(session_dir, model_path, cache, tape):
    backend = FrankensteinBackend.__new__(FrankensteinBackend)
    backend._session_dir = Path(session_dir)
    backend._model_path = str(model_path)
    backend._cache_options = {
        "paged": False,
        "page_size": 256,
        "top_k_pages": 16,
        "resident_pages": 24,
        "spill_dir": None,
        "min_context": 16384,
    }
    backend._startup_settings = {}
    backend.thinking_enabled = False
    backend.engine = SessionEngine(cache, tape, model_path)
    return backend


def _live_kv(offset, seed=0):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    keys = mx.zeros((1, 2, offset, 4)) + seed
    values = mx.zeros((1, 2, offset, 4)) + seed
    cache = KVCache()
    cache.state = (keys, values)
    return cache


def _read_embedded(session_dir, name):
    import mlx.core as mx

    tensors, file_metadata = mx.load(
        str(Path(session_dir) / name / "cache.safetensors"),
        return_metadata=True,
    )
    raw = file_metadata["macqwen_session"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return tensors, json.loads(raw)


def _write_tampered(session_dir, name, tensors, metadata):
    """Write a crafted snapshot under a fresh name.

    The new file goes to a fresh path, never over a file that a loaded
    array still memory-maps. Overwriting such a file corrupts later reads.
    """
    import mlx.core as mx

    copies = {key: mx.array(value) for key, value in tensors.items()}
    mx.eval(list(copies.values()))
    encoded = json.dumps(metadata, separators=(",", ":"))
    directory = Path(session_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(directory / "cache.safetensors"),
        copies,
        metadata={"macqwen_session": encoded},
    )
    (directory / "meta.json").write_text(encoded)


class SessionFingerprintTests(unittest.TestCase):
    def test_save_writes_fingerprint_and_tensor_table(self):
        import mlx.core as mx

        with tempfile.TemporaryDirectory() as root:
            backend = _session_backend(
                root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14]
            )
            result = backend.save_session("work")
            self.assertTrue(result.startswith("saved work"))
            tensors, metadata = _read_embedded(root, "work")
            fingerprint = metadata.get("fingerprint")
            self.assertIsInstance(fingerprint, dict)
            for key in ("model", "tokenizer", "engine", "profile"):
                self.assertIn(key, fingerprint)
            table = metadata.get("tensors")
            self.assertIsInstance(table, dict)
            self.assertTrue(table)
            for key, described in table.items():
                value = tensors[key]
                self.assertEqual(
                    [int(part) for part in value.shape], described["shape"]
                )
                self.assertEqual(str(value.dtype), described["dtype"])
            mx.eval(list(tensors.values()))

    def test_round_trip_restores_tape_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            saved = _session_backend(
                root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14]
            )
            self.assertTrue(saved.save_session("work").startswith("saved"))
            fresh = _session_backend(
                root, Path(root) / "model", [_live_kv(2, seed=5)], [21, 22]
            )
            result = fresh.load_session("work")
            self.assertTrue(result.startswith("loaded work"))
            self.assertEqual(fresh.engine.tape, [11, 12, 13, 14])
            self.assertEqual(fresh.engine.cache[0].offset, 4)

    def test_fingerprint_mismatch_leaves_live_conversation_intact(self):
        mutations = {
            "model": ("different checkpoint", lambda fp: fp.update(model="0" * 64)),
            "tokenizer": (
                "different tokenizer",
                lambda fp: fp.update(tokenizer="0" * 64),
            ),
            "engine": (
                "different engine code",
                lambda fp: fp.update(engine="0" * 64),
            ),
            "profile": (
                "different generation profile",
                lambda fp: fp["profile"].update(paged=True),
            ),
        }
        for field, (reason, mutate) in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                saved = _session_backend(
                    root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14]
                )
                self.assertTrue(saved.save_session("work").startswith("saved"))
                tensors, metadata = _read_embedded(root, "work")
                mutate(metadata["fingerprint"])
                _write_tampered(root, "tampered", tensors, metadata)

                live_cache = _live_kv(2, seed=5)
                backend = _session_backend(
                    root, Path(root) / "model", [live_cache], [21, 22]
                )
                backend.engine.pending = [99]
                result = backend.load_session("tampered")
                self.assertIn("could not load session", result)
                self.assertIn(reason, result)
                self.assertIs(backend.engine.cache[0], live_cache)
                self.assertEqual(backend.engine.tape, [21, 22])
                self.assertEqual(backend.engine.pending, [99])

    def test_tensor_shape_and_dtype_mismatch_leave_live_cache_intact(self):
        with tempfile.TemporaryDirectory() as root:
            saved = _session_backend(
                root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14]
            )
            self.assertTrue(saved.save_session("work").startswith("saved"))

            tensors, metadata = _read_embedded(root, "work")
            first = next(iter(metadata["tensors"]))
            metadata["tensors"][first]["shape"][-1] += 1
            _write_tampered(root, "bad-shape", tensors, metadata)
            live_cache = _live_kv(2, seed=5)
            backend = _session_backend(
                root, Path(root) / "model", [live_cache], [21, 22]
            )
            result = backend.load_session("bad-shape")
            self.assertIn("could not load session", result)
            self.assertIn("invalid shape", result)
            self.assertIs(backend.engine.cache[0], live_cache)
            self.assertEqual(backend.engine.tape, [21, 22])

            tensors, metadata = _read_embedded(root, "work")
            first = next(iter(metadata["tensors"]))
            metadata["tensors"][first]["dtype"] = "int32"
            _write_tampered(root, "bad-dtype", tensors, metadata)
            live_cache = _live_kv(2, seed=5)
            backend = _session_backend(
                root, Path(root) / "model", [live_cache], [21, 22]
            )
            result = backend.load_session("bad-dtype")
            self.assertIn("could not load session", result)
            self.assertIn("invalid dtype", result)
            self.assertIs(backend.engine.cache[0], live_cache)
            self.assertEqual(backend.engine.tape, [21, 22])


class ConversationDelegationTests(unittest.TestCase):
    def test_conversation_calls_reach_the_engine(self):
        backend = FrankensteinBackend.__new__(FrankensteinBackend)
        calls = []

        class DelegatingEngine:
            def open_conversation(self, *args, **kwargs):
                calls.append(("open_conversation", args, kwargs))
                return "opened"

            def append_user(self, *args, **kwargs):
                calls.append(("append_user", args, kwargs))
                return "user"

            def append_text(self, *args, **kwargs):
                calls.append(("append_text", args, kwargs))
                return "text"

            def encode(self, *args, **kwargs):
                calls.append(("encode", args, kwargs))
                return [1, 2]

            def append_tokens(self, *args, **kwargs):
                calls.append(("append_tokens", args, kwargs))
                return "tokens"

            def append_tool_results(self, *args, **kwargs):
                calls.append(("append_tool_results", args, kwargs))
                return "tools"

        backend.engine = DelegatingEngine()
        self.assertEqual(backend.open_conversation("hi"), "opened")
        self.assertEqual(backend.append_user("hi"), "user")
        self.assertEqual(backend.append_text("hi"), "text")
        self.assertEqual(backend.encode("hi"), [1, 2])
        self.assertEqual(backend.append_tokens([1]), "tokens")
        self.assertEqual(backend.append_tool_results("r"), "tools")
        self.assertEqual(
            [name for name, _, _ in calls],
            [
                "open_conversation",
                "append_user",
                "append_text",
                "encode",
                "append_tokens",
                "append_tool_results",
            ],
        )


if __name__ == "__main__":
    unittest.main()
