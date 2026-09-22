"""Synthetic FrankensteinBackend tests. No checkpoint. No live model.

Covers item 13 at the backend level: cancellable prefill/decode, stop
behavior passthrough, session save/load, fingerprint mismatch, malformed
session data, tool-result continuation, and recoverable generation errors.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from macqwen.backends.base import CANCELLABLE_PREFILL_STEP_SIZE
from macqwen.backends.frankenstein import FrankensteinBackend
from models.qwen27b import frankenstein_engine as engine_module


class FakeTokenizer:
    eos_token_ids = [999]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, enable_thinking=True,
                            reasoning_effort="xhigh"):
        from macqwen.conversation import IM_END, IM_START
        return "\n".join(
            f"{IM_START}{m['role']}\n{m['content']}{IM_END}" for m in messages)

    def decode(self, ids):
        return "".join(chr(c) for c in ids)


class SessionEngine:
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
        return_metadata=True)
    raw = file_metadata["macqwen_session"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return tensors, json.loads(raw)


def _write_tampered(session_dir, name, tensors, metadata):
    import mlx.core as mx
    copies = {key: mx.array(value) for key, value in tensors.items()}
    mx.eval(list(copies.values()))
    encoded = json.dumps(metadata, separators=(",", ":"))
    directory = Path(session_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(directory / "cache.safetensors"), copies,
        metadata={"macqwen_session": encoded})
    (directory / "meta.json").write_text(encoded)


def _real_engine():
    from models.qwen27b.frankenstein_engine import FrankensteinEngine
    eng = FrankensteinEngine.__new__(FrankensteinEngine)
    eng.tokenizer = FakeTokenizer()
    eng.model = object()
    eng.cache = [SimpleNamespace(offset=0, nbytes=0)]
    eng.sampler = None
    eng.logits_processors = None
    eng.prefill_step_size = 512
    eng.kv_bits = None
    eng.kv_group_size = 64
    eng.quantized_kv_start = 8192
    eng.loop_guard = False
    eng.turn = 0
    eng.stats = []
    eng.tape = []
    eng.pending = []
    eng.turn_closed = True
    eng._replay_needed = False
    eng._user_codec = None
    eng.cache_bytes = lambda: (0, 0)
    eng._replace_cache = lambda: setattr(
        eng, "cache", [SimpleNamespace(offset=0, nbytes=0)])
    return eng


def _backend_with_engine(engine):
    backend = FrankensteinBackend.__new__(FrankensteinBackend)
    backend.engine = engine
    backend._interactive_budgets = None
    backend.thinking_enabled = False
    return backend


def _stream(tokens, finish="stop", record=None, cache=None):
    def responses(_model, _tokenizer, prompt, **kwargs):
        if record is not None:
            record.append((len(prompt), kwargs.get("prefill_step_size")))
        if cache is not None:
            cache[0].offset += len(prompt)
        for i, tok in enumerate(tokens):
            if cache is not None:
                cache[0].offset += 1
            yield SimpleNamespace(
                token=tok, text=f"t{tok}", prompt_tps=5.0,
                generation_tps=1.0, peak_memory=0.0,
                finish_reason=finish if i == len(tokens) - 1 else None)
    return responses


def _patches():
    return (
        patch.object(engine_module, "host_mem", return_value=(0.0, 0.0)),
        patch.object(engine_module.mx, "get_active_memory", return_value=0),
        patch.object(engine_module.mx, "get_cache_memory", return_value=0),
    )


class BackendCancellationTests(unittest.TestCase):
    def test_prefill_cancel_reports_interrupted_and_replays(self):
        eng = _real_engine()
        eng.open_conversation("sys", "hello")
        backend = _backend_with_engine(eng)
        tape_len = None

        def responses(_model, _tokenizer, prompt, **kwargs):
            eng.cache[0].offset = 1
            kwargs["prompt_progress_callback"](1, len(prompt))
            return iter(())

        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", responses):
            text, stats = backend.generate(
                4, should_cancel=lambda: True)
        self.assertEqual(stats.finish, "interrupted")
        self.assertTrue(eng._replay_needed)
        self.assertFalse(eng.turn_closed)
        tape_len = len(eng.tape)
        self.assertTrue(tape_len)

        # Next turn replays the full tape plus new tokens.
        eng.pending = [ord("x")]
        seen = []

        def responses2(_model, _tokenizer, prompt, **kwargs):
            seen.append(len(prompt))
            eng.cache[0].offset += len(prompt)
            return iter(())

        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", responses2):
            backend.generate(4)
        self.assertEqual(seen, [tape_len + 1])

    def test_decode_cancel_reports_interrupted_and_keeps_partial(self):
        eng = _real_engine()
        eng.open_conversation("sys", "hello")
        backend = _backend_with_engine(eng)
        calls = {"n": 0}

        def should_cancel():
            # False during prefill progress, True once decoding starts.
            return calls["n"] > 1

        orig_stream = _stream([101, 102, 103], finish="stop", cache=eng.cache)

        def responses(_model, _tokenizer, prompt, **kwargs):
            calls["n"] += 1  # progress call
            kwargs["prompt_progress_callback"](len(prompt), len(prompt))
            for item in orig_stream(_model, _tokenizer, prompt, **kwargs):
                calls["n"] += 1
                yield item

        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", responses):
            _text, stats = backend.generate(10, should_cancel=should_cancel)
        self.assertEqual(stats.finish, "interrupted")
        self.assertFalse(eng._replay_needed)
        self.assertFalse(eng.turn_closed)
        self.assertTrue(eng.check_invariant())

    def test_cancellable_prefill_uses_short_steps(self):
        eng = _real_engine()
        eng.open_conversation("sys", "hello")
        backend = _backend_with_engine(eng)
        seen = []
        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                _stream([101], finish="stop", record=seen, cache=eng.cache)):
            backend.generate(4, should_cancel=lambda: False)
        self.assertEqual(seen[0][1], CANCELLABLE_PREFILL_STEP_SIZE)

    def test_stop_finish_passes_through(self):
        for finish, closed in (("stop", True), ("length", False)):
            with self.subTest(finish=finish):
                eng = _real_engine()
                eng.open_conversation("sys", "hello")
                backend = _backend_with_engine(eng)
                h1, h2, h3 = _patches()
                with h1, h2, h3, patch.object(
                        engine_module, "stream_generate",
                        _stream([101], finish=finish, cache=eng.cache)):
                    _text, stats = backend.generate(4)
                self.assertEqual(stats.finish, finish)
                self.assertEqual(eng.turn_closed, closed)

    def test_tool_result_continuation_uses_only_new_tokens(self):
        eng = _real_engine()
        eng.open_conversation("sys", "work")
        backend = _backend_with_engine(eng)
        prompts = []
        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                _stream([101], finish="stop", record=prompts, cache=eng.cache)):
            backend.generate(4)
        backend.append_tool_results(['{"ok": true}'])
        added = len(eng.pending)
        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                _stream([102], finish="stop", record=prompts, cache=eng.cache)):
            _text, stats = backend.generate(4)
        self.assertEqual(prompts[1][0], added)
        self.assertEqual(stats.finish, "stop")
        self.assertTrue(backend.check_invariant())

    def test_runtime_error_restores_budgets_and_pending(self):
        eng = _real_engine()
        eng.open_conversation("sys", "hello")
        backend = _backend_with_engine(eng)
        backend._interactive_budgets = (5, None)
        backend.thinking_enabled = False
        eng._interactive_budgets = "old-budgets"
        eng._thinking_enabled = "old-thinking"
        pending_before = list(eng.pending)

        def boom(*_a, **_k):
            raise RuntimeError("boom")

        h1, h2, h3 = _patches()
        with h1, h2, h3, patch.object(engine_module, "stream_generate", boom):
            with self.assertRaises(RuntimeError):
                backend.generate(4)
        self.assertEqual(eng.pending, pending_before)
        self.assertEqual(eng._interactive_budgets, "old-budgets")
        self.assertEqual(eng._thinking_enabled, "old-thinking")


class SessionRoundTripTests(unittest.TestCase):
    def test_save_load_preserves_flags_and_clears_pending(self):
        with tempfile.TemporaryDirectory() as root:
            saved = _session_backend(
                root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14])
            saved.engine.turn_closed = False
            saved.thinking_enabled = True
            saved.engine.pending = [99]
            saved.engine.stats.append("old")
            self.assertTrue(saved.save_session("work").startswith("saved"))
            fresh = _session_backend(
                root, Path(root) / "model", [_live_kv(2, seed=5)], [21, 22])
            fresh.engine.pending = [7]
            result = fresh.load_session("work")
            self.assertTrue(result.startswith("loaded work"))
            self.assertEqual(fresh.engine.tape, [11, 12, 13, 14])
            self.assertEqual(fresh.engine.pending, [])
            self.assertFalse(fresh.engine.turn_closed)
            self.assertTrue(fresh.thinking_enabled)
            self.assertEqual(fresh.engine.stats, [])


class FingerprintMismatchTests(unittest.TestCase):
    def _tampered(self, root, mutate):
        saved = _session_backend(
            root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14])
        self.assertTrue(saved.save_session("work").startswith("saved"))
        tensors, metadata = _read_embedded(root, "work")
        mutate(metadata)
        _write_tampered(root, "tampered", tensors, metadata)

    def test_version_mismatch_leaves_live_state(self):
        with tempfile.TemporaryDirectory() as root:
            self._tampered(root, lambda m: m["fingerprint"].update(version=999))
            live = _live_kv(2, seed=5)
            backend = _session_backend(
                root, Path(root) / "model", [live], [21, 22])
            backend.engine.pending = [99]
            result = backend.load_session("tampered")
            self.assertIn("could not load session", result)
            self.assertIn("unsupported session fingerprint", result)
            self.assertIs(backend.engine.cache[0], live)
            self.assertEqual(backend.engine.tape, [21, 22])
            self.assertEqual(backend.engine.pending, [99])

    def test_invalid_fingerprint_type_leaves_live_state(self):
        with tempfile.TemporaryDirectory() as root:
            self._tampered(root, lambda m: m.update(fingerprint="nope"))
            live = _live_kv(2, seed=5)
            backend = _session_backend(
                root, Path(root) / "model", [live], [21, 22])
            result = backend.load_session("tampered")
            self.assertIn("could not load session", result)
            self.assertIn("fingerprint is invalid", result)
            self.assertIs(backend.engine.cache[0], live)
            self.assertEqual(backend.engine.tape, [21, 22])


class MalformedSessionTests(unittest.TestCase):
    def _live(self, root):
        live = _live_kv(2, seed=5)
        backend = _session_backend(root, Path(root) / "model", [live], [21, 22])
        backend.engine.pending = [99]
        return backend, live

    def _save_and_tamper(self, root, mutate):
        saved = _session_backend(
            root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14])
        self.assertTrue(saved.save_session("work").startswith("saved"))
        tensors, metadata = _read_embedded(root, "work")
        mutate(metadata)
        _write_tampered(root, "bad", tensors, metadata)

    def assert_intact(self, backend, live, result, fragment):
        self.assertIn("could not load session", result)
        self.assertIn(fragment, result)
        self.assertIs(backend.engine.cache[0], live)
        self.assertEqual(backend.engine.tape, [21, 22])
        self.assertEqual(backend.engine.pending, [99])

    def test_missing_tape(self):
        with tempfile.TemporaryDirectory() as root:
            self._save_and_tamper(root, lambda m: m.pop("tape"))
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"), "no token tape")

    def test_tape_not_a_list(self):
        with tempfile.TemporaryDirectory() as root:
            self._save_and_tamper(root, lambda m: m.update(tape="nope"))
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"), "no token tape")

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as root:
            self._save_and_tamper(root, lambda m: m.update(format=99))
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"),
                "unsupported session format")

    def test_cache_list_length_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            self._save_and_tamper(root, lambda m: m.update(caches=[]))
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"),
                "does not match this model")

    def test_cache_tape_offset_disagreement(self):
        with tempfile.TemporaryDirectory() as root:
            self._save_and_tamper(
                root, lambda m: m.update(tape=[11, 12, 13, 14, 15, 16]))
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"), "disagree")

    def test_invalid_cache_tree_node(self):
        with tempfile.TemporaryDirectory() as root:
            def mutate(m):
                m["caches"][0]["state"] = {"type": "bogus"}
            self._save_and_tamper(root, mutate)
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"),
                "unsupported cache tree node")

    def test_tensor_size_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            def mutate(m):
                first = next(iter(m["tensors"]))
                m["tensors"][first]["nbytes"] += 1
            self._save_and_tamper(root, mutate)
            backend, live = self._live(root)
            self.assert_intact(
                backend, live, backend.load_session("bad"), "invalid size")

    def test_missing_tensor(self):
        with tempfile.TemporaryDirectory() as root:
            saved = _session_backend(
                root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14])
            self.assertTrue(saved.save_session("work").startswith("saved"))
            tensors, metadata = _read_embedded(root, "work")
            tensors.pop(next(iter(tensors)))
            _write_tampered(root, "bad", tensors, metadata)
            backend, live = self._live(root)
            result = backend.load_session("bad")
            self.assertIn("could not load session", result)
            self.assertIs(backend.engine.cache[0], live)
            self.assertEqual(backend.engine.tape, [21, 22])

    def test_missing_session_files(self):
        with tempfile.TemporaryDirectory() as root:
            backend, live = self._live(root)
            result = backend.load_session("absent")
            self.assertIn("could not load session", result)
            self.assertIs(backend.engine.cache[0], live)
            self.assertEqual(backend.engine.tape, [21, 22])

    def test_embedded_metadata_invalid(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as root:
            saved = _session_backend(
                root, Path(root) / "model", [_live_kv(4)], [11, 12, 13, 14])
            self.assertTrue(saved.save_session("work").startswith("saved"))
            tensors, _metadata = _read_embedded(root, "work")
            copies = {k: mx.array(v) for k, v in tensors.items()}
            mx.eval(list(copies.values()))
            directory = Path(root) / "bad"
            directory.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(
                str(directory / "cache.safetensors"), copies,
                metadata={"macqwen_session": "not-json{"})
            (directory / "meta.json").write_text("{}")
            backend, live = self._live(root)
            result = backend.load_session("bad")
            self.assertIn("could not load session", result)
            self.assertIn("embedded session metadata is invalid", result)
            self.assertIs(backend.engine.cache[0], live)


if __name__ == "__main__":
    unittest.main()
