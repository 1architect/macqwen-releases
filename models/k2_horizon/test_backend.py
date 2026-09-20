from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from macqwen.backends.base import (
    CANCELLABLE_PREFILL_STEP_SIZE,
    GenerationCancelled,
)
from macqwen.conversation import EXTRA_REASONING
from mlx_lm.models.cache import KVCache, RotatingKVCache

from models.k2_horizon.backend import K2HorizonBackend


class FakeTokenizer:
    eos_token_ids = [1]
    added_tokens_decoder = {}

    def __len__(self):
        return 1000

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)

    def convert_tokens_to_ids(self, token):
        return 999 if token == "<|ifm|im_end|>" else None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return "rendered"


class K2BoundaryTokenizer(FakeTokenizer):
    """Records safe-encoder chunks; structural framing bypasses __call__."""

    def __init__(self, markers):
        from types import SimpleNamespace

        self.added_tokens_decoder = {
            code: SimpleNamespace(content=text) for text, code in markers.items()
        }
        self.chunks = []

    def __call__(self, text, add_special_tokens=False):
        self.chunks.append(text)
        return super().__call__(text, add_special_tokens=add_special_tokens)

    def apply_chat_template(self, messages, **options):
        from models.k2_horizon.backend import IM_END, IM_START

        self.messages = messages
        self.options = options
        return "\n".join(
            f"{IM_START}{message['role']}\n{message['content']}{IM_END}"
            for message in messages
        )


class BackendTests(unittest.TestCase):
    def backend(self, **options):
        tokenizer = FakeTokenizer()
        checkpoint_dir = tempfile.TemporaryDirectory()
        self.addCleanup(checkpoint_dir.cleanup)
        checkpoint = Path(checkpoint_dir.name)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ):
            (checkpoint / name).write_text(f"test {name}")
        patches = (
            patch(
                "models.k2_horizon.backend.resolve_k2_horizon",
                return_value=checkpoint,
            ),
            patch("mlx_lm.load", return_value=(object(), tokenizer)),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return K2HorizonBackend("k2", **options), tokenizer

    def test_template_maps_xhigh_and_keeps_k2_options_inside_the_model(self):
        backend, tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", tools=[], enable_thinking=True,
            reasoning_effort="xhigh",
        )
        self.assertEqual(tokenizer.options["reasoning_effort"], "high")
        self.assertEqual(tokenizer.options["tool_call_format"], "json")

    def test_server_style_high_prompt_is_restored_to_native_k2_high(self):
        backend, tokenizer = self.backend()
        backend.tokenizer.apply_chat_template(
            [{
                "role": "system",
                "content": EXTRA_REASONING["high"] + "\n\nsystem",
            }],
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        self.assertEqual(tokenizer.messages[0]["content"], "system")
        self.assertEqual(tokenizer.options["reasoning_effort"], "high")

    def test_no_thinking_uses_the_matching_k2_tag(self):
        backend, _tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", enable_thinking=False,
            reasoning_effort="medium",
        )
        rendered = "".join(chr(value) for value in backend.pending)
        self.assertTrue(rendered.endswith(
            "<|ifm|im_start|>assistant\n<ifm|think_fast>\n"
            "</ifm|think_fast>"
        ))

    def test_server_history_adds_the_thinking_field_k2_requires(self):
        backend, tokenizer = self.backend()
        backend.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "again"},
            ],
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        self.assertEqual(tokenizer.messages[1]["think_fast"], "")

    def test_stop_is_rewound_so_the_shared_tape_contract_stays_unchanged(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 66, 999):
                backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            text, stats = backend.generate(3)

        self.assertEqual(text, "AB")
        self.assertEqual(stats.finish, "stop")
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(backend.tape, [10, 11, 65, 66])
        self.assertTrue(backend.check_invariant())

    def test_tool_continuation_prefills_only_added_tokens_after_stop(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0
        prompts = []
        outputs = iter(((65, 999), (66, 999)))

        def generate_step(prompt, _model, **options):
            prompts.append(len(prompt))
            for item in backend.cache:
                item.offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in next(outputs):
                for item in backend.cache:
                    item.offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            backend.generate(2)
            backend.append_tool_results(["tool failed"])
            added = len(backend.pending)
            backend.generate(2)

        self.assertEqual(prompts, [2, added])
        self.assertEqual(backend.tape[-1], 66)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

    def test_manual_cancellation_keeps_cache_and_prefills_only_new_turn(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0
        prompts = []
        prefill_steps = []
        checks = [0]

        def generate_step(prompt, _model, **options):
            prompts.append(len(prompt))
            prefill_steps.append(options["prefill_step_size"])
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            # Mirror generate_step's one-ahead cache update for the yielded
            # token. The next loop check must stop before another update.
            backend.cache[0].offset += 1
            yield 65, None

        def should_cancel():
            checks[0] += 1
            return checks[0] > 3

        with patch("mlx_lm.generate.generate_step", generate_step):
            with self.assertRaises(GenerationCancelled):
                backend.generate(2, should_cancel=should_cancel)

        self.assertEqual(backend.tape, [10, 65])
        self.assertFalse(backend._replay_needed)
        self.assertFalse(backend.turn_closed)
        self.assertTrue(backend.check_invariant())

        backend.append_user("next")
        added = len(backend.pending)
        with patch("mlx_lm.generate.generate_step", generate_step):
            backend.generate(1)

        self.assertEqual(prompts, [1, added])
        self.assertEqual(
            prefill_steps,
            [CANCELLABLE_PREFILL_STEP_SIZE, backend.prefill_step_size],
        )

    def test_cache_invariant_checks_every_layer_offset(self):
        backend, _tokenizer = self.backend()
        backend.tape = [10, 11]
        backend.cache = [KVCache(), KVCache()]
        backend.cache[0].offset = 2
        backend.cache[1].offset = 1
        self.assertFalse(backend.check_invariant())
        backend.cache[1].offset = 2
        self.assertTrue(backend.check_invariant())

    def test_cached_append_keeps_all_offsets_aligned_after_stop(self):
        backend, _tokenizer = self.backend()
        backend.tape = [10]
        backend.pending = [11]
        backend.cache = [KVCache(), KVCache()]
        for item in backend.cache:
            item.offset = 1

        def generate_step(prompt, _model, **options):
            for item in backend.cache:
                item.offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 999):
                for item in backend.cache:
                    item.offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            backend.generate(2)

        self.assertEqual(backend.tape, [10, 11, 65])
        self.assertTrue(backend.check_invariant())

    def test_synchronous_generation_setup_failure_replays_tape(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]

        def fail(*_args, **_kwargs):
            raise RuntimeError("setup failed")

        with patch("mlx_lm.generate.generate_step", fail):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                backend.generate(1)

        self.assertEqual(backend.tape, [10])
        self.assertEqual(backend.pending, [])
        self.assertTrue(backend._replay_needed)
        self.assertFalse(backend.turn_closed)

    def test_rotating_cache_is_rejected_on_reset(self):
        backend, _tokenizer = self.backend()
        with patch(
            "mlx_lm.models.cache.make_prompt_cache",
            return_value=[RotatingKVCache(max_size=4)],
        ):
            with self.assertRaisesRegex(TypeError, "ordinary MLX-LM KVCache"):
                backend.reset()

    def test_prefill_step_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "prefill_step_size"):
            self.backend(prefill_step_size=0)

    def test_allocator_cache_limit_is_restored_after_generation(self):
        backend, _tokenizer = self.backend(allocator_cache_mb=256)
        backend.pending = [10]

        def generate_step(prompt, _model, **options):
            options["prompt_progress_callback"](len(prompt), len(prompt))
            yield 65, None

        with patch("mlx_lm.generate.generate_step", generate_step), patch(
            "mlx.core.set_cache_limit", side_effect=[123, None]
        ) as set_cache_limit:
            backend.generate(1)

        self.assertEqual(
            set_cache_limit.call_args_list,
            [call(256 * 1024 * 1024), call(123)],
        )

    def test_post_generation_clear_is_opt_in(self):
        backend, _tokenizer = self.backend(clear_cache_after_generate=True)
        backend.pending = [10]

        def generate_step(prompt, _model, **options):
            options["prompt_progress_callback"](len(prompt), len(prompt))
            yield 65, None

        with patch("mlx_lm.generate.generate_step", generate_step), patch(
            "mlx.core.clear_cache"
        ) as clear_cache:
            backend.generate(1)

        clear_cache.assert_called_once_with()

    def test_wired_limit_wraps_consumption_and_cleanup(self):
        backend, _tokenizer = self.backend(wired_limit_enabled=True)
        backend.pending = [10]
        events = []

        class Steps:
            def __iter__(self):
                events.append("iterate")
                yield 65, None

            def close(self):
                events.append("close")

        def generate_step(_prompt, _model, **options):
            options["prompt_progress_callback"](1, 1)
            return Steps()

        @contextmanager
        def wired(_model, streams):
            self.assertEqual(len(streams), 1)
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        with patch("mlx_lm.generate.generate_step", generate_step), patch(
            "mlx_lm.generate.wired_limit", wired
        ), patch(
            "mlx.core.synchronize", side_effect=lambda *_args: events.append("sync")
        ):
            backend.generate(1)

        self.assertEqual(events, ["enter", "iterate", "close", "sync", "exit"])

    def test_callback_exception_closes_generator_and_requires_replay(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        state = {"closed": False}

        class Steps:
            def __iter__(self):
                yield 65, None
                yield 66, None

            def close(self):
                state["closed"] = True

        def generate_step(_prompt, _model, **options):
            options["prompt_progress_callback"](1, 1)
            return Steps()

        def fail(*_args):
            raise RuntimeError("callback failed")

        with patch("mlx_lm.generate.generate_step", generate_step):
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                backend.generate(2, on_decode_token=fail)

        self.assertTrue(state["closed"])
        self.assertTrue(backend._replay_needed)
        self.assertFalse(backend.turn_closed)
        self.assertEqual(backend.tape, [10, 65])
        self.assertFalse(backend.check_invariant())
        backend.append_user("next")
        self.assertTrue("<|ifm|im_end|>" in "".join(
            chr(value) for value in backend.pending
        ))

    def test_session_replays_the_saved_tape(self):
        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.turn_closed = False
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            backend.reset()
            self.assertTrue(backend.load_session("work").startswith("loaded work"))
        self.assertEqual(backend.tape, [1, 2, 3])
        self.assertFalse(backend.turn_closed)
        self.assertTrue(backend._replay_needed)

    def test_session_rejects_changed_checkpoint_identity(self):
        from models.k2_horizon.backend import _config_identity

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.pending = [4]
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            live_tape, live_pending = list(backend.tape), list(backend.pending)
            baseline = _config_identity(backend.model_path)
            self.assertIsNotNone(baseline)
            checkpoint = Path(backend.model_path)
            (checkpoint / "tokenizer.json").write_text("changed tokenizer")
            self.assertNotEqual(_config_identity(backend.model_path), baseline)
            self.assertIn("could not load", backend.load_session("work"))
            self.assertEqual(backend.tape, live_tape)
            self.assertEqual(backend.pending, live_pending)

    def test_session_rejects_invalid_token_ids(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.pending = [4]
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            payload_path = Path(directory) / "work.json"
            payload = json.loads(payload_path.read_text())
            live_tape, live_pending = list(backend.tape), list(backend.pending)
            for bad_tape in ([1, True, 3], [1, "3", 2], [1, 5000], [1, -2]):
                payload["tape"] = bad_tape
                payload_path.write_text(json.dumps(payload))
                self.assertIn("could not load", backend.load_session("work"))
                self.assertEqual(backend.tape, live_tape)
                self.assertEqual(backend.pending, live_pending)
            payload["tape"] = [1, 2, 3]
            for bad_pending in ([True], ["4"], [5000], [-1]):
                payload["pending"] = bad_pending
                payload_path.write_text(json.dumps(payload))
                self.assertIn("could not load", backend.load_session("work"))
                self.assertEqual(backend.tape, live_tape)
                self.assertEqual(backend.pending, live_pending)

    def test_session_rejects_malformed_schema(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            payload_path = Path(directory) / "work.json"
            payload = json.loads(payload_path.read_text())
            live_tape, live_pending = list(backend.tape), list(backend.pending)
            for bad in ("[1, 2, 3]", "null", '"tape"'):
                payload_path.write_text(bad)
                self.assertIn("could not load", backend.load_session("work"))
                self.assertEqual(backend.tape, live_tape)
                self.assertEqual(backend.pending, live_pending)
            payload["schema"] = 999
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["schema"] = 1
            payload["turn_closed"] = "false"
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["turn_closed"] = False
            payload["thinking"] = 1
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["thinking"] = False
            payload["thinking_tag"] = "bogus-tag"
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            self.assertEqual(backend.tape, live_tape)
            self.assertEqual(backend.pending, live_pending)

    def test_session_round_trips_pending_with_live_state_intact(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.pending = [4, 5]
            backend.turn_closed = False
            backend.thinking_enabled = True
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            payload = json.loads((Path(directory) / "work.json").read_text())
            self.assertEqual(payload["pending"], [4, 5])
            self.assertEqual(payload["schema"], 1)
            self.assertIn("config_sha256", payload)
            backend.tape = [9, 9, 9]
            backend.pending = [8]
            backend.turn_closed = True
            backend.thinking_enabled = False
            self.assertTrue(backend.load_session("work").startswith("loaded work"))
            self.assertEqual(backend.tape, [1, 2, 3])
            self.assertEqual(backend.pending, [4, 5])
            self.assertFalse(backend.turn_closed)
            self.assertTrue(backend.thinking_enabled)
            self.assertTrue(backend._replay_needed)

    def test_hostile_user_text_survives_the_tokenizer_wrapper(self):
        from types import SimpleNamespace

        backend, _tokenizer = self.backend()
        wrapped = backend.tokenizer._tokenizer
        wrapped.added_tokens_decoder = {
            9998: SimpleNamespace(content="</think>"),
        }
        try:
            backend.append_user("paste </think> verbatim")
        finally:
            wrapped.added_tokens_decoder = {}
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("paste </think> verbatim", text)


class K2SafetyTests(unittest.TestCase):
    def backend_with(self, tokenizer):
        patches = (
            patch(
                "models.k2_horizon.backend.resolve_k2_horizon",
                return_value=Path("/models/k2"),
            ),
            patch("mlx_lm.load", return_value=(object(), tokenizer)),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return K2HorizonBackend("k2"), tokenizer

    def test_tokenizer_forwards_call_and_len(self):
        backend, tokenizer = self.backend_with(K2BoundaryTokenizer({}))
        wrapped = backend.tokenizer
        self.assertEqual(
            wrapped("hi", add_special_tokens=False)["input_ids"],
            [ord(character) for character in "hi"],
        )
        self.assertEqual(len(wrapped), len(tokenizer))
        self.assertIn("hi", tokenizer.chunks)

    def test_append_text_keeps_think_close_verbatim(self):
        backend, _tokenizer = self.backend_with(K2BoundaryTokenizer({}))
        backend._thinking_tag = "ifm|think_fast"
        backend.append_text("a </think> b")
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("</think>", text)
        self.assertNotIn("</ifm|think_fast>", text)

    def test_user_paste_with_k2_markers_splits_at_boundaries(self):
        tokenizer = K2BoundaryTokenizer({
            "</think>": 11, "<|ifm|im_end|>": 12, "<|ifm|im_start|>": 13,
            "<ifm|think_fast>": 14, "</ifm|think_fast>": 15,
        })
        backend, _ = self.backend_with(tokenizer)
        backend.append_user(
            "say </think> and <|ifm|im_end|> plus "
            "<ifm|think_fast>x</ifm|think_fast> ok"
        )
        for chunk in tokenizer.chunks:
            self.assertNotIn("</think>", chunk)
            self.assertNotIn("<|ifm|im_end|>", chunk)
            self.assertNotIn("<|ifm|im_start|>", chunk)
            self.assertNotIn("<ifm|think_fast>", chunk)
            self.assertNotIn("</ifm|think_fast>", chunk)
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("say </think> and <|ifm|im_end|> plus", text)
        self.assertIn("<ifm|think_fast>x</ifm|think_fast>", text)

    def test_marker_free_user_text_encodes_jointly(self):
        tokenizer = K2BoundaryTokenizer({"</think>": 11})
        backend, _ = self.backend_with(tokenizer)
        backend.append_user("hello")
        self.assertEqual(tokenizer.chunks, [])
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("<|ifm|im_start|>user\nhello<|ifm|im_end|>", text)

    def test_k2_markers_split_without_added_tokens(self):
        backend, tokenizer = self.backend_with(K2BoundaryTokenizer({}))
        backend.append_user("paste <|ifm|im_start|> verbatim")
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            self.assertNotIn("<|ifm|im_start|>", chunk)
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("paste <|ifm|im_start|> verbatim", text)

    def test_qwen_markers_split_without_added_tokens(self):
        backend, tokenizer = self.backend_with(K2BoundaryTokenizer({}))
        backend.append_user("paste <|im_start|> and <think>x</think> ok")
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            self.assertNotIn("<|im_start|>", chunk)
            self.assertNotIn("<think>", chunk)
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("paste <|im_start|> and <think>x</think> ok", text)

    def test_tool_result_with_markers_splits_at_boundaries(self):
        tokenizer = K2BoundaryTokenizer({"<|ifm|im_end|>": 12})
        backend, _ = self.backend_with(tokenizer)
        backend.append_tool_results(['source = "<|ifm|im_end|>"'])
        for chunk in tokenizer.chunks:
            self.assertNotIn("<|ifm|im_end|>", chunk)
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn('source = "<|ifm|im_end|>"', text)

    def test_open_conversation_splits_pasted_markers(self):
        tokenizer = K2BoundaryTokenizer({
            "</think>": 11, "<|im_end|>": 12,
        })
        backend, _ = self.backend_with(tokenizer)
        backend.open_conversation("sys </think> s", "hi <|im_end|> u")
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            self.assertNotIn("</think>", chunk)
            self.assertNotIn("<|im_end|>", chunk)
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("sys </think> s", text)
        self.assertIn("hi <|im_end|> u", text)

    def test_open_conversation_stays_joint_without_markers(self):
        tokenizer = K2BoundaryTokenizer({"</think>": 11})
        backend, _ = self.backend_with(tokenizer)
        backend.open_conversation("sys", "hello")
        self.assertEqual(tokenizer.chunks, [])
        text = "".join(chr(code) for code in backend.pending)
        self.assertIn("sys", text)
        self.assertIn("hello", text)

    def test_tool_results_are_framed_one_block_each(self):
        backend, _ = self.backend_with(K2BoundaryTokenizer({}))
        backend.append_tool_results(["first", "second"])
        text = "".join(chr(code) for code in backend.pending)
        self.assertEqual(text.count("<|ifm|im_start|>tool"), 2)
        self.assertIn("first", text)
        self.assertIn("second", text)


if __name__ == "__main__":
    unittest.main()
