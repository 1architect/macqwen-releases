from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from macqwen.conversation import EXTRA_REASONING
from mlx_lm.models.cache import KVCache, RotatingKVCache

from models.bonsai2.backend import BonsaiBackend


class FakeTokenizer:
    eos_token_ids = [1]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)

    def convert_tokens_to_ids(self, token):
        return 999 if token == "<|im_end|>" else None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return "rendered"


class BackendTests(unittest.TestCase):
    def backend(self, **options):
        import sys
        import types

        tokenizer = FakeTokenizer()
        fake_model = types.SimpleNamespace(language_model=object())
        fake_artifact = types.ModuleType("vision_artifact")
        fake_artifact.load_vl_model = lambda *args, **kwargs: (
            fake_model, None, {"modules": []},
        )
        sys.modules.setdefault("vision_artifact", fake_artifact)
        patches = (
            patch(
                "models.bonsai2.backend.resolve_bonsai2",
                return_value=Path("/models/b2"),
            ),
            patch(
                "models.bonsai2.backend.runtime_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"vision_artifact": fake_artifact}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return BonsaiBackend("b2", **options), tokenizer

    def test_template_keeps_xhigh_native_for_bonsai(self):
        backend, tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", tools=[], enable_thinking=True,
            reasoning_effort="xhigh",
        )
        self.assertEqual(tokenizer.options["reasoning_effort"], "xhigh")
        self.assertEqual(tokenizer.options["tool_call_format"], "json")

    def test_low_effort_folds_to_medium(self):
        backend, tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", tools=[], enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertEqual(tokenizer.options["reasoning_effort"], "medium")

    def test_no_thinking_uses_the_matching_think_tag(self):
        backend, _tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", enable_thinking=False,
            reasoning_effort="medium",
        )
        rendered = "".join(chr(value) for value in backend.pending)
        self.assertTrue(rendered.endswith(
            "<|im_start|>assistant\n<think>\n"
            "</think>"
        ))

    def test_server_history_adds_the_thinking_field_bonsai_requires(self):
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
        self.assertEqual(tokenizer.messages[1]["think"], "")

    def test_stop_discards_cache_for_replay_on_hybrid_state(self):
        # KV rewind alone leaves the 48 offset-free GDN states one step
        # ahead, so a stop must drop the whole cache and replay the tape.
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
        self.assertTrue(backend._replay_needed)
        self.assertFalse(backend.turn_closed)

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
        self.assertTrue(backend._replay_needed)

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

    def test_quantized_kv_option_converts_full_attention_caches(self):
        import sys
        import types

        tokenizer = FakeTokenizer()
        fake_model = types.SimpleNamespace(language_model=object())
        fake_artifact = types.ModuleType("vision_artifact")
        fake_artifact.load_vl_model = lambda *args, **kwargs: (
            fake_model, None, {"modules": []},
        )
        patches = (
            patch(
                "models.bonsai2.backend.resolve_bonsai2",
                return_value=Path("/models/b2"),
            ),
            patch(
                "models.bonsai2.backend.runtime_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"vision_artifact": fake_artifact}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[KVCache()],
            ),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        backend = BonsaiBackend("b2", quantized_kv=(8, 64))
        self.assertEqual(
            [type(item).__name__ for item in backend.cache],
            ["QuantizedKVCache"],
        )
        with self.assertRaisesRegex(ValueError, "quantized_kv"):
            BonsaiBackend("b2", quantized_kv=(2, 64))

    def test_rotating_cache_is_rejected_on_reset(self):
        backend, _tokenizer = self.backend()
        with patch(
            "mlx_lm.models.cache.make_prompt_cache",
            return_value=[RotatingKVCache(max_size=4)],
        ):
            with self.assertRaisesRegex(TypeError, "ArraysCache/KVCache"):
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
        self.assertTrue("<|im_end|>" in "".join(
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


if __name__ == "__main__":
    unittest.main()
