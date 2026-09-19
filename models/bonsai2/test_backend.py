from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
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
        if text == "</think>":
            return [9998]
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
            patch(
                "models.bonsai2.backend._load_text_model",
                return_value=(fake_model, {"modules": []}),
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

    def test_low_effort_reaches_the_native_template(self):
        backend, tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", tools=[], enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertEqual(tokenizer.options["reasoning_effort"], "low")

    def test_unknown_effort_fails_closed(self):
        backend, _tokenizer = self.backend()
        with self.assertRaisesRegex(ValueError, "unsupported Bonsai-2"):
            backend.tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                add_generation_prompt=True,
                reasoning_effort="high",
            )

    def test_no_thinking_reaches_the_official_template(self):
        backend, tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", enable_thinking=False,
            reasoning_effort="medium",
        )
        # The wrapper must not invent its own assistant suffix: the official
        # template renders the closed think block, including its newlines.
        self.assertFalse(tokenizer.options["enable_thinking"])
        self.assertTrue(tokenizer.options["add_generation_prompt"])

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

    def test_im_end_stop_is_retained_with_the_live_cache(self):
        # The close token is already consumed into every cache layer, so
        # retaining it keeps tape and cache identical with no replay.
        backend, _tokenizer = self.backend(retain_stop=True)
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
        self.assertEqual(backend.tape, [10, 11, 65, 66, 999])
        self.assertTrue(backend.turn_closed)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

    def test_retention_stays_off_by_default(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 999):
                backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(2)

        self.assertEqual(stats.finish, "stop")
        self.assertEqual(backend.tape, [10, 11, 65])
        self.assertFalse(backend.turn_closed)
        self.assertTrue(backend._replay_needed)

    def test_other_stops_keep_the_replay_recovery_path(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 1):
                backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(3)

        self.assertEqual(stats.finish, "stop")
        self.assertEqual(backend.tape, [10, 11, 65])
        self.assertFalse(backend.turn_closed)
        self.assertTrue(backend._replay_needed)

    def test_cache_invariant_checks_every_layer_offset(self):
        backend, _tokenizer = self.backend()
        backend.tape = [10, 11]
        backend.cache = [KVCache(), KVCache()]
        backend.cache[0].offset = 2
        backend.cache[1].offset = 1
        self.assertFalse(backend.check_invariant())
        backend.cache[1].offset = 2
        self.assertTrue(backend.check_invariant())

    def test_pristine_replay_state_is_valid(self):
        # After a normal stop the replacement cache is empty while the tape
        # is authoritative. That state must read valid: the next turn
        # replays the tape instead of continuing a broken cache.
        backend, _tokenizer = self.backend()
        backend.tape = [10, 11]
        backend.cache = []
        backend._replay_needed = True
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
        self.assertFalse(backend.turn_closed)
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
            patch(
                "models.bonsai2.backend._load_text_model",
                return_value=(fake_model, {"modules": []}),
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

    def test_text_loader_rejects_foreign_schema(self):
        import json
        import tempfile

        from models.bonsai2.backend import _load_text_model

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
            with self.assertRaisesRegex(ValueError, "Unsupported packed model schema"):
                _load_text_model(path)

    def test_text_loader_validates_language_weights_strictly(self):
        # The loader must never silently retain initialized values for
        # missing weights. Pin the strict call by source contract: the live
        # strict load against the real checkpoint is verified manually.
        from models.bonsai2 import backend as backend_module

        source = Path(backend_module.__file__).read_text()
        self.assertIn("strict=True", source)
        self.assertNotIn("strict=False", source)

    def test_weight_loader_merges_shards_and_rejects_duplicates(self):
        import json
        import tempfile

        from models.bonsai2.backend import _load_weight_tensors

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"},
            }))
            shard_a = {"a": "tensor-a"}
            shard_b = {"b": "tensor-b"}

            def fake_load(name):
                return dict(shard_a if name.endswith("s1.safetensors") else shard_b)

            with patch("mlx.core.load", side_effect=fake_load):
                merged = _load_weight_tensors(path)
            self.assertEqual(merged, {"a": "tensor-a", "b": "tensor-b"})

            with patch(
                "mlx.core.load", side_effect=[{"a": 1}, {"a": 2}]
            ):
                with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
                    _load_weight_tensors(path)

    def test_packed_geometry_validation_rejects_bad_records(self):
        import mlx.core as mx
        from mlx.nn import Linear
        from models.bonsai2.backend import _validate_packed_record

        def arrays(rows, width, signs=True):
            out = [
                mx.zeros((rows, width // 16), dtype=mx.uint32),
                mx.ones((rows, width // 128), dtype=mx.float16),
                mx.zeros((rows, width // 128), dtype=mx.float16),
            ]
            sign = mx.ones((width,), dtype=mx.float32) if signs else None
            return out, sign

        rows, width = 8, 256
        module = Linear(width, rows)
        record = {"embedding": False}
        good, signs = arrays(rows, width)
        _validate_packed_record(module, record, good, signs, 64)

        bad_shapes, _ = arrays(rows + 8, width)
        with self.assertRaisesRegex(ValueError, "packed weight"):
            _validate_packed_record(module, record, bad_shapes, signs, 64)

        _, no_signs = arrays(rows, width, signs=False)
        with self.assertRaisesRegex(ValueError, "sign vector"):
            _validate_packed_record(module, record, good, no_signs, 64)

        wrong_kind = dict(record, embedding=True)
        with self.assertRaisesRegex(ValueError, "kind mismatch"):
            _validate_packed_record(module, wrong_kind, good, signs, 64)

        with self.assertRaisesRegex(ValueError, "linear layer"):
            _validate_packed_record(object(), record, good, signs, 64)

    def test_think_budget_forces_closure_and_answer_budget_caps(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.thinking_enabled = True
        backend._interactive_budgets = (5, 3)
        seen = []

        def generate_step(prompt, _model, **options):
            options["prompt_progress_callback"](len(prompt), len(prompt))
            index = 0
            while True:
                index += 1
                yield 100 + (index % 800), None

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(100)

        # 3 thinking tokens with the last forced to the </think> close,
        # then 5 answer tokens before the cap stops generation.
        self.assertEqual(stats.tokens, 8)
        self.assertEqual(stats.finish, "length")
        self.assertFalse(backend.turn_closed)
        self.assertIn(9998, backend.tape)

    def test_answer_budget_caps_after_forced_closure(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.thinking_enabled = True
        backend._interactive_budgets = (4, 1000)

        def generate_step(prompt, _model, **options):
            options["prompt_progress_callback"](len(prompt), len(prompt))
            index = 0
            while True:
                index += 1
                yield 100 + (index % 800), None

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(100000)

        self.assertEqual(stats.tokens, 1004)
        self.assertEqual(stats.finish, "length")
        self.assertIn(9998, backend.tape)

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
        self.assertTrue(backend.check_invariant())
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


CHECKPOINT = Path(os.environ.get(
    "MACQWEN_MODEL_ROOT", "~/models")).expanduser() / "Ternary-Bonsai-2-27B-mlx-2bit"


def _needs_checkpoint(test):
    return unittest.skipUnless(
        (CHECKPOINT / "tokenizer.json").is_file(),
        "needs the local Bonsai-2 checkpoint tokenizer",
    )(test)


class TemplateParityTests(unittest.TestCase):
    """Incremental construction must equal one-shot official renders."""

    def official(self, messages, generation_prompt=True, **options):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(CHECKPOINT), fix_mistral_regex=True
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt,
            tool_presentation_format="markdown", tool_call_format="json",
            **options
        )
        return tokenizer.encode(text, add_special_tokens=False)

    @_needs_checkpoint
    def test_second_user_turn_matches(self):
        from macqwen.conversation import Conversation
        from transformers import AutoTokenizer

        first = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
        ]
        options = {"enable_thinking": False, "reasoning_effort": "medium"}
        backend = BonsaiBackend.__new__(BonsaiBackend)
        Conversation.__init__(backend, backend_tokenizer_for_test())
        backend.turn_closed = True
        backend.append_user("Thanks.", enable_thinking=False)
        head = self.official(first, generation_prompt=False, **options)
        full = self.official(
            first + [{"role": "user", "content": "Thanks."}], **options
        )
        self.assertEqual(list(head) + list(backend.pending), list(full))

    @_needs_checkpoint
    def test_tool_results_match_in_each_count(self):
        from macqwen.conversation import Conversation
        from transformers import AutoTokenizer

        for count in (1, 3):
            with self.subTest(count=count):
                first = [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "List it."},
                    {"role": "assistant", "content": "Here."},
                ]
                options = {"enable_thinking": False, "reasoning_effort": "medium"}
                backend = BonsaiBackend.__new__(BonsaiBackend)
                Conversation.__init__(backend, backend_tokenizer_for_test())
                backend.turn_closed = True
                backend.thinking_enabled = False
                results = [f"outcome {index}" for index in range(count)]
                backend.append_tool_results(results, enable_thinking=False)
                head = self.official(first, generation_prompt=False, **options)
                body = "".join(
                    f"\n<tool_response>\n{result}\n</tool_response>"
                    for result in results
                )
                full = self.official(
                    first + [{"role": "user", "content": body}], **options
                )
                self.assertEqual(list(head) + list(backend.pending), list(full))

    @_needs_checkpoint
    def test_thinking_prefix_matches(self):
        from macqwen.conversation import Conversation
        from transformers import AutoTokenizer

        first = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
        ]
        options = {"enable_thinking": True, "reasoning_effort": "medium"}
        backend = BonsaiBackend.__new__(BonsaiBackend)
        Conversation.__init__(backend, backend_tokenizer_for_test())
        backend.turn_closed = True
        backend.reasoning_effort = "medium"
        backend.append_user("More.", enable_thinking=True)
        head = self.official(first, generation_prompt=False, **options)
        full = self.official(
            first + [{"role": "user", "content": "More."}], **options
        )
        self.assertEqual(list(head) + list(backend.pending), list(full))


def backend_tokenizer_for_test():
    from models.bonsai2.backend import BonsaiTokenizer
    from transformers import AutoTokenizer

    return BonsaiTokenizer(
        AutoTokenizer.from_pretrained(str(CHECKPOINT), fix_mistral_regex=True)
    )


if __name__ == "__main__":
    unittest.main()
