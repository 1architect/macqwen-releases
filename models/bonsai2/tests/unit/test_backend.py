from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import call, patch

from macqwen.backends.base import (
    CANCELLABLE_PREFILL_STEP_SIZE,
    GenerationCancelled,
)
from macqwen.conversation import EXTRA_REASONING
from mlx_lm.models.cache import KVCache, RotatingKVCache

from models.bonsai2.backend import BonsaiBackend


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
        # A real directory with identity files, so session fingerprinting
        # runs its strict path instead of skipping for a missing checkpoint.
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
                "models.bonsai2.backend.resolve_bonsai2",
                return_value=checkpoint,
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
        options.setdefault("prepared_qmm_metadata", False)
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

    def test_im_end_stop_stays_out_of_the_live_cache(self):
        # The stop is sampled but never fed through the recurrent model, so
        # the next tool/user append can reuse the live cache directly.
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0
        prompts = []

        def generate_step(prompt, _model, **options):
            prompts.append(len(prompt))
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 66, 999):
                if value not in backend.stops:
                    backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            text, stats = backend.generate(3)

        self.assertEqual(text, "AB")
        self.assertEqual(stats.finish, "stop")
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(backend.tape, [10, 11, 65, 66])
        self.assertFalse(backend.turn_closed)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

        backend.append_tool_results(["FileNotFoundError: missing.txt"])
        added = len(backend.pending)
        with patch("mlx_lm.generate.generate_step", generate_step):
            backend.generate(3)

        self.assertEqual(prompts, [2, added])
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

    def test_stop_aware_model_skips_only_post_prefill_stop_calls(self):
        import mlx.core as mx

        from models.bonsai2.backend import _NonConsumingStopModel

        calls = []

        def model(inputs, cache=None):
            del cache
            calls.append(inputs.tolist())
            return mx.zeros((1, inputs.shape[-1], 7))

        wrapped = _NonConsumingStopModel(
            model, prompt_length=3, prefill_step_size=2, stops={999}
        )
        wrapped(mx.array([[10]], dtype=mx.uint32))
        wrapped(mx.array([[11]], dtype=mx.uint32))
        wrapped(mx.array([[999]], dtype=mx.uint32))
        wrapped(mx.array([[12]], dtype=mx.uint32))

        self.assertEqual(calls, [[[10]], [[11]], [[12]]])

    def test_stop_aware_model_keeps_logits_metadata_only(self):
        import mlx.core as mx

        from models.bonsai2.backend import _NonConsumingStopModel

        def model(inputs, cache=None):
            del cache
            return mx.ones((1, inputs.shape[-1], 7), dtype=mx.float16)

        wrapped = _NonConsumingStopModel(
            model, prompt_length=2, prefill_step_size=2, stops={999}
        )
        wrapped(mx.array([[10]], dtype=mx.uint32))
        self.assertEqual(wrapped._last_logits_shape, (1, 1, 7))
        self.assertEqual(wrapped._last_logits_dtype, mx.float16)
        self.assertFalse(hasattr(wrapped, "_last_logits"))
        with patch.object(mx, "zeros_like", side_effect=AssertionError):
            skipped = wrapped(mx.array([[999]], dtype=mx.uint32))
        self.assertEqual(tuple(skipped.shape), (1, 1, 7))
        self.assertEqual(skipped.dtype, mx.float16)

    def test_q4_attention_tiles_absolute_causal_rows_without_touching_cache(self):
        import mlx.core as mx

        from models.bonsai2.backend import _bonsai_quantized_attention

        class Cache:
            bits = 4
            group_size = 64
            offset = 10

        queries = mx.ones((1, 4, 6, 2), dtype=mx.float16)
        keys = (
            mx.zeros((1, 2, 10, 1), dtype=mx.uint32),
            mx.ones((1, 2, 10, 1), dtype=mx.float16),
            mx.zeros((1, 2, 10, 1), dtype=mx.float16),
        )
        values = keys
        calls = []

        def stock(tile, tile_keys, tile_values, *, cache, scale, mask):
            calls.append({
                "shape": tuple(tile.shape),
                "keys": tuple(item.shape for item in tile_keys),
                "values": tuple(item.shape for item in tile_values),
                "cache": cache,
                "scale": scale,
                "mask": mask,
            })
            return tile

        output = _bonsai_quantized_attention(
            stock, queries, keys, values, Cache(), 1.0, "causal", 1
        )
        self.assertEqual(tuple(output.shape), tuple(queries.shape))
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call["keys"][0][-2] == 10 for call in calls))
        self.assertTrue(all(call["values"][0][-2] == 10 for call in calls))
        self.assertEqual(calls[0]["mask"].tolist()[0], [True] * 5 + [False] * 5)
        self.assertEqual(calls[-1]["mask"].tolist()[0], [True] * 10)
        self.assertEqual(Cache.offset, 10)

    def test_fused_rejection_preserves_enabled_memory_tiling(self):
        import mlx_vlm.models.qwen3_5.language as language

        from models.bonsai2 import backend as backend_module

        marker = object()
        old_stock_marker = getattr(language, "_bonsai_stock_attention", marker)
        old_dispatch = language.scaled_dot_product_attention

        def stock(*args, **kwargs):
            return "stock"

        try:
            language._bonsai_stock_attention = stock
            language.scaled_dot_product_attention = stock
            with patch.object(
                backend_module, "_bonsai_fused_q4_attention", return_value=None
            ), patch.object(
                backend_module, "_bonsai_quantized_attention", return_value="tiled"
            ) as tiled:
                backend_module._install_q4_attention_tiling(
                    True, 123, fused_q4_attention=True
                )
                result = language.scaled_dot_product_attention(
                    "queries", "keys", "values", "cache", 1.0, "causal"
                )
            self.assertEqual(result, "tiled")
            tiled.assert_called_once_with(
                stock, "queries", "keys", "values", "cache", 1.0,
                "causal", 123,
            )
        finally:
            language.scaled_dot_product_attention = old_dispatch
            if old_stock_marker is marker:
                delattr(language, "_bonsai_stock_attention")
            else:
                language._bonsai_stock_attention = old_stock_marker

    def test_deferred_fused_compiler_failure_falls_back(self):
        from types import SimpleNamespace

        import mlx.core as mx

        from models.bonsai2 import backend as backend_module

        queries = mx.zeros((1, 24, 1, 256), dtype=mx.float32)
        packed = mx.zeros((1, 4, 1, 32), dtype=mx.uint32)
        metadata = mx.zeros((1, 4, 1, 4), dtype=mx.float16)
        cache = SimpleNamespace(bits=4, group_size=64, offset=1)
        counters = backend_module._new_attention_counters()
        backend_module._FUSED_Q4_VALIDATED.clear()
        self.addCleanup(backend_module._FUSED_Q4_VALIDATED.clear)
        with patch(
            "models.bonsai2.q4_attention_kernel.fused_q4_attention",
            return_value=mx.zeros(queries.shape, dtype=queries.dtype),
        ), patch("mlx.core.eval", side_effect=Exception("deferred compile")):
            result = backend_module._bonsai_fused_q4_attention(
                queries, (packed, metadata, metadata),
                (packed, metadata, metadata), cache, 1.0, "causal",
                counters=counters,
            )
        self.assertIsNone(result)
        self.assertEqual(counters["fallback_reasons"], {"compiler_rejection": 1})

    def test_fused_attention_counters_record_selection_and_isolate_runs(self):
        from types import SimpleNamespace

        import mlx.core as mx

        from models.bonsai2 import backend as backend_module

        queries = mx.zeros((1, 24, 1, 256), dtype=mx.float32)
        packed = mx.zeros((1, 4, 1, 32), dtype=mx.uint32)
        metadata = mx.zeros((1, 4, 1, 4), dtype=mx.float16)
        cache = SimpleNamespace(bits=4, group_size=64, offset=1)
        first = backend_module._new_attention_counters()
        second = backend_module._new_attention_counters()
        backend_module._FUSED_Q4_VALIDATED.clear()
        self.addCleanup(backend_module._FUSED_Q4_VALIDATED.clear)
        with patch(
            "models.bonsai2.q4_attention_kernel.fused_q4_attention",
            return_value=mx.zeros(queries.shape, dtype=queries.dtype),
        ), patch("mlx.core.eval"):
            result = backend_module._bonsai_fused_q4_attention(
                queries, (packed, metadata, metadata),
                (packed, metadata, metadata), cache, 1.0, "causal",
                counters=first,
            )
        self.assertEqual(result.dtype, mx.float32)
        self.assertEqual(first["fused_selected"]["total"], 1)
        self.assertEqual(first["fused_fallbacks"]["total"], 0)
        self.assertEqual(second["fused_selected"]["total"], 0)

    def test_fused_attention_counters_record_unsupported_fallback_reason(self):
        from types import SimpleNamespace

        import mlx.core as mx

        from models.bonsai2 import backend as backend_module

        queries = mx.zeros((1, 24, 1, 256), dtype=mx.float32)
        packed = mx.zeros((1, 4, 1, 32), dtype=mx.uint32)
        metadata = mx.zeros((1, 4, 1, 4), dtype=mx.float16)
        cache = SimpleNamespace(bits=8, group_size=64, offset=1)
        counters = backend_module._new_attention_counters()
        result = backend_module._bonsai_fused_q4_attention(
            queries, (packed, metadata, metadata),
            (packed, metadata, metadata), cache, 1.0, "causal",
            counters=counters,
        )
        self.assertIsNone(result)
        self.assertEqual(counters["fused_fallbacks"]["total"], 1)
        self.assertEqual(
            counters["fallback_reasons"], {"cache is not affine Q4": 1}
        )

    def test_fused_attention_counters_record_compiler_rejection(self):
        from types import SimpleNamespace

        import mlx.core as mx

        from models.bonsai2 import backend as backend_module

        queries = mx.zeros((1, 24, 1, 256), dtype=mx.float32)
        packed = mx.zeros((1, 4, 1, 32), dtype=mx.uint32)
        metadata = mx.zeros((1, 4, 1, 4), dtype=mx.float16)
        cache = SimpleNamespace(bits=4, group_size=64, offset=1)
        counters = backend_module._new_attention_counters()
        with patch(
            "models.bonsai2.q4_attention_kernel.fused_q4_attention",
            side_effect=RuntimeError("invalid metal compiler input"),
        ):
            result = backend_module._bonsai_fused_q4_attention(
                queries, (packed, metadata, metadata),
                (packed, metadata, metadata), cache, 1.0, "causal",
                counters=counters,
            )
        self.assertIsNone(result)
        self.assertEqual(
            counters["fallback_reasons"], {"compiler_rejection": 1}
        )

    def test_attention_counters_reset_per_generation(self):
        first, _tokenizer = self.backend()
        second, _tokenizer = self.backend()
        first.attention_counters["fused_selected"]["total"] = 7
        first._reset_attention_counters()
        self.assertEqual(first.attention_counters["fused_selected"]["total"], 0)
        self.assertEqual(second.attention_counters["fused_selected"]["total"], 0)

    def test_eos_stop_reuses_the_live_cache(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 999):
                if value not in backend.stops:
                    backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(2)

        self.assertEqual(stats.finish, "stop")
        self.assertEqual(backend.tape, [10, 11, 65])
        self.assertFalse(backend.turn_closed)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

    def test_other_stops_reuse_the_live_cache(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 1):
                if value not in backend.stops:
                    backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(3)

        self.assertEqual(stats.finish, "stop")
        self.assertEqual(backend.tape, [10, 11, 65])
        self.assertFalse(backend.turn_closed)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

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
        # A fallback replay state remains valid when a runtime reports that
        # its cache cannot continue from the tape.
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
                if value not in backend.stops:
                    for item in backend.cache:
                        item.offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            backend.generate(2)

        self.assertEqual(backend.tape, [10, 11, 65])
        self.assertFalse(backend.turn_closed)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

    def test_generation_on_a_new_thread_replays_instead_of_reusing_cache(self):
        # MLX binds materialized cache state to the generating thread's
        # stream. The chat loop runs each turn on a fresh worker thread,
        # so a live cache from a previous turn is rebuilt here instead
        # of evaluated on the wrong thread. Same-thread generation keeps
        # reusing the live cache.
        import threading

        backend, _tokenizer = self.backend()
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            backend.cache[0].offset += 1
            yield 65, None

        def on_fresh_thread(function):
            result, error = [], []

            def worker():
                try:
                    result.append(function())
                except BaseException as exc:
                    error.append(exc)

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            if error:
                raise error[0]
            return result[0]

        with patch(
            "mlx_lm.models.cache.make_prompt_cache",
            side_effect=lambda *args: [KVCache()],
        ) as make_cache:
            with patch("mlx_lm.generate.generate_step", generate_step):
                backend.pending = [10]
                _text, stats = on_fresh_thread(lambda: backend.generate(2))
                self.assertEqual(stats.tokens, 1)
                self.assertEqual(make_cache.call_count, 0)
                cache_a = backend.cache

                def turns_b_and_c():
                    backend.pending = [12]
                    first = backend.generate(2)
                    rebuilt = backend.cache
                    backend.pending = [13]
                    second = backend.generate(2)
                    return first, second, rebuilt

                (text_b, stats_b), (text_c, stats_c), cache_b = on_fresh_thread(
                    turns_b_and_c
                )

        self.assertEqual(stats_b.tokens, 1)
        self.assertEqual(stats_c.tokens, 1)
        self.assertEqual(make_cache.call_count, 1)
        self.assertIsNot(cache_b, cache_a)
        self.assertIs(backend.cache, cache_b)
        self.assertEqual(backend.tape, [10, 65, 12, 65, 13, 65])
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

    def test_cancellation_keeps_the_replay_recovery_path(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            backend.cache[0].offset += 1
            yield 65, None

        def interrupt(*_args):
            raise KeyboardInterrupt

        with patch("mlx_lm.generate.generate_step", generate_step):
            with self.assertRaises(KeyboardInterrupt):
                backend.generate(2, on_decode_token=interrupt)

        self.assertEqual(backend.tape, [10, 65])
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
        backend = BonsaiBackend(
            "b2", quantized_kv=(8, 64), prepared_qmm_metadata=False
        )
        self.assertEqual(
            [type(item).__name__ for item in backend.cache],
            ["QuantizedKVCache"],
        )
        with self.assertRaisesRegex(ValueError, "quantized_kv"):
            BonsaiBackend(
                "b2", quantized_kv=(2, 64), prepared_qmm_metadata=False
            )

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
            # Mirror generate_step's one-ahead order: the next token is
            # sampled before the current one is yielded, so sampler flags
            # always describe the upcoming token, never the yielded one.
            import mlx.core as mx

            options["prompt_progress_callback"](len(prompt), len(prompt))
            sampler = options["sampler"]
            index = 0

            def sample():
                nonlocal index
                index += 1
                seen.append(index)
                candidate = 100 + (index % 800)
                return sampler(
                    mx.where(mx.arange(1000) == candidate, 1.0, 0.0)
                )

            current = sample()
            while True:
                upcoming = sample()
                yield current, None
                current = upcoming

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(100)

        # 3 thinking tokens with the last forced to the </think> close,
        # then 5 answer tokens before the cap stops generation. Because the
        # double samples one token ahead, the forced close must not shift
        # the phase early: it still closes thinking, not an answer slot.
        self.assertEqual(stats.tokens, 8)
        self.assertEqual(stats.finish, "length")
        self.assertFalse(backend.turn_closed)
        self.assertEqual(
            backend.tape[-8:], [101, 102, 9998, 104, 105, 106, 107, 108]
        )
        # The cap breaks after accepting its last token: every pulled token
        # is taped, so tape and cache agree and no replay is marked.
        self.assertEqual(len(seen), stats.tokens + 1)
        self.assertFalse(backend._replay_needed)

    def test_natural_close_before_forced_boundary_keeps_one_close_and_answer_budget(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.thinking_enabled = True
        backend._interactive_budgets = (2, 3)
        backend.cache = [KVCache()]
        backend.cache[0].offset = 0

        def generate_step(prompt, _model, **options):
            import mlx.core as mx

            options["prompt_progress_callback"](len(prompt), len(prompt))
            backend.cache[0].offset += len(prompt)
            sampler = options["sampler"]
            candidates = iter((101, 9998, 103, 104, 105))

            def sample():
                candidate = next(candidates)
                return sampler(
                    mx.where(mx.arange(10000) == candidate, 1.0, 0.0)
                )

            current = sample()
            while True:
                upcoming = sample()
                backend.cache[0].offset += 1
                yield current, None
                current = upcoming

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(100)

        # The natural close is sampled immediately before the old forced
        # boundary. Lookahead must disarm forcing before it samples again.
        self.assertEqual(stats.tokens, 4)
        self.assertEqual(backend.tape[-4:], [101, 9998, 103, 104])
        self.assertEqual(backend.tape[-4:].count(9998), 1)
        self.assertFalse(backend._replay_needed)
        self.assertTrue(backend.check_invariant())

    def test_unlimited_budget_reaches_generate_step_without_a_cap(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.thinking_enabled = True
        backend._interactive_budgets = (-1, -1)
        seen = []

        def generate_step(prompt, _model, **options):
            seen.append(options["max_tokens"])
            options["prompt_progress_callback"](len(prompt), len(prompt))
            yield 65, None
            yield 1, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            text, stats = backend.generate(-1)

        self.assertEqual(seen, [-1])
        self.assertEqual(text, "A")
        self.assertEqual(stats.finish, "stop")

    def test_capped_answer_emits_its_last_accepted_token(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.thinking_enabled = True
        backend._interactive_budgets = (2, 3)
        decoded = []

        def generate_step(prompt, _model, **options):
            import mlx.core as mx

            options["prompt_progress_callback"](len(prompt), len(prompt))
            sampler = options["sampler"]
            index = 0

            def sample():
                nonlocal index
                index += 1
                candidate = 200 + index
                return sampler(
                    mx.where(mx.arange(1000) == candidate, 1.0, 0.0)
                )

            current = sample()
            while True:
                upcoming = sample()
                yield current, None
                current = upcoming

        with patch("mlx_lm.generate.generate_step", generate_step):
            text, stats = backend.generate(
                100, on_decode_token=lambda value, piece: decoded.append(value)
            )

        # 3 thinking tokens (last forced shut) + 2 answer tokens.
        self.assertEqual(stats.tokens, 5)
        self.assertEqual(stats.finish, "length")
        last = backend.tape[-1]
        # Reply, callbacks, and tape agree at the boundary: the last
        # accepted token is emitted, not dropped.
        self.assertEqual(decoded[-1], last)
        self.assertTrue(text.endswith(chr(last)))
        self.assertEqual(len(decoded), stats.tokens)

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

    def test_tool_argument_strings_normalize_to_objects(self):
        from models.bonsai2.backend import BonsaiTokenizer

        histories = [
            # Chat Completions shape.
            [{"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": '{"path": "a.txt"}',
                }}]}],
            # Flattened shape.
            [{"role": "assistant", "content": "", "tool_calls": [{
                "name": "read_file", "arguments": '{"path": "a.txt"}',
            }]}],
        ]
        for messages in histories:
            with self.subTest(messages=messages):
                tokenizer = BonsaiTokenizer(FakeTokenizer())
                tokenizer.apply_chat_template(messages)
                for message in tokenizer.messages:
                    for call in message["tool_calls"]:
                        function = call.get("function", call)
                        self.assertEqual(
                            function["arguments"], {"path": "a.txt"}
                        )
        # Already an object: untouched. Malformed or non-object: rejected.
        tokenizer = BonsaiTokenizer(FakeTokenizer())
        good = [{"role": "assistant", "content": "", "tool_calls": [{
            "name": "read_file", "arguments": {"path": "a.txt"}}]}]
        tokenizer.apply_chat_template(good)
        for bad in ('{"path":', '[1, 2]', '42'):
            with self.subTest(bad=bad):
                tokenizer = BonsaiTokenizer(FakeTokenizer())
                with self.assertRaises(ValueError):
                    tokenizer.apply_chat_template([{
                        "role": "assistant", "content": "", "tool_calls": [{
                            "name": "read_file", "arguments": bad}]}])

    def test_malformed_budgets_fail_before_tape_moves(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.thinking_enabled = True
        backend._interactive_budgets = ("nope", 3)
        with self.assertRaises(ValueError):
            backend.generate(100)
        self.assertEqual(backend.pending, [10, 11])
        self.assertEqual(backend.tape, [])

    def test_answer_budget_caps_after_forced_closure(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10]
        backend.thinking_enabled = True
        backend._interactive_budgets = (4, 1000)

        def generate_step(prompt, _model, **options):
            import mlx.core as mx

            options["prompt_progress_callback"](len(prompt), len(prompt))
            sampler = options["sampler"]
            index = 0

            def sample():
                nonlocal index
                index += 1
                candidate = 100 + (index % 800)
                return sampler(
                    mx.where(mx.arange(1000) == candidate, 1.0, 0.0)
                )

            current = sample()
            while True:
                upcoming = sample()
                yield current, None
                current = upcoming

        with patch("mlx_lm.generate.generate_step", generate_step):
            _text, stats = backend.generate(100000)

        self.assertEqual(stats.tokens, 1004)
        self.assertEqual(stats.finish, "length")
        self.assertIn(9998, backend.tape)

    def test_forcing_sampler_emits_close_at_budget_then_disarms(self):
        import mlx.core as mx

        from models.bonsai2.backend import _ForcingSampler

        inner = lambda logits: mx.argmax(logits.reshape(-1), axis=-1).reshape(1)
        sampler = _ForcingSampler(inner, 3, 9998)
        peak = lambda want: mx.where(mx.arange(1000) == want, 1.0, 0.0)

        self.assertEqual(int(sampler(peak(101))), 101)
        self.assertFalse(sampler.forced_last)
        self.assertEqual(int(sampler(peak(102))), 102)
        self.assertFalse(sampler.forced_last)
        # Budget boundary: the close token comes out of the sampler, so
        # generate_step consumes it into cache instead of a substituted id.
        self.assertEqual(int(sampler(peak(103))), 9998)
        self.assertTrue(sampler.forced_last)
        # One-shot: later calls pass through to the inner sampler.
        self.assertFalse(sampler.armed)
        self.assertEqual(int(sampler(peak(104))), 104)
        self.assertFalse(sampler.forced_last)

    def test_forcing_sampler_disarms_on_natural_close(self):
        import mlx.core as mx

        from models.bonsai2.backend import _ForcingSampler

        inner = lambda logits: mx.argmax(logits.reshape(-1), axis=-1).reshape(1)
        sampler = _ForcingSampler(inner, 1000, 9998)
        peak = lambda want: mx.where(mx.arange(1000) == want, 1.0, 0.0)

        self.assertEqual(int(sampler(peak(101))), 101)
        sampler.end_thinking()
        self.assertFalse(sampler.armed)
        self.assertEqual(int(sampler(peak(102))), 102)
        self.assertFalse(sampler.forced_last)

    def test_forcing_sampler_disarms_when_natural_close_is_sampled_ahead(self):
        import mlx.core as mx

        from models.bonsai2.backend import _ForcingSampler

        inner = lambda logits: mx.argmax(logits.reshape(-1), axis=-1).reshape(1)
        sampler = _ForcingSampler(inner, 3, 9998)
        peak = lambda want: mx.where(mx.arange(10000) == want, 1.0, 0.0)

        self.assertEqual(int(sampler(peak(101))), 101)
        self.assertEqual(int(sampler(peak(9998))), 9998)
        self.assertFalse(sampler.armed)
        self.assertEqual(int(sampler(peak(103))), 103)
        self.assertFalse(sampler.forced_last)

    def test_checkpoint_identity_covers_tokenizer_and_template(self):
        from models.bonsai2.backend import _IDENTITY_FILES, _config_identity

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in _IDENTITY_FILES:
                (root / name).write_text(f"v1 {name}")
            (root / "runtime").mkdir()
            (root / "runtime" / "runtime.py").write_text("v1 runtime")
            baseline = _config_identity(str(root))
            self.assertIsNotNone(baseline)
            # Unrelated files do not affect the fingerprint.
            (root / "notes.txt").write_text("noise")
            self.assertEqual(_config_identity(str(root)), baseline)
            # Any identity input change invalidates saved tapes.
            for name in list(_IDENTITY_FILES) + ["runtime/runtime.py"]:
                target = root / name
                target.write_text(target.read_text() + " changed")
                self.assertNotEqual(
                    _config_identity(str(root)), baseline, msg=name
                )
                target.write_text(target.read_text().replace(" changed", ""))
            self.assertEqual(_config_identity(str(root)), baseline)
            self.assertIsNone(_config_identity(str(root / "missing")))

    def test_session_round_trips_interactive_budgets(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend._interactive_budgets = (5, 3)
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            backend._interactive_budgets = (9, 9)
            self.assertTrue(backend.load_session("work").startswith("loaded work"))
            # Restored budgets replace whatever a later chat set.
            self.assertEqual(backend._interactive_budgets, (5, 3))
            payload = json.loads((Path(directory) / "work.json").read_text())
            payload["budgets"] = [5, True]
            (Path(directory) / "work.json").write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["budgets"] = [5]
            (Path(directory) / "work.json").write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))

    def test_session_rejects_non_object_payload_and_wild_ids(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.pending = []
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            payload_path = Path(directory) / "work.json"
            payload = json.loads(payload_path.read_text())
            live_tape, live_pending = backend.tape, backend.pending
            for bad in ("[1, 2, 3]", "null", '"tape"'):
                payload_path.write_text(bad)
                self.assertIn("could not load", backend.load_session("work"))
                # Failed loads leave live state untouched.
                self.assertEqual(backend.tape, live_tape)
                self.assertEqual(backend.pending, live_pending)
            payload["tape"] = [1, 5000]
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["tape"] = [1, -2]
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))

    def test_session_round_trips_pending_with_strict_types(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.pending = [4]
            backend.turn_closed = False
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            payload_path = Path(directory) / "work.json"
            payload = json.loads(payload_path.read_text())
            self.assertEqual(payload["pending"], [4])
            self.assertEqual(payload["schema"], 1)
            backend.reset()
            self.assertTrue(backend.load_session("work").startswith("loaded work"))
            self.assertEqual(backend.tape, [1, 2, 3])
            self.assertEqual(backend.pending, [4])

    def test_session_rejects_coerced_types_and_unknown_schema(self):
        import json

        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            payload_path = Path(directory) / "work.json"
            payload = json.loads(payload_path.read_text())

            payload["tape"] = [1, True, 3]
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["tape"] = [1, "3", 2]
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["tape"] = [1, 2, 3]
            payload["turn_closed"] = "false"
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))
            payload["turn_closed"] = False
            payload["schema"] = 999
            payload_path.write_text(json.dumps(payload))
            self.assertIn("could not load", backend.load_session("work"))

    def test_cache_recognition_uses_real_classes_from_both_stacks(self):
        from mlx_lm.models.cache import (
            ArraysCache as LmArrays,
            KVCache as LmKv,
        )
        from mlx_vlm.models.cache import (
            ArraysCache as VlmArrays,
            KVCache as VlmKv,
        )
        from models.bonsai2.backend import BonsaiBackend

        BonsaiBackend._validate_cache([VlmArrays(size=2), VlmKv()])
        BonsaiBackend._validate_cache([LmArrays(size=2), LmKv()])

    def test_vlm_quantized_cache_is_recognized(self):
        from mlx_vlm.models.cache import QuantizedKVCache

        from models.bonsai2.backend import BonsaiBackend

        cache = QuantizedKVCache(group_size=64, bits=8)
        BonsaiBackend._validate_cache([cache])

    def test_kv_environment_toggle_maps_and_defers_to_argument(self):
        import os

        backend, _tokenizer = self.backend()
        self.assertIsNone(backend.quantized_kv)
        with patch.dict(os.environ, {"MACQWEN_BONSAI2_KV": "8"}):
            backend, _tokenizer = self.backend()
            self.assertEqual(backend.quantized_kv, (8, 64))
        with patch.dict(os.environ, {"MACQWEN_BONSAI2_KV": "4"}):
            backend, _tokenizer = self.backend()
            self.assertEqual(backend.quantized_kv, (4, 64))
        with patch.dict(os.environ, {"MACQWEN_BONSAI2_KV": "8"}):
            backend, _tokenizer = self.backend(quantized_kv=(4, 64))
            self.assertEqual(backend.quantized_kv, (4, 64))
        with patch.dict(os.environ, {"MACQWEN_BONSAI2_KV": "2"}):
            with self.assertRaisesRegex(ValueError, "MACQWEN_BONSAI2_KV"):
                self.backend()

    def test_configure_reports_effective_runtime_and_kv_settings(self):
        backend, _tokenizer = self.backend()
        text = backend.configure("")
        self.assertIn("fused-fwht          on", text)
        self.assertIn("kv-cache            fp32", text)
        self.assertEqual(backend.configure("kv-cache"), "kv-cache            fp32")

        backend.tape = [10]
        backend.turn_closed = True
        self.assertEqual(
            backend.configure("kv-cache 8"),
            "kv-cache            8-bit (group 64)",
        )
        self.assertEqual(backend.quantized_kv, (8, 64))
        self.assertTrue(backend._replay_needed)
        self.assertTrue(backend.turn_closed)

        self.assertEqual(
            backend.configure("kv-cache off"),
            "kv-cache            fp32",
        )
        self.assertIsNone(backend.quantized_kv)

        backend.quantized_kv = (8, 64)
        text = backend.configure("")
        self.assertIn("kv-cache            8-bit (group 64)", text)
        self.assertEqual(
            backend.configure("kv-cache"),
            "kv-cache            8-bit (group 64)",
        )
        with self.assertRaisesRegex(ValueError, "expects 4, 8"):
            backend.configure("kv-cache nonsense")

    def test_fused_q4_attention_is_opt_in_and_recorded(self):
        stock, _tokenizer = self.backend()
        self.assertFalse(stock.fused_q4_attention)
        backend, _tokenizer = self.backend(fused_q4_attention=True)
        self.assertTrue(backend.fused_q4_attention)
        self.assertTrue(backend.runtime_settings()["fused_q4_attention"])
        self.assertIn("fused-q4-attention on", backend.configure("all"))

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

    def test_prepared_qmm_default_is_on_and_off_differs(self):
        import inspect

        from models.bonsai2.backend import BonsaiBackend

        self.assertTrue(
            inspect.signature(BonsaiBackend.__init__).parameters[
                "prepared_qmm_metadata"
            ].default
        )
        with patch("models.bonsai2.qmm_metadata.install", return_value={}):
            on_backend, _tokenizer = self.backend(prepared_qmm_metadata=True)
        off_backend, _tokenizer = self.backend(prepared_qmm_metadata=False)
        self.assertTrue(on_backend.prepared_qmm_metadata)
        self.assertFalse(off_backend.prepared_qmm_metadata)
        self.assertNotEqual(
            on_backend.prepared_qmm_metadata, off_backend.prepared_qmm_metadata
        )

    def test_configure_reports_prepared_qmm_value(self):
        with patch("models.bonsai2.qmm_metadata.install", return_value={}):
            on_backend, _tokenizer = self.backend(prepared_qmm_metadata=True)
        off_backend, _tokenizer = self.backend(prepared_qmm_metadata=False)
        self.assertIn("prepared-qmm        on", on_backend.configure("all"))
        self.assertIn("prepared-qmm        off", off_backend.configure("all"))
        self.assertEqual(on_backend.configure("prepared-qmm"), "prepared-qmm        on")
        self.assertEqual(off_backend.configure("prepared-qmm"), "prepared-qmm        off")

    def test_prepared_qmm_live_write_fails_closed_with_restart(self):
        backend, _tokenizer = self.backend(prepared_qmm_metadata=False)
        for text in ("prepared-qmm on", "prepared-qmm off"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "(?i)restart"):
                    backend.configure(text)
                with self.assertRaisesRegex(ValueError, "(?i)startup"):
                    backend.configure(text)
        self.assertFalse(backend.prepared_qmm_metadata)


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
