"""Generation speed must exclude streaming callback time."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from models.qwen27b import frankenstein_engine as engine_module


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class GenerationTimingTests(unittest.TestCase):
    def run_generation(self, fade):
        clock = FakeClock()
        engine = engine_module.FrankensteinEngine.__new__(
            engine_module.FrankensteinEngine
        )
        engine.pending = [1]
        engine.tape = []
        engine.model = object()
        engine.tokenizer = object()
        engine.cache = []
        engine.sampler = None
        engine.logits_processors = None
        engine.prefill_step_size = 1
        engine.kv_bits = None
        engine.kv_group_size = 64
        engine.quantized_kv_start = 8192
        engine.loop_guard = False
        engine.turn = 0
        engine.stats = []
        engine.cache_bytes = lambda: (0, 0)

        def responses(*_args, **_kwargs):
            for token in (10, 11):
                clock.advance(0.35)
                yield SimpleNamespace(
                    token=token,
                    text="word",
                    prompt_tps=5.0,
                    generation_tps=1.0,
                    peak_memory=0.0,
                    finish_reason=None,
                )

        def on_token(count, _response):
            clock.advance(fade)
            return count < 2

        with patch.object(engine_module, "stream_generate", responses), \
                patch.object(engine_module.time, "perf_counter", clock.read), \
                patch.object(engine_module, "host_mem", return_value=(0.0, 0.0)), \
                patch.object(engine_module.mx, "get_active_memory", return_value=0), \
                patch.object(engine_module.mx, "get_cache_memory", return_value=0):
            _text, stats = engine.generate(
                max_tokens=10, echo=False, on_token=on_token
            )
        return stats

    def test_callback_delay_does_not_change_token_rate(self):
        fast = self.run_generation(0.0)
        slow = self.run_generation(0.2)

        self.assertEqual(fast.gen_tokens, 2)
        self.assertEqual(slow.gen_tokens, 2)
        self.assertAlmostEqual(fast.gen_tps, 1 / 0.35)
        self.assertAlmostEqual(slow.gen_tps, fast.gen_tps)

    def test_retained_stop_reuses_cache_for_tool_error(self):
        engine = engine_module.FrankensteinEngine.__new__(
            engine_module.FrankensteinEngine
        )
        engine.pending = [1, 2]
        engine.tape = []
        engine.model = object()

        class Tokenizer:
            eos_token_ids = [999]

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [ord(character) for character in text]

        engine.tokenizer = Tokenizer()
        engine.cache = [SimpleNamespace(offset=0, nbytes=0)]
        engine.sampler = None
        engine.logits_processors = None
        engine.prefill_step_size = 1
        engine.kv_bits = None
        engine.kv_group_size = 64
        engine.quantized_kv_start = 8192
        engine.loop_guard = False
        engine.turn = 0
        engine.stats = []
        engine.turn_closed = True
        engine.cache_bytes = lambda: (0, 0)
        prompts = []
        outputs = iter(((65, 999), (66, 999)))

        def responses(_model, _tokenizer, prompt, **_kwargs):
            prompts.append(len(prompt))
            engine.cache[0].offset += len(prompt)
            for value in next(outputs):
                engine.cache[0].offset += 1
                yield SimpleNamespace(
                    token=value, text="", prompt_tps=5.0,
                    generation_tps=1.0, peak_memory=0.0,
                    finish_reason="stop" if value == 999 else None,
                )

        with patch.object(engine_module, "stream_generate", responses), \
                patch.object(engine_module, "host_mem", return_value=(0.0, 0.0)), \
                patch.object(engine_module.mx, "get_active_memory", return_value=0), \
                patch.object(engine_module.mx, "get_cache_memory", return_value=0):
            engine.generate(max_tokens=4, echo=False)
            engine.append_tool_results(["FileNotFoundError: missing.txt"])
            added = len(engine.pending)
            engine.generate(max_tokens=4, echo=False)

        self.assertEqual(prompts, [2, added])
        self.assertTrue(engine.turn_closed)
        self.assertEqual(engine.cache_tokens, len(engine.tape))

    def test_partial_prefill_replays_before_the_next_turn(self):
        engine = engine_module.FrankensteinEngine.__new__(
            engine_module.FrankensteinEngine
        )
        engine.pending = [1, 2, 3]
        engine.tape = []
        engine.model = object()
        engine.tokenizer = object()
        engine.cache = [SimpleNamespace(offset=0, nbytes=0)]
        engine.sampler = None
        engine.logits_processors = None
        engine.prefill_step_size = 1
        engine.kv_bits = None
        engine.kv_group_size = 64
        engine.quantized_kv_start = 8192
        engine.loop_guard = False
        engine.turn = 0
        engine.stats = []
        engine.turn_closed = True
        engine._replay_needed = False
        engine.cache_bytes = lambda: (0, 0)
        prompts = []

        def replace_cache():
            engine.cache = [SimpleNamespace(offset=0, nbytes=0)]

        engine._replace_cache = replace_cache

        def responses(_model, _tokenizer, prompt, **kwargs):
            prompts.append(len(prompt))
            if len(prompts) == 1:
                engine.cache[0].offset = 1
                kwargs["prompt_progress_callback"](1, len(prompt))
            else:
                engine.cache[0].offset += len(prompt)
            return ()

        def cancel_progress(_done, _total):
            raise engine_module.GenerationCancelled

        with patch.object(engine_module, "stream_generate", responses), \
                patch.object(engine_module, "host_mem", return_value=(0.0, 0.0)), \
                patch.object(engine_module.mx, "get_active_memory", return_value=0), \
                patch.object(engine_module.mx, "get_cache_memory", return_value=0):
            engine.generate(max_tokens=4, echo=False, progress=cancel_progress)

            self.assertTrue(engine._replay_needed)
            self.assertTrue(engine.check_invariant())
            self.assertEqual(engine.tape, [1, 2, 3])

            engine.pending = [4]
            engine.generate(max_tokens=4, echo=False)

        self.assertEqual(prompts, [3, 4])
        self.assertFalse(engine._replay_needed)
        self.assertEqual(engine.cache_tokens, len(engine.tape))


if __name__ == "__main__":
    unittest.main()
