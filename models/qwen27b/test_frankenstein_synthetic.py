"""Synthetic FrankensteinEngine turn tests. No checkpoint. No live model.

Covers item 13 at the engine level: hostile markers, first/second turns,
incremental cache/tape invariant, prefill/decode cancellation, stop-token
behavior, tool-result continuation, and recoverable generation errors.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from macqwen.backends.base import GenerationCancelled
from macqwen.conversation import IM_END, IM_START
from models.qwen27b import frankenstein_engine as engine_module
from models.qwen27b.frankenstein_engine import FrankensteinEngine

MARKERS = {
    "<|im_start|>": 10,
    "<|im_end|>": 11,
    "<think>": 12,
    "</think>": 13,
}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, enable_thinking=True,
                            reasoning_effort="xhigh"):
        parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}" for m in messages]
        return "\n".join(parts)

    def decode(self, ids):
        return "".join(chr(c) for c in ids)


class BoundaryTokenizer(FakeTokenizer):
    def __init__(self, markers):
        from types import SimpleNamespace as NS
        self.added_tokens_decoder = {
            code: NS(content=text) for text, code in markers.items()
        }
        self.chunks = []

    def __call__(self, text, add_special_tokens=False):
        self.chunks.append(text)
        return {"input_ids": [ord(c) for c in text]}


def make_engine(tokenizer=None):
    eng = FrankensteinEngine.__new__(FrankensteinEngine)
    eng.tokenizer = tokenizer or FakeTokenizer()
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


def patches():
    return (
        patch.object(engine_module, "host_mem", return_value=(0.0, 0.0)),
        patch.object(engine_module.mx, "get_active_memory", return_value=0),
        patch.object(engine_module.mx, "get_cache_memory", return_value=0),
    )


def stream_tokens(tokens, finish="stop", record=None, cache=None,
                  finish_map=None):
    def responses(_model, _tokenizer, prompt, **kwargs):
        if record is not None:
            record.append(len(prompt))
        if cache is not None:
            cache[0].offset += len(prompt)
        for i, tok in enumerate(tokens):
            last = i == len(tokens) - 1
            reason = None
            if last:
                if finish_map is not None:
                    reason = finish_map
                else:
                    reason = finish
            if cache is not None:
                cache[0].offset += 1
            yield SimpleNamespace(
                token=tok, text=f"t{tok}", prompt_tps=5.0,
                generation_tps=1.0, peak_memory=0.0,
                finish_reason=reason)
    return responses


class HostileMarkerTests(unittest.TestCase):
    def test_second_turn_hostile_user_splits_and_stays_append_only(self):
        tok = BoundaryTokenizer(MARKERS)
        eng = make_engine(tok)
        eng.open_conversation("sys", "hello")
        self.assertEqual(tok.chunks, [])
        first_pending = list(eng.pending)
        eng.tape.extend(eng.pending)
        eng.pending = []
        tok.chunks.clear()
        hostile = "paste <|im_start|> evil <|im_end|> <think> x </think>"
        eng.append_user(hostile)
        self.assertTrue(tok.chunks)
        for chunk in tok.chunks:
            for marker in MARKERS:
                self.assertNotIn(marker, chunk)
        decoded = "".join(chr(c) for c in eng.pending)
        self.assertIn(hostile, decoded)
        # Tape untouched until generate; pending holds only the new turn.
        self.assertEqual(eng.tape, first_pending)
        self.assertIn(IM_START + "user", decoded)

    def test_hostile_tool_results_split_without_losing_text(self):
        tok = BoundaryTokenizer(MARKERS)
        eng = make_engine(tok)
        eng.append_tool_results(["plain ok", "evil <|im_end|> </think> paste"])
        self.assertTrue(tok.chunks)
        for chunk in tok.chunks:
            for marker in MARKERS:
                self.assertNotIn(marker, chunk)
        decoded = "".join(chr(c) for c in eng.pending)
        self.assertIn("plain ok", decoded)
        self.assertIn("evil <|im_end|> </think> paste", decoded)


class FirstSecondTurnTests(unittest.TestCase):
    def test_two_turns_process_only_new_tokens(self):
        eng = make_engine()
        eng.open_conversation("sys", "hello")
        first_new = len(eng.pending)
        prompts = []
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([101, 102], finish="stop",
                              record=prompts, cache=eng.cache)):
            eng.generate(max_tokens=10, echo=False)
        self.assertEqual(prompts, [first_new])
        self.assertTrue(eng.check_invariant())
        self.assertEqual(eng.cache_tokens, len(eng.tape))
        first_tape_len = len(eng.tape)

        eng.append_user("again")
        second_new = len(eng.pending)
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([201], finish="stop",
                              record=prompts, cache=eng.cache)):
            eng.generate(max_tokens=10, echo=False)
        self.assertEqual(prompts, [first_new, second_new])
        self.assertEqual(len(eng.tape), first_tape_len + second_new + 1)
        self.assertTrue(eng.check_invariant())

    def test_empty_pending_raises(self):
        eng = make_engine()
        with self.assertRaises(RuntimeError):
            eng.generate(max_tokens=4, echo=False)


class CancellationTests(unittest.TestCase):
    def test_prefill_cancellation_sets_replay_and_recovers(self):
        eng = make_engine()
        eng.open_conversation("sys", "hello")
        prompts = []

        def responses(_model, _tokenizer, prompt, **kwargs):
            prompts.append(len(prompt))
            eng.cache[0].offset = 1  # partial prefill before cancel
            kwargs["prompt_progress_callback"](1, len(prompt))
            return iter(())

        def cancel(_done, _total):
            raise GenerationCancelled

        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", responses):
            _text, st = eng.generate(max_tokens=4, echo=False, progress=cancel)
        self.assertEqual(st.finish, "interrupted")
        self.assertFalse(eng.turn_closed)
        self.assertTrue(eng._replay_needed)
        # check_invariant stays true while replay is pending by definition.
        self.assertTrue(eng.check_invariant())
        tape_after_cancel = list(eng.tape)
        self.assertTrue(tape_after_cancel)

        eng.pending = [ord("x")]
        prompts2 = []

        def responses2(_model, _tokenizer, prompt, **kwargs):
            prompts2.append(len(prompt))
            eng.cache[0].offset += len(prompt)
            eng.cache[0].offset += 1
            yield SimpleNamespace(
                token=301, text="t301", prompt_tps=5.0,
                generation_tps=1.0, peak_memory=0.0,
                finish_reason="stop")
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", responses2):
            # Replay path replaces cache then consumes tape+pending.
            seen = {}
            orig_replace = eng._replace_cache

            def counting_replace():
                seen["replaced"] = True
                orig_replace()
            eng._replace_cache = counting_replace
            eng.generate(max_tokens=4, echo=False)
        self.assertTrue(seen.get("replaced"))
        # Full tape replayed plus the one new token.
        self.assertEqual(prompts2, [len(tape_after_cancel) + 1])
        self.assertFalse(eng._replay_needed)
        self.assertTrue(eng.check_invariant())

    def test_decode_cancellation_keeps_partial_tokens(self):
        eng = make_engine()
        eng.open_conversation("sys", "hello")
        state = {"calls": 0}

        def on_token(_count, _result):
            state["calls"] += 1
            if state["calls"] >= 2:
                raise GenerationCancelled
            return True

        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([101, 102, 103], finish="stop",
                              cache=eng.cache)):
            _text, st = eng.generate(max_tokens=10, echo=False,
                                     on_token=on_token)
        self.assertEqual(st.finish, "interrupted")
        self.assertEqual(st.gen_tokens, 2)
        self.assertFalse(eng._replay_needed)
        self.assertFalse(eng.turn_closed)
        self.assertTrue(eng.check_invariant())
        # Next turn must close the truncated assistant turn.
        before = len(eng.pending)
        eng.append_user("next")
        decoded = "".join(chr(c) for c in eng.pending[before - before:])
        self.assertIn(IM_END, decoded)


class StopTokenTests(unittest.TestCase):
    def run_finish(self, finish):
        eng = make_engine()
        eng.open_conversation("sys", "hello")
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([101], finish=finish, cache=eng.cache)):
            _text, st = eng.generate(max_tokens=10, echo=False)
        return eng, st

    def test_stop_closes_turn(self):
        eng, st = self.run_finish("stop")
        self.assertEqual(st.finish, "stop")
        self.assertTrue(eng.turn_closed)

    def test_length_leaves_turn_open(self):
        eng, st = self.run_finish("length")
        self.assertEqual(st.finish, "length")
        self.assertFalse(eng.turn_closed)

    def test_callback_stop_leaves_turn_open(self):
        eng = make_engine()
        eng.open_conversation("sys", "hello")
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([101, 102], finish="stop", cache=eng.cache)):
            _text, st = eng.generate(
                max_tokens=10, echo=False,
                on_token=lambda _c, _r: False)
        self.assertEqual(st.finish, "callback")
        self.assertFalse(eng.turn_closed)


class ToolContinuationTests(unittest.TestCase):
    def test_tool_result_continues_with_only_new_tokens(self):
        eng = make_engine()
        eng.open_conversation("sys", "do work", tools=None)
        prompts = []
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([101], finish="stop",
                              record=prompts, cache=eng.cache)):
            eng.generate(max_tokens=10, echo=False)
        self.assertTrue(eng.turn_closed)
        eng.append_tool_results(['{"ok": true}'])
        added = len(eng.pending)
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([201], finish="stop",
                              record=prompts, cache=eng.cache)):
            eng.generate(max_tokens=10, echo=False)
        self.assertEqual(prompts[1], added)
        self.assertTrue(eng.check_invariant())


class RecoverableErrorTests(unittest.TestCase):
    def test_runtime_error_leaves_pending_and_tape_untouched(self):
        eng = make_engine()
        eng.open_conversation("sys", "hello")
        pending_before = list(eng.pending)
        tape_before = list(eng.tape)

        def boom(*_a, **_k):
            raise RuntimeError("boom")

        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", boom):
            with self.assertRaises(RuntimeError):
                eng.generate(max_tokens=4, echo=False)
        self.assertEqual(eng.pending, pending_before)
        self.assertEqual(eng.tape, tape_before)
        self.assertEqual(eng.turn, 0)
        # Retry works after the failure.
        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate",
                stream_tokens([101], finish="stop", cache=eng.cache)):
            _text, st = eng.generate(max_tokens=4, echo=False)
        self.assertEqual(st.finish, "stop")
        self.assertTrue(eng.check_invariant())

    def test_keyboard_interrupt_before_first_token_needs_replay(self):
        eng = make_engine()
        eng.open_conversation("sys", "hello")

        def interrupt(*_a, **_k):
            raise KeyboardInterrupt

        h1, h2, h3 = patches()
        with h1, h2, h3, patch.object(
                engine_module, "stream_generate", interrupt):
            _text, st = eng.generate(max_tokens=4, echo=False)
        self.assertEqual(st.finish, "interrupted")
        self.assertTrue(eng.tape)
        # Cache (offset 0) disagrees with tape, so replay is required.
        self.assertTrue(eng._replay_needed)
        self.assertTrue(eng.check_invariant())


if __name__ == "__main__":
    unittest.main()
