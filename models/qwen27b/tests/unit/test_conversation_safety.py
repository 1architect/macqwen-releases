"""Hostile-marker safety for FrankensteinEngine conversation builders.

The engine inherits open_conversation/append_user/append_tool_results from
macqwen.conversation.Conversation. These tests prove pasted control markers
never become chat structure, BoundaryTokenizer style.
"""
from __future__ import annotations

import unittest

from macqwen.conversation import IM_END, IM_START, Conversation
from models.qwen27b.frankenstein_engine import FrankensteinEngine


MARKERS = {
    "<|im_start|>": 10,
    "<|im_end|>": 11,
    "<think>": 12,
    "</think>": 13,
}


class FakeTokenizer:
    """One id per character, so token counts stay readable."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, enable_thinking=True,
                            reasoning_effort="xhigh"):
        parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}" for m in messages]
        if tools:
            parts.insert(0, f"<tools>{len(tools)}</tools>")
        return "\n".join(parts)

    def decode(self, ids):
        return "".join(chr(c) for c in ids)


class BoundaryTokenizer(FakeTokenizer):
    """Records chunks the safe content encoder produces."""

    def __init__(self, markers):
        from types import SimpleNamespace

        self.added_tokens_decoder = {
            code: SimpleNamespace(content=text) for text, code in markers.items()
        }
        self.chunks = []

    def __call__(self, text, add_special_tokens=False):
        self.chunks.append(text)
        return {"input_ids": [ord(c) for c in text]}


def make_engine(tokenizer):
    eng = FrankensteinEngine.__new__(FrankensteinEngine)
    eng.tokenizer = tokenizer
    eng.tape = []
    eng.pending = []
    eng.turn_closed = True
    eng._user_codec = None
    return eng


class EngineConversationSafetyTests(unittest.TestCase):
    def test_delegates_to_shared_conversation(self):
        self.assertIs(FrankensteinEngine.open_conversation, Conversation.open_conversation)
        self.assertIs(FrankensteinEngine.append_user, Conversation.append_user)
        self.assertIs(
            FrankensteinEngine.append_tool_results, Conversation.append_tool_results
        )
        self.assertIs(FrankensteinEngine.append_text, Conversation.append_text)

    def test_user_paste_with_all_markers_splits(self):
        tokenizer = BoundaryTokenizer(MARKERS)
        eng = make_engine(tokenizer)
        text = "a <|im_start|> b <|im_end|> c <think> d </think> e"
        eng.append_user(text)
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            for marker in MARKERS:
                self.assertNotIn(marker, chunk)
        decoded = "".join(chr(c) for c in eng.pending)
        self.assertIn(text, decoded)

    def test_marker_free_user_text_encodes_jointly(self):
        tokenizer = BoundaryTokenizer(MARKERS)
        eng = make_engine(tokenizer)
        eng.append_user("hello")
        self.assertEqual(tokenizer.chunks, [])
        decoded = "".join(chr(c) for c in eng.pending)
        self.assertIn(f"{IM_START}user\nhello{IM_END}", decoded)

    def test_tool_results_with_all_markers_split(self):
        tokenizer = BoundaryTokenizer(MARKERS)
        eng = make_engine(tokenizer)
        results = [
            "out <|im_start|> x",
            "out <|im_end|> y",
            "out <think> z",
            "out </think> w",
        ]
        eng.append_tool_results(results)
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            for marker in MARKERS:
                self.assertNotIn(marker, chunk)
        decoded = "".join(chr(c) for c in eng.pending)
        for result in results:
            self.assertIn(result, decoded)

    def test_mixed_tool_results_keep_plain_part_joint(self):
        tokenizer = BoundaryTokenizer(MARKERS)
        eng = make_engine(tokenizer)
        eng.append_tool_results(["plain", "hostile <think> paste"])
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            self.assertNotIn("<think>", chunk)
        decoded = "".join(chr(c) for c in eng.pending)
        self.assertIn("plain", decoded)
        self.assertIn("hostile <think> paste", decoded)

    def test_open_conversation_splits_pasted_markers(self):
        tokenizer = BoundaryTokenizer(MARKERS)
        eng = make_engine(tokenizer)
        eng.open_conversation(
            "sys <|im_start|> s <think> t",
            "hi <|im_end|> u </think> v",
        )
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            for marker in MARKERS:
                self.assertNotIn(marker, chunk)
        decoded = "".join(chr(c) for c in eng.pending)
        self.assertIn("sys <|im_start|> s <think> t", decoded)
        self.assertIn("hi <|im_end|> u </think> v", decoded)

    def test_open_conversation_stays_joint_without_markers(self):
        tokenizer = BoundaryTokenizer(MARKERS)
        eng = make_engine(tokenizer)
        eng.open_conversation("sys", "hello")
        self.assertEqual(tokenizer.chunks, [])
        plain = FakeTokenizer()
        expected = plain.encode(plain.apply_chat_template([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]))
        self.assertEqual(eng.pending, expected)


if __name__ == "__main__":
    unittest.main()
