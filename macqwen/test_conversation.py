from __future__ import annotations

import unittest

from macqwen.conversation import IM_END, IM_START, Conversation


class FakeTokenizer:
    """One id per character, so token counts are readable in assertions."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, enable_thinking=True,
                            reasoning_effort="xhigh"):
        parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}" for m in messages]
        if tools:
            parts.insert(0, f"<tools>{len(tools)}</tools>")
        return "\n".join(parts)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.chat = Conversation(FakeTokenizer())

    def test_open_then_reopen_is_refused(self):
        self.chat.open_conversation("sys", "hello")
        with self.assertRaises(RuntimeError):
            self.chat.open_conversation("sys", "again")

    def test_tools_reach_the_template(self):
        self.chat.open_conversation("sys", "hi", tools=[{"a": 1}, {"b": 2}])
        text = "".join(chr(c) for c in self.chat.pending)
        self.assertIn("<tools>2</tools>", text)

    def test_thinking_prefix_switches(self):
        self.assertEqual(
            self.chat._assistant_prefix(True), f"{IM_START}assistant\n<think>\n")
        self.assertIn("</think>", self.chat._assistant_prefix(False))

    def test_open_turn_is_closed_before_the_next(self):
        self.chat.turn_closed = False
        self.chat.append_user("next")
        text = "".join(chr(c) for c in self.chat.pending)
        self.assertTrue(text.startswith(IM_END))

    def test_closed_turn_is_not_closed_twice(self):
        self.chat.turn_closed = True
        self.chat.append_user("next")
        text = "".join(chr(c) for c in self.chat.pending)
        self.assertFalse(text.startswith(IM_END))

    def test_tool_results_are_framed_one_block_each(self):
        self.chat.append_tool_results(["first", "second"])
        text = "".join(chr(c) for c in self.chat.pending)
        self.assertEqual(text.count("<tool_response>"), 2)
        self.assertIn("first", text)
        self.assertIn("second", text)

    def test_invariant_notices_drift(self):
        self.chat.tape = [1, 2, 3]
        self.assertTrue(self.chat.check_invariant())
        self.chat.cache_tokens_override = 2
        type(self.chat).cache_tokens = property(lambda s: 2)
        self.assertFalse(self.chat.check_invariant())
        del type(self.chat).cache_tokens


if __name__ == "__main__":
    unittest.main()
