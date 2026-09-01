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


class ReasoningEffortTests(unittest.TestCase):
    """`high` fills the gap the chat template leaves.

    The template maps effort to one sentence of system text. `xhigh` asks for
    validation and alternatives, `low` asks for brevity, and `medium` is empty.
    `high` keeps the validation and adds a stopping rule, so it has to reach
    the model as system text under an effort name the template accepts.
    """

    def test_known_levels_pass_through_untouched(self):
        from macqwen.conversation import reasoning_system_text

        for level in ("low", "medium", "xhigh"):
            text, effort = reasoning_system_text("You are helpful.", level)
            self.assertEqual(text, "You are helpful.")
            self.assertEqual(effort, level)

    def test_high_rides_in_the_system_turn(self):
        from macqwen.conversation import reasoning_system_text

        text, effort = reasoning_system_text("You are helpful.", "high")
        self.assertEqual(effort, "medium")
        self.assertTrue(text.endswith("You are helpful."))
        self.assertIn("validate key assumptions", text)
        self.assertIn("choose one", text)

    def test_high_uses_a_template_effort_the_template_accepts(self):
        from macqwen.conversation import TEMPLATE_EFFORT

        self.assertIn(TEMPLATE_EFFORT["high"], ("low", "medium", "xhigh"))

    def test_high_survives_an_empty_system_prompt(self):
        from macqwen.conversation import reasoning_system_text

        text, _ = reasoning_system_text("", "high")
        self.assertIn("validate key assumptions", text)
        self.assertFalse(text.startswith("\n"))

    def test_the_preference_accepts_high(self):
        from macqwen.preferences import SCHEMA

        _, valid = SCHEMA["effort"]
        self.assertTrue(valid("high"))
        self.assertFalse(valid("enormous"))
