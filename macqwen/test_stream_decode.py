"""A character whose UTF-8 bytes span several tokens must not print as U+FFFD.

The chat decodes tokens as they arrive. Decoding one token of a multi-token
character yields the replacement character, which reached the terminal as a
filled box. Emoji are the common case: U+1F604 is two tokens.
"""
from __future__ import annotations

import unittest


class _Tokenizer:
    """Two tokens that only form a character together, as a real BPE does."""

    PIECES = {1: "a", 2: b"\xf0\x9f\x98", 3: b"\x84", 4: "b"}

    def decode(self, ids):
        raw = b""
        for value in ids:
            piece = self.PIECES[value]
            raw += piece.encode() if isinstance(piece, str) else piece
        return raw.decode("utf-8", errors="replace")


from macqwen.text import (
    CompletedTextBuffer,
    ThinkingStreamFilter,
    ToolCallStreamFilter,
    stream_decode,
)


def stream(tokenizer, ids, limit=8):
    """Drive the shared decoder the way the chat loop does."""
    out, partial = [], []
    for value in ids:
        out.append(stream_decode(tokenizer, partial, value, limit))
    return "".join(out)


class StreamDecodeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _Tokenizer()

    def test_split_character_survives(self):
        ids = [1, 2, 3, 4]
        self.assertEqual(self.tokenizer.decode(ids), "a\U0001F604b")
        self.assertEqual(stream(self.tokenizer, ids), "a\U0001F604b")

    def test_per_token_decode_is_what_broke(self):
        joined = "".join(self.tokenizer.decode([i]) for i in [1, 2, 3, 4])
        self.assertIn("�", joined)

    def test_incomplete_tail_does_not_stall_forever(self):
        # a dangling partial character must still flush once the cap is hit
        out = stream(self.tokenizer, [1, 2], limit=2)
        self.assertTrue(out.startswith("a"))


class CompletedTextTests(unittest.TestCase):
    def test_partial_word_stays_hidden(self):
        buffer = CompletedTextBuffer()
        self.assertEqual(buffer.feed("hel"), [])
        self.assertEqual(buffer.feed("lo "), ["hello "])

    def test_final_word_flushes_when_generation_stops(self):
        buffer = CompletedTextBuffer()
        self.assertEqual(buffer.feed("done"), [])
        self.assertEqual(buffer.finish(), ["done"])

    def test_punctuation_completes_a_phrase(self):
        buffer = CompletedTextBuffer()
        self.assertEqual(buffer.feed("Ready."), ["Ready."])


class ThinkingStreamTests(unittest.TestCase):
    def test_split_tags_do_not_leak(self):
        stream_filter = ThinkingStreamFilter(False, False)
        output = [stream_filter.feed(piece) for piece in (
            "<thi", "nk>secret</thi", "nk>answer"
        )]
        output.append(stream_filter.finish())
        self.assertEqual("".join(output), "answer")

    def test_visible_thinking_has_one_blank_line_before_answer(self):
        stream_filter = ThinkingStreamFilter(True, True)
        output = [stream_filter.feed(piece) for piece in (
            "\nreasoning\n", "\n</think>\n", "\nanswer"
        )]
        output.append(stream_filter.finish())
        self.assertEqual("".join(output), "reasoning\n\nanswer")


class ToolCallStreamTests(unittest.TestCase):
    def test_split_tool_marker_and_protocol_stay_hidden(self):
        stream_filter = ToolCallStreamFilter()
        output = []
        events = []
        for piece in (
            "I will check.\n<tool_", "call>\n<function=list_dir>", "</tool_call>"
        ):
            output.append(stream_filter.feed(piece))
            events.append(stream_filter.take_events())
        output.append(stream_filter.finish())
        self.assertEqual("".join(output), "I will check.\n")
        self.assertIn((True, "list_dir"), events)

    def test_normal_answer_is_unchanged(self):
        stream_filter = ToolCallStreamFilter()
        output = stream_filter.feed("normal answer") + stream_filter.finish()
        self.assertEqual(output, "normal answer")


if __name__ == "__main__":
    unittest.main()
