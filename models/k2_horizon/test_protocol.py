from __future__ import annotations

import unittest

from macqwen.text import ThinkingStreamFilter, ToolCallStreamFilter
from macqwen.tools import parse_tool_calls

from models.k2_horizon.protocol import ProtocolTranslator


class ProtocolTests(unittest.TestCase):
    def test_reasoning_tags_are_adapted_before_the_shared_filter(self):
        translator = ProtocolTranslator()
        translated = "".join(
            translator.feed(piece)
            for piece in ("secret</ifm|thi", "nk_fast>answer")
        ) + translator.finish()
        stream_filter = ThinkingStreamFilter(True, False)
        visible = stream_filter.feed(translated) + stream_filter.finish()
        self.assertEqual(visible, "answer")

    def test_json_tool_calls_are_adapted_to_the_existing_parser(self):
        translator = ProtocolTranslator()
        first = translator.feed("checking<ifm|tool_")
        opened = translator.feed("calls><ifm|tool_call>")
        second = translator.feed(
            '{"name":"read_file","arguments":{"path":"README.md",'
            '"start_line":2}}</ifm|tool_call></ifm|tool_calls>'
        )
        translated = first + opened + second + translator.finish()
        self.assertEqual(first, "checking")
        self.assertEqual(opened, "<tool_call>")
        self.assertEqual(
            parse_tool_calls(translated),
            [("read_file", {"path": "README.md", "start_line": 2})],
        )
        stream_filter = ToolCallStreamFilter()
        self.assertEqual(stream_filter.feed(translated), "checking")


if __name__ == "__main__":
    unittest.main()
