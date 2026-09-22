from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from macqwen.agent import run_agent

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

    def test_native_xml_tool_calls_are_adapted_to_the_existing_parser(self):
        translator = ProtocolTranslator()
        translated = translator.feed(
            "<ifm|tool_calls>\n<ifm|tool_call>api_docs\n"
            "<ifm|arg_key>library</ifm|arg_key>\n"
            "<ifm|arg_value>sketchup</ifm|arg_value>\n"
            "<ifm|arg_key>topic</ifm|arg_key>\n"
            "<ifm|arg_value>Model extrude</ifm|arg_value>\n"
            "</ifm|tool_call>\n<ifm|tool_call>read_file\n"
            "<ifm|arg_key>path</ifm|arg_key>\n"
            "<ifm|arg_value>README.md</ifm|arg_value>\n"
            "</ifm|tool_call>\n</ifm|tool_calls>"
        ) + translator.finish()
        self.assertEqual(
            parse_tool_calls(translated),
            [
                ("api_docs", {"library": "sketchup", "topic": "Model extrude"}),
                ("read_file", {"path": "README.md"}),
            ],
        )

    def test_xml_calls_survive_split_markers_and_preserve_types(self):
        raw = (
            '<ifm|tool_calls><ifm|tool_call>read_file'
            '<ifm|arg_key>path</ifm|arg_key>'
            '<ifm|arg_type>string</ifm|arg_type>'
            '<ifm|arg_value>README.md</ifm|arg_value>'
            '<ifm|arg_key>start_line</ifm|arg_key>'
            '<ifm|arg_type>integer</ifm|arg_type>'
            '<ifm|arg_value>2</ifm|arg_value>'
            '</ifm|tool_call><ifm|tool_call>list_dir</ifm|tool_call>'
            '</ifm|tool_calls>'
        )
        for size in (1, 3, 17, len(raw)):
            with self.subTest(size=size):
                translator = ProtocolTranslator()
                text = ''.join(translator.feed(raw[i:i + size])
                               for i in range(0, len(raw), size)) + translator.finish()
                self.assertEqual(parse_tool_calls(text), [
                    ('read_file', {'path': 'README.md', 'start_line': 2}),
                    ('list_dir', {}),
                ])

    def test_invalid_call_payloads_do_not_produce_partial_calls(self):
        for raw in (
            '[]', 'null', '42',
            'read_file<ifm|arg_key>path</ifm|arg_key>',
            'read_file<ifm|arg_key>path</ifm|arg_key>'
            '<ifm|arg_value>README.md',
            'read_file<ifm|arg_key>path</ifm|arg_key>'
            '<ifm|arg_value>README.md</ifm|arg_value>junk',
            'list_dir<ifm|arg_key>path</ifm|arg_key>'
            '<ifm|arg_value>a</ifm|arg_value>'
            '<ifm|arg_key>path</ifm|arg_key>'
            '<ifm|arg_value>b</ifm|arg_value>',
        ):
            with self.subTest(raw=raw):
                translator = ProtocolTranslator()
                text = translator.feed(
                    '<ifm|tool_calls><ifm|tool_call>' + raw
                    + '</ifm|tool_call></ifm|tool_calls>'
                ) + translator.finish()
                self.assertEqual(parse_tool_calls(text), [])

    def test_native_xml_tool_calls_reach_the_agent_loop(self):
        class Stats(SimpleNamespace):
            pass

        turns = [
            ("<ifm|tool_calls>\n<ifm|tool_call>list_dir\n"
             "<ifm|arg_key>path</ifm|arg_key>\n"
             "<ifm|arg_value>.</ifm|arg_value>\n"
             "</ifm|tool_call>\n</ifm|tool_calls>",
             Stats(finish="stop", host_free_gb=None, swap_gb=None)),
            ("done", Stats(finish="stop", host_free_gb=None, swap_gb=None)),
        ]

        class Loop:
            thinking_enabled = False
            pending = []

            def __init__(self):
                self.turns = list(turns)
                self.tool_results = []
                self.tool_thinking_flags = []

            def generate(self, max_tokens, out, **_kwargs):
                raw, stats = self.turns.pop(0)
                translator = ProtocolTranslator()
                text = translator.feed(raw) + translator.finish()
                return text, stats

            def check_invariant(self):
                return True

            def append_tool_results(self, results, enable_thinking=True):
                self.tool_results.append(results)
                self.tool_thinking_flags.append(enable_thinking)

        loop = Loop()
        repo = Mock()
        repo.call.return_value = {"entries": []}
        reason = run_agent(loop, repo, lambda _text: None)
        self.assertEqual(reason, "answer")
        repo.call.assert_called_once_with("list_dir", {"path": "."})
        self.assertEqual(len(loop.tool_results), 1)
        self.assertEqual(loop.tool_thinking_flags, [False])
        self.assertEqual(loop.turns, [])


if __name__ == "__main__":
    unittest.main()
