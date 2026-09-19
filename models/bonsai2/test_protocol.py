from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from macqwen.agent import run_agent

from macqwen.text import ThinkingStreamFilter, ToolCallStreamFilter
from macqwen.tools import parse_tool_calls

from models.bonsai2.protocol import ProtocolTranslator


class ProtocolTests(unittest.TestCase):
    def test_reasoning_tags_pass_through_to_the_shared_filter(self):
        translator = ProtocolTranslator()
        translated = "".join(
            translator.feed(piece)
            for piece in ("secret</th", "ink>answer")
        ) + translator.finish()
        stream_filter = ThinkingStreamFilter(True, False)
        visible = stream_filter.feed(translated) + stream_filter.finish()
        self.assertEqual(visible, "answer")

    def test_json_tool_calls_are_adapted_to_the_existing_parser(self):
        translator = ProtocolTranslator()
        first = translator.feed("checking<tool_")
        opened = translator.feed("call>")
        second = translator.feed(
            '{"name":"read_file","arguments":{"path":"README.md",'
            '"start_line":2}}</tool_call>'
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

    def test_native_qwen_xml_calls_pass_through_intact(self):
        # Regression: the first translator ate non-JSON blocks, hiding the
        # function name from the tool filter so the turn ended with no answer.
        raw = ('<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n'
               '</parameter>\n</function>\n</tool_call>')
        for size in (1, 7, len(raw)):
            with self.subTest(size=size):
                translator = ProtocolTranslator()
                text = ''.join(translator.feed(raw[i:i + size])
                               for i in range(0, len(raw), size)) + translator.finish()
                self.assertEqual(
                    parse_tool_calls(text), [('list_dir', {'path': '.'})])

    def test_split_markers_preserve_typed_arguments(self):
        raw = (
            '<tool_call>{"name":"read_file","arguments":'
            '{"path":"README.md","start_line":2}}</tool_call>'
            '<tool_call>{"name":"list_dir","arguments":{}}</tool_call>'
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

    def test_truncated_outer_close_still_yields_the_completed_call(self):
        translator = ProtocolTranslator()
        text = translator.feed(
            "<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>"
        ) + translator.finish()
        self.assertEqual(
            parse_tool_calls(text), [("list_dir", {"path": "."})]
        )

    def test_truncated_garbage_without_a_function_stays_dropped(self):
        translator = ProtocolTranslator()
        text = translator.feed("<tool_call>\nno function here") + translator.finish()
        self.assertEqual(parse_tool_calls(text), [])

    def test_invalid_call_payloads_do_not_produce_partial_calls(self):
        for raw in ('[]', 'null', '42', 'not-json{{{'):
            with self.subTest(raw=raw):
                translator = ProtocolTranslator()
                text = translator.feed(
                    '<tool_call>' + raw + '</tool_call>'
                ) + translator.finish()
                self.assertEqual(parse_tool_calls(text), [])

    def test_tool_calls_reach_the_agent_loop(self):
        class Stats(SimpleNamespace):
            pass

        turns = [
            ('<tool_call>{"name":"list_dir","arguments":{"path":"."}}'
             "</tool_call>",
             Stats(finish="stop", host_free_gb=None, swap_gb=None)),
            ("done", Stats(finish="stop", host_free_gb=None, swap_gb=None)),
        ]

        class Loop:
            thinking_enabled = False
            pending = []

            def __init__(self):
                self.turns = list(turns)
                self.tool_results = []

            def generate(self, max_tokens, out, **_kwargs):
                raw, stats = self.turns.pop(0)
                translator = ProtocolTranslator()
                text = translator.feed(raw) + translator.finish()
                return text, stats

            def check_invariant(self):
                return True

            def append_tool_results(self, results):
                self.tool_results.append(results)

        loop = Loop()
        repo = Mock()
        repo.call.return_value = {"entries": []}
        reason = run_agent(loop, repo, lambda _text: None)
        self.assertEqual(reason, "answer")
        repo.call.assert_called_once_with("list_dir", {"path": "."})
        self.assertEqual(len(loop.tool_results), 1)
        self.assertEqual(loop.turns, [])


if __name__ == "__main__":
    unittest.main()
