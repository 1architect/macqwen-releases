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

    def test_native_calls_stream_without_buffering_text(self):
        translator = ProtocolTranslator()
        first = translator.feed("checking<tool_")
        opened = translator.feed("call>")
        second = translator.feed(
            '\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n'
            '<parameter=start_line>\n2\n</parameter>\n</function>\n</tool_call>'
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
            '<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n'
            '</parameter>\n<parameter=start_line>\n2\n</parameter>\n'
            '</function>\n</tool_call>'
            '<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n'
            '</parameter>\n</function>\n</tool_call>'
        )
        for size in (1, 3, 17, len(raw)):
            with self.subTest(size=size):
                translator = ProtocolTranslator()
                text = ''.join(translator.feed(raw[i:i + size])
                               for i in range(0, len(raw), size)) + translator.finish()
                self.assertEqual(parse_tool_calls(text), [
                    ('read_file', {'path': 'README.md', 'start_line': 2}),
                    ('list_dir', {'path': '.'}),
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

    def test_delimiter_text_inside_arguments_round_trips(self):
        # A coding tool can legitimately write protocol text inside a
        # file-content argument. Value-embedded delimiters are escaped in
        # transit and restored on extraction.
        hostile = "write </tool_call> then </function> then </parameter> end"
        translator = ProtocolTranslator()
        text = translator.feed(
            '<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n'
            '</parameter>\n<parameter=content>\n' + hostile + '\n</parameter>\n'
            '</function>\n</tool_call>'
        ) + translator.finish()
        self.assertEqual(
            parse_tool_calls(text),
            [("write_file", {"path": "a.txt", "content": hostile})],
        )

    def test_split_blocks_with_delimiters_still_parse(self):
        hostile = "x</tool_call>y"
        raw = ('<tool_call>\n<function=write_file>\n<parameter=path>\na\n'
               '</parameter>\n<parameter=content>\n' + hostile + '\n</parameter>\n'
               '</function>\n</tool_call>')
        for size in (1, 7, 31):
            with self.subTest(size=size):
                translator = ProtocolTranslator()
                text = "".join(
                    translator.feed(raw[i:i + size])
                    for i in range(0, len(raw), size)
                ) + translator.finish()
                calls = parse_tool_calls(text)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1]["content"], hostile)

    def test_invalid_call_payloads_do_not_produce_partial_calls(self):
        for raw in ('[]', 'null', '42', 'not-json{{{'):
            with self.subTest(raw=raw):
                translator = ProtocolTranslator()
                text = translator.feed(
                    '<tool_call>' + raw + '</tool_call>'
                ) + translator.finish()
                self.assertEqual(parse_tool_calls(text), [])

    def test_nested_element_does_not_close_the_outer_call(self):
        # A complete short element inside a value (<b>x</b>) must not
        # satisfy the terminator rule: only the outer function completes
        # the block. Split across chunk sizes to cover streaming.
        content = "see <b>x</b> then </tool_call> done"
        raw = (
            "<tool_call>\n<function=write_file>\n<parameter=path>\na\n"
            "</parameter>\n<parameter=content>\n" + content + "\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        for size in (1, 9, len(raw)):
            with self.subTest(size=size):
                translator = ProtocolTranslator()
                text = "".join(
                    translator.feed(raw[i:i + size])
                    for i in range(0, len(raw), size)
                ) + translator.finish()
                calls = parse_tool_calls(text)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1]["content"], content)

    def test_short_form_function_still_closes(self):
        raw = "<tool_call><list_dir><path>.</path></list_dir></tool_call>"
        for size in (1, 7, len(raw)):
            with self.subTest(size=size):
                translator = ProtocolTranslator()
                text = "".join(
                    translator.feed(raw[i:i + size])
                    for i in range(0, len(raw), size)
                ) + translator.finish()
                self.assertEqual(
                    parse_tool_calls(text), [("list_dir", {"path": "."})]
                )

    def test_bare_ampersands_pass_through_unescaped(self):
        # Transport escapes only structural closers. A bare & is payload,
        # so a&b.txt must not become a&amp;b.txt in transit.
        translator = ProtocolTranslator()
        text = translator.feed(
            "<tool_call>\n<function=read_file>\n<parameter=path>\na&b.txt\n"
            "</parameter>\n</function>\n</tool_call>"
        ) + translator.finish()
        self.assertIn("a&b.txt", text)
        self.assertNotIn("a&amp;b.txt", text)
        self.assertEqual(
            parse_tool_calls(text), [("read_file", {"path": "a&b.txt"})]
        )

    def test_json_payloads_are_dropped_as_non_native_syntax(self):
        # Native XML is the only accepted tool-call syntax. A complete JSON
        # object block no longer converts; it yields no calls.
        translator = ProtocolTranslator()
        text = translator.feed(
            '<tool_call>{"name":"list_dir","arguments":{"path":"."}}'
            "</tool_call>"
        ) + translator.finish()
        self.assertEqual(parse_tool_calls(text), [])

    def test_truncated_json_at_eof_stays_dropped(self):
        # EOF recovery never synthesizes content: truncated JSON (even with
        # unterminated strings and braces) is dropped, not parsed.
        for raw in (
            '<tool_call>{"name":"read_file","arguments":{"path":"READ',
            '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}',
            '<tool_call>{"name":"read_file","arguments":',
            '<tool_call>{"name":',
        ):
            with self.subTest(raw=raw):
                translator = ProtocolTranslator()
                text = translator.feed(raw) + translator.finish()
                self.assertEqual(parse_tool_calls(text), [])

    def test_tool_calls_reach_the_agent_loop(self):
        class Stats(SimpleNamespace):
            pass

        turns = [
            ('<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n'
             '</parameter>\n</function>\n</tool_call>',
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
