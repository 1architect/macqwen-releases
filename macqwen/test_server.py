from __future__ import annotations

from types import SimpleNamespace
import unittest

from macqwen import preferences
from macqwen.server import ModelService, _parse_tool_calls


class FakeTokenizer:
    def __init__(self):
        self.rendered = "rendered prompt"

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return self.rendered


class FakeBackend:
    def __init__(self, pieces):
        self.tokenizer = FakeTokenizer()
        self.pieces = pieces
        self.pending = []
        self.tape = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.pending = []
        self.tape = []

    def encode(self, text):
        return [ord(character) for character in text]

    def common_prefix(self, ids):
        limit = min(len(ids), len(self.tape))
        index = 0
        while index < limit and self.tape[index] == ids[index]:
            index += 1
        return index

    def append_text(self, text):
        self.pending.extend(self.encode(text))

    def append_tokens(self, ids):
        self.pending.extend(int(token) for token in ids)

    def generate(self, max_tokens, out=None):
        text = "".join(self.pieces)
        for piece in self.pieces:
            if out:
                out(piece)
        self.tape.extend(self.pending)
        self.pending = []
        # a real client echoes the reply back in the next request
        self.tape.extend(self.encode(text))
        return text, SimpleNamespace(
            finish="stop", tokens=min(max_tokens, 3), prompt_tokens=4
        )


class FakeSession:
    def __init__(self, pieces):
        self.backend = FakeBackend(pieces)
        self.preferences = dict(preferences.DEFAULTS, model="flashnext")
        self.opened = True

    def reset(self):
        self.backend.reset()
        self.opened = False


class ModelServiceTests(unittest.TestCase):
    def test_the_first_request_builds_the_cache_once(self):
        session = FakeSession(["hello"])
        service = ModelService(session)
        result = service.complete([{"role": "user", "content": "test"}], [], 10)
        self.assertEqual(result.text, "hello")
        self.assertEqual(session.backend.reset_count, 1)
        self.assertEqual((service.reused, service.rebuilt), (0, 1))

    def test_a_growing_conversation_keeps_the_cache(self):
        session = FakeSession(["hello"])
        service = ModelService(session)
        service.complete([{"role": "user", "content": "one"}], [], 10)
        tape = list(session.backend.tape)
        session.backend.tokenizer.rendered = "rendered prompthelloMORE"
        service.complete([{"role": "user", "content": "two"}], [], 10)
        self.assertEqual((service.reused, service.rebuilt), (1, 1))
        self.assertEqual(session.backend.reset_count, 1)
        self.assertEqual(session.backend.tape[: len(tape)], tape)

    def test_a_diverging_conversation_rebuilds_the_cache(self):
        session = FakeSession(["hello"])
        service = ModelService(session)
        service.complete([{"role": "user", "content": "one"}], [], 10)
        session.backend.tokenizer.rendered = "different prompt"
        service.complete([{"role": "user", "content": "two"}], [], 10)
        self.assertEqual((service.reused, service.rebuilt), (0, 2))
        self.assertEqual(session.backend.reset_count, 2)

    def test_a_failed_generation_drops_the_cache(self):
        session = FakeSession(["hello"])
        service = ModelService(session)

        def explode(max_tokens, out=None):
            raise RuntimeError("generation failed")

        session.backend.generate = explode
        with self.assertRaises(RuntimeError):
            service.complete([{"role": "user", "content": "test"}], [], 10)
        self.assertEqual(session.backend.tape, [])
        self.assertEqual(session.backend.pending, [])

    def test_stream_releases_only_complete_words(self):
        session = FakeSession(["Hel", "lo ", "wor", "ld"])
        output = []
        ModelService(session).complete(
            [{"role": "user", "content": "test"}], [], 10, output.append
        )
        self.assertEqual(output, ["Hello ", "world"])

    def test_tool_call_parser_returns_openai_arguments(self):
        text, calls = _parse_tool_calls(
            '<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>'
        )
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(calls[0]["arguments"], '{"path": "a"}')

    def test_tool_call_parser_accepts_qwen_xml_for_client_tools(self):
        tool = {"type": "function", "function": {
            "name": "client_tool",
            "parameters": {"type": "object", "properties": {
                "count": {"type": "integer"},
            }},
        }}
        text, calls = _parse_tool_calls(
            "<tool_call><function=client_tool><parameter=count>3</parameter>"
            "</function></tool_call>",
            [tool],
        )
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["name"], "client_tool")
        self.assertEqual(calls[0]["arguments"], '{"count": 3}')


if __name__ == "__main__":
    unittest.main()
