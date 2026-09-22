from __future__ import annotations

import io
import json
from types import SimpleNamespace
import unittest

from macqwen import preferences
from macqwen.server import (
    MacqwenHandler,
    ModelService,
    RequestError,
    _anthropic_stop,
    _parse_tool_calls,
    capabilities,
    model_card,
)
from macqwen.conversation import content_sentinel


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
        self._replay_needed = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.pending = []
        self.tape = []
        self._replay_needed = False

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

    def test_invalid_replay_state_rebuilds_even_with_a_matching_prefix(self):
        session = FakeSession(["hello"])
        service = ModelService(session)
        service.complete([{"role": "user", "content": "one"}], [], 10)
        session.backend.tokenizer.rendered = "rendered prompthelloMORE"
        session.backend._replay_needed = True

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

    def test_custom_schema_sharing_a_builtin_name_keeps_extra_params(self):
        import json

        tool = {"type": "function", "function": {
            "name": "search",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "depth": {"type": "integer"},
            }},
        }}
        _text, calls = _parse_tool_calls(
            "<tool_call><function=search><parameter=query>hi</parameter>"
            "<parameter=depth>2</parameter></function></tool_call>",
            [tool],
        )
        self.assertEqual(
            json.loads(calls[0]["arguments"]), {"query": "hi", "depth": 2}
        )

    def test_transport_escaping_reverses_exactly_once(self):
        import json

        _text, calls = _parse_tool_calls(
            "<tool_call><function=read_file><parameter=path>a&amp;b.txt"
            "</parameter></function></tool_call>"
        )
        self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "a&b.txt"})

    def test_string_payloads_keep_their_bytes(self):
        import json

        _text, calls = _parse_tool_calls(
            "<tool_call><function=write_file><parameter=path>a.txt"
            "</parameter><parameter=content>  indented\nline2\n\n"
            "</parameter></function></tool_call>"
        )
        self.assertEqual(
            json.loads(calls[0]["arguments"])["content"], "  indented\nline2\n"
        )

    def test_hostile_user_content_stays_content_not_structure(self):
        from types import SimpleNamespace

        class EchoTokenizer(FakeTokenizer):
            added_tokens_decoder = {
                1: SimpleNamespace(content="</think>"),
                2: SimpleNamespace(content="<|im_end|>"),
            }

            def __init__(self):
                super().__init__()
                self.chunks = []

            def __call__(self, text, add_special_tokens=False):
                self.chunks.append(text)
                return {"input_ids": [ord(c) for c in text]}

            def apply_chat_template(self, messages, **options):
                self.messages = messages
                self.options = options
                return (
                    "HEAD:"
                    + "|".join(m.get("content", "") for m in messages)
                    + ":TAIL"
                )

        session = FakeSession(["done"])
        session.backend.tokenizer = EchoTokenizer()
        ModelService(session).complete(
            [{"role": "user", "content": "paste </think> verbatim"}], [], 10
        )
        tokenizer = session.backend.tokenizer
        # Non-empty chunks prove the split path ran, not joint fallback.
        self.assertTrue(tokenizer.chunks)
        for chunk in tokenizer.chunks:
            self.assertNotIn("</think>", chunk)
        tape_text = "".join(chr(c) for c in session.backend.tape)
        self.assertIn("paste </think> verbatim", tape_text)

    def test_tool_history_content_uses_the_safe_encoder(self):
        from types import SimpleNamespace

        class EchoTokenizer(FakeTokenizer):
            added_tokens_decoder = {
                1: SimpleNamespace(content="</think>"),
                2: SimpleNamespace(content="<|im_end|>"),
            }

            def __init__(self):
                super().__init__()
                self.chunks = []

            def __call__(self, text, add_special_tokens=False):
                self.chunks.append(text)
                return {"input_ids": [ord(c) for c in text]}

            def apply_chat_template(self, messages, **options):
                self.messages = messages
                self.options = options
                return "HEAD:" + "|".join(
                    m.get("content", "") for m in messages
                ) + ":TAIL"

        content = "file </think> and <|im_end|> bytes"
        session = FakeSession(["done"])
        session.backend.tokenizer = EchoTokenizer()
        ModelService(session).complete(
            [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function", "function": {
                        "name": "read_file", "arguments": '{"path":"a.txt"}',
                    },
                }]},
                {"role": "tool", "content": content},
                {"role": "tool", "content": content + " again"},
            ],
            [],
            10,
        )
        tokenizer = session.backend.tokenizer
        self.assertTrue(tokenizer.chunks)
        self.assertTrue(all(
            "</think>" not in chunk and "<|im_end|>" not in chunk
            for chunk in tokenizer.chunks
        ))
        tape_text = "".join(chr(c) for c in session.backend.tape)
        self.assertIn(content, tape_text)
        self.assertIn(content + " again", tape_text)

    def test_failed_placeholder_split_fails_closed(self):
        from types import SimpleNamespace

        class BrokenTokenizer(FakeTokenizer):
            added_tokens_decoder = {1: SimpleNamespace(content="</think>")}

            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) for c in text]}

            def apply_chat_template(self, messages, **options):
                sentinel = content_sentinel(0)
                if self.mode == "missing":
                    return "HEAD:missing:TAIL"
                return f"HEAD:{sentinel}{sentinel}:TAIL"

        for mode in ("missing", "repeated"):
            with self.subTest(mode=mode):
                session = FakeSession(["done"])
                session.backend.tokenizer = BrokenTokenizer(mode)
                with self.assertRaisesRegex(
                    RequestError, "safely encode content placeholders"
                ):
                    ModelService(session).complete(
                        [{"role": "tool", "content": "file </think>"}],
                        [],
                        10,
                    )
                self.assertEqual(session.backend.tape, [])
                self.assertEqual(session.backend.pending, [])


class XmlCoercionTests(unittest.TestCase):
    """The server XML parser shares coerce_scalar with macqwen.tools."""

    SCHEMA = {"type": "function", "function": {
        "name": "client_tool",
        "parameters": {"type": "object", "properties": {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "items": {"type": "array"},
            "config": {"type": "object"},
            "label": {"type": "string"},
        }},
    }}

    def _call(self, params):
        body = "".join(
            f"<parameter={key}>{value}</parameter>"
            for key, value in params.items()
        )
        return (f"<tool_call><function=client_tool>{body}</function>"
                "</tool_call>")

    def _args(self, params):
        _text, calls = _parse_tool_calls(self._call(params), [self.SCHEMA])
        return json.loads(calls[0]["arguments"])

    def test_valid_table(self):
        args = self._args({
            "count": "3", "ratio": "3.7", "flag": "true",
            "items": "[1, 2]", "config": '{"a": 1}', "label": "banana",
        })
        self.assertEqual(args, {
            "count": 3, "ratio": 3.7, "flag": True,
            "items": [1, 2], "config": {"a": 1}, "label": "banana",
        })

    def test_false_and_zero_words(self):
        for raw, expected in (("false", False), ("0", False), ("no", False),
                              ("off", False), ("1", True), ("yes", True),
                              ("on", True)):
            with self.subTest(raw=raw):
                self.assertIs(self._args({"flag": raw})["flag"], expected)

    def test_invalid_table_raises_request_error(self):
        cases = [
            ("count", "3.0"), ("count", "3.7"), ("count", "banana"),
            ("count", ""), ("ratio", "banana"), ("ratio", ""),
            ("ratio", "nan"), ("ratio", "inf"), ("flag", "banana"),
            ("flag", ""), ("items", "banana"), ("items", '{"a": 1}'),
            ("config", "[1]"), ("config", "banana"),
        ]
        for key, raw in cases:
            with self.subTest(key=key, raw=raw):
                with self.assertRaises(RequestError):
                    _parse_tool_calls(self._call({key: raw}), [self.SCHEMA])

    def test_error_names_tool_and_parameter(self):
        with self.assertRaisesRegex(
            RequestError, "client_tool.*count.*integer"
        ):
            _parse_tool_calls(self._call({"count": "3.7"}), [self.SCHEMA])

    def test_plain_strings_pass_through(self):
        self.assertEqual(self._args({"label": "banana"})["label"], "banana")
        self.assertEqual(self._args({"label": "3.7"})["label"], "3.7")


def _make_handler(complete):
    service = SimpleNamespace(model="macqwen-test", complete=complete)
    handler = MacqwenHandler.__new__(MacqwenHandler)
    handler.server = SimpleNamespace(service=service, allowed_origins=())
    handler.headers = {}
    handler.wfile = io.BytesIO()
    handler._sse_open = False
    handler.status_calls = []
    handler.error_calls = []
    handler.json_calls = []

    def send_response(status, message=None):
        handler.status_calls.append(status)

    def send_header(*args, **kwargs):
        pass

    def end_headers():
        pass

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    orig_json = MacqwenHandler._json.__get__(handler, MacqwenHandler)
    orig_error = MacqwenHandler._error.__get__(handler, MacqwenHandler)

    def rec_json(value, status=200):
        handler.json_calls.append((value, status))
        return orig_json(value, status)

    def rec_error(message, status=400):
        handler.error_calls.append((str(message), status))
        return orig_error(message, status)

    handler._json = rec_json
    handler._error = rec_error
    return handler


def _stub_result(finish, tool_calls, text="hello"):
    return SimpleNamespace(
        text=text,
        tool_calls=tool_calls,
        stats=SimpleNamespace(finish=finish, tokens=5, prompt_tokens=7),
    )


_STUB_TOOL = {"id": "call_1", "name": "tool_a", "arguments": '{"x": 1}'}


class SseErrorTests(unittest.TestCase):
    def test_chat_stream_error_sends_error_then_done(self):
        def boom(messages, tools, max_tokens, on_text=None):
            raise RuntimeError("boom-chat")

        handler = _make_handler(boom)
        handler._chat({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        out = handler.wfile.getvalue().decode()
        self.assertIn("boom-chat", out)
        self.assertIn('"error"', out)
        self.assertIn("data: [DONE]", out)
        self.assertEqual(handler.status_calls, [200])
        self.assertEqual(handler.error_calls, [])
        self.assertEqual(handler.json_calls, [])

    def test_anthropic_stream_error_sends_error_then_stop(self):
        def boom(messages, tools, max_tokens, on_text=None):
            raise RuntimeError("boom-anthropic")

        handler = _make_handler(boom)
        handler._anthropic({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        out = handler.wfile.getvalue().decode()
        self.assertIn("boom-anthropic", out)
        self.assertIn("event: error", out)
        self.assertIn("event: message_stop", out)
        self.assertEqual(handler.status_calls, [200])
        self.assertEqual(handler.error_calls, [])
        self.assertEqual(handler.json_calls, [])

    def test_responses_stream_error_sends_failed(self):
        def boom(messages, tools, max_tokens, on_text=None):
            raise RuntimeError("boom-responses")

        handler = _make_handler(boom)
        handler._responses({"input": "hi", "stream": True})
        out = handler.wfile.getvalue().decode()
        self.assertIn("boom-responses", out)
        self.assertIn("response.failed", out)
        self.assertNotIn("response.completed", out)
        self.assertEqual(handler.status_calls, [200])
        self.assertEqual(handler.error_calls, [])
        self.assertEqual(handler.json_calls, [])

    def test_limit_validated_before_sse_start(self):
        def ok(messages, tools, max_tokens, on_text=None):
            return _stub_result("stop", [])

        for method, payload in (
            ("_chat", {"messages": [{"role": "user", "content": "hi"}],
                       "stream": True, "max_tokens": 0}),
            ("_anthropic", {"messages": [{"role": "user", "content": "hi"}],
                            "stream": True, "max_tokens": 0}),
            ("_responses", {"input": "hi",
                            "stream": True, "max_tokens": 0}),
        ):
            with self.subTest(method=method):
                handler = _make_handler(ok)
                with self.assertRaises(RequestError):
                    getattr(handler, method)(payload)
                self.assertEqual(handler.status_calls, [])
                self.assertEqual(handler.wfile.getvalue(), b"")


class AnthropicStopTests(unittest.TestCase):
    def test_stop_length_wins_over_tools(self):
        result = _stub_result("length", [_STUB_TOOL])
        self.assertEqual(_anthropic_stop(result), "max_tokens")

    def test_stop_length_without_tools(self):
        result = _stub_result("length", [])
        self.assertEqual(_anthropic_stop(result), "max_tokens")

    def test_stop_tool_use(self):
        result = _stub_result("stop", [_STUB_TOOL])
        self.assertEqual(_anthropic_stop(result), "tool_use")

    def test_stop_end_turn(self):
        result = _stub_result("stop", [])
        self.assertEqual(_anthropic_stop(result), "end_turn")

    def test_stream_stop_reasons(self):
        cases = [
            ("length", True, "max_tokens"),
            ("length", False, "max_tokens"),
            ("stop", True, "tool_use"),
            ("stop", False, "end_turn"),
        ]
        for finish, has_tools, expected in cases:
            with self.subTest(finish=finish, has_tools=has_tools):
                tools = [_STUB_TOOL] if has_tools else []

                def complete(messages, tool_arg, max_tokens, on_text=None,
                             _finish=finish, _tools=tools):
                    return _stub_result(_finish, list(_tools))

                handler = _make_handler(complete)
                handler._anthropic({
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                })
                out = handler.wfile.getvalue().decode()
                self.assertIn(f'"stop_reason": "{expected}"', out)
                self.assertIn("event: message_stop", out)
                self.assertEqual(handler.status_calls, [200])

    def test_nonstream_stop_reasons(self):
        cases = [
            ("length", True, "max_tokens"),
            ("length", False, "max_tokens"),
            ("stop", True, "tool_use"),
            ("stop", False, "end_turn"),
        ]
        for finish, has_tools, expected in cases:
            with self.subTest(finish=finish, has_tools=has_tools):
                tools = [_STUB_TOOL] if has_tools else []

                def complete(messages, tool_arg, max_tokens, on_text=None,
                             _finish=finish, _tools=tools):
                    return _stub_result(_finish, list(_tools))

                handler = _make_handler(complete)
                handler._anthropic({
                    "messages": [{"role": "user", "content": "hi"}],
                })
                self.assertEqual(len(handler.json_calls), 1)
                self.assertEqual(
                    handler.json_calls[0][0]["stop_reason"], expected)


class FakeFlashNextTokenizer(FakeTokenizer):
    model_max_length = 131072


class FakeFlashNextBackend(FakeBackend):
    def __init__(self, pieces):
        super().__init__(pieces)
        self.tokenizer = FakeFlashNextTokenizer()


class FakeBonsaiBackend(FakeBackend):
    context_window = 65536


def _make_get_handler(service, path="/v1/models"):
    handler = MacqwenHandler.__new__(MacqwenHandler)
    handler.server = SimpleNamespace(
        service=service, api_key=None, allowed_origins=())
    handler.headers = {}
    handler.path = path
    handler.wfile = io.BytesIO()
    handler._sse_open = False
    handler.status_calls = []
    handler.json_calls = []

    def send_response(status, message=None):
        handler.status_calls.append(status)

    def send_header(*args, **kwargs):
        pass

    def end_headers():
        pass

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    orig_json = MacqwenHandler._json.__get__(handler, MacqwenHandler)

    def rec_json(value, status=200):
        handler.json_calls.append((value, status))
        return orig_json(value, status)

    handler._json = rec_json
    return handler


def _service_with_backend(slug, backend):
    session = SimpleNamespace(backend=backend)
    return SimpleNamespace(model=slug, session=session,
                           complete=lambda *args, **kwargs: None)


class ModelCardTests(unittest.TestCase):
    def test_flashnext_card_reads_tokenizer_window(self):
        backend = FakeFlashNextBackend(["hi"])
        service = _service_with_backend("macqwen-flashnext", backend)
        caps = capabilities(service)
        self.assertEqual(caps["display_name"], "MACQWEN Flash-Next")
        self.assertEqual(caps["context_window"], 131072)
        card = model_card(service)
        self.assertEqual(card["slug"], "macqwen-flashnext")
        self.assertEqual(card["display_name"], "MACQWEN Flash-Next")
        self.assertEqual(card["context_window"], 131072)

    def test_bonsai_card_reads_backend_window(self):
        backend = FakeBonsaiBackend(["hi"])
        service = _service_with_backend("macqwen-bonsai2", backend)
        caps = capabilities(service)
        self.assertEqual(caps["display_name"], "MACQWEN Bonsai-2")
        self.assertEqual(caps["context_window"], 65536)

    def test_missing_config_falls_back_to_default_window(self):
        service = _service_with_backend("macqwen-flashnext", FakeBackend(["hi"]))
        self.assertEqual(capabilities(service)["context_window"], 32768)

    def test_sentinel_tokenizer_window_falls_back(self):
        backend = FakeBackend(["hi"])
        backend.tokenizer.model_max_length = int(1e30)
        service = _service_with_backend("macqwen-flashnext", backend)
        self.assertEqual(capabilities(service)["context_window"], 32768)

    def test_models_route_renders_from_helper(self):
        backend = FakeBonsaiBackend(["hi"])
        service = _service_with_backend("macqwen-bonsai2", backend)
        handler = _make_get_handler(service)
        handler.do_GET()
        self.assertEqual(len(handler.json_calls), 1)
        body = handler.json_calls[0][0]
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], "macqwen-bonsai2")
        self.assertEqual(body["models"][0], model_card(service))
        self.assertEqual(body["models"][0]["display_name"], "MACQWEN Bonsai-2")
        self.assertEqual(body["models"][0]["context_window"], 65536)

    def test_limit_caps_at_backend_window(self):
        backend = FakeBonsaiBackend(["hi"])
        service = _service_with_backend("macqwen-bonsai2", backend)
        handler = _make_get_handler(service)
        self.assertEqual(handler._limit({"max_tokens": 200000}), 65536)
        self.assertEqual(handler._limit({"max_tokens": 100}), 100)
        with self.assertRaises(RequestError):
            handler._limit({"max_tokens": 0})

    def test_limit_falls_back_without_backend(self):
        handler = _make_handler(
            lambda *args, **kwargs: _stub_result("stop", []))
        self.assertEqual(handler._limit({"max_tokens": 200000}), 32768)


if __name__ == "__main__":
    unittest.main()
