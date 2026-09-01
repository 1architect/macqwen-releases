"""Local OpenAI and Anthropic compatible HTTP server for MACQWEN."""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
import json
import re
import time
import uuid
from urllib.parse import urlsplit

from macqwen.text import CompletedTextBuffer, ThinkingStreamFilter
from macqwen.profiles import system_prompt


MAX_BODY = 16 * 1024 * 1024
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
XML_FUNCTION = re.compile(
    r"\s*<function=([^>\s]+)>(.*?)(?:</function>|</>)?\s*$", re.DOTALL
)
XML_PARAMETER = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL
)


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Completion:
    text: str
    tool_calls: list[dict]
    stats: object


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "input_text", "output_text"):
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _openai_tools(tools) -> list[dict]:
    result = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            result.append(tool)
            continue
        name = tool.get("name")
        if name:
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            })
    return result


def _normalize_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError(f"unsupported message role: {role}")
        item = {"role": "system" if role == "developer" else role,
                "content": _text_content(message.get("content", ""))}
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            item["tool_call_id"] = message["tool_call_id"]
        normalized.append(item)
    return normalized


def _anthropic_messages(payload: dict) -> list[dict]:
    messages = []
    system = _text_content(payload.get("system", ""))
    if system:
        messages.append({"role": "system", "content": system})
    for source in payload.get("messages", []):
        role = source.get("role")
        content = source.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text = []
        calls = []
        for block in content if isinstance(content, list) else []:
            kind = block.get("type")
            if kind == "text":
                text.append(str(block.get("text", "")))
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif kind == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _text_content(block.get("content", "")),
                })
        if text or calls:
            item = {"role": role, "content": "".join(text)}
            if calls:
                item["tool_calls"] = calls
            messages.append(item)
    return _normalize_messages(messages)


def _responses_messages(payload: dict) -> list[dict]:
    messages = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})
    source = payload.get("input", "")
    if isinstance(source, str):
        messages.append({"role": "user", "content": source})
    elif isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                continue
            kind = item.get("type", "message")
            if kind == "message":
                messages.append({
                    "role": item.get("role", "user"),
                    "content": _text_content(item.get("content", "")),
                })
            elif kind == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": str(item.get("output", "")),
                })
            elif kind == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
    return _normalize_messages(messages)


def _coerce_argument(value: str, kind: str | None):
    value = value.strip()
    try:
        if kind == "integer":
            return int(value)
        if kind == "number":
            return float(value)
        if kind == "boolean":
            return value.lower() in ("true", "1", "yes")
        if kind in ("object", "array"):
            return json.loads(value)
    except (TypeError, ValueError):
        pass
    return value


def _parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    calls = []
    schemas = {}
    for tool in _openai_tools(tools):
        function = tool["function"]
        properties = function.get("parameters", {}).get("properties", {})
        schemas[function["name"]] = {
            name: value.get("type") for name, value in properties.items()
        }
    blocks = [(match.group(1), match) for match in TOOL_CALL.finditer(text)]
    fallback = False
    if not blocks and "<tool_call>" in text:
        tail = text.rsplit("<tool_call>", 1)[1]
        if "</function>" in tail:
            blocks = [(tail, None)]
            fallback = True
    for block, _match in blocks:
        try:
            value = json.loads(block)
        except (TypeError, ValueError):
            function = XML_FUNCTION.match(block)
            if not function:
                continue
            name, body = function.groups()
            schema = schemas.get(name, {})
            arguments = {
                key: _coerce_argument(raw, schema.get(key))
                for key, raw in XML_PARAMETER.findall(body)
            }
            calls.append({"id": f"call_{uuid.uuid4().hex}", "name": name,
                          "arguments": json.dumps(arguments)})
        else:
            name = value.get("name")
            if not name:
                continue
            arguments = value.get("arguments", value.get("parameters", {}))
            calls.append({
                "id": value.get("id", f"call_{uuid.uuid4().hex}"),
                "name": name,
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            })
    clean = text.rsplit("<tool_call>", 1)[0] if fallback else TOOL_CALL.sub("", text)
    return clean.strip(), calls


def _model_tokenizer(backend):
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        tokenizer = backend.engine.tokenizer
    return tokenizer


class ModelService:
    """Own one loaded model and serve requests from one warm cache."""

    def __init__(self, session):
        self.session = session
        self.model = f"macqwen-{session.preferences['model']}"
        self.reused = 0
        self.rebuilt = 0

    def _adopt(self, rendered: str) -> bool:
        """Feed the prompt, keeping the cache when it holds a prefix.

        A client resends the whole conversation on every call. When the new
        prompt extends what the cache already holds, only the new tokens need
        a prefill. When it diverges the cache has to go: the recurrent layers
        carry state that cannot be rewound.
        """
        session = self.session
        backend = session.backend
        encode = getattr(backend, "encode", None)
        common_prefix = getattr(backend, "common_prefix", None)
        if encode is None or common_prefix is None:
            session.reset()
            backend.append_text(rendered)
            self.rebuilt += 1
            return False
        ids = encode(rendered)
        if not backend.pending and backend.tape:
            shared = common_prefix(ids)
            if shared == len(backend.tape) and shared < len(ids):
                backend.append_tokens(ids[shared:])
                self.reused += 1
                return True
        session.reset()
        backend.append_tokens(ids)
        self.rebuilt += 1
        return False

    def complete(self, messages, tools, max_tokens, on_text=None) -> Completion:
        session = self.session
        backend = session.backend
        try:
            normalized = _normalize_messages(messages)
            if not any(message["role"] == "system" for message in normalized):
                normalized.insert(0, {
                    "role": "system",
                    "content": session.preferences["system_prompt"] or system_prompt(
                        "plain", session.preferences["workspace"]
                    ),
                })
            tokenizer = _model_tokenizer(backend)
            rendered = tokenizer.apply_chat_template(
                normalized,
                tools=_openai_tools(tools) or None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=session.preferences["thinking_enabled"],
                reasoning_effort=session.preferences["effort"],
            )
            self._adopt(rendered)
            thinking = ThinkingStreamFilter(
                session.preferences["thinking_enabled"], False
            )
            words = CompletedTextBuffer()
            hold = bool(tools)

            def streamed(piece: str) -> None:
                visible = thinking.feed(piece)
                for complete in words.feed(visible):
                    if on_text is not None:
                        on_text(None if hold else complete)

            text, stats = backend.generate(
                max_tokens=max_tokens,
                out=streamed if on_text is not None else None,
            )
            if on_text is not None:
                tail = thinking.finish()
                for complete in words.feed(tail) + words.finish():
                    on_text(None if hold else complete)

            final_filter = ThinkingStreamFilter(
                session.preferences["thinking_enabled"], False
            )
            visible = final_filter.feed(text) + final_filter.finish()
            visible, calls = _parse_tool_calls(visible, tools)
            if hold and on_text is not None and visible:
                final_words = CompletedTextBuffer()
                for complete in final_words.feed(visible) + final_words.finish():
                    on_text(complete)
            return Completion(visible, calls, stats)
        except BaseException:
            session.reset()
            raise


class MacqwenHTTPServer(HTTPServer):
    def __init__(self, address, service, api_key=None, allowed_origins=()):
        super().__init__(address, MacqwenHandler)
        self.service = service
        self.api_key = api_key
        self.allowed_origins = tuple(allowed_origins)


class MacqwenHandler(BaseHTTPRequestHandler):
    server: MacqwenHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        # constant time, so a wrong key cannot be discovered byte by byte
        offered = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if offered.startswith(prefix) and hmac.compare_digest(
                offered[len(prefix):], key):
            return True
        return hmac.compare_digest(self.headers.get("x-api-key", "") or "", key)

    def _allowed_origin(self) -> str | None:
        """Which Origin, if any, may read this reply.

        A browser page on any site can reach a server bound to localhost.
        Answering every Origin with `*` would let any site you visit use this
        model and read its output, so an Origin is echoed back only when it
        was allowed explicitly with --allow-origin. Requests without an
        Origin header, which is every non-browser client, are unaffected.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return None
        allowed = self.server.allowed_origins
        if "*" in allowed or origin in allowed:
            return origin
        return None

    def _reject_foreign_origin(self) -> bool:
        """Refuse a browser request from an Origin that was not allowed."""
        if self.headers.get("Origin") and self._allowed_origin() is None:
            self._error(
                "this origin is not allowed; start the server with "
                "--allow-origin to permit browser clients",
                403,
            )
            return True
        return False

    def _headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        origin = self._allowed_origin()
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self._headers(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=400):
        self._json({"error": {"message": str(message), "type": "invalid_request_error"}}, status)

    def _payload(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError("invalid Content-Length") from exc
        if length < 1 or length > MAX_BODY:
            raise RequestError("request body size is invalid", 413)
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RequestError("request body must contain JSON") from exc
        if not isinstance(value, dict):
            raise RequestError("request body must be an object")
        return value

    def _sse_start(self):
        self._headers(content_type="text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, value, event=None):
        prefix = f"event: {event}\n" if event else ""
        self.wfile.write((prefix + "data: " + json.dumps(value) + "\n\n").encode())
        self.wfile.flush()

    def _heartbeat(self):
        self.wfile.write(b": generating\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        self._headers(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self._reject_foreign_origin():
            return
        if not self._authorized():
            self._error("invalid API key", 401)
            return
        path = urlsplit(self.path).path
        if path == "/health":
            self._json({"status": "ok", "model": self.server.service.model})
        elif path == "/v1/models":
            model = self.server.service.model
            self._json({"object": "list", "data": [{
                "id": model,
                "object": "model",
                "owned_by": "macqwen",
            }], "models": [{
                "slug": model,
                "display_name": "MACQWEN Flash-Next",
                "description": "Local low-memory Qwen runtime",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Low reasoning effort"},
                    {"effort": "medium", "description": "Medium reasoning effort"},
                    {"effort": "high", "description": "High reasoning effort"},
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "minimal_client_version": [0, 0, 0],
                "supported_in_api": True,
                "priority": 1,
                "upgrade": None,
                "base_instructions": "",
                "support_verbosity": False,
                "default_verbosity": None,
                "apply_patch_tool_type": None,
                "truncation_policy": {"mode": "bytes", "limit": 100000},
                "supports_parallel_tool_calls": False,
                "supports_image_detail_original": False,
                "context_window": 32768,
                "experimental_supported_tools": [],
            }]})
        else:
            self._error("route not found", 404)

    def do_POST(self):
        if self._reject_foreign_origin():
            return
        if not self._authorized():
            self._error("invalid API key", 401)
            return
        try:
            payload = self._payload()
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self._chat(payload)
            elif path == "/v1/messages":
                self._anthropic(payload)
            elif path == "/v1/responses":
                self._responses(payload)
            else:
                self._error("route not found", 404)
        except RequestError as exc:
            self._error(exc, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._error(str(exc), 500)

    @staticmethod
    def _limit(payload, default=4096):
        value = payload.get(
            "max_output_tokens",
            payload.get("max_completion_tokens", payload.get("max_tokens", default)),
        )
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RequestError("max_tokens must be a positive integer")
        return min(value, 32768)

    def _chat(self, payload):
        service = self.server.service
        messages = _normalize_messages(payload.get("messages"))
        tools = payload.get("tools") or []
        ident = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if payload.get("stream"):
            self._sse_start()

            def chunk(delta, finish=None):
                self._sse({
                    "id": ident, "object": "chat.completion.chunk", "created": created,
                    "model": service.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                })

            chunk({"role": "assistant", "content": ""})

            def on_text(text):
                self._heartbeat() if text is None else chunk({"content": text})

            result = service.complete(messages, tools, self._limit(payload), on_text)
            for index, call in enumerate(result.tool_calls):
                chunk({"tool_calls": [{
                    "index": index, "id": call["id"], "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }]})
            chunk({}, "tool_calls" if result.tool_calls else result.stats.finish)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        result = service.complete(messages, tools, self._limit(payload))
        message = {"role": "assistant", "content": result.text or None}
        if result.tool_calls:
            message["tool_calls"] = [{
                "id": call["id"], "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            } for call in result.tool_calls]
        self._json({
            "id": ident, "object": "chat.completion", "created": created,
            "model": service.model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if result.tool_calls else result.stats.finish}],
            "usage": {"prompt_tokens": result.stats.prompt_tokens,
                      "completion_tokens": result.stats.tokens,
                      "total_tokens": result.stats.prompt_tokens + result.stats.tokens},
        })

    def _anthropic(self, payload):
        service = self.server.service
        messages = _anthropic_messages(payload)
        tools = payload.get("tools") or []
        ident = f"msg_{uuid.uuid4().hex}"
        if payload.get("stream"):
            self._sse_start()
            self._sse({"type": "message_start", "message": {
                "id": ident, "type": "message", "role": "assistant",
                "model": service.model, "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
            }}, "message_start")
            text_open = [False]

            def on_text(text):
                if text is None:
                    self._heartbeat()
                    return
                if not text_open[0]:
                    self._sse({"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}},
                              "content_block_start")
                    text_open[0] = True
                self._sse({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": text}},
                          "content_block_delta")

            result = service.complete(messages, tools, self._limit(payload), on_text)
            index = 0
            if text_open[0]:
                self._sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
                index = 1
            for call in result.tool_calls:
                self._sse({"type": "content_block_start", "index": index,
                           "content_block": {"type": "tool_use", "id": call["id"],
                                             "name": call["name"], "input": {}}},
                          "content_block_start")
                self._sse({"type": "content_block_delta", "index": index,
                           "delta": {"type": "input_json_delta",
                                     "partial_json": call["arguments"]}},
                          "content_block_delta")
                self._sse({"type": "content_block_stop", "index": index},
                          "content_block_stop")
                index += 1
            stop = "tool_use" if result.tool_calls else "end_turn"
            self._sse({"type": "message_delta", "delta": {"stop_reason": stop,
                       "stop_sequence": None}, "usage": {"output_tokens": result.stats.tokens}},
                      "message_delta")
            self._sse({"type": "message_stop"}, "message_stop")
            return
        result = service.complete(messages, tools, self._limit(payload))
        content = []
        if result.text:
            content.append({"type": "text", "text": result.text})
        for call in result.tool_calls:
            try:
                arguments = json.loads(call["arguments"])
            except ValueError:
                arguments = {"raw": call["arguments"]}
            content.append({"type": "tool_use", "id": call["id"],
                            "name": call["name"], "input": arguments})
        self._json({
            "id": ident, "type": "message", "role": "assistant",
            "model": service.model, "content": content,
            "stop_reason": "tool_use" if result.tool_calls else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": result.stats.prompt_tokens,
                      "output_tokens": result.stats.tokens},
        })

    def _responses(self, payload):
        service = self.server.service
        messages = _responses_messages(payload)
        tools = payload.get("tools") or []
        ident = f"resp_{uuid.uuid4().hex}"
        if payload.get("stream"):
            self._sse_start()
            base = {"id": ident, "object": "response", "created_at": int(time.time()),
                    "status": "in_progress", "error": None,
                    "incomplete_details": None, "model": service.model, "output": []}
            sequence = [0]

            def event(kind, **values):
                sequence[0] += 1
                self._sse({"type": kind, "sequence_number": sequence[0], **values}, kind)

            event("response.created", response=base)
            event("response.in_progress", response=base)
            item_id = f"msg_{uuid.uuid4().hex}"
            opened = [False]

            def on_text(text):
                if text is None:
                    self._heartbeat()
                    return
                if not opened[0]:
                    item = {"id": item_id, "type": "message", "status": "in_progress",
                            "role": "assistant", "content": []}
                    event("response.output_item.added", output_index=0, item=item)
                    event("response.content_part.added", item_id=item_id,
                          output_index=0, content_index=0,
                          part={"type": "output_text", "text": "", "annotations": []})
                    opened[0] = True
                event("response.output_text.delta", item_id=item_id,
                      output_index=0, content_index=0, delta=text)

            result = service.complete(messages, tools, self._limit(payload), on_text)
            output = []
            if result.text:
                message = {"id": item_id, "type": "message", "status": "completed",
                           "role": "assistant", "content": [{"type": "output_text",
                           "text": result.text, "annotations": []}]}
                output.append(message)
                event("response.output_text.done", item_id=item_id, output_index=0,
                      content_index=0, text=result.text)
                event("response.content_part.done", item_id=item_id, output_index=0,
                      content_index=0, part=message["content"][0])
                event("response.output_item.done", output_index=0, item=message)
            for index, call in enumerate(result.tool_calls, start=len(output)):
                item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": call["id"], "name": call["name"],
                        "arguments": call["arguments"], "status": "completed"}
                output.append(item)
                pending = dict(item, arguments="", status="in_progress")
                event("response.output_item.added", output_index=index, item=pending)
                event("response.function_call_arguments.delta", item_id=item["id"],
                      output_index=index, delta=call["arguments"])
                event("response.function_call_arguments.done", item_id=item["id"],
                      output_index=index, arguments=call["arguments"])
                event("response.output_item.done", output_index=index, item=item)
            completed = dict(base, status="completed", output=output,
                             usage={"input_tokens": result.stats.prompt_tokens,
                                    "output_tokens": result.stats.tokens,
                                    "total_tokens": result.stats.prompt_tokens + result.stats.tokens})
            event("response.completed", response=completed)
            return
        result = service.complete(messages, tools, self._limit(payload))
        output = []
        if result.text:
            output.append({"id": f"msg_{uuid.uuid4().hex}", "type": "message",
                           "status": "completed", "role": "assistant",
                           "content": [{"type": "output_text", "text": result.text,
                                        "annotations": []}]})
        for call in result.tool_calls:
            output.append({"type": "function_call", "id": f"fc_{uuid.uuid4().hex}",
                           "call_id": call["id"], "name": call["name"],
                           "arguments": call["arguments"], "status": "completed"})
        self._json({
            "id": ident, "object": "response", "created_at": int(time.time()),
            "status": "completed", "model": service.model, "output": output,
            "usage": {"input_tokens": result.stats.prompt_tokens,
                      "output_tokens": result.stats.tokens,
                      "total_tokens": result.stats.prompt_tokens + result.stats.tokens},
        })


def serve(session, host: str, port: int, api_key: str | None = None,
          allowed_origins=()) -> None:
    """Reset the model when the server starts and when it stops."""
    session.reset()
    server = MacqwenHTTPServer(
        (host, port), ModelService(session), api_key, allowed_origins)
    print(f"server ready at http://{host}:{server.server_port}")
    if allowed_origins:
        print(f"browser origins allowed: {', '.join(allowed_origins)}")
    print("OpenAI: /v1/chat/completions and /v1/responses")
    print("Anthropic: /v1/messages")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped")
    finally:
        server.server_close()
        session.reset()
