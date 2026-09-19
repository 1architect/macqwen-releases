"""Adapt Bonsai-2 output to the shared chat protocol.

Bonsai-2 derives from Qwen3.8 and uses standard ``<think>`` reasoning tags.
Tool calls use the native XML form (``<function=name>``) the model emits;
that is the only accepted tool-call syntax. JSON object payloads no longer
convert: a second syntax doubles the parser's misparse surface for a form
the model does not natively emit.
"""
from __future__ import annotations

import json
import re

from macqwen.tools import parse_tool_calls


TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
FUNCTION_CLOSE = "</function>"
SHORT_ELEMENT = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>.*?</\1>", re.S)
SHORT_OPEN = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")
# A closer is structural only when the next structural tag or the block
# end follows; anything else means it sits inside a parameter value.
EMBEDDED_PARAM = re.compile(r"</parameter>(?!\s*(?:<parameter=|</function>|\Z))")
EMBEDDED_FUNCTION = re.compile(
    r"</function>(?!\s*(?:</tool_call>|<tool_call>|<function=|\Z))"
)
MARKERS = {
    "<think>": "<think>",
    "</think>": "</think>",
    TOOL_START: None,
}


def _partial_marker(text: str) -> int:
    keep = 0
    for marker in MARKERS:
        maximum = min(len(text), len(marker) - 1)
        for size in range(maximum, keep, -1):
            if text.endswith(marker[:size]):
                keep = size
                break
    return keep


def _escape_xml_text(value: str) -> str:
    """Escape protocol delimiters inside a parsed argument value.

    Only applied to values already extracted by the shared parser, where
    data and structure are unambiguous, so full escaping is safe here.
    The parser unescapes these sequences on extraction.
    """
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _tool_value(value) -> str:
    if isinstance(value, str):
        return _escape_xml_text(value)
    return _escape_xml_text(json.dumps(value, ensure_ascii=False))


def _has_function(block: str) -> bool:
    """Whether the block holds a native function element worth keeping."""
    if "<function=" in block:
        return True
    for match in SHORT_OPEN.finditer(block):
        if match.group(0) != TOOL_START:
            return True
    return False


def _candidate_closes(prefix: str) -> bool:
    """Whether a `</tool_call>` candidate really ends the block.

    Only accept a candidate preceded by a complete function element. A
    value-embedded `</tool_call>` has no `</function>` before it, so it is
    skipped instead of truncating the block. Values containing a literal
    `</function>` before the real close can still misfire; such bytes are
    the model's escaping duty.
    """
    if FUNCTION_CLOSE in prefix:
        return True
    return SHORT_ELEMENT.search(prefix) is not None


def _find_block_end(buffered: str) -> int:
    """Locate the tool-block terminator that actually ends the block.

    Scans `</tool_call>` candidates and accepts the first one preceded by
    a complete function element (see `_candidate_closes`). Anything else
    is a value-embedded delimiter, so buffering continues.
    """
    position = buffered.find(TOOL_END)
    while position >= 0:
        if _candidate_closes(buffered[:position]):
            return position
        position = buffered.find(TOOL_END, position + len(TOOL_END))
    return -1


def _escape_embedded(block: str) -> str:
    """Escape value-embedded closers, keep structural ones.

    The translator works on raw native XML, so data and structure are not
    yet separated. A closer followed by the next structural tag (or the
    block end) is structural; any other follower means the closer sits
    inside a parameter value and must not reach the shared parser raw.
    A value-embedded closer followed by structural-looking text can still
    misfire; such bytes are the model's escaping duty.
    """
    block = EMBEDDED_PARAM.sub("&lt;/parameter&gt;", block)
    block = EMBEDDED_FUNCTION.sub("&lt;/function&gt;", block)
    return block.replace(TOOL_END, "&lt;/tool_call&gt;")


def _render_calls(block: str) -> str:
    """Pass a native XML tool block through, drop anything else.

    Returns "" for blocks without a function element so the caller drops
    them instead of leaking raw payload text into the transcript.
    Schema-known calls re-render through the shared parser, which
    identifies values structurally and escapes embedded delimiters; the
    transcript then parses byte-identically. Anything the shared parser
    rejects (unknown tools, truncated values) passes through raw so a
    hallucinated call stays visible instead of vanishing silently.
    Value-embedded closers are pre-escaped by `_escape_embedded` before
    parsing: the accepted terminator was sliced off before this point, so
    a remaining `</tool_call>` always sits inside a value, and
    `</parameter>` / `</function>` occurrences are classified by what
    follows them.
    """
    if not _has_function(block):
        return ""
    probe = _escape_embedded(block)
    calls = parse_tool_calls(TOOL_START + probe + TOOL_END)
    if not calls:
        return block + TOOL_END
    rendered = []
    for index, (name, arguments) in enumerate(calls):
        if index:
            rendered.append(TOOL_START)
        rendered.append(f"<function={name}>")
        for key, value in arguments.items():
            rendered.append(
                f"<parameter={key}>\n{_tool_value(value)}\n</parameter>"
            )
        rendered.append("</function>")
    rendered.append(TOOL_END)
    return "".join(rendered)


class ProtocolTranslator:
    """Stream normal text while buffering only tool-call payloads."""

    def __init__(self):
        self.pending = ""
        self.in_tool = False

    def feed(self, piece: str) -> str:
        self.pending += piece
        output = []
        while self.pending:
            if self.in_tool:
                end = _find_block_end(self.pending)
                if end < 0:
                    break
                output.append(self._close_tool(self.pending[:end]))
                self.pending = self.pending[end + len(TOOL_END):]
                self.in_tool = False
                continue

            matches = [
                (index, marker)
                for marker in MARKERS
                if (index := self.pending.find(marker)) >= 0
            ]
            if matches:
                index, marker = min(matches, key=lambda item: item[0])
                output.append(self.pending[:index])
                self.pending = self.pending[index + len(marker):]
                replacement = MARKERS[marker]
                if replacement is None:
                    output.append("<tool_call>")
                    self.in_tool = True
                else:
                    output.append(replacement)
                continue

            keep = _partial_marker(self.pending)
            output.append(self.pending[:-keep] if keep else self.pending)
            self.pending = self.pending[-keep:] if keep else ""
            break
        return "".join(output)

    @staticmethod
    def _close_tool(inner: str) -> str:
        # The opening tag was already emitted; append the validated inner
        # markup plus the close, or nothing when the block holds no call.
        return _render_calls(inner)

    def finish(self) -> str:
        if self.in_tool:
            payload = self.pending
            self.pending = ""
            self.in_tool = False
            if FUNCTION_CLOSE in payload or SHORT_ELEMENT.search(payload):
                # Generation stopped after a complete native function but
                # before the outer close. Synthesize the close and let the
                # canonical path validate it instead of dropping the call.
                # Anything shorter (truncated JSON, unterminated values)
                # stays dropped: synthesizing content fabricates arguments.
                return self._close_tool(payload)
            return ""
        tail = self.pending
        self.pending = ""
        return tail
