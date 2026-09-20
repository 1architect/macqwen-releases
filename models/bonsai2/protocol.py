"""Adapt Bonsai-2 output to the shared chat protocol.

Bonsai-2 derives from Qwen3.8 and uses standard ``<think>`` reasoning tags.
Tool calls use the native XML form (``<function=name>``) the model emits;
that is the only accepted tool-call syntax. JSON object payloads no longer
convert: a second syntax doubles the parser's misparse surface for a form
the model does not natively emit.
"""
from __future__ import annotations

import re


TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
FUNCTION_CLOSE = "</function>"
# A short-form function element opens the block itself. Anchored: a nested
# element inside a value must never satisfy this, no matter how complete
# it looks.
SHORT_FUNCTION_OPEN = re.compile(r"\A\s*<([A-Za-z_][A-Za-z0-9_]*)\s*>")
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


def _short_function_closed(prefix: str) -> bool:
    """Whether the block is a completed short-form function element."""
    match = SHORT_FUNCTION_OPEN.match(prefix)
    if match is None or match.group(1) in ("tool_call", "function"):
        return False
    return f"</{match.group(1)}>" in prefix[match.end():]


def _has_function(block: str) -> bool:
    """Whether the block holds a native function element worth keeping."""
    if "<function=" in block:
        return True
    return SHORT_FUNCTION_OPEN.match(block) is not None


def _candidate_closes(prefix: str) -> bool:
    """Whether a `</tool_call>` candidate really ends the block.

    Only accept a candidate preceded by a complete OUTER function element:
    a structural `</function>`, or a short-form `<name>...</name>` element
    opening the block itself. A nested element inside a parameter value
    (for example `<b>x</b>`) never qualifies, so a value-embedded
    `</tool_call>` after it keeps buffering instead of truncating the
    block. Values containing a literal `</function>` before the real close
    can still misfire; such bytes are the model's escaping duty.
    """
    if FUNCTION_CLOSE in prefix:
        return True
    return _short_function_closed(prefix)


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
        # A block with no function syntax is malformed.  Its first outer
        # delimiter is still a boundary; discard only that block and let the
        # caller resume ordinary text after it.
        prefix = buffered[:position]
        if "<function=" not in prefix and SHORT_FUNCTION_OPEN.match(prefix) is None:
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
    Transport stays independent of the built-in tool registry: unknown
    tools, extra parameters, and original whitespace pass through
    untouched, so a custom API schema sharing a built-in name loses
    nothing. Value-embedded closers are pre-escaped by `_escape_embedded`
    before this point: the accepted terminator was sliced off, so a
    remaining `</tool_call>` always sits inside a value, and
    `</parameter>` / `</function>` occurrences are classified by what
    follows them. Downstream parsers reverse that escaping on extraction.
    """
    if not _has_function(block):
        return ""
    return TOOL_START + _escape_embedded(block) + TOOL_END


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
        # Hold the complete block until validation so malformed syntax cannot
        # leave an orphan opening tag in the visible transcript.
        return _render_calls(inner)

    def finish(self) -> str:
        if self.in_tool:
            payload = self.pending
            self.pending = ""
            self.in_tool = False
            if FUNCTION_CLOSE in payload or _short_function_closed(payload):
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
