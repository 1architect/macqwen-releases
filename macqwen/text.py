"""Turning user text into tokens without letting it become chat structure."""
from __future__ import annotations

import re


_TEXT_BOUNDARY = re.compile(r".*?(?:\s+|[.!?,;:]+)", re.DOTALL)


def build_user_encoder(tokenizer):
    """Encode user text as content. Control markers stay literal text.

    A pasted transcript often carries <think>, </think> or <|im_end|>. The
    tokenizer turns those strings into control tokens, and the model then
    reads the paste as chat structure and drops most of it from the turn.

    Returns (pattern, encode). `pattern` finds markers so a caller can tell
    whether the slow path is needed; text without markers should be encoded
    whole, which keeps its tokenization byte-identical to the plain call.
    """
    markers = sorted(
        {token.content for token in tokenizer.added_tokens_decoder.values()},
        key=len,
        reverse=True,
    )
    pattern = re.compile(
        "|".join(re.escape(marker) for marker in markers) if markers else r"(?!)"
    )

    def plain(chunk: str) -> list[int]:
        if not chunk:
            return []
        return tokenizer(chunk, add_special_tokens=False)["input_ids"]

    def encode(text: str) -> list[int]:
        ids: list[int] = []
        position = 0
        for match in pattern.finditer(text):
            marker = match.group(0)
            ids.extend(plain(text[position:match.start()]))
            ids.extend(plain(marker[:1]))
            ids.extend(plain(marker[1:]))
            position = match.end()
        ids.extend(plain(text[position:]))
        return ids

    return pattern, encode


def stream_decode(tokenizer, partial: list[int], value: int, limit: int = 8):
    """Decode a streamed token, holding back incomplete characters.

    A character whose UTF-8 bytes span several tokens decodes to U+FFFD when
    each token is decoded alone. U+1F604 is two tokens and printed as two
    filled boxes. Returns the text to emit, or "" while more tokens are
    needed. `partial` is the caller's buffer and is cleared on emit.
    """
    partial.append(value)
    piece = tokenizer.decode(partial)
    if "�" in piece and len(partial) < limit:
        return ""
    partial.clear()
    return piece


class ThinkingStreamFilter:
    """Hide thinking without leaking tags split across streamed pieces."""

    def __init__(self, inside: bool, show: bool):
        self.inside = inside
        self.show = show
        self.pending = ""
        self._trim_leading = True

    @staticmethod
    def _partial_marker(text: str, marker: str) -> int:
        maximum = min(len(text), len(marker) - 1)
        for size in range(maximum, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def feed(self, piece: str) -> str:
        self.pending += piece
        visible = []
        while self.pending:
            if self._trim_leading:
                self.pending = self.pending.lstrip("\r\n")
                if not self.pending:
                    break
                self._trim_leading = False
            marker = "</think>" if self.inside else "<think>"
            index = self.pending.find(marker)
            if index >= 0:
                before = self.pending[:index]
                if self.inside and self.show:
                    visible.append(before.rstrip("\r\n"))
                    visible.append("\n\n")
                elif not self.inside:
                    visible.append(before)
                self.pending = self.pending[index + len(marker):]
                self.inside = not self.inside
                self._trim_leading = True
                continue
            keep = self._partial_marker(self.pending, marker)
            safe = self.pending[:-keep] if keep else self.pending
            remainder = self.pending[-keep:] if keep else ""
            if self.inside and self.show:
                stripped = safe.rstrip("\r\n")
                remainder = safe[len(stripped):] + remainder
                safe = stripped
            if self.show or not self.inside:
                visible.append(safe)
            self.pending = remainder
            break
        return "".join(visible)

    def finish(self) -> str:
        text = self.pending if self.show or not self.inside else ""
        self.pending = ""
        return text


class ToolCallStreamFilter:
    """Hide model protocol markup while preserving text before a tool call."""

    MARKER = "<tool_call>"
    FUNCTION = re.compile(r"<function=([^>\s]+)>")

    def __init__(self):
        self.pending = ""
        self.hidden = False
        self.hidden_text = ""
        self._started_event = False
        self._name_event = None
        self._name_found = False

    def feed(self, piece: str) -> str:
        if self.hidden:
            self._inspect_hidden(piece)
            return ""
        self.pending += piece
        index = self.pending.find(self.MARKER)
        if index >= 0:
            visible = self.pending[:index]
            tail = self.pending[index + len(self.MARKER):]
            self.pending = ""
            self.hidden = True
            self._started_event = True
            self._inspect_hidden(tail)
            return visible
        keep = ThinkingStreamFilter._partial_marker(self.pending, self.MARKER)
        visible = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        return visible

    def _inspect_hidden(self, piece: str) -> None:
        if self._name_found:
            return
        self.hidden_text = (self.hidden_text + piece)[-512:]
        match = self.FUNCTION.search(self.hidden_text)
        if match:
            self._name_found = True
            self._name_event = match.group(1)

    def take_events(self) -> tuple[bool, str | None]:
        """Return and clear pending protocol-start and function-name events."""
        events = self._started_event, self._name_event
        self._started_event = False
        self._name_event = None
        return events

    def finish(self) -> str:
        """Release buffered normal text and discard buffered tool protocol."""
        visible = "" if self.hidden else self.pending
        self.pending = ""
        return visible


class CompletedTextBuffer:
    """Release complete words or phrases and hold an unfinished word."""

    def __init__(self):
        self.pending = ""

    def feed(self, piece: str) -> list[str]:
        self.pending += piece
        complete = []
        position = 0
        for match in _TEXT_BOUNDARY.finditer(self.pending):
            complete.append(match.group(0))
            position = match.end()
        if position:
            self.pending = self.pending[position:]
        return complete

    def finish(self) -> list[str]:
        if not self.pending:
            return []
        final = self.pending
        self.pending = ""
        return [final]
