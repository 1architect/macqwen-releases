"""Adapt K2-Horizon output to the shared chat protocol."""
from __future__ import annotations

import json
import re


TOOL_START = "<ifm|tool_calls>"
TOOL_END = "</ifm|tool_calls>"
CALL = re.compile(r"<ifm\|tool_call>\s*(.*?)\s*</ifm\|tool_call>", re.S)
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MARKERS = {
    "<ifm|think>": "<think>",
    "<ifm|think_fast>": "<think>",
    "<ifm|think_faster>": "<think>",
    "</ifm|think>": "</think>",
    "</ifm|think_fast>": "</think>",
    "</ifm|think_faster>": "</think>",
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


def _tool_value(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _render_calls(block: str) -> str:
    calls = []
    for raw in CALL.findall(block):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            continue
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(name, str) or not NAME.fullmatch(name):
            continue
        if not isinstance(arguments, dict) or not all(
            isinstance(key, str) and NAME.fullmatch(key) for key in arguments
        ):
            continue
        calls.append((name, arguments))

    rendered = []
    for index, (name, arguments) in enumerate(calls):
        if index:
            rendered.append("<tool_call>")
        rendered.append(f"<function={name}>")
        for key, value in arguments.items():
            rendered.append(
                f"<parameter={key}>\n{_tool_value(value)}\n</parameter>"
            )
        rendered.append("</function></tool_call>")
    return "".join(rendered) if rendered else "</tool_call>"


class ProtocolTranslator:
    """Stream normal text while buffering only K2 tool-call payloads."""

    def __init__(self):
        self.pending = ""
        self.in_tool = False

    def feed(self, piece: str) -> str:
        self.pending += piece
        output = []
        while self.pending:
            if self.in_tool:
                end = self.pending.find(TOOL_END)
                if end < 0:
                    break
                output.append(_render_calls(self.pending[:end]))
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

    def finish(self) -> str:
        if self.in_tool:
            self.pending = ""
            return ""
        tail = self.pending
        self.pending = ""
        return tail
