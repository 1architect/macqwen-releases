"""Adapt Bonsai-2 output to the shared chat protocol.

Bonsai-2 derives from Qwen3.8 and uses standard ``<think>`` reasoning tags.
Tool-call wire format stays provisional until we capture live output; this
translator passes reasoning text through and converts ``<tool_call>`` JSON
blocks to the shared contract.
"""
from __future__ import annotations

import json
import re


TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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


def _tool_value(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _as_call(value) -> tuple | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments", {})
    if not isinstance(name, str) or not NAME.fullmatch(name):
        return None
    if not isinstance(arguments, dict) or not all(
        isinstance(key, str) and NAME.fullmatch(key) for key in arguments
    ):
        return None
    return (name, arguments)


def _render_calls(block: str) -> str:
    calls = []
    try:
        single = _as_call(json.loads(block))
    except (TypeError, ValueError):
        single = None
    if single is not None:
        calls.append(single)
    else:
        for raw in CALL.findall(block):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                continue
            call = _as_call(value)
            if call is not None:
                calls.append(call)

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
    """Stream normal text while buffering only tool-call payloads."""

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
                inner = self.pending[:end]
                rendered = _render_calls(inner)
                if rendered == "</tool_call>" and inner.strip():
                    # Not JSON. The native Qwen XML call already matches the
                    # shared contract, so pass it through instead of eating it.
                    # The opening tag was already emitted; re-emit only the
                    # inner markup plus the close. Eating a valid call hides
                    # the function name from the tool filter and the turn ends
                    # with no visible answer.
                    output.append(inner + TOOL_END)
                else:
                    output.append(rendered)
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
