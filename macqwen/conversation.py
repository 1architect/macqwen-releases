"""Building the transcript a chat model reads, shared by every model.

Both Qwen models use the same turn markers and the same tool-response
framing, so assembling the conversation is not model-specific work. A
backend supplies the tokenizer and the generation; this holds the tape.

Two pieces of state, and the difference matters. `tape` is every token the
model has seen. `pending` is what has been appended but not yet fed through
the cache. A backend consumes `pending` when it generates and moves those
tokens onto `tape`, so `check_invariant` can catch a cache that has drifted
out of step with the transcript.
"""
from __future__ import annotations

from macqwen.text import build_user_encoder

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# The chat template turns `reasoning_effort` into a single sentence of system
# text. `xhigh` asks the model to "validate key assumptions, consider plausible
# alternatives", `low` asks it to be brief, and `medium` is an empty string.
# There is nothing between the two instructions, which is why the two levels
# behave so differently.
#
# Measured on a SketchUp Ruby task: `xhigh` validated its assumptions and
# caught that `String#to_l` raises instead of returning nil, then looped on
# questions where both branches were correct, because "consider plausible
# alternatives" has no stopping rule. `medium` wrote clean prose and skipped
# the `Face#valid?` guard that the task actually needs.
#
# `high` keeps what `xhigh` buys and adds the stopping rule it lacks. It rides
# in the system turn because the template raises on any effort name it does
# not know.
EXTRA_REASONING = {
    "high": (
        "Think carefully through the task and validate key assumptions "
        "before you answer. When two options are both correct, choose one "
        "and move on. Do not re-ask a question you have already answered."
    ),
}
TEMPLATE_EFFORT = {"high": "medium"}


def content_sentinel(index: int) -> str:
    """Placeholder for one untrusted content field inside a template render."""
    return f"\ue000macqwen-content-{index}\ue001"


def split_content_slots(rendered: str, count: int) -> list[str] | None:
    """Split a template render at content sentinels.

    Returns [head, slot_0, ..., slot_{count-1}, tail], or None when a
    sentinel is missing or repeated (the template transformed it): the
    caller falls back to joint encoding.
    """
    parts = []
    position = 0
    for index in range(count):
        sentinel = content_sentinel(index)
        found = rendered.find(sentinel, position)
        if found < 0 or rendered.find(sentinel, found + len(sentinel)) >= 0:
            return None
        parts.append(rendered[position:found])
        position = found + len(sentinel)
    parts.append(rendered[position:])
    return parts


def reasoning_system_text(system: str, reasoning_effort: str) -> tuple[str, str]:
    """Return the system text and the effort name the template accepts."""
    extra = EXTRA_REASONING.get(reasoning_effort, "")
    template_effort = TEMPLATE_EFFORT.get(reasoning_effort, reasoning_effort)
    if not extra:
        return system, template_effort
    return (f"{extra}\n\n{system}" if system else extra), template_effort


class Conversation:
    """The token tape and the turn framing around it."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tape: list[int] = []
        self.pending: list[int] = []
        # a turn that stopped before <|im_end|> has to be closed before the
        # next one opens, or the model reads two turns as one
        self.turn_closed = True
        # (pattern, encode) for untrusted content, built on first use.
        # False when the tokenizer cannot report added tokens: without the
        # marker list there is nothing to split on, so content encodes
        # plainly exactly as before.
        self._user_codec = None

    def _codec(self):
        if self._user_codec is None:
            try:
                self._user_codec = build_user_encoder(self.tokenizer)
            except (AttributeError, TypeError):
                self._user_codec = False
        return self._user_codec

    def _content_ids(self, text: str) -> list[int] | None:
        """Token ids for untrusted content, or None for the joint path.

        Marker-free text returns None so the caller encodes it jointly with
        its framing, keeping tokenization byte-identical. Text carrying
        control markers encodes through the safe encoder, which splits at
        marker boundaries so a paste can never become chat structure.
        """
        codec = self._codec()
        if codec is False or codec[0].search(text) is None:
            return None
        return codec[1](text)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def append_text(self, text: str) -> int:
        ids = self.encode(text)
        self.pending.extend(ids)
        return len(ids)

    def common_prefix(self, ids) -> int:
        """How many leading tokens of `ids` the tape already holds.

        A server client resends the whole conversation each turn. When the
        tape is a prefix of the new prompt, the cache is still valid and only
        the new tokens need a prefill.
        """
        limit = min(len(ids), len(self.tape))
        index = 0
        while index < limit and self.tape[index] == ids[index]:
            index += 1
        return index

    def append_tokens(self, ids) -> int:
        """Append token IDs produced by this exact tokenizer."""
        self.pending.extend(int(token) for token in ids)
        return len(ids)

    def _close(self) -> str:
        return "" if self.turn_closed else IM_END

    def _separator(self) -> str:
        """Newline separating the transcript from the next turn, if needed.

        The template ends every message with `<|im_end|>\\n`, so a turn
        join is always `<|im_end|>\\n<|im_start|>`: exactly one newline.
        When this side appends the close itself (truncated turn), the
        newline is unconditional: trailing newlines in the generated text
        belong to the message content, not to the turn join, and skipping
        it diverges from the template. When the turn already closed, the
        tape ends with the `<|im_end|>` token and the newline is still
        needed; only an empty transcript or a tape that somehow already
        ends with a newline skips it.
        """
        if not self.turn_closed:
            return "\n"
        tail = self.pending or self.tape
        if not tail:
            return ""
        try:
            ending = self.tokenizer.decode(tail[-2:])
        except Exception:
            return "\n"
        return "" if ending.endswith("\n") else "\n"

    @staticmethod
    def _assistant_prefix(enable_thinking: bool = True) -> str:
        if enable_thinking:
            return f"{IM_START}assistant\n<think>\n"
        return f"{IM_START}assistant\n<think>\n\n</think>\n\n"

    def open_conversation(self, system, user, tools=None, enable_thinking=True,
                          reasoning_effort="xhigh") -> int:
        """System turn with the tool contract, first user turn, generation prompt.

        Untrusted content carrying control markers encodes through the safe
        encoder: the template renders with sentinels standing in for the
        system and user fields, then structure encodes jointly around them
        while the genuine template markers stay structural. Marker-free
        prompts take the joint path, keeping tokenization byte-identical.
        """
        if self.tape or self.pending:
            raise RuntimeError("conversation already open")
        system, reasoning_effort = reasoning_system_text(system, reasoning_effort)
        codec = self._codec()
        hostile = (
            codec is not False
            and isinstance(system, str)
            and isinstance(user, str)
            and (
                codec[0].search(system) is not None
                or codec[0].search(user) is not None
            )
        )
        def render(messages):
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True,
                tokenize=False, enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort)
        if not hostile:
            return self.append_text(render([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]))
        parts = split_content_slots(render([
            {"role": "system", "content": content_sentinel(0)},
            {"role": "user", "content": content_sentinel(1)},
        ]), 2)
        if parts is None:
            return self.append_text(render([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]))
        safe = codec[1]
        ids = (
            self.encode(parts[0])
            + safe(system)
            + self.encode(parts[1])
            + safe(user)
            + self.encode(parts[2])
        )
        self.pending.extend(ids)
        return len(ids)

    def append_user(self, text: str, enable_thinking: bool = True) -> int:
        before = f"{self._close()}{self._separator()}{IM_START}user\n"
        after = f"{IM_END}\n{self._assistant_prefix(enable_thinking)}"
        content = self._content_ids(text)
        if content is None:
            return self.append_text(before + text + after)
        ids = self.encode(before) + content + self.encode(after)
        self.pending.extend(ids)
        return len(ids)

    def append_tool_results(self, results, enable_thinking: bool = True) -> int:
        """Return tool output as a user turn, one block per result."""
        head = f"{self._close()}{self._separator()}{IM_START}user"
        tail = f"{IM_END}\n{self._assistant_prefix(enable_thinking)}"
        coded = [self._content_ids(result) for result in results]
        if all(part is None for part in coded):
            body = "".join(
                f"\n<tool_response>\n{result}\n</tool_response>" for result in results
            )
            return self.append_text(f"{head}{body}{tail}")
        ids = self.encode(head)
        for result, content in zip(results, coded):
            ids.extend(self.encode("\n<tool_response>\n"))
            ids.extend(content if content is not None else self.encode(result))
            ids.extend(self.encode("\n</tool_response>"))
        ids.extend(self.encode(tail))
        self.pending.extend(ids)
        return len(ids)

    @property
    def cache_tokens(self) -> int:
        """How many tokens the backend's cache holds. Backends override this."""
        return len(self.tape)

    def check_invariant(self) -> bool:
        """The cache must hold exactly the tape. Drift means a lost turn."""
        return self.cache_tokens == len(self.tape)
