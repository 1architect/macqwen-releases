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

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class Conversation:
    """The token tape and the turn framing around it."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tape: list[int] = []
        self.pending: list[int] = []
        # a turn that stopped before <|im_end|> has to be closed before the
        # next one opens, or the model reads two turns as one
        self.turn_closed = True

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

    @staticmethod
    def _assistant_prefix(enable_thinking: bool = True) -> str:
        if enable_thinking:
            return f"{IM_START}assistant\n<think>\n"
        return f"{IM_START}assistant\n<think>\n\n</think>\n\n"

    def open_conversation(self, system, user, tools=None, enable_thinking=True,
                          reasoning_effort="xhigh") -> int:
        """System turn with the tool contract, first user turn, generation prompt."""
        if self.tape or self.pending:
            raise RuntimeError("conversation already open")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
        return self.append_text(text)

    def append_user(self, text: str, enable_thinking: bool = True) -> int:
        return self.append_text(
            f"{self._close()}\n{IM_START}user\n{text}{IM_END}\n"
            f"{self._assistant_prefix(enable_thinking)}")

    def append_tool_results(self, results, enable_thinking: bool = True) -> int:
        """Return tool output as a user turn, one block per result."""
        body = "".join(
            f"\n<tool_response>\n{result}\n</tool_response>" for result in results
        )
        return self.append_text(
            f"{self._close()}\n{IM_START}user{body}{IM_END}\n"
            f"{self._assistant_prefix(enable_thinking)}")

    @property
    def cache_tokens(self) -> int:
        """How many tokens the backend's cache holds. Backends override this."""
        return len(self.tape)

    def check_invariant(self) -> bool:
        """The cache must hold exactly the tape. Drift means a lost turn."""
        return self.cache_tokens == len(self.tape)
