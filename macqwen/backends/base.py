"""What every backend must provide, and the timing they share."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Protocol


class DecodeTimer:
    """Measure model time, not terminal time.

    The streaming callback runs inside the decode loop, so its cost lands in
    any span measured across that loop. The terminal fade and the writes
    behind it vary per word, which made the reported rate wander for reasons
    that had nothing to do with the model. Wrap each callback in `emitting`
    and the spans come back clean.
    """

    def __init__(self, clock=None):
        self._clock = clock or time.perf_counter
        self._began = self._clock()
        self.emitted = 0.0

    @contextmanager
    def emitting(self):
        began = self._clock()
        try:
            yield
        finally:
            self.emitted += self._clock() - began

    def mark(self):
        """A point to measure from later, carrying the emit total with it."""
        return self._clock(), self.emitted

    def since(self, mark) -> float:
        began, emitted = mark
        return max(0.0, self._clock() - began - (self.emitted - emitted))

    def elapsed(self) -> float:
        return self.since((self._began, 0.0))


class Backend(Protocol):
    tape: list[int]
    pending: list[int]
    thinking_enabled: bool

    def open_conversation(self, system, task, tools=None, **options) -> int: ...
    def append_user(self, text: str, enable_thinking: bool = True) -> int: ...
    def append_text(self, text: str) -> int: ...
    def append_tool_results(self, results, enable_thinking: bool = True) -> int: ...
    def generate(
        self,
        max_tokens: int,
        out: Callable[[str], None] | None = None,
        on_prefilled: Callable[[], None] | None = None,
        on_prefill_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[str, Any]: ...
    def check_invariant(self) -> bool: ...
    def reset(self) -> None: ...
    def save_session(self, name: str) -> str: ...
    def load_session(self, name: str) -> str: ...
    def list_sessions(self) -> str: ...
    def delete_session(self, name: str) -> str: ...
    def configure(self, argument: str) -> str: ...
