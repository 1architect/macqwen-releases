"""Shared test-provider types and metric labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


CommandScript = Callable[[Any, Path], list[str]]
EnvironmentScript = Callable[[Any], dict[str, str]]
InterpretScript = Callable[[int, str, list[dict]], str | None]
LiveParser = Callable[[str], dict[str, Any] | None]


@dataclass(frozen=True)
class TestSpec:
    id: str
    title: str
    category: str
    explanation: str
    why: str
    script: CommandScript | None = None
    metrics: tuple[str, ...] = ()
    controls: dict[str, str] = field(default_factory=dict)
    source: str = ""
    status: str = "runnable"
    promotion: bool = False
    environment: EnvironmentScript | None = None
    interpret: InterpretScript | None = None
    live_parser: LiveParser | None = None
    canonical: bool = True

    @property
    def runnable(self) -> bool:
        return self.status == "runnable" and callable(self.script)


COMMON_METRICS = (
    "generation and tail rate",
    "physical MB/token",
    "active memory",
    "token digest",
    "paired effect and resolution band",
)
IO_METRICS = COMMON_METRICS + (
    "submission-to-worker-start delay",
    "positioned-read wall time",
    "total I/O wait",
)
