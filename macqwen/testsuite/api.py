"""Runtime-neutral API exposed to model test providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
CommandScript = Callable[[Any, Path], list[str]]


@dataclass
class TestContext:
    runtime: str
    checkpoint: str
    python: str
    tokens: int = 32
    pairs: int = 6
    workers: int = 16
    result_root: Path = ROOT / "docs"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def results_dir(self) -> str:
        return str(self.result_root / self.runtime / "measurements")

    @property
    def canonical_environment(self) -> dict[str, str]:
        if self.runtime != "flashnext":
            return {}
        from models.flashnext.settings.launch import CHAT_ENV

        environment = dict(CHAT_ENV)
        environment["FLASHNEXT_IO_WORKERS"] = str(self.workers)
        environment.update({
            "FLASHNEXT_PROFILE_BOUNDARIES": "0",
            "FLASHNEXT_PROFILE_SCORE_SYNC": "0",
        })
        return environment
