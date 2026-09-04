"""Opt-in timing for MLX producer and consumer boundaries.

The profiler is disabled by default.  An enabled sample measures the host
time needed to issue one operation, then the host time needed to complete its
result.  The completion call is deliberate.  It exposes where queued GPU work
is paid, but changes the scheduling topology for that diagnostic run.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable


PROFILE_ENV = "FLASHNEXT_PROFILE_BOUNDARIES"
SELECTED_PROFILE_ENV = "FLASHNEXT_PROFILE_BOUNDARY"
BOUNDARY_LABELS = (
    "gate_qmv",
    "up_qmv",
    "swiglu",
    "down_qmv",
    "fused_down",
)


@dataclass(frozen=True)
class BoundarySample:
    """One issue and completion measurement for a named graph boundary."""

    label: str
    issue_ms: float
    completion_ms: float

    @property
    def total_ms(self) -> float:
        return self.issue_ms + self.completion_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "issue_ms": self.issue_ms,
            "completion_ms": self.completion_ms,
            "total_ms": self.total_ms,
        }


def profile_enabled(value: bool | None = None) -> bool:
    """Resolve the opt-in flag without reading environment state per sample."""
    if value is not None:
        return bool(value)
    return (
        os.environ.get(PROFILE_ENV) == "1"
        or bool(os.environ.get(SELECTED_PROFILE_ENV))
    )


def _validate_label(label: str) -> str:
    if label not in BOUNDARY_LABELS:
        raise ValueError(f"unknown boundary label: {label}")
    return label


def selected_boundaries(
    enabled: bool | None = None,
    boundary: str | None = None,
) -> tuple[str, ...]:
    """Resolve the one-boundary diagnostic arm, or all boundaries explicitly."""
    if boundary is None and enabled is None:
        boundary = os.environ.get(SELECTED_PROFILE_ENV)
        if boundary:
            _validate_label(boundary)
            return (boundary,)
        if os.environ.get(PROFILE_ENV) == "1":
            return BOUNDARY_LABELS
        return ()
    if enabled is False:
        return ()
    if boundary is not None:
        _validate_label(boundary)
        return (boundary,)
    return BOUNDARY_LABELS if enabled else ()


class BoundaryProfiler:
    """Collect bounded issue/completion samples for one executor."""

    def __init__(
        self,
        enabled: bool | None = None,
        *,
        boundary: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.selected = selected_boundaries(enabled, boundary)
        self.enabled = bool(self.selected)
        self._clock = clock or time.perf_counter
        self._samples: list[BoundarySample] = []

    @property
    def samples(self) -> tuple[BoundarySample, ...]:
        return tuple(self._samples)

    def selected_for(self, label: str) -> bool:
        """Return whether this diagnostic arm forces completion for ``label``."""
        return _validate_label(label) in self.selected

    def reset(self) -> None:
        self._samples.clear()

    def record(self, label: str, issue_ms: float, completion_ms: float) -> None:
        _validate_label(label)
        if not self.enabled:
            return
        if label not in self.selected:
            return
        self._samples.append(
            BoundarySample(label, float(issue_ms), float(completion_ms))
        )

    def measure(
        self,
        label: str,
        operation: Callable[[], Any],
        complete: Callable[[Any], None],
    ) -> Any:
        """Run and optionally complete one operation, preserving its result."""
        _validate_label(label)
        if label not in self.selected:
            return operation()
        began = self._clock()
        result = operation()
        issued = self._clock()
        complete_began = issued
        complete(result)
        completed = self._clock()
        self.record(
            label,
            (issued - began) * 1000.0,
            (completed - complete_began) * 1000.0,
        )
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-compatible events and per-label totals."""
        totals: dict[str, dict[str, Any]] = {
            label: {
                "count": 0,
                "issue_ms": 0.0,
                "completion_ms": 0.0,
                "total_ms": 0.0,
            }
            for label in BOUNDARY_LABELS
        }
        for sample in self._samples:
            total = totals[sample.label]
            total["count"] += 1
            total["issue_ms"] += sample.issue_ms
            total["completion_ms"] += sample.completion_ms
            total["total_ms"] += sample.total_ms
        return {
            "enabled": self.enabled,
            "selected": list(self.selected),
            "events": [sample.as_dict() for sample in self._samples],
            "totals": totals,
        }


__all__ = [
    "BOUNDARY_LABELS",
    "BoundaryProfiler",
    "BoundarySample",
    "PROFILE_ENV",
    "profile_enabled",
    "selected_boundaries",
    "SELECTED_PROFILE_ENV",
]
