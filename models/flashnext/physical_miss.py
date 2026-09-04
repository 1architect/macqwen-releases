"""Physical-miss evidence and opt-in slab allocation.

The normal slab allocator uses route frequency.  This module adds a separate
diagnostic allocator for evidence collected during a long, constrained run.
It never runs unless ``FLASHNEXT_SLAB_POLICY=physical-miss`` is selected.

The evidence must contain measured physical bytes for a ``(layer, expert)``
pair.  Route counts alone are not accepted as physical-miss evidence.  A
calibration run should use one I/O worker and one expert per read when it
needs attribution.  Concurrent process-wide disk counters cannot identify
which expert caused a physical read.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


PROFILE_VERSION = 1
DEFAULT_PROFILE = "~/.cache/flashnext/physical-misses.json"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TRACE_LOCK = threading.Lock()
_ACTIVE_TRACE: "PhysicalMissTrace | None" = None


def empty_profile(source: str = "") -> dict:
    """Return an empty, versioned evidence document."""
    return {
        "version": PROFILE_VERSION,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokens": 0,
        "layers": {},
    }


class PhysicalMissTrace:
    """Collect per-expert evidence during a constrained calibration run.

    The process-wide disk counter is only attributable when reads are
    serialized.  Use ``FLASHNEXT_IO_WORKERS=1`` and
    ``FLASHNEXT_PREAD_CHUNK=1`` for a calibration trace.  Concurrent traces
    remain useful as a diagnostic, but their per-expert attribution is not
    exact and must not justify a default change.
    """

    def __init__(self, source: str = "calibration"):
        self.profile = empty_profile(source)

    def record(self, name: str, expert: int, physical_bytes: int, requested_bytes: int) -> None:
        match = _LAYER_RE.search(name)
        if match is None:
            return
        add_observation(
            self.profile,
            int(match.group(1)),
            int(expert),
            physical_miss_bytes=max(0, int(physical_bytes)),
            requested_bytes=max(0, int(requested_bytes)),
        )

    def write(self, path: str | Path | None = None) -> Path:
        return write_profile(self.profile, path)


def start_trace(source: str = "calibration") -> PhysicalMissTrace:
    """Start one process-local physical-miss trace."""
    global _ACTIVE_TRACE
    with _TRACE_LOCK:
        if _ACTIVE_TRACE is not None:
            raise RuntimeError("a physical-miss trace is already active")
        _ACTIVE_TRACE = PhysicalMissTrace(source)
        return _ACTIVE_TRACE


def stop_trace(path: str | Path | None = None) -> Path | None:
    """Stop the active trace and optionally write its evidence."""
    global _ACTIVE_TRACE
    with _TRACE_LOCK:
        trace = _ACTIVE_TRACE
        _ACTIVE_TRACE = None
    return trace.write(path) if trace is not None and path is not None else None


def record_trace_read(name: str, expert: int, physical_bytes: int, requested_bytes: int) -> None:
    """Record one read when a diagnostic trace is active."""
    with _TRACE_LOCK:
        trace = _ACTIVE_TRACE
        if trace is not None:
            trace.record(name, expert, physical_bytes, requested_bytes)


def add_observation(
    profile: MutableMapping[str, Any],
    layer: int,
    expert: int,
    *,
    physical_miss_bytes: int,
    requested_bytes: int = 0,
    route_count: int = 1,
    samples: int = 1,
) -> None:
    """Add one measured observation to an evidence document.

    ``physical_miss_bytes`` must come from a physical-read counter.  The
    requested byte count is retained for diagnostics and is not used as a
    substitute when physical evidence is absent.
    """
    if physical_miss_bytes < 0 or requested_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    if route_count < 0 or samples < 0:
        raise ValueError("counts must be non-negative")
    layers = profile.setdefault("layers", {})
    layer_data = layers.setdefault(str(int(layer)), {})
    experts = layer_data.setdefault("experts", {})
    row = experts.setdefault(
        str(int(expert)),
        {
            "physical_miss_bytes": 0,
            "requested_bytes": 0,
            "route_count": 0,
            "samples": 0,
        },
    )
    row["physical_miss_bytes"] += int(physical_miss_bytes)
    row["requested_bytes"] += int(requested_bytes)
    row["route_count"] += int(route_count)
    row["samples"] += int(samples)


def load_profile(path: str | Path | None = None) -> dict:
    """Load and validate physical-miss evidence without importing MLX."""
    profile_path = Path(path or DEFAULT_PROFILE).expanduser()
    with profile_path.open() as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict) or profile.get("version") != PROFILE_VERSION:
        raise ValueError(
            f"unsupported physical-miss profile: {profile_path} "
            f"(expected version {PROFILE_VERSION})"
        )
    layers = profile.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("physical-miss profile must contain a layers object")
    for layer, data in layers.items():
        if not isinstance(data, dict) or not isinstance(data.get("experts", {}), dict):
            raise ValueError(f"malformed physical-miss layer: {layer}")
        for expert, row in data.get("experts", {}).items():
            if not isinstance(row, dict):
                raise ValueError(f"malformed physical-miss row: {layer}/{expert}")
            value = int(row.get("physical_miss_bytes", 0))
            if value < 0:
                raise ValueError(f"negative physical bytes: {layer}/{expert}")
    return profile


def _rows(profile: Mapping[str, Any], min_samples: int) -> list[tuple[float, int, int]]:
    result = []
    for layer_text, layer_data in profile.get("layers", {}).items():
        layer = int(layer_text)
        for expert_text, row in layer_data.get("experts", {}).items():
            if int(row.get("samples", 0)) < min_samples:
                continue
            bytes_value = int(row.get("physical_miss_bytes", 0))
            if bytes_value > 0:
                result.append((float(bytes_value), layer, int(expert_text)))
    return result


def allocate_physical_miss_slots(
    profile: Mapping[str, Any],
    total_slots: int,
    *,
    min_slots: int = 4,
    max_slots: int = 6,
    num_layers: int = 12,
    min_samples: int = 1,
) -> Dict[int, list[int]]:
    """Allocate slots by observed physical bytes, with skew constraints.

    The first allocation gives each selected layer ``min_slots`` experts.
    Layers rank by the sum of their best measured physical misses.  Remaining
    slots go to the largest unused expert evidence, capped at ``max_slots``.
    This mirrors the existing skew topology while changing only its utility
    signal.  It returns an empty mapping when no measured physical evidence
    exists, preventing route frequency from being silently reused.
    """
    total_slots = int(total_slots)
    min_slots = int(min_slots)
    max_slots = int(max_slots)
    num_layers = int(num_layers)
    if total_slots <= 0 or min_slots <= 0 or max_slots < min_slots:
        raise ValueError("invalid physical-miss slot constraints")
    rows = _rows(profile, min_samples)
    if not rows:
        return {}
    by_layer: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for score, layer, expert in rows:
        by_layer[layer].append((score, expert))
    for layer in by_layer:
        by_layer[layer].sort(reverse=True)
    layer_order = sorted(
        by_layer,
        key=lambda layer: (sum(score for score, _ in by_layer[layer][:min_slots]), layer),
        reverse=True,
    )
    layer_order = layer_order[: min(num_layers, total_slots // min_slots)]
    allocation: Dict[int, list[int]] = {}
    for layer in layer_order:
        allocation[layer] = [expert for _, expert in by_layer[layer][:min_slots]]
    allocated = sum(len(experts) for experts in allocation.values())
    if allocated < total_slots:
        candidates = []
        for layer in layer_order:
            for position, (score, expert) in enumerate(by_layer[layer]):
                if position >= min_slots and position < max_slots:
                    candidates.append((score, layer, expert))
        candidates.sort(reverse=True)
        used = {(layer, expert) for layer, experts in allocation.items() for expert in experts}
        for score, layer, expert in candidates:
            if allocated >= total_slots:
                break
            if (layer, expert) in used:
                continue
            allocation[layer].append(expert)
            used.add((layer, expert))
            allocated += 1
    return allocation


def allocation_summary(profile: Mapping[str, Any], allocation: Mapping[int, Sequence[int]]) -> dict:
    """Return evidence totals for a proposed allocation."""
    selected = {
        (int(layer), int(expert))
        for layer, experts in allocation.items()
        for expert in experts
    }
    total = 0
    selected_bytes = 0
    for layer_text, data in profile.get("layers", {}).items():
        for expert_text, row in data.get("experts", {}).items():
            value = int(row.get("physical_miss_bytes", 0))
            total += value
            if (int(layer_text), int(expert_text)) in selected:
                selected_bytes += value
    return {
        "profile_physical_miss_bytes": total,
        "selected_physical_miss_bytes": selected_bytes,
        "selected_fraction": selected_bytes / total if total else 0.0,
        "selected_slots": len(selected),
        "selected_layers": len(allocation),
    }


def write_profile(profile: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Write evidence atomically and return its expanded path."""
    target = Path(path or DEFAULT_PROFILE).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target
