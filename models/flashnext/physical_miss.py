"""Physical-miss evidence and opt-in slab allocation.

The normal slab allocator uses route frequency.  This module adds a separate
diagnostic allocator for evidence collected during a long, constrained run.
The guarded runtime probe uses ``FLASHNEXT_SLAB_POLICY=physical-miss-hybrid``.
The original full replacement remains historical and is not a runtime policy.

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
from typing import Any, Dict, Mapping, MutableMapping, Sequence


PROFILE_VERSION = 1
DEFAULT_PROFILE = "~/.cache/flashnext/physical-misses.json"
HYBRID_POLICY = "physical-miss-hybrid"
HISTORICAL_POLICY = "physical-miss"
DEFAULT_SAFETY_MARGIN = 0.10
DEFAULT_MAX_PROFILE_AGE_HOURS = 24.0
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TRACE_LOCK = threading.Lock()
_ACTIVE_TRACE: "PhysicalMissTrace | None" = None
_LAST_HYBRID_SUMMARY: dict[str, Any] | None = None


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


def _row_bytes(row: Mapping[str, Any], *, fallback_per_route: float = 0.0) -> float:
    """Estimate the bytes represented by one evidence row.

    Positive physical bytes are the objective.  Requested bytes and route
    counts only provide a displacement cost when a canonical expert was
    resident during calibration and therefore recorded zero physical bytes.
    """
    physical = max(0.0, float(row.get("physical_miss_bytes", 0)))
    if physical:
        return physical
    requested = max(0.0, float(row.get("requested_bytes", 0)))
    if requested:
        return requested
    return max(0.0, float(row.get("route_count", 0))) * fallback_per_route


def _profile_fallback_per_route(profile: Mapping[str, Any]) -> float:
    requested = 0.0
    routes = 0.0
    for layer_data in profile.get("layers", {}).values():
        for row in layer_data.get("experts", {}).values():
            requested += max(0.0, float(row.get("requested_bytes", 0)))
            routes += max(0.0, float(row.get("route_count", 0)))
    return requested / routes if routes else 0.0


def _canonical_core(
    canonical_allocation: Mapping[int | str, Sequence[int]],
    min_slots: int,
    num_layers: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Split a canonical allocation into its preserved core and extensions."""
    normalized = {
        int(layer): [int(expert) for expert in experts]
        for layer, experts in canonical_allocation.items()
    }
    if len(normalized) != num_layers:
        raise ValueError(
            f"canonical allocation must contain {num_layers} layers, "
            f"got {len(normalized)}"
        )
    if any(len(experts) < min_slots for experts in normalized.values()):
        raise ValueError("canonical allocation does not contain the preserved core")
    if any(len(set(experts)) != len(experts) for experts in normalized.values()):
        raise ValueError("canonical allocation contains duplicate experts")
    core = {layer: experts[:min_slots] for layer, experts in normalized.items()}
    extensions = {
        layer: experts[min_slots:] for layer, experts in normalized.items()
    }
    return core, extensions


def _profile_is_stale(profile: Mapping[str, Any], max_age_hours: float) -> bool:
    created = profile.get("created_at")
    if not created or max_age_hours <= 0:
        return not bool(created)
    try:
        timestamp = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError:
        return True
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    return age.total_seconds() > max_age_hours * 3600.0


def allocate_physical_miss_hybrid_slots(
    profile: Mapping[str, Any],
    canonical_allocation: Mapping[int | str, Sequence[int]],
    total_slots: int = 60,
    *,
    min_slots: int = 4,
    max_slots: int = 6,
    num_layers: int = 12,
    min_samples: int = 1,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    max_age_hours: float = DEFAULT_MAX_PROFILE_AGE_HOURS,
    require_core_calibration: bool = False,
) -> Dict[int, list[int]]:
    """Preserve the canonical core and replace only safe extensions.

    The canonical allocation supplies four established experts in each of the
    twelve hot layers.  Physical evidence may replace only the remaining
    extension slots.  A candidate must beat the displaced extension by the
    safety margin.  Missing or stale evidence leaves the canonical extension
    unchanged.
    """
    global _LAST_HYBRID_SUMMARY

    total_slots = int(total_slots)
    min_slots = int(min_slots)
    max_slots = int(max_slots)
    num_layers = int(num_layers)
    safety_margin = float(safety_margin)
    if total_slots <= 0 or min_slots <= 0 or max_slots < min_slots:
        raise ValueError("invalid physical-miss slot constraints")
    if safety_margin < 0:
        raise ValueError("safety margin must be non-negative")
    core, canonical_extensions = _canonical_core(
        canonical_allocation, min_slots, num_layers
    )
    normalized_canonical = {
        int(layer): [int(expert) for expert in experts]
        for layer, experts in canonical_allocation.items()
    }
    canonical_total = sum(
        len(experts) for experts in core.values()
    ) + sum(len(experts) for experts in canonical_extensions.values())
    if canonical_total != total_slots:
        raise ValueError(
            f"canonical allocation has {canonical_total} slots, expected {total_slots}"
        )
    if any(len(core[layer]) + len(canonical_extensions[layer]) > max_slots
           for layer in core):
        raise ValueError("canonical allocation exceeds max slots per layer")

    fallback_per_route = _profile_fallback_per_route(profile)
    rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    for layer_text, layer_data in profile.get("layers", {}).items():
        layer = int(layer_text)
        for expert_text, row in layer_data.get("experts", {}).items():
            if int(row.get("samples", 0)) >= min_samples:
                rows[(layer, int(expert_text))] = row

    allocation = {
        layer: list(experts) for layer, experts in normalized_canonical.items()
    }
    calibration = profile.get("calibration", {})
    invalid_calibration = require_core_calibration and not (
        calibration.get("canonical_core_slots") == 48
        and calibration.get("equal_residency") is True
    )
    if _profile_is_stale(profile, max_age_hours) or invalid_calibration:
        _LAST_HYBRID_SUMMARY = hybrid_allocation_summary(
            profile, normalized_canonical, allocation,
            changed=(), safety_margin=safety_margin,
        )
        _LAST_HYBRID_SUMMARY["fallback_reason"] = (
            "profile is not an equal-residency 48-slot core calibration"
            if invalid_calibration else "missing or stale evidence"
        )
        return allocation
    used = {(layer, expert) for layer, experts in allocation.items() for expert in experts}
    changed: list[dict[str, float | int]] = []

    for layer in sorted(core):
        extension_count = len(canonical_extensions[layer])
        if not extension_count:
            continue
        core_experts = set(core[layer])
        candidates = []
        for (candidate_layer, expert), row in rows.items():
            if candidate_layer != layer or expert in core_experts:
                continue
            benefit = float(row.get("physical_miss_bytes", 0))
            if benefit <= 0:
                continue
            candidates.append((benefit, expert, row))
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        incumbents = []
        for position, incumbent in enumerate(canonical_extensions[layer]):
            incumbent_row = rows.get((layer, incumbent))
            if incumbent_row is None:
                continue
            displacement = _row_bytes(
                incumbent_row, fallback_per_route=fallback_per_route
            )
            if displacement <= 0:
                continue
            incumbents.append((displacement, position, incumbent))
        incumbents.sort()
        for displacement, position, incumbent in incumbents:
            selected = None
            for benefit, candidate, row in candidates:
                if (layer, candidate) in used or candidate in core_experts:
                    continue
                required = displacement * (1.0 + safety_margin)
                if benefit <= 0 or benefit <= required:
                    continue
                selected = (benefit, candidate, row)
                break
            if selected is None:
                continue
            benefit, candidate, row = selected
            old_index = min_slots + position
            allocation[layer][old_index] = candidate
            used.discard((layer, incumbent))
            used.add((layer, candidate))
            candidates = [item for item in candidates if item[1] != candidate]
            changed.append({
                "layer": layer,
                "old_expert": incumbent,
                "new_expert": candidate,
                "candidate_benefit_bytes": benefit,
                "displacement_cost_bytes": displacement,
                "net_value_bytes": benefit - displacement,
            })

    if any(len(set(experts)) != len(experts) for experts in allocation.values()):
        raise ValueError("hybrid allocation contains duplicate experts")
    _LAST_HYBRID_SUMMARY = hybrid_allocation_summary(
        profile,
        canonical_allocation,
        allocation,
        changed=changed,
        safety_margin=safety_margin,
    )
    return allocation


def hybrid_allocation_summary(
    profile: Mapping[str, Any],
    canonical_allocation: Mapping[int | str, Sequence[int]],
    allocation: Mapping[int | str, Sequence[int]],
    *,
    changed: Sequence[Mapping[str, Any]] | None = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, Any]:
    """Return provenance for a hybrid allocation decision."""
    canonical = {
        (int(layer), int(expert))
        for layer, experts in canonical_allocation.items()
        for expert in experts
    }
    selected = {
        (int(layer), int(expert))
        for layer, experts in allocation.items()
        for expert in experts
    }
    preserved = canonical & selected
    core = {
        (int(layer), int(expert))
        for layer, experts in canonical_allocation.items()
        for expert in list(experts)[:4]
    }
    fallback_per_route = _profile_fallback_per_route(profile)
    changed_rows = list(changed or [])
    if not changed_rows:
        for layer, expert in sorted(canonical - selected):
            replacement = next(
                (new for new in selected if new[0] == layer and new not in canonical),
                None,
            )
            if replacement is None:
                continue
            old_row = profile.get("layers", {}).get(str(layer), {}).get(
                "experts", {}
            ).get(str(expert), {})
            new_row = profile.get("layers", {}).get(str(layer), {}).get(
                "experts", {}
            ).get(str(replacement[1]), {})
            changed_rows.append({
                "layer": layer,
                "old_expert": expert,
                "new_expert": replacement[1],
                "candidate_benefit_bytes": float(new_row.get("physical_miss_bytes", 0)),
                "displacement_cost_bytes": _row_bytes(
                    old_row, fallback_per_route=fallback_per_route
                ),
                "net_value_bytes": float(new_row.get("physical_miss_bytes", 0))
                - _row_bytes(old_row, fallback_per_route=fallback_per_route),
            })
    benefit = sum(float(row.get("candidate_benefit_bytes", 0)) for row in changed_rows)
    cost = sum(float(row.get("displacement_cost_bytes", 0)) for row in changed_rows)
    return {
        "policy": HYBRID_POLICY,
        "profile_source": profile.get("source", ""),
        "profile_created_at": profile.get("created_at", ""),
        "profile_tokens": int(profile.get("tokens", 0)),
        "total_slots": len(selected),
        "preserved_core_slots": len(core & selected),
        "preserved_slots": len(preserved),
        "changed_extension_slots": len(changed_rows),
        "candidate_benefit_bytes": benefit,
        "displacement_cost_bytes": cost,
        "net_value_bytes": benefit - cost,
        "safety_margin": float(safety_margin),
        "changes": changed_rows,
    }


def last_hybrid_summary() -> dict[str, Any] | None:
    """Return the latest hybrid provenance without exposing mutable state."""
    return dict(_LAST_HYBRID_SUMMARY) if _LAST_HYBRID_SUMMARY is not None else None


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
