"""Offline slab-topology scoring from measured physical-miss evidence.

This module has no MLX dependency.  It consumes a calibration profile and
produces candidates for analysis only.  A logical route hit never contributes
to the objective.  Runtime selection remains the canonical skew allocation or
the guarded ``physical-miss-hybrid`` policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .physical_miss import (
    DEFAULT_SAFETY_MARGIN,
    allocate_physical_miss_hybrid_slots,
    hybrid_allocation_summary,
    load_profile,
)


TOPOLOGY_NAMES = ("current", "depth6", "depth8", "depth10", "canonical-core-hybrid")
DEFAULT_CEILING_MB_PER_TOKEN = 20.0


def canonical_skew_allocation_from_pins(
    path: str | Path,
    total_slots: int = 60,
    *,
    min_slots: int = 4,
    max_slots: int = 6,
    num_layers: int = 12,
) -> dict[int, list[int]]:
    """Reproduce the runtime skew allocation without importing MLX."""
    payload = json.loads(Path(path).expanduser().read_text())
    ranked = payload.get("ranked_counts") or payload.get("ranked_scores", {})
    if not ranked:
        raise ValueError("pin profile has no ranked expert evidence")
    layer_scores = sorted(
        (
            (sum(float(score) for _expert, score in pairs[:min_slots]), int(layer))
            for layer, pairs in ranked.items()
        ),
        reverse=True,
    )
    selected = [layer for _score, layer in layer_scores[:num_layers]]
    allocation = {
        layer: [int(expert) for expert, _score in ranked[str(layer)][:min_slots]]
        for layer in selected
    }
    while sum(len(experts) for experts in allocation.values()) < total_slots:
        best = None
        for layer in selected:
            count = len(allocation[layer])
            pairs = ranked[str(layer)]
            if count >= max_slots or count >= len(pairs):
                continue
            expert, score = pairs[count]
            candidate = (float(score), -layer, int(expert), layer)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise ValueError("pin profile cannot fill the canonical slot budget")
        allocation[best[3]].append(best[2])
    return allocation


def _rows(profile: Mapping[str, Any], min_samples: int = 1):
    for layer_text, layer_data in profile.get("layers", {}).items():
        layer = int(layer_text)
        for expert_text, row in layer_data.get("experts", {}).items():
            if int(row.get("samples", 0)) < min_samples:
                continue
            physical = max(0, int(row.get("physical_miss_bytes", 0)))
            if physical:
                yield layer, int(expert_text), physical


def physical_miss_score(
    profile: Mapping[str, Any],
    allocation: Mapping[int | str, Sequence[int]],
    *,
    min_samples: int = 1,
) -> int:
    """Score selected slots by measured physical-miss bytes only."""
    selected = {
        (int(layer), int(expert))
        for layer, experts in allocation.items()
        for expert in experts
    }
    return sum(
        physical
        for layer, expert, physical in _rows(profile, min_samples)
        if (layer, expert) in selected
    )


def _ranked_layers(profile: Mapping[str, Any], min_samples: int) -> list[int]:
    totals: dict[int, int] = {}
    for layer, _expert, physical in _rows(profile, min_samples):
        totals[layer] = totals.get(layer, 0) + physical
    return [layer for layer, _total in sorted(
        totals.items(), key=lambda item: (item[1], -item[0]), reverse=True
    )]


def _ranked_experts(profile: Mapping[str, Any], layer: int, min_samples: int) -> list[int]:
    candidates = [
        (physical, expert)
        for row_layer, expert, physical in _rows(profile, min_samples)
        if row_layer == layer
    ]
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [expert for _physical, expert in candidates]


def _depth_allocation(
    profile: Mapping[str, Any],
    depth: int,
    total_slots: int,
    *,
    min_samples: int,
) -> dict[int, list[int]]:
    """Build a diagnostic fixed-slot shape over the top ``depth`` layers.

    Depth candidates intentionally model concentrated layer shapes.  They are
    offline hypotheses and may exceed production's 4..6 per-layer limit.
    The hybrid policy is the constrained candidate used by runtime tests.
    """
    layers = _ranked_layers(profile, min_samples)[:depth]
    if not layers:
        return {}
    ranked = {layer: _ranked_experts(profile, layer, min_samples) for layer in layers}
    allocation = {layer: [] for layer in layers}
    cursor = 0
    while cursor < total_slots:
        made_progress = False
        for layer in layers:
            experts = ranked[layer]
            if len(allocation[layer]) >= len(experts):
                continue
            allocation[layer].append(experts[len(allocation[layer])])
            cursor += 1
            made_progress = True
            if cursor >= total_slots:
                break
        if not made_progress:
            break
    return {layer: experts for layer, experts in allocation.items() if experts}


def canonical_core_calibration(
    canonical_allocation: Mapping[int | str, Sequence[int]],
    profile: Mapping[str, Any] | None = None,
    *,
    min_slots: int = 4,
    min_samples: int = 1,
) -> dict[str, Any]:
    """Describe a 48-slot core run and expose one extension pool per layer.

    The core keeps four canonical experts in each of twelve layers.  The
    extension pools are reported separately so a calibration can test all
    twelve extensions under equal residency conditions.
    """
    normalized = {
        int(layer): [int(expert) for expert in experts]
        for layer, experts in canonical_allocation.items()
    }
    if len(normalized) != 12 or any(len(experts) < min_slots for experts in normalized.values()):
        raise ValueError("canonical calibration needs twelve layers with four core slots")
    core = {layer: experts[:min_slots] for layer, experts in normalized.items()}
    core_pairs = {(layer, expert) for layer, experts in core.items() for expert in experts}
    extensions: dict[int, list[int]] = {}
    for layer in sorted(core):
        if profile is None:
            extensions[layer] = list(normalized[layer][min_slots:])
            continue
        extensions[layer] = [
            expert for expert in _ranked_experts(profile, layer, min_samples)
            if (layer, expert) not in core_pairs
        ]
    return {
        "policy": "canonical-core-calibration",
        "slots": sum(len(experts) for experts in core.values()),
        "layers": sorted(core),
        "allocation": core,
        "extension_candidates": extensions,
        "equal_residency": True,
    }


def offline_ceiling_gate(
    selected_physical_bytes: int | float,
    tokens: int,
    *,
    minimum_mb_per_token: float = DEFAULT_CEILING_MB_PER_TOKEN,
) -> dict[str, Any]:
    """Reject offline proposals below the configured physical-saving ceiling."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if minimum_mb_per_token < 0:
        raise ValueError("minimum ceiling must be non-negative")
    mb_per_token = float(selected_physical_bytes) / 1_000_000.0 / tokens
    return {
        "minimum_mb_per_token": float(minimum_mb_per_token),
        "predicted_mb_per_token": mb_per_token,
        "passes": mb_per_token >= minimum_mb_per_token,
    }


def simulate_topologies(
    profile: Mapping[str, Any],
    canonical_allocation: Mapping[int | str, Sequence[int]],
    *,
    total_slots: int = 60,
    min_slots: int = 4,
    max_slots: int = 6,
    num_layers: int = 12,
    min_samples: int = 1,
    minimum_mb_per_token: float = DEFAULT_CEILING_MB_PER_TOKEN,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> dict[str, dict[str, Any]]:
    """Score current, depth, and constrained hybrid shapes offline."""
    allocations: dict[str, Mapping[int, Sequence[int]]] = {
        "current": canonical_allocation,
        "depth6": _depth_allocation(
            profile, 6, total_slots, min_samples=min_samples
        ),
        "depth8": _depth_allocation(
            profile, 8, total_slots, min_samples=min_samples
        ),
        "depth10": _depth_allocation(
            profile, 10, total_slots, min_samples=min_samples
        ),
        "canonical-core-hybrid": allocate_physical_miss_hybrid_slots(
            profile,
            canonical_allocation,
            total_slots,
            min_slots=min_slots,
            max_slots=max_slots,
            num_layers=num_layers,
            min_samples=min_samples,
            safety_margin=safety_margin,
        ),
    }
    tokens = int(profile.get("tokens", 0))
    current_score = physical_miss_score(
        profile, canonical_allocation, min_samples=min_samples
    )
    result = {}
    for name, allocation in allocations.items():
        selected = physical_miss_score(profile, allocation, min_samples=min_samples)
        predicted_saving = max(0, selected - current_score)
        gate = offline_ceiling_gate(
            predicted_saving, tokens,
            minimum_mb_per_token=minimum_mb_per_token,
        ) if tokens else {
            "minimum_mb_per_token": minimum_mb_per_token,
            "predicted_mb_per_token": 0.0,
            "passes": False,
        }
        result[name] = {
            "name": name,
            "allocation": {
                str(layer): [int(expert) for expert in experts]
                for layer, experts in sorted(allocation.items())
            },
            "slots": sum(len(experts) for experts in allocation.values()),
            "layers": len(allocation),
            "physical_miss_bytes": selected,
            "predicted_saving_physical_bytes": predicted_saving,
            "offline_ceiling": gate,
            "objective": "measured physical-miss bytes",
        }
        if name == "canonical-core-hybrid":
            result[name]["provenance"] = hybrid_allocation_summary(
                profile, canonical_allocation, allocation,
                safety_margin=safety_margin,
            )
    return result


def window_evidence(records: Sequence[Mapping[str, Any]], window: int = 32) -> list[dict[str, Any]]:
    """Normalize product benchmark evidence while preserving phase labels."""
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    for index, record in enumerate(records, start=1):
        if record.get("type") != "window":
            continue
        row = dict(record)
        row.setdefault("window", index)
        row.setdefault("tokens", window)
        row.setdefault("phase", "unknown")
        row["phase"] = str(row["phase"])
        result.append(row)
    return result


def _load_canonical(path: Path) -> dict[int, list[int]]:
    data = json.loads(path.read_text())
    allocation = data.get("allocation", data)
    return {
        int(layer): [int(expert) for expert in experts]
        for layer, experts in allocation.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--canonical-allocation", type=Path)
    parser.add_argument(
        "--pin-profile", type=Path,
        default=Path("~/.cache/flashnext/pins.json").expanduser(),
        help="derive canonical skew allocation when no allocation JSON is given",
    )
    parser.add_argument("--slots", type=int, default=60)
    parser.add_argument("--min-slots", type=int, default=4)
    parser.add_argument("--max-slots", type=int, default=6)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--minimum-mb-token", type=float, default=DEFAULT_CEILING_MB_PER_TOKEN
    )
    args = parser.parse_args()
    profile = load_profile(args.profile)
    canonical = (
        _load_canonical(args.canonical_allocation)
        if args.canonical_allocation
        else canonical_skew_allocation_from_pins(
            args.pin_profile, args.slots,
            min_slots=args.min_slots,
            max_slots=args.max_slots,
            num_layers=args.layers,
        )
    )
    print(json.dumps(simulate_topologies(
        profile,
        canonical,
        total_slots=args.slots,
        min_slots=args.min_slots,
        max_slots=args.max_slots,
        num_layers=args.layers,
        min_samples=args.min_samples,
        minimum_mb_per_token=args.minimum_mb_token,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
