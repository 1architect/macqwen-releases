#!/usr/bin/env python3
"""Analyze a long-run physical-miss profile and propose slab slots.

This is a diagnostic analysis tool.  It does not load a model or measure
anything.  The profile must come from a constrained calibration run that can
attribute physical bytes to individual ``(layer, expert)`` reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.flashnext.physical_miss import (
    allocation_summary,
    allocate_physical_miss_slots,
    load_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="~/.cache/flashnext/physical-misses.json",
        help="measured physical-miss evidence JSON",
    )
    parser.add_argument("--slots", type=int, default=60)
    parser.add_argument("--min-slots", type=int, default=4)
    parser.add_argument("--max-slots", type=int, default=6)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--write-allocation",
        type=Path,
        help="optional JSON file for the proposed allocation",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    allocation = allocate_physical_miss_slots(
        profile,
        args.slots,
        min_slots=args.min_slots,
        max_slots=args.max_slots,
        num_layers=args.layers,
        min_samples=args.min_samples,
    )
    if not allocation:
        raise SystemExit(
            "No measured physical-miss evidence met the requested sample threshold."
        )
    summary = allocation_summary(profile, allocation)
    print("physical-miss slab allocation (diagnostic)")
    print(f"profile: {Path(args.profile).expanduser()}")
    print(f"slots: {summary['selected_slots']} across {summary['selected_layers']} layers")
    print(
        f"selected physical misses: {summary['selected_physical_miss_bytes'] / 1e9:.3f} GB "
        f"of {summary['profile_physical_miss_bytes'] / 1e9:.3f} GB "
        f"({summary['selected_fraction'] * 100:.1f}%)"
    )
    for layer in sorted(allocation):
        print(f"  layer {layer:02d}: {', '.join(map(str, allocation[layer]))}")
    if args.write_allocation:
        args.write_allocation.parent.mkdir(parents=True, exist_ok=True)
        args.write_allocation.write_text(
            json.dumps(
                {
                    "version": 1,
                    "policy": "physical-miss",
                    "source": str(Path(args.profile).expanduser()),
                    "allocation": {
                        str(layer): experts
                        for layer, experts in sorted(allocation.items())
                    },
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote: {args.write_allocation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
