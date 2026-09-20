"""Migrate retained legacy measurement files into canonical JSONL."""
from __future__ import annotations

import argparse
from pathlib import Path

from .measurement import migrate_legacy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    migrate_legacy(
        args.source, args.destination,
        runtime=args.runtime, experiment=args.experiment,
    )
    print(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
