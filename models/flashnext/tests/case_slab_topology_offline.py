"""Offline physical-evidence topology gate."""
from pathlib import Path

from .api import FLASHNEXT, TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [
        str(config.python), "-m", "models.flashnext.slab_topology",
        "--profile", str(Path("~/.cache/flashnext/physical-misses.json").expanduser()),
        "--pin-profile", str(Path("~/.cache/flashnext/pins.json").expanduser()),
        "--slots", "60", "--min-slots", "4", "--max-slots", "6",
        "--layers", "12", "--minimum-mb-token", "20",
    ]


TEST = TestSpec(
    id="slab-topology-offline",
    title="Offline slab topology and physical ceiling gate",
    category="diagnostic",
    explanation=(
        "Scores current, depth 6, depth 8, depth 10, and guarded hybrid "
        "60-slot shapes from equal-residency physical evidence."
    ),
    why=(
        "A model comparison is too expensive when measured evidence predicts "
        "less than 20 MB/token of physical-read savings."
    ),
    script=command,
    metrics=(
        "predicted physical MB/token saving", "20 MB/token premise gate",
        "slot and layer totals", "hybrid replacement provenance",
    ),
    controls={
        "objective": "physical misses, never logical hit rate",
        "slots": "60",
        "canonical core": "48 slots across 12 layers",
        "runtime": "no model load",
    },
    source="models/flashnext/slab_topology.py",
    promotion=False,
    canonical=False,
)
