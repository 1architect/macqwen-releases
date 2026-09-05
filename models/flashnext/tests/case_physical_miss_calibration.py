"""Long-run serialized physical-miss evidence collection."""
from __future__ import annotations

from pathlib import Path

from .api import FLASHNEXT, TestSpec
from .case_product_long import live_parser


def command(config, _result_path: Path) -> list[str]:
    profile = Path("~/.cache/flashnext/physical-misses.json").expanduser()
    return [
        str(config.python), str(FLASHNEXT / "bench_product_long.py"),
        "--tokens", "256", "--window", "32", "--rounds", "1",
        "--paths", "core48-calibration", "--phase", "answer",
        "--trace-profile", str(profile),
    ]


TEST = TestSpec(
    id="physical-miss-calibration",
    title="Long-run physical-miss calibration",
    category="diagnostic",
    explanation=(
        "Runs one canonical product generation with serialized expert reads "
        "and writes measured physical bytes for each layer and expert."
    ),
    why=(
        "The optimized long run loses slab hits and gains physical reads. "
        "Route frequency cannot identify which requests reach NVMe."
    ),
    script=command,
    metrics=(
        "generation rate per 32-token window",
        "physical MB/token per window",
        "slab hit rate per window",
        "per-expert physical miss bytes",
        "answer phase",
    ),
    controls={
        "startup": "canonical 48-slot core with Frontier 8A",
        "I/O workers": "1 for attributable physical counters",
        "pread chunk": "1 expert",
        "profile": "~/.cache/flashnext/physical-misses.json",
        "status": "diagnostic only; serialized I/O changes throughput",
        "rounds": "one",
    },
    source="models/flashnext/bench_product_long.py and physical_miss.py",
    promotion=False,
    live_parser=live_parser,
    canonical=False,
)
