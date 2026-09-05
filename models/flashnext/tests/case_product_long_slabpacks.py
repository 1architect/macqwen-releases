"""Long-horizon slab capacity comparison."""
from __future__ import annotations

from pathlib import Path

from .api import FLASHNEXT, TestSpec
from .case_product_long import interpret, live_parser


def command(config, _result_path: Path) -> list[str]:
    return [
        str(config.python), str(FLASHNEXT / "bench_product_long.py"),
        "--tokens", "256", "--window", "32",
        "--rounds", "1", "--phase", "answer",
        "--paths", "slabpack60", "slabpack48",
    ]


TEST = TestSpec(
    id="product-long-slabpacks",
    title="Single long-answer slabpack48 versus slabpack60 pair",
    category="performance",
    explanation=(
        "Runs one answer-focused arm for each slab capacity. It prints one "
        "record for every 32-token window."
    ),
    why=(
        "The 32-token capacity sweep cannot show whether route locality "
        "decays after the working set expands. This is a separate horizon."
    ),
    script=command,
    metrics=(
        "generation rate per 32-token window",
        "physical MB/token per window",
        "slab hit rate per window",
        "active memory per window",
        "context size per window",
        "thinking and answer phase counts",
        "exact token digest",
    ),
    controls={
        "horizon": "256 generated tokens",
        "window": "32 generated tokens",
        "control": "60-slot skew pack, Frontier 8A",
        "candidate": "48-slot skew pack, Frontier 8A",
        "decode": "greedy",
        "digest": "required",
        "rounds": "one pair; directional validation only",
    },
    source="models/flashnext/bench_product_long.py",
    promotion=False,
    live_parser=live_parser,
    interpret=interpret,
    canonical=False,
)
