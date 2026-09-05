"""Corrected decode-only Section 17 control."""
from pathlib import Path

from .api import FLASHNEXT, TestSpec


def command(config, _result_path: Path) -> list[str]:
    evidence = Path(config.results_dir).expanduser() / "io-section17-control.json"
    return [
        str(config.python), str(FLASHNEXT / "bench_io_scheduling.py"),
        "--mode", "control", "--tokens", "32",
        "--rounds", str(max(3, config.pairs)), "--json", str(evidence),
    ]


TEST = TestSpec(
    id="io-section17-control",
    title="Corrected decode-only Section 17 control",
    category="diagnostic",
    explanation=(
        "Measures the 16-worker projection-task control after prefill and "
        "records complete Frontier 5 attribution."
    ),
    why=(
        "The historical queue result included prefill. The worker sweep must "
        "remain blocked until corrected queue residence is material."
    ),
    script=command,
    metrics=(
        "queue residence", "positioned-read time", "task overhead",
        "layer completion", "total I/O wait", "physical MB/token",
        "generation and tail rate", "active RAM", "exact digest",
    ),
    controls={
        "tokens": "32",
        "workers": "16",
        "topology": "projection",
        "profiling": "decode-only Frontier 5 attribution",
    },
    source="models/flashnext/bench_io_scheduling.py",
    promotion=False,
)
