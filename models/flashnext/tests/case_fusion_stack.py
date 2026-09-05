"""Frontier 8A plus Up-QMV/SwiGLU versus Frontier 8B plus Up."""
from pathlib import Path

from .api import FLASHNEXT, IO_METRICS, TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [
        str(config.python), str(FLASHNEXT / "bench_slab_production.py"),
        "--arms", "slabpack60_skew_8a_up,slabpack60_skew_8b_up",
        "--tokens", str(config.tokens), "--pairs", str(config.pairs),
        "--settle-seconds", "0",
    ]


TEST = TestSpec(
    id="fusion-stack",
    title="Frontier 8A plus Up versus Frontier 8B plus Up",
    category="performance",
    explanation=(
        "Compares the 8A finalization stack with Up-QMV/SwiGLU against the "
        "additional 8B shared multiply, with all slab controls fixed."
    ),
    why=(
        "8B was previously measured without the retained Up fusion. This "
        "isolates the complete fusion stack rather than mixing partial stacks."
    ),
    script=command,
    metrics=IO_METRICS,
    controls={
        "control": "60-slot skew, Frontier 8A, Up-QMV/SwiGLU on",
        "candidate": "60-slot skew, Frontier 8B, Up-QMV/SwiGLU on",
        "decode": "greedy",
        "I/O profiler": "off",
        "digest": "required",
    },
    source="models/flashnext/bench_slab_production.py",
    promotion=False,
)
