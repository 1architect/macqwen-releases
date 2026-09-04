from pathlib import Path
from .api import FLASHNEXT, IO_METRICS, TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), str(FLASHNEXT / "bench_slab_production.py"), "--arms", "slabpack60_skew,slabpack60_skew_f5c2,slabpack60_skew_f5c3", "--tokens", str(config.tokens), "--pairs", str(config.pairs), "--settle-seconds", "0"]


TEST = TestSpec(
    id="stream-records", title="Frontier 5 streamed records", category="performance",
    explanation="Compares standard destinations with expert-major chunks two and three.",
    why="It tests destination coalescing and task grouping while source reads stay unchanged.",
    script=command, metrics=IO_METRICS, controls={"slab": "60 skew slots", "Frontier": "8A"},
    source="models/flashnext/bench_slab_production.py", promotion=True,
)
