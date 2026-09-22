from pathlib import Path
from macqwen.testsuite.api import IO_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", bench_module("flashnext", "bench_slab_production"), "--arms", "slabpack60_skew,slabpack60_skew_f5c2,slabpack60_skew_f5c3", "--tokens", str(config.tokens), "--pairs", str(config.pairs), "--settle-seconds", "0"]


TEST = TestSpec(
    id="stream-records", title="Frontier 5 streamed records", category="performance",
    explanation="Compares standard destinations with expert-major chunks two and three.",
    why="It tests destination coalescing and task grouping while source reads stay unchanged.",
    script=command, metrics=IO_METRICS, controls={"slab": "60 skew slots", "Frontier": "8A"},
    source="models/flashnext/tests/bench/bench_slab_production.py", promotion=True,
)
