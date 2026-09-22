from pathlib import Path
from macqwen.testsuite.api import IO_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", bench_module("flashnext", "bench_slab_production"), "--arms", "slabpack60_skew_8a,slabpack60_skew_8b", "--tokens", str(config.tokens), "--pairs", str(config.pairs), "--settle-seconds", "0"]


TEST = TestSpec(
    id="frontier-8b", title="Frontier 8B shared multiply", category="performance",
    explanation="Compares Frontier 8A finalization with the additional shared multiply fusion.",
    why="It tests whether removing another elementwise boundary adds value beyond routed-output finalization.",
    script=command, metrics=IO_METRICS, controls={"slab": "60 skew slots", "8A": "control", "I/O profiler": "off"},
    source="models/flashnext/tests/bench/bench_slab_production.py", promotion=True,
)
