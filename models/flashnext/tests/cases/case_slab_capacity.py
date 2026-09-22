from pathlib import Path
from macqwen.testsuite.api import IO_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", bench_module("flashnext", "bench_slab_production"), "--arms", "slabpack56_skew,slabpack60_skew,slabpack64_skew", "--tokens", str(config.tokens), "--pairs", str(config.pairs), "--settle-seconds", "0"]


TEST = TestSpec(
    id="slab-capacity", title="56/60/64 skew capacity", category="performance",
    explanation="Compares three resident capacities with corrected decode-only counters.",
    why="It tests whether more logical resident hits reduce physical I/O after the high-value set is captured.",
    script=command, metrics=IO_METRICS, controls={"foundation": "Frontier 8A", "digest": "required", "I/O profiler": "off"},
    source="models/flashnext/tests/bench/bench_slab_production.py", promotion=True,
)
