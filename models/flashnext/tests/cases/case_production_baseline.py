from pathlib import Path
from macqwen.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", bench_module("flashnext", "bench_production"), "--compare", "none", "--tokens", str(config.tokens), "--arms", str(config.pairs), "--min-arms", str(config.pairs), "--drop", "0"]


TEST = TestSpec(
    id="production-baseline", title="Canonical production baseline", category="performance",
    explanation="Measures the current canonical greedy decode without a target condition.",
    why="Every optimization needs a current rate, physical-byte, memory, and drift reference.",
    script=command, metrics=COMMON_METRICS, controls={"foundation": "60-slot skew + Frontier 8A"},
    source="models/flashnext/tests/bench/bench_production.py", promotion=False,
)
