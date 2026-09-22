from pathlib import Path
from macqwen.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", bench_module("flashnext", "bench_slab_production"), "--arms", "slabpack60_skew,slabpack60_skew_f10_up", "--tokens", str(config.tokens), "--pairs", str(config.pairs), "--settle-seconds", "0"]


TEST = TestSpec(
    id="up-swiglu", title="Up-QMV to SwiGLU fusion", category="performance",
    explanation="Compares the standard Up and SwiGLU graph with the fused Up epilogue.",
    why="It tests removal of the Up output materialization and separate SwiGLU boundary.",
    script=command, metrics=COMMON_METRICS, controls={"slab": "60 skew slots", "Frontier": "8A", "I/O profiler": "off", "baseline judgment": "user"},
    source="models/flashnext/tests/bench/bench_slab_production.py", promotion=True,
)
