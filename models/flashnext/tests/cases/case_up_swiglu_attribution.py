from pathlib import Path
from macqwen.testsuite.api import IO_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", bench_module("flashnext", "bench_slab_production"), "--arms", "slabpack60_skew,slabpack60_skew_f10_up", "--tokens", str(config.tokens), "--pairs", str(config.pairs), "--settle-seconds", "0", "--profile-io"]


TEST = TestSpec(
    id="up-swiglu-attribution", title="Up-QMV/SwiGLU I/O attribution", category="diagnostic",
    explanation="Runs the same comparison with perturbing per-read Frontier 5 instrumentation enabled.",
    why="It attributes queue and positioned-read time after the uninstrumented production rate is known.",
    script=command, metrics=IO_METRICS,
    controls={"slab": "60 skew slots", "Frontier": "8A", "I/O profiler": "on; rate is diagnostic"},
    source="models/flashnext/tests/bench/bench_slab_production.py", promotion=False,
)
