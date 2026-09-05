from pathlib import Path

from .api import TestSpec, ROOT, FLASHNEXT


def _command(config, result_path: Path) -> list[str]:
    evidence = Path(config.results_dir).expanduser() / "io-worker-sweep.json"
    control = Path(config.results_dir).expanduser() / "io-section17-control.json"
    return [
        str(config.python), str(FLASHNEXT / "bench_io_scheduling.py"),
        "--mode", "workers",
        "--tokens", "32",
        "--rounds", str(max(3, config.pairs)),
        "--control-json", str(control),
        "--json", str(evidence),
    ]


TEST = TestSpec(
    id="io-worker-sweep",
    title="Decode-only I/O worker sweep",
    category="diagnostic",
    explanation=(
        "Runs the corrected 32-token decode-only Section 17 control at 8, 16, "
        "24, and 32 workers."
    ),
    why=(
        "Queue residence is large, but worker saturation, storage limits, "
        "scheduling, and Python contention remain unisolated."
    ),
    script=_command,
    metrics=(
        "queue residence per token",
        "positioned-read wall time per token",
        "task overhead and layer completion",
        "total I/O wait",
        "physical MB/token",
        "generation and tail rate",
        "active RAM",
        "exact token digest",
    ),
    controls={
        "tokens": "32, unchanged Section 17 horizon",
        "workers": "8, 16, 24, 32",
        "topology": "current projection-major tasks",
        "slab": "60-slot skew pack",
        "fusion": "Frontier 8A and current Up-QMV/SwiGLU setting",
        "reads": "pread, chunk 2, profiling enabled only for attribution",
        "order": "interleaved forward and reverse worker order",
    },
    source="models/flashnext/bench_io_scheduling.py",
    promotion=False,
)
