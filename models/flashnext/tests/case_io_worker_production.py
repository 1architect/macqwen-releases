"""Unprofiled confirmation of the diagnostic worker-width candidate."""
from pathlib import Path

from .api import COMMON_METRICS, FLASHNEXT, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = Path(config.results_dir).expanduser() / "io-worker-production.json"
    return [
        str(config.python), str(FLASHNEXT / "bench_io_scheduling.py"),
        "--mode", "production", "--tokens", "32",
        "--rounds", str(max(3, config.pairs)), "--json", str(evidence),
    ]


TEST = TestSpec(
    id="io-worker-production",
    title="Unprofiled 8 versus 16 I/O workers",
    category="performance",
    explanation=(
        "Compares eight workers with the current sixteen-worker runtime in "
        "fresh processes. Each arm generates 32 tokens without I/O profiling."
    ),
    why=(
        "Eight workers led the diagnostic sweep slightly. Profiling changed "
        "memory pressure, so that result cannot select a production default."
    ),
    script=command,
    metrics=COMMON_METRICS + ("VM deltas", "effective worker count", "allocation digest"),
    controls={
        "control": "16 workers",
        "candidate": "8 workers",
        "runtime": "60 skew slots, Frontier 8A, Up-QMV/SwiGLU on",
        "reads": "pread, chunk 2, projection tasks, profiling off",
        "horizon": "32 tokens, at least three reversed pairs",
        "pins": "identical private snapshot per arm",
        "digest": "same token and slab allocation digests across processes",
        "scope": "short comparison; no long-generation gain claimed",
    },
    source="models/flashnext/bench_io_scheduling.py",
    promotion=False,
)
