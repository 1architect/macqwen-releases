from pathlib import Path

from .api import TestSpec, FLASHNEXT


def _command(config, result_path: Path) -> list[str]:
    evidence = Path(config.results_dir).expanduser() / "io-worker-sweep.json"
    output = Path(config.results_dir).expanduser() / "io-task-topology.json"
    return [
        str(config.python), str(FLASHNEXT / "bench_io_scheduling.py"),
        "--mode", "topology",
        "--tokens", "32",
        "--rounds", str(max(3, config.pairs)),
        "--worker-json", str(evidence),
        "--json", str(output),
    ]


TEST = TestSpec(
    id="io-task-topology",
    title="I/O task topology comparison",
    category="diagnostic",
    explanation=(
        "Compares current projection-major read tasks with one task per expert, "
        "where each task reads three projections sequentially."
    ),
    why=(
        "Task coalescing may reduce queue residence, but it can also serialize "
        "storage service and remove useful concurrency."
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
        "tokens": "32, unchanged baseline horizon",
        "workers": "configured suite width, default 16",
        "topology": "projection-major versus one-task-per-expert",
        "reads": "same pread reads, chunk 2, same destinations",
        "slab": "same 60-slot skew allocation",
        "gate": "runs only when worker sweep shows material queue residence",
        "order": "reversed interleaved topology order",
    },
    source="models/flashnext/bench_io_scheduling.py",
    promotion=False,
)
