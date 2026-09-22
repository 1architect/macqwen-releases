"""Screen prepared QMM metadata with paging counters retained."""
from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def live_parser(line: str) -> dict | None:
    if line.startswith("BonsaiProgress "):
        return {"progress": line}
    if line.startswith("BonsaiStop "):
        return {"stop": line}
    return None


def command(config, result_path: Path) -> list[str]:
    evidence = config.output("arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.tests.bench.bench",
        "--checkpoint", config.checkpoint,
        "--compare", "prepared-qmm-metadata", "--fixture", "context-1k",
        "--horizon", "short", "--window", "32", "--rounds", "2",
        "--sampling", "greedy", "--seed", "7",
        "--experiment-id", "bonsai2-production-qmm-metadata-context-1k-256m-v1",
        "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="prepared-qmm-metadata-1k",
    title="Prepared QMM metadata at 1K",
    category="performance",
    explanation=(
        "Measures complete prefill and short greedy decode with one-time "
        "FP32 affine-metadata preparation against stock; paging is measured, not assumed absent."
    ),
    why="Keeps the prefill probe bounded after the 3.3K attempt entered memory pressure.",
    script=command,
    metrics=COMMON_METRICS + (
        "prefill time", "time to first token", "metadata residency",
        "paging counters", "phase progress",
    ),
    controls={
        "control": "stock FP16 scales/biases with MLX per-call promotion",
        "candidate": "one-time FP32 scales/biases preparation",
        "prompt": "context-1k; 32 records; under 1,000 rendered prompt tokens",
        "ordering": "stock/candidate then candidate/stock; two rounds; four total arms",
        "sampling": "greedy; complete token digest required",
        "profile": "interactive-production; 256 MiB allocator cache; actual cancellable prefill callback with a 64-token cap",
        "experiment_id": "bonsai2-production-qmm-metadata-context-1k-256m-v1",
        "scope": "bounded screen only; no 2K, 8K, or long-context extension",
    },
    source="models/bonsai2/backend.py, models/bonsai2/qmm_metadata.py, and models/bonsai2/tests/bench/bench.py",
    promotion=False,
    live_parser=live_parser,
)
