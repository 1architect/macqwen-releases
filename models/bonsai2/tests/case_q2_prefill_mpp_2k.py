"""Screen the opt-in packed-Q2 prefill candidate with paging counters retained."""
from pathlib import Path

from macqwen.testsuite.spec import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = result_path.with_name(f"{result_path.stem}-arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.bench",
        "--checkpoint", config.checkpoint, "--compare", "q2-prefill-mpp",
        "--fixture", "context-1k", "--horizon", "short", "--window", "32",
        "--rounds", "2", "--sampling", "greedy", "--seed", "7",
        "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="q2-prefill-mpp-1k",
    title="Opt-in packed-Q2 prefill at 1K",
    category="diagnostic",
    explanation=(
        "Runs the temporary-staged Q2/G128 MPP candidate through the real "
        "Packed projection path against stock prefill."
    ),
    why="Screens complete prefill while retaining VM evidence for any paging that occurs.",
    script=command,
    metrics=COMMON_METRICS + ("prefill wall time", "paired two-standard-error band"),
    controls={
        "control": "stock mx.quantized_matmul Q2/G128",
        "candidate": "temporary packed-Q2 MPP affine regrouping",
        "fixture": "context-1k (32 records; under 1,000 rendered prompt tokens)",
        "sampling": "greedy; complete digest required for promotion",
        "ordering": "two reverse-interleaved rounds; four total arms; fresh processes",
        "scope": "diagnostic until exactness and paired speed gates pass",
    },
    source="models/bonsai2/bench.py and models/bonsai2/q2_kernel.py",
    promotion=False,
)
