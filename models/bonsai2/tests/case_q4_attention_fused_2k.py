"""Screen fused affine-Q4 attention with bounded prefill context."""
from pathlib import Path

from macqwen.testsuite.spec import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = result_path.with_name(f"{result_path.stem}-arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.bench",
        "--checkpoint", config.checkpoint,
        "--compare", "q4-attention-fused", "--fixture", "context-1k",
        "--horizon", "short", "--window", "32", "--rounds", "2",
        "--sampling", "greedy", "--seed", "7", "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="q4-attention-fused-1k",
    title="Fused affine-Q4 attention at 1K",
    category="performance",
    explanation=(
        "Compares the opt-in fused affine-Q4/G64 attention candidate with "
        "the untiled stock Q4 control at a bounded 1K context."
    ),
    why="Screens the candidate while retaining VM evidence for prefill pressure.",
    script=command,
    metrics=COMMON_METRICS + ("prefill rate", "paired two-standard-error band"),
    controls={
        "control": "untiled stock Q4/G64 attention",
        "candidate": "32-token-page fused affine-Q4/G64 attention",
        "cache": "Q4/G64; existing cache tuple and session behavior",
        "tracing": "off in both timing arms",
        "ordering": "two reverse-interleaved rounds; four total arms; fresh child processes",
        "sampling": "greedy; complete token digest required",
        "scope": "1K screen; no promotion from an unresolved exactness result",
    },
    source="models/bonsai2/bench.py",
    promotion=False,
)
