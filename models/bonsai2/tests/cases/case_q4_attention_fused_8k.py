"""Compare fused affine-Q4 attention with the untiled stock control at 8K."""
from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = config.output("arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.tests.bench.bench",
        "--checkpoint", config.checkpoint,
        "--compare", "q4-attention-fused", "--fixture", "context-8k",
        "--horizon", "short", "--window", "32", "--rounds", "3",
        "--sampling", "greedy", "--seed", "7", "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="q4-attention-fused-8k",
    title="Fused affine-Q4 attention at 8K",
    category="performance",
    explanation=(
        "Compares the opt-in fused affine-Q4/G64 attention candidate with "
        "the untiled stock Q4 control at an 8K context."
    ),
    why="It measures the long-context target only after the same paired gate is available at 2K.",
    script=command,
    metrics=COMMON_METRICS + ("prefill rate", "paired two-standard-error band"),
    controls={
        "control": "untiled stock Q4/G64 attention",
        "candidate": "32-token-page fused affine-Q4/G64 attention",
        "cache": "Q4/G64; existing cache tuple and session behavior",
        "tracing": "off in both timing arms",
        "ordering": "three reverse-interleaved rounds; fresh child processes",
        "sampling": "greedy; complete token digest required",
        "scope": "8K measurement; 16K remains deferred until shorter gates pass",
    },
    source="models/bonsai2/tests/bench/bench.py",
    promotion=False,
    status="deferred",
)
