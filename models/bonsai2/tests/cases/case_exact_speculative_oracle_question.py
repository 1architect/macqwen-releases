"""Measure the greedy perfect-draft verifier ceiling on the short question."""
from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = config.output("arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.tests.bench.bench",
        "--checkpoint", config.checkpoint,
        "--compare", "exact-speculative-oracle-4",
        "--fixture", "question-only", "--horizon", "short", "--window", "32",
        "--rounds", "3", "--sampling", "greedy", "--seed", "7",
        "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="exact-speculative-oracle-question",
    title="Exact speculative verifier ceiling on question-only decode",
    category="diagnostic",
    explanation=(
        "Uses a prior target transcript as a labelled perfect draft and "
        "measures target verification plus recurrent/KV rollback costs."
    ),
    why="Establishes the ceiling before any compatible draft model is considered.",
    script=command,
    metrics=COMMON_METRICS + ("verification blocks", "accepted tokens", "verify/commit time"),
    controls={
        "control": "ordinary greedy target decode",
        "candidate": "greedy exact verifier with block size 4",
        "draft": "perfect target transcript; diagnostic only",
        "prompt": "question-only; 23 rendered prompt tokens",
        "ordering": "three reverse-interleaved rounds; fresh processes",
        "scope": "not achieved speculative decoding or a promotion gate",
    },
    source="models/bonsai2/backend.py and models/bonsai2/tests/bench/bench.py",
    promotion=False,
)
