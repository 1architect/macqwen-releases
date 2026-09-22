"""Check sustained greedy verifier behavior without increasing context."""
from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = config.output("arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.tests.bench.bench",
        "--checkpoint", config.checkpoint,
        "--compare", "exact-speculative-oracle-4",
        "--fixture", "question-only", "--horizon", "tg128", "--window", "32",
        "--rounds", "3", "--sampling", "greedy", "--seed", "7",
        "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="exact-speculative-oracle-question-longer",
    title="Exact speculative verifier sustained decode",
    category="diagnostic",
    explanation=(
        "Repeats the question-only target with a 128-token output horizon "
        "to expose sustained verification and commit costs."
    ),
    why="A matching 32-token prefix alone is not sufficient evidence for decode promotion.",
    script=command,
    metrics=COMMON_METRICS + ("verification blocks", "accepted tokens", "verify/commit time"),
    controls={
        "control": "ordinary greedy target decode",
        "candidate": "greedy exact verifier with block size 4",
        "draft": "perfect target transcript; diagnostic only",
        "context": "question-only; no materially long-context run",
        "horizon": "128 generated tokens",
        "ordering": "three reverse-interleaved rounds; fresh processes",
        "scope": "not achieved speculative decoding or a promotion gate",
    },
    source="models/bonsai2/backend.py and models/bonsai2/tests/bench/bench.py",
    promotion=False,
)
