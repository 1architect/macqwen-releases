"""Run the shortest Bonsai-2 speed arm with FlashNext's reference question."""
from pathlib import Path

from macqwen.testsuite.spec import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = result_path.with_name(f"{result_path.stem}-arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.bench",
        "--checkpoint", config.checkpoint, "--compare", "baseline",
        "--fixture", "question-only", "--horizon", "short", "--window", "32",
        "--rounds", "3", "--sampling", "greedy", "--seed", "7",
        "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="baseline-question-short",
    title="Bonsai-2 question-only short speed check",
    category="diagnostic",
    explanation=(
        "Runs three fresh greedy Bonsai-2 control arms with FlashNext's "
        "question-only reference prompt and a 32-token horizon."
    ),
    why="Measures the lowest-context live speed without claiming a product-context rate.",
    script=command,
    metrics=COMMON_METRICS,
    controls={
        "prompt": "Explique a fotossintese em duas frases.",
        "context": "question only; empty system content",
        "sampling": "greedy; complete token digest required",
        "horizon": "32 tokens; three fresh control arms",
        "ordering": "fresh processes; no unnecessary warmup",
        "scope": "short workload diagnostic; not a promotion gate",
    },
    source="models/bonsai2/bench.py and models/flashnext/bench_chat_parity.py",
    promotion=False,
)
