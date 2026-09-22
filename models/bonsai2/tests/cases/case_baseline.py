from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = config.output("arms.jsonl")
    return [
        config.python, "-m", "models.bonsai2.tests.bench.bench", "--checkpoint", config.checkpoint,
        "--compare", "baseline", "--fixture", "context-1k", "--horizon", "short",
        "--window", "32", "--rounds", "2", "--sampling", "greedy",
        "--seed", "7", "--experiment-id", "baseline-short-context-1k-v2",
        "--jsonl", str(evidence),
    ]


TEST = TestSpec(
    id="baseline-short", title="Bonsai-2 bounded short baseline", category="performance",
    explanation="Runs a bounded 32-token greedy Bonsai-2 control and records paging counters.",
    why="Provides a quick live control for the selected Bonsai-2 checkpoint.",
    script=command, metrics=COMMON_METRICS,
    controls={"identity": "baseline-short-context-1k-v2", "sampling": "greedy",
              "digest": "required", "horizon": "32 tokens",
              "fixture": "context-1k; two reverse-interleaved rounds; two total arms"},
    source="models/bonsai2/tests/bench/bench.py", promotion=False,
)
