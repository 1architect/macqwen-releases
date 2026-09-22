from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    scratch = config.output("arms.jsonl")
    return [
        config.python, "-m", "models.k2_horizon.tests.bench.bench", "--checkpoint", config.checkpoint,
        "--compare", "baseline", "--fixture", "context-2k", "--horizon", "short",
        "--window", "32", "--rounds", "3", "--sampling", "greedy",
        "--seed", "7", "--jsonl", str(scratch),
    ]


TEST = TestSpec(
    id="baseline-short", title="K2-Horizon short baseline", category="performance",
    explanation="Runs the retained 32-token greedy K2-Horizon control.",
    why="Provides a canonical live control for the selected K2-Horizon checkpoint.",
    script=command, metrics=COMMON_METRICS,
    controls={"sampling": "greedy", "digest": "required", "horizon": "32 tokens"},
    source="models/k2_horizon/tests/bench/bench.py", promotion=True,
)
