from pathlib import Path

from macqwen.testsuite.spec import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    scratch = Path("/tmp") / f"macqwen-bonsai2-{result_path.stem}.jsonl"
    return [
        config.python, "-m", "models.bonsai2.bench", "--checkpoint", config.checkpoint,
        "--compare", "baseline", "--fixture", "context-2k", "--horizon", "short",
        "--window", "32", "--rounds", "3", "--sampling", "greedy",
        "--seed", "7", "--jsonl", str(scratch),
    ]


TEST = TestSpec(
    id="baseline-short", title="Bonsai-2 short baseline", category="performance",
    explanation="Runs the retained 32-token greedy Bonsai-2 control.",
    why="Provides a canonical live control for the selected Bonsai-2 checkpoint.",
    script=command, metrics=COMMON_METRICS,
    controls={"sampling": "greedy", "digest": "required", "horizon": "32 tokens"},
    source="models/bonsai2/bench.py", promotion=True,
)
