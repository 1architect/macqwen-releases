from pathlib import Path

from macqwen.testsuite.spec import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    log = Path("/tmp") / f"macqwen-qwen27b-{result_path.stem}.log"
    return [
        config.python, "models/qwen27b/bench_decode.py", "--model", config.checkpoint,
        "--decode-tokens", str(config.tokens), "--log", str(log),
    ]


TEST = TestSpec(
    id="decode-context-ladder", title="Qwen27B decode context ladder", category="diagnostic",
    explanation="Runs the retained Qwen27B context/decode ladder.",
    why="Provides a project-level entry point for the existing live Qwen27B probe.",
    script=command, metrics=COMMON_METRICS,
    controls={"sampling": "runtime default", "digest": "diagnostic", "horizon": "configured"},
    source="models/qwen27b/bench_decode.py", promotion=False,
)
