from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    log = config.output("decode.log")
    return [
        config.python, "-m", "models.qwen27b.tests.bench.bench_decode", "--model", config.checkpoint,
        "--decode-tokens", str(config.tokens), "--log", str(log),
    ]


TEST = TestSpec(
    id="decode-context-ladder", title="Qwen27B decode context ladder", category="diagnostic",
    explanation="Runs the retained Qwen27B context/decode ladder.",
    why="Provides a project-level entry point for the existing live Qwen27B probe.",
    script=command, metrics=COMMON_METRICS,
    controls={"sampling": "runtime default", "digest": "diagnostic", "horizon": "configured"},
    source="models/qwen27b/tests/bench/bench_decode.py", promotion=False,
)
