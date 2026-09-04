from pathlib import Path

from .api import FLASHNEXT, TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", "unittest", "discover", "-s", str(FLASHNEXT), "-p", "test_*.py", "-q"]


TEST = TestSpec(
    id="verification-unit", title="FlashNext unit and integration suite", category="verification",
    explanation="Runs the checkpoint-free FlashNext unittest discovery suite.",
    why="It detects runtime, routing, Metal contract, session, and benchmark regressions before performance work is trusted.",
    script=command, metrics=("tests run", "failures", "errors", "elapsed time"),
    controls={"model generation": "off", "quality": "not evaluated"},
    source="models/flashnext/test_*.py", canonical=False,
)
