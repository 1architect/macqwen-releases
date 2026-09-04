from pathlib import Path
from .api import FLASHNEXT, TestSpec


def command(config, result_path: Path) -> list[str]:
    capture = result_path.with_suffix(".gputrace")
    return [str(config.python), str(FLASHNEXT / "capture_dispatches.py"), "--out", str(capture), "--tokens", "2"]


def environment(_config) -> dict[str, str]:
    return {"MTL_CAPTURE_ENABLED": "1"}


TEST = TestSpec(
    id="dispatch-capture", title="Metal dispatch capture", category="diagnostic",
    explanation="Captures two steady decode tokens for Xcode GPU dispatch inspection.",
    why="It identifies launch-bound copies, reductions, elementwise work, and matrix kernels inside command buffers.",
    script=command, environment=environment,
    metrics=("dispatch count", "command buffers", "kernel groups", "capture path"),
    controls={"captured tokens": "2", "timing": "diagnostic only"},
    source="models/flashnext/capture_dispatches.py",
)
