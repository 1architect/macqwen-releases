from pathlib import Path

from .api import FLASHNEXT, TestSpec


def command(config, result_path: Path) -> list[str]:
    output = str(result_path or (FLASHNEXT / "results" / "g64-divergence"))
    model = config.checkpoint or "${MACQWEN_FLASHNEXT_MODEL}"
    return [
        str(config.python), str(FLASHNEXT / "diagnose_g64_divergence.py"),
        "--model", model, "--output", output, "--tokens", "16",
    ]


TEST = TestSpec(
    id="g64-divergence",
    title="G64 first divergence diagnostic",
    category="diagnostic",
    explanation="Runs one short teacher-forced shadow decode on one loaded model.",
    why="Locates the first G64 output mismatch after the complete-model digest failure.",
    script=command,
    controls={"reference": "same inputs, routes, and loaded weights", "bound": "32 decode tokens"},
    source="models/flashnext/diagnose_g64_divergence.py",
    promotion=False,
)
