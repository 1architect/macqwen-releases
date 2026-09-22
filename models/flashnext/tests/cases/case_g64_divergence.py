from pathlib import Path

from macqwen.testsuite.api import TestSpec, bench_module


def command(config, result_path: Path) -> list[str]:
    output = str(config.output("g64-divergence"))
    model = config.checkpoint or "${MACQWEN_FLASHNEXT_MODEL}"
    return [
        str(config.python), "-m", bench_module("flashnext", "diagnose_g64_divergence"),
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
    source="models/flashnext/tests/bench/diagnose_g64_divergence.py",
    promotion=False,
)
