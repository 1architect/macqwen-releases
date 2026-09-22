"""Compare broad and narrow expert pinning on the REAP reference path."""
from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    prompt = config.input("chat-workload.txt")
    if not prompt.is_file() or not prompt.read_text().strip():
        raise RuntimeError(f"Provide the everyday prompt in {prompt}")
    return [str(config.python), "-m", "models.flashnext.tests.bench.bench_chat_parity",
            "--mode", "pins", "--rounds", str(max(3, config.pairs)),
            "--prompt-file", str(prompt), "--json", str(config.output("chat-pin-budget.json"))]


TEST = TestSpec(
    id="chat-pin-budget", title="REAP 32 versus 8 pinned experts",
    category="performance",
    explanation=(
        "Compares 32 pinned experts with eight experts per layer on the same "
        "generic MLX REAP path. Packed G64 residency stays off."
    ),
    why=(
        "Recovered REAP records show about 3.95 to 4.06 GB pinned. The pin "
        "operation itself is only 0.11 to 0.18 seconds, so a useful result "
        "must improve memory allocation, physical reads, or sustained decode."
    ),
    script=command,
    metrics=COMMON_METRICS + ("actual pinned MB", "VM deltas", "effective pin count"),
    controls={
        "control": "32 pinned experts per layer",
        "candidate": "8 pinned experts per layer",
        "runtime": "generic MLX Q4/G64, packed residency off, 16 workers",
        "workload": "same everyday prompt, 32 greedy tokens, closed thinking, renderer on",
        "order": "reversed pairs; identical private initial pin profiles",
        "digest": "identical prompt and output required; routes and scores unchanged",
        "decision": "adopt only with lower physical reads or better sustained throughput",
    },
    source="models/flashnext/tests/bench/bench_chat_parity.py",
    promotion=False,
)
