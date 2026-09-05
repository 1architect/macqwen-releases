"""Compare broad expert pinning with a smaller set while retaining the slab."""
from pathlib import Path

from .api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    folder = Path(config.results_dir).expanduser()
    prompt = folder / "chat-workload.txt"
    if not prompt.is_file() or not prompt.read_text().strip():
        raise RuntimeError(f"Provide the everyday prompt in {prompt}")
    return [str(config.python), "-m", "models.flashnext.bench_chat_parity",
            "--mode", "pins", "--rounds", str(max(3, config.pairs)),
            "--prompt-file", str(prompt), "--json", str(folder / "chat-pin-budget.json")]


TEST = TestSpec(
    id="chat-pin-budget", title="32 versus 8 pinned experts with the 60-slot slab",
    category="performance",
    explanation=(
        "Compares the current 32-expert pin set with eight experts per layer. "
        "Both keep the same 60-slot slab and generate the same 32 tokens."
    ),
    why=(
        "Our pin diagnostic locks 4.65 GB and records compression and swap "
        "during cold pinning. A smaller pin set may relieve memory pressure."
    ),
    script=command,
    metrics=COMMON_METRICS + ("actual pinned MB", "VM deltas", "effective pin count"),
    controls={
        "control": "32 pinned experts per layer",
        "candidate": "8 pinned experts per layer",
        "runtime": "60-slot slab, 16 workers, exact-quality routing, both profilers off",
        "workload": "same everyday prompt, 32 greedy tokens, closed thinking, renderer on",
        "order": "reversed pairs; identical private initial pin profiles",
        "digest": "identical prompt, output, and slab allocation required",
    },
    source="models/flashnext/bench_chat_parity.py",
    promotion=False,
)
