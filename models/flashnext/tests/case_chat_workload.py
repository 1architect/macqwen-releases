"""Compare our reference prompt with an explicitly supplied everyday prompt."""
from pathlib import Path

from .api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    folder = Path(config.results_dir).expanduser()
    prompt = folder / "chat-workload.txt"
    if not prompt.is_file() or not prompt.read_text().strip():
        raise RuntimeError(f"Provide an everyday prompt in {prompt} before this comparison")
    return [str(config.python), "-m", "models.flashnext.bench_chat_parity",
            "--mode", "workload", "--rounds", "3", "--prompt-file", str(prompt),
            "--json", str(folder / "chat-workload.json")]


TEST = TestSpec(
    id="chat-workload", title="Reference versus everyday prompt",
    category="diagnostic",
    explanation=(
        "Compares photosynthesis with our supplied everyday prompt through "
        "the same chat renderer. Both use 32 greedy tokens and closed thinking."
    ),
    why=(
        "Our terminal comparison only establishes parity on photosynthesis. "
        "Everyday prompts may route to different experts and read more from SSD."
    ),
    script=command, metrics=COMMON_METRICS + ("prompt digest", "VM deltas"),
    controls={
        "horizon": "32 tokens, three reversed pairs, fresh conversations",
        "runtime": "60-slot preset, 16 workers, I/O profiling off, actual renderer",
        "sampler": "greedy, closed thinking, saved effort",
        "pins": "same private snapshot per process",
        "digest": "stable within each prompt; different output across prompts is expected",
        "scope": "prompt-content attribution; no optimization or quality claim",
    },
    source="models/flashnext/bench_chat_parity.py",
    promotion=False,
)
