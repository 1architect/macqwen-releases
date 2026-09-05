"""Attribute the chat workload gap to sampling and thinking at 32 tokens."""
from pathlib import Path

from .api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = Path(config.results_dir).expanduser() / "chat-settings.json"
    prompt = Path(config.results_dir).expanduser() / "chat-workload.txt"
    if not prompt.is_file() or not prompt.read_text().strip():
        raise RuntimeError(f"Provide an everyday prompt in {prompt} before this comparison")
    return [str(config.python), "-m", "models.flashnext.bench_chat_parity",
            "--mode", "settings", "--rounds", "3",
            "--prompt-file", str(prompt), "--json", str(evidence)]


TEST = TestSpec(
    id="chat-settings", title="32-token sampling and thinking attribution",
    category="diagnostic",
    explanation=(
        "Runs four real-chat conditions: greedy answer, sampled answer, greedy "
        "thinking, and sampled thinking. All use a 32-token total limit."
    ),
    why=(
        "Our short greedy chat matches benchmark speed. This restores normal "
        "sampling and thinking separately without changing generation length."
    ),
    script=command,
    metrics=COMMON_METRICS + ("thinking and answer token counts", "VM deltas"),
    controls={
        "horizon": "32 total tokens, three reversed rounds",
        "runtime": "60-slot preset, 16 workers, I/O profiling off, actual renderer",
        "sampler": "greedy or saved sampling settings; fixed seed 42",
        "thinking": "closed or open thinking block; saved effort",
        "digest": "stable within each condition; output differences across settings are expected",
        "pins": "same private snapshot per process",
        "scope": "workload attribution; no quality test or optimization promotion",
    },
    source="models/flashnext/bench_chat_parity.py and macqwen/session.py",
    promotion=False,
)
