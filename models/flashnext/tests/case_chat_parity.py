"""Compare the short benchmark workload with the real plain-chat path."""
from pathlib import Path

from .api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    evidence = Path(config.results_dir).expanduser() / "chat-parity.json"
    return [str(config.python), "-m", "models.flashnext.bench_chat_parity",
            "--rounds", "3", "--json", str(evidence)]


TEST = TestSpec(
    id="chat-parity", title="32-token chat versus benchmark attribution",
    category="diagnostic",
    explanation=(
        "Runs raw-prompt silent decode, formatted-chat silent decode, and "
        "formatted chat with its actual terminal renderer. All start through chat.sh."
    ),
    why=(
        "The chat is slower than the short benchmark. This separates rendering "
        "from prompt changes before treating generation length as the cause."
    ),
    script=command,
    metrics=COMMON_METRICS + ("prompt digest", "callback and wall time", "VM deltas"),
    controls={
        "horizon": "32 tokens in every arm; three reversed rounds",
        "runtime": "current 60-slot preset, 16 workers, I/O profiling off",
        "sampling": "greedy; closed thinking block in every arm",
        "rendering pair": "identical formatted prompt IDs and generated token IDs required",
        "raw versus formatted": "different workload; no same-trajectory speed claim",
        "pins": "same private snapshot per process",
        "terminal": "PTY keeps the actual renderer active under suite capture",
    },
    source="models/flashnext/bench_chat_parity.py and macqwen/session.py",
    promotion=False,
)
