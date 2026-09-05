"""Locate VM movement during the file-backed expert pin operation."""
from pathlib import Path

from .api import TestSpec
from .case_chat_workload import command as workload_command


def command(config, result_path: Path) -> list[str]:
    result = workload_command(config, result_path)
    result[result.index("--json") + 1] = str(
        Path(config.results_dir).expanduser() / "chat-pin-memory.json"
    )
    return result + ["--profile-pins"]


TEST = TestSpec(
    id="chat-pin-memory", title="Chat expert-pin memory attribution",
    category="diagnostic",
    explanation=(
        "Records the file-backed expert pin size, pin duration, physical reads, "
        "and VM movement inside the pin operation."
    ),
    why=(
        "Every workload arm showed swap activity. MLX active memory omits "
        "the separate file-backed expert pins. We need their measured cost."
    ),
    script=command,
    metrics=("pin time", "pin physical reads", "pin VM deltas", "pinned MB",
             "MLX active and cached MB", "generation and tail time", "token digests"),
    controls={
        "workload": "same two prompts, 32 tokens, three reversed pairs",
        "runtime": "unchanged 60-slot preset and 16 workers",
        "profile": "pin boundaries only; per-read I/O profiling off",
        "scope": "diagnostic overhead recorded; no performance promotion",
    },
    source="models/flashnext/bench_chat_parity.py",
    promotion=False,
)
