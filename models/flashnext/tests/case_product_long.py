"""Long-generation product comparison through the real chat launcher."""
from __future__ import annotations

import json
from pathlib import Path

from .api import FLASHNEXT, TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [
        str(config.python), str(FLASHNEXT / "bench_product_long.py"),
        "--tokens", "256", "--window", "32",
        "--rounds", "1", "--paths", "current", "--phase", "answer",
    ]


def live_parser(line: str) -> dict | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if record.get("type") == "window":
        return {
            "arm": record.get("path", "path"),
            "progress": (
                f"window {record.get('window', '?')}  "
                f"ctx {record.get('context_tokens', 0)}  "
                f"phase {record.get('phase', '?')}  "
                f"think {record.get('thinking_tokens', 0)}  "
                f"answer {record.get('answer_tokens', 0)}"
            ),
            "gen": record.get("generation_tps"),
            "physical": record.get("physical_mb_token"),
            "active": record.get("active_mb"),
            "hit": record.get("slab_hit_pct"),
            "complete": True,
        }
    return None


def interpret(returncode: int, output: str, _arms: list[dict]) -> str:
    if returncode:
        return "Invalid run. A chat.sh startup path failed."
    records = []
    for line in output.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "complete":
            records.append(row)
    if not records:
        return "Incomplete run. No long-answer path completed."
    if len(records) > 1:
        digests = {row.get("digest") for row in records if row.get("digest")}
        if len(digests) > 1:
            return "Observed result: the long-answer paths produced different token digests."
        return (
            "Observed result: the single directional long-answer pair completed. "
            "Inspect every 32-token window. This pair cannot promote a change."
        )
    return (
        "Observed result: the single long-answer validation completed. Inspect "
        "every 32-token window. This run cannot promote a performance change."
    )


TEST = TestSpec(
    id="product-long-answer",
    title="Single 256-token answer validation",
    category="performance",
    explanation=(
        "Starts one current-runtime chat process after a closed thinking block. "
        "It reports metrics after every 32 answer tokens."
    ),
    why=(
        "The 32-token comparisons select candidates. This single long answer "
        "checks sustained behavior without repeating the workload."
    ),
    script=command,
    metrics=(
        "generation rate per 32-token window",
        "physical MB/token per window",
        "slab hit rate per window",
        "active memory per window",
        "context size per window",
        "answer phase count",
        "token digest",
    ),
    controls={
        "horizon": "256 generated tokens",
        "window": "32 generated tokens",
        "path": "current 60-slot Frontier 8A plus Up-QMV/SwiGLU",
        "phase": "answer from a closed thinking block",
        "rounds": "one; long runs never provide promotion statistics",
        "decode": "greedy",
        "digest": "required",
        "quality": "user via chat.sh, sampling and xhigh",
    },
    source="models/flashnext/bench_product_long.py and chat.sh",
    promotion=False,
    live_parser=live_parser,
    interpret=interpret,
    canonical=False,
)
