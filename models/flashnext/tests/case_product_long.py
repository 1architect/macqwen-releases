"""Long-generation product comparison through the real chat launcher."""
from __future__ import annotations

import json
from pathlib import Path

from .api import FLASHNEXT, TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [
        str(config.python), str(FLASHNEXT / "bench_product_long.py"),
        "--tokens", "256", "--window", "32",
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
                f"phase {record.get('phase', '?')}"
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
    if len(records) < 2:
        return "Incomplete run. Both startup paths must produce a completion record."
    digests = {row.get("digest") for row in records if row.get("digest")}
    if len(digests) > 1:
        return "Observed result: the startup paths produced different token digests."
    return (
        "Observed result: both startup paths completed. Compare every 32-token "
        "window for rate decay, physical I/O, slab hits, memory, context, and phase. "
        "This product horizon has no 32-token baseline or promotion decision."
    )


TEST = TestSpec(
    id="product-long-256",
    title="256-token product generation windows",
    category="performance",
    explanation=(
        "Starts two real chat.sh processes and reports metrics after every "
        "32-token window of a 256-token generation."
    ),
    why=(
        "The existing 32-token benchmark cannot show route-locality decay. "
        "This separate product horizon compares normal startup with canonical "
        "60-slot Frontier 8A startup."
    ),
    script=command,
    metrics=(
        "generation rate per 32-token window",
        "physical MB/token per window",
        "slab hit rate per window",
        "active memory per window",
        "context size per window",
        "thinking and answer phase counts",
        "token digest per startup path",
    ),
    controls={
        "horizon": "256 generated tokens",
        "window": "32 generated tokens",
        "normal": "chat.sh startup with inherited environment",
        "canonical": "60-slot skew slab pack + Frontier 8A controls",
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
