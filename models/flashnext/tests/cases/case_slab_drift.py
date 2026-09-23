"""Fresh-launch comparison of rolling and frozen normal-chat slab profiles."""
from __future__ import annotations

from pathlib import Path

from macqwen.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(config, _result_path: Path) -> list[str]:
    return [
        str(config.python), "-m", bench_module("flashnext", "bench_slab_drift"),
        "--tokens", "32", "--pairs", "3",
        "--json", str(config.output("slab-drift.json")),
        "--record", str(config.output("slab-drift-arms.jsonl")),
    ]


def environment(_config) -> dict[str, str]:
    return {
        "FLASHNEXT_GPU_KEEPWARM": "0",
        "FLASHNEXT_NORM_WEIGHT_CACHE": "0",
        "FLASHNEXT_SLAB_COUNTS": "turn",
        "FLASHNEXT_PREWARM": "0",
    }


TEST = TestSpec(
    id="slab-drift", title="Normal chat slab drift across launches",
    category="diagnostic",
    explanation=(
        "Runs three reversed pairs of fresh 32-token Vontra launches with "
        "8 pins and a 60-slot skew slab. Each pair uses the same prompt."
    ),
    why="Measures pack churn and whether a fixed profile changes hits, reads, or speed.",
    script=command, environment=environment,
    metrics=COMMON_METRICS + (
        "slab hit rate", "allocation digest", "pack path", "load time", "pin digest",
    ),
    controls={
        "conditions": "private rolling history versus one frozen snapshot",
        "routing": "exact-quality, 8 resident experts",
        "slab": "60 skew slots, Frontier 8A, Up-QMV/SwiGLU",
        "decode": "greedy, exact digest within each pair",
        "ordering": "three reversed pairs, fresh process per arm",
    },
    source="models/flashnext/tests/bench/bench_slab_drift.py",
    promotion=False,
)
