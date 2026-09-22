"""Chunk comparison gated on a completed worker-width diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

from macqwen.testsuite.api import IO_METRICS, TestSpec, bench_module


def _worker_sweep_exists(config) -> bool:
    evidence = config.latest("io-worker-sweep.json")
    if evidence is None:
        return False
    try:
        payload = json.loads(evidence.read_text())
    except (OSError, ValueError):
        return False
    return bool(payload.get("premise", {}).get("topology_eligible"))


def command(config, _result_path: Path) -> list[str]:
    if not _worker_sweep_exists(config):
        raise RuntimeError(
            "premise gate: run the worker-width diagnostic first; "
            "chunk comparison is blocked until worker selection is recorded"
        )
    return [
        str(config.python), "-m", bench_module("flashnext", "bench_production"),
        "--compare", "buffer-chunk2-vs-4", "--tokens", str(config.tokens),
        "--arms", str(config.pairs), "--min-arms", str(config.pairs),
        "--drop", "0",
    ]


def environment(config) -> dict[str, str]:
    evidence = config.latest("io-worker-sweep.json")
    payload = json.loads(evidence.read_text())
    return {"FLASHNEXT_IO_WORKERS": str(payload["selected_workers"])}


TEST = TestSpec(
    id="chunk-after-workers",
    title="Chunk 2 versus chunk 4 after worker selection",
    category="performance",
    explanation=(
        "Compares shared-read chunk sizes two and four after the worker-width "
        "diagnostic records the selected worker count."
    ),
    why=(
        "Chunking changes task grouping and storage concurrency. The worker "
        "premise must be established before this narrower comparison."
    ),
    script=command,
    metrics=IO_METRICS,
    controls={
        "control": "shared buffer, pread chunk 2",
        "candidate": "shared buffer, pread chunk 4",
        "worker count": "selected by completed worker-width diagnostic",
        "digest": "required",
    },
    source="models/flashnext/tests/bench/bench_production.py",
    promotion=False,
    environment=environment,
)
