"""Shared durable records for live model measurements."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile
import uuid
from typing import Any, Iterable


SCHEMA = 1
RECORD_TYPES = {"run", "arm", "validation", "failure", "summary", "legacy_payload"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return uuid.uuid4().hex


def canonical_measurement_path(root: Path, runtime: str, experiment: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "-".join(part for part in experiment.strip().lower().replace("_", "-").split() if part)
    slug = slug or "run"
    return root / "docs" / runtime / "measurements" / f"{stamp}-{slug}.jsonl"


def validate_path(path: Path, root: Path, runtime: str) -> Path:
    path = path.expanduser().resolve()
    expected = (root / "docs" / runtime / "measurements").resolve()
    try:
        path.relative_to(expected)
    except ValueError as exc:
        raise ValueError(
            f"measurement must be stored under {expected}; got {path}"
        ) from exc
    if path.suffix != ".jsonl":
        raise ValueError("canonical measurements must use the .jsonl extension")
    return path


def append_record(path: Path, record: dict[str, Any]) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError(f"record schema must be {SCHEMA}")
    if record.get("type") not in RECORD_TYPES:
        raise ValueError(f"unknown measurement record type: {record.get('type')!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MeasurementRun:
    """Append-only lifecycle writer used by every live test adapter."""

    def __init__(self, path: Path, *, runtime: str, experiment: str,
                 metadata: dict[str, Any] | None = None,
                 run: str | None = None):
        self.path = path
        self.runtime = runtime
        self.experiment = experiment
        self.run = run or run_id()
        self.metadata = metadata or {}
        self.started = False

    def start(self) -> None:
        append_record(self.path, {
            "schema": SCHEMA, "type": "run", "run_id": self.run,
            "runtime": self.runtime, "experiment": self.experiment,
            "status": "started", "started_at": now_iso(),
            "metadata": self.metadata,
        })
        self.started = True

    def arm(self, *, arm_id: str, condition: str, round_index: int,
            command: list[str], status: str = "raw",
            metrics: dict[str, Any] | None = None, **values: Any) -> None:
        append_record(self.path, {
            "schema": SCHEMA, "type": "arm", "run_id": self.run,
            "runtime": self.runtime, "experiment": self.experiment,
            "arm_id": arm_id, "condition": condition,
            "round": round_index, "status": status,
            "command": command, "recorded_at": now_iso(),
            "metrics": metrics or {}, **values,
        })

    def validation(self, arm_id: str, status: str, **checks: Any) -> None:
        append_record(self.path, {
            "schema": SCHEMA, "type": "validation", "run_id": self.run,
            "arm_id": arm_id, "status": status, "checks": checks,
            "recorded_at": now_iso(),
        })

    def failure(self, message: str, *, arm_id: str | None = None,
                error_type: str = "failure", **details: Any) -> None:
        append_record(self.path, {
            "schema": SCHEMA, "type": "failure", "run_id": self.run,
            "runtime": self.runtime, "experiment": self.experiment,
            "arm_id": arm_id, "status": "failed", "error_type": error_type,
            "message": message, "details": details, "recorded_at": now_iso(),
        })

    def finish(self, status: str = "completed", **summary: Any) -> None:
        append_record(self.path, {
            "schema": SCHEMA, "type": "summary", "run_id": self.run,
            "runtime": self.runtime, "experiment": self.experiment,
            "status": status, "summary": summary, "recorded_at": now_iso(),
        })


def migrate_legacy(path: Path, destination: Path, *, runtime: str,
                   experiment: str) -> None:
    """Wrap an old JSON artifact without fabricating missing provenance."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=f".{destination.name}.",
                                      dir=destination.parent)[1])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        writer = MeasurementRun(destination, runtime=runtime, experiment=experiment,
                                metadata={
                                    "legacy": True,
                                    "source_path": str(path),
                                    "source_sha256": source_hash(path),
                                    "provenance": "unknown where absent",
                                })
        writer.path = temporary
        writer.start()
        append_record(temporary, {
            "schema": SCHEMA, "type": "legacy_payload", "run_id": writer.run,
            "runtime": runtime, "experiment": experiment,
            "source_path": str(path), "payload": payload,
        })
        writer.finish("legacy", source_path=str(path))
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
