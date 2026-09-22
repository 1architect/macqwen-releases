"""Project-level live-test runner and canonical JSONL writer."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from macqwen.measurement import MeasurementRun, canonical_measurement_path, validate_path
from macqwen.results import ENVIRONMENT_KEY, LOG_NAME
from .api import ROOT, TestContext


ARM_RE = re.compile(r"Arm\s+(\d+)\s*/\s*(\d+):\s+Running\s+(\S+)")
DIGEST_RE = re.compile(r"(?:Digest:|token digest[^:]*:)\s*([0-9a-f]{16,64})", re.I)
_CANCELLATION_CODES = {
    -int(signal.SIGINT), 128 + int(signal.SIGINT),
    -int(signal.SIGTERM), 128 + int(signal.SIGTERM),
}
_STOP_GRACE_SECONDS = 1.0


def _is_cancellation_returncode(returncode: int) -> bool:
    try:
        return int(returncode) in _CANCELLATION_CODES
    except (TypeError, ValueError):
        return False


def _signal_process_group(process, signal_number: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except (OSError, ProcessLookupError, PermissionError):
        pass


def _stop_process_group(process) -> int:
    """Stop the controller and every descendant in its owned process group."""

    _signal_process_group(process, signal.SIGINT)
    try:
        returncode = int(process.wait(timeout=_STOP_GRACE_SECONDS))
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGTERM)
        try:
            returncode = int(process.wait(timeout=_STOP_GRACE_SECONDS))
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            try:
                returncode = int(process.wait(timeout=_STOP_GRACE_SECONDS))
            except subprocess.TimeoutExpired:
                poll = getattr(process, "poll", None)
                value = poll() if callable(poll) else None
                returncode = int(value) if value is not None else 128 + int(signal.SIGKILL)

    # The controller can handle SIGINT and exit while a descendant ignores it.
    # A real Popen has poll(); the small test doubles used by callers may not.
    # Escalate the owned group after controller exit as well, so that a worker
    # cannot outlive a cleanly handled controller cancellation.
    if callable(getattr(process, "poll", None)):
        _signal_process_group(process, signal.SIGTERM)
        _signal_process_group(process, signal.SIGKILL)
    return returncode


@dataclass
class Runner:
    context: TestContext
    output: Any = sys.stdout

    def _path(self, test_id: str) -> Path:
        path = canonical_measurement_path(ROOT, self.context.runtime, test_id)
        while path.exists():
            path = path.with_name(f"{path.stem}-retry{int(time.time())}{path.suffix}")
        return validate_path(path, ROOT, self.context.runtime)

    def run(self, spec) -> dict[str, Any]:
        path = self._path(spec.id)
        # Every artifact of this run goes in the record's folder. The case
        # sees it as context.run_dir, the child as MACQWEN_RESULTS_DIR.
        self.context.run_dir = path.parent
        command = spec.script(self.context, path)
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise TypeError(f"{spec.id} returned an invalid command")
        run = MeasurementRun(
            path, runtime=self.context.runtime, experiment=spec.id,
            metadata={
                "title": spec.title,
                "category": spec.category,
                "checkpoint": self.context.checkpoint,
                "python": self.context.python,
                "controls": spec.controls,
                "source": spec.source,
                "promotion": spec.promotion,
            },
        )
        run.start()
        environment = os.environ.copy()
        environment.update(self.context.canonical_environment)
        if self.context.checkpoint:
            environment["MACQWEN_FLASHNEXT_MODEL"] = self.context.checkpoint
        if getattr(spec, "environment", None) is not None:
            environment.update(spec.environment(self.context))
        environment[ENVIRONMENT_KEY] = str(path.parent)
        log = (path.parent / LOG_NAME).open("a", encoding="utf-8")
        started = time.perf_counter()
        lines: list[str] = []
        live: list[dict[str, Any]] = []
        process = subprocess.Popen(
            command, cwd=str(ROOT), env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        interrupted = False
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
                log.write(raw)
                log.flush()
                if hasattr(spec, "live_parser") and spec.live_parser is not None:
                    parsed = spec.live_parser(line)
                    if parsed:
                        live.append(parsed)
                match = ARM_RE.search(line)
                if match:
                    self.output.write(f"arm {match.group(1)}/{match.group(2)} {match.group(3)}\n")
                else:
                    self.output.write(line + "\n")
                self.output.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            returncode = _stop_process_group(process)
            if process.stdout is not None:
                try:
                    remainder = process.stdout.read()
                except (OSError, ValueError):
                    remainder = ""
                if remainder:
                    lines.extend(remainder.rstrip("\n").splitlines())
            lines.append("interrupted by user")
        except BaseException:
            # A parser, output, or filesystem exception must not strand the
            # command's process group either.  Preserve the original error
            # after the bounded cleanup so callers still see its cause.
            _stop_process_group(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if interrupted:
                log.write("interrupted by user\n")
            log.close()
        interrupted = interrupted or _is_cancellation_returncode(returncode)
        text = "\n".join(lines)
        digest = None
        for line in lines:
            match = DIGEST_RE.search(line)
            if match:
                digest = match.group(1)
        status = "interrupted" if interrupted else "passed" if returncode == 0 else "failed"
        arm_id = f"terminal-{spec.id}"
        run.arm(
            arm_id=arm_id, condition=spec.id, round_index=0, command=command,
            status=status if interrupted else "raw" if lines else "failed",
            metrics={"common": {"returncode": returncode, "elapsed_seconds": time.perf_counter() - started,
                                "token_digest": digest}, "terminal": {"live": live}},
            output=text,
        )
        interpretation = "Interrupted by user; partial evidence was preserved." if interrupted else None
        if not interrupted and getattr(spec, "interpret", None) is not None:
            interpretation = spec.interpret(returncode, text, live)
        if interpretation is None:
            interpretation = "Completed." if returncode == 0 else "Invalid run. The process failed."
        run.validation(arm_id, status, returncode=returncode,
                       interpretation=interpretation, interrupted=interrupted)
        run.finish("interrupted" if interrupted else "completed" if returncode == 0 else "completed_with_failures",
                   returncode=returncode, interpretation=interpretation,
                   interrupted=interrupted)
        return {"path": path, "returncode": returncode,
                "interpretation": interpretation, "interrupted": interrupted}
