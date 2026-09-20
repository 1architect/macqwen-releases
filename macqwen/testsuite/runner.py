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
from .api import ROOT, TestContext


ARM_RE = re.compile(r"Arm\s+(\d+)\s*/\s*(\d+):\s+Running\s+(\S+)")
DIGEST_RE = re.compile(r"(?:Digest:|token digest[^:]*:)\s*([0-9a-f]{16,64})", re.I)


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
        started = time.perf_counter()
        lines: list[str] = []
        live: list[dict[str, Any]] = []
        process = subprocess.Popen(
            command, cwd=str(ROOT), env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
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
            os.killpg(process.pid, signal.SIGINT)
            returncode = process.wait()
            lines.append("interrupted by user")
        finally:
            if process.stdout is not None:
                process.stdout.close()
        text = "\n".join(lines)
        digest = None
        for line in lines:
            match = DIGEST_RE.search(line)
            if match:
                digest = match.group(1)
        status = "passed" if returncode == 0 else "failed"
        arm_id = f"terminal-{spec.id}"
        run.arm(
            arm_id=arm_id, condition=spec.id, round_index=0, command=command,
            status="raw" if lines else "failed",
            metrics={"common": {"returncode": returncode, "elapsed_seconds": time.perf_counter() - started,
                                "token_digest": digest}, "terminal": {"live": live}},
            output=text,
        )
        interpretation = None
        if getattr(spec, "interpret", None) is not None:
            interpretation = spec.interpret(returncode, text, live)
        if interpretation is None:
            interpretation = "Completed." if returncode == 0 else "Invalid run. The process failed."
        run.validation(arm_id, status, returncode=returncode, interpretation=interpretation)
        run.finish("completed" if returncode == 0 else "completed_with_failures",
                   returncode=returncode, interpretation=interpretation)
        return {"path": path, "returncode": returncode, "interpretation": interpretation}
