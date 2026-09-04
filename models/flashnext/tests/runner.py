"""Subprocess runner with live arm metrics and result interpretation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from macqwen.ui import C

from .api import ROOT, TestSpec


@dataclass
class SuiteConfig:
    python: str = sys.executable
    checkpoint: str = ""
    tokens: int = 32
    pairs: int = 6
    workers: int = 16
    purge_file_cache: bool = False
    settle_seconds: float = 0.0
    settle_timeout: float = 900.0
    max_load: float = 2.5
    max_compressor_rate: float = 128.0
    results_dir: str = str(Path(__file__).resolve().parent / "results")

    @property
    def canonical_environment(self) -> dict[str, str]:
        return {
            "FLASHNEXT_IO_WORKERS": str(self.workers),
            "FLASHNEXT_METAL_RUNTIME": "1",
            "FLASHNEXT_SLAB": "0",
            "FLASHNEXT_SLAB_GLOBAL": "60",
            "FLASHNEXT_SLAB_PACK": "1",
            "FLASHNEXT_SLAB_POLICY": "skew",
            "FLASHNEXT_SLAB_MIN_SLOTS": "4",
            "FLASHNEXT_SLAB_MAX_SLOTS": "6",
            "FLASHNEXT_SLAB_NUM_LAYERS": "12",
            "FLASHNEXT_FUSED_SHARED": "1",
            "FLASHNEXT_FUSED_SHARED_PARTS": "0",
            "FLASHNEXT_FUSED_UP_SWIGLU": "0",
            "FLASHNEXT_STREAM_PACK": "0",
            "FLASHNEXT_PROFILE_BOUNDARIES": "0",
            "FLASHNEXT_PROFILE_SCORE_SYNC": "0",
        }


@dataclass
class LiveState:
    arm: str = "waiting"
    progress: str = ""
    gen: float | None = None
    tail: float | None = None
    physical: float | None = None
    active: float | None = None
    io_wait: float | None = None
    hit: float | None = None
    digest: str = ""

    def line(self) -> str:
        values = [self.arm]
        if self.progress:
            values.append(self.progress)
        if self.gen is not None:
            values.append(f"gen {self.gen:.2f} tok/s")
        if self.tail is not None:
            values.append(f"tail {self.tail:.2f}")
        if self.physical is not None:
            values.append(f"phys {self.physical:.1f} MB/tok")
        if self.io_wait is not None:
            values.append(f"I/O {self.io_wait:.1f} ms/tok")
        if self.active is not None:
            values.append(f"active {self.active:.0f} MB")
        if self.hit is not None:
            values.append(f"hits {self.hit:.1f}%")
        return "  ".join(values)


@dataclass
class RunRecord:
    test_id: str
    title: str
    started_at: str
    command: list[str]
    controls: dict[str, str]
    returncode: int
    elapsed_seconds: float
    arms: list[dict] = field(default_factory=list)
    interpretation: str = ""
    output: str = ""


ARM_RE = re.compile(r"Arm\s+(\d+)\s*/\s*(\d+):\s+Running\s+(\S+)")
SLAB_METRIC_RE = re.compile(
    r"Gen:\s*([\d.]+).*?Tail:\s*([\d.]+).*?Phys:\s*([\d.]+).*?"
    r"Active:\s*([\d.]+).*?Hits:\s*([\d.]+)%.*?IO wait:\s*([\d.]+)",
)
PRODUCTION_RE = re.compile(
    r"^\s*(\S+)\s+gen median\s+([\d.]+).*?tail\s+([\d.]+)\s+([\d.]+) MB/tok",
)
CONTEXT_RE = re.compile(
    r"round\s+(\d+)\s+ctx\s+(\d+)\s+([\d.]+) tok/s\s+([\d.]+) MB/token",
)
DIGEST_RE = re.compile(r"(?:Digest:|token digest[^:]*:)\s*([0-9a-f]{16,64})", re.I)
BAND_RE = re.compile(r"(?:band[^:]*:|above)\s*([\d.]+)\s*(?:percent|%)", re.I)


class LiveMetrics:
    def __init__(self, output=None, parser=None):
        self.output = output or sys.stdout
        self.parser = parser
        self.state = LiveState()
        self.arms: list[dict] = []
        self._current: dict = {}
        self._live = False

    def _render(self) -> None:
        if not self.output.isatty():
            return
        self._live = True
        self.output.write(f"\r{C['clear']}{C['c']}{self.state.line()}{C['0']}\033[K")
        self.output.flush()

    def finish_line(self) -> None:
        if self._live:
            self.output.write("\n")
            self.output.flush()
            self._live = False

    def feed(self, line: str) -> bool:
        custom = self.parser(line) if self.parser is not None else None
        if custom:
            for name in ("arm", "progress", "gen", "tail", "physical", "active", "io_wait", "hit", "digest"):
                if name in custom:
                    setattr(self.state, name, custom[name])
            self._current.update(custom)
            if custom.get("complete"):
                self.arms.append(dict(self._current))
                self._current = {}
            self._render()
            return True
        arm = ARM_RE.search(line)
        if arm:
            if self._current:
                self.arms.append(dict(self._current))
            self._current = {"index": int(arm.group(1)), "total": int(arm.group(2)), "condition": arm.group(3)}
            self.state = LiveState(arm=arm.group(3), progress=f"arm {arm.group(1)}/{arm.group(2)}")
            self._render()
            return True
        metric = SLAB_METRIC_RE.search(line)
        if metric:
            values = [float(item) for item in metric.groups()]
            self.state.gen, self.state.tail, self.state.physical, self.state.active, self.state.hit, self.state.io_wait = values
            self._current.update({
                "gen": values[0], "tail": values[1], "physical_mb_token": values[2],
                "active_mb": values[3], "hit_percent": values[4], "io_wait_ms_token": values[5],
            })
            digest = DIGEST_RE.search(line)
            if digest:
                self.state.digest = digest.group(1)
                self._current["digest"] = digest.group(1)
            self._render()
            return True
        production = PRODUCTION_RE.search(line)
        if production:
            condition, gen, tail, physical = production.groups()
            self.state = LiveState(
                arm=condition, gen=float(gen), tail=float(tail), physical=float(physical)
            )
            self.arms.append({
                "condition": condition, "gen": float(gen), "tail": float(tail),
                "physical_mb_token": float(physical),
            })
            self._render()
            return True
        context = CONTEXT_RE.search(line)
        if context:
            round_id, length, gen, physical = context.groups()
            self.state = LiveState(
                arm=f"context {length}", progress=f"round {round_id}",
                gen=float(gen), physical=float(physical),
            )
            self.arms.append({
                "round": int(round_id), "context": int(length),
                "gen": float(gen), "physical_mb_token": float(physical),
            })
            self._render()
            return True
        digest = DIGEST_RE.search(line)
        if digest:
            self.state.digest = digest.group(1)
            self._current["digest"] = digest.group(1)
        return False

    def close(self) -> None:
        if self._current:
            self.arms.append(dict(self._current))
            self._current = {}
        self.finish_line()


def expand_command(spec: TestSpec, config: SuiteConfig, result_path: Path) -> list[str]:
    if not callable(spec.script):
        return []
    command = spec.script(config, result_path)
    if not isinstance(command, list) or not command:
        raise TypeError(f"{spec.id} script must return a non-empty list")
    if not all(isinstance(item, str) for item in command):
        raise TypeError(f"{spec.id} script must return only strings")
    return command


def interpret(spec: TestSpec, returncode: int, output: str) -> str:
    lowered = output.lower()
    if returncode != 0:
        return "Invalid run. The process failed, so no performance conclusion is allowed."
    if "refused" in lowered or "different token" in lowered or "digest" in lowered and "mismatch" in lowered:
        return "Invalid run. The control premise or exact token trajectory failed."
    bands = [float(value) for value in BAND_RE.findall(output)]
    if bands and max(bands) > 10.0:
        return (
            f"Environmentally unresolved. The largest reported band is {max(bands):.1f}%, "
            "which cannot decide a small optimization."
        )
    if "demonstrates a regression" in lowered or "resolved regression" in lowered:
        return "Observed result: the target shows a resolved performance regression. The user decides its status."
    if "resolved improvement" in lowered or "so this one stands" in lowered:
        return "Observed result: the measured improvement resolved. The user decides promotion and final quality evaluation."
    if "inside" in lowered and "band" in lowered or "unresolved" in lowered or "do not separate" in lowered:
        return "Observed result: the comparison is statistically unresolved. The user decides its status."
    if spec.promotion:
        return "Diagnostic completion only. This output lacks enough paired evidence for promotion."
    return "Diagnostic completed. Use the reported mechanism and metrics; do not infer a production speedup from this run alone."


class Runner:
    def __init__(self, config: SuiteConfig, output=None):
        self.config = config
        self.output = output or sys.stdout
        self.process: subprocess.Popen | None = None

    def _preflight(self) -> None:
        from models.flashnext.system_state import (
            purge_file_cache, wait_for_quiescence,
        )
        if self.config.purge_file_cache:
            self.output.write(f"{C['dim']}purging file cache once{C['0']}\n")
            self.output.flush()
            purge_file_cache()
        wait_for_quiescence(
            self.config.settle_seconds,
            self.config.settle_timeout,
            self.config.max_load,
            self.config.max_compressor_rate,
        )

    def run(self, spec: TestSpec) -> RunRecord:
        if not spec.runnable:
            raise ValueError(f"{spec.id} is {spec.status}, not runnable")
        results_dir = Path(self.config.results_dir).expanduser()
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_path = results_dir / f"{stamp}-{spec.id.replace(':', '-')}.json"
        command = expand_command(spec, self.config, result_path)
        self._preflight()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if spec.canonical:
            env.update(self.config.canonical_environment)
        if spec.environment is not None:
            env.update(spec.environment(self.config))
        if self.config.checkpoint:
            env["MACQWEN_FLASHNEXT_MODEL"] = self.config.checkpoint
        began = time.perf_counter()
        started_at = datetime.now().astimezone().isoformat()
        live = LiveMetrics(self.output, spec.live_parser)
        collected: list[str] = []
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        try:
            assert self.process.stdout is not None
            for raw in self.process.stdout:
                line = raw.rstrip("\n")
                collected.append(line)
                is_metric = live.feed(line)
                if is_metric:
                    live.finish_line()
                    self.output.write(f"{C['g']}{line}{C['0']}\n")
                elif line:
                    color = C["r"] if any(word in line.lower() for word in ("error", "failed", "refused")) else C["gray"]
                    self.output.write(f"{color}{line}{C['0']}\n")
                self.output.flush()
            returncode = self.process.wait()
        except KeyboardInterrupt:
            os.killpg(self.process.pid, signal.SIGINT)
            returncode = self.process.wait()
            collected.append("interrupted by user")
        finally:
            live.close()
            self.process = None
        text = "\n".join(collected)
        interpretation = (
            spec.interpret(returncode, text, live.arms)
            if spec.interpret is not None else None
        ) or interpret(spec, returncode, text)
        record = RunRecord(
            test_id=spec.id,
            title=spec.title,
            started_at=started_at,
            command=command,
            controls={
                **(self.config.canonical_environment if spec.canonical else {}),
                **spec.controls,
            },
            returncode=returncode,
            elapsed_seconds=round(time.perf_counter() - began, 3),
            arms=live.arms,
            interpretation=interpretation,
            output=text,
        )
        result_path.write_text(json.dumps(asdict(record), indent=2) + "\n")
        return record


def command_text(command: list[str]) -> str:
    return shlex.join(command)
