"""Interactive terminal using the same visual language as chat.sh."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import os
import sys
import time

from macqwen.terminal import read_prompt
from macqwen.ui import C, IngestGlow

from .api import TestSpec
from .catalog import build_catalog
from .runner import Runner, SuiteConfig, command_text, expand_command


CONFIG_PATH = Path("~/.cache/flashnext/test-suite.json").expanduser()


HELP = """  /help                         show commands
  /list [CATEGORY]              list runnable tests
  /show TEST                    explain one test and its controls
  /run TEST                     run one test after confirmation
  /research [TEXT]              search all historical research evidence
  /controls                     show enabled model and protocol controls
  /config [NAME VALUE]          read or change suite controls
  /results                      list saved result records
  /status                       show suite state
  /quit                         leave the suite"""


def _load_config() -> SuiteConfig:
    if not CONFIG_PATH.is_file():
        return SuiteConfig()
    try:
        data = json.loads(CONFIG_PATH.read_text())
        known = SuiteConfig.__dataclass_fields__
        return SuiteConfig(**{key: value for key, value in data.items() if key in known})
    except Exception:
        return SuiteConfig()


def _save_config(config: SuiteConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config.__dict__, indent=2) + "\n")


def _wrap(text: str, indent: str = "  ") -> str:
    import textwrap
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


class SuiteTerminal:
    def __init__(self, output=None):
        self.output = output or sys.stdout
        self.config = _load_config()
        self.catalog = build_catalog()
        self.runner = Runner(self.config, self.output)
        self.running = True

    def write(self, text: str = "", color: str = "0") -> None:
        self.output.write(f"{C[color]}{text}{C['0']}\n")
        self.output.flush()

    def controls(self) -> str:
        rows = [
            ("foundation", "60-slot skew slab pack + Frontier 8A"),
            ("decode", "greedy"),
            ("digest", "required"),
            ("quality", "user via chat.sh, sampling, xhigh"),
            ("tokens", str(self.config.tokens)),
            ("pairs", str(self.config.pairs)),
            ("I/O workers", str(self.config.workers)),
            ("file-cache purge", "on" if self.config.purge_file_cache else "off"),
            (
                "quiescence",
                "advisory/off" if self.config.settle_seconds <= 0 else
                f"{self.config.settle_seconds:.0f}s clean within {self.config.settle_timeout:.0f}s",
            ),
            ("maximum load", f"{self.config.max_load:.2f}"),
            ("compressor limit", f"{self.config.max_compressor_rate:.1f} pages/s"),
            ("checkpoint", self.config.checkpoint or "automatic"),
            ("Python", self.config.python),
            ("Frontier 8B", "off"),
            ("streamed records", "off"),
            ("Up-QMV/SwiGLU", "off"),
            ("boundary profiler", "off unless selected test enables it"),
            ("reboot", "never required"),
        ]
        width = max(len(name) for name, _ in rows) + 2
        return "\n".join(f"  {name:<{width}}{value}" for name, value in rows)

    def show(self, spec: TestSpec) -> None:
        state = "runnable" if spec.runnable else spec.status
        self.write(spec.title, "b")
        self.write(f"  id: {spec.id}  category: {spec.category}  state: {state}", "dim")
        self.write(_wrap(spec.explanation))
        self.write("\nWhy it was proposed:", "c")
        self.write(_wrap(spec.why))
        if spec.metrics:
            self.write("\nMetrics:", "c")
            for metric in spec.metrics:
                self.write(f"  - {metric}")
        if spec.controls:
            self.write("\nTest controls:", "c")
            for name, value in spec.controls.items():
                self.write(f"  {name}: {value}")
        self.write(f"\n  source: {spec.source}", "dim")
        if spec.runnable:
            result_path = Path(self.config.results_dir) / "RESULT.json"
            command = expand_command(spec, self.config, result_path)
            self.write(f"  command: {command_text(command)}", "dim")

    def list_specs(self, category: str = "") -> None:
        specs = [
            spec for spec in self.catalog.values()
            if spec.runnable and (not category or spec.category == category)
        ]
        specs.sort(key=lambda item: (item.category, item.id))
        if not specs:
            self.write(f"no runnable tests in category {category}", "y")
            return
        width = min(34, max(len(spec.id) for spec in specs) + 2)
        current = None
        for spec in specs:
            if spec.category != current:
                current = spec.category
                self.write(f"\n{current}", "c")
            self.write(f"  {spec.id:<{width}}{spec.title}")

    def research(self, query: str) -> None:
        words = query.lower().split()
        rows = [spec for spec in self.catalog.values() if spec.category == "history"]
        if words:
            rows = [
                spec for spec in rows
                if all(word in f"{spec.title} {spec.explanation} {spec.why}".lower() for word in words)
            ]
        rows.sort(key=lambda item: int(item.id.split("L", 1)[1]))
        cap = 80
        for spec in rows[:cap]:
            self.write(f"  {spec.id:<15}{spec.title[:78]}")
        if len(rows) > cap:
            self.write(f"  +{len(rows) - cap} more; narrow the search text", "dim")
        self.write(f"\n  {len(rows)} historical records matched", "dim")

    def configure(self, argument: str) -> None:
        if not argument:
            self.write(self.controls())
            self.write("\n  names: tokens pairs workers purge settle timeout max-load compressor-rate checkpoint results python", "dim")
            return
        parts = shlex.split(argument)
        if len(parts) < 2:
            self.write("usage: /config NAME VALUE", "y")
            return
        name, value = parts[0], " ".join(parts[1:])
        try:
            if name == "tokens":
                self.config.tokens = max(1, int(value))
            elif name == "pairs":
                self.config.pairs = max(3, int(value))
            elif name == "workers":
                self.config.workers = max(1, int(value))
            elif name == "purge":
                if value not in {"on", "off"}:
                    raise ValueError("purge accepts on or off")
                self.config.purge_file_cache = value == "on"
            elif name == "settle":
                self.config.settle_seconds = max(0.0, float(value))
            elif name == "timeout":
                self.config.settle_timeout = max(1.0, float(value))
            elif name == "max-load":
                self.config.max_load = max(0.0, float(value))
            elif name == "compressor-rate":
                self.config.max_compressor_rate = max(0.0, float(value))
            elif name == "checkpoint":
                self.config.checkpoint = "" if value in {"auto", "automatic"} else value
            elif name == "results":
                self.config.results_dir = str(Path(value).expanduser())
            elif name == "python":
                self.config.python = str(Path(value).expanduser())
            else:
                raise ValueError(f"unknown config name {name}")
        except ValueError as exc:
            self.write(str(exc), "y")
            return
        _save_config(self.config)
        self.runner.config = self.config
        self.write(f"{name}: {value}", "g")

    def results(self) -> None:
        directory = Path(self.config.results_dir).expanduser()
        files = sorted(directory.glob("*.json"), reverse=True) if directory.is_dir() else []
        for path in files[:30]:
            try:
                row = json.loads(path.read_text())
                self.write(
                    f"  {path.name:<48}{row.get('test_id', ''):<28}{row.get('interpretation', '')[:55]}"
                )
            except Exception:
                self.write(f"  {path.name:<48}unreadable", "y")
        if not files:
            self.write("no saved results", "dim")

    def run_spec(self, spec: TestSpec) -> None:
        if not spec.runnable:
            self.show(spec)
            self.write("\nThis record is evidence only. Its implementation is not retained.", "y")
            return
        self.show(spec)
        self.write("\nEnabled suite controls:", "c")
        self.write(self.controls())
        answer = read_prompt(f"{C['b']}run>{C['0']} type yes to start: ")
        if answer.strip().lower() != "yes":
            self.write("cancelled", "dim")
            return
        glow = IngestGlow(self.output)
        glow.start(label=f"Preparing {spec.title}")
        time.sleep(0.20)
        glow.finish()
        self.write(f"\n{spec.title}", "b")
        try:
            record = self.runner.run(spec)
        except Exception as exc:
            self.write(f"Failed: {exc}", "r")
            return
        color = "g" if record.returncode == 0 else "r"
        self.write(f"\nInterpretation: {record.interpretation}", color)
        self.write(f"  elapsed {record.elapsed_seconds:.1f}s  saved in {self.config.results_dir}", "dim")

    def dispatch(self, text: str) -> None:
        parts = text.strip().split(maxsplit=1)
        command = parts[0]
        argument = parts[1] if len(parts) > 1 else ""
        if command == "/help":
            self.write(HELP)
        elif command == "/list":
            self.list_specs(argument.strip())
        elif command == "/show":
            spec = self.catalog.get(argument.strip())
            self.show(spec) if spec else self.write(f"unknown test {argument}", "y")
        elif command == "/run":
            spec = self.catalog.get(argument.strip())
            self.run_spec(spec) if spec else self.write(f"unknown test {argument}", "y")
        elif command == "/research":
            self.research(argument)
        elif command in {"/controls", "/status"}:
            self.write(self.controls())
            self.write(f"\n  runnable tests: {sum(spec.runnable for spec in self.catalog.values())}", "dim")
            self.write(f"  historical records: {sum(spec.category == 'history' for spec in self.catalog.values())}", "dim")
        elif command == "/config":
            self.configure(argument)
        elif command == "/results":
            self.results()
        elif command in {"/quit", "/exit"}:
            self.running = False
        else:
            matches = [spec for spec in self.catalog.values() if text.lower() in spec.title.lower()]
            if matches:
                for spec in matches[:20]:
                    self.write(f"  {spec.id:<34}{spec.title}")
            else:
                self.write(f"unknown command {command}  use /help", "y")

    def start(self) -> int:
        began = time.perf_counter()
        self.write(f"{C['dim']}loading FlashNext test catalog...{C['0']}")
        self.write(
            f"ready in {time.perf_counter() - began:.1f}s  flashnext / tests  use /help for commands\n",
            "dim",
        )
        self.write(HELP + "\n")
        if os.environ.get("TERM_PROGRAM") != "Apple_Terminal":
            self.write(
                "  warning: trusted performance runs use Apple Terminal; "
                "embedded application terminals can consume WindowServer and GPU time\n",
                "y",
            )
        while self.running:
            try:
                text = read_prompt(f"{C['b']}you>{C['0']} ")
            except (EOFError, KeyboardInterrupt):
                self.write()
                return 0
            if text.strip():
                self.dispatch(text.strip())
                if self.running:
                    self.write()
        return 0


def main() -> int:
    return SuiteTerminal().start()
