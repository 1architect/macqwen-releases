"""Interactive project-level terminal for runtime live tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from macqwen.checkpoints import installed_checkpoints
from macqwen.terminal import read_prompt
from macqwen.ui import C

from .api import ROOT, TestContext
from .catalog import build_catalog, runtime_directories
from .runner import Runner


HELP = """  /help                         show commands
  /list [CATEGORY]              list tests for the selected model
  /show TEST                    explain one test
  /run TEST                     run one test after confirmation
  /results                      show this model's results folder
  /status                       show selected model and test count
  /quit                         leave the test terminal"""


def choose_model(model: str | None, checkpoint: str | None) -> tuple[str, Path]:
    choices = installed_checkpoints()
    if model:
        wanted = model.replace("-", "_")
        choices = [item for item in choices if item[0].replace("-", "_") == wanted]
    if checkpoint:
        requested = Path(checkpoint).expanduser().resolve()
        choices = [item for item in choices if item[0] == model and item[1].resolve() == requested]
        if not choices:
            if not model:
                raise SystemExit("--checkpoint requires --model for the test terminal")
            return model, requested
    if not choices:
        raise SystemExit("no compatible installed checkpoint is available")
    if len(choices) == 1:
        return choices[0]
    print("Multiple compatible checkpoints found:", file=sys.stderr)
    for index, (name, path) in enumerate(choices, 1):
        print(f"  {index}. {name}: {path}", file=sys.stderr)
    try:
        index = int(input(f"Choose a model [1-{len(choices)}]: ").strip()) - 1
    except (EOFError, ValueError):
        raise SystemExit("invalid model choice") from None
    if not 0 <= index < len(choices):
        raise SystemExit("invalid model choice")
    return choices[index]


class TestTerminal:
    def __init__(self, model: str | None = None, checkpoint: str | None = None,
                 output=None):
        self.output = output or sys.stdout
        self.model, self.checkpoint = choose_model(model, checkpoint)
        from macqwen.cli import _interpreter

        self.python = str(_interpreter(self.model))
        runtime = self.model.replace("-", "_")
        self.context = TestContext(runtime, str(self.checkpoint), self.python)
        self.catalog = build_catalog(runtime)
        self.runner = Runner(self.context, self.output)
        self.running = True

    def write(self, text: str = "", color: str = "0") -> None:
        self.output.write(f"{C[color]}{text}{C['0']}\n")
        self.output.flush()

    def list_specs(self, category: str = "") -> None:
        specs = [item for item in self.catalog.values() if item.runnable and
                 (not category or item.category == category)]
        for spec in sorted(specs, key=lambda item: (item.category, item.id)):
            self.write(f"  {spec.category:<16}{spec.id:<34}{spec.title}")
        if not specs:
            self.write("no matching runnable tests", "y")

    def show(self, spec) -> None:
        self.write(spec.title, "b")
        self.write(f"  id: {spec.id}  category: {spec.category}", "dim")
        self.write(f"  {spec.explanation}")
        self.write(f"  why: {spec.why}")
        self.write(f"  source: {spec.source}", "dim")

    def dispatch(self, text: str) -> None:
        command, _, argument = text.partition(" ")
        if command == "/help":
            self.write(HELP)
        elif command == "/list":
            self.list_specs(argument.strip())
        elif command == "/show":
            spec = self.catalog.get(argument.strip())
            self.show(spec) if spec else self.write("unknown test", "y")
        elif command == "/run":
            spec = self.catalog.get(argument.strip())
            if not spec or not spec.runnable:
                self.write("unknown or non-runnable test", "y")
                return
            self.show(spec)
            if read_prompt("type yes to start: ").strip().lower() != "yes":
                self.write("cancelled", "dim")
                return
            try:
                result = self.runner.run(spec)
            except Exception as exc:
                self.write(f"Failed: {exc}", "r")
                return
            color = "g" if result["returncode"] == 0 else "r"
            self.write(f"{result['interpretation']}\nsaved in {result['path']}", color)
        elif command == "/results":
            self.write(str(Path(self.context.results_dir).resolve()))
        elif command in {"/status", "/controls"}:
            self.write(f"model: {self.model}\ncheckpoint: {self.checkpoint}\npython: {self.python}")
            self.write(f"runnable tests: {sum(spec.runnable for spec in self.catalog.values())}", "dim")
        elif command in {"/quit", "/exit"}:
            self.running = False
        else:
            self.write("unknown command; use /help", "y")

    def start(self) -> int:
        self.write(f"ready: {self.model} / tests", "dim")
        self.write(HELP + "\n")
        while self.running:
            try:
                text = read_prompt(f"{C['b']}you>{C['0']} ")
            except (EOFError, KeyboardInterrupt):
                self.write()
                return 0
            if text.strip():
                self.dispatch(text.strip())
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="macqwen tests")
    model_choices = sorted({name for name in runtime_directories()} | {
        name.replace("_", "-") for name in runtime_directories()
    })
    parser.add_argument("--model", choices=model_choices)
    parser.add_argument("--checkpoint", "--model-path")
    args = parser.parse_args(argv)
    return TestTerminal(args.model, args.checkpoint).start()
