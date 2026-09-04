"""Stable plugin API for terminal-discovered FlashNext tests."""
from __future__ import annotations

from dataclasses import dataclass, field
import ast
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
FLASHNEXT = ROOT / "models" / "flashnext"

CommandScript = Callable[[Any, Path], list[str]]
EnvironmentScript = Callable[[Any], dict[str, str]]
InterpretScript = Callable[[int, str, list[dict]], str | None]
LiveParser = Callable[[str], dict[str, Any] | None]


@dataclass(frozen=True)
class TestSpec:
    """One self-contained terminal test supplied by a case module."""

    id: str
    title: str
    category: str
    explanation: str
    why: str
    script: CommandScript | None = None
    metrics: tuple[str, ...] = ()
    controls: dict[str, str] = field(default_factory=dict)
    source: str = ""
    status: str = "runnable"
    promotion: bool = False
    environment: EnvironmentScript | None = None
    interpret: InterpretScript | None = None
    live_parser: LiveParser | None = None
    canonical: bool = True

    @property
    def runnable(self) -> bool:
        return self.status == "runnable" and callable(self.script)


COMMON_METRICS = (
    "generation and tail rate",
    "physical MB/token",
    "active memory",
    "token digest",
    "paired effect and resolution band",
)
IO_METRICS = COMMON_METRICS + (
    "submission-to-worker-start delay",
    "positioned-read wall time",
    "total I/O wait",
)


def benchmark_script(filename: str, *arguments) -> CommandScript:
    """Build a command function while preserving late config values."""
    def command(config, _result_path: Path) -> list[str]:
        values = {
            "tokens": str(config.tokens),
            "pairs": str(config.pairs),
            "workers": str(config.workers),
        }
        command = [str(config.python), str(FLASHNEXT / filename)]
        for argument in arguments:
            if argument == "{model_args}":
                if config.checkpoint:
                    command.extend(("--model", config.checkpoint))
                continue
            command.append(values.get(argument.strip("{}"), argument))
        return command
    return command


def script_case(
    *, test_id: str, title: str, category: str, explanation: str, why: str,
    filename: str, arguments: tuple = (), metrics: tuple[str, ...] = (),
    controls: dict[str, str] | None = None, promotion: bool = False,
) -> TestSpec:
    return TestSpec(
        id=test_id, title=title, category=category,
        explanation=explanation, why=why,
        script=benchmark_script(filename, *arguments),
        metrics=metrics or (IO_METRICS if category in {"performance", "diagnostic"} else COMMON_METRICS),
        controls=controls or {"decode": "greedy", "digest": "required"},
        source=f"models/flashnext/{filename}", promotion=promotion,
    )


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def production_case(name: str, why: str) -> TestSpec:
    """Create one production comparison from its retained benchmark definition."""
    benchmark = FLASHNEXT / "bench_production.py"
    comparisons = _literal_assignment(benchmark, "COMPARISONS") or {}
    if name not in comparisons:
        raise ValueError(f"unknown production comparison {name}")
    conditions = comparisons[name]
    load_time = set(_literal_assignment(benchmark, "LOAD_TIME_SETTINGS") or ())
    settings = {key for environment in conditions.values() for key in environment}

    def command(config, _result_path: Path) -> list[str]:
        result = [
            str(config.python), str(benchmark), "--compare", name,
            "--tokens", str(config.tokens), "--arms", str(config.pairs),
            "--min-arms", str(config.pairs), "--drop", "0",
        ]
        if settings & load_time:
            result.append("--fresh-arms")
        return result

    controls = {
        condition: ", ".join(f"{key}={value}" for key, value in environment.items()) or "canonical"
        for condition, environment in conditions.items()
    }
    return TestSpec(
        id=f"production-{name}", title=f"Production comparison: {name}", category="performance",
        explanation=f"Runs complete greedy decode across: {', '.join(conditions)}.",
        why=why, script=command, metrics=COMMON_METRICS, controls=controls,
        source="models/flashnext/bench_production.py", promotion=True,
    )
