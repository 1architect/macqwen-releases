"""Runtime-neutral API for test cases in ``models/<model>/tests/cases/``.

A case file imports everything it needs from here. Benchmarks run as modules
(``python -m models.<model>.tests.bench.<script>``), so no command depends on
the working directory.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Callable

from macqwen import results
from .spec import (  # noqa: F401  re-exported for case files
    COMMON_METRICS, IO_METRICS, CommandScript, EnvironmentScript,
    InterpretScript, LiveParser, TestSpec,
)


ROOT = Path(__file__).resolve().parents[2]


def bench_module(runtime: str, script: str) -> str:
    """Dotted module name of a benchmark script in ``tests/bench``."""
    stem = Path(script).stem
    return f"models.{runtime}.tests.bench.{stem}"


def bench_path(runtime: str, script: str) -> Path:
    return ROOT / "models" / runtime / "tests" / "bench" / f"{Path(script).stem}.py"


def inputs_dir(runtime: str) -> Path:
    """Operator-supplied inputs, such as an everyday prompt file."""
    return ROOT / "models" / runtime / "tests" / "inputs"


@dataclass
class TestContext:
    runtime: str
    checkpoint: str
    python: str
    tokens: int = 32
    pairs: int = 6
    workers: int = 16
    result_root: Path = results.RESULTS_ROOT
    extra: dict[str, Any] = field(default_factory=dict)
    # Set by the runner before the case builds its command.
    run_dir: Path | None = None

    @property
    def results_dir(self) -> str:
        """This run's folder, or the model's results folder outside a run."""
        if self.run_dir is not None:
            return str(self.run_dir)
        return str(results.model_root(self.runtime, self.result_root))

    def output(self, filename: str) -> Path:
        """Destination for one artifact of the current run."""
        if self.run_dir is None:
            raise RuntimeError("output paths exist only while a test runs")
        return Path(self.run_dir) / filename

    def latest(self, filename: str) -> Path | None:
        """Newest ``filename`` recorded by an earlier run of this model."""
        return results.latest(self.runtime, filename, self.result_root)

    def input(self, filename: str) -> Path:
        return inputs_dir(self.runtime) / filename

    @property
    def canonical_environment(self) -> dict[str, str]:
        """The model's launch environment, from its ``tests`` package hook."""
        try:
            package = importlib.import_module(f"models.{self.runtime}.tests")
        except ImportError:
            return {}
        hook = getattr(package, "canonical_environment", None)
        return dict(hook(self)) if callable(hook) else {}


def benchmark_script(runtime: str, script: str, *arguments) -> CommandScript:
    """Build a command that runs one benchmark module with late config values.

    ``{tokens}``, ``{pairs}`` and ``{workers}`` take the terminal's settings;
    ``{model_args}`` expands to ``--model CHECKPOINT`` when one is chosen.
    """
    def command(config, _result_path: Path) -> list[str]:
        values = {
            "tokens": str(config.tokens),
            "pairs": str(config.pairs),
            "workers": str(config.workers),
        }
        result = [str(config.python), "-m", bench_module(runtime, script)]
        for argument in arguments:
            if argument == "{model_args}":
                if config.checkpoint:
                    result.extend(("--model", config.checkpoint))
                continue
            result.append(values.get(argument.strip("{}"), argument))
        return result
    return command


def script_case(
    *, runtime: str, test_id: str, title: str, category: str, explanation: str,
    why: str, filename: str, arguments: tuple = (),
    metrics: tuple[str, ...] = (), controls: dict[str, str] | None = None,
    promotion: bool = False,
) -> TestSpec:
    return TestSpec(
        id=test_id, title=title, category=category,
        explanation=explanation, why=why,
        script=benchmark_script(runtime, filename, *arguments),
        metrics=metrics or (
            IO_METRICS if category in {"performance", "diagnostic"} else COMMON_METRICS
        ),
        controls=controls or {"decode": "greedy", "digest": "required"},
        source=str(bench_path(runtime, filename).relative_to(ROOT)),
        promotion=promotion,
    )


def literal_assignment(path: Path, name: str):
    """Read a module-level literal without importing the module."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def production_case(
    name: str, why: str, *, runtime: str = "flashnext",
    script: str = "bench_production",
) -> TestSpec:
    """One comparison from a benchmark's ``COMPARISONS`` table."""
    benchmark = bench_path(runtime, script)
    comparisons = literal_assignment(benchmark, "COMPARISONS") or {}
    if name not in comparisons:
        raise ValueError(f"unknown production comparison {name}")
    conditions = comparisons[name]
    load_time = set(literal_assignment(benchmark, "LOAD_TIME_SETTINGS") or ())
    settings = {key for environment in conditions.values() for key in environment}

    def command(config, _result_path: Path) -> list[str]:
        result = [
            str(config.python), "-m", bench_module(runtime, script),
            "--compare", name,
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
        source=str(benchmark.relative_to(ROOT)), promotion=True,
    )


EnvironmentHook = Callable[[TestContext], dict[str, str]]
