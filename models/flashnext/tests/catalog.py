"""Automatic case-module discovery and validation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from .api import TestSpec


CASE_PATTERN = "case_*.py"


def _load_module(path: Path) -> ModuleType:
    name = f"models.flashnext.tests.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test case module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provided(module: ModuleType) -> list[TestSpec]:
    values = []
    if hasattr(module, "TEST"):
        values.append(module.TEST)
    if hasattr(module, "TESTS"):
        values.extend(module.TESTS)
    provider = getattr(module, "get_tests", None)
    if callable(provider):
        values.extend(provider())
    if not values:
        raise RuntimeError(
            f"{module.__name__} must provide TEST, TESTS, or get_tests()"
        )
    runnable = [case for case in values if getattr(case, "status", None) == "runnable"]
    if len(runnable) > 1:
        raise RuntimeError(
            f"{module.__name__} provides multiple runnable tests; use one case file per test"
        )
    return values


def _validate(case: TestSpec, path: Path) -> None:
    if not isinstance(case, TestSpec):
        raise TypeError(f"{path} returned {type(case).__name__}, expected TestSpec")
    missing = [
        name for name in ("id", "title", "category", "explanation", "why", "source")
        if not str(getattr(case, name, "")).strip()
    ]
    if missing:
        raise ValueError(f"{path} omits required fields: {', '.join(missing)}")
    if case.status == "runnable" and not callable(case.script):
        raise ValueError(f"{path} must provide an executable script function")


def build_catalog(directory: Path | None = None) -> dict[str, TestSpec]:
    folder = directory or Path(__file__).resolve().parent
    result: dict[str, TestSpec] = {}
    for path in sorted(folder.glob(CASE_PATTERN)):
        module = _load_module(path)
        for case in _provided(module):
            _validate(case, path)
            if case.id in result:
                raise ValueError(f"duplicate test id {case.id} in {path}")
            result[case.id] = case
    return result
