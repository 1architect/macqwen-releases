"""Discover live-test providers below runtime folders."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .api import ROOT


def runtime_directories(root: Path = ROOT) -> list[str]:
    models = root / "models"
    result = []
    for path in sorted(models.iterdir() if models.is_dir() else ()):
        if path.is_dir() and (path / "tests").is_dir() and any(
            (path / "tests").glob("case_*.py")
        ):
            result.append(path.name)
    return result


def _module_name(runtime: str) -> str:
    return f"models.{runtime.replace('-', '_')}.tests.catalog"


def build_catalog(runtime: str, root: Path = ROOT) -> dict[str, Any]:
    """Return one runtime's catalog without importing any model backend."""
    runtime = runtime.replace("-", "_")
    if runtime not in runtime_directories(root):
        return {}
    module = importlib.import_module(_module_name(runtime))
    builder = getattr(module, "build_catalog", None)
    if not callable(builder):
        raise RuntimeError(f"{runtime} tests must expose build_catalog()")
    return builder()
