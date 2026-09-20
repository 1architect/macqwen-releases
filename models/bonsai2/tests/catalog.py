from __future__ import annotations

import importlib.util
from pathlib import Path


def build_catalog() -> dict:
    result = {}
    folder = Path(__file__).resolve().parent
    for path in sorted(folder.glob("case_*.py")):
        name = f"models.bonsai2.tests.{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load test case {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case = getattr(module, "TEST", None)
        if case is None:
            raise RuntimeError(f"{path} must provide TEST")
        if case.id in result:
            raise RuntimeError(f"duplicate test id {case.id}")
        result[case.id] = case
    return result
