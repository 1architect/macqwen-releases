"""Discover setting_*.py providers without importing MLX."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from macqwen.backend_settings import Setting, SettingRegistry


FOLDER = Path(__file__).resolve().parent


def _load(path: Path) -> ModuleType:
    name = f"models.flashnext.settings.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load settings provider {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provided(module: ModuleType) -> list[Setting]:
    values = []
    if hasattr(module, "SETTING"):
        values.append(module.SETTING)
    if hasattr(module, "SETTINGS"):
        values.extend(module.SETTINGS)
    provider = getattr(module, "get_settings", None)
    if callable(provider):
        values.extend(provider())
    if not values:
        raise ValueError(f"{module.__name__} must provide SETTING, SETTINGS, or get_settings()")
    return values


def get_registry(directory: Path | None = None) -> SettingRegistry:
    folder = directory or FOLDER
    settings = []
    for path in sorted(folder.glob("setting_*.py")):
        settings.extend(_provided(_load(path)))
    for setting in settings:
        if not isinstance(setting, Setting):
            raise TypeError(f"{setting!r} is not a Setting")
    return SettingRegistry(settings)
