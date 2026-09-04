"""Discover Qwen27B setting providers without loading the model."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from macqwen.backend_settings import Setting, SettingRegistry


FOLDER = Path(__file__).resolve().parent


def get_registry() -> SettingRegistry:
    settings = []
    for path in sorted(FOLDER.glob("setting_*.py")):
        name = f"models.qwen27b.settings.{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load settings provider {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "SETTING"):
            settings.append(module.SETTING)
        if hasattr(module, "SETTINGS"):
            settings.extend(module.SETTINGS)
        provider = getattr(module, "get_settings", None)
        if callable(provider):
            settings.extend(provider())
    return SettingRegistry(settings)
