"""Shared pure-Python backend setting metadata and registry behavior."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Setting:
    name: str
    aliases: tuple[str, ...]
    default: Any
    parser: Callable[[str], Any]
    lifecycle: str
    group: str
    visibility: str
    backend: str
    reader: Callable[[Any], Any]
    setter: Callable[[Any, Any], None] | None = None
    active: Callable[[Any], bool] | None = None
    warning: Callable[[Any], str | None] | None = None
    env_key: str | None = None
    source: str = ""
    attribute: str | None = None
    cli_flags: tuple[str, ...] = ()
    cli_dest: str | None = None
    cli_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lifecycle not in {"live", "next-turn", "startup", "read-only"}:
            raise ValueError(f"invalid lifecycle for {self.name}: {self.lifecycle}")
        if self.visibility not in {"public", "internal", "research-only"}:
            raise ValueError(f"invalid visibility for {self.name}: {self.visibility}")
        if not self.name or not callable(self.parser) or not callable(self.reader):
            raise ValueError(f"malformed setting: {self.name!r}")

    def parse(self, raw: str) -> Any:
        return self.parser(raw)

    def value(self, backend: Any) -> Any:
        return self.reader(backend)

    def is_active(self, backend: Any) -> bool:
        return bool(self.active(backend)) if self.active else True

    def warning_text(self, backend: Any) -> str | None:
        return self.warning(backend) if self.warning else None

    def source_for(self, backend: Any) -> str:
        sources = getattr(backend, "_setting_sources", {})
        if self.name in sources:
            return sources[self.name]
        if self.env_key:
            import os
            if self.env_key in os.environ:
                return "environment"
        return "default"


class SettingRegistry:
    def __init__(self, settings: Iterable[Setting]):
        self.settings = tuple(settings)
        self._by_backend_name: dict[tuple[str, str], Setting] = {}
        for setting in self.settings:
            for name in (setting.name, *setting.aliases):
                key = (setting.backend, name)
                if key in self._by_backend_name:
                    raise ValueError(
                        f"duplicate setting name or alias: {key[0]}:{key[1]}"
                    )
                self._by_backend_name[key] = setting

    def for_backend(self, backend: str) -> tuple[Setting, ...]:
        return tuple(setting for setting in self.settings if setting.backend == backend)

    def get(self, backend: str, name: str) -> Setting:
        try:
            return self._by_backend_name[(backend, name)]
        except KeyError as exc:
            raise ValueError(f"unknown {backend} setting: {name}") from exc

    def defaults(self, backend: str) -> dict[str, Any]:
        return {setting.name: setting.default for setting in self.for_backend(backend)}

    def cli_values(self, args: Any, backend: str) -> dict[str, Any]:
        values = {}
        for setting in self.for_backend(backend):
            if setting.cli_dest and hasattr(args, setting.cli_dest):
                value = getattr(args, setting.cli_dest)
                if value is not None:
                    values[setting.attribute or setting.name.replace("-", "_")] = value
        return values

    def configure(self, backend_obj: Any, argument: str, backend: str) -> str:
        text = argument.strip()
        settings = self.for_backend(backend)
        if not text:
            return self.render(backend_obj, backend)
        if text == "all":
            return self.render(backend_obj, backend, include_research=True)
        if text == "defaults":
            for setting in settings:
                if setting.setter and setting.lifecycle in {"live", "next-turn"}:
                    setting.setter(backend_obj, setting.default)
            if hasattr(backend_obj, "_rebuild_routing"):
                backend_obj._rebuild_routing()
            backend_obj._setting_sources = {
                setting.name: "default" for setting in settings
            }
            return f"{backend} settings restored to defaults\n{self.render(backend_obj, backend)}"
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("use /config model NAME VALUE or /config model defaults")
        setting = self.get(backend, parts[0])
        if not setting.setter or setting.lifecycle in {"startup", "read-only"}:
            raise ValueError(
                f"{setting.name} applies at {setting.lifecycle}; restart the model"
            )
        value = setting.parse(parts[1])
        if setting.name == "fusion-block" and value < getattr(backend_obj, "fusion_min_block", 1):
            raise ValueError("fusion-min-block cannot exceed fusion-block")
        if setting.name == "fusion-min-block" and value > getattr(backend_obj, "fusion_block", value):
            raise ValueError("fusion-min-block cannot exceed fusion-block")
        setting.setter(backend_obj, value)
        if hasattr(backend_obj, "_rebuild_routing"):
            backend_obj._rebuild_routing()
        sources = dict(getattr(backend_obj, "_setting_sources", {}))
        sources[setting.name] = "live state"
        backend_obj._setting_sources = sources
        warning = setting.warning_text(backend_obj)
        result = f"{setting.name}: {value}  applies on the next turn"
        return result + (f"\nwarning: {warning}" if warning else "")

    def render(self, backend_obj: Any, backend: str, include_research: bool = False) -> str:
        title = "Flash-Next settings" if backend == "flashnext" else "Qwen27B startup settings"
        rows = [title]
        groups: dict[str, list[Setting]] = {}
        for setting in self.for_backend(backend):
            if setting.visibility == "internal":
                continue
            if setting.visibility == "research-only" and not include_research:
                continue
            groups.setdefault(setting.group, []).append(setting)
        for group, settings in groups.items():
            rows.append(f"  [{group}]")
            for setting in settings:
                value = setting.value(backend_obj)
                active = setting.is_active(backend_obj)
                lifecycle = (
                    "startup, restart required"
                    if setting.lifecycle == "startup" else setting.lifecycle
                )
                text = (
                    f"  {setting.name:<24} {value!s:<18} "
                    f"{'active' if active else 'inactive'}, "
                    f"{lifecycle}, {setting.source_for(backend_obj)}"
                )
                if setting.aliases:
                    text += f"  aliases: {', '.join(setting.aliases)}"
                warning = setting.warning_text(backend_obj)
                if warning:
                    text += f"  warning: {warning}"
                rows.append(text if active else f"\033[2m{text}\033[0m")
        rows.append(
            "legend: dim = inactive; source = default, CLI, environment, or live state"
        )
        if backend == "flashnext":
            rows.extend([
                "usage: /config model NAME VALUE | /config model defaults | /config model all",
                "routing: standard, fast, fast-quality, exact-quality, cache-aware, fused-quality",
                "research-only settings are visible with /config model all",
            ])
        if backend == "qwen27b":
            rows.append("change these with CLI options, then restart the model")
        return "\n".join(rows)
