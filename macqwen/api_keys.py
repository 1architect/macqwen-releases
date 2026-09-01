"""Private API keys stored outside the repository."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile


DEFAULT_PATH = "~/Library/Application Support/MACQWEN/api_keys.json"


@dataclass(frozen=True)
class Service:
    name: str
    environment: str
    required: bool


SERVICES = {
    "tavily": Service("Tavily", "TAVILY_API_KEY", True),
    "context7": Service("Context7", "CONTEXT7_API_KEY", False),
}

_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY", re.I)


def sanitized_environment() -> dict[str, str]:
    """Return the process environment without secret-like variables."""
    return {
        name: value
        for name, value in os.environ.items()
        if not _SECRET_NAME.search(name)
    }


class KeyStore:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path).expanduser()
        self._external = {
            service.environment: os.environ.get(service.environment)
            for service in SERVICES.values()
        }
        self.values = self._load()
        self._secure_existing()
        self.apply()

    def _secure_existing(self) -> None:
        if not self.path.is_file():
            return
        os.chmod(self.path, 0o600)
        os.chmod(self.path.parent, 0o700)

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, TypeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            name: value.strip()
            for name, value in raw.items()
            if name in SERVICES and isinstance(value, str) and value.strip()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".api_keys.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(self.values, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def apply(self) -> None:
        for name, value in self.values.items():
            os.environ[SERVICES[name].environment] = value

    def resolve(self, name: str) -> str:
        key = name.strip().lower()
        if key not in SERVICES:
            raise ValueError(f"unknown service {name!r}; use: {', '.join(SERVICES)}")
        return key

    def set(self, name: str, value: str) -> None:
        key = self.resolve(name)
        secret = value.strip()
        if not secret or "\n" in secret or "\r" in secret:
            raise ValueError("API key must be one non-empty line")
        self.values[key] = secret
        self._save()
        os.environ[SERVICES[key].environment] = secret

    def delete(self, name: str) -> bool:
        key = self.resolve(name)
        if key not in self.values:
            return False
        del self.values[key]
        self._save()
        environment = SERVICES[key].environment
        external = self._external.get(environment)
        if external is None:
            os.environ.pop(environment, None)
        else:
            os.environ[environment] = external
        return True

    def source(self, name: str) -> str:
        key = self.resolve(name)
        if key in self.values:
            return "Application Support"
        if os.environ.get(SERVICES[key].environment, "").strip():
            return "environment"
        return "missing"

    def status(self) -> str:
        rows = []
        for name, service in SERVICES.items():
            source = self.source(name)
            need = "required" if service.required else "optional"
            state = f"configured in {source}" if source != "missing" else "not configured"
            rows.append(f"  {name:<10} {state} ({need})")
        return "\n".join(rows)
