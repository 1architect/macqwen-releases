"""Discover the supported K2-Horizon MLX checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .settings import ALIASES, CHECKPOINT_ENV


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def model_root() -> Path:
    return Path(os.environ.get("MACQWEN_MODEL_ROOT", "~/models")).expanduser()


def compatible(path: Path) -> bool:
    config = _json(path / "config.json")
    index = _json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    required = (
        "model.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    if (
        config.get("model_type") != "k2_horizon"
        or config.get("model_file") != "model.py"
        or not isinstance(weight_map, dict)
        or not all((path / name).is_file() for name in required)
    ):
        return False
    shards = set(weight_map.values())
    return bool(shards) and all((path / shard).is_file() for shard in shards)


def installed(root: Path | None = None) -> list[Path]:
    root = root or model_root()
    children = root.iterdir() if root.is_dir() else ()
    return [path for path in sorted(children) if path.is_dir() and compatible(path)]


def resolve_k2_horizon(requested: str | os.PathLike[str] | None = None) -> Path:
    root = model_root()
    value = str(requested or os.environ.get(CHECKPOINT_ENV, "")).strip()
    if value and value != "auto":
        path = Path(ALIASES.get(value.lower(), value)).expanduser()
        if not path.is_absolute():
            path = root / path
        if not compatible(path):
            raise ValueError(
                f"incomplete or incompatible K2-Horizon checkpoint: {path}"
            )
        return path.resolve()

    choices = installed(root)
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise ValueError(
            "no complete K2-Horizon checkpoint found; use --checkpoint PATH"
        )
    lines = "\n".join(f"  --checkpoint {path}" for path in choices)
    raise ValueError("choose a K2-Horizon checkpoint:\n" + lines)
