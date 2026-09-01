"""Find compatible local model checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path


FLASHNEXT_ALIASES = {
    "oq3": "Qwen3.8-Flash-Next-MLX-oQ3-MTP",
    "oq3-mtp": "Qwen3.8-Flash-Next-MLX-oQ3-MTP",
    "oq4": "Qwen3.8-Flash-Next-MLX-oQ4",
}


def model_root() -> Path:
    return Path(os.environ.get("MACQWEN_MODEL_ROOT", "~/models")).expanduser()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def flashnext_compatible(path: Path, complete: bool = True) -> bool:
    config = _json(path / "config.json")
    index = _json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return False
    nested = [config.get("text_config"), config.get("llm_config")]
    model_types = {config.get("model_type")}
    model_types.update(
        item.get("model_type") for item in nested if isinstance(item, dict)
    )
    if not ({"qwen4_exp", "qwen4_exp_text"} & model_types):
        return False
    shards = set(weight_map.values())
    return not complete or bool(shards) and all((path / shard).is_file() for shard in shards)


def installed_flashnext(root: Path | None = None) -> list[Path]:
    root = root or model_root()
    children = root.iterdir() if root.is_dir() else ()
    return [path for path in sorted(children) if path.is_dir() and flashnext_compatible(path)]


def resolve_flashnext(requested: str | os.PathLike[str] | None = None) -> Path:
    root = model_root()
    value = str(requested or os.environ.get("MACQWEN_FLASHNEXT_MODEL", "")).strip()
    if value and value != "auto":
        candidate = FLASHNEXT_ALIASES.get(value.lower(), value)
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = root / path
        if not flashnext_compatible(path):
            raise ValueError(f"incomplete or incompatible Flash-Next checkpoint: {path}")
        return path.resolve()

    choices = installed_flashnext(root)
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise ValueError("no complete Flash-Next checkpoint found; use --checkpoint PATH")
    lines = "\n".join(f"  --checkpoint {path}" for path in choices)
    raise ValueError("choose a Flash-Next checkpoint:\n" + lines)


def qwen27b_compatible(path: Path) -> bool:
    return _json(path / "config.json").get("vocab_size") == 248320


def installed_qwen27b(root: Path | None = None) -> list[Path]:
    root = root or model_root()
    children = root.iterdir() if root.is_dir() else ()
    return [path for path in sorted(children) if path.is_dir() and qwen27b_compatible(path)]


def resolve_qwen27b(requested: str | os.PathLike[str] | None = None) -> Path:
    root = model_root()
    value = str(requested or os.environ.get("MACQWEN_QWEN27B_MODEL", "")).strip()
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if not qwen27b_compatible(path):
            raise ValueError(f"incompatible Qwen27B checkpoint: {path}")
        return path.resolve()
    choices = installed_qwen27b(root)
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise ValueError("no compatible Qwen27B checkpoint found; use --model-path PATH")
    lines = "\n".join(f"  --model-path {path}" for path in choices)
    raise ValueError("choose a Qwen27B checkpoint:\n" + lines)
