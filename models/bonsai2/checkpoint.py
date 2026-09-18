"""Discover the supported Bonsai-2 ternary MLX checkpoint."""
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


def runtime_available(path: Path) -> bool:
    """Check for the bundled ternary runtime loader.

    Bonsai-2 declares ``model_type: prism_hadamard_qwen35`` and requires the
    loader bundled in its ``runtime/`` directory. Stock MLX loaders skip the
    activation Hadamard transform and return wrong output silently, so the
    backend must refuse to run when these files are absent.
    """
    runtime = path / "runtime"
    return (runtime / "requirements.txt").is_file() or any(
        runtime.glob("*.py")
    )


def compatible(path: Path) -> bool:
    config = _json(path / "config.json")
    if config.get("model_type") != "prism_hadamard_qwen35":
        return False
    required = (
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if not all((path / name).is_file() for name in required):
        return False
    index = _json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        shards = set(weight_map.values())
        if not shards or not all((path / shard).is_file() for shard in shards):
            return False
    elif not (path / "model.safetensors").is_file():
        return False
    return True


def installed(root: Path | None = None) -> list[Path]:
    root = root or model_root()
    children = root.iterdir() if root.is_dir() else ()
    return [path for path in sorted(children) if path.is_dir() and compatible(path)]


def resolve_bonsai2(requested: str | os.PathLike[str] | None = None) -> Path:
    root = model_root()
    value = str(requested or os.environ.get(CHECKPOINT_ENV, "")).strip()
    if value and value != "auto":
        path = Path(ALIASES.get(value.lower(), value)).expanduser()
        if not path.is_absolute():
            path = root / path
        if not compatible(path):
            raise ValueError(
                f"incomplete or incompatible Bonsai-2 checkpoint: {path}"
            )
        return path.resolve()

    choices = installed(root)
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise ValueError(
            "no complete Bonsai-2 checkpoint found; use --checkpoint PATH"
        )
    lines = "\n".join(f"  --checkpoint {path}" for path in choices)
    raise ValueError("choose a Bonsai-2 checkpoint:\n" + lines)
