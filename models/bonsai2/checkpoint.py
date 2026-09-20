"""Discover the supported Bonsai-2 ternary MLX checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .settings import ALIASES, CHECKPOINT_ENV

_RUNTIME_FILES = (
    "runtime/runtime.py",
    "runtime/artifact.py",
    "runtime/vision_artifact.py",
    "runtime/codec.py",
)


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
    backend must refuse to run when these files are absent. A requirements
    file or an unrelated script is not enough: both entry points our loader
    imports must exist, and the transform module must define Packed.
    """
    runtime = path / "runtime"
    if not all((path / name).is_file() for name in _RUNTIME_FILES):
        return False
    try:
        return "class Packed" in (runtime / "runtime.py").read_text()
    except OSError:
        return False


def _sane_shard_name(value) -> bool:
    """Accept only plain shard filenames from an index weight map."""
    return (
        isinstance(value, str)
        and value.endswith(".safetensors")
        and "/" not in value
        and "\\" not in value
        and value not in (".", "..")
        and not value.startswith(".")
        and not Path(value).is_absolute()
    )


def compatible(path: Path) -> bool:
    config = _json(path / "config.json")
    if config.get("model_type") != "prism_hadamard_qwen35":
        return False
    required = (
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if not all((path / name).is_file() for name in required) or not runtime_available(path):
        return False
    index = _json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        # Values must be plain filenames before set() touches them: a
        # list or dict value would crash discovery with TypeError instead
        # of marking the checkpoint incompatible.
        values = list(weight_map.values())
        if not all(isinstance(value, str) for value in values):
            return False
        shards = set(values)
        if (
            not shards
            or not all(_sane_shard_name(shard) for shard in shards)
            or not all((path / shard).is_file() for shard in shards)
        ):
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
            missing = [
                name for name in ("config.json", *required_files(path))
                if not (path / name).is_file()
            ]
            if _json(path / "config.json").get("model_type") != "prism_hadamard_qwen35":
                missing.append("compatible config.json")
            raise ValueError(
                f"incomplete or incompatible Bonsai-2 checkpoint: {path}\n"
                f"missing or invalid: {', '.join(sorted(set(missing)))}\n"
                f"repair: resume the checkpoint download into {path} and try again"
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


def required_files(path: Path) -> list[str]:
    """Return files that make the checkpoint loadable, including its loader."""
    required = ["tokenizer.json", "tokenizer_config.json", *_RUNTIME_FILES]
    index = _json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        values = list(weight_map.values())
        if all(isinstance(value, str) and _sane_shard_name(value) for value in values):
            required.extend(sorted(set(values)))
        else:
            required.append("model.safetensors.index.json")
    elif not (path / "model.safetensors").is_file():
        required.append("model.safetensors")
    return required
