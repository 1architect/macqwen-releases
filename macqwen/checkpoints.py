"""Find compatible local model checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path


FLASHNEXT_ALIASES = {
    "oq3": "Qwen3.8-Flash-Next-MLX-oQ3-MTP",
    "oq3-mtp": "Qwen3.8-Flash-Next-MLX-oQ3-MTP",
    "oq4": "Qwen3.8-Flash-Next-MLX-oQ4",
    "vontra-mtp": "Qwen3.8-Flash-Next-MLX-4bit-MTP",
}
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def _sane_shard_name(value) -> bool:
    return (
        isinstance(value, str)
        and value.endswith(".safetensors")
        and "/" not in value
        and "\\" not in value
        and value not in (".", "..")
        and not value.startswith(".")
        and not Path(value).is_absolute()
    )


def _missing_shards(path: Path) -> list[str]:
    index_path = path / "model.safetensors.index.json"
    index = _json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [index_path.name]
    values = list(weight_map.values())
    if not all(_sane_shard_name(value) for value in values):
        return [index_path.name]
    return [name for name in sorted(set(values)) if not (path / name).is_file()]


def _checkpoint_error(label: str, path: Path, missing: list[str]) -> ValueError:
    details = ", ".join(missing) if missing else "model metadata"
    return ValueError(
        f"incomplete or incompatible {label} checkpoint: {path}\n"
        f"missing or invalid: {details}\n"
        f"repair: resume the checkpoint download into {path} and try again"
    )


def _has_chat_template(path: Path) -> bool:
    if (path / "chat_template.jinja").is_file():
        return True
    return bool(_json(path / "tokenizer_config.json").get("chat_template"))


def _valid_bf16_asset(path: Path, shape) -> bool:
    return (
        isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(value, int) and value > 0 for value in shape)
        and path.is_file()
        and path.stat().st_size == shape[0] * shape[1] * 2
    )


def model_root() -> Path:
    return Path(os.environ.get("MACQWEN_MODEL_ROOT", "~/models")).expanduser()


def installed_checkpoints(root: Path | None = None) -> list[tuple[str, Path]]:
    """Return every complete checkpoint that this checkout can load."""
    from models.bonsai2.checkpoint import installed as installed_bonsai2
    from models.k2_horizon.checkpoint import installed as installed_k2

    choices = [("flashnext", path) for path in installed_flashnext(root)]
    choices.extend(("bonsai2", path) for path in installed_bonsai2(root))
    choices.extend(("k2-horizon", path) for path in installed_k2(root))
    choices.extend(("qwen27b", path) for path in installed_qwen27b(root))
    return sorted(choices, key=lambda item: (item[0], str(item[1])))


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def flashnext_compatible(path: Path, complete: bool = True) -> bool:
    config = _json(path / "config.json")
    if not all((path / name).is_file() for name in _TOKENIZER_FILES):
        return False
    nested = [config.get("text_config"), config.get("llm_config")]
    model_types = {config.get("model_type")}
    model_types.update(
        item.get("model_type") for item in nested if isinstance(item, dict)
    )
    if not ({"qwen4_exp", "qwen4_exp_text"} & model_types):
        return False
    index = _json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    values = list(weight_map.values())
    if not all(_sane_shard_name(value) for value in values):
        return False
    return not complete or all((path / shard).is_file() for shard in set(values))


def installed_flashnext(root: Path | None = None) -> list[Path]:
    root = root or model_root()
    children = root.iterdir() if root.is_dir() else ()
    return [path for path in sorted(children) if path.is_dir() and flashnext_compatible(path)]


def resolve_flashnext(
    requested: str | os.PathLike[str] | None = None,
    *,
    allow_stale_fallback: bool = False,
) -> Path:
    root = model_root()
    value = str(requested or os.environ.get("MACQWEN_FLASHNEXT_MODEL", "")).strip()
    if value and value != "auto":
        candidate = FLASHNEXT_ALIASES.get(value.lower(), value)
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = root / path
        if not flashnext_compatible(path):
            if allow_stale_fallback and not path.exists():
                choices = installed_flashnext(root)
                if len(choices) == 1:
                    return choices[0].resolve()
            missing = [name for name in _TOKENIZER_FILES if not (path / name).is_file()]
            if path.is_dir() and _json(path / "config.json"):
                missing.extend(_missing_shards(path))
            raise _checkpoint_error("Flash-Next", path, sorted(set(missing)))
        return path.resolve()

    choices = installed_flashnext(root)
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise ValueError("no complete Flash-Next checkpoint found; use --checkpoint PATH")
    lines = "\n".join(f"  --checkpoint {path}" for path in choices)
    raise ValueError("choose a Flash-Next checkpoint:\n" + lines)


def qwen27b_compatible(path: Path) -> bool:
    config = _json(path / "config.json")
    if config.get("vocab_size") != 248320:
        return False
    if not all((path / name).is_file() for name in _TOKENIZER_FILES) or not _has_chat_template(path):
        return False
    if _missing_shards(path):
        return False
    assets = path / "bf16-ends"
    meta = _json(assets / "meta.json")
    return all(
        _valid_bf16_asset(assets / filename, meta.get(shape_name))
        for filename, shape_name in (
            ("embed.bf16", "embed_shape"),
            ("head.bf16", "head_shape"),
        )
    )


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
            missing = [name for name in _TOKENIZER_FILES if not (path / name).is_file()]
            if not _has_chat_template(path):
                missing.append("chat template")
            if path.is_dir() and _json(path / "config.json").get("vocab_size") == 248320:
                missing.extend(_missing_shards(path))
                meta = _json(path / "bf16-ends" / "meta.json")
                missing.extend(
                    f"bf16-ends/{name}"
                    for name in ("embed.bf16", "head.bf16", "meta.json")
                    if not (path / "bf16-ends" / name).is_file()
                )
                if not _valid_bf16_asset(
                    path / "bf16-ends" / "embed.bf16", meta.get("embed_shape")
                ):
                    missing.append("bf16-ends/embed.bf16")
                if not _valid_bf16_asset(
                    path / "bf16-ends" / "head.bf16", meta.get("head_shape")
                ):
                    missing.append("bf16-ends/head.bf16")
                if not meta:
                    missing.append("bf16-ends/meta.json")
            if _json(path / "config.json").get("vocab_size") != 248320:
                missing.append("compatible config.json")
            raise _checkpoint_error("Qwen27B", path, sorted(set(missing)))
        return path.resolve()
    choices = installed_qwen27b(root)
    if len(choices) == 1:
        return choices[0].resolve()
    if not choices:
        raise ValueError("no compatible Qwen27B checkpoint found; use --model-path PATH")
    lines = "\n".join(f"  --model-path {path}" for path in choices)
    raise ValueError("choose a Qwen27B checkpoint:\n" + lines)
