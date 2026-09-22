"""Checkpoint capability detection for FlashNext.

Derive layout facts from config, index, and tensor names. Never guess from
directory names, group size, or expert count.

MTP source is one of: ``none``, ``sidecar``, ``indexed``.
If both sidecar and indexed MTP exist, fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

MTP_PREFIX = "language_model.mtp."
MTP_SIDECAR_NAME = "model-mtp.safetensors"

MTP_NONE = "none"
MTP_SIDECAR = "sidecar"
MTP_INDEXED = "indexed"

# Structural core the runtime needs. Names are prefixes/suffixes, not a
# rename table. Missing entries mean the checkpoint cannot drive mtp.py.
REQUIRED_MTP_SUBSTRINGS = (
    "language_model.mtp.pre_fc_norm_embedding.weight",
    "language_model.mtp.pre_fc_norm_hidden.weight",
    "language_model.mtp.fc_embedding.weight",
    "language_model.mtp.fc_hidden.weight",
    "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
    "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight",
    "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight",
)


@dataclass
class CheckpointCaps:
    compatible: bool = False
    model_type: str = ""
    layers: int = 0
    hidden_size: int = 0
    experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0
    backbone_group: int = 0
    backbone_bits: int = 0
    mtp_source: str = MTP_NONE
    mtp_keys: list = field(default_factory=list)
    mtp_shards: list = field(default_factory=list)
    norm_convention: str = ""  # "", "one", or "zero"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def mtp_keys_from_weight_map(weight_map: dict) -> list[str]:
    if not isinstance(weight_map, dict):
        return []
    return sorted(key for key in weight_map if key.startswith(MTP_PREFIX))


def validate_mtp_structure(available_keys) -> list[str]:
    """Return missing required MTP entries. Empty means structurally complete."""
    available = set(available_keys)
    return [name for name in REQUIRED_MTP_SUBSTRINGS if name not in available]


def detect_mtp_source(model_dir: str | os.PathLike, weight_map: dict | None = None) -> tuple[str, list[str], list[str]]:
    """Detect MTP storage form without loading weights.

    Returns (source, mtp_keys, mtp_shards). Raises ValueError when both
    sidecar and indexed MTP exist, because silently picking one is unsafe.
    """
    path = Path(os.path.expanduser(str(model_dir)))
    sidecar_exists = (path / MTP_SIDECAR_NAME).is_file()
    if weight_map is None:
        weight_map = _read_json(path / "model.safetensors.index.json").get("weight_map", {})
    if not isinstance(weight_map, dict):
        weight_map = {}
    indexed_keys = mtp_keys_from_weight_map(weight_map)
    indexed_shards = sorted({str(weight_map[key]) for key in indexed_keys if isinstance(weight_map.get(key), str)})
    if sidecar_exists and indexed_keys:
        raise ValueError(
            f"checkpoint has both {MTP_SIDECAR_NAME} and {len(indexed_keys)} indexed "
            f"{MTP_PREFIX}* tensors; refusing to pick one. Remove or prove "
            "they are identical, then retry."
        )
    if sidecar_exists:
        return MTP_SIDECAR, [], [MTP_SIDECAR_NAME]
    if indexed_keys:
        return MTP_INDEXED, indexed_keys, indexed_shards
    return MTP_NONE, [], []


def is_mtp_key(name: str) -> bool:
    return name.startswith(MTP_PREFIX)


def describe_checkpoint(model_dir: str | os.PathLike) -> CheckpointCaps:
    """Describe one FlashNext checkpoint from metadata only. No MLX import."""
    path = Path(os.path.expanduser(str(model_dir)))
    caps = CheckpointCaps()
    config = _read_json(path / "config.json")
    text = config.get("text_config", {})
    if not isinstance(text, dict):
        text = {}
    model_types = {config.get("model_type"), text.get("model_type")}
    caps.model_type = str(text.get("model_type") or config.get("model_type") or "")
    caps.compatible = bool({"qwen4_exp", "qwen4_exp_text"} & {m for m in model_types if m})
    if not caps.compatible:
        return caps
    try:
        caps.layers = int(text.get("num_hidden_layers") or 0)
    except (TypeError, ValueError):
        caps.layers = 0
    try:
        caps.hidden_size = int(text.get("hidden_size") or 0)
    except (TypeError, ValueError):
        caps.hidden_size = 0
    try:
        caps.experts = int(text.get("num_experts") or 0)
    except (TypeError, ValueError):
        caps.experts = 0
    try:
        caps.top_k = int(text.get("num_experts_per_tok") or 0)
    except (TypeError, ValueError):
        caps.top_k = 0
    try:
        caps.moe_intermediate = int(text.get("moe_intermediate_size") or 0)
    except (TypeError, ValueError):
        caps.moe_intermediate = 0
    quant = config.get("quantization") or {}
    if isinstance(quant, dict):
        try:
            caps.backbone_group = int(quant.get("group_size") or 0)
        except (TypeError, ValueError):
            caps.backbone_group = 0
        try:
            caps.backbone_bits = int(quant.get("bits") or 0)
        except (TypeError, ValueError):
            caps.backbone_bits = 0
    for source in (config, text):
        if isinstance(source, dict):
            raw = source.get("norm_convention") or source.get("rms_norm_convention")
            if isinstance(raw, str):
                normalized = raw.lower().replace("_", "-")
                if normalized in {"zero", "zero-centered"}:
                    caps.norm_convention = "zero"
                elif normalized in {"one", "one-centered"}:
                    caps.norm_convention = "one"
    weight_map = _read_json(path / "model.safetensors.index.json").get("weight_map", {})
    try:
        caps.mtp_source, caps.mtp_keys, caps.mtp_shards = detect_mtp_source(path, weight_map)
    except ValueError:
        raise
    return caps
