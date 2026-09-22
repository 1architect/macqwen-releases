"""Checkpoint-specific runtime policy overrides, keyed by content identity.

The generic FlashNext defaults apply unless a checkpoint's content identity
carries a measured override. The identity hashes the index, the config and
every referenced shard's name, size and safetensors header. It never uses the
directory path, inode or timestamps, so the same checkpoint keeps its policy
under another spelling of its path (APFS ignores case), after a copy or
re-download, and after a metadata-only change such as chmod.

This differs on purpose from ``slab_pack.checkpoint_identity``, which keys
local cache files and includes the location.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

_CONTENT_IDENTITY_VERSION = b"flashnext-checkpoint-content-v1"
# A safetensors header larger than this is not a real header.
_HEADER_LIMIT = 100 * 1024 * 1024

# Vontra Qwen3.8-Flash-Next-MLX-4bit-MTP. Measured 2026-09-22: 8
# exact-quality pins beat 32 at identical tokens/routes, and 8 ties
# 0, on 72-token photosynthesis arms across three fresh processes.
VONTRA_4BIT_MTP_IDENTITY = (
    "0e8c98dd61cf8aefc56b9c1a8f6dabfbc4f978824060ea1c8f0f3111a22dae6b"
)

_CHECKPOINT_POLICIES = {
    VONTRA_4BIT_MTP_IDENTITY: {"resident_experts": 8},
}


def policy_for_identity(identity: str | None) -> dict:
    """Return the override map for a checkpoint identity, or {}."""
    if not identity:
        return {}
    return dict(_CHECKPOINT_POLICIES.get(str(identity), {}))


def content_identity(model_dir) -> str:
    """Hash what the checkpoint contains, not where it is.

    Reads the index, the config and each shard's safetensors header, a few
    megabytes at most. Tensor payloads are not read: the header names every
    tensor with its dtype, shape and byte range, and the file size bounds the
    payload.
    """
    model_path = Path(os.path.expanduser(str(model_dir)))
    index_bytes = (model_path / "model.safetensors.index.json").read_bytes()
    shards = sorted(set(json.loads(index_bytes).get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"checkpoint index has no shards: {model_path}")
    digest = hashlib.sha256(_CONTENT_IDENTITY_VERSION)
    digest.update(index_bytes)
    config_path = model_path / "config.json"
    if config_path.is_file():
        digest.update(b"config.json")
        digest.update(config_path.read_bytes())
    for name in shards:
        path = model_path / str(name)
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"truncated safetensors shard: {path}")
            length = struct.unpack("<Q", raw_length)[0]
            if length > _HEADER_LIMIT:
                raise ValueError(f"invalid safetensors header: {path}")
            header = handle.read(length)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        digest.update(header)
    return digest.hexdigest()


def identity_for_checkpoint(model_path) -> str | None:
    """Resolve a checkpoint's content identity without loading the model."""
    try:
        return content_identity(model_path)
    except (OSError, TypeError, ValueError):
        return None


def resolve_resident_experts(explicit, model_path=None):
    """Resolve effective resident-experts with explicit > policy > None.

    Returns None when neither an explicit value nor a checkpoint
    policy applies, so the caller keeps the generic default.
    """
    if explicit is not None:
        return int(explicit)
    if model_path is None:
        return None
    policy = policy_for_identity(identity_for_checkpoint(model_path))
    if "resident_experts" in policy:
        return int(policy["resident_experts"])
    return None
