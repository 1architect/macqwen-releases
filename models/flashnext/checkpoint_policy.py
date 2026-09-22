"""Checkpoint-specific runtime policy overrides, keyed by identity.

The generic FlashNext defaults apply unless a verified checkpoint
identity carries a measured override. Identity comes from checkpoint
content hashes (see slab_pack.checkpoint_identity), never from
directory or file basenames.
"""
from __future__ import annotations

import os

# Vontra Qwen3.8-Flash-Next-MLX-4bit-MTP. Measured 2026-09-22: 8
# exact-quality pins beat 32 at identical tokens/routes, and 8 ties
# 0, on 72-token photosynthesis arms across three fresh processes.
VONTRA_4BIT_MTP_IDENTITY = (
    "b59d036b02d6425aeeb2019ca53a21934f2208e629f2f063a1f786b3edd346a6"
)

_CHECKPOINT_POLICIES = {
    VONTRA_4BIT_MTP_IDENTITY: {"resident_experts": 8},
}


def policy_for_identity(identity: str | None) -> dict:
    """Return the override map for a checkpoint identity, or {}."""
    if not identity:
        return {}
    return dict(_CHECKPOINT_POLICIES.get(str(identity), {}))


def identity_for_checkpoint(model_path) -> str | None:
    """Resolve a checkpoint identity without loading the model."""
    from .slab_pack import checkpoint_identity

    try:
        return checkpoint_identity(os.path.expanduser(str(model_path)))
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
