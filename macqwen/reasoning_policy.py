"""Model-specific reasoning safeguards for interactive chat."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def has_reap_prune(checkpoint: str | os.PathLike[str] | None) -> bool:
    """Return whether a checkpoint declares REAP pruning metadata."""
    if not checkpoint:
        return False
    try:
        config = json.loads(Path(checkpoint).expanduser().joinpath("config.json").read_text())
    except (OSError, TypeError, ValueError):
        return False
    return isinstance(config.get("reap_prune"), dict)


def effective_reasoning_effort(
    requested: str,
    checkpoint: str | os.PathLike[str] | None = None,
    *,
    allow_reap_xhigh: bool | None = None,
) -> str:
    """Return the requested effort. REAP keeps xhigh and caps its budget."""
    return requested


def reap_xhigh_guard_active(
    requested: str,
    checkpoint: str | os.PathLike[str] | None = None,
) -> bool:
    """Return whether the default REAP xhigh budget cap applies."""
    return (
        requested == "xhigh"
        and has_reap_prune(checkpoint)
        and not _enabled(os.environ.get("MACQWEN_ALLOW_REAP_XHIGH"))
    )


def reasoning_effort_status(
    requested: str,
    checkpoint: str | os.PathLike[str] | None = None,
) -> str:
    """Describe requested and effective effort for `/status`."""
    return f"effort={requested}"


def _reap_budget_cap() -> int:
    raw = os.environ.get("MACQWEN_REAP_XHIGH_THINK_BUDGET", "4096")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 4096
    return value if value > 0 else 4096


def effective_think_budget(
    requested: int | None,
    effort: str,
    checkpoint: str | os.PathLike[str] | None = None,
) -> int | None:
    """Cap excessive REAP xhigh reasoning while preserving the effort level."""
    # An explicit -1 is the user's no-cap setting. Keep the older ``None``
    # sentinel available for callers that still request a shared total.
    if requested is not None and requested < 0:
        return requested
    if not reap_xhigh_guard_active(effort, checkpoint):
        return requested
    cap = _reap_budget_cap()
    # ``None`` is the preferences sentinel for an unlimited shared budget.
    # REAP xhigh must remain bounded even when the caller uses that sentinel.
    return cap if requested is None else min(requested, cap)


def think_budget_status(
    requested: int | None,
    effort: str,
    checkpoint: str | os.PathLike[str] | None = None,
) -> str:
    effective = effective_think_budget(requested, effort, checkpoint)
    if effective == requested:
        return (
            "think-tokens=unlimited"
            if requested is None or requested < 0 else f"think-tokens={requested}"
        )
    requested_text = "unlimited" if requested is None else str(requested)
    return f"think-tokens={effective} (requested {requested_text}; REAP xhigh budget cap)"
