"""FlashNext routing profile settings."""
from __future__ import annotations

import os

from macqwen.backend_settings import Setting


def _float(raw: str) -> float:
    return float(raw)


def _budget(raw: str) -> float:
    value = _float(raw)
    if not 0 <= value <= 64:
        raise ValueError("pin-budget-gb must be between 0 and 64")
    return value


def _threshold(raw: str) -> float:
    value = _float(raw)
    if not 0.01 <= value <= 1.0:
        raise ValueError("threshold must be between 0.01 and 1.0")
    return value


def _threshold_display(backend):
    configured = backend.threshold
    if backend.routing_profile == "fast":
        return f"0.2 (configured {configured:g}; config ignored)"
    if backend.routing_profile == "fast-quality":
        return f"{configured:g} (warmup; tail threshold 0.2)"
    return configured


def _epsilon(raw: str) -> float:
    value = _float(raw)
    if not 0 <= value <= 1.0:
        raise ValueError("swap-epsilon must be between 0 and 1.0")
    return value


def _integer(raw: str, name: str, low: int = 1, high: int = 512) -> int:
    value = int(raw)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _routing(raw: str) -> str:
    options = ("standard", "fast", "fast-quality", "exact-quality", "cache-aware", "fused-quality")
    if raw not in options:
        raise ValueError(f"routing must be one of: {', '.join(options)}")
    return raw


def _set_attr(name):
    def setter(backend, value):
        setattr(backend, name, value)
    return setter


def _quality(backend):
    return backend.routing_profile in {"fast-quality", "exact-quality", "cache-aware", "fused-quality"}


def _cache_aware(backend):
    return backend.routing_profile == "cache-aware"


def _warn_cache(backend):
    return (
        "changes expert choices; exact-quality gave better answers in the current quality check"
        if _cache_aware(backend) else None
    )


def _warn_fused(backend):
    if backend.routing_profile != "fused-quality":
        return None
    warning = "experimental; its reasoning quality gate failed"
    if getattr(backend, "tape", None):
        warning += "; use /reset to enable its one-shot draft"
    return warning


def _warn_routing(backend):
    return _warn_cache(backend) or _warn_fused(backend)


def _effective_swap_epsilon(backend):
    store = getattr(backend, "store", None)
    captured = getattr(store, "_flashnext_env", None)
    if captured is not None:
        if captured.get("FLASHNEXT_SWAP_RESIDENT") == "1":
            return float(captured.get("FLASHNEXT_SWAP_EPSILON", backend.swap_epsilon))
        return getattr(backend, "swap_epsilon", 0.02)
    if os.environ.get("FLASHNEXT_SWAP_RESIDENT") == "1":
        return float(os.environ.get("FLASHNEXT_SWAP_EPSILON", backend.swap_epsilon))
    return getattr(backend, "swap_epsilon", 0.02)


def _swap_active(backend):
    store = getattr(backend, "store", None)
    captured = getattr(store, "_flashnext_env", None)
    flag = (
        captured.get("FLASHNEXT_SWAP_RESIDENT", "0")
        if captured is not None else os.environ.get("FLASHNEXT_SWAP_RESIDENT", "0")
    )
    return backend.routing_profile == "cache-aware" or flag == "1"


SETTINGS = (
    Setting(
        "routing", ("mode", "profile"), "exact-quality", _routing, "live",
        "routing", "public", "flashnext", lambda b: b.routing_profile,
        _set_attr("routing_profile"), lambda b: True, _warn_routing, None,
        __file__, "routing_profile", ("--routing-profile",), "routing_profile",
        {"choices": ("standard", "fast", "fast-quality", "exact-quality", "cache-aware", "fused-quality")},
    ),
    # The backend writes FLASHNEXT_TOPK_THRESHOLD internally while loading.
    # Keep provenance tied to CLI/default/live state instead of that internal
    # implementation variable.
    Setting("threshold", (), 0.85, _threshold, "next-turn", "routing", "public", "flashnext", _threshold_display, _set_attr("threshold"), lambda b: b.routing_profile != "fast", None, None, __file__, "threshold", ("--threshold",), "threshold", {"type": float}),
    Setting("swap-epsilon", (), 0.02, _epsilon, "next-turn", "routing", "public", "flashnext", _effective_swap_epsilon, _set_attr("swap_epsilon"), _swap_active, _warn_cache, "FLASHNEXT_SWAP_EPSILON", __file__, "swap_epsilon", ("--swap-epsilon",), "swap_epsilon", {"type": float}),
    Setting("resident-experts", ("pinned-experts",), 32, lambda raw: _integer(raw, "resident-experts"), "next-turn", "routing", "public", "flashnext", lambda b: b.resident_experts, _set_attr("resident_experts"), _quality, None, None, __file__, "resident_experts", ("--resident-experts", "--pinned-experts"), "resident_experts", {"type": int}),
    Setting("pin-budget-gb", (), 6.0, _budget, "next-turn", "routing", "public", "flashnext", lambda b: b.pin_budget_gb, _set_attr("pin_budget_gb"), _quality, None, None, __file__, "pin_budget_gb", ("--pin-budget-gb",), "pin_budget_gb", {"type": float}),
    Setting("tail-experts", (), 6, lambda raw: _integer(raw, "tail-experts"), "next-turn", "routing", "public", "flashnext", lambda b: b.tail_experts, _set_attr("tail_experts"), lambda b: b.routing_profile == "fast-quality", None, None, __file__, "tail_experts", ("--tail-experts",), "tail_experts", {"type": int}),
    Setting("tail-warmup", (), 8, lambda raw: _integer(raw, "tail-warmup", 1, 4096), "next-turn", "routing", "public", "flashnext", lambda b: b.tail_warmup, _set_attr("tail_warmup"), _quality, None, None, __file__, "tail_warmup", ("--tail-warmup",), "tail_warmup", {"type": int}),
    Setting("pinned-now", (), 0, lambda raw: int(raw), "read-only", "routing", "public", "flashnext", lambda b: sum(len(v) for v in getattr(getattr(b, "routing", None), "pinned", {}).values()), None, _quality, None, None, __file__),
)
