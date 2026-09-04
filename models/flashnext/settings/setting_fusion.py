"""FlashNext one-shot fusion settings."""
from __future__ import annotations

import os

from macqwen.backend_settings import Setting


def _int(raw, name, low=1, high=262144):
    value = int(raw)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _margin(raw):
    value = float(raw)
    if value < 0:
        raise ValueError("fusion-min-margin must be zero or greater")
    return value


def _active(b):
    return b.routing_profile == "fused-quality"


def _set(name):
    return lambda b, v: setattr(b, name, v)


SETTINGS = (
    Setting("fusion-block", (), 23, lambda v: _int(v, "fusion-block", 1, 128), "next-turn", "fusion", "public", "flashnext", lambda b: b.fusion_block, _set("fusion_block"), _active, None, None, __file__, "fusion_block", ("--fusion-block",), "fusion_block", {"type": int}),
    Setting("fusion-min-margin", (), 1.0, _margin, "next-turn", "fusion", "public", "flashnext", lambda b: b.fusion_min_margin, _set("fusion_min_margin"), _active, None, None, __file__, "fusion_min_margin", ("--fusion-min-margin",), "fusion_min_margin", {"type": float}),
    Setting("fusion-min-block", (), 20, lambda v: _int(v, "fusion-min-block", 1, 128), "next-turn", "fusion", "public", "flashnext", lambda b: b.fusion_min_block, _set("fusion_min_block"), _active, None, None, __file__, "fusion_min_block", ("--fusion-min-block",), "fusion_min_block", {"type": int}),
    Setting("fusion-margin-tokens", (), 8, lambda v: _int(v, "fusion-margin-tokens", 0, 128), "next-turn", "fusion", "public", "flashnext", lambda b: b.fusion_margin_tokens, _set("fusion_margin_tokens"), _active, None, None, __file__, "fusion_margin_tokens", ("--fusion-margin-tokens",), "fusion_margin_tokens", {"type": int}),
    Setting("fusion-max-prompt", (), 512, lambda v: _int(v, "fusion-max-prompt", 1, 262144), "next-turn", "fusion", "public", "flashnext", lambda b: b.fusion_max_prompt, _set("fusion_max_prompt"), _active, None, None, __file__, "fusion_max_prompt", ("--fusion-max-prompt",), "fusion_max_prompt", {"type": int}),
    Setting("fusion-model", (), "", str, "next-turn", "fusion", "public", "flashnext", lambda b: b.fusion_model, lambda b, v: setattr(b, "fusion_model", os.path.expanduser(v)), _active, None, None, __file__, "fusion_model", ("--fusion-model",), "fusion_model", {}),
)
