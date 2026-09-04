"""Startup settings for the Qwen27B backend."""
from __future__ import annotations

from macqwen.backend_settings import Setting


def _env(name, default):
    return lambda backend: getattr(backend, "_startup_settings", {}).get(name, default)


def _int(raw):
    return int(raw)


SETTINGS = tuple(
    Setting(name, (), default, parser, "startup", "startup", "public", "qwen27b", _env(name, default), None, lambda _b: True, None, None, __file__, name.replace("-", "_"), (f"--{name}",), name.replace("-", "_"), kwargs)
    for name, default, parser, kwargs in (
        ("prefill-step-size", 512, _int, {"type": int}),
        ("kv-bits", "off", str, {}),
        ("kv-group-size", 64, _int, {"type": int}),
        ("quantized-kv-start", 8192, _int, {"type": int}),
        ("paged", False, lambda raw: raw == "1", {"action": "store_true"}),
        ("page-size", 256, _int, {"type": int}),
        ("top-k-pages", 16, _int, {"type": int}),
        ("resident-pages", 24, _int, {"type": int}),
        ("min-context", 16384, _int, {"type": int}),
        ("temperature", 0.0, float, {"type": float}),
        ("repetition-penalty", 1.12, float, {"type": float}),
        ("repetition-context-size", 512, _int, {"type": int}),
        ("backtrack-bias", 0.0, float, {"type": float}),
        ("shortlist-k", 1024, _int, {"type": int}),
        ("lm-head-opt", False, lambda raw: raw == "1", {"action": "store_true"}),
        ("layer-indices", None, str, {}),
        ("wired-limit-gb", None, float, {"type": float}),
        ("spill-dir", "/tmp/frankenstein_pages", str, {}),
        ("session-dir", "~/.frankenstein/sessions", str, {}),
        ("bf16-ends", False, lambda raw: raw == "1", {"action": "store_true"}),
    )
)
