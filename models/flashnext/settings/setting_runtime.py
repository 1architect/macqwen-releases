"""Storage and Metal controls visible to FlashNext."""
from __future__ import annotations

import os

from macqwen.backend_settings import Setting


def env_reader(key, default, parser=str):
    def read(_backend):
        backend = _backend
        store = getattr(backend, "store", None)
        if store is not None and hasattr(store, "_flashnext_env"):
            raw = store._flashnext_env.get(key)
        else:
            raw = os.environ.get(key)
        return parser(raw) if raw is not None else default
    return read


def env_active(key):
    def active(backend):
        store = getattr(backend, "store", None)
        if store is not None and hasattr(store, "_flashnext_env"):
            return store._flashnext_env.get(key, "0") == "1"
        return os.environ.get(key, "0") == "1"
    return active


def store_reader(attribute, key, default, parser=str):
    def read(backend):
        store = getattr(backend, "store", None)
        if store is not None and hasattr(store, attribute):
            return getattr(store, attribute)
        return env_reader(key, default, parser)(backend)
    return read


def live_env_reader(key, default, parser=str):
    return lambda _backend: parser(os.environ.get(key, default))


def env_setter(key, attribute=None):
    def setter(backend, value):
        os.environ[key] = str(value)
        store = getattr(backend, "store", None)
        if attribute and store is not None:
            setattr(store, attribute, value)
    return setter


def choice(options, name):
    def parse(raw):
        if raw not in options:
            raise ValueError(f"{name} must be one of: {', '.join(options)}")
        return raw
    return parse


def integer(raw):
    return int(raw)


SETTINGS = (
    Setting("metal-runtime", ("FLASHNEXT_METAL_RUNTIME",), "0", choice(("0", "1"), "metal-runtime"), "live", "runtime", "public", "flashnext", live_env_reader("FLASHNEXT_METAL_RUNTIME", "0"), env_setter("FLASHNEXT_METAL_RUNTIME"), lambda _b: os.environ.get("FLASHNEXT_METAL_RUNTIME", "0") == "1", None, "FLASHNEXT_METAL_RUNTIME", __file__),
    Setting("slab-global", ("FLASHNEXT_SLAB_GLOBAL",), 0, integer, "startup", "storage", "public", "flashnext", env_reader("FLASHNEXT_SLAB_GLOBAL", 0, integer), None, lambda _b: True, None, "FLASHNEXT_SLAB_GLOBAL", __file__),
    Setting("slab-pack", ("FLASHNEXT_SLAB_PACK",), "0", choice(("0", "1"), "slab-pack"), "startup", "storage", "public", "flashnext", env_reader("FLASHNEXT_SLAB_PACK", "0"), None, env_active("FLASHNEXT_SLAB_PACK"), None, "FLASHNEXT_SLAB_PACK", __file__),
    Setting("slab-policy", ("FLASHNEXT_SLAB_POLICY",), "skew", str, "startup", "storage", "public", "flashnext", env_reader("FLASHNEXT_SLAB_POLICY", "skew"), None, lambda _b: True, None, "FLASHNEXT_SLAB_POLICY", __file__),
    Setting("fused-shared", ("FLASHNEXT_FUSED_SHARED",), "1", choice(("0", "1"), "fused-shared"), "startup", "runtime", "public", "flashnext", env_reader("FLASHNEXT_FUSED_SHARED", "1"), None, env_active("FLASHNEXT_FUSED_SHARED"), None, "FLASHNEXT_FUSED_SHARED", __file__),
    Setting("fused-shared-parts", ("FLASHNEXT_FUSED_SHARED_PARTS",), "0", choice(("0", "1"), "fused-shared-parts"), "startup", "runtime", "public", "flashnext", env_reader("FLASHNEXT_FUSED_SHARED_PARTS", "0"), None, env_active("FLASHNEXT_FUSED_SHARED_PARTS"), None, "FLASHNEXT_FUSED_SHARED_PARTS", __file__),
    Setting("fused-up-swiglu", ("FLASHNEXT_FUSED_UP_SWIGLU",), "0", choice(("0", "1"), "fused-up-swiglu"), "startup", "runtime", "public", "flashnext", env_reader("FLASHNEXT_FUSED_UP_SWIGLU", "0"), None, env_active("FLASHNEXT_FUSED_UP_SWIGLU"), None, "FLASHNEXT_FUSED_UP_SWIGLU", __file__),
    Setting("stream-pack", ("FLASHNEXT_STREAM_PACK",), "0", str, "startup", "storage", "public", "flashnext", env_reader("FLASHNEXT_STREAM_PACK", "0"), None, lambda _b: True, None, "FLASHNEXT_STREAM_PACK", __file__),
    Setting("pread-chunk", ("FLASHNEXT_PREAD_CHUNK",), 2, integer, "live", "storage", "public", "flashnext", store_reader("_pread_chunk", "FLASHNEXT_PREAD_CHUNK", 2, integer), env_setter("FLASHNEXT_PREAD_CHUNK", "_pread_chunk"), lambda _b: True, None, "FLASHNEXT_PREAD_CHUNK", __file__),
    Setting("io-workers", ("FLASHNEXT_IO_WORKERS",), 16, integer, "startup", "storage", "public", "flashnext", env_reader("FLASHNEXT_IO_WORKERS", 16, integer), None, lambda _b: True, None, "FLASHNEXT_IO_WORKERS", __file__),
    Setting("read-mode", ("FLASHNEXT_READ",), "pread", str, "live", "storage", "public", "flashnext", store_reader("_read_mode", "FLASHNEXT_READ", "pread"), env_setter("FLASHNEXT_READ", "_read_mode"), lambda _b: True, None, "FLASHNEXT_READ", __file__),
    Setting("model-path", (), "", str, "read-only", "runtime", "public", "flashnext", lambda b: getattr(b, "model_path", ""), None, lambda _b: True, None, None, __file__),
    Setting("session-dir", (), "~/.cache/flashnext/sessions", str, "startup", "runtime", "public", "flashnext", lambda b: getattr(b, "session_dir", "~/.cache/flashnext/sessions"), None, lambda _b: True, None, None, __file__),
    Setting("pinned-bytes", (), 0, int, "read-only", "routing", "public", "flashnext", lambda b: getattr(getattr(b, "routing", None), "pinned_bytes", 0), None, lambda _b: True, None, None, __file__),
)
