"""Backend-owned environment preset for the normal FlashNext chat."""
from __future__ import annotations


CHAT_ENV = {
    "FLASHNEXT_METAL_RUNTIME": "1",
    "FLASHNEXT_SLAB": "0",
    "FLASHNEXT_SLAB_GLOBAL": "60",
    "FLASHNEXT_SLAB_PACK": "1",
    "FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING": "0",
    "FLASHNEXT_SLAB_POLICY": "skew",
    "FLASHNEXT_SLAB_MIN_SLOTS": "4",
    "FLASHNEXT_SLAB_MAX_SLOTS": "6",
    "FLASHNEXT_SLAB_NUM_LAYERS": "12",
    "FLASHNEXT_FUSED_SHARED": "1",
    "FLASHNEXT_FUSED_SHARED_PARTS": "0",
    "FLASHNEXT_FUSED_UP_SWIGLU": "1",
    "FLASHNEXT_STREAM_PACK": "0",
    "FLASHNEXT_PROFILE_IO": "0",
    "FLASHNEXT_PREAD_CHUNK": "2",
    "FLASHNEXT_IO_WORKERS": "16",
    "FLASHNEXT_READ": "pread",
}


def apply_chat_environment(environment: dict[str, str]) -> dict[str, str]:
    """Apply defaults while preserving explicit process environment values."""
    for key, value in CHAT_ENV.items():
        environment.setdefault(key, value)
    return environment
