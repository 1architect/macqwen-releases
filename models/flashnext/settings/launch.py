"""Backend-owned environment preset for the normal FlashNext chat."""
from __future__ import annotations


CHAT_ENV = {
    "FLASHNEXT_METAL_RUNTIME": "1",
    # The 20260917 comparison measured +13.4% paired mean decode speed with
    # identical token digests, so the Metal executor is the default now.
    "FLASHNEXT_METAL_G64": "1",
    "FLASHNEXT_SLAB_G64": "0",
    "FLASHNEXT_SLAB": "0",
    "FLASHNEXT_SLAB_GLOBAL": "60",
    "FLASHNEXT_SLAB_PACK": "1",
    "FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING": "0",
    "FLASHNEXT_SLAB_POLICY": "skew",
    "FLASHNEXT_SLAB_PROFILE": "frozen",
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
    # Hold the GPU clock during expert reads. 2026-09-22: +23.1% decode over
    # three paired 128-token arms at 8 pins, identical digests, and no thermal
    # warning over a 384-token answer. Preferences can turn it off.
    "FLASHNEXT_GPU_KEEPWARM": "1",
}


def apply_chat_environment(environment: dict[str, str]) -> dict[str, str]:
    """Apply defaults while preserving explicit process environment values."""
    for key, value in CHAT_ENV.items():
        environment.setdefault(key, value)
    return environment
