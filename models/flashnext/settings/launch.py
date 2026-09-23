"""Backend-owned environment preset for the normal FlashNext chat."""
from __future__ import annotations


CHAT_ENV = {
    "FLASHNEXT_METAL_RUNTIME": "1",
    # The 20260917 comparison measured +13.4% paired mean decode speed with
    # identical token digests, so the Metal executor is the default now.
    "FLASHNEXT_METAL_G64": "1",
    "FLASHNEXT_SLAB_G64": "0",
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
    "FLASHNEXT_PROFILE_IO": "0",
    "FLASHNEXT_PREAD_CHUNK": "2",
    "FLASHNEXT_READ": "pread",
    # Hold the GPU clock during expert reads. 2026-09-22: +23.1% decode over
    # three paired 128-token arms at 8 pins, identical digests, and no thermal
    # warning over a 384-token answer. Preferences can turn it off.
    "FLASHNEXT_GPU_KEEPWARM": "1",
    # Exact opt-in bundle, promoted on 2026-09-23 at the user's request after
    # two fresh-process pairs at 128 tokens: +4.6% and +5.0%, identical digest
    # e19af44d5268e9d1, no resolution band yet. Every member keeps the token
    # digest; research.md records each one's own measurement.
    "FLASHNEXT_STREAM_EMBED": "1",
    "FLASHNEXT_COMPILE_HC": "1",
    "FLASHNEXT_COMPILE_NORM": "1",
    "FLASHNEXT_NORM_WEIGHT_CACHE": "1",
    "FLASHNEXT_IO_QOS": "user-interactive",
    # 16 since 2026-09-23: a decode token's 16 n-gram rows go to the read
    # pool too (part of the host and GPU-glue stack below).
    "FLASHNEXT_NGRAM_PARALLEL_MIN": "16",
    "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "1",
    "FLASHNEXT_QSA_SCATTER_DECODE": "1",
    "FLASHNEXT_OVERLAP": "0",
    "FLASHNEXT_RDAHEAD": "0",
    "FLASHNEXT_IO_WORKERS": "8",
    "FLASHNEXT_STREAM_PACK": "1",
    "FLASHNEXT_STREAM_PACK_CHUNK": "2",
    # Host and GPU-glue stack, promoted on 2026-09-23 at the user's request:
    # one sync per decode layer and compiled GDN q/k normalization and gated
    # norm (0 mismatches on 144 captured calls each). With the n-gram change
    # above it won 3 of 3 fresh-process pairs at 128 tokens, +6.6% inside an
    # 8.0% band, identical digest e19af44d5268e9d1.
    "FLASHNEXT_ONE_SYNC": "1",
    "FLASHNEXT_COMPILE_GDN": "1",
}


def apply_chat_environment(environment: dict[str, str]) -> dict[str, str]:
    """Apply defaults while preserving explicit process environment values."""
    for key, value in CHAT_ENV.items():
        environment.setdefault(key, value)
    return environment
