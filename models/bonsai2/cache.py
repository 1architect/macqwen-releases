"""Bonsai-2-local cache policy helpers."""
from __future__ import annotations

from collections.abc import Iterable

from mlx_lm.models.cache import KVCache


SUPPORTED_CACHE_STEPS = (256, 1024)


def set_kv_cache_step(caches: KVCache | Iterable[object], step: int) -> None:
    """Set the growth step on Bonsai-2's ordinary cache instances.

    The upstream ``KVCache.step`` is a class default.  Assigning on each
    instance keeps the process-global default unchanged and leaves other cache
    types, such as sliding-window caches, alone.
    """
    if step not in SUPPORTED_CACHE_STEPS:
        raise ValueError(
            f"unsupported Bonsai-2 KV cache step {step}; "
            f"choose one of {SUPPORTED_CACHE_STEPS}"
        )
    if isinstance(caches, KVCache):
        caches = (caches,)
    for cache in caches:
        if isinstance(cache, KVCache):
            cache.step = step
