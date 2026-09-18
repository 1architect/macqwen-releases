"""Bonsai-2-local cache policy helpers."""
from __future__ import annotations

from collections.abc import Iterable


def _kv_cache_classes() -> tuple:
    """Return the KV cache classes our runtime can actually construct.

    The loaded model builds mlx-vlm cache objects, while checkpoint-free
    tests construct mlx-lm ones. Both expose the same ``step`` growth
    policy; anything else is rejected.
    """
    from mlx_lm.models.cache import KVCache as LmkvCache

    try:
        from mlx_vlm.models.cache import KVCache as VlmKvCache
    except ImportError:
        VlmKvCache = None
    classes = [LmkvCache]
    if VlmKvCache is not None and VlmKvCache is not LmkvCache:
        classes.append(VlmKvCache)
    return tuple(classes)


SUPPORTED_CACHE_STEPS = (256, 1024)


def set_kv_cache_step(caches, step: int) -> None:
    """Set the growth step on Bonsai-2's ordinary cache instances.

    The upstream ``KVCache.step`` is a class default.  Assigning on each
    instance keeps the process-global default unchanged and leaves other cache
    types, such as sliding-window caches, alone.
    """
    classes = _kv_cache_classes()
    if step not in SUPPORTED_CACHE_STEPS:
        raise ValueError(
            f"unsupported Bonsai-2 KV cache step {step}; "
            f"choose one of {SUPPORTED_CACHE_STEPS}"
        )
    if isinstance(caches, classes):
        caches = (caches,)
    for cache in caches:
        if isinstance(cache, classes):
            cache.step = step
