"""Utilities for short-lived resident draft models."""
from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map


def materialize_model(model) -> None:
    """Copy mapped parameters into anonymous Metal-backed memory."""
    original = model.parameters()
    copied = tree_map(
        lambda value: value + mx.zeros((), dtype=value.dtype),
        original,
    )
    mx.eval(copied)
    model.update(copied)
    del original, copied
    mx.clear_cache()
