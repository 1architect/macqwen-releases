"""N-gram embedding rows read straight from the checkpoint.

The hashed trigram table is 19.2 GB across 128 shards, but a row is only 60
bytes and a token touches a handful of them. Holding a shard resident to read
one row wastes 150 MB, so rows are fetched individually and dequantized.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .store import SafeTensorStore


def infer_quantization(store: SafeTensorStore, prefix: str, dims: int):
    """Recover (group_size, bits) from the packed shapes."""
    packed = store.shape(f"{prefix}.weight")[-1]
    groups = store.shape(f"{prefix}.scales")[-1]
    bits = packed * 32 // dims
    group_size = dims // groups
    return group_size, bits


class StreamingQuantizedEmbedding(nn.Module):
    """Quantized embedding whose rows live on disk."""

    def __init__(
        self,
        store: SafeTensorStore,
        prefix: str,
        dims: int,
        mode: str = "affine",
        capacity: int = 0,
    ):
        super().__init__()
        self.store = store
        self.prefix = prefix
        self.dims = dims
        self.mode = mode
        self.group_size, self.bits = infer_quantization(store, prefix, dims)
        self.capacity = capacity
        self._cache: "OrderedDict[int, mx.array]" = OrderedDict()

    def _rows(self, rows):
        if self.capacity <= 0:
            weight = self.store.rows(f"{self.prefix}.weight", rows)
            scales = self.store.rows(f"{self.prefix}.scales", rows)
            biases = self.store.rows(f"{self.prefix}.biases", rows)
            return mx.dequantize(
                weight,
                scales,
                biases,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )

        out, missing = [], []
        for row in rows:
            cached = self._cache.get(row)
            if cached is None:
                missing.append(row)
            out.append(cached)
        if missing:
            weight = self.store.rows(f"{self.prefix}.weight", missing)
            scales = self.store.rows(f"{self.prefix}.scales", missing)
            biases = self.store.rows(f"{self.prefix}.biases", missing)
            fresh = mx.dequantize(
                weight,
                scales,
                biases,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            for slot, row in enumerate(missing):
                self._cache[row] = fresh[slot]
                if len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
        return mx.stack(
            [value if value is not None else self._cache[row] for value, row in zip(out, rows)]
        )

    def __call__(self, indices: mx.array) -> mx.array:
        flat = indices.reshape(-1)
        mx.eval(flat)
        rows = [int(v) for v in flat.tolist()]
        if not rows:
            return mx.zeros((0, self.dims), dtype=mx.bfloat16)
        values = self._rows(rows)
        return values.reshape(*indices.shape, self.dims)
