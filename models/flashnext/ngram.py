"""N-gram embedding rows read straight from the checkpoint.

The hashed trigram table spans 128 shards, but each lookup needs one small row.
A token touches only a few rows. Holding a whole shard resident wastes memory,
so the runtime fetches and dequantizes rows individually.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import os
import time

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
            else:
                self._cache.move_to_end(row)
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
            # Hold this call's rows directly. Reading them back from the
            # cache would fail when more rows are missing than the cache can
            # keep, because the eviction below discards rows this same call
            # still has to return.
            produced = {}
            for slot, row in enumerate(missing):
                produced[row] = fresh[slot]
                self._cache[row] = fresh[slot]
                if len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
            out = [
                value if value is not None else produced[row]
                for value, row in zip(out, rows)
            ]
        return mx.stack(out)

    def __call__(self, indices: mx.array) -> mx.array:
        flat = indices.reshape(-1)
        mx.eval(flat)
        rows = [int(v) for v in flat.tolist()]
        if not rows:
            return mx.zeros((0, self.dims), dtype=mx.bfloat16)
        values = self._rows(rows)
        return values.reshape(*indices.shape, self.dims)


class StreamingShardedEmbedding(nn.Module):
    """Read only shards that own at least one requested n-gram row."""

    def __init__(self, shards, shard_sizes, dims):
        super().__init__()
        self.shards = shards
        self.shard_sizes = tuple(int(size) for size in shard_sizes)
        offsets = [0]
        for size in self.shard_sizes:
            offsets.append(offsets[-1] + size)
        self.shard_offsets = tuple(offsets)
        self.dims = int(dims)
        self.direct = os.environ.get("FLASHNEXT_NGRAM_DIRECT", "1") == "1"

    def _groups(self, global_rows):
        groups = {}
        for position, row in enumerate(global_rows):
            if row < 0 or row >= self.shard_offsets[-1]:
                shard_index = 0
                local = row
            else:
                shard_index = bisect_right(self.shard_offsets, row) - 1
                local = row - self.shard_offsets[shard_index]
            local = min(max(local, 0), self.shard_sizes[shard_index] - 1)
            positions, rows = groups.setdefault(shard_index, ([], []))
            positions.append(position)
            rows.append(local)
        return groups

    def _legacy(self, flat):
        result = None
        for shard, start, end in zip(
            self.shards,
            self.shard_offsets[:-1],
            self.shard_offsets[1:],
        ):
            local = mx.clip(flat - start, 0, end - start - 1)
            values = shard(local)
            mask = (flat >= start) & (flat < end)
            result = (
                values
                if result is None
                else mx.where(mask[:, None], values, result)
            )
        return result

    def _direct(self, flat):
        from models.flashnext.expert_cache import _PROFILE, _TIMERS

        if _PROFILE:
            began = time.perf_counter()
            result = self._direct_rows(flat)
            _TIMERS["ngram_wait"] += time.perf_counter() - began
            return result
        return self._direct_rows(flat)

    def _direct_rows(self, flat):
        mx.eval(flat)
        global_rows = [int(value) for value in flat.tolist()]
        groups = self._groups(global_rows)

        blocks = []
        packed_positions = []
        for shard_index, (_, rows) in groups.items():
            blocks.append(self.shards[shard_index]._rows(rows))
        for positions, _ in groups.values():
            packed_positions.extend(positions)
        packed = blocks[0] if len(blocks) == 1 else mx.concatenate(blocks, axis=0)
        inverse = [0] * len(global_rows)
        for packed_index, position in enumerate(packed_positions):
            inverse[position] = packed_index
        return packed[mx.array(inverse, dtype=mx.uint32)]

    def __call__(self, indices: mx.array) -> mx.array:
        flat = indices.reshape(-1)
        if not flat.size:
            return mx.zeros((*indices.shape, self.dims), dtype=mx.bfloat16)
        values = self._direct(flat) if self.direct else self._legacy(flat)
        return values.reshape(*indices.shape, self.dims)
