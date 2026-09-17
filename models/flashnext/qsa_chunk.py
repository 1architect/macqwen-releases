"""Bound Qwen Sparse Attention masks without chunking the whole model."""
from __future__ import annotations

import math
import os

import mlx.core as mx


QSA_CHUNK_THRESHOLD = int(
    os.environ.get("FLASHNEXT_QSA_CHUNK_THRESHOLD", "2048")
)
QSA_QUERY_CHUNK = int(os.environ.get("FLASHNEXT_QSA_QUERY_CHUNK", "2048"))
# Upstream builds a dense block-selection mask with this shape:
# (batch, query, block_topk, key). Route to our sparse mask before that
# temporary crosses the allocator budget, even when query is below the old
# length-only threshold.
QSA_DENSE_MASK_MAX_BYTES = int(os.environ.get(
    "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES", str(512 * 1024 * 1024)
))

QSA_CACHE_POOLED_KEYS = os.environ.get("FLASHNEXT_QSA_CACHE_POOLED_KEYS", "0") == "1"
QSA_SCATTER_DECODE = os.environ.get("FLASHNEXT_QSA_SCATTER_DECODE", "0") == "1"

_ORIGINAL_CALL = None
_ORIGINAL_INDEXER_CALL = None


def _dense_mask_bytes(batch, query, past, block_topk) -> int:
    """Estimate upstream QSA's dense block-selection mask allocation."""
    return int(batch) * int(query) * int(block_topk) * (int(past) + int(query))


def _pool_keys(indexer, raw_keys, position_ids, start, end):
    pooled_keys = raw_keys[
        :, start * indexer.compress_ratio : end * indexer.compress_ratio
    ].reshape(raw_keys.shape[0], end - start, indexer.compress_ratio, indexer.head_dim)
    pooled_keys = mx.expand_dims(
        indexer.k_layernorm(
            mx.mean(pooled_keys.astype(mx.float32), axis=2).astype(raw_keys.dtype)
        ),
        axis=1,
    )
    block_starts = mx.arange(start, end) * indexer.compress_ratio
    return indexer._apply_rope(pooled_keys, position_ids[..., block_starts])


def _reusable_pool(indexer, cache):
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache

    if type(cache) not in (QSAKVCache, QSAQuantizedKVCache):
        return None
    saved = getattr(cache, "_flashnext_pooled_keys", None)
    if saved is None:
        return None
    owner, keys, positions, ratio, pooled = saved
    if (
        owner is indexer
        and keys is cache.index_keys
        and positions is cache.index_position_ids
        and ratio == indexer.compress_ratio
        and cache.offset == keys.shape[1]
    ):
        return pooled
    return None


def _prepare_indexer(indexer, hidden_states, cache, position_ids):
    batch, seq_len, _ = hidden_states.shape
    past_len = int(cache.offset) if cache is not None else 0
    pooled_keys = None
    if QSA_CACHE_POOLED_KEYS:
        pooled_keys = _reusable_pool(indexer, cache)
    if position_ids is None:
        position_ids = indexer._default_position_ids(batch, past_len, seq_len)

    qk = indexer.index_qk_proj(hidden_states).reshape(
        batch,
        seq_len,
        indexer.n_heads + indexer.kv_heads,
        indexer.head_dim,
    )
    query = qk[:, :, : indexer.n_heads]
    raw_keys = qk[:, :, indexer.n_heads :].squeeze(2)
    query = indexer.q_layernorm(query).transpose(0, 2, 1, 3)
    if cache is not None:
        raw_keys, full_position_ids = cache.update_indexer(raw_keys, position_ids)
    else:
        full_position_ids = position_ids

    key_len = int(raw_keys.shape[1])
    max_complete_blocks = key_len // indexer.compress_ratio
    if max_complete_blocks <= indexer.block_topk:
        return query, None, key_len, past_len, max_complete_blocks
    query = indexer._apply_rope(query, position_ids)
    completed = 0 if pooled_keys is None else pooled_keys.shape[2]
    if completed < max_complete_blocks:
        new_keys = _pool_keys(
            indexer, raw_keys, full_position_ids, completed, max_complete_blocks
        )
        pooled_keys = (
            new_keys if pooled_keys is None
            else mx.concatenate([pooled_keys, new_keys], axis=2)
        )
    if QSA_CACHE_POOLED_KEYS and cache is not None:
        cache._flashnext_pooled_keys = (
            indexer, raw_keys, full_position_ids, indexer.compress_ratio, pooled_keys
        )
    return query, pooled_keys, key_len, past_len, max_complete_blocks


def _chunk_mask(
    indexer,
    query,
    pooled_keys,
    key_len: int,
    past_len: int,
    max_complete_blocks: int,
    start: int,
    end: int,
    scatter: bool = True,
):
    query_chunk = query[:, :, start:end]
    scores = query_chunk @ pooled_keys.transpose(0, 1, 3, 2)
    scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1)
    scores = scores / math.sqrt(indexer.head_dim)

    query_ends = past_len + mx.arange(start, end) + 1
    complete_counts = query_ends // indexer.compress_ratio
    valid_blocks = (
        mx.arange(max_complete_blocks)[None, None, :]
        < complete_counts[None, :, None]
    )
    scores = mx.where(valid_blocks, scores, -mx.inf)
    selected_blocks = mx.argpartition(
        scores,
        kth=-indexer.block_topk,
        axis=-1,
    )[..., -indexer.block_topk :]

    if not scatter:
        token_indices = mx.arange(key_len)
        token_blocks = token_indices // indexer.compress_ratio
        selected_tokens = mx.any(
            token_blocks[None, None, None, :] == selected_blocks[..., None], axis=2
        )
        tail_starts = complete_counts * indexer.compress_ratio
        tail = (token_indices[None, None, :] >= tail_starts[None, :, None]) & (
            token_indices[None, None, :] < query_ends[None, :, None]
        )
        causal = token_indices[None, None, :] < query_ends[None, :, None]
        use_sparse = complete_counts > indexer.block_topk
        return mx.where(
            use_sparse[None, :, None], selected_tokens | tail, causal
        )[:, None]

    offsets = mx.arange(indexer.compress_ratio)
    selected_indices = (
        selected_blocks[..., None] * indexer.compress_ratio + offsets
    ).reshape(*selected_blocks.shape[:-1], -1)
    selected_tokens = mx.zeros(
        (*selected_indices.shape[:-1], key_len),
        dtype=mx.bool_,
    )
    selected_tokens = mx.put_along_axis(
        selected_tokens,
        selected_indices,
        mx.ones(selected_indices.shape, dtype=mx.bool_),
        axis=-1,
    )

    tail_starts = complete_counts * indexer.compress_ratio
    tail_indices = tail_starts[None, :, None] + offsets[None, None, :]
    tail_valid = tail_indices < query_ends[None, :, None]
    fallback = selected_indices[..., :1]
    tail_indices = mx.where(tail_valid, tail_indices, fallback)
    selected_tokens = mx.put_along_axis(
        selected_tokens,
        tail_indices,
        mx.ones(tail_indices.shape, dtype=mx.bool_),
        axis=-1,
    )

    sparse_start = (indexer.block_topk + 1) * indexer.compress_ratio
    first_query_end = past_len + start + 1
    last_query_end = past_len + end
    if first_query_end >= sparse_start:
        return selected_tokens[:, None]

    token_indices = mx.arange(key_len)
    causal = token_indices[None, None, :] < query_ends[None, :, None]
    if last_query_end < sparse_start:
        return causal[:, None]
    use_sparse = complete_counts > indexer.block_topk
    return mx.where(
        use_sparse[None, :, None],
        selected_tokens,
        causal,
    )[:, None]


def _indexer_call(self, hidden_states, cache, position_ids):
    if not (QSA_CACHE_POOLED_KEYS or QSA_SCATTER_DECODE):
        return _ORIGINAL_INDEXER_CALL(self, hidden_states, cache, position_ids)
    if (
        (not QSA_CACHE_POOLED_KEYS and hidden_states.shape[1] != 1)
        or (cache is not None and not isinstance(cache.offset, int))
    ):
        return _ORIGINAL_INDEXER_CALL(self, hidden_states, cache, position_ids)
    query, pooled_keys, key_len, past_len, complete_blocks = _prepare_indexer(
        self, hidden_states, cache, position_ids
    )
    if complete_blocks <= self.block_topk:
        return None
    return _chunk_mask(
        self, query, pooled_keys, key_len, past_len, complete_blocks,
        0, hidden_states.shape[1],
        scatter=QSA_SCATTER_DECODE and hidden_states.shape[1] == 1,
    )


def _chunked_call(
    self,
    x,
    mask=None,
    cache=None,
    position_ids=None,
    position_embeddings=None,
):
    length = int(x.shape[1])
    supported_mask = mask is None or (
        isinstance(mask, str) and mask == "causal"
    )
    past_len = getattr(cache, "offset", None)
    dense_mask_bytes = (
        _dense_mask_bytes(x.shape[0], length, past_len, self.indexer.block_topk)
        if isinstance(past_len, int)
        else 0
    )
    if (
        (length <= QSA_CHUNK_THRESHOLD and dense_mask_bytes <= QSA_DENSE_MASK_MAX_BYTES)
        or not supported_mask
        or cache is None
        or not isinstance(past_len, int)
        or (past_len + length) // self.indexer.compress_ratio
        <= self.indexer.block_topk
    ):
        return _ORIGINAL_CALL(
            self,
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    (
        index_query,
        pooled_keys,
        key_len,
        saved_past_len,
        max_complete_blocks,
    ) = _prepare_indexer(self.indexer, x, cache, position_ids)

    q_proj_output, keys, values = (
        self.q_proj(x),
        self.k_proj(x),
        self.v_proj(x),
    )
    queries, keys, values, gate, _ = self._prepare_projected_qkv(
        q_proj_output,
        keys,
        values,
        cache,
        position_ids,
        position_embeddings,
        mask,
    )

    from mlx_vlm.models.qwen3_5.language import scaled_dot_product_attention

    outputs = []
    for start in range(0, length, QSA_QUERY_CHUNK):
        end = min(start + QSA_QUERY_CHUNK, length)
        qsa_mask = _chunk_mask(
            self.indexer,
            index_query,
            pooled_keys,
            key_len,
            saved_past_len,
            max_complete_blocks,
            start,
            end,
        )
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end],
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=qsa_mask,
            )
        )
    output = mx.concatenate(outputs, axis=2)
    output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], length, -1)
    return self.o_proj(output * mx.sigmoid(gate))


def apply() -> bool:
    """Patch only QSA calls whose dense mask would approach Metal limits."""
    global _ORIGINAL_CALL, _ORIGINAL_INDEXER_CALL
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention, Qwen4ExpQSAIndexer

    if getattr(Qwen4ExpAttention, "_flashnext_chunked_qsa", False):
        return False
    if QSA_QUERY_CHUNK < 1 or QSA_CHUNK_THRESHOLD < 1:
        raise ValueError("QSA chunk sizes must be positive")
    if QSA_DENSE_MASK_MAX_BYTES < 1:
        raise ValueError("QSA dense mask byte budget must be positive")
    _ORIGINAL_CALL = Qwen4ExpAttention.__call__
    _ORIGINAL_INDEXER_CALL = Qwen4ExpQSAIndexer.__call__
    Qwen4ExpQSAIndexer.__call__ = _indexer_call
    Qwen4ExpAttention.__call__ = _chunked_call
    Qwen4ExpAttention._flashnext_chunked_qsa = True
    return True
