"""Opt-in fused affine-Q4/G64 attention candidate for Bonsai-2.

The candidate intentionally keeps the existing ``QuantizedKVCache`` tuple
layout.  One threadgroup owns one query row/head, walks logical 32-token
pages, and performs QK, online softmax, and PV in one Metal launch.  It is a
narrow feasibility probe, not a replacement for MLX's composed attention:
callers must keep the stock path available for every unsupported contract.
"""
from __future__ import annotations

from functools import lru_cache


PAGE_TOKENS = 32
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
GROUP_SIZE = 64
BITS = 4


def page_ranges(tokens: int, page_tokens: int = PAGE_TOKENS):
    """Yield logical page ``(start, end)`` pairs, including a partial tail."""
    if tokens < 0 or page_tokens <= 0:
        raise ValueError("tokens and page_tokens must be non-negative/positive")
    return tuple(
        (start, min(start + page_tokens, tokens))
        for start in range(0, tokens, page_tokens)
    )


def kv_head_for_query(query_head: int, query_heads: int = QUERY_HEADS,
                      kv_heads: int = KV_HEADS) -> int:
    """Map a query head to the repeated GQA KV head."""
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query_heads must be a positive multiple of kv_heads")
    if query_head < 0 or query_head >= query_heads:
        raise ValueError("query_head is outside the query-head range")
    return query_head // (query_heads // kv_heads)


def causal_visible_tokens(query_offset: int, query_row: int,
                          key_rows: int) -> int:
    """Return the inclusive causal prefix visible to one query row."""
    if query_offset < 0 or query_row < 0 or key_rows < 0:
        raise ValueError("causal positions must be non-negative")
    return min(key_rows, query_offset + query_row + 1)


def unpack_q4_codes(packed, head_dim: int = HEAD_DIM):
    """Extract eight low-to-high affine-Q4 codes from each uint32 word."""
    import mlx.core as mx

    if packed.ndim < 1 or packed.dtype != mx.uint32:
        raise ValueError("packed Q4 storage must be a uint32 array")
    if int(packed.shape[-1]) * 8 != head_dim:
        raise ValueError("packed Q4 width does not match head_dim")
    shifts = mx.arange(8, dtype=mx.uint32) * 4
    return mx.reshape(
        mx.bitwise_and(mx.right_shift(packed[..., None], shifts), 15),
        packed.shape[:-1] + (head_dim,),
    )


def affine_q4_reference(packed, scales, biases, group_size: int = GROUP_SIZE):
    """Reference affine correction for tests and exactness diagnostics."""
    import mlx.core as mx

    if group_size <= 0 or int(packed.shape[-1]) * 8 % group_size:
        raise ValueError("Q4 packed width is not divisible by group_size")
    codes = unpack_q4_codes(packed, int(packed.shape[-1]) * 8)
    groups = codes.shape[-1] // group_size
    if tuple(scales.shape) != packed.shape[:-1] + (groups,):
        raise ValueError("Q4 scales have the wrong shape")
    if tuple(biases.shape) != tuple(scales.shape):
        raise ValueError("Q4 biases have the wrong shape")
    return (
        codes.astype(scales.dtype)
        * mx.repeat(scales, group_size, axis=-1)
        + mx.repeat(biases, group_size, axis=-1)
    )


def _scalar_offset(cache):
    offset = getattr(cache, "offset", None)
    if hasattr(offset, "ndim") and int(offset.ndim) != 0:
        return None
    try:
        return int(offset.item()) if hasattr(offset, "item") else int(offset)
    except (AttributeError, TypeError, ValueError):
        return None


def unsupported_reason(queries, keys, values, cache, mask) -> str | None:
    """Return a reason when the fused kernel must not be selected."""
    if not hasattr(cache, "bits") or int(cache.bits) != BITS:
        return "cache is not affine Q4"
    if int(getattr(cache, "group_size", -1)) != GROUP_SIZE:
        return "cache group size is not 64"
    if not isinstance(mask, (str, type(None))) or (
        isinstance(mask, str) and mask != "causal"
    ):
        return "mask is not causal or absent"
    if not isinstance(keys, (tuple, list)) or len(keys) != 3:
        return "keys are not an affine quantized tuple"
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        return "values are not an affine quantized tuple"
    if queries.ndim != 4 or tuple(queries.shape[:2]) != (1, QUERY_HEADS):
        return "only batch one with 24 query heads is supported"
    if int(queries.shape[-1]) != HEAD_DIM:
        return "query head dimension is not 256"
    if mask is None and int(queries.shape[-2]) != 1:
        return "unmasked multi-row attention is unsupported"
    if any(item.ndim != 4 for item in (keys[0], values[0])):
        return "packed KV arrays are not rank 4"
    if any(str(item.dtype) != "mlx.core.uint32" for item in (keys[0], values[0])):
        return "packed KV arrays are not uint32"
    batch, kv_heads, key_rows, packed_width = map(int, keys[0].shape)
    if (batch, kv_heads, packed_width) != (1, KV_HEADS, HEAD_DIM // 8):
        return "KV geometry is not 1 x 4 x 256"
    if tuple(values[0].shape) != tuple(keys[0].shape):
        return "K/V packed shapes differ"
    groups = HEAD_DIM // GROUP_SIZE
    expected_meta = (1, KV_HEADS, key_rows, groups)
    if tuple(keys[1].shape) != expected_meta or tuple(keys[2].shape) != expected_meta:
        return "K metadata shape is not G64"
    if tuple(values[1].shape) != expected_meta or tuple(values[2].shape) != expected_meta:
        return "V metadata shape is not G64"
    offset = _scalar_offset(cache)
    if offset is None or offset < 0 or key_rows != offset:
        return "cache offset does not match the visible key rows"
    if int(queries.shape[-2]) <= 0:
        return "query rows are empty"
    if offset - int(queries.shape[-2]) < 0:
        return "query rows start before position zero"
    if int(queries.shape[-2]) > 0 and int(queries.shape[-2]) > offset:
        return "query rows exceed the cache offset"
    return None


_FUSED_Q4_SOURCE = r"""
{
  uint d = thread_position_in_threadgroup.x;
  uint query_head = threadgroup_position_in_grid.y;
  uint batch_row = threadgroup_position_in_grid.z;
  uint batch = batch_row / p[2];
  uint query_row = batch_row % p[2];

  const uint query_rows = p[2];
  const uint head_dim = p[3];
  const uint key_rows = p[4];
  const uint kv_heads = p[5];
  const uint packed_width = p[6];
  const uint groups = p[7];
  const uint query_offset = p[8];
  const uint query_per_kv = p[1] / kv_heads;
  const uint kv_head = query_head / query_per_kv;
  const uint visible = min(key_rows, query_offset + query_row + 1);
  const float query_scale = scale[0];
  const uint simd_lane = thread_index_in_simdgroup;
  const uint simd_group = simdgroup_index_in_threadgroup;

  threadgroup float score_partials[32 * 8];
  threadgroup float page_probability[32];
  threadgroup float page_rescale;
  threadgroup float running_max;
  threadgroup float running_sum;

  if (d == 0) {
    running_max = -INFINITY;
    running_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float numerator = 0.0f;
  const ulong query_origin =
      (ulong(batch) * p[1] + query_head) * query_rows * head_dim +
      ulong(query_row) * head_dim;
  const ulong output_origin = query_origin;
  const uint page_count = (visible + 31) / 32;

  // One threadgroup owns one query row/head.  Its eight simdgroups reduce the
  // QK dot product into eight partials per token; thread zero performs the
  // online-softmax update, and all 256 threads then perform PV in parallel.
  // The score tensor never leaves this small threadgroup scratch allocation.
  for (uint page = 0; page < page_count; ++page) {
    uint token_begin = page * 32;
    uint token_end = min(token_begin + 32, visible);
    for (uint token_slot = 0; token_slot < 32; ++token_slot) {
      uint token = token_begin + token_slot;
      if (token >= token_end)
        continue;
      ulong key_origin =
          (ulong(batch) * kv_heads + kv_head) * key_rows * packed_width +
          ulong(token) * packed_width;
      ulong key_meta =
          (ulong(batch) * kv_heads + kv_head) * key_rows * groups +
          ulong(token) * groups;
      uint word = keys[key_origin + d / 8];
      uint code = (word >> ((d & 7) * 4)) & 15u;
      uint group = d / 64;
      float key_value = float(key_scales[key_meta + group]) * float(code) +
                        float(key_biases[key_meta + group]);
      float partial = simd_sum(float(queries[query_origin + d]) * key_value);
      if (simd_lane == 0)
        score_partials[(token - token_begin) * 8 + simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d == 0) {
      float local_max = -INFINITY;
      for (uint token = token_begin; token < token_end; ++token) {
        float score = 0.0f;
        for (uint group = 0; group < 8; ++group)
          score += score_partials[(token - token_begin) * 8 + group];
        score *= query_scale;
        local_max = max(local_max, score);
        page_probability[token - token_begin] = score;
      }
      float next_max = max(running_max, local_max);
      page_rescale = running_sum == 0.0f
          ? 0.0f : metal::precise::exp(running_max - next_max);
      float local_sum = 0.0f;
      for (uint token = token_begin; token < token_end; ++token) {
        float probability = metal::precise::exp(
            page_probability[token - token_begin] - next_max);
        page_probability[token - token_begin] = probability;
        local_sum += probability;
      }
      running_sum = running_sum * page_rescale + local_sum;
      running_max = next_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    numerator *= page_rescale;
    for (uint token = token_begin; token < token_end; ++token) {
      ulong value_origin =
          (ulong(batch) * kv_heads + kv_head) * key_rows * packed_width +
          ulong(token) * packed_width;
      ulong value_meta =
          (ulong(batch) * kv_heads + kv_head) * key_rows * groups +
          ulong(token) * groups;
      uint value_word = values[value_origin + d / 8];
      uint value_code = (value_word >> ((d & 7) * 4)) & 15u;
      uint value_group = d / 64;
      float value = float(value_scales[value_meta + value_group]) *
                    float(value_code) +
                    float(value_biases[value_meta + value_group]);
      numerator += page_probability[token - token_begin] * value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[output_origin + d] = running_sum > 0.0f
      ? static_cast<T>(numerator / running_sum)
      : T(0);
}
"""


@lru_cache(maxsize=None)
def _get_kernel(dtype_name: str):
    import mlx.core as mx

    dtype = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"fused Q4 attention does not support {dtype_name}")
    return mx.fast.metal_kernel(
        name=f"bonsai2_fused_q4_attention_{dtype_name}",
        input_names=[
            "queries", "keys", "key_scales", "key_biases", "values",
            "value_scales", "value_biases", "p", "scale",
        ],
        output_names=["out"],
        source=_FUSED_Q4_SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def fused_q4_attention(queries, keys, values, cache, scale, mask,
                       diagnostics=None):
    """Return the fused candidate, or ``None`` for the stock fallback."""
    import mlx.core as mx

    def reject(reason):
        if diagnostics is not None:
            diagnostics["reason"] = reason
        return None

    reason = unsupported_reason(queries, keys, values, cache, mask)
    if reason is not None:
        return reject(reason)
    dtype_name = str(queries.dtype).split(".")[-1]
    if dtype_name not in ("float16", "bfloat16", "float32"):
        return reject("query dtype is not supported")
    key_packed, key_scales, key_biases = keys
    value_packed, value_scales, value_biases = values
    batch, query_heads, query_rows, head_dim = map(int, queries.shape)
    key_rows = int(key_packed.shape[-2])
    kv_heads = int(key_packed.shape[-3])
    packed_width = int(key_packed.shape[-1])
    groups = int(key_scales.shape[-1])
    offset = _scalar_offset(cache)
    query_offset = offset - query_rows
    parameters = mx.array(
        [
            batch, query_heads, query_rows, head_dim, key_rows, kv_heads,
            packed_width, groups, query_offset,
        ],
        dtype=mx.uint32,
    )
    kernel = _get_kernel(dtype_name)
    result = kernel(
        inputs=[
            queries, key_packed, key_scales, key_biases, value_packed,
            value_scales, value_biases, parameters,
            mx.array([float(scale)], dtype=mx.float32),
        ],
        template=[("T", queries.dtype)],
        grid=(head_dim, query_heads, batch * query_rows),
        threadgroup=(256, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return result[0] if isinstance(result, (tuple, list)) else result
