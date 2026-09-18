"""Fused sign+Hadamard+downcast kernel for Bonsai-2 ternary projections.

The stock path per packed module runs three launches: a sign multiply, the
library Hadamard transform, and a downcast. This module folds all three into
one dispatch over 1024-element blocks. Anything the kernel does not cover
falls back to the stock ``runtime.fwht``.

The kernel stays opt-in behind ``BONSAI2_FUSED_FWHT=1`` until a controlled
comparison with matching greedy digests promotes it.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

_FUSED_FWHT_SOURCE = """
// One threadgroup per BLOCK slice of one row. Thread i owns element i.
// This is a kernel body fragment: mx.fast.metal_kernel supplies the wrapper
// and the x/signs/out buffers. Activations are fp16; signs are fp32.
{
  constexpr uint N = BLOCK;
  threadgroup float buf[N];
  // Blocks tile the flattened tensor, so the sign index wraps by row width:
  // signs cover one row (WIDTH elements), matching the stock broadcast.
  uint base = threadgroup_position_in_grid.x * N;
  uint width = WIDTH;
  uint i = thread_position_in_threadgroup.x;
  buf[i] = float(x[base + i]) * signs[(base + i) % width];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1; stride < N; stride <<= 1) {
    uint full = stride << 1;
    uint j = (i & ~(full - 1)) | (i & (stride - 1));
    float a = buf[j];
    float b = buf[j + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[i] = ((i & stride) == 0) ? (a + b) : (a - b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[base + i] = half(buf[i] * SCALE);
}
"""


def _dtype_tag(dtype) -> str:
    name = str(getattr(dtype, "value", dtype))
    if "bfloat16" in name:
        return "bfloat16"
    return "half"


@lru_cache(maxsize=None)
def _get_kernel(block: int, width: int, dtype_tag: str):
    import mlx.core as mx

    if dtype_tag != "half":
        raise ValueError(f"fused FWHT supports fp16 activations, got {dtype_tag}")
    source = _FUSED_FWHT_SOURCE.replace("BLOCK", str(block))
    source = source.replace("WIDTH", str(width))
    scale = 1.0 / math.sqrt(block)
    source = source.replace("SCALE", repr(float(scale)))
    return mx.fast.metal_kernel(
        name=f"bonsai2_fused_fwht_b{block}_w{width}_{dtype_tag}",
        input_names=["x", "signs"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def fused_fwht_enabled() -> bool:
    return os.environ.get("BONSAI2_FUSED_FWHT") == "1"


_STOCK_FWHT = None
_MEMO = None


def share_fwht_enabled() -> bool:
    return os.environ.get("BONSAI2_SHARE_FWHT") == "1"


def arm_memo() -> None:
    """Open a call-scoped transform cache. The wrapper arms before each model
    forward and disarms after it returns; entries hold their inputs alive so
    object ids cannot be reused while cached, and clearing per forward bounds
    memory. Hits return the identical computed array, so shared calls are
    bit-identical by construction."""
    global _MEMO
    _MEMO = {}


def disarm_memo() -> None:
    global _MEMO
    _MEMO = None


def install_share_hook() -> bool:
    """Memoize forward transforms across modules sharing one input object.

    Same-width modules carry byte-identical sign vectors, so a shared input
    means a shared result. The memo consults whatever transform sits
    underneath (fused kernel when enabled, stock otherwise).
    """
    import sys

    if not share_fwht_enabled():
        return False
    module = sys.modules.get("runtime")
    if module is None:
        return False
    if getattr(module, "_bonsai2_shared", False):
        return True
    inner = module.fwht

    def shared(x, block, signs, inverse=False):
        memo = _MEMO
        if memo is None or inverse:
            return inner(x, block, signs, inverse=inverse)
        # Same-width sign vectors are byte-identical (verified per loaded
        # model in the backend), so width identifies the signs. Inputs are
        # held alive by their entries, so ids cannot be reused mid-forward.
        key = (id(x), x.shape[-1], tuple(x.shape), block)
        hit = memo.get(key)
        if hit is not None:
            return hit[2]
        result = inner(x, block, signs, inverse=inverse)
        memo[key] = (x, signs, result)
        return result

    module.fwht = shared
    module._bonsai2_shared = True
    return True


def install_packed_hook() -> bool:
    """Route the checkpoint runtime's forward transform through the kernel.

    ``Packed.__call__`` resolves ``fwht`` from its own module globals, so
    patching that attribute redirects every forward transform without
    touching checkpoint code. The inverse embedding path keeps the stock
    implementation. Returns whether the hook is active.
    """
    import sys

    global _STOCK_FWHT
    if not fused_fwht_enabled():
        return False
    module = sys.modules.get("runtime")
    if module is None:
        return False
    if getattr(module, "_bonsai2_fused", False):
        return True
    _STOCK_FWHT = module.fwht

    def hooked(x, block, signs, inverse=False):
        if inverse:
            return _STOCK_FWHT(x, block, signs, inverse=inverse)
        return fused_fwht(x, signs, block)

    module.fwht = hooked
    module._bonsai2_fused = True
    return True


def fused_fwht(x, signs, block: int):
    """Apply sign multiply, Hadamard transform, and downcast in one launch.

    Falls back to the stock runtime path when the row width is not a
    multiple of ``block`` or the fused path is disabled.
    """
    import mlx.core as mx

    width = x.shape[-1]
    if not fused_fwht_enabled() or width % block != 0:
        if _STOCK_FWHT is not None:
            return _STOCK_FWHT(
                x.astype(mx.float32), block, signs, inverse=False
            ).astype(x.dtype)
        from runtime import fwht as stock_fwht

        return stock_fwht(
            x.astype(mx.float32), block, signs, inverse=False
        ).astype(x.dtype)
    kernel = _get_kernel(block, width, _dtype_tag(x.dtype))
    # MLX grid counts total threads, not threadgroups: one thread per
    # element, grouped in BLOCK-wide threadgroups of 1024 threads.
    outputs = kernel(
        inputs=[x, signs],
        grid=(x.size, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[list(x.shape)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]
