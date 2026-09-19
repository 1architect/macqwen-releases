"""Experimental M=1 Q2/G128 GEMV for Bonsai-2 decode (D2).

Calls MLX's own qmv_fast vector helper directly, bypassing dispatch
selection, to test whether a specialized launch beats stock
quantized_matmul on our shapes. Opt-in only; promotion needs full-model
gains with matching digests.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_BODY = """
{
  uint3 group = threadgroup_position_in_grid;
  uint simd_lid = thread_index_in_simdgroup;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint3 tid = uint3(group.x, group.y, 0);
  qmv_fast_impl<half, 128, 2>(
      w, scales, biases, x, y, in_vec_size[0], out_vec_size[0],
      tid, simd_gid, simd_lid);
}
"""


def _wheel_header() -> str:
    import mlx.core as mx

    path = (
        Path(mx.__file__).resolve().parent
        / "include/mlx/backend/metal/kernels/quantized.h"
    )
    text = path.read_text()
    helper_end = text.index(
        "template <typename U, int values_per_thread, int bits>\ninline void\nqouter"
    )
    start = text.index(
        "template <typename T, int group_size, int bits>\n"
        "METAL_FUNC void qmv_fast_impl"
    )
    end = text.index("template <typename T, int group_size, int bits>\n"
                     "METAL_FUNC void qmv_impl", start)
    return text[:helper_end] + text[start:end]


@lru_cache(maxsize=None)
def _get_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="bonsai2_gemv_q2g128",
        input_names=["w", "scales", "biases", "x", "in_vec_size", "out_vec_size"],
        output_names=["y"],
        source=_BODY,
        header=_wheel_header(),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def gemv_enabled() -> bool:
    return os.environ.get("BONSAI2_GEMV") == "1"


def install_gemv_hook() -> bool:
    """Route single-row forward projections through the custom GEMV kernel.

    Patches the checkpoint runtime's Packed call path, reusing its stock
    transform (fused when that flag is on) and falling back to the original
    implementation for embeddings, batched inputs, and unsupported shapes.
    """
    import sys

    if not gemv_enabled():
        return False
    import runtime

    if getattr(runtime.Packed, "_bonsai2_gemv", False):
        return True
    original = runtime.Packed.__call__
    runtime.Packed._bonsai2_original = original

    def gemv_call(self, x):
        if not getattr(self, "embedding", False) and getattr(self, "block", 0):
            try:
                from runtime import fwht as transform

                out = gemv_q2(transform(x, self.block, self.signs), self.weight, self.scales, self.biases)
            except Exception:
                out = None
            if out is not None:
                return out
        return original(self, x)

    runtime.Packed.__call__ = gemv_call
    runtime.Packed._bonsai2_gemv = True
    return True


def uninstall_gemv_hook() -> bool:
    """Restore the stock Packed call path. Used by tests."""
    import sys

    module = sys.modules.get("runtime")
    if module is None:
        return False
    original = getattr(module.Packed, "_bonsai2_original", None)
    if original is None:
        return False
    module.Packed.__call__ = original
    delattr(module.Packed, "_bonsai2_original")
    delattr(module.Packed, "_bonsai2_gemv")
    return True


def gemv_q2(x, weight, scales, biases):
    """Single-row Q2/G128 matvec. Returns None when shapes are unsupported."""
    import mlx.core as mx

    if not gemv_enabled():
        return None
    if x.ndim != 2 or x.shape[0] != 1:
        return None
    out_vec_size, in_packed = weight.shape
    in_vec_size = x.shape[1]
    if out_vec_size % 8 != 0:
        return None
    if in_packed * 16 != in_vec_size:
        return None
    if in_vec_size % 512 != 0:
        return None
    kernel = _get_kernel()
    # MLX grid counts total threads: 64 threads (2 simdgroups) per 8-row
    # group, matching the helper's tid.y/simd indexing.
    out = kernel(
        inputs=[
            weight, scales, biases, x,
            mx.array([in_vec_size], dtype=mx.int32),
            mx.array([out_vec_size], dtype=mx.int32),
        ],
        grid=(64, (out_vec_size + 7) // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, out_vec_size)],
        output_dtypes=[x.dtype],
    )
    return out[0] if isinstance(out, (tuple, list)) else out
