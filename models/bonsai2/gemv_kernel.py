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


#: Weight shape of mlp gate/up projections: the only narrow-F16 route.
NARROW_GATEUP_WEIGHT_SHAPE = (17408, 320)

_narrow_counters: dict | None = None


def new_narrow_counters() -> dict:
    """Fresh counters for the narrow-F16 gate/up route."""
    return {"narrow_calls": 0, "narrow_fallbacks": 0, "fallback_reasons": {}}


def _narrow_fallback(reason: str) -> None:
    counters = _narrow_counters
    if counters is None:
        return
    counters["narrow_fallbacks"] += 1
    reasons = counters["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def install_narrow_hook(counters: dict | None = None) -> bool:
    """Route gate/up single-row projections through F16-metadata GEMV.

    Keeps the stock transform (fused FWHT when enabled), reshapes the
    single-row 3D activation for the 2D GEMV helper, and falls back to
    the wrapped call path for every other shape.  Callers must exclude
    gate/up modules from prepared-F32 installation so this route reads
    the original F16 scales and biases.
    """
    import runtime

    global _narrow_counters
    if counters is None:
        counters = new_narrow_counters()
    _narrow_counters = counters
    if getattr(runtime.Packed, "_bonsai2_narrow", False):
        return True
    wrapped = runtime.Packed.__call__
    runtime.Packed._bonsai2_narrow_wrapped = wrapped

    def narrow_call(self, x):
        if (
            not bool(getattr(self, "embedding", False))
            and bool(getattr(self, "block", 0))
            and tuple(int(v) for v in self.weight.shape) == NARROW_GATEUP_WEIGHT_SHAPE
            and str(self.scales.dtype) == "mlx.core.float16"
            and str(self.biases.dtype) == "mlx.core.float16"
        ):
            try:
                from runtime import fwht as transform

                transformed = transform(x, self.block, self.signs)
                flat = (
                    transformed.reshape(1, -1)
                    if transformed.ndim == 3 and transformed.shape[1] == 1
                    else transformed
                )
                out = narrow_qmm_f32(flat, self.weight, self.scales, self.biases)
            except Exception:
                _narrow_fallback("exception")
                return wrapped(self, x)
            if out is not None:
                if _narrow_counters is not None:
                    _narrow_counters["narrow_calls"] += 1
                if x.ndim == 3:
                    return out.reshape(x.shape[0], x.shape[1], -1)
                return out
            _narrow_fallback("kernel_none")
            return wrapped(self, x)
        _narrow_fallback("shape_or_dtype")
        return wrapped(self, x)

    runtime.Packed.__call__ = narrow_call
    runtime.Packed._bonsai2_narrow = True
    return True


def uninstall_narrow_hook() -> bool:
    """Restore the wrapped Packed call path. Used by tests."""
    import sys

    module = sys.modules.get("runtime")
    if module is None:
        return False
    wrapped = getattr(module.Packed, "_bonsai2_narrow_wrapped", None)
    if wrapped is None:
        return False
    module.Packed.__call__ = wrapped
    delattr(module.Packed, "_bonsai2_narrow_wrapped")
    delattr(module.Packed, "_bonsai2_narrow")
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
    if not gemv_enabled():
        return None
    return _gemv_q2_impl(x, weight, scales, biases)


@lru_cache(maxsize=None)
def _get_narrow_kernel():
    """FP32-input narrow QMM tail for gate/up (D3 prototype)."""
    import mlx.core as mx

    source = (
        Path(__file__).resolve().parent / "metal" / "qmv_f32_narrow.metal"
    ).read_text()
    return mx.fast.metal_kernel(
        name="bonsai2_narrow_qmm_f32",
        input_names=["w", "scales", "biases", "x", "in_vec_size", "out_vec_size"],
        output_names=["y"],
        source=source,
        header=_wheel_header(),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


_TERNARY_TABLE: object | None = None


def _ternary_table() -> object:
    """256x5 base-243 decode table, built once and cached."""
    import mlx.core as mx

    global _TERNARY_TABLE
    if _TERNARY_TABLE is None:
        table = [[(v // (3 ** j)) % 3 for j in range(5)] for v in range(256)]
        flat = [d for row in table for d in row]
        _TERNARY_TABLE = mx.array(bytes(flat), dtype=mx.uint8)
    return _TERNARY_TABLE


def _get_ternary_kernel(variant: str):
    import mlx.core as mx

    if variant not in ("table", "divmod"):
        raise ValueError("ternary variant must be table or divmod")
    inputs = ["tpack", "scales", "biases", "x", "in_vec_size", "out_vec_size"]
    if variant == "table":
        inputs = ["tpack", "stable", "scales", "biases", "x", "in_vec_size", "out_vec_size"]
    source = (
        Path(__file__).resolve().parent / "metal" / f"qmv_ternary_{variant}.metal"
    ).read_text()
    return mx.fast.metal_kernel(
        name=f"bonsai2_ternary_gateup_{variant}",
        input_names=inputs,
        output_names=["y"],
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def ternary_gateup_qmm(x, tpack, scales, biases, variant: str = "table"):
    """Gate/up ternary QMM for FP32 single-row x. None when unsupported."""
    import mlx.core as mx

    if x.ndim != 2 or x.shape[0] != 1 or int(x.shape[1]) != 5120:
        return None
    if str(x.dtype) != "mlx.core.float32":
        return None
    out_vec_size = 17408
    args: list = [tpack, scales, biases, x]
    if variant == "table":
        args = [tpack, _ternary_table(), scales, biases, x]
    kernel = _get_ternary_kernel(variant)
    out = kernel(
        inputs=args + [
            mx.array([5120], dtype=mx.int32),
            mx.array([out_vec_size], dtype=mx.int32),
        ],
        grid=(out_vec_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, out_vec_size)],
        output_dtypes=[x.dtype],
    )
    return out[0] if isinstance(out, (tuple, list)) else out


@lru_cache(maxsize=None)
def _get_ternary_qmv_kernel():
    import mlx.core as mx

    source = (
        Path(__file__).resolve().parent / "metal" / "qmv_ternary_qmv.metal"
    ).read_text()
    return mx.fast.metal_kernel(
        name="bonsai2_ternary_gateup_qmv",
        input_names=["tpack", "scales", "biases", "x", "in_vec_size", "out_vec_size"],
        output_names=["y"],
        source=source,
        header=_wheel_header(),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def ternary_qmv_qmm(x, tpack, scales, biases):
    """Gate/up ternary QMM with qmv-structured traversal. None if unsupported."""
    import mlx.core as mx

    if x.ndim != 2 or x.shape[0] != 1 or int(x.shape[1]) != 5120:
        return None
    if str(x.dtype) != "mlx.core.float32":
        return None
    if str(scales.dtype) != "mlx.core.float16":
        return None
    if str(biases.dtype) != "mlx.core.float16":
        return None
    if tuple(int(v) for v in tpack.shape) != (17408, 1040):
        return None
    out_vec_size = 17408
    kernel = _get_ternary_qmv_kernel()
    out = kernel(
        inputs=[
            tpack, scales, biases, x,
            mx.array([5120], dtype=mx.int32),
            mx.array([out_vec_size], dtype=mx.int32),
        ],
        grid=(64, (out_vec_size + 7) // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, out_vec_size)],
        output_dtypes=[x.dtype],
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def narrow_qmm_f32(x, weight, scales, biases):
    """Single-row Q2/G128 matvec for FP32 x with F16 metadata.

    Returns None when shapes or dtypes are unsupported.  Gate/up only;
    all other projections keep the stock path.
    """
    import mlx.core as mx

    if x.ndim != 2 or x.shape[0] != 1:
        return None
    if str(x.dtype) != "mlx.core.float32":
        return None
    if str(scales.dtype) != "mlx.core.float16":
        return None
    if str(biases.dtype) != "mlx.core.float16":
        return None
    out_vec_size, in_packed = weight.shape
    in_vec_size = x.shape[1]
    if (out_vec_size, in_packed) != NARROW_GATEUP_WEIGHT_SHAPE:
        return None
    if in_packed * 16 != in_vec_size:
        return None
    if in_vec_size % 512 != 0:
        return None
    kernel = _get_narrow_kernel()
    # Same launch geometry as the wheel fast path: 64 threads
    # (2 simdgroups) per 8-row group.
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


def _gemv_q2_impl(x, weight, scales, biases):
    """Shape-gated Q2/G128 matvec without the experiment flag check."""
    import mlx.core as mx

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
