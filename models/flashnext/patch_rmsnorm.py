"""Fix the RMSNorm gain in mlx-vlm's qwen4_exp.

`Qwen4ExpRMSNorm` applies `y * (1.0 + weight)`, documented as "Qwen4 RMSNorm,
whose checkpoint weights are centered at zero". The checkpoint says otherwise.
Measured over every norm family in Qwen3.8-Flash-Next-MLX-oQ4:

    norm_conv       mean +0.8906   min +0.543
    norm_key        mean +0.8933   min +0.672
    q/k_layernorm   mean +0.9306   min +0.500
    q/k_norm        mean +1.3576   min +0.098
    hc_norm         mean +3.7498   min +0.633

Every minimum is positive. A zero-centered tensor would be half negative. The
weights are one-centered, so the `1.0 +` doubles the gain of every norm in the
model. Output stays free of NaN and overflow, which is why the failure reads as
incoherent text rather than a crash.

Import `apply()` before building the model.
"""
from __future__ import annotations

import os
from functools import partial

import mlx.core as mx

_applied = False

# A GPU capture of one decode token showed the compute shader launch limiter
# pinned near 100% with occupancy and ALU utilisation low, so the GPU is
# launch-bound at about 120 dispatches per layer. In that capture the two dtype
# conversions below, `v_copybfloat16float32` and `v_copyfloat32bfloat16`, ranked
# third and fifth of all kernels by SIMD groups, together larger than the routed
# expert gather. They come from this function, which spends nine dispatches on a
# chain that fuses into far fewer.
#
# Off by default. The float32 accumulation is deliberate, so the acceptance test
# is an identical token digest, not a rate.
_COMPILE = [os.environ.get("FLASHNEXT_COMPILE_NORM", "0") != "0"]


def compile_norm() -> bool:
    return _COMPILE[0]


def set_compile_norm(enabled) -> None:
    _COMPILE[0] = bool(int(enabled)) if isinstance(enabled, str) else bool(enabled)


@partial(mx.compile, shapeless=True)
def _fused(y, weight, eps):
    """The same operations in the same order, as one graph."""
    y = y.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
    return y * weight.astype(mx.float32)


def _rms_norm(self, x: mx.array) -> mx.array:
    dtype = x.dtype
    if self.group_size is not None:
        # Reshape before the cast rather than after. An elementwise cast
        # commutes with a reshape, so the values are unchanged and the compiled
        # graph gets the whole chain including both conversions.
        y = x.reshape(*x.shape[:-1], -1, self.group_size)
        weight = self.weight.reshape(-1, self.group_size)
    else:
        y = x
        weight = self.weight
    # The original Flash-Next conversion stores one-centered gains, while
    # some newer converters store the zero-centered gains expected by
    # mlx-vlm. Keep this decision on each norm instance. This allows callers
    # to load both checkpoint families in one process.
    if getattr(self, "_flashnext_one_centered", True):
        weight = weight.astype(mx.float32)
    else:
        weight = 1.0 + weight.astype(mx.float32)
    if _COMPILE[0]:
        return _fused(y, weight, self.eps).reshape(x.shape).astype(dtype)
    y = y.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + self.eps)
    y = y * weight
    return y.reshape(x.shape).astype(dtype)


def apply() -> bool:
    """Patch Qwen4ExpRMSNorm. Returns True if this call did the patching."""
    global _applied
    if _applied:
        return False
    from mlx_vlm.models.qwen4_exp import language

    language.Qwen4ExpRMSNorm.__call__ = _rms_norm
    _applied = True
    return True


def configure(model, *, one_centered: bool) -> int:
    """Set the RMSNorm gain convention for one constructed model.

    ``apply`` patches the shared class method before model construction.
    Convention selection happens after construction, so it must be stored on
    each norm rather than in module-global state.
    """
    count = 0
    for _name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen4ExpRMSNorm":
            module._flashnext_one_centered = bool(one_centered)
            count += 1
    return count
