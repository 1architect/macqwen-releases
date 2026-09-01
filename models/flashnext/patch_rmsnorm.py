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

import mlx.core as mx

_applied = False


def _rms_norm(self, x: mx.array) -> mx.array:
    dtype = x.dtype
    y = x.astype(mx.float32)
    if self.group_size is not None:
        y = y.reshape(*y.shape[:-1], -1, self.group_size)
        weight = self.weight.reshape(-1, self.group_size)
    else:
        weight = self.weight
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + self.eps)
    y = y * weight.astype(mx.float32)
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
