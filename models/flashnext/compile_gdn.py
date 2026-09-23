"""Compile two elementwise chains inside each GatedDeltaNet layer (research).

At P15 the GPU half of a decode token is launch-bound (about 5,800 dispatches,
about 19 us each). Each of the 36 GatedDeltaNet layers runs two chains of
small kernels that ``mx.compile`` can fuse:

- the Qwen4 q/k L2 normalization: square, sum, add, rsqrt and multiply for
  both q and k, plus the query scale;
- the gated output norm after ``mx.fast.rms_norm``: two float32 casts, the
  gate activation, a multiply and the cast back.

``FLASHNEXT_COMPILE_GDN=1`` runs both compiled for one-row decode calls;
prefill keeps the plain chain so no prompt length retraces. Off by default.
A fused kernel can round a transcendental differently from separate kernels
(compiled hyper-connections changed the digest on 2026-09-22), so each chain
must pass ``bench_gdn_exact`` on captured real inputs before a speed run.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

FLAG = "FLASHNEXT_COMPILE_GDN"
ENABLED = [os.environ.get(FLAG, "0") == "1"]
_APPLIED = [False]
_ORIGINAL = {}


def set_enabled(value: bool) -> None:
    ENABLED[0] = bool(value)


def _normalize_qk(q, k):
    # Same operations, order and constants as Qwen4ExpGatedDeltaNet.
    scale = q.shape[-1] ** -0.5
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    return q * scale, k


def _gate_sigmoid(y, gate):
    return y.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))


def _gate_silu(y, gate):
    return y.astype(mx.float32) * nn.silu(gate.astype(mx.float32))


compiled_normalize_qk = mx.compile(_normalize_qk)
compiled_gate = {
    "sigmoid": mx.compile(_gate_sigmoid),
    "silu": mx.compile(_gate_silu),
}


def normalize_qk(self, q, k):
    if ENABLED[0] and q.shape[1] == 1:
        return compiled_normalize_qk(q, k)
    return _ORIGINAL["normalize_qk"](self, q, k)


def gated_norm(self, x, gate):
    if ENABLED[0] and x.shape[1] == 1:
        dtype = x.dtype
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        fused = compiled_gate["sigmoid" if self.activation == "sigmoid" else "silu"]
        return fused(y, gate).astype(dtype)
    return _ORIGINAL["gated_norm"](self, x, gate)


def apply() -> bool:
    """Patch the Qwen4 GatedDeltaNet chains once (flag read at call time)."""
    if _APPLIED[0]:
        return False
    from mlx_vlm.models.qwen4_exp import language

    _ORIGINAL["normalize_qk"] = language.Qwen4ExpGatedDeltaNet._normalize_qk
    _ORIGINAL["gated_norm"] = language.Qwen4ExpRMSNormGated.__call__
    language.Qwen4ExpGatedDeltaNet._normalize_qk = normalize_qk
    language.Qwen4ExpRMSNormGated.__call__ = gated_norm
    _APPLIED[0] = True
    return True
