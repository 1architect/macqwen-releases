"""Compile the decode glue around each decoder layer's two branches.

Each Qwen4Exp decoder layer runs two hyper-connections (norm, two low-rank
projections, SiLU, sigmoid, a mean over the streams and the injection
weights) and two injections (broadcast multiply, reshape, add) around its
attention and MoE branches. That is roughly a third of the ~5,800 dispatches
of a decode token, on a GPU that the Metal capture showed launch-bound.

With ``FLASHNEXT_COMPILE_HC=1`` a decode step (one query row) runs both
injections as one compiled graph each. The hyper-connections themselves stay
plain: compiling them fused ``2 * sigmoid(x / hc_count)`` into one kernel whose
rounding differed from the separate kernels on 1 of 48 real layers, which
changed the token digest. The injections have no transcendental operation.
Prefill keeps the plain chain so no prompt length triggers a new trace. Off by
default.
"""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = [os.environ.get("FLASHNEXT_COMPILE_HC", "0") == "1"]
_APPLIED = [False]
_ORIGINAL = {}


def set_enabled(value: bool) -> None:
    ENABLED[0] = bool(value)


def _inject(hyper_input, branch, weights):
    injection = branch[..., None, :] * weights[..., None]
    return hyper_input + injection.reshape(*hyper_input.shape)


_compiled_inject = mx.compile(_inject)


def _decoder_call(self, hidden_states, input_ids, mask, cache, position_ids):
    if not ENABLED[0] or hidden_states.shape[1] != 1:
        return _ORIGINAL["decoder"](
            self, hidden_states, input_ids, mask, cache, position_ids
        )
    if "ple" in self:
        hidden_states = hidden_states + self.ple(hidden_states, input_ids, cache, mask)
    mixed, hyper_input, weights = self.attn_hyper_connection(hidden_states)
    if self.is_linear:
        branch = self.linear_attn(mixed, mask=mask, cache=cache)
    else:
        branch = self.self_attn(mixed, mask=mask, cache=cache, position_ids=position_ids)
    hidden_states = _compiled_inject(hyper_input, branch, weights)
    mixed, hyper_input, weights = self.mlp_hyper_connection(hidden_states)
    branch = self.mlp(mixed)
    return _compiled_inject(hyper_input, branch, weights)


def apply() -> bool:
    """Patch Qwen4ExpDecoderLayer once. Returns True if this call patched it."""
    if _APPLIED[0]:
        return False
    from mlx_vlm.models.qwen4_exp import language

    _ORIGINAL["decoder"] = language.Qwen4ExpDecoderLayer.__call__
    language.Qwen4ExpDecoderLayer.__call__ = _decoder_call
    _APPLIED[0] = True
    return True
