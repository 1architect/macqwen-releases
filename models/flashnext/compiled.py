"""Compile the elementwise chains that sit between this model's matmuls.

`mx.compile` was probed on three chains and never applied. The probe was
bit-exact under `mx.array_equal` and measured 28% on the PLE gate chain, 5.7%
on `_normalize_qk` and 2.6% on the router top-k chain. The research note then
estimated 10 to 20 ms per token if applied everywhere.

That estimate does not survive the probe's own numbers. The three measured
chains save 0.073, 0.013 and 0.010 ms per call, and they run 1, 36 and 48
times per token, so they sum to about 1.0 ms. This module exists to measure
the real figure in the complete runtime rather than argue about it, and the
arithmetic above says to expect a result inside the resolution band.

Compilation fuses elementwise work. It does not help sorts, reductions over
large axes, or matmuls, which is most of what a token costs. `mx.compile` is
also already applied upstream: `gated_delta.py` decorates its recurrence with
`@partial(mx.compile, shapeless=True)` and `swiglu` carries the same
decorator. Nothing here touches either.

Set `FLASHNEXT_COMPILE=1`, or call `install()`. Every chain is checked against
its plain form with `mx.array_equal` before it is installed, so an inexact
compile cannot reach a benchmark.
"""
from __future__ import annotations

import math
import os
from functools import partial

import mlx.core as mx

ENABLED = os.environ.get("FLASHNEXT_COMPILE") == "1"

_INSTALLED = [False]
_ORIGINALS: dict = {}
_ROUTER_CACHE: dict = {}


# --- the chains, plain and compiled -----------------------------------------


def _normalize_qk_plain(q, k):
    scale = q.shape[-1] ** -0.5
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    return q * scale, k


# Not shapeless: `scale` is read from `q.shape`, so a shapeless trace would
# bake one head dimension into the graph and stay wrong for any other. One
# trace per shape is correct and costs a compile the first time each is seen.
_normalize_qk_compiled = mx.compile(_normalize_qk_plain)


def _ple_gate_plain(keys, queries, values, hidden_size):
    gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(hidden_size)
    gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
    return mx.sigmoid(gate) * values[..., None, :]


def _router_plain(logits, k):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    return inds, scores, scores.sum(axis=-1, keepdims=True)


def _renorm_plain(scores, topk_mass, blend):
    selected_mass = scores.sum(axis=-1, keepdims=True)
    normalizer = topk_mass + blend * (selected_mass - topk_mass)
    return scores / normalizer, normalizer


_renorm_compiled = mx.compile(_renorm_plain, shapeless=True)


def _combine_plain(expert_values, scores, shared_gate, shared):
    y = (expert_values * scores[..., None]).sum(axis=-2)
    return y + shared_gate * shared


_combine_compiled = mx.compile(_combine_plain, shapeless=True)


def ple_gate(keys, queries, values, hidden_size):
    """Compiled PLE gate, keyed on the static hidden size."""
    fn = _ROUTER_CACHE.get(("ple", hidden_size))
    if fn is None:
        fn = mx.compile(
            partial(_ple_gate_plain, hidden_size=hidden_size), shapeless=True
        )
        _ROUTER_CACHE[("ple", hidden_size)] = fn
    return fn(keys, queries, values)


def router(logits, k):
    """Compiled router top-k chain, keyed on the static expert count.

    `k` is a Python int, so it becomes a constant inside the compiled graph.
    One compiled function per distinct `k` keeps that correct.
    """
    fn = _ROUTER_CACHE.get(("router", k))
    if fn is None:
        # Not shapeless: `[..., -k:]` cannot infer its output shape without
        # one, and MLX refuses the trace.
        fn = mx.compile(partial(_router_plain, k=k))
        _ROUTER_CACHE[("router", k)] = fn
    return fn(logits)


def renorm(scores, topk_mass, blend):
    return _renorm_compiled(scores, topk_mass, mx.array(blend, scores.dtype))


def combine(expert_values, scores, shared_gate, shared):
    return _combine_compiled(expert_values, scores, shared_gate, shared)


# --- exactness gate ---------------------------------------------------------


def verify() -> list:
    """Check every chain against its plain form. Returns the failures."""
    failures = []
    q = mx.random.normal((1, 4, 1, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, 1, 128)).astype(mx.bfloat16)
    plain = _normalize_qk_plain(q, k)
    fused = _normalize_qk_compiled(q, k)
    mx.eval(plain, fused)
    if not all(mx.array_equal(a, b) for a, b in zip(plain, fused)):
        failures.append("_normalize_qk")

    hidden = 2560
    keys = mx.random.normal((1, 1, 4, hidden)).astype(mx.bfloat16)
    queries = mx.random.normal((1, 1, 4, hidden)).astype(mx.bfloat16)
    values = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    plain = _ple_gate_plain(keys, queries, values, hidden)
    fused = ple_gate(keys, queries, values, hidden)
    mx.eval(plain, fused)
    if not mx.array_equal(plain, fused):
        failures.append("ple_gate")

    logits = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    plain = _router_plain(logits, 10)
    fused = router(logits, 10)
    mx.eval(plain, fused)
    if not all(mx.array_equal(a, b) for a, b in zip(plain, fused)):
        failures.append("router")

    scores = mx.random.uniform(shape=(1, 1, 10)).astype(mx.bfloat16)
    mass = scores.sum(axis=-1, keepdims=True)
    plain = _renorm_plain(scores, mass, mx.array(1.0, scores.dtype))
    fused = renorm(scores, mass, 1.0)
    mx.eval(plain, fused)
    if not all(mx.array_equal(a, b) for a, b in zip(plain, fused)):
        failures.append("renorm")

    values = mx.random.normal((1, 1, 10, 2560)).astype(mx.bfloat16)
    weights = mx.random.uniform(shape=(1, 1, 10)).astype(mx.bfloat16)
    gate = mx.random.uniform(shape=(1, 1, 1)).astype(mx.bfloat16)
    shared = mx.random.normal((1, 1, 2560)).astype(mx.bfloat16)
    plain = _combine_plain(values, weights, gate, shared)
    fused = combine(values, weights, gate, shared)
    mx.eval(plain, fused)
    if not mx.array_equal(plain, fused):
        failures.append("combine")
    return failures


# --- installation -----------------------------------------------------------


def install(verify_first: bool = True) -> bool:
    """Patch the upstream chains. Returns whether anything changed."""
    if _INSTALLED[0]:
        return False
    if verify_first:
        failures = verify()
        if failures:
            raise SystemExit(
                "compiled chains are not bit-exact: "
                + ", ".join(failures)
                + ". Refusing to install."
            )
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedDeltaNet,
        Qwen4ExpPLELayer,
    )

    _ORIGINALS["_normalize_qk"] = Qwen4ExpGatedDeltaNet._normalize_qk
    Qwen4ExpGatedDeltaNet._normalize_qk = staticmethod(_normalize_qk_compiled)

    _ORIGINALS["ple_call"] = Qwen4ExpPLELayer.__call__
    Qwen4ExpPLELayer.__call__ = _ple_call
    _INSTALLED[0] = True
    return True


def uninstall() -> bool:
    if not _INSTALLED[0]:
        return False
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedDeltaNet,
        Qwen4ExpPLELayer,
    )

    Qwen4ExpGatedDeltaNet._normalize_qk = _ORIGINALS["_normalize_qk"]
    Qwen4ExpPLELayer.__call__ = _ORIGINALS["ple_call"]
    _INSTALLED[0] = False
    return True


def installed() -> bool:
    return _INSTALLED[0]


def _ple_call(self, hidden_states, input_ids, cache, mask):
    """`Qwen4ExpPLELayer.__call__` with the gate chain compiled.

    The body is copied from upstream. Only the three lines that build `gate`
    and `gated_values` move into `ple_gate`; every other line and its order is
    unchanged, so a divergence here would be a copy error rather than a
    compilation effect. `verify()` compares the moved chain against the same
    arithmetic before this is installed.
    """
    embeddings = self.ple_embedding(input_ids, cache)
    keys = self.norm_key(self.key_proj(embeddings)).reshape(
        *hidden_states.shape[:-1], self.hc_count, self.hidden_size
    )
    values = self.value_proj(embeddings)
    queries = self.norm_query(hidden_states).reshape(
        *hidden_states.shape[:-1], self.hc_count, self.hidden_size
    )
    gated_values = ple_gate(keys, queries, values, self.hidden_size)
    gated_values = gated_values.reshape(*hidden_states.shape)
    normed = self.norm_conv(gated_values)
    if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
        gated_values = mx.where(mask[..., None], gated_values, 0)
        normed = mx.where(mask[..., None], normed, 0)
    return gated_values + self._short_conv(normed, cache)
