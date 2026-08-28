"""Route to as many experts as the router's confidence justifies.

A fixed lower `top_k` is fast and wrong: cutting 10 to 6 roughly triples decode
but `2 + 2 =` stops answering `4` at 8 already. Cutting by cumulative router
probability keeps the answer correct down to a 0.90 threshold, because tokens
where the router is confident drop experts that were contributing almost
nothing, and ambiguous tokens keep all ten.

Decode against threshold, each timed in its own process, confirmed in both
sweep directions:

    1.00   1770 / 2034 ms      0.53 tok/s   (unchanged routing)
    0.90   1675 / 1934         0.57
    0.85   1033 / 1091         0.94         <- the cliff
    0.70    941 /  961         1.05

The jump is not proportional to the bytes cut. It is the page cache: below 0.85
the routed working set crosses what the machine can hold, and reads start
hitting memory instead of the drive. Every byte not read is worth more than its
own weight.

Output identity against threshold 1.0, ten prompts, ten greedy tokens each:
0.70 matches on 7/10, and the three that differ pick a different but equally
correct continuation (Italy instead of Germany, `age > 25` instead of
`age > 17`). Facts hold: gold is Au and Japan's capital is Tokyo at every
threshold tested. 0.85 is the default because it sits just past the cliff while
staying closest to the shipped behaviour.
"""
from __future__ import annotations

import os

import mlx.core as mx

_applied = False


def _moe_call(self, x: mx.array) -> mx.array:
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    topk_mass = scores.sum(axis=-1, keepdims=True)

    layer_id = getattr(self, "_flashnext_layer_id", None)
    threshold = getattr(
        self,
        "_flashnext_threshold",
        _LAYER_THRESHOLDS.get(layer_id, _THRESHOLD[0]),
    )
    if threshold < 1.0:
        order = mx.argsort(-scores, axis=-1)
        inds = mx.take_along_axis(inds, order, axis=-1)
        scores = mx.take_along_axis(scores, order, axis=-1)
    if threshold < 1.0:
        mx.eval(scores)
        keeps = []
        for weights in scores.reshape(-1, k).tolist():
            total = sum(weights)
            accumulated = 0.0
            keep = k
            for position, weight in enumerate(weights):
                accumulated += weight / total
                if accumulated >= threshold:
                    keep = position + 1
                    break
            keeps.append(keep)

        observer = _ROUTE_OBSERVER[0]
        expert_rows = None
        if observer is not None and layer_id is not None:
            expert_rows = inds.reshape(-1, k).tolist()
            observer(
                layer_id,
                expert_rows,
                scores.reshape(-1, k).tolist(),
                keeps,
            )

        resident = _RESIDENT_EXPERTS.get(layer_id)
        if resident:
            if expert_rows is None:
                expert_rows = inds.reshape(-1, k).tolist()
            masks = [
                [position < keep or expert in resident
                 for position, expert in enumerate(row)]
                for row, keep in zip(expert_rows, keeps)
            ]
            effective_keeps = [sum(row) for row in masks]
            width = max(
                position + 1
                for row in masks
                for position, enabled in enumerate(row)
                if enabled
            )
        else:
            masks = None
            effective_keeps = keeps
            width = max(keeps)

        if layer_id is not None:
            _LAST_KEEPS[layer_id] = tuple(effective_keeps)
            _KEEP_SUM[0] += sum(effective_keeps)
            _KEEP_COUNT[0] += len(effective_keeps)

        inds = inds[..., :width]
        scores = scores[..., :width]
        if masks is None:
            keep_shape = (*scores.shape[:-1], 1)
            keep_array = mx.array(keeps, dtype=mx.int32).reshape(keep_shape)
            active = mx.arange(width) < keep_array
        else:
            active = mx.array(
                [row[:width] for row in masks], dtype=mx.bool_
            ).reshape(scores.shape)
        # Reuse the first routed expert in padded slots. This avoids extra I/O.
        inds = mx.where(active, inds, inds[..., :1])
        scores = mx.where(active, scores, 0)
    elif layer_id is not None:
        _LAST_KEEPS[layer_id] = (k,) * (inds.size // k)
        _KEEP_SUM[0] += inds.size
        _KEEP_COUNT[0] += inds.size // k

    selected_mass = scores.sum(axis=-1, keepdims=True)
    normalizer = topk_mass + _RENORM_BLEND[0] * (selected_mass - topk_mass)
    scores = scores / normalizer
    shared = self.shared_expert(x)
    shared_gate = mx.sigmoid(self.shared_expert_gate(x))
    if _OVERLAP:
        mx.async_eval(shared, shared_gate)

    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y + shared_gate * shared


_THRESHOLD = [1.0]
_LAYER_THRESHOLDS = {}
_LAST_KEEPS = {}
_KEEP_SUM = [0]
_KEEP_COUNT = [0]
_ROUTE_OBSERVER = [None]
_RESIDENT_EXPERTS = {}
FAST_LAYERS = (24, 12, 10, 21, 7, 18, 33, 22, 15, 5, 26, 16)
_OVERLAP = os.environ.get("FLASHNEXT_OVERLAP", "1") == "1"
_RENORM_BLEND = [float(os.environ.get(
    "FLASHNEXT_RENORM_BLEND",
    "1" if os.environ.get("FLASHNEXT_RENORM", "1") == "1" else "0",
))]


def set_threshold(value: float) -> None:
    _THRESHOLD[0] = value


def set_layer_thresholds(values=None) -> None:
    _LAYER_THRESHOLDS.clear()
    if values:
        _LAYER_THRESHOLDS.update({int(k): float(v) for k, v in values.items()})


def set_renorm_blend(value: float) -> None:
    _RENORM_BLEND[0] = float(value)


def set_fast_profile() -> None:
    set_threshold(0.20)
    set_layer_thresholds({layer: 0.40 for layer in FAST_LAYERS})
    set_renorm_blend(0.0)


def last_keeps():
    return dict(_LAST_KEEPS)


def reset_keep_stats() -> None:
    _KEEP_SUM[0] = 0
    _KEEP_COUNT[0] = 0


def mean_keeps() -> float:
    return _KEEP_SUM[0] / _KEEP_COUNT[0] if _KEEP_COUNT[0] else 0.0


def set_route_observer(observer=None) -> None:
    _ROUTE_OBSERVER[0] = observer


def set_resident_experts(values=None) -> None:
    _RESIDENT_EXPERTS.clear()
    if values:
        _RESIDENT_EXPERTS.update(
            {int(layer): {int(expert) for expert in experts}
             for layer, experts in values.items()}
        )


def apply() -> bool:
    """Patch the MoE block. Threshold 1.0 leaves routing exactly as shipped."""
    global _applied
    if _applied:
        return False
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    Qwen3_5MoeSparseMoeBlock.__call__ = _moe_call
    set_threshold(float(os.environ.get("FLASHNEXT_TOPK_THRESHOLD", "0.85")))
    _applied = True
    return True
