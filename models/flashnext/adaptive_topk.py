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
import time

import mlx.core as mx

_applied = False


def _keep_for_mass(weights, threshold: float) -> int:
    total = sum(weights)
    accumulated = 0.0
    for position, weight in enumerate(weights):
        accumulated += weight / total
        if accumulated >= threshold:
            return position + 1
    return len(weights)


def _tail_features(fast_y, shared_y, inputs, missing_fraction):
    scale = missing_fraction.astype(mx.float32)
    return mx.stack(
        (
            scale * fast_y.astype(mx.float32),
            scale * shared_y.astype(mx.float32),
            scale * inputs.astype(mx.float32),
        ),
        axis=-1,
    )


def _collect_tail_fit(layer_id, features, target) -> None:
    hidden = int(target.shape[-1])
    rows = features.reshape(-1, hidden, 3)
    values = target.astype(mx.float32).reshape(-1, hidden)
    gram = mx.einsum("nhf,nhg->hfg", rows, rows)
    rhs = mx.einsum("nhf,nh->hf", rows, values)
    previous = _TAIL_STATS.get(layer_id)
    if previous is not None:
        gram = gram + previous[0]
        rhs = rhs + previous[1]
    _TAIL_STATS[layer_id] = (gram, rhs)


def _moe_call(self, x: mx.array) -> mx.array:
    from models.flashnext import compiled, hostwindow
    from models.flashnext.expert_cache import (
        _EARLY_SUBMIT,
        _PROFILE,
        _TIMERS,
        _WARM_ON,
        record_layer,
        warm_layer,
    )

    warm_layer(self.switch_mlp, getattr(self, "_flashnext_layer_id", None))

    k = self.top_k
    if compiled.installed():
        inds, scores, topk_mass = compiled.router(self.gate(x), k)
    else:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
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
        # `inds` is only needed on the host when a prefetch, the warm
        # predictor, or cache-aware routing will read it. Forcing it every
        # layer costs a materialisation 48 times per token for a feature that
        # is off, so it joins the same sync as `scores` only when wanted.
        swap_active = (
            _SWAP_RESIDENT[0] is not None
            and scores.size // k <= _SWAP_MAX_ROWS[0]
        )
        needed = _EARLY_SUBMIT or _WARM_ON or swap_active
        if _PROFILE:
            began = time.perf_counter()
            mx.eval(scores, inds) if needed else mx.eval(scores)
            _TIMERS["score_sync"] += time.perf_counter() - began
            python_began = time.perf_counter()
        elif needed:
            mx.eval(scores, inds)
        else:
            mx.eval(scores)
        keeps = []
        fit_keeps = []
        minimum = max(
            1,
            min(k, _LAYER_MIN_KEEPS.get(layer_id, _MIN_KEEP[0])),
        )
        # `scores` was evaluated one line above, so this loop is a host copy
        # and pure Python. No kernel runs and no read is in flight.
        with hostwindow.window("keep_loop"):
            for weights in scores.reshape(-1, k).tolist():
                keep = _keep_for_mass(weights, threshold)
                keeps.append(max(minimum, keep))
                if _TAIL_MODE[0] == "collect" and layer_id is not None:
                    fit_threshold = (
                        _TAIL_SENSITIVE[0]
                        if layer_id in FAST_LAYERS
                        else _TAIL_THRESHOLD[0]
                    )
                    fit_keeps.append(_keep_for_mass(weights, fit_threshold))

        observer = _ROUTE_OBSERVER[0]
        expert_rows = None
        if observer is not None and layer_id is not None:
            limit = _OBSERVER_MAX_ROWS[0]
            if limit is not None and scores.size // k > limit:
                # A pin observer stops after a fixed number of rows per layer.
                # Handing it a whole prefill batch builds two lists of one row
                # per prompt token, 48 times, and discards nearly all of them.
                observer(
                    layer_id,
                    inds.reshape(-1, k)[:limit].tolist(),
                    scores.reshape(-1, k)[:limit].tolist(),
                    keeps[:limit],
                )
            else:
                expert_rows = inds.reshape(-1, k).tolist()
                observer(
                    layer_id,
                    expert_rows,
                    scores.reshape(-1, k).tolist(),
                    keeps,
                )

        swap = _SWAP_RESIDENT[0] if swap_active else None
        if swap is not None and layer_id is not None:
            if expert_rows is None:
                expert_rows = inds.reshape(-1, k).tolist()
            weight_rows = scores.reshape(-1, k).tolist()
            epsilon = _SWAP_EPSILON[0]
            moved = False
            swapped_experts, swapped_weights = [], []
            for row, weights, keep in zip(expert_rows, weight_rows, keeps):
                new_row, new_weights = swap_row(
                    row, weights, keep,
                    lambda expert: swap(layer_id, expert),
                    epsilon,
                )
                moved = moved or new_row is not row
                swapped_experts.append(new_row)
                swapped_weights.append(new_weights)
            if moved:
                shape = inds.shape
                inds = mx.array(swapped_experts, dtype=inds.dtype).reshape(shape)
                scores = mx.array(
                    swapped_weights, dtype=scores.dtype
                ).reshape(scores.shape)
                expert_rows = swapped_experts

        resident = _RESIDENT_EXPERTS.get(layer_id)
        if resident:
            if expert_rows is None:
                # `inds` joined the sync above only when something asked for
                # it. Otherwise this call forces its own, so the enclosing
                # window is not host-only and must not be counted as free.
                if not needed:
                    hostwindow.note_eval()
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

        if masks is None:
            if _EARLY_SUBMIT or _WARM_ON:
                rows = expert_rows
                if rows is None:
                    rows = inds.reshape(-1, k).tolist()
                routed = [
                    row[position] if position < keep else row[0]
                    for row, keep in zip(rows, keeps)
                    for position in range(width)
                ]
                wanted = list(dict.fromkeys(routed))
                record_layer(layer_id, wanted)
                prefetch = getattr(self.switch_mlp, "prefetch", None)
                if prefetch is not None:
                    prefetch(wanted)

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
        if _PROFILE:
            _TIMERS["topk_python"] += time.perf_counter() - python_began
    elif layer_id is not None:
        _LAST_KEEPS[layer_id] = (k,) * (inds.size // k)
        _KEEP_SUM[0] += inds.size
        _KEEP_COUNT[0] += inds.size // k
        # At threshold 1.0 nothing is dropped, but the observer still has to
        # run: exact-quality selects its resident experts from it, so without
        # this the profile pinned nothing and silently lost its own feature.
        observer = _ROUTE_OBSERVER[0]
        if observer is not None:
            limit = _OBSERVER_MAX_ROWS[0]
            rows_here = inds.size // k
            if limit is not None and rows_here > limit:
                rows_here = limit
                rows = inds.reshape(-1, k)[:limit].tolist()
                score_rows = scores.reshape(-1, k)[:limit].tolist()
            else:
                rows = inds.reshape(-1, k).tolist()
                score_rows = scores.reshape(-1, k).tolist()
            observer(layer_id, rows, score_rows, [k] * rows_here)

    if compiled.installed():
        scores, normalizer = compiled.renorm(scores, topk_mass, _RENORM_BLEND[0])
    else:
        selected_mass = scores.sum(axis=-1, keepdims=True)
        normalizer = topk_mass + _RENORM_BLEND[0] * (selected_mass - topk_mass)
        scores = scores / normalizer
    shared_began = time.perf_counter() if _PROFILE else 0.0
    shared = self.shared_expert(x)
    shared_gate = mx.sigmoid(self.shared_expert_gate(x))
    if _OVERLAP:
        mx.async_eval(shared, shared_gate)
    if _PROFILE:
        _TIMERS["shared_expert"] += time.perf_counter() - shared_began

    expert_values = self.switch_mlp(x, inds)
    y = (expert_values * scores[..., None]).sum(axis=-2)
    shared_y = shared_gate * shared

    predictor_mode = _TAIL_MODE[0]
    if predictor_mode in ("collect", "apply") and layer_id is not None:
        predictor_keeps = fit_keeps if predictor_mode == "collect" else keeps
        predictor_shape = (*scores.shape[:-1], 1)
        predictor_keep_array = mx.array(
            predictor_keeps, dtype=mx.int32
        ).reshape(predictor_shape)
        predictor_active = mx.arange(scores.shape[-1]) < predictor_keep_array
        fast_scores = mx.where(
            predictor_active,
            scores * normalizer / topk_mass,
            0,
        )
        fast_y = (expert_values * fast_scores[..., None]).sum(axis=-2)
        fast_mass = mx.where(predictor_active, scores * normalizer, 0).sum(
            axis=-1,
            keepdims=True,
        )
        missing_fraction = mx.clip(
            1.0 - fast_mass / topk_mass,
            0.0,
            1.0,
        )
        features = _tail_features(fast_y, shared_y, x, missing_fraction)
        if predictor_mode == "collect":
            _collect_tail_fit(layer_id, features, y - fast_y)
        else:
            coefficients = _TAIL_COEFFICIENTS.get(layer_id)
            if coefficients is not None:
                correction = (
                    features * coefficients[None, ...]
                ).sum(axis=-1).astype(y.dtype)
                y = y + correction
    return y + shared_y


_THRESHOLD = [1.0]
_LAYER_THRESHOLDS = {}
_MIN_KEEP = [1]
_LAYER_MIN_KEEPS = {}
_LAST_KEEPS = {}
_KEEP_SUM = [0]
_KEEP_COUNT = [0]
_ROUTE_OBSERVER = [None]
# Cache-aware routing. Adaptive top-k scores k experts and keeps the first few,
# so the rest are already scored and discarded. When a kept expert has to come
# off the drive and a discarded one is already in memory within `epsilon` of
# its score, taking the resident one removes a physical read. Measured
# opportunity: 13.9% of cold reads at epsilon 0.02, for 0.15% of the kept
# routed mass, against the 15% adaptive top-k already discards by design.
# `_SWAP_RESIDENT[0]` is a predicate (layer, expert) -> bool. None disables.
_SWAP_RESIDENT = [None]
_SWAP_EPSILON = [0.02]
# The swap spends Python time per token to remove a physical read. Prefill
# reads one expert row for a whole batch of tokens, so the read it removes is
# already amortised while the Python cost is paid per token. A cache-aware
# chat turn prefilled at 35.1 tok/s against 45.0 exact. Above this many rows
# in one call the swap stands down and the batch routes exactly.
_SWAP_MAX_ROWS = [int(os.environ.get("FLASHNEXT_SWAP_MAX_ROWS", "4"))]
_RESIDENT_EXPERTS = {}
# Rows handed to the route observer per call. None means every row.
_OBSERVER_MAX_ROWS = [None]
_TAIL_MODE = ["off"]
_TAIL_STATS = {}
_TAIL_COEFFICIENTS = {}
_TAIL_THRESHOLD = [0.20]
_TAIL_SENSITIVE = [0.40]
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


def set_min_keep(value: int, layer_values=None) -> None:
    _MIN_KEEP[0] = max(1, int(value))
    _LAYER_MIN_KEEPS.clear()
    if layer_values:
        _LAYER_MIN_KEEPS.update(
            {int(layer): max(1, int(keep)) for layer, keep in layer_values.items()}
        )


def set_fast_profile() -> None:
    base = float(os.environ.get("FLASHNEXT_FAST_THRESHOLD", "0.20"))
    sensitive = float(os.environ.get("FLASHNEXT_FAST_SENSITIVE", "0.40"))
    top_count = int(os.environ.get(
        "FLASHNEXT_FAST_TOP_COUNT",
        os.environ.get("FLASHNEXT_FAST_PROTECTED", "0"),
    ))
    top_threshold = float(os.environ.get(
        "FLASHNEXT_FAST_TOP_THRESHOLD",
        os.environ.get("FLASHNEXT_FAST_PROTECTED_THRESHOLD", "0.85"),
    ))
    mid_count = int(os.environ.get("FLASHNEXT_FAST_MID_COUNT", "0"))
    mid_threshold = float(
        os.environ.get("FLASHNEXT_FAST_MID_THRESHOLD", str(sensitive))
    )
    layer_thresholds = {layer: sensitive for layer in FAST_LAYERS}
    layer_thresholds.update(
        {layer: top_threshold for layer in FAST_LAYERS[:top_count]}
    )
    layer_thresholds.update(
        {
            layer: mid_threshold
            for layer in FAST_LAYERS[top_count:top_count + mid_count]
        }
    )
    minimum = int(os.environ.get("FLASHNEXT_FAST_MIN_KEEP", "1"))
    sensitive_minimum = int(
        os.environ.get("FLASHNEXT_FAST_SENSITIVE_MIN_KEEP", str(minimum))
    )
    top_minimum = int(os.environ.get(
        "FLASHNEXT_FAST_TOP_MIN_KEEP",
        str(sensitive_minimum),
    ))
    mid_minimum = int(os.environ.get(
        "FLASHNEXT_FAST_MID_MIN_KEEP",
        str(sensitive_minimum),
    ))
    layer_minimums = {
        layer: sensitive_minimum for layer in FAST_LAYERS
    }
    layer_minimums.update(
        {layer: top_minimum for layer in FAST_LAYERS[:top_count]}
    )
    layer_minimums.update(
        {
            layer: mid_minimum
            for layer in FAST_LAYERS[top_count:top_count + mid_count]
        }
    )
    set_threshold(base)
    set_layer_thresholds(layer_thresholds)
    set_min_keep(minimum, layer_minimums)
    set_renorm_blend(float(os.environ.get("FLASHNEXT_FAST_RENORM", "0")))


def last_keeps():
    return dict(_LAST_KEEPS)


def reset_keep_stats() -> None:
    _KEEP_SUM[0] = 0
    _KEEP_COUNT[0] = 0


def mean_keeps() -> float:
    return _KEEP_SUM[0] / _KEEP_COUNT[0] if _KEEP_COUNT[0] else 0.0


def set_swap_resident(predicate=None, epsilon: float | None = None,
                      max_rows: int | None = None) -> None:
    """Route to a resident expert when a cold one scores no better.

    `predicate(layer, expert)` reports whether the expert's rows are already
    in memory. Passing None restores exact routing. This changes what the
    model computes, so it is gated by the reasoning quality gate rather than
    by token identity. `max_rows` is the largest batch the swap runs on, so
    that prefill keeps exact routing and pays no Python cost.
    """
    _SWAP_RESIDENT[0] = predicate
    if epsilon is not None:
        _SWAP_EPSILON[0] = float(epsilon)
    if max_rows is not None:
        _SWAP_MAX_ROWS[0] = int(max_rows)


def swap_row(experts, weights, keep, resident, epsilon):
    """Exchange cold kept experts for resident discarded ones.

    Returns new expert and weight rows. The cheapest kept expert is offered
    first, because giving up the least routed mass is the point, and each
    discarded expert can only serve once.
    """
    kept = list(range(keep))
    spare = [
        position
        for position in range(keep, len(experts))
        if resident(experts[position])
    ]
    if not spare:
        return experts, weights
    cold = [position for position in kept if not resident(experts[position])]
    if not cold:
        return experts, weights
    experts = list(experts)
    weights = list(weights)
    used = set()
    for position in sorted(cold, key=lambda index: weights[index]):
        best = None
        for candidate in spare:
            if candidate in used:
                continue
            if weights[position] - weights[candidate] <= epsilon:
                if best is None or weights[candidate] > weights[best]:
                    best = candidate
        if best is None:
            continue
        used.add(best)
        experts[position], experts[best] = experts[best], experts[position]
        weights[position], weights[best] = weights[best], weights[position]
    return experts, weights


def set_route_observer(observer=None, max_rows: int | None = None) -> None:
    """Watch routing decisions. `max_rows` caps the rows handed over per
    call, for an observer that stops after a fixed count. A measurement that
    wants every token must leave it None."""
    _ROUTE_OBSERVER[0] = observer
    _OBSERVER_MAX_ROWS[0] = None if max_rows is None else int(max_rows)


def set_resident_experts(values=None) -> None:
    _RESIDENT_EXPERTS.clear()
    if values:
        _RESIDENT_EXPERTS.update(
            {int(layer): {int(expert) for expert in experts}
             for layer, experts in values.items()}
        )


def start_tail_fit(threshold: float = 0.20, sensitive: float = 0.40) -> None:
    _TAIL_STATS.clear()
    _TAIL_COEFFICIENTS.clear()
    _TAIL_THRESHOLD[0] = float(threshold)
    _TAIL_SENSITIVE[0] = float(sensitive)
    _TAIL_MODE[0] = "collect"


def finish_tail_fit(
    ridge: float = 0.01,
    clip: float = 4.0,
    per_channel: bool = True,
) -> int:
    if not _TAIL_STATS:
        _TAIL_MODE[0] = "off"
        return 0
    pending = [value for pair in _TAIL_STATS.values() for value in pair]
    mx.eval(*pending)
    identity = mx.eye(3, dtype=mx.float32)[None]
    for layer_id, (gram, rhs) in _TAIL_STATS.items():
        if not per_channel:
            gram = gram.sum(axis=0, keepdims=True)
            rhs = rhs.sum(axis=0, keepdims=True)
        scale = (
            gram[:, 0, 0] + gram[:, 1, 1] + gram[:, 2, 2]
        ) / 3.0
        regularized = gram + identity * (
            float(ridge) * scale[:, None, None] + 1e-6
        )
        coefficients = mx.linalg.solve(
            regularized,
            rhs[..., None],
            stream=mx.cpu,
        )[..., 0]
        coefficients = mx.clip(
            coefficients, -float(clip), float(clip)
        )
        _TAIL_COEFFICIENTS[layer_id] = (
            coefficients if per_channel else coefficients[0]
        )
    mx.eval(*_TAIL_COEFFICIENTS.values())
    _TAIL_STATS.clear()
    _TAIL_MODE[0] = "apply"
    return len(_TAIL_COEFFICIENTS)


def disable_tail_predictor() -> None:
    _TAIL_MODE[0] = "off"
    _TAIL_STATS.clear()
    _TAIL_COEFFICIENTS.clear()


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
