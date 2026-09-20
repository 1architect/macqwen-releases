"""Opt-in one-time preparation for Bonsai's packed QMM metadata.

The checkpoint stores FP16 scales and biases while the real packed projection
call sites receive FP32 activations.  MLX promotes the metadata on every
``quantized_matmul`` call.  This module materializes FP32 metadata once and
keeps the packed uint32 weights and stock QMM arithmetic unchanged.
"""
from __future__ import annotations

import importlib
import math
import os
import time


PREPARATION_GROUP_SIZE = 8
PREPARATION_DEADLINE_SECONDS = 300.0
PREPARATION_MAX_ADDITIONAL_BYTES = 1 << 30


class QMMResourceError(RuntimeError):
    """Metadata preparation exceeded its explicit resource budget."""


def new_counters() -> dict[str, int]:
    return {
        "prepared_calls": 0,
        "fp32_calls": 0,
        "lower_precision_calls": 0,
        "unsupported_calls": 0,
        "phase_calls": {"unknown": 0, "prefill": 0, "decode": 0},
        "phase_prepared_calls": {"unknown": 0, "prefill": 0, "decode": 0},
        "phase_fp32_calls": {"unknown": 0, "prefill": 0, "decode": 0},
        "phase_lower_precision_calls": {"unknown": 0, "prefill": 0, "decode": 0},
        "phase_unsupported_calls": {"unknown": 0, "prefill": 0, "decode": 0},
        "module_calls": {},
        "admission_calls": 0,
        "groups_considered": 0,
        "groups_admitted": 0,
        "groups_rejected": 0,
        "rejected_modules": 0,
        "rejection_reasons": {},
        "admission_failures": [],
        "preparation_failures": [],
        "current_phase": "unknown",
    }


def set_phase(counters: dict, phase: str) -> None:
    if not isinstance(counters, dict):
        return
    counters["current_phase"] = phase if phase in ("prefill", "decode") else "unknown"


def _is_packed(module, packed_type) -> bool:
    import mlx.core as mx

    return (
        isinstance(module, packed_type)
        and not bool(getattr(module, "embedding", False))
        and getattr(module, "weight", None) is not None
        and getattr(module, "scales", None) is not None
        and getattr(module, "biases", None) is not None
        and module.weight.dtype == mx.uint32
        and module.scales.dtype == mx.float16
        and module.biases.dtype == mx.float16
    )


def _stock_projection(runtime, module, x, scales, biases):
    import mlx.core as mx

    if module.block:
        x = runtime.fwht(x, module.block, module.signs)
    return mx.quantized_matmul(
        x,
        module.weight,
        scales,
        biases,
        transpose=True,
        group_size=128,
        bits=2,
    )


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _dtype_itemsize(dtype) -> int:
    try:
        itemsize = getattr(dtype, "itemsize")
        itemsize = itemsize() if callable(itemsize) else itemsize
        return int(itemsize)
    except (AttributeError, TypeError, ValueError):
        return {
            "mlx.core.float16": 2,
            "mlx.core.bfloat16": 2,
            "mlx.core.float32": 4,
        }.get(str(dtype), 4)


def _group_plan(index: int, items) -> dict:
    source_bytes = sum(item[3] for item in items)
    replacement_bytes = sum(item[4] for item in items)
    return {
        "group_index": index,
        "names": [item[0] for item in items],
        "source_bytes": source_bytes,
        "replacement_bytes": replacement_bytes,
        "persistent_growth_bytes": replacement_bytes - source_bytes,
        "temporary_coexistence_bytes": source_bytes + replacement_bytes,
        "items": items,
    }


def _record_group_rejection(
    stats: dict,
    counters: dict,
    group: dict,
    reason: str,
    message: str | None = None,
) -> None:
    failure = {
        "group_index": int(group["group_index"]),
        "names": list(group["names"]),
        "reason": reason,
    }
    if message:
        failure["message"] = message
    stats["admission_failures"].append(failure)
    stats["rejected_groups"].append(failure)
    stats["failure"] = failure
    counters["groups_rejected"] += 1
    counters["rejected_modules"] += len(group["names"])
    reasons = counters["rejection_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    counters["admission_failures"].append(failure)


def _memory_snapshot() -> tuple[int | None, int | None]:
    try:
        import mlx.core as mx

        active = getattr(mx, "get_active_memory", None)
        peak = getattr(mx, "get_peak_memory", None)
        return (
            int(active()) if callable(active) else None,
            int(peak()) if callable(peak) else None,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None, None


def _resource_check(
    started: float,
    stats: dict,
    *,
    max_seconds: float,
    max_additional_bytes: int,
    pressure_check=None,
    clock=time.perf_counter,
) -> None:
    elapsed = clock() - started
    active, peak = _memory_snapshot()
    if active is not None:
        stats["preparation_peak_active_bytes"] = max(
            int(stats.get("preparation_peak_active_bytes", 0)), active
        )
    if peak is not None:
        stats["preparation_peak_bytes"] = max(
            int(stats.get("preparation_peak_bytes", 0)), peak
        )
    stats["elapsed_seconds"] = elapsed
    if elapsed > max_seconds:
        raise QMMResourceError(
            f"metadata preparation deadline exceeded ({elapsed:.3f}s > {max_seconds:.3f}s)"
        )
    if int(stats.get("additional_bytes", 0)) > max_additional_bytes:
        raise QMMResourceError(
            "metadata preparation additional allocation exceeded "
            f"{max_additional_bytes} bytes"
        )
    if pressure_check is not None and pressure_check(stats):
        raise QMMResourceError("metadata preparation pressure admission failed")


def install(
    model,
    enabled: bool,
    counters: dict[str, int] | None = None,
    *,
    group_size: int = PREPARATION_GROUP_SIZE,
    max_seconds: float | None = None,
    max_additional_bytes: int | None = None,
    max_temporary_bytes: int | None = None,
    pressure_check=None,
    clock=time.perf_counter,
) -> dict:
    """Prepare eligible packed projections and install the opt-in dispatch.

    The original FP16 metadata is replaced after the prepared arrays have
    been evaluated.  A lower-precision call explicitly casts those values
    back to FP16 before the unchanged stock QMM operation, so preparation is
    never silently applied as a precision change.
    """
    stats = {
        "enabled": bool(enabled),
        "prepared_modules": 0,
        "prepared_arrays": 0,
        "source_bytes": 0,
        "prepared_bytes": 0,
        "projected_prepared_bytes": 0,
        "additional_bytes": 0,
        "projected_persistent_growth_bytes": 0,
        "projected_persistent_bytes": 0,
        "temporary_coexistence_bytes": 0,
        "temporary_original_replacement_bytes": 0,
        "elapsed_seconds": 0.0,
        "activation_dtype": "mlx.core.float32",
        "eligible_module_manifest": [],
        "prepared_module_manifest": [],
        "preparation_peak_bytes": 0,
        "preparation_peak_active_bytes": 0,
        "group_size": int(group_size),
        "deadline_seconds": (
            _env_float("MACQWEN_BONSAI2_QMM_DEADLINE_S", PREPARATION_DEADLINE_SECONDS)
            if max_seconds is None else float(max_seconds)
        ),
        "max_additional_bytes": (
            int(_env_float(
                "MACQWEN_BONSAI2_QMM_MAX_ADDITIONAL_MB",
                PREPARATION_MAX_ADDITIONAL_BYTES / (1024 * 1024),
            ) * 1024 * 1024)
            if max_additional_bytes is None else int(max_additional_bytes)
        ),
        "max_temporary_bytes": (
            None if max_temporary_bytes is None else int(max_temporary_bytes)
        ),
        "groups": [],
        "rejected_groups": [],
        "admission_failures": [],
        "admission_status": "not_attempted",
        "failure": None,
        "pressure_callback": (
            getattr(pressure_check, "__name__", type(pressure_check).__name__)
            if pressure_check is not None else None
        ),
    }
    if counters is None:
        counters = new_counters()
    else:
        counters.clear()
        counters.update(new_counters())
    try:
        runtime = importlib.import_module("runtime")
    except ImportError:
        if not enabled:
            return stats
        raise
    packed_type = runtime.Packed
    original = getattr(packed_type, "_bonsai_qmm_original_call", None)
    if original is None:
        original = packed_type.__call__
        packed_type._bonsai_qmm_original_call = original

    # Remove the earlier Q2 wrapper, if one was installed in this interpreter,
    # before composing a fresh metadata wrapper for the newly loaded model.
    q2_stock = getattr(packed_type, "_bonsai_stock_call", None)
    if q2_stock is not None:
        packed_type.__call__ = q2_stock
        delattr(packed_type, "_bonsai_stock_call")
    packed_type.__call__ = original
    if not enabled:
        return stats

    import mlx.core as mx

    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("metadata preparation group_size must be positive")
    if not math.isfinite(float(stats["deadline_seconds"])) or stats["deadline_seconds"] < 0:
        raise ValueError("metadata preparation deadline must be finite and non-negative")
    if stats["max_additional_bytes"] < 0:
        raise ValueError("metadata preparation additional budget must be non-negative")
    if (
        stats["max_temporary_bytes"] is not None
        and stats["max_temporary_bytes"] < 0
    ):
        raise ValueError("metadata preparation temporary budget must be non-negative")
    started = clock()
    eligible = []
    for name, module in model.named_modules():
        if not _is_packed(module, packed_type):
            continue
        if tuple(module.scales.shape) != tuple(module.biases.shape):
            raise ValueError(f"packed metadata shape mismatch at {name}")
        if module.scales.ndim != 2:
            raise ValueError(f"packed metadata rank mismatch at {name}")
        elements = int(module.scales.size)
        source_bytes = elements * (
            _dtype_itemsize(module.scales.dtype)
            + _dtype_itemsize(module.biases.dtype)
        )
        replacement_bytes = elements * _dtype_itemsize(mx.float32) * 2
        stats["eligible_module_manifest"].append({
            "name": name,
            "shape": [int(value) for value in module.scales.shape],
            "elements": elements,
            "source_bytes": source_bytes,
            "replacement_bytes": replacement_bytes,
            "persistent_growth_bytes": replacement_bytes - source_bytes,
        })
        stats["source_bytes"] += source_bytes
        stats["projected_prepared_bytes"] += replacement_bytes
        eligible.append((name, module, elements, source_bytes, replacement_bytes))
    stats["eligible_modules"] = len(eligible)
    stats["projected_persistent_bytes"] = stats["projected_prepared_bytes"]
    stats["projected_persistent_growth_bytes"] = (
        stats["projected_prepared_bytes"] - stats["source_bytes"]
    )
    groups = [
        _group_plan(index, eligible[start:start + group_size])
        for index, start in enumerate(range(0, len(eligible), group_size))
    ]
    for group in groups:
        stats["groups"].append({
            key: value for key, value in group.items() if key != "items"
        })
    stats["temporary_coexistence_bytes"] = max(
        (group["temporary_coexistence_bytes"] for group in groups),
        default=0,
    )
    stats["temporary_original_replacement_bytes"] = stats[
        "temporary_coexistence_bytes"
    ]
    if not eligible:
        stats["admission_status"] = "empty"
        return stats

    # A zero budget is an explicit no-op.  In particular, do not even build a
    # replacement array: callers use this to disable preparation safely.
    if stats["max_additional_bytes"] == 0:
        counters["groups_considered"] += len(groups)
        for group in groups:
            _record_group_rejection(
                stats, counters, group, "persistent_budget",
                "zero persistent-growth budget",
            )
        stats["admission_status"] = "rejected"
        return stats
    if stats["projected_persistent_growth_bytes"] > stats["max_additional_bytes"]:
        counters["groups_considered"] += len(groups)
        for group in groups:
            _record_group_rejection(
                stats, counters, group, "persistent_budget",
                "projected persistent growth exceeds the admission budget",
            )
        stats["admission_status"] = "rejected"
        return stats

    # Do all admission before the first cast/eval/replacement.  This leaves a
    # rejected candidate or group with its original module objects intact.
    admitted = []
    for group in groups:
        counters["groups_considered"] += 1
        stats.update({
            "current_group_index": group["group_index"],
            "current_group_names": list(group["names"]),
            "current_group_source_bytes": group["source_bytes"],
            "current_group_replacement_bytes": group["replacement_bytes"],
            "current_group_persistent_growth_bytes": group[
                "persistent_growth_bytes"
            ],
            "current_group_temporary_coexistence_bytes": group[
                "temporary_coexistence_bytes"
            ],
            # The original group is already resident, so this is the actual
            # incremental allocation at the replacement peak.
            "current_group_temporary_allocation_bytes": group[
                "replacement_bytes"
            ],
        })
        try:
            _resource_check(
                started,
                stats,
                max_seconds=float(stats["deadline_seconds"]),
                max_additional_bytes=int(stats["max_additional_bytes"]),
                clock=clock,
            )
        except QMMResourceError as exc:
            for remaining in groups[group["group_index"]:]:
                _record_group_rejection(
                    stats, counters, remaining, "deadline", str(exc)
                )
            stats["admission_status"] = "rejected"
            return stats
        if (
            stats["max_temporary_bytes"] is not None
            and group["temporary_coexistence_bytes"]
            > stats["max_temporary_bytes"]
        ):
            _record_group_rejection(
                stats, counters, group, "temporary_budget",
                "original and replacement metadata exceed the temporary budget",
            )
            continue
        if pressure_check is not None:
            counters["admission_calls"] += 1
            stats.pop("pressure_rejection_reason", None)
            try:
                rejected = bool(pressure_check(stats))
            except Exception as exc:
                _record_group_rejection(
                    stats, counters, group, "pressure_callback_error", str(exc)
                )
                raise QMMResourceError(
                    f"metadata pressure callback failed for group {group['group_index']}"
                ) from exc
            if rejected:
                _record_group_rejection(
                    stats, counters, group,
                    stats.get("pressure_rejection_reason", "pressure_callback"),
                    "pressure admission callback rejected the group",
                )
                # Pressure is a live admission failure, not a reason to skip
                # one group and keep allocating later groups.  Preserve every
                # untouched group as rejected and leave their modules alone.
                for remaining in groups[group["group_index"] + 1:]:
                    _record_group_rejection(
                        stats, counters, remaining, "pressure_callback",
                        "pressure admission stopped subsequent groups",
                    )
                break
        admitted.append(group)
        counters["groups_admitted"] += 1

    if not admitted:
        stats["admission_status"] = "rejected"
        return stats

    module_names = {id(item[1]): item[0] for item in eligible}
    prepared_records = []
    try:
        for group in admitted:
            pending = []
            stats.update({
                "current_group_index": group["group_index"],
                "current_group_names": list(group["names"]),
            })
            for name, module, _elements, _source_bytes, _replacement_bytes in group[
                "items"
            ]:
                scales = module.scales.astype(mx.float32)
                biases = module.biases.astype(mx.float32)
                pending.append((name, module, scales, biases))
            # Admission happened before these casts.  Only this bounded group
            # has both the old and replacement metadata live at once.
            mx.eval([
                value
                for _name, _module, scales, biases in pending
                for value in (scales, biases)
            ])
            _resource_check(
                started,
                stats,
                max_seconds=float(stats["deadline_seconds"]),
                max_additional_bytes=int(stats["max_additional_bytes"]),
                clock=clock,
            )
            for name, module, scales, biases in pending:
                old_scales, old_biases = module.scales, module.biases
                module.scales = scales
                module.biases = biases
                module._bonsai_qmm_prepared = True
                prepared_records.append(
                    (module, old_scales, old_biases, scales, biases)
                )
                stats["prepared_modules"] += 1
                stats["prepared_arrays"] += 2
                stats["prepared_module_manifest"].append(name)
            stats["prepared_bytes"] += group["replacement_bytes"]
            stats["additional_bytes"] += group["persistent_growth_bytes"]
            pending.clear()
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

        _resource_check(
            started,
            stats,
            max_seconds=float(stats["deadline_seconds"]),
            max_additional_bytes=int(stats["max_additional_bytes"]),
            clock=clock,
        )
    except BaseException as exc:
        for module, old_scales, old_biases, _scales, _biases in reversed(
            prepared_records
        ):
            module.scales = old_scales
            module.biases = old_biases
            if hasattr(module, "_bonsai_qmm_prepared"):
                delattr(module, "_bonsai_qmm_prepared")
        failure = {
            "type": "preparation",
            "group_index": stats.get("current_group_index"),
            "message": str(exc),
        }
        stats["failure"] = failure
        counters["preparation_failures"].append(failure)
        raise

    stats["admission_status"] = (
        "partial" if stats["admission_failures"] else "accepted"
    )

    def dispatch(module, x):
        if not getattr(module, "_bonsai_qmm_prepared", False):
            return original(module, x)
        counters["prepared_calls"] += 1
        phase = counters.get("current_phase", "unknown")
        phase = phase if phase in counters["phase_calls"] else "unknown"
        counters["phase_calls"][phase] += 1
        counters["module_calls"][module_names.get(id(module), "<unknown>")] = (
            counters["module_calls"].get(module_names.get(id(module), "<unknown>"), 0) + 1
        )
        counters["phase_prepared_calls"][phase] += 1
        if str(x.dtype).endswith("float32"):
            counters["fp32_calls"] += 1
            counters["phase_fp32_calls"][phase] += 1
            scales, biases = module.scales, module.biases
        elif str(x.dtype).endswith("float16"):
            counters["lower_precision_calls"] += 1
            counters["phase_lower_precision_calls"][phase] += 1
            scales = module.scales.astype(mx.float16)
            biases = module.biases.astype(mx.float16)
        else:
            counters["unsupported_calls"] += 1
            counters["phase_unsupported_calls"][phase] += 1
            scales = module.scales.astype(mx.float16)
            biases = module.biases.astype(mx.float16)
        return _stock_projection(runtime, module, x, scales, biases)

    packed_type.__call__ = dispatch
    return stats
