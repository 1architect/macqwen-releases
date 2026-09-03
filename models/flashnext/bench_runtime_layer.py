#!/usr/bin/env python3
"""Benchmark one real Flash-Next MoE layer against an optional executor.

This is the small experiment for the remaining runtime question. It reads the
packed gate, up, and down tensors from a safetensors checkpoint, runs the
current MLX ``gather_qmm`` implementation, and optionally runs a custom
executor on the same arrays and route. The custom executor is loaded lazily so
the benchmark still imports and reports the MLX arm when that experiment is
not installed.

The custom executor can be supplied as ``module:attribute`` with
``--custom-executor``. A builder may accept ``packs``, ``group_size``, ``bits``
and ``mode`` and return a callable object. A runner may accept ``x``,
``packs``/``weights``, ``indices``/``rhs_indices`` and the quantization
arguments. The adapter also exposes individual ``gate``, ``up`` and ``down``
packs for small experimental modules.

Each miss arm uses the same route width. Hot rows come from the first expert
pool and cold rows come from a disjoint pool. Hot rows are pinned when
possible, while every row still comes from the real checkpoint. Timing starts
after the rows are loaded and evaluated. The load time and physical-read
counter are reported separately, so compute and I/O do not get conflated.
If the bundled executor falls back to its reference path, the benchmark omits
its timing instead of labeling MLX work as custom work.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import inspect
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


PARTS = ("weight", "scales", "biases")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_MISSES = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Cell:
    """One deterministic route and its hot/cold split."""

    miss: float
    route: tuple[int, ...]
    hot: tuple[int, ...]
    cold: tuple[int, ...]


def parse_misses(value: str) -> tuple[float, ...]:
    """Parse comma-separated miss fractions and reject values outside [0, 1]."""
    result = []
    for item in value.split(","):
        miss = float(item.strip())
        if not 0.0 <= miss <= 1.0:
            raise ValueError(f"miss fraction must be between 0 and 1: {miss}")
        result.append(miss)
    if not result:
        raise ValueError("at least one miss fraction is required")
    return tuple(result)


def make_cell(miss: float, width: int, cold_base: int | None = None) -> Cell:
    """Build a route whose cold experts never overlap the pinned hot pool."""
    cold_count = int(round(width * miss))
    hot_count = width - cold_count
    hot = tuple(range(hot_count))
    base = width if cold_base is None else cold_base
    cold = tuple(range(base, base + cold_count))
    return Cell(miss, hot + cold, hot, cold)


def _load_target(spec: str):
    """Load ``module:attribute`` without importing custom code at module load."""
    module_name, separator, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    if separator:
        return getattr(module, attribute)
    for name in (
        "build_executor", "create_executor", "MetalMoEExecutor", "Executor",
        "execute", "run",
    ):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(
        f"{spec} has no executor builder or runner attribute"
    )


def load_custom_executor(spec: str | None):
    """Return a custom target, or ``(None, reason)`` when it is unavailable."""
    if not spec:
        return None, "not requested"
    try:
        return _load_target(spec), None
    except (ImportError, ModuleNotFoundError, AttributeError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _invoke(fn: Callable[..., Any], values: dict[str, Any]) -> Any:
    """Call an experimental function using only names in its signature.

    This keeps the harness compatible with early custom executors that use a
    smaller API. Positional-only parameters use the matching canonical name.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(
            values["x"], values["packs"], values["indices"],
            values["group_size"], values["bits"], values["mode"],
        )

    params = list(signature.parameters.values())
    accepts_any = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    kwargs = (
        values
        if accepts_any
        else {p.name: values[p.name] for p in params
              if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY)
              and p.name in values}
    )
    positional = []
    for param in params:
        if param.kind != inspect.Parameter.POSITIONAL_ONLY:
            continue
        if param.name not in values:
            if param.default is inspect.Parameter.empty:
                raise TypeError(f"custom executor needs unsupported argument {param.name!r}")
            continue
        positional.append(values[param.name])
    return fn(*positional, **kwargs)


def _runner_from_target(target, context: dict[str, Any]):
    """Construct a runner from a module, builder, class, or callable."""
    if inspect.ismodule(target):
        for name in ("build_executor", "create_executor", "MetalMoEExecutor", "Executor"):
            if hasattr(target, name):
                return _runner_from_target(getattr(target, name), context)
        for name in ("execute", "run"):
            if hasattr(target, name):
                return getattr(target, name)
        raise TypeError("custom module has no executor entry point")
    if inspect.isclass(target):
        return _invoke(target, context)
    if hasattr(target, "build_executor"):
        return _invoke(target.build_executor, context)
    if hasattr(target, "create_executor"):
        return _invoke(target.create_executor, context)
    if hasattr(target, "execute") or hasattr(target, "run"):
        return target
    if callable(target):
        names = set(inspect.signature(target).parameters)
        runner_names = {"x", "inputs", "indices", "rhs_indices", "route"}
        if names & runner_names:
            return target
        return _invoke(target, context)
    raise TypeError(f"custom target is not callable: {target!r}")


def _run_custom(runner, context: dict[str, Any]):
    """Run an executor object or function with the canonical context."""
    # MetalMoEExecutor uses a compact flattened API and returns one output per
    # routed slot. Adapt that API without imposing its shape on other runners.
    if type(runner).__name__ == "MetalMoEExecutor":
        values = dict(context)
        x = values["x"]
        values["x"] = x.reshape(-1, x.shape[-1])
        values["routes"] = values["indices"].reshape(-1, values["top_k"])
        values["scores"] = values["scores"].reshape(-1, values["top_k"])
        values["projections"] = values["packs"]
        # This is the bundled executor's production call shape. Avoid the
        # generic signature adapter here because repeated inspect.signature
        # work would be benchmark overhead, not runtime work.
        result = runner.execute(
            values["x"], values["routes"], values["projections"],
            scores=values["scores"],
        )
        scores = values["scores"]
        if result.ndim == scores.ndim + 1:
            return (result * scores[..., None]).sum(axis=-2)
        return result
    if hasattr(runner, "execute"):
        return _invoke(runner.execute, context)
    if hasattr(runner, "run"):
        return _invoke(runner.run, context)
    return _invoke(runner, context)


def custom_uses_native_path(runner) -> bool:
    """Return whether a runner reports a real custom backend execution.

    Experimental runners without a path marker remain eligible. The bundled
    Metal executor marks its CPU or MLX fallback as ``reference``.
    """
    path = getattr(runner, "last_path", None)
    return path is None or path == "custom-metal"


def _normalise_output(output):
    """Accept an array, a one-item tuple, or a mapping with an output key."""
    if isinstance(output, dict):
        for key in ("output", "out", "y", "result"):
            if key in output:
                return output[key]
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise TypeError("custom executor must return one output array")
        return output[0]
    return output


def compare_outputs(reference, candidate, atol: float, rtol: float) -> dict[str, Any]:
    """Return exact and tolerance checks without requiring MLX at import time."""
    def as_numpy(value):
        try:
            return np.asarray(value)
        except ValueError as exc:
            if "bfloat16" not in str(exc):
                raise
            import mlx.core as mx

            return np.asarray(value.astype(mx.float32))

    ref = as_numpy(reference)
    got = as_numpy(candidate)
    # Combine-only prototypes often omit a singleton sequence dimension.
    # Permit that representation difference, but never reshape data.
    if ref.shape != got.shape and ref.size == got.size:
        ref = np.squeeze(ref)
        got = np.squeeze(got)
    if ref.shape != got.shape:
        return {
            "shape_equal": False, "exact": False, "within_tolerance": False,
            "shape": (ref.shape, got.shape), "max_abs": float("inf"),
            "max_rel": float("inf"),
        }
    delta = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return {
        "shape_equal": True,
        "exact": bool(np.array_equal(ref, got)),
        "within_tolerance": bool(np.allclose(ref, got, atol=atol, rtol=rtol)),
        "shape": ref.shape,
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


def comparison_stats(mlx_runs, custom_runs) -> tuple[float, float]:
    """Return custom speed gain and a two-standard-error resolution band."""
    if not mlx_runs or not custom_runs:
        return float("nan"), float("inf")
    base = statistics.median(mlx_runs)
    other = statistics.median(custom_runs)
    gain = (base - other) / base * 100.0 if base else 0.0
    if len(mlx_runs) < 2 or len(custom_runs) < 2 or not base:
        return gain, float("inf")
    spread = math.sqrt(
        statistics.stdev(mlx_runs) ** 2 / len(mlx_runs)
        + statistics.stdev(custom_runs) ** 2 / len(custom_runs)
    )
    return gain, 2.0 * spread / base * 100.0


def worst_validation(checks: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the worst arm so one passing final arm cannot hide a failure."""
    checks = tuple(checks)
    if not checks:
        return None
    return max(
        checks,
        key=lambda item: (
            not item["within_tolerance"], item["max_abs"], item["max_rel"]
        ),
    )


def _mlx_moe(mx, activation, x, packs, indices, group_size, bits, mode,
             scores=None):
    """Run the current three-call MLX MoE path on already resident packs."""
    from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort

    del _gather_sort, _scatter_unsort  # Keep this path visibly unsorted.
    expanded = mx.expand_dims(x, (-2, -3))
    common = dict(
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=False,
    )
    gate = mx.gather_qmm(expanded, *packs["gate_proj"], **common)
    up = mx.gather_qmm(expanded, *packs["up_proj"], **common)
    down_input = activation(up, gate)
    out = mx.gather_qmm(down_input, *packs["down_proj"], **common)
    out = out.squeeze(-2)
    if scores is not None:
        out = (out * mx.expand_dims(scores, -1)).sum(axis=-2)
    return out


def _load_packs(mx, store, prefix: str, route: Iterable[int], read_mode: str):
    """Read and evaluate all three real checkpoint projections for a route."""
    route = list(route)
    packs = {}
    for projection in PROJECTIONS:
        parts = []
        for part in PARTS:
            name = f"{prefix}.{projection}.{part}"
            parts.append(store.to_mx(name, store.rows_np(name, route, read_mode)))
        packs[projection] = tuple(parts)
    mx.eval([array for pack in packs.values() for array in pack])
    return packs


def _load_controlled_packs(mx, store, prefix: str, cell: Cell):
    """Load pinned rows from memory and cold rows through ``F_NOCACHE``.

    This makes the requested miss fraction a measured condition. A normal
    ``pread`` can hit pages left by an earlier arm and silently turn a cold
    cell into a mixed one.
    """
    packs = {}
    hot_count = len(cell.hot)
    command = getattr(fcntl, "F_NOCACHE", 48)
    for projection in PROJECTIONS:
        parts = []
        for part in PARTS:
            name = f"{prefix}.{projection}.{part}"
            ref = store.refs[name]
            out = store.empty_rows(name, len(cell.route))
            if cell.hot:
                store.rows_into(
                    name, cell.hot, out[:hot_count], "shared_mmap"
                )
            if cell.cold:
                path = os.path.join(store.dir, ref.shard)
                fd = os.open(path, os.O_RDONLY)
                try:
                    fcntl.fcntl(fd, command, 1)
                    for slot, expert in enumerate(cell.cold, hot_count):
                        offset = ref.start + expert * ref.row_bytes
                        read = os.preadv(
                            fd, [memoryview(out[slot]).cast("B")], offset
                        )
                        if read != ref.row_bytes:
                            raise OSError(f"short controlled read for {name}")
                finally:
                    os.close(fd)
            parts.append(store.to_mx(name, out))
        packs[projection] = tuple(parts)
    mx.eval([array for pack in packs.values() for array in pack])
    return packs


def expected_cold_bytes(store, prefix: str, cold_count: int) -> int:
    """Return bytes that the controlled loader must read from the device."""
    return cold_count * sum(
        store.refs[f"{prefix}.{projection}.{part}"].row_bytes
        for projection in PROJECTIONS
        for part in PARTS
    )


def _context(x, packs, indices, cell, args, scores=None, expert_outputs=None):
    """Build aliases used by current and experimental executor APIs."""
    return {
        "x": x, "inputs": x, "hidden_states": x,
        "packs": packs, "weights": packs, "expert_weights": packs,
        "projections": packs,
        "gate": packs["gate_proj"], "up": packs["up_proj"],
        "down": packs["down_proj"], "gate_proj": packs["gate_proj"],
        "up_proj": packs["up_proj"], "down_proj": packs["down_proj"],
        "indices": indices, "rhs_indices": indices, "expert_indices": indices,
        "route": cell.route, "expert_ids": cell.route,
        "miss_fraction": cell.miss, "miss": cell.miss,
        "scores": scores, "gating_scores": scores,
        # ``routes`` indexes the packed expert_outputs array. The actual
        # checkpoint IDs stay available as ``route`` and ``expert_ids``.
        "routes": np.arange(len(cell.route), dtype=np.int32).reshape(1, -1),
        "route_indices": np.arange(len(cell.route), dtype=np.int32).reshape(1, -1),
        # A combine-only executor, such as the first Metal prototype, can
        # consume the projected expert values instead of packed QMM weights.
        "expert_outputs": expert_outputs, "expert_values": expert_outputs,
        "group_size": args.group_size, "bits": args.bits, "mode": args.mode,
        "activation": args.activation,
        # Packed arrays contain only this route. The checkpoint IDs remain in
        # ``expert_ids``; custom kernels index the compact local bank.
        "expert_count": len(cell.route), "hidden_size": args.hidden_size,
        "top_k": args.width,
    }


def _read_counter():
    try:
        from models.flashnext.diskio import disk_bytes_read
        return disk_bytes_read()
    except (ImportError, OSError):
        return None


def _pin_hot(store, prefix: str, hot: Iterable[int]) -> int:
    total = 0
    for expert in hot:
        for projection in PROJECTIONS:
            for part in PARTS:
                total += store.pin_rows(
                    f"{prefix}.{projection}.{part}", [expert]
                )
    return total


def _time_runner(mx, runner: Callable[..., Any], xs, contexts) -> float:
    """Time distinct calls under one eval, matching bench_gather_qmm."""
    began = time.perf_counter()
    outputs = []
    for x in xs:
        context = dict(contexts)
        context["x"] = x
        context["inputs"] = x
        context["hidden_states"] = x
        outputs.append(_normalise_output(runner(context)))
    mx.eval(outputs)
    return (time.perf_counter() - began) * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--arms", type=int, default=3)
    parser.add_argument("--misses", default=",".join(map(str, DEFAULT_MISSES)))
    parser.add_argument(
        "--custom-executor", "--executor", "--custom", dest="custom",
        default=os.environ.get(
            "FLASHNEXT_CUSTOM_EXECUTOR", "models.flashnext.metal_runtime"
        ),
        help="module[:attribute] for the custom executor; the default module "
             "is optional and may be absent",
    )
    parser.add_argument("--require-custom", action="store_true")
    parser.add_argument(
        "--read-mode",
        choices=("controlled", "pread", "preadv", "shared_mmap"),
        default="controlled",
        help="controlled pins hot rows and uses F_NOCACHE for cold rows",
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-hot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.activation = None

    if args.arms < 3:
        parser.error("--arms must be at least 3 for interleaved measurements")
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.read_mode == "controlled" and not args.pin_hot:
        parser.error("controlled reads require --pin-hot")

    try:
        misses = parse_misses(args.misses)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import mlx.core as mx
        from mlx_vlm.models.switch_layers import SwiGLU
        from macqwen.checkpoints import resolve_flashnext
        from models.flashnext.store import SafeTensorStore
    except ImportError as exc:
        print(f"MLX benchmark unavailable: {exc}", file=sys.stderr)
        return 2

    model = args.model or str(resolve_flashnext())
    store = SafeTensorStore(model)
    prefix = f"language_model.model.layers.{args.layer}.mlp.switch_mlp"
    required = f"{prefix}.gate_proj.weight"
    if required not in store.refs:
        print(f"no switch_mlp at layer {args.layer} in {model}", file=sys.stderr)
        return 2
    shape = store.shape(required)
    experts = shape[0]
    if args.width < 1 or args.width * 2 > experts:
        print(f"width must be in [1, {experts // 2}] for disjoint miss pools",
              file=sys.stderr)
        return 2
    args.width = min(args.width, experts)
    args.activation = SwiGLU()

    custom_target, custom_error = load_custom_executor(args.custom)
    if custom_error:
        print(f"custom executor: unavailable ({custom_error})")
        if args.require_custom:
            return 2
    elif custom_target is not None:
        print(f"custom executor: {args.custom}")

    # Use one real checkpoint hidden vector as the activation seed. This gives
    # the test a real tensor even when it runs without a full model forward.
    hidden = store.shape(f"{prefix}.down_proj.weight")[1]
    # The custom executor sees the compact route bank, not all checkpoint
    # experts. ``_context`` passes this local count during construction.
    args.expert_count = args.width
    args.hidden_size = hidden
    x_name = f"language_model.model.layers.{args.layer}.input_layernorm.weight"
    if x_name in store.refs and store.shape(x_name)[0] == hidden:
        # LayerNorm weight is a one-dimensional checkpoint tensor. Use the
        # whole vector, because ``rows_np(name, [0])`` would load one scalar.
        x = store.to_mx(x_name, store.whole_np(x_name))
        x = x.reshape(1, 1, hidden)
    else:
        x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16) * 0.02
    mx.eval(x)
    xs = [x * (1.0 + i * 0.001) for i in range(args.reps)]
    mx.eval(xs)
    indices = mx.arange(args.width, dtype=mx.uint32).reshape(1, 1, args.width)
    mx.eval(indices)
    scores = mx.array(
        np.linspace(1.0, 2.0, args.width, dtype=np.float32),
    ).reshape(1, 1, args.width)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    mx.eval(scores)

    print(f"model        {model}")
    print(f"layer        {args.layer}, experts={experts}, width={args.width}")
    print(f"read         {args.read_mode}, arms={args.arms}, reps={args.reps}")
    print(f"tolerance    atol={args.atol:g}, rtol={args.rtol:g}")
    print("\nmiss  route              load_ms  read_MB  mlx_ms  custom_ms  gain  band  exact  tol  max_abs  max_rel")

    for miss in misses:
        # Use a distant expert pool for controlled cold reads. Low-numbered
        # rows are often already cached by setup and unit-test probes.
        cold_base = experts - args.width if args.read_mode == "controlled" else None
        cell = make_cell(miss, args.width, cold_base=cold_base)
        rows = cell.route
        mlx_runs = []
        custom_runs = []
        checks = []
        load_runs = []
        read_runs = []
        for arm in range(args.arms):
            store.unpin_all()
            pinned = 0
            if args.pin_hot and cell.hot:
                try:
                    pinned = _pin_hot(store, prefix, cell.hot)
                except (OSError, ValueError) as exc:
                    print(f"pin hot: unavailable ({exc})", file=sys.stderr)
            before = _read_counter()
            began = time.perf_counter()
            packs = (
                _load_controlled_packs(mx, store, prefix, cell)
                if args.read_mode == "controlled"
                else _load_packs(mx, store, prefix, rows, args.read_mode)
            )
            load_runs.append((time.perf_counter() - began) * 1000.0)
            after = _read_counter()
            if before is not None and after is not None:
                read_runs.append(max(0, after - before) / 1e6)

            # The raw expert values are also useful to a combine-only custom
            # executor. They still come from all three real checkpoint packs.
            raw_ref = _mlx_moe(
                mx, args.activation, xs[0], packs, indices,
                args.group_size, args.bits, args.mode, scores=None,
            )
            mx.eval(raw_ref)
            ref = (raw_ref * mx.expand_dims(scores, -1)).sum(axis=-2)
            mx.eval(ref)
            context = _context(
                xs[0], packs, indices, cell, args, scores=scores,
                expert_outputs=np.asarray(raw_ref.astype(mx.float32)).squeeze(-3),
            )
            context["activation"] = args.activation
            custom = None
            if custom_target is not None:
                custom = _runner_from_target(custom_target, context)

            if custom is not None:
                candidate = _run_custom(custom, context)
                candidate = _normalise_output(candidate)
                mx.eval([ref, candidate])
                if custom_uses_native_path(custom):
                    checks.append(compare_outputs(
                        ref, candidate, args.atol, args.rtol
                    ))
                else:
                    reason = getattr(custom, "fallback_reason", "reference path")
                    print(
                        f"custom executor: skipped reference fallback ({reason})",
                        file=sys.stderr,
                    )
                    custom_target = None
                    custom = None

            # Reverse the order on alternate arms. This limits a fixed warmup
            # or thermal bias while keeping each pair on identical resident
            # packs, routes, inputs, and validation.
            order = ("mlx", "custom") if arm % 2 == 0 else ("custom", "mlx")
            for runner_name in order:
                if runner_name == "mlx":
                    mlx_runs.append(_time_runner(
                        mx,
                        lambda c: _mlx_moe(
                            mx, args.activation, c["x"], packs, indices,
                            args.group_size, args.bits, args.mode, scores=scores,
                        ),
                        xs, context,
                    ) / args.reps)
                elif custom is not None:
                    custom_runs.append(_time_runner(
                        mx, lambda c: _run_custom(custom, c), xs, context
                    ) / args.reps)
            if pinned:
                pass
        store.unpin_all()

        if args.read_mode == "controlled" and read_runs:
            expected = expected_cold_bytes(store, prefix, len(cell.cold)) / 1e6
            observed = statistics.median(read_runs)
            tolerance_mb = max(1.0, expected * 0.20)
            if abs(observed - expected) > tolerance_mb:
                print(
                    f"REFUSED: requested miss={miss} needs about "
                    f"{expected:.1f} MB, observed {observed:.1f} MB",
                    file=sys.stderr,
                )
                return 2

        # Report the worst validation arm. Checking only the last arm could
        # hide an earlier tolerance failure behind a passing final arm.
        check = worst_validation(checks)
        exact = "-" if check is None else ("yes" if check["exact"] else "no")
        tolerance = "-" if check is None else ("yes" if check["within_tolerance"] else "no")
        max_abs = "-" if check is None else f"{check['max_abs']:.3g}"
        max_rel = "-" if check is None else f"{check['max_rel']:.3g}"
        custom_ms = "-" if not custom_runs else f"{statistics.median(custom_runs):.2f}"
        gain, band = comparison_stats(mlx_runs, custom_runs)
        gain_text = "-" if math.isnan(gain) else f"{gain:+.1f}%"
        band_text = "-" if math.isinf(band) else f"{band:.1f}%"
        read_mb = "-" if not read_runs else f"{statistics.median(read_runs):.1f}"
        print(f"{miss:4.2f}  {','.join(map(str, rows)):18s} "
              f"{statistics.median(load_runs):7.1f}  {read_mb:>7s}  "
              f"{statistics.median(mlx_runs):6.2f}  {custom_ms:>9s}  "
              f"{gain_text:>6s}  {band_text:>5s}  "
              f"{exact:>5s}  {tolerance:>3s}  {max_abs:>7s}  {max_rel:>7s}")
        if any(not item["within_tolerance"] for item in checks):
            print(f"  REFUSED: custom output exceeds tolerance at miss={miss}",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
