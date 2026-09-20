"""Packed-Q2/G128 prefill probe and opt-in runtime hook.

The candidate reads Bonsai's existing ``uint32`` Q2 tensors, stages one output
tile in temporary threadgroup storage, and asks Metal Performance Primitives
to multiply that staged ``uint2b_format`` tile.  If the installed MPP compiler
cannot consume that representation, the runtime falls back to stock QMM and
the benchmark records the reason.  No dequantized or persistent-Q4 route is
added.

The affine regrouping is intentionally not called exact.  MLX's reference
QMM rounds each dequantized weight to the activation dtype before its tiled
accumulation; this probe accumulates the mathematically regrouped expression
and converts the result at the end.  ``q2_rounding_diagnostics`` exposes that
distinction for tests and research notes.  The runtime switch therefore stays
opt-in until a separate exactness gate passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


Q2_BITS = 2
Q2_GROUP_SIZE = 128
Q2_TILE_ROWS = 32
Q2_TILE_COLUMNS = 128
Q2_WORDS_PER_GROUP = Q2_GROUP_SIZE // 16
Q2_PREFILL_BATCH_SIZES = (32, 128, 256, 512)
Q2_PROJECTION_GEOMETRIES = {
    "gate_proj": (5120, 17408),
    "up_proj": (5120, 17408),
    "down_proj": (17408, 5120),
}


class Q2MPPInfeasible(RuntimeError):
    """The device/compiler cannot run the temporary packed-Q2 MPP route."""


@dataclass(frozen=True)
class Q2ProbeRecord:
    """Small serializable result for a single feasibility attempt."""

    feasible: bool
    exact: bool
    rows: int
    input_size: int
    output_size: int
    reason: str


def _new_q2_counters() -> dict[str, object]:
    return {
        "q2_candidate_calls": 0,
        "q2_eligible_calls": 0,
        "q2_policy_excluded_calls": 0,
        "q2_policy_exclusion_reasons": {},
        "q2_phase_policy_excluded": {"unknown": 0, "prefill": 0, "decode": 0},
        "q2_attempts": 0,
        "q2_selected": 0,
        "q2_fallbacks": 0,
        "q2_multi_row": 0,
        "q2_single_row": 0,
        "q2_phase_calls": {"unknown": 0, "prefill": 0, "decode": 0},
        "q2_phase_selected": {"unknown": 0, "prefill": 0, "decode": 0},
        "q2_phase_fallbacks": {"unknown": 0, "prefill": 0, "decode": 0},
        "q2_eligible_geometries": {},
        "q2_current_phase": "unknown",
        "fallback_reasons": {},
    }


def new_q2_counters() -> dict[str, object]:
    """Return counters scoped to one backend/generation run."""
    return _new_q2_counters()


def _record_q2_call(counters, rows: int) -> None:
    if counters is None:
        return
    counters["q2_candidate_calls"] += 1
    counters["q2_eligible_calls"] += 1
    counters["q2_multi_row" if rows > 1 else "q2_single_row"] += 1


def _record_q2_policy_exclusion(counters, reason: str = "policy") -> None:
    if counters is not None:
        counters["q2_policy_excluded_calls"] += 1
        reasons = counters["q2_policy_exclusion_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        phase = counters.get("q2_current_phase", "unknown")
        phase = phase if phase in counters["q2_phase_policy_excluded"] else "unknown"
        counters["q2_phase_policy_excluded"][phase] += 1


def _record_q2_attempt(counters, geometry) -> None:
    if counters is None:
        return
    counters["q2_attempts"] += 1
    phase = counters.get("q2_current_phase", "unknown")
    phase = phase if phase in counters["q2_phase_calls"] else "unknown"
    counters["q2_phase_calls"][phase] += 1
    geometries = counters["q2_eligible_geometries"]
    key = f"{geometry[0]}x{geometry[1]}"
    geometries[key] = geometries.get(key, 0) + 1


def _record_q2_selection(counters) -> None:
    if counters is not None:
        counters["q2_selected"] += 1
        phase = counters.get("q2_current_phase", "unknown")
        phase = phase if phase in counters["q2_phase_selected"] else "unknown"
        counters["q2_phase_selected"][phase] += 1


def _record_q2_fallback(counters, reason: str) -> None:
    if counters is None:
        return
    counters["q2_fallbacks"] += 1
    phase = counters.get("q2_current_phase", "unknown")
    phase = phase if phase in counters["q2_phase_fallbacks"] else "unknown"
    counters["q2_phase_fallbacks"][phase] += 1
    reasons = counters["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def unpack_q2_codes(weight):
    """Extract the sixteen low-to-high Q2 codes in every packed word.

    ``weight`` has Bonsai's ``[output, input / 16]`` uint32 layout.  The
    returned array is only a reference/probe aid; the Metal route never
    materializes it.
    """
    import mlx.core as mx

    if weight.ndim != 2 or weight.dtype != mx.uint32:
        raise ValueError("Q2 weights must be a rank-2 uint32 array")
    shifts = mx.arange(16, dtype=mx.uint32) * 2
    return mx.reshape(
        mx.bitwise_and(mx.right_shift(weight[..., None], shifts), 3),
        (weight.shape[0], weight.shape[1] * 16),
    )


def q2_affine_reference(x, weight, scales, biases):
    """Reference the affine Q2 formula without changing the packed weights.

    This is intentionally a transparent diagnostic implementation.  It keeps
    the group correction explicit so tests can compare each intermediate with
    ``mx.quantized_matmul`` while the MPP probe remains independently packed.
    """
    import mlx.core as mx

    if x.ndim != 2:
        raise ValueError("Q2 reference input must be rank 2")
    codes = unpack_q2_codes(weight)
    rows, input_size = codes.shape
    if x.shape[-1] != input_size or input_size % Q2_GROUP_SIZE:
        raise ValueError("Q2 input width does not match G128 weights")
    if tuple(scales.shape) != (rows, input_size // Q2_GROUP_SIZE):
        raise ValueError("Q2 scale shape does not match the packed weights")
    if tuple(biases.shape) != tuple(scales.shape):
        raise ValueError("Q2 bias shape does not match the scales")

    result = mx.zeros((x.shape[0], rows), dtype=mx.float32)
    x_float = x.astype(mx.float32)
    code_float = codes.astype(mx.float32)
    for group in range(input_size // Q2_GROUP_SIZE):
        start = group * Q2_GROUP_SIZE
        end = start + Q2_GROUP_SIZE
        dots = mx.matmul(x_float[:, start:end], code_float[:, start:end].T)
        group_sum = mx.sum(x_float[:, start:end], axis=-1, keepdims=True)
        result = result + (
            dots * scales[:, group].astype(mx.float32)[None, :]
            + group_sum * biases[:, group].astype(mx.float32)[None, :]
        )
    return result.astype(x.dtype)


def q2_dequantized_reference(x, weight, scales, biases):
    """Reference MLX's dtype ordering: promote metadata before dequantizing.

    This is diagnostic only.  It explains the first numerical difference
    between the regrouped affine candidate and ``mx.quantized_matmul``; it is
    not used by the runtime hook and does not expand checkpoint weights.
    """
    import mlx.core as mx

    if x.ndim != 2:
        raise ValueError("Q2 reference input must be rank 2")
    promoted_scales = scales.astype(x.dtype)
    promoted_biases = biases.astype(x.dtype)
    dense = mx.dequantize(
        weight, promoted_scales, promoted_biases,
        group_size=Q2_GROUP_SIZE, bits=Q2_BITS,
    ).astype(x.dtype)
    return mx.matmul(x, dense.T)


def q2_rounding_diagnostics(x, weight, scales, biases) -> dict[str, float | int]:
    """Compare regrouping, per-weight rounding, and stock QMM intermediates."""
    import mlx.core as mx
    import numpy as np

    affine = q2_affine_reference(x, weight, scales, biases)
    dequantized = q2_dequantized_reference(x, weight, scales, biases)
    stock = mx.quantized_matmul(
        x, weight, scales, biases, transpose=True,
        group_size=Q2_GROUP_SIZE, bits=Q2_BITS,
    )
    mx.eval(affine, dequantized, stock)

    def max_abs(left, right) -> float:
        return float(mx.max(mx.abs(left - right)).item())

    return {
        "affine_vs_stock_max_abs": max_abs(affine, stock),
        "dequantized_vs_stock_max_abs": max_abs(dequantized, stock),
        "affine_vs_dequantized_max_abs": max_abs(affine, dequantized),
        "affine_differing_values": int(np.sum(np.asarray(affine) != np.asarray(stock))),
        "dequantized_differing_values": int(
            np.sum(np.asarray(dequantized) != np.asarray(stock))
        ),
    }


_Q2_MPP_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""


_Q2_MPP_BODY = """
{
  constexpr ushort TileM = 32;
  constexpr ushort TileN = 128;
  constexpr ushort GroupK = 128;
  constexpr ushort WordsPerGroup = 8;

  uint row_tile = threadgroup_position_in_grid.x;
  uint output_tile = threadgroup_position_in_grid.y;
  uint lane = thread_index_in_simdgroup;
  uint simd_group = simdgroup_index_in_threadgroup;
  uint input_size = uint(in_vec_size[0]);
  uint output_size = uint(out_vec_size[0]);
  uint input_groups = input_size / GroupK;
  uint output_origin = output_tile * TileN;
  uint row_origin = row_tile * TileM;

  // Eight uint32 words per output column hold one G128 group.  This is the
  // only Q2 -> MPP staging allocation; weights remain packed in device memory.
  threadgroup uint32_t staged[TileN * WordsPerGroup];
  threadgroup float input_sums[TileM];
  uint thread_index = simd_group * 32 + lane;

  auto a = tensor(const_cast<device half *>(
                      x + ulong(row_origin) * input_size),
                  dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, GroupK, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.slice<128, 32>(0, 0);
  auto first_b = tensor<threadgroup uint2b_format,
                        dextents<int, 2>, tensor_inline>(
      reinterpret_cast<threadgroup uchar *>(staged),
      dextents<int, 2>{GroupK, TileN}, array<int, 2>{1, GroupK});
  auto b0 = first_b.slice<128, 128>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
  for (ushort index = 0; index < accumulated.get_capacity(); ++index) {
    accumulated[index] = 0.0f;
  }

  for (uint group = 0; group < input_groups; ++group) {
    for (uint index = thread_index; index < TileN * WordsPerGroup;
         index += 256) {
      uint column = index / WordsPerGroup;
      uint word = index % WordsPerGroup;
      uint output = output_origin + column;
      staged[index] = output < output_size
          ? w[ulong(output) * (input_size / 16) +
              ulong(group) * WordsPerGroup + word]
          : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = simd_group; row < TileM; row += 8) {
      float partial_sum = 0.0f;
      uint origin = (row_origin + row) * input_size + group * GroupK;
      for (uint k = lane; k < GroupK; k += 32)
        partial_sum += float(x[origin + k]);
      partial_sum = simd_sum(partial_sum);
      if (lane == 0)
        input_sums[row] = partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto a_slice = a.slice<128, 32>(group * GroupK, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b0), float>();
    operation.run(a_slice, b0, partial);
    for (ushort index = 0; index < accumulated.get_capacity(); ++index) {
      if (!accumulated.is_valid_element(index))
        continue;
      auto coordinates = accumulated.get_multidimensional_index(index);
      uint column = coordinates[0];
      uint row = coordinates[1];
      uint output = output_origin + column;
      ulong parameter = ulong(output) * input_groups + group;
      accumulated[index] +=
          partial[index] * float(scales[parameter]) +
          input_sums[row] * float(biases[parameter]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (ushort index = 0; index < accumulated.get_capacity(); ++index) {
    if (!accumulated.is_valid_element(index))
      continue;
    auto coordinates = accumulated.get_multidimensional_index(index);
    uint column = coordinates[0];
    uint row = coordinates[1];
    if (row < TileM && output_origin + column < output_size)
      y[ulong(row_origin + row) * output_size + output_origin + column] =
          half(accumulated[index]);
  }
}
"""


_Q2_VALIDATED = set()
_Q2_INFEASIBLE_DTYPES = set()


@lru_cache(maxsize=None)
def _get_mpp_kernel_for_dtype(dtype_name: str):
    import mlx.core as mx

    try:
        if dtype_name not in ("float16", "float32"):
            raise Q2MPPInfeasible(
                f"unsupported MPP activation dtype: {dtype_name}"
            )
        source = _Q2_MPP_BODY.replace(
            "half", "float" if dtype_name == "float32" else "half"
        )
        return mx.fast.metal_kernel(
            name=f"bonsai2_q2_g128_prefill_mpp_{dtype_name}",
            input_names=[
                "w", "scales", "biases", "x", "in_vec_size", "out_vec_size"
            ],
            output_names=["y"],
            source=source,
            header=_Q2_MPP_HEADER,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    except Exception as exc:  # Metal compiler errors are probe results.
        raise Q2MPPInfeasible(f"MPP Q2 tile compilation failed: {exc}") from exc


def _get_mpp_kernel():
    """Keep the original half-precision test seam."""
    return _get_mpp_kernel_for_dtype("float16")


def _ensure_mpp_kernel(dtype) -> object:
    dtype_name = "float16" if str(dtype).endswith("float16") else "float32"
    if dtype_name in _Q2_INFEASIBLE_DTYPES:
        raise Q2MPPInfeasible(
            f"MPP Q2 tile is unavailable for {dtype_name} activations"
        )
    try:
        return (
            _get_mpp_kernel()
            if dtype_name == "float16"
            else _get_mpp_kernel_for_dtype(dtype_name)
        )
    except Q2MPPInfeasible:
        _Q2_INFEASIBLE_DTYPES.add(dtype_name)
        raise


def _validate_mpp_shapes(x, weight, scales, biases) -> tuple[int, int, int]:
    import mlx.core as mx

    if x.ndim != 2 or x.dtype not in (mx.float16, mx.float32):
        raise Q2MPPInfeasible(
            "the Q2 MPP probe requires rank-2 fp16 or fp32 input"
        )
    if weight.ndim != 2 or weight.dtype != mx.uint32:
        raise Q2MPPInfeasible("Q2 weights must be rank-2 uint32")
    rows, packed_width = map(int, weight.shape)
    input_size = packed_width * 16
    if input_size != int(x.shape[-1]) or input_size % Q2_GROUP_SIZE:
        raise Q2MPPInfeasible("input width is not a packed Q2/G128 geometry")
    if rows % Q2_TILE_COLUMNS:
        raise Q2MPPInfeasible(
            "output geometry must use 128-column tiles; input uses 32-row tiles"
        )
    expected = (rows, input_size // Q2_GROUP_SIZE)
    if tuple(scales.shape) != expected or tuple(biases.shape) != expected:
        raise Q2MPPInfeasible("Q2 affine metadata has the wrong shape")
    return int(x.shape[0]), input_size, rows


def q2_mpp_matmul(x, weight, scales, biases):
    """Run the temporary-staged Q2 MPP candidate or raise infeasibility."""
    import mlx.core as mx

    input_rows, input_size, output_size = _validate_mpp_shapes(
        x, weight, scales, biases
    )
    tile_rows = (input_rows + Q2_TILE_ROWS - 1) // Q2_TILE_ROWS
    padded_rows = tile_rows * Q2_TILE_ROWS
    padded = x
    if padded_rows != input_rows:
        # One bounded partial tile.  This is deliberately temporary and is
        # included in the complete prefill timing by the runtime hook.
        padded = mx.pad(x, ((0, padded_rows - input_rows), (0, 0)))
    kernel = _ensure_mpp_kernel(x.dtype)
    output = kernel(
        inputs=[
            weight,
            scales,
            biases,
            padded,
            mx.array([input_size], dtype=mx.int32),
            mx.array([output_size], dtype=mx.int32),
        ],
        # MLX's grid is expressed in threads.  The X dimension therefore
        # carries the 256-thread group width; Y is already a tile count.
        grid=(tile_rows * 256, output_size // Q2_TILE_COLUMNS, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[padded.shape[:-1] + (output_size,)],
        output_dtypes=[padded.dtype],
    )
    output = output[0] if isinstance(output, (tuple, list)) else output
    return output[:input_rows] if padded_rows != input_rows else output


_Q2_PROJECTION_SHAPES = {
    (17408, 5120),  # gate/up: [out, in]
    (5120, 17408),  # down: [out, in]
}


def install_q2_prefill_hook(enabled: bool, counters=None) -> None:
    """Install the opt-in hook on the checkpoint-owned ``runtime.Packed``.

    The hook transforms activations exactly as the stock ``Packed`` call does,
    then routes only the three dense MLP geometries with more than one prompt
    row through the temporary Q2 MPP candidate.  M=1 calls always remain on
    stock QMM.  Reinstalling the hook is safe for checkpoint-free tests and
    for constructing control/candidate backends in one interpreter.
    """
    import importlib

    try:
        runtime = importlib.import_module("runtime")
    except ImportError:
        if enabled:
            raise
        return
    packed = runtime.Packed
    stock = getattr(packed, "_bonsai_stock_call", None)
    if stock is None:
        stock = packed.__call__
        packed._bonsai_stock_call = stock
    if not enabled:
        packed.__call__ = stock
        return

    def dispatch(self, x):
        import mlx.core as mx

        weight = getattr(self, "weight", None)
        if (
            bool(getattr(self, "embedding", False))
            or not bool(getattr(self, "block", 0))
            or getattr(x, "ndim", 0) < 2
            or weight is None
            or int(x.shape[-1]) != int(weight.shape[1]) * 16
        ):
            return stock(self, x)
        geometry = (int(weight.shape[0]), int(weight.shape[1]) * 16)
        if geometry not in _Q2_PROJECTION_SHAPES:
            return stock(self, x)
        if x.dtype not in (mx.float16, mx.float32):
            _record_q2_policy_exclusion(counters, "unsupported_activation_dtype")
            return stock(self, x)

        rows = 1
        for dimension in x.shape[:-1]:
            rows *= int(dimension)
        if rows <= 1:
            # The candidate is prefill-only; stock single-token decode is a
            # hard boundary rather than a fallback that can be promoted.
            _record_q2_policy_exclusion(counters, "single_row_decode")
            return stock(self, x)

        _record_q2_call(counters, rows)
        _record_q2_attempt(counters, geometry)
        try:
            if x.dtype not in (mx.float16, mx.float32):
                raise Q2MPPInfeasible(
                    f"unsupported activation dtype at Packed call: {x.dtype}"
                )
            # Compile/validate before the Hadamard transform.  A rejected
            # compiler route must not make the safe stock fallback transform
            # the same prompt rows twice.
            _ensure_mpp_kernel(x.dtype)
            # Keep the transform and all packed checkpoint arrays exactly as
            # the stock runtime sees them.  Only the projection arithmetic is
            # changed for this opt-in diagnostic.
            transformed = runtime.fwht(x, self.block, self.signs)
            flat = transformed.reshape(-1, transformed.shape[-1])
            output = q2_mpp_matmul(
                flat, self.weight, self.scales, self.biases
            )
            key = (
                str(transformed.dtype), int(transformed.shape[-1]),
                int(self.weight.shape[0]),
            )
            if key not in _Q2_VALIDATED:
                # Metal compilation is deferred.  Validate the first
                # signature inside the selection policy so a compiler failure
                # cannot escape later from a lazy evaluation.
                mx.eval(output)
                _Q2_VALIDATED.add(key)
            _record_q2_selection(counters)
            return output.reshape(*x.shape[:-1], output.shape[-1])
        except Exception as exc:
            key = str(exc).lower()
            reason = (
                "compiler_rejection"
                if "compile" in key or "metal" in key
                else "mpp_infeasible"
                if isinstance(exc, Q2MPPInfeasible)
                else "runtime_rejection"
            )
            if reason == "compiler_rejection":
                dtype_name = "float16" if str(x.dtype).endswith("float16") else "float32"
                # ``mx.fast.metal_kernel`` defers compilation until eval.  A
                # rejected signature is stable for this process, so do not
                # pay the same deferred compile cost at every Packed call.
                _Q2_INFEASIBLE_DTYPES.add(dtype_name)
            _record_q2_fallback(counters, reason)
            return stock(self, x)

    packed.__call__ = dispatch


def probe_q2_mpp(x, weight, scales, biases) -> Q2ProbeRecord:
    """Attempt one Q2 route and retain an explicit infeasibility reason."""
    rows = int(x.shape[0]) if getattr(x, "ndim", 0) == 2 else -1
    input_size = int(x.shape[-1]) if getattr(x, "ndim", 0) else -1
    output_size = int(weight.shape[0]) if getattr(weight, "ndim", 0) == 2 else -1
    try:
        result = q2_mpp_matmul(x, weight, scales, biases)
        # A probe record is only useful after the device has actually run it.
        import mlx.core as mx

        mx.eval(result)
    except (Q2MPPInfeasible, RuntimeError, ValueError, TypeError) as exc:
        return Q2ProbeRecord(False, False, rows, input_size, output_size, str(exc))

    # Exactness is a promotion gate, not an assumption.  The comparison uses
    # the same packed weights and includes the temporary-staging launch cost;
    # a numerically close result still remains diagnostic-only.
    import numpy as np

    import mlx.core as mx

    reference = mx.quantized_matmul(
        x, weight, scales, biases, transpose=True,
        group_size=Q2_GROUP_SIZE, bits=Q2_BITS,
    )
    mx.eval(result, reference)
    exact = bool(np.array_equal(np.asarray(result), np.asarray(reference)))
    if not exact:
        return Q2ProbeRecord(
            False, False, rows, input_size, output_size,
            "MPP tile executed but differs from mx.quantized_matmul; diagnostic-only",
        )
    return Q2ProbeRecord(True, True, rows, input_size, output_size, "exact MPP tile")


def probe_q2_projection_geometries(
    projections, batch_sizes: tuple[int, ...] = Q2_PREFILL_BATCH_SIZES
):
    """Run the packed probe for gate/up/down tensors at prefill row counts.

    ``projections`` maps the names in ``Q2_PROJECTION_GEOMETRIES`` to
    ``(packed_weight, scales, biases)`` tuples taken directly from a loaded
    Bonsai checkpoint.  Inputs are deterministic fp16 diagnostics; the
    checkpoint tensors stay packed and are never repacked to Q4.  Each call
    includes the candidate's temporary staging launch, while callers can
    time the returned batch of launches when measuring promotion speed.
    """
    import mlx.core as mx

    missing = set(Q2_PROJECTION_GEOMETRIES) - set(projections)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing Q2 projection tensors: {names}")
    if any(int(rows) <= 0 for rows in batch_sizes):
        raise ValueError("Q2 probe batch sizes must be positive")

    results = {}
    for name, (weight, scales, biases) in projections.items():
        if name not in Q2_PROJECTION_GEOMETRIES:
            raise ValueError(f"unknown Q2 projection: {name}")
        input_size, _output_size = Q2_PROJECTION_GEOMETRIES[name]
        pattern = (mx.arange(input_size, dtype=mx.float32) % 17 - 8) / 17
        rows = []
        for batch in batch_sizes:
            activations = mx.tile(
                pattern[None, :], (int(batch), 1)
            ).astype(mx.float16)
            rows.append(
                probe_q2_mpp(activations, weight, scales, biases)
            )
        results[name] = tuple(rows)
    return results
