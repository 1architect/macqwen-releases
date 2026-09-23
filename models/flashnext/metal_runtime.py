"""Bounded Level-1 Metal MoE experiment for FlashNext Q4/G32 experts.

This module is opt-in and does not change the production loader. It accepts
flattened decode inputs and runs the three routed projections. It loads the
Q4/G32 vector helpers from the active MLX wheel so the supported production
shapes use the same reduction order and return bit-identical output.

The experiment is deliberately bounded.  It supports at most eight tokens,
ten routed slots, 512 experts, and 16,384 projection outputs per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .boundary_profiler import BoundaryProfiler


MAX_TOKENS = 8
MAX_EXPERTS = 512
MAX_TOP_K = 10
MAX_WIDTH = 16_384
GROUP_SIZE = 32
SUPPORTED_GROUP_SIZES = (32, 64)
# Full-model digest gate passed after the MLX score-combine correction on
# 20260908. The 20260917 paired comparison measured +13.4% decode speed with
# identical digests, so FLASHNEXT_METAL_G64 defaults to on.
G64_RUNTIME_READY = True
BITS = 4


@dataclass(frozen=True)
class Q4G32Projection:
    """Packed expert rows consumed by one gate, up, or down projection."""

    weight: Any
    scales: Any
    biases: Any


@dataclass(frozen=True)
class MetalCapabilities:
    """Backend probe result exposed as attributes and mapping-like keys."""

    available: bool
    supports_custom_moe: bool
    backend: str
    reason: str

    @property
    def custom_metal(self) -> bool:
        return self.supports_custom_moe

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def probe_capabilities(backend: Any = None) -> MetalCapabilities:
    """Return capability flags without compiling or allocating a kernel.

    ``backend`` may be ``"metal"``, ``"reference"``, or an object exposing
    ``metal_kernel``.  The object form makes the experiment testable without
    requiring an Apple GPU.
    """
    if backend == "reference":
        return MetalCapabilities(False, False, "reference", "reference backend requested")
    if backend is not None and hasattr(backend, "available"):
        available = bool(getattr(backend, "available"))
        supports = bool(getattr(backend, "supports_custom_moe", False))
        reason = "injected backend capability"
        if not available:
            reason = "injected backend is unavailable"
        elif not supports:
            reason = "backend does not provide custom MoE"
        return MetalCapabilities(available and supports, supports, "injected", reason)
    if hasattr(backend, "metal_kernel"):
        return MetalCapabilities(True, True, "injected", "injected backend provides metal_kernel")
    if backend not in (None, "metal"):
        return MetalCapabilities(False, False, str(backend), "unknown backend")
    if sys.platform != "darwin":
        return MetalCapabilities(False, False, "metal", "custom Metal requires macOS")
    try:
        import mlx.core as mx

        if mx.default_device().type != mx.DeviceType.gpu:
            return MetalCapabilities(False, False, "metal", "MLX is not using a GPU device")
        return MetalCapabilities(True, True, "metal", "MLX GPU device detected")
    except Exception as exc:  # pragma: no cover - platform dependent
        return MetalCapabilities(False, False, "metal", f"MLX unavailable: {exc}")


def _as_projection(value: Any) -> Q4G32Projection:
    if isinstance(value, Q4G32Projection):
        return value
    if isinstance(value, Mapping):
        return Q4G32Projection(value["weight"], value["scales"], value["biases"])
    if isinstance(value, Sequence) and len(value) == 3:
        return Q4G32Projection(value[0], value[1], value[2])
    raise TypeError("projection must be (weight, scales, biases)")


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _slab_layout_constants(group_size: int, header_size: int = 4096) -> tuple[int, int]:
    from .slab_pack import get_slab_layout

    return header_size, get_slab_layout(group_size).record_stride


def _validate_projection(
    projection: Q4G32Projection,
    expert_count: int | None,
    input_width: int,
    output_width: int,
    group_size: int = GROUP_SIZE,
) -> None:
    weight_shape = _shape(projection.weight)
    scale_shape = _shape(projection.scales)
    bias_shape = _shape(projection.biases)
    count = weight_shape[0] if expert_count is None else expert_count
    expected_weight = (count, output_width, input_width // 8)
    expected_meta = (count, output_width, input_width // group_size)
    if weight_shape != expected_weight:
        raise ValueError(f"weight shape {weight_shape} != {expected_weight}")
    if scale_shape != expected_meta or bias_shape != expected_meta:
        raise ValueError(
            f"Q4/G{group_size} scales and biases have an invalid shape"
        )
    dtype = getattr(projection.weight, "dtype", None)
    if str(dtype) not in ("mlx.core.uint32", "uint32", "<class 'numpy.uint32'>"):
        # MLX prints ``mlx.core.uint32``; NumPy prints ``uint32``.
        if np.dtype(dtype) != np.dtype(np.uint32):
            raise TypeError("Q4 weights must use uint32 packed values")


def weighted_combine(expert_outputs: Any, routes: Any, scores: Any) -> Any:
    """Combine routed expert outputs using the router scores.

    The normal form is ``expert_outputs[..., slots, hidden]``.  If outputs
    instead contain every expert in ``[..., experts, hidden]``, ``routes``
    selects the slots first.  The route values do not affect the normal form,
    but this function validates their slot shape and range when possible.
    """
    output_shape = _shape(expert_outputs)
    route_shape = _shape(routes)
    score_shape = _shape(scores)
    if len(output_shape) < 2 or len(route_shape) == 0:
        raise ValueError("expert_outputs and routes need at least one dimension")
    if score_shape != route_shape:
        raise ValueError("routes and scores must have identical shapes")
    slots = route_shape[-1]
    if output_shape[:-1] == route_shape:
        selected = expert_outputs
    elif len(output_shape) == len(route_shape) + 1 and output_shape[-2] >= slots:
        try:
            route_values = np.asarray(routes)
            if route_values.size and (
                route_values.min() < 0 or route_values.max() >= output_shape[-2]
            ):
                raise ValueError("routes contain an expert outside expert_outputs")
        except TypeError:
            pass
        try:
            import mlx.core as mx

            selected = mx.take_along_axis(
                expert_outputs,
                mx.expand_dims(routes, -1),
                axis=-2,
            )
        except (ImportError, TypeError):
            selected = np.take_along_axis(
                np.asarray(expert_outputs), np.expand_dims(routes, -1), axis=-2
            )
    else:
        raise ValueError(
            f"expert_outputs shape {output_shape} does not match routes {route_shape}"
        )
    if _shape(selected)[:-1] != route_shape:
        raise ValueError("selected expert outputs do not match route slots")
    weights = scores[..., None]
    return (selected * weights).sum(axis=-2)


# Expert row pointers. A packed slab route sets bit 31 and addresses the
# expert-major record in the slab pack (or the expert pool, which shares its
# layout); anything else indexes the streamed
# weight arrays, or the expert-major stream pack when it is enabled.
_POINTERS = r"""
#if SLAB_PACK_ENABLED
bool in_slab = ((raw_expert & 0x80000000u) != 0);
uint expert = in_slab ? (raw_expert & 0x7FFFFFFFu) : raw_expert;
#if STREAM_PACK_ENABLED
const device char* record_base = in_slab
    ? ((const device char*)slab_pack) + SLAB_HEADER_SIZE + (ulong)expert * SLAB_RECORD_STRIDE
    : ((const device char*)stream_pack) + (ulong)expert * SLAB_RECORD_STRIDE;
decltype(weight) w_ptr = (decltype(weight))(record_base + W_OFFSET);
decltype(scales) s_ptr = (decltype(scales))(record_base + S_OFFSET);
decltype(biases) b_ptr = (decltype(biases))(record_base + B_OFFSET);
#else
// 64-bit: an expert pool can exceed 4 GiB of records.
ulong expert_offset = SLAB_HEADER_SIZE + (ulong)expert * SLAB_RECORD_STRIDE;
decltype(weight) w_ptr = in_slab
    ? (decltype(weight))(((const device char*)slab_pack) + expert_offset + W_OFFSET)
    : (weight + expert * OUT_WIDTH * (IN_WIDTH / 8));
decltype(scales) s_ptr = in_slab
    ? (decltype(scales))(((const device char*)slab_pack) + expert_offset + S_OFFSET)
    : (scales + expert * OUT_WIDTH * (IN_WIDTH / GROUP_SIZE));
decltype(biases) b_ptr = in_slab
    ? (decltype(biases))(((const device char*)slab_pack) + expert_offset + B_OFFSET)
    : (biases + expert * OUT_WIDTH * (IN_WIDTH / GROUP_SIZE));
#endif
#else
uint expert = raw_expert;
decltype(weight) w_ptr = weight + expert * OUT_WIDTH * (IN_WIDTH / 8);
decltype(scales) s_ptr = scales + expert * OUT_WIDTH * (IN_WIDTH / GROUP_SIZE);
decltype(biases) b_ptr = biases + expert * OUT_WIDTH * (IN_WIDTH / GROUP_SIZE);
#endif
"""


_PROJECTION_BODY = r"""
uint pair = threadgroup_position_in_grid.z;
uint token = pair / SLOTS;
uint slot = pair % SLOTS;
uint raw_expert = routes[pair];
const int in_size = IN_WIDTH;
const int out_size = OUT_WIDTH;
const device T* input = x +
    ((SLOT_INPUT != 0) ? pair * IN_WIDTH : token * IN_WIDTH);
device T* output = out + pair * OUT_WIDTH;
uint3 tid = uint3(0, threadgroup_position_in_grid.y, pair);
""" + _POINTERS + r"""
    QMV_MIXED_IMPL<T, GROUP_SIZE, 4>(
    w_ptr, s_ptr, b_ptr,
    input, output, in_size, out_size, tid,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


_FUSED_UP_SWIGLU_BODY = r"""
uint pair = threadgroup_position_in_grid.z;
uint token = pair / SLOTS;
uint slot = pair % SLOTS;
uint raw_expert = routes[pair];
const int in_size = IN_WIDTH;
const int out_size = OUT_WIDTH;
const device T* input = x + token * IN_WIDTH;
device T* output = out + pair * OUT_WIDTH;
const device T* gate_values = gate_out + pair * OUT_WIDTH;
uint3 tid = uint3(0, threadgroup_position_in_grid.y, pair);
""" + _POINTERS + r"""
    QMV_MIXED_IMPL<T, GROUP_SIZE, 4>(
    w_ptr, s_ptr, b_ptr,
    input, output, in_size, out_size, tid,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

// MLX's bfloat16 sigmoid helper uses this stable, rounded sequence.
uint out_base = threadgroup_position_in_grid.y * 8 +
    simdgroup_index_in_threadgroup * 4;
if (thread_index_in_simdgroup == 0) {
#pragma unroll
    for (uint row = 0; row < 4; ++row) {
        uint col = out_base + row;
        if (col < OUT_WIDTH) {
            uint index = pair * OUT_WIDTH + col;
            T gate_value = gate_values[col];
            T up_value = output[col];
            T magnitude = metal::abs(gate_value);
            T exponent = metal::exp(magnitude);
            T denominator = T(1) + exponent;
            T small_sigmoid = T(1) / denominator;
            T sigmoid = (gate_value < T(0)) ? small_sigmoid :
                T(1) - small_sigmoid;
            output[col] = (gate_value * sigmoid) * up_value;
        }
    }
}
"""


_FUSED_DOWN_COMBINE_BODY = r"""
#pragma clang fp contract(off)
uint3 group = threadgroup_position_in_grid;
uint simd_lid = thread_index_in_simdgroup;
uint simd_gid = simdgroup_index_in_threadgroup;
uint token = group.z;
uint out_base = group.x * 8 + simd_gid * 4;
if (token >= TOKENS || out_base >= OUT_WIDTH) return;
float combined[4] = {0.0f, 0.0f, 0.0f, 0.0f};

for (uint slot = 0; slot < SLOTS; ++slot) {
    uint raw_expert = routes[token * SLOTS + slot];
    float slot_score = float(scores[token * SLOTS + slot]);
    const int in_size = IN_WIDTH;
    const int out_size = OUT_WIDTH;
    uint3 tid = uint3(0, group.x, token);
""" + _POINTERS + r"""
    QMV_ACCUMULATE_IMPL<T, GROUP_SIZE, 4>(
        w_ptr, s_ptr, b_ptr,
        x + (token * SLOTS + slot) * IN_WIDTH,
        combined, slot_score, in_size, out_size, tid,
        simd_gid, simd_lid);
}

if (simd_lid == 0) {
#pragma unroll
    for (uint row = 0; row < 4; ++row) {
        uint col = out_base + row;
        if (col < OUT_WIDTH) {
            uint idx = token * OUT_WIDTH + col;
            T routed = static_cast<T>(combined[row]);
#if HAS_SHARED_PARTS
            T shared_component = static_cast<T>(
                float(shared[idx]) * float(shared_gate[token]));
            float final_value = float(routed) + float(shared_component);
            out[idx] = static_cast<T>(final_value);
#elif HAS_SHARED_Y
            float final_value = float(routed) + float(shared_y[idx]);
            out[idx] = static_cast<T>(final_value);
#else
            out[idx] = routed;
#endif
        }
    }
}
"""

_BODIES = {
    "projection": _PROJECTION_BODY,
    "up_swiglu": _FUSED_UP_SWIGLU_BODY,
    "down_combine": _FUSED_DOWN_COMBINE_BODY,
}
_QMV_LABELS = {"gate_proj": "gate_qmv", "up_proj": "up_qmv", "down_proj": "down_qmv"}
_KERNEL_TOKEN = re.compile(r"\b[A-Z][A-Z0-9_]*\b")


def _render(source: str, values: Mapping[str, str]) -> str:
    """Substitute whole-word placeholders; anything else stays as written."""
    return _KERNEL_TOKEN.sub(lambda match: values.get(match.group(0), match.group(0)), source)


@lru_cache(maxsize=1)
def _mlx_qmv_header() -> str:
    """Load the exact Q4 vector helpers shipped with the active MLX wheel."""
    import mlx.core as mx

    path = (
        Path(mx.__file__).resolve().parent
        / "include/mlx/backend/metal/kernels/quantized.h"
    )
    text = path.read_text()
    helper_end = text.index(
        "template <typename U, int values_per_thread, int bits>\ninline void\nqouter"
    )
    qmv_start = text.index(
        "template <typename T, int group_size, int bits>\n"
        "METAL_FUNC void qmv_fast_impl"
    )
    qmv_end = text.index("// Affine analog of fp_qmv_wide", qmv_start)
    header = text[:helper_end] + text[qmv_start:qmv_end]
    # The stock MLX helpers use one type for activations and affine metadata.
    # FlashNext stores float32 activations with bfloat16 scales and biases.
    # Keep the MLX reduction body, but use a separate metadata type so MLX
    # does not insert per-projection metadata conversion kernels.
    impl_start = text.index(
        "template <typename T, int group_size, int bits>\n"
        "METAL_FUNC void qmv_impl",
        qmv_start,
    )
    fast = text[qmv_start:impl_start]
    impl = text[impl_start:qmv_end]
    fast = fast.replace(
        "template <typename T, int group_size, int bits>",
        "template <typename T, int group_size, int bits, typename M>",
        1,
    ).replace("qmv_fast_impl", "qmv_fast_mixed_impl", 1)
    impl = impl.replace(
        "template <typename T, int group_size, int bits>",
        "template <typename T, int group_size, int bits, typename M>",
        1,
    ).replace("qmv_impl", "qmv_mixed_impl", 1)
    for old, new in (
        ("const device T* scales", "const device M* scales"),
        ("const device T* biases", "const device M* biases"),
        ("const device T* sl", "const device M* sl"),
        ("const device T* bl", "const device M* bl"),
    ):
        fast = fast.replace(old, new)
        impl = impl.replace(old, new)
    header += fast + impl

    def _make_accumulate(src: str, old_name: str, new_name: str) -> str:
        acc = src.replace(old_name, new_name, 1)
        acc = acc.replace("device T* y,", "thread float* combined, float score,", 1)
        acc = acc.replace("y += tid.x * out_vec_size + out_row;", "// y unused", 1)
        acc = acc.replace("y += tid.x * out_vec_size + used_out_row;", "// y unused", 1)
        acc = acc.replace(
            "y[row] = static_cast<T>(result[row]);",
            "combined[row] += float(static_cast<T>(result[row])) * score;",
        )
        return acc

    fast_acc = _make_accumulate(fast, "qmv_fast_mixed_impl", "qmv_fast_accumulate_impl")
    impl_acc = _make_accumulate(impl, "qmv_mixed_impl", "qmv_accumulate_impl")
    header += fast_acc + impl_acc
    # Custom-kernel helpers receive small local dimensions, not buffer
    # address-space constants. Pass them by value so older Metal compilers do
    # not reject references without an explicit address space.
    return header.replace("const constant int&", "const int")


class MetalMoEExecutor:
    """Bounded three-projection executor with an explicit reference path."""

    def __init__(
        self,
        expert_count: int,
        hidden_size: int,
        top_k: int,
        backend: Any = None,
        *,
        max_tokens: int = MAX_TOKENS,
        max_width: int = MAX_WIDTH,
        profile_boundaries: bool | None = None,
        profile_boundary: str | None = None,
        fused_up_swiglu: bool | None = None,
        group_size: int = GROUP_SIZE,
        header_size: int = 4096,
    ) -> None:
        if not 1 <= expert_count <= MAX_EXPERTS:
            raise ValueError(f"expert_count must be in 1..{MAX_EXPERTS}")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be in 1..{MAX_TOP_K}")
        if top_k > expert_count:
            raise ValueError("top_k cannot exceed expert_count")
        if not 1 <= max_tokens <= MAX_TOKENS:
            raise ValueError(f"max_tokens must be in 1..{MAX_TOKENS}")
        if not 1 <= max_width <= MAX_WIDTH:
            raise ValueError(f"max_width must be in 1..{MAX_WIDTH}")
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"group_size must be one of {SUPPORTED_GROUP_SIZES}"
            )
        self.expert_count = int(expert_count)
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)
        self.max_width = int(max_width)
        self.group_size = int(group_size)
        self.header_size = int(header_size)
        self.backend = backend
        self.capabilities = probe_capabilities(backend)
        self.last_path = "reference"
        self.fallback_reason = self.capabilities["reason"]
        self._kernels: dict[tuple[Any, ...], Any] = {}
        self._boundary_profiler = BoundaryProfiler(
            profile_boundaries, boundary=profile_boundary
        )
        if fused_up_swiglu is None:
            fused_up_swiglu = os.environ.get("FLASHNEXT_FUSED_UP_SWIGLU") == "1"
        self.fused_up_swiglu = bool(fused_up_swiglu)

    @property
    def boundary_profile(self) -> dict[str, Any]:
        """Return boundary events and totals for the latest executor calls."""
        return self._boundary_profiler.snapshot()

    def reset_boundary_profile(self) -> None:
        """Discard collected boundary events without changing the opt-in flag."""
        self._boundary_profiler.reset()

    @staticmethod
    def _complete_boundary(value: Any) -> None:
        # Injected backends can return NumPy arrays in checkpoint-free tests.
        # They have no deferred device graph to drain.
        if isinstance(value, np.ndarray):
            return
        import mlx.core as mx

        mx.eval(value)

    @property
    def available(self) -> bool:
        return bool(self.capabilities["available"])

    def _kernel(
        self,
        kind: str,
        x: Any,
        tokens: int,
        slots: int,
        input_width: int,
        output_width: int,
        *,
        proj_name: str,
        slot_input: bool = False,
        has_slab_pack: bool = False,
        has_stream_pack: bool = False,
        has_shared_y: bool = False,
        has_shared_parts: bool = False,
    ) -> Any:
        """Build, or reuse, one specialised Metal kernel."""
        key = (
            kind, self.group_size, str(getattr(x, "dtype", None)), tokens, slots,
            input_width, output_width, proj_name, slot_input, has_slab_pack,
            has_stream_pack, has_shared_y, has_shared_parts, self.header_size,
        )
        kernel = self._kernels.get(key)
        if kernel is not None:
            return kernel
        fast = input_width % 512 == 0
        values = {
            "TOKENS": str(tokens),
            "SLOTS": str(slots),
            "IN_WIDTH": str(input_width),
            "OUT_WIDTH": str(output_width),
            "GROUP_SIZE": str(self.group_size),
            "SLOT_INPUT": "1" if slot_input else "0",
            "SLAB_PACK_ENABLED": "1" if has_slab_pack else "0",
            "STREAM_PACK_ENABLED": "1" if has_stream_pack else "0",
            "HAS_SHARED_Y": "1" if has_shared_y else "0",
            "HAS_SHARED_PARTS": "1" if has_shared_parts else "0",
            "QMV_MIXED_IMPL": "qmv_fast_mixed_impl" if fast else "qmv_mixed_impl",
            "QMV_ACCUMULATE_IMPL": (
                "qmv_fast_accumulate_impl" if fast else "qmv_accumulate_impl"
            ),
        }
        if has_slab_pack:
            from .slab_pack import get_slab_layout

            header_size, record_stride = _slab_layout_constants(
                self.group_size, self.header_size
            )
            layout = get_slab_layout(self.group_size)
            values.update({
                "SLAB_HEADER_SIZE": f"{header_size}u",
                "SLAB_RECORD_STRIDE": f"{record_stride}u",
                "W_OFFSET": f"{layout.offset(proj_name, 'weight')}u",
                "S_OFFSET": f"{layout.offset(proj_name, 'scales')}u",
                "B_OFFSET": f"{layout.offset(proj_name, 'biases')}u",
            })
        input_names = ["x", "weight", "scales", "biases", "routes"]
        if kind == "down_combine":
            input_names.append("scores")
        if has_slab_pack:
            input_names.append("slab_pack")
            if has_stream_pack:
                input_names.append("stream_pack")
        if kind == "up_swiglu":
            input_names.append("gate_out")
        if has_shared_y:
            input_names.append("shared_y")
        elif has_shared_parts:
            input_names.extend(["shared", "shared_gate"])

        base = f"flashnext_level1_q4g{self.group_size}"
        if kind == "projection":
            name = f"{base}_{proj_name}_pack" if has_slab_pack else base
            if has_stream_pack:
                name += "_stream"
            label = _QMV_LABELS.get(proj_name)
        elif kind == "up_swiglu":
            name = f"{base}_up_swiglu_pack" if has_slab_pack else f"{base}_up_swiglu"
            label = None
        else:
            name = f"{base}_down_combine"
            if has_slab_pack:
                name += "_pack_stream" if has_stream_pack else "_pack"
            if has_shared_y:
                name += "_shared"
            elif has_shared_parts:
                name += "_shared_parts"
            label = "fused_down"
        if (
            label is not None
            and self._boundary_profiler.enabled
            and self._boundary_profiler.selected_for(label)
        ):
            name += f"_boundary_{label}"
        maker = getattr(self.backend, "metal_kernel", None)
        if maker is None:
            import mlx.core as mx

            maker = mx.fast.metal_kernel
        kernel = maker(
            name=name,
            input_names=input_names,
            output_names=["out"],
            source=_render(_BODIES[kind], values),
            header=_mlx_qmv_header(),
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
        self._kernels[key] = kernel
        return kernel

    def _launch(self, label, kernel, inputs, x, grid, threadgroup, output_shape):
        """Dispatch one kernel, timing it only when its boundary is profiled."""
        profiler = self._boundary_profiler
        if label is None or not profiler.enabled or not profiler.selected_for(label):
            result = kernel(
                inputs=inputs, template=[("T", x.dtype)], grid=grid,
                threadgroup=threadgroup, output_shapes=[output_shape],
                output_dtypes=[x.dtype],
            )
            return result[0] if isinstance(result, (tuple, list)) else result

        def issue():
            result = kernel(
                inputs=inputs, template=[("T", x.dtype)], grid=grid,
                threadgroup=threadgroup, output_shapes=[output_shape],
                output_dtypes=[x.dtype],
            )
            return result[0] if isinstance(result, (tuple, list)) else result

        return profiler.measure(label, issue, self._complete_boundary)

    @staticmethod
    def _pack_inputs(inputs, slab_pack, stream_pack):
        if stream_pack is not None and slab_pack is None:
            raise ValueError("stream pack requires a resident slab pack")
        if slab_pack is not None:
            inputs.append(slab_pack)
            if stream_pack is not None:
                inputs.append(stream_pack)
        return inputs

    def _metal_projection(
        self,
        x: Any,
        routes: Any,
        projection: Q4G32Projection,
        output_width: int,
        slot_input: bool,
        slab_pack: Any = None,
        stream_pack: Any = None,
        proj_name: str = "",
    ) -> Any:
        tokens, input_width = _shape(x)[0], _shape(x)[-1]
        slots = _shape(routes)[1]
        inputs = self._pack_inputs(
            [x, projection.weight, projection.scales, projection.biases, routes],
            slab_pack, stream_pack,
        )
        kernel = self._kernel(
            "projection", x, tokens, slots, input_width, output_width,
            proj_name=proj_name, slot_input=slot_input,
            has_slab_pack=slab_pack is not None,
            has_stream_pack=stream_pack is not None,
        )
        return self._launch(
            _QMV_LABELS.get(proj_name), kernel, inputs, x,
            (32, ((output_width + 7) // 8) * 2, tokens * slots), (32, 2, 1),
            (tokens, slots, output_width),
        )

    def _metal_fused_up_swiglu(
        self,
        x: Any,
        routes: Any,
        gate_out: Any,
        up: Q4G32Projection,
        output_width: int,
        slab_pack: Any = None,
        stream_pack: Any = None,
    ) -> Any:
        """Run Up QMV and consume gate output in the same Metal dispatch."""
        tokens, input_width = _shape(x)[0], _shape(x)[-1]
        slots = _shape(routes)[1]
        inputs = self._pack_inputs(
            [x, up.weight, up.scales, up.biases, routes], slab_pack, stream_pack,
        )
        # Keeping gate_out as the final input makes the dependency explicit.
        inputs.append(gate_out)
        kernel = self._kernel(
            "up_swiglu", x, tokens, slots, input_width, output_width,
            proj_name="up_proj", has_slab_pack=slab_pack is not None,
            has_stream_pack=stream_pack is not None,
        )
        return self._launch(
            "swiglu", kernel, inputs, x,
            (32, ((output_width + 7) // 8) * 2, tokens * slots), (32, 2, 1),
            (tokens, slots, output_width),
        )

    def _metal_fused_down_combine(
        self, x, routes, scores, down, output_width, slab_pack=None,
        stream_pack=None, shared_y=None, shared=None, shared_gate=None,
    ):
        tokens, input_width = _shape(x)[0], _shape(x)[-1]
        slots = _shape(routes)[1]
        if (shared is None) != (shared_gate is None):
            raise ValueError("shared and shared_gate must be provided together")
        if shared_y is not None and shared is not None:
            raise ValueError("provide shared_y or shared parts, not both")
        inputs = self._pack_inputs(
            [x, down.weight, down.scales, down.biases, routes, scores],
            slab_pack, stream_pack,
        )
        if shared_y is not None:
            inputs.append(shared_y.reshape(tokens, output_width))
        elif shared is not None:
            inputs.extend([
                shared.reshape(tokens, output_width),
                shared_gate.reshape(tokens),
            ])
        kernel = self._kernel(
            "down_combine", x, tokens, slots, input_width, output_width,
            proj_name="down_proj", has_slab_pack=slab_pack is not None,
            has_stream_pack=stream_pack is not None,
            has_shared_y=shared_y is not None,
            has_shared_parts=shared is not None,
        )
        return self._launch(
            "fused_down", kernel, inputs, x,
            (((output_width + 7) // 8) * 64, 1, tokens), (64, 1, 1),
            (tokens, output_width),
        )

    def _reference_projection(
        self,
        x: Any,
        routes: Any,
        projection: Q4G32Projection,
        output_width: int,
        slot_input: bool,
    ) -> Any:
        # Keep a NumPy reference for tests and non-MLX callers.  Its unpacking
        # mirrors the Metal kernel and uses float32 accumulators.
        if isinstance(x, np.ndarray):
            return _numpy_projection(
                x, np.asarray(routes), projection, output_width, slot_input,
                self.group_size,
            )
        import mlx.core as mx

        # MLX's dequantize uses the same affine Q4 representation.  This
        # fallback preserves values and dtypes while avoiding custom Metal.
        dense = mx.dequantize(
            projection.weight,
            projection.scales,
            projection.biases,
            group_size=self.group_size,
            bits=BITS,
            mode="affine",
        )
        selected = mx.take(dense, routes, axis=0)
        if slot_input:
            return mx.einsum("tki,tkoi->tko", x, selected)
        return mx.einsum("ti,tkoi->tko", x, selected)

    def execute(
        self,
        x: Any,
        routes: Any,
        projections: Mapping[str, Any] | Sequence[Any] | Any = None,
        *,
        return_all: bool = False,
        scores: Any = None,
        slab_pack: Any = None,
        stream_pack: Any = None,
        shared_y: Any = None,
        shared: Any = None,
        shared_gate: Any = None,
    ) -> Any:
        """Run gate, up, activation, and down for bounded flattened inputs.

        ``x`` has shape ``(tokens, hidden_size)`` and ``routes`` has shape
        ``(tokens, top_k)``.  Projections are a mapping with ``gate_proj``,
        ``up_proj``, and ``down_proj`` keys, or a three-item sequence.
        """
        # Level-1 combine-only probes pass already projected values as the
        # first argument and scores as the third.  Keep this small API useful
        # for validating route math without allocating packed expert rows.
        if projections is not None and not isinstance(projections, (Mapping, list, tuple)):
            if not self.capabilities.supports_custom_moe:
                raise RuntimeError("custom MoE is unavailable on this backend")
            self.last_path = "custom-metal"
            self.fallback_reason = None
            return weighted_combine(x, routes, projections)

        if projections is None and scores is not None:
            if not self.capabilities.supports_custom_moe:
                raise RuntimeError("custom MoE is unavailable on this backend")
            self.last_path = "custom-metal"
            self.fallback_reason = None
            return weighted_combine(x, routes, scores)

        if projections is None:
            raise TypeError("packed execution needs gate, up, and down projections")
        x_shape = _shape(x)
        route_shape = _shape(routes)
        if len(x_shape) != 2 or x_shape[1] != self.hidden_size:
            raise ValueError("x must have shape (tokens, hidden_size)")
        if len(route_shape) != 2 or route_shape[1] != self.top_k:
            raise ValueError("routes must have shape (tokens, top_k)")
        if scores is not None and _shape(scores) != route_shape:
            raise ValueError("scores must have the same shape as routes")
        if (shared is None) != (shared_gate is None):
            raise ValueError("shared and shared_gate must be provided together")
        if shared_y is not None and shared is not None:
            raise ValueError("provide shared_y or shared parts, not both")
        if stream_pack is not None and slab_pack is None:
            raise ValueError("stream pack requires a resident slab pack")
        tokens = x_shape[0]
        if route_shape[0] != tokens:
            raise ValueError("x and routes must have the same token count")
        if not 1 <= tokens <= self.max_tokens:
            raise ValueError(f"tokens must be in 1..{self.max_tokens}")
        # Reading MLX route values with ``np.asarray`` forces a host sync.
        # The production route tensor is already bounded by the router.  Do
        # value validation only for NumPy callers, which are synchronous.
        if isinstance(routes, np.ndarray):
            if np.any(routes < 0):
                raise ValueError("routes contain an expert outside the configured bank")
        if isinstance(projections, Mapping):
            gate = _as_projection(projections["gate_proj"])
            up = _as_projection(projections["up_proj"])
            down = _as_projection(projections["down_proj"])
        else:
            if len(projections) != 3:
                raise ValueError("projections must contain gate, up, and down")
            gate, up, down = (_as_projection(item) for item in projections)

        gate_width = _shape(gate.weight)[1]
        inter_width = _shape(up.weight)[1]
        if gate_width != inter_width or gate_width > self.max_width:
            raise ValueError("gate and up projection widths must match the bound")
        _validate_projection(
            gate, None, self.hidden_size, gate_width, self.group_size
        )
        _validate_projection(
            up, None, self.hidden_size, inter_width, self.group_size
        )
        if self.hidden_size % self.group_size or inter_width % self.group_size:
            raise ValueError(
                f"Q4/G{self.group_size} projection inputs must be group aligned"
            )
        _validate_projection(
            down, None, inter_width, self.hidden_size, self.group_size
        )
        if slab_pack is not None and (
            self.hidden_size != 2560 or gate_width != 640 or inter_width != 640
        ):
            raise ValueError(
                "packed slab execution requires hidden=2560 and intermediate=640"
            )

        if not self.available:
            if stream_pack is not None:
                raise RuntimeError("stream pack requires custom Metal")
            gate_out, up_out, down_out = self._reference_all(x, routes, gate, up, down)
            if scores is not None and not return_all:
                down_out = weighted_combine(down_out, routes, scores)
                if shared_y is not None:
                    down_out = down_out + shared_y
                elif shared is not None:
                    down_out = down_out + shared_gate * shared
            self.last_path = "reference"
            self.fallback_reason = self.capabilities["reason"]
            return (gate_out, up_out, down_out) if return_all else down_out

        # Separate gate and up projections outperform the fused variant on
        # M4. Fusion raises register pressure enough to exceed the saved
        # launch. The down-plus-router fusion remains profitable.
        gate_out = self._metal_projection(
            x, routes, gate, gate_width, False, slab_pack=slab_pack,
            stream_pack=stream_pack, proj_name="gate_proj",
        )
        # return_all exposes the raw Up projection for verification. Keep
        # that diagnostic API unchanged when the fusion flag is enabled.
        up_out = None
        if self.fused_up_swiglu and not return_all:
            activation = self._metal_fused_up_swiglu(
                x, routes, gate_out, up, inter_width,
                slab_pack=slab_pack, stream_pack=stream_pack,
            )
        else:
            up_out = self._metal_projection(
                x, routes, up, inter_width, False, slab_pack=slab_pack,
                stream_pack=stream_pack, proj_name="up_proj",
            )
            from mlx_vlm.models.activations import swiglu

            if (
                self._boundary_profiler.enabled
                and self._boundary_profiler.selected_for("swiglu")
            ):
                activation = self._boundary_profiler.measure(
                    "swiglu",
                    lambda: swiglu(gate_out, up_out),
                    self._complete_boundary,
                )
            else:
                activation = swiglu(gate_out, up_out)
        if scores is not None and not return_all and self.group_size != 64:
            down_out = self._metal_fused_down_combine(
                activation, routes, scores, down, self.hidden_size,
                slab_pack=slab_pack, stream_pack=stream_pack,
                shared_y=shared_y, shared=shared, shared_gate=shared_gate,
            )
        else:
            down_out = self._metal_projection(
                activation, routes, down, self.hidden_size, True,
                slab_pack=slab_pack, stream_pack=stream_pack,
                proj_name="down_proj",
            )
            if scores is not None and not return_all:
                # Keep the G64 reduction in MLX. The generic path uses the
                # score dtype for the product and sum, while the fused kernel
                # accumulates in float32 before its final output cast. These
                # orders differ for BF16 scores.
                down_out = weighted_combine(down_out, routes, scores)
                if shared_y is not None:
                    down_out = down_out + shared_y
                elif shared is not None:
                    down_out = down_out + shared_gate * shared
        self.last_path = "custom-metal"
        self.fallback_reason = None
        return (gate_out, up_out, down_out) if return_all else down_out

    def _reference_all(self, x: Any, routes: Any, gate: Any, up: Any, down: Any):
        gate_out = self._reference_projection(x, routes, gate, _shape(gate.weight)[1], False)
        up_out = self._reference_projection(x, routes, up, _shape(up.weight)[1], False)
        activation = up_out * (gate_out / (1.0 + _exp(-gate_out)))
        down_out = self._reference_projection(
            activation, routes, down, self.hidden_size, True
        )
        return gate_out, up_out, down_out


def _exp(value: Any) -> Any:
    try:
        import mlx.core as mx

        if not isinstance(value, np.ndarray):
            return mx.exp(value)
    except ImportError:
        pass
    return np.exp(value)


def _numpy_projection(
    x: np.ndarray,
    routes: np.ndarray,
    projection: Q4G32Projection,
    output_width: int,
    slot_input: bool,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    packed = np.asarray(projection.weight)
    scales = np.asarray(projection.scales)
    biases = np.asarray(projection.biases)
    tokens, input_width = x.shape[0], x.shape[-1]
    slots = routes.shape[1]
    out = np.zeros((tokens, slots, output_width), dtype=np.float32)
    for token in range(tokens):
        for slot in range(slots):
            expert = int(routes[token, slot])
            source = x[token, slot] if slot_input else x[token]
            for row in range(output_width):
                result = 0.0
                for group in range(input_width // group_size):
                    qsum = 0.0
                    xsum = 0.0
                    for i in range(group_size):
                        k = group * group_size + i
                        q = (int(packed[expert, row, k // 8]) >> ((k & 7) * 4)) & 15
                        value = float(source[k])
                        qsum += value * q
                        xsum += value
                    result += float(scales[expert, row, group]) * qsum
                    result += float(biases[expert, row, group]) * xsum
                out[token, slot, row] = result
    return out.astype(x.dtype, copy=False)


__all__ = [
    "BITS",
    "GROUP_SIZE",
    "MAX_EXPERTS",
    "MAX_TOKENS",
    "MAX_TOP_K",
    "MAX_WIDTH",
    "BoundaryProfiler",
    "MetalCapabilities",
    "MetalMoEExecutor",
    "Q4G32Projection",
    "SUPPORTED_GROUP_SIZES",
    "probe_capabilities",
    "weighted_combine",
]
