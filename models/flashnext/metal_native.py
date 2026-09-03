"""Loader for the native Metal dependency-chain probe.

Unlike ``mx.fast.metal_kernel``, this probe owns its ``MTLCommandQueue`` and
``MTLCommandBuffer``.  It is a scheduler experiment, not a Q4 executor yet.
It runs a bounded float32 chain with optional buffer barriers and returns a
clean unavailable status when compilation or Metal is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


MAX_WIDTH = 16_384
MAX_STEPS = 64
STRATEGIES = ("serial", "barrier", "fence")


@dataclass(frozen=True)
class NativeMetalStatus:
    """Availability and build details for the native bridge."""

    available: bool
    reason: str
    library: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


_HANDLE: ctypes.CDLL | None = None
_STATUS: NativeMetalStatus | None = None


def _source_path() -> Path:
    return Path(__file__).with_name("metal_runtime_native.mm")


def _library_path(source: Path) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / "macqwen-flashnext"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"metal_runtime_native-{digest}.dylib"


def load_native() -> tuple[ctypes.CDLL | None, NativeMetalStatus]:
    """Build and load the bridge once.  No build occurs on non-macOS hosts."""
    global _HANDLE, _STATUS
    if _STATUS is not None:
        return _HANDLE, _STATUS
    if sys.platform != "darwin":
        _STATUS = NativeMetalStatus(False, "native Metal requires macOS")
        return None, _STATUS
    source = _source_path()
    library = _library_path(source)
    if not library.exists():
        command = [
            "xcrun", "clang++", "-std=c++17", "-fobjc-arc", "-dynamiclib",
            str(source), "-framework", "Foundation", "-framework", "Metal",
            "-o", str(library),
        ]
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _STATUS = NativeMetalStatus(False, f"native build failed: {exc}")
            return None, _STATUS
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            _STATUS = NativeMetalStatus(False, f"native build failed: {detail}")
            return None, _STATUS
    try:
        handle = ctypes.CDLL(str(library))
        handle.flashnext_native_available.argtypes = []
        handle.flashnext_native_available.restype = ctypes.c_int
        handle.flashnext_native_chain.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p,
        ]
        handle.flashnext_native_chain.restype = ctypes.c_int
        handle.flashnext_native_get_last_error.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        handle.flashnext_native_get_last_error.restype = ctypes.c_int
        handle.flashnext_native_init_moe.argtypes = [ctypes.c_char_p]
        handle.flashnext_native_init_moe.restype = ctypes.c_int
        handle.flashnext_native_moe_execute.argtypes = [
            ctypes.POINTER(FlashNextMoEArgs),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        handle.flashnext_native_moe_execute.restype = ctypes.c_int
        if handle.flashnext_native_available() != 1:
            _STATUS = NativeMetalStatus(False, "Metal device is unavailable", str(library))
            return None, _STATUS
    except OSError as exc:
        _STATUS = NativeMetalStatus(False, f"native load failed: {exc}", str(library))
        return None, _STATUS
    _HANDLE = handle
    _STATUS = NativeMetalStatus(True, "native Metal bridge loaded", str(library))
    return _HANDLE, _STATUS


class FlashNextMoEArgs(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p),
        ("gate_weight", ctypes.c_void_p),
        ("gate_scales", ctypes.c_void_p),
        ("gate_biases", ctypes.c_void_p),
        ("up_weight", ctypes.c_void_p),
        ("up_scales", ctypes.c_void_p),
        ("up_biases", ctypes.c_void_p),
        ("down_weight", ctypes.c_void_p),
        ("down_scales", ctypes.c_void_p),
        ("down_biases", ctypes.c_void_p),
        ("routes", ctypes.c_void_p),
        ("scores", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("tokens", ctypes.c_uint32),
        ("slots", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("inter_size", ctypes.c_uint32),
        ("expert_count", ctypes.c_uint32),
    ]


def probe_native() -> NativeMetalStatus:
    """Compile, load, and report the native bridge capability."""
    return load_native()[1]


def run_dependency_chain(
    input_values: Any,
    *,
    steps: int = 8,
    strategy: str = "serial",
    barrier_every: int | None = None,
) -> np.ndarray:
    """Run ``output = input + steps`` through explicit Metal encoders.

    ``serial`` uses one normal encoder.  ``barrier`` uses one concurrent
    encoder and a buffer barrier after every dispatch.  ``fence`` uses one
    encoder per dispatch with explicit fence waits and updates.  The legacy
    ``barrier_every`` argument maps zero to ``serial`` and non-zero to
    ``barrier``.  The function never changes the caller's input array.
    """
    values = np.ascontiguousarray(input_values, dtype=np.float32)
    if values.ndim != 1 or not 1 <= values.size <= MAX_WIDTH:
        raise ValueError(f"input must be a 1-D array with 1..{MAX_WIDTH} values")
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be in 1..{MAX_STEPS}")
    if barrier_every is not None:
        if barrier_every < 0 or barrier_every > MAX_STEPS:
            raise ValueError("barrier_every must be zero or at most MAX_STEPS")
        strategy = "barrier" if barrier_every else "serial"
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    handle, status = load_native()
    if handle is None:
        raise RuntimeError(status.reason)
    output = np.empty_like(values)
    code = handle.flashnext_native_chain(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(values.size), ctypes.c_uint32(steps),
        strategy.encode("ascii"),
    )
    if code != 0:
        raise RuntimeError(f"native Metal chain failed with status {code}")
    return output


_MOE_METAL_SOURCE: str | None = None
_MOE_INITIALIZED = False


def get_moe_metal_source() -> str:
    """Generate the Metal source containing mixed QMV helpers and MoE kernels."""
    global _MOE_METAL_SOURCE
    if _MOE_METAL_SOURCE is not None:
        return _MOE_METAL_SOURCE
    from .metal_runtime import _mlx_qmv_header

    header = _mlx_qmv_header().replace("const int&", "const int")
    shaders = r"""
kernel void flashnext_qmv_proj(
    const device float* x [[buffer(0)]],
    const device uint32_t* weight [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    const device uint32_t* routes [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& slots [[buffer(7)]],
    constant uint& in_width [[buffer(8)]],
    constant uint& out_width [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    uint pair = group.z;
    uint token = pair / slots;
    if (token >= tokens) return;
    uint expert = routes[pair];
    const int in_size = in_width;
    const int out_size = out_width;
    const device float* input = x + token * in_width;
    device float* output = out + pair * out_width;
    uint3 tid = uint3(0, group.y, pair);
    qmv_fast_mixed_impl<float, 32, 4, bfloat>(
        weight + expert * out_width * (in_width / 8),
        scales + expert * out_width * (in_width / 32),
        biases + expert * out_width * (in_width / 32),
        input, output, in_size, out_size, tid, simd_gid, simd_lid);
}

kernel void flashnext_swiglu(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < total_elements) {
        float g = gate[id];
        float u = up[id];
        float sig = 1.0f / (1.0f + metal::exp(-g));
        out[id] = u * (g * sig);
    }
}

kernel void flashnext_fused_down_combine(
    const device float* x [[buffer(0)]],
    const device uint32_t* weight [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    const device uint32_t* routes [[buffer(4)]],
    const device float* scores [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    constant uint& slots [[buffer(9)]],
    constant uint& in_width [[buffer(10)]],
    constant uint& out_width [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
#pragma clang fp contract(off)
    uint token = group.z;
    uint out_base = group.x * 8 + simd_gid * 4;
    if (token >= tokens || out_base >= out_width) return;
    float combined[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint slot = 0; slot < slots; ++slot) {
        uint expert = routes[token * slots + slot];
        const int in_size = in_width;
        const int out_size = out_width;
        uint3 tid = uint3(0, group.x, token);
        device float* slot_output = scratch +
            (token * slots + slot) * out_width;
        qmv_mixed_impl<float, 32, 4, bfloat>(
            weight + expert * out_width * (in_width / 8),
            scales + expert * out_width * (in_width / 32),
            biases + expert * out_width * (in_width / 32),
            x + (token * slots + slot) * in_width,
            slot_output, in_size, out_size, tid,
            simd_gid, simd_lid);
        threadgroup_barrier(mem_flags::mem_device);
        if (simd_lid == 0) {
#pragma unroll
            for (uint row = 0; row < 4; ++row) {
                if (out_base + row >= out_width) continue;
                float product = float(slot_output[out_base + row]) *
                                float(scores[token * slots + slot]);
                combined[row] += product;
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    if (simd_lid == 0) {
#pragma unroll
        for (uint row = 0; row < 4; ++row)
            if (out_base + row < out_width)
                out[token * out_width + out_base + row] = combined[row];
    }
}
"""
    _MOE_METAL_SOURCE = header + shaders
    return _MOE_METAL_SOURCE


def init_native_moe() -> None:
    """Compile and register the native Q4 MoE kernels with the Metal device."""
    global _MOE_INITIALIZED
    if _MOE_INITIALIZED:
        return
    handle, status = load_native()
    if handle is None:
        raise RuntimeError(status.reason)
    source = get_moe_metal_source()
    code = handle.flashnext_native_init_moe(source.encode("utf-8"))
    if code != 0:
        buf = ctypes.create_string_buffer(1024)
        handle.flashnext_native_get_last_error(buf, len(buf))
        err_msg = buf.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"flashnext_native_init_moe failed (code {code}): {err_msg}")
    _MOE_INITIALIZED = True


def _ptr(arr: Any) -> int:
    """Extract a C data pointer from an MLX or NumPy array without copying."""
    try:
        import mlx.core as mx

        if isinstance(arr, mx.array):
            if arr.dtype == mx.bfloat16:
                arr = arr.view(mx.uint16)
            arr = np.array(arr, copy=False)
    except ImportError:
        pass
    if not isinstance(arr, np.ndarray):
        arr = np.ascontiguousarray(arr)
    return arr.ctypes.data


def run_native_moe(
    x: Any,
    routes: Any,
    scores: Any,
    gate_pack: tuple[Any, Any, Any],
    up_pack: tuple[Any, Any, Any],
    down_pack: tuple[Any, Any, Any],
    *,
    strategy: str = "serial",
    expert_count: int = 512,
) -> tuple[np.ndarray, float]:
    """Execute a 3-projection Q4 MoE block directly via native MTLCommandBuffer.

    Returns ``(output, gpu_time_ms)``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    init_native_moe()
    handle, _ = load_native()
    if handle is None:
        raise RuntimeError("native Metal library handle unavailable")

    x_ptr = _ptr(x)
    routes_ptr = _ptr(routes)
    scores_ptr = _ptr(scores)
    gw_ptr = _ptr(gate_pack[0])
    gs_ptr = _ptr(gate_pack[1])
    gb_ptr = _ptr(gate_pack[2])
    uw_ptr = _ptr(up_pack[0])
    us_ptr = _ptr(up_pack[1])
    ub_ptr = _ptr(up_pack[2])
    dw_ptr = _ptr(down_pack[0])
    ds_ptr = _ptr(down_pack[1])
    db_ptr = _ptr(down_pack[2])

    tokens = 1 if len(x.shape) == 1 else x.shape[0]
    hidden_size = x.shape[-1]
    slots = routes.shape[-1]
    inter_size = gate_pack[0].shape[1]

    output = np.empty((tokens, hidden_size), dtype=np.float32)
    output_ptr = output.ctypes.data

    args = FlashNextMoEArgs(
        x=ctypes.c_void_p(x_ptr),
        gate_weight=ctypes.c_void_p(gw_ptr),
        gate_scales=ctypes.c_void_p(gs_ptr),
        gate_biases=ctypes.c_void_p(gb_ptr),
        up_weight=ctypes.c_void_p(uw_ptr),
        up_scales=ctypes.c_void_p(us_ptr),
        up_biases=ctypes.c_void_p(ub_ptr),
        down_weight=ctypes.c_void_p(dw_ptr),
        down_scales=ctypes.c_void_p(ds_ptr),
        down_biases=ctypes.c_void_p(db_ptr),
        routes=ctypes.c_void_p(routes_ptr),
        scores=ctypes.c_void_p(scores_ptr),
        output=ctypes.c_void_p(output_ptr),
        tokens=tokens,
        slots=slots,
        hidden_size=hidden_size,
        inter_size=inter_size,
        expert_count=expert_count,
    )

    gpu_ms = ctypes.c_double(0.0)
    code = handle.flashnext_native_moe_execute(
        ctypes.byref(args),
        strategy.encode("ascii"),
        ctypes.byref(gpu_ms),
    )
    if code != 0:
        buf = ctypes.create_string_buffer(1024)
        handle.flashnext_native_get_last_error(buf, len(buf))
        err_msg = buf.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"flashnext_native_moe_execute failed ({code}): {err_msg}")

    return output, gpu_ms.value


__all__ = [
    "MAX_STEPS",
    "MAX_WIDTH",
    "STRATEGIES",
    "FlashNextMoEArgs",
    "NativeMetalStatus",
    "get_moe_metal_source",
    "init_native_moe",
    "load_native",
    "probe_native",
    "run_dependency_chain",
    "run_native_moe",
]

