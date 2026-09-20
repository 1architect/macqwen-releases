"""Fused sign+Hadamard+downcast kernel for Bonsai-2 ternary projections.

The stock path per packed module runs three launches: a sign multiply, the
library Hadamard transform, and a downcast. This module folds all three into
one dispatch over 1024-element blocks. Anything the kernel does not cover
falls back to the stock ``runtime.fwht``.

The backend enables the kernel by default through ``fused_fwht=True``. The
stock path remains available with ``fused_fwht=False``; the internal
``BONSAI2_FUSED_FWHT`` flag is managed by ``apply_runtime_hooks``.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

_FUSED_FWHT_SOURCE = """
// One threadgroup per BLOCK slice of one row. Thread i owns element i.
// This is a kernel body fragment: mx.fast.metal_kernel supplies the wrapper
// and the x/signs/out buffers. Activations are fp16 or fp32; signs are fp32.
{
  constexpr uint N = BLOCK;
  threadgroup float buf[N];
  // Blocks tile the flattened tensor, so the sign index wraps by row width:
  // signs cover one row (WIDTH elements), matching the stock broadcast.
  uint base = threadgroup_position_in_grid.x * N;
  uint width = WIDTH;
  uint i = thread_position_in_threadgroup.x;
  buf[i] = float(x[base + i]) * signs[(base + i) % width];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1; stride < N; stride <<= 1) {
    uint full = stride << 1;
    uint j = (i & ~(full - 1)) | (i & (stride - 1));
    float a = buf[j];
    float b = buf[j + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[i] = ((i & stride) == 0) ? (a + b) : (a - b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[base + i] = OUTPUT_TYPE(buf[i] * SCALE);
}
"""


def _dtype_tag(dtype) -> str:
    name = str(getattr(dtype, "value", dtype))
    if "bfloat16" in name:
        return "bfloat16"
    if "float16" in name or name == "half":
        return "half"
    if "float32" in name or name == "float":
        return "float32"
    return "unsupported:" + name


@lru_cache(maxsize=None)
def _get_kernel(block: int, width: int, dtype_tag: str):
    import mlx.core as mx

    if dtype_tag not in ("half", "float32"):
        raise ValueError(
            f"fused FWHT supports fp16 or fp32 activations, got {dtype_tag}"
        )
    source = _FUSED_FWHT_SOURCE.replace("BLOCK", str(block))
    source = source.replace("WIDTH", str(width))
    scale = 1.0 / math.sqrt(block)
    source = source.replace("SCALE", repr(float(scale)))
    source = source.replace("OUTPUT_TYPE", "half" if dtype_tag == "half" else "float")
    return mx.fast.metal_kernel(
        name=f"bonsai2_fused_fwht_b{block}_w{width}_{dtype_tag}",
        input_names=["x", "signs"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def fused_fwht_enabled() -> bool:
    return os.environ.get("BONSAI2_FUSED_FWHT") == "1"


_STOCK_FWHT = None
_STOCK_OWNER = None
_MEMO = None
_ORIGINALS: dict = {}
_INSTALLED = None
_ACTIVE_COUNTERS = None


def new_fwht_counters() -> dict:
    return {
        "requested": False,
        "attempts": 0,
        "selected": 0,
        "fallbacks": 0,
        "executed": False,
        "input_dtypes": {},
        "fallback_reasons": {},
    }


def set_fwht_counters(counters) -> None:
    global _ACTIVE_COUNTERS
    _ACTIVE_COUNTERS = counters


def _fwht_event(name: str, *, dtype: str | None = None, reason: str | None = None) -> None:
    if not isinstance(_ACTIVE_COUNTERS, dict):
        return
    if name in ("attempts", "selected", "fallbacks"):
        _ACTIVE_COUNTERS[name] += 1
    if name == "selected":
        _ACTIVE_COUNTERS["executed"] = True
    if dtype:
        inputs = _ACTIVE_COUNTERS["input_dtypes"]
        inputs[dtype] = inputs.get(dtype, 0) + 1
    if reason:
        reasons = _ACTIVE_COUNTERS["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1


def _tracked_runtime_module(checkpoint_path=None):
    """Return the checkpoint runtime module, verifying provenance.

    A foreign `runtime` module already in sys.modules (another checkpoint,
    a test double without a matching file, or anything outside the selected
    checkpoint directory) refuses patching instead of silently patching the
    wrong code. Returns None only when no runtime module is loaded at all.
    """
    import sys
    from pathlib import Path

    module = sys.modules.get("runtime")
    if module is None:
        return None
    if checkpoint_path is None:
        return module
    anchor = getattr(module, "__file__", "")
    try:
        inside = (
            Path(anchor).resolve().is_relative_to(
                Path(checkpoint_path).resolve() / "runtime"
            )
            if anchor
            else False
        )
    except (OSError, ValueError):
        inside = False
    if not inside:
        raise RuntimeError(
            "Refusing to patch a foreign runtime module: "
            f"{anchor!r} is not inside {checkpoint_path}"
        )
    return module


def restore_runtime_hooks(checkpoint_path=None) -> bool:
    """Undo transform patches and return the module to stock behavior.

    Clears per-module patch flags and drops saved originals so a later
    backend constructed without the flags genuinely runs stock code.
    """
    global _STOCK_FWHT, _STOCK_OWNER, _INSTALLED
    # Provenance-checked like the install path: restoring a runtime module
    # that does not belong to this checkpoint would corrupt another
    # checkpoint's code.
    module = _tracked_runtime_module(checkpoint_path)
    restored = False
    if module is not None:
        saved = _ORIGINALS.pop(id(module), None)
        if saved is not None:
            module.fwht = saved
            restored = True
        if getattr(module, "_bonsai2_fused", False):
            delattr(module, "_bonsai2_fused")
            restored = True
        if getattr(module, "_bonsai2_shared", False):
            delattr(module, "_bonsai2_shared")
            restored = True
        if _STOCK_OWNER == id(module):
            _STOCK_FWHT = None
            _STOCK_OWNER = None
    _INSTALLED = None
    disarm_memo()
    return restored


def apply_runtime_hooks(checkpoint_path=None, fused=False, share=False) -> None:
    """Build the runtime transform composition in one canonical order.

    Stock restores first whenever the requested composition differs, then
    fused installs innermost and share outermost. Separate
    install-then-install call sites made the result depend on
    construction order: a fused install after a share install silently
    dropped the share layer, and flags from an earlier backend leaked into
    later ones through the process environment. Keying on module identity
    plus both flags makes repeat construction a verified no-op instead of
    a second patch layer.
    """
    global _INSTALLED
    module = _tracked_runtime_module(checkpoint_path)
    key = (id(module) if module is not None else None, bool(fused), bool(share))
    if _INSTALLED == key and module is not None:
        if getattr(module, "_bonsai2_fused", False) == bool(
            fused
        ) and getattr(module, "_bonsai2_shared", False) == bool(share):
            return
    restore_runtime_hooks(checkpoint_path)
    if module is None:
        _INSTALLED = key
        return
    if fused:
        os.environ["BONSAI2_FUSED_FWHT"] = "1"
    else:
        os.environ.pop("BONSAI2_FUSED_FWHT", None)
    if share:
        os.environ["BONSAI2_SHARE_FWHT"] = "1"
    else:
        os.environ.pop("BONSAI2_SHARE_FWHT", None)
    if fused:
        install_packed_hook(checkpoint_path)
    if share:
        install_share_hook(checkpoint_path)
    _INSTALLED = key


def share_fwht_enabled() -> bool:
    return os.environ.get("BONSAI2_SHARE_FWHT") == "1"


def arm_memo() -> None:
    """Open a call-scoped transform cache. The wrapper arms before each model
    forward and disarms after it returns; entries hold their inputs alive so
    object ids cannot be reused while cached, and clearing per forward bounds
    memory. Hits return the identical computed array, so shared calls are
    bit-identical by construction."""
    global _MEMO
    _MEMO = {}


def disarm_memo() -> None:
    global _MEMO
    _MEMO = None


def install_share_hook(checkpoint_path=None) -> bool:
    """Memoize forward transforms across modules sharing one input object.

    Same-width modules carry byte-identical sign vectors, so a shared input
    means a shared result. The memo consults whatever transform sits
    underneath (fused kernel when enabled, stock otherwise).
    """
    if not share_fwht_enabled():
        return False
    module = _tracked_runtime_module(checkpoint_path)
    if module is None:
        return False
    if getattr(module, "_bonsai2_shared", False):
        return True
    inner = module.fwht
    _ORIGINALS.setdefault(id(module), inner)

    def shared(x, block, signs, inverse=False):
        memo = _MEMO
        if memo is None or inverse:
            return inner(x, block, signs, inverse=inverse)
        # Same-width sign vectors are byte-identical (verified per loaded
        # model in the backend), so width identifies the signs. Inputs are
        # held alive by their entries, so ids cannot be reused mid-forward.
        key = (id(x), x.shape[-1], tuple(x.shape), block)
        hit = memo.get(key)
        if hit is not None:
            return hit[2]
        result = inner(x, block, signs, inverse=inverse)
        memo[key] = (x, signs, result)
        return result

    module.fwht = shared
    module._bonsai2_shared = True
    return True


def install_packed_hook(checkpoint_path=None) -> bool:
    """Route the checkpoint runtime's forward transform through the kernel.

    ``Packed.__call__`` resolves ``fwht`` from its own module globals, so
    patching that attribute redirects every forward transform without
    touching checkpoint code. The inverse embedding path keeps the stock
    implementation. Returns whether the hook is active.
    """
    global _STOCK_FWHT, _STOCK_OWNER
    if not fused_fwht_enabled():
        return False
    module = _tracked_runtime_module(checkpoint_path)
    if module is None:
        return False
    if getattr(module, "_bonsai2_fused", False):
        return True
    _STOCK_FWHT = module.fwht
    _STOCK_OWNER = id(module)
    _ORIGINALS.setdefault(id(module), module.fwht)
    stock = module.fwht

    def hooked(x, block, signs, inverse=False):
        # Captured, not global: a later install on another module replaces
        # _STOCK_FWHT, and this hook must keep calling its own stock.
        if inverse:
            return stock(x, block, signs, inverse=inverse)
        return fused_fwht(x, signs, block)

    module.fwht = hooked
    module._bonsai2_fused = True
    return True


def fused_fwht(x, signs, block: int):
    """Apply sign multiply, Hadamard transform, and downcast in one launch.

    Falls back to the stock runtime path when the row width is not a
    multiple of ``block``, the block exceeds the device threadgroup limit,
    the activation dtype is not fp16/fp32, or the fused path is disabled.
    """
    import mlx.core as mx

    width = x.shape[-1]
    dtype_tag = _dtype_tag(x.dtype)
    _fwht_event("attempts", dtype=dtype_tag)
    if (
        not fused_fwht_enabled()
        or width % block != 0
        or block > 1024
        or dtype_tag not in ("half", "float32")
    ):
        reason = (
            "disabled" if not fused_fwht_enabled()
            else "width_not_divisible" if width % block
            else "block_too_large" if block > 1024
            else "unsupported_dtype"
        )
        _fwht_event("fallbacks", reason=reason)
        if _STOCK_FWHT is not None:
            return _STOCK_FWHT(
                x.astype(mx.float32), block, signs, inverse=False
            ).astype(x.dtype)
        from runtime import fwht as stock_fwht

        return stock_fwht(
            x.astype(mx.float32), block, signs, inverse=False
        ).astype(x.dtype)
    _fwht_event("selected")
    kernel = _get_kernel(block, width, dtype_tag)
    # MLX grid counts total threads, not threadgroups: one thread per
    # element, grouped in BLOCK-wide threadgroups of 1024 threads.
    outputs = kernel(
        inputs=[x, signs],
        grid=(x.size, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[list(x.shape)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]
