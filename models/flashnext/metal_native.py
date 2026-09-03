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
        if handle.flashnext_native_available() != 1:
            _STATUS = NativeMetalStatus(False, "Metal device is unavailable", str(library))
            return None, _STATUS
    except OSError as exc:
        _STATUS = NativeMetalStatus(False, f"native load failed: {exc}", str(library))
        return None, _STATUS
    _HANDLE = handle
    _STATUS = NativeMetalStatus(True, "native Metal bridge loaded", str(library))
    return _HANDLE, _STATUS


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


__all__ = [
    "MAX_STEPS",
    "MAX_WIDTH",
    "STRATEGIES",
    "NativeMetalStatus",
    "load_native",
    "probe_native",
    "run_dependency_chain",
]
