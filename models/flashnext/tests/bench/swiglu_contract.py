"""Check bfloat16 rounding contracts for the FlashNext SwiGLU activation.

The production activation uses the MLX ``SwiGLU`` implementation.  This module
keeps that implementation as the reference and compares candidate Metal-style
rounding sequences without loading a checkpoint.

The exhaustive helper enumerates every 16-bit bfloat16 gate encoding.  It uses
bounded MLX batches so a diagnostic run does not create one large graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
from typing import Callable, Iterable, Mapping

import numpy as np
import mlx.core as mx
import mlx.nn as nn


TOTAL_GATE_PATTERNS = 1 << 16
DEFAULT_BATCH_SIZE = 4096
DEFAULT_MAX_MISMATCHES = 12
METAL_CANDIDATE = "metal_swiglu"
METAL_HEADER_CANDIDATE = "metal_mlx_header_bf16"
METAL_PRECISE_CANDIDATE = "metal_precise_exp_bf16"


@dataclass(frozen=True)
class SwiGLUMismatch:
    """One bit-pattern mismatch in a candidate result."""

    gate_bits: int
    expected_bits: int
    actual_bits: int


@dataclass(frozen=True)
class CandidateReport:
    """Mismatch totals and examples for one candidate sequence."""

    name: str
    total: int
    mismatches: int
    examples: tuple[SwiGLUMismatch, ...]

    @property
    def exact(self) -> bool:
        return self.mismatches == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total": self.total,
            "mismatches": self.mismatches,
            "exact": self.exact,
            "examples": [
                {
                    "gate_bits": f"0x{item.gate_bits:04x}",
                    "expected_bits": f"0x{item.expected_bits:04x}",
                    "actual_bits": f"0x{item.actual_bits:04x}",
                }
                for item in self.examples
            ],
        }


@dataclass(frozen=True)
class ExhaustiveReport:
    """Result of comparing all gate encodings for one bfloat16 up value."""

    total: int
    batch_size: int
    up_bits: int
    candidates: tuple[CandidateReport, ...]

    def candidate(self, name: str) -> CandidateReport:
        """Return a candidate report by name."""
        for report in self.candidates:
            if report.name == name:
                return report
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "batch_size": self.batch_size,
            "up_bits": f"0x{self.up_bits:04x}",
            "candidates": [item.as_dict() for item in self.candidates],
        }

    def format(self) -> str:
        """Return a compact human-readable mismatch report."""
        lines = [
            f"gate patterns: {self.total} (batch size {self.batch_size})",
            f"up bits: 0x{self.up_bits:04x}",
        ]
        for report in self.candidates:
            state = "exact" if report.exact else f"{report.mismatches} mismatches"
            lines.append(f"{report.name}: {state} of {report.total}")
            for item in report.examples:
                lines.append(
                    "  gate=0x{0:04x} expected=0x{1:04x} actual=0x{2:04x}".format(
                        item.gate_bits, item.expected_bits, item.actual_bits
                    )
                )
        return "\n".join(lines)


Candidate = Callable[[mx.array, mx.array], mx.array]


def _bfloat16(value: object) -> mx.array:
    """Convert a value to an MLX bfloat16 array."""
    return mx.array(value).astype(mx.bfloat16)


def bfloat16_from_bits(bits: object) -> mx.array:
    """View uint16 bfloat16 encodings as an MLX bfloat16 array."""
    raw = np.asarray(bits, dtype=np.uint16)
    return mx.array(raw).view(mx.bfloat16)


def bfloat16_bits(values: mx.array) -> np.ndarray:
    """Materialise bfloat16 values as uint16 bit patterns."""
    if values.dtype != mx.bfloat16:
        values = values.astype(mx.bfloat16)
    mx.eval(values)
    return np.asarray(values.view(mx.uint16), dtype=np.uint16)


def mlx_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Run the model's MLX SwiGLU reference on bfloat16 inputs."""
    gate = gate.astype(mx.bfloat16)
    up = up.astype(mx.bfloat16)
    try:
        from mlx_vlm.models.activations import swiglu
    except ImportError:
        return (nn.silu(gate) * up).astype(mx.bfloat16)
    return swiglu(gate, up).astype(mx.bfloat16)


def _bf16_silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    gate = gate.astype(mx.bfloat16)
    up = up.astype(mx.bfloat16)
    return (nn.silu(gate) * up).astype(mx.bfloat16)


def _fp32_silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    return (nn.silu(gate) * up).astype(mx.bfloat16)


def _fp32_silu_bf16_product(gate: mx.array, up: mx.array) -> mx.array:
    gate = gate.astype(mx.float32)
    up = up.astype(mx.bfloat16)
    return (nn.silu(gate).astype(mx.bfloat16) * up).astype(mx.bfloat16)


def _bf16_sigmoid_mul(gate: mx.array, up: mx.array) -> mx.array:
    gate = gate.astype(mx.bfloat16)
    up = up.astype(mx.bfloat16)
    return (gate * mx.sigmoid(gate) * up).astype(mx.bfloat16)


def _fp32_sigmoid_mul(gate: mx.array, up: mx.array) -> mx.array:
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    return (gate * mx.sigmoid(gate) * up).astype(mx.bfloat16)


_METAL_KERNELS: dict[str, object] = {}


def metal_swiglu_available() -> tuple[bool, str]:
    """Return whether MLX can launch the standalone Metal candidate."""
    if sys.platform != "darwin":
        return False, "Metal SwiGLU requires macOS"
    try:
        if mx.default_device().type != mx.DeviceType.gpu:
            return False, "MLX is not using a GPU device"
    except Exception as exc:  # pragma: no cover - platform dependent
        return False, f"MLX GPU probe failed: {exc}"
    return True, "MLX GPU device detected"


def _get_metal_swiglu_kernel(variant: str):
    kernel = _METAL_KERNELS.get(variant)
    if kernel is None:
        source = r"""
#pragma clang fp contract(off)
uint id = thread_position_in_grid.x;
T gate_value = gate[id];
T up_value = up[id];
"""
        if variant == "fp32":
            source += r"""
float gate_float = float(gate_value);
float sigmoid = 1.0f / (1.0f + metal::exp(-gate_float));
out[id] = static_cast<T>(float(up_value) * (gate_float * sigmoid));
"""
        elif variant == "precise_exp":
            source += r"""
float gate_float = float(gate_value);
float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate_float));
out[id] = static_cast<T>(float(up_value) * (gate_float * sigmoid));
"""
        elif variant == "mlx_header":
            source += r"""
T magnitude = metal::abs(gate_value);
T exponent = metal::exp(magnitude);
T denominator = T(1) + exponent;
T small_sigmoid = T(1) / denominator;
T sigmoid = (gate_value < T(0)) ? small_sigmoid : T(1) - small_sigmoid;
T silu = gate_value * sigmoid;
out[id] = silu * up_value;
"""
        elif variant == "bf16_silu_f32_product":
            source += r"""
float gate_float = float(gate_value);
float sigmoid = 1.0f / (1.0f + metal::exp(-gate_float));
T silu = static_cast<T>(gate_float * sigmoid);
out[id] = static_cast<T>(float(silu) * float(up_value));
"""
        else:
            raise ValueError(f"unknown Metal SwiGLU variant: {variant}")
        kernel = mx.fast.metal_kernel(
            name=f"flashnext_contract_swiglu_{variant}",
            input_names=["gate", "up"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
        _METAL_KERNELS[variant] = kernel
    return kernel


def _run_metal_swiglu(gate: mx.array, up: mx.array, variant: str) -> mx.array:
    """Run the planned Up-kernel SwiGLU epilogue in a Metal kernel.

    The QMV outputs already use bfloat16.  The epilogue reads those values,
    computes sigmoid and the product in float32, then stores bfloat16.
    """
    available, reason = metal_swiglu_available()
    if not available:
        raise RuntimeError(reason)
    gate = gate.astype(mx.bfloat16)
    up = up.astype(mx.bfloat16)
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    flat_gate = gate.reshape(-1)
    flat_up = up.reshape(-1)
    kernel = _get_metal_swiglu_kernel(variant)
    elements = flat_gate.size
    result = kernel(
        inputs=[flat_gate, flat_up],
        template=[("T", mx.bfloat16)],
        grid=(elements, 1, 1),
        threadgroup=(min(elements, 256), 1, 1),
        output_shapes=[flat_gate.shape],
        output_dtypes=[mx.bfloat16],
    )
    result = result[0] if isinstance(result, (tuple, list)) else result
    return result.reshape(gate.shape)


def metal_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Run the float32 Metal SwiGLU candidate."""
    return _run_metal_swiglu(gate, up, "fp32")


def metal_mlx_header_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Run the bfloat16 sequence used by MLX's Metal sigmoid helper."""
    return _run_metal_swiglu(gate, up, "mlx_header")


def metal_precise_exp_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Run the float32 candidate with ``metal::precise::exp``."""
    return _run_metal_swiglu(gate, up, "precise_exp")


def metal_bf16_silu_product_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Round SiLU to bfloat16 before a float32 product and final store."""
    return _run_metal_swiglu(gate, up, "bf16_silu_f32_product")


CANDIDATES: Mapping[str, Candidate] = {
    "bf16_silu_mul": _bf16_silu_mul,
    "fp32_silu_mul": _fp32_silu_mul,
    "fp32_silu_bf16_product": _fp32_silu_bf16_product,
    "bf16_sigmoid_mul": _bf16_sigmoid_mul,
    "fp32_sigmoid_mul": _fp32_sigmoid_mul,
}
METAL_CANDIDATES: Mapping[str, Candidate] = {
    METAL_CANDIDATE: metal_swiglu,
    METAL_HEADER_CANDIDATE: metal_mlx_header_swiglu,
    METAL_PRECISE_CANDIDATE: metal_precise_exp_swiglu,
    "metal_bf16_silu_f32_product": metal_bf16_silu_product_swiglu,
}
ALL_CANDIDATES: Mapping[str, Candidate] = {**CANDIDATES, **METAL_CANDIDATES}
DEFAULT_CANDIDATES = tuple(CANDIDATES)


def compare_swiglu(
    gate: mx.array,
    up: mx.array,
    *,
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> tuple[CandidateReport, ...]:
    """Compare candidate output bits against MLX output bits.

    ``gate`` and ``up`` must have equal shapes.  Inputs are converted to
    bfloat16 because that is the production QMV output contract.
    """
    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative")
    gate = gate.astype(mx.bfloat16)
    up = up.astype(mx.bfloat16)
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    names = tuple(candidates)
    if len(set(names)) != len(names):
        raise ValueError("candidates must not contain duplicates")
    unknown = tuple(name for name in names if name not in ALL_CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    expected = mlx_swiglu(gate, up)
    actual = [ALL_CANDIDATES[name](gate, up) for name in names]
    mx.eval([expected, *actual])
    expected_bits = bfloat16_bits(expected).reshape(-1)
    gate_bits = bfloat16_bits(gate).reshape(-1)
    reports = []
    for name, result in zip(names, actual):
        actual_bits = bfloat16_bits(result).reshape(-1)
        differing = np.flatnonzero(expected_bits != actual_bits)
        examples = tuple(
            SwiGLUMismatch(
                gate_bits=int(gate_bits[index]),
                expected_bits=int(expected_bits[index]),
                actual_bits=int(actual_bits[index]),
            )
            for index in differing[:max_mismatches]
        )
        reports.append(
            CandidateReport(
                name=name,
                total=int(expected_bits.size),
                mismatches=int(differing.size),
                examples=examples,
            )
        )
    return tuple(reports)


def exhaustive_gate_patterns(
    up_value: float = 1.0,
    *,
    up_bits: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> ExhaustiveReport:
    """Compare every bfloat16 gate encoding in bounded batches.

    ``up_value`` is converted to bfloat16 once and broadcast in each batch.
    Use ``up_bits`` when a precise bfloat16 input encoding is required.
    """
    if not 1 <= batch_size <= TOTAL_GATE_PATTERNS:
        raise ValueError(
            f"batch_size must be in 1..{TOTAL_GATE_PATTERNS}"
        )
    if up_bits is not None:
        if not 0 <= up_bits <= 0xFFFF:
            raise ValueError("up_bits must be a uint16 bfloat16 encoding")
        up = bfloat16_from_bits(np.array([up_bits], dtype=np.uint16)).reshape(())
    else:
        up = _bfloat16(up_value).reshape(())
    names = tuple(candidates)
    if len(set(names)) != len(names):
        raise ValueError("candidates must not contain duplicates")
    unknown = tuple(name for name in names if name not in ALL_CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    mismatch_counts = {name: 0 for name in names}
    examples: dict[str, list[SwiGLUMismatch]] = {name: [] for name in names}

    for start in range(0, TOTAL_GATE_PATTERNS, batch_size):
        stop = min(start + batch_size, TOTAL_GATE_PATTERNS)
        gate = bfloat16_from_bits(np.arange(start, stop, dtype=np.uint16))
        reports = compare_swiglu(
            gate,
            mx.broadcast_to(up, gate.shape),
            candidates=names,
            max_mismatches=max_mismatches,
        )
        for report in reports:
            mismatch_counts[report.name] += report.mismatches
            if len(examples[report.name]) >= max_mismatches:
                continue
            room = max_mismatches - len(examples[report.name])
            examples[report.name].extend(
                report.examples[:room]
            )

    return ExhaustiveReport(
        total=TOTAL_GATE_PATTERNS,
        batch_size=batch_size,
        up_bits=int(bfloat16_bits(up).reshape(-1)[0]),
        candidates=tuple(
            CandidateReport(
                name=name,
                total=TOTAL_GATE_PATTERNS,
                mismatches=mismatch_counts[name],
                examples=tuple(examples[name]),
            )
            for name in names
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up", type=float, default=1.0)
    parser.add_argument("--up-bits", type=lambda value: int(value, 0))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-mismatches", type=int, default=DEFAULT_MAX_MISMATCHES)
    parser.add_argument("--candidate", action="append", dest="candidates")
    args = parser.parse_args(argv)
    if set(args.candidates or ()) & set(METAL_CANDIDATES):
        available, reason = metal_swiglu_available()
        if not available:
            print(f"skip: {reason}")
            return 0
    report = exhaustive_gate_patterns(
        args.up,
        up_bits=args.up_bits,
        batch_size=args.batch_size,
        candidates=args.candidates or DEFAULT_CANDIDATES,
        max_mismatches=args.max_mismatches,
    )
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATES",
    "ALL_CANDIDATES",
    "CandidateReport",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CANDIDATES",
    "ExhaustiveReport",
    "METAL_CANDIDATE",
    "METAL_CANDIDATES",
    "METAL_HEADER_CANDIDATE",
    "METAL_PRECISE_CANDIDATE",
    "SwiGLUMismatch",
    "bfloat16_bits",
    "bfloat16_from_bits",
    "compare_swiglu",
    "exhaustive_gate_patterns",
    "mlx_swiglu",
    "metal_swiglu",
    "metal_mlx_header_swiglu",
    "metal_precise_exp_swiglu",
    "metal_bf16_silu_product_swiglu",
    "metal_swiglu_available",
]
