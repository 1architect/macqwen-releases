"""Single-model, teacher-forced Q4/G64 divergence diagnostic.

The probe runs the reference and custom MoE call back-to-back for the same
``StreamingSwitchGLU`` inputs. It keeps the custom result in the model graph,
so downstream layers see the candidate output. The reference result only
serves as an oracle for the current call.

This module is diagnostic code. It never changes the production readiness
constant or the default runtime environment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _materialize(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    # NumPy has no PEP 3118 format for MLX bfloat16. Convert only after the
    # device evaluation, preserving the MLX dtype separately in each record.
    if "bfloat16" in str(value.dtype):
        return np.asarray(value.astype(mx.float32))
    return np.asarray(value)


class _StopProbe(RuntimeError):
    """Internal control flow after the first decode mismatch."""


class G64DivergenceProbe:
    """Bounded per-call oracle for one loaded model."""

    def __init__(self, output_dir: str | os.PathLike[str], max_records: int = 48 * 32,
                 stop_on_divergence: bool = True):
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.output_dir = Path(output_dir)
        self.max_records = int(max_records)
        self.stop_on_divergence = bool(stop_on_divergence)
        self.decode_started = False
        self.records: list[dict[str, Any]] = []
        self.first_divergence: dict[str, Any] | None = None
        self.error: str | None = None
        self._original = None

    @staticmethod
    def _digest(value: np.ndarray) -> str:
        return hashlib.sha256(value.tobytes(order="C")).hexdigest()[:24]

    def _record(self, block, reference, candidate, inputs):
        if len(self.records) >= self.max_records:
            return
        reference_dtype = str(reference.dtype)
        candidate_dtype = str(candidate.dtype)
        inputs_dtype = str(inputs.dtype)
        ref = _materialize(reference)
        got = _materialize(candidate)
        x = _materialize(inputs)
        equal = ref.dtype == got.dtype and np.array_equal(ref, got)
        delta = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        mismatch = int(np.count_nonzero(ref != got))
        record = {
            "layer": int(getattr(block, "_flashnext_layer_id", -1)),
            "shape": list(got.shape),
            "dtype": candidate_dtype,
            "reference_dtype": reference_dtype,
            "inputs_dtype": inputs_dtype,
            "equal": bool(equal),
            "mismatch_count": mismatch,
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "reference_digest": self._digest(ref),
            "candidate_digest": self._digest(got),
        }
        self.records.append(record)
        if not equal and self.first_divergence is None:
            self.first_divergence = record
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.output_dir / "first_divergence.npz",
                     inputs=x, reference=ref, candidate=got)
            if self.stop_on_divergence:
                raise _StopProbe(
                    f"first G64 divergence at layer {record['layer']}"
                )

    def begin_decode(self) -> None:
        """Enable shadow calls after prompt prefill completes."""
        self.decode_started = True

    @contextmanager
    def installed(self, model):
        """Install a temporary class wrapper and restore it on exit."""
        blocks = [
            layer.mlp for layer in model.language.model.layers
            if hasattr(getattr(layer, "mlp", None), "switch_mlp")
        ]
        if not blocks:
            raise ValueError("model has no sparse MoE blocks")
        block_type = type(blocks[0])
        original = block_type.__call__
        self._original = original
        probe = self

        def shadow(block, inputs, *args, **kwargs):
            switch = getattr(block, "switch_mlp", None)
            is_g64 = getattr(getattr(switch, "gate_proj", None), "group_size", None) == 64
            if not probe.decode_started:
                previous = os.environ.get("FLASHNEXT_METAL_RUNTIME")
                os.environ["FLASHNEXT_METAL_RUNTIME"] = "0"
                try:
                    return original(block, inputs, *args, **kwargs)
                finally:
                    if previous is None:
                        os.environ.pop("FLASHNEXT_METAL_RUNTIME", None)
                    else:
                        os.environ["FLASHNEXT_METAL_RUNTIME"] = previous
            if not is_g64 or len(probe.records) >= probe.max_records:
                return original(block, inputs, *args, **kwargs)
            previous = os.environ.get("FLASHNEXT_METAL_RUNTIME")
            os.environ["FLASHNEXT_METAL_RUNTIME"] = "0"
            try:
                reference = original(block, inputs, *args, **kwargs)
            finally:
                if previous is None:
                    os.environ.pop("FLASHNEXT_METAL_RUNTIME", None)
                else:
                    os.environ["FLASHNEXT_METAL_RUNTIME"] = previous
            from models.flashnext import adaptive_topk
            observer = adaptive_topk._ROUTE_OBSERVER[0]
            adaptive_topk._ROUTE_OBSERVER[0] = None
            try:
                candidate = original(block, inputs, *args, **kwargs)
            finally:
                adaptive_topk._ROUTE_OBSERVER[0] = observer
            probe._record(block, reference, candidate, inputs)
            # Keep the reference trajectory for every later decode call.
            return reference

        block_type.__call__ = shadow
        try:
            yield self
        finally:
            block_type.__call__ = original
            self._original = None

    def write_report(self, checkpoint: str, prompt: str, tokens: int) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source = Path(__file__)
        report = {
            "diagnostic": "g64-divergence",
            "checkpoint": str(checkpoint),
            "prompt": prompt,
            "max_tokens": int(tokens),
            "max_records": self.max_records,
            "records": self.records,
            "first_divergence": self.first_divergence,
            "error": self.error,
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "environment": {
                key: os.environ.get(key)
                for key in ("FLASHNEXT_METAL_RUNTIME", "FLASHNEXT_METAL_G64")
            },
            "diagnostic_settings": {
                "FLASHNEXT_METAL_RUNTIME": "1",
                "FLASHNEXT_METAL_G64": "1",
                "FLASHNEXT_SLAB_G64": "0",
                "FLASHNEXT_SLAB_PACK": "1",
                "FLASHNEXT_SLAB": "0",
                "FLASHNEXT_SLAB_GLOBAL": "60",
                "shadow_boundary": "Qwen3_5MoeSparseMoeBlock",
                "return_trajectory": "reference",
            },
        }
        path = self.output_dir / "report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return path


def run_diagnostic(checkpoint: str, output_dir: str, prompt: str, tokens: int) -> G64DivergenceProbe:
    """Run a short greedy decode with one loaded reference model."""
    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext import metal_runtime

    # The readiness gate remains false in production. This local, scoped
    # diagnostic opt-in is required to exercise the rejected candidate.
    old_ready = metal_runtime.G64_RUNTIME_READY
    env_keys = (
        "FLASHNEXT_METAL_RUNTIME", "FLASHNEXT_METAL_G64", "FLASHNEXT_SLAB_G64",
        "FLASHNEXT_SLAB_PACK", "FLASHNEXT_SLAB", "FLASHNEXT_SLAB_GLOBAL",
    )
    old_env = {key: os.environ.get(key) for key in env_keys}
    metal_runtime.G64_RUNTIME_READY = True
    os.environ["FLASHNEXT_METAL_RUNTIME"] = "1"
    os.environ["FLASHNEXT_METAL_G64"] = "1"
    os.environ["FLASHNEXT_SLAB_G64"] = "0"
    os.environ["FLASHNEXT_SLAB_PACK"] = "1"
    os.environ["FLASHNEXT_SLAB"] = "0"
    os.environ["FLASHNEXT_SLAB_GLOBAL"] = "60"
    probe = G64DivergenceProbe(output_dir)
    backend = None
    failure = None
    try:
        backend = FlashNextBackend(model_path=checkpoint)
        backend.append_text(prompt)
        with probe.installed(backend):
            try:
                backend.generate(
                    max_tokens=min(int(tokens), 32),
                    on_prefilled=probe.begin_decode,
                )
            except _StopProbe:
                pass
    except BaseException as exc:
        failure = exc
        probe.error = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            try:
                backend.store.close()
            except (AttributeError, OSError):
                pass
            del backend
        metal_runtime.G64_RUNTIME_READY = old_ready
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    probe.write_report(checkpoint, prompt, min(int(tokens), 32))
    if failure is not None:
        raise failure
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--prompt", default=(
        "<|im_start|>user\nExplique a fotossintese em duas frases."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ))
    args = parser.parse_args()
    if not 1 <= args.tokens <= 32:
        parser.error("--tokens must be between 1 and 32")
    probe = run_diagnostic(args.model, args.output, args.prompt, args.tokens)
    status = "diverged" if probe.first_divergence else "matched"
    print(json.dumps({"status": status, "records": len(probe.records), "report": str(Path(args.output) / "report.json")}))


if __name__ == "__main__":
    main()
