from __future__ import annotations
import argparse
import cProfile
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import pstats
from pathlib import Path
from queue import Empty, Queue
import resource
import signal
import statistics as st
import subprocess
import sys
from threading import Thread
import time
import uuid
from types import SimpleNamespace
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
SHORT, PRODUCT, WINDOW, SEED = 32, 256, 32, 7
RESOURCE_POLICY = {
    # The screen is deliberately bounded; these are abort limits, not claims
    # about the machine's steady-state memory behavior.
    "max_prefill_seconds": 240.0,
    "max_generation_seconds": 600.0,
    "max_swap_pages": 65536,
    "max_pageout_pages": 65536,
}


class ResourceAbort(RuntimeError):
    """The current arm exceeded the benchmark's resource admission policy."""

    def __init__(self, phase: str, reason: str, details: dict[str, Any]):
        self.phase, self.reason, self.details = phase, reason, details
        super().__init__(f"{phase} resource limit: {reason}")


def resource_policy() -> dict[str, float | int]:
    policy = dict(RESOURCE_POLICY)
    for name, default in RESOURCE_POLICY.items():
        value = os.environ.get(f"MACQWEN_BONSAI2_{name.upper()}")
        if value is None:
            continue
        try:
            parsed = float(value) if isinstance(default, float) else int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            policy[name] = parsed
    return policy
COMPARISONS = {
    "baseline": {"control": {}},
    "profile": {"control": {}, "cprofile": {"profile": True}},
    "allocator": {"control": {}, "allocator-256": {"allocator_cache_mb": 256}},
    "clear-cache": {"control": {}, "clear-cache-after": {"clear_cache_after_generate": True}},
    "wired": {"control": {}, "wired-limit": {"wired_limit_enabled": True}},
    "cache-step": {"step-256": {"cache_step": 256}, "step-1024": {"cache_step": 1024}},
    "prefill": {"prefill-512": {"prefill_step_size": 512}, "prefill-256": {"prefill_step_size": 256}},
    "prefill-wide": {"prefill-512": {"prefill_step_size": 512}, "prefill-1024": {"prefill_step_size": 1024}, "prefill-2048": {"prefill_step_size": 2048}},
    "fused-fwht": {"control": {"fused_fwht": False}, "fused": {"fused_fwht": True}},
    "quant-kv8": {"control": {}, "qkv8": {"quantized_kv": [8, 64]}},
    "quant-kv4": {"control": {}, "qkv4": {"quantized_kv": [4, 64]}},
    "q4-attention": {
        "q4-untiled": {
            "quantized_kv": [4, 64], "allocator_cache_mb": 256,
            "prefill_step_size": 512, "q4_attention_tiling": False,
            "trace_memory": True,
        },
        "q4-tiled": {
            "quantized_kv": [4, 64], "allocator_cache_mb": 256,
            "prefill_step_size": 512, "q4_attention_tiling": True,
            "trace_memory": True,
        },
    },
    "q4-attention-fused": {
        "q4-stock": {
            "quantized_kv": [4, 64], "allocator_cache_mb": 256,
            "prefill_step_size": 512, "q4_attention_tiling": False,
            "fused_q4_attention": False, "trace_memory": False,
        },
        "q4-fused": {
            "quantized_kv": [4, 64], "allocator_cache_mb": 256,
            "prefill_step_size": 512, "q4_attention_tiling": False,
            "fused_q4_attention": True, "trace_memory": False,
        },
    },
    "q2-prefill-mpp": {
        "q2-stock": {
            "q2_prefill_mpp": False, "q4_attention_tiling": False,
            "fused_q4_attention": False, "trace_memory": False,
        },
        "q2-mpp": {
            "q2_prefill_mpp": True, "q4_attention_tiling": False,
            "fused_q4_attention": False, "trace_memory": False,
        },
    },
    "prepared-qmm-metadata": {
        "qmm-stock": {
            "prepared_qmm_metadata": False, "q4_attention_tiling": False,
            "fused_q4_attention": False, "trace_memory": False,
        },
        "qmm-prepared": {
            "prepared_qmm_metadata": True, "q4_attention_tiling": False,
            "fused_q4_attention": False, "trace_memory": False,
        },
    },
    "exact-speculative-oracle-2": {
        "spec-stock": {"exact_speculative_decode": False},
        "spec-oracle-2": {
            "exact_speculative_decode": True, "speculative_block_size": 2,
        },
    },
    "exact-speculative-oracle-4": {
        "spec-stock": {"exact_speculative_decode": False},
        "spec-oracle-4": {
            "exact_speculative_decode": True, "speculative_block_size": 4,
        },
    },
    "exact-speculative-oracle-8": {
        "spec-stock": {"exact_speculative_decode": False},
        "spec-oracle-8": {
            "exact_speculative_decode": True, "speculative_block_size": 8,
        },
    },
    "share-fwht": {"control": {}, "shared": {"share_fwht": True}},}
DIAGNOSTIC_COMPARISONS = {"profile"}
PRODUCTION_COMPARISONS = {
    "baseline", "q4-attention", "q4-attention-fused", "prepared-qmm-metadata",
}
_ANALYSIS_REQUEST = "Using the numbered records, write a detailed neutral analysis of at least 300 words covering the observed patterns and exceptions."
def _context_fixture(records: int) -> tuple[str, str, None, None]:
    context = "\n".join(f"Record {i:04d}: category {i % 16}; value {(i * 37) % 100}; status {('stable', 'review')[i % 2]}; note segment {i % 7}." for i in range(1, records + 1))
    return "Answer from the records.", f"{context}\n\n{_ANALYSIS_REQUEST}", None, None
FIXTURES = {
    "context-1k": _context_fixture(32),
    "context-2k": _context_fixture(128), "context-8k": _context_fixture(384),
    "context-16k": _context_fixture(768),
    "smoke": (
        "Answer from the records.",
        "Record 0001: category 5; value 42; status stable.\n\n"
        "Summarize the single record in one sentence.",
        None, None,
    ),
    # Keep the shortest live speed arm aligned with FlashNext's reference
    # question; the empty system content adds only the model's chat framing.
    "question-only": ("", "Explique a fotossintese em duas frases.", None, None),
    "cached-tool-result": ("Use tool results as context.", _ANALYSIS_REQUEST, '{"setting":"example","value":42}', None),
    "repeated-turn": ("Keep prior context.", _ANALYSIS_REQUEST, None, "Now repeat the recorded value with a detailed analysis of at least 300 words."),
}
DEFAULT_FIXTURE = "context-1k"


def operating_profile(comparison: str) -> str:
    return "interactive-production" if comparison in PRODUCTION_COMPARISONS else "diagnostic"
def append_jsonl(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    try:
        lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except (TypeError, ValueError):
            pass
    return rows
def sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


_CHECKPOINT_FILES = {
    "config.json": True,
    # A monolithic safetensors file does not need the index or model shim.
    "model.py": False,
    "model.safetensors.index.json": False,
    "tokenizer.json": True,
    "tokenizer_config.json": True,
    "chat_template.jinja": True,
    "hadamard.json": True,
    "runtime/runtime.py": True,
    "runtime/artifact.py": True,
    "runtime/vision_artifact.py": True,
    "runtime/codec.py": True,
}


def _checkpoint_file_identity(item: Path, required: bool) -> dict[str, Any]:
    try:
        if not item.is_file():
            return {
                "present": False,
                "required": required,
                "state": "unknown" if required else "absent",
                "reason": (
                    "missing_required_identity"
                    if required else "optional_file_absent"
                ),
            }
        size = item.stat().st_size
        digest = sha256(item)
        if digest is None:
            return {
                "present": True,
                "required": required,
                "state": "unknown",
                "reason": "unreadable",
                "bytes": size,
            }
        return {
            "present": True,
            "required": required,
            "state": "present",
            "bytes": size,
            "sha256": digest,
        }
    except OSError:
        return {
            "present": False,
            "required": required,
            "state": "unknown",
            "reason": "unreadable_path",
        }


def checkpoint_info(requested: str) -> dict[str, Any]:
    try:
        from .checkpoint import resolve_bonsai2
        path = resolve_bonsai2(requested)
    except (OSError, TypeError, ValueError):
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = Path(os.environ.get("MACQWEN_MODEL_ROOT", "~/models")) / path
        path = path.resolve()
    files = {}
    for name, required in _CHECKPOINT_FILES.items():
        files[name] = _checkpoint_file_identity(path / name, required)
    # The index is optional for a monolithic checkpoint.  In that case the
    # single weight file is the required model identity; sharded checkpoints
    # identify their required files through the index and shard manifest.
    if not (path / "model.safetensors.index.json").is_file():
        weight = path / "model.safetensors"
        try:
            if weight.is_file():
                # Do not reread multi-gigabyte weights on every arm.  The
                # required identity here is the confirmed file size; the
                # small config/runtime identities carry the content hashes.
                files["model.safetensors"] = {
                    "present": True, "required": True, "state": "present",
                    "bytes": weight.stat().st_size,
                }
            else:
                files["model.safetensors"] = _checkpoint_file_identity(weight, True)
        except OSError:
            files["model.safetensors"] = _checkpoint_file_identity(weight, True)
    shards = []
    try:
        shards = [{"name": item.name, "bytes": item.stat().st_size}
                  for item in sorted(path.glob("*.safetensors"))]
    except OSError:
        pass
    identity = {"path": str(path), "files": files, "shards": shards}
    return {"requested": requested, **identity,
            "identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
def source_fingerprints() -> dict[str, str | None]:
    names = (
        "models/bonsai2/backend.py", "models/bonsai2/bench.py",
        "models/bonsai2/cache.py", "models/bonsai2/checkpoint.py",
        "models/bonsai2/protocol.py", "models/bonsai2/settings.py",
        "models/bonsai2/ternary_kernel.py", "models/bonsai2/q2_kernel.py",
        "models/bonsai2/q4_attention_kernel.py", "models/bonsai2/qmm_metadata.py",
        "macqwen/backends/base.py", "macqwen/conversation.py",
        "macqwen/sampling.py", "macqwen/text.py", "macqwen/agent.py",
    )
    return {name: sha256(ROOT / name) for name in names}


_DEPENDENCY_PACKAGES = ("mlx", "mlx_lm", "mlx_vlm", "transformers")
_DEPENDENCY_MODULES = (
    "mlx.core",
    "mlx_lm.generate",
    "mlx_lm.models.cache",
    "mlx_vlm.models.cache",
    "mlx_vlm.models.qwen3_5.speculative_verifier",
    "mlx_vlm.speculative.mtp",
)


def _module_origin(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin and spec.origin not in ("built-in", "frozen"):
            origin = Path(spec.origin)
            if origin.is_file():
                return origin
        parts = name.split(".")
        package = importlib.util.find_spec(parts[0])
        roots = (
            [Path(path) for path in package.submodule_search_locations]
            if package and package.submodule_search_locations
            else []
        )
        relative = Path(*parts[1:])
        candidates = [
            candidate
            for root in roots
            for candidate in (
                (root / relative).with_suffix(".py"),
                root / relative / "__init__.py",
            )
        ]
        if parts[-1] != parts[0]:
            candidates += [
                candidate
                for root in roots
                for candidate in (root / relative.parent).glob(relative.name + ".*")
            ]
        return next((candidate for candidate in candidates if candidate.is_file()), None)
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return None


def _module_root(name: str) -> Path | None:
    """Return a package root even when the package is namespace-only."""
    try:
        spec = importlib.util.find_spec(name)
        locations = getattr(spec, "submodule_search_locations", None) if spec else None
        if locations:
            return Path(next(iter(locations)))
        origin = getattr(spec, "origin", None) if spec else None
        if origin and origin not in ("built-in", "frozen"):
            path = Path(origin)
            if path.is_file():
                return path.parent
    except (ImportError, OSError, StopIteration, TypeError, ValueError):
        pass
    origin = _module_origin(name)
    return origin.parent if origin else None


def dependency_info() -> dict[str, Any]:
    result = {}
    for name in _DEPENDENCY_PACKAGES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            origin = _module_origin(name)
            root = _module_root(name)
            spec = importlib.util.find_spec(name)
            namespace = bool(
                spec is not None
                and getattr(spec, "origin", None) in (None, "built-in", "frozen")
                and getattr(spec, "submodule_search_locations", None)
            )
            source = sha256(origin) if origin and origin.is_file() else None
        except (ImportError, OSError, TypeError, ValueError):
            root, source, namespace = None, None, False
        if namespace:
            # Namespace packages intentionally have no __init__.py to hash.
            # Their search root and explicit kind are the confirmed identity;
            # the concrete binary submodule (mlx.core) is fingerprinted below.
            result[name] = {
                "version": version,
                "package_kind": "namespace",
                "root": str(root) if root else "",
            }
        else:
            result[name] = {
                "version": version,
                "package_kind": "package",
                "root": str(root) if root else None,
                "source_sha256": source,
            }
    for name in _DEPENDENCY_MODULES:
        origin = _module_origin(name)
        result[name] = {"origin": str(origin) if origin else None,
                        "source_sha256": sha256(origin) if origin else None}
    return result


def provenance_manifest(checkpoint: str) -> dict[str, Any]:
    """Capture the identities that make two benchmark arms comparable."""
    return {
        "checkpoint": checkpoint_info(checkpoint),
        "runtime_source_fingerprints": source_fingerprints(),
        "dependency_info": dependency_info(),
        "harness_fingerprint": sha256(Path(__file__).resolve()),
    }


def _provenance_paths(expected: Any, observed: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    changed, unknown = [], []
    if isinstance(expected, dict) and isinstance(observed, dict):
        expected_state = expected.get("state")
        observed_state = observed.get("state")
        if expected_state is not None or observed_state is not None:
            if expected_state == observed_state == "absent":
                return [], []
            if "unknown" in (expected_state, observed_state):
                return [], [prefix]
            if expected_state != observed_state:
                return [prefix], []
        if "present" in expected or "present" in observed:
            expected_present = expected.get("present")
            observed_present = observed.get("present")
            required = bool(expected.get("required", observed.get("required", True)))
            if expected_present is False and observed_present is False:
                if not required:
                    return [], []
                return [], [f"{prefix}.present"]
            if expected_present != observed_present:
                return (
                    ([f"{prefix}.present"], [])
                    if not required else ([], [f"{prefix}.present"])
                )
        for key in sorted(set(expected) | set(observed)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in observed:
                unknown.append(path)
                continue
            left, right = expected[key], observed[key]
            if left is None or right is None:
                unknown.append(path)
            else:
                child_changed, child_unknown = _provenance_paths(left, right, path)
                changed.extend(child_changed)
                unknown.extend(child_unknown)
        return changed, unknown
    if expected is None or observed is None:
        unknown.append(prefix)
    elif expected != observed:
        changed.append(prefix)
    return changed, unknown


def validate_provenance(
    expected: dict[str, Any], before: dict[str, Any] | None,
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    changed, unknown = _provenance_paths(expected, before, "before")
    if after is not None and before is not None:
        later_changed, later_unknown = _provenance_paths(before, after, "during")
        changed.extend(later_changed)
        unknown.extend(later_unknown)
    status = "changed" if changed else "unknown" if unknown else "matched"
    return {"status": status, "changed": sorted(set(changed)),
            "unknown": sorted(set(unknown))}
def metadata(checkpoint: str, comparison: str, fixture: str, horizon: int,
             window: int, thinking: bool, effort: str, sampling: str,
             prefill_step_size: int, seed: int = SEED,
             experiment_id: str | None = None) -> dict[str, Any]:
    provenance = provenance_manifest(checkpoint)
    return {
        "schema": 1, "comparison": comparison, "checkpoint": provenance["checkpoint"],
        "experiment_id": experiment_id,
        "operating_profile": operating_profile(comparison),
        "fixture": fixture, "horizon_tokens": horizon, "window_tokens": window,
        "reasoning": {"enabled": thinking, "effort": effort},
        "sampling_control": sampling,
        "sampled_chat": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
        "greedy_control": {"temperature": 0.0, "digest_required": sampling == "greedy"},
        "prefill_step_size": prefill_step_size, "seed": seed, "interpreter": sys.executable,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "runtime_source_fingerprints": provenance["runtime_source_fingerprints"],
        "dependency_info": provenance["dependency_info"],
        "harness_fingerprint": provenance["harness_fingerprint"],
        "provenance_manifest": provenance,
        "resource_policy": resource_policy(),
    }
def _memory() -> dict[str, int | None]:
    rss = peak_rss = None
    try:
        if sys.platform.startswith("linux"):
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGESIZE")
        elif sys.platform == "darwin":
            peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        else:
            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss = value * 1024
    except (OSError, IndexError, TypeError, ValueError):
        pass
    footprint = None
    if sys.platform == "darwin":
        try:
            import ctypes
            buffer = (ctypes.c_uint8 * 512)()
            lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
            if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer)) == 0:
                footprint = int.from_bytes(bytes(buffer[72:80]), "little")
                rss = int.from_bytes(bytes(buffer[64:72]), "little")
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return {"rss_bytes": rss, "peak_rss_bytes": peak_rss, "physical_footprint_bytes": footprint}
def _mlx_memory() -> dict[str, int | None]:
    result = {"active_bytes": None, "cache_bytes": None, "peak_bytes": None}
    try:
        import mlx.core as mx
        for key, name in (("active_bytes", "get_active_memory"),
                          ("cache_bytes", "get_cache_memory"),
                          ("peak_bytes", "get_peak_memory")):
            function = getattr(mx, name, None)
            if function:
                result[key] = int(function())
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    return result
def _reset_mlx_peak() -> None:
    try:
        import mlx.core as mx
        reset = getattr(mx, "reset_peak_memory", None)
        if reset is not None:
            reset()
    except (ImportError, AttributeError, RuntimeError):
        pass
def _disk() -> int | None:
    try:
        from models.flashnext.diskio import disk_bytes_read
        value = int(disk_bytes_read())
        return value if value >= 0 else None
    except (ImportError, OSError, TypeError, ValueError):
        return None
def _vm() -> dict[str, int]:
    try:
        from models.flashnext.diskio import vm_counters
        return {str(k): int(v) for k, v in vm_counters().items()}
    except (ImportError, OSError, TypeError, ValueError):
        return {}
def _shape(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(x) for x in value.shape)
    except (AttributeError, TypeError, ValueError):
        return None


def _cache_array(value: Any) -> Any:
    """Return the shaped storage array for full or quantized KV storage."""
    if _shape(value) is not None:
        return value
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    return None


def cache_metrics(backend: Any) -> dict[str, Any]:
    try:
        caches = list(getattr(backend, "cache", ()) or ())
    except TypeError:
        caches = []
    offsets, capacities, dtypes, bits, group_sizes = [], [], [], [], []
    payload = allocated = capacity_bytes = 0
    for cache in caches:
        try:
            offset = int(cache.offset)
        except (AttributeError, TypeError, ValueError):
            offset = None
        keys, values = getattr(cache, "keys", None), getattr(cache, "values", None)
        key_array, value_array = _cache_array(keys), _cache_array(values)
        key_shape, value_shape = _shape(key_array), _shape(value_array)
        capacity = key_shape[-2] if key_shape and len(key_shape) >= 2 else None
        offsets.append(offset); capacities.append(capacity)
        dtypes.append(str(getattr(key_array, "dtype", "")) or None)
        bits.append(getattr(cache, "bits", None))
        group_sizes.append(getattr(cache, "group_size", None))
        try:
            nbytes = int(cache.nbytes)
            allocated += nbytes
        except (AttributeError, TypeError, ValueError, RuntimeError):
            nbytes = None
        if nbytes is not None and capacity:
            capacity_bytes += nbytes
            payload += nbytes * min(max(offset or 0, 0), capacity) // capacity
            continue
        if offset is None or not key_shape or not value_shape:
            continue
        try:
            key_size = int(key_array.dtype.itemsize)
            value_size = int(value_array.dtype.itemsize)
        except (AttributeError, TypeError, ValueError):
            continue
        per_token = (math.prod(key_shape[:-2]) * key_shape[-1] * key_size +
                     math.prod(value_shape[:-2]) * value_shape[-1] * value_size)
        payload += offset * per_token
        if capacity is not None:
            capacity_bytes += capacity * per_token
    return {"count": len(caches), "offsets": offsets, "capacities": capacities,
            "dtypes": dtypes, "bits": bits, "group_sizes": group_sizes,
            "allocated_bytes": allocated, "capacity_bytes": capacity_bytes,
            "payload_bytes": payload}
def snapshot(backend: Any, phase: str, *, os_probes: bool = True) -> dict[str, Any]:
    mlx, cache, process = _mlx_memory(), cache_metrics(backend), _memory()
    result = {"phase": phase, "time_ns": time.time_ns(), "mlx": mlx,
              "cache": cache, "process": process, "physical_read_bytes": None,
              "vm_counters": {}}
    if os_probes:
        result["physical_read_bytes"], result["vm_counters"] = _disk(), _vm()
    return result
def _digest(tokens: list[int]) -> str:
    data = b"".join(int(x).to_bytes(4, "little", signed=False) for x in tokens)
    return hashlib.sha256(data).hexdigest()
def _windows(arrivals: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    result = []
    for start in range(0, len(arrivals), window):
        group = arrivals[start:start + window]
        if not group:
            continue
        clock_name = "decode_at_s" if "decode_at_s" in group[-1] else "at_s"
        intervals = [group[i][clock_name] - group[i - 1][clock_name] for i in range(1, len(group))]
        boundary = arrivals[start - 1][clock_name] if start else 0.0
        elapsed = group[-1][clock_name] - boundary
        ordered = sorted(intervals)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else None
        median = st.median(intervals) if intervals else None
        result.append({"window": start // window + 1, "start_token": start + 1, "end_token": start + len(group),
                       "tokens": len(group), "is_product_tail": start + 1 >= 33,
                       "rate_tps": len(group) / elapsed if elapsed > 0 else 0.0, "boundary_elapsed_s": elapsed,
                       "first_token_latency_s": group[0]["at_s"],
                       "decode_relative_first_token_latency_s": group[0].get(
                           "decode_at_s", group[0]["at_s"]
                       ),
                       "arrival_intervals_s": intervals,
                       "interval_median_s": median, "interval_p95_s": p95,
                       "ms_per_token_median": 1000.0 * median if median else None})
    return result
def _vm_delta(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(right.get(key, 0)) - int(left.get(key, 0)) for key in sorted(set(left) | set(right))}


def _resource_failure(
    started: float,
    baseline: dict[str, Any],
    current: dict[str, Any],
    phase: str,
    policy: dict[str, float | int],
    *,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Return a measured breach, or ``None`` when the policy is satisfied."""
    elapsed = (time.perf_counter() if now is None else now) - started
    limit_key = "max_prefill_seconds" if phase == "prefill" else "max_generation_seconds"
    limit = float(policy.get(limit_key, float("inf")))
    if elapsed > limit:
        return {
            "phase": phase,
            "reason": "deadline",
            "elapsed_seconds": elapsed,
            "limit_seconds": limit,
        }
    before_vm = baseline.get("vm_counters", {}) if isinstance(baseline, dict) else {}
    after_vm = current.get("vm_counters", {}) if isinstance(current, dict) else {}
    before_vm = before_vm if isinstance(before_vm, dict) else {}
    after_vm = after_vm if isinstance(after_vm, dict) else {}
    vm_delta = _vm_delta(before_vm, after_vm)
    swap_pages = max(vm_delta.get("swapin", 0), vm_delta.get("swapout", 0))
    pageout_pages = vm_delta.get("pageout", 0)
    if swap_pages > int(policy.get("max_swap_pages", 2**63 - 1)):
        return {
            "phase": phase,
            "reason": "swap_pressure",
            "swap_pages": swap_pages,
            "limit_pages": int(policy.get("max_swap_pages", 0)),
            "vm_delta": vm_delta,
        }
    if pageout_pages > int(policy.get("max_pageout_pages", 2**63 - 1)):
        return {
            "phase": phase,
            "reason": "pageout_pressure",
            "pageout_pages": pageout_pages,
            "limit_pages": int(policy.get("max_pageout_pages", 0)),
            "vm_delta": vm_delta,
        }
    return None
def _attention_events(backend: Any) -> list[dict[str, Any]]:
    events = getattr(backend, "attention_events", None)
    return list(events) if isinstance(events, (list, tuple)) else []


def _attention_counters(backend: Any) -> dict[str, Any]:
    counters = getattr(backend, "attention_counters", None)
    if not isinstance(counters, dict):
        return {}
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in counters.items()
    }


def _q2_counters(backend: Any) -> dict[str, Any]:
    counters = getattr(backend, "q2_counters", None)
    if not isinstance(counters, dict):
        return {}
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in counters.items()
    }


def _qmm_metadata_counters(backend: Any) -> dict[str, Any]:
    counters = getattr(backend, "qmm_metadata_counters", None)
    return dict(counters) if isinstance(counters, dict) else {}


def _qmm_metadata_stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "qmm_metadata_stats", None)
    return dict(stats) if isinstance(stats, dict) else {}


def _speculative_stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "speculative_stats", None)
    if not isinstance(stats, dict):
        return {}
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in stats.items()
    }


def _fused_fwht_stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "fused_fwht_counters", None)
    if not isinstance(stats, dict):
        return {}
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in stats.items()
    }


def _stop_token_sync_stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "stop_token_sync_stats", None)
    return dict(stats) if isinstance(stats, dict) else {}


def _progress(arm_id: str, phase: str, **values) -> None:
    payload = {"arm": arm_id, "pid": os.getpid(), "phase": phase, **values}
    print("BonsaiProgress " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def _progress_observations(output: str) -> list[dict[str, Any]]:
    """Recover durable-enough progress when a child dies before its arm row."""
    observations = []
    for line in str(output or "").splitlines():
        if not line.startswith("BonsaiProgress "):
            continue
        try:
            value = json.loads(line[len("BonsaiProgress "):])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            observations.append(value)
    return observations


_ARM_STOP_GRACE_SECONDS = 1.0
_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.25
_PARENT_TIMEOUT_ENV = "MACQWEN_BONSAI2_PARENT_TIMEOUT_SECONDS"


def _parent_timeout_seconds(command: list[str]) -> float:
    override = os.environ.get(_PARENT_TIMEOUT_ENV)
    if override is not None:
        try:
            value = float(override)
            if value >= 0:
                return value
        except (TypeError, ValueError):
            pass
    try:
        policy = json.loads(command[command.index("--resource-policy-json") + 1])
        return max(0.0, float(policy["max_prefill_seconds"])) + max(
            0.0, float(policy["max_generation_seconds"])
        ) + _ARM_STOP_GRACE_SECONDS
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        policy = resource_policy()
        return float(policy["max_prefill_seconds"]) + float(
            policy["max_generation_seconds"]
        ) + _ARM_STOP_GRACE_SECONDS


def _stop_worker(process) -> int:
    """Bounded cleanup for a worker when the benchmark is run directly."""
    send_signal = getattr(process, "send_signal", None)
    if not callable(send_signal):
        poll = getattr(process, "poll", None)
        value = poll() if callable(poll) else None
        return int(value) if value is not None else 128 + int(signal.SIGINT)
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            send_signal(signal_number)
        except (OSError, ProcessLookupError, PermissionError):
            pass
        try:
            return int(process.wait(timeout=_ARM_STOP_GRACE_SECONDS))
        except subprocess.TimeoutExpired:
            continue
    try:
        send_signal(signal.SIGKILL)
    except (OSError, ProcessLookupError, PermissionError):
        pass
    try:
        return int(process.wait(timeout=_ARM_STOP_GRACE_SECONDS))
    except subprocess.TimeoutExpired:
        poll = getattr(process, "poll", None)
        value = poll() if callable(poll) else None
        return int(value) if value is not None else 128 + int(signal.SIGKILL)


def _stream_child(
    command: list[str], env: dict[str, str], arm_id: str,
    timeout_seconds: float | None = None,
):
    """Run one child in the controller's process group.

    The outer live-test runner owns that group.  Do not create a second
    session here: a second session is exactly how an interrupted worker used
    to survive its controller.
    """
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    output = Queue()

    def read_output():
        try:
            if process.stdout is not None:
                for raw in process.stdout:
                    output.put(("line", raw))
        except BaseException as error:
            output.put(("error", error))
        finally:
            output.put(("eof", None))

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    timeout_seconds = (
        _parent_timeout_seconds(command)
        if timeout_seconds is None else max(0.0, float(timeout_seconds))
    )
    deadline = time.perf_counter() + timeout_seconds
    timed_out = False
    returncode = None
    try:
        print(f"BonsaiArm arm={arm_id} pid={process.pid} phase=started", flush=True)
        while True:
            remaining = deadline - time.perf_counter()
            try:
                kind, value = output.get(timeout=max(0.0, remaining))
            except Empty:
                if getattr(process, "poll", lambda: None)() is not None:
                    break
                timed_out = True
                returncode = _stop_worker(process)
                lines.append(
                    f"BonsaiParentTimeout arm={arm_id} seconds={timeout_seconds:.3f}"
                )
                break
            if kind == "line":
                line = value.rstrip("\n")
                lines.append(line)
                print(line, flush=True)
            elif kind == "error":
                raise value
            else:
                break
        if returncode is None:
            value = getattr(process, "poll", lambda: None)()
            if value is not None:
                returncode = int(value)
            else:
                remaining = max(0.0, deadline - time.perf_counter())
                try:
                    returncode = int(process.wait(timeout=remaining))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = _stop_worker(process)
                    lines.append(
                        f"BonsaiParentTimeout arm={arm_id} seconds={timeout_seconds:.3f}"
                    )
    except KeyboardInterrupt:
        # The outer runner owns the full group, but direct CLI callers still
        # need the worker itself to stop before this controller returns.
        returncode = _stop_worker(process)
        lines.append("KeyboardInterrupt")
    except BaseException:
        if getattr(process, "poll", lambda: None)() is None:
            _stop_worker(process)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        reader.join(timeout=_ARM_STOP_GRACE_SECONDS)
    return SimpleNamespace(
        returncode=int(returncode), stderr="\n".join(lines), timed_out=timed_out
    )
class GenerationProfile:
    def __init__(self):
        self.profiles = {phase: cProfile.Profile() for phase in ("prefill", "decode")}
        self.phase = "prefill"

    def start(self):
        self.profiles[self.phase].enable()

    def start_decode(self):
        self.stop()
        self.phase = "decode"
        self.start()

    def stop(self):
        self.profiles[self.phase].disable()

    def save(self, record_path: str, arm_id: str) -> dict[str, Any]:
        self.stop()
        phases = {}
        target = Path(record_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        invocation = uuid.uuid4().hex
        for phase, profiler in self.profiles.items():
            path = target.with_name(f"{target.stem}-{arm_id}-{invocation}-{phase}.pstats")
            profiler.create_stats()
            if not profiler.stats:
                phases[phase] = {"status": "empty", "total_calls": 0, "total_s": 0.0}
                continue
            profiler.dump_stats(str(path))
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(pstats.SortKey.TIME).print_stats(30)
            phases[phase] = {"status": "captured", "pstats_path": str(path), "top30": stream.getvalue(),
                             "total_calls": stats.total_calls, "total_s": stats.total_tt}
        return {"scope": "backend.generate; excludes load and terminal UI",
                "clock": "wall; native calls include GPU waits, not GPU kernel timing",
                "boundary": "prefill includes completion snapshot and decode lookahead submission; decode includes cleanup",
                "phases": phases}


def _constructor_options(
    options: dict[str, Any], default_prefill_step: int,
    profile: str = "diagnostic",
) -> tuple[dict[str, Any], int]:
    constructor = {k: v for k, v in options.items() if k not in ("cache_step", "profile")}
    if profile == "interactive-production":
        from .backend import PRODUCTION_ALLOCATOR_CACHE_MB

        constructor["allocator_cache_mb"] = PRODUCTION_ALLOCATOR_CACHE_MB
    return constructor, int(constructor.pop("prefill_step_size", default_prefill_step))


def _oracle_tokens_for_arm(
    record_path: str, round_index: int, horizon: int
) -> tuple[list[int] | None, str | None]:
    """Load the latest same-run stock transcript for an oracle arm."""
    candidates = []
    for row in read_jsonl(record_path):
        if (
            row.get("type") == "arm"
            and row.get("condition") == "spec-stock"
            and row.get("status") == "raw"
            and int(row.get("round", -1)) <= int(round_index)
            and isinstance(row.get("tokens"), list)
            and len(row["tokens"]) >= int(horizon)
        ):
            try:
                candidates.append((int(row.get("round", -1)), row))
            except (TypeError, ValueError):
                continue
    if not candidates:
        return None, None
    _round, row = max(candidates, key=lambda item: item[0])
    return [int(value) for value in row["tokens"][:horizon]], str(row.get("arm_id"))


def child_arm(*, checkpoint: str, arm_id: str, condition: str, options: dict[str, Any],
              record_path: str, fixture: str, horizon: int, window: int,
              thinking: bool, effort: str, sampling: str, prefill_step_size: int,
              round_index: int, seed: int = SEED,
              resource_limits: dict[str, float | int] | None = None,
              operating_profile_name: str = "diagnostic",
              expected_provenance: dict[str, Any] | None = None) -> int:
    started, backend, arrivals, raw_written = time.perf_counter(), None, [], False
    policy = dict(resource_limits or resource_policy())
    profiler = GenerationProfile() if options.get("profile") else None
    record = {"type": "arm", "schema": 1, "arm_id": arm_id, "condition": condition,
              "round": round_index, "status": "failed", "tokens": [],
              "token_digest": _digest([]), "error": None, "snapshots": {}, "windows": [],
              "seed": seed, "resource_policy": policy, "checkpoint": checkpoint}
    record["qmm_projection_coverage"] = _expected_qmm_projection_coverage(checkpoint)
    try:
        provenance_before = provenance_manifest(checkpoint)
    except BaseException as error:
        provenance_before = {"error": {"type": type(error).__name__, "message": str(error)}}
    record["provenance_before"] = provenance_before
    if expected_provenance is not None:
        before_check = validate_provenance(expected_provenance, provenance_before)
        record["provenance_validation"] = before_check
        if before_check["status"] != "matched":
            error_type = (
                "ProvenanceChanged"
                if before_check["status"] == "changed"
                else "ProvenanceUnknown"
            )
            record["error"] = {
                "type": error_type,
                "message": "arm provenance is not comparable before execution",
                "details": before_check,
            }
            append_jsonl(record_path, record)
            return 1
    _progress(arm_id, "loading")
    try:
        from macqwen.sampling import Sampling
        from .backend import (
            BonsaiBackend, PRODUCTION_CANCELLABLE_PREFILL_STEP_SIZE,
        )
        system, user, tool_result, repeat_user = FIXTURES[fixture]
        constructor, effective_prefill_step = _constructor_options(
            options, prefill_step_size, operating_profile_name
        )
        record["prefill_step_size"] = effective_prefill_step
        record["operating_profile"] = operating_profile_name
        record["production_profile"] = {
            "allocator_cache_mb": constructor.get("allocator_cache_mb"),
            "configured_prefill_step_size": effective_prefill_step,
            "cancellation_callback": operating_profile_name == "interactive-production",
            "effective_prefill_step_size": (
                min(
                    effective_prefill_step,
                    PRODUCTION_CANCELLABLE_PREFILL_STEP_SIZE,
                )
                if operating_profile_name == "interactive-production" else effective_prefill_step
            ),
        }
        backend = BonsaiBackend(checkpoint, prefill_step_size=effective_prefill_step, **constructor)
        _progress(
            arm_id,
            "loaded",
            prepared_qmm=bool(getattr(backend, "prepared_qmm_metadata", False)),
            prepared_modules=_qmm_metadata_stats(backend).get("prepared_modules", 0),
        )
        effective_kv = getattr(backend, "quantized_kv", None)
        record["effective_quantized_kv"] = (
            list(effective_kv)
            if isinstance(effective_kv, (tuple, list)) else None
        )
        record["runtime_settings"] = (
            backend.runtime_settings()
            if hasattr(backend, "runtime_settings")
            else {"prefill_step_size": effective_prefill_step, **constructor}
        )
        record["qmm_metadata"] = _qmm_metadata_stats(backend)
        record["qmm_metadata"]["expected_projection_manifest"] = record[
            "qmm_projection_coverage"
        ].get("expected", [])
        record["qmm_metadata"]["excluded_projection_manifest"] = record[
            "qmm_projection_coverage"
        ].get("excluded", [])
        record["preparation_memory_peak_bytes"] = record["qmm_metadata"].get(
            "preparation_peak_bytes"
        )
        oracle_tokens = None
        if options.get("exact_speculative_decode"):
            oracle_tokens, oracle_arm = _oracle_tokens_for_arm(
                record_path, round_index, horizon
            )
            record["speculative_oracle_arm"] = oracle_arm
        import mlx.core as mx
        mx.random.seed(seed)
        record["snapshots"]["load"] = snapshot(backend, "load")
        if options.get("cache_step") is not None:
            from .cache import set_kv_cache_step
            set_kv_cache_step(backend.cache, int(options["cache_step"]))
        backend.sampling = Sampling.greedy_settings() if sampling == "greedy" else Sampling()
        backend.thinking_enabled, backend.reasoning_effort = bool(thinking), effort
        template_tokens = backend.open_conversation(system, user, tools=None,
            enable_thinking=thinking, reasoning_effort=effort)
        setup_tokens = 0
        if tool_result is not None:
            # Populate the cache and close the first turn before the tool
            # results land: appending onto a never-generated prompt neither
            # exercises a live cache nor frames the turn correctly.
            _text, setup_stats = backend.generate(max_tokens=1)
            setup_tokens += int(getattr(setup_stats, "tokens", 0) or 0)
            setup_tokens += backend.append_tool_results(
                [tool_result], enable_thinking=thinking
            )
        if repeat_user is not None:
            _text, setup_stats = backend.generate(max_tokens=1)
            setup_tokens += int(getattr(setup_stats, "tokens", 0) or 0)
            backend.append_user(repeat_user, enable_thinking=thinking)
        record.update({"template_tokens": template_tokens, "setup_tokens": setup_tokens,
                       "prompt_tokens": len(backend.pending), "options": options})
        _progress(arm_id, "prefill", done=0, total=len(backend.pending))
        prefill_at, generation_start = [None], time.perf_counter()
        observation_overhead_s = 0.0
        _reset_mlx_peak()
        record["snapshots"]["generation-start"] = snapshot(backend, "generation-start")
        read_start = _disk()
        generation_snapshot = record["snapshots"]["generation-start"]
        prefill_chunks = []
        previous_progress = 0
        previous_progress_at = generation_start
        previous_vm = generation_snapshot.get("vm_counters", {})
        previous_read = generation_snapshot.get("physical_read_bytes")
        last_resource_probe_at = [generation_start - _RESOURCE_SAMPLE_INTERVAL_SECONDS]
        last_resource_probe = [generation_snapshot]

        def resource_failure_for(
            phase: str, now: float | None = None, *, force: bool = False
        ):
            now = time.perf_counter() if now is None else now
            if (
                not force
                and phase == "generation"
                and now - last_resource_probe_at[0] < _RESOURCE_SAMPLE_INTERVAL_SECONDS
            ):
                current = last_resource_probe[0]
            else:
                current = {"vm_counters": _vm()}
                last_resource_probe_at[0] = now
                last_resource_probe[0] = current
            failure = _resource_failure(
                generation_start,
                generation_snapshot,
                current,
                phase,
                policy,
                now=now,
            )
            if failure is not None:
                record["resource_abort"] = failure
            return failure

        def on_prefill_progress(done, total):
            nonlocal previous_progress, previous_progress_at, previous_vm, previous_read
            nonlocal observation_overhead_s
            if done <= previous_progress:
                return
            now = time.perf_counter()
            phase = "completion" if done >= total else "chunk"
            probe_started = time.perf_counter()
            current = snapshot(
                backend, f"prefill-{phase}-{done}", os_probes=True
            )
            observation_overhead_s += time.perf_counter() - probe_started
            current["observation_overhead_s"] = observation_overhead_s
            resource_failure = _resource_failure(
                generation_start,
                generation_snapshot,
                current,
                "prefill",
                policy,
                now=now,
            )
            last_resource_probe_at[0] = now
            last_resource_probe[0] = current
            current_vm = current.get("vm_counters", {})
            current_read = current.get("physical_read_bytes")
            current["progress"] = {
                "done": int(done),
                "total": int(total),
                "chunk_tokens": int(done - previous_progress),
                "duration_s": now - previous_progress_at,
            }
            current["vm_counters_delta"] = _vm_delta(
                previous_vm, current_vm
            )
            current["physical_read_bytes_delta"] = (
                current_read - previous_read
                if current_read is not None and previous_read is not None
                else None
            )
            prefill_chunks.append(current)
            if resource_failure is not None:
                record["resource_abort"] = resource_failure
                raise ResourceAbort(
                    "prefill",
                    str(resource_failure["reason"]),
                    resource_failure,
                )
            _progress(
                arm_id,
                "prefill",
                done=int(done),
                total=int(total),
                chunk_tokens=int(done - previous_progress),
            )
            previous_progress = int(done)
            previous_progress_at = now
            previous_vm = current_vm
            previous_read = current_read

        def on_prefilled():
            nonlocal observation_overhead_s
            prefill_at[0] = time.perf_counter()
            probe_started = time.perf_counter()
            record["snapshots"]["prefill"] = snapshot(backend, "prefill", os_probes=False)
            observation_overhead_s += time.perf_counter() - probe_started
            _progress(arm_id, "decode", generated=0)
            if profiler is not None:
                profiler.start_decode()
        def on_token(value, _piece):
            arrived = time.perf_counter()
            decode_origin = (
                prefill_at[0]
                if prefill_at[0] is not None else generation_start
            )
            arrivals.append({
                "token": int(value),
                "token_arrival_s": arrived,
                "at_s": arrived - generation_start,
                "decode_at_s": arrived - decode_origin,
            })
            if prefill_at[0] is not None:
                resource_failure = resource_failure_for("generation", arrived)
                if resource_failure is not None:
                    raise ResourceAbort(
                        "generation",
                        str(resource_failure["reason"]),
                        resource_failure,
                    )
            if len(arrivals) == 1 or len(arrivals) % 8 == 0:
                _progress(arm_id, "decode", generated=len(arrivals))

        def resource_guard():
            phase = "generation" if prefill_at[0] is not None else "prefill"
            failure = resource_failure_for(phase)
            if failure is not None:
                raise ResourceAbort(phase, str(failure["reason"]), failure)

        def should_cancel():
            resource_guard()
            return False
        if profiler is not None:
            profiler.start()
        try:
            generation_options = {
                "max_tokens": horizon,
                "on_prefilled": on_prefilled,
                "on_prefill_progress": on_prefill_progress,
                "on_decode_token": on_token,
                "resource_check": resource_guard,
            }
            if options.get("exact_speculative_decode"):
                # An empty transcript deliberately reaches the backend's
                # recorded fallback instead of silently turning an oracle arm
                # into a stock arm.
                generation_options["speculative_draft"] = oracle_tokens or []
            if operating_profile_name == "interactive-production":
                # Passing the callback is part of the interactive contract;
                # Bonsai then applies the shared 64-token cancellation cap.
                generation_options["should_cancel"] = should_cancel
            text, model_stats = backend.generate(**generation_options)
        finally:
            if profiler is not None:
                profiler.stop()
        generation_done = time.perf_counter()
        sync_start = time.perf_counter()
        try:
            import mlx.core as mx
            mx.synchronize()
        except (ImportError, AttributeError, RuntimeError):
            pass
        sync_done = time.perf_counter()
        record["snapshots"]["decode"] = snapshot(backend, "decode")
        record["prefill_chunks"] = prefill_chunks
        record["attention_events"] = _attention_events(backend)
        record["attention_counters"] = _attention_counters(backend)
        record["qmm_metadata"] = _qmm_metadata_stats(backend)
        record["qmm_metadata_counters"] = _qmm_metadata_counters(backend)
        record["qmm_metadata"].update({
            "expected_projection_manifest": record[
                "qmm_projection_coverage"
            ].get("expected", []),
            "excluded_projection_manifest": record[
                "qmm_projection_coverage"
            ].get("excluded", []),
        })
        record["q2_counters"] = _q2_counters(backend)
        record["speculative_stats"] = _speculative_stats(backend)
        record["fused_fwht"] = _fused_fwht_stats(backend)
        record["stop_token_sync"] = _stop_token_sync_stats(backend)
        try:
            provenance_after = provenance_manifest(checkpoint)
        except BaseException as error:
            provenance_after = {
                "error": {"type": type(error).__name__, "message": str(error)}
            }
        record["provenance_after"] = provenance_after
        record["provenance_validation"] = validate_provenance(
            expected_provenance or provenance_before,
            provenance_before,
            provenance_after,
        )
        start = record["snapshots"]["generation-start"]
        decode = record["snapshots"]["decode"]
        resource_failure = _resource_failure(
            generation_start,
            start,
            decode,
            "generation",
            policy,
        )
        tokens = [item["token"] for item in arrivals]
        count = int(getattr(model_stats, "tokens", 0) or 0)
        if count and len(tokens) != count:
            tokens = [int(x) for x in backend.tape[-count:]]
        record.update({"status": "raw", "text": text, "tokens": tokens,
            "token_digest": _digest(tokens), "profile": bool(profiler), "stats": {
                "finish": getattr(model_stats, "finish", None), "tokens": count,
                "seconds": float(getattr(model_stats, "seconds", 0.0) or 0.0),
                "rate_tps": float(getattr(model_stats, "rate", 0.0) or 0.0),
                "prompt_tokens": int(getattr(model_stats, "prompt_tokens", 0) or 0),
                "prefill_seconds": float(getattr(model_stats, "prefill_seconds", 0.0) or 0.0),
                "speculative": _speculative_stats(backend)},
            "timing": {"arm_wall_s": time.perf_counter() - started,
                       "generation_wall_s": generation_done - generation_start,
                       "minimally_instrumented_generation_wall_s": max(
                           0.0,
                           generation_done - generation_start - observation_overhead_s,
                       ),
                       "observation_overhead_s": observation_overhead_s,
                       "measurement_mode": "diagnostic-instrumented",
                       "timestamps": {
                           "generation_start_s": generation_start,
                           "prefill_complete_s": prefill_at[0],
                           "first_token_s": (
                               arrivals[0]["token_arrival_s"]
                               if arrivals else None
                           ),
                           "token_arrival_s": (
                               arrivals[0]["token_arrival_s"]
                               if arrivals else None
                           ),
                           "token_arrivals_s": [
                               item["token_arrival_s"] for item in arrivals
                           ],
                           "generation_done_s": generation_done,
                       },
                       "prefill_complete_relative_s": (
                           prefill_at[0] - generation_start
                           if prefill_at[0] is not None else None
                       ),
                       "sync_s": sync_done - sync_start,
                       "first_token_latency_s": arrivals[0]["at_s"] if arrivals else None,
                       "decode_relative_first_token_latency_s": (
                           arrivals[0]["decode_at_s"] if arrivals else None
                       )},
            "windows": _windows(arrivals, window),
            "runtime_source_fingerprints": provenance_after.get(
                "runtime_source_fingerprints", source_fingerprints()
            ),
            "dependency_info": provenance_after.get(
                "dependency_info", dependency_info()
            ),
            "metrics": {
                "physical_read_bytes": (
                    decode.get("physical_read_bytes") - read_start
                    if decode.get("physical_read_bytes") is not None and read_start is not None
                    else None
                ),
                "physical_read_bytes_generation": (
                    decode.get("physical_read_bytes") - start.get("physical_read_bytes")
                    if decode.get("physical_read_bytes") is not None
                    and start.get("physical_read_bytes") is not None
                    else None
                ),
                "vm_delta": _vm_delta(
                    start.get("vm_counters", {}), decode.get("vm_counters", {})
                ),
                "cache": decode.get("cache"),
                "mlx": decode.get("mlx"),
                "process": decode.get("process"),
                "memory_peaks": {
                    "preparation_bytes": record["qmm_metadata"].get(
                        "preparation_peak_bytes"
                    ),
                    "generation_mlx_bytes": (
                        decode.get("mlx", {}) or {}
                    ).get("peak_bytes"),
                    "generation_process_bytes": (
                        decode.get("process", {}) or {}
                    ).get("peak_rss_bytes"),
                },
            }})
        append_jsonl(record_path, record); raw_written = True
        if record["provenance_validation"]["status"] != "matched":
            append_jsonl(record_path, {
                "type": "validation", "arm_id": arm_id, "status": "failed",
                "check": "provenance",
                "provenance": record["provenance_validation"],
            })
            return 1
        if resource_failure is not None:
            append_jsonl(record_path, {
                "type": "validation", "arm_id": arm_id, "status": "failed",
                "failure": {"type": "resource_abort", **resource_failure},
            })
            return 1
        _progress(
            arm_id,
            "path",
            generated=count,
            prepared_qmm_calls=record["qmm_metadata_counters"].get("prepared_calls", 0),
            q2_selected=record["q2_counters"].get("q2_selected", 0),
        )
        if count <= 0:
            append_jsonl(record_path, {"type": "validation", "arm_id": arm_id, "status": "failed",
                                       "failure": {"type": "no_tokens", "message": "no generated tokens"}})
            return 1
        if count != horizon:
            append_jsonl(record_path, {"type": "validation", "arm_id": arm_id, "status": "failed",
                                       "failure": {"type": "incomplete_horizon", "expected": horizon, "actual": count}})
            return 1
        if not backend.check_invariant():
            append_jsonl(record_path, {"type": "validation", "arm_id": arm_id, "status": "failed",
                                       "failure": {"type": "cache_invariant", "message": "cache does not match tape"}})
            return 1
        return 0
    except BaseException as error:
        if "prefill_chunks" in locals():
            record["prefill_chunks"] = prefill_chunks
        if backend is not None:
            record["attention_events"] = _attention_events(backend)
            record["attention_counters"] = _attention_counters(backend)
            record["qmm_metadata"] = _qmm_metadata_stats(backend)
            record["qmm_metadata_counters"] = _qmm_metadata_counters(backend)
            record["qmm_metadata"].update({
                "expected_projection_manifest": record[
                    "qmm_projection_coverage"
                ].get("expected", []),
                "excluded_projection_manifest": record[
                    "qmm_projection_coverage"
                ].get("excluded", []),
            })
            record["q2_counters"] = _q2_counters(backend)
            record["speculative_stats"] = _speculative_stats(backend)
            record["fused_fwht"] = _fused_fwht_stats(backend)
            record["stop_token_sync"] = _stop_token_sync_stats(backend)
        if arrivals:
            record["tokens"] = [item["token"] for item in arrivals]
            record["token_digest"] = _digest(record["tokens"])
            record["windows"] = _windows(arrivals, window)
        if isinstance(error, ResourceAbort):
            record["resource_abort"] = {
                "phase": error.phase,
                "reason": error.reason,
                **error.details,
            }
        record["error"] = {"type": type(error).__name__, "message": str(error) or repr(error)}
        record["timing"] = {"arm_wall_s": time.perf_counter() - started}
        _progress(arm_id, "failed", error=type(error).__name__, message=str(error)[:160])
        if not raw_written:
            append_jsonl(record_path, record)
        else:
            append_jsonl(record_path, {"type": "validation", "arm_id": arm_id,
                                       "status": "failed", "failure": record["error"]})
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if profiler is not None:
            profiler.stop()
            try:
                profile_record = {"type": "profile", "arm_id": arm_id,
                                  "condition": condition, "round": round_index,
                                  "status": "captured", **profiler.save(record_path, arm_id)}
            except Exception as error:
                profile_record = {"type": "profile", "arm_id": arm_id, "status": "failed",
                                  "error": {"type": type(error).__name__, "message": str(error)}}
            append_jsonl(record_path, profile_record)
            if profile_record["status"] == "failed":
                append_jsonl(record_path, {"type": "validation", "arm_id": arm_id,
                                          "status": "failed", "failure": profile_record["error"]})
        cleanup = {"type": "cleanup", "arm_id": arm_id, "status": "complete"}
        try:
            if backend is not None:
                try:
                    import mlx.core as mx
                    mx.synchronize()
                except (ImportError, AttributeError, RuntimeError):
                    pass
                cleanup["snapshot"] = snapshot(backend, "final")
            import gc
            gc.collect()
        except BaseException as error:
            cleanup["status"] = "failed"
            cleanup["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            append_jsonl(record_path, cleanup)
        except OSError:
            pass
def ordered_conditions(names: list[str], rounds: int) -> list[tuple[int, str]]:
    if not names or rounds < 2:
        raise ValueError("need at least two reverse-interleaved rounds")
    return [(r, name) for r in range(rounds)
            for name in (names if r % 2 == 0 else names[::-1])]
def sign_test(values: list[float]) -> dict[str, Any]:
    from math import comb
    wins, losses = sum(x > 0 for x in values), sum(x < 0 for x in values)
    ties, total = len(values) - wins - losses, wins + losses
    if not total:
        p = 1.0
    else:
        lower = sum(comb(total, k) for k in range(wins + 1)) / 2 ** total
        upper = sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total
        p = min(1.0, 2 * min(lower, upper))
    return {"wins": wins, "losses": losses, "ties": ties, "p_two_sided": p}
def paired_stats(control: list[dict[str, Any]], candidate: list[dict[str, Any]], control_name="control", candidate_name="candidate") -> dict[str, Any]:
    left = {int(x.get("round", i)): x for i, x in enumerate(control) if x.get("status") == "raw"}
    right = {int(x.get("round", i)): x for i, x in enumerate(candidate) if x.get("status") == "raw"}
    pairs, values = [], []
    for round_index in sorted(set(left) & set(right)):
        a = float(left[round_index].get("stats", {}).get("rate_tps", 0) or 0)
        b = float(right[round_index].get("stats", {}).get("rate_tps", 0) or 0)
        if a <= 0 or b <= 0:
            continue
        delta = (b - a) / a * 100
        values.append(delta)
        pairs.append({"round": round_index, "control_rate_tps": a, "candidate_rate_tps": b,
                      "delta_pct": delta, "direction": "improvement" if delta > 0 else "regression" if delta < 0 else "tie"})
    mean = st.mean(values) if values else None
    band = 2 * st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"control": control_name, "candidate": candidate_name, "pairs": pairs,
            "pair_count": len(pairs), "mean_delta_pct": mean,
            "median_delta_pct": st.median(values) if values else None,
            "two_se_band_pct": band,
            "direction": "improvement" if mean and mean > 0 else "regression" if mean and mean < 0 else "tie",
            "resolved": bool(mean is not None and band is not None and abs(mean) > band),
            **sign_test(values)}


def paired_prefill_stats(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    control_name="control",
    candidate_name="candidate",
) -> dict[str, Any]:
    """Report diagnostic prefill deltas from provenance-valid raw arms.

    This deliberately does not make a quality or promotion decision.  A
    Q2 arm whose output digest fails can still show whether complete prompt
    processing got faster, while the normal paired decode statistics exclude
    that arm.
    """

    def eligible(row):
        provenance = row.get("provenance_validation")
        return (
            row.get("status") == "raw"
            and isinstance(provenance, dict)
            and provenance.get("status") == "matched"
        )

    left = {
        int(x.get("round", i)): x for i, x in enumerate(control)
        if eligible(x)
    }
    right = {
        int(x.get("round", i)): x for i, x in enumerate(candidate)
        if eligible(x)
    }
    pairs, values = [], []
    for round_index in sorted(set(left) & set(right)):
        a = float(left[round_index].get("stats", {}).get("prefill_seconds", 0) or 0)
        b = float(right[round_index].get("stats", {}).get("prefill_seconds", 0) or 0)
        if a <= 0 or b <= 0:
            continue
        delta = (a - b) / a * 100
        values.append(delta)
        pairs.append({
            "round": round_index,
            "control_prefill_seconds": a,
            "candidate_prefill_seconds": b,
            "delta_pct": delta,
            "direction": "improvement" if delta > 0 else "regression" if delta < 0 else "tie",
        })
    mean = st.mean(values) if values else None
    band = 2 * st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {
        "control": control_name,
        "candidate": candidate_name,
        "pairs": pairs,
        "pair_count": len(pairs),
        "mean_delta_pct": mean,
        "median_delta_pct": st.median(values) if values else None,
        "two_se_band_pct": band,
        "direction": "improvement" if mean and mean > 0 else "regression" if mean and mean < 0 else "tie",
        "resolved": bool(mean is not None and band is not None and abs(mean) > band),
        **sign_test(values),
    }


_FUSED_PATH_GATE = {
    "minimum_selected_calls": 2,
    "maximum_fallback_fraction": 0.10,
    "requires_attempt_on_every_call": True,
}
_QMM_METADATA_PATH_GATE = {
    "requires_expected_projection_set": True,
    "minimum_prepared_calls": 1,
    "maximum_unsupported_calls": 0,
    "maximum_lower_precision_calls": 0,
}
_Q2_PATH_GATE = {
    "minimum_selected_calls": 1,
    "maximum_fallback_fraction": 0.0,
}
# These are the two distinct packed MLP geometries that the Q2 hook is
# expected to see.  The runtime counters are observations; this set is the
# benchmark's independent coverage contract.
_Q2_EXPECTED_PROJECTION_GEOMETRIES = frozenset({
    "17408x5120", "5120x17408",
})
_SPECULATIVE_PATH_GATE = {
    "minimum_verification_blocks": 2,
    "maximum_fallbacks": 0,
}
_FUSED_FWHT_PATH_GATE = {
    "minimum_selected_calls": 1,
    "maximum_fallbacks": 0,
    "required_input_dtype": "float32",
}
_CANCELLATION_CODES = {
    -int(signal.SIGINT), 128 + int(signal.SIGINT),
    -int(signal.SIGTERM), 128 + int(signal.SIGTERM),
}


def _is_cancellation_returncode(returncode: int) -> bool:
    try:
        return int(returncode) in _CANCELLATION_CODES
    except (TypeError, ValueError):
        return False


def _is_cancelled_row(row: dict[str, Any]) -> bool:
    error = row.get("error")
    return isinstance(error, dict) and error.get("type") == "KeyboardInterrupt"


def _counter_total(counters: dict[str, Any], name: str) -> int:
    value = counters.get(name)
    if not isinstance(value, dict):
        return 0
    try:
        return int(value.get("total", 0))
    except (TypeError, ValueError):
        return 0


def _expected_qmm_projection_coverage(checkpoint: str) -> dict[str, Any]:
    """Build the QMM coverage oracle from checkpoint structure, not runtime lists."""
    try:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = Path(checkpoint_info(checkpoint)["path"])
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "source": "checkpoint.config.modules",
            "valid": False,
            "expected": [],
            "excluded": [],
            "errors": [{"reason": "unreadable_checkpoint_structure", "message": str(error)}],
        }
    modules = config.get("modules") if isinstance(config, dict) else None
    if not isinstance(modules, list):
        return {
            "source": "checkpoint.config.modules",
            "valid": False,
            "expected": [],
            "excluded": [],
            "errors": [{"reason": "missing_modules"}],
        }
    expected, excluded, errors, seen = [], [], [], set()
    for index, item in enumerate(modules):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append({"index": index, "reason": "invalid_module_identity"})
            continue
        name = item["path"]
        if name in seen:
            errors.append({"name": name, "reason": "duplicate_module_identity"})
            continue
        seen.add(name)
        entry = {"name": name, "index": index}
        if bool(item.get("embedding", False)):
            excluded.append({**entry, "reason": "embedding_module"})
        else:
            expected.append(entry)
    return {
        "source": "checkpoint.config.modules",
        "valid": not errors,
        "expected": expected,
        "excluded": excluded,
        "errors": errors,
    }
def _execution_path_validation(comparison: str, row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") != "raw":
        return None
    if row.get("child_returncode", 0) != 0 or row.get("child_validation_failures"):
        return None
    if comparison == "prepared-qmm-metadata":
        counters = row.get("qmm_metadata_counters")
        if not isinstance(counters, dict):
            return {"passed": False, "reason": "missing_qmm_metadata_counters"}
        metadata = row.get("qmm_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        manifest = metadata.get("eligible_module_manifest")
        prepared_manifest = metadata.get("prepared_module_manifest")
        if not isinstance(manifest, list) or not isinstance(prepared_manifest, list):
            return {
                "passed": False,
                "reason": "missing_qmm_module_manifest",
            }
        prepared = int(counters.get("prepared_calls", 0) or 0)
        fp32 = int(counters.get("fp32_calls", 0) or 0)
        lower = int(counters.get("lower_precision_calls", 0) or 0)
        unsupported = int(counters.get("unsupported_calls", 0) or 0)
        phase_calls = counters.get("phase_calls")
        phase_prepared = counters.get("phase_prepared_calls")
        phase_fp32 = counters.get("phase_fp32_calls")
        phase_lower = counters.get("phase_lower_precision_calls")
        phase_unsupported = counters.get("phase_unsupported_calls")
        module_calls = counters.get("module_calls")
        phase_calls = phase_calls if isinstance(phase_calls, dict) else {}
        phase_prepared = phase_prepared if isinstance(phase_prepared, dict) else {}
        phase_fp32 = phase_fp32 if isinstance(phase_fp32, dict) else {}
        phase_lower = phase_lower if isinstance(phase_lower, dict) else {}
        phase_unsupported = phase_unsupported if isinstance(phase_unsupported, dict) else {}
        module_calls = module_calls if isinstance(module_calls, dict) else {}
        coverage = _expected_qmm_projection_coverage(str(row.get("checkpoint", "")))
        expected_manifest = coverage.get("expected", [])
        excluded_manifest = coverage.get("excluded", [])
        expected_names = [
            item["name"] for item in expected_manifest
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        expected_name_set = set(expected_names)
        manifest_names = [
            item.get("name") for item in manifest if isinstance(item, dict)
        ]
        manifest_invalid = sum(not isinstance(name, str) for name in manifest_names)
        prepared_invalid = sum(
            not isinstance(name, str) for name in prepared_manifest
        )
        prepared_names = [name for name in prepared_manifest if isinstance(name, str)]
        manifest_name_set = {
            name for name in manifest_names if isinstance(name, str)
        }
        prepared_name_set = set(prepared_names)
        manifest_duplicates = sorted(
            name for name in manifest_name_set if manifest_names.count(name) > 1
        )
        prepared_duplicates = sorted(
            name for name in prepared_name_set if prepared_names.count(name) > 1
        )
        missing_modules = sorted(name for name in expected_name_set if not module_calls.get(name))
        unexpected_modules = sorted(
            str(name) for name, value in module_calls.items()
            if name not in expected_name_set and value
        )
        unexpected_manifest = sorted(manifest_name_set - expected_name_set)
        unexpected_prepared = sorted(prepared_name_set - expected_name_set)
        expected_projection_valid = bool(coverage.get("valid")) and bool(expected_names)
        phase_coverage = {
            phase: {
                "calls": int(phase_calls.get(phase, 0) or 0),
                "prepared_calls": int(phase_prepared.get(phase, 0) or 0),
                "fp32_calls": int(phase_fp32.get(phase, 0) or 0),
                "lower_precision_calls": int(phase_lower.get(phase, 0) or 0),
                "unsupported_calls": int(phase_unsupported.get(phase, 0) or 0),
            }
            for phase in ("prefill", "decode", "unknown")
        }
        phase_total = sum(item["calls"] for item in phase_coverage.values())
        module_total = 0
        module_values_valid = True
        for value in module_calls.values():
            try:
                module_total += int(value or 0)
            except (TypeError, ValueError):
                module_values_valid = False
        counters_consistent = (
            prepared == fp32 + lower + unsupported
            and phase_total == prepared
            and sum(item["prepared_calls"] for item in phase_coverage.values()) == prepared
            and sum(item["fp32_calls"] for item in phase_coverage.values()) == fp32
            and module_total == prepared
            and module_values_valid
        )
        checks = {
            "prepared_calls": prepared,
            "fp32_calls": fp32,
            "lower_precision_calls": lower,
            "unsupported_calls": unsupported,
            "eligible_modules": len(manifest),
            "prepared_modules": len(prepared_manifest),
            "prepared_module_set_complete": (
                len(prepared_names) == len(expected_names)
                and prepared_name_set == expected_name_set
                and not prepared_duplicates
            ),
            "expected_projection_manifest_valid": expected_projection_valid,
            "expected_projection_modules": len(expected_names),
            "covered_projection_modules": len(expected_names) - len(missing_modules),
            "excluded_projection_manifest": excluded_manifest,
            "manifest_duplicates": manifest_duplicates,
            "prepared_duplicates": prepared_duplicates,
            "manifest_invalid_entries": manifest_invalid,
            "prepared_invalid_entries": prepared_invalid,
            "unexpected_manifest": unexpected_manifest,
            "unexpected_prepared": unexpected_prepared,
            "missing_modules": missing_modules,
            "unexpected_modules": unexpected_modules,
            "projection_coverage_complete": (
                expected_projection_valid
                and len(manifest_names) == len(expected_names)
                and manifest_name_set == expected_name_set
                and not manifest_duplicates
                and not prepared_duplicates
                and manifest_invalid == 0
                and prepared_invalid == 0
                and not unexpected_manifest
                and not unexpected_prepared
                and not missing_modules
                and not unexpected_modules
            ),
            "module_calls_total": module_total,
            "phase_coverage": phase_coverage,
            "counters_consistent": counters_consistent,
        }
        if row.get("condition") == "qmm-stock":
            passed = (
                prepared == 0 and fp32 == 0 and lower == 0 and unsupported == 0
                and phase_total == 0
            )
            return {
                **checks,
                "passed": passed,
                "reason": "stock_control_selected" if passed else "stock_control_path_invalid",
            }
        if row.get("condition") != "qmm-prepared":
            return None
        passed = (
            checks["expected_projection_manifest_valid"]
            and checks["prepared_module_set_complete"]
            and checks["projection_coverage_complete"]
            and not missing_modules
            and prepared >= _QMM_METADATA_PATH_GATE["minimum_prepared_calls"]
            and fp32 == prepared
            and lower <= _QMM_METADATA_PATH_GATE["maximum_lower_precision_calls"]
            and unsupported <= _QMM_METADATA_PATH_GATE["maximum_unsupported_calls"]
            and counters_consistent
        )
        return {
            **checks,
            "coverage": fp32 / prepared if prepared else 0.0,
            "passed": passed,
            "reason": "prepared_qmm_selected" if passed else "prepared_qmm_not_selected",
        }
    if comparison == "q2-prefill-mpp":
        counters = row.get("q2_counters")
        if not isinstance(counters, dict):
            return {"passed": False, "reason": "missing_q2_counters"}
        candidate_calls = int(counters.get("q2_candidate_calls", 0) or 0)
        calls = int(
            counters.get("q2_eligible_calls", candidate_calls) or 0
        )
        attempts = int(counters.get("q2_attempts", calls) or 0)
        selected = int(counters.get("q2_selected", 0) or 0)
        fallbacks = int(counters.get("q2_fallbacks", 0) or 0)
        excluded = int(counters.get("q2_policy_excluded_calls", 0) or 0)
        phase_calls = counters.get("q2_phase_calls")
        phase_selected = counters.get("q2_phase_selected")
        phase_fallbacks = counters.get("q2_phase_fallbacks")
        phase_excluded = counters.get("q2_phase_policy_excluded")
        phase_calls = phase_calls if isinstance(phase_calls, dict) else {}
        phase_selected = phase_selected if isinstance(phase_selected, dict) else {}
        phase_fallbacks = phase_fallbacks if isinstance(phase_fallbacks, dict) else {}
        phase_excluded = phase_excluded if isinstance(phase_excluded, dict) else {}
        geometries = counters.get("q2_eligible_geometries")
        geometry_values = {}
        geometry_counts_valid = isinstance(geometries, dict)
        if geometry_counts_valid:
            for name, value in geometries.items():
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    geometry_counts_valid = False
                    continue
                if value < 0:
                    geometry_counts_valid = False
                geometry_values[str(name)] = value
        observed_geometries = {
            name for name, value in geometry_values.items() if value > 0
        }
        missing_geometries = sorted(
            _Q2_EXPECTED_PROJECTION_GEOMETRIES - observed_geometries
        )
        geometry_total = sum(geometry_values.values())
        exclusion_reasons = counters.get("q2_policy_exclusion_reasons")
        exclusion_reasons = exclusion_reasons if isinstance(exclusion_reasons, dict) else {}
        exclusion_reason_total = 0
        exclusion_reasons_valid = True
        for value in exclusion_reasons.values():
            try:
                value = int(value)
            except (TypeError, ValueError):
                exclusion_reasons_valid = False
                continue
            if value < 0:
                exclusion_reasons_valid = False
            exclusion_reason_total += value
        exclusion_phase_total = 0
        exclusion_phase_valid = True
        for value in phase_excluded.values():
            try:
                value = int(value)
            except (TypeError, ValueError):
                exclusion_phase_valid = False
                continue
            if value < 0:
                exclusion_phase_valid = False
            exclusion_phase_total += value
        exclusions = int(counters.get("q2_policy_excluded_calls", 0) or 0)
        exclusions_consistent = (
            exclusions >= 0
            and exclusion_reasons_valid
            and exclusion_phase_valid
            and exclusion_reason_total == exclusions
            and exclusion_phase_total == exclusions
        )
        phase_coverage = {
            phase: {
                "eligible_calls": int(phase_calls.get(phase, 0) or 0),
                "selected": int(phase_selected.get(phase, 0) or 0),
                "fallbacks": int(phase_fallbacks.get(phase, 0) or 0),
                "policy_excluded": int(phase_excluded.get(phase, 0) or 0),
            }
            for phase in ("prefill", "decode", "unknown")
        }
        phase_attempts = sum(item["eligible_calls"] for item in phase_coverage.values())
        phase_selected_total = sum(item["selected"] for item in phase_coverage.values())
        phase_fallback_total = sum(item["fallbacks"] for item in phase_coverage.values())
        counters_consistent = (
            candidate_calls == calls
            and attempts == calls
            and phase_attempts == attempts
            and phase_selected_total == selected
            and phase_fallback_total == fallbacks
            and selected + fallbacks == attempts
        )
        checks = {
            "candidate_calls": candidate_calls,
            "eligible_calls": calls,
            "attempts": attempts,
            "selected": selected,
            "fallbacks": fallbacks,
            "policy_excluded_calls": excluded,
            "policy_exclusion_reasons": exclusion_reasons,
            "policy_exclusions_consistent": exclusions_consistent,
            "expected_projection_geometries": sorted(
                _Q2_EXPECTED_PROJECTION_GEOMETRIES
            ),
            "observed_projection_geometries": sorted(observed_geometries),
            "missing_projection_geometries": missing_geometries,
            "projection_geometry_counts": geometry_values,
            "projection_geometry_total": geometry_total,
            "projection_coverage": (
                len(observed_geometries)
                / len(_Q2_EXPECTED_PROJECTION_GEOMETRIES)
                if geometry_counts_valid else 0.0
            ),
            "phase_coverage": phase_coverage,
            "counters_consistent": counters_consistent,
            "projection_coverage_present": geometry_counts_valid,
        }
        if row.get("condition") == "q2-stock":
            passed = (
                calls == 0 and selected == 0 and fallbacks == 0
                and counters_consistent
            )
            return {
                **checks, "passed": passed,
                "reason": "stock_control_selected" if passed else "stock_control_path_invalid",
            }
        if row.get("condition") != "q2-mpp":
            return None
        passed = (
            calls > 0
            and counters_consistent
            and selected >= _Q2_PATH_GATE["minimum_selected_calls"]
            and selected == calls
            and fallbacks <= _Q2_PATH_GATE["maximum_fallback_fraction"] * calls
            and geometry_counts_valid
            and geometry_total == calls
            and not missing_geometries
            and exclusions_consistent
        )
        selection_valid = (
            calls > 0
            and counters_consistent
            and selected >= _Q2_PATH_GATE["minimum_selected_calls"]
            and selected == calls
            and fallbacks <= _Q2_PATH_GATE["maximum_fallback_fraction"] * calls
        )
        if not selection_valid:
            reason = "q2_candidate_not_selected"
        elif not exclusions_consistent:
            reason = "q2_exclusion_accounting_invalid"
        elif not geometry_counts_valid or geometry_total != calls:
            reason = "q2_projection_coverage_invalid"
        elif missing_geometries:
            reason = "q2_projection_coverage_incomplete"
        else:
            reason = "q2_candidate_selected"
        return {
            **checks,
            "coverage": selected / calls if calls else 0.0,
            "fallback_fraction": fallbacks / calls if calls else 1.0,
            "passed": passed,
            "reason": reason if passed else (
                reason if reason != "q2_candidate_selected"
                else "q2_candidate_not_selected"
            ),
        }
    if comparison.startswith("exact-speculative-oracle-"):
        stats = row.get("speculative_stats")
        if not isinstance(stats, dict):
            return {"passed": False, "reason": "missing_speculative_stats"}
        selected = int(stats.get("selected", 0) or 0)
        blocks = int(stats.get("verification_blocks", 0) or 0)
        fallbacks = int(stats.get("fallbacks", 0) or 0)
        capability = stats.get("capability")
        capability_status = (
            capability.get("status") if isinstance(capability, dict) else "unknown"
        )
        checks = {
            "selected": selected,
            "verification_blocks": blocks,
            "fallbacks": fallbacks,
            "capability": capability,
            "capability_status": capability_status,
        }
        if row.get("condition") == "spec-stock":
            passed = selected == 0 and blocks == 0
            return {
                **checks, "passed": passed,
                "reason": "stock_control_selected" if passed else "stock_control_path_invalid",
            }
        if not row.get("condition", "").startswith("spec-oracle-"):
            return None
        passed = (
            bool(stats.get("oracle"))
            and capability_status == "supported"
            and selected > 0
            and blocks >= _SPECULATIVE_PATH_GATE["minimum_verification_blocks"]
            and fallbacks <= _SPECULATIVE_PATH_GATE["maximum_fallbacks"]
        )
        return {
            **checks,
            "passed": passed,
            "reason": "oracle_verifier_selected" if passed else "oracle_verifier_not_selected",
        }
    if comparison == "fused-fwht":
        stats = row.get("fused_fwht")
        if not isinstance(stats, dict):
            return {"passed": False, "reason": "missing_fused_fwht_counters"}
        requested = bool(stats.get("requested"))
        selected = int(stats.get("selected", 0) or 0)
        fallbacks = int(stats.get("fallbacks", 0) or 0)
        attempts = int(stats.get("attempts", 0) or 0)
        checks = {
            "requested": requested,
            "attempts": attempts,
            "selected": selected,
            "fallbacks": fallbacks,
            "executed": bool(stats.get("executed")),
            "input_dtypes": stats.get("input_dtypes", {}),
            "fallback_reasons": stats.get("fallback_reasons", {}),
        }
        input_dtypes = checks["input_dtypes"]
        fp32_inputs = int(
            input_dtypes.get("float32", 0)
            if isinstance(input_dtypes, dict) else 0
        )
        checks["required_input_dtype_calls"] = fp32_inputs
        if row.get("condition") == "control":
            passed = not requested and selected == 0
            reason = "stock_control_selected" if passed else "stock_control_path_invalid"
        else:
            passed = (
                requested
                and selected >= _FUSED_FWHT_PATH_GATE["minimum_selected_calls"]
                and fallbacks <= _FUSED_FWHT_PATH_GATE["maximum_fallbacks"]
                and fp32_inputs > 0
            )
            reason = "fused_fwht_selected" if passed else (
                "fused_fwht_fp32_not_executed"
                if fp32_inputs == 0 else "fused_fwht_not_executed"
            )
        return {**checks, "passed": passed, "reason": reason}
    if comparison != "q4-attention-fused":
        return None
    counters = row.get("attention_counters")
    if not isinstance(counters, dict):
        return {"passed": False, "reason": "missing_attention_counters"}
    total = _counter_total(counters, "attention_calls")
    attempts = _counter_total(counters, "fused_attempts")
    selected = _counter_total(counters, "fused_selected")
    fallbacks = _counter_total(counters, "fused_fallbacks")
    stock = _counter_total(counters, "stock_selected")
    coverage = selected / total if total else 0.0
    fallback_fraction = fallbacks / total if total else 1.0
    checks = {
        "attention_calls": total,
        "fused_attempts": attempts,
        "fused_selected": selected,
        "fused_fallbacks": fallbacks,
        "stock_selected": stock,
        "coverage": coverage,
        "fallback_fraction": fallback_fraction,
    }
    if row.get("condition") == "q4-stock":
        passed = total > 0 and attempts == 0 and selected == 0 and stock > 0
        return {
            **checks,
            "passed": passed,
            "reason": "stock_control_selected" if passed else "stock_control_path_invalid",
        }
    if row.get("condition") != "q4-fused":
        return None
    if total == 0:
        reason = "no_attention_calls"
        passed = False
    elif _FUSED_PATH_GATE["requires_attempt_on_every_call"] and attempts != total:
        reason = "fused_attempts_incomplete"
        passed = False
    elif selected < _FUSED_PATH_GATE["minimum_selected_calls"]:
        reason = "insufficient_fused_calls"
        passed = False
    elif coverage < 1.0 - _FUSED_PATH_GATE["maximum_fallback_fraction"]:
        reason = "fused_coverage_below_gate"
        passed = False
    elif fallback_fraction > _FUSED_PATH_GATE["maximum_fallback_fraction"]:
        reason = "fused_fallback_coverage_above_gate"
        passed = False
    else:
        reason = "fused_candidate_selected"
        passed = True
    return {**checks, "passed": passed, "reason": reason}


def run_comparison(*, checkpoint: str, comparison: str, record_path: str, fixture=DEFAULT_FIXTURE, horizon=SHORT, window=WINDOW, rounds=3,
                   thinking=False, effort="medium", sampling="greedy", prefill_step_size=512, seed=SEED,
                   runner=None, experiment_id: str | None = None) -> dict[str, Any]:
    if comparison not in COMPARISONS or fixture not in FIXTURES:
        raise ValueError("unknown comparison or fixture")
    if horizon <= 0 or window <= 0 or horizon % window or rounds < 2:
        raise ValueError("invalid horizon, window, or rounds")
    if comparison == "prepared-qmm-metadata" and experiment_id is None:
        experiment_id = "bonsai2-production-qmm-metadata-256m-v1"
    conditions, meta = COMPARISONS[comparison], metadata(
        checkpoint, comparison, fixture, horizon, window, thinking, effort,
        sampling, prefill_step_size, seed, experiment_id)
    if comparison in DIAGNOSTIC_COMPARISONS:
        meta["purpose"] = "profiler overhead diagnostic; not an optimization comparison"
    if comparison == "q4-attention-fused":
        meta["execution_path_gate"] = dict(_FUSED_PATH_GATE)
    elif comparison == "prepared-qmm-metadata":
        meta["execution_path_gate"] = dict(_QMM_METADATA_PATH_GATE)
        meta["candidate"] = "one-time FP32 scales/biases preparation"
        meta["control"] = "stock FP16 metadata with MLX per-call promotion"
        meta["production_profile"] = {
            "allocator_cache_mb": 256.0,
            "configured_prefill_step_size": prefill_step_size,
            "cancellable_prefill_step_size": 64,
            "cancellation_callback": "resource_admission_guard",
            "arms_differ_only_by": "prepared_qmm_metadata",
        }
        meta["expected_projection_coverage"] = _expected_qmm_projection_coverage(
            checkpoint
        )
    elif comparison == "q2-prefill-mpp":
        meta["execution_path_gate"] = dict(_Q2_PATH_GATE)
        meta["exactness_gate"] = "candidate must match the greedy digest before promotion"
        meta["numerical_mode"] = "output-changing affine regrouping diagnostic"
    elif comparison.startswith("exact-speculative-oracle-"):
        meta["execution_path_gate"] = dict(_SPECULATIVE_PATH_GATE)
        meta["oracle_draft"] = True
        meta["promotion_note"] = "perfect-draft ceiling only; not achieved decoding"
    append_jsonl(record_path, {"type": "run", "status": "started", "metadata": meta})
    rows = {name: [] for name in conditions}
    interrupted = False
    interruption = None
    stop_reason = None
    names = list(conditions)
    expected = None
    validation_failures = 0
    execution_path_failures = 0
    path_validated = set()
    digest_validated = set()
    for round_index, condition in ordered_conditions(list(conditions), rounds):
        arm_id = f"round-{round_index + 1}-{condition}"
        current_provenance = provenance_manifest(checkpoint)
        manifest_check = validate_provenance(
            meta["provenance_manifest"], current_provenance
        )
        if manifest_check["status"] != "matched":
            stop_reason = {
                "arm_id": arm_id,
                "type": (
                    "provenance_changed_before_arm"
                    if manifest_check["status"] == "changed"
                    else "provenance_unknown_before_arm"
                ),
                "failure": manifest_check,
            }
            row = {
                "type": "arm", "schema": 1, "arm_id": arm_id,
                "condition": condition, "round": round_index,
                "status": "failed", "tokens": [],
                "token_digest": _digest([]),
                "error": {
                    "type": "ProvenanceChanged"
                    if manifest_check["status"] == "changed"
                    else "ProvenanceUnknown",
                    "message": "arm provenance is not comparable before execution",
                    "details": manifest_check,
                },
                "provenance_validation": manifest_check,
            }
            rows[condition].append(row)
            append_jsonl(record_path, row)
            append_jsonl(record_path, {
                "type": "validation", "arm_id": arm_id, "status": "failed",
                "check": "provenance", "failure": manifest_check,
            })
            validation_failures += 1
            break
        command = [sys.executable, "-m", "models.bonsai2.bench", "--child",
                   "--checkpoint", checkpoint, "--arm-id", arm_id, "--condition", condition,
                   "--round", str(round_index), "--options-json", json.dumps(conditions[condition], sort_keys=True),
                   "--record", str(Path(record_path).expanduser()), "--fixture", fixture,
                   "--horizon", str(horizon), "--window", str(window), "--effort", effort,
                   "--sampling", sampling, "--prefill-step-size", str(prefill_step_size),
                   "--seed", str(seed), "--resource-policy-json",
                   json.dumps(meta["resource_policy"], sort_keys=True),
                   "--provenance-json",
                   json.dumps(meta["provenance_manifest"], sort_keys=True),
                   "--operating-profile", str(meta["operating_profile"])]
        if thinking:
            command.append("--thinking")
        before = read_jsonl(record_path)
        env = dict(os.environ)
        # Precision rides the constructor options, never the ambient
        # environment: an inherited MACQWEN_BONSAI2_KV would silently turn
        # the full-precision control into a quantized arm.
        env.pop("MACQWEN_BONSAI2_KV", None)
        env["PYTHONPATH"] = os.pathsep.join(x for x in (str(ROOT), env.get("PYTHONPATH", "")) if x)
        cancelled = False
        cancellation = None
        parent_timeout = False
        try:
            result = (
                runner(command, cwd=str(ROOT), env=env,
                       capture_output=True, text=True, check=False)
                if runner is not None
                else _stream_child(command, env, arm_id)
            )
            returncode, stderr = int(getattr(result, "returncode", 0)), str(getattr(result, "stderr", "") or "")
            parent_timeout = bool(getattr(result, "timed_out", False))
            if _is_cancellation_returncode(returncode):
                cancelled = True
                cancellation = {
                    "type": "subprocess_termination",
                    "returncode": returncode,
                }
        except KeyboardInterrupt as error:
            cancelled = True
            returncode, stderr = 128 + int(signal.SIGINT), (
                f"KeyboardInterrupt: {error}" if str(error) else "KeyboardInterrupt"
            )
            cancellation = {"type": "KeyboardInterrupt", "message": stderr}
        except BaseException as error:
            returncode, stderr = 1, f"{type(error).__name__}: {error}"
        new = read_jsonl(record_path)[len(before):]
        row = next((x for x in new if x.get("type") == "arm" and x.get("arm_id") == arm_id), None)
        if row is not None and _is_cancelled_row(row):
            cancelled = True
            error_info = row.get("error")
            cancellation = {
                "type": "KeyboardInterrupt",
                "message": (
                    error_info.get("message", "KeyboardInterrupt")
                    if isinstance(error_info, dict)
                    else "KeyboardInterrupt"
                ),
            }
        if row is None:
            row = {"type": "arm", "schema": 1, "arm_id": arm_id, "condition": condition,
                   "round": round_index, "status": "interrupted" if cancelled else "failed",
                   "tokens": [], "token_digest": _digest([]),
                   "error": {"type": "cancellation" if cancelled else "parent_timeout" if parent_timeout else "child_process",
                              "message": stderr or f"child exited {returncode}",
                              "returncode": returncode},
                   "observations": _progress_observations(stderr)}
            append_jsonl(record_path, row)
        elif stderr:
            row.setdefault("observations", _progress_observations(stderr))
        row.setdefault("round", round_index)
        row["child_returncode"], row["child_stderr"] = returncode, stderr[-2000:]
        if parent_timeout:
            row["parent_timeout"] = True
        row["child_validation_failures"] = sum(
            item.get("status") == "failed" for item in new
            if item.get("type") == "validation" and item.get("arm_id") == arm_id)
        validation_failures += int(row["child_validation_failures"] or 0)
        rows[condition].append(row)
        if cancelled:
            cancellation = cancellation or {
                "type": "cancellation", "message": "benchmark interrupted"
            }
            row["interrupted"] = True
            row["cancellation"] = cancellation
            append_jsonl(record_path, {
                "type": "validation", "arm_id": arm_id, "status": "interrupted",
                "failure": cancellation,
            })
            interrupted = True
            interruption = {"arm_id": arm_id, **cancellation}
            break

        error_info = row.get("error") if isinstance(row, dict) else None
        is_resource_failure = (
            isinstance(row.get("resource_abort"), dict)
            or isinstance(error_info, dict)
            and error_info.get("type") in {"ResourceAbort", "QMMResourceError"}
        )
        if parent_timeout:
            stop_reason = {
                "arm_id": arm_id,
                "type": "parent_timeout",
                "returncode": returncode,
                "message": stderr[-1000:] or "silent child exceeded parent timeout",
            }
        elif returncode != 0:
            stop_reason = {
                "arm_id": arm_id,
                "type": "resource_abort" if is_resource_failure else "child_failure",
                "returncode": returncode,
                "message": stderr[-1000:] or f"child exited {returncode}",
            }
            if is_resource_failure:
                stop_reason["resource_abort"] = row.get("resource_abort", error_info)
        elif row.get("status") != "raw" or row.get("child_validation_failures"):
            stop_reason = {
                "arm_id": arm_id,
                "type": "arm_validation_failure",
                "message": "child arm was not a complete raw arm",
            }
        else:
            provenance_check = row.get("provenance_validation")
            if isinstance(provenance_check, dict):
                provenance_status = provenance_check.get("status")
                append_jsonl(record_path, {
                    "type": "validation", "arm_id": arm_id,
                    "status": "passed" if provenance_status == "matched" else "failed",
                    "check": "provenance",
                    "provenance_status": provenance_status,
                    "provenance": provenance_check,
                })
                if provenance_status != "matched":
                    validation_failures += 1
                    stop_reason = {
                        "arm_id": arm_id,
                        "type": (
                            "provenance_changed_during_arm"
                            if provenance_status == "changed"
                            else "provenance_unknown_during_arm"
                        ),
                        "failure": provenance_check,
                    }
            path_validation = _execution_path_validation(comparison, row)
            if stop_reason is None and path_validation is not None:
                passed = bool(path_validation["passed"])
                row["execution_path_validation"] = path_validation
                row["execution_path_valid"] = passed
                path_validated.add(arm_id)
                if not passed:
                    execution_path_failures += 1
                    validation_failures += 1
                    stop_reason = {
                        "arm_id": arm_id,
                        "type": "execution_path_failure",
                        "reason": path_validation.get("reason"),
                    }
                append_jsonl(record_path, {
                    "type": "validation", "arm_id": arm_id,
                    "status": "passed" if passed else "failed",
                    "check": "execution_path", **path_validation,
                })
            if stop_reason is None and sampling == "greedy":
                if condition == names[0] and expected is None:
                    expected = row.get("token_digest")
                elif expected is not None and row.get("token_digest") != expected:
                    digest_validated.add(arm_id)
                    validation_failures += 1
                    row["validation"] = {
                        "passed": False,
                        "reason": "token_mismatch",
                    }
                    append_jsonl(record_path, {
                        "type": "validation", "arm_id": arm_id,
                        "status": "failed", "check": "greedy_digest",
                        **row["validation"],
                    })
                    stop_reason = {
                        "arm_id": arm_id,
                        "type": "digest_failure",
                        "reason": "token_mismatch",
                    }
                elif expected is not None:
                    digest_validated.add(arm_id)
                    row["validation"] = {
                        "passed": True,
                        "reason": "digest matched",
                    }
                    append_jsonl(record_path, {
                        "type": "validation", "arm_id": arm_id,
                        "status": "passed", "check": "greedy_digest",
                        **row["validation"],
                    })

        if stop_reason is not None:
            print(
                "BonsaiStop " + json.dumps(stop_reason, sort_keys=True),
                flush=True,
            )
            append_jsonl(record_path, {
                "type": "validation", "arm_id": arm_id,
                "status": "failed", "check": "comparison_stop",
                "failure": stop_reason,
            })
            break

    for row in (item for name in names for item in rows[name]):
        if row.get("arm_id") in path_validated:
            continue
        path_validation = _execution_path_validation(comparison, row)
        if path_validation is None:
            continue
        passed = bool(path_validation["passed"])
        row["execution_path_validation"] = path_validation
        row["execution_path_valid"] = passed
        if not passed:
            execution_path_failures += 1
            validation_failures += 1
        append_jsonl(record_path, {
            "type": "validation", "arm_id": row["arm_id"],
            "status": "passed" if passed else "failed",
            "check": "execution_path", **path_validation,
        })

    def clean_arm(row):
        provenance = row.get("provenance_validation")
        return (
            row.get("status") == "raw"
            and row.get("child_returncode", 0) == 0
            and not row.get("child_validation_failures")
            and row.get("execution_path_valid", True)
            and isinstance(provenance, dict)
            and provenance.get("status") == "matched"
        )

    # A raw row can still have failed child validation. Never let such a row
    # define the digest expected from the clean reference arm.
    reference = next((row for row in rows[names[0]] if clean_arm(row)), None)
    if sampling == "greedy" and reference is not None and expected is None:
        expected = reference.get("token_digest")
    for row in (item for name in names for item in rows[name]):
        if sampling != "greedy" or not clean_arm(row) or row.get("arm_id") in digest_validated:
            continue
        passed = expected is not None and row.get("token_digest") == expected
        validation_failures += not passed
        row["validation"] = {"passed": passed, "reason": "digest matched" if passed else "token_mismatch"}
        append_jsonl(record_path, {"type": "validation", "arm_id": row["arm_id"],
                                   "status": "passed" if passed else "failed", **row["validation"]})

    def valid_for_stats(row):
        # Performance conclusions admit only arms that ran clean: raw
        # status alone still includes nonzero exits, child validation
        # failures, execution-path failures, and (under greedy) digest
        # mismatches.
        if (row.get("status") != "raw" or row.get("child_returncode", 0) != 0
                or row.get("child_validation_failures")
                or not row.get("execution_path_valid", True)
                or not isinstance(row.get("provenance_validation"), dict)
                or row["provenance_validation"].get("status") != "matched"):
            return False
        if sampling != "greedy":
            return True
        return bool(row.get("validation", {}).get("passed", False))

    valid = {name: [row for row in rows[name] if valid_for_stats(row)] for name in names}
    paired = {name: paired_stats(valid[names[0]], valid[name], names[0], name) for name in names[1:]}
    diagnostic_prefill = {
        name: paired_prefill_stats(
            [
                row for row in rows[names[0]]
                if row.get("execution_path_valid", True)
                and row.get("status") == "raw"
                and row.get("child_returncode", 0) == 0
                and not row.get("child_validation_failures")
                and isinstance(row.get("provenance_validation"), dict)
                and row["provenance_validation"].get("status") == "matched"
            ],
            [
                row for row in rows[name]
                if row.get("execution_path_valid", True)
                and row.get("status") == "raw"
                and row.get("child_returncode", 0) == 0
                and not row.get("child_validation_failures")
                and isinstance(row.get("provenance_validation"), dict)
                and row["provenance_validation"].get("status") == "matched"
            ],
            names[0], name,
        )
        for name in names[1:]
    }
    failed = sum(row.get("status") != "raw" or row.get("child_returncode", 0) != 0 or
                 row.get("child_validation_failures", 0) for values in rows.values() for row in values)
    summary = {"type": "summary", "status": "completed_with_failures" if failed or validation_failures or stop_reason else "completed",
               "metadata": meta, "conditions": rows, "paired": paired, "failed_arms": failed,
               "diagnostic_prefill": diagnostic_prefill,
               "validation_failures": validation_failures,
               "execution_path_failures": execution_path_failures,
               "expected_greedy_digest": expected}
    if interrupted:
        summary["status"] = "interrupted"
        summary["interruption"] = interruption
    elif stop_reason is not None:
        summary["stop_reason"] = stop_reason
    append_jsonl(record_path, summary)
    return summary
def child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    for flags, kwargs in ((
        (("--child",), {"action": "store_true"}), (("--checkpoint",), {"required": True}),
        (("--arm-id",), {"required": True}), (("--condition",), {"required": True}),
        (("--round",), {"type": int, "required": True}), (("--options-json",), {"required": True}),
        (("--record",), {"required": True}), (("--fixture",), {"choices": FIXTURES, "required": True}),
        (("--horizon",), {"type": int, "required": True}), (("--window",), {"type": int, "required": True}),
        (("--thinking",), {"action": "store_true"}), (("--effort",), {"default": "medium"}),
        (("--sampling",), {"choices": ("greedy", "sampled"), "default": "greedy"}),
        (("--prefill-step-size",), {"type": int, "default": 512}),
        (("--seed",), {"type": int, "default": SEED}),
        (("--resource-policy-json",), {"default": None}),
        (("--provenance-json",), {"default": None}),
        (("--operating-profile",), {"default": "diagnostic"}),
    )):
        parser.add_argument(*flags, **kwargs)
    args = parser.parse_args(argv)
    options = json.loads(args.options_json)
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    resource_limits = (
        json.loads(args.resource_policy_json)
        if args.resource_policy_json else resource_policy()
    )
    if not isinstance(resource_limits, dict):
        raise ValueError("resource policy must be an object")
    expected_provenance = (
        json.loads(args.provenance_json) if args.provenance_json else None
    )
    if expected_provenance is not None and not isinstance(expected_provenance, dict):
        raise ValueError("provenance must be an object")
    return child_arm(checkpoint=args.checkpoint, arm_id=args.arm_id, condition=args.condition,
        options=options, record_path=args.record, fixture=args.fixture, horizon=args.horizon,
        window=args.window, thinking=args.thinking, effort=args.effort, sampling=args.sampling,
        prefill_step_size=args.prefill_step_size, round_index=args.round, seed=args.seed,
        resource_limits=resource_limits, operating_profile_name=args.operating_profile,
        expected_provenance=expected_provenance)
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--child" in argv:
        return child_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("MACQWEN_BONSAI2_MODEL", "b2"))
    parser.add_argument("--compare", choices=sorted(COMPARISONS), default="allocator")
    parser.add_argument("--jsonl", default="bonsai2-benchmark.jsonl")
    parser.add_argument("--fixture", choices=FIXTURES, default=DEFAULT_FIXTURE)
    parser.add_argument("--horizon", choices=("short", "product", "tg128", "tg256"), default="short")
    parser.add_argument("--window", type=int, default=WINDOW); parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thinking", action="store_true"); parser.add_argument("--effort", default="medium")
    parser.add_argument("--sampling", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args(argv)
    horizons = {"short": SHORT, "product": PRODUCT, "tg128": 128, "tg256": 256}
    result = run_comparison(checkpoint=args.checkpoint, comparison=args.compare, record_path=args.jsonl,
        fixture=args.fixture, horizon=horizons[args.horizon],
        window=args.window, rounds=args.rounds, thinking=args.thinking, effort=args.effort,
        sampling=args.sampling, prefill_step_size=args.prefill_step_size, seed=args.seed,
        experiment_id=args.experiment_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "interrupted":
        return 128 + int(signal.SIGINT)
    return 0 if result["status"] == "completed" else 1
if __name__ == "__main__":
    raise SystemExit(main())
