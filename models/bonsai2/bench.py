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
import resource
import statistics as st
import subprocess
import sys
import time
import uuid
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
SHORT, PRODUCT, WINDOW, SEED = 32, 256, 32, 7
COMPARISONS = {
    "baseline": {"control": {}},
    "profile": {"control": {}, "cprofile": {"profile": True}},
    "allocator": {"control": {}, "allocator-256": {"allocator_cache_mb": 256}},
    "clear-cache": {"control": {}, "clear-cache-after": {"clear_cache_after_generate": True}},
    "wired": {"control": {}, "wired-limit": {"wired_limit_enabled": True}},
    "cache-step": {"step-256": {"cache_step": 256}, "step-1024": {"cache_step": 1024}},
    "prefill": {"prefill-512": {"prefill_step_size": 512}, "prefill-256": {"prefill_step_size": 256}},
    "prefill-wide": {"prefill-512": {"prefill_step_size": 512}, "prefill-1024": {"prefill_step_size": 1024}, "prefill-2048": {"prefill_step_size": 2048}},
    "fused-fwht": {"control": {"fused_fwht": False}, "fused": {"fused_fwht": True}},    "quant-kv8": {"control": {}, "qkv8": {"quantized_kv": [8, 64]}},
    "share-fwht": {"control": {}, "shared": {"share_fwht": True}},
    "gemv-decode": {"control": {}, "gemv": {"gemv": True}},}
DIAGNOSTIC_COMPARISONS = {"profile"}
_ANALYSIS_REQUEST = "Using the numbered records, write a detailed neutral analysis of at least 300 words covering the observed patterns and exceptions."
def _context_fixture(records: int) -> tuple[str, str, None, None]:
    context = "\n".join(f"Record {i:04d}: category {i % 16}; value {(i * 37) % 100}; status {('stable', 'review')[i % 2]}; note segment {i % 7}." for i in range(1, records + 1))
    return "Answer from the records.", f"{context}\n\n{_ANALYSIS_REQUEST}", None, None
FIXTURES = {
    "context-2k": _context_fixture(128), "context-8k": _context_fixture(384),
    "context-16k": _context_fixture(768),
    "smoke": (
        "Answer from the records.",
        "Record 0001: category 5; value 42; status stable.\n\n"
        "Summarize the single record in one sentence.",
        None, None,
    ),
    "cached-tool-result": ("Use tool results as context.", _ANALYSIS_REQUEST, '{"setting":"example","value":42}', None),
    "repeated-turn": ("Keep prior context.", _ANALYSIS_REQUEST, None, "Now repeat the recorded value with a detailed analysis of at least 300 words."),
}
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
    for name in ("config.json", "model.py", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "hadamard.json", "runtime/runtime.py", "runtime/artifact.py", "runtime/vision_artifact.py", "runtime/codec.py"):
        item = path / name
        try:
            files[name] = {"bytes": item.stat().st_size, "sha256": sha256(item)}
        except OSError:
            files[name] = {"bytes": None, "sha256": None}
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
    names = ("models/bonsai2/backend.py", "models/bonsai2/cache.py", "models/bonsai2/checkpoint.py", "models/bonsai2/protocol.py", "models/bonsai2/settings.py", "models/bonsai2/ternary_kernel.py")
    return {name: sha256(ROOT / name) for name in names}
def _module_origin(name: str) -> Path | None:
    try:
        parts = name.split(".")
        spec = importlib.util.find_spec(parts[0])
        roots = [Path(path) for path in spec.submodule_search_locations] if spec and spec.submodule_search_locations else ([Path(spec.origin).parent] if spec and spec.origin else [])
        relative = Path(*parts[1:])
        candidates = [candidate for root in roots for candidate in ((root / relative).with_suffix(".py"), root / relative / "__init__.py")]
        candidates += [candidate for root in roots for candidate in (root / relative.parent).glob(relative.name + ".*")]
        return next((candidate for candidate in candidates if candidate.is_file()), None)
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return None
def dependency_info() -> dict[str, Any]:
    result = {}
    for name in ("mlx", "mlx_lm", "transformers"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            origin = _module_origin(name)
            root = origin.parent if origin else None
            source = sha256(origin) if origin and origin.is_file() else None
        except (ImportError, OSError, TypeError, ValueError):
            root, source = None, None
        result[name] = {"version": version, "root": str(root) if root else None,
                        "source_sha256": source}
    for name in ("mlx.core", "mlx_lm.generate", "mlx_lm.models.cache"):
        origin = _module_origin(name)
        result[name] = {"origin": str(origin) if origin else None,
                        "source_sha256": sha256(origin) if origin else None}
    return result
def metadata(checkpoint: str, comparison: str, fixture: str, horizon: int,
             window: int, thinking: bool, effort: str, sampling: str,
             prefill_step_size: int, seed: int = SEED) -> dict[str, Any]:
    return {
        "schema": 1, "comparison": comparison, "checkpoint": checkpoint_info(checkpoint),
        "fixture": fixture, "horizon_tokens": horizon, "window_tokens": window,
        "reasoning": {"enabled": thinking, "effort": effort},
        "sampling_control": sampling,
        "sampled_chat": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
        "greedy_control": {"temperature": 0.0, "digest_required": sampling == "greedy"},
        "prefill_step_size": prefill_step_size, "seed": seed, "interpreter": sys.executable,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "runtime_source_fingerprints": source_fingerprints(),
        "dependency_info": dependency_info(),
        "harness_fingerprint": sha256(Path(__file__).resolve()),
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
def cache_metrics(backend: Any) -> dict[str, Any]:
    try:
        caches = list(getattr(backend, "cache", ()) or ())
    except TypeError:
        caches = []
    offsets, capacities, dtypes = [], [], []
    payload = allocated = capacity_bytes = 0
    for cache in caches:
        try:
            offset = int(cache.offset)
        except (AttributeError, TypeError, ValueError):
            offset = None
        keys, values = getattr(cache, "keys", None), getattr(cache, "values", None)
        key_shape, value_shape = _shape(keys), _shape(values)
        capacity = key_shape[-2] if key_shape and len(key_shape) >= 2 else None
        offsets.append(offset); capacities.append(capacity)
        dtypes.append(str(getattr(keys, "dtype", "")) or None)
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
            key_size, value_size = int(keys.dtype.itemsize), int(values.dtype.itemsize)
        except (AttributeError, TypeError, ValueError):
            continue
        per_token = (math.prod(key_shape[:-2]) * key_shape[-1] * key_size +
                     math.prod(value_shape[:-2]) * value_shape[-1] * value_size)
        payload += offset * per_token
        if capacity is not None:
            capacity_bytes += capacity * per_token
    return {"count": len(caches), "offsets": offsets, "capacities": capacities, "dtypes": dtypes,
            "allocated_bytes": allocated, "capacity_bytes": capacity_bytes, "payload_bytes": payload}
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
        intervals = [group[i]["at_s"] - group[i - 1]["at_s"] for i in range(1, len(group))]
        boundary = arrivals[start - 1]["at_s"] if start else 0.0
        elapsed = group[-1]["at_s"] - boundary
        ordered = sorted(intervals)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else None
        median = st.median(intervals) if intervals else None
        result.append({"window": start // window + 1, "start_token": start + 1, "end_token": start + len(group),
                       "tokens": len(group), "is_product_tail": start + 1 >= 33,
                       "rate_tps": len(group) / elapsed if elapsed > 0 else 0.0, "boundary_elapsed_s": elapsed,
                       "first_token_latency_s": group[0]["at_s"], "arrival_intervals_s": intervals,
                       "interval_median_s": median, "interval_p95_s": p95,
                       "ms_per_token_median": 1000.0 * median if median else None})
    return result
def _vm_delta(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(right.get(key, 0)) - int(left.get(key, 0)) for key in sorted(set(left) | set(right))}
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


def _constructor_options(options: dict[str, Any], default_prefill_step: int) -> tuple[dict[str, Any], int]:
    constructor = {k: v for k, v in options.items() if k not in ("cache_step", "profile")}
    return constructor, int(constructor.pop("prefill_step_size", default_prefill_step))
def child_arm(*, checkpoint: str, arm_id: str, condition: str, options: dict[str, Any],
              record_path: str, fixture: str, horizon: int, window: int,
              thinking: bool, effort: str, sampling: str, prefill_step_size: int,
              round_index: int, seed: int = SEED) -> int:
    started, backend, arrivals, raw_written = time.perf_counter(), None, [], False
    profiler = GenerationProfile() if options.get("profile") else None
    record = {"type": "arm", "schema": 1, "arm_id": arm_id, "condition": condition,
              "round": round_index, "status": "failed", "tokens": [],
              "token_digest": _digest([]), "error": None, "snapshots": {}, "windows": [], "seed": seed}
    try:
        from macqwen.sampling import Sampling
        from .backend import BonsaiBackend
        system, user, tool_result, repeat_user = FIXTURES[fixture]
        constructor, effective_prefill_step = _constructor_options(options, prefill_step_size)
        record["prefill_step_size"] = effective_prefill_step
        backend = BonsaiBackend(checkpoint, prefill_step_size=effective_prefill_step, **constructor)
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
            setup_tokens += backend.append_tool_results([tool_result])
        if repeat_user is not None:
            _text, setup_stats = backend.generate(max_tokens=1)
            setup_tokens += int(getattr(setup_stats, "tokens", 0) or 0)
            backend.append_user(repeat_user, enable_thinking=thinking)
        record.update({"template_tokens": template_tokens, "setup_tokens": setup_tokens,
                       "prompt_tokens": len(backend.pending), "options": options})
        prefill_at, generation_start = [None], time.perf_counter()
        record["snapshots"]["generation-start"] = snapshot(backend, "generation-start")
        read_start = _disk()
        def on_prefilled():
            prefill_at[0] = time.perf_counter()
            record["snapshots"]["prefill"] = snapshot(backend, "prefill", os_probes=False)
            if profiler is not None:
                profiler.start_decode()
        def on_token(value, _piece):
            arrivals.append({"token": int(value), "at_s": time.perf_counter() -
                             (prefill_at[0] or generation_start)})
        if profiler is not None:
            profiler.start()
        try:
            text, model_stats = backend.generate(max_tokens=horizon, on_prefilled=on_prefilled,
                                                 on_decode_token=on_token)
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
        tokens = [item["token"] for item in arrivals]
        count = int(getattr(model_stats, "tokens", 0) or 0)
        if count and len(tokens) != count:
            tokens = [int(x) for x in backend.tape[-count:]]
        start, decode = record["snapshots"]["generation-start"], record["snapshots"]["decode"]
        record.update({"status": "raw", "text": text, "tokens": tokens,
            "token_digest": _digest(tokens), "profile": bool(profiler), "stats": {
                "finish": getattr(model_stats, "finish", None), "tokens": count,
                "seconds": float(getattr(model_stats, "seconds", 0.0) or 0.0),
                "rate_tps": float(getattr(model_stats, "rate", 0.0) or 0.0),
                "prompt_tokens": int(getattr(model_stats, "prompt_tokens", 0) or 0),
                "prefill_seconds": float(getattr(model_stats, "prefill_seconds", 0.0) or 0.0)},
            "timing": {"arm_wall_s": time.perf_counter() - started,
                       "generation_wall_s": generation_done - generation_start,
                       "sync_s": sync_done - sync_start,
                       "first_token_latency_s": arrivals[0]["at_s"] if arrivals else None},
            "windows": _windows(arrivals, window),
            "runtime_source_fingerprints": source_fingerprints(),
            "dependency_info": dependency_info(),
            "metrics": {"physical_read_bytes": decode["physical_read_bytes"] - read_start if decode["physical_read_bytes"] is not None and read_start is not None else None,
                        "physical_read_bytes_generation": decode["physical_read_bytes"] - start["physical_read_bytes"] if decode["physical_read_bytes"] is not None and start["physical_read_bytes"] is not None else None,
                        "vm_delta": _vm_delta(start["vm_counters"], decode["vm_counters"]),
                        "cache": decode["cache"], "mlx": decode["mlx"], "process": decode["process"]}})
        append_jsonl(record_path, record); raw_written = True
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
        if arrivals:
            record["tokens"] = [item["token"] for item in arrivals]
            record["token_digest"] = _digest(record["tokens"])
            record["windows"] = _windows(arrivals, window)
        record["error"] = {"type": type(error).__name__, "message": str(error) or repr(error)}
        record["timing"] = {"arm_wall_s": time.perf_counter() - started}
        if not raw_written:
            append_jsonl(record_path, record)
        else:
            append_jsonl(record_path, {"type": "validation", "arm_id": arm_id,
                                       "status": "failed", "failure": record["error"]})
        return 1
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
def run_comparison(*, checkpoint: str, comparison: str, record_path: str, fixture="context-2k", horizon=SHORT, window=WINDOW, rounds=3,
                   thinking=False, effort="medium", sampling="greedy", prefill_step_size=512, seed=SEED, runner=None) -> dict[str, Any]:
    if comparison not in COMPARISONS or fixture not in FIXTURES:
        raise ValueError("unknown comparison or fixture")
    if horizon <= 0 or window <= 0 or horizon % window or rounds < 2:
        raise ValueError("invalid horizon, window, or rounds")
    conditions, meta = COMPARISONS[comparison], metadata(
        checkpoint, comparison, fixture, horizon, window, thinking, effort,
        sampling, prefill_step_size, seed)
    if comparison in DIAGNOSTIC_COMPARISONS:
        meta["purpose"] = "profiler overhead diagnostic; not an optimization comparison"
    append_jsonl(record_path, {"type": "run", "status": "started", "metadata": meta})
    rows = {name: [] for name in conditions}
    for round_index, condition in ordered_conditions(list(conditions), rounds):
        arm_id = f"round-{round_index + 1}-{condition}"
        command = [sys.executable, "-m", "models.bonsai2.bench", "--child",
                   "--checkpoint", checkpoint, "--arm-id", arm_id, "--condition", condition,
                   "--round", str(round_index), "--options-json", json.dumps(conditions[condition], sort_keys=True),
                   "--record", str(Path(record_path).expanduser()), "--fixture", fixture,
                   "--horizon", str(horizon), "--window", str(window), "--effort", effort,
                   "--sampling", sampling, "--prefill-step-size", str(prefill_step_size),
                   "--seed", str(seed)]
        if thinking:
            command.append("--thinking")
        before = read_jsonl(record_path)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(x for x in (str(ROOT), env.get("PYTHONPATH", "")) if x)
        try:
            result = (runner or subprocess.run)(command, cwd=str(ROOT), env=env,
                                                capture_output=True, text=True, check=False)
            returncode, stderr = int(getattr(result, "returncode", 0)), str(getattr(result, "stderr", "") or "")
        except BaseException as error:
            returncode, stderr = 1, f"{type(error).__name__}: {error}"
        new = read_jsonl(record_path)[len(before):]
        row = next((x for x in new if x.get("type") == "arm" and x.get("arm_id") == arm_id), None)
        if row is None:
            row = {"type": "arm", "schema": 1, "arm_id": arm_id, "condition": condition,
                   "round": round_index, "status": "failed", "tokens": [], "token_digest": _digest([]),
                   "error": {"type": "child_process", "message": stderr or f"child exited {returncode}",
                              "returncode": returncode}}
            append_jsonl(record_path, row)
        row.setdefault("round", round_index)
        row["child_returncode"], row["child_stderr"] = returncode, stderr[-2000:]
        row["child_validation_failures"] = sum(
            item.get("status") == "failed" for item in new
            if item.get("type") == "validation" and item.get("arm_id") == arm_id)
        rows[condition].append(row)
    names, expected, validation_failures = list(conditions), None, 0
    for row in (item for name in names for item in rows[name]):
        if sampling != "greedy" or row.get("status") != "raw":
            continue
        passed = expected is None or row.get("token_digest") == expected
        expected = expected or row.get("token_digest")
        validation_failures += not passed
        row["validation"] = {"passed": passed, "reason": "digest matched" if passed else "token_mismatch"}
        append_jsonl(record_path, {"type": "validation", "arm_id": row["arm_id"],
                                   "status": "passed" if passed else "failed", **row["validation"]})
    paired = {name: paired_stats(rows[names[0]], rows[name], names[0], name) for name in names[1:]}
    failed = sum(row.get("status") != "raw" or row.get("child_returncode", 0) != 0 or
                 row.get("child_validation_failures", 0) for values in rows.values() for row in values)
    summary = {"type": "summary", "status": "completed_with_failures" if failed or validation_failures else "completed",
               "metadata": meta, "conditions": rows, "paired": paired, "failed_arms": failed,
               "validation_failures": validation_failures, "expected_greedy_digest": expected}
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
    )):
        parser.add_argument(*flags, **kwargs)
    args = parser.parse_args(argv)
    options = json.loads(args.options_json)
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    return child_arm(checkpoint=args.checkpoint, arm_id=args.arm_id, condition=args.condition,
        options=options, record_path=args.record, fixture=args.fixture, horizon=args.horizon,
        window=args.window, thinking=args.thinking, effort=args.effort, sampling=args.sampling,
        prefill_step_size=args.prefill_step_size, round_index=args.round, seed=args.seed)
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--child" in argv:
        return child_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("MACQWEN_BONSAI2_MODEL", "b2"))
    parser.add_argument("--compare", choices=sorted(COMPARISONS), default="allocator")
    parser.add_argument("--jsonl", default="bonsai2-benchmark.jsonl")
    parser.add_argument("--fixture", choices=FIXTURES, default="context-2k")
    parser.add_argument("--horizon", choices=("short", "product", "tg128", "tg256"), default="short")
    parser.add_argument("--window", type=int, default=WINDOW); parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--thinking", action="store_true"); parser.add_argument("--effort", default="medium")
    parser.add_argument("--sampling", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    horizons = {"short": SHORT, "product": PRODUCT, "tg128": 128, "tg256": 256}
    result = run_comparison(checkpoint=args.checkpoint, comparison=args.compare, record_path=args.jsonl,
        fixture=args.fixture, horizon=horizons[args.horizon],
        window=args.window, rounds=args.rounds, thinking=args.thinking, effort=args.effort,
        sampling=args.sampling, prefill_step_size=args.prefill_step_size, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0
if __name__ == "__main__":
    raise SystemExit(main())
