#!/usr/bin/env python3
"""The standard production benchmark. One protocol, comparable numbers.

Decode rate on this machine varies with checkpoint residency and other
conditions. A rate published without its physical read volume therefore
cannot be compared against another rate, and a single two-arm A/B is not
enough to resolve an effect.

This harness enforces the protocol:

* alternates conditions to reduce ordering bias without assuming that drift
  affects every arm equally,
* discards the first arms of each condition, which establish the initial
  cache state,
* stops as soon as the median settles, instead of running a fixed count. The
  machine is fanless, and a longer run can introduce changing conditions;
  extra arms buy noise rather than confidence,
* records every arm with the seconds since the run began, and reports the
  correlation between rate and elapsed time as a drift diagnostic. The
  correlation does not identify its cause or make the comparison immune to
  environmental changes,
* reports median and range, never a bare mean,
* reports physical MB per token beside every rate,
* asserts identical token IDs across arms, so a changed runtime cannot pass
  by producing different work.

Examples:

    # what the code does right now
    python models/flashnext/tests/bench/bench_production.py --arms 12

    # does isolating n-gram traffic protect expert residency?
    python models/flashnext/tests/bench/bench_production.py --compare ngram-nocache

    # does warming last session's expert set help the first turn?
    python models/flashnext/tests/bench/bench_production.py --compare prewarm --fresh-arms
"""
from __future__ import annotations


import argparse
import atexit
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics as st
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from macqwen.results import output_path

from macqwen.measurement import MeasurementRun, validate_path

PROMPT = ("<|im_start|>user\nExplique a fotossintese em duas frases."
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

# A benchmark can stop before its final summary (for example, when an arm
# changes the token trajectory). Keep a small, JSON-serializable snapshot
# alive so the process can publish that failure from ``atexit``. The parent
# benchmark does not load a model merely by importing this module.
_ACTIVE_EVIDENCE = None
_MEASUREMENT_RUN = None
_MEASUREMENT_ARMS = set()


def write_evidence(path, payload: dict) -> None:
    """Atomically write benchmark evidence, including incomplete runs."""
    from pathlib import Path

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _flush_incomplete_evidence() -> None:
    state = _ACTIVE_EVIDENCE
    if state is None:
        return
    payload = state["payload"]
    if payload.get("status") != "running":
        return
    payload["status"] = "failed"
    payload.setdefault("failure", {
        "type": "incomplete_exit",
        "message": "benchmark exited before producing a complete summary",
    })
    try:
        current = runtime_source_fingerprints()
        payload["source_fingerprints_at_failure"] = current
        expected = payload.get("source_fingerprints")
        if expected is not None and current != expected:
            payload["source_changed"] = True
    except (OSError, TypeError, ValueError):
        pass
    try:
        write_evidence(state["path"], payload)
    except OSError:
        # Never hide the original benchmark error while trying to save its
        # evidence. A best-effort write is still useful for normal paths.
        pass


def _record_terminal_failure(error: BaseException) -> None:
    """Persist the exception that stopped a command-line benchmark."""
    global _MEASUREMENT_RUN, _MEASUREMENT_ARMS
    state = _ACTIVE_EVIDENCE
    if state is None:
        if _MEASUREMENT_RUN is not None:
            _MEASUREMENT_RUN.failure(
                str(error) or "benchmark terminated before completion",
                error_type=type(error).__name__,
            )
            _MEASUREMENT_RUN.finish("completed_with_failures")
            _MEASUREMENT_RUN = None
            _MEASUREMENT_ARMS.clear()
        return
    payload = state["payload"]
    payload["status"] = "failed"
    terminal_failure = {
        "type": type(error).__name__,
        "message": str(error) or "benchmark terminated before completion",
    }
    # A validation path can persist a specific failure (for example,
    # ``token_mismatch``) and then raise SystemExit. Keep that causal record;
    # replacing it here with the wrapper's generic exception loses the useful
    # evidence that the artifact was designed to preserve.
    if not isinstance(payload.get("failure"), dict):
        payload["failure"] = terminal_failure
    elif payload["failure"].get("type") in {None, "incomplete_exit"}:
        payload["failure"] = terminal_failure
    else:
        payload["terminal_failure"] = terminal_failure
    try:
        current = runtime_source_fingerprints()
        payload["source_fingerprints_at_failure"] = current
        expected = payload.get("source_fingerprints")
        if expected is not None and current != expected:
            payload["source_changed"] = True
    except (OSError, TypeError, ValueError):
        pass
    try:
        write_evidence(state["path"], payload)
    except OSError:
        pass
    if _MEASUREMENT_RUN is not None:
        _MEASUREMENT_RUN.failure(
            str(error) or "benchmark terminated before completion",
            error_type=type(error).__name__,
        )
        _MEASUREMENT_RUN.finish("completed_with_failures")
        _MEASUREMENT_RUN = None
        _MEASUREMENT_ARMS.clear()


atexit.register(_flush_incomplete_evidence)


def effective_chat_environment(process_environment=None) -> dict[str, str]:
    """Return normal-chat defaults while preserving explicit overrides."""
    from models.flashnext.settings.launch import CHAT_ENV

    source = os.environ if process_environment is None else process_environment
    environment = dict(CHAT_ENV)
    for key in CHAT_ENV:
        if key in source:
            environment[key] = str(source[key])
    return environment


def benchmark_harness_fingerprint(path=None) -> str:
    """Hash this benchmark harness separately from the model runtime."""
    source = Path(path or __file__).expanduser().resolve()
    digest = hashlib.sha256(b"flashnext-production-benchmark-v1")
    digest.update(source.name.encode("utf-8") + b"\0")
    digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def runtime_source_fingerprints() -> dict[str, str]:
    """Return fingerprints for the complete runtime and benchmark harness."""
    from models.flashnext.tests.bench.bench_chat_parity import source_fingerprint

    return {
        # The shared helper covers local Python/Metal runtime sources and
        # dependencies used by the FlashNext chat path. It intentionally
        # excludes bench_* files, so the production harness is recorded below.
        "runtime": source_fingerprint(),
        "benchmark": benchmark_harness_fingerprint(),
    }


def benchmark_provenance(checkpoint) -> dict:
    """Return provenance that applies to every production benchmark arm.

    The checkpoint identity only reads its config/index and shard metadata;
    it never opens model tensors.  ``source_fingerprint`` is the shared
    FlashNext runtime fingerprint used by the chat-parity harness and covers
    the local Python/Metal runtime sources.  Keep both the descriptive name
    and the historical alias in artifacts so old evidence remains readable.
    """
    from models.flashnext.slab_pack import checkpoint_identity

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    fingerprints = runtime_source_fingerprints()
    runtime_fingerprint = fingerprints["runtime"]
    identity = checkpoint_identity(checkpoint_path)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_identity": str(identity),
        "runtime_source_fingerprint": runtime_fingerprint,
        "source_fingerprint": runtime_fingerprint,
        "benchmark_source_fingerprint": fingerprints["benchmark"],
        "source_fingerprints": fingerprints,
    }

# These comparisons change routed experts and can change the token trajectory.
# They need a separate quality interpretation when their digests differ.
ROUTING_ALTERING_COMPARISONS = {
    "swap-resident", "swap-epsilon", "swap-epsilon-wide",
}

COMPARISONS = {
    "none": {"baseline": {}},
    "qsa-cache": {
        "baseline": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "0", "FLASHNEXT_QSA_SCATTER_DECODE": "0",
        },
        "cache-only": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "1", "FLASHNEXT_QSA_SCATTER_DECODE": "0",
        },
    },
    "qsa-scatter": {
        "baseline": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "0", "FLASHNEXT_QSA_SCATTER_DECODE": "0",
        },
        "scatter-only": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "0", "FLASHNEXT_QSA_SCATTER_DECODE": "1",
        },
    },
    "qsa": {
        "baseline": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "0", "FLASHNEXT_QSA_SCATTER_DECODE": "0",
        },
        "cache-only": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "1", "FLASHNEXT_QSA_SCATTER_DECODE": "0",
        },
        "scatter-only": {
            "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "0", "FLASHNEXT_QSA_SCATTER_DECODE": "1",
        },
    },
    "wired": {
        "wired0": {"FLASHNEXT_WIRED_GB": "0"},
        "wired2": {"FLASHNEXT_WIRED_GB": "2"},
    },
    "gpu-keepwarm": {
        "baseline": {"FLASHNEXT_GPU_KEEPWARM": "0"},
        "keepwarm": {"FLASHNEXT_GPU_KEEPWARM": "1"},
    },
    # Glue compilation at the top GPU clock. Both arms keep the GPU warm, so
    # dispatch savings are measured at P15 rather than at a collapsed clock.
    "glue-p15": {
        "baseline": {
            "FLASHNEXT_GPU_KEEPWARM": "1", "FLASHNEXT_COMPILE_HC": "0",
            "FLASHNEXT_COMPILE_NORM": "0", "FLASHNEXT_NORM_WEIGHT_CACHE": "0",
        },
        "compiled": {
            "FLASHNEXT_GPU_KEEPWARM": "1", "FLASHNEXT_COMPILE_HC": "1",
            "FLASHNEXT_COMPILE_NORM": "1", "FLASHNEXT_NORM_WEIGHT_CACHE": "1",
        },
    },
    # The input embedding streamed from the checkpoint instead of resident.
    # Load-time, so it needs --fresh-arms.
    "stream-embed": {
        "resident": {"FLASHNEXT_GPU_KEEPWARM": "1", "FLASHNEXT_STREAM_EMBED": "0"},
        "streamed": {"FLASHNEXT_GPU_KEEPWARM": "1", "FLASHNEXT_STREAM_EMBED": "1"},
    },
    "io-qos": {
        "default": {"FLASHNEXT_GPU_KEEPWARM": "1", "FLASHNEXT_IO_QOS": "default"},
        "interactive": {
            "FLASHNEXT_GPU_KEEPWARM": "1", "FLASHNEXT_IO_QOS": "user-interactive",
        },
    },
    "prewarm": {
        "baseline": {"FLASHNEXT_PREWARM": "0"},
        "prewarm": {"FLASHNEXT_PREWARM": "1"},
    },
    # The shared buffer removes the concatenate but is written by 16 workers
    # scattering across it, where the concatenate was one sequential copy. The
    # research log flags that difference as the unmeasured explanation for why
    # its 35 ms saving came back as GPU time. Chunk 2 and 4 keep most of the
    # queue depth while each worker writes a contiguous run, which is the
    # point the two settings were never tested at together.
    "buffer-chunk": {
        "concat-chunk1": {
            "FLASHNEXT_SHARED_READ_BUFFER": "0", "FLASHNEXT_PREAD_CHUNK": "1",
        },
        "buffer-chunk2": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "2",
        },
        "buffer-chunk4": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "4",
        },
    },
    # The three-way sweep above cannot report a band or a sign test, so the
    # winner is settled head to head against the current default.
    "buffer-chunk2": {
        "concat-chunk1": {
            "FLASHNEXT_SHARED_READ_BUFFER": "0", "FLASHNEXT_PREAD_CHUNK": "1",
        },
        "buffer-chunk2": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "2",
        },
    },
    "buffer-chunk2-vs-4": {
        "buffer-chunk2": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "2",
        },
        "buffer-chunk4": {
            "FLASHNEXT_SHARED_READ_BUFFER": "1", "FLASHNEXT_PREAD_CHUNK": "4",
        },
    },
    "metal-runtime": {
        # This is an MLX reference versus the opt-in custom executor on a
        # Q4/G32 checkpoint. It is intentionally not called "stock": both
        # arms are FlashNext and a Q4/G64 checkpoint would route both arms to
        # MLX while the guarded G64 flag remains disabled.
        "mlx-reference": {
            "FLASHNEXT_METAL_RUNTIME": "0",
        },
        "custom-runtime": {
            "FLASHNEXT_METAL_RUNTIME": "1",
        },
    },
    "g64-kernel": {
        "g64-reference": {
            "FLASHNEXT_METAL_RUNTIME": "1",
            "FLASHNEXT_METAL_G64": "0",
            "FLASHNEXT_SLAB_GLOBAL": "0",
            "FLASHNEXT_SLAB_PACK": "0",
            "FLASHNEXT_SLAB_G64": "0",
            "FLASHNEXT_STREAM_PACK": "0",
        },
        "g64-metal": {
            "FLASHNEXT_METAL_RUNTIME": "1",
            "FLASHNEXT_METAL_G64": "1",
            "FLASHNEXT_SLAB_GLOBAL": "0",
            "FLASHNEXT_SLAB_PACK": "0",
            "FLASHNEXT_SLAB_G64": "0",
            "FLASHNEXT_STREAM_PACK": "0",
        },
    },
    "slab-global": {
        "baseline": {"FLASHNEXT_METAL_RUNTIME": "1", "FLASHNEXT_SLAB_GLOBAL": "0"},
        "global48": {"FLASHNEXT_METAL_RUNTIME": "1", "FLASHNEXT_SLAB_GLOBAL": "48"},
    },
    # Compile the elementwise chains around the matmuls. Bit-exact, checked
    # with mx.array_equal before install. The probe's own per-call figures
    # sum to about 1 ms per token, so expect this inside the band.
    "compile": {
        "plain": {"FLASHNEXT_COMPILE": "0"},
        "compiled": {"FLASHNEXT_COMPILE": "1"},
    },
    # Cache-aware routing: take a resident expert when a cold one scores no
    # better. This changes what the model computes, so compare the text as
    # well as the rate. The tracker must be on for the gate to know anything.
    "swap-resident": {
        "exact": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "0",
        },
        "cache-aware": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
        },
    },
    # How far the swap may reach for a resident expert. At 0.02 it replaces
    # 13.9 percent of cold reads and at 0.005 it replaces 11.9. Above 0.02 is
    # unmeasured. Cache-aware measures 2.91 gen at 0.02, so 3.0 needs about 5
    # percent fewer bytes, from 360 to near 341.
    #
    # A wider epsilon takes experts the router scored further from its choice.
    # This trades answer quality for bytes, so run the quality gate on any
    # value that wins here. Do not promote one on rate alone.
    "swap-epsilon": {
        "e=0.02": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.02",
        },
        "e=0.05": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.05",
        },
    },
    "swap-epsilon-wide": {
        "e=0.02": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.02",
        },
        "e=0.10": {
            "FLASHNEXT_TRACK_RESIDENT": "1",
            "FLASHNEXT_SWAP_RESIDENT": "1",
            "FLASHNEXT_SWAP_EPSILON": "0.10",
        },
    },
    # Two changes that are each too small to resolve alone. Both act on how
    # much memory is left for the page cache, so they may not be independent:
    # pinning scales only frees 3.6 GB of mlock, and prewarm needs cache to
    # fill. Measuring the pair costs one comparison instead of two and can
    # resolve a combined effect that neither shows on its own.
    "stacked": {
        "neither": {"FLASHNEXT_PIN_PARTS": "all", "FLASHNEXT_PREWARM": "0"},
        "both": {"FLASHNEXT_PIN_PARTS": "scales", "FLASHNEXT_PREWARM": "1"},
    },
    # Spend the pin budget on scales and biases across many experts instead of
    # whole experts across few. Needs resident_experts raised and a candidate
    # pool that large, so pass --hot through the tail benchmark to compare
    # depths honestly.
    "pin-parts": {
        "whole-experts": {"FLASHNEXT_PIN_PARTS": "all"},
        "scales-only": {"FLASHNEXT_PIN_PARTS": "scales"},
    },
}


# Every setting a condition can flip, and how to make it real on a live
# backend. A setting read once at import or in a constructor cannot be changed
# by writing the environment, so each one is applied explicitly and then read
# back. The read-back is the point: a comparison that silently measured the
# same thing twice already produced one wrong result in this project.
def _qsa_bool(value):
    if value not in ("0", "1"):
        raise ValueError("QSA flags must be 0 or 1")
    return value == "1"


def _set_qsa_flag(name, value):
    from models.flashnext import qsa_chunk

    setattr(qsa_chunk, name, _qsa_bool(value))


def _get_qsa_flag(name):
    from models.flashnext import qsa_chunk

    return getattr(qsa_chunk, name)


LIVE_SETTINGS = {
    "FLASHNEXT_QSA_CACHE_POOLED_KEYS": (
        lambda backend, value: _set_qsa_flag("QSA_CACHE_POOLED_KEYS", value),
        lambda backend: _get_qsa_flag("QSA_CACHE_POOLED_KEYS"),
        _qsa_bool,
    ),
    "FLASHNEXT_QSA_SCATTER_DECODE": (
        lambda backend, value: _set_qsa_flag("QSA_SCATTER_DECODE", value),
        lambda backend: _get_qsa_flag("QSA_SCATTER_DECODE"),
        _qsa_bool,
    ),
    "FLASHNEXT_READ": (
        lambda backend, value: setattr(backend.store, "_read_mode", value),
        lambda backend: backend.store._read_mode,
        str,
    ),
    "FLASHNEXT_GPU_KEEPWARM": (
        # Read at call time from a list; spins change no model value.
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_gpu_keepwarm"]
        ).set_gpu_keepwarm(value == "1"),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["gpu_keepwarm"]
        ).gpu_keepwarm(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_COMPILE_HC": (
        lambda backend, value: __import__(
            "models.flashnext.compile_glue", fromlist=["set_enabled"]
        ).set_enabled(value == "1"),
        lambda backend: __import__(
            "models.flashnext.compile_glue", fromlist=["ENABLED"]
        ).ENABLED[0],
        lambda value: value == "1",
    ),
    "FLASHNEXT_COMPILE_NORM": (
        lambda backend, value: __import__(
            "models.flashnext.patch_rmsnorm", fromlist=["set_compile_norm"]
        ).set_compile_norm(value == "1"),
        lambda backend: __import__(
            "models.flashnext.patch_rmsnorm", fromlist=["compile_norm"]
        ).compile_norm(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_NORM_WEIGHT_CACHE": (
        lambda backend, value: __import__(
            "models.flashnext.patch_rmsnorm", fromlist=["set_weight_cache"]
        ).set_weight_cache(value == "1"),
        lambda backend: __import__(
            "models.flashnext.patch_rmsnorm", fromlist=["_WEIGHT_CACHE"]
        )._WEIGHT_CACHE[0],
        lambda value: value == "1",
    ),
    "FLASHNEXT_IO_QOS": (
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_io_qos"]
        ).set_io_qos(value),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["io_qos"]
        ).io_qos(),
        str,
    ),
    "FLASHNEXT_WIRED_GB": (
        # MLX wires nothing by default, so every buffer handed to the GPU is
        # evictable. The setter reads the limit back through Metal itself.
        lambda backend, value: __import__(
            "models.flashnext.loader", fromlist=["set_wired_gb"]
        ).set_wired_gb(value),
        lambda backend: __import__(
            "models.flashnext.loader", fromlist=["wired_gb"]
        ).wired_gb(),
        lambda value: float(value),
    ),
    "FLASHNEXT_SWAP_RESIDENT": (
        lambda backend, value: os.environ.__setitem__(
            "FLASHNEXT_SWAP_RESIDENT", value
        ),
        lambda backend: __import__(
            "models.flashnext.routing", fromlist=["swap_enabled"]
        ).swap_enabled(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_PIN_PARTS": (
        # read at call time by routing.pin_parts(); the environment is the
        # state, so setting it is enough, but it is still read back.
        lambda backend, value: os.environ.__setitem__("FLASHNEXT_PIN_PARTS", value),
        lambda backend: __import__(
            "models.flashnext.routing", fromlist=["pin_parts"]
        ).pin_parts(),
        lambda value: ("scales", "biases") if value == "scales"
        else ("weight", "scales", "biases"),
    ),
    "FLASHNEXT_TRACK_RESIDENT": (
        lambda backend, value: setattr(
            backend.store, "_track_residency", value == "1"
        ),
        lambda backend: backend.store._track_residency,
        lambda value: value == "1",
    ),
    "FLASHNEXT_SHARED_READ_BUFFER": (
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_shared_buffer"]
        ).set_shared_buffer(value == "1"),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["shared_buffer"]
        ).shared_buffer(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_PREAD_CHUNK": (
        lambda backend, value: setattr(
            backend.store, "_pread_chunk", int(value)
        ),
        lambda backend: backend.store._pread_chunk,
        lambda value: int(value),
    ),
    "FLASHNEXT_COMPILE": (
        lambda backend, value: (
            __import__(
                "models.flashnext.compiled", fromlist=["install"]
            ).install()
            if value == "1"
            else __import__(
                "models.flashnext.compiled", fromlist=["uninstall"]
            ).uninstall()
        ),
        lambda backend: __import__(
            "models.flashnext.compiled", fromlist=["installed"]
        ).installed(),
        lambda value: value == "1",
    ),
    "FLASHNEXT_SWAP_EPSILON": (
        # The environment is what `begin_decode` reads, but the router uses
        # the module value, so set both and read back the one that decides.
        lambda backend, value: (
            os.environ.__setitem__("FLASHNEXT_SWAP_EPSILON", value),
            __import__(
                "models.flashnext.adaptive_topk", fromlist=["_SWAP_EPSILON"]
            )._SWAP_EPSILON.__setitem__(0, float(value)),
        ),
        lambda backend: __import__(
            "models.flashnext.adaptive_topk", fromlist=["_SWAP_EPSILON"]
        )._SWAP_EPSILON[0],
        lambda value: float(value),
    ),
    "FLASHNEXT_METAL_RUNTIME": (
        lambda backend, value: __import__(
            "models.flashnext.expert_cache", fromlist=["set_metal_runtime"]
        ).set_metal_runtime(value == "1"),
        lambda backend: __import__(
            "models.flashnext.expert_cache", fromlist=["metal_runtime"]
        ).metal_runtime(),
        lambda value: value == "1",
    ),
}

# Settings that only take effect while the backend is built. A condition using
# one of these needs --fresh-arms; applying it to a live backend is a no-op.
LOAD_TIME_SETTINGS = {
    "FLASHNEXT_STREAM_EMBED",
    "FLASHNEXT_PREWARM",
    "FLASHNEXT_SLAB_GLOBAL",
    "FLASHNEXT_SLAB_MIN_SLOTS",
    "FLASHNEXT_METAL_G64",
    "FLASHNEXT_SLAB_G64",
    "FLASHNEXT_SLAB_PACK",
    "FLASHNEXT_STREAM_PACK",
    # Each arm gets a fresh backend so stale executor objects from the other
    # arm cannot make a live flip look like a distinct runtime comparison.
    "FLASHNEXT_METAL_RUNTIME",
}


def g64_kernel_status() -> str:
    """Return the executor verification state without compiling a kernel."""
    from models.flashnext import metal_runtime

    return "ready" if getattr(metal_runtime, "G64_RUNTIME_READY", False) else "verification"


def inspect_g64_runtime(backend, enabled: bool, phase: str = "after") -> dict:
    """Inspect every loaded layer before accepting a G64 benchmark arm."""
    capable_layers = 0
    switch_layers = 0
    paths = set()
    executor_count = 0
    executor_layers = set()
    slab_objects = 0
    group_sizes = set()
    for layer_index, layer in enumerate(backend.language.model.layers):
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if switch is None:
            continue
        switch_layers += 1
        if getattr(switch, "metal_runtime_capable", False):
            capable_layers += 1
        for projection in (
            getattr(switch, "gate_proj", None),
            getattr(switch, "up_proj", None),
            getattr(switch, "down_proj", None),
        ):
            if projection is not None and hasattr(projection, "group_size"):
                group_sizes.add(int(projection.group_size))
        for projection in (
            getattr(switch, "gate_proj", None),
            getattr(switch, "up_proj", None),
            getattr(switch, "down_proj", None),
        ):
            if getattr(projection, "slab", None) is not None:
                slab_objects += 1
        if getattr(switch, "slab_pack", None) is not None:
            slab_objects += 1
        executors = getattr(switch, "_metal_executors", {})
        executor_count += len(executors)
        if executors:
            executor_layers.add(layer_index)
        paths.update(
            getattr(executor, "last_path", "unknown")
            for executor in executors.values()
        )
    state = {
        "enabled": bool(enabled),
        "switch_layers": switch_layers,
        "capable_layers": capable_layers,
        "group_sizes": sorted(group_sizes),
        "executor_count": executor_count,
        "executor_layers": sorted(executor_layers),
        "paths": sorted(paths),
        "slab_objects": slab_objects,
    }
    expected_paths = {"custom-metal"} if enabled and phase != "before" else set()
    expected_capable = 48 if enabled else 0
    if state["switch_layers"] != 48:
        raise SystemExit(f"invalid FlashNext layer count: {state}")
    if state["capable_layers"] != expected_capable:
        raise SystemExit(f"invalid G64 capability state: {state}")
    if state["group_sizes"] != [64]:
        raise SystemExit(f"invalid G64 projection metadata: {state}")
    if phase == "before" and state["executor_count"]:
        raise SystemExit(f"G64 executors started before generation: {state}")
    expected_executor_layers = set(range(1, 48)) if enabled and phase != "before" else set()
    if set(state["executor_layers"]) != expected_executor_layers:
        raise SystemExit(f"incomplete G64 executor coverage: {state}")
    if set(state["paths"]) != expected_paths:
        raise SystemExit(f"invalid G64 executor paths: {state}")
    if state["slab_objects"]:
        raise SystemExit(f"G64 benchmark arm created slab state: {state}")
    return state


def check_g64_kernel_checkpoint(checkpoint: str) -> dict:
    """Validate a Q4/G64 checkpoint before creating a model backend."""
    from models.flashnext.tests.bench.bench_chat_parity import checkpoint_runtime_capability

    capability = checkpoint_runtime_capability(checkpoint)
    if capability.get("group_size") != 64:
        raise SystemExit(
            "g64-kernel requires a Q4/G64 checkpoint; refusing to load "
            f"{checkpoint}"
        )
    return {"status": g64_kernel_status(), "capability": capability}


def check_metal_runtime_checkpoint(checkpoint: str) -> dict:
    """Require a checkpoint on which the two runtime arms can differ.

    The generic custom executor currently has a Q4/G32 production contract.
    Q4/G64 remains behind its separate guarded comparison. Running the
    generic runtime comparison against Q4/G64 with that guard off would make
    both arms use MLX and would report an invalid A/A result.
    """
    from models.flashnext.tests.bench.bench_chat_parity import checkpoint_runtime_capability

    capability = checkpoint_runtime_capability(checkpoint)
    if capability.get("group_size") != 32:
        raise SystemExit(
            "metal-runtime requires a Q4/G32 checkpoint so its custom arm "
            "has a distinct executor; the selected checkpoint is "
            f"Q4/G{capability.get('group_size')} (use g64-kernel for the "
            "separate guarded experiment)"
        )
    return capability


def inspect_metal_runtime(backend, enabled: bool, phase: str = "after") -> dict:
    """Verify the actual generic Metal executor path for one benchmark arm."""
    switch_layers = 0
    capable_layers = 0
    paths = set()
    executor_layers = set()
    executor_count = 0
    group_sizes = set()
    for layer_index, layer in enumerate(backend.language.model.layers):
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if switch is None:
            continue
        switch_layers += 1
        if getattr(switch, "metal_runtime_capable", False):
            capable_layers += 1
        for projection in (
            getattr(switch, "gate_proj", None),
            getattr(switch, "up_proj", None),
            getattr(switch, "down_proj", None),
        ):
            if projection is not None and hasattr(projection, "group_size"):
                group_sizes.add(int(projection.group_size))
        executors = getattr(switch, "_metal_executors", {})
        executor_count += len(executors)
        if executors:
            executor_layers.add(layer_index)
        paths.update(
            getattr(executor, "last_path", "unknown")
            for executor in executors.values()
        )
    state = {
        "enabled": bool(enabled),
        "switch_layers": switch_layers,
        "capable_layers": capable_layers,
        "group_sizes": sorted(group_sizes),
        "executor_count": executor_count,
        "executor_layers": sorted(executor_layers),
        "paths": sorted(paths),
    }
    if switch_layers != 48:
        raise SystemExit(f"invalid FlashNext layer count: {state}")
    if enabled and capable_layers != switch_layers:
        raise SystemExit(
            "custom Metal runtime was requested but the checkpoint has no "
            f"executor capability on every routed layer: {state}"
        )
    expected_paths = {"custom-metal"} if enabled and phase != "before" else set()
    if set(state["paths"]) != expected_paths:
        raise SystemExit(f"invalid generic Metal executor paths: {state}")
    expected_layers = set(range(1, 48)) if expected_paths else set()
    if set(state["executor_layers"]) != expected_layers:
        raise SystemExit(f"incomplete generic Metal executor coverage: {state}")
    return state


def apply_condition(backend, env: dict) -> None:
    """Make a condition real on a live backend, then prove it took."""
    for key, value in env.items():
        if key in LOAD_TIME_SETTINGS:
            continue
        entry = LIVE_SETTINGS.get(key)
        if entry is None:
            continue
        setter, getter, expected = entry
        setter(backend, value)
        if getter(backend) != expected(value):
            raise SystemExit(
                f"{key}={value} did not take effect: "
                f"the runtime reports {getter(backend)!r}. "
                f"Refusing to report a comparison that did not happen."
            )


def check_load_time(env: dict, fresh_arms: bool) -> None:
    needed = LOAD_TIME_SETTINGS & set(env)
    if needed and not fresh_arms:
        raise SystemExit(
            f"{', '.join(sorted(needed))} only applies while the backend loads. "
            f"Pass --fresh-arms, or the two arms measure the same thing."
        )


def load_prompt(path=None):
    if path is None:
        prompt = PROMPT
        source = "PROMPT"
    else:
        source = str(Path(path).expanduser().resolve())
        prompt = Path(source).read_bytes().decode("utf-8")
        if not prompt.strip():
            raise ValueError("prompt file must not be empty")
    return prompt, {
        "prompt_source": source,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "chat_template": "preformatted; no additional template",
        "sampling": "greedy",
        "seed_policy": "deterministic argmax; no sampling seed",
    }


def require_source_freeze(provenance):
    expected = provenance.get("source_fingerprints") if provenance else None
    if expected is not None and runtime_source_fingerprints() != expected:
        raise RuntimeError("runtime source or benchmark harness changed before model execution")


def memory_snapshot():
    import resource
    import subprocess

    state = {"rss_mb": None, "rss_process_peak_mb": None,
             "active_mb": None, "mlx_process_peak_mb": None, "mlx_cache_mb": None}
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=True, timeout=5,
        )
        state["rss_mb"] = int(result.stdout.strip()) * 1024 / 1e6
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        state["rss_process_peak_mb"] = peak * (1 if sys.platform == "darwin" else 1024) / 1e6
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import mlx.core as mx
        for key, getter in (("active_mb", "get_active_memory"),
                            ("mlx_process_peak_mb", "get_peak_memory"),
                            ("mlx_cache_mb", "get_cache_memory")):
            method = getattr(mx, getter, None)
            if method is not None:
                state[key] = float(method()) / 1e6
    except (AttributeError, ImportError, TypeError):
        pass
    return state


def qsa_runtime_state(backend):
    from models.flashnext import qsa_chunk
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention, Qwen4ExpQSAIndexer

    rows = []
    caches = getattr(backend, "cache", ())
    layers = getattr(getattr(getattr(backend, "language", None), "model", None), "layers", ())
    for index, layer in enumerate(layers):
        indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if indexer is None:
            continue
        cache = caches[index] if index < len(caches) else None
        saved = getattr(cache, "_flashnext_pooled_keys", None)
        pooled = saved[4] if saved is not None else None
        offset = int(getattr(cache, "offset", 0))
        rows.append({
            "layer": index, "context_tokens": offset,
            "sparse_active": offset // indexer.compress_ratio > indexer.block_topk,
            "pooled_shape": list(pooled.shape) if pooled is not None else None,
            "pooled_bytes": int(pooled.nbytes) if pooled is not None else 0,
            "pool_reusable": pooled is not None and qsa_chunk._reusable_pool(indexer, cache) is pooled,
        })
    return {
        "cache_pooled_keys": qsa_chunk.QSA_CACHE_POOLED_KEYS,
        "scatter_decode": qsa_chunk.QSA_SCATTER_DECODE,
        "patch_installed": Qwen4ExpQSAIndexer.__call__ is qsa_chunk._indexer_call
        and Qwen4ExpAttention.__call__ is qsa_chunk._chunked_call,
        "layers": rows,
        "derived_bytes": sum(row["pooled_bytes"] for row in rows),
    }


def validate_qsa_state(state, condition, phase="after"):
    for name, key in (("cache_pooled_keys", "FLASHNEXT_QSA_CACHE_POOLED_KEYS"),
                      ("scatter_decode", "FLASHNEXT_QSA_SCATTER_DECODE")):
        if state[name] != _qsa_bool(condition[key]):
            raise RuntimeError(f"QSA flag mismatch: {name}")
    if not state["patch_installed"] or not state["layers"]:
        raise RuntimeError("QSA runtime patch or layers are missing")
    if phase == "before":
        if state["derived_bytes"] or any(row["context_tokens"] for row in state["layers"]):
            raise RuntimeError("QSA arm inherited cache state")
        return
    if not all(row["sparse_active"] for row in state["layers"]):
        raise RuntimeError("QSA comparison requires a longer prompt to activate sparse attention")
    if state["cache_pooled_keys"]:
        if not all(row["pooled_bytes"] > 0 and row["pool_reusable"] for row in state["layers"]):
            raise RuntimeError("QSA pooled cache did not retain reusable arrays on every layer")
    elif state["derived_bytes"]:
        raise RuntimeError("QSA cache-off arm retained pooled arrays")


def arm(backend, tokens, meter, run_began, condition=None,
        validate_metal_runtime: bool = False, on_raw_result=None,
        provenance: dict | None = None, prompt: str = PROMPT,
        validate_g64_runtime: bool = False):
    from models.flashnext.diskio import free_memory_mb, vm_counters

    require_source_freeze(provenance)

    free = free_memory_mb()
    backend.reset()
    if condition:
        # RoutingProfile.reset restores defaults. Apply the live condition at
        # the point where this arm starts, then validate its effective value.
        apply_condition(backend, condition)
        if validate_metal_runtime and "FLASHNEXT_METAL_RUNTIME" in condition:
            inspect_metal_runtime(
                backend, condition["FLASHNEXT_METAL_RUNTIME"] == "1", phase="before"
            )
        elif validate_g64_runtime and "FLASHNEXT_METAL_G64" in condition:
            inspect_g64_runtime(
                backend, condition["FLASHNEXT_METAL_G64"] == "1", phase="before"
            )
    qsa_condition = condition and "FLASHNEXT_QSA_CACHE_POOLED_KEYS" in condition
    if qsa_condition:
        validate_qsa_state(qsa_runtime_state(backend), condition, phase="before")
    backend.append_text(prompt)
    prompt_ids = tuple(getattr(backend, "pending", ()))
    if qsa_condition:
        for layer in backend.language.model.layers:
            indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
            if indexer is not None and len(prompt_ids) // indexer.compress_ratio <= indexer.block_topk:
                raise RuntimeError("QSA comparison needs a prompt above the sparse attention threshold")
        if backend.resident_experts != 32 or backend.routing_profile != "exact-quality":
            raise RuntimeError("QSA comparison requires exact-quality with 32 resident experts")
    memory_before = memory_snapshot()
    vm_before = vm_counters()
    vm_prefilled = {}
    prefill_memory = {}
    prefill_read = -1
    prefill_wall = None
    meter.reset()
    prefilled = False

    def on_prefilled():
        nonlocal prefilled, vm_prefilled, prefill_memory, prefill_read, prefill_wall
        prefill_wall = time.perf_counter() - began
        prefill_read = meter.bytes_since()
        vm_prefilled = vm_counters()
        prefill_memory = memory_snapshot()
        # Generation counters must exclude prompt reads. The backend invokes
        # this callback after prefill and before the first decoded token.
        meter.reset()
        prefilled = True

    began = time.perf_counter()
    _text, stats = backend.generate(
        max_tokens=tokens, on_prefilled=on_prefilled
    )
    wall = time.perf_counter() - began
    read = meter.bytes_since()
    generated_tokens = int(getattr(stats, "tokens", 0) or 0)
    generated_rate = float(getattr(stats, "rate", 0.0) or 0.0)
    ids = tuple(backend.tape[-generated_tokens:]) if generated_tokens else ()
    tail_tokens = int(getattr(stats, "tail_tokens", 0) or 0)
    tail_seconds = float(getattr(stats, "tail_seconds", 0.0) or 0.0)
    tail = tail_tokens / tail_seconds if tail_seconds else 0.0
    vm_after = vm_counters()
    memory_after = memory_snapshot()

    # Preserve the measurements before any post-generation validation.  In
    # particular, inspect_* can reject a path after generation has produced
    # useful tokens; the caller must still be able to write this raw arm.
    raw = {
        "elapsed_s": time.perf_counter() - run_began,
        "gen_tokens": generated_tokens,
        "gen_rate": generated_rate,
        "tail_rate": tail,
        "mb_per_token": (
            read / generated_tokens / 1e6
            if generated_tokens and read >= 0 else -1.0
        ),
        "pinned_gb": getattr(stats, "pinned_bytes", 0) / 1e9,
        **memory_after,
        "memory_before": memory_before,
        "memory_prefilled": prefill_memory,
        "vm_before": vm_before,
        "vm_prefilled": vm_prefilled,
        "vm_after": vm_after,
        "vm_counters": {key: vm_after[key] - value
                        for key, value in vm_prefilled.items() if key in vm_after},
        "prefill_vm_counters": {key: vm_prefilled[key] - value
                                for key, value in vm_before.items() if key in vm_prefilled},
        "prefill_wall_s": prefill_wall,
        "prefill_seconds": float(getattr(stats, "prefill_seconds", 0.0) or 0.0),
        "prefill_read_bytes": prefill_read,
        "decode_read_bytes": read,
        "prompt_tokens": int(getattr(stats, "prompt_tokens", len(prompt_ids)) or 0),
        "prompt_token_sha256": hashlib.sha256(str(prompt_ids).encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_sha256": hashlib.sha256(str(ids).encode("utf-8")).hexdigest(),
        "capable_layers": None,
        "actual_path": [],
        "free_mb_before": free,
        "wall": wall,
        "ids": ids,
    }
    if provenance:
        raw.update(provenance)
    if on_raw_result is not None:
        on_raw_result(raw)
    if qsa_condition:
        raw["qsa"] = qsa_runtime_state(backend)
        raw["qsa_derived_bytes"] = raw["qsa"]["derived_bytes"]

    # Only after the raw snapshot is published do we reject malformed or
    # incomplete measurements and validate the concrete executor path.
    if generated_tokens and not prefilled:
        raise RuntimeError("backend did not report the prefill boundary")
    if generated_tokens <= 0 or generated_rate <= 0:
        raise RuntimeError(
            f"generation produced no measurable tokens or rate: "
            f"tokens={generated_tokens}, rate={generated_rate}"
        )
    if read < 0:
        raise RuntimeError("physical read telemetry is unavailable")
    if qsa_condition:
        validate_qsa_state(raw["qsa"], condition)
    state = None
    if validate_metal_runtime and condition and "FLASHNEXT_METAL_RUNTIME" in condition:
        state = inspect_metal_runtime(
            backend, condition["FLASHNEXT_METAL_RUNTIME"] == "1"
        )
    elif validate_g64_runtime and condition and "FLASHNEXT_METAL_G64" in condition:
        state = inspect_g64_runtime(
            backend, condition["FLASHNEXT_METAL_G64"] == "1"
        )
    raw.update({
        "capable_layers": state["capable_layers"] if state else None,
        "actual_path": state["paths"] if state else [],
    })
    return raw


def settled(rates, window, tolerance) -> bool:
    """True once the median stops moving, so the run can stop early."""
    if len(rates) < window * 2:
        return False
    before = st.median(rates[:-1])
    now = st.median(rates)
    return abs(now - before) / now < tolerance


def elapsed_rate_correlation(arms):
    """Correlate rate with elapsed time without assigning a cause."""
    if len(arms) < 4:
        return 0.0
    xs = [a["elapsed_s"] for a in arms]
    ys = [a["gen_rate"] for a in arms]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


# Keep the old import name for small downstream diagnostics. New reports use
# the neutral ``elapsed_rate_correlation`` field instead of calling this
# measurement thermal drift.
thermal_drift = elapsed_rate_correlation


def report(name, arms, drop):
    kept = arms[drop:]
    rates = [a["gen_rate"] for a in kept]
    tails = [a["tail_rate"] for a in kept]
    mb = [a["mb_per_token"] for a in kept if a["mb_per_token"] >= 0]
    line = {
        "condition": name,
        "arms_run": len(arms),
        "arms_kept": len(kept),
        "gen_median": round(st.median(rates), 3),
        "gen_min": round(min(rates), 3),
        "gen_max": round(max(rates), 3),
        "gen_sd": round(st.stdev(rates), 3) if len(rates) > 1 else 0.0,
        "tail_median": round(st.median(tails), 3),
        "mb_per_token_median": round(st.median(mb), 1) if mb else -1.0,
        "free_mb_first": round(kept[0]["free_mb_before"], 0) if kept else -1,
        "elapsed_rate_correlation": round(elapsed_rate_correlation(kept), 2),
        "arms": [
            {k: (round(v, 3) if isinstance(v, float) else v)
             for k, v in a.items() if k != "ids"}
            for a in arms
        ],
    }
    print(
        f"  {name:<12} gen median {line['gen_median']:5.2f}  "
        f"range {line['gen_min']:.2f}-{line['gen_max']:.2f}  sd {line['gen_sd']:.3f}  "
        f"tail {line['tail_median']:5.2f}  "
        f"{line['mb_per_token_median']:6.1f} MB/tok  n={len(kept)}",
        flush=True,
    )
    return line


def resolution_note(base, other, drop: int = 0) -> str:
    """Report a matched paired two-SE band around the paired mean effect."""
    from math import sqrt

    base_arms = base["arms"][drop:]
    other_arms = other["arms"][drop:]
    if len(base_arms) != len(other_arms):
        return "matched paired resolution band unavailable: unequal pair counts"
    diffs = [
        (y["gen_rate"] - x["gen_rate"]) / x["gen_rate"] * 100
        for x, y in zip(base_arms, other_arms)
        if x["gen_rate"] > 0 and y["gen_rate"] > 0
    ]
    if len(diffs) != len(base_arms) or len(diffs) < 3:
        return "matched paired resolution band unavailable"
    effect = st.mean(diffs)
    band = 2 * st.stdev(diffs) / sqrt(len(diffs))
    if effect == 0.0 or band == 0.0:
        return (
            f"paired mean effect {effect:+.1f}% and band {band:.1f}% "
            "cannot resolve an effect"
        )
    if band > 10.0:
        return (
            f"matched paired two-SE band is {band:.1f}%, too wide for a "
            f"small-effect decision (mean {effect:+.1f}%)."
        )
    if abs(effect) >= band:
        return (
            f"paired mean effect {effect:+.1f}% resolves differences above "
            f"{band:.1f} percent."
        )
    return (
        f"paired mean effect {effect:+.1f}% is inside the matched paired "
        f"two-SE band of {band:.1f} percent, so this remains unresolved."
    )


def _two_sided_sign_p(diffs: list[float]) -> tuple[float, int, int, int]:
    """Return an exact two-sided sign-test p-value and its pair counts.

    Zero differences are ties and do not contribute to the null distribution.
    Keeping them out of the effective sample size avoids treating an exact tie
    as either an improvement or a regression while still reporting it to the
    caller.
    """
    from math import comb

    wins = sum(1 for difference in diffs if difference > 0)
    losses = sum(1 for difference in diffs if difference < 0)
    ties = len(diffs) - wins - losses
    total = wins + losses
    if total == 0:
        return 1.0, wins, losses, ties

    lower_tail = sum(comb(total, k) for k in range(wins + 1)) / 2 ** total
    upper_tail = sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total
    p_value = min(1.0, 2 * min(lower_tail, upper_tail))
    return p_value, wins, losses, ties


def report_paired(results, drop: int = 0) -> None:
    """Compare arm pairs taken at matched run positions.

    Alternating arms controls for shared run position, but it does not prove
    that environmental variation was eliminated. A sign test over the pairs
    needs no assumption about the spread.
    """
    if len(results) != 2:
        return
    base, other = results
    pairs = [
        (x["gen_rate"], y["gen_rate"])
        for x, y in zip(base["arms"][drop:], other["arms"][drop:])
    ]
    if len(pairs) < 3 or len(base["arms"][drop:]) != len(other["arms"][drop:]):
        print("  matched paired result unavailable: need three equal pairs")
        return
    diffs = [(y - x) / x * 100 for x, y in pairs]
    p_value, wins, losses, ties = _two_sided_sign_p(diffs)
    total = len(diffs)
    directional_total = wins + losses
    bytes_down = sum(
        1 for x, y in zip(base["arms"][drop:], other["arms"][drop:])
        if y["mb_per_token"] < x["mb_per_token"]
    )
    print()
    print(f"  paired over {total} arms: mean {st.mean(diffs):+.1f} percent, "
          f"median {st.median(diffs):+.1f}")
    print(f"  {other['condition']} improved in {wins} of {directional_total} "
          f"non-tied pairs and regressed in {losses}; ties: {ties}")
    print(f"  two-sided sign test p = {p_value:.3f}")
    print(f"  fewer bytes in {bytes_down} of {total} pairs")
    if directional_total == 0:
        print("  All matched pairs tied; there is no directional effect to resolve.")
    elif p_value > 0.05:
        if losses == 0:
            direction = "improvement"
        elif wins == 0:
            direction = "regression"
        else:
            direction = "improvement versus regression"
        print(
            f"  The {direction} direction is not statistically resolved at "
            f"p = {p_value:.3f}; collect more matched pairs."
        )


def report_drift(results) -> None:
    """Report elapsed-time correlations without inferring their cause.

    Conditions are interleaved, so a common time-varying influence can affect
    every arm. A condition-specific correlation is a diagnostic to investigate,
    not proof that the condition caused the change.
    """
    sliding = [r for r in results if r["elapsed_rate_correlation"] < -0.6]
    if not sliding:
        return
    print()
    if len(sliding) == len(results):
        print("  every condition has a negative elapsed-time correlation.")
        print("  This indicates a shared time-varying influence; its cause is")
        print("  not established, and absolute rates should be treated cautiously.")
        return
    for row in sliding:
        steady = [r for r in results if r not in sliding]
        print(f"  {row['condition']} has a negative elapsed-time correlation "
              f"(r={row['elapsed_rate_correlation']:+.2f}) while "
              f"{', '.join(r['condition'] for r in steady)} does not.")
        print("  This is a diagnostic association, not a causal conclusion;")
        print("  inspect the environment before interpreting the arm effect.")


def main() -> None:
    global _ACTIVE_EVIDENCE, _MEASUREMENT_RUN, _MEASUREMENT_ARMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=int, default=8,
                        help="ceiling on arms per condition; the run stops "
                             "earlier once the median settles")
    parser.add_argument("--min-arms", type=int, default=5,
                        help="arms per condition before early stopping applies")
    parser.add_argument("--tolerance", type=float, default=0.015,
                        help="stop once the median moves less than this fraction")
    parser.add_argument("--drop", type=int, default=2,
                        help="cold arms discarded per condition")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--prompt-file", help="UTF-8 preformatted prompt; used verbatim")
    parser.add_argument("--compare", choices=sorted(COMPARISONS), default="none")
    parser.add_argument("--fresh-arms", action="store_true",
                        help="reload the model for every arm in reversed rounds; "
                             "needed for settings that take effect at load")
    parser.add_argument("--json", default="", help="write the summary here")
    parser.add_argument(
        "--resident-experts", default="32",
        help="exact-quality pins per layer: an integer, or 'policy' for the "
             "checkpoint policy that normal chat applies (8 on Vontra)",
    )
    parser.add_argument(
        "--record", default="",
        help="write canonical append-only JSONL evidence under FlashNext measurements",
    )
    args = parser.parse_args()
    args.json = str(output_path("flashnext", "bench_production", "production.json", args.json))

    conditions = COMPARISONS[args.compare]
    routing_altering = args.compare in ROUTING_ALTERING_COMPARISONS
    effective_environment = effective_chat_environment()
    if args.compare in {"qsa", "qsa-cache", "qsa-scatter"}:
        effective_environment.update({
            "FLASHNEXT_METAL_RUNTIME": "1", "FLASHNEXT_METAL_G64": "0",
            "FLASHNEXT_SLAB_GLOBAL": "0",
            "FLASHNEXT_SLAB_PACK": "0", "FLASHNEXT_SLAB_G64": "0",
            "FLASHNEXT_STREAM_PACK": "0", "FLASHNEXT_PREWARM": "0",
        })
    prompt, prompt_provenance = load_prompt(args.prompt_file)
    evidence = {
        **prompt_provenance,
        "token_limit": args.tokens,
        "quality_gate": "not run; exact token equality only",
        "status": "running",
        "comparison": args.compare,
        "fresh_arms": args.fresh_arms,
        "routing_altering": routing_altering,
        "condition_definitions": conditions,
        "effective_environment": effective_environment,
        "conditions": [],
        "raw_arms": {name: [] for name in conditions},
    }
    if args.json:
        _ACTIVE_EVIDENCE = {"path": args.json, "payload": evidence}
        write_evidence(args.json, evidence)

    if args.arms <= 0:
        parser.error("--arms must be positive")
    if args.min_arms <= 0 or args.min_arms > args.arms:
        parser.error("--min-arms must be positive and no greater than --arms")
    if args.drop < 0 or args.drop >= args.min_arms:
        parser.error("--drop must be non-negative and less than --min-arms")
    if args.min_arms - args.drop < 3:
        parser.error("--min-arms must exceed --drop by at least three")
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    for env in conditions.values():
        check_load_time(env, args.fresh_arms)

    # Capture provenance for ordinary baseline runs too.  This metadata is
    # collected before a backend is constructed and checkpoint_identity only
    # reads metadata/stat information, so it does not run model inference.
    from macqwen.checkpoints import resolve_flashnext

    checkpoint = resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL"))
    if args.resident_experts == "policy":
        from models.flashnext.checkpoint_policy import resolve_resident_experts

        resident_experts = resolve_resident_experts(None, str(checkpoint)) or 32
    else:
        resident_experts = int(args.resident_experts)
    evidence["resident_experts"] = resident_experts
    provenance = benchmark_provenance(checkpoint)
    evidence.update(provenance)
    evidence["runtime_source_fingerprints"] = provenance["source_fingerprints"]
    if args.record:
        record_path = validate_path(
            Path(args.record), Path(__file__).resolve().parents[4], "flashnext"
        )
        _MEASUREMENT_RUN = MeasurementRun(
            record_path, runtime="flashnext", experiment=args.compare,
            metadata={
                "comparison": args.compare,
                "tokens": args.tokens,
                "arms": args.arms,
                "drop": args.drop,
                "fresh_arms": args.fresh_arms,
                "conditions": conditions,
                "provenance": provenance,
            },
        )
        _MEASUREMENT_RUN.start()
    if args.json:
        write_evidence(args.json, evidence)
    g64_preflight = None
    metal_preflight = None
    if args.compare == "g64-kernel":
        g64_preflight = check_g64_kernel_checkpoint(str(checkpoint))
        if g64_preflight["status"] != "ready":
            raise SystemExit(
                "g64-kernel executor is pending numerical verification; "
                "refusing to create a benchmark backend"
            )
    elif args.compare == "metal-runtime":
        metal_preflight = check_metal_runtime_checkpoint(str(checkpoint))

    from models.flashnext.diskio import ReadMeter

    meter = ReadMeter()
    run_began = time.perf_counter()
    results, collected, first_ids = [], {name: [] for name in conditions}, None
    if routing_altering:
        print(
            "  routing-altering comparison: digest changes are expected; "
            "apply the separate quality gate.",
            flush=True,
        )

    condition_keys = set(effective_environment).union(
        *(env.keys() for env in conditions.values())
    )
    initial_values = {
        key: effective_environment.get(key, os.environ.get(key))
        for key in condition_keys
    }

    def set_condition_environment(env, pin_path=None):
        for key in condition_keys:
            value = env.get(key, initial_values[key])
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if pin_path is not None:
            os.environ["FLASHNEXT_PIN_CACHE"] = str(pin_path)

    def persist_progress(round_index=None, name=None, failure=None):
        if _ACTIVE_EVIDENCE is None:
            return
        payload = _ACTIVE_EVIDENCE["payload"]
        payload["raw_arms"] = collected
        if round_index is not None and name is not None:
            payload["current_arm"] = {
                "round": round_index + 1,
                "condition": name,
                "environment": {
                    key: conditions[name].get(key, initial_values.get(key))
                    for key in condition_keys
                },
            }
        if failure is not None:
            payload["status"] = "failed"
            payload["failure"] = failure
        write_evidence(_ACTIVE_EVIDENCE["path"], payload)

    def publish_measurement_arm(round_index, name, row):
        if _MEASUREMENT_RUN is None:
            return
        key = (round_index, name)
        if key in _MEASUREMENT_ARMS:
            return
        _MEASUREMENT_ARMS.add(key)
        _MEASUREMENT_RUN.arm(
            arm_id=f"round-{round_index + 1}-{name}", condition=name,
            round_index=round_index,
            command=["models/flashnext/tests/bench/bench_production.py"],
            metrics={
                "common": {
                    "generation_rate_tps": row.get("gen_rate"),
                    "tail_rate_tps": row.get("tail_rate"),
                    "physical_mb_per_token": row.get("mb_per_token"),
                    "elapsed_seconds": row.get("elapsed_s"),
                },
                "flashnext": row,
            },
            tokens=list(row.get("ids", ())),
            token_digest=row.get("token_sha256"),
        )

    if args.fresh_arms:
        # Fresh mode creates a new backend for every arm, while the arm order
        # still alternates in reversed rounds. Stop only after a full round so
        # every condition has the same number of matched observations.
        from models.flashnext.routing import prewarm_enabled

        source_pin = os.environ.get(
            "FLASHNEXT_PIN_CACHE", "~/.cache/flashnext/pins.json"
        )
        source_path = os.path.expanduser(source_pin)
        source_bytes = b"{}"
        if os.path.isfile(source_path):
            with open(source_path, "rb") as handle:
                source_bytes = handle.read()

        with tempfile.TemporaryDirectory(prefix="flashnext-production-pins-") as pin_dir:
            cond_keys = list(conditions.keys())
            rounds_run = 0
            for round_index in range(args.arms):
                order = (
                    cond_keys if round_index % 2 == 0 else list(reversed(cond_keys))
                )
                for name in order:
                    env = conditions[name]
                    private_pin = os.path.join(
                        pin_dir, f"round-{round_index + 1}-{name}.json"
                    )
                    with open(private_pin, "wb") as handle:
                        handle.write(source_bytes)
                    set_condition_environment(env, private_pin)
                    persist_progress(round_index, name)
                    if "FLASHNEXT_PREWARM" in env:
                        want = env["FLASHNEXT_PREWARM"] == "1"
                        if prewarm_enabled() != want:
                            raise SystemExit(
                                f"condition {name} asked for prewarm={want} but the "
                                f"runtime reports {prewarm_enabled()}. Refusing to "
                                f"report a comparison that did not happen."
                            )
                    # Allocation caches are process globals and historically
                    # keyed only by slot count. Clear them between fresh arms
                    # so a private profile cannot inherit another arm's map.
                    try:
                        from models.flashnext import expert_cache
                        expert_cache._GLOBAL_SLAB_CACHE.clear()
                    except ImportError:
                        pass
                    # Import after the first condition environment is active.
                    from macqwen.backends.flashnext import FlashNextBackend
                    require_source_freeze(provenance)
                    backend = FlashNextBackend(resident_experts=resident_experts)
                    def preserve_raw(row, arm_name=name, arm_round=round_index):
                        collected[arm_name].append(row)
                        publish_measurement_arm(arm_round, arm_name, row)
                        persist_progress(arm_round, arm_name)
                    try:
                        row = arm(
                            backend, args.tokens, meter, run_began, condition=env,
                            validate_metal_runtime=args.compare == "metal-runtime",
                            validate_g64_runtime=args.compare == "g64-kernel",
                            on_raw_result=preserve_raw, provenance=provenance, prompt=prompt,
                        )
                    finally:
                        store = backend.store
                        del backend
                        gc.collect()
                        store.close()
                        del store
                        gc.collect()
                        try:
                            import mlx.core as mx
                            mx.clear_cache()
                        except AttributeError:
                            mx.metal.clear_cache()
                    # ``arm`` already published this same dict before its
                    # post-generation checks. Persist once more to capture
                    # validated executor metadata on successful completion.
                    persist_progress(round_index, name)
                    if first_ids is None:
                        first_ids = row["ids"]
                    if row["ids"] != first_ids:
                        message = f"  !! round {round_index + 1} of {name} produced different tokens"
                        if routing_altering:
                            print(message + " (expected for routing-altering run)")
                        else:
                            persist_progress(
                                round_index, name,
                                {"type": "token_mismatch", "message": message},
                            )
                            raise SystemExit(message + "; exact comparison rejected")
                rounds_run += 1
                if rounds_run >= args.min_arms:
                    if all(
                        settled(
                            [a["gen_rate"] for a in rows[args.drop:]],
                            args.min_arms // 2, args.tolerance,
                        )
                        for rows in collected.values()
                    ):
                        print(f"  every fresh median settled after {rounds_run} rounds")
                        break
    else:
        set_condition_environment(next(iter(conditions.values())))
        from macqwen.backends.flashnext import FlashNextBackend

        require_source_freeze(provenance)
        backend = FlashNextBackend(resident_experts=resident_experts)
        cond_keys = list(conditions.keys())
        print(f"  system load average before: {os.getloadavg()}", flush=True)

        rounds_run = 0
        for round_index in range(args.arms):
            order = (
                cond_keys if round_index % 2 == 0 else list(reversed(cond_keys))
            )
            for name in order:
                set_condition_environment(conditions[name])
                persist_progress(round_index, name)
                def preserve_raw(row, arm_name=name, arm_round=round_index):
                    collected[arm_name].append(row)
                    publish_measurement_arm(arm_round, arm_name, row)
                    persist_progress(arm_round, arm_name)
                row = arm(
                    backend, args.tokens, meter, run_began,
                    condition=conditions[name],
                    validate_metal_runtime=args.compare == "metal-runtime",
                    validate_g64_runtime=args.compare == "g64-kernel",
                    on_raw_result=preserve_raw, provenance=provenance, prompt=prompt,
                )
                # ``arm`` already published this same dict before its
                # post-generation checks. Persist once more after validation.
                persist_progress(round_index, name)
                if first_ids is None:
                    first_ids = row["ids"]
                if row["ids"] != first_ids:
                    message = (
                        f"  !! round {round_index + 1} of {name} "
                        "produced different tokens"
                    )
                    if routing_altering:
                        print(message + " (expected for routing-altering run)")
                    else:
                        persist_progress(
                            round_index, name,
                            {"type": "token_mismatch", "message": message},
                        )
                        raise SystemExit(message + "; exact comparison rejected")
            rounds_run += 1
            if rounds_run >= args.min_arms and all(
                settled(
                    [a["gen_rate"] for a in rows[args.drop:]],
                    args.min_arms // 2, args.tolerance,
                )
                for rows in collected.values()
            ):
                print(f"  every median settled after {rounds_run} rounds")
                break

    completion_fingerprints = runtime_source_fingerprints()
    if completion_fingerprints != provenance["source_fingerprints"]:
        failure = {
            "type": "source_changed",
            "message": (
                "runtime source or benchmark harness changed during the run"
            ),
        }
        if args.json:
            payload = _ACTIVE_EVIDENCE["payload"]
            payload["source_fingerprints_at_completion"] = completion_fingerprints
            persist_progress(failure=failure)
        raise RuntimeError(failure["message"])
    if args.json:
        _ACTIVE_EVIDENCE["payload"]["source_fingerprints_at_completion"] = (
            completion_fingerprints
        )

    print()
    for name, arms in collected.items():
        results.append(report(name, arms, args.drop))
        kept = arms[args.drop:]
        if kept and kept[0]["ids"]:
            h = hashlib.sha256(bytes(str(kept[0]["ids"]), "utf-8")).hexdigest()[:16]
            print(f"    token digest ({name}): {h}", flush=True)
    print(f"  system load average after: {os.getloadavg()}", flush=True)

    pairs = ([results[0], candidate] for candidate in results[1:])
    for base, other in pairs:
        report_paired([base, other], args.drop)
        change = (other["gen_median"] - base["gen_median"]) / base["gen_median"] * 100
        print(f"\n  {other['condition']} vs {base['condition']}: {change:+.1f}% gen median")
        print(f"  {resolution_note(base, other, args.drop)}")
    report_drift(results)

    if args.json:
        payload = _ACTIVE_EVIDENCE["payload"]
        payload.update({
            "status": "completed",
            "g64_preflight": g64_preflight,
            "g64_status": g64_kernel_status() if args.compare == "g64-kernel" else None,
            "metal_preflight": metal_preflight,
            "conditions": results,
            "raw_arms": collected,
        })
        write_evidence(args.json, payload)
        _ACTIVE_EVIDENCE = None
        print(f"\n  wrote {args.json}")
    if _MEASUREMENT_RUN is not None:
        _MEASUREMENT_RUN.finish(
            "completed", conditions=results, raw_arms=collected,
        )
        _MEASUREMENT_RUN = None
        _MEASUREMENT_ARMS.clear()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        _record_terminal_failure(error)
        raise
