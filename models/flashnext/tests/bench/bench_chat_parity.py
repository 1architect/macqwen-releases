#!/usr/bin/env python3
"""Attribute chat workload differences at the fixed 32-token horizon.

The parent imports no MLX module. Children start through chat.sh and use the
same interpreter, preset, private pin snapshot, and token cap. Parity and
workload modes use greedy sampling. Settings mode varies sampling and thinking.
Only formatted versus rendered is a same-trajectory comparison. Raw versus
formatted changes prompt tokens and measures a workload difference.
"""
from __future__ import annotations

from macqwen.results import output_path

import argparse
from collections import deque
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import pty
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time

from models.flashnext.settings.launch import CHAT_ENV

ROOT = Path(__file__).resolve().parents[4]
PROMPT = "Explique a fotossintese em duas frases."
CONDITIONS = ("raw", "formatted", "rendered")
SETTINGS_CONDITIONS = ("rendered", "sampled", "thinking", "thinking-sampled")
WORKLOAD_CONDITIONS = ("reference", "everyday")
PIN_CONDITIONS = ("pins32", "pins8")


def condition_settings(condition: str) -> tuple[bool, bool]:
    """Return the requested thinking and sampling state for one condition."""
    if condition not in set(CONDITIONS + SETTINGS_CONDITIONS + WORKLOAD_CONDITIONS + PIN_CONDITIONS):
        raise ValueError("unknown chat comparison condition")
    return condition in {"thinking", "thinking-sampled"}, condition in {
        "sampled", "thinking-sampled",
    }


def token_digest(ids) -> str:
    return hashlib.sha256(b"".join(
        int(token).to_bytes(4, "little") for token in ids
    )).hexdigest()


def source_fingerprint(root: Path = ROOT) -> str:
    """Hash repository runtime sources, excluding results and unrelated tests."""
    paths = {root / "chat.sh", root / "models/flashnext/tests/bench/bench_chat_parity.py"}
    for folder in (root / "macqwen", root / "models/flashnext"):
        for path in folder.rglob("*"):
            if path.suffix not in {".py", ".mm", ".metal", ".h", ".hpp"}:
                continue
            if set(path.relative_to(folder).parts) & {"tests", "__pycache__"}:
                continue
            if path.name.startswith(("test_", "bench_")):
                continue
            paths.add(path)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def benchmark_harness_fingerprint(root: Path = ROOT) -> str:
    digest = hashlib.sha256(b"flashnext-chat-parity-benchmark-v1")
    for name in ("bench_chat_parity.py", "bench_production.py"):
        path = root / "models/flashnext/tests/bench" / name
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def require_unchanged_harness(expected: str) -> None:
    if benchmark_harness_fingerprint() != expected:
        raise RuntimeError("Benchmark harness changed during the comparison")


def condition_environment(condition: str, capability=None) -> dict[str, str]:
    environment = dict(CHAT_ENV)
    if condition in PIN_CONDITIONS and capability is not None and capability["group_size"] == 64:
        environment.update({
            "FLASHNEXT_METAL_G64": "0",
            "FLASHNEXT_SLAB": "0",
            "FLASHNEXT_SLAB_GLOBAL": "0",
            "FLASHNEXT_SLAB_PACK": "0",
            "FLASHNEXT_SLAB_G64": "0",
            "FLASHNEXT_STREAM_PACK": "0",
        })
    return environment


def require_unchanged_source(expected: str, root: Path = ROOT) -> None:
    if source_fingerprint(root) != expected:
        raise RuntimeError(
            "Runtime source changed during the comparison. Stop editing before "
            "starting a new comparison; do not combine these arms with a later run."
        )


def write_evidence(path: Path, payload: dict) -> None:
    """Publish a complete JSON snapshot, including incomplete-run status."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def vm_warnings(records: list[dict]) -> list[dict]:
    """Report contaminated arms without silently dropping pairs."""
    warnings = []
    for row in records:
        counters = row.get("vm_counters", {})
        limit = max(256, int(row["tokens"]) * 8)
        excess = {key: counters[key] for key in ("swapin", "swapout", "pageout")
                  if counters.get(key, 0) > limit}
        missing = [key for key in ("swapin", "swapout", "pageout") if key not in counters]
        if excess or missing:
            warnings.append({
                "condition": row["condition"], "round": row.get("round"),
                "limit_pages": limit, "excess_pages": excess, "missing_counters": missing,
            })
    return warnings


def checkpoint_runtime_capability(checkpoint: str | Path) -> dict[str, object]:
    """Infer the slab capability from checkpoint metadata without loading MLX."""
    path = Path(checkpoint).expanduser()
    try:
        config = json.loads((path / "config.json").read_text())
        index = json.loads((path / "model.safetensors.index.json").read_text())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot inspect checkpoint metadata: {path}") from error
    model_type = config.get("model_type")
    quantization = config.get("quantization")
    if model_type not in {"qwen4_exp", "qwen4_exp_text"} or not isinstance(quantization, dict):
        raise ValueError(f"unsupported Flash-Next checkpoint metadata: {path}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weights: {path}")
    shards = set(weight_map.values())
    if not all((path / str(shard)).is_file() for shard in shards):
        raise ValueError(f"checkpoint is incomplete: {path}")
    try:
        bits = int(quantization["bits"])
        group_size = int(quantization["group_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint has malformed quantization metadata: {path}") from error
    if bits != 4 or group_size not in {32, 64} or quantization.get("mode", "affine") != "affine":
        raise ValueError(f"unsupported Flash-Next quantization layout: bits={bits}, group_size={group_size}")
    # Confirm every routed projection's header shape. This catches mixed layouts
    # that a top-level quantization setting cannot describe.
    routed = {
        name: shard for name, shard in weight_map.items()
        if ".mlp.switch_mlp." in name
    }
    nested = config.get("text_config") or config.get("llm_config") or config
    try:
        layer_count = int(nested["num_hidden_layers"])
        expert_count = int(nested["num_experts"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint lacks routed model dimensions: {path}") from error
    if layer_count != 48 or expert_count <= 0 or int(nested.get("hidden_size", 0)) != 2560 \
            or int(nested.get("moe_intermediate_size", 0)) != 640:
        raise ValueError(f"unsupported routed model dimensions: layers={layer_count}, experts={expert_count}")
    required = {
        f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.{part}"
        for layer in range(layer_count)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for part in ("weight", "scales", "biases")
    }
    if set(routed) != required:
        raise ValueError("checkpoint has incomplete routed projection headers")
    headers: dict[str, dict] = {}
    for name, shard in routed.items():
        shard_path = path / str(shard)
        if str(shard) not in headers:
            with shard_path.open("rb") as handle:
                raw_size = handle.read(8)
                if len(raw_size) != 8:
                    raise ValueError(f"checkpoint shard has no safetensors header: {shard_path}")
                header_size = struct.unpack("<Q", raw_size)[0]
                if header_size > 16 * 1024 * 1024:
                    raise ValueError(f"safetensors header is too large: {shard_path}")
                raw_header = handle.read(header_size)
            try:
                headers[str(shard)] = json.loads(raw_header)
            except (TypeError, ValueError) as error:
                raise ValueError(f"checkpoint shard has malformed safetensors header: {shard_path}") from error
        tensor = headers[str(shard)].get(name)
        if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
            raise ValueError(f"checkpoint lacks routed tensor header: {name}")
        projection, part = name.rsplit(".", 2)[-2:]
        gate_groups = 80 * 32 // group_size
        down_groups = 20 * 32 // group_size
        expected = {
            "gate_proj": {"weight": (expert_count, 640, 320), "scales": (expert_count, 640, gate_groups), "biases": (expert_count, 640, gate_groups)},
            "up_proj": {"weight": (expert_count, 640, 320), "scales": (expert_count, 640, gate_groups), "biases": (expert_count, 640, gate_groups)},
            "down_proj": {"weight": (expert_count, 2560, 80), "scales": (expert_count, 2560, down_groups), "biases": (expert_count, 2560, down_groups)},
        }
        shape = tuple(tensor["shape"])
        expected_dtype = "U32" if part == "weight" else "BF16"
        if (projection not in expected or shape != expected[projection][part]
                or tensor.get("dtype") != expected_dtype):
            raise ValueError(f"checkpoint has mixed routed layout: {name}")
    if (bits, group_size) == (4, 32):
        return {"group_size": 32, "slab_mode": "packed", "allocated_slots": 60, "mlock_ok": True}
    if (bits, group_size) == (4, 64):
        return {"group_size": 64, "slab_mode": "reference", "allocated_slots": 0, "mlock_ok": False}
    raise ValueError(f"unsupported Flash-Next quantization layout: bits={bits}, group_size={group_size}")


def _runtime_capability(records: list[dict], expected_runtime) -> dict[str, object] | None:
    """Resolve an explicitly requested capability for offline evidence recovery."""
    if expected_runtime is None:
        return None
    if expected_runtime == "checkpoint":
        checkpoints = {row.get("checkpoint") for row in records}
        if len(checkpoints) != 1 or not next(iter(checkpoints), None):
            raise ValueError("checkpoint capability inference requires one checkpoint")
        return checkpoint_runtime_capability(next(iter(checkpoints)))
    if isinstance(expected_runtime, (str, Path)):
        return checkpoint_runtime_capability(expected_runtime)
    raise ValueError("invalid expected runtime capability")


def run_chat_child(session, condition: str, prompt: str) -> None:
    """Measure the backend call inside the real plain-chat function."""
    import mlx.core as mx
    from macqwen import preferences
    from macqwen.sampling import Sampling
    from macqwen.session import open_or_continue, run_turn_plain
    from macqwen.ui import IngestGlow
    from models.flashnext.diskio import disk_bytes_read, vm_counters
    from models.flashnext.expert_cache import _POOL, profile_enabled

    thinking, sampled = condition_settings(condition)
    backend = session.backend
    backend.sampling = (
        Sampling.from_preferences(session.preferences) if sampled else Sampling.greedy_settings()
    )
    if sampled and backend.sampling.greedy:
        raise ValueError("the sampled condition requires non-greedy saved sampling settings")
    # Repeated conditions replay one sampled workload. This is not a quality gate.
    mx.random.seed(42)
    # This is a temporary diagnostic condition. Never save these preferences.
    session.preferences = dict(session.preferences)
    session.preferences.update(max_tokens=32, think_budget=-1, thinking_enabled=thinking)
    backend.thinking_enabled = thinking
    if preferences.generation_limit(session.preferences, 32) != 32:
        raise RuntimeError("chat parity requires the same 32-token total cap")
    original_generate = backend.generate
    original_pin = backend.routing._pin_candidates
    profile_pins = os.environ.get("FLASHNEXT_PROFILE_PINS") == "1"
    pin_events = []
    result = {}

    def measured_pin():
        wrapper_began = time.perf_counter()
        before_vm = vm_counters()
        before_bytes = disk_bytes_read()
        began = time.perf_counter()
        resident = original_pin()
        seconds = time.perf_counter() - began
        after_bytes = disk_bytes_read()
        after_vm = vm_counters()
        event = {
            "after_generated_tokens": len(backend.tape) - result["prompt_tokens"],
            "seconds": seconds,
            "physical_mb": (after_bytes - before_bytes) / 1e6
            if before_bytes >= 0 and after_bytes >= before_bytes else None,
            "pinned_mb": backend.routing.pinned_bytes / 1e6,
            "vm_counters": {key: after_vm[key] - before_vm[key]
                            for key in before_vm.keys() & after_vm.keys()},
            "active_mb": mx.get_active_memory() / 1e6,
            "cache_mb": mx.get_cache_memory() / 1e6,
        }
        event["diagnostic_overhead_seconds"] = time.perf_counter() - wrapper_began - seconds
        pin_events.append(event)
        return resident

    def measured_generate(*args, **kwargs):
        result["prompt_tokens"] = len(backend.pending)
        result["prompt_digest"] = token_digest(backend.pending)
        original_prefilled = kwargs.pop("on_prefilled", None)
        original_token = kwargs.pop("on_decode_token", None)
        start = {}
        phase_counts = {"thinking_tokens": 0, "answer_tokens": 0}
        inside_thinking = thinking

        def count_phase(value, piece):
            nonlocal inside_thinking
            phase_counts["thinking_tokens" if inside_thinking else "answer_tokens"] += 1
            if "</think>" in piece:
                inside_thinking = False
            if original_token is not None:
                original_token(value, piece)

        def begin_decode():
            if original_prefilled is not None:
                original_prefilled()
            start["vm"] = vm_counters()
            start["bytes"] = disk_bytes_read()
            start["time"] = time.perf_counter()

        text, stats = original_generate(
            *args, on_prefilled=begin_decode, on_decode_token=count_phase, **kwargs
        )
        ended = time.perf_counter()
        after_bytes = disk_bytes_read()
        after_vm = vm_counters()
        reads_valid = bool(start) and start["bytes"] >= 0 and after_bytes >= start["bytes"]
        vm_valid = bool(start.get("vm")) and bool(after_vm)
        pack = getattr(backend.store, "_slab_pack", None)
        switches = [
            layer.mlp.switch_mlp for layer in backend.language.model.layers
            if getattr(getattr(layer, "mlp", None), "switch_mlp", None) is not None
        ]
        slab_objects = sum(
            getattr(projection, "slab", None) is not None
            for switch in switches
            for projection in (switch.gate_proj, switch.up_proj, switch.down_proj)
        ) + sum(getattr(switch, "slab_pack", None) is not None for switch in switches)
        ids = backend.tape[-stats.tokens:] if stats.tokens else []
        result.update({
            "type": "chat-parity", "condition": condition,
            "tokens": stats.tokens, "digest": token_digest(ids),
            "gen_rate": stats.rate,
            "tail_rate": stats.tail_tokens / stats.tail_seconds if stats.tail_seconds else 0.0,
            "tail_tokens": stats.tail_tokens, "tail_seconds": stats.tail_seconds,
            "pinned_mb": stats.pinned_bytes / 1e6,
            "pinned_signature": stats.pinned_signature,
            "pin_budget_mb": backend.routing.pin_budget / 1e6,
            "resident_experts": backend.routing.resident_experts,
            "pin_warmup_tokens": backend.routing.warmup,
            "profile_pins": profile_pins, "pin_events": pin_events,
            "decode_wall_seconds": ended - start["time"] if start else None,
            "callback_seconds": stats.ui_seconds,
            "prefill_seconds": stats.prefill_seconds,
            "physical_mb_token": (after_bytes - start["bytes"]) / 1e6 / stats.tokens
            if stats.tokens and reads_valid else None,
            "active_mb": mx.get_active_memory() / 1e6,
            "cache_mb": mx.get_cache_memory() / 1e6,
            "vm_counters": {key: after_vm[key] - start["vm"][key]
                             for key in start.get("vm", {}).keys() & after_vm.keys()},
            "slab_objects": slab_objects,
            "executor_count": sum(len(getattr(switch, "_metal_executors", {})) for switch in switches),
            "environment": {key: os.environ.get(key) for key in CHAT_ENV},
            "allocation_digest": getattr(pack, "allocation_digest", None),
            "allocated_slots": getattr(pack, "expert_count", 0),
            "mlock_ok": bool(pack and pack.is_locked),
            "io_workers": _POOL._max_workers, "profile_io": profile_enabled(),
            "python": sys.executable, "checkpoint": backend.model_path,
            "sampling": "configured" if sampled else "greedy", "thinking": thinking,
            "sampling_settings": dict(vars(backend.sampling)), "seed": 42,
            **phase_counts,
            "effort": session.preferences["effort"],
            "animate": session.preferences["animate"],
            "stream_answers": session.preferences["stream_answers"],
            "render_tty": sys.stderr.isatty(),
        })
        if not reads_valid or not vm_valid:
            print(json.dumps(result), flush=True)
            raise RuntimeError("decode-only physical-read or VM measurement unavailable")
        return text, stats

    backend.generate = measured_generate
    if profile_pins:
        backend.routing._pin_candidates = measured_pin
    began = time.perf_counter()
    try:
        if condition in SETTINGS_CONDITIONS:
            # Includes the real glow, filters, animator, callback, and finish.
            with contextlib.redirect_stdout(sys.stderr):
                run_turn_plain(session, prompt, IngestGlow())
        else:
            if condition == "raw":
                backend.append_text(
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
            else:
                open_or_continue(session, prompt)
            backend.generate(max_tokens=32)
    finally:
        backend.generate = original_generate
        if profile_pins:
            backend.routing._pin_candidates = original_pin
    result["turn_wall_seconds"] = time.perf_counter() - began
    print(json.dumps(result), flush=True)
    if not result.get("tokens"):
        raise RuntimeError("chat parity arm produced no measured tokens")


def summarize(records: list[dict], mode: str = "parity", expected_runtime=None) -> dict:
    if mode not in {"parity", "settings", "workload", "pins"}:
        raise ValueError("unknown comparison mode")
    conditions = {"parity": CONDITIONS, "settings": SETTINGS_CONDITIONS,
                  "workload": WORKLOAD_CONDITIONS, "pins": PIN_CONDITIONS}[mode]
    if any(row["condition"] not in conditions for row in records):
        raise ValueError("unexpected comparison condition")
    grouped = {name: [row for row in records if row["condition"] == name]
               for name in conditions}
    counts = {len(rows) for rows in grouped.values()}
    if len(counts) != 1 or min(counts) < 3:
        raise ValueError("chat parity needs at least three complete reversed rounds")
    for name, rows in grouped.items():
        expected_thinking, expected_sampled = condition_settings(name)
        for row in rows:
            if row.get("thinking") != expected_thinking or row.get("sampling") != (
                    "configured" if expected_sampled else "greedy"):
                raise ValueError(f"{name} did not apply its thinking and sampling settings")
        if {row["tokens"] for row in rows} != {32}:
            raise ValueError(f"{name} did not complete the 32-token control")
        for field in ("prompt_digest", "digest"):
            if len({row[field] for row in rows}) != 1 or not rows[0][field]:
                raise ValueError(f"{name} changed {field} across rounds")
    for field in ("python", "checkpoint", "effort"):
        if len({row[field] for row in records}) != 1 or not records[0][field]:
            raise ValueError(f"startup mismatch: {field}")
    capability = _runtime_capability(records, expected_runtime)
    allocation_values = {row.get("allocation_digest") for row in records}
    if len(allocation_values) != 1:
        raise ValueError("startup mismatch: allocation_digest")
    if any(row["io_workers"] != 16 or row["profile_io"] for row in records):
        raise ValueError("runtime controls do not match the current preset")
    if capability is None:
        if any(row["allocated_slots"] != 60 or not row["mlock_ok"]
               or not row.get("allocation_digest") for row in records):
            raise ValueError("runtime controls do not match the current preset")
    elif capability["slab_mode"] == "packed":
        if any(not row.get("allocation_digest") or row["allocated_slots"] != 60
               or not row["mlock_ok"] for row in records):
            raise ValueError("runtime controls do not match the packed slab preset")
    else:
        if any("allocation_digest" not in row or row["allocation_digest"] is not None
               or "allocated_slots" not in row or row["allocated_slots"] != 0
               or "mlock_ok" not in row or row["mlock_ok"] for row in records):
            raise ValueError("reference-streaming evidence has mixed slab state")
    if mode == "pins" and capability is not None and capability["group_size"] == 64:
        controls = condition_environment("pins32", capability)
        keys = ("FLASHNEXT_METAL_RUNTIME", "FLASHNEXT_METAL_G64", "FLASHNEXT_SLAB",
                "FLASHNEXT_SLAB_GLOBAL", "FLASHNEXT_SLAB_PACK", "FLASHNEXT_SLAB_G64",
                "FLASHNEXT_STREAM_PACK")
        if any(row.get("environment", {}).get(key) != controls[key]
               for row in records for key in keys):
            raise ValueError("REAP pin runtime controls did not take effect")
        if any(row.get("slab_objects") != 0 or row.get("executor_count") != 0
               for row in records):
            raise ValueError("REAP pin comparison requires reference execution without slabs")
    rendered_rows = [row for row in records
                     if row["condition"] in SETTINGS_CONDITIONS + WORKLOAD_CONDITIONS + PIN_CONDITIONS]
    if any(not row["render_tty"] for row in rendered_rows):
        raise ValueError("rendering comparison did not use a terminal")
    result = {
        "conditions": {
            name: {key: statistics.median(row[key] for row in rows) for key in (
                "gen_rate", "tail_rate", "physical_mb_token", "active_mb",
                "decode_wall_seconds", "turn_wall_seconds", "callback_seconds",
            )} for name, rows in grouped.items()
        },
        "scope": "Raw versus formatted changes workload. No optimization promotion.",
    }
    if capability is not None:
        result["runtime_validation"] = {
            "source": "checkpoint config, index, and safetensors headers",
            "slab_mode": capability["slab_mode"],
            "allocated_slots": capability["allocated_slots"],
            "mlock_ok": capability["mlock_ok"],
        }
    warnings = vm_warnings(records)
    result["vm_warnings"] = warnings
    result["attribution_status"] = (
        "Memory-state contamination prevents causal attribution. Retain all raw arms."
        if warnings else
        "Swap/pageout advisory limit passed. Compressor counters still require inspection."
    )
    if any(row.get("profile_pins") for row in records):
        result["pin_profile_note"] = (
            "Pin timing is diagnostic. VM sampling adds overhead to the decode window. "
            "Inspect pin_events; do not use this run for a small-effect speed decision."
        )
    if mode == "parity":
        effects = []
        for quiet, rendered in zip(grouped["formatted"], grouped["rendered"]):
            if (quiet["prompt_digest"], quiet["digest"]) != (
                    rendered["prompt_digest"], rendered["digest"]):
                raise ValueError("rendering comparison changed prompt or generated token IDs")
            effects.append((rendered["gen_rate"] / quiet["gen_rate"] - 1) * 100)
        result.update({
            "rendering_mean_percent": statistics.mean(effects),
            "rendering_two_se_percent": 2 * statistics.stdev(effects) / math.sqrt(len(effects)),
            "rendering_paired_effects_percent": effects,
        })
    elif mode == "settings":
        for left, right in (("rendered", "sampled"), ("thinking", "thinking-sampled")):
            if grouped[left][0]["prompt_digest"] != grouped[right][0]["prompt_digest"]:
                raise ValueError("sampling comparison changed prompt tokens")
        if any(row.get("seed") != 42 for row in records):
            raise ValueError("settings comparison requires the fixed seed")
        configured = [row["sampling_settings"] for row in records
                      if row["sampling"] == "configured"]
        if any(value != configured[0] for value in configured):
            raise ValueError("saved sampling settings changed across arms")
        result["scope"] = (
            "Fixed-seed workload attribution at 32 tokens. Sampling and thinking "
            "can change generated tokens and expert reads. No quality or optimization claim."
        )
    elif mode == "workload":
        if grouped["reference"][0]["prompt_digest"] == grouped["everyday"][0]["prompt_digest"]:
            raise ValueError("everyday prompt matches the reference; workload premise failed")
        effects = [(daily["gen_rate"] / reference["gen_rate"] - 1) * 100
                   for reference, daily in zip(grouped["reference"], grouped["everyday"])]
        result.update({
            "workload_mean_percent": statistics.mean(effects),
            "workload_two_se_percent": 2 * statistics.stdev(effects) / math.sqrt(len(effects)),
            "workload_paired_effects_percent": effects,
            "scope": (
                "Prompt-content comparison at 32 tokens in fresh conversations. "
                "Both use the actual renderer, greedy sampling, and closed thinking. "
                "Different output tokens and expert reads are expected. No optimization claim."
            ),
        })
    else:
        effects, physical_deltas = [], []
        for control, candidate in zip(grouped["pins32"], grouped["pins8"]):
            if (control["prompt_digest"], control["digest"]) != (
                    candidate["prompt_digest"], candidate["digest"]):
                raise ValueError("pin comparison changed prompt or generated tokens")
            if control.get("resident_experts") != 32 or candidate.get("resident_experts") != 8:
                raise ValueError("pin count did not take effect")
            if not 0 < candidate.get("pinned_mb", 0) < control.get("pinned_mb", 0):
                raise ValueError("candidate did not reduce pinned memory")
            if control.get("profile_pins") or candidate.get("profile_pins"):
                raise ValueError("pin performance comparison requires pin profiling off")
            effects.append((candidate["gen_rate"] / control["gen_rate"] - 1) * 100)
            physical_deltas.append(control["physical_mb_token"] - candidate["physical_mb_token"])
        from models.flashnext.tests.bench.bench_production import _two_sided_sign_p

        p_value, wins, losses, ties = _two_sided_sign_p(effects)
        result.update({
            "pin_mean_percent": statistics.mean(effects),
            "pin_median_percent": statistics.median(effects),
            "pin_two_se_percent": 2 * statistics.stdev(effects) / math.sqrt(len(effects)),
            "pin_paired_effects_percent": effects,
            "pin_wins": wins,
            "pin_losses": losses,
            "pin_ties": ties,
            "pin_non_tied_pairs": wins + losses,
            "pin_sign_p": p_value,
            "paired_physical_reductions_mb_token": physical_deltas,
            "pinned_mb_medians": {name: statistics.median(row["pinned_mb"] for row in grouped[name])
                                  for name in PIN_CONDITIONS},
            "scope": (
                "Same-token comparison of 32 versus 8 pinned experts; "
                + ("REAP slabs and G64 executor remain off."
                   if capability is not None and capability["group_size"] == 64
                   else "60-slot slab remains fixed.")
            ),
            "attribution_status": (
                "Pin amount changes memory pressure by design. VM deltas are an outcome "
                "and remain system-wide. Inspect paired rates, reads, and VM activity together."
            ),
        })
    return result


def capture_arm(command: list[str], environment: dict[str, str]):
    """Keep the real renderer active even when the suite captures stdout."""
    master, slave = pty.openpty()
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=slave, text=True,
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)

    stderr_tail = deque(maxlen=16)

    def forward_terminal():
        try:
            while True:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    break
                if not data:
                    break
                stderr_tail.append(data)
                sys.stderr.buffer.write(data)
                sys.stderr.buffer.flush()
        finally:
            os.close(master)

    output_thread = threading.Thread(target=forward_terminal, daemon=True)
    output_thread.start()
    try:
        stdout, _ = process.communicate()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        output_thread.join()
    return process.returncode, stdout, b"".join(stderr_tail).decode(errors="replace")[-32768:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mode", choices=("parity", "settings", "workload", "pins"), default="parity")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--profile-pins", action="store_true",
                        help="measure expert pinning and VM movement; diagnostic only")
    args = parser.parse_args()
    args.json = output_path(
        "flashnext", "bench_chat_parity", f"chat-{args.mode}.json", args.json
    )
    if args.mode == "pins" and args.profile_pins:
        parser.error("pin performance comparison requires --profile-pins off")
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.expanduser().read_text().strip()
    if args.mode != "parity" and not args.prompt:
        parser.error("provide an everyday prompt with --prompt or --prompt-file")
    args.prompt = args.prompt or PROMPT
    if args.rounds < 3:
        parser.error("at least three reversed rounds are required")
    pins = Path(os.environ.get("FLASHNEXT_PIN_CACHE", "~/.cache/flashnext/pins.json")).expanduser().read_bytes()
    records = []
    conditions = {"parity": CONDITIONS, "settings": SETTINGS_CONDITIONS,
                  "workload": WORKLOAD_CONDITIONS, "pins": PIN_CONDITIONS}[args.mode]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    harness_fingerprint = benchmark_harness_fingerprint()
    payload = {
        "mode": args.mode, "planned_rounds": args.rounds, "status": "running",
        "source_fingerprint": fingerprint,
        "benchmark_source_fingerprint": harness_fingerprint, "records": records,
    }
    write_evidence(args.json, payload)
    try:
        capability = None
        checkpoint = None
        if args.mode == "pins":
            from macqwen.checkpoints import resolve_flashnext

            checkpoint = resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL"))
            capability = checkpoint_runtime_capability(checkpoint)
            payload["checkpoint"] = str(checkpoint)
            payload["runtime_capability"] = capability
        with tempfile.TemporaryDirectory(prefix="flashnext-chat-parity-") as folder:
            for index in range(args.rounds):
                order = conditions if index % 2 == 0 else tuple(reversed(conditions))
                for name in order:
                    payload["current_arm"] = {"round": index + 1, "condition": name}
                    require_unchanged_source(fingerprint)
                    require_unchanged_harness(harness_fingerprint)
                    pin_path = Path(folder) / f"pins-{index}-{name}.json"
                    pin_path.write_bytes(pins)
                    environment = dict(os.environ)
                    environment.update(condition_environment(name, capability))
                    if checkpoint is not None:
                        environment["MACQWEN_FLASHNEXT_MODEL"] = str(checkpoint)
                    environment.update({
                        "FLASHNEXT_PIN_CACHE": str(pin_path),
                        "FLASHNEXT_PROFILE_BOUNDARIES": "0",
                        "FLASHNEXT_PROFILE_SCORE_SYNC": "0",
                        "FLASHNEXT_IO_TASK_TOPOLOGY": "projection",
                        "FLASHNEXT_PROFILE_PINS": "1" if args.profile_pins else "0",
                    })
                    thinking, _sampled = condition_settings(name)
                    child_condition = "rendered" if args.mode in {"workload", "pins"} else name
                    prompt = PROMPT if args.mode == "workload" and name == "reference" else args.prompt
                    command = [str(ROOT / "chat.sh"), "--model", "flashnext", "--profile", "plain",
                               "--exact-quality", "--think" if thinking else "--no-think",
                               "--benchmark-chat-parity", child_condition,
                               "--benchmark-prompt", prompt]
                    if args.mode == "pins":
                        command.extend(("--resident-experts", "32" if name == "pins32" else "8"))
                    payload["current_arm"]["environment"] = {
                        key: environment.get(key) for key in CHAT_ENV
                    }
                    write_evidence(args.json, payload)
                    print(f"Round {index + 1}/{args.rounds}: {name}", flush=True)
                    returncode, stdout, stderr_tail = capture_arm(command, environment)
                    payload["last_child"] = {
                        "returncode": returncode, "stdout": stdout, "stderr_tail": stderr_tail,
                    }
                    write_evidence(args.json, payload)
                    if returncode:
                        payload["child_failure"] = payload["last_child"]
                        raise RuntimeError(f"{name} exited with {returncode}")
                    rows = []
                    for line in stdout.splitlines():
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row.get("type") == "chat-parity":
                            rows.append(row)
                    if len(rows) != 1:
                        raise RuntimeError(f"{name} did not return one measurement")
                    row = rows[0]
                    row["condition"] = name
                    row["round"] = index + 1
                    row["source_fingerprint"] = fingerprint
                    row["benchmark_source_fingerprint"] = harness_fingerprint
                    row["source_verified"] = False
                    records.append(row)
                    require_unchanged_source(fingerprint)
                    require_unchanged_harness(harness_fingerprint)
                    row["source_verified"] = True
                    print(json.dumps(row), flush=True)
                    write_evidence(args.json, payload)
        require_unchanged_source(fingerprint)
        require_unchanged_harness(harness_fingerprint)
        summary = summarize(records, args.mode, expected_runtime="checkpoint")
    except BaseException as error:
        payload["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        payload["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_evidence(args.json, payload)
        raise
    payload.update(status="complete", summary=summary)
    write_evidence(args.json, payload)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
