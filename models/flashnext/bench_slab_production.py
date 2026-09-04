#!/usr/bin/env python3
"""Controlled production A/B comparison of the winning selective slab.

Compares:
  - baseline: FLASHNEXT_METAL_RUNTIME=1, FLASHNEXT_SLAB=0
  - slab12:   FLASHNEXT_METAL_RUNTIME=1, FLASHNEXT_SLAB=4, FLASHNEXT_SLAB_LAYERS=12

Methodological controls:
1. Interleaved reversed pairs ([baseline, slab12], [slab12, baseline]...) to cancel thermal drift.
2. Separate instances cleanly closed and garbage-collected per arm.
3. Live physical disk telemetry via proc_pid_rusage (ReadMeter).
4. Full token digest tracking to guarantee exact numerical determinism.
5. Slab-pack allocation, locking, and VM telemetry.
6. Paired statistical sign test and resolution band reporting.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from math import comb
import os
from pathlib import Path
import re
import statistics as st
import struct
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import mlx.core as mx

from models.flashnext.boundary_profiler import BOUNDARY_LABELS

PROMPT = (
    "<|im_start|>user\nExplique a fotossintese em duas frases."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

CAPACITY_ARMS = (
    "slabpack56_skew",
    "slabpack60_skew",
    "slabpack64_skew",
)
CAPACITY_MANIFEST = Path("~/.cache/flashnext/capacity-sweep-manifest.json").expanduser()
CAPACITY_PINS = Path("~/.cache/flashnext/capacity-sweep-pins.json").expanduser()
CAPACITY_OBSERVED_PINS = Path(
    "~/.cache/flashnext/capacity-sweep-observed.json"
).expanduser()


def aggregate_boundary_profiles(
    snapshots: list[dict], tokens: int,
) -> dict:
    """Combine executor boundary totals and report milliseconds per token."""
    totals = {
        label: {
            "count": 0,
            "issue_ms": 0.0,
            "completion_ms": 0.0,
            "total_ms": 0.0,
        }
        for label in BOUNDARY_LABELS
    }
    selected = []
    enabled = False
    for snapshot in snapshots:
        enabled = enabled or bool(snapshot.get("enabled", False))
        for label in snapshot.get("selected", ()):
            if label in BOUNDARY_LABELS and label not in selected:
                selected.append(label)
        for label, values in snapshot.get("totals", {}).items():
            if label not in totals:
                continue
            total = totals[label]
            for key in ("count", "issue_ms", "completion_ms", "total_ms"):
                total[key] += values.get(key, 0)
    per_token = {}
    divisor = float(tokens) if tokens > 0 else 0.0
    for label, values in totals.items():
        per_token[label] = {
            "issue_ms": round(values["issue_ms"] / divisor, 3) if divisor else 0.0,
            "completion_ms": round(values["completion_ms"] / divisor, 3)
            if divisor else 0.0,
            "total_ms": round(values["total_ms"] / divisor, 3) if divisor else 0.0,
        }
    return {
        "enabled": enabled,
        "selected": selected,
        "executor_count": len(snapshots),
        "totals": totals,
        "per_token": per_token,
    }


def collect_boundary_profiles(backend) -> list[dict]:
    """Collect snapshots from every cached Metal executor in the backend."""
    snapshots = []
    language = getattr(backend, "language", None)
    layers = getattr(language, "layers", ()) if language is not None else ()
    seen = set()
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        switch = getattr(mlp, "switch_mlp", None)
        executors = getattr(switch, "_metal_executors", None)
        if not hasattr(executors, "values"):
            continue
        for executor in executors.values():
            if id(executor) in seen:
                continue
            seen.add(id(executor))
            profile = getattr(executor, "boundary_profile", None)
            if profile is not None:
                snapshots.append(profile)
    return snapshots


def system_boot_time() -> int:
    """Return the Darwin boot epoch used to enforce post-preparation reboot."""
    result = subprocess.run(
        ["sysctl", "-n", "kern.boottime"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    match = re.search(r"sec\s*=\s*(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse kern.boottime: {result.stdout!r}")
    return int(match.group(1))


def write_capacity_manifest(prepared: dict) -> None:
    """Record all prepared packs and the boot that created them."""
    CAPACITY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "boot_time": system_boot_time(),
        "prepared_at": time.time(),
        "routing_profile": {
            "path": str(CAPACITY_PINS),
            "digest": hashlib.sha256(CAPACITY_PINS.read_bytes()).hexdigest()[:16],
        },
        "packs": prepared,
    }
    temp_path = CAPACITY_MANIFEST.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, CAPACITY_MANIFEST)


def allocation_directory_digest(allocation: dict) -> str:
    """Match SlabPack.allocation_digest without opening the pack."""
    entries = []
    slot = 0
    for layer_id in sorted(allocation):
        for expert_id in allocation[layer_id]:
            entries.append((int(layer_id), int(expert_id), slot))
            slot += 1
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", len(entries), 3072000))
    for layer_id, expert_id, global_slot in sorted(entries):
        digest.update(struct.pack("<HHI", layer_id, expert_id, global_slot))
    return digest.hexdigest()[:16]


def require_prepared_capacity_packs(allow_same_boot: bool = False) -> None:
    """Refuse generation unless all capacity packs predate this boot."""
    if not CAPACITY_MANIFEST.is_file():
        raise RuntimeError(
            "Capacity manifest is missing. Run --capacity-sweep --prepare-only, "
            "then reboot."
        )
    payload = json.loads(CAPACITY_MANIFEST.read_text())
    if int(payload.get("version", 0)) != 2:
        raise RuntimeError(
            "Capacity manifest is obsolete. Run --capacity-sweep "
            "--prepare-only again."
        )
    routing_profile = payload.get("routing_profile", {})
    pin_path = Path(routing_profile.get("path", "")).expanduser()
    if not pin_path.is_file():
        raise RuntimeError(f"Capacity routing profile is missing: {pin_path}")
    pin_digest = hashlib.sha256(pin_path.read_bytes()).hexdigest()[:16]
    if pin_digest != routing_profile.get("digest"):
        raise RuntimeError(
            f"Capacity routing profile changed: got {pin_digest}, "
            f"expected {routing_profile.get('digest')}"
        )
    os.environ["FLASHNEXT_PIN_CACHE"] = str(pin_path.resolve())
    packs = payload.get("packs", {})
    os.environ["FLASHNEXT_SLAB_MIN_SLOTS"] = "4"
    os.environ["FLASHNEXT_SLAB_MAX_SLOTS"] = "6"
    os.environ["FLASHNEXT_SLAB_NUM_LAYERS"] = "12"
    os.environ["FLASHNEXT_WARM"] = "0"
    os.environ["FLASHNEXT_EARLY_SUBMIT"] = "0"
    from models.flashnext.expert_cache import get_skew_slab_allocation

    for name, budget in zip(CAPACITY_ARMS, (56, 60, 64)):
        info = packs.get(name)
        if not info:
            raise RuntimeError(f"Capacity manifest is missing {name}")
        path = Path(info["path"])
        if not path.is_file():
            raise RuntimeError(f"Prepared slab pack is missing: {path}")
        if path.stat().st_size != int(info["pack_bytes"]):
            raise RuntimeError(f"Prepared slab pack size changed: {path}")
        os.environ[f"FLASHNEXT_CAPACITY_PACK_{budget}"] = str(path.resolve())
        allocation = get_skew_slab_allocation(budget, min_slots=4)
        digest = allocation_directory_digest(allocation)
        if digest != info["allocation_digest"]:
            raise RuntimeError(
                f"Allocation changed for {name}: got {digest}, "
                f"expected {info['allocation_digest']}"
            )
    os.environ["FLASHNEXT_PIN_CACHE"] = str(CAPACITY_OBSERVED_PINS)
    prepared_boot = int(payload.get("boot_time", 0))
    current_boot = system_boot_time()
    if current_boot <= prepared_boot and not allow_same_boot:
        raise RuntimeError(
            "Capacity packs were prepared during this boot. Reboot before "
            "the trusted capacity sweep."
        )


def configure_arm(
    slab: int, layers: int, global_budget: int, slab_pack: bool,
    policy: str, fuse_shared: bool, fuse_shared_parts: bool, stream_pack: int,
    require_existing_pack: bool, boundary_profile: str | None = None,
    fuse_up_swiglu: bool = False,
) -> None:
    """Apply one complete slab configuration before importing the backend."""
    os.environ["FLASHNEXT_METAL_RUNTIME"] = "1"
    os.environ["FLASHNEXT_SLAB"] = str(slab)
    os.environ["FLASHNEXT_SLAB_LAYERS"] = str(layers)
    os.environ["FLASHNEXT_SLAB_GLOBAL"] = str(global_budget)
    os.environ["FLASHNEXT_SLAB_PACK"] = "1" if slab_pack else "0"
    os.environ["FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING"] = (
        "1" if slab_pack and require_existing_pack else "0"
    )
    expected_pack = os.environ.get(
        f"FLASHNEXT_CAPACITY_PACK_{global_budget}", ""
    )
    os.environ["FLASHNEXT_SLAB_PACK_EXPECTED_PATH"] = (
        expected_pack if slab_pack and require_existing_pack else ""
    )
    os.environ["FLASHNEXT_SLAB_POLICY"] = policy
    os.environ["FLASHNEXT_SLAB_MIN_SLOTS"] = "4"
    os.environ["FLASHNEXT_SLAB_MAX_SLOTS"] = "6"
    os.environ["FLASHNEXT_SLAB_NUM_LAYERS"] = "12"
    os.environ["FLASHNEXT_FUSED_SHARED"] = "1" if fuse_shared else "0"
    os.environ["FLASHNEXT_FUSED_SHARED_PARTS"] = (
        "1" if fuse_shared_parts else "0"
    )
    os.environ["FLASHNEXT_STREAM_PACK"] = "1" if stream_pack else "0"
    os.environ["FLASHNEXT_STREAM_PACK_CHUNK"] = str(max(0, stream_pack))
    os.environ["FLASHNEXT_FUSED_UP_SWIGLU"] = "1" if fuse_up_swiglu else "0"
    os.environ["FLASHNEXT_PREWARM"] = "0"
    os.environ["FLASHNEXT_WARM"] = "0"
    os.environ["FLASHNEXT_EARLY_SUBMIT"] = "0"
    os.environ["FLASHNEXT_PROFILE_IO"] = "1"
    os.environ.pop("FLASHNEXT_PROFILE_BOUNDARIES", None)
    if boundary_profile is None:
        os.environ.pop("FLASHNEXT_PROFILE_BOUNDARY", None)
    else:
        if boundary_profile not in BOUNDARY_LABELS:
            raise ValueError(f"unknown boundary profile: {boundary_profile}")
        os.environ["FLASHNEXT_PROFILE_BOUNDARY"] = boundary_profile


def close_store(store) -> None:
    """Close the store after its owning backend references are gone."""
    gc.collect()
    store.close()
    del store
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()


def validate_pack(backend, global_budget: int) -> dict:
    """Return trusted pack metadata or raise before generation starts."""
    store = getattr(backend, "store", None)
    pack = getattr(store, "_slab_pack", None)
    allocation = getattr(store, "_slab_alloc", None) or {}
    allocation_slots = sum(len(experts) for experts in allocation.values())
    if pack is None:
        raise RuntimeError("slab pack was not created")
    if not pack.is_locked:
        raise RuntimeError(f"mlock failed for {pack.path}")
    if allocation_slots != global_budget:
        raise RuntimeError(
            f"allocation contains {allocation_slots} slots, expected {global_budget}"
        )
    if pack.expert_count != global_budget:
        raise RuntimeError(
            f"pack contains {pack.expert_count} slots, expected {global_budget}"
        )
    if len(allocation) != 12:
        raise RuntimeError(
            f"allocation contains {len(allocation)} layers, expected 12"
        )
    for layer_id, experts in allocation.items():
        if not 4 <= len(experts) <= 6:
            raise RuntimeError(
                f"layer {layer_id} has {len(experts)} slots, expected 4..6"
            )
        if len(set(experts)) != len(experts):
            raise RuntimeError(f"layer {layer_id} contains duplicate experts")
    return {
        "allocation_digest": pack.allocation_digest,
        "allocated_slots": allocation_slots,
        "pack_bytes": int(pack.size),
        "pack_mib": pack.size / (1024 * 1024),
        "path": str(pack.path),
        "mlock_ok": True,
    }


def run_arm(
    slab: int, layers: int, tokens: int, global_budget: int = 0, slab_pack: bool = False,
    policy: str = "skew", fuse_shared: bool = True,
    fuse_shared_parts: bool = False, stream_pack: int = 0,
    boundary_profile: str | None = None,
    fuse_up_swiglu: bool = False,
) -> dict:
    configure_arm(
        slab, layers, global_budget, slab_pack, policy, fuse_shared,
        fuse_shared_parts, stream_pack,
        require_existing_pack=True,
        boundary_profile=boundary_profile,
        fuse_up_swiglu=fuse_up_swiglu,
    )

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext.diskio import ReadMeter, free_memory_mb, vm_counters
    from models.flashnext.expert_cache import (
        profile_enabled, profile_totals, reset_profile, set_profile,
    )

    set_profile(True)
    if not profile_enabled():
        raise RuntimeError("I/O profiling could not be enabled")

    free_before = free_memory_mb()
    meter = ReadMeter()
    backend = FlashNextBackend()

    if slab_pack:
        try:
            pack_info = validate_pack(backend, global_budget)
        except Exception:
            store = backend.store
            del backend
            close_store(store)
            raise
    else:
        pack_info = {
            "allocation_digest": "none",
            "allocated_slots": 0,
            "pack_bytes": 0,
            "pack_mib": 0.0,
            "mlock_ok": False,
        }

    backend.reset()
    backend.append_text(PROMPT)
    meter.reset()
    reset_profile()
    vm_before = vm_counters()

    began = time.perf_counter()
    _text, stats = backend.generate(max_tokens=tokens)
    wall = time.perf_counter() - began
    phys_bytes = meter.bytes_since()
    if phys_bytes < 0:
        store = backend.store
        del backend
        close_store(store)
        raise RuntimeError("Physical read telemetry is unavailable")
    vm_after = vm_counters()
    profile = profile_totals()
    boundary_summary = None
    if boundary_profile is not None:
        boundary_summary = aggregate_boundary_profiles(
            collect_boundary_profiles(backend), stats.tokens
        )

    # Collect slab statistics
    hits, misses = 0, 0
    language = getattr(backend, "language", None)
    if language is not None and hasattr(language, "layers"):
        for layer in language.layers:
            mlp = getattr(layer, "mlp", None)
            s_mlp = getattr(mlp, "switch_mlp", None)
            if s_mlp is not None and hasattr(s_mlp, "hits"):
                hits += s_mlp.hits
                misses += s_mlp.misses

    hit_pct = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0.0
    active_bytes = mx.metal.get_active_memory()
    phys_mb_tok = (phys_bytes / stats.tokens / 1e6) if stats.tokens and phys_bytes > 0 else 0.0

    ids = tuple(backend.tape[-stats.tokens:]) if stats.tokens else ()
    digest = hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest()[:16] if ids else "none"
    tail_rate = (stats.tail_tokens / stats.tail_seconds) if stats.tail_seconds else 0.0

    io_wait_s = float(profile.get("io_wait", 0.0))
    profile_ms_tok = {
        key: round(float(profile.get(key, 0.0)) / stats.tokens * 1000, 3)
        if stats.tokens else 0.0
        for key in (
            "critical_queue", "critical_pread", "critical_task_overhead",
            "completion_overhead", "queue_delay_sum", "task_service_sum",
            "pread_service_sum",
        )
    }
    vm_delta = {
        key: vm_after.get(key, 0) - vm_before.get(key, 0)
        for key in sorted(set(vm_before) | set(vm_after))
    }
    vm_per_token = {
        key: round(value / stats.tokens, 3) if stats.tokens else 0.0
        for key, value in vm_delta.items()
    }

    language = None
    layer = mlp = s_mlp = None
    store = backend.store
    del backend
    close_store(store)

    result = {
        "slab": slab,
        "layers": layers,
        "global_budget": global_budget,
        "tokens": stats.tokens,
        "gen_rate": round(stats.rate, 3),
        "tail_rate": round(tail_rate, 3),
        "phys_mb_tok": round(phys_mb_tok, 1),
        "free_mb_before": round(free_before, 0),
        "active_mb": round(active_bytes / 1e6, 1),
        "hit_pct": round(hit_pct, 1),
        "digest": digest,
        "allocation_digest": pack_info["allocation_digest"],
        "allocated_slots": pack_info["allocated_slots"],
        "pack_bytes": pack_info["pack_bytes"],
        "pack_mib": round(pack_info["pack_mib"], 2),
        "mlock_ok": pack_info["mlock_ok"],
        "io_wait_s": round(io_wait_s, 6),
        "io_wait_ms_tok": round(io_wait_s / stats.tokens * 1000, 3) if stats.tokens else 0.0,
        "io_breakdown_ms_tok": profile_ms_tok,
        "read_tasks_tok": round(float(profile.get("read_tasks", 0)) / stats.tokens, 3)
        if stats.tokens else 0.0,
        "pread_calls_tok": round(float(profile.get("pread_calls", 0)) / stats.tokens, 3)
        if stats.tokens else 0.0,
        "pread_mb_tok": round(float(profile.get("pread_bytes", 0)) / stats.tokens / 1e6, 3)
        if stats.tokens else 0.0,
        "vm_counters": vm_delta,
        "vm_per_token": vm_per_token,
        "wall_s": round(wall, 2),
    }
    if boundary_summary is not None:
        result["boundary_profile"] = boundary_summary
    return result


def prepare_pack(name: str, definition: tuple) -> dict:
    """Create and validate one cached pack without running generation."""
    (
        slab, layers, global_budget, slab_pack, policy, fuse_shared,
        fuse_shared_parts, stream_pack,
    ) = definition
    if not slab_pack:
        raise ValueError(f"{name} is not a slab-pack configuration")
    configure_arm(
        slab, layers, global_budget, slab_pack, policy, fuse_shared,
        fuse_shared_parts, stream_pack,
        require_existing_pack=False,
    )
    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend()
    try:
        return validate_pack(backend, global_budget)
    finally:
        store = backend.store
        del backend
        close_store(store)


def calibrate_routing_profile() -> None:
    """Refresh pins from the benchmark prompt before extracting slab packs."""
    CAPACITY_PINS.parent.mkdir(parents=True, exist_ok=True)
    temporary_pins = CAPACITY_PINS.with_suffix(".calibrating.json")
    temporary_pins.unlink(missing_ok=True)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(temporary_pins)
    configure_arm(
        slab=0, layers=0, global_budget=0, slab_pack=False,
        policy="uniform", fuse_shared=True, fuse_shared_parts=False,
        stream_pack=False,
        require_existing_pack=False,
    )
    os.environ["FLASHNEXT_PREWARM"] = "0"
    from macqwen.backends.flashnext import FlashNextBackend

    backend = None
    try:
        print("Calibrating routing profile from benchmark PROMPT...", flush=True)
        backend = FlashNextBackend(
            routing_profile="exact-quality",
            resident_experts=32,
            pin_budget_gb=6.0,
        )
        backend.reset()
        backend.append_text(PROMPT)
        _text, stats = backend.generate(max_tokens=16)
        if stats.tokens < 9:
            raise RuntimeError(
                "Routing calibration stopped before the nine-token warmup"
            )
        if not temporary_pins.is_file():
            raise RuntimeError(
                f"Routing calibration did not write {temporary_pins}"
            )
        payload = json.loads(temporary_pins.read_text())
        ranked_counts = payload.get("ranked_counts", {})
        if payload.get("mode") != "exact-quality" or len(ranked_counts) != 48:
            raise RuntimeError("Routing calibration wrote an incomplete profile")
        if any(not ranked_counts.get(str(layer)) for layer in range(48)):
            raise RuntimeError("Routing calibration omitted one or more layers")
        os.replace(temporary_pins, CAPACITY_PINS)
        os.environ["FLASHNEXT_PIN_CACHE"] = str(CAPACITY_PINS)
        pin_digest = hashlib.sha256(CAPACITY_PINS.read_bytes()).hexdigest()[:16]
        print(
            f"  -> Calibrated with {stats.tokens} tokens: {CAPACITY_PINS} "
            f"({pin_digest})",
            flush=True,
        )
    finally:
        if backend is not None:
            store = backend.store
            del backend
            close_store(store)


def print_paired_analysis(control_name: str, target_name: str, collected) -> None:
    """Report one target against the first condition from matched rounds."""
    base_arms = collected[control_name]
    target_arms = collected[target_name]
    base_rates = [arm["gen_rate"] for arm in base_arms]
    target_rates = [arm["gen_rate"] for arm in target_arms]
    base_phys = [arm["phys_mb_tok"] for arm in base_arms if arm["phys_mb_tok"] > 0]
    target_phys = [arm["phys_mb_tok"] for arm in target_arms if arm["phys_mb_tok"] > 0]

    pairs = list(zip(base_rates, target_rates))
    diffs_pct = [(target - base) / base * 100 for base, target in pairs]
    wins = sum(1 for difference in diffs_pct if difference > 0)
    pair_count = len(pairs)
    p_value = sum(
        comb(pair_count, k) for k in range(wins, pair_count + 1)
    ) / (2 ** pair_count)
    median_difference = st.median(diffs_pct)
    mean_difference = st.mean(diffs_pct)
    span = max(base_rates) - min(base_rates)
    band = span / st.median(base_rates) * 100 if st.median(base_rates) else 0.0
    physical_saved = (
        st.median(base_phys) - st.median(target_phys)
        if base_phys and target_phys else 0.0
    )

    print(f"\nPaired analysis over {pair_count} rounds ({target_name} vs {control_name}):")
    print(f"  Mean paired speedup: {mean_difference:+.1f}%")
    print(f"  Median paired speedup: {median_difference:+.1f}%")
    print(f"  Physical read reduction: {physical_saved:+.1f} MB/token")
    print(
        f"  {target_name} ahead in {wins} of {pair_count} rounds, "
        f"sign test p = {p_value:.3f}"
    )
    print(f"  Resolution band: {band:.1f}%")
    if abs(median_difference) < band:
        print("  -> Speedup is inside the resolution band.")
    elif median_difference > 0:
        print(f"  -> {target_name} demonstrates a resolved improvement.")
    else:
        print(f"  -> {target_name} demonstrates a regression.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="Tokens per arm")
    parser.add_argument(
        "--pairs", type=int, default=None,
        help="Reversed rounds. Defaults to 4 for the capacity sweep and 6 otherwise",
    )
    parser.add_argument("--pause", type=float, default=2.5, help="Pause seconds between arms")
    parser.add_argument(
        "--boundary-profile",
        choices=BOUNDARY_LABELS,
        default=None,
        help="Force completion at one Metal graph boundary per arm",
    )
    parser.add_argument(
        "--capacity-sweep", action="store_true",
        help="Run the trusted 56, 60, and 64-slot skew sweep",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Build and validate selected pack files without generating tokens",
    )
    parser.add_argument(
        "--allow-same-boot", action="store_true",
        help="Run the capacity sweep without the required post-preparation reboot",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="slabpack56_skew",
        choices=[
            "baseline", "slab12", "global48", "global56", "slabpack48",
            "slabpack56_uniform", "slabpack56_skew", "slabpack56_skew_unfused",
            "slabpack60_skew", "slabpack60_skew_f5",
            "slabpack60_skew_f5c2", "slabpack60_skew_f5c3",
            "slabpack60_skew_f10_up",
            "slabpack60_skew_8a", "slabpack60_skew_8b",
            "slabpack60_skew_unfused", "slabpack64_skew",
        ],
        help="Control configuration (default: 56-slot 8A skew pack)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="slabpack60_skew",
        choices=[
            "baseline", "slab12", "global48", "global56", "slabpack48",
            "slabpack56_uniform", "slabpack56_skew", "slabpack56_skew_unfused",
            "slabpack60_skew", "slabpack60_skew_f5",
            "slabpack60_skew_f5c2", "slabpack60_skew_f5c3",
            "slabpack60_skew_f10_up",
            "slabpack60_skew_8a", "slabpack60_skew_8b",
            "slabpack60_skew_unfused", "slabpack64_skew",
        ],
        help="Target slab configuration (default: standard 60-slot 8A pack)",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Comma-separated configurations to compare",
    )
    args = parser.parse_args()

    cond_defs = {
        "baseline": (0, 0, 0, False, "uniform", True, False, False),
        "slab12": (4, 12, 0, False, "uniform", True, False, False),
        "global48": (0, 0, 48, False, "uniform", True, False, False),
        "global56": (0, 0, 56, False, "uniform", True, False, False),
        "slabpack48": (0, 0, 48, True, "uniform", True, False, False),
        "slabpack56_uniform": (0, 0, 56, True, "uniform", True, False, False),
        "slabpack56_skew": (0, 0, 56, True, "skew", True, False, False),
        "slabpack56_skew_unfused": (0, 0, 56, True, "skew", False, False, False),
        "slabpack60_skew": (0, 0, 60, True, "skew", True, False, False),
        "slabpack60_skew_f5": (0, 0, 60, True, "skew", True, False, -1),
        "slabpack60_skew_f5c2": (0, 0, 60, True, "skew", True, False, 2),
        "slabpack60_skew_f5c3": (0, 0, 60, True, "skew", True, False, 3),
        "slabpack60_skew_f10_up": (0, 0, 60, True, "skew", True, False, False),
        "slabpack60_skew_8a": (0, 0, 60, True, "skew", True, False, False),
        "slabpack60_skew_8b": (0, 0, 60, True, "skew", True, True, False),
        "slabpack60_skew_unfused": (0, 0, 60, True, "skew", False, False, False),
        "slabpack64_skew": (0, 0, 64, True, "skew", True, False, False),
    }

    if args.capacity_sweep and args.arms:
        parser.error("--capacity-sweep and --arms cannot be used together")
    if args.capacity_sweep:
        arm_names = list(CAPACITY_ARMS)
    elif args.arms:
        arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
        for a in arm_names:
            if a not in cond_defs:
                parser.error(f"Unknown arm '{a}'. Available: {list(cond_defs.keys())}")
    else:
        arm_names = [args.control, args.target]

    rounds = args.pairs
    if rounds is None:
        rounds = 4 if args.capacity_sweep else 6
    if rounds < 1:
        parser.error("--pairs must be at least 1")

    if args.prepare_only:
        if args.capacity_sweep:
            calibrate_routing_profile()
            from models.flashnext.expert_cache import _GLOBAL_SLAB_CACHE

            _GLOBAL_SLAB_CACHE.clear()
        prepared = {}
        for name in arm_names:
            if not cond_defs[name][3]:
                parser.error(
                    f"--prepare-only requires slab-pack arms, got {name}"
                )
            print(f"Preparing {name}...", flush=True)
            info = prepare_pack(name, cond_defs[name])
            prepared[name] = info
            print(
                f"  -> Slots: {info['allocated_slots']} | "
                f"Pack: {info['pack_bytes']} B ({info['pack_mib']:.2f} MiB) | "
                f"mlock: ok | Alloc: {info['allocation_digest']}",
                flush=True,
            )
        if args.capacity_sweep:
            write_capacity_manifest(prepared)
        print("Pack preparation changes page-cache state. Reboot before a trusted run.")
        return

    if args.capacity_sweep:
        require_prepared_capacity_packs(allow_same_boot=args.allow_same_boot)

    print(f"=== Controlled Production Benchmark: {' vs '.join(arm_names)} ===")
    print(f"Tokens per arm: {args.tokens}")
    print(f"Rounds: {rounds} (Total arms: {rounds * len(arm_names)})")
    print(f"System load average: {os.getloadavg()}\n")

    schedule = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            schedule.extend(arm_names)
        else:
            schedule.extend(list(reversed(arm_names)))

    collected = {name: [] for name in arm_names}

    for idx, name in enumerate(schedule, 1):
        (
            slab, layers, global_b, s_pack, policy, f_shared,
            f_shared_parts, stream_pack,
        ) = cond_defs[name]
        pack_str = f", PACK=1, POLICY={policy}" if s_pack else ""
        fuse_str = "" if f_shared else ", FUSE_SHARED=0"
        if f_shared and not f_shared_parts:
            fuse_str = ", FUSE_SHARED_PARTS=0"
        if stream_pack:
            chunk = "all" if stream_pack < 0 else str(stream_pack)
            fuse_str += f", STREAM_PACK={chunk}"
        fuse_up_swiglu = name == "slabpack60_skew_f10_up"
        if fuse_up_swiglu:
            fuse_str += ", FUSED_UP_SWIGLU=1"
        print(f"Arm {idx:2d}/{len(schedule)}: Running {name:<24} (SLAB={slab}, LAYERS={layers}, GLOBAL={global_b}{pack_str}{fuse_str})...", flush=True)
        res = run_arm(
            slab, layers, args.tokens, global_b, slab_pack=s_pack,
            policy=policy, fuse_shared=f_shared,
            fuse_shared_parts=f_shared_parts,
            stream_pack=stream_pack,
            boundary_profile=args.boundary_profile,
            fuse_up_swiglu=fuse_up_swiglu,
        )
        collected[name].append(res)
        print(
            f"  -> Gen: {res['gen_rate']:.2f} t/s | Tail: {res['tail_rate']:.2f} t/s | "
            f"Phys: {res['phys_mb_tok']:.1f} MB/tok | Active: {res['active_mb']:.1f} MB | "
            f"Hits: {res['hit_pct']:.1f}% | IO wait: {res['io_wait_ms_tok']:.2f} ms/tok | "
            f"Slots: {res['allocated_slots']} | Pack: {res['pack_bytes']} B ({res['pack_mib']:.2f} MiB) | "
            f"mlock: {'ok' if res['mlock_ok'] else 'no'} | "
            f"Alloc: {res['allocation_digest']} | VM: {res['vm_counters']} | "
            f"Digest: {res['digest']}",
            flush=True,
        )
        breakdown = res["io_breakdown_ms_tok"]
        print(
            "     Frontier 5: "
            f"pread={breakdown['critical_pread']:.2f}, "
            f"queue={breakdown['critical_queue']:.2f}, "
            f"task-overhead={breakdown['critical_task_overhead']:.2f}, "
            f"completion={breakdown['completion_overhead']:.2f} ms/tok | "
            f"pread-sum={breakdown['pread_service_sum']:.2f} ms/tok | "
            f"calls={res['pread_calls_tok']:.2f}/tok | "
            f"requested={res['pread_mb_tok']:.1f} MB/tok",
            flush=True,
        )
        if args.boundary_profile is not None:
            selected = res["boundary_profile"]["per_token"][args.boundary_profile]
            print(
                f"     Boundary {args.boundary_profile}: "
                f"issue={selected['issue_ms']:.3f}, "
                f"completion={selected['completion_ms']:.3f}, "
                f"total={selected['total_ms']:.3f} ms/tok",
                flush=True,
            )
        if idx < len(schedule):
            time.sleep(args.pause)

    # Summary Table
    print("\n=== Production Comparison Summary ===")
    print(f"{'Condition':<20} | {'Gen med':<8} | {'Range':<14} | {'Tail med':<9} | {'Phys MB/tok':<12} | {'Active MB':<10} | {'IO ms/tok':<10} | {'Hit %':<7} | {'Slots':<5} | {'Pack MiB':<9} | {'mlock':<5} | {'Digest':<16}")
    print("-" * 173)
    for name in arm_names:
        arms = collected[name]
        rates = [a["gen_rate"] for a in arms]
        tails = [a["tail_rate"] for a in arms]
        phys = [a["phys_mb_tok"] for a in arms if a["phys_mb_tok"] > 0]
        hits = [a["hit_pct"] for a in arms]
        active = [a["active_mb"] for a in arms]
        io_wait = [a["io_wait_ms_tok"] for a in arms]
        slots = arms[0]["allocated_slots"] if arms else 0
        pack_mib = arms[0]["pack_mib"] if arms else 0.0
        mlock = "ok" if arms and arms[0]["mlock_ok"] else "no"
        digest = arms[0]["digest"] if arms else "none"
        print(
            f"{name:<20} | {st.median(rates):8.2f} | "
            f"{min(rates):.2f}..{max(rates):.2f}     | {st.median(tails):9.2f} | "
            f"{(st.median(phys) if phys else 0.0):12.1f} | {st.median(active):10.1f} | "
            f"{st.median(io_wait):10.2f} | {st.median(hits):6.1f}% | {slots:5d} | "
            f"{pack_mib:9.2f} | {mlock:<5} | {digest:<16}"
        )
        breakdown_keys = (
            "critical_pread", "critical_queue", "critical_task_overhead",
            "completion_overhead",
        )
        medians = {
            key: st.median(
                arm["io_breakdown_ms_tok"][key] for arm in arms
            )
            for key in breakdown_keys
        }
        print(
            " " * 23
            + "Frontier 5 median ms/tok: "
            + ", ".join(f"{key}={value:.2f}" for key, value in medians.items())
        )

    control_name = arm_names[0]
    for target_name in arm_names[1:]:
        print_paired_analysis(control_name, target_name, collected)


if __name__ == "__main__":
    main()
