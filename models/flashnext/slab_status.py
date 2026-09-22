"""Loaded-object slab residency report. No environment inference.

Usage:
    python -m models.flashnext.slab_status --model ~/models/CHECKPOINT

Loads the checkpoint target-only through the streaming loader and
reports the slab/pin state proven from the constructed objects:
pin-profile compatibility and identity, slab allocation validity,
pack creation, active layers, slot counts, and per-layer mappings.
"""
from __future__ import annotations

import argparse
import json
import os


def summarize_slab(store, switch_modules) -> dict:
    """Summarize slab state from a loaded store and its MoE modules."""
    from . import expert_cache as ec
    from .routing import pin_profile_compatible, checkpoint_identity_for_store
    from .slab_pack import validate_slab_allocation

    profile = ec._load_pin_profile()
    report: dict = {
        "pin_profile_present": profile is not None,
        "pin_compatible": None,
        "pin_reason": None,
        "pin_identity": None,
        "pin_group_size": None,
        "store_identity": None,
        "slab_alloc_layers": None,
        "slab_alloc_valid": None,
        "slab_pack_created": False,
        "slab_pack_digest": None,
        "slab_pack_bytes": None,
        "active_slab_layers": 0,
        "total_packed_slots": 0,
        "per_layer_slots": {},
        "disabled_reasons": {},
    }
    try:
        report["store_identity"] = checkpoint_identity_for_store(store)
    except Exception:
        report["store_identity"] = None
    if profile is not None:
        report["pin_identity"] = profile.get("checkpoint_identity")
        report["pin_group_size"] = profile.get("group_size")
        try:
            ok, reason = pin_profile_compatible(
                store, profile, expected_group_size=32)
        except Exception as exc:
            ok, reason = False, str(exc)
        report["pin_compatible"] = bool(ok)
        report["pin_reason"] = reason
    alloc = getattr(store, "_slab_alloc", None)
    if alloc is not None:
        report["slab_alloc_layers"] = sorted(int(k) for k in alloc)
        try:
            report["slab_alloc_valid"] = bool(
                validate_slab_allocation(store, alloc, layout=32))
        except Exception:
            report["slab_alloc_valid"] = False
    pack = getattr(store, "_slab_pack", None)
    report["slab_pack_created"] = pack is not None
    if pack is not None:
        digest = getattr(pack, "allocation_digest", None)
        report["slab_pack_digest"] = (
            digest() if callable(digest) else digest)
        report["slab_pack_bytes"] = getattr(pack, "size", None)
    active = 0
    total = 0
    for index, module in enumerate(switch_modules):
        slots = getattr(module, "slab_expert_to_slot", None) or {}
        if slots:
            active += 1
            total += len(slots)
            report["per_layer_slots"][index] = len(slots)
        reason = getattr(module, "_slab_pack_disabled_reason", None)
        if reason:
            report["disabled_reasons"][index] = reason
    report["active_slab_layers"] = active
    report["total_packed_slots"] = total
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Report loaded-object slab residency for a checkpoint.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    from .loader import load_streaming

    model, _config, store = load_streaming(
        os.path.expanduser(args.model), expert_capacity=0, verbose=False,
        keep_vision=False, use_mtp=False)
    switches = []
    for layer in model.language_model.model.layers:
        block = getattr(layer, "mlp", None)
        switch = getattr(block, "switch_mlp", None)
        if switch is not None and hasattr(switch, "slab_expert_to_slot"):
            switches.append(switch)
    print(json.dumps(summarize_slab(store, switches), indent=2,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
