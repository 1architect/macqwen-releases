#!/usr/bin/env python3
"""Build the small-row sidecar for the slab-pack layers of normal chat.

Loads no model. The layers come from the chat's frozen slab allocation, the
only layers that stream through the expert-major record path. Every record is
written and read back with F_NOCACHE and compared with the checkpoint.

    python -m models.flashnext.tests.bench.build_small_sidecar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    record = output_path("flashnext", "small-sidecar", "build.json", "")

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext import small_sidecar
    from models.flashnext.expert_cache import _slab_allocation
    from models.flashnext.routing import checkpoint_identity_for_store
    from models.flashnext.store import SafeTensorStore

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    store = SafeTensorStore(checkpoint)
    identity = checkpoint_identity_for_store(store)
    allocation = _slab_allocation(store, int(os.environ["FLASHNEXT_SLAB_GLOBAL"]), 32)
    layers = sorted(int(layer) for layer in allocation)
    if not layers:
        raise SystemExit("no slab allocation; run one chat turn first")
    directory = Path(os.path.expanduser(
        args.out or f"~/.cache/flashnext/small-sidecar-{identity[:16]}"
    ))
    began = time.perf_counter()
    written = small_sidecar.build(
        store, layers, directory,
        lambda layer: f"language_model.model.layers.{layer}.mlp.switch_mlp",
    )
    manifest = {
        "format": small_sidecar.FORMAT,
        "checkpoint": checkpoint,
        "checkpoint_identity": identity,
        "layers": layers,
        "experts": {str(layer): count for layer, count in written.items()},
        "record_bytes": small_sidecar.RECORD_BYTES,
        "slab_allocation": {str(layer): allocation[layer] for layer in layers},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    summary = {
        "directory": str(directory), "layers": layers,
        "bytes": sum(written.values()) * small_sidecar.RECORD_BYTES,
        "seconds": round(time.perf_counter() - began, 1), "verified": True,
    }
    record.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
