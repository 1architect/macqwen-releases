#!/usr/bin/env python3
"""Where the normal-chat process keeps its memory.

Loads the normal-chat backend with the checkpoint's resident-expert policy,
then records a memory snapshot after load and after one short greedy answer:
MLX active, cache and peak memory, the ``footprint`` category summary and the
``vmmap --summary`` region table. Clean file-backed pages (the checkpoint and
slab mappings) can be dropped by macOS; dirty and anonymous pages compete with
the page cache the expert stream depends on. The raw tool output is kept so a
later change can be checked against the same categories.

    python -m models.flashnext.tests.bench.bench_footprint --tokens 32
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402

from models.flashnext.tests.bench.bench_long_states import PROMPT  # noqa: E402


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def snapshot(label: str, folder: Path) -> dict:
    import mlx.core as mx

    pid = os.getpid()
    footprint = _run(["footprint", "-p", str(pid)])
    vmmap = _run(["vmmap", "--summary", str(pid)])
    (folder / f"footprint-{label}.txt").write_text(footprint)
    (folder / f"vmmap-{label}.txt").write_text(vmmap)
    return {
        "label": label,
        "mlx_active_mb": mx.get_active_memory() / 1e6,
        "mlx_cache_mb": mx.get_cache_memory() / 1e6,
        "mlx_peak_mb": mx.get_peak_memory() / 1e6,
        "swap": _run(["sysctl", "-n", "vm.swapusage"]).strip(),
        "footprint_head": footprint.splitlines()[:40],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "footprint", "footprint.json", args.json)
    folder = target.parent

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    # Private copies keep the user's pin history unchanged.
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-footprint-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32

    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    evidence = {
        "checkpoint": checkpoint, "resident_experts": resident,
        "tokens": args.tokens, "snapshots": [],
        "slab_pack": bool(getattr(backend.store, "_slab_pack", None)),
    }
    evidence["snapshots"].append(snapshot("load", folder))
    backend.reset()
    backend.append_text(PROMPT)
    _text, stats = backend.generate(args.tokens)
    evidence["decode_rate"] = round(stats.rate, 3)
    evidence["snapshots"].append(snapshot("decode", folder))
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    for item in evidence["snapshots"]:
        print(
            f"{item['label']}: MLX active {item['mlx_active_mb']:.0f} MB, "
            f"cache {item['mlx_cache_mb']:.0f} MB, peak {item['mlx_peak_mb']:.0f} MB",
            flush=True,
        )
    print(f"wrote {target}", flush=True)
    shutil.rmtree(private, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
