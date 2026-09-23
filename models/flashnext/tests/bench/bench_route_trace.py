#!/usr/bin/env python3
"""Record the expert reads of real chat turns for the offline cache simulator.

Runs the normal-chat backend (checkpoint pin policy, frozen slab, keep-warm
off) over a few predeclared prompts in one process, the way a conversation
would touch the page cache. For every streamed MoE call it records the layer
and the distinct routed experts, marks prefill and decode, and records the
physical bytes each decode token read. The slab contents and the pinned set
are saved so the simulator can treat them as always resident.

    python -m models.flashnext.tests.bench.bench_route_trace --tokens 256
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402

PROMPTS = {
    "photosynthesis": (
        "Explique a fotossintese em detalhes, em portugues, cobrindo etapas, "
        "estruturas celulares, reagentes, produtos e fatores que alteram a taxa."
    ),
    "sketchup": (
        "crie uma extensão para sketchup que extrude várias faces ao mesmo "
        "tempo até uma altura definida pelo usuário. produza o código para eu "
        "salvar em um arquivo .rb"
    ),
    "open": (
        "Conte uma historia longa sobre uma cidade costeira que muda ao longo "
        "de cem anos. Continue sem resumir."
    ),
}


def formatted(text: str) -> str:
    return (f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prompts", nargs="+", choices=sorted(PROMPTS),
                        default=["photosynthesis", "sketchup", "open"])
    parser.add_argument("--resident-experts", default="policy")
    args = parser.parse_args()
    target = output_path("flashnext", "route-trace", "route-trace.json.gz")

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_GPU_KEEPWARM"] = "0"
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-route-trace-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = (
        resolve_resident_experts(None, checkpoint) or 32
        if args.resident_experts == "policy" else int(args.resident_experts)
    )

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext import expert_cache
    from models.flashnext.diskio import ReadMeter, free_memory_mb

    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    meter = ReadMeter()
    events: list = []
    state = {"phase": "prefill", "turn": 0, "token": 0}

    def trace(layer_id, routed):
        events.append((state["turn"], state["phase"], state["token"], int(layer_id),
                       sorted(set(int(e) for e in routed))))

    slab = {}
    record_bytes = None
    for index, layer in enumerate(backend.language.model.layers):
        block = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        mapping = getattr(block, "slab_expert_to_slot", None) or {}
        if mapping:
            slab[index] = sorted(int(e) for e in mapping)
        if block is not None and record_bytes is None:
            prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
            store = backend.store
            record_bytes = sum(
                store.pin_size(f"{prefix}.{projection}.{part}", [0])
                for projection in ("gate_proj", "up_proj", "down_proj")
                for part in ("weight", "scales", "biases")
            )

    turns = []
    expert_cache.set_route_trace(trace)
    try:
        for turn, name in enumerate(args.prompts):
            state.update(turn=turn, phase="prefill", token=0)
            backend.reset()
            backend.append_text(formatted(PROMPTS[name]))
            per_token = []
            marks = {"read": 0}

            def on_prefilled():
                state["phase"] = "decode"
                meter.reset()

            def on_token(_value, _piece):
                read = meter.bytes_since()
                per_token.append(read)
                meter.reset()
                state["token"] += 1

            free_before = free_memory_mb()
            _text, stats = backend.generate(
                args.tokens, on_prefilled=on_prefilled, on_decode_token=on_token,
            )
            turns.append({
                "prompt": name, "tokens": stats.tokens,
                "prompt_tokens": stats.prompt_tokens,
                "gen_rate": round(stats.rate, 3),
                "free_mb_before": free_before,
                "physical_bytes_per_token": per_token,
                "pinned": {str(k): sorted(v) for k, v in backend.routing.pinned.items()},
                "pinned_bytes": stats.pinned_bytes,
            })
            decode_mb = sum(max(0, b) for b in per_token) / 1e6 / max(1, len(per_token))
            print(f"{name}: {stats.tokens} tok at {stats.rate:.2f} tok/s, "
                  f"{decode_mb:.1f} MB/token physical, {len(events)} events", flush=True)
    finally:
        expert_cache.set_route_trace(None)

    payload = {
        "checkpoint": checkpoint, "resident_experts": resident,
        "record_bytes": record_bytes, "slab": {str(k): v for k, v in slab.items()},
        "turns": turns, "events": events,
    }
    with gzip.open(target, "wt") as handle:
        json.dump(payload, handle)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
