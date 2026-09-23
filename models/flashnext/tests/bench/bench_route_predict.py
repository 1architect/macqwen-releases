#!/usr/bin/env python3
"""How well does a layer's MoE input predict the next layers' experts?

A read issued one or two layers early could overlap with compute, but only if
the experts are known in time. At each decode MoE call this applies the routers
of layers L+1 and L+2 to layer L's MoE input, keeps the top-m experts, and
later compares them with the experts those layers actually route. The
previous token's route at the same layer is recorded as the old baseline.

Slab-pack experts are never read, so recall is also reported over streamed
experts only. Predictions run beside the model and change no value; the token
digest is recorded so the run can be matched to a normal decode.

    python -m models.flashnext.tests.bench.bench_route_predict --tokens 128
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402
from models.flashnext.tests.bench.bench_long_states import PROMPT  # noqa: E402

SKETCHUP = (
    "<|im_start|>user\ncrie uma extensão para sketchup que extrude várias faces "
    "ao mesmo tempo até uma altura definida pelo usuário. produza o código para "
    "eu salvar em um arquivo .rb<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
    "</think>\n\n"
)
PROMPTS = {"photosynthesis": PROMPT, "sketchup": SKETCHUP}
LEADS = (1, 2)
WIDTHS = (8, 10, 16, 24)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompts", nargs="+", choices=tuple(PROMPTS),
                        default=list(PROMPTS))
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "route-predict", "route-predict.json", args.json)

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-route-predict-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    import mlx.core as mx
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from macqwen.backends.flashnext import FlashNextBackend
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext import expert_cache
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32
    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    layers = backend.language.model.layers
    blocks = {
        index: layer.mlp for index, layer in enumerate(layers)
        if getattr(layer.mlp, "switch_mlp", None) is not None
    }
    slab = {
        index: set(block.switch_mlp.slab_expert_to_slot)
        for index, block in blocks.items()
    }
    widest = max(WIDTHS)

    state = {"on": False, "token": -1}
    predicted = {}   # (token, layer) -> {lead: ranked expert list}
    actual = {}      # (token, layer) -> set of routed experts

    original = Qwen3_5MoeSparseMoeBlock.__call__

    def call(self, x, _original=original):
        layer = getattr(self, "_flashnext_layer_id", None)
        if state["on"] and layer is not None and x.size // x.shape[-1] == 1:
            if layer == min(blocks):
                state["token"] += 1
            for lead in LEADS:
                ahead = blocks.get(layer + lead)
                if ahead is None:
                    continue
                gates = ahead.gate(x).reshape(-1)
                ranked = mx.argsort(-gates)[:widest]
                predicted.setdefault((state["token"], layer + lead), {})[lead] = (
                    ranked.tolist()
                )
        return _original(self, x)

    def trace(layer, routed):
        if state["on"] and len(routed) <= 10:
            actual[(state["token"], layer)] = set(routed)

    Qwen3_5MoeSparseMoeBlock.__call__ = call
    expert_cache.set_route_trace(trace)

    evidence = {
        "checkpoint": checkpoint, "resident_experts": resident,
        "tokens": args.tokens, "leads": LEADS, "widths": WIDTHS,
        "slab_experts": sum(len(v) for v in slab.values()),
        "prompts": {}, "status": "running",
    }

    def save():
        target.write_text(json.dumps(evidence, indent=1) + "\n")

    save()
    for name in args.prompts:
        predicted.clear()
        actual.clear()
        state["token"] = -1
        backend.reset()
        backend.append_text(PROMPTS[name])

        def on_prefilled():
            state["on"] = True

        _text, stats = backend.generate(args.tokens, on_prefilled=on_prefilled)
        state["on"] = False
        ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []

        # Recall of the actual routed set, all experts and streamed only, and
        # the streamed experts a prediction would read but not use.
        sums = defaultdict(lambda: [0, 0, 0, 0, 0])
        previous = defaultdict(lambda: [0, 0, 0, 0])
        for (token, layer), used in actual.items():
            streamed = used - slab.get(layer, set())
            for lead, ranked in predicted.get((token, layer), {}).items():
                for width in WIDTHS:
                    guess = set(ranked[:width])
                    streamed_guess = guess - slab.get(layer, set())
                    row = sums[(lead, width)]
                    row[0] += len(used & guess)
                    row[1] += len(used)
                    row[2] += len(streamed & streamed_guess)
                    row[3] += len(streamed)
                    row[4] += len(streamed_guess - streamed)
            before = actual.get((token - 1, layer))
            if before is not None:
                streamed_before = before - slab.get(layer, set())
                row = previous[layer >= 0]
                row[0] += len(used & before)
                row[1] += len(used)
                row[2] += len(streamed & streamed_before)
                row[3] += len(streamed)
        samples = len(actual)
        result = {
            "tokens": stats.tokens,
            "gen_rate": round(stats.rate, 3),
            "digest": hashlib.sha256(repr(ids).encode()).hexdigest()[:16],
            "layer_samples": samples,
            "predictors": {
                f"lead{lead}-top{width}": {
                    "recall": round(row[0] / max(1, row[1]), 4),
                    "streamed_recall": round(row[2] / max(1, row[3]), 4),
                    "wasted_streamed_per_layer": round(row[4] / max(1, samples), 3),
                }
                for (lead, width), row in sorted(sums.items())
            },
            "previous_token": {
                "recall": round(previous[True][0] / max(1, previous[True][1]), 4),
                "streamed_recall": round(
                    previous[True][2] / max(1, previous[True][3]), 4
                ),
            },
            "streamed_per_layer": round(
                sum(len(used - slab.get(layer, set()))
                    for (_t, layer), used in actual.items()) / max(1, samples), 3
            ),
        }
        evidence["prompts"][name] = result
        save()
        print(f"== {name}: {stats.tokens} tokens, digest {result['digest']}, "
              f"streamed/layer {result['streamed_per_layer']}", flush=True)
        print(f"  previous token  {result['previous_token']}", flush=True)
        for key, value in result["predictors"].items():
            print(f"  {key:14s} {value}", flush=True)
    expert_cache.set_route_trace(None)
    Qwen3_5MoeSparseMoeBlock.__call__ = original
    evidence["status"] = "completed"
    save()
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
