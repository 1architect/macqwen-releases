#!/usr/bin/env python3
"""Long greedy answers with GPU and CPU clock residency per window.

Loads the normal-chat backend once, with the checkpoint's resident-expert
policy (8 on Vontra), then generates one long answer per condition. Every
``--window`` tokens it records the decode rate, physical MB/token, the GPU
and CPU-complex performance-state residency from IOReport, and the
``pmset -g therm`` warning lines. A condition that throttles shows up as a
falling mean GPU state or a thermal warning late in its answer.

The conditions only flip process-level switches that the runtime reads live,
so one loaded model serves every arm. All arms must produce the same tokens.

    python -m models.flashnext.tests.bench.bench_long_states \
        --tokens 384 --conditions off keepwarm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402

PROMPT = (
    "<|im_start|>user\nExplique a fotossintese em detalhes, em portugues, com "
    "pelo menos 800 palavras. Cubra as etapas, as estruturas celulares, os "
    "reagentes, os produtos e os fatores que alteram a taxa do processo."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# Each condition maps to live setters. Keep the list short and explicit.
CONDITIONS = ("off", "keepwarm", "qos", "qos-keepwarm", "keepwarm-nooverlap", "bundle")
# ``bundle``: keep-warm, QoS user-interactive and shared-expert overlap off.
# Its load-time members come from the process environment, so compare it
# against ``keepwarm`` in separate processes.


def apply_condition(name: str) -> None:
    from models.flashnext import adaptive_topk, expert_cache

    expert_cache.set_gpu_keepwarm(
        name in ("keepwarm", "qos-keepwarm", "keepwarm-nooverlap", "bundle")
    )
    # Submit the shared expert early (FLASHNEXT_OVERLAP, default on) or not.
    adaptive_topk.set_overlap(name not in ("keepwarm-nooverlap", "bundle"))
    setter = getattr(expert_cache, "set_io_qos", None)
    if setter is not None:
        setter(
            "user-interactive" if name in ("qos", "qos-keepwarm", "bundle")
            else "default"
        )
    elif name in ("qos", "qos-keepwarm", "bundle"):
        raise SystemExit("this runtime has no set_io_qos; cannot run a QoS arm")


def therm_lines() -> list[str]:
    try:
        text = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ["unavailable"]
    return [line.strip() for line in text.splitlines() if line.strip()]


class Clocks:
    """GPU and CPU-complex performance-state residency between two marks."""

    def __init__(self):
        from models.flashnext.tests.bench.gpu_pstates import GpuStates

        self.gpu = GpuStates()
        self.cpu = GpuStates("CPU Stats", "CPU Complex Performance States")
        self._marks = None

    def mark(self):
        previous = self._marks
        self._marks = (self.gpu.sample(), self.cpu.sample())
        if previous is None:
            return None
        from models.flashnext.tests.bench.gpu_pstates import summarize

        result = {}
        for meter, before, after in (
            (self.gpu, previous[0], self._marks[0]),
            (self.cpu, previous[1], self._marks[1]),
        ):
            for channel, states in meter.delta(before, after).items():
                summary = summarize(states)
                result[channel] = {
                    "active_share": round(summary["active_share"], 4),
                    "mean_active_state": round(summary["mean_active_state"], 3),
                }
            meter.release(before)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["off", "keepwarm"])
    parser.add_argument("--resident-experts", default="policy",
                        help="'policy' uses the checkpoint policy, else an integer")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "long-states", "long-states.json", args.json)

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_GPU_KEEPWARM"] = "0"

    # Private copies of the pin history and the frozen slab snapshots keep
    # the user's profile unchanged while the slab matches normal chat.
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-long-states-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    if args.resident_experts == "policy":
        resident = resolve_resident_experts(None, checkpoint) or 32
    else:
        resident = int(args.resident_experts)

    from macqwen.backends.flashnext import FlashNextBackend
    from models.flashnext.diskio import ReadMeter, free_memory_mb, vm_counters

    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    clocks = Clocks()
    meter = ReadMeter()
    evidence = {
        "checkpoint": checkpoint, "resident_experts": resident,
        "tokens": args.tokens, "window": args.window,
        "conditions": args.conditions, "arms": [], "status": "running",
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("FLASHNEXT_")
        },
    }

    def save():
        target.write_text(json.dumps(evidence, indent=1) + "\n")

    save()
    first_digest = None
    for name in args.conditions:
        apply_condition(name)
        backend.reset()
        backend.append_text(PROMPT)
        windows = []
        state = {"count": 0, "t": None, "bytes": 0}
        vm_before = vm_counters()
        arm = {
            "condition": name, "free_mb_before": free_memory_mb(),
            "therm_before": therm_lines(), "load_before": os.getloadavg(),
        }
        print(f"== {name}: free {arm['free_mb_before']:.0f} MB", flush=True)

        def on_prefilled():
            state["t"] = time.perf_counter()
            meter.reset()
            clocks.mark()

        def on_token(_value, _piece):
            state["count"] += 1
            if state["count"] % args.window:
                return
            now = time.perf_counter()
            read = meter.bytes_since()
            meter.reset()
            seconds = now - state["t"]
            state["t"] = now
            window = {
                "end_token": state["count"],
                "tok_s": round(args.window / seconds, 3),
                "mb_per_token": round(read / 1e6 / args.window, 1),
                "clocks": clocks.mark(),
                "therm": [line for line in therm_lines() if "level" in line.lower()],
            }
            windows.append(window)
            gpu = window["clocks"].get("GPUPH", {})
            pcpu = window["clocks"].get("PCPU", {})
            print(
                f"  {name:12s} tok {state['count']:4d}  {window['tok_s']:.2f} tok/s  "
                f"{window['mb_per_token']:6.1f} MB/tok  gpu state "
                f"{gpu.get('mean_active_state', 0):5.2f}  pcpu state "
                f"{pcpu.get('mean_active_state', 0):5.2f}",
                flush=True,
            )

        text, stats = backend.generate(
            args.tokens, on_prefilled=on_prefilled, on_decode_token=on_token,
        )
        ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
        digest = hashlib.sha256(repr(ids).encode()).hexdigest()[:16]
        arm.update({
            "gen_rate": round(stats.rate, 3),
            "tail_rate": round(stats.tail_tokens / stats.tail_seconds, 3)
            if stats.tail_seconds else None,
            "tokens": stats.tokens, "digest": digest, "windows": windows,
            "therm_after": therm_lines(), "load_after": os.getloadavg(),
            "vm": {key: value - vm_before.get(key, 0) for key, value in vm_counters().items()},
            "pinned_bytes": stats.pinned_bytes,
        })
        evidence["arms"].append(arm)
        save()
        print(f"  {name}: gen {arm['gen_rate']} tok/s, digest {digest}", flush=True)
        if first_digest is None:
            first_digest = digest
        elif digest != first_digest:
            evidence["status"] = "digest_mismatch"
            save()
            raise SystemExit(f"{name} produced different tokens")
    evidence["status"] = "completed"
    save()
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
