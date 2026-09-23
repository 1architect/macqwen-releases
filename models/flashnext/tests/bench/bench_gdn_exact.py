#!/usr/bin/env python3
"""Exactness gate for FLASHNEXT_COMPILE_GDN on captured real inputs.

Loads the normal-chat backend, decodes ``--capture`` tokens with the plain
chains while recording every decode call's inputs to the Qwen4 q/k
normalization and the gated output norm, then compares the plain and compiled
outputs bit for bit on each call. Finally it decodes ``--tokens`` tokens with
the flag off and on and compares the token digests.

    python -m models.flashnext.tests.bench.bench_gdn_exact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from macqwen.results import output_path  # noqa: E402
from models.flashnext.tests.bench.bench_long_states import PROMPT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    target = output_path("flashnext", "gdn-exact", "gdn-exact.json", args.json)

    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_COMPILE_GDN"] = "0"
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    private = Path(tempfile.mkdtemp(prefix="flashnext-gdn-exact-"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, private / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(private / "pins.json")

    import mlx.core as mx

    from macqwen.backends.flashnext import FlashNextBackend
    from macqwen.checkpoints import resolve_flashnext
    from models.flashnext import compile_gdn
    from models.flashnext.checkpoint_policy import resolve_resident_experts

    checkpoint = str(resolve_flashnext(os.environ.get("MACQWEN_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32
    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    from mlx_vlm.models.qwen4_exp import language

    captured = {"qk": [], "norm": []}
    state = {"on": False}
    patched_qk = language.Qwen4ExpGatedDeltaNet._normalize_qk
    patched_norm = language.Qwen4ExpRMSNormGated.__call__

    def capture_qk(self, q, k):
        if state["on"] and q.shape[1] == 1:
            mx.eval(q, k)
            captured["qk"].append((self, q, k))
        return patched_qk(self, q, k)

    def capture_norm(self, x, gate):
        if state["on"] and x.shape[1] == 1:
            mx.eval(x, gate)
            captured["norm"].append((self, x, gate))
        return patched_norm(self, x, gate)

    language.Qwen4ExpGatedDeltaNet._normalize_qk = capture_qk
    language.Qwen4ExpRMSNormGated.__call__ = capture_norm
    backend.reset()
    backend.append_text(PROMPT)
    backend.generate(args.capture, on_prefilled=lambda: state.update(on=True))
    state["on"] = False
    language.Qwen4ExpGatedDeltaNet._normalize_qk = patched_qk
    language.Qwen4ExpRMSNormGated.__call__ = patched_norm

    results = {}
    for name, calls in captured.items():
        mismatched, worst = 0, 0.0
        for module, first, second in calls:
            compile_gdn.set_enabled(False)
            plain = (patched_qk if name == "qk" else patched_norm)(module, first, second)
            compile_gdn.set_enabled(True)
            fused = (patched_qk if name == "qk" else patched_norm)(module, first, second)
            compile_gdn.set_enabled(False)
            plain = plain if isinstance(plain, tuple) else (plain,)
            fused = fused if isinstance(fused, tuple) else (fused,)
            same = all(
                a.dtype == b.dtype and mx.array_equal(a, b).item()
                for a, b in zip(plain, fused)
            )
            if not same:
                mismatched += 1
                worst = max(worst, max(
                    mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
                    for a, b in zip(plain, fused)
                ))
        results[name] = {"calls": len(calls), "mismatched": mismatched,
                         "max_abs_difference": worst}
        print(f"{name:5s} calls {len(calls):4d}  mismatched {mismatched:4d}  "
              f"max |diff| {worst:g}", flush=True)

    digests = {}
    for enabled in (False, True):
        compile_gdn.set_enabled(enabled)
        backend.reset()
        backend.append_text(PROMPT)
        _text, stats = backend.generate(args.tokens)
        ids = list(backend.tape[-stats.tokens:])
        digests["on" if enabled else "off"] = hashlib.sha256(repr(ids).encode()).hexdigest()[:16]
    compile_gdn.set_enabled(False)
    print(f"digests {digests}", flush=True)
    evidence = {
        "checkpoint": checkpoint, "capture_tokens": args.capture,
        "chains": results, "digests": digests,
        "exact": all(r["mismatched"] == 0 for r in results.values())
        and digests["off"] == digests["on"],
    }
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"exact: {evidence['exact']}  wrote {target}")
    return 0 if evidence["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
