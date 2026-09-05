"""Compare canonical skew slabs with the guarded physical-miss hybrid."""
from __future__ import annotations

import json
from pathlib import Path

from .api import FLASHNEXT, TestSpec
from .case_product_long import interpret as interpret_product
from .case_product_long import live_parser


def command(config, _result_path: Path) -> list[str]:
    profile = Path("~/.cache/flashnext/physical-misses.json").expanduser()
    return [
        str(config.python), str(FLASHNEXT / "bench_product_long.py"),
        "--tokens", "256", "--window", "32",
        "--rounds", "1", "--phase", "answer",
        "--paths", "canonical", "physical-miss-hybrid",
        "--physical-miss-profile", str(profile),
    ]


def interpret(returncode: int, output: str, arms: list[dict]) -> str:
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        gate = record.get("offline_ceiling", {})
        if record.get("type") == "premise" and not gate.get("passes", False):
            return (
                "Premise blocked. Predicted physical saving is "
                f"{gate.get('predicted_mb_per_token', 0.0):.2f} MB/token, below "
                f"the required {gate.get('minimum_mb_per_token', 0.0):.2f} "
                "MB/token. No model comparison ran."
            )
    return interpret_product(returncode, output, arms)


TEST = TestSpec(
    id="product-long-physical-miss",
    title="Canonical versus physical-miss hybrid slabs",
    category="performance",
    explanation=(
        "Compares canonical skew slabs with a 60-slot allocation selected "
        "from serialized long-run physical-read evidence. The old full replacement "
        "policy is not a normal selection."
    ),
    why=(
        "The full physical-miss replacement lost 8.4% overall and every window. "
        "Its calibration evidence is censored by resident canonical hits, so it "
        "cannot price displaced extensions fairly. This comparison keeps equal "
        "60-slot residency while testing the guarded hybrid."
    ),
    script=command,
    metrics=(
        "generation rate per 32-token window",
        "physical MB/token per window",
        "slab hit rate per window",
        "active memory and context",
        "thinking and answer phase",
    ),
    controls={
        "control": "canonical 60-slot skew Frontier 8A",
        "candidate": "60-slot physical-miss-hybrid allocation",
        "evidence": "~/.cache/flashnext/physical-misses.json",
        "allocation identity": "frozen pin-profile copies",
        "rounds": "one pair; directional validation only",
    },
    source="models/flashnext/bench_product_long.py and physical_miss.py",
    promotion=False,
    live_parser=live_parser,
    interpret=interpret,
    canonical=False,
)
