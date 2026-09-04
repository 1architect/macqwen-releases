from pathlib import Path
import sys

from .api import TestSpec


def command(config, _result_path: Path) -> list[str]:
    return [str(config.python), "-m", "models.flashnext.tests.case_swiglu_contract", "--execute"]


def main(argv=None) -> int:
    if "--execute" not in (argv or sys.argv[1:]):
        return 0
    from models.flashnext.swiglu_contract import exhaustive_gate_patterns

    for up_bits in (0x3F80, 0x3F9E):
        report = exhaustive_gate_patterns(
            up_bits=up_bits,
            candidates=("metal_swiglu", "metal_mlx_header_bf16"),
        )
        print(report.format(), flush=True)
    return 0


TEST = TestSpec(
    id="swiglu-contract", title="Exhaustive bfloat16 SwiGLU contract", category="verification",
    explanation="Compares every bfloat16 gate encoding against the retained MLX SwiGLU result.",
    why="It was proposed after an apparently small sigmoid rounding change altered exact output.",
    script=command, metrics=("gate patterns", "mismatch count", "example bit patterns"),
    controls={"candidate": "float32 and MLX-header bfloat16"},
    source="models/flashnext/swiglu_contract.py", canonical=False,
)


if __name__ == "__main__":
    raise SystemExit(main())
