#!/usr/bin/env python3
"""Select one model runtime, then execute the shared chat in its environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macqwen import preferences


QWEN27B_PYTHON = "~/mlx-qwen38-kernel-lab/bin/python3"
FLASHNEXT_PYTHON = "~/models/.venv-qwen4exp/bin/python"
V4_ROOT = Path("~/.lmstudio/models/gioma").expanduser()
V4_PREFIX = "Qwen3.8-27B-Apple-MLX-V4-"


def _split_build(argv: list[str]) -> tuple[str | None, list[str]]:
    if argv and not argv[0].startswith("-"):
        return argv[0], argv[1:]
    return None, argv


def _v4_path(build: str) -> Path:
    path = V4_ROOT / f"{V4_PREFIX}{build}"
    if not path.is_dir():
        raise SystemExit(f"no such V4 build: {path}")
    return _validate_qwen27b(path)


def _validate_qwen27b(path: Path) -> Path:
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid model config: {path / 'config.json'}: {exc}") from exc
    if config.get("vocab_size") != 248320:
        raise SystemExit(
            "refusing: the BF16 embedding and head require Qwen3.8-27B "
            "with vocab 248320"
        )
    return path


def _qwen27b_path(requested: str | None, build: str | None) -> Path:
    if build:
        return _v4_path(build)
    value = requested or os.environ.get("MACQWEN_MODEL")
    if value:
        path = Path(value).expanduser()
        if path.is_dir():
            return _validate_qwen27b(path)
        raise SystemExit(f"no such model: {path}")
    builds = sorted(V4_ROOT.glob(f"{V4_PREFIX}*"))
    choices = "\n".join(
        f"  ./chat.sh {path.name[len(V4_PREFIX):]}" for path in builds if path.is_dir()
    )
    raise SystemExit("choose a 27B build:\n" + (choices or "  no V4 builds found"))


def command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    argv = ["--server" if value == "/server" else value for value in argv]
    build, argv = _split_build(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=("flashnext", "qwen27b"))
    parser.add_argument("--profile", choices=("plain", "agent"))
    parser.add_argument("--model-path")
    parser.add_argument("--preferences-file", default=preferences.DEFAULT_PATH)
    parser.add_argument("--v4", action="store_true")
    known, remaining = parser.parse_known_args(argv)

    model = known.model or ("qwen27b" if build else "flashnext")
    python = (
        os.environ.get("MACQWEN_QWEN27B_PYTHON", QWEN27B_PYTHON)
        if model == "qwen27b"
        else os.environ.get("MACQWEN_FLASHNEXT_PYTHON", FLASHNEXT_PYTHON)
    )
    interpreter = Path(python).expanduser()
    if not interpreter.is_file():
        raise SystemExit(f"missing Python environment: {interpreter}")

    chat_args = [
        str(interpreter), "-u", str(ROOT / "macqwen" / "session.py"),
        "--model", model,
        "--preferences-file", known.preferences_file,
    ]
    if known.profile:
        chat_args += ["--profile", known.profile]
    if model == "qwen27b":
        model_path = _qwen27b_path(known.model_path, build)
        chat_args += [
            "--model-path", str(model_path),
            "--bf16-ends",
            "--kv-bits", os.environ.get("KV_BITS", "4"),
            "--quantized-kv-start", os.environ.get("KV_START", "0"),
            "--prefill-step-size", os.environ.get("PREFILL_STEP", "256"),
        ]
    elif known.model_path:
        chat_args += ["--model-path", known.model_path]
    if os.environ.get("MACQWEN_WORKSPACE") and "--workspace" not in remaining:
        chat_args += ["--workspace", os.environ["MACQWEN_WORKSPACE"]]
    if os.environ.get("SPEED_LAYER_INDICES") and "--layer-indices" not in remaining:
        chat_args += ["--layer-indices", os.environ["SPEED_LAYER_INDICES"]]
    if os.environ.get("WIRED") and "--wired-limit-gb" not in remaining:
        chat_args += ["--wired-limit-gb", os.environ["WIRED"]]
    if os.environ.get("PAGED") == "1" and "--paged" not in remaining:
        chat_args.append("--paged")
    chat_args.extend(remaining)

    environment = dict(os.environ)
    if model == "qwen27b":
        environment.setdefault("MLX_QMM_BM", "64")
        environment.setdefault("MLX_QMM_BK", "32")
        environment.setdefault("MLX_QMM_BN", "64")
    return ["/usr/bin/caffeinate", "-i", *chat_args], environment


def main() -> int:
    executable, environment = command(sys.argv[1:])
    os.execvpe(executable[0], executable, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
