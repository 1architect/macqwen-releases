#!/usr/bin/env python3
"""Select one model runtime, then execute the shared chat in its environment."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macqwen import preferences
from macqwen.checkpoints import resolve_qwen27b


PYTHON_ENV = {
    "flashnext": "MACQWEN_FLASHNEXT_PYTHON",
    "qwen27b": "MACQWEN_QWEN27B_PYTHON",
}
REQUIRED_MODULES = {
    "flashnext": ("mlx", "mlx_vlm", "transformers"),
    "qwen27b": ("mlx",),
}


def branch_sync_warning(root: Path = ROOT) -> str:
    """Warn when a checkout lacks commits from its known origin/main."""
    try:
        current = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor",
             "origin/main", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return ""
    if current.returncode != 1:
        return ""
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip() or "current branch"
    return (
        f"warning: {branch} does not include known origin/main; "
        "fetch and merge origin/main before validating chat behavior"
    )


def _split_build(argv: list[str]) -> tuple[str | None, list[str]]:
    if argv and not argv[0].startswith("-"):
        return argv[0], argv[1:]
    return None, argv


def _qwen27b_path(requested: str | None, build: str | None) -> Path:
    try:
        return resolve_qwen27b(requested or build or os.environ.get("MACQWEN_MODEL"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _supports_current_python(model: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES[model])


def _supports_python(path: Path, model: str) -> bool:
    imports = "; ".join(f"import {name}" for name in REQUIRED_MODULES[model])
    try:
        result = subprocess.run(
            [str(path), "-c", imports],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _interpreter(model: str) -> Path:
    override = os.environ.get(PYTHON_ENV[model])
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path
        raise SystemExit(f"missing Python environment: {path}")

    candidates = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python")
    candidates.append(ROOT / (".venv-qwen27b" if model == "qwen27b" else ".venv") / "bin" / "python")
    for path in candidates:
        if path.is_file() and _supports_python(path, model):
            return path
    if _supports_current_python(model):
        return Path(sys.executable)
    variable = PYTHON_ENV[model]
    raise SystemExit(
        f"no compatible Python environment found for {model}; run './chat.sh setup' "
        f"or set {variable}"
    )


def setup_environment(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="macqwen setup")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    args = parser.parse_args(argv)
    target = args.venv.expanduser().resolve()
    creator = sys.executable
    if sys.version_info < (3, 12):
        creator = shutil.which("python3.12") or ""
    if not creator:
        raise SystemExit("Python 3.12 is required; install it, then run setup again")
    subprocess.check_call([creator, "-m", "venv", str(target)])
    python = target / "bin" / "python"
    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(python), "-m", "pip", "install", "-e", f"{ROOT}[flashnext]"])
    print(f"MACQWEN environment ready: {target}")
    return 0


def command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    argv = ["--server" if value == "/server" else value for value in argv]
    build, argv = _split_build(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=("flashnext", "qwen27b"))
    parser.add_argument("--profile", choices=("plain", "agent"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path")
    parser.add_argument("--preferences-file", default=preferences.DEFAULT_PATH)
    parser.add_argument("--v4", action="store_true")
    known, remaining = parser.parse_known_args(argv)

    model = known.model or ("qwen27b" if build else "flashnext")
    interpreter = _interpreter(model)

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
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        return setup_environment(sys.argv[2:])
    warning = branch_sync_warning()
    if warning:
        print(warning, file=sys.stderr)
    executable, environment = command(sys.argv[1:])
    os.execvpe(executable[0], executable, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
