#!/usr/bin/env python3
"""Select one model runtime, then execute the shared chat in its environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macqwen import preferences
from macqwen.checkpoints import installed_checkpoints, resolve_qwen27b
from models.bonsai2.settings import (
    MODEL_NAME as BONSAI2,
    PYTHON_ENV as BONSAI2_PYTHON_ENV,
)
from models.k2_horizon.settings import (
    MODEL_NAME as K2_HORIZON,
    PYTHON_ENV as K2_HORIZON_PYTHON_ENV,
)


PYTHON_ENV = {
    "shared": "MACQWEN_PYTHON",
    "flashnext": "MACQWEN_FLASHNEXT_PYTHON",
    BONSAI2: BONSAI2_PYTHON_ENV,
    K2_HORIZON: K2_HORIZON_PYTHON_ENV,
    "qwen27b": "MACQWEN_QWEN27B_PYTHON",
}
MANAGED_ENV = ROOT / ".venv"
MANAGED_PYTHON = MANAGED_ENV / "bin" / "python"
RUNTIME_VERSIONS = {
    "mlx": "0.32.2",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.6.17",
    "transformers": "5.16.1",
    "numpy": "2.5.2",
    "requests": "2.34.2",
    "huggingface-hub": "1.29.0",
}
RUNTIME_PROBES = {
    "flashnext": """
from mlx_vlm.utils import (apply_generation_config_defaults, get_model_and_args,
                           load_config, update_module_configs)
""",
    BONSAI2: """
from mlx_vlm.models.qwen3_5 import Model, ModelConfig
from mlx_vlm.utils import load_config
""",
    K2_HORIZON: """
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
""",
    "qwen27b": """
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import (ArraysCache, KVCache, QuantizedKVCache,
                                 make_prompt_cache)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load_tokenizer
""",
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


def _runtime_probe(model: str) -> str:
    versions = repr(RUNTIME_VERSIONS)
    return f"""
from importlib.metadata import PackageNotFoundError, version
import sys

expected = {versions}
for package, wanted in expected.items():
    try:
        found = version(package)
    except PackageNotFoundError:
        raise SystemExit(f"missing package {{package}}")
    if found != wanted:
        raise SystemExit(f"{{package}} {{found}}, expected {{wanted}}")

import mlx.core as mx
import mlx.nn as nn
if not hasattr(mx, "array") or not hasattr(mx, "fast") or not hasattr(nn, "Module"):
    raise SystemExit("MLX backend API is incomplete")
{RUNTIME_PROBES[model]}
"""


def _python_runtime_error(path: Path, model: str) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "-c", _runtime_probe(model)],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip().splitlines()
    return detail[-1][:240] if detail else f"probe exited with {result.returncode}"


def _supports_python(path: Path, model: str) -> bool:
    return _python_runtime_error(path, model) is None


def _setup_failure(operation: str, error: BaseException) -> SystemExit:
    detail = getattr(error, "stderr", None) or str(error)
    return SystemExit(
        f"MACQWEN setup failed during {operation}: {detail}\n"
        "Retry with './chat.sh setup'."
    )


def _run_setup_step(operation: str, command: list[str]) -> None:
    print(f"MACQWEN setup: {operation}", flush=True)
    try:
        subprocess.check_call(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _setup_failure(operation, exc) from exc


def _interpreter(model: str) -> Path:
    override_name = next(
        (name for name in (PYTHON_ENV[model], PYTHON_ENV["shared"])
         if os.environ.get(name)),
        None,
    )
    override = os.environ.get(override_name) if override_name else None
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"missing Python environment override {override_name}: {path}")
        if not _supports_python(path, model):
            reason = _python_runtime_error(path, model) or "runtime probe failed"
            raise SystemExit(
                f"Python environment override {override_name} is incompatible "
                f"with {model}: {reason}"
            )
        return path

    if not _supports_python(MANAGED_PYTHON, model):
        setup_environment(["--venv", str(MANAGED_ENV)])
    if not _supports_python(MANAGED_PYTHON, model):
        reason = _python_runtime_error(MANAGED_PYTHON, model) or "runtime probe failed"
        raise SystemExit(
            f"managed MACQWEN environment is incompatible with {model}: {reason}\n"
            "Retry with './chat.sh setup'."
        )
    return MANAGED_PYTHON


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
    _run_setup_step("create the managed Python environment", [creator, "-m", "venv", str(target)])
    python = target / "bin" / "python"
    _run_setup_step(
        "upgrade pip",
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
    )
    _run_setup_step(
        "install MACQWEN and its shared runtime",
        [str(python), "-m", "pip", "install", "-e", str(ROOT)],
    )
    print(f"MACQWEN environment ready: {target}")
    return 0


def _default_checkpoint_args(argv: list[str]) -> list[str]:
    """Select the only installed checkpoint, or ask before loading one."""
    if argv and (argv[0] == "setup" or not argv[0].startswith("-")):
        return argv
    if any(
        value in ("--model", "--model-path", "--checkpoint")
        or value.startswith(("--model=", "--model-path=", "--checkpoint="))
        for value in argv
    ):
        return argv

    choices = installed_checkpoints()
    if not choices:
        return argv
    if len(choices) == 1:
        model, path = choices[0]
        return [*argv, "--model", model, "--checkpoint", str(path)]

    print("Multiple compatible checkpoints found:", file=sys.stderr)
    for index, (model, path) in enumerate(choices, 1):
        print(f"  {index}. {model}: {path}", file=sys.stderr)
    if not sys.stdin.isatty():
        raise SystemExit(
            "choose a checkpoint with --model and --checkpoint before loading"
        )
    try:
        answer = input(f"Choose a checkpoint [1-{len(choices)}]: ").strip()
        index = int(answer) - 1
    except (EOFError, ValueError):
        raise SystemExit("invalid checkpoint choice") from None
    if not 0 <= index < len(choices):
        raise SystemExit("invalid checkpoint choice")
    model, path = choices[index]
    return [*argv, "--model", model, "--checkpoint", str(path)]


def command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    argv = ["--server" if value == "/server" else value for value in argv]
    build, argv = _split_build(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=("flashnext", BONSAI2, K2_HORIZON, "qwen27b"))
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
    if model == "flashnext":
        from models.flashnext.settings.launch import apply_chat_environment

        apply_chat_environment(environment)
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
    executable, environment = command(_default_checkpoint_args(sys.argv[1:]))
    os.execvpe(executable[0], executable, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
