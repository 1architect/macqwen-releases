"""One preferences file, shared by every model and profile.

Settings are global on purpose: thinking, effort and approval describe how
you want to work, not which model is loaded. Switching models must not
silently change them.

Each setting is declared once, with its default and its validator together.
That is the rule issue #1 exists to enforce: four settings were defined in
two places last night and silently discarded. Adding a setting here is one
line, and there is nowhere else to disagree with it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

DEFAULT_PATH = "~/.macqwen/preferences.json"
DEFAULT_ANSWER_TOKENS = 2048
DEFAULT_PLAIN_ANSWER_TOKENS = 4096
DEFAULT_THINK_TOKENS = 512
# read once at migration time, in this order, and then left alone
LEGACY_PATHS = (
    "~/.cache/flashnext/preferences.json",
    "~/.frankenstein/preferences.json",
)


def _boolean(default):
    return default, lambda v: isinstance(v, bool)


def _choice(default, options):
    return default, lambda v: v in options


# One source of truth. `/effort` used to carry its own copy of this tuple and
# silently rejected `high` after the schema gained it.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh")


def _text(default):
    return default, lambda v: isinstance(v, str) and bool(v.strip())


def _string(default):
    return default, lambda v: isinstance(v, str)


def _token_limit(default):
    # -1 means no limit. bool is an int subclass, so reject it explicitly.
    return default, lambda v: (
        isinstance(v, int) and not isinstance(v, bool) and (v == -1 or v > 0)
    )


def _number(default, low, high):
    return default, lambda v: (
        isinstance(v, (int, float)) and not isinstance(v, bool) and low <= v <= high
    )


def _whole(default):
    return default, lambda v: (
        isinstance(v, int) and not isinstance(v, bool) and v >= 0
    )


SCHEMA = {
    # how the model answers
    "thinking_enabled": _boolean(False),
    "show_thinking": _boolean(False),
    "think_budget": _token_limit(DEFAULT_THINK_TOKENS),
    "effort": _choice("medium", EFFORT_LEVELS),
    "max_tokens": _token_limit(-1),
    # Qwen's card recommends these for thinking mode. This runtime decoded
    # with argmax, which is temperature 0 and none of the rest, and greedy
    # decoding resolves a tie the same way every time. Set temperature to 0
    # for greedy, which the benchmarks need to compare token IDs.
    "temperature": _number(1.0, 0.0, 2.0),
    "top_p": _number(0.95, 0.0, 1.0),
    "top_k": _whole(20),
    "min_p": _number(0.0, 0.0, 1.0),
    # "adjust between 0 and 2 to reduce endless repetition. However, using a
    # higher value may occasionally result in language mixing"
    "presence_penalty": _number(0.0, 0.0, 2.0),
    # how the chat behaves
    "stream_answers": _boolean(True),
    "animate": _boolean(True),
    "code_only": _boolean(False),
    "system_prompt": _string(""),
    # agent profile only, ignored by the plain profile
    "approval": _choice("ask", ("ask", "auto")),
    "spec_enabled": _boolean(False),
    "workspace": _text(str(Path.cwd())),
    # what to start with
    "model": _text("flashnext"),
    "flashnext_checkpoint": _string(""),
    "profile": _choice("plain", ("plain", "agent")),
}

DEFAULTS = {name: default for name, (default, _) in SCHEMA.items()}


def answer_limit(values: dict, default: int = DEFAULT_ANSWER_TOKENS) -> int:
    """Return the answer allowance after resolving the default sentinel."""
    saved = values.get("max_tokens", default)
    return saved if saved > 0 else default


def think_limit(values: dict) -> int:
    """Return extra reasoning capacity, or zero when it is disabled."""
    if not values.get("thinking_enabled", False):
        return 0
    saved = values.get("think_budget", DEFAULT_THINK_TOKENS)
    return saved if saved > 0 else 0


def generation_limit(values: dict, default: int = DEFAULT_ANSWER_TOKENS) -> int:
    """Keep reasoning tokens from consuming the answer allowance."""
    return answer_limit(values, default) + think_limit(values)

# the two chats disagreed on this name; the explicit one wins
RENAMED = {"show_think": "show_thinking"}


def _accept(saved: dict) -> dict:
    """Keep the values that pass their own validator, drop the rest."""
    values = dict(DEFAULTS)
    for key, value in saved.items():
        key = RENAMED.get(key, key)
        entry = SCHEMA.get(key)
        if entry is None:
            continue
        _, valid = entry
        if valid(value):
            values[key] = value.strip() if isinstance(value, str) else value
    return values


def load(path: str | Path = DEFAULT_PATH) -> dict:
    """Read preferences.

    Only the default location falls back to the legacy files. An explicit
    path means that file and nothing else, so callers and tests get exactly
    what they asked for.
    """
    target = Path(path).expanduser()
    try:
        return _accept(json.loads(target.read_text()))
    except (OSError, ValueError, TypeError):
        pass
    if target != Path(DEFAULT_PATH).expanduser():
        return dict(DEFAULTS)
    merged: dict = {}
    for legacy in LEGACY_PATHS:
        try:
            saved = json.loads(Path(legacy).expanduser().read_text())
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(saved, dict):
            merged.update(saved)
    return _accept(merged)


def save(values: dict, path: str | Path = DEFAULT_PATH) -> None:
    """Write atomically, private to the user, leaving no temporary behind."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(target.parent, 0o700)
    except OSError:
        pass
    keep = {k: v for k, v in values.items() if k in SCHEMA}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False,
            prefix=".preferences.", suffix=".tmp",
        ) as handle:
            json.dump(keep, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
