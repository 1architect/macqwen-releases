"""Model-independent chat profiles."""
from __future__ import annotations

from macqwen.profiles.agent import SYSTEM_TOOLS, environment_block
from macqwen.profiles.plain import SYSTEM_PLAIN
from macqwen.tools import TOOLS


PROFILES = {
    "plain": {"system": SYSTEM_PLAIN, "tools": None},
    "agent": {"system": SYSTEM_TOOLS, "tools": TOOLS},
}


def system_prompt(profile: str, workspace: str | None = None) -> str:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    text = PROFILES[profile]["system"]
    if workspace and profile == "agent":
        text = f"{text}\n\n{environment_block(workspace)}"
    return text


def tools_for(profile: str):
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    return PROFILES[profile]["tools"]
