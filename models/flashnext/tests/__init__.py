"""Flash-Next tests: ``unit/`` (checkpoint-free), ``bench/`` (harness scripts)
and ``cases/`` (test-terminal cards). Results go to ``results/flashnext/``."""
from __future__ import annotations


def canonical_environment(context) -> dict[str, str]:
    """Launch environment the terminal gives every Flash-Next case."""
    from models.flashnext.settings.launch import CHAT_ENV

    environment = dict(CHAT_ENV)
    environment["FLASHNEXT_IO_WORKERS"] = str(context.workers)
    environment.update({
        "FLASHNEXT_PROFILE_BOUNDARIES": "0",
        "FLASHNEXT_PROFILE_SCORE_SYNC": "0",
    })
    return environment
