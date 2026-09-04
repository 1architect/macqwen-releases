"""Compatibility defaults derived from the backend setting registry.

Dynamic CLI argument construction remains deferred. Existing flags stay owned
by the shared session parser for compatibility.
"""

from models.flashnext.settings import get_registry


_REGISTRY_DEFAULTS = get_registry().defaults("flashnext")
FLASHNEXT_DEFAULTS = {
    name.replace("-", "_"): value
    for name, value in _REGISTRY_DEFAULTS.items()
    if name in {
        "routing", "swap-epsilon", "threshold", "resident-experts",
        "pin-budget-gb", "tail-experts", "tail-warmup", "fusion-block",
        "fusion-min-margin", "fusion-min-block", "fusion-margin-tokens",
        "fusion-max-prompt", "fusion-model",
    }
}
