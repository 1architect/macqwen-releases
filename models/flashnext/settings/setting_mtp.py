"""Native MTP controls for FlashNext. Off by default until gates pass."""
from __future__ import annotations

import os

from macqwen.backend_settings import Setting

from .setting_runtime import choice, env_reader


def _mtp_depth_parser(raw) -> int:
    value = int(raw)
    if value < 1 or value > 8:
        raise ValueError("mtp-depth must be between 1 and 8")
    return value


def _native_mtp_reader(backend) -> str:
    """Report the state the backend was constructed with.

    A startup setting cannot affect an already-built model, so prefer
    the constructed attribute over a later environment change.
    """
    value = getattr(backend, "native_mtp", None)
    if value in ("off", "on"):
        return value
    return env_reader("FLASHNEXT_NATIVE_MTP", "off")(backend)


def _mtp_depth_reader(backend) -> int:
    """Report the constructed/next-turn depth, not a later env change."""
    try:
        return _mtp_depth_parser(getattr(backend, "mtp_depth", 3))
    except (TypeError, ValueError):
        pass
    raw = os.environ.get("FLASHNEXT_MTP_DEPTH")
    if raw is not None:
        try:
            return _mtp_depth_parser(raw)
        except (TypeError, ValueError):
            pass
    return 3


def _mtp_depth_setter(backend, value) -> None:
    parsed = _mtp_depth_parser(value)
    backend.mtp_depth = parsed
    os.environ["FLASHNEXT_MTP_DEPTH"] = str(parsed)


def _native_mtp_active(backend) -> bool:
    """Active only when the constructed model actually carries MTP."""
    if hasattr(backend, "_mtp_active"):
        return bool(backend._mtp_active)
    return getattr(backend, "native_mtp", "off") == "on" or os.environ.get(
        "FLASHNEXT_NATIVE_MTP", "off"
    ) == "on"


SETTINGS = (
    Setting(
        "native-mtp",
        ("FLASHNEXT_NATIVE_MTP",),
        "off",
        choice(("off", "on"), "native-mtp"),
        "startup",
        "runtime",
        "public",
        "flashnext",
        _native_mtp_reader,
        None,
        _native_mtp_active,
        None,
        "FLASHNEXT_NATIVE_MTP",
        __file__,
    ),
    Setting(
        "mtp-depth",
        ("FLASHNEXT_MTP_DEPTH",),
        3,
        _mtp_depth_parser,
        "next-turn",
        "runtime",
        "public",
        "flashnext",
        _mtp_depth_reader,
        _mtp_depth_setter,
        lambda _b: True,
        None,
        "FLASHNEXT_MTP_DEPTH",
        __file__,
    ),
)
