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


def _mtp_depth_reader(backend) -> int:
    raw = os.environ.get("FLASHNEXT_MTP_DEPTH")
    if raw is not None:
        try:
            return _mtp_depth_parser(raw)
        except (TypeError, ValueError):
            pass
    return int(getattr(backend, "mtp_depth", 3))


def _mtp_depth_setter(backend, value) -> None:
    parsed = _mtp_depth_parser(value)
    backend.mtp_depth = parsed
    os.environ["FLASHNEXT_MTP_DEPTH"] = str(parsed)


def _native_mtp_active(backend) -> bool:
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
        env_reader("FLASHNEXT_NATIVE_MTP", "off"),
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
