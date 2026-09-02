"""Physical disk bytes for this process.

`getrusage` returns 0 for `ru_inblock` on Darwin, so the count comes from
`proc_pid_rusage` with `RUSAGE_INFO_V4`.

This is the instrument that tells a cold run from a warm one. Decode rate on
this machine spans about 21 percent depending on how much of the checkpoint
the page cache still holds, so a rate reported without its physical read
volume cannot be compared against another rate.
"""
from __future__ import annotations

import ctypes
import os

_RUSAGE_INFO_V4 = 4
_BYTESREAD_OFFSET = 16 + 16 * 8  # after ri_uuid[16] and 16 uint64 fields
_LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.dylib")
_BUFFER = (ctypes.c_uint8 * 512)()


def disk_bytes_read() -> int:
    """Physical bytes this process has read, or -1 when unavailable."""
    if _LIBSYSTEM.proc_pid_rusage(
        os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(_BUFFER)
    ) != 0:
        return -1
    raw = bytes(_BUFFER[_BYTESREAD_OFFSET : _BYTESREAD_OFFSET + 8])
    return int.from_bytes(raw, "little")


def free_memory_mb() -> float:
    """Free pages, in MB. This is an inverse proxy for page-cache warmth.

    Measured correlation with decode rate across 13 arms was -0.84: more free
    memory means less of the checkpoint is cached, which means a slower run.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.split("\n"):
            if line.startswith("Pages free"):
                pages = int(line.split(":")[1].strip().rstrip("."))
                return pages * 16384 / 1048576
    except Exception:
        pass
    return -1.0


class ReadMeter:
    """Physical bytes and wall time across one measured span."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._bytes = disk_bytes_read()

    def bytes_since(self) -> int:
        now = disk_bytes_read()
        if now < 0 or self._bytes < 0:
            return -1
        return now - self._bytes


_VM_KEYS = {
    "Pages reactivated": "reactivated",
    "Pageins": "pagein",
    "Pageouts": "pageout",
    "Compressions": "compress",
    "Decompressions": "decompress",
    "Swapins": "swapin",
    "Swapouts": "swapout",
}


def vm_counters() -> dict:
    """What the memory system did, beside what the drive served.

    Physical bytes say how much came off the device. These say what macOS did
    to make room for it: pull pages back off the inactive queue, push them
    through the compressor, or swap. A cost that tracks these rather than the
    byte count is the VM managing the cache, not the drive serving it.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return {}
    found = {}
    for line in out.split("\n"):
        label, _, value = line.partition(":")
        name = _VM_KEYS.get(label.strip())
        if name:
            found[name] = int(value.strip().rstrip("."))
    return found


def vm_delta(before: dict, after: dict, tokens: int) -> str:
    """Per-token counter movement, as a printable line."""
    if not before or not after:
        return "vm: unavailable"
    parts = [
        f"{name}={((after.get(name, 0) - value) / tokens):.0f}"
        for name, value in sorted(before.items())
    ]
    return "vm/token " + " ".join(parts)
