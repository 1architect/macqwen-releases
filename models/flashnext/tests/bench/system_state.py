"""Lightweight macOS preparation checks that never import MLX."""
from __future__ import annotations

import os
import subprocess
import time


def purge_file_cache() -> None:
    """Request a cold file cache without changing anonymous memory or swap."""
    subprocess.run(["purge"], check=True, timeout=120)


def wait_for_quiescence(
    window_seconds: float,
    timeout_seconds: float,
    max_load: float,
    max_compressor_pages_per_second: float = 128.0,
    sample_seconds: float = 5.0,
) -> None:
    """Optionally wait for bounded background VM activity."""
    from models.flashnext.diskio import vm_counters

    if window_seconds <= 0:
        return
    began = time.monotonic()
    quiet_began = None
    previous = vm_counters()
    if not previous:
        raise RuntimeError("VM counters are unavailable")
    watched = ("swapin", "swapout", "pageout", "compress", "decompress")
    while time.monotonic() - began < timeout_seconds:
        time.sleep(sample_seconds)
        current = vm_counters()
        if not current:
            quiet_began = None
            previous = current
            continue
        deltas = {
            key: current.get(key, 0) - previous.get(key, 0)
            for key in watched
        }
        load = os.getloadavg()[0]
        critical_idle = all(
            deltas[key] == 0 for key in ("swapin", "swapout", "pageout")
        )
        compressor_rate = (
            deltas["compress"] + deltas["decompress"]
        ) / sample_seconds
        quiet = (
            critical_idle
            and compressor_rate <= max_compressor_pages_per_second
            and load <= max_load
        )
        if quiet:
            quiet_began = quiet_began or time.monotonic()
            if time.monotonic() - quiet_began >= window_seconds:
                print(
                    f"Quiescence gate passed: {window_seconds:.0f}s clean, "
                    f"load={load:.2f}, compressor={compressor_rate:.1f} pages/s, "
                    f"vm={deltas}",
                    flush=True,
                )
                return
        else:
            quiet_began = None
            print(
                f"Waiting for quiescence: load={load:.2f}, vm={deltas}",
                flush=True,
            )
        previous = current
    raise RuntimeError(
        f"System did not provide a {window_seconds:.0f}s clean window within "
        f"{timeout_seconds:.0f}s"
    )
