"""Relative GPU utilization from IOKit, without sudo and without forking.

Every timing in this project so far is wall-clock on the host: `perf_counter`
around an `mx.eval`, which measures the main thread blocked, not the GPU
running. That distinction is why five hypotheses about the unattributed part
of `score_sync` all came back negative. The component table accounts for about
58 ms of a token while the eval blocks for 172, and no amount of host-side
timing can say whether the difference is kernels or waiting.

`ioreg -c IOAccelerator` publishes `Device Utilization %` and needs no
privileges, but forking it costs 12.7 ms per sample, which both caps the rate
near 79 Hz and perturbs the thing being measured. This reads the same property
through IOKit directly, so a sample costs microseconds.

`powermetrics` would also answer this and is not usable here: it needs
administrator access, which the research log already recorded when it could
not measure energy.

Read `Device Utilization %` as a relative signal for comparing runs. It is not
validated kernel time, and it does not separate one kernel from another. The
counter undercounts short kernels, so never convert this value to milliseconds
or present it as absolute GPU time. Use Metal System Trace or another validated
source for absolute GPU measurements.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import plistlib
import threading
import time

_IOKIT = ctypes.CDLL(
    "/System/Library/Frameworks/IOKit.framework/IOKit", use_errno=True
)
_CF = ctypes.CDLL(
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
)

_CF.CFStringCreateWithCString.restype = ctypes.c_void_p
_CF.CFStringCreateWithCString.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32
]
_CF.CFPropertyListCreateData.restype = ctypes.c_void_p
_CF.CFPropertyListCreateData.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_void_p,
]
_CF.CFDataGetLength.restype = ctypes.c_long
_CF.CFDataGetLength.argtypes = [ctypes.c_void_p]
_CF.CFDataGetBytePtr.restype = ctypes.c_void_p
_CF.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
_CF.CFRelease.argtypes = [ctypes.c_void_p]

_IOKIT.IOServiceMatching.restype = ctypes.c_void_p
_IOKIT.IOServiceMatching.argtypes = [ctypes.c_char_p]
_IOKIT.IOServiceGetMatchingServices.restype = ctypes.c_int
_IOKIT.IOServiceGetMatchingServices.argtypes = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
]
_IOKIT.IOIteratorNext.restype = ctypes.c_uint32
_IOKIT.IOIteratorNext.argtypes = [ctypes.c_uint32]
_IOKIT.IOObjectRelease.restype = ctypes.c_int
_IOKIT.IOObjectRelease.argtypes = [ctypes.c_uint32]
_IOKIT.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
_IOKIT.IORegistryEntryCreateCFProperty.argtypes = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
]

_UTF8 = 0x08000100
_XML = 100
_KEY = "PerformanceStatistics"


def _cfstr(text: str):
    return _CF.CFStringCreateWithCString(None, text.encode(), _UTF8)


def _accelerators():
    """Every IOAccelerator service, as registry entry ids."""
    matching = _IOKIT.IOServiceMatching(b"IOAccelerator")
    it = ctypes.c_uint32(0)
    if _IOKIT.IOServiceGetMatchingServices(0, matching, ctypes.byref(it)) != 0:
        return []
    found = []
    while True:
        entry = _IOKIT.IOIteratorNext(it)
        if not entry:
            break
        found.append(entry)
    _IOKIT.IOObjectRelease(it)
    return found


class GPUMeter:
    """Poll the relative `Device Utilization %` signal on a background thread.

    The thread is a sampler, not a load. Each sample is one IOKit property
    read plus a plist parse of a small dictionary, so it does not compete with
    the model for the drive or the GPU.
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self._entries = _accelerators()
        self._key = _cfstr(_KEY)
        self._stop = threading.Event()
        self._thread = None
        self.samples: list[float] = []

    def available(self) -> bool:
        return bool(self._entries) and self.read() is not None

    def read(self):
        """One reading of Device Utilization %, or None."""
        for entry in self._entries:
            ref = _IOKIT.IORegistryEntryCreateCFProperty(
                entry, self._key, None, 0
            )
            if not ref:
                continue
            data = _CF.CFPropertyListCreateData(None, ref, _XML, 0, None)
            _CF.CFRelease(ref)
            if not data:
                continue
            length = _CF.CFDataGetLength(data)
            ptr = _CF.CFDataGetBytePtr(data)
            raw = ctypes.string_at(ptr, length)
            _CF.CFRelease(data)
            try:
                stats = plistlib.loads(raw)
            except Exception:
                continue
            value = stats.get("Device Utilization %")
            if value is not None:
                return float(value)
        return None

    def _run(self):
        while not self._stop.is_set():
            value = self.read()
            if value is not None:
                self.samples.append(value)
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.summary()

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        ordered = sorted(self.samples)
        mean = sum(self.samples) / len(self.samples)
        return {
            "samples": len(self.samples),
            "mean": mean,
            "median": ordered[len(ordered) // 2],
            "p90": ordered[int(len(ordered) * 0.9)],
            "max": ordered[-1],
            # This fraction supports comparisons between runs only. It is not
            # a duration and must not be multiplied by wall-clock time.
            "relative_busy_fraction": mean / 100.0,
            "relative_only": True,
        }
