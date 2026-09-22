"""GPU performance-state residency from IOReport, without sudo.

IOReport publishes, per GPU performance state, how long the GPU spent in it.
Two samples and their delta give the residency of each state over an
interval. A state's share of the active time tells whether the GPU ran at a
low or a high clock while the benchmark worked, which is the question behind
the drive-loaded GPU hump: the same kernels took 1.9 times longer per command
buffer at 25 to 50 percent misses.

The library lives in the dyld shared cache, so ``ctypes`` loads it by path
even though no file exists there.

    python -m models.flashnext.tests.bench.gpu_pstates --seconds 2
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from ctypes import c_char_p, c_int, c_int64, c_uint32, c_void_p

_CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_IOR = ctypes.CDLL("/usr/lib/libIOReport.dylib")

_UTF8 = 0x08000100
_CF.CFStringCreateWithCString.restype = c_void_p
_CF.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
_CF.CFStringGetCString.restype = ctypes.c_bool
_CF.CFStringGetCString.argtypes = [c_void_p, c_char_p, ctypes.c_long, c_uint32]
_CF.CFDictionaryGetValue.restype = c_void_p
_CF.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
_CF.CFArrayGetCount.restype = ctypes.c_long
_CF.CFArrayGetCount.argtypes = [c_void_p]
_CF.CFArrayGetValueAtIndex.restype = c_void_p
_CF.CFArrayGetValueAtIndex.argtypes = [c_void_p, ctypes.c_long]
_CF.CFRelease.argtypes = [c_void_p]

_IOR.IOReportCopyChannelsInGroup.restype = c_void_p
_IOR.IOReportCopyChannelsInGroup.argtypes = [c_void_p, c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
_IOR.IOReportCreateSubscription.restype = c_void_p
_IOR.IOReportCreateSubscription.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_void_p), ctypes.c_uint64, c_void_p]
_IOR.IOReportCreateSamples.restype = c_void_p
_IOR.IOReportCreateSamples.argtypes = [c_void_p, c_void_p, c_void_p]
_IOR.IOReportCreateSamplesDelta.restype = c_void_p
_IOR.IOReportCreateSamplesDelta.argtypes = [c_void_p, c_void_p, c_void_p]
for _name in ("IOReportChannelGetGroup", "IOReportChannelGetSubGroup",
              "IOReportChannelGetChannelName"):
    getattr(_IOR, _name).restype = c_void_p
    getattr(_IOR, _name).argtypes = [c_void_p]
_IOR.IOReportStateGetCount.restype = c_int
_IOR.IOReportStateGetCount.argtypes = [c_void_p]
_IOR.IOReportStateGetNameForIndex.restype = c_void_p
_IOR.IOReportStateGetNameForIndex.argtypes = [c_void_p, c_int]
_IOR.IOReportStateGetResidency.restype = c_int64
_IOR.IOReportStateGetResidency.argtypes = [c_void_p, c_int]


def _cfstr(text: str) -> c_void_p:
    return _CF.CFStringCreateWithCString(None, text.encode(), _UTF8)


def _text(value) -> str:
    if not value:
        return ""
    buffer = ctypes.create_string_buffer(256)
    _CF.CFStringGetCString(value, buffer, 256, _UTF8)
    return buffer.value.decode(errors="replace")


class GpuStates:
    """Subscribe once; ``sample()`` returns an opaque handle for ``delta``."""

    def __init__(self, group: str = "GPU Stats",
                 subgroup: str = "GPU Performance States"):
        self._group = _cfstr(group)
        self._subgroup = _cfstr(subgroup)
        channels = _IOR.IOReportCopyChannelsInGroup(self._group, self._subgroup, 0, 0, 0)
        if not channels:
            raise RuntimeError("IOReport has no GPU performance-state channels")
        self._subscribed = c_void_p()
        self._subscription = _IOR.IOReportCreateSubscription(
            None, channels, ctypes.byref(self._subscribed), 0, None)
        if not self._subscription:
            raise RuntimeError("IOReport subscription failed")
        self._channels_key = _cfstr("IOReportChannels")

    def sample(self):
        return _IOR.IOReportCreateSamples(self._subscription, self._subscribed, None)

    def delta(self, before, after) -> dict[str, dict[str, int]]:
        """Residency per channel and state between two samples, in ticks."""
        difference = _IOR.IOReportCreateSamplesDelta(before, after, None)
        result: dict[str, dict[str, int]] = {}
        try:
            array = _CF.CFDictionaryGetValue(difference, self._channels_key)
            for index in range(_CF.CFArrayGetCount(array) if array else 0):
                channel = _CF.CFArrayGetValueAtIndex(array, index)
                name = _text(_IOR.IOReportChannelGetChannelName(channel))
                states = {}
                for state in range(_IOR.IOReportStateGetCount(channel)):
                    label = _text(_IOR.IOReportStateGetNameForIndex(channel, state))
                    states[label] = int(_IOR.IOReportStateGetResidency(channel, state))
                result[name] = states
        finally:
            _CF.CFRelease(difference)
        return result

    @staticmethod
    def release(sample) -> None:
        if sample:
            _CF.CFRelease(sample)


def summarize(states: dict[str, int]) -> dict:
    """Active share per state, and the mean active state index.

    State names order from idle to the highest clock. ``OFF`` and ``IDLE``
    are not active time. The mean index over active time is a clock proxy
    that needs no frequency table.
    """
    def level(name: str, position: int) -> int:
        # "P7" is level 7. Fall back to the position for unnamed states.
        digits = name[1:] if name[:1].upper() == "P" else ""
        return int(digits) if digits.isdigit() else position

    active = [
        (level(name, position), name, ticks)
        for position, (name, ticks) in enumerate(states.items())
        if name.upper() not in {"OFF", "IDLE", "DOWN"}
    ]
    total = sum(states.values())
    active_ticks = sum(ticks for _, _, ticks in active)
    mean_index = (
        sum(index * ticks for index, _, ticks in active) / active_ticks
        if active_ticks else 0.0
    )
    return {
        "active_share": active_ticks / total if total else 0.0,
        "mean_active_state": mean_index,
        "active_residency": {
            name: ticks / active_ticks for _, name, ticks in active if active_ticks
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    meter = GpuStates()
    before = meter.sample()
    time.sleep(args.seconds)
    after = meter.sample()
    delta = meter.delta(before, after)
    meter.release(before)
    meter.release(after)
    print(json.dumps({name: {"states": states, **summarize(states)}
                      for name, states in delta.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
