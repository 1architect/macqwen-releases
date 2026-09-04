"""Internal and research-only runtime switches used by benchmarks."""
from __future__ import annotations

import os

from macqwen.backend_settings import Setting


def env(key, default="0"):
    return lambda _backend: os.environ.get(key, default)


def flag(key):
    return lambda _backend: os.environ.get(key, "0") == "1"


def text(raw):
    return raw


def integer(raw):
    return int(raw)


def research(name, key, default="0", lifecycle="startup", parser=text):
    return Setting(
        name, (key,), default, parser, lifecycle, "research", "research-only", "flashnext",
        env(key, default), None, flag(key), None, key, __file__,
    )


SETTINGS = (
    research("prewarm", "FLASHNEXT_PREWARM", lifecycle="startup"),
    research("slab", "FLASHNEXT_SLAB", lifecycle="startup"),
    research("slab-layers", "FLASHNEXT_SLAB_LAYERS", "0", "startup", integer),
    research("slab-min-slots", "FLASHNEXT_SLAB_MIN_SLOTS", "4", "startup", integer),
    research("slab-max-slots", "FLASHNEXT_SLAB_MAX_SLOTS", "6", "startup", integer),
    research("slab-num-layers", "FLASHNEXT_SLAB_NUM_LAYERS", "12", "startup", integer),
    research("slab-pack-require-existing", "FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING", "0", "startup"),
    research("slab-pack-expected-path", "FLASHNEXT_SLAB_PACK_EXPECTED_PATH", "", "startup"),
    research("ngram-nocache", "FLASHNEXT_NGRAM_NOCACHE"),
    research("buffer-arena", "FLASHNEXT_BUFFER_ARENA", "0", "startup", integer),
    research("sort-reads", "FLASHNEXT_SORT_READS"),
    research("swap-resident", "FLASHNEXT_SWAP_RESIDENT"),
    research("track-resident", "FLASHNEXT_TRACK_RESIDENT"),
    research("pin-parts", "FLASHNEXT_PIN_PARTS", "all"),
    research("shared-read-buffer", "FLASHNEXT_SHARED_READ_BUFFER"),
    research("one-sync", "FLASHNEXT_ONE_SYNC"),
    research("compile", "FLASHNEXT_COMPILE"),
    research("wired-gb", "FLASHNEXT_WIRED_GB", "0", "startup", float),
    research("swap-max-rows", "FLASHNEXT_SWAP_MAX_ROWS", "4", "startup", integer),
    research("warm", "FLASHNEXT_WARM"),
    research("early-submit", "FLASHNEXT_EARLY_SUBMIT"),
    research("profile-io", "FLASHNEXT_PROFILE_IO"),
    research("profile-boundaries", "FLASHNEXT_PROFILE_BOUNDARIES"),
    research("profile-boundary", "FLASHNEXT_PROFILE_BOUNDARY", ""),
    research("stream-pack-chunk", "FLASHNEXT_STREAM_PACK_CHUNK", "0", "startup", integer),
    research("physical-miss-trace", "FLASHNEXT_PHYSICAL_MISS_TRACE", "0", "startup"),
    research("physical-miss-profile", "FLASHNEXT_PHYSICAL_MISS_PROFILE", "~/.cache/flashnext/physical-misses.json", "startup"),
    research("physical-miss-min-samples", "FLASHNEXT_PHYSICAL_MISS_MIN_SAMPLES", "1", "startup", integer),
)
