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


def positive_integer(raw):
    value = int(raw)
    if value <= 0:
        raise ValueError("value must be positive")
    return value


def choice(options, name):
    def parse(raw):
        if raw not in options:
            raise ValueError(f"{name} must be one of: {', '.join(options)}")
        return raw
    return parse


def norm_convention(raw):
    value = str(raw).strip().lower()
    if value in {"", "auto"}:
        return ""
    if value in {"one", "zero"}:
        return value
    raise ValueError("norm-convention must be auto, one, or zero")


def research(name, key, default="0", lifecycle="startup", parser=text, unset=None):
    """``default`` is the chat default; ``unset`` is what the runtime does
    when the variable is absent, when the two differ."""
    return Setting(
        name, (key,), default, parser, lifecycle, "research", "research-only", "flashnext",
        env(key, default if unset is None else unset), None, flag(key), None, key, __file__,
    )


SETTINGS = (
    research("prewarm", "FLASHNEXT_PREWARM", lifecycle="startup"),
    research("slab-min-slots", "FLASHNEXT_SLAB_MIN_SLOTS", "4", "startup", integer),
    research("slab-max-slots", "FLASHNEXT_SLAB_MAX_SLOTS", "6", "startup", integer),
    research("slab-num-layers", "FLASHNEXT_SLAB_NUM_LAYERS", "12", "startup", integer),
    research("slab-pack-require-existing", "FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING", "0", "startup"),
    research("slab-pack-expected-path", "FLASHNEXT_SLAB_PACK_EXPECTED_PATH", "", "startup"),
    research("swap-resident", "FLASHNEXT_SWAP_RESIDENT"),
    research("track-resident", "FLASHNEXT_TRACK_RESIDENT"),
    research("pin-parts", "FLASHNEXT_PIN_PARTS", "all"),
    research("shared-read-buffer", "FLASHNEXT_SHARED_READ_BUFFER"),
    research("compile", "FLASHNEXT_COMPILE"),
    research("qsa-cache-pooled-keys", "FLASHNEXT_QSA_CACHE_POOLED_KEYS", "1", parser=choice(("0", "1"), "qsa-cache-pooled-keys"), unset="0"),
    research("qsa-scatter-decode", "FLASHNEXT_QSA_SCATTER_DECODE", "1", parser=choice(("0", "1"), "qsa-scatter-decode"), unset="0"),
    research("overlap", "FLASHNEXT_OVERLAP", "0", parser=choice(("0", "1"), "overlap"), unset="1"),
    research("expert-lock-gb", "FLASHNEXT_EXPERT_LOCK_GB", "0", "startup", text),
    research("rdahead", "FLASHNEXT_RDAHEAD", "0", parser=choice(("0", "1"), "rdahead"), unset="1"),
    research("wired-gb", "FLASHNEXT_WIRED_GB", "0", "startup", float),
    research("swap-max-rows", "FLASHNEXT_SWAP_MAX_ROWS", "4", "startup", integer),
    research("profile-io", "FLASHNEXT_PROFILE_IO"),
    research("profile-boundaries", "FLASHNEXT_PROFILE_BOUNDARIES"),
    research("profile-boundary", "FLASHNEXT_PROFILE_BOUNDARY", ""),
    research("stream-pack-chunk", "FLASHNEXT_STREAM_PACK_CHUNK", "2", "startup", integer, unset="0"),
    research("physical-miss-trace", "FLASHNEXT_PHYSICAL_MISS_TRACE", "0", "startup"),
    research("physical-miss-profile", "FLASHNEXT_PHYSICAL_MISS_PROFILE", "~/.cache/flashnext/physical-misses.json", "startup"),
    research("physical-miss-min-samples", "FLASHNEXT_PHYSICAL_MISS_MIN_SAMPLES", "1", "startup", integer),
    research("gpu-keepwarm-iters", "FLASHNEXT_GPU_KEEPWARM_ITERS", "60000", "startup", integer),
    research("gpu-keepwarm-period-ms", "FLASHNEXT_GPU_KEEPWARM_PERIOD_MS", "0.5", "startup", float),
    research("gpu-keepwarm-stream-pack", "FLASHNEXT_GPU_KEEPWARM_STREAM_PACK", "1", "startup"),
    research("small-sidecar", "FLASHNEXT_SMALL_SIDECAR", "", "startup"),
    research("one-sync", "FLASHNEXT_ONE_SYNC"),
    research("compile-gdn", "FLASHNEXT_COMPILE_GDN"),
    research("compile-hc", "FLASHNEXT_COMPILE_HC", "1", parser=choice(("0", "1"), "compile-hc"), unset="0"),
    research("compile-norm", "FLASHNEXT_COMPILE_NORM", "1", parser=choice(("0", "1"), "compile-norm"), unset="0"),
    research("stream-embed", "FLASHNEXT_STREAM_EMBED", "1", "startup", choice(("0", "1"), "stream-embed"), unset="0"),
    research("ngram-parallel-min", "FLASHNEXT_NGRAM_PARALLEL_MIN", "64", "startup", integer, unset="0"),
    research("io-qos", "FLASHNEXT_IO_QOS", "user-interactive", "startup", choice(("default", "user-interactive", "user-initiated", "utility"), "io-qos"), unset="default"),
    research("norm-weight-cache", "FLASHNEXT_NORM_WEIGHT_CACHE", "1", parser=choice(("0", "1"), "norm-weight-cache"), unset="0"),
    research("slab-counts", "FLASHNEXT_SLAB_COUNTS", "turn", "startup", choice(("turn", "cumulative"), "slab-counts")),
    research("slab-profile", "FLASHNEXT_SLAB_PROFILE", "frozen", "startup", choice(("rolling", "frozen"), "slab-profile"), unset="rolling"),
    research("slab-counts-decay", "FLASHNEXT_SLAB_COUNTS_DECAY", "0.9", "startup", float),
    research("slab-pack-max-age-days", "FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS", "14", "startup", float),
    Setting(
        "metal-g64", ("FLASHNEXT_METAL_G64",), "1", choice(("0", "1"), "metal-g64"),
        "startup", "runtime", "public", "flashnext",
        env("FLASHNEXT_METAL_G64", "1"), None,
        lambda _backend: os.environ.get("FLASHNEXT_METAL_G64", "1") == "1",
        None, "FLASHNEXT_METAL_G64", __file__,
    ),
    Setting(
        "slab-g64", ("FLASHNEXT_SLAB_G64",), "0", choice(("0", "1"), "slab-g64"),
        "startup", "storage", "research-only", "flashnext",
        env("FLASHNEXT_SLAB_G64", "0"), None,
        flag("FLASHNEXT_SLAB_G64"), None, "FLASHNEXT_SLAB_G64", __file__,
    ),
    Setting(
        "norm-convention", ("FLASHNEXT_NORM_CONVENTION",), "", norm_convention,
        "startup", "runtime", "research-only", "flashnext",
        env("FLASHNEXT_NORM_CONVENTION", ""), None,
        lambda _backend: bool(os.environ.get("FLASHNEXT_NORM_CONVENTION")),
        None, "FLASHNEXT_NORM_CONVENTION", __file__,
    ),
    Setting(
        "qsa-dense-mask-max-bytes", ("FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES",),
        512 * 1024 * 1024, positive_integer, "startup", "runtime", "research-only",
        "flashnext",
        lambda _backend: positive_integer(os.environ.get(
            "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES", str(512 * 1024 * 1024)
        )),
        None, lambda _backend: "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES" in os.environ,
        None, "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES", __file__,
    ),
)
