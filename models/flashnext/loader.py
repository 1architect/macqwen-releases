"""Build Qwen3.8-Flash-Next so it runs from disk instead of from memory.

Stock mlx-vlm materializes every weight. These checkpoints exceed 16 GB of
unified memory, so macOS swaps heavily. Two large tensor families stay on disk:

    MoE experts   ->  routed rows read from the checkpoint
    n-gram table  ->  rows read per lookup

The loader infers each selected checkpoint's quantization layout.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.utils import (
    apply_generation_config_defaults,
    get_model_and_args,
    load_config,
    update_module_configs,
)

from .expert_cache import StreamingSwitchGLU
from .adaptive_topk import apply as apply_adaptive_topk
from .patch_rmsnorm import apply as apply_rmsnorm_fix
from .patch_rmsnorm import configure as configure_rmsnorm
from .qsa_chunk import apply as apply_qsa_chunk
from .compile_glue import apply as apply_compile_glue
from .ngram import (
    StreamingQuantizedEmbedding,
    StreamingShardedEmbedding,
)
from .store import SafeTensorStore
from .checkpoint_compat import (
    MTP_INDEXED,
    MTP_NONE,
    MTP_SIDECAR,
    detect_mtp_source,
    is_mtp_key,
    validate_mtp_structure,
)

STREAMED = (".switch_mlp.", ".ngram_embedding.")
_REAP_NORM_KEY = "language_model.model.hyper_connection_mixer.hc_norm.weight"
_REAP_NORM_FINGERPRINT = (
    "59f7a6ebc4e47b0dba46d5bd7f10c156bc08cb3709347e5c5dec5077e914ee79"
)


def _is_streamed(key: str) -> bool:
    return any(marker in key for marker in STREAMED)


_WIRED_GB = [0.0]


def set_wired_gb(value) -> None:
    """Set the Metal wired limit and prove it took.

    `set_wired_limit` returns the previous value, so calling it twice reads
    back what actually applied. Metal clamps above the device maximum, and a
    clamped arm would measure the wrong thing silently.
    """
    target = int(float(value) * 1e9)
    mx.set_wired_limit(target)
    got = mx.set_wired_limit(target)
    if got != target:
        raise SystemExit(
            f"wired limit clamped to {got} from {target}; refusing to report "
            f"an arm whose setting did not apply"
        )
    _WIRED_GB[0] = float(value)


def wired_gb() -> float:
    return _WIRED_GB[0]


def apply_wired_limit() -> None:
    """Ask Metal to keep some allocations GPU-resident.

    MLX wires nothing by default: `wired_limit_` is 0 in the allocator, and
    `MLX_RESIDENCY_SET_MAX_PCT` only partitions that budget rather than
    setting it. So every buffer this runtime hands the GPU is evictable, on a
    machine that sits at 80 to 135 MB free. When a residency set loses
    residency under pressure its allocations have to be made resident again,
    and Metal locks a command buffer's residency at commit, so that cost lands
    inside the GPU spans.

    Wired memory competes with the page cache the expert stream depends on, so
    a limit that is too high should show up as more physical reads per token.
    Off by default.
    """
    want = os.environ.get("FLASHNEXT_WIRED_GB")
    if not want:
        return
    set_wired_gb(want)
    print(f"wired limit: {float(want):.2f} GB", flush=True)


def _weight_map_for_detector(path: Path, store: SafeTensorStore) -> dict:
    """Rebuild an index-style weight map from loaded store refs.

    The detector needs key -> shard without reading the index twice.
    """
    return {name: ref.shard for name, ref in store.refs.items()}


def load_streaming(
    model_dir: str,
    verbose: bool = True,
    keep_vision: bool = True,
    use_mtp: bool = True,
) -> Tuple[nn.Module, dict, SafeTensorStore]:
    apply_wired_limit()
    apply_rmsnorm_fix()
    apply_adaptive_topk()
    apply_qsa_chunk()
    apply_compile_glue()
    from . import compiled

    if compiled.ENABLED:
        # The research switch was read here once and then never acted on:
        # only bench_production installed the chains. Install verifies each
        # chain with mx.array_equal first and refuses an inexact one.
        compiled.install()
    path = Path(os.path.expanduser(model_dir))
    store = SafeTensorStore(str(path))
    mtp_path = path / "model-mtp.safetensors"
    # Capability detection reads config/index/tensor layout, never the
    # directory name. Both sidecar and indexed MTP present fails closed.
    index_map = _weight_map_for_detector(path, store)
    mtp_source, indexed_mtp_keys, _indexed_mtp_shards = detect_mtp_source(
        str(path), index_map
    )
    if use_mtp and mtp_source == MTP_SIDECAR:
        store.add_shard(mtp_path.name)
    elif use_mtp and mtp_source == MTP_NONE:
        use_mtp = False
    elif use_mtp and mtp_source == MTP_INDEXED:
        # Indexed MTP tensors already live in store.refs. Register nothing.
        missing = validate_mtp_structure(store.refs)
        if missing:
            raise ValueError(
                "indexed MTP checkpoint is structurally incomplete; missing: "
                + ", ".join(missing)
            )
    elif not use_mtp:
        # Target-only mode. Keep detector source for reporting, but exclude
        # indexed MTP tensors from the resident set below.
        pass

    config = load_config(path)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_class, _ = get_model_and_args(config=config, model_path=path)
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config, ["text", "vision", "perceiver", "projector", "audio"]
    )
    model_config = apply_generation_config_defaults(model_config, config)
    model = model_class.Model(model_config)
    if not keep_vision:
        model.vision_tower = None
    if use_mtp:
        from .mtp import attach

        attach(model.language_model)
    configure_rmsnorm(model, one_centered=_norm_one_centered(config, store))

    # Lazy: shapes are known, nothing is read yet.
    weights = {}
    for shard in sorted(glob.glob(str(path / "*.safetensors"))):
        if not use_mtp and Path(shard).name == mtp_path.name:
            continue
        weights.update(mx.load(shard))

    quantization = config.get("quantization")
    mode = quantization.get("mode", "affine")

    def class_predicate(module_path, module):
        if _is_streamed(f"{module_path}."):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.size % 64 != 0:
            return False
        weight_key = f"{module_path}.weight"
        scales_key = f"{module_path}.scales"
        if weight_key not in weights or scales_key not in weights:
            return False

        # The MTP sidecar uses mixed 4/8-bit weights but has no config entry.
        # Infer every module from its logical and packed tensor shapes.
        logical_in = int(module.weight.shape[-1])
        packed_in = int(weights[weight_key].shape[-1])
        groups = int(weights[scales_key].shape[-1])
        bits = packed_in * 32 // logical_in
        group_size = logical_in // groups
        return {"group_size": group_size, "bits": bits, "mode": mode}

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=mode,
        class_predicate=class_predicate,
    )

    stream_embed = _swap_embedding(model, store, mode, use_mtp)
    swapped_experts = _swap_experts(model, store, mode)
    swapped_ngram = _swap_ngram(model, store, mode)
    if use_mtp:
        from .mtp import swap_streaming

        swap_streaming(model.language_model, store, mode)

    resident = {
        k: v
        for k, v in weights.items()
        if not _is_streamed(k) and (use_mtp or not is_mtp_key(k))
        and not (stream_embed and k.startswith(_EMBED_PREFIX + "."))
    }
    if not keep_vision:
        # Decode speed tracks how much of the expert pool the page cache can
        # hold, so RAM given to the vision tower is RAM taken from that cache.
        # A text-only agent never runs it.
        resident = {k: v for k, v in resident.items() if not k.startswith("vision_tower")}
    _sanitize_conv1d_weights(model, resident)
    model.load_weights(list(resident.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()

    if verbose:
        streamed_bytes = sum(
            store.refs[k].shape and _nbytes(store, k)
            for k in weights
            if _is_streamed(k) and (use_mtp or not is_mtp_key(k))
        )
        print(
            f"  resident  : {len(resident)} tensors, "
            f"{sum(_nbytes(store, k) for k in resident)/1e9:.2f} GB"
        )
        print(f"  streaming : {streamed_bytes/1e9:.2f} GB")
        print(f"  streamed  : {swapped_experts} MoE blocks, {swapped_ngram} n-gram shards")
        if use_mtp:
            print(f"  MTP       : on ({mtp_source}), experts on the drive")
        else:
            print(f"  MTP       : off (source: {mtp_source})")

    return model, config, store


_EMBED_PREFIX = "language_model.model.embed_tokens"


def _swap_embedding(model, store, mode, use_mtp) -> bool:
    """Serve input-embedding rows from the checkpoint (FLASHNEXT_STREAM_EMBED=1).

    Decode looks up one row per token, so the 397 MB quantized table is almost
    entirely cold. Resident, it is anonymous Metal memory that macOS
    compresses or swaps under pressure. Streamed, its rows stay in the page
    cache and can be dropped. The rows go through the same ``mx.dequantize``
    as ``QuantizedEmbedding``, so the values are identical. MTP needs the
    whole table as its tied head, so it keeps the resident table.
    """
    if os.environ.get("FLASHNEXT_STREAM_EMBED", "0") != "1" or use_mtp:
        return False
    if f"{_EMBED_PREFIX}.scales" not in store.refs:
        return False
    inner = model.language_model.model
    dims = int(store.shape(f"{_EMBED_PREFIX}.weight")[-1]) * 32
    dims //= infer_embed_bits(store)
    inner.embed_tokens = StreamingQuantizedEmbedding(store, _EMBED_PREFIX, dims, mode)
    return True


def infer_embed_bits(store) -> int:
    """Bits of the quantized embedding from its packed and scale shapes."""
    packed = int(store.shape(f"{_EMBED_PREFIX}.weight")[-1])
    groups = int(store.shape(f"{_EMBED_PREFIX}.scales")[-1])
    for bits in (4, 8, 2, 3, 5, 6):
        logical = packed * 32 // bits
        if logical % groups == 0 and logical // groups in (32, 64, 128):
            return bits
    raise ValueError("cannot infer embedding quantization")


def _sanitize_conv1d_weights(model: nn.Module, weights: dict) -> None:
    """Convert Conv1d weights only when the target module needs it.

    Upstream Qwen sanitization moves every Conv1d tensor whose last dimension
    is not one. REAP contains both layouts in one checkpoint: linear-attention tensors already
    match MLX, while the PLE depthwise tensor needs its kernel and channel
    axes exchanged. Compare each tensor with its actual target shape so old
    checkpoints remain unchanged and compatible tensors are not transposed.
    """
    targets = {
        f"{path}.weight": module.weight.shape
        for path, module in model.named_modules()
        if path and isinstance(module, nn.Conv1d) and hasattr(module, "weight")
    }
    for key, value in list(weights.items()):
        expected = targets.get(key)
        if expected is None or value.shape == expected:
            continue
        if value.ndim == 3:
            converted = value.moveaxis(2, 1)
            if converted.shape == expected:
                weights[key] = converted
                continue
        raise ValueError(
            f"incompatible Conv1d weight {key}: checkpoint {value.shape}, "
            f"target {expected}"
        )


def _nbytes(store: SafeTensorStore, key: str) -> int:
    ref = store.refs[key]
    total = 1
    for dim in ref.shape:
        total *= dim
    from .store import _DTYPES

    return total * _DTYPES[ref.dtype][0].itemsize


def _swap_experts(model, store, mode) -> int:
    layers = model.language_model.model.layers
    count = 0
    for index, layer in enumerate(layers):
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "switch_mlp"):
            continue
        prefix = f"language_model.model.layers.{index}.mlp.switch_mlp"
        if f"{prefix}.gate_proj.weight" not in store.refs:
            continue
        old = block.switch_mlp
        block._flashnext_layer_id = index
        group_size, bits = infer_switch_quantization(store, prefix)
        nxt = f"language_model.model.layers.{index + 1}.mlp.switch_mlp"
        block.switch_mlp = StreamingSwitchGLU(
            store, prefix, group_size, bits, mode, old.activation,
            layer_id=index,
            next_prefix=nxt if f"{nxt}.gate_proj.weight" in store.refs else prefix,
        )
        count += 1
    return count


def infer_switch_quantization(store: SafeTensorStore, prefix: str):
    """Infer one expert block's quantization from its tensor shapes.

    The top-level config describes the export in aggregate, but converted
    checkpoints can contain layers with different packed layouts.  The
    streaming loader already derives the runtime arguments from the tensors
    for each layer; pin-profile provenance must use this same calculation.
    """
    # ``down_proj.weight`` stores [experts, hidden, packed_intermediate].
    # Its second dimension is therefore the logical input width shared by the
    # gate/up projections.  Recover bits from the packed gate width and the
    # group size from the gate scale count.
    packed = int(store.shape(f"{prefix}.gate_proj.weight")[-1])
    groups = int(store.shape(f"{prefix}.gate_proj.scales")[-1])
    hidden = int(store.shape(f"{prefix}.down_proj.weight")[1])
    if packed <= 0 or groups <= 0 or hidden <= 0:
        raise ValueError(f"invalid switch quantization shapes at {prefix}")
    packed_bits = packed * 32
    if packed_bits % hidden or hidden % groups:
        raise ValueError(f"incompatible switch quantization shapes at {prefix}")
    bits = packed_bits // hidden
    group_size = hidden // groups
    return group_size, bits


# Keep the historical private name for focused compatibility tests and local
# callers while making the shared implementation explicit for new consumers.
_infer_switch_quant = infer_switch_quantization


def _swap_ngram(model, store, mode) -> int:
    count = 0
    layers = model.language_model.model.layers
    for index, layer in enumerate(layers):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        table = getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
        if table is None:
            continue
        base = f"language_model.model.layers.{index}.ple.ple_embedding.ngram_embedding"
        shards = []
        for shard_index in range(len(table.shards)):
            prefix = _ngram_shard_prefix(store, base, shard_index)
            if prefix is None:
                break
            shards.append(
                StreamingQuantizedEmbedding(store, prefix, table.dims, mode)
            )
            count += 1
        if len(shards) != len(table.shards):
            raise ValueError(
                f"incomplete n-gram shard set at layer {index}: "
                f"found {len(shards)} of {len(table.shards)}"
            )
        ple.ple_embedding.ngram_embedding = StreamingShardedEmbedding(
            shards, table.shard_sizes, table.dims
        )
    return count


def _ngram_shard_prefix(store, base: str, shard_index: int) -> str | None:
    """Return the on-disk name for an n-gram shard.

    Older converted checkpoints use ``shard_N``. REAP exports use the
    upstream ``shards.N`` spelling. Probe the weight key only, so incomplete
    shards fail closed and the existing replacement is not installed.
    """
    candidates = (
        f"{base}.shards.{shard_index}",
        f"{base}.shard_{shard_index}",
    )
    for prefix in candidates:
        if f"{prefix}.weight" in store.refs:
            return prefix
    return None


def _norm_one_centered(config: dict, store: SafeTensorStore) -> bool:
    """Select the norm convention without guessing from a model family.

    The historical conversion has one-centered gains and remains the default.
    A converter can declare the newer zero-centered convention in config.
    ``FLASHNEXT_NORM_CONVENTION`` is an explicit escape hatch for checkpoints
    whose metadata predates the convention field.
    """
    override = os.environ.get("FLASHNEXT_NORM_CONVENTION", "").strip().lower()
    if override == "auto":
        override = ""
    if override in {"one", "one-centered", "one_centered"}:
        return True
    if override in {"zero", "zero-centered", "zero_centered"}:
        return False
    if override:
        raise ValueError("FLASHNEXT_NORM_CONVENTION must be one or zero")
    for source in (config, config.get("text_config", {})):
        value = source.get("norm_convention") or source.get("rms_norm_convention")
        if isinstance(value, str):
            value = value.lower().replace("_", "-")
            if value in {"zero", "zero-centered"}:
                return False
            if value in {"one", "one-centered"}:
                return True
    # The sh0wie REAP export is the only known corrected artifact. Fingerprint
    # one small, stable norm tensor. Unknown derivatives retain legacy mode.
    key = _REAP_NORM_KEY
    ref = store.refs.get(key)
    if ref is not None:
        import hashlib

        size = _nbytes(store, key)
        with open(os.path.join(store.dir, ref.shard), "rb") as handle:
            handle.seek(ref.start)
            digest = hashlib.sha256(handle.read(size)).hexdigest()
        if digest == _REAP_NORM_FINGERPRINT:
            return False
    return True
