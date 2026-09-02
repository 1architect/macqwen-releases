"""Build Qwen3.8-Flash-Next so it runs from disk instead of from memory.

Stock mlx-vlm materializes every weight. These checkpoints exceed 16 GB of
unified memory, so macOS swaps heavily. Two large tensor families stay on disk:

    MoE experts   ->  bounded LRU of routed experts
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
from .qsa_chunk import apply as apply_qsa_chunk
from .ngram import (
    StreamingQuantizedEmbedding,
    StreamingShardedEmbedding,
)
from .store import SafeTensorStore

STREAMED = (".switch_mlp.", ".ngram_embedding.")


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


def load_streaming(
    model_dir: str,
    expert_capacity: int = 32,
    ngram_capacity: int = 0,
    verbose: bool = True,
    keep_vision: bool = True,
    use_mtp: bool = True,
) -> Tuple[nn.Module, dict, SafeTensorStore]:
    apply_wired_limit()
    apply_rmsnorm_fix()
    apply_adaptive_topk()
    apply_qsa_chunk()
    path = Path(os.path.expanduser(model_dir))
    store = SafeTensorStore(str(path))
    mtp_path = path / "model-mtp.safetensors"
    use_mtp = bool(use_mtp and mtp_path.is_file())
    if use_mtp:
        store.add_shard(mtp_path.name)

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

    swapped_experts = _swap_experts(model, store, expert_capacity, mode)
    swapped_ngram = _swap_ngram(model, store, ngram_capacity, mode)
    if use_mtp:
        from .mtp import swap_streaming

        swap_streaming(model.language_model, store, mode, expert_capacity)

    resident = {k: v for k, v in weights.items() if not _is_streamed(k)}
    if not keep_vision:
        # Decode speed tracks how much of the expert pool the page cache can
        # hold, so RAM given to the vision tower is RAM taken from that cache.
        # A text-only agent never runs it.
        resident = {k: v for k, v in resident.items() if not k.startswith("vision_tower")}
    model.load_weights(list(resident.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()

    if verbose:
        streamed_bytes = sum(
            store.refs[k].shape and _nbytes(store, k) for k in weights if _is_streamed(k)
        )
        print(
            f"  resident  : {len(resident)} tensors, "
            f"{sum(_nbytes(store, k) for k in resident)/1e9:.2f} GB"
        )
        print(f"  streaming : {streamed_bytes/1e9:.2f} GB")
        print(f"  streamed  : {swapped_experts} MoE blocks, {swapped_ngram} n-gram shards")
        print(f"  MTP       : {'on, experts on the drive' if use_mtp else 'off'}")

    return model, config, store


def _nbytes(store: SafeTensorStore, key: str) -> int:
    ref = store.refs[key]
    total = 1
    for dim in ref.shape:
        total *= dim
    from .store import _DTYPES

    return total * _DTYPES[ref.dtype][0].itemsize


def _swap_experts(model, store, capacity, mode) -> int:
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
        group_size, bits = _infer_switch_quant(store, prefix)
        nxt = f"language_model.model.layers.{index + 1}.mlp.switch_mlp"
        block.switch_mlp = StreamingSwitchGLU(
            store, prefix, group_size, bits, mode, capacity, old.activation,
            layer_id=index,
            next_prefix=nxt if f"{nxt}.gate_proj.weight" in store.refs else prefix,
        )
        count += 1
    return count


def _infer_switch_quant(store: SafeTensorStore, prefix: str):
    # gate_proj input dim is the model hidden size; recover bits from packing.
    packed = store.shape(f"{prefix}.gate_proj.weight")[-1]
    groups = store.shape(f"{prefix}.gate_proj.scales")[-1]
    down_out = store.shape(f"{prefix}.down_proj.weight")[1]
    bits = packed * 32 // down_out
    group_size = down_out // groups
    return group_size, bits


def _swap_ngram(model, store, capacity, mode) -> int:
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
            prefix = f"{base}.shard_{shard_index}"
            if f"{prefix}.weight" not in store.refs:
                break
            shards.append(
                StreamingQuantizedEmbedding(
                    store, prefix, table.dims, mode, capacity
                )
            )
            count += 1
        if len(shards) == len(table.shards):
            ple.ple_embedding.ngram_embedding = StreamingShardedEmbedding(
                shards,
                table.shard_sizes,
                table.dims,
            )
    return count
