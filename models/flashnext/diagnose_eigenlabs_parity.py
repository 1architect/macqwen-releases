"""Stock-vs-MACQWEN parity diagnostic for Flash-Next checkpoints.

Compares unmodified upstream ``mlx_vlm`` arithmetic against the MACQWEN
streaming runtime on the same checkpoint, layer by layer, under a hard
memory budget: no child may exceed 8 GiB RSS on the 16 GiB reference
machine. Normal target is below 6 GiB.

Memory rules enforced by construction:

- The stock arm never builds a full expert bank. Routed experts are
  served sparsely per call (tens of rows, not 512 experts).
- PLE n-gram tables are replaced immediately after layer construction,
  before quantization or evaluation, and served sparsely row by row.
- Shard files are never loaded whole; all reads are positioned.
- Every child runs under a parent watchdog that kills it past 8 GiB.

Modes (each runs in its own fresh child process):

- ``dump-macqwen``: streaming reference captures (embedding, per-layer
  inputs/outputs, per-norm records, routes, PLE ids, final, logits).
- ``dump-stock-embedding``: stock embedding tables only.
- ``dump-stock-layer``: one stock decoder layer with sparse experts.
- ``dump-stock-final``: stock final mixer only.
- ``dump-stock-head``: stock lm_head only.
- ``verify-rows``: positioned pread bytes versus ``SafeTensorStore``.
- ``norm-sweep``: both RMSNorm formulas against captured records.
- ``compare``: PASS/FAIL per stage over a sweep root.
- ``sweep``: driver running staged children in gate order, stopping at
  the first divergence.

This module is diagnostic code. The runtime never imports it. It changes
no defaults and enables no optimizations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = "Hello, world! This is a parity check."
DEFAULT_IDS = 4

# Hard kill threshold for any diagnostic child. Breaching it fails the arm;
# the limit must never be raised to make a run pass.
MAX_CHILD_RSS_GB = 8.0
_WATCHDOG_POLL_SECONDS = 0.2
_WATCHDOG_TERM_GRACE_SECONDS = 2.0
_CHILD_TIMEOUT_SECONDS = 1800

# Parity children must not inherit experimental FlashNext settings. Norm
# convention stays automatic (the variable is removed) so fingerprint and
# config selection run exactly as in normal use.
CHILD_ENV = {
    "FLASHNEXT_METAL_RUNTIME": "0",
    "FLASHNEXT_METAL_G64": "0",
    "FLASHNEXT_PHASE_STREAM": "0",
    "FLASHNEXT_TOPK_THRESHOLD": "1.0",
    "FLASHNEXT_SLAB": "0",
    "FLASHNEXT_SLAB_GLOBAL": "0",
    "FLASHNEXT_SLAB_PACK": "0",
    "FLASHNEXT_SLAB_G64": "0",
    "FLASHNEXT_PREWARM": "0",
    "FLASHNEXT_STREAM_PACK": "0",
    "FLASHNEXT_SWAP_RESIDENT": "0",
    "FLASHNEXT_EARLY_SUBMIT": "0",
    "FLASHNEXT_WARM": "0",
    "FLASHNEXT_QSA_CACHE_POOLED_KEYS": "0",
    "FLASHNEXT_QSA_SCATTER_DECODE": "0",
}

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _materialize(value):
    import mlx.core as mx

    mx.eval(value)
    dtype = str(value.dtype)
    if "bfloat16" in dtype:
        return np.asarray(value.astype(mx.float32)), dtype
    return np.asarray(value), dtype


def _save_array(path: Path, value) -> dict:
    import mlx.core as mx

    mx.eval(value)
    dtype = str(value.dtype)
    arr, _ = _materialize(value)
    np.save(path, arr)
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:24]
    return {"file": path.name, "shape": list(arr.shape), "dtype": dtype, "digest": digest}


def _short(prompt: str, count: int, tokenizer) -> list[int]:
    ids = tokenizer.encode(prompt)
    if len(ids) < count:
        raise ValueError(f"prompt encodes to {len(ids)} tokens, need {count}")
    return [int(v) for v in ids[:count]]


def _rss_gb() -> float:
    """Peak RSS so far; diagnostic metadata only, never a safety gate."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _mx_mem_gb() -> dict:
    """Active/cache Metal memory when the installed MLX exposes it."""
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover
        return {}
    info = {}
    getter = getattr(mx, "get_active_memory", None)
    if callable(getter):
        try:
            info["mx_active_gb"] = round(float(getter()) / 1e9, 3)
        except Exception:  # pragma: no cover
            pass
    getter = getattr(mx, "get_cache_memory", None)
    if callable(getter):
        try:
            info["mx_cache_gb"] = round(float(getter()) / 1e9, 3)
        except Exception:  # pragma: no cover
            pass
    return info


def _log_rss(stage: str) -> None:
    detail = " ".join(f"{k}={v}" for k, v in sorted(_mx_mem_gb().items()))
    suffix = f" {detail}" if detail else ""
    print(f"rss {stage}: {_rss_gb():.2f} GiB{suffix}", flush=True)


def _clear_eager() -> None:
    import mlx.core as mx

    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _fresh_caches(flags):
    """Mirror Qwen4Exp LanguageModel.make_cache: ArraysCache(4) for PLE
    linear layers, ArraysCache(2) for other linear layers, QSAKVCache
    for full-attention layers. Flags are (is_linear, has_ple) pairs."""
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    return [ArraysCache(4 if ple else 2) if linear else QSAKVCache()
            for linear, ple in flags]


# ---------------------------------------------------------------------------
# dump-macqwen (streaming reference; captures stream straight to disk)
# ---------------------------------------------------------------------------

def cmd_dump_macqwen(args) -> Path:
    for key, value in CHILD_ENV.items():
        os.environ[key] = value
    os.environ.pop("FLASHNEXT_NORM_CONVENTION", None)
    if getattr(args, "norm_convention", "auto") != "auto":
        # Explicit diagnostic control only; primary runs stay automatic.
        os.environ["FLASHNEXT_NORM_CONVENTION"] = args.norm_convention
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer, Qwen4ExpRMSNorm

    from models.flashnext.loader import load_streaming

    _log_rss("macqwen process start")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, config, _store = load_streaming(
        args.model, expert_capacity=0, ngram_capacity=0, verbose=False,
        keep_vision=False, use_mtp=False,
    )
    _log_rss("macqwen model loaded")
    language = model.language_model
    qwen = language.model

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = _short(args.prompt, args.tokens, tokenizer)
    del tokenizer

    is_linear = [layer.is_linear for layer in qwen.layers]
    has_ple = ["ple" in layer for layer in qwen.layers]
    records: dict = {"ids": ids, "captures": {}}

    orig_layer_call = Qwen4ExpDecoderLayer.__call__
    layer_order = {id(layer): index for index, layer in enumerate(qwen.layers)}

    def layer_call(layer_self, hidden_states, input_ids, mask=None, cache=None,
                   position_ids=None):
        mx.eval(hidden_states, input_ids)
        out_value = orig_layer_call(
            layer_self, hidden_states, input_ids, mask=mask, cache=cache,
            position_ids=position_ids,
        )
        mx.eval(out_value)
        lid = layer_order[id(layer_self)]
        # Stream to disk immediately; keep only filenames, never arrays.
        rec_in = _save_array(out / f"layer_{lid:02d}_in.npy", hidden_states)
        rec_out = _save_array(out / f"layer_{lid:02d}_out.npy", out_value)
        if mask is None or isinstance(mask, str):
            rec_in["mask"] = {"kind": "none" if mask is None else "flag",
                              "value": mask}
        else:
            mx.eval(mask)
            arr_m, _ = _materialize(mask)
            np.save(out / f"layer_{lid:02d}_mask.npy", arr_m)
            rec_in["mask"] = {"kind": "array"}
        records["captures"][f"layer_{lid:02d}_in"] = rec_in
        records["captures"][f"layer_{lid:02d}_out"] = rec_out
        _clear_eager()
        return out_value

    orig_norm_call = Qwen4ExpRMSNorm.__call__
    norm_manifest: list[dict] = []

    def norm_call(norm_self, x):
        out_value = orig_norm_call(norm_self, x)
        mx.eval(x, out_value)
        arr_x, _ = _materialize(x)
        arr_o, _ = _materialize(out_value)
        arr_w, _ = _materialize(norm_self.weight)
        idx = len(norm_manifest)
        entry = {
            "path": str(getattr(norm_self, "_parity_path", "?")),
            "group_size": norm_self.group_size,
            "eps": float(norm_self.eps),
        }
        for key, arr in (("x", arr_x), ("out", arr_o), ("weight", arr_w)):
            entry[key] = _save_array(out / f"norm_{idx:03d}_{key}.npy", arr)
        norm_manifest.append(entry)
        del arr_x, arr_o, arr_w
        _clear_eager()
        return out_value

    for name, module in qwen.named_modules():
        if module.__class__.__name__ == "Qwen4ExpRMSNorm":
            module._parity_path = name
    mixer = qwen.hyper_connection_mixer
    for name, module in mixer.named_modules():
        if module.__class__.__name__ == "Qwen4ExpRMSNorm":
            module._parity_path = f"mixer.{name}"

    Qwen4ExpDecoderLayer.__call__ = layer_call
    Qwen4ExpRMSNorm.__call__ = norm_call
    routes: dict = {}
    ple_ids: dict = {}
    try:
        from models.flashnext.adaptive_topk import _ROUTE_OBSERVER
        from models.flashnext.ngram import StreamingShardedEmbedding

        def observer(layer_id, rows, score_rows, keeps):
            del keeps
            routes[str(layer_id)] = {"inds": rows, "scores": score_rows}

        previous_observer = _ROUTE_OBSERVER[0]
        _ROUTE_OBSERVER[0] = observer
        orig_sharded_call = StreamingShardedEmbedding.__call__

        def sharded_call(sharded_self, indices):
            flat = indices.reshape(-1)
            mx.eval(flat)
            ple_ids.setdefault(str(len(ple_ids)), []).append(
                [int(v) for v in flat.tolist()])
            return orig_sharded_call(sharded_self, indices)

        StreamingShardedEmbedding.__call__ = sharded_call
        observer_on = True
    except ImportError:
        observer_on = False
    try:
        hidden = qwen.embed_tokens(mx.array([ids], dtype=mx.uint32))
        mx.eval(hidden)
        records["captures"]["embed_raw"] = _save_array(out / "embed_raw.npy", hidden)
        tiled = mx.tile(hidden, (1, 1, qwen.args.hc_count))
        mx.eval(tiled)
        records["captures"]["embed_tiled"] = _save_array(out / "embed_tiled.npy", tiled)
        del hidden, tiled
        caches = _fresh_caches(list(zip(is_linear, has_ple)))
        final = qwen(mx.array([ids], dtype=mx.uint32), cache=caches)
        mx.eval(final)
        records["captures"]["final"] = _save_array(out / "final.npy", final)
        logits = language.lm_head(final)
        mx.eval(logits)
        records["captures"]["logits"] = _save_array(out / "logits.npy", logits)
        del final, logits
    finally:
        Qwen4ExpDecoderLayer.__call__ = orig_layer_call
        Qwen4ExpRMSNorm.__call__ = orig_norm_call
        if observer_on:
            _ROUTE_OBSERVER[0] = previous_observer
            StreamingShardedEmbedding.__call__ = orig_sharded_call
    records["routes"] = routes
    records["ple_ids"] = ple_ids
    records["norms"] = norm_manifest
    records["env_norm_convention"] = os.environ.get("FLASHNEXT_NORM_CONVENTION")
    try:
        records["layer0_hc_norm_one_centered"] = bool(
            qwen.layers[0].attn_hyper_connection.hc_norm._flashnext_one_centered)
    except AttributeError:
        records["layer0_hc_norm_one_centered"] = None
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    _log_rss("macqwen before exit")
    print(json.dumps({"dumped": str(out),
                      "layers": sum(1 for k in records["captures"] if k.startswith("layer_") and k.endswith("_out")),
                      "norms": len(norm_manifest), "ids": ids}))
    return out


# ---------------------------------------------------------------------------
# stock arm (pure upstream; never imports macqwen runtime modules)
# ---------------------------------------------------------------------------

def _stock_config(model_dir: str):
    from mlx_vlm.utils import get_model_and_args, load_config, update_module_configs
    from mlx_vlm.utils import apply_generation_config_defaults

    path = Path(model_dir).expanduser()
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
    return path, config, model_class, model_config


def _module_overrides(config: dict) -> tuple[dict, int, int, str]:
    table: dict = {}
    for source in (config.get("quantization_config", {}) or {},
                   config.get("quantization", {}) or {}):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, dict) and "bits" in value:
                table[key] = value
    quant = config.get("quantization", {}) or {}
    return table, int(quant.get("group_size", 64)), int(quant.get("bits", 4)), str(quant.get("mode", "affine"))


def _stock_predicate(prefix: str, subset: dict, table: dict, mode_default: str = "affine"):
    """Per-module quantization geometry, inferred from tensor shapes.

    Checkpoint metadata carries no per-module overrides here, so geometry
    comes from the stored shapes exactly like the loader infers it:
    bits from packed width, group size from scale count. Anything that
    does not divide cleanly is skipped and fails closed in _assert_loaded.
    """
    def predicate(path, module):
        full = f"{prefix}.{path}" if path else prefix
        if ".switch_mlp." in full or ".ngram_embedding." in full:
            # Routed experts are served sparsely per call and PLE tables
            # sparsely row by row; never quantize their random placeholders.
            return False
        override = table.get(full)
        if isinstance(override, dict):
            return override
        if not hasattr(module, "to_quantized"):
            return False
        if not hasattr(module, "weight") or module.weight.size % 64 != 0:
            return False
        weight_key, scales_key = f"{full}.weight", f"{full}.scales"
        if weight_key not in subset or scales_key not in subset:
            return False
        logical = int(module.weight.shape[-1])
        packed = int(subset[weight_key].shape[-1])
        groups = int(subset[scales_key].shape[-1])
        if logical <= 0 or packed * 32 % logical or logical % groups:
            return False
        return {"group_size": logical // groups, "bits": packed * 32 // logical,
                "mode": mode_default}
    return predicate


def _sparse_quant_geometry(store, prefix: str, table: dict) -> tuple[int, int, str, int, int, int]:
    """Quantization geometry and dims for one MoE block, from metadata only.

    No tensor is read: expert count, widths, bits, group, and mode all come
    from shapes and config overrides.
    """
    gate_shape = tuple(store.shape(f"{prefix}.gate_proj.weight"))
    down_shape = tuple(store.shape(f"{prefix}.down_proj.weight"))
    num_experts, hidden_dims = int(gate_shape[0]), int(gate_shape[1])
    input_dims = int(down_shape[1])
    override = table.get(f"{prefix}.gate_proj")
    if isinstance(override, dict):
        group_size, bits = int(override["group_size"]), int(override["bits"])
        mode = str(override.get("mode", "affine"))
    else:
        groups = int(store.shape(f"{prefix}.gate_proj.scales")[-1])
        packed = int(gate_shape[-1])
        if packed * 32 % input_dims or input_dims % groups:
            raise ValueError(f"incompatible MoE quantization shapes at {prefix}")
        bits = packed * 32 // input_dims
        group_size = input_dims // groups
        mode = "affine"
    if bits <= 0 or group_size <= 0 or input_dims <= 0 or hidden_dims <= 0:
        raise ValueError(f"invalid sparse MoE geometry at {prefix}")
    return group_size, bits, mode, num_experts, input_dims, hidden_dims


class SparseStockSwitchGLU:
    """Upstream-arithmetic MoE that stores no experts.

    Diagnostic-only. It exposes the ``switch_mlp(x, indices)`` contract the
    upstream block expects, but for each call it evaluates the routed ids,
    loads only those experts' rows, computes per-expert outputs with the
    normal ``QuantizedLinear`` implementation, and discards everything.
    Routing, scores, renormalization, and the shared expert stay owned by
    the upstream ``Qwen3_5MoeSparseMoeBlock``.
    """

    def __init__(self, store, prefix: str, group_size: int, bits: int,
                 mode: str, num_experts: int, input_dims: int,
                 hidden_dims: int, activation):
        self.store = store
        self.prefix = prefix
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.num_experts = num_experts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.activation = activation

    def _expert_linears(self, expert: int):
        import mlx.core as mx
        import mlx.nn as nn

        linears = []
        for proj, out_dims, in_dims in (
            ("gate_proj", self.hidden_dims, self.input_dims),
            ("up_proj", self.hidden_dims, self.input_dims),
            ("down_proj", self.input_dims, self.hidden_dims),
        ):
            module = nn.QuantizedLinear(
                in_dims, out_dims, bias=False,
                group_size=self.group_size, bits=self.bits, mode=self.mode,
            )
            for part in ("weight", "scales", "biases"):
                row = self.store.rows(f"{self.prefix}.{proj}.{part}", [expert])
                setattr(module, part, row[0])
            linears.append(module)
        mx.eval([leaf for module in linears for leaf in
                 (module.weight, module.scales, module.biases)])
        return linears

    def __call__(self, x, indices):
        import mlx.core as mx

        mx.eval(indices)
        host = np.asarray(indices).reshape(-1).tolist()
        unique = sorted(set(int(v) for v in host))
        for expert in unique:
            if expert < 0 or expert >= self.num_experts:
                raise ValueError(f"routed expert out of range: {expert}")
        flat_in = x.reshape(-1, self.input_dims)
        slots_per_token = int(indices.shape[-1])
        slots: dict[int, object] = {}
        for expert in unique:
            gate, up, down = self._expert_linears(expert)
            positions = [i for i, v in enumerate(host) if int(v) == expert]
            token_rows = [position // slots_per_token for position in positions]
            taken = mx.take(flat_in, mx.array(token_rows, dtype=mx.uint32), axis=0)
            hidden = self.activation(up(taken), gate(taken))
            slots[expert] = (positions, down(hidden))
            del gate, up, down, taken, hidden
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        parts, order = [], []
        for expert in unique:
            positions, values = slots[expert]
            parts.append(values)
            order.extend(positions)
            del values
        packed = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        inverse = [0] * len(host)
        for packed_index, position in enumerate(order):
            inverse[position] = packed_index
        out = packed[mx.array(inverse, dtype=mx.uint32)]
        out = out.reshape(*indices.shape, self.input_dims)
        mx.eval(out)
        del parts, packed, slots
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        return out


def _load_subset(store, wanted_prefixes) -> dict:
    """Fetch only small non-streamed tensors through positioned reads.

    Routed experts and PLE tables are always excluded; they are served
    sparsely per call instead. Peak memory is the layer's dense tensors.
    """
    import mlx.core as mx

    subset: dict = {}
    for key in store.refs:
        if ".switch_mlp." in key or ".ngram_embedding." in key:
            continue
        if not any(key == prefix or key.startswith(prefix + ".")
                   for prefix in wanted_prefixes):
            continue
        ref = store.refs[key]
        count = ref.shape[0] if ref.shape else 1
        subset[key] = store.rows(key, range(count))
    mx.eval(list(subset.values()))
    return subset


def _quantize_stock(module, prefix: str, subset: dict, table: dict,
                    group_size: int, bits: int, mode: str):
    """Quantize like upstream, including the root module.

    ``nn.quantize`` only visits sub-modules, so a bare embedding or head
    root would stay float. Apply the same predicate to the root first.
    """
    import mlx.nn as nn

    predicate = _stock_predicate(prefix, subset, table, mode)
    decision = predicate("", module)
    if decision and hasattr(module, "to_quantized"):
        kwargs = (dict(decision) if isinstance(decision, dict)
                  else {"group_size": group_size, "bits": bits, "mode": mode})
        module = module.to_quantized(**kwargs)
    nn.quantize(
        module, group_size=group_size, bits=bits, mode=mode,
        class_predicate=predicate,
    )
    return module


class _RecordingNgramTables:
    """Stand-in that collects queried ids with zero embeddings.

    Id computation precedes the lookup, so dummy values cannot change
    the queried set. Only the recording run uses this.
    """

    def __init__(self, dims: int):
        self.dims = int(dims)
        self.seen: list[int] = []

    def __call__(self, indices):
        import mlx.core as mx

        flat = indices.reshape(-1)
        mx.eval(flat)
        self.seen.extend(int(v) for v in flat.tolist())
        return mx.zeros((*indices.shape, self.dims), dtype=mx.bfloat16)


class _SparseNgramTables:
    """Serve preloaded dequantized rows for a known id set.

    Ownership mirrors ShardedEmbedding (contiguous offsets); every
    queried id must be preloaded, otherwise the run fails closed.
    """

    def __init__(self, dims: int, vectors: dict[int, object]):
        import mlx.core as mx

        self.dims = int(dims)
        self.vectors = dict(vectors)
        self.queried: list[int] = []
        self._mx = mx

    def __call__(self, indices):
        mx = self._mx
        flat = indices.reshape(-1)
        mx.eval(flat)
        ids = [int(v) for v in flat.tolist()]
        self.queried.extend(ids)
        missing = [v for v in ids if v not in self.vectors]
        if missing:
            raise ValueError(f"ngram id not preloaded: {missing[:5]}")
        out = mx.stack([self.vectors[v] for v in ids])
        return out.reshape(*indices.shape, self.dims)


def _ngram_dequant_params(store, base: str, table: dict, dims: int):
    """Dequant geometry plus on-disk shard-name spelling.

    Checkpoints use either upstream ``shards.N`` or legacy ``shard_N``;
    probe exactly like the loader instead of assuming one spelling.
    """
    for spelling in ("shards", "shard"):
        probe = f"{base}.{spelling}.0" if spelling == "shards" else f"{base}.shard_0"
        if f"{probe}.weight" in store.refs:
            override = table.get(probe) or table.get(base)
            if isinstance(override, dict):
                return (int(override["group_size"]), int(override["bits"]),
                        str(override.get("mode", "affine")), spelling)
            packed = int(store.shape(f"{probe}.weight")[-1])
            groups = int(store.shape(f"{probe}.scales")[-1])
            if dims <= 0 or dims % groups or packed * 32 % dims:
                raise ValueError(f"incompatible n-gram shapes at {probe}")
            return dims // groups, packed * 32 // dims, "affine", spelling
    raise ValueError(f"no n-gram shards found at {base}")


def _strip_prefix(subset: dict, prefix: str) -> list:
    return [(key[len(prefix) + 1:], value) for key, value in subset.items()]


def _assert_loaded(layer, subset: dict, prefix: str) -> None:
    """Fail closed when a quantized triple did not land on its module.

    ``load_weights(strict=False)`` silently skips shape mismatches, which
    would leave random weights behind and fake a divergence. Every triple
    present in the checkpoint must match its module shapes exactly.
    Routed experts and PLE rows are served sparsely; verify-rows covers
    their raw bytes instead.
    """
    problems = []
    seen: set[str] = set()
    for path, module in layer.named_modules():
        full = f"{prefix}.{path}" if path else prefix
        if ".switch_mlp." in full or ".ngram_embedding." in full:
            continue
        triple = [f"{full}.{part}" for part in ("weight", "scales", "biases")]
        if triple[0] not in subset:
            continue
        seen.add(triple[0])
        if triple[1] not in subset or triple[2] not in subset:
            continue  # unquantized tensor (norms, biases, router)
        for key, attr in zip(triple, ("weight", "scales", "biases")):
            seen.add(key)
            module_array = getattr(module, attr, None)
            if module_array is None:
                problems.append(f"{key}: module has no {attr} (not quantized?)")
            elif tuple(subset[key].shape) != tuple(module_array.shape):
                problems.append(
                    f"{key}: checkpoint {subset[key].shape} vs module {module_array.shape}"
                )
    missing = [k for k in subset if k not in seen and k.endswith(".weight")]
    if missing:
        problems.append(f"weights with no module: {missing[:5]}")
    if problems:
        raise ValueError("stock weight binding failed:\n" + "\n".join(problems))


def _install_sparse_switch(layer, store, layer_prefix: str, table: dict):
    """Replace the routed expert module before quantization or evaluation.

    Saves only metadata (activation, dims, count, geometry) from the
    original module, then deletes it so no full expert bank can
    materialize. Returns the installed sparse module.
    """
    original = layer.mlp.switch_mlp
    activation = original.activation
    group_size, bits, mode, num_experts, input_dims, hidden_dims = \
        _sparse_quant_geometry(store, f"{layer_prefix}.mlp.switch_mlp", table)
    sparse = SparseStockSwitchGLU(
        store, f"{layer_prefix}.mlp.switch_mlp", group_size, bits, mode,
        num_experts, input_dims, hidden_dims, activation,
    )
    layer.mlp.switch_mlp = sparse
    del original
    gc.collect()
    return sparse


def _detach_ngram_tables(layer):
    """Replace PLE tables immediately; return metadata for sparse serving."""
    ngram_mod = layer.ple.ple_embedding
    real_tables = ngram_mod.ngram_embedding
    meta = (real_tables.dims, [int(v) for v in real_tables.shard_offsets])
    recorder = _RecordingNgramTables(real_tables.dims)
    ngram_mod.ngram_embedding = recorder
    del real_tables
    gc.collect()
    return recorder, meta


def cmd_dump_stock_layer(args) -> Path:
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    _log_rss("stock process start")
    model_dir, config, _model_class, model_config = _stock_config(args.model)
    manifest = json.loads((Path(args.title_from) / "manifest.json").read_text())
    layer_id = int(args.layer)
    text_config = model_config.text_config
    is_linear = bool(text_config.layer_types[layer_id] == "linear_attention")
    has_ple = bool(text_config.ple_layer_ids and (layer_id + 1) in list(text_config.ple_layer_ids))

    layer = Qwen4ExpDecoderLayer(text_config, layer_idx=layer_id)
    _log_rss("stock after layer construction")
    layer_prefix = f"language_model.model.layers.{layer_id}"
    from models.flashnext.store import SafeTensorStore

    store = SafeTensorStore(str(model_dir))
    table, group_size, bits, mode = _module_overrides(config)
    subset = _load_subset(store, (layer_prefix,))
    _install_sparse_switch(layer, store, layer_prefix, table)
    recorder, ngram_meta = (None, None)
    if has_ple:
        recorder, ngram_meta = _detach_ngram_tables(layer)
    _log_rss("stock after giant modules replaced")

    layer = _quantize_stock(layer, layer_prefix, subset, table,
                            group_size, bits, mode)
    layer.load_weights(
        [(key, value) for key, value in _strip_prefix(subset, layer_prefix)
         if ".switch_mlp." not in key and ".ngram_embedding." not in key],
        strict=False,
    )
    _assert_loaded(layer, subset, layer_prefix)
    del subset
    gc.collect()
    mx.eval(layer.parameters())
    _log_rss("stock after quantization/binding")

    hidden = np.load(Path(args.title_from) / manifest["captures"][f"layer_{layer_id:02d}_in"]["file"])
    ids = manifest["ids"]
    # Activations are bf16 on both arms. The dumps store exact float32
    # copies; cast back so the stock kernels take the identical path.
    hidden_mx = mx.array(hidden).astype(mx.bfloat16)
    ids_mx = mx.array([ids], dtype=mx.uint32)
    caches = _fresh_caches([(is_linear, has_ple)])
    # Replay the exact mask the full-model run used. Mask builders read
    # cache state, so a standalone fresh cache can produce a different
    # mask than the one the layer actually ran with.
    mask_rec = manifest["captures"][f"layer_{layer_id:02d}_in"].get("mask", {"kind": "rebuild"})
    if mask_rec.get("kind") == "array":
        mask = mx.array(np.load(Path(args.title_from) / f"layer_{layer_id:02d}_mask.npy"))
    elif mask_rec.get("kind") == "flag":
        mask = mask_rec.get("value")
    elif mask_rec.get("kind") == "none":
        mask = None
    elif is_linear:
        from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_ssm_mask
        mask = _create_qwen3_5_ssm_mask(hidden_mx, caches[0])
    else:
        from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_attention_mask
        mask = _create_qwen3_5_attention_mask(hidden_mx, caches[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if has_ple:
        from bisect import bisect_right

        dims, offsets = ngram_meta
        phase1 = layer(hidden_mx, ids_mx, mask=mask, cache=caches[0], position_ids=None)
        mx.eval(phase1)
        del phase1
        gc.collect()
        _log_rss("stock ngram discovery done")
        wanted: list[int] = list(dict.fromkeys(recorder.seen))
        base = f"{layer_prefix}.ple.ple_embedding.ngram_embedding"
        group_size_ng, bits_ng, mode_ng, spelling = _ngram_dequant_params(
            store, base, table, dims)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for gid in wanted:
            shard = bisect_right(offsets, gid) - 1
            if shard < 0 or gid >= offsets[-1]:
                raise ValueError(f"ngram id out of range: {gid}")
            by_shard.setdefault(shard, []).append((gid, gid - offsets[shard]))
        vectors: dict[int, object] = {}
        for shard, pairs in sorted(by_shard.items()):
            triple = (f"{base}.shards.{shard}" if spelling == "shards"
                      else f"{base}.shard_{shard}")
            locals_ = [local for _, local in pairs]
            deq = mx.dequantize(
                store.rows(f"{triple}.weight", locals_),
                store.rows(f"{triple}.scales", locals_),
                store.rows(f"{triple}.biases", locals_),
                group_size=group_size_ng, bits=bits_ng, mode=mode_ng,
            )
            mx.eval(deq)
            for (gid, _), slot in zip(pairs, range(len(pairs))):
                vectors[gid] = deq[slot]
            del deq
            gc.collect()
        server = _SparseNgramTables(dims, vectors)
        layer.ple.ple_embedding.ngram_embedding = server
        np.save(out / f"stock_layer_{layer_id:02d}_ngram_ids.npy", np.array(wanted, dtype=np.int64))
        # Phase 2 reuses the same mask and fresh caches: the PLE cache
        # mutated during phase 1, but the mask is replayed verbatim.
        caches = _fresh_caches([(is_linear, has_ple)])

    route_data: dict = {}
    orig_moe_call = Qwen3_5MoeSparseMoeBlock.__call__

    def moe_call(moe_self, x):
        out_moe = orig_moe_call(moe_self, x)
        mx.eval(x, out_moe)
        gate_logits = moe_self.gate(x)
        mx.eval(gate_logits)
        gates = mx.softmax(gate_logits, axis=-1, precise=True)
        k = moe_self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        mx.eval(gates, inds, scores)
        route_data["gate"] = _save_array(out / f"stock_layer_{layer_id:02d}_gate.npy", gate_logits)
        route_data["inds"] = _save_array(out / f"stock_layer_{layer_id:02d}_inds.npy", inds)
        route_data["scores"] = _save_array(out / f"stock_layer_{layer_id:02d}_scores.npy", scores)
        return out_moe

    Qwen3_5MoeSparseMoeBlock.__call__ = moe_call
    _log_rss("stock before forward")
    try:
        out_value = layer(hidden_mx, ids_mx, mask=mask, cache=caches[0], position_ids=None)
        mx.eval(out_value)
    finally:
        Qwen3_5MoeSparseMoeBlock.__call__ = orig_moe_call
    _log_rss("stock after forward")
    if has_ple:
        if set(server.queried) != set(wanted):
            raise ValueError("phase-2 ngram ids differ from recorded set")
    rec = _save_array(out / f"stock_layer_{layer_id:02d}_out.npy", out_value)
    (out / "manifest.json").write_text(json.dumps(rec, indent=2) + "\n")
    try:
        store.close()
    except (AttributeError, OSError):
        pass
    del layer
    gc.collect()
    _log_rss("stock before exit")
    print(json.dumps({"dumped": str(out), "layer": layer_id}))
    return out


def _stock_end_common(args, kind: str):
    import mlx.core as mx
    import mlx.nn as nn

    _log_rss(f"stock {kind} process start")
    model_dir, config, _model_class, model_config = _stock_config(args.model)
    manifest = json.loads((Path(args.title_from) / "manifest.json").read_text())
    table, group_size, bits, mode = _module_overrides(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return model_dir, config, model_config, manifest, table, group_size, bits, mode, out


def _quantize_and_load_end(model_dir, module, prefix: str, table: dict,
                           group_size: int, bits: int, mode: str):
    import mlx.core as mx
    from models.flashnext.store import SafeTensorStore

    store = SafeTensorStore(str(model_dir))
    subset = _load_subset(store, (prefix,))
    module = _quantize_stock(module, prefix, subset, table,
                             group_size, bits, mode)
    module.load_weights(_strip_prefix(subset, prefix), strict=False)
    _assert_loaded(module, subset, prefix)
    del subset
    store.close()
    gc.collect()
    mx.eval(module.parameters())
    _log_rss("stock end module live")
    return module


def cmd_dump_stock_embedding(args) -> Path:
    import mlx.core as mx
    import mlx.nn as nn

    model_dir, _config, model_config, manifest, table, group_size, bits, mode, out = \
        _stock_end_common(args, "embedding")
    ids = manifest["ids"]
    embed = nn.Embedding(int(model_config.text_config.vocab_size),
                         int(model_config.text_config.hidden_size))
    embed = _quantize_and_load_end(str(model_dir), embed,
                                   "language_model.model.embed_tokens",
                                   table, group_size, bits, mode)
    raw = embed(mx.array([ids], dtype=mx.uint32))
    records = {"embed_raw": _save_array(out / "stock_embed_raw.npy", raw)}
    tiled = mx.tile(raw, (1, 1, int(model_config.text_config.hc_count)))
    records["embed_tiled"] = _save_array(out / "stock_embed_tiled.npy", tiled)
    del embed
    gc.collect()
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    _log_rss("stock embedding before exit")
    print(json.dumps({"dumped": str(out)}))
    return out


def cmd_dump_stock_final(args) -> Path:
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual

    model_dir, _config, model_config, manifest, table, group_size, bits, mode, out = \
        _stock_end_common(args, "final")
    mixer = Qwen4ExpGatedResidual(model_config.text_config, use_combine=False)
    mixer = _quantize_and_load_end(str(model_dir), mixer,
                                   "language_model.model.hyper_connection_mixer",
                                   table, group_size, bits, mode)
    hidden = np.load(Path(args.title_from) / manifest["captures"]["layer_47_out"]["file"])
    mixed = mixer(mx.array(hidden).astype(mx.bfloat16))
    records = {"final": _save_array(out / "stock_final.npy", mixed)}
    del mixer
    gc.collect()
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    _log_rss("stock final before exit")
    print(json.dumps({"dumped": str(out)}))
    return out


def cmd_dump_stock_head(args) -> Path:
    import mlx.core as mx
    import mlx.nn as nn

    model_dir, _config, model_config, manifest, table, group_size, bits, mode, out = \
        _stock_end_common(args, "head")
    head = nn.Linear(int(model_config.text_config.hidden_size),
                     int(model_config.text_config.vocab_size), bias=False)
    head = _quantize_and_load_end(str(model_dir), head,
                                  "language_model.lm_head",
                                  table, group_size, bits, mode)
    hidden = np.load(Path(args.title_from) / manifest["captures"]["layer_47_out"]["file"])
    logits = head(mx.array(hidden).astype(mx.bfloat16))
    records = {"logits": _save_array(out / "stock_logits.npy", logits)}
    del head
    gc.collect()
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    _log_rss("stock head before exit")
    print(json.dumps({"dumped": str(out)}))
    return out


# ---------------------------------------------------------------------------
# verify-rows (positioned pread only; never loads a shard)
# ---------------------------------------------------------------------------

_PREAD_DTYPES = {
    "U32": ("<u4", 4), "BF16": ("<u2", 2), "F16": ("<f2", 2),
    "F32": ("<f4", 4), "U8": ("u1", 1), "I8": ("i1", 1),
    "U16": ("<u2", 2), "I16": ("<i2", 2), "I32": ("<i4", 4),
    "I64": ("<i8", 8), "U64": ("<u8", 8), "BOOL": ("?", 1),
}


def _read_tensor_bytes(model_dir: Path, shard: str, start: int, nbytes: int) -> bytes:
    """Read an exact byte range with positioned reads (no shard load)."""
    path = str(model_dir / shard)
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        remaining = nbytes
        offset = start
        while remaining > 0:
            piece = os.pread(fd, min(remaining, 1 << 26), offset)
            if not piece:
                raise ValueError(f"short read in {shard} at {offset}")
            chunks.append(piece)
            offset += len(piece)
            remaining -= len(piece)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _shard_headers(model_dir: Path) -> dict:
    """Map tensor name to (shard, dtype, shape, data_start, data_offset)."""
    import struct

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict = index["weight_map"]
    headers: dict[str, dict] = {}
    cache: dict[str, dict] = {}
    for key in sorted(set(weight_map)):
        shard = weight_map[key]
        if shard not in cache:
            with open(model_dir / shard, "rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            cache[shard] = {"header": header, "data_start": data_start}
        entry = cache[shard]["header"].get(key)
        if entry is None:
            raise ValueError(f"{key} missing from {shard} header")
        headers[key] = {
            "shard": shard,
            "dtype": entry["dtype"],
            "shape": tuple(entry["shape"]),
            "start": cache[shard]["data_start"] + entry["data_offsets"][0],
            "end": cache[shard]["data_start"] + entry["data_offsets"][1],
        }
    return headers


def _stored_as_float32(view, dtype: str):
    """Decode a raw storage view to float32 without model arithmetic.

    BF16 travels as uint16 bits; shifting restores exact float32 values.
    Other dtypes compare as plain numpy arrays.
    """
    arr = np.asarray(view)
    if dtype == "BF16":
        return (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def cmd_verify_rows(args) -> bool:
    from models.flashnext.store import SafeTensorStore

    model_dir = Path(args.model).expanduser()
    store = SafeTensorStore(str(model_dir))
    headers = _shard_headers(model_dir)
    layer_id = int(args.layer)
    prefixes = [
        f"language_model.model.layers.{layer_id}.mlp.switch_mlp.gate_proj",
        f"language_model.model.layers.{layer_id}.mlp.switch_mlp.up_proj",
        f"language_model.model.layers.{layer_id}.mlp.switch_mlp.down_proj",
    ]
    base = f"language_model.model.layers.{layer_id}.ple.ple_embedding.ngram_embedding"
    token = None
    for candidate in (".shards.", ".shard_"):
        if any(key.startswith(base + candidate) for key in headers):
            token = candidate
            break
    if token is not None:
        shards = sorted({key.split(base + token)[1].split(".")[0]
                         for key in headers if key.startswith(base + token)})
        for shard in shards:
            prefixes.append(f"{base}{token}{shard}")
    ok = True
    for prefix in prefixes:
        for part in ("weight", "scales", "biases"):
            key = f"{prefix}.{part}"
            if key not in headers:
                continue
            meta = headers[key]
            try:
                code, itemsize = _PREAD_DTYPES[meta["dtype"]]
            except KeyError:
                print(f"{key} UNSUPPORTED DTYPE {meta['dtype']}")
                ok = False
                continue
            raw = _read_tensor_bytes(
                model_dir, meta["shard"], meta["start"], meta["end"] - meta["start"])
            expect = _stored_as_float32(
                np.frombuffer(raw, dtype=np.dtype(code)).reshape(meta["shape"]),
                meta["dtype"])
            streamed = _stored_as_float32(store._view(key), meta["dtype"])
            match = streamed.shape == expect.shape and np.array_equal(streamed, expect)
            print(f"{key} {'OK' if match else 'MISMATCH shape_streamed=' + str(streamed.shape) + ' shape_direct=' + str(expect.shape)}")
            ok = ok and match
            del raw, expect
            gc.collect()
    try:
        store.close()
    except (AttributeError, OSError):
        pass
    print("verify-rows " + ("PASS" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------------------
# norm-sweep (no model; both formulas against captured records)
# ---------------------------------------------------------------------------

def cmd_norm_sweep(args) -> None:
    base = Path(args.title_from)
    manifest = json.loads((base / "manifest.json").read_text())
    every = manifest["norms"] if args.layer == "all" else [
        n for n in manifest["norms"] if f".{args.layer}." in n["path"]
        or (args.layer == "final" and n["path"].startswith("mixer."))
    ]
    print(f"{'family':42s} {'mean':>9s} {'min':>9s} {'match':>22s}")
    for entry in every:
        x = np.load(base / entry["x"]["file"]).astype(np.float64)
        w = np.load(base / entry["weight"]["file"]).astype(np.float64)
        got = np.load(base / entry["out"]["file"]).astype(np.float64)
        eps = float(entry.get("eps", 1e-6))
        group = entry.get("group_size")
        if group is not None:
            # Mirror Qwen4ExpRMSNorm grouped path: reshape before the cast.
            x = x.reshape(*x.shape[:-1], -1, int(group))
            w = w.reshape(-1, int(group))
        var = (x * x).mean(axis=-1, keepdims=True)
        base_norm = x / np.sqrt(var + eps)
        cand_a = (base_norm * w).reshape(got.shape)
        cand_b = (base_norm * (1.0 + w)).reshape(got.shape)
        err_a = float(np.abs(cand_a - got).max())
        err_b = float(np.abs(cand_b - got).max())
        winner = "A:direct" if err_a < err_b else "B:one-plus"
        print(f"{entry['path']:42s} {w.mean():9.4f} {w.min():9.4f} {winner:>22s} "
              f"errA={err_a:.3e} errB={err_b:.3e}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _compare_pair(base: Path, macqwen_rec: dict, stock_file: str, label: str) -> bool:
    got = np.load(base / macqwen_rec["file"]).astype(np.float32)
    ref = np.load(stock_file).astype(np.float32)
    same_shape = got.shape == ref.shape
    equal = same_shape and bool(np.array_equal(got, ref))
    if same_shape:
        delta = np.abs(got - ref)
        max_abs = float(delta.max())
        mean_abs = float(delta.mean())
    else:
        max_abs, mean_abs = float("nan"), float("nan")
    status = "PASS" if equal else "FAIL"
    extra = "" if equal else f" max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} shape_macqwen={list(got.shape)} shape_stock={list(ref.shape)}"
    print(f"{label} {status}{extra}")
    return equal


def cmd_compare(args) -> bool:
    """Compare a sweep root: macqwen/ + stock_embedding/ + stock_layer_NN/."""
    root = Path(args.stock)
    base = root / "macqwen"
    embed = root / "stock_embedding"
    manifest = json.loads((base / "manifest.json").read_text())
    caps = manifest["captures"]
    ok = _compare_pair(base, caps["embed_tiled"], str(embed / "stock_embed_tiled.npy"), "embedding")
    if not ok:
        return False
    for layer_id in range(int(args.layers)):
        layer_dir = root / f"stock_layer_{layer_id:02d}"
        if not _compare_pair(base, caps[f"layer_{layer_id:02d}_out"],
                             str(layer_dir / f"stock_layer_{layer_id:02d}_out.npy"),
                             f"layer {layer_id}"):
            return False
    final = root / "stock_final"
    head = root / "stock_head"
    ok = _compare_pair(base, caps["final"], str(final / "stock_final.npy"), "final norm") and ok
    ok = _compare_pair(base, caps["logits"], str(head / "stock_logits.npy"), "logits") and ok
    return ok


# ---------------------------------------------------------------------------
# sweep driver (sequential fresh children under the memory watchdog)
# ---------------------------------------------------------------------------

class _ChildResult:
    def __init__(self, returncode: int, stdout: str, stderr: str,
                 killed_for_memory: bool = False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.killed_for_memory = killed_for_memory


def _child_rss_gb(pid: int) -> float | None:
    """Current child RSS via ps (KiB). None when the PID is gone."""
    try:
        proc = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip()) / 1048576.0
    except ValueError:
        return None


def _enforce_child_limit(child: "subprocess.Popen", mode: str,
                         limit_gb: float, timeout: float) -> _ChildResult:
    """Watch current RSS; SIGTERM then SIGKILL past the limit. Timeout kept."""
    started = time.time()
    killed = False
    while True:
        try:
            _, _ = child.communicate(timeout=_WATCHDOG_POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
        rss = _child_rss_gb(child.pid)
        if rss is not None and rss > limit_gb:
            print(f"watchdog: mode={mode} pid={child.pid} rss={rss:.2f} GiB "
                  f"over {limit_gb:.1f} GiB; SIGTERM", flush=True)
            killed = True
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=_WATCHDOG_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"watchdog: pid={child.pid} ignored SIGTERM; SIGKILL",
                      flush=True)
                child.kill()
            break
        if time.time() - started > timeout:
            child.kill()
            out, err = child.communicate()
            return _ChildResult(124, out or "", err or "", False)
    out, err = child.communicate()
    return _ChildResult(child.returncode if not killed else 128 + signal.SIGTERM,
                        out or "", err or "", killed)


def _run(mode: str, extra: list[str],
         memory_limit_gb: float = MAX_CHILD_RSS_GB,
         timeout: float = _CHILD_TIMEOUT_SECONDS) -> _ChildResult:
    cmd = [sys.executable, str(Path(__file__).resolve()), mode, *extra]
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(CHILD_ENV)
    env.pop("FLASHNEXT_NORM_CONVENTION", None)
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env)
    return _enforce_child_limit(child, mode, memory_limit_gb, timeout)


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if text else ""


def _compare_arrays(label: str, macqwen_file: Path, stock_file: Path) -> bool:
    got = np.load(macqwen_file).astype(np.float32)
    ref = np.load(stock_file).astype(np.float32)
    same_shape = got.shape == ref.shape
    equal = same_shape and bool(np.array_equal(got, ref))
    if same_shape:
        delta = np.abs(got - ref)
        extra = "" if equal else (
            f" max_abs={float(delta.max()):.3e} mean_abs={float(delta.mean()):.3e}"
            f" shape_macqwen={list(got.shape)} shape_stock={list(ref.shape)}"
        )
    else:
        extra = f" shape_macqwen={list(got.shape)} shape_stock={list(ref.shape)}"
    print(f"{label} {'PASS' if equal else 'FAIL'}{extra}")
    return equal


def _run_stage(root: Path, mode: str, extra: list[str], label: str) -> bool:
    print(f"--- {label}")
    proc = _run(mode, extra)
    print(_tail(proc.stdout or "", 1000))
    if proc.killed_for_memory:
        print(f"FAILED: {label} exceeded {MAX_CHILD_RSS_GB:.0f} GiB and was killed")
        return False
    if proc.returncode != 0:
        print(_tail(proc.stderr or "", 3000))
        return False
    return True


def cmd_sweep(args) -> int:
    """Gate order: macqwen, embedding, layers in order, final, head."""
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    macqwen_dir = root / "macqwen"
    if (macqwen_dir / "manifest.json").is_file():
        print("--- dump-macqwen present, reusing")
    elif not _run_stage(root, "dump-macqwen",
                        ["--model", args.model, "--out", str(macqwen_dir),
                         "--prompt", args.prompt, "--tokens", str(args.tokens)],
                        f"dump-macqwen ({args.model})"):
        return 2
    manifest = json.loads((macqwen_dir / "manifest.json").read_text())
    caps = manifest["captures"]
    layers = int(args.layers)
    start = int(args.start)
    embed_dir = root / "stock_embedding"
    if (embed_dir / "manifest.json").is_file():
        print("--- dump-stock-embedding present, reusing")
    elif not _run_stage(root, "dump-stock-embedding",
                        ["--model", args.model, "--from", str(macqwen_dir),
                         "--out", str(embed_dir)],
                        "dump-stock-embedding"):
        return 2
    if start == 0:
        if not _compare_arrays("embedding", macqwen_dir / caps["embed_tiled"]["file"],
                               embed_dir / "stock_embed_tiled.npy"):
            return 1
    for layer_id in range(start, layers):
        layer_dir = root / f"stock_layer_{layer_id:02d}"
        if not (layer_dir / f"stock_layer_{layer_id:02d}_out.npy").is_file():
            if not _run_stage(root, "dump-stock-layer",
                              ["--model", args.model, "--layer", str(layer_id),
                               "--from", str(macqwen_dir), "--out", str(layer_dir)],
                              f"dump-stock-layer {layer_id}"):
                return 2
        if not _compare_arrays(f"layer {layer_id}",
                               macqwen_dir / caps[f"layer_{layer_id:02d}_out"]["file"],
                               layer_dir / f"stock_layer_{layer_id:02d}_out.npy"):
            return 1
        ngram_ids_file = layer_dir / f"stock_layer_{layer_id:02d}_ngram_ids.npy"
        if ngram_ids_file.is_file():
            flat_logged = [i for call in manifest.get("ple_ids", {}).values()
                           for lst in call for i in lst]
            stock_ids = [int(v) for v in np.load(ngram_ids_file).tolist()]
            match = sorted(flat_logged) == sorted(stock_ids)
            print(f"layer {layer_id} ngram ids {'PASS' if match else 'FAIL'}"
                  f" ({len(stock_ids)} ids)")
            if not match:
                return 1
    final_dir = root / "stock_final"
    if (final_dir / "manifest.json").is_file():
        print("--- dump-stock-final present, reusing")
    elif not _run_stage(root, "dump-stock-final",
                        ["--model", args.model, "--from", str(macqwen_dir),
                         "--out", str(final_dir)],
                        "dump-stock-final"):
        return 2
    if not _compare_arrays("final norm", macqwen_dir / caps["final"]["file"],
                           final_dir / "stock_final.npy"):
        return 1
    head_dir = root / "stock_head"
    if (head_dir / "manifest.json").is_file():
        print("--- dump-stock-head present, reusing")
    elif not _run_stage(root, "dump-stock-head",
                        ["--model", args.model, "--from", str(macqwen_dir),
                         "--out", str(head_dir)],
                        "dump-stock-head"):
        return 2
    ok = _compare_arrays("logits", macqwen_dir / caps["logits"]["file"],
                         head_dir / "stock_logits.npy")
    print("sweep " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


MODES = {
    "dump-macqwen": cmd_dump_macqwen,
    "dump-stock-layer": cmd_dump_stock_layer,
    "dump-stock-embedding": cmd_dump_stock_embedding,
    "dump-stock-final": cmd_dump_stock_final,
    "dump-stock-head": cmd_dump_stock_head,
    "verify-rows": cmd_verify_rows,
    "norm-sweep": cmd_norm_sweep,
    "compare": cmd_compare,
    "sweep": cmd_sweep,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--out", required=True)

    take = argparse.ArgumentParser(add_help=False)
    take.add_argument("--from", dest="title_from", required=True)

    dump = sub.add_parser("dump-macqwen", parents=[common])
    dump.add_argument("--prompt", default=DEFAULT_PROMPT)
    dump.add_argument("--tokens", type=int, default=DEFAULT_IDS)
    dump.add_argument("--norm-convention", default="auto",
                      choices=("auto", "one", "zero"))

    layer = sub.add_parser("dump-stock-layer", parents=[common, take])
    layer.add_argument("--layer", required=True)

    embed = sub.add_parser("dump-stock-embedding", parents=[common, take])

    final = sub.add_parser("dump-stock-final", parents=[common, take])

    head = sub.add_parser("dump-stock-head", parents=[common, take])

    rows = sub.add_parser("verify-rows", parents=[common])
    rows.add_argument("--layer", required=True)

    sweep_norm = sub.add_parser("norm-sweep", parents=[take])
    sweep_norm.add_argument("--layer", default="all")

    comp = sub.add_parser("compare")
    comp.add_argument("--stock", required=True, help="sweep root with macqwen/, stock_embedding/, stock_layer_NN/, stock_final/, stock_head/")
    comp.add_argument("--layers", default="48")

    sweep = sub.add_parser("sweep", parents=[common])
    sweep.add_argument("--prompt", default=DEFAULT_PROMPT)
    sweep.add_argument("--tokens", type=int, default=DEFAULT_IDS)
    sweep.add_argument("--layers", default="48")
    sweep.add_argument("--start", default="0")

    args = parser.parse_args()
    if np is None:
        raise SystemExit("numpy is required")
    handler = MODES.get(args.mode)
    if handler is None:
        raise SystemExit(f"unknown mode {args.mode}")
    result = handler(args)
    if isinstance(result, bool):
        return 0 if result else 1
    if result is None or isinstance(result, Path):
        return 0
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
