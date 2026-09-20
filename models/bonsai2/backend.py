"""Resident MLX runtime for Bonsai-2 ternary 27B (text-only)."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from macqwen.backends.base import DecodeTimer, GenerationCancelled
from macqwen.conversation import Conversation, EXTRA_REASONING
from macqwen.sampling import Sampler, Sampling
from macqwen.text import stream_decode

from .checkpoint import resolve_bonsai2, runtime_available
from .protocol import ProtocolTranslator
from .settings import SESSION_DIR


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_TAGS = {
    "medium": "think",
    "low": "think",
    "xhigh": "think",
}
THINK_FIELDS = {
    "medium": "think",
    "low": "think",
    "xhigh": "think",
}
_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SESSION_SCHEMA = 1
_Q4_ATTENTION_DEFAULT_TEMP_MB = 256.0


_IDENTITY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def _config_identity(model_path: str) -> str | None:
    """Identify the checkpoint without loading weights.

    Sessions store token IDs, not text, so the fingerprint must cover
    everything that maps between them: model config, tokenizer data and
    options, the chat template, and the bundled runtime loader. A changed
    tokenizer or template silently redefines old tapes, so sessions
    failing this check are rejected instead of replayed.
    """
    import hashlib

    def feed(handle) -> None:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    try:
        digest = hashlib.sha256()
        root = Path(model_path).expanduser()
        for name in _IDENTITY_FILES:
            with open(root / name, "rb") as handle:
                feed(handle)
        runtime = root / "runtime"
        if runtime.is_dir():
            for path in sorted(runtime.rglob("*.py")):
                digest.update(path.name.encode("utf-8"))
                with open(path, "rb") as handle:
                    feed(handle)
        return digest.hexdigest()
    except OSError:
        return None


@contextmanager
def _transformers_import_environment():
    """Hide the irrelevant PyTorch advisory during the tokenizer import."""
    key = "TRANSFORMERS_NO_ADVISORY_WARNINGS"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def _cache_limit(mx, megabytes: float | None):
    if megabytes is None:
        yield
        return
    previous = mx.set_cache_limit(int(megabytes * 1024 * 1024))
    try:
        yield
    finally:
        mx.set_cache_limit(previous)


@dataclass
class Stats:
    finish: str = "stop"
    tokens: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    host_free_gb: float | None = None
    swap_gb: float | None = None

    @property
    def rate(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def prompt_rate(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0


def _quantized_kv_from_environment():
    """Read the chat-side KV toggle without touching chat code.

    `MACQWEN_BONSAI2_KV=8` selects 8-bit groups of 64, `=4` selects 4-bit.
    Unset or empty means full-precision caches. Anything else raises
    instead of silently picking a precision.
    """
    raw = (os.environ.get("MACQWEN_BONSAI2_KV") or "").strip()
    if not raw:
        return None
    if raw not in ("4", "8"):
        raise ValueError("MACQWEN_BONSAI2_KV must be 4 or 8")
    return (int(raw), 64)


def _mlx_dtype_itemsize(dtype) -> int:
    """Return the storage width used by the attention estimate."""
    try:
        return int(dtype.itemsize)
    except (AttributeError, TypeError, ValueError):
        return {
            "mlx.core.float16": 2,
            "mlx.core.bfloat16": 2,
            "mlx.core.float32": 4,
        }.get(str(dtype), 4)


def _attention_memory_snapshot() -> dict[str, int | None]:
    import mlx.core as mx

    result = {}
    for key, name in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        function = getattr(mx, name, None)
        result[key] = int(function()) if function is not None else None
    return result


def _q4_attention_tile_plan(queries, keys, mask, budget_bytes: int):
    """Estimate a score/probability tile without changing model batch size."""
    batch, q_heads, query_rows, head_dim = map(int, queries.shape)
    key_rows = int(keys[0].shape[-2])
    dtype_bytes = _mlx_dtype_itemsize(queries.dtype)
    mask_bytes = 1 if mask is None or isinstance(mask, str) else _mlx_dtype_itemsize(mask.dtype)
    per_query_bytes = (
        2 * batch * q_heads * key_rows * dtype_bytes
        + batch * key_rows * mask_bytes
        + 2 * batch * q_heads * head_dim * dtype_bytes
    )
    # Quantized matmul workspace is backend-dependent; leave a conservative
    # margin instead of pretending the score estimate is a hard allocator cap.
    per_query_bytes = max(1, math.ceil(per_query_bytes * 1.25))
    tile_rows = max(1, min(query_rows, budget_bytes // per_query_bytes))
    return tile_rows, {
        "batch": batch,
        "query_heads": q_heads,
        "query_rows": query_rows,
        "key_rows": key_rows,
        "head_dim": head_dim,
        "dtype": str(queries.dtype),
        "dtype_bytes": dtype_bytes,
        "estimated_bytes_per_query": per_query_bytes,
        "estimated_tile_bytes": per_query_bytes * tile_rows,
        "tile_rows": tile_rows,
    }


def _q4_attention_tile_mask(mask, query_offset: int, start: int, end: int, key_rows: int):
    import mlx.core as mx

    if mask is None:
        return None
    if isinstance(mask, str):
        if mask != "causal":
            return mask
        query_positions = mx.arange(query_offset + start, query_offset + end)
        key_positions = mx.arange(key_rows)
        return query_positions[:, None] >= key_positions[None, :]
    if mask.ndim < 2:
        return mask
    query_axis = int(mask.shape[-2])
    if query_axis == 1:
        return mask[..., :1, :key_rows]
    if query_axis < end:
        raise ValueError("attention mask is shorter than its query tile")
    return mask[..., start:end, :key_rows]


def _bonsai_quantized_attention(
    stock_attention,
    queries,
    keys,
    values,
    cache,
    scale,
    mask,
    budget_bytes: int,
):
    import mlx.core as mx

    if not hasattr(cache, "bits") or queries.shape[-2] <= 1:
        return stock_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    if isinstance(mask, str) and mask != "causal":
        return stock_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    offset = getattr(cache, "offset", None)
    if hasattr(offset, "ndim") and int(offset.ndim) != 0:
        return stock_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    try:
        offset = int(offset.item()) if hasattr(offset, "item") else int(offset)
        key_rows = int(keys[0].shape[-2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return stock_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    query_offset = offset - int(queries.shape[-2])
    if query_offset < 0 or key_rows != offset:
        return stock_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    tile_rows, plan = _q4_attention_tile_plan(
        queries, keys, mask, budget_bytes
    )
    if tile_rows >= queries.shape[-2]:
        return stock_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    trace = getattr(cache, "_bonsai_attention_trace", None)
    layer = getattr(cache, "_bonsai_attention_layer", None)
    outputs = []
    for start in range(0, int(queries.shape[-2]), tile_rows):
        end = min(start + tile_rows, int(queries.shape[-2]))
        tile_plan = dict(plan, tile_start=start, tile_end=end)
        tile_mask = _q4_attention_tile_mask(
            mask, query_offset, start, end, key_rows
        )
        started = time.perf_counter() if trace is not None else None
        before = _attention_memory_snapshot() if trace is not None else None
        output = stock_attention(
            queries[..., start:end, :],
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=tile_mask,
        )
        # The point of tiling is lost if all outputs remain one lazy graph.
        mx.eval(output)
        after = _attention_memory_snapshot() if trace is not None else None
        if trace is not None:
            trace.append({
                "layer": layer,
                "bits": int(cache.bits),
                "group_size": int(cache.group_size),
                **tile_plan,
                "duration_s": time.perf_counter() - started,
                "memory_before": before,
                "memory_after": after,
            })
        outputs.append(output)
    return mx.concatenate(outputs, axis=-2)


def _install_q4_attention_tiling(enabled: bool, budget_bytes: int) -> None:
    """Patch only Bonsai's imported Qwen3.5 attention symbol."""
    import mlx_vlm.models.qwen3_5.language as language

    stock = getattr(language, "_bonsai_stock_attention", None)
    if stock is None:
        stock = language.scaled_dot_product_attention
        language._bonsai_stock_attention = stock

    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):
        if not enabled or sinks is not None:
            return stock(
                queries, keys, values, cache=cache, scale=scale,
                mask=mask, sinks=sinks
            )
        return _bonsai_quantized_attention(
            stock, queries, keys, values, cache, scale, mask, budget_bytes
        )

    language.scaled_dot_product_attention = dispatch


def _quantize_kv_caches(cache, bits: int, group_size: int):
    """Replace full-attention caches with quantized versions in place.

    GDN linear layers keep their fp32 recurrent state; only the 16 KVCache
    layers quantize. Runs at construction and on every replay so restored
    caches never silently return to fp32.
    """
    for index, item in enumerate(cache):
        if type(item).__name__ == "KVCache":
            cache[index] = item.to_quantized(
                group_size=group_size, bits=bits
            )
    return cache


def _verify_shared_signs(model) -> None:
    """Prove same-width sign vectors are byte-identical in this checkpoint.

    The share hook keys the memo by input width, which is exact only when
    every module of that width carries the same signs. Refuse to run shared
    otherwise instead of risking cross-wired transforms.
    """
    import hashlib

    import mlx.core as mx
    import numpy as np

    seen: dict[int, str] = {}
    for _name, module in model.named_modules():
        signs = getattr(module, "signs", None)
        if signs is None:
            continue
        width = int(signs.shape[0])
        mx.eval(signs)
        digest = hashlib.sha256(bytes(np.asarray(signs).tobytes())).hexdigest()
        if width in seen and seen[width] != digest:
            raise RuntimeError(
                f"Bonsai-2 sign vectors differ at width {width}; "
                "refusing shared transforms"
            )
        seen[width] = digest


def _load_weight_tensors(directory: Path) -> dict:
    """Load every weight shard into one tensor map.

    Single-file packs read `model.safetensors` directly. Sharded packs
    follow the index weight map in file order and reject duplicate tensor
    names across shards instead of silently keeping the last copy.
    """
    import mlx.core as mx

    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        return dict(mx.load(str(directory / "model.safetensors")))
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Shard index has no weight map")
    shard_names = list(dict.fromkeys(weight_map.values()))
    merged: dict = {}
    for shard in shard_names:
        for key, value in mx.load(str(directory / shard)).items():
            if key in merged:
                raise ValueError(f"Duplicate tensor across shards: {key}")
            merged[key] = value
    return merged


def _validate_packed_record(original, record, arrays, signs, block) -> None:
    """Validate packed tensors against the original module geometry.

    Runs before the original module is replaced, so a corrupted pack fails
    here instead of executing against unchecked shapes.
    """
    import mlx.core as mx
    from mlx.nn import Embedding, Linear

    weight, scales, biases = arrays
    if not isinstance(original, (Linear, Embedding)):
        raise ValueError("Packed module target is not a linear layer")
    if record["embedding"] != isinstance(original, Embedding):
        raise ValueError("Packed module kind mismatch")
    rows, width = original.weight.shape
    if width % 128:
        raise ValueError("Invalid packed width")
    if tuple(weight.shape) != (rows, width // 16) or str(weight.dtype) != "mlx.core.uint32":
        raise ValueError("Invalid packed weight shape or storage dtype")
    for array in (scales, biases):
        if tuple(array.shape) != (rows, width // 128):
            raise ValueError("Invalid packed metadata shape")
        if str(array.dtype) not in (
            "mlx.core.float16", "mlx.core.float32", "mlx.core.bfloat16",
        ):
            raise ValueError("Invalid affine dtype")
        if not mx.all(mx.isfinite(array)).item():
            raise ValueError("Non-finite affine parameters")
    if block:
        if width % block:
            raise ValueError("Transform block does not divide input width")
        if signs is None or tuple(signs.shape) != (width,):
            raise ValueError("Invalid sign vector")
        if not mx.all((signs == 1) | (signs == -1)).item():
            raise ValueError("Invalid sign values")
    elif signs is not None:
        raise ValueError("Unexpected sign vector")


def _load_text_model(path):
    """Load the language model without materializing the vision tower.

    Mirrors the bundled ``vision_artifact.load_vl_model`` construction and
    Packed validation, but filters vision tensors out before loading and
    drops the tower module afterward. Text-only milestone: vision input
    stays unsupported. Raises identically on schema, duplicate, shape, and
    sign violations.
    """
    import json

    import mlx.core as mx

    directory = Path(path)
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "prism_hadamard_qwen35":
        raise ValueError("Unsupported packed model schema")
    if config.get("base_model_type") != "qwen3_5":
        raise ValueError("Unsupported base model type")

    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    model = Model(ModelConfig.from_dict(config))
    weights = _load_weight_tensors(directory)

    from runtime import Packed

    lm = model.language_model
    seen = set()
    for record in config["modules"]:
        path_ = record["path"]
        if path_ in seen:
            raise ValueError("Duplicate packed module")
        seen.add(path_)
        parts = path_.split(".")
        parent = lm
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = "language_model." + path_
        arrays = [weights[key + "." + s] for s in ("weight", "scales", "biases")]
        if record["dtype"] != "float16":
            raise ValueError("Unsupported activation dtype")
        block = record["block"]
        if block and block not in (512, 1024, 2048, 4096):
            raise ValueError("Unsupported block size")
        signs = weights.get(key + ".signs")
        original = getattr(parent, parts[-1])
        _validate_packed_record(original, record, arrays, signs, block)
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"], mx.float16))

    text_weights = {
        key: value for key, value in weights.items()
        if not key.startswith("vision_tower")
    }
    # Drop the tower before loading so the strict call below validates every
    # remaining parameter. A lenient load could silently retain initialized
    # values for missing auxiliary weights.
    model.vision_tower = None
    model.load_weights(list(text_weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    mx.clear_cache()
    return model, config


class _ForcingSampler:
    """Emit the reasoning-close token at the think budget boundary.

    generate_step() consumes each yielded token into cache before yielding
    the next, so substituting a token after the fact corrupts GDN state:
    the tape would record </think> while the cache holds an unrelated
    token. Forcing inside the sampler makes the close token genuinely
    consumed, so tape and cache always agree. Natural closes are detected
    from the sampled token itself before lookahead can make another decision;
    the backend also observes yielded ids as a fallback. One-shot per arming.
    """

    def __init__(self, sampler, think_budget, close_token):
        self._sampler = sampler
        self._remaining = think_budget
        self._close_token = close_token
        self.armed = think_budget is not None and think_budget > 0
        self.forced_last = False

    def __call__(self, logits):
        import mlx.core as mx

        self.forced_last = False
        if self.armed:
            if self._remaining <= 1:
                self.armed = False
                self.forced_last = True
                return mx.array([self._close_token], dtype=mx.uint32)
            self._remaining -= 1
            token = self._sampler(logits)
            # The generator can sample one token ahead of yielding the
            # previous one. Observe a natural close at sampling time, before
            # that lookahead can reach the budget boundary and force a second
            # close.
            if int(token) == self._close_token:
                self.armed = False
            return token
        return self._sampler(logits)

    def observe(self, value):
        self._sampler.observe(value)
        if int(value) == self._close_token:
            self.armed = False

    def end_thinking(self):
        self.armed = False


class _TextModelWrapper:
    """Expose the VL language model as a plain logits module.

    The bundled loader returns an mlx-vlm model whose language model answers
    ``LanguageModelOutput``. The shared ``generate_step`` helper expects plain
    logit arrays, so unwrap ``.logits`` at this boundary.
    """

    def __init__(self, language_model, share: bool = False):
        self._language_model = language_model
        self._share = bool(share)

    def __call__(self, inputs, cache=None, **options):
        from .ternary_kernel import arm_memo, disarm_memo

        if not self._share:
            return self._language_model(inputs, cache=cache, **options).logits
        arm_memo()
        try:
            return self._language_model(inputs, cache=cache, **options).logits
        finally:
            disarm_memo()

    def __getattr__(self, name):
        return getattr(self.__dict__["_language_model"], name)


class _NonConsumingStopModel:
    """Keep sampled stop IDs out of recurrent state.

    ``mlx_lm.generate_step`` runs one model call ahead of each yielded token.
    The call for a stop ID is unnecessary because the backend immediately
    ends the turn. Skipping only those post-prefill one-token calls preserves
    every accepted token in every cache layer while leaving prompt content
    and structural stop IDs in the prefill path untouched.
    """

    def __init__(self, model, prompt_length: int, prefill_step_size: int, stops):
        self._model = model
        prefill_tokens = max(0, int(prompt_length) - 1)
        self._initial_calls = (
            (prefill_tokens + prefill_step_size - 1) // prefill_step_size + 1
        )
        self._calls = 0
        self._stops = {int(value) for value in stops}
        self._last_logits_shape = None
        self._last_logits_dtype = None

    def __call__(self, inputs, cache=None, **options):
        import mlx.core as mx

        self._calls += 1
        if self._calls > self._initial_calls and inputs.shape[-1] == 1:
            value = int(inputs.reshape(-1)[0].item())
            if value in self._stops:
                if self._last_logits_shape is None:
                    raise RuntimeError("stop-aware generation has no logits shape")
                return mx.zeros(
                    self._last_logits_shape, dtype=self._last_logits_dtype
                )
        logits = self._model(inputs, cache=cache, **options)
        self._last_logits_shape = tuple(int(value) for value in logits.shape)
        self._last_logits_dtype = logits.dtype
        return logits


class BonsaiTokenizer:
    """Present the Bonsai-2 template through the interface the shared chat already uses."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.thinking_tag = THINK_TAGS["medium"]

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, text, **options):
        # Explicit: implicit dunder lookup bypasses __getattr__, so the
        # safe content encoder's tokenizer(chunk) call needs this forwarder.
        return self._tokenizer(text, **options)

    def __len__(self):
        # Same dunder limitation for session token-bound validation.
        return len(self._tokenizer)

    def _normalize_effort(self, messages, effort: str):
        messages = [dict(message) for message in messages]
        # No Bonsai-specific reinterpretation: the shared policy resolves
        # MACQWEN levels before this point, and the template natively
        # supports xhigh, medium, and low. Unknown levels fail closed.
        if effort not in THINK_TAGS:
            raise ValueError(f"unsupported Bonsai-2 reasoning effort: {effort}")
        thinking_fields = {
            "think", "think_fast", "think_faster", "reasoning",
            "reasoning_content",
        }
        for message in messages:
            if message.get("role") == "assistant" and not (
                thinking_fields & message.keys()
            ):
                message[THINK_FIELDS[effort]] = ""
        self.thinking_tag = THINK_TAGS[effort]
        return messages, effort

    @staticmethod
    def _normalize_tool_arguments(messages) -> None:
        """Parse JSON argument strings into objects before rendering.

        API histories carry function.arguments as a JSON string, but the
        installed template iterates arguments as a mapping. A follow-up
        request containing a previous tool call would fail inside the
        template, so valid objects normalize here and anything else fails
        closed with a clear error instead of a template traceback.
        """
        for message in messages:
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("tool call history must hold objects")
                function = call.get("function")
                if isinstance(function, dict):
                    target, key = function, "arguments"
                elif isinstance(call.get("arguments"), str):
                    target, key = call, "arguments"
                else:
                    continue
                arguments = target.get(key, {})
                if isinstance(arguments, dict):
                    continue
                if not isinstance(arguments, str):
                    raise ValueError(
                        "tool call arguments must be an object or JSON text"
                    )
                try:
                    parsed = json.loads(arguments)
                except (TypeError, ValueError):
                    raise ValueError(
                        "tool call arguments are not valid JSON"
                    ) from None
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "tool call arguments must decode to an object"
                    )
                target[key] = parsed

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="medium",
        **options,
    ):
        messages, effort = self._normalize_effort(messages, reasoning_effort)
        self._normalize_tool_arguments(messages)
        text = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=enable_thinking,
            reasoning_effort=effort,
            tool_presentation_format="markdown",
            tool_call_format="json",
            **options,
        )
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text


class BonsaiBackend(Conversation):
    """Adapt the checkpoint-owned Bonsai-2 runtime to the shared backend contract."""

    def __init__(
        self,
        model_path: str,
        *,
        prefill_step_size: int = 512,
        allocator_cache_mb: float | None = None,
        clear_cache_after_generate: bool = False,
        wired_limit_enabled: bool = False,
        fused_fwht: bool = True,
        share_fwht: bool = False,
        retain_stop: bool = False,
        quantized_kv: tuple | list | None = None,
        q4_attention_tiling: bool = True,
        q4_attention_temp_mb: float = _Q4_ATTENTION_DEFAULT_TEMP_MB,
        trace_memory: bool = False,
        session_dir: str = SESSION_DIR,
    ):
        prefill_step_size = int(prefill_step_size)
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be greater than zero")
        if allocator_cache_mb is not None:
            allocator_cache_mb = float(allocator_cache_mb)
            if not math.isfinite(allocator_cache_mb) or allocator_cache_mb < 0:
                raise ValueError("allocator_cache_mb must be a finite non-negative number")
        q4_attention_temp_mb = float(q4_attention_temp_mb)
        if not math.isfinite(q4_attention_temp_mb) or q4_attention_temp_mb <= 0:
            raise ValueError("q4_attention_temp_mb must be a finite positive number")

        import mlx_lm
        from mlx_lm.models.cache import make_prompt_cache

        path = resolve_bonsai2(model_path)
        if not runtime_available(path):
            raise RuntimeError(
                "Bonsai-2 requires its bundled runtime/ loader "
                f"(missing in {path}); stock loaders return wrong output "
                "silently, so refusing to run"
            )
        for candidate in (path / "runtime", Path(__file__).resolve().parent / "runtime"):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        import vision_artifact  # noqa: F401 (proves the bundled runtime loads)

        from .ternary_kernel import apply_runtime_hooks

        vl_model, _pack_config = _load_text_model(path)
        model = vl_model.language_model
        if share_fwht:
            _verify_shared_signs(model)
        apply_runtime_hooks(path, fused=fused_fwht, share=share_fwht)
        with _transformers_import_environment():
            from transformers import AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(path), fix_mistral_regex=True
                )
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
        super().__init__(BonsaiTokenizer(tokenizer))
        self.model = model
        self._text_model = _TextModelWrapper(model, share=share_fwht)
        self.model_path = str(path)
        self.fused_fwht = bool(fused_fwht)
        self.share_fwht = bool(share_fwht)
        self.q4_attention_tiling = bool(q4_attention_tiling)
        self.q4_attention_temp_mb = q4_attention_temp_mb
        self.trace_memory = bool(trace_memory)
        self.attention_events = []
        _install_q4_attention_tiling(
            self.q4_attention_tiling,
            int(self.q4_attention_temp_mb * 1024 * 1024),
        )
        self.cache = make_prompt_cache(model)
        if quantized_kv is None:
            quantized_kv = _quantized_kv_from_environment()
        self.quantized_kv = (
            (int(quantized_kv[0]), int(quantized_kv[1]))
            if quantized_kv is not None
            else None
        )
        if self.quantized_kv is not None:
            bits, group_size = self.quantized_kv
            if bits not in (4, 8) or group_size <= 0:
                raise ValueError("quantized_kv needs (bits, group_size) with bits 4 or 8")
            _quantize_kv_caches(self.cache, bits, group_size)
        self._validate_cache(self.cache)
        self._attach_attention_trace()
        self.prefill_step_size = prefill_step_size
        self.allocator_cache_mb = allocator_cache_mb
        self.clear_cache_after_generate = bool(clear_cache_after_generate)
        self.wired_limit_enabled = bool(wired_limit_enabled)
        # Kept for constructor compatibility. Stops are now always rejected
        # before the one-ahead model call, so retaining a consumed stop is no
        # longer a separate mode.
        self.retain_stop = bool(retain_stop)
        self.session_dir = Path(session_dir).expanduser()
        self.thinking_enabled = False
        self.reasoning_effort = "medium"
        self.think_budget = 0
        self.answer_budget = 0
        self._interactive_budgets = None
        self.sampling = Sampling.greedy_settings()
        self._thinking_tag = THINK_TAGS["medium"]
        self._replay_needed = False
        eos = getattr(tokenizer, "eos_token_ids", None)
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", ())
        if isinstance(eos, int):
            eos = (eos,)
        self.stops = {int(value) for value in (eos or ()) if value is not None}
        end = tokenizer.convert_tokens_to_ids(IM_END)
        self._im_end_id = int(end) if end is not None else None
        if end is not None:
            self.stops.add(int(end))

    @property
    def cache_tokens(self) -> int:
        offsets = [int(item.offset) for item in self.cache if hasattr(item, "offset")]
        return offsets[0] if offsets else 0

    def _attach_attention_trace(self) -> None:
        for layer, item in enumerate(self.cache):
            if item is None:
                continue
            if self.trace_memory:
                item._bonsai_attention_trace = self.attention_events
                item._bonsai_attention_layer = layer
            else:
                for name in ("_bonsai_attention_trace", "_bonsai_attention_layer"):
                    if hasattr(item, name):
                        delattr(item, name)

    def runtime_settings(self) -> dict[str, object]:
        return {
            "prefill_step_size": self.prefill_step_size,
            "allocator_cache_mb": self.allocator_cache_mb,
            "fused_fwht": self.fused_fwht,
            "share_fwht": self.share_fwht,
            "quantized_kv": (
                list(self.quantized_kv) if self.quantized_kv is not None else None
            ),
            "q4_attention_tiling": self.q4_attention_tiling,
            "q4_attention_temp_mb": self.q4_attention_temp_mb,
            "trace_memory": self.trace_memory,
        }

    @staticmethod
    def _validate_cache(cache) -> None:
        from mlx_lm.models.cache import ArraysCache as LmArrays
        from mlx_lm.models.cache import KVCache as LmKv
        from mlx_lm.models.cache import QuantizedKVCache as LmQuant

        try:
            from mlx_vlm.models.cache import ArraysCache as VlmArrays
            from mlx_vlm.models.cache import KVCache as VlmKv
            from mlx_vlm.models.cache import QuantizedKVCache as VlmQuant
        except ImportError:
            VlmArrays = VlmKv = VlmQuant = None
        recognized = (LmArrays, LmKv, LmQuant) + tuple(
            cls for cls in (VlmArrays, VlmKv, VlmQuant) if cls is not None
        )
        kinds = {type(item) for item in cache}
        if not kinds <= set(recognized):
            raise TypeError(
                "Bonsai-2 requires ArraysCache/KVCache objects"
            )
        full_attention = (LmKv, LmQuant) + tuple(
            cls for cls in (VlmKv, VlmQuant) if cls is not None
        )
        if kinds and not any(
            issubclass(kind, full_attention) for kind in kinds
        ):
            raise TypeError("Bonsai-2 requires full-attention KVCache layers")

    def check_invariant(self) -> bool:
        # A replay-needed state is valid, not broken: the tape is
        # authoritative and the next turn rebuilds the cache from it. Only
        # a live cache must match the tape offsets.
        if self._replay_needed:
            return True
        offsets = [int(item.offset) for item in self.cache if hasattr(item, "offset")]
        return (
            not self.cache and not self.tape
        ) or (
            bool(offsets)
            and all(offset == len(self.tape) for offset in offsets)
        )

    def _mark_replay_needed(self) -> None:
        """Drop a partially consumed cache; the next turn replays the tape."""
        from mlx_lm.models.cache import make_prompt_cache

        self.cache = make_prompt_cache(self.model)
        if self.quantized_kv is not None:
            bits, group_size = self.quantized_kv
            _quantize_kv_caches(self.cache, bits, group_size)
        self._validate_cache(self.cache)
        self._attach_attention_trace()
        self._replay_needed = bool(self.tape)
        self.turn_closed = False

    def generate(
        self,
        max_tokens: int,
        out=None,
        on_prefilled=None,
        on_prefill_progress=None,
        on_decode_token=None,
        should_cancel=None,
    ) -> tuple[str, Stats]:
        if not self.pending:
            return "", Stats()

        import mlx.core as mx
        from mlx_lm.generate import generate_step, generation_stream, wired_limit

        if self.trace_memory:
            self.attention_events.clear()

        # Validate budgets before pending tokens move into the tape: an
        # early validation exception must not leave moved-but-unprocessed
        # state behind.
        interactive_budgets = getattr(self, "_interactive_budgets", None)
        budget_answer = budget_think = 0
        close_token = None
        if interactive_budgets is not None:
            budget_answer, budget_think = interactive_budgets
            budget_answer = int(budget_answer)
            budget_think = (
                None
                if budget_think is None or int(budget_think) < 0
                else int(budget_think)
            )
            if self.thinking_enabled and budget_think is not None:
                close_ids = self.encode("</think>")
                if len(close_ids) != 1:
                    raise ValueError(
                        "interactive reasoning closure must encode as one token"
                    )
                close_token = int(close_ids[0])
        separate_budgets = (
            interactive_budgets is not None
            and self.thinking_enabled
            and budget_think is not None
            and budget_think > 0
        )
        prompt = list(self.tape) + self.pending if self._replay_needed else list(self.pending)
        prompt_tokens = len(prompt)
        self.tape.extend(self.pending)
        self.pending = []
        self._replay_needed = False
        sampler = Sampler(self.sampling)
        decoding_sampler = (
            _ForcingSampler(sampler, budget_think, close_token)
            if separate_budgets
            else sampler
        )
        prefill_began = time.perf_counter()
        prefill_seconds = 0.0
        timer = None
        prefilled = False

        def check_cancel():
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled

        def progress(done, total):
            nonlocal prefill_seconds, timer, prefilled
            check_cancel()
            if on_prefill_progress is not None:
                on_prefill_progress(done, total)
            if done >= total and not prefilled:
                prefilled = True
                prefill_seconds = time.perf_counter() - prefill_began
                timer = DecodeTimer()
                if on_prefilled is not None:
                    on_prefilled()

        steps = None
        text_model = _NonConsumingStopModel(
            self._text_model, prompt_tokens, self.prefill_step_size, self.stops
        )
        try:
            check_cancel()
            steps = generate_step(
                mx.array(prompt),
                text_model,
                max_tokens=max_tokens,
                sampler=decoding_sampler,
                prompt_cache=self.cache,
                prefill_step_size=self.prefill_step_size,
                prompt_progress_callback=progress,
            )
        except BaseException:
            try:
                mx.synchronize(generation_stream)
            finally:
                self._mark_replay_needed()
            raise
        produced: list[int] = []
        pieces: list[str] = []
        partial: list[int] = []
        protocol = ProtocolTranslator()
        finish = "length"
        stop_seen = False
        interrupted = False
        cleaned = False
        answer_limited = False

        phase = "thinking" if self.thinking_enabled else "answer"
        thinking_count = answer_count = 0

        cache_limit = _cache_limit(mx, self.allocator_cache_mb)


        def cleanup_steps():
            nonlocal cleaned, interrupted
            if cleaned:
                return
            cleaned = True
            close = getattr(steps, "close", None)
            try:
                if close is not None:
                    close()
            finally:
                # generate_step schedules one prediction ahead.  Synchronize
                # the same stream before wired_limit restores its old limit.
                mx.synchronize(generation_stream)

        try:
            with cache_limit:
                try:
                    residency = (
                        wired_limit(self.model, [generation_stream])
                        if self.wired_limit_enabled
                        else nullcontext()
                    )
                    with residency:
                        try:
                            # One-ahead invariant: generate_step() feeds each
                            # accepted token back through the model before
                            # yielding the next. The stop-aware wrapper keeps
                            # protocol stops out of that call, so every token
                            # taped here is present in every cache layer.
                            for token, _logprobs in steps:
                                check_cancel()
                                value = int(token)
                                if value in self.stops:
                                    stop_seen = True
                                    finish = "stop"
                                    # Stop-aware generation keeps this token
                                    # out of every cache layer. The next
                                    # append supplies the structural close as
                                    # pending input, so only the new framing
                                    # and tool/user content are prefilled.
                                    self.turn_closed = False
                                    break
                                self.tape.append(value)
                                produced.append(value)
                                decoding_sampler.observe(value)
                                raw = stream_decode(self.tokenizer, partial, value)
                                piece = protocol.feed(raw) if raw else ""
                                if separate_budgets:
                                    if phase == "thinking":
                                        thinking_count += 1
                                    else:
                                        answer_count += 1
                                    if (
                                        phase == "thinking"
                                        and separate_budgets
                                        and value == close_token
                                    ):
                                        # The yielded close token itself ends
                                        # thinking, forced or natural: it was
                                        # genuinely consumed, so tape, cache,
                                        # and phase agree. Sampler-side flags
                                        # describe the lookahead token already
                                        # sampled for the next step, never the
                                        # token yielded here, so they must not
                                        # drive the phase.
                                        phase = "answer"
                                        decoding_sampler.end_thinking()
                                    if (
                                        phase == "answer"
                                        and budget_answer >= 0
                                        and answer_count >= budget_answer
                                    ):
                                        # Break after accepting: this token is
                                        # already consumed into cache and
                                        # taped, and the sampled-but-unfed
                                        # next token is safe to discard. Tape
                                        # and cache stay aligned, so unlike a
                                        # mid-stream stop this needs no replay
                                        # and saves one forward. The accepted
                                        # token still emits below so reply,
                                        # stream, callbacks, and tape agree.
                                        answer_limited = True
                                        finish = "length"
                                        self.turn_closed = False
                                if on_decode_token is not None:
                                    on_decode_token(value, piece)
                                if piece:
                                    pieces.append(piece)
                                    if out is not None:
                                        with timer.emitting():
                                            out(piece)
                                if answer_limited:
                                    break
                            else:
                                self.turn_closed = False
                        except BaseException:
                            interrupted = True
                            raise
                        finally:
                            cleanup_steps()
                except BaseException:
                    interrupted = True
                    raise
                finally:
                    if not cleaned:
                        cleanup_steps()
                    steps = None
                    if self.clear_cache_after_generate:
                        mx.clear_cache()
        finally:
            if not cleaned:
                interrupted = True
                cleanup_steps()
            steps = None
            try:
                if stop_seen and not self.turn_closed and not self.check_invariant():
                    # The production wrapper above prevents stop consumption.
                    # Keep the old recovery path for alternate generators or
                    # a future MLX change that violates that boundary.
                    self._mark_replay_needed()
            finally:
                if interrupted:
                    self._mark_replay_needed()

        raw_tail = self.tokenizer.decode(partial) if partial else ""
        tail = protocol.feed(raw_tail) + protocol.finish()
        if tail:
            pieces.append(tail)
            if out is not None:
                with timer.emitting():
                    out(tail)
        if not prefilled:
            progress(prompt_tokens, prompt_tokens)
        return "".join(pieces), Stats(
            finish=finish,
            tokens=len(produced),
            seconds=timer.elapsed(),
            prompt_tokens=prompt_tokens,
            prefill_seconds=prefill_seconds,
        )

    def _session_path(self, name: str) -> Path:
        if not _SESSION_NAME.fullmatch(name):
            raise ValueError("invalid name; use 1 to 64 letters, numbers, dots, _ or -")
        return self.session_dir / f"{name}.json"

    def save_session(self, name: str) -> str:
        temporary = None
        try:
            path = self._session_path(name)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            payload = json.dumps({
                "schema": SESSION_SCHEMA,
                "model_path": self.model_path,
                "config_sha256": _config_identity(self.model_path),
                "tape": self.tape,
                "pending": self.pending,
                "turn_closed": self.turn_closed,
                "thinking": self.thinking_enabled,
                "thinking_tag": self._thinking_tag,
                "budgets": (
                    None
                    if self._interactive_budgets is None
                    else [
                        int(self._interactive_budgets[0]),
                        (
                            None
                            if self._interactive_budgets[1] is None
                            else int(self._interactive_budgets[1])
                        ),
                    ]
                ),
            }, separators=(",", ":"))
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False,
                prefix=f".{path.name}.", suffix=".tmp",
            ) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            os.replace(temporary, path)
            temporary = None
        except (OSError, TypeError, ValueError) as exc:
            return f"could not save session: {exc}"
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return f"saved {name}  {len(self.tape)} tokens"

    def load_session(self, name: str) -> str:
        def valid_tokens(values) -> list[int] | None:
            if not isinstance(values, list):
                return None
            try:
                vocab_size = len(self.tokenizer)
            except TypeError:
                vocab_size = None
            clean = []
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool):
                    return None
                if vocab_size is not None and not 0 <= value < vocab_size:
                    return None
                clean.append(value)
            return clean

        def valid_flag(value) -> bool | None:
            return value if isinstance(value, bool) else None

        try:
            payload = json.loads(self._session_path(name).read_text())
            if not isinstance(payload, dict):
                raise ValueError("session payload must be an object")
            if payload.get("schema") != SESSION_SCHEMA:
                raise ValueError("session schema is not supported here")
            if payload.get("model_path") != self.model_path:
                raise ValueError("session belongs to another checkpoint")
            expected_config = _config_identity(self.model_path)
            if expected_config is None:
                raise ValueError(
                    "session checkpoint identity is unavailable"
                )
            if payload.get("config_sha256") != expected_config:
                raise ValueError("session checkpoint config changed")
            tape = valid_tokens(payload.get("tape"))
            if tape is None:
                raise ValueError("session token tape is invalid")
            pending = valid_tokens(payload.get("pending", []))
            if pending is None:
                raise ValueError("session pending tokens are invalid")
            tag = payload.get("thinking_tag", THINK_TAGS["medium"])
            if not isinstance(tag, str) or tag not in THINK_TAGS.values():
                raise ValueError("session thinking tag is invalid")
            turn_closed = valid_flag(payload.get("turn_closed", True))
            thinking = valid_flag(payload.get("thinking", False))
            if turn_closed is None or thinking is None:
                raise ValueError("session flags are invalid")
            budgets = payload.get("budgets", None)
            if budgets is not None:
                # Budgets ride the session so a restored conversation keeps
                # its reasoning contract instead of inheriting whatever a
                # later chat set. Strict shapes only: bools are ints in
                # Python and must not smuggle in as token counts.
                if (
                    not isinstance(budgets, list)
                    or len(budgets) != 2
                    or isinstance(budgets[0], bool)
                    or not isinstance(budgets[0], int)
                    or budgets[0] < -1
                    or (
                        budgets[1] is not None
                        and (
                            isinstance(budgets[1], bool)
                            or not isinstance(budgets[1], int)
                            or budgets[1] < -1
                        )
                    )
                ):
                    raise ValueError("session budgets are invalid")
                budgets = (budgets[0], budgets[1])
            self.reset()
            self.tape = tape
            self.pending = pending
            self.turn_closed = turn_closed
            self.thinking_enabled = thinking
            self._thinking_tag = tag
            self._interactive_budgets = budgets
            self._replay_needed = bool(self.tape or self.pending)
        except (OSError, TypeError, ValueError) as exc:
            return f"could not load session: {exc}"
        return f"loaded {name}  {len(self.tape)} tokens; cache will replay once"

    def list_sessions(self) -> str:
        try:
            paths = sorted(self.session_dir.glob("*.json"))
        except OSError as exc:
            return f"could not list sessions: {exc}"
        if not paths:
            return "no saved sessions"
        rows = []
        for path in paths:
            try:
                count = len(json.loads(path.read_text())["tape"])
                rows.append(f"  {path.stem:<24} {count:>7} tok")
            except (OSError, KeyError, TypeError, ValueError):
                rows.append(f"  {path.stem:<24} invalid")
        return "\n".join(rows)

    def delete_session(self, name: str) -> str:
        try:
            path = self._session_path(name)
            if not path.exists():
                return f"{name} not found"
            path.unlink()
        except (OSError, ValueError) as exc:
            return f"could not delete session: {exc}"
        return f"{name} deleted"

    def reset(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self.cache = make_prompt_cache(self.model)
        if self.quantized_kv is not None:
            bits, group_size = self.quantized_kv
            _quantize_kv_caches(self.cache, bits, group_size)
        self._validate_cache(self.cache)
        self._attach_attention_trace()
        self.tape = []
        self.pending = []
        self.turn_closed = True
        self._replay_needed = False
        mx.clear_cache()

    def configure(self, argument: str) -> str:
        text = argument.strip()

        def kv_description() -> str:
            if self.quantized_kv is None:
                return "fp32"
            bits, group_size = self.quantized_kv
            return f"{bits}-bit (group {group_size})"

        if text == "kv-cache":
            return f"kv-cache            {kv_description()}"
        if text.startswith("kv-cache "):
            value = text.split(None, 1)[1].strip().lower()
            if value in ("off", "fp32"):
                selected = None
            elif value in ("4", "8"):
                selected = (int(value), 64)
            else:
                raise ValueError("kv-cache expects 4, 8, fp32, or off")
            if selected != self.quantized_kv:
                turn_closed = self.turn_closed
                self.quantized_kv = selected
                self._mark_replay_needed()
                self.turn_closed = turn_closed
            return f"kv-cache            {kv_description()}"
        if text in ("", "all"):
            return (
                "Bonsai-2 settings\n"
                f"  checkpoint          {self.model_path}\n"
                f"  prefill-step-size   {self.prefill_step_size}\n"
                f"  allocator-cache-mb  {self.allocator_cache_mb if self.allocator_cache_mb is not None else 'off'}\n"
                f"  fused-fwht          {'on' if self.fused_fwht else 'off'}\n"
                f"  kv-cache            {kv_description()}\n"
                f"  q4-attention       {'tiled' if self.q4_attention_tiling else 'stock'}\n"
                f"  q4-attention-mb    {self.q4_attention_temp_mb:g}\n"
                f"  clear-cache-after   {'on' if self.clear_cache_after_generate else 'off'}\n"
                f"  wired-limit         {'on' if self.wired_limit_enabled else 'off'}"
            )
        raise ValueError("Bonsai-2 has no model-specific runtime settings")

    def settings_state(self, include_research: bool = False) -> str:
        del include_research
        return self.configure("all")
