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

from macqwen.backends.base import DecodeTimer
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


def _config_identity(model_path: str) -> str | None:
    """Identify the checkpoint config without loading weights."""
    import hashlib

    try:
        digest = hashlib.sha256()
        with open(Path(model_path).expanduser() / "config.json", "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
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


class BonsaiTokenizer:
    """Present the Bonsai-2 template through the interface the shared chat already uses."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.thinking_tag = THINK_TAGS["medium"]

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

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
        gemv: bool = False,
        retain_stop: bool = False,
        quantized_kv: tuple | list | None = None,
        session_dir: str = SESSION_DIR,
    ):
        prefill_step_size = int(prefill_step_size)
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be greater than zero")
        if allocator_cache_mb is not None:
            allocator_cache_mb = float(allocator_cache_mb)
            if not math.isfinite(allocator_cache_mb) or allocator_cache_mb < 0:
                raise ValueError("allocator_cache_mb must be a finite non-negative number")

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

        from .ternary_kernel import (
            install_packed_hook,
            install_share_hook,
            restore_runtime_hooks,
        )

        if fused_fwht:
            os.environ["BONSAI2_FUSED_FWHT"] = "1"
            install_packed_hook(path)
        if share_fwht:
            os.environ["BONSAI2_SHARE_FWHT"] = "1"
        if not fused_fwht and not share_fwht:
            restore_runtime_hooks(path)
        vl_model, _pack_config = _load_text_model(path)
        model = vl_model.language_model
        if share_fwht:
            _verify_shared_signs(model)
            install_share_hook(path)
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
        self.prefill_step_size = prefill_step_size
        self.allocator_cache_mb = allocator_cache_mb
        self.clear_cache_after_generate = bool(clear_cache_after_generate)
        self.wired_limit_enabled = bool(wired_limit_enabled)
        # Experimental: retain a consumed <|im_end|> close in the tape and
        # continue on the live cache instead of replaying history. Off by
        # default until user-turn and tool-turn continuation checks pass.
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
        self._replay_needed = bool(self.tape)
        self.turn_closed = False

    def generate(
        self,
        max_tokens: int,
        out=None,
        on_prefilled=None,
        on_prefill_progress=None,
        on_decode_token=None,
    ) -> tuple[str, Stats]:
        if not self.pending:
            return "", Stats()

        import mlx.core as mx
        from mlx_lm.generate import generate_step, generation_stream, wired_limit

        prompt = list(self.tape) + self.pending if self._replay_needed else list(self.pending)
        prompt_tokens = len(prompt)
        self.tape.extend(self.pending)
        self.pending = []
        self._replay_needed = False
        sampler = Sampler(self.sampling)
        prefill_began = time.perf_counter()
        prefill_seconds = 0.0
        timer = None
        prefilled = False

        def progress(done, total):
            nonlocal prefill_seconds, timer, prefilled
            if on_prefill_progress is not None:
                on_prefill_progress(done, total)
            if done >= total and not prefilled:
                prefilled = True
                prefill_seconds = time.perf_counter() - prefill_began
                timer = DecodeTimer()
                if on_prefilled is not None:
                    on_prefilled()

        steps = None
        try:
            steps = generate_step(
                mx.array(prompt),
                self._text_model,
                max_tokens=max_tokens,
                sampler=sampler,
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

        interactive_budgets = getattr(self, "_interactive_budgets", None)
        budget_answer = budget_think = 0
        close_token = None
        if interactive_budgets is not None:
            budget_answer, budget_think = interactive_budgets
            budget_answer = max(0, int(budget_answer))
            budget_think = (
                None if budget_think is None else max(0, int(budget_think))
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
                            for token, _logprobs in steps:
                                value = int(token)
                                force_close = (
                                    separate_budgets
                                    and phase == "thinking"
                                    and thinking_count == budget_think - 1
                                )
                                if value in self.stops:
                                    stop_seen = True
                                    finish = "stop"
                                    if (
                                        self.retain_stop
                                        and self._im_end_id is not None
                                        and value == self._im_end_id
                                    ):
                                        # The close token is already consumed
                                        # into every cache layer through the
                                        # one-ahead lookahead. Retain it in
                                        # the tape and close the turn: the
                                        # next-turn builders omit a second
                                        # close, so the combined sequence is
                                        # identical while the live cache
                                        # survives. Other stops keep the
                                        # replay recovery path below.
                                        self.tape.append(value)
                                        self.turn_closed = True
                                    else:
                                        self.turn_closed = False
                                    break
                                if force_close:
                                    value = close_token
                                if (
                                    separate_budgets
                                    and phase == "answer"
                                    and answer_count >= budget_answer
                                ):
                                    answer_limited = True
                                    finish = "length"
                                    self.turn_closed = False
                                    break
                                self.tape.append(value)
                                produced.append(value)
                                sampler.observe(value)
                                raw = stream_decode(self.tokenizer, partial, value)
                                piece = protocol.feed(raw) if raw else ""
                                if separate_budgets:
                                    if phase == "thinking":
                                        thinking_count += 1
                                    else:
                                        answer_count += 1
                                    if phase == "thinking" and (
                                        force_close or "</think>" in piece
                                    ):
                                        phase = "answer"
                                if on_decode_token is not None:
                                    on_decode_token(value, piece)
                                if piece:
                                    pieces.append(piece)
                                    if out is not None:
                                        with timer.emitting():
                                            out(piece)
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
                if stop_seen and not self.turn_closed:
                    # Rewinding KV offsets is not enough: the 48 GDN linear
                    # states have no offset and already absorbed the stop
                    # token through the one-ahead lookahead. Continuing from
                    # them contaminates the next turn, so drop the whole
                    # cache and replay the tape instead. A retained
                    # <|im_end|> close leaves turn_closed true and skips
                    # this path with the live cache intact.
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
            clean = []
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool):
                    return None
                clean.append(value)
            return clean

        def valid_flag(value) -> bool | None:
            return value if isinstance(value, bool) else None

        try:
            payload = json.loads(self._session_path(name).read_text())
            if payload.get("schema") != SESSION_SCHEMA:
                raise ValueError("session schema is not supported here")
            if payload.get("model_path") != self.model_path:
                raise ValueError("session belongs to another checkpoint")
            expected_config = _config_identity(self.model_path)
            if (
                expected_config is not None
                and payload.get("config_sha256") != expected_config
            ):
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
            self.reset()
            self.tape = tape
            self.pending = pending
            self.turn_closed = turn_closed
            self.thinking_enabled = thinking
            self._thinking_tag = tag
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
        self.tape = []
        self.pending = []
        self.turn_closed = True
        self._replay_needed = False
        mx.clear_cache()

    def configure(self, argument: str) -> str:
        if argument.strip() in ("", "all"):
            return (
                "Bonsai-2 settings\n"
                f"  checkpoint          {self.model_path}\n"
                f"  prefill-step-size   {self.prefill_step_size}\n"
                f"  allocator-cache-mb  {self.allocator_cache_mb if self.allocator_cache_mb is not None else 'off'}\n"
                f"  clear-cache-after   {'on' if self.clear_cache_after_generate else 'off'}\n"
                f"  wired-limit         {'on' if self.wired_limit_enabled else 'off'}"
            )
        raise ValueError("Bonsai-2 has no model-specific runtime settings")

    def settings_state(self, include_research: bool = False) -> str:
        del include_research
        return self.configure("all")
