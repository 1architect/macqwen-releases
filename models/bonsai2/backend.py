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
    "high": "think",
    "medium": "think",
    "low": "think",
    "xhigh": "think",
}
THINK_FIELDS = {
    "high": "think",
    "medium": "think",
    "low": "think",
    "xhigh": "think",
}
_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


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
        if block and signs is None:
            raise ValueError("Missing sign vector")
        if signs is not None and not mx.all((signs == 1) | (signs == -1)).item():
            raise ValueError("Invalid sign values")
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
        # The model card marks `low` unsupported (behaves close to `xhigh`);
        # keep `xhigh` native and fold `low` to `medium`. MACQWEN `high` is
        # project-specific, so map it to the native `xhigh` level.
        if effort == "low":
            effort = "medium"
        elif effort == "high":
            effort = "xhigh"
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
            add_generation_prompt=add_generation_prompt and enable_thinking,
            tokenize=False,
            reasoning_effort=effort,
            tool_presentation_format="markdown",
            tool_call_format="json",
            **options,
        )
        if add_generation_prompt and not enable_thinking:
            tag = self.thinking_tag
            text += f"{IM_START}assistant\n<{tag}>\n</{tag}>"
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
        fused_fwht: bool = False,
        share_fwht: bool = False,
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

        if fused_fwht:
            os.environ["BONSAI2_FUSED_FWHT"] = "1"
            from .ternary_kernel import install_packed_hook

            install_packed_hook()
        if share_fwht:
            os.environ["BONSAI2_SHARE_FWHT"] = "1"
        vl_model, _pack_config = _load_text_model(path)
        model = vl_model.language_model
        if share_fwht:
            from .ternary_kernel import install_share_hook

            _verify_shared_signs(model)
            install_share_hook()
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

    @staticmethod
    def _effort(effort: str) -> str:
        if effort == "low":
            return "medium"
        return effort

    def _assistant_prefix(self, enable_thinking: bool) -> str:
        tag = self._thinking_tag
        prefix = f"{IM_START}assistant\n<{tag}>\n"
        return prefix if enable_thinking else f"{prefix}</{tag}>"

    def _select_thinking_tag(self, effort: str) -> None:
        self._thinking_tag = THINK_TAGS[self._effort(effort)]

    def open_conversation(
        self,
        system,
        user,
        tools=None,
        enable_thinking=True,
        reasoning_effort="high",
    ) -> int:
        if self.tape or self.pending:
            raise RuntimeError("conversation already open")
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        self._thinking_tag = self.tokenizer.thinking_tag
        return self.append_text(text)

    def append_user(self, text: str, enable_thinking: bool = True) -> int:
        self._select_thinking_tag(self.reasoning_effort)
        close = "" if self.turn_closed else IM_END
        return self.append_text(
            f"{close}{IM_START}user\n{text}{IM_END}"
            f"{self._assistant_prefix(enable_thinking)}"
        )

    def append_tool_results(self, results, enable_thinking: bool | None = None) -> int:
        if enable_thinking is None:
            enable_thinking = self.thinking_enabled
        close = "" if self.turn_closed else IM_END
        body = "".join(f"{IM_START}tool\n{result}{IM_END}" for result in results)
        return self.append_text(close + body + self._assistant_prefix(enable_thinking))

    def append_text(self, text: str) -> int:
        text = text.replace("</think>", f"</{self._thinking_tag}>")
        return super().append_text(text)

    @property
    def cache_tokens(self) -> int:
        offsets = [int(item.offset) for item in self.cache if hasattr(item, "offset")]
        return offsets[0] if offsets else 0

    @staticmethod
    def _validate_cache(cache) -> None:
        kinds = {type(item).__name__ for item in cache}
        if not kinds <= {"ArraysCache", "KVCache", "QuantizedKVCache"}:
            raise TypeError(
                "Bonsai-2 requires ArraysCache/KVCache objects"
            )
        if kinds and "KVCache" not in kinds and "QuantizedKVCache" not in kinds:
            raise TypeError("Bonsai-2 requires full-attention KVCache layers")

    def check_invariant(self) -> bool:
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
                                self.tape.append(value)
                                produced.append(value)
                                sampler.observe(value)
                                raw = stream_decode(self.tokenizer, partial, value)
                                piece = protocol.feed(raw) if raw else ""
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
                "model_path": self.model_path,
                "tape": self.tape,
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
        try:
            payload = json.loads(self._session_path(name).read_text())
            if payload.get("model_path") != self.model_path:
                raise ValueError("session belongs to another checkpoint")
            tape = payload.get("tape")
            if not isinstance(tape, list):
                raise ValueError("session token tape is invalid")
            tag = payload.get("thinking_tag", THINK_TAGS["medium"])
            if tag not in THINK_TAGS.values():
                raise ValueError("session thinking tag is invalid")
            self.reset()
            self.tape = [int(value) for value in tape]
            self.turn_closed = bool(payload.get("turn_closed", True))
            self.thinking_enabled = bool(payload.get("thinking", False))
            self._thinking_tag = tag
            self._replay_needed = bool(self.tape)
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
