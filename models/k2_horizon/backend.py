"""Resident MLX-LM runtime for K2-Horizon 7B."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from macqwen.backends.base import DecodeTimer
from macqwen.conversation import Conversation, EXTRA_REASONING
from macqwen.sampling import Sampler, Sampling
from macqwen.text import stream_decode

from .checkpoint import resolve_k2_horizon
from .protocol import ProtocolTranslator
from .settings import SESSION_DIR


IM_START = "<|ifm|im_start|>"
IM_END = "<|ifm|im_end|>"
THINK_TAGS = {
    "high": "ifm|think",
    "medium": "ifm|think_fast",
    "low": "ifm|think_faster",
}
THINK_FIELDS = {
    "high": "think",
    "medium": "think_fast",
    "low": "think_faster",
}
_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


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


class K2Tokenizer:
    """Present K2's template through the interface the shared chat already uses."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.thinking_tag = THINK_TAGS["medium"]

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def _normalize_effort(self, messages, effort: str):
        messages = [dict(message) for message in messages]
        high_instruction = EXTRA_REASONING["high"]
        if effort == "medium" and messages and messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if content == high_instruction or content.startswith(high_instruction + "\n\n"):
                messages[0]["content"] = content[len(high_instruction):].lstrip("\n")
                effort = "high"
        if effort == "xhigh":
            effort = "high"
        if effort not in THINK_TAGS:
            raise ValueError(f"unsupported K2-Horizon reasoning effort: {effort}")
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


class K2HorizonBackend(Conversation):
    """Adapt the checkpoint-owned K2 runtime to the shared backend contract."""

    def __init__(
        self,
        model_path: str,
        *,
        prefill_step_size: int = 512,
        allocator_cache_mb: float | None = None,
        clear_cache_after_generate: bool = False,
        wired_limit_enabled: bool = False,
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

        path = resolve_k2_horizon(model_path)
        model, tokenizer = mlx_lm.load(
            str(path), tokenizer_config={"trust_remote_code": True}
        )
        super().__init__(K2Tokenizer(tokenizer))
        self.model = model
        self.model_path = str(path)
        self.cache = make_prompt_cache(model)
        self._validate_cache(self.cache)
        self.prefill_step_size = prefill_step_size
        self.allocator_cache_mb = allocator_cache_mb
        self.clear_cache_after_generate = bool(clear_cache_after_generate)
        self.wired_limit_enabled = bool(wired_limit_enabled)
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
        if end is not None:
            self.stops.add(int(end))

    @staticmethod
    def _effort(effort: str) -> str:
        return "high" if effort in ("high", "xhigh") else effort

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

    def _rewind_stop_token(self) -> None:
        for item in self.cache:
            if hasattr(item, "offset"):
                if item.offset < 1:
                    raise RuntimeError("K2-Horizon cache cannot rewind its stop token")
                item.offset -= 1

    @staticmethod
    def _validate_cache(cache) -> None:
        from mlx_lm.models.cache import KVCache

        if any(type(item) is not KVCache for item in cache):
            raise TypeError("K2-Horizon requires ordinary MLX-LM KVCache objects")

    def check_invariant(self) -> bool:
        offsets = [int(item.offset) for item in self.cache if hasattr(item, "offset")]
        return (
            not self.cache and not self.tape
        ) or (
            bool(offsets)
            and len(offsets) == len(self.cache)
            and all(offset == len(self.tape) for offset in offsets)
        )

    def _mark_replay_needed(self) -> None:
        """Drop a partially consumed cache; the next turn replays the tape."""
        from mlx_lm.models.cache import make_prompt_cache

        self.cache = make_prompt_cache(self.model)
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
                self.model,
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
                if stop_seen:
                    self._rewind_stop_token()
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
        self._validate_cache(self.cache)
        self.tape = []
        self.pending = []
        self.turn_closed = True
        self._replay_needed = False
        mx.clear_cache()

    def configure(self, argument: str) -> str:
        if argument.strip() in ("", "all"):
            return (
                "K2-Horizon settings\n"
                f"  checkpoint          {self.model_path}\n"
                f"  prefill-step-size   {self.prefill_step_size}\n"
                f"  allocator-cache-mb  {self.allocator_cache_mb if self.allocator_cache_mb is not None else 'off'}\n"
                f"  clear-cache-after   {'on' if self.clear_cache_after_generate else 'off'}\n"
                f"  wired-limit         {'on' if self.wired_limit_enabled else 'off'}"
            )
        raise ValueError("K2-Horizon has no model-specific runtime settings")

    def settings_state(self, include_research: bool = False) -> str:
        del include_research
        return self.configure("all")
