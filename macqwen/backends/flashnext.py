"""Qwen3.8-Flash-Next as an agent backend.

Supplies generation and cache accounting; the transcript comes from
Conversation and the loop from macqwen.agent. Streams experts from the
checkpoint, so a token costs an SSD read rather than resident memory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import time

import mlx.core as mx

from macqwen.backends.base import DecodeTimer
from macqwen.checkpoints import resolve_flashnext
from macqwen.conversation import Conversation
from macqwen.model_settings import FLASHNEXT_DEFAULTS
from macqwen.sampling import Sampler, Sampling
from macqwen.text import stream_decode
from models.flashnext.settings import get_registry

DEFAULT_FUSION_MODEL = FLASHNEXT_DEFAULTS["fusion_model"]
_TRANSFORMERS_ADVISORY_ENV = "TRANSFORMERS_NO_ADVISORY_WARNINGS"


@contextmanager
def _transformers_import_environment():
    """Hide Transformers' irrelevant PyTorch advisory during its import.

    Flash-Next uses MLX and only imports Transformers tokenizer utilities.
    Keep the environment change limited to the import, so later warnings and
    all real errors still reach the user.
    """
    previous = os.environ.get(_TRANSFORMERS_ADVISORY_ENV)
    os.environ[_TRANSFORMERS_ADVISORY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_TRANSFORMERS_ADVISORY_ENV, None)
        else:
            os.environ[_TRANSFORMERS_ADVISORY_ENV] = previous


def _load_transformers_tokenizer():
    with _transformers_import_environment():
        from transformers import AutoTokenizer
    return AutoTokenizer


@dataclass
class Stats:
    """What the agent loop reads back from a turn.

    `host_free_gb` and `swap_gb` stay None: this backend does not measure
    them, and the loop skips those guards rather than treating None as zero.
    """

    finish: str = "stop"
    tokens: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    tail_tokens: int = 0
    tail_seconds: float = 0.0
    ui_seconds: float = 0.0
    pinned_bytes: int = 0
    pinned_signature: str = ""
    host_free_gb: float | None = None
    swap_gb: float | None = None

    @property
    def rate(self) -> float:
        """Tokens per second of model time.

        `seconds` excludes the streaming callback. The terminal fade and the
        writes behind it run inside the decode loop, and their cost varies
        per word, so counting them made the reported rate wander for reasons
        that had nothing to do with the model.
        """
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def prompt_rate(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0


class FlashNextBackend(Conversation):
    def settings_registry(self):
        return get_registry()

    def settings_state(self, include_research: bool = False) -> str:
        return self.settings_registry().render(self, "flashnext", include_research)

    def __init__(self, model_path: str | None = None,
                 threshold: float = FLASHNEXT_DEFAULTS["threshold"],
                 resident_experts: int = FLASHNEXT_DEFAULTS["resident_experts"],
                 pin_budget_gb: float = FLASHNEXT_DEFAULTS["pin_budget_gb"],
                 routing_profile: str = FLASHNEXT_DEFAULTS["routing"],
                 swap_epsilon: float = FLASHNEXT_DEFAULTS["swap_epsilon"],
                 tail_experts: int = FLASHNEXT_DEFAULTS["tail_experts"],
                 tail_warmup: int = FLASHNEXT_DEFAULTS["tail_warmup"],
                 fusion_block: int = FLASHNEXT_DEFAULTS["fusion_block"],
                 fusion_min_margin: float = FLASHNEXT_DEFAULTS["fusion_min_margin"],
                 fusion_min_block: int = FLASHNEXT_DEFAULTS["fusion_min_block"],
                 fusion_margin_tokens: int = FLASHNEXT_DEFAULTS["fusion_margin_tokens"],
                 fusion_max_prompt: int = FLASHNEXT_DEFAULTS["fusion_max_prompt"],
                 fusion_model: str = DEFAULT_FUSION_MODEL,
                 session_dir: str = "~/.cache/flashnext/sessions"):
        AutoTokenizer = _load_transformers_tokenizer()
        from models.flashnext.loader import load_streaming
        from models.flashnext.qsa_chunk import apply as apply_qsa

        load_threshold = 0.20 if routing_profile == "fast" else threshold
        os.environ["FLASHNEXT_TOPK_THRESHOLD"] = str(load_threshold)
        try:
            path = str(resolve_flashnext(model_path))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        model, _, self.store = load_streaming(
            path, expert_capacity=0, verbose=False, keep_vision=False,
            use_mtp=False)
        apply_qsa()
        super().__init__(AutoTokenizer.from_pretrained(path))
        self.language = model.language_model
        self.model_path = path
        self.threshold = threshold
        self.resident_experts = resident_experts
        self.pin_budget_gb = pin_budget_gb
        self.tail_experts = tail_experts
        self.tail_warmup = tail_warmup
        self.routing_profile = routing_profile
        self.swap_epsilon = swap_epsilon
        self.fusion_block = fusion_block
        self.fusion_min_margin = fusion_min_margin
        self.fusion_min_block = fusion_min_block
        self.fusion_margin_tokens = fusion_margin_tokens
        self.fusion_max_prompt = fusion_max_prompt
        self.fusion_model = os.path.expanduser(fusion_model)
        self._setting_sources = {}
        self.session_dir = session_dir
        self.thinking_enabled = False
        # Greedy by default so no benchmark can sample by accident. Every
        # comparison here proves a change left the trajectory alone by
        # matching token IDs across arms. The chat sets this from
        # preferences to Qwen's recommended thinking-mode sampler.
        self.sampling = Sampling.greedy_settings()
        # Mirrored from preferences for display, the way `thinking_enabled`
        # is. `/effort`, `/thinking` and `/think-budget` remain the writers,
        # so there is one owner per value and nothing to drift.
        self.reasoning_effort = "medium"
        self.think_budget = 0
        self.answer_budget = 0
        self._store = None
        self._decoder = None
        self._fused_pending = routing_profile == "fused-quality"
        from models.flashnext.routing import RoutingProfile, prewarm_enabled

        self.routing = RoutingProfile(
            routing_profile, self.store, self.language,
            threshold=threshold, resident_experts=resident_experts,
            tail_experts=tail_experts, warmup=tail_warmup,
            pin_budget_gb=pin_budget_gb, swap_epsilon_value=swap_epsilon,
        )
        if routing_profile == "cache-aware":
            self.store.set_residency_tracking(True)
        if prewarm_enabled():
            self.prewarmed_rows = self.routing.prewarm()
        self.cache = self.language.make_cache()
        self.stops = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        } - {None}

    def _settings_text(self) -> str:
        return get_registry().render(self, "flashnext")

    def _rebuild_routing(self) -> None:
        from models.flashnext.routing import RoutingProfile, prewarm_enabled

        if self._decoder is not None:
            self.cache = self._decoder.target_cache
        self._decoder = None
        self._fused_pending = self.routing_profile == "fused-quality" and not self.tape
        self.routing = RoutingProfile(
            self.routing_profile,
            self.store,
            self.language,
            threshold=self.threshold,
            resident_experts=self.resident_experts,
            tail_experts=self.tail_experts,
            warmup=self.tail_warmup,
            pin_budget_gb=self.pin_budget_gb,
            swap_epsilon_value=self.swap_epsilon,
        )
        self.store.set_residency_tracking(
            self.routing_profile == "cache-aware"
            or os.environ.get("FLASHNEXT_TRACK_RESIDENT") == "1"
        )
        self._store = None

    def configure(self, argument: str) -> str:
        return get_registry().configure(self, argument, "flashnext")

    # ---- session persistence -------------------------------------------
    # Exact snapshots: every DeltaNet state, QSA cache and indexer array is
    # written, so a restored session continues without replaying the prompt.
    # The 27B backend persists prompt-cache images instead; both satisfy the
    # same four methods, so the chat never learns the difference.

    def _session_profile(self):
        profile = self.routing.session_profile(self.stops)
        if self.routing_profile == "fused-quality":
            profile.update({
                "fused_quality": True,
                "speculative_fast": True,
                "draft_depth": self.fusion_block,
                "draft_model": os.path.basename(self.fusion_model),
                "fusion_block": self.fusion_block,
                "fusion_min_margin": self.fusion_min_margin,
                "fusion_min_block": self.fusion_min_block,
                "fusion_margin_tokens": self.fusion_margin_tokens,
                "fusion_max_prompt": self.fusion_max_prompt,
            })
        return profile

    def _sessions(self):
        from models.flashnext.sessions import SessionStore

        if getattr(self, "_store", None) is None:
            self._store = SessionStore(
                os.path.expanduser(self.session_dir),
                self.model_path,
                self._session_profile(),
                self.language,
            )
        return self._store

    def save_session(self, name: str) -> str:
        from models.flashnext.sessions import SessionError

        try:
            summary = self._sessions().save(
                name, self.cache, self.tape, not self.tape,
                self.thinking_enabled)
        except (OSError, SessionError, ValueError) as exc:
            return f"could not save session: {exc}"
        return (f"saved {summary.name}  {summary.cached_tokens} tokens, "
                f"thinking={'on' if summary.thinking else 'off'}")

    def load_session(self, name: str) -> str:
        from models.flashnext.sessions import SessionError, SessionStore

        try:
            sessions = self._sessions()
            saved_profile = sessions.saved_profile(name)
            loader = sessions
            if saved_profile != sessions.profile:
                loader = SessionStore(
                    os.path.expanduser(self.session_dir),
                    self.model_path,
                    saved_profile,
                    self.language,
                )
            loaded = loader.load(name)
        except (OSError, SessionError, ValueError) as exc:
            return f"could not load session: {exc}"
        self.cache = loaded.cache
        self.tape = list(loaded.token_ids)
        self.pending = []
        self.turn_closed = True
        self.thinking_enabled = loaded.thinking
        self.language._position_ids = loaded.position_ids
        self.language._rope_deltas = loaded.rope_deltas
        saved_mode = saved_profile.get("mode", "unknown")
        mode_note = (
            "" if saved_mode == self.routing_profile
            else f", saved as {saved_mode}, continuing as {self.routing_profile}"
        )
        return (f"loaded {name}  {len(self.tape)} tokens, "
                f"thinking={'on' if loaded.thinking else 'off'}{mode_note}, "
                "no old prefill")

    def list_sessions(self) -> str:
        from models.flashnext.sessions import SessionError

        try:
            saved = self._sessions().list()
        except (OSError, SessionError, ValueError) as exc:
            return f"could not list sessions: {exc}"
        if not saved:
            return "no saved sessions"
        rows = []
        for item in saved:
            if item.valid:
                rows.append(f"  {item.name:<24} {item.cached_tokens:>7} tok  "
                            f"thinking {'on' if item.thinking else 'off'}")
            else:
                rows.append(f"  {item.name:<24} invalid: {item.error}")
        return "\n".join(rows)

    def delete_session(self, name: str) -> str:
        from models.flashnext.sessions import SessionError

        try:
            removed = self._sessions().delete(name)
        except (OSError, SessionError, ValueError) as exc:
            return f"could not delete session: {exc}"
        return f"{name} {'deleted' if removed else 'not found'}"

    def reset(self) -> None:
        if self._decoder is not None:
            self.cache = self._decoder.target_cache
        self._decoder = None
        self._fused_pending = self.routing_profile == "fused-quality"
        self.cache = self.language.make_cache()
        self.tape = []
        self.pending = []
        self.turn_closed = True
        self.language._position_ids = None
        self.language._rope_deltas = None

    def _start_fused_decoder(self):
        if (
            not self._fused_pending
            or self.tape
            or len(self.pending) > self.fusion_max_prompt
        ):
            self._fused_pending = False
            return None

        from mlx_vlm import load as load_mlx_vlm
        from models.flashnext.speculative import FastDraftGreedy
        from models.flashnext.transient import materialize_model
        AutoTokenizer = _load_transformers_tokenizer()

        if not self.fusion_model:
            raise RuntimeError(
                "fused-quality needs a draft model. Set one with "
                "/settings fusion-model <path>"
            )
        if not os.path.isdir(self.fusion_model):
            raise RuntimeError(f"fused draft model not found: {self.fusion_model}")
        draft_model, draft_processor = load_mlx_vlm(self.fusion_model)
        draft_language = draft_model.language_model
        if int(draft_language.args.vocab_size) != int(self.language.args.vocab_size):
            raise RuntimeError("fused draft and target vocabularies differ")
        draft_tokenizer = AutoTokenizer.from_pretrained(self.fusion_model)
        if draft_tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise RuntimeError("fused draft and target token IDs differ")
        draft_tokenizer = None
        draft_processor = None
        materialize_model(draft_language)
        draft_model = None
        decoder = FastDraftGreedy(
            self.language,
            self.store,
            depth=self.fusion_block,
            draft_language=draft_language,
            fallback_on_reject=True,
            release_draft_before_verify=True,
            draft_min_margin=self.fusion_min_margin,
            draft_min_block=self.fusion_min_block,
            draft_margin_tokens=self.fusion_margin_tokens,
        )
        decoder.target_cache = self.cache
        self._decoder = decoder
        self._fused_pending = False
        return decoder

    @property
    def cache_tokens(self) -> int:
        for entry in self.cache:
            offset = getattr(entry, "offset", None)
            if offset is not None:
                return int(offset)
        return len(self.tape)

    def generate(self, max_tokens: int, out=None, on_prefilled=None,
                 on_prefill_progress=None, on_decode_token=None) -> tuple[str, Stats]:
        """Feed everything pending through the model, then decode a reply.

        `on_prefilled` fires once the prompt is in the cache and before the
        first token is decoded, so a caller can stop a prefill animation
        rather than leaving it running over the answer.
        """
        from models.flashnext.expert_cache import set_prefill_progress
        from models.flashnext.prefill import prefill_language

        if not self.pending:
            return "", Stats(finish="stop")
        decoder = self._decoder
        if decoder is None and self.routing_profile == "fused-quality":
            decoder = self._start_fused_decoder()
        ids = mx.array(self.pending)[None]
        prompt_tokens = int(ids.shape[1])
        self.tape.extend(self.pending)
        self.pending = []

        prefill_began = time.perf_counter()
        self.routing.reset()
        streamed_layers = sum(
            hasattr(
                getattr(getattr(layer, "mlp", None), "switch_mlp", None),
                "layer_id",
            )
            for layer in self.language.model.layers
        )
        completed_layers = set()

        def layer_completed(layer_id):
            completed_layers.add(layer_id)
            if on_prefill_progress is not None and streamed_layers:
                confirmed = max(0, len(completed_layers) - 1)
                done = prompt_tokens * confirmed // streamed_layers
                on_prefill_progress(done, prompt_tokens)

        if on_prefill_progress is not None:
            on_prefill_progress(0, prompt_tokens)
        set_prefill_progress(layer_completed)
        try:
            if decoder is None:
                _, token = prefill_language(
                    self.language, ids, self.cache, sampler=Sampler(self.sampling)
                )
            else:
                decoder.append(ids)
                self.cache = decoder.target_cache
        finally:
            set_prefill_progress(None)
        prefill_seconds = time.perf_counter() - prefill_began
        if on_prefill_progress is not None:
            on_prefill_progress(prompt_tokens, prompt_tokens)
        if on_prefilled is not None:
            on_prefilled()
        produced, partial, pieces = [], [], []
        tail_began = None
        tail_index = 0
        timer = DecodeTimer()
        finish = "length"
        self.routing.begin_decode()
        if decoder is not None:
            decoder.set_route_observer(self.routing._observe)

        sampler = Sampler(self.sampling)

        def standard_tokens():
            nonlocal token
            for _index in range(max_tokens):
                value = int(token.item())
                if value in self.stops:
                    return
                yield value
                sampler.observe(value)
                step = self.language(token[None], cache=self.cache)
                token = sampler(step.logits[:, -1, :])
                mx.eval(token)

        tokens = (
            standard_tokens()
            if decoder is None
            else decoder.generate(max_tokens, self.stops)
        )
        try:
            for index, value in enumerate(tokens):
                produced.append(value)
                self.tape.append(value)
                if self.routing.after_token(index + 1, max_tokens):
                    tail_began = timer.mark()
                    # Token warmup+1 was generated before pinning. The next
                    # model call produces the first pinned-tail output.
                    tail_index = len(produced)
                piece = stream_decode(self.tokenizer, partial, value)
                if on_decode_token is not None:
                    on_decode_token(value, piece)
                if not piece:
                    continue
                pieces.append(piece)
                if out is not None:
                    with timer.emitting():
                        out(piece)
            if len(produced) < max_tokens:
                finish = "stop"
                self.turn_closed = True
            else:
                self.turn_closed = False
        finally:
            if decoder is not None:
                decoder.set_route_observer(None)
                self.cache = decoder.target_cache
            self.routing.finish_decode()

        tail_tokens = len(produced) - tail_index if tail_began is not None else 0
        tail_seconds = timer.since(tail_began) if tail_began is not None else 0.0
        return "".join(pieces), Stats(
            finish=finish, tokens=len(produced), seconds=timer.elapsed(),
            prompt_tokens=prompt_tokens, prefill_seconds=prefill_seconds,
            tail_tokens=tail_tokens, tail_seconds=tail_seconds,
            ui_seconds=timer.emitted,
            pinned_bytes=self.routing.pinned_bytes,
            pinned_signature=self.routing.pinned_signature,
        )
