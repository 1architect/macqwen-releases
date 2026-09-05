"""Exact greedy Lightning MTP decoding for the streamed Qwen4 model."""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx

from . import mtp
from .qsa_chunk import QSA_CHUNK_THRESHOLD

# A large prefill leaves several GB in the MLX allocator; holding it through
# decode starves the page cache the expert bank depends on. Target prefills
# use the shared helper below. Draft-specific releases remain local.
from .prefill import prefill_target
from .routing import DEFAULT_READ_MODE


def _copy_state(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def snapshot_cache(cache):
    return [
        (_copy_state(entry.state), entry.meta_state)
        for entry in cache
    ]


def restore_cache(cache, snapshot) -> None:
    for entry, (state, meta_state) in zip(cache, snapshot):
        entry.state = _copy_state(state)
        entry.meta_state = meta_state


def trim_cache(cache, offset: int) -> None:
    for entry in cache:
        current = int(getattr(entry, "offset", 0))
        if current > offset:
            entry.trim(current - offset)


@dataclass
class MTPStats:
    cycles: int = 0
    drafted: int = 0
    accepted: int = 0
    replayed: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


@dataclass
class FastDraftStats:
    cycles: int = 0
    drafted: int = 0
    accepted: int = 0
    replayed: int = 0
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    rollback_seconds: float = 0.0
    release_seconds: float = 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


class FastDraftGreedy:
    """Use fast routing for drafts and exact routing for every committed token."""

    def __init__(
        self,
        language,
        store,
        depth: int = 4,
        draft_language=None,
        fallback_on_reject: bool = False,
        release_draft_before_verify: bool = False,
        draft_min_margin: float | None = None,
        draft_min_block: int = 1,
        draft_margin_tokens: int | None = None,
    ):
        from mlx_vlm.models.qwen3_5 import language as qwen35_language

        from .qwen4_verifier import Qwen4ExactSpeculativeVerifier

        self.language = language
        self.draft_language = draft_language or language
        self.external_draft = draft_language is not None
        self.fallback_on_reject = bool(fallback_on_reject)
        self.release_draft_before_verify = bool(
            release_draft_before_verify
        )
        self.depth = max(1, int(depth))
        self.draft_min_block = max(1, int(draft_min_block))
        self.draft_margin_tokens = (
            self.depth
            if draft_margin_tokens is None
            else max(0, int(draft_margin_tokens))
        )
        self.draft_disabled = False
        self.store = store
        self.target_cache = language.make_cache()
        self.draft_cache = self.draft_language.make_cache()
        self.next_main = None
        self.draft_next = None
        self.route_observer = None
        self.stats = FastDraftStats()
        self.draft_min_margin = (
            float(os.environ.get("FLASHNEXT_DRAFT_MIN_MARGIN", "0"))
            if draft_min_margin is None
            else float(draft_min_margin)
        )
        self.draft_fused_argmax = (
            os.environ.get("FLASHNEXT_DRAFT_FUSED_ARGMAX", "0") == "1"
        )
        self.verifier = Qwen4ExactSpeculativeVerifier()
        qwen35_language._EXACT_SPECULATIVE_VERIFIER = self.verifier

    def reset(self) -> None:
        self.target_cache = self.language.make_cache()
        self.draft_cache = (
            self.draft_language.make_cache()
            if self.draft_language is not None
            else None
        )
        self.next_main = None
        self.draft_next = None
        self.route_observer = None
        self.stats = FastDraftStats()
        self.draft_disabled = self.external_draft and self.draft_language is None
        if self.external_draft and self.draft_language is not None:
            self.draft_language._position_ids = None
            self.draft_language._rope_deltas = None

    def load_cache(self, cache, token_ids=None) -> None:
        self.target_cache = cache
        self.draft_cache = (
            self.draft_language.make_cache()
            if self.draft_language is not None
            else None
        )
        self.next_main = None
        self.draft_next = None
        self.draft_disabled = self.external_draft and self.draft_language is None
        if self.external_draft and self.draft_language is not None:
            self.draft_language._position_ids = None
            self.draft_language._rope_deltas = None
            if token_ids:
                ids = mx.array(token_ids, dtype=mx.uint32).reshape(1, -1)
                out = self.draft_language(ids, cache=self.draft_cache)
                self.draft_next = mx.argmax(
                    out.logits[:, -1, :], axis=-1
                ).astype(mx.uint32)
                mx.eval(self.draft_next)

    def _release_transient_draft(self) -> None:
        """Release a one-shot external draft exactly once."""
        if (
            self.draft_disabled
            and self.draft_language is None
            and self.draft_cache is None
            and self.draft_next is None
        ):
            return
        release_started = time.perf_counter()
        self.draft_next = None
        self.draft_cache = None
        self.draft_language = None
        self.draft_disabled = True
        gc.collect()
        mx.clear_cache()
        self.stats.release_seconds += time.perf_counter() - release_started

    def set_route_observer(self, observer) -> None:
        self.route_observer = observer

    def _exact_profile(self) -> None:
        from .adaptive_topk import (
            set_layer_thresholds,
            set_renorm_blend,
            set_resident_experts,
            set_route_observer,
            set_threshold,
        )

        # Take the store's configured default, exactly as RoutingProfile
        # does. Hardcoding "pread" here made FLASHNEXT_READ silently useless
        # on the fused path.
        self.store._read_mode = DEFAULT_READ_MODE
        set_resident_experts(None)
        set_route_observer(self.route_observer)
        set_threshold(0.85)
        set_layer_thresholds({})
        set_renorm_blend(1.0)

    def _fast_profile(self) -> None:
        from .adaptive_topk import (
            FAST_LAYERS,
            set_layer_thresholds,
            set_renorm_blend,
            set_resident_experts,
            set_route_observer,
            set_threshold,
        )

        self.store._read_mode = "shared_mmap"
        set_resident_experts(None)
        set_route_observer(None)
        set_threshold(0.20)
        set_layer_thresholds({layer: 0.40 for layer in FAST_LAYERS})
        set_renorm_blend(0.0)

    def append(self, ids) -> None:
        """Append prompt tokens with the exact target profile."""
        long_prefill = int(ids.shape[1]) > QSA_CHUNK_THRESHOLD
        if self.external_draft and not self.draft_disabled and long_prefill:
            self._release_transient_draft()
        self._exact_profile()
        _logits, _hidden, self.next_main = prefill_target(
            self.language, ids, self.target_cache
        )
        if self.external_draft and not self.draft_disabled:
            draft_out = self.draft_language(ids, cache=self.draft_cache)
            self.draft_next = mx.argmax(
                draft_out.logits[:, -1, :], axis=-1
            ).astype(mx.uint32)
            mx.eval(self.draft_next)

    def _replay(self, snapshot, values):
        restore_cache(self.target_cache, snapshot)
        if not values:
            return None
        ids = mx.array(values, dtype=mx.uint32).reshape(1, -1)
        out = self.language(
            ids,
            cache=self.target_cache,
            speculative_verify=True,
            return_hidden=True,
            skip_logits=True,
        )
        token = self.language.speculative_argmax_from_hidden(
            out.hidden_states[-1][:, -1:]
        ).reshape(-1).astype(mx.uint32)
        mx.eval(token)
        self.stats.replayed += len(values)
        return token

    def _rollback_verified(self, snapshot, verify_out, block_size, keep_count):
        """Keep an exact prefix of a verified block without replaying it."""
        if keep_count <= 0:
            restore_cache(self.target_cache, snapshot)
            return
        self.language.rollback_speculative_cache(
            self.target_cache,
            verify_out.gdn_states,
            keep_count - 1,
            block_size,
        )
        ssm_caches = [
            entry
            for entry in self.target_cache
            if entry is not None
            and not entry.is_trimmable()
            and not hasattr(entry, "zero_row_tail")
        ]
        restored = []
        for entry, state in zip(ssm_caches, verify_out.gdn_states):
            if len(state) <= 12 or state[12] is None:
                continue
            token_history, context_len, conv_input, conv_state_len = state[12]
            entry[3] = mx.contiguous(
                token_history[:, keep_count : keep_count + context_len]
            )
            entry[2] = mx.contiguous(
                conv_input[:, keep_count : keep_count + conv_state_len]
            )
            restored.extend((entry[2], entry[3]))
        if restored:
            mx.eval(*restored)

    def _external_replay(self, snapshot, next_snapshot, values):
        restore_cache(self.draft_cache, snapshot)
        if not values:
            self.draft_next = next_snapshot
            return
        ids = mx.array(values, dtype=mx.uint32).reshape(1, -1)
        out = self.draft_language(ids, cache=self.draft_cache)
        self.draft_next = mx.argmax(
            out.logits[:, -1, :], axis=-1
        ).astype(mx.uint32)
        mx.eval(self.draft_next)

    def _align_external_full_block(self, last_value: int) -> None:
        token = mx.array([[last_value]], dtype=mx.uint32)
        out = self.draft_language(token, cache=self.draft_cache)
        self.draft_next = mx.argmax(
            out.logits[:, -1, :], axis=-1
        ).astype(mx.uint32)
        mx.eval(self.draft_next)

    def _generate_exact_tail(
        self,
        max_tokens: int,
        stops: set[int],
        produced: int,
    ):
        while produced < max_tokens and self.next_main is not None:
            value = int(self.next_main.item())
            if value in stops:
                self.next_main = None
                return
            yield value
            token = mx.array([[value]], dtype=mx.uint32)
            output = self.language(token, cache=self.target_cache)
            self.next_main = mx.argmax(
                output.logits[:, -1, :], axis=-1
            ).astype(mx.uint32)
            mx.eval(self.next_main)
            produced += 1

    def _generate_external(self, max_tokens: int, stops: set[int]):
        produced = 0
        while (
            produced < max_tokens
            and self.next_main is not None
            and self.draft_next is not None
        ):
            first = int(self.next_main.item())
            if first in stops:
                self.next_main = None
                return
            target_snapshot = snapshot_cache(self.target_cache)
            draft_snapshot = snapshot_cache(self.draft_cache)
            draft_next_snapshot = self.draft_next

            # Anchor every target block with the exact token already known.
            # The draft predicts only later positions. A rejected draft then
            # becomes the next anchor instead of requiring a one-token replay.
            candidates = [first]
            draft_margins = []
            draft_cache_has_last = False
            token = self.next_main
            room = min(self.depth, max_tokens - produced)
            draft_started = time.perf_counter()
            trace_draft = os.environ.get("FLASHNEXT_SPEC_TRACE") == "1"
            draft_token_arrays = []
            for draft_index in range(1, room):
                check_margin = (
                    self.draft_min_margin > 0
                    and draft_index <= self.draft_margin_tokens
                )
                defer_draft_sync = not check_margin and not trace_draft
                input_token = token
                out = None
                logits = None
                sampled = None
                if (
                    self.draft_fused_argmax
                    and not check_margin
                    and not trace_draft
                ):
                    sampled = self.draft_language.fused_greedy_decode(
                        input_token.reshape(1, 1),
                        cache=self.draft_cache,
                    )
                if sampled is None:
                    out = self.draft_language(
                        input_token.reshape(1, 1), cache=self.draft_cache
                    )
                    logits = out.logits[:, -1, :]
                    token = mx.argmax(logits, axis=-1).astype(mx.uint32)
                else:
                    token = sampled.reshape(-1).astype(mx.uint32)
                margin = None
                if defer_draft_sync:
                    mx.async_eval(
                        token,
                        [entry.state for entry in self.draft_cache],
                    )
                    draft_token_arrays.append(token)
                else:
                    top = mx.topk(logits, k=2, axis=-1)
                    mx.eval(token, top)
                    pair = sorted(float(value) for value in top[0].tolist())
                    margin = pair[1] - pair[0]
                    draft_margins.append(margin)
                if (
                    margin is not None
                    and self.draft_min_margin > 0
                    and margin < self.draft_min_margin
                ):
                    draft_cache_has_last = True
                    break
                if not defer_draft_sync:
                    candidates.append(int(token.item()))
            if draft_token_arrays:
                mx.eval(
                    draft_token_arrays,
                    [entry.state for entry in self.draft_cache],
                )
                candidates.extend(
                    int(value.item()) for value in draft_token_arrays
                )
            self.stats.draft_seconds += time.perf_counter() - draft_started

            if self.release_draft_before_verify:
                out = None
                logits = None
                token = None
                input_token = None
                sampled = None
                top = None
                draft_snapshot = None
                draft_next_snapshot = None
                draft_token_arrays = None
                self._release_transient_draft()

            self.stats.cycles += 1
            self.stats.drafted += len(candidates) - 1
            if (
                self.release_draft_before_verify
                and len(candidates) < self.draft_min_block
            ):
                target_snapshot = None
                draft_snapshot = None
                draft_next_snapshot = None
                gc.collect()
                mx.clear_cache()
                self._exact_profile()
                yield from self._generate_exact_tail(
                    max_tokens,
                    stops,
                    produced,
                )
                return
            block_ids = mx.array(candidates, dtype=mx.uint32).reshape(1, -1)
            self._exact_profile()
            verify_started = time.perf_counter()
            out = self.language(
                block_ids,
                cache=self.target_cache,
                speculative_verify=True,
                return_hidden=True,
                skip_logits=True,
            )
            target_ids = self.language.speculative_argmax_from_hidden(
                out.hidden_states[-1]
            )[0].astype(mx.uint32)
            mx.eval(target_ids)
            self.stats.verify_seconds += time.perf_counter() - verify_started
            host_targets = [int(value) for value in target_ids.tolist()]

            accepted = 0
            for index, proposed in enumerate(candidates[1:]):
                if host_targets[index] != proposed:
                    break
                accepted += 1
            self.stats.accepted += accepted
            all_accepted = accepted == len(candidates) - 1
            if all_accepted:
                committed = candidates
                following = mx.array([host_targets[-1]], dtype=mx.uint32)
            else:
                correction = host_targets[accepted]
                committed = candidates[: accepted + 1]
                following = mx.array([correction], dtype=mx.uint32)
            if os.environ.get("FLASHNEXT_SPEC_TRACE") == "1":
                print(
                    f"spec block={candidates} target={host_targets} "
                    f"margins={[round(x, 3) for x in draft_margins]} "
                    f"accepted={accepted} committed={committed}",
                    flush=True,
                )

            stop_at = next(
                (index for index, value in enumerate(committed) if value in stops),
                None,
            )
            remaining = max_tokens - produced
            keep = len(committed) if stop_at is None else stop_at
            keep = min(keep, remaining)
            finishing = (
                stop_at is not None or keep < len(committed) or keep == remaining
            )

            verified_prefix = accepted + 1
            full_block_committed = all_accepted and keep == len(candidates)
            if full_block_committed:
                if not self.draft_disabled and not draft_cache_has_last:
                    self._align_external_full_block(candidates[-1])
            else:
                kept_verified = min(keep, verified_prefix)
                rollback_started = time.perf_counter()
                self._rollback_verified(
                    target_snapshot,
                    out,
                    len(candidates),
                    kept_verified,
                )
                self.stats.rollback_seconds += (
                    time.perf_counter() - rollback_started
                )
                if not self.draft_disabled and not (
                    self.fallback_on_reject and not all_accepted
                ):
                    self._external_replay(
                        draft_snapshot,
                        draft_next_snapshot,
                        committed[:kept_verified],
                    )
            if self.draft_disabled:
                mx.eval([entry.state for entry in self.target_cache])
                out = None
                target_ids = None
                host_targets = None
                block_ids = None
                target_snapshot = None
                draft_snapshot = None
                draft_next_snapshot = None
                draft_margins = None
                gc.collect()
                mx.clear_cache()
            if finishing:
                self.next_main = None
                for value in committed[:keep]:
                    produced += 1
                    yield value
                return

            self.next_main = following
            mx.eval(self.next_main)
            for value in committed:
                produced += 1
                yield value
            if self.draft_disabled or (
                self.fallback_on_reject and not all_accepted
            ):
                self.draft_disabled = True
                yield from self._generate_exact_tail(
                    max_tokens,
                    stops,
                    produced,
                )
                return

    def _draft(self, snapshot):
        restore_cache(self.draft_cache, snapshot)
        self._fast_profile()
        token = self.next_main
        drafts = []
        for _ in range(self.depth):
            out = self.language(token.reshape(1, 1), cache=self.draft_cache)
            token = mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.uint32)
            mx.eval(token)
            drafts.append(int(token.item()))
        return drafts

    def generate(self, max_tokens: int, stops: Iterable[int]):
        """Yield the same greedy tokens as exact routing."""
        stops = set(stops)
        if self.external_draft:
            try:
                if self.draft_disabled:
                    yield from self._generate_exact_tail(max_tokens, stops, 0)
                else:
                    yield from self._generate_external(max_tokens, stops)
            finally:
                if self.release_draft_before_verify:
                    self._release_transient_draft()
                self._exact_profile()
                self.route_observer = None
            return
        produced = 0
        try:
            while produced < max_tokens and self.next_main is not None:
                first = int(self.next_main.item())
                if first in stops:
                    self.next_main = None
                    return

                snapshot = snapshot_cache(self.target_cache)
                drafts = self._draft(snapshot)
                self.stats.cycles += 1
                self.stats.drafted += len(drafts)

                block = [first, *drafts]
                block_ids = mx.array(block, dtype=mx.uint32).reshape(1, -1)
                self._exact_profile()
                out = self.language(
                    block_ids,
                    cache=self.target_cache,
                    speculative_verify=True,
                    return_hidden=True,
                    skip_logits=True,
                )
                target_ids = self.language.speculative_argmax_from_hidden(
                    out.hidden_states[-1]
                )[0].astype(mx.uint32)
                mx.eval(target_ids)
                host_targets = target_ids.tolist()

                accepted = 0
                for expected, proposed in zip(host_targets, drafts):
                    if int(expected) != proposed:
                        break
                    accepted += 1
                self.stats.accepted += accepted
                exact = block[: accepted + 1]

                stop_at = next(
                    (index for index, value in enumerate(exact) if value in stops),
                    None,
                )
                room = max_tokens - produced
                keep = len(exact) if stop_at is None else stop_at
                keep = min(keep, room)
                finishing = stop_at is not None or keep < len(exact) or keep == room

                if finishing:
                    block_is_committed = (
                        accepted == len(drafts) and keep == len(block)
                    )
                    if not block_is_committed:
                        self._replay(snapshot, exact[:keep])
                    self.next_main = None
                    for value in exact[:keep]:
                        produced += 1
                        yield value
                    return

                if accepted < len(drafts):
                    self._replay(snapshot, exact)
                    self.next_main = mx.array(
                        [int(host_targets[accepted])], dtype=mx.uint32
                    )
                else:
                    self.next_main = mx.array(
                        [int(host_targets[-1])], dtype=mx.uint32
                    )
                mx.eval(self.next_main)

                for value in exact:
                    produced += 1
                    yield value
        finally:
            self._exact_profile()
            self.route_observer = None


class MTPGreedy:
    """Maintain exact target and MTP caches across appended chat turns."""

    def __init__(self, language, depth: int = 4):
        self.language = language
        self.depth = max(1, int(depth))
        self.target_cache = language.make_cache()
        self.mtp_cache = mtp.make_cache(language)
        self.anchor = None
        self.next_main = None
        self.draft_logits = None
        self.draft_hidden = None
        self.hist_offset = 0
        self.stats = MTPStats()

    def reset(self) -> None:
        self.target_cache = self.language.make_cache()
        self.mtp_cache = mtp.make_cache(self.language)
        self.anchor = None
        self.next_main = None
        self.draft_logits = None
        self.draft_hidden = None
        self.hist_offset = 0
        self.stats = MTPStats()

    def _target_capture(self, ids):
        logits, hidden_states, _token = prefill_target(
            self.language,
            ids,
            self.target_cache,
            want_logits=True,
            want_hidden=True,
        )
        hidden = hidden_states[0]
        expected = self.language.args.hidden_size * self.language.args.hc_count
        if hidden.shape[-1] != expected:
            raise RuntimeError(
                f"MTP hidden width is {hidden.shape[-1]}, expected {expected}"
            )
        return logits, hidden

    def _target_replay(self, snapshot, ids) -> None:
        restore_cache(self.target_cache, snapshot)
        if ids.shape[1]:
            out = self.language(ids, cache=self.target_cache)
            mx.eval(out.logits)
            self.stats.replayed += int(ids.shape[1])

    def _fold(self, hidden_rows, tokens):
        logits, head_hidden = mtp.forward(
            self.language,
            hidden_rows,
            tokens,
            self.mtp_cache,
            return_hidden=True,
        )
        self.hist_offset += int(tokens.shape[1])
        return logits, head_hidden

    def append(self, ids) -> None:
        """Append prompt tokens and seed the next MTP verify cycle."""
        logits, hidden = self._target_capture(ids)
        mx.eval(logits, hidden)

        if self.anchor is None:
            pair_hidden = hidden[:, :-1]
            pair_tokens = ids[:, 1:]
        else:
            pair_hidden = mx.concatenate([self.anchor, hidden[:, :-1]], axis=1)
            pair_tokens = ids
        if pair_tokens.shape[1]:
            folded, _ = self._fold(pair_hidden, pair_tokens)
            mx.eval(folded)

        self.anchor = hidden[:, -1:]
        self.next_main = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(self.next_main)
        self.draft_logits, self.draft_hidden = self._fold(
            self.anchor,
            self.next_main.reshape(1, 1),
        )
        mx.eval(self.draft_logits, self.draft_hidden)

    def _draft_chain(self):
        drafts = []
        logits = self.draft_logits
        hidden = self.draft_hidden[:, -1:]
        for index in range(self.depth):
            token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
            value = int(token.item())
            drafts.append(value)
            if index + 1 < self.depth:
                logits, hidden = mtp.forward(
                    self.language,
                    hidden,
                    token.reshape(1, 1),
                    self.mtp_cache,
                    return_hidden=True,
                )
        return drafts

    def _finish_prefix(self, snapshot, verify_hidden, keep_ids, base_hist) -> None:
        ids = mx.array(keep_ids, dtype=mx.uint32).reshape(1, -1)
        self._target_replay(snapshot, ids)
        trim_cache(self.mtp_cache, base_hist)

        # next_main already occupies the last committed MTP history slot.
        extra = keep_ids[1:]
        if extra:
            hidden_rows = verify_hidden[:, : len(extra)]
            tokens = mx.array(extra, dtype=mx.uint32).reshape(1, -1)
            folded, _ = self._fold(hidden_rows, tokens)
            mx.eval(folded)
        self.anchor = verify_hidden[:, len(keep_ids) - 1 : len(keep_ids)]
        self.next_main = None

    def generate(self, max_tokens: int, stops: Iterable[int]):
        """Yield exact greedy tokens. The target cache always trails no output."""
        stops = set(stops)
        produced = 0
        while produced < max_tokens and self.next_main is not None:
            if int(self.next_main.item()) in stops:
                trim_cache(self.mtp_cache, self.hist_offset - 1)
                self.hist_offset -= 1
                self.next_main = None
                return

            target_snapshot = snapshot_cache(self.target_cache)
            base_hist = self.hist_offset
            drafts = self._draft_chain()
            self.stats.cycles += 1
            self.stats.drafted += len(drafts)

            block = [int(self.next_main.item()), *drafts]
            block_ids = mx.array(block, dtype=mx.uint32).reshape(1, -1)
            logits, hidden = self._target_capture(block_ids)
            target_ids = mx.argmax(logits[0], axis=-1).astype(mx.uint32)
            host_targets = target_ids.tolist()

            accepted = 0
            for expected, proposed in zip(host_targets, drafts):
                if int(expected) != proposed:
                    break
                accepted += 1
            self.stats.accepted += accepted

            processed = block[: accepted + 1]
            stop_at = next(
                (i for i, token in enumerate(processed) if token in stops),
                None,
            )
            room = max_tokens - produced
            keep = len(processed) if stop_at is None else stop_at
            keep = min(keep, room)
            finishing = stop_at is not None or keep < len(processed) or keep == room

            if finishing:
                if keep:
                    self._finish_prefix(
                        target_snapshot,
                        hidden,
                        processed[:keep],
                        base_hist,
                    )
                    for token in processed[:keep]:
                        produced += 1
                        yield token
                else:
                    restore_cache(self.target_cache, target_snapshot)
                    trim_cache(self.mtp_cache, base_hist - 1)
                    self.hist_offset = base_hist - 1
                    self.next_main = None
                return

            if accepted < len(drafts):
                self._target_replay(
                    target_snapshot, block_ids[:, : accepted + 1]
                )
            trim_cache(self.mtp_cache, base_hist)

            correction = int(host_targets[accepted])
            committed = [*drafts[:accepted], correction]
            fold_hidden = hidden[:, : accepted + 1]
            fold_tokens = mx.array(committed, dtype=mx.uint32).reshape(1, -1)
            self.draft_logits, self.draft_hidden = self._fold(
                fold_hidden,
                fold_tokens,
            )
            self.anchor = hidden[:, accepted : accepted + 1]
            self.next_main = mx.array([correction], dtype=mx.uint32)
            mx.eval(self.draft_logits, self.draft_hidden, self.next_main)

            for token in processed:
                produced += 1
                yield token
