"""Exact greedy Lightning MTP decoding for the streamed Qwen4 model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx

from . import mtp


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
        out = self.language(
            ids,
            cache=self.target_cache,
            capture_layer_ids=[],
            return_hidden=True,
        )
        hidden = out.hidden_states[0]
        expected = self.language.args.hidden_size * self.language.args.hc_count
        if hidden.shape[-1] != expected:
            raise RuntimeError(
                f"MTP hidden width is {hidden.shape[-1]}, expected {expected}"
            )
        return out.logits, hidden

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
                self._target_replay(target_snapshot, block_ids[:, : accepted + 1])
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
