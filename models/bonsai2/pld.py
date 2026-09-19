"""Prompt Lookup Decoding for Bonsai-2 (D8).

Drafts continuations from context n-grams without a draft model, then
verifies them with target forwards. No MLX calls here: find_draft is pure
and unit-tested. The backend wires it into generation behind an explicit
opt-in flag. Speculative tables stay separate from base-runtime tables.
"""
from __future__ import annotations


def find_draft(
    tape: list[int], ngram: int = 4, max_draft: int = 8
) -> list[int]:
    """Propose a draft continuation from context matches.

    Takes the last `ngram` tape tokens, finds their most recent earlier
    occurrence, and returns up to `max_draft` following tokens. Returns []
    when nothing matches. The tape itself is never modified.
    """
    if ngram < 1 or max_draft < 1 or len(tape) < ngram + 1:
        return []
    needle = tuple(tape[-ngram:])
    context = tape[:-1]
    for start in range(len(context) - ngram, -1, -1):
        if tuple(context[start : start + ngram]) == needle:
            following = context[start + ngram :]
            return list(following[:max_draft])
    return []


def accepted_prefix(draft: list[int], verified: list[int]) -> list[int]:
    """Longest draft prefix matching the verifier output."""
    accepted = []
    for proposed, actual in zip(draft, verified):
        if proposed != actual:
            break
        accepted.append(proposed)
    return accepted
