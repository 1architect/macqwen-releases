"""Choosing the next token.

We decoded with `argmax`, which is temperature 0 and none of the rest. Qwen's
model card recommends thinking mode at `temperature=1.0`, `top_p=0.95`,
`top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and instruct mode at
`temperature=0.7`, `top_p=0.80`, `presence_penalty=1.5`.

Greedy isn't an oversight in the benchmarks. They prove a change left the
trajectory alone by comparing token IDs across arms, and sampling makes that
impossible, so greedy stays for measurement and the recommended sampler serves
the chat.

The card also gives the remedy for the problem that sent us here:

    you can adjust the presence_penalty parameter between 0 and 2 to reduce
    endless repetition. However, using a higher value may occasionally result
    in language mixing and a slight decrease in model performance.

Greedy breaks a tie the same way every time, so a model that lands between two
near-equal continuations can stay there. We saw it on a SketchUp Ruby task: it
asked whether one API existed about forty times in identical words, on two
routing profiles and two effort levels.
"""
from __future__ import annotations

from dataclasses import dataclass

# Qwen's recommended thinking-mode values, from the model card's Best Practices.
THINKING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
}
# Instruct mode. Kept for reference; the chat runs thinking mode.
INSTRUCT = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}


@dataclass(frozen=True)
class Sampling:
    """One decode's sampling settings. Temperature 0 means greedy."""

    temperature: float = THINKING["temperature"]
    top_p: float = THINKING["top_p"]
    top_k: int = THINKING["top_k"]
    min_p: float = THINKING["min_p"]
    presence_penalty: float = THINKING["presence_penalty"]

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @classmethod
    def greedy_settings(cls) -> "Sampling":
        return cls(temperature=0.0)

    @classmethod
    def from_preferences(cls, values: dict) -> "Sampling":
        return cls(
            temperature=float(values.get("temperature", THINKING["temperature"])),
            top_p=float(values.get("top_p", THINKING["top_p"])),
            top_k=int(values.get("top_k", THINKING["top_k"])),
            min_p=float(values.get("min_p", THINKING["min_p"])),
            presence_penalty=float(
                values.get("presence_penalty", THINKING["presence_penalty"])
            ),
        )

    def describe(self) -> str:
        if self.greedy:
            return "greedy"
        return (
            f"temp {self.temperature:g}  top-p {self.top_p:g}  "
            f"top-k {self.top_k}  min-p {self.min_p:g}  "
            f"presence {self.presence_penalty:g}"
        )


class Sampler:
    """Turn one row of logits into one token id.

    Keeps the ids produced so far, since `presence_penalty` applies to tokens
    already in the reply. Call `reset()` between generations.
    """

    def __init__(self, settings: Sampling | None = None):
        self.settings = settings or Sampling()
        self._seen: set[int] = set()

    def reset(self) -> None:
        self._seen.clear()

    def observe(self, token: int) -> None:
        if self.settings.presence_penalty:
            self._seen.add(int(token))

    def __call__(self, logits):
        """`logits` is the final row, shape (1, vocab) or (vocab,)."""
        import mlx.core as mx

        s = self.settings
        row = logits.reshape(-1)
        if s.greedy:
            # Shape (1,), not a scalar: the decode loop feeds `token[None]`
            # to the model and needs (1, 1).
            return mx.argmax(row, axis=-1).reshape(1).astype(mx.uint32)

        if s.presence_penalty and self._seen:
            seen = mx.array(sorted(self._seen), dtype=mx.int32)
            hits = mx.zeros(row.shape[0], dtype=row.dtype)
            hits = hits.at[seen].add(
                mx.full((seen.shape[0],), s.presence_penalty, dtype=row.dtype)
            )
            row = row - hits

        row = row.astype(mx.float32) / s.temperature

        if s.top_k and 0 < s.top_k < row.shape[0]:
            kept = mx.argpartition(row, kth=-s.top_k, axis=-1)[-s.top_k:]
            floor = mx.min(mx.take(row, kept, axis=-1))
            row = mx.where(row < floor, -mx.inf, row)

        probabilities = mx.softmax(row, axis=-1)

        if s.min_p > 0.0:
            floor = s.min_p * mx.max(probabilities)
            row = mx.where(probabilities < floor, -mx.inf, row)
            probabilities = mx.softmax(row, axis=-1)

        if 0.0 < s.top_p < 1.0:
            order = mx.argsort(-probabilities, axis=-1)
            ordered = mx.take(probabilities, order, axis=-1)
            carried = mx.cumsum(ordered, axis=-1)
            # Keep the first token whose cumulative mass crosses top_p, so the
            # nucleus is never empty when one token already exceeds it.
            keep = carried - ordered < s.top_p
            allowed = mx.zeros(row.shape[0], dtype=mx.bool_)
            allowed = allowed.at[order].add(keep)
            row = mx.where(allowed, row, -mx.inf)

        token = mx.random.categorical(row)
        return token.reshape(1).astype(mx.uint32)
