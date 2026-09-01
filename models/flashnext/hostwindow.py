"""Find host work that runs while both the drive and the GPU are idle.

Every rejected overlap experiment moved SSD DMA next to GPU work and lost. The
mechanism is unified-memory contention and it is a hardware property. Small
host bookkeeping is a different case: it starts no storage traffic, so moving
it cannot invoke that mechanism. The early-submit control is the evidence.
Adding `mx.eval(scores, inds)`, a `tolist()` and host list work measured 498.5
ms/token against a 499.3 ms/token baseline, which is free.

Before moving anything, this measures how much host time actually runs with
both devices idle. That interval is dead time on both units, so removing it
cannot be absorbed the way the read-path savings were.

How each device is judged idle:

  drive   Measured. `reads_begin` and `reads_done` bracket every submission to
          the expert read pool, so `pending()` is the count of reads in
          flight. A window is exclusive only when nothing is pending at either
          end and no submission happened inside it.

  GPU     Structural. MLX runs one default stream and `mx.eval` drains it, so
          no kernel executes between an `mx.eval` returning and the next eval
          call. Building a lazy graph runs no work. Every window here opens
          right after an eval returns and closes before the next one, so the
          GPU is idle throughout by construction. Where a region can force its
          own eval, the call site calls `note_eval()` and the enclosing window
          is counted as shared rather than free.

Off unless `FLASHNEXT_HOST_WINDOW=1`. With the flag off, `window` is bound to
a function returning one shared `nullcontext`, so the decode path pays a call
and nothing else.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager, nullcontext

ENABLED = os.environ.get("FLASHNEXT_HOST_WINDOW") == "1"

_LOCK = threading.Lock()
_PENDING = [0]
_SUBMITS = [0]
_EVALS = [0]
# name -> [exclusive seconds, shared seconds, calls, exclusive calls]
_TOTALS: dict[str, list] = {}


def reads_begin(count: int = 1) -> None:
    """One or more expert reads have been handed to the pool."""
    with _LOCK:
        _PENDING[0] += count
        _SUBMITS[0] += count


def reads_done(count: int = 1) -> None:
    """A read has landed. Safe to call from the pool's worker threads."""
    with _LOCK:
        _PENDING[0] -= count


def track(future):
    """Count one pool future from submission until it completes."""
    reads_begin()
    future.add_done_callback(lambda _: reads_done())
    return future


def pending() -> int:
    with _LOCK:
        return _PENDING[0]


def note_eval() -> None:
    """Record that the caller drove the GPU, so enclosing windows are shared."""
    _EVALS[0] += 1


@contextmanager
def _window(name: str):
    """Time one host region and say whether it had both devices to itself."""
    with _LOCK:
        pending_in, submits_in = _PENDING[0], _SUBMITS[0]
    evals_in = _EVALS[0]
    began = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - began
        with _LOCK:
            pending_out, submits_out = _PENDING[0], _SUBMITS[0]
        exclusive = (
            pending_in == 0
            and pending_out == 0
            and submits_out == submits_in
            and _EVALS[0] == evals_in
        )
        slot = _TOTALS.get(name)
        if slot is None:
            slot = _TOTALS[name] = [0.0, 0.0, 0, 0]
        slot[0 if exclusive else 1] += elapsed
        slot[2] += 1
        slot[3] += 1 if exclusive else 0


# Bound once at import. The decode path enters five of these per layer, 240
# times a token, and building a generator for each costs real time on a path
# that is about to be timed. When the flag is off this is one call returning a
# shared nullcontext, which the interpreter handles without allocating.
_NULL = nullcontext()


def _no_window(name: str):
    return _NULL


window = _window if ENABLED else _no_window


def reset() -> None:
    _TOTALS.clear()
    _EVALS[0] = 0
    with _LOCK:
        _SUBMITS[0] = 0


def totals() -> dict:
    """Copy of the accumulated windows, in seconds."""
    return {name: list(slot) for name, slot in _TOTALS.items()}


def report(tokens: int) -> str:
    """One table of exclusive against shared host time per token."""
    lines = [
        f"  {'window':22s} {'exclusive':>10} {'shared':>9} "
        f"{'calls':>7} {'excl%':>6}",
    ]
    exclusive_total = 0.0
    shared_total = 0.0
    for name, (exclusive, shared, calls, exclusive_calls) in sorted(
        _TOTALS.items(), key=lambda item: -item[1][0]
    ):
        exclusive_total += exclusive
        shared_total += shared
        share = 100.0 * exclusive_calls / calls if calls else 0.0
        lines.append(
            f"  {name:22s} {exclusive / tokens * 1000:9.2f}ms "
            f"{shared / tokens * 1000:8.2f}ms {calls / tokens:7.1f} "
            f"{share:5.0f}%"
        )
    lines.append(
        f"  {'TOTAL':22s} {exclusive_total / tokens * 1000:9.2f}ms "
        f"{shared_total / tokens * 1000:8.2f}ms"
    )
    return "\n".join(lines)
