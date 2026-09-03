"""The agent loop, shared by every model.

Generate, parse tool calls, run them, feed the results back, repeat. None of
that depends on which model is answering, so it lives here and drives any
backend that satisfies the shared `Backend` interface.

The loop was already model-agnostic inside frankenstein_engine.mode_agent;
it touched the engine through six members and nothing else. This makes that
boundary explicit so a second model can use it.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from macqwen.backends.base import Backend

from macqwen.tools import (
    MUTATING_TOOLS,
    parse_tool_calls,
    render_tool_result,
    split_think,
)


@dataclass
class Limits:
    """Where the loop gives up.

    Memory limits exist because this runs on a 16 GB machine where swap, not
    speed, is what ends a session badly.
    """

    max_turns: int = 24
    max_tokens: int = 2048
    think_tokens: int = 512
    max_forced_closes: int = 2
    min_free_gb: float = 0.0
    max_swap_growth_gb: float = 8.0
    tool_format: str = "pretty"


# why the loop stopped, in the order they are checked
STOP_REASONS = (
    "answer",      # the model finished with no pending call
    "invariant",   # cache and transcript disagree
    "memory",      # host memory fell below the floor
    "swap",        # swap grew past the allowance
    "truncated",   # generation ended mid-turn with nothing to run
    "max-turns",   # the turn budget ran out
)


def run_agent(engine: Backend, repo, out, limits: Limits = Limits(),
              host_memory=None, approve=None, model_out=None,
              model_done=None, ui=None, on_stats=None) -> str:
    """Drive one agent session. Returns why it stopped.

    `host_memory` returns (free_gb, swap_gb) and may be None when a runtime
    cannot report it; the memory guards are then skipped rather than guessed.
    """
    swap_start = host_memory()[1] if host_memory else 0.0
    forced = 0

    def stop(reason):
        if ui is not None:
            ui.finish()
        return reason

    for turn in range(1, limits.max_turns + 1):
        if ui is not None:
            ui.start_turn(turn, len(engine.pending))
        else:
            out(f"\n{'#' * 30} TURN {turn} {'#' * 30}")
            out(f"[appending {len(engine.pending)} new tokens]")
        generation_limit = limits.max_tokens
        if getattr(engine, "thinking_enabled", False):
            generation_limit += limits.think_tokens
        text, stats = engine.generate(
            max_tokens=generation_limit,
            out=model_out or out,
            on_prefilled=ui.prefilled if ui is not None else None,
            on_prefill_progress=(
                ui.prefill_progress if ui is not None else None
            ),
        )
        if model_done is not None:
            model_done()
        if on_stats is not None:
            on_stats(stats)

        if not engine.check_invariant():
            out("!! INVARIANT BROKEN: cache and transcript disagree")
            return stop("invariant")
        free = getattr(stats, "host_free_gb", None)
        if free and limits.min_free_gb and free < limits.min_free_gb:
            out(f"!! host free memory {free:.2f} GB below the floor; stopping")
            return stop("memory")
        swap = getattr(stats, "swap_gb", None)
        if swap is not None and swap - swap_start > limits.max_swap_growth_gb:
            out(f"!! swap grew {swap - swap_start:.2f} GB since start; stopping")
            return stop("swap")

        _, content = split_think(text)
        calls = parse_tool_calls(content) or parse_tool_calls(text)
        if not calls:
            if getattr(stats, "finish", None) == "stop":
                if ui is None:
                    out("\n[final answer produced]")
                return stop("answer")
            # Qwen sometimes runs out of room inside <think> and never closes
            # it. Closing the block for it recovers the turn; doing that
            # forever would loop, so it is bounded.
            if "</think>" not in text and forced < limits.max_forced_closes:
                forced += 1
                if ui is None:
                    out(f"\n[turn ended inside <think>; forcing closure {forced}]")
                engine.append_text("\n</think>\n\n")
                continue
            if ui is None:
                out("\n[turn ended with no tool call]")
            return stop("truncated")

        forced = 0
        results = []
        for name, args in calls:
            preview = json.dumps(args, ensure_ascii=False)[:200]
            if ui is not None:
                ui.tool_started(name, args)
            else:
                out(f"\n[tool] {name}({preview})")
            if name in MUTATING_TOOLS and approve is not None:
                if ui is not None:
                    ui.tool_approval()
                if not approve(name, args):
                    results.append(json.dumps({
                        "error": "User denied this change. Do not retry it unless "
                                 "the user asks."
                    }))
                    if ui is not None:
                        ui.tool_finished(error=True)
                    else:
                        out("[tool denied]")
                    continue
            try:
                if ui is not None:
                    ui.tool_executing()
                raw = repo.call(name, args)
                results.append(render_tool_result(name, raw, limits.tool_format))
                if ui is not None:
                    ui.tool_finished(result=raw)
            except Exception as exc:
                results.append(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
                if ui is not None:
                    ui.tool_finished(error=True)
                else:
                    out(f"[tool error] {exc}")
        engine.append_tool_results(results)
    return stop("max-turns")
