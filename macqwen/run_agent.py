#!/usr/bin/env python3
"""Run one agent session against a chosen model.

    python -m macqwen.run_agent --task "list the python files and summarise one"

This is the hand-test entry point for the agent profile on Flash-Next. The
loop, the tools and the prompts are shared; only the backend is chosen here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macqwen.agent import Limits, run_agent
from macqwen.preferences import DEFAULT_ANSWER_TOKENS, DEFAULT_THINK_TOKENS
from macqwen.profiles import system_prompt, tools_for
from macqwen.text import ToolCallStreamFilter
from macqwen.tools.repo import Repo
from macqwen.tools.toolbox import Toolbox
from macqwen.ui import AgentUI, C


def build_backend(name: str, args):
    if name == "flashnext":
        from macqwen.backends.flashnext import FlashNextBackend

        return FlashNextBackend(
            model_path=args.model_path,
            threshold=args.threshold,
            resident_experts=args.resident_experts,
        )
    raise SystemExit(
        f"unknown model {name!r}. This hand-test entry point supports flashnext. "
        "Use chat.sh for the qwen27b backend."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="flashnext")
    parser.add_argument("--profile", default="agent", choices=("agent", "plain"))
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--resident-experts", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_ANSWER_TOKENS)
    parser.add_argument("--think-budget", type=int, default=DEFAULT_THINK_TOKENS)
    parser.add_argument("--quiet", action="store_true",
                        help="hide the model's reasoning as it streams")
    args = parser.parse_args()

    def out(text=""):
        sys.stdout.write(str(text))
        sys.stdout.flush()

    quiet = (lambda text="": None) if args.quiet else out
    protocol = [ToolCallStreamFilter()]

    def model_out(piece=""):
        quiet(protocol[0].feed(str(piece)))

    def model_done():
        quiet(protocol[0].finish())
        protocol[0] = ToolCallStreamFilter()

    began = time.time()
    print(f"{C['dim']}loading {args.model}...{C['0']}", flush=True)
    engine = build_backend(args.model, args)
    tools = Toolbox.build(Repo(args.workspace))
    print(f"{C['dim']}ready in {time.time() - began:.1f}s  "
          f"profile={args.profile}  workspace={tools.repo.root}{C['0']}\n")

    engine.open_conversation(
        system_prompt(args.profile, args.workspace),
        args.task,
        tools=tools_for(args.profile),
        reasoning_effort="medium",
    )
    engine.thinking_enabled = True
    began = time.time()
    reason = run_agent(
        engine, tools, quiet,
        Limits(
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            think_tokens=args.think_budget,
        ),
        model_out=model_out,
        model_done=model_done,
        ui=AgentUI(),
    )
    print(f"\n\n{C['dim']}stopped: {reason}  "
          f"{time.time() - began:.1f}s  {len(engine.tape)} tokens{C['0']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
