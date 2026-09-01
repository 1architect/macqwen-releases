#!/usr/bin/env python3
"""The chat. One interface, any model, either profile.

    python -m macqwen.session                      plain profile, flashnext
    python -m macqwen.session --profile agent      tools and the repository prompt

Commands come from macqwen.commands, prompts from macqwen.profiles, and the
model from a backend. Nothing here knows which model is answering.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macqwen import commands, preferences
from macqwen.api_keys import (
    DEFAULT_PATH as DEFAULT_API_KEYS_PATH,
    KeyStore,
    sanitized_environment,
)
from macqwen.agent import Limits, run_agent
from macqwen.model_settings import FLASHNEXT_DEFAULTS
from macqwen.profiles import system_prompt, tools_for
from macqwen.tools.repo import Repo
from macqwen.tools.toolbox import Toolbox
from macqwen.terminal import read_prompt
from macqwen.text import ThinkingStreamFilter, ToolCallStreamFilter
from macqwen.ui import (
    AgentUI,
    AsyncWordAnimator,
    C,
    IngestGlow,
    filter_thinking,
    rss_gb,
)


def ask_approval(name: str, args: dict, input_fn=input) -> bool:
    """Require a reply typed after the approval prompt becomes visible."""
    try:
        if sys.stdin.isatty():
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
    subject = args.get("path") or args.get("command") or str(args)
    prompts = (
        f"{C['y']}approve {name}: {str(subject)[:140]}? [y/s/N] {C['0']}",
        f"{C['y']}type y to approve or n to deny: {C['0']}",
    )
    for prompt in prompts:
        try:
            answer = input_fn(prompt)
        except EOFError:
            return False
        letters = re.sub(r"[^a-z]", "", answer.lower())
        if letters in ("y", "yes", "s", "sim"):
            return True
        if letters in ("n", "no", "nao"):
            return False
    return False


class Session:
    """What a command sees: the settings, the backend and the workspace."""

    def __init__(
        self,
        backend,
        profile: str,
        prefs: dict,
        path: str,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        migrate_system_prompt: bool = True,
    ):
        self.backend = backend
        self.profile = profile
        self.preferences = prefs
        self.preferences_path = path
        self.running = True
        self.tools = None
        self.api_keys = KeyStore(api_keys_path)
        if migrate_system_prompt:
            self._migrate_system_prompt()
        self._build_tools()
        self.opened = False
        self.server_requested = False

    @property
    def repo(self):
        return self.tools.repo if self.tools is not None else None

    def _build_tools(self):
        if self.profile == "agent":
            self.tools = Toolbox.build(Repo(self.preferences["workspace"]))
        else:
            self.tools = None

    def save_preferences(self):
        preferences.save(self.preferences, self.preferences_path)
        self.backend.thinking_enabled = self.preferences["thinking_enabled"]

    def stop(self):
        self.running = False

    def start_server(self):
        self.server_requested = True
        self.running = False

    def current_system_prompt(self):
        try:
            saved = self.system_prompt_path().read_text().strip()
        except OSError:
            saved = ""
        return saved or system_prompt(
            self.profile, self.preferences["workspace"]
        )

    def system_prompt_path(self) -> Path:
        return Path(self.preferences_path).expanduser().resolve().with_name(
            f"system-prompt-{self.profile}.txt"
        )

    def _migrate_system_prompt(self):
        legacy = self.preferences.get("system_prompt", "").strip()
        if not legacy:
            return
        path = self.system_prompt_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_text(legacy + "\n")
            os.chmod(path, 0o600)
        self.preferences["system_prompt"] = ""
        preferences.save(self.preferences, self.preferences_path)

    def set_system_prompt(self, value: str):
        value = value.strip()
        path = self.system_prompt_path()
        if value:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_text(value + "\n")
            os.chmod(path, 0o600)
        else:
            path.unlink(missing_ok=True)
        self.preferences["system_prompt"] = ""
        self.save_preferences()

    def edit_system_prompt(self):
        path = self.system_prompt_path()
        if not path.is_file():
            self.set_system_prompt(self.current_system_prompt())
        subprocess.call(
            [os.environ.get("EDITOR", "nano"), str(path)],
            env=sanitized_environment(),
        )

    def set_profile(self, profile: str):
        changed = profile != self.profile
        self.profile = profile
        self.preferences["profile"] = profile
        self.save_preferences()
        if changed:
            self.reset()
        return changed

    def list_api_keys(self):
        return self.api_keys.status()

    def set_api_key(self, service: str):
        try:
            secret = getpass.getpass(f"{service} API key: ")
            self.api_keys.set(service, secret)
        except (EOFError, KeyboardInterrupt):
            return "API key input cancelled"
        except (OSError, ValueError) as exc:
            return f"could not save API key: {exc}"
        self._build_tools()
        return f"{service.lower()} API key saved in Application Support"

    def delete_api_key(self, service: str):
        try:
            deleted = self.api_keys.delete(service)
        except (OSError, ValueError) as exc:
            return f"could not delete API key: {exc}"
        self._build_tools()
        return f"{service.lower()} API key {'deleted' if deleted else 'not stored'}"

    def reset(self):
        self.backend.reset()
        self._build_tools()
        self.opened = False

    def save_session(self, name):
        return self.backend.save_session(name)

    def load_session(self, name):
        result = self.backend.load_session(name)
        self.opened = bool(self.backend.tape)
        return result

    def list_sessions(self):
        return self.backend.list_sessions()

    def delete_session(self, name):
        return self.backend.delete_session(name)

    def model_settings(self, argument: str):
        configure = getattr(self.backend, "configure", None)
        if configure is None:
            return "this model does not expose configurable settings"
        try:
            return configure(argument)
        except ValueError as exc:
            return f"could not change settings: {exc}"

    def status(self):
        prefs = self.preferences
        routing = getattr(self.backend, "routing_profile", None)
        default_answer = (
            preferences.DEFAULT_PLAIN_ANSWER_TOKENS
            if self.profile == "plain"
            else preferences.DEFAULT_ANSWER_TOKENS
        )
        lines = [
            f"{C['b']}profile{C['0']}   {self.profile}"
            f"  model {prefs['model']}"
            f"{'  routing ' + routing if routing else ''}"
            f"{'  workspace ' + str(self.repo.root) if self.repo else ''}",
            f"{C['b']}options{C['0']}   "
            f"thinking={'on' if prefs['thinking_enabled'] else 'off'}  "
            f"display={'show' if prefs['show_thinking'] else 'hide'}  "
            f"animate={'on' if prefs['animate'] else 'off'}  "
            f"effort={prefs['effort']}  "
            f"answer-tokens={preferences.answer_limit(prefs, default_answer)}  "
            f"think-tokens={preferences.think_limit(prefs)}",
            f"{C['b']}context{C['0']}   {len(self.backend.tape)} tokens",
            f"{C['b']}memory{C['0']}    RSS {rss_gb():.2f} GB",
        ]
        try:
            import mlx.core as mx

            lines.append(f"{C['b']}mlx{C['0']}       "
                         f"active {mx.get_active_memory() / 1e9:.2f} GB, "
                         f"cache {mx.get_cache_memory() / 1e9:.2f} GB")
        except Exception:
            pass
        return "\n".join(lines)


def build_backend(name: str, args, prefs: dict):
    if name == "flashnext":
        from macqwen.backends.flashnext import FlashNextBackend

        backend = FlashNextBackend(
            model_path=args.model_path,
            threshold=args.threshold,
            resident_experts=args.resident_experts,
            pin_budget_gb=args.pin_budget_gb,
            routing_profile=args.routing_profile,
            swap_epsilon=args.swap_epsilon,
            tail_experts=args.tail_experts,
            tail_warmup=args.tail_warmup,
            fusion_block=args.fusion_block,
            fusion_min_margin=args.fusion_min_margin,
            fusion_min_block=args.fusion_min_block,
            fusion_margin_tokens=args.fusion_margin_tokens,
            fusion_max_prompt=args.fusion_max_prompt,
            fusion_model=args.fusion_model,
            session_dir=(args.session_dir or "~/.cache/flashnext/sessions"),
        )
        backend.thinking_enabled = prefs["thinking_enabled"]
        return backend
    if name == "qwen27b":
        if not args.model_path:
            raise SystemExit("--model-path is required for model 'qwen27b'")
        from macqwen.backends.frankenstein import FrankensteinBackend

        backend = FrankensteinBackend(
            model_path=args.model_path,
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
            backtrack_bias=args.backtrack_bias,
            paged=args.paged,
            page_size=args.page_size,
            top_k_pages=args.top_k_pages,
            resident_pages=args.resident_pages,
            spill_dir=args.spill_dir,
            min_context=args.min_context,
            lm_head_last=args.lm_head_opt,
            wired_limit_gb=args.wired_limit_gb,
            layer_indices=args.layer_indices,
            bf16_ends=args.bf16_ends,
            shortlist_k=args.shortlist_k,
            session_dir=(args.session_dir or "~/.frankenstein/sessions"),
        )
        backend.thinking_enabled = prefs["thinking_enabled"]
        return backend
    raise SystemExit(
        f"unknown model {name!r}. Use 'flashnext' or 'qwen27b'.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=("flashnext", "qwen27b"))
    parser.add_argument("--profile", default=None, choices=("plain", "agent"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", default=None)
    parser.add_argument(
        "--threshold", type=float, default=FLASHNEXT_DEFAULTS["threshold"]
    )
    parser.add_argument(
        "--resident-experts", "--pinned-experts", dest="resident_experts", type=int,
        default=FLASHNEXT_DEFAULTS["resident_experts"],
    )
    parser.add_argument(
        "--pin-budget-gb", type=float,
        default=FLASHNEXT_DEFAULTS["pin_budget_gb"],
    )
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--standard", dest="routing_profile",
                         action="store_const", const="standard")
    routing.add_argument("--fast", dest="routing_profile",
                         action="store_const", const="fast")
    routing.add_argument("--fast-quality", dest="routing_profile",
                         action="store_const", const="fast-quality")
    routing.add_argument("--exact-quality", dest="routing_profile",
                         action="store_const", const="exact-quality")
    routing.add_argument("--cache-aware", dest="routing_profile",
                         action="store_const", const="cache-aware")
    routing.add_argument("--fused-quality", dest="routing_profile",
                         action="store_const", const="fused-quality")
    parser.set_defaults(routing_profile=FLASHNEXT_DEFAULTS["routing"])
    parser.add_argument(
        "--swap-epsilon", type=float,
        default=FLASHNEXT_DEFAULTS["swap_epsilon"],
    )
    parser.add_argument(
        "--tail-experts", type=int, default=FLASHNEXT_DEFAULTS["tail_experts"]
    )
    parser.add_argument(
        "--tail-warmup", type=int, default=FLASHNEXT_DEFAULTS["tail_warmup"]
    )
    parser.add_argument(
        "--fusion-block", type=int, default=FLASHNEXT_DEFAULTS["fusion_block"]
    )
    parser.add_argument(
        "--fusion-min-margin", type=float,
        default=FLASHNEXT_DEFAULTS["fusion_min_margin"],
    )
    parser.add_argument(
        "--fusion-min-block", type=int,
        default=FLASHNEXT_DEFAULTS["fusion_min_block"],
    )
    parser.add_argument(
        "--fusion-margin-tokens", type=int,
        default=FLASHNEXT_DEFAULTS["fusion_margin_tokens"],
    )
    parser.add_argument(
        "--fusion-max-prompt", type=int,
        default=FLASHNEXT_DEFAULTS["fusion_max_prompt"],
    )
    parser.add_argument(
        "--fusion-model", default=FLASHNEXT_DEFAULTS["fusion_model"]
    )
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--repetition-context-size", type=int, default=512)
    parser.add_argument("--backtrack-bias", type=float, default=0.0)
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--top-k-pages", type=int, default=16)
    parser.add_argument("--resident-pages", type=int, default=24)
    parser.add_argument("--min-context", type=int, default=16384)
    parser.add_argument("--spill-dir", default="/tmp/frankenstein_pages")
    parser.add_argument("--wired-limit-gb", type=float, default=None)
    parser.add_argument("--bf16-ends", action="store_true")
    parser.add_argument("--shortlist-k", type=int, default=1024)
    parser.add_argument("--lm-head-opt", action="store_true")
    parser.add_argument("--layer-indices", default=None)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--think-budget", type=int, default=None)
    parser.add_argument("--think", dest="thinking_enabled",
                        action="store_true", default=None)
    parser.add_argument("--no-think", dest="thinking_enabled",
                        action="store_false")
    parser.add_argument("--show-think", dest="show_thinking",
                        action="store_true", default=None)
    parser.add_argument("--hide-think", dest="show_thinking",
                        action="store_false")
    parser.add_argument("--preferences-file", default=preferences.DEFAULT_PATH)
    parser.add_argument("--api-keys-file", default=DEFAULT_API_KEYS_PATH)
    parser.add_argument("--server", action="store_true",
                        help="run a local API server without terminal chat")
    parser.add_argument("--allow-origin", action="append", default=[],
                        metavar="ORIGIN",
                        help="let a browser page from ORIGIN use the server; "
                             "repeatable, or '*' for any. Without this, "
                             "requests carrying an Origin header are refused, "
                             "so a site you visit cannot use your model")
    parser.add_argument("--host", default="127.0.0.1",
                        help="server bind address; default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080,
                        help="server port; default: 8080")
    parser.add_argument("--server-api-key",
                        default=os.environ.get("MACQWEN_SERVER_API_KEY"),
                        help="optional Bearer or x-api-key value")
    parser.add_argument("--benchmark-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-prompt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.benchmark_json and not args.benchmark_prompt:
        parser.error("--benchmark-json requires --benchmark-prompt")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.host not in ("127.0.0.1", "localhost", "::1") \
            and not args.server_api_key:
        parser.error("a non-local server requires --server-api-key")
    if args.max_tokens is not None and args.max_tokens != -1 and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive or -1")
    if args.think_budget is not None \
            and args.think_budget != -1 and args.think_budget <= 0:
        parser.error("--think-budget must be positive or -1")
    prefs = preferences.load(args.preferences_file)
    if (
        args.tail_experts < 1
        or args.resident_experts < 1
        or args.tail_warmup < 1
    ):
        parser.error("Flash-Next expert counts and warmup must be positive")
    if not 0.01 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.01 and 1.0")
    if not 0 <= args.swap_epsilon <= 1.0:
        parser.error("--swap-epsilon must be between 0 and 1.0")
    if not 0 <= args.pin_budget_gb <= 64:
        parser.error("--pin-budget-gb must be between 0 and 64")
    if (
        args.fusion_block < 1
        or args.fusion_min_block < 1
        or args.fusion_margin_tokens < 0
        or args.fusion_max_prompt < 1
        or args.fusion_min_margin < 0
    ):
        parser.error("Flash-Next fusion settings are outside their valid range")
    if args.fusion_min_block > args.fusion_block:
        parser.error("--fusion-min-block cannot exceed --fusion-block")
    # a flag beats the saved value, and becomes the saved value
    if args.model:
        prefs["model"] = args.model
    if args.profile:
        prefs["profile"] = args.profile
    if args.workspace:
        prefs["workspace"] = args.workspace
    if args.max_tokens is not None:
        prefs["max_tokens"] = args.max_tokens
    if args.think_budget is not None:
        prefs["think_budget"] = args.think_budget
    if args.thinking_enabled is not None:
        prefs["thinking_enabled"] = args.thinking_enabled
        if args.thinking_enabled and args.show_thinking is None:
            prefs["show_thinking"] = True
    if args.show_thinking is not None:
        prefs["show_thinking"] = args.show_thinking
    if prefs["model"] == "flashnext":
        from macqwen.checkpoints import resolve_flashnext

        environment_checkpoint = os.environ.get("MACQWEN_FLASHNEXT_MODEL")
        choice = args.model_path or environment_checkpoint \
            or prefs["flashnext_checkpoint"] or None
        try:
            args.model_path = str(resolve_flashnext(choice))
        except ValueError as exc:
            parser.error(str(exc))
        if args.model_path and not environment_checkpoint:
            prefs["flashnext_checkpoint"] = args.model_path
    if args.benchmark_json and prefs["profile"] != "plain":
        parser.error("--benchmark-json requires --profile plain")
    # A benchmark uses temporary command-line conditions. It must not change
    # the user's next interactive chat, as an earlier 20-token probe did.
    if not args.benchmark_json:
        preferences.save(prefs, args.preferences_file)

    signal.signal(signal.SIGTSTP, signal.SIG_IGN)

    began = time.time()
    display = sys.stderr if args.benchmark_json else sys.stdout
    print(f"{C['dim']}loading {prefs['model']}...{C['0']}", flush=True, file=display)
    if args.benchmark_json:
        with contextlib.redirect_stdout(sys.stderr):
            backend = build_backend(prefs["model"], args, prefs)
    else:
        backend = build_backend(prefs["model"], args, prefs)
    session = Session(
        backend,
        prefs["profile"],
        prefs,
        args.preferences_file,
        args.api_keys_file,
        migrate_system_prompt=not args.benchmark_json,
    )
    ready_seconds = time.time() - began
    routing = getattr(backend, "routing_profile", None)
    print(f"{C['dim']}ready in {time.time() - began:.1f}s  "
          f"model={prefs['model']}  profile={prefs['profile']}"
          f"  thinking={'on' if prefs['thinking_enabled'] else 'off'}"
          f"{'  routing=' + routing if routing else ''}"
          f"{_swap_banner()}  "
          f"RSS {rss_gb():.2f} GB{C['0']}\n", file=display)
    if args.benchmark_json:
        print(json.dumps(run_benchmark(session, args.benchmark_prompt, ready_seconds)))
        return 0
    if args.server:
        from macqwen.server import serve

        serve(session, args.host, args.port, args.server_api_key,
              args.allow_origin)
        return 0
    print(commands.render_help(session.profile) + "\n")

    glow = IngestGlow()
    while session.running:
        try:
            prompt = read_prompt(f"{C['b']}you>{C['0']} ")
        except KeyboardInterrupt:
            print(f"\n{C['dim']}(Ctrl+C again or /quit to leave){C['0']}")
            try:
                prompt = read_prompt(f"{C['b']}you>{C['0']} ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
        except EOFError:
            print()
            return 0
        if not prompt.strip():
            continue

        try:
            answered = commands.dispatch(session, prompt)
            if answered is not None:
                if answered:
                    print(answered + "\n")
                continue

            if session.profile == "agent":
                run_turn_agent(session, prompt)
            else:
                run_turn_plain(session, prompt, glow)
        except KeyboardInterrupt:
            glow.finish()
            session.reset()
            print(f"\n{C['dim']}generation interrupted; conversation reset{C['0']}\n")
    if session.server_requested:
        from macqwen.server import serve

        serve(session, args.host, args.port, args.server_api_key,
              args.allow_origin)
    return 0


def _swap_banner() -> str:
    """Show cache-aware routing on the ready line. It changes the reply, so
    it must be visible rather than inferred from an environment variable."""
    try:
        from models.flashnext.routing import swap_enabled, swap_epsilon
    except ImportError:
        return ""
    return f"  swap=on/{swap_epsilon():g}" if swap_enabled() else ""


def open_or_continue(session, prompt: str) -> None:
    """First turn carries the system prompt; later turns just add the user."""
    if not session.opened:
        session.backend.open_conversation(
            session.current_system_prompt(),
            prompt,
            tools=tools_for(session.profile),
            enable_thinking=session.preferences["thinking_enabled"],
            reasoning_effort=session.preferences["effort"])
        session.opened = True
    else:
        session.backend.append_user(
            prompt, enable_thinking=session.preferences["thinking_enabled"])


def run_turn_plain(session, prompt: str, glow: IngestGlow) -> None:
    open_or_continue(session, prompt)
    prefs = session.preferences
    limit = preferences.generation_limit(
        prefs, preferences.DEFAULT_PLAIN_ANSWER_TOKENS
    )
    thinking = ThinkingStreamFilter(
        prefs["thinking_enabled"], prefs["show_thinking"]
    )
    animator = AsyncWordAnimator(enabled=prefs["animate"])

    def show(piece: str) -> None:
        visible = thinking.feed(piece)
        if visible:
            animator.feed(visible, C["gray"] if thinking.inside else "")

    glow.start(len(session.backend.pending))
    began = time.time()
    try:
        # the glow belongs to the prefill; decoding prints the answer over it
        text, stats = session.backend.generate(
            max_tokens=limit,
            out=show if prefs["stream_answers"] else None,
            on_prefilled=glow.finish,
            on_prefill_progress=glow.update,
        )
    except KeyboardInterrupt:
        animator.cancel()
        session.reset()
        print(f"\n{C['dim']}generation interrupted; conversation reset{C['0']}\n")
        return
    finally:
        glow.finish()
    if prefs["stream_answers"]:
        tail = thinking.finish()
        if tail:
            animator.feed(tail, C["gray"] if thinking.inside else "")
        animator.finish(C["gray"] if thinking.inside else "")
    if not prefs["stream_answers"]:
        visible, _ = filter_thinking(
            text, prefs["thinking_enabled"], prefs["show_thinking"]
        )
        if visible:
            animator.feed(visible)
            animator.finish()
    elapsed = time.time() - began
    tail_tokens = getattr(stats, "tail_tokens", 0)
    tail_seconds = getattr(stats, "tail_seconds", 0.0)
    tail_text = (
        f" | tail {tail_tokens} @ {tail_tokens / tail_seconds:.1f} t/s"
        if tail_tokens and tail_seconds else ""
    )
    print(f"\n\n{C['dim']}{stats.prompt_tokens} new tok @ "
          f"{stats.prompt_rate:.1f} t/s | gen {stats.tokens} @ {stats.rate:.1f} t/s"
          f"{tail_text} | "
          f"ctx {len(session.backend.tape)} | {elapsed:.1f}s{C['0']}\n")


def run_benchmark(session, prompt: str, ready_seconds: float) -> dict:
    open_or_continue(session, prompt)
    prefs = session.preferences
    limit = preferences.generation_limit(
        prefs, preferences.DEFAULT_PLAIN_ANSWER_TOKENS
    )
    began = time.time()
    text, stats = session.backend.generate(max_tokens=limit)
    turn_seconds = time.time() - began
    produced = session.backend.tape[-stats.tokens:] if stats.tokens else []
    token_bytes = b"".join(
        int(value).to_bytes(4, "little", signed=False) for value in produced
    )
    tail_seconds = getattr(stats, "tail_seconds", 0.0)
    result = {
        "profile": getattr(session.backend, "routing_profile", session.profile),
        "model": prefs["model"],
        "ready_seconds": ready_seconds,
        "prompt_tokens": stats.prompt_tokens,
        "prompt_tps": stats.prompt_rate,
        "generated_tokens": stats.tokens,
        "decode_tps": stats.rate,
        "tail_tokens": getattr(stats, "tail_tokens", 0),
        "tail_tps": (
            getattr(stats, "tail_tokens", 0) / tail_seconds if tail_seconds else 0.0
        ),
        "pinned_bytes": getattr(stats, "pinned_bytes", 0),
        "pinned_signature": getattr(stats, "pinned_signature", ""),
        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "token_ids": produced,
        "output_text": text,
        "turn_seconds": turn_seconds,
        "complete_tps": stats.tokens / turn_seconds if turn_seconds else 0.0,
    }
    try:
        import mlx.core as mx

        result.update({
            "mlx_active_gb": mx.get_active_memory() / 1e9,
            "mlx_cache_gb": mx.get_cache_memory() / 1e9,
            "mlx_peak_gb": mx.get_peak_memory() / 1e9,
        })
    except Exception:
        pass
    return result


def token_stats_text(stats_items, context: int, elapsed: float) -> str:
    """Combine every model segment from one agent request."""
    prompt_tokens = sum(getattr(item, "prompt_tokens", 0) for item in stats_items)
    output_tokens = sum(getattr(item, "tokens", 0) for item in stats_items)
    prefill_seconds = sum(
        getattr(item, "prefill_seconds", 0.0) for item in stats_items
    )
    decode_seconds = sum(getattr(item, "seconds", 0.0) for item in stats_items)
    tail_tokens = sum(getattr(item, "tail_tokens", 0) for item in stats_items)
    tail_seconds = sum(getattr(item, "tail_seconds", 0.0) for item in stats_items)
    prompt_rate = prompt_tokens / prefill_seconds if prefill_seconds else 0.0
    output_rate = output_tokens / decode_seconds if decode_seconds else 0.0
    tail_text = (
        f" | tail {tail_tokens:,} @ {tail_tokens / tail_seconds:.1f} tok/s"
        if tail_tokens and tail_seconds else ""
    )
    return (
        f"{prompt_tokens:,} new tok @ {prompt_rate:.1f} tok/s | "
        f"gen {output_tokens:,} @ {output_rate:.1f} tok/s"
        f"{tail_text} | "
        f"ctx {context:,} | {elapsed:.1f}s"
    )


def run_turn_agent(session, prompt: str) -> None:
    open_or_continue(session, prompt)
    prefs = session.preferences

    def out(text=""):
        sys.stdout.write(str(text))
        sys.stdout.flush()

    thinking = [ThinkingStreamFilter(
        prefs["thinking_enabled"], prefs["show_thinking"]
    )]
    protocol = [ToolCallStreamFilter()]
    animator = [AsyncWordAnimator(enabled=prefs["animate"])]
    ui = AgentUI()
    turn_stats = []

    def feed_model_text(piece=""):
        visible = protocol[0].feed(thinking[0].feed(str(piece)))
        if visible:
            if animator[0] is None:
                animator[0] = AsyncWordAnimator(enabled=prefs["animate"])
            animator[0].feed(
                visible, C["gray"] if thinking[0].inside else ""
            )
        started, name = protocol[0].take_events()
        if started or name:
            if animator[0] is not None:
                animator[0].finish(C["gray"] if thinking[0].inside else "")
                animator[0] = None
            ui.tool_pending(name)

    def model_out(piece=""):
        feed_model_text(piece)

    def model_done():
        tail = thinking[0].finish()
        if tail:
            feed_model_text(tail)
        visible = protocol[0].finish()
        if visible:
            if animator[0] is None:
                animator[0] = AsyncWordAnimator(enabled=prefs["animate"])
            animator[0].feed(
                visible, C["gray"] if thinking[0].inside else ""
            )
        if animator[0] is not None:
            animator[0].finish(C["gray"] if thinking[0].inside else "")
        thinking[0] = ThinkingStreamFilter(
            prefs["thinking_enabled"], prefs["show_thinking"]
        )
        protocol[0] = ToolCallStreamFilter()
        animator[0] = None

    began = time.time()
    try:
        reason = run_agent(
            session.backend, session.tools, out,
            Limits(
                max_tokens=preferences.answer_limit(prefs),
                think_tokens=preferences.think_limit(prefs),
            ),
            approve=ask_approval if prefs["approval"] == "ask" else None,
            model_out=model_out,
            model_done=model_done,
            ui=ui,
            on_stats=turn_stats.append,
        )
    except KeyboardInterrupt:
        ui.finish()
        if animator[0] is not None:
            animator[0].cancel()
        raise
    elapsed = time.time() - began
    print(f"\n{C['dim']}{token_stats_text(turn_stats, len(session.backend.tape), elapsed)}"
          f"\nstopped: {reason}{C['0']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
