#!/usr/bin/env python3
"""Interactive terminal chat for Qwen3.8-Flash-Next running from disk.

Usage:
    ./flashnext/chat.sh
    ./flashnext/chat.sh --threshold 0.70      # faster, drifts more
    ./flashnext/chat.sh --threshold 1.0       # routing exactly as shipped
    ./flashnext/chat.sh --exact-quality       # exact output, 1.24 GB pinned
    ./flashnext/chat.sh --fast-quality        # fast tail with 0.93 GB pinned

Session commands:
    /salvar NAME      save the exact live model state
    /carregar NAME    restore it without re-running the old prefill
    /sessoes          list saved sessions
    /apagar NAME      delete one saved session

Chat controls:
    /thinking on|off
    /thinking show|hide
    /max-tokens N|off
    /status

Ctrl+C stops the current reply and clears its live context.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

DEFAULT_MODEL = "~/models/Qwen3.8-Flash-Next-MLX-oQ4"
NEXT_TURN_THINK = (
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
NEXT_TURN_DIRECT = (
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

C = {
    "dim": "\033[2m",
    "gray": "\033[90m",
    "b": "\033[1m",
    "g": "\033[32m",
    "y": "\033[33m",
    "r": "\033[31m",
    "c": "\033[36m",
    "clear": "\033[2K\r",
    "0": "\033[0m",
}

PREFERENCE_DEFAULTS = {
    "thinking_enabled": False,
    "show_thinking": False,
    "max_tokens": -1,
}


class IngestGlow:
    """Animate prefill with the same moving foreground glow as MACQWEN."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._active = False
        self._total = 0
        self._started = 0.0

    @staticmethod
    def colorize(text: str, center: float) -> str:
        base = (92, 101, 122)
        peak = (220, 248, 255)
        radius = 8.0
        out = []
        previous = None
        for index, char in enumerate(text):
            strength = max(0.0, 1.0 - abs(index - center) / radius)
            strength *= strength
            level = round(strength * 4)
            strength = level / 4
            rgb = tuple(
                round(start + (end - start) * strength)
                for start, end in zip(base, peak)
            )
            if rgb != previous:
                out.append(f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                previous = rgb
            out.append(char)
        return "".join(out) + C["0"]

    def start(self, total: int) -> None:
        self.finish()
        with self._lock:
            self._total = total
            self._started = time.perf_counter()
            self._active = True
            self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def finish(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=0.25)
        sys.stdout.write(C["clear"])
        sys.stdout.flush()

    def _animate(self) -> None:
        while not self._stop.wait(0.10):
            with self._lock:
                if not self._active:
                    return
                total = self._total
                elapsed = time.perf_counter() - self._started
            line = f"ingest {total:>6} tok"
            travel = len(line) + 16
            center = ((elapsed / 1.35) * travel) % travel - 8
            sys.stdout.write(
                "\r" + self.colorize(line, center) + "\033[K"
            )
            sys.stdout.flush()


def load_preferences(path: str | Path) -> dict:
    values = dict(PREFERENCE_DEFAULTS)
    try:
        saved = json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError, TypeError):
        return values
    if isinstance(saved.get("thinking_enabled"), bool):
        values["thinking_enabled"] = saved["thinking_enabled"]
    if isinstance(saved.get("show_thinking"), bool):
        values["show_thinking"] = saved["show_thinking"]
    maximum = saved.get("max_tokens")
    if (
        isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and (maximum == -1 or maximum > 0)
    ):
        values["max_tokens"] = maximum
    return values


def write_preferences(path: str | Path, values: dict) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(target.parent, 0o700)
    except OSError:
        pass
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
            prefix=".preferences.",
            suffix=".tmp",
        ) as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def human_size(size: int) -> str:
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    return f"{size / 1_000_000:.0f} MB"


def rss_gb() -> float:
    return int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()) / 1048576


def token_limit_text(limit: int) -> str:
    return "off" if limit < 0 else str(limit)


def print_commands() -> None:
    print(
        f"{C['dim']}"
        "/thinking on|off       enable or disable model reasoning\n"
        "/thinking show|hide    show or hide model reasoning\n"
        "/max-tokens N|off      set the reply limit\n"
        "/status                show current settings and memory\n"
        "/save [name]  /load [name]  /sessions  /delete <name>\n"
        "/reset  /quit          Ctrl+C stops a reply and clears its context"
        f"{C['0']}\n"
    )


def filter_thinking(piece: str, inside: bool, show: bool) -> tuple[str, bool]:
    """Remove thinking tags and optionally their contents from one token."""
    visible = ""
    while piece:
        if inside:
            end = piece.find("</think>")
            if end < 0:
                return visible + (piece if show else ""), True
            if show:
                visible += piece[:end]
            piece = piece[end + len("</think>") :]
            inside = False
        else:
            start = piece.find("<think>")
            if start < 0:
                return visible + piece, False
            visible += piece[:start]
            piece = piece[start + len("<think>") :]
            inside = True
    return visible, inside


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=os.environ.get("FLASHNEXT_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="router mass to keep; an explicit value selects standard mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="maximum reply tokens; -1 waits for EOS or Ctrl+C",
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="use threshold routing without selective RAM",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the approximate low-I/O routing profile",
    )
    parser.add_argument(
        "--fast-quality",
        action="store_true",
        help="use a quality warmup, then recover exact tail experts from RAM",
    )
    parser.add_argument(
        "--exact-quality",
        action="store_true",
        help="pin recurring experts without output drift; this is the default",
    )
    parser.add_argument("--tail-experts", type=int, default=6)
    parser.add_argument("--resident-experts", type=int, default=8)
    parser.add_argument("--tail-warmup", type=int, default=8)
    parser.add_argument(
        "--mtp-depth",
        type=int,
        default=int(os.environ.get("FLASHNEXT_MTP_DEPTH", "0")),
        help="MTP draft tokens per target verification; 0 disables it",
    )
    parser.add_argument(
        "--think",
        dest="thinking_enabled",
        action="store_true",
        default=None,
        help="enable and show reasoning blocks",
    )
    parser.add_argument(
        "--no-think",
        dest="thinking_enabled",
        action="store_false",
        help="disable reasoning blocks",
    )
    parser.add_argument(
        "--show-think",
        dest="show_thinking",
        action="store_true",
        default=None,
        help="show reasoning text",
    )
    parser.add_argument(
        "--hide-think",
        dest="show_thinking",
        action="store_false",
        help="hide reasoning text",
    )
    parser.add_argument(
        "--session-dir",
        default=os.environ.get(
            "FLASHNEXT_SESSION_DIR", "~/.cache/flashnext/sessions"
        ),
        help="directory for exact session snapshots",
    )
    parser.add_argument(
        "--preferences-file",
        default=os.environ.get(
            "FLASHNEXT_PREFERENCES", "~/.cache/flashnext/preferences.json"
        ),
        help="persistent terminal settings",
    )
    args = parser.parse_args()
    preferences = load_preferences(args.preferences_file)
    if args.max_tokens is None:
        args.max_tokens = int(
            os.environ.get(
                "FLASHNEXT_MAX_TOKENS", str(preferences["max_tokens"])
            )
        )
    if args.max_tokens == 0 or args.max_tokens < -1:
        parser.error("--max-tokens must be -1 or a positive integer")

    thinking_enabled = (
        preferences["thinking_enabled"]
        if args.thinking_enabled is None
        else args.thinking_enabled
    )
    show_thinking = (
        preferences["show_thinking"]
        if args.show_thinking is None
        else args.show_thinking
    )
    if args.thinking_enabled is True and args.show_thinking is None:
        show_thinking = True

    profile_count = sum(
        (args.standard, args.fast, args.fast_quality, args.exact_quality)
    )
    if profile_count > 1:
        parser.error("use only one performance profile")
    threshold_override = args.threshold is not None or (
        "FLASHNEXT_TOPK_THRESHOLD" in os.environ
    )
    if profile_count == 0 and not threshold_override and not args.mtp_depth:
        args.exact_quality = True
    quality_mode = args.fast_quality or args.exact_quality
    if quality_mode and args.mtp_depth:
        parser.error("quality profiles cannot be combined with --mtp-depth")

    if args.threshold is None:
        args.threshold = float(
            os.environ.get("FLASHNEXT_TOPK_THRESHOLD", "0.85")
        )
    if quality_mode:
        args.threshold = 0.85
    if args.fast:
        args.threshold = 0.20
        os.environ["FLASHNEXT_RENORM"] = "0"
        os.environ["FLASHNEXT_READ"] = "shared_mmap"
    os.environ["FLASHNEXT_TOPK_THRESHOLD"] = str(args.threshold)

    from transformers import AutoTokenizer

    from flashnext.loader import load_streaming
    from flashnext.speculative import MTPGreedy
    if args.fast:
        from flashnext.adaptive_topk import set_fast_profile

        set_fast_profile()

    path = os.path.expanduser(args.model)
    print(
        f"{C['dim']}loading {os.path.basename(path)}...{C['0']}",
        flush=True,
    )
    started = time.time()
    model, _, store = load_streaming(
        path,
        expert_capacity=0,
        verbose=True,
        keep_vision=False,
        use_mtp=args.mtp_depth > 0,
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    language = model.language_model
    threshold = args.threshold
    profile = (
        "exact-quality"
        if args.exact_quality
        else "fast-quality"
        if args.fast_quality
        else "fast" if args.fast else "standard"
    )
    print(
        f"{C['g']}ready{C['0']} in {time.time() - started:.1f}s   "
        f"profile={profile}  threshold={threshold}  RSS {rss_gb():.2f} GB\n"
    )
    stops = {tokenizer.eos_token_id, 248044, 248046}
    decoder = MTPGreedy(language, depth=args.mtp_depth) if args.mtp_depth else None
    cache = language.make_cache() if decoder is None else None
    first_turn = True
    context_ids: list[int] = []

    from flashnext.sessions import SessionError, SessionStore

    mode = (
        "exact-quality"
        if args.exact_quality
        else "fast-quality"
        if args.fast_quality
        else "fast" if args.fast else "standard"
    )
    default_renorm = float(
        os.environ.get(
            "FLASHNEXT_RENORM_BLEND",
            "1" if os.environ.get("FLASHNEXT_RENORM", "1") == "1" else "0",
        )
    )
    session_profile = {
        "mode": mode,
        "threshold": float(args.threshold),
        "think": bool(thinking_enabled),
        "stop_ids": sorted(int(value) for value in stops if value is not None),
        "prompt_protocol": {
            "first_turn": "tokenizer.apply_chat_template",
            "next_turn_direct": NEXT_TURN_DIRECT,
            "next_turn_think": NEXT_TURN_THINK,
        },
        "renorm": (
            {"warmup": 1.0, "tail": 0.1}
            if args.fast_quality
            else 0.0
            if args.fast
            else 1.0
            if args.exact_quality
            else default_renorm
        ),
        "tail_warmup": int(args.tail_warmup) if quality_mode else None,
        "tail_experts": int(args.tail_experts) if args.fast_quality else None,
        "resident_experts": (
            int(args.resident_experts) if args.exact_quality else None
        ),
        "mtp_depth": int(args.mtp_depth),
    }
    sessions = SessionStore(args.session_dir, path, session_profile, language)

    def save_ui_preferences() -> None:
        write_preferences(
            args.preferences_file,
            {
                "thinking_enabled": bool(thinking_enabled),
                "show_thinking": bool(show_thinking),
                "max_tokens": int(args.max_tokens),
            },
        )

    save_ui_preferences()
    atexit.register(save_ui_preferences)
    ingest_glow = IngestGlow()

    print(
        f"{C['dim']}saved options: "
        f"thinking={'on' if thinking_enabled else 'off'}, "
        f"display={'show' if show_thinking else 'hide'}, "
        f"max-tokens={token_limit_text(args.max_tokens)}, "
        f"profile={mode}{C['0']}\n"
    )
    print_commands()

    if quality_mode:
        from flashnext.adaptive_topk import (
            FAST_LAYERS,
            set_layer_thresholds,
            set_renorm_blend,
            set_resident_experts,
            set_route_observer,
            set_threshold,
        )

        def quality_profile():
            store.unpin_all()
            set_resident_experts(None)
            set_route_observer(None)
            store._read_mode = "pread"
            set_threshold(0.85)
            set_layer_thresholds({})
            set_renorm_blend(1.0)

        def fast_keep(scores, threshold):
            total = sum(scores)
            accumulated = 0.0
            for position, score in enumerate(scores):
                accumulated += score / total
                if accumulated >= threshold:
                    return position + 1
            return len(scores)

        def pin_candidates(candidates):
            count = (
                args.resident_experts if args.exact_quality else args.tail_experts
            )
            resident = {
                layer: {expert for expert, _ in values.most_common(count)}
                for layer, values in candidates.items()
            }
            pinned = 0
            try:
                for layer_number, experts in resident.items():
                    block = language.model.layers[layer_number].mlp.switch_mlp
                    prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for part in ("weight", "scales", "biases"):
                            pinned += store.pin_rows(
                                f"{prefix}.{projection}.{part}", sorted(experts)
                            )
            except OSError:
                store.unpin_all()
                raise
            if args.fast_quality:
                set_resident_experts(resident)
                store._read_mode = "shared_mmap"
                set_threshold(0.20)
                set_layer_thresholds({layer: 0.40 for layer in FAST_LAYERS})
                set_renorm_blend(0.10)
            return pinned
    else:
        quality_profile = None

    while True:
        try:
            prompt = input(f"{C['b']}you>{C['0']} ").strip()
        except KeyboardInterrupt:
            print(
                f"\n{C['dim']}(Ctrl+C again at an empty prompt to quit, "
                f"or type /quit){C['0']}"
            )
            try:
                prompt = input(f"{C['b']}you>{C['0']} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
        except EOFError:
            print()
            return
        if not prompt:
            continue
        if prompt in ("/sair", "/quit", "/exit", "/q"):
            return
        command_parts = prompt.split(maxsplit=1)
        command = command_parts[0]
        argument = command_parts[1].strip().lower() if len(command_parts) == 2 else ""
        command = {
            "/save": "/salvar",
            "/load": "/carregar",
            "/sessions": "/sessoes",
            "/delete": "/apagar",
        }.get(command, command)

        if command in ("/ajuda", "/help"):
            print_commands()
            continue
        if command == "/status":
            print(
                f"{C['b']}options{C['0']}   profile={mode}  "
                f"threshold={args.threshold}  "
                f"thinking={'on' if thinking_enabled else 'off'}  "
                f"display={'show' if show_thinking else 'hide'}  "
                f"max-tokens={token_limit_text(args.max_tokens)}\n"
                f"{C['b']}context{C['0']}   {len(context_ids)} tokens in cache\n"
                f"{C['b']}model{C['0']}     RSS {rss_gb():.2f} GB\n"
                f"{C['b']}config{C['0']}    "
                f"{Path(args.preferences_file).expanduser()}\n"
            )
            continue
        if command in ("/thinking", "/think"):
            if not argument:
                print(
                    f"thinking: {'on' if thinking_enabled else 'off'}, "
                    f"display: {'show' if show_thinking else 'hide'}\n"
                )
                continue
            if argument in ("on", "enable"):
                thinking_enabled = True
            elif argument in ("off", "disable"):
                thinking_enabled = False
            elif argument == "show":
                show_thinking = True
            elif argument == "hide":
                show_thinking = False
            else:
                print(f"{C['y']}usage: /thinking on|off|show|hide{C['0']}\n")
                continue
            save_ui_preferences()
            print(
                f"thinking: {'on' if thinking_enabled else 'off'}, "
                f"display: {'show' if show_thinking else 'hide'}\n"
            )
            continue
        if command in ("/max-tokens", "/limite"):
            if argument in ("off", "none", "unlimited", "-1"):
                args.max_tokens = -1
            else:
                try:
                    limit = int(argument)
                except ValueError:
                    limit = 0
                if limit <= 0:
                    print(f"{C['y']}usage: /max-tokens N|off{C['0']}\n")
                    continue
                args.max_tokens = limit
            save_ui_preferences()
            print(f"max tokens: {token_limit_text(args.max_tokens)}\n")
            continue
        if command == "/reset":
            if decoder is None:
                cache = language.make_cache()
            else:
                decoder.reset()
            first_turn = True
            context_ids.clear()
            language._position_ids = None
            language._rope_deltas = None
            if quality_mode:
                quality_profile()
            print(f"{C['g']}conversation reset{C['0']}  model stayed loaded\n")
            continue

        if command in ("/salvar", "/carregar") and decoder is not None:
            print(f"{C['y']}sessions do not support MTP yet{C['0']}\n")
            continue
        if command == "/salvar":
            name = command_parts[1].strip() if len(command_parts) == 2 else "last"
            try:
                began = time.time()
                summary = sessions.save(
                    name, cache, context_ids, first_turn
                )
                print(
                    f"{C['g']}saved{C['0']} {summary.name}  "
                    f"{summary.cached_tokens} tokens, "
                    f"{human_size(summary.size_bytes)}, {time.time() - began:.1f}s\n"
                )
            except (OSError, SessionError, ValueError) as exc:
                print(f"{C['r']}could not save session: {exc}{C['0']}\n")
            continue
        if command == "/carregar":
            name = command_parts[1].strip() if len(command_parts) == 2 else "last"
            try:
                began = time.time()
                loaded = sessions.load(name)
                if quality_mode:
                    quality_profile()
                cache = loaded.cache
                context_ids = loaded.token_ids
                first_turn = loaded.first_turn
                language._position_ids = loaded.position_ids
                language._rope_deltas = loaded.rope_deltas
                print(
                    f"{C['g']}loaded{C['0']} {name}  "
                    f"{len(context_ids)} tokens, {human_size(loaded.size_bytes)}, "
                    f"{time.time() - began:.1f}s, no old prefill\n"
                )
            except (OSError, SessionError, ValueError) as exc:
                print(f"{C['r']}could not load session: {exc}{C['0']}\n")
            continue
        if command == "/sessoes":
            if len(command_parts) != 1:
                print(f"{C['y']}usage: /sessions{C['0']}\n")
                continue
            try:
                saved = sessions.list()
            except (OSError, SessionError, ValueError) as exc:
                print(f"{C['r']}could not list sessions: {exc}{C['0']}\n")
                continue
            if not saved:
                print("no saved sessions\n")
            else:
                for item in saved:
                    if item.valid:
                        print(
                            f"  {item.name:<24} {item.cached_tokens:>7} tok  "
                            f"{human_size(item.size_bytes):>9}"
                        )
                    else:
                        print(f"  {item.name:<24} invalid: {item.error}")
                print()
            continue
        if command == "/apagar":
            if len(command_parts) != 2:
                print(f"{C['y']}usage: /delete NAME{C['0']}\n")
                continue
            try:
                deleted = sessions.delete(command_parts[1])
                result = "deleted" if deleted else "not found"
                color = C["g"] if deleted else C["y"]
                print(f"{color}{command_parts[1]} {result}{C['0']}\n")
            except (OSError, SessionError, ValueError) as exc:
                print(f"{C['r']}could not delete session: {exc}{C['0']}\n")
            continue
        if command.startswith("/"):
            print(f"{C['y']}unknown command{C['0']}  use /help\n")
            continue

        if first_turn:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_enabled,
            )
        else:
            template = NEXT_TURN_THINK if thinking_enabled else NEXT_TURN_DIRECT
            text = template.format(prompt)
        input_ids = tokenizer(text)["input_ids"]
        ids = mx.array(input_ids)[None]

        if quality_mode:
            quality_profile()
        turn_started = time.time()
        prefill_started = time.time()
        ingest_glow.start(int(ids.shape[1]))
        try:
            if decoder is None:
                out = language(ids, cache=cache)
                logits = out.logits
                mx.eval(logits)
                context_ids.extend(int(value) for value in input_ids)
            else:
                decoder.append(ids)
        finally:
            ingest_glow.finish()
        prefill = time.time() - prefill_started

        produced: list[int] = []
        pending = ""
        inside_thinking = thinking_enabled
        began = time.time()
        interrupted = False
        tail_started = None
        tail_token_index = 0
        pinned_bytes = 0
        status_visible = False
        status_updated = 0.0
        generation_rss = rss_gb()
        try:
            accepted_before = decoder.stats.accepted if decoder else 0
            drafted_before = decoder.stats.drafted if decoder else 0
            if decoder is None:
                token = mx.argmax(logits[:, -1, :], axis=-1)

                candidates = {layer: Counter() for layer in range(48)}

                def collect_candidates(layer, experts, scores, keeps):
                    threshold = 0.40 if layer in FAST_LAYERS else 0.20
                    for expert_row, score_row, normal_keep in zip(
                        experts, scores, keeps
                    ):
                        mass = sum(score_row)
                        if args.exact_quality:
                            for expert, score in zip(
                                expert_row[:normal_keep], score_row[:normal_keep]
                            ):
                                candidates[layer][expert] += score / mass
                            continue
                        keep = fast_keep(score_row, threshold)
                        for expert, score in zip(
                            expert_row[keep:], score_row[keep:]
                        ):
                            candidates[layer][expert] += score / mass

                if quality_mode:
                    set_route_observer(collect_candidates)

                def standard_tokens():
                    nonlocal token, tail_started, tail_token_index, pinned_bytes
                    try:
                        indices = (
                            itertools.count()
                            if args.max_tokens < 0
                            else range(args.max_tokens)
                        )
                        for index in indices:
                            value = int(token.item())
                            if value in stops:
                                return
                            yield value
                            step = language(token[None], cache=cache)
                            token = mx.argmax(step.logits[:, -1, :], axis=-1)
                            mx.eval(token)
                            context_ids.append(value)
                            if (
                                quality_mode
                                and index + 1 == args.tail_warmup
                                and (
                                    args.max_tokens < 0
                                    or index + 1 < args.max_tokens
                                )
                            ):
                                set_route_observer(None)
                                pinned_bytes = pin_candidates(candidates)
                                tail_token_index = index + 1
                                tail_started = time.time()
                    finally:
                        if quality_mode:
                            set_route_observer(None)

                tokens = standard_tokens()
            else:
                mtp_limit = args.max_tokens if args.max_tokens > 0 else sys.maxsize
                tokens = decoder.generate(mtp_limit, stops)
            for value in tokens:
                produced.append(value)
                was_thinking = inside_thinking
                raw_piece = tokenizer.decode([value])
                piece, inside_thinking = filter_thinking(
                    raw_piece, inside_thinking, show_thinking
                )
                if was_thinking and show_thinking and piece:
                    sys.stdout.write(C["gray"] + piece + C["0"])
                    sys.stdout.flush()
                    piece = ""
                elif was_thinking and not show_thinking:
                    if inside_thinking:
                        now = time.time()
                        if now - status_updated >= 0.4:
                            status_updated = now
                            status_visible = True
                            rate = len(produced) / max(now - began, 1e-6)
                            line = (
                                f"thinking {len(produced):>5} tok  "
                                f"{rate:5.1f} t/s  {now - began:5.1f}s  "
                                f"ctx {len(context_ids) + 1}  "
                                f"model {generation_rss:5.2f} GB"
                            )
                            sys.stdout.write(
                                "\r" + C["dim"] + line[:108] + C["0"]
                            )
                            sys.stdout.flush()
                    elif status_visible:
                        sys.stdout.write(C["clear"])
                        sys.stdout.flush()
                        status_visible = False
                if piece:
                    pending += piece
                    if len(pending) > 8 or "\n" in pending:
                        print(pending, end="", flush=True)
                        pending = ""
        except KeyboardInterrupt:
            if decoder is None:
                cache = language.make_cache()
            else:
                decoder.reset()
            first_turn = True
            context_ids.clear()
            language._position_ids = None
            language._rope_deltas = None
            if quality_mode:
                quality_profile()
            interrupted = True
            if status_visible:
                sys.stdout.write(C["clear"])
                status_visible = False
            print(
                f"{C['y']}[stopped; context cleared]{C['0']}",
                end="",
            )
        if status_visible:
            sys.stdout.write(C["clear"])
            sys.stdout.flush()
        if pending:
            print(pending, end="", flush=True)

        hit_limit = (
            not interrupted
            and args.max_tokens > 0
            and len(produced) >= args.max_tokens
        )
        if hit_limit:
            print(
                f"\n{C['y']}[reply reached {args.max_tokens} tokens]{C['0']}",
                end="",
            )

        elapsed = time.time() - began
        rate = len(produced) / elapsed if elapsed else 0.0
        drafted = decoder.stats.drafted - drafted_before if decoder else 0
        accepted = decoder.stats.accepted - accepted_before if decoder else 0
        acceptance = accepted / drafted if drafted else 0.0
        mtp_text = f" | MTP {acceptance:.0%}" if decoder else ""
        tail_text = ""
        if tail_started is not None:
            tail_count = len(produced) - tail_token_index
            tail_elapsed = time.time() - tail_started
            tail_rate = tail_count / tail_elapsed if tail_elapsed else 0.0
            tail_text = (
                f" | tail {tail_rate:.1f} t/s"
                f" | pinned {pinned_bytes / 1e9:.2f} GB"
            )
        if quality_mode:
            quality_profile()
        prompt_rate = ids.shape[1] / prefill if prefill else 0.0
        print()
        print(
            f"\n{C['dim']}{ids.shape[1]} new tok @ {prompt_rate:.1f} t/s | "
            f"gen {len(produced)} @ {rate:.1f} t/s | "
            f"ctx {len(context_ids)} | model {rss_gb():.2f} GB | "
            f"{time.time() - turn_started:.1f}s{tail_text}{mtp_text}{C['0']}"
        )
        if not interrupted:
            first_turn = False


if __name__ == "__main__":
    main()
