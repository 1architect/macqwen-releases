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

Ctrl+C stops the current reply without ending the session.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import os
import sys
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


def human_size(size: int) -> str:
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    return f"{size / 1_000_000:.0f} MB"


def rss_gb() -> float:
    return int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()) / 1048576


def token_limit_text(limit: int) -> str:
    return "sem limite" if limit < 0 else str(limit)


def print_commands() -> None:
    print(
        "\nComandos:\n"
        "  /thinking on|off       ativa ou desativa o thinking\n"
        "  /thinking show|hide    mostra ou oculta o thinking\n"
        "  /max-tokens N|off      define o limite da resposta\n"
        "  /status                mostra a configuração atual\n"
        "  /salvar NOME           salva a sessão exata\n"
        "  /carregar NOME         carrega sem refazer o prefill\n"
        "  /sessoes               lista as sessões\n"
        "  /apagar NOME           apaga uma sessão\n"
        "  /reset                 limpa o contexto\n"
        "  /sair                  encerra o chat\n"
        "  Ctrl+C                 interrompe e limpa o contexto\n"
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
        default=float(os.environ.get("FLASHNEXT_TOPK_THRESHOLD", "0.85")),
        help="router mass to keep; lower is faster (default 0.85)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("FLASHNEXT_MAX_TOKENS", "-1")),
        help="maximum reply tokens; -1 waits for EOS or Ctrl+C",
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
        help="keep normal routing and pin recurring experts without output drift",
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
        "--think", action="store_true", help="enable and show reasoning blocks"
    )
    parser.add_argument(
        "--session-dir",
        default=os.environ.get(
            "FLASHNEXT_SESSION_DIR", "~/.cache/flashnext/sessions"
        ),
        help="directory for exact session snapshots",
    )
    args = parser.parse_args()
    if args.max_tokens == 0 or args.max_tokens < -1:
        parser.error("--max-tokens must be -1 or a positive integer")

    thinking_enabled = bool(args.think)
    show_thinking = bool(args.think)

    quality_mode = args.fast_quality or args.exact_quality
    if sum((args.fast, args.fast_quality, args.exact_quality)) > 1:
        parser.error("use only one performance profile")
    if quality_mode and args.mtp_depth:
        parser.error("quality profiles cannot be combined with --mtp-depth")

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
    print(f"carregando {os.path.basename(path)} ...", flush=True)
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
        "   perfil exact-quality"
        if args.exact_quality
        else "   perfil fast-quality"
        if args.fast_quality
        else "   perfil fast" if args.fast else ""
    )
    print(
        f"pronto em {time.time() - started:.1f}s   RSS {rss_gb():.2f} GB"
        f"   limiar {threshold}{profile}"
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
        "think": bool(args.think),
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

    print(
        f"thinking {'on' if thinking_enabled else 'off'}, "
        f"exibição {'on' if show_thinking else 'off'}, "
        f"limite {token_limit_text(args.max_tokens)}"
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
            prompt = input("\033[1mvoce>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt in ("/sair", "/quit", "/exit"):
            return
        command_parts = prompt.split(maxsplit=1)
        command = command_parts[0]
        argument = command_parts[1].strip().lower() if len(command_parts) == 2 else ""

        if command in ("/ajuda", "/help"):
            print_commands()
            continue
        if command == "/status":
            print(
                f"  perfil {mode}, limiar {args.threshold}, RSS {rss_gb():.1f} GB\n"
                f"  thinking {'on' if thinking_enabled else 'off'}, "
                f"exibição {'on' if show_thinking else 'off'}\n"
                f"  limite {token_limit_text(args.max_tokens)}, "
                f"contexto {len(context_ids)} tokens\n"
            )
            continue
        if command in ("/thinking", "/think"):
            if not argument:
                print(
                    f"  thinking {'on' if thinking_enabled else 'off'}, "
                    f"exibição {'on' if show_thinking else 'off'}\n"
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
                print("  uso: /thinking on|off|show|hide\n")
                continue
            print(
                f"  thinking {'on' if thinking_enabled else 'off'}, "
                f"exibição {'on' if show_thinking else 'off'}\n"
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
                    print("  uso: /max-tokens N|off\n")
                    continue
                args.max_tokens = limit
            print(f"  limite {token_limit_text(args.max_tokens)}\n")
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
            print("  contexto limpo\n")
            continue

        if command in ("/salvar", "/carregar") and decoder is not None:
            print("  sessões ainda não suportam MTP\n")
            continue
        if command == "/salvar":
            if len(command_parts) != 2:
                print("  uso: /salvar NOME\n")
                continue
            try:
                began = time.time()
                summary = sessions.save(
                    command_parts[1], cache, context_ids, first_turn
                )
                print(
                    f"  sessão {summary.name} salva: {summary.cached_tokens} tok, "
                    f"{human_size(summary.size_bytes)}, {time.time() - began:.1f}s\n"
                )
            except (OSError, SessionError, ValueError) as exc:
                print(f"  erro ao salvar sessão: {exc}\n")
            continue
        if command == "/carregar":
            if len(command_parts) != 2:
                print("  uso: /carregar NOME\n")
                continue
            try:
                began = time.time()
                loaded = sessions.load(command_parts[1])
                if quality_mode:
                    quality_profile()
                cache = loaded.cache
                context_ids = loaded.token_ids
                first_turn = loaded.first_turn
                language._position_ids = loaded.position_ids
                language._rope_deltas = loaded.rope_deltas
                print(
                    f"  sessão {command_parts[1]} carregada: "
                    f"{len(context_ids)} tok, {human_size(loaded.size_bytes)}, "
                    f"{time.time() - began:.1f}s, sem prefill antigo\n"
                )
            except (OSError, SessionError, ValueError) as exc:
                print(f"  erro ao carregar sessão: {exc}\n")
            continue
        if command == "/sessoes":
            if len(command_parts) != 1:
                print("  uso: /sessoes\n")
                continue
            try:
                saved = sessions.list()
            except (OSError, SessionError, ValueError) as exc:
                print(f"  erro ao listar sessões: {exc}\n")
                continue
            if not saved:
                print("  nenhuma sessão salva\n")
            else:
                for item in saved:
                    if item.valid:
                        print(
                            f"  {item.name:<24} {item.cached_tokens:>7} tok  "
                            f"{human_size(item.size_bytes):>9}"
                        )
                    else:
                        print(f"  {item.name:<24} inválida: {item.error}")
                print()
            continue
        if command == "/apagar":
            if len(command_parts) != 2:
                print("  uso: /apagar NOME\n")
                continue
            try:
                deleted = sessions.delete(command_parts[1])
                result = "apagada" if deleted else "não encontrada"
                print(f"  sessão {command_parts[1]} {result}\n")
            except (OSError, SessionError, ValueError) as exc:
                print(f"  erro ao apagar sessão: {exc}\n")
            continue
        if command.startswith("/"):
            print(f"  comando desconhecido: {command}. Use /ajuda.\n")
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
        began = time.time()
        if decoder is None:
            out = language(ids, cache=cache)
            logits = out.logits
            mx.eval(logits)
            context_ids.extend(int(value) for value in input_ids)
        else:
            decoder.append(ids)
        prefill = time.time() - began
        print(f"\033[2m  prefill {ids.shape[1]} tok em {prefill:.1f}s\033[0m")

        produced: list[int] = []
        pending = ""
        inside_thinking = thinking_enabled
        began = time.time()
        interrupted = False
        tail_started = None
        tail_token_index = 0
        pinned_bytes = 0
        print("\033[1mqwen>\033[0m ", end="", flush=True)
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
                piece = tokenizer.decode([value])
                piece, inside_thinking = filter_thinking(
                    piece, inside_thinking, show_thinking
                )
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
            print("\033[2m  [interrompido; contexto limpo]\033[0m", end="")
        if pending:
            print(pending, end="", flush=True)

        hit_limit = (
            not interrupted
            and args.max_tokens > 0
            and len(produced) >= args.max_tokens
        )
        if hit_limit:
            print(
                f"\n\033[33m  [limite de {args.max_tokens} tokens atingido]\033[0m",
                end="",
            )

        elapsed = time.time() - began
        rate = len(produced) / elapsed if elapsed else 0.0
        drafted = decoder.stats.drafted - drafted_before if decoder else 0
        accepted = decoder.stats.accepted - accepted_before if decoder else 0
        acceptance = accepted / drafted if drafted else 0.0
        mtp_text = f"   MTP {acceptance:.0%}" if decoder else ""
        tail_text = ""
        if tail_started is not None:
            tail_count = len(produced) - tail_token_index
            tail_elapsed = time.time() - tail_started
            tail_rate = tail_count / tail_elapsed if tail_elapsed else 0.0
            tail_text = (
                f"   tail {tail_rate:.2f} tok/s"
                f"   RAM {pinned_bytes / 1e9:.2f} GB"
            )
        if quality_mode:
            quality_profile()
        print(f"\n\033[2m  {len(produced)} tok em {elapsed:.1f}s = {rate:.2f} tok/s"
              f"{tail_text}{mtp_text}   RSS {rss_gb():.1f} GB\033[0m\n")
        if not interrupted:
            first_turn = False


if __name__ == "__main__":
    main()
