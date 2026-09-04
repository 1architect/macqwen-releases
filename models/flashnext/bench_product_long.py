#!/usr/bin/env python3
"""Compare real chat.sh startup paths over a 256-token generation.

Each child process starts through ``chat.sh``. The child emits one JSON record
after every 32 generated tokens. The normal path inherits the caller's
environment. The canonical path adds the current 60-slot Frontier 8A controls.
This keeps startup effects visible and does not alter the established
32-token production benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from models.flashnext.settings.launch import CHAT_ENV


ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "chat.sh"
PROMPT = "Explique a fotossintese em duas frases, com uma resposta clara."

def run_path(name: str, args) -> list[dict]:
    environment = os.environ.copy()
    if name == "canonical":
        environment.update(CHAT_ENV)
    command = [
        str(CHAT), "--model", "flashnext", "--profile", "plain",
        "--exact-quality", "--think", "--benchmark-product-json",
        "--benchmark-prompt", args.prompt, "--benchmark-window", str(args.window),
        "--benchmark-tokens", str(args.tokens), "--benchmark-label", name,
    ]
    print(f"START {name}: {' '.join(command)}", flush=True)
    records = []
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line:
                print(line, flush=True)
            continue
        records.append(record)
        print(json.dumps(record), flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{name} chat.sh exited with status {returncode}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--paths", nargs="+", default=["normal", "canonical"],
                        choices=("normal", "canonical"))
    args = parser.parse_args()
    if args.tokens <= 0 or args.window <= 0 or args.tokens % args.window:
        parser.error("tokens must be a positive multiple of window")
    all_records = {name: run_path(name, args) for name in args.paths}
    completes = {
        name: next((row for row in rows if row.get("type") == "complete"), {})
        for name, rows in all_records.items()
    }
    print("SUMMARY", flush=True)
    for name, rows in all_records.items():
        windows = [row for row in rows if row.get("type") == "window"]
        rates = [row.get("generation_tps", 0.0) for row in windows]
        print(json.dumps({
            "path": name,
            "windows": len(windows),
            "generation_tps": rates,
            "digest": completes[name].get("digest", ""),
            "generated_tokens": completes[name].get("generated_tokens", 0),
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
