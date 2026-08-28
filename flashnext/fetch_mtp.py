#!/usr/bin/env python3
"""Extract the Qwen4 MTP tensors from one remote safetensors shard."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import requests


REPO = "Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP"
PREFIX = "language_model.mtp."


def get_range(session, url: str, start: int, end: int):
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    if response.status_code != 206:
        response.close()
        raise RuntimeError(f"server ignored Range request: HTTP {response.status_code}")
    return response


def remote_header(session, shard_url: str):
    response = get_range(session, shard_url, 0, 7)
    raw = response.content
    response.close()
    if len(raw) != 8:
        raise RuntimeError(f"invalid safetensors prefix: {len(raw)} bytes")
    header_len = struct.unpack("<Q", raw)[0]
    response = get_range(session, shard_url, 8, 7 + header_len)
    raw = response.content
    response.close()
    if len(raw) != header_len:
        raise RuntimeError(f"invalid safetensors header: {len(raw)} bytes")
    return header_len, json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    session = requests.Session()
    index_response = session.get(
        f"{base}/model.safetensors.index.json", timeout=60
    )
    index_response.raise_for_status()
    weight_map = index_response.json()["weight_map"]
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(PREFIX)}
    if not selected:
        raise RuntimeError("the remote checkpoint has no MTP tensors")
    shards = set(selected.values())
    if len(shards) != 1:
        raise RuntimeError(f"MTP tensors span {len(shards)} shards")
    shard = shards.pop()
    shard_url = f"{base}/{shard}"
    source_header_len, source_header = remote_header(session, shard_url)

    tensors = []
    for name in selected:
        meta = source_header[name]
        start, end = meta["data_offsets"]
        tensors.append((start, end, name, meta))
    tensors.sort()

    output_header = {}
    output_offset = 0
    for start, end, name, meta in tensors:
        size = end - start
        output_header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [output_offset, output_offset + size],
        }
        output_offset += size

    runs = []
    for start, end, *_ in tensors:
        if runs and runs[-1][1] == start:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))

    print(f"MTP tensors : {len(tensors)}")
    print(f"source shard: {shard}")
    print(f"data        : {output_offset / 1e9:.3f} GB in {len(runs)} range(s)")
    if args.inspect:
        return

    free = shutil.disk_usage(args.output.parent).free
    if free < output_offset + 64 * 1024 * 1024:
        raise RuntimeError(
            f"insufficient free space: {free / 1e9:.2f} GB available"
        )

    encoded = json.dumps(output_header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    data_start = 8 + source_header_len
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")

    try:
        with temporary.open("wb") as target:
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for run_start, run_end in runs:
                response = get_range(
                    session,
                    shard_url,
                    data_start + run_start,
                    data_start + run_end - 1,
                )
                copied = 0
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        target.write(chunk)
                        copied += len(chunk)
                response.close()
                expected = run_end - run_start
                if copied != expected:
                    raise RuntimeError(
                        f"short range: expected {expected}, received {copied}"
                    )
        os.replace(temporary, args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"wrote       : {args.output} ({args.output.stat().st_size / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()
