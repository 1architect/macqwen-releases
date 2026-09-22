#!/usr/bin/env python3
"""Build and restore exact MLX prompt-cache images for repository files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.utils import load


CACHE_ROOT = Path.home() / "Library/Application Support/QwenContextImages"
CACHE_LIMIT_BYTES = 8 * 1024**3
ACK = "Repository file context loaded."


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True).encode())


def repo_key(root: Path) -> str:
    suffix = digest_bytes(str(root).encode())[:12]
    return f"{root.name}-{suffix}"


def model_fingerprint(model: Path) -> str:
    rows = []
    for path in sorted(model.glob("*.safetensors")):
        stat = path.stat()
        rows.append((path.name, stat.st_size, stat.st_mtime_ns))
    for name in ("config.json", "tokenizer.json", "chat_template.jinja"):
        path = model / name
        if path.is_file():
            rows.append((name, digest_bytes(path.read_bytes())))
    return digest_json(rows)[:16]


def image_paths(root: Path, source: Path, model: Path):
    relative = str(source.relative_to(root))
    file_key = digest_bytes(relative.encode())[:20]
    directory = CACHE_ROOT / repo_key(root) / model_fingerprint(model)
    return directory / f"{file_key}.safetensors", directory / f"{file_key}.json"


def prefix_text(tokenizer, system, tools, source, body, effort):
    content = f"// REPOSITORY FILE: {source}\n{body}\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
        {"role": "assistant", "content": ACK},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=effort,
    )


def expected_metadata(model, root, source, system, tools, effort):
    data = source.read_bytes()
    return {
        "version": 2,
        "model_fingerprint": model_fingerprint(model),
        "repo": str(root),
        "source": str(source),
        "source_sha256": digest_bytes(data),
        "contract_sha256": digest_json([system, tools, effort]),
        "effort": effort,
    }


def image_status(model_path, root, source, system, tools, effort="medium"):
    model_path = Path(model_path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    cache_path, meta_path = image_paths(root, source, model_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return "missing", None
    metadata = json.loads(meta_path.read_text())
    expected = expected_metadata(model_path, root, source, system, tools, effort)
    if any(metadata.get(key) != value for key, value in expected.items()):
        return "stale", metadata
    return "current", metadata


def build_image(
    model_path,
    root,
    source,
    system,
    tools,
    effort="medium",
    step=256,
    kv_bits=0,
    loaded=None,
    progress=None,
):
    model_path = Path(model_path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    metadata = expected_metadata(model_path, root, source, system, tools, effort)
    cache_path, meta_path = image_paths(root, source, model_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model, tokenizer = loaded if loaded is not None else load(str(model_path))
    body = source.read_text(errors="replace")
    rendered = prefix_text(tokenizer, system, tools, source, body, effort)
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    source_tokens = len(tokenizer.encode(body, add_special_tokens=False))
    cache = make_prompt_cache(model)

    for offset in range(0, len(tokens), step):
        chunk = mx.array(tokens[offset : offset + step], dtype=mx.int32)[None]
        model(chunk, cache=cache)
        mx.eval([item.state for item in cache])
        mx.clear_cache()
        done = min(offset + step, len(tokens))
        if progress is None:
            print(f"prefill {done}/{len(tokens)}", flush=True)
        else:
            progress(done, len(tokens))

    if kv_bits:
        for index, item in enumerate(cache):
            if hasattr(item, "to_quantized"):
                cache[index] = item.to_quantized(group_size=64, bits=kv_bits)
        mx.eval([item.state for item in cache])

    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent, suffix=".safetensors", delete=False
    ) as handle:
        temporary_cache = Path(handle.name)
    try:
        save_prompt_cache(str(temporary_cache), cache, {"kind": "repo-context-image"})
        temporary_cache.replace(cache_path)
    finally:
        temporary_cache.unlink(missing_ok=True)

    metadata.update(
        {
            "prefix_tokens": len(tokens),
            "source_tokens": source_tokens,
            "tokens": tokens,
            "cache_bytes": cache_path.stat().st_size,
            "build_seconds": time.perf_counter() - started,
            "kv_bits": kv_bits,
            "kv_group_size": 64 if kv_bits else None,
        }
    )
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(metadata, separators=(",", ":")))
    temporary_meta.replace(meta_path)
    prune_images()
    return metadata, cache_path


def prune_images(limit_bytes=CACHE_LIMIT_BYTES):
    entries = []
    total = 0
    for meta_path in CACHE_ROOT.glob("*/*/*.json"):
        if meta_path.stem.endswith(".fp16"):
            continue
        cache_path = meta_path.with_suffix(".safetensors")
        if not cache_path.is_file():
            continue
        size = cache_path.stat().st_size + meta_path.stat().st_size
        total += size
        entries.append((cache_path.stat().st_mtime_ns, size, cache_path, meta_path))
    for _, size, cache_path, meta_path in sorted(entries):
        if total <= limit_bytes:
            break
        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        total -= size
    return total


def restore_image(engine, root, source, system, tools, effort="medium"):
    model_path = engine.path.resolve()
    root = Path(root).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    status, metadata = image_status(
        model_path, root, source, system, tools, effort
    )
    if status != "current":
        return None
    cache_path, meta_path = image_paths(root, source, model_path)

    started = time.perf_counter()
    restored = load_prompt_cache(str(cache_path))
    mx.eval([item.state for item in restored])
    for item in engine.cache:
        if hasattr(item, "close"):
            item.close()
    engine.cache = restored
    engine.tape = [int(token) for token in metadata["tokens"]]
    engine.pending = []
    engine.turn_closed = True
    engine.turn = 0
    if not engine.check_invariant():
        raise RuntimeError("context image cache/tape mismatch")
    metadata["load_seconds"] = time.perf_counter() - started
    os.utime(cache_path, None)
    os.utime(meta_path, None)
    return metadata


def compact_image(model_path, root, source, bits=4, group_size=64):
    model_path = Path(model_path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    cache_path, meta_path = image_paths(root, source, model_path)
    if not cache_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("build the context image first")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("kv_bits") == bits:
        return metadata, cache_path

    cache = load_prompt_cache(str(cache_path))
    for index, item in enumerate(cache):
        if hasattr(item, "to_quantized"):
            cache[index] = item.to_quantized(group_size=group_size, bits=bits)
    mx.eval([item.state for item in cache])

    backup_cache = cache_path.with_name(cache_path.stem + ".fp16.safetensors")
    backup_meta = meta_path.with_name(meta_path.stem + ".fp16.json")
    if not backup_cache.exists():
        shutil.copy2(cache_path, backup_cache)
        shutil.copy2(meta_path, backup_meta)

    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent, suffix=".safetensors", delete=False
    ) as handle:
        temporary_cache = Path(handle.name)
    try:
        save_prompt_cache(str(temporary_cache), cache, {"kind": "repo-context-image"})
        temporary_cache.replace(cache_path)
    finally:
        temporary_cache.unlink(missing_ok=True)

    metadata["kv_bits"] = bits
    metadata["kv_group_size"] = group_size
    metadata["cache_bytes"] = cache_path.stat().st_size
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(metadata, separators=(",", ":")))
    temporary_meta.replace(meta_path)
    return metadata, cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "compact"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    from macqwen.profiles import SYSTEM_TOOLS
    from macqwen.tools import TOOLS

    root = Path(args.repo).expanduser().resolve()
    source = Path(args.file).expanduser()
    if not source.is_absolute():
        source = root / source
    if args.command == "build":
        metadata, path = build_image(
            args.model,
            root,
            source,
            SYSTEM_TOOLS,
            TOOLS,
            args.effort,
            args.step,
            args.bits,
        )
    else:
        metadata, path = compact_image(args.model, root, source, args.bits)
    print(
        f"ready {metadata['source_tokens']} source tokens, "
        f"{metadata['cache_bytes']/1024**2:.1f} MiB, {path}"
    )


if __name__ == "__main__":
    main()
