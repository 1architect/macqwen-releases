"""Exact persistent session snapshots for FlashNext text generation.

The upstream disk APC currently treats QSAKVCache as a plain KVCache. It drops
the QSA indexer state. This module saves that state and every recurrent cache
slot explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import uuid

import mlx.core as mx


SCHEMA = "flashnext-session-v2"
MANIFEST_KEY = "flashnext_manifest"
DEFAULT_SESSION_DIR = "~/.cache/flashnext/sessions"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_TEMP_NAME = re.compile(
    r"^\.[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.\d+\.[0-9a-f]{32}\.safetensors$"
)
_HEADER_LIMIT = 16 * 1024 * 1024
_CHECKSUM_PLACEHOLDER = "UNSET-" + "0" * 58


class SessionError(RuntimeError):
    """A snapshot is invalid or incompatible with this runtime."""


@dataclass(frozen=True)
class LoadedSession:
    cache: list[Any]
    token_ids: list[int]
    first_turn: bool
    thinking: bool
    position_ids: Any
    rope_deltas: Any
    created_at: str
    size_bytes: int


@dataclass(frozen=True)
class SessionSummary:
    name: str
    cached_tokens: int
    created_at: str
    size_bytes: int
    thinking: bool = False
    valid: bool = True
    error: str = ""


def validate_session_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise SessionError(
            "invalid name; use 1 to 64 letters, numbers, dots, _ or -"
        )
    return name


def _read_file_header(path: Path) -> tuple[dict[str, Any], int, set[str]]:
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise SessionError("truncated session file")
            header_size = int.from_bytes(raw_size, "little")
            if header_size <= 0 or header_size > _HEADER_LIMIT:
                raise SessionError("invalid session header")
            raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise SessionError("truncated session header")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise SessionError("invalid session header")
        metadata = header.get("__metadata__")
        if not isinstance(metadata, dict):
            raise SessionError("invalid session metadata")
        encoded = metadata.get(MANIFEST_KEY)
        if not isinstance(encoded, str):
            raise SessionError("missing session manifest")
        if not encoded:
            raise SessionError("missing session manifest")
        manifest = json.loads(encoded)
    except SessionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise SessionError(f"could not read session: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SessionError("invalid session manifest")
    if manifest.get("schema") != SCHEMA:
        raise SessionError("incompatible session version")

    tensor_names: set[str] = set()
    maximum_end = 0
    try:
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise SessionError("invalid safetensors entry")
            offsets = entry.get("data_offsets")
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not isinstance(shape, list)
                or not isinstance(dtype, str)
            ):
                raise SessionError("invalid safetensors descriptor")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or any(int(dim) < 0 for dim in shape):
                raise SessionError("invalid safetensors offset")
            maximum_end = max(maximum_end, end)
            tensor_names.add(name)
        data_start = 8 + header_size
        if path.stat().st_size != data_start + maximum_end:
            raise SessionError("session payload is truncated or oversized")
        if int(manifest.get("payload_bytes", -1)) != maximum_end:
            raise SessionError("payload size does not match the manifest")
    except SessionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise SessionError(f"invalid session layout: {exc}") from exc
    return manifest, data_start, tensor_names


def _read_manifest(path: Path) -> dict[str, Any]:
    return _read_file_header(path)[0]


def _payload_sha256(path: Path, data_start: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            handle.seek(data_start)
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SessionError(f"could not verify payload: {exc}") from exc
    return digest.hexdigest()


def _install_payload_checksum(path: Path) -> str:
    manifest, data_start, _ = _read_file_header(path)
    if manifest.get("payload_sha256") != _CHECKSUM_PLACEHOLDER:
        raise SessionError("missing checksum marker")
    checksum = _payload_sha256(path, data_start)
    with path.open("r+b") as handle:
        raw_size = handle.read(8)
        header_size = int.from_bytes(raw_size, "little")
        raw_header = handle.read(header_size)
        marker = _CHECKSUM_PLACEHOLDER.encode()
        if raw_header.count(marker) != 1:
            raise SessionError("invalid checksum marker")
        raw_header = raw_header.replace(marker, checksum.encode(), 1)
        handle.seek(8)
        handle.write(raw_header)
        handle.flush()
        os.fsync(handle.fileno())
    return checksum


def _verify_payload(path: Path, manifest: dict[str, Any], data_start: int) -> None:
    expected = manifest.get("payload_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SessionError("invalid payload checksum")
    actual = _payload_sha256(path, data_start)
    if not hmac.compare_digest(actual, expected):
        raise SessionError("payload checksum does not match")


def _hash_file(digest: Any, path: Path, label: str, full: bool = True) -> None:
    stat = path.stat()
    digest.update(label.encode())
    digest.update(str(stat.st_size).encode())
    with path.open("rb") as handle:
        if full or stat.st_size <= 131072:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            return
        digest.update(handle.read(65536))
        handle.seek(max(0, stat.st_size - 65536))
        digest.update(handle.read(65536))


def _hash_relocated_engine_file(digest: Any, path: Path) -> None:
    data = path.read_bytes().replace(b"models.flashnext.", b"flashnext.")
    digest.update(path.name.encode())
    digest.update(str(len(data)).encode())
    digest.update(data)


def _model_fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256(b"flashnext-model-v2")
    digest.update(str(model_dir).encode())
    found = False
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
    ):
        path = model_dir / name
        if path.is_file():
            stat = path.stat()
            digest.update(
                f"{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:"
                f"{stat.st_ctime_ns}".encode()
            )
            _hash_file(digest, path, name)
            found = True
    for path in sorted(model_dir.glob("*.safetensors")):
        if path.name == "model-mtp.safetensors":
            continue
        stat = path.stat()
        digest.update(
            f"{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:"
            f"{stat.st_ctime_ns}".encode()
        )
        _hash_file(digest, path, path.name, full=False)
        found = True
    if not found:
        raise SessionError(f"checkpoint not found: {model_dir}")
    return digest.hexdigest()


def _tokenizer_fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256(b"flashnext-tokenizer-v1")
    found = False
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    ):
        path = model_dir / name
        if path.is_file():
            _hash_file(digest, path, name)
            found = True
    if not found:
        raise SessionError(f"tokenizer not found: {model_dir}")
    return digest.hexdigest()


def _engine_fingerprint(language: Any) -> str:
    digest = hashlib.sha256(b"flashnext-engine-v1")
    for package in ("mlx", "mlx-vlm", "transformers"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        digest.update(f"{package}={version}".encode())

    paths: set[Path] = set()
    local_paths: set[Path] = set()
    local = Path(__file__).resolve().parent
    for name in (
        # Every local file that decides what lands in the cache. `qsa_chunk`
        # replaces `Qwen4ExpAttention.__call__`, so a change to its mask moves
        # the attention state; `prefill` chooses which path ingests the prompt.
        # A session restored against changed code would be silently wrong, so
        # both belong in the fingerprint.
        "adaptive_topk.py",
        "expert_cache.py",
        "loader.py",
        "ngram.py",
        "patch_rmsnorm.py",
        "prefill.py",
        "qsa_chunk.py",
        "store.py",
    ):
        path = local / name
        if path.is_file():
            paths.add(path)
            local_paths.add(path)
    for cls in type(language).__mro__:
        try:
            source = inspect.getsourcefile(cls)
        except (OSError, TypeError):
            source = None
        if source and "mlx_vlm" in source:
            path = Path(source).resolve()
            if path.is_file():
                paths.add(path)
    for entry in language.make_cache():
        for cls in type(entry).__mro__:
            try:
                source = inspect.getsourcefile(cls)
            except (OSError, TypeError):
                source = None
            if source and "mlx_vlm" in source:
                path = Path(source).resolve()
                if path.is_file():
                    paths.add(path)
    for path in sorted(paths, key=str):
        if path in local_paths:
            _hash_relocated_engine_file(digest, path)
        else:
            _hash_file(digest, path, path.name)
    return digest.hexdigest()


def _copy_array(value: Any) -> Any:
    return mx.contiguous(mx.array(value, dtype=value.dtype))


def _tensor_description(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "bytes": int(value.nbytes),
    }


class SessionStore:
    """Save and restore one model's exact prompt-cache state."""

    def __init__(
        self,
        directory: str | Path,
        model_dir: str | Path,
        profile: dict[str, Any],
        language: Any,
    ):
        self.directory = Path(directory).expanduser()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.profile = json.loads(json.dumps(profile, sort_keys=True))
        self.language = language
        self._compatibility_cache: dict[str, Any] | None = None

    def _path(self, name: str) -> Path:
        return self.directory / f"{validate_session_name(name)}.safetensors"

    def saved_profile(self, name: str) -> dict[str, Any]:
        """Return the exact generation profile recorded by one snapshot."""
        manifest = _read_manifest(self._path(name))
        compatibility = manifest.get("compatibility")
        profile = compatibility.get("profile") if isinstance(compatibility, dict) else None
        if not isinstance(profile, dict):
            raise SessionError("missing generation profile")
        return json.loads(json.dumps(profile, sort_keys=True))

    def _prepare_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        cutoff = time.time() - 24 * 60 * 60
        for path in self.directory.iterdir():
            try:
                if (
                    path.is_file()
                    and _TEMP_NAME.fullmatch(path.name)
                    and path.stat().st_mtime < cutoff
                ):
                    path.unlink()
            except OSError:
                continue

    def _compatibility(self) -> dict[str, Any]:
        if self._compatibility_cache is None:
            self._compatibility_cache = {
                "model": _model_fingerprint(self.model_dir),
                "tokenizer": _tokenizer_fingerprint(self.model_dir),
                "engine": _engine_fingerprint(self.language),
                "profile": self.profile,
            }
        return self._compatibility_cache

    @staticmethod
    def _put(arrays: dict[str, Any], key: str, value: Any) -> str:
        arrays[key] = value
        return key

    def _snapshot_layer(
        self,
        layer_index: int,
        entry: Any,
        arrays: dict[str, Any],
        cached_tokens: int,
    ) -> dict[str, Any]:
        prefix = f"cache.l{layer_index:02d}"
        class_name = type(entry).__name__
        if all(hasattr(entry, field) for field in ("keys", "values", "index_keys")):
            offset = int(entry.offset)
            layer: dict[str, Any] = {
                "kind": "qsa",
                "class": class_name,
                "offset": offset,
            }
            if offset != cached_tokens:
                raise SessionError(
                    f"QSA cache {layer_index} has {offset} tokens; expected {cached_tokens}"
                )
            if entry.keys is None:
                if (
                    offset != 0
                    or entry.values is not None
                    or entry.index_keys is not None
                    or entry.index_position_ids is not None
                ):
                    raise SessionError(f"invalid empty QSA cache at layer {layer_index}")
                layer["empty"] = True
                return layer
            if offset <= 0 or entry.values is None:
                raise SessionError(f"invalid QSA cache at layer {layer_index}")
            if entry.index_keys is None or entry.index_position_ids is None:
                raise SessionError(f"missing QSA auxiliary state at layer {layer_index}")
            if (
                entry.keys.ndim != 4
                or entry.values.ndim != 4
                or entry.keys.shape[2] < offset
                or entry.values.shape[2] < offset
                or entry.index_keys.ndim < 2
                or entry.index_keys.shape[1] < offset
                or entry.index_position_ids.ndim < 2
                or entry.index_position_ids.shape[-1] < offset
            ):
                raise SessionError(f"invalid QSA shape at layer {layer_index}")
            layer["keys"] = self._put(
                arrays, f"{prefix}.keys", entry.keys[..., :offset, :]
            )
            layer["values"] = self._put(
                arrays, f"{prefix}.values", entry.values[..., :offset, :]
            )
            layer["index_keys"] = self._put(
                arrays, f"{prefix}.index_keys", entry.index_keys[:, :offset]
            )
            layer["index_position_ids"] = self._put(
                arrays,
                f"{prefix}.index_position_ids",
                entry.index_position_ids[..., :offset],
            )
            return layer

        if hasattr(entry, "cache") and isinstance(entry.cache, list):
            slots: list[str | None] = []
            for slot, value in enumerate(entry.cache):
                key = f"{prefix}.array{slot}"
                slots.append(None if value is None else self._put(arrays, key, value))
            layer = {
                "kind": "arrays",
                "class": class_name,
                "size": len(entry.cache),
                "slots": slots,
                "left_padding_advance": int(
                    getattr(entry, "_left_padding_advance", 0)
                ),
                "lengths_advance": int(getattr(entry, "_lengths_advance", 0)),
            }
            left_padding = getattr(entry, "_left_padding", None)
            lengths = getattr(entry, "_lengths", None)
            layer["left_padding"] = (
                None
                if left_padding is None
                else self._put(arrays, f"{prefix}.left_padding", left_padding)
            )
            layer["lengths"] = (
                None
                if lengths is None
                else self._put(arrays, f"{prefix}.lengths", lengths)
            )
            return layer

        raise SessionError(
            f"unsupported cache type at layer {layer_index}: {class_name}"
        )

    def save(
        self,
        name: str,
        cache: list[Any],
        token_ids: list[int],
        first_turn: bool,
        thinking: bool = False,
    ) -> SessionSummary:
        try:
            return self._save(name, cache, token_ids, first_turn, thinking)
        except SessionError:
            raise
        except Exception as exc:
            raise SessionError(f"failed to save session: {exc}") from exc

    def _save(
        self,
        name: str,
        cache: list[Any],
        token_ids: list[int],
        first_turn: bool,
        thinking: bool = False,
    ) -> SessionSummary:
        path = self._path(name)
        cached_tokens = len(token_ids)
        if first_turn != (cached_tokens == 0):
            raise SessionError("inconsistent session boundary")
        self._prepare_directory()
        compatibility = self._compatibility()
        if path.exists():
            if not path.is_file():
                raise SessionError(f"session destination is not a file: {path}")
            try:
                previous = _read_manifest(path)
            except SessionError as exc:
                raise SessionError(
                    "an invalid session already has this name; delete it first"
                ) from exc
            if previous.get("compatibility") != compatibility:
                raise SessionError(
                    "this name belongs to another model or profile; use another name"
                )

        arrays: dict[str, Any] = {
            "session.token_ids": mx.array(token_ids, dtype=mx.int32)
        }
        layers = [
            self._snapshot_layer(index, entry, arrays, cached_tokens)
            for index, entry in enumerate(cache)
        ]
        position_ids = getattr(self.language, "_position_ids", None)
        rope_deltas = getattr(self.language, "_rope_deltas", None)
        language_state = {
            "position_ids": (
                None
                if position_ids is None
                else self._put(arrays, "language.position_ids", position_ids)
            ),
            "rope_deltas": (
                None
                if rope_deltas is None
                else self._put(arrays, "language.rope_deltas", rope_deltas)
            ),
        }
        created_at = datetime.now(timezone.utc).isoformat()
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": created_at,
            "cached_tokens": cached_tokens,
            "first_turn": first_turn,
            "thinking": bool(thinking),
            "compatibility": compatibility,
            "layer_count": len(cache),
            "layers": layers,
            "language": language_state,
        }
        manifest["tensors"] = {
            key: _tensor_description(value) for key, value in arrays.items()
        }
        manifest["payload_bytes"] = sum(
            item["bytes"] for item in manifest["tensors"].values()
        )
        manifest["payload_sha256"] = _CHECKSUM_PLACEHOLDER

        mx.eval(list(arrays.values()))
        temporary = self.directory / (
            f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.safetensors"
        )
        try:
            mx.save_safetensors(
                str(temporary),
                arrays,
                metadata={
                    MANIFEST_KEY: json.dumps(
                        manifest, sort_keys=True, separators=(",", ":")
                    )
                },
            )
            os.chmod(temporary, 0o600)
            checksum = _install_payload_checksum(temporary)
            written = _read_manifest(temporary)
            if written.get("payload_sha256") != checksum:
                raise SessionError("checksum was not written")
            os.replace(temporary, path)
            try:
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        size = path.stat().st_size
        return SessionSummary(
            name, cached_tokens, created_at, size, thinking=bool(thinking)
        )

    @staticmethod
    def _validate_tensor(
        arrays: dict[str, Any], manifest: dict[str, Any], key: str
    ) -> Any:
        if key not in arrays:
            raise SessionError(f"missing tensor: {key}")
        value = arrays[key]
        expected = manifest["tensors"].get(key)
        if not isinstance(expected, dict):
            raise SessionError(f"undeclared tensor: {key}")
        if [int(item) for item in value.shape] != expected.get("shape"):
            raise SessionError(f"invalid shape: {key}")
        if str(value.dtype) != expected.get("dtype"):
            raise SessionError(f"invalid dtype: {key}")
        if int(value.nbytes) != int(expected.get("bytes", -1)):
            raise SessionError(f"invalid size: {key}")
        return value

    def _restore_qsa(
        self,
        entry: Any,
        layer: dict[str, Any],
        arrays: dict[str, Any],
        manifest: dict[str, Any],
        cached_tokens: int,
    ) -> list[Any]:
        offset = int(layer.get("offset", -1))
        if offset != cached_tokens:
            raise SessionError("QSA offset does not match the context")
        if layer.get("empty"):
            if offset != 0:
                raise SessionError("empty QSA cache has an invalid offset")
            entry.keys = entry.values = None
            entry.index_keys = entry.index_position_ids = None
            entry.offset = 0
            return []

        names = [
            layer.get("keys"),
            layer.get("values"),
            layer.get("index_keys"),
            layer.get("index_position_ids"),
        ]
        if not all(isinstance(name, str) for name in names):
            raise SessionError("invalid QSA references")
        keys = self._validate_tensor(arrays, manifest, names[0])
        values = self._validate_tensor(arrays, manifest, names[1])
        index_keys = self._validate_tensor(arrays, manifest, names[2])
        positions = self._validate_tensor(
            arrays, manifest, names[3]
        )
        if keys.ndim != 4 or values.ndim != 4:
            raise SessionError("QSA cache has an invalid rank")
        if keys.shape[2] != offset or values.shape[2] != offset:
            raise SessionError("invalid QSA K/V length")
        if index_keys.ndim < 2 or index_keys.shape[1] != offset:
            raise SessionError("invalid index_keys length")
        if positions.ndim < 2 or positions.shape[-1] != offset:
            raise SessionError("invalid index_position_ids length")

        keys = _copy_array(keys)
        values = _copy_array(values)
        step = max(1, int(getattr(entry, "step", 256)))
        capacity = (offset // step + 1) * step
        spare = capacity - offset
        if spare:
            key_shape = list(keys.shape)
            value_shape = list(values.shape)
            key_shape[2] = spare
            value_shape[2] = spare
            keys = mx.concatenate(
                [keys, mx.zeros(key_shape, dtype=keys.dtype)], axis=2
            )
            values = mx.concatenate(
                [values, mx.zeros(value_shape, dtype=values.dtype)], axis=2
            )
        entry.keys = keys
        entry.values = values
        entry.offset = offset
        entry.index_keys = _copy_array(index_keys)
        entry.index_position_ids = _copy_array(positions)
        return [keys, values, entry.index_keys, entry.index_position_ids]

    def _restore_arrays(
        self,
        entry: Any,
        layer: dict[str, Any],
        arrays: dict[str, Any],
        manifest: dict[str, Any],
    ) -> list[Any]:
        size = int(layer.get("size", -1))
        slots = layer.get("slots")
        if size != len(entry.cache) or not isinstance(slots, list) or len(slots) != size:
            raise SessionError("incompatible ArraysCache layout")
        restored = []
        targets = []
        for key in slots:
            if key is None:
                restored.append(None)
            else:
                if not isinstance(key, str):
                    raise SessionError("invalid ArraysCache reference")
                value = _copy_array(self._validate_tensor(arrays, manifest, key))
                restored.append(value)
                targets.append(value)
        entry.cache = restored
        for field in ("left_padding", "lengths"):
            key = layer.get(field)
            value = (
                None
                if key is None
                else _copy_array(self._validate_tensor(arrays, manifest, key))
            )
            setattr(entry, f"_{field}", value)
            if value is not None:
                targets.append(value)
        entry._left_padding_advance = int(layer.get("left_padding_advance", 0))
        entry._lengths_advance = int(layer.get("lengths_advance", 0))
        return targets

    def load(self, name: str) -> LoadedSession:
        try:
            return self._load(name)
        except SessionError:
            raise
        except Exception as exc:
            raise SessionError(f"invalid session: {exc}") from exc

    def _load(self, name: str) -> LoadedSession:
        path = self._path(name)
        if not path.is_file():
            raise SessionError(f"session not found: {name}")
        manifest, data_start, tensor_names = _read_file_header(path)
        if manifest.get("compatibility") != self._compatibility():
            actual = manifest.get("compatibility") or {}
            expected = self._compatibility()
            if not isinstance(actual, dict):
                reason = "invalid metadata"
            elif actual.get("model") != expected["model"]:
                reason = "different checkpoint"
            elif actual.get("tokenizer") != expected["tokenizer"]:
                reason = "different tokenizer"
            elif actual.get("engine") != expected["engine"]:
                reason = "different engine code"
            else:
                reason = "different generation profile"
            raise SessionError(f"incompatible session: {reason}")

        cached_tokens = int(manifest.get("cached_tokens", -1))
        first_turn = manifest.get("first_turn")
        thinking = manifest.get("thinking", False)
        if (
            cached_tokens < 0
            or not isinstance(first_turn, bool)
            or not isinstance(thinking, bool)
        ):
            raise SessionError("invalid context metadata")
        if first_turn != (cached_tokens == 0):
            raise SessionError("invalid session boundary")

        declared = manifest.get("tensors")
        if not isinstance(declared, dict) or set(declared) != tensor_names:
            raise SessionError("invalid session tensor list")
        _verify_payload(path, manifest, data_start)

        try:
            arrays = mx.load(str(path))
        except Exception as exc:
            raise SessionError(f"invalid session payload: {exc}") from exc
        if set(arrays) != set(declared):
            raise SessionError("invalid session tensor list")
        token_array = self._validate_tensor(
            arrays, manifest, "session.token_ids"
        )
        if token_array.ndim != 1 or token_array.shape[0] != cached_tokens:
            raise SessionError("invalid session token list")

        cache = self.language.make_cache()
        layers = manifest.get("layers")
        if (
            manifest.get("layer_count") != len(cache)
            or not isinstance(layers, list)
            or len(layers) != len(cache)
        ):
            raise SessionError("incompatible layer count")
        eval_targets: list[Any] = []
        for entry, layer in zip(cache, layers):
            if not isinstance(layer, dict) or type(entry).__name__ != layer.get("class"):
                raise SessionError("incompatible cache type")
            if all(
                hasattr(entry, field)
                for field in ("keys", "values", "index_keys")
            ):
                expected_kind = "qsa"
            elif hasattr(entry, "cache") and isinstance(entry.cache, list):
                expected_kind = "arrays"
            else:
                raise SessionError("unsupported cache type")
            if layer.get("kind") != expected_kind:
                raise SessionError("incompatible cache layout")
            if expected_kind == "qsa":
                eval_targets.extend(
                    self._restore_qsa(
                        entry, layer, arrays, manifest, cached_tokens
                    )
                )
            else:
                eval_targets.extend(
                    self._restore_arrays(entry, layer, arrays, manifest)
                )

        language_state = manifest.get("language")
        if not isinstance(language_state, dict):
            raise SessionError("missing position state")
        position_key = language_state.get("position_ids")
        rope_key = language_state.get("rope_deltas")
        if position_key is not None and not isinstance(position_key, str):
            raise SessionError("invalid position_ids reference")
        if rope_key is not None and not isinstance(rope_key, str):
            raise SessionError("invalid rope_deltas reference")
        position_ids = (
            None
            if position_key is None
            else _copy_array(
                self._validate_tensor(arrays, manifest, position_key)
            )
        )
        rope_deltas = (
            None
            if rope_key is None
            else _copy_array(self._validate_tensor(arrays, manifest, rope_key))
        )
        if position_ids is not None:
            eval_targets.append(position_ids)
        if rope_deltas is not None:
            eval_targets.append(rope_deltas)
        eval_targets.append(token_array)
        try:
            mx.eval(eval_targets)
            token_ids = [int(value) for value in token_array.tolist()]
        except Exception as exc:
            raise SessionError(f"corrupted session data: {exc}") from exc
        return LoadedSession(
            cache=cache,
            token_ids=token_ids,
            first_turn=first_turn,
            thinking=thinking,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            created_at=str(manifest.get("created_at", "")),
            size_bytes=path.stat().st_size,
        )

    def list(self) -> list[SessionSummary]:
        if not self.directory.is_dir():
            return []
        sessions = []
        for path in sorted(self.directory.glob("*.safetensors")):
            if path.name.startswith("."):
                continue
            name = path.stem
            try:
                manifest = _read_manifest(path)
                sessions.append(
                    SessionSummary(
                        name=name,
                        cached_tokens=int(manifest.get("cached_tokens", 0)),
                        created_at=str(manifest.get("created_at", "")),
                        size_bytes=path.stat().st_size,
                        thinking=bool(manifest.get("thinking", False)),
                    )
                )
            except Exception as exc:
                sessions.append(
                    SessionSummary(
                        name=name,
                        cached_tokens=0,
                        created_at="",
                        size_bytes=path.stat().st_size if path.exists() else 0,
                        valid=False,
                        error=str(exc),
                    )
                )
        return sessions

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
