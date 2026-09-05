"""Qwen3.8-27B V4 as a shared-chat backend.

The legacy engine still owns model loading and generation. This adapter gives
it the same generation statistics and session methods as Flash-Next.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

from models.qwen27b.settings import get_registry


_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SESSION_FORMAT = 2


def _is_mlx_array(value):
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and hasattr(value, "nbytes"))


def _encode_tree(value, tensors, path=()):
    """Encode nested cache state while keeping arrays in safetensors."""
    if value is None:
        return {"type": "none"}
    if _is_mlx_array(value):
        key = "t_" + "_".join(str(part) for part in path)
        if key in tensors:
            raise ValueError(f"duplicate cache tensor path {key}")
        tensors[key] = value
        return {"type": "array", "key": key}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [
            _encode_tree(item, tensors, path + (index,))
            for index, item in enumerate(value)
        ]}
    if isinstance(value, list):
        return {"type": "list", "items": [
            _encode_tree(item, tensors, path + (index,))
            for index, item in enumerate(value)
        ]}
    if isinstance(value, dict):
        return {"type": "dict", "items": [
            [str(key), _encode_tree(item, tensors, path + (str(key),))]
            for key, item in value.items()
        ]}
    if isinstance(value, (str, int, float, bool)):
        return {"type": "value", "value": value}
    raise TypeError(f"unsupported cache state value {type(value).__name__}")


def _decode_tree(node, tensors):
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise ValueError("invalid cache tree node")
    kind = node["type"]
    if kind == "none":
        return None
    if kind == "array":
        key = node.get("key")
        if not isinstance(key, str) or key not in tensors:
            raise ValueError(f"cache tensor {key!r} is missing")
        return tensors[key]
    if kind in ("tuple", "list"):
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid cache sequence")
        values = [_decode_tree(item, tensors) for item in items]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid cache mapping")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError("invalid cache mapping item")
            result[item[0]] = _decode_tree(item[1], tensors)
        return result
    if kind == "value" and isinstance(node.get("value"), (str, int, float, bool)):
        return node["value"]
    raise ValueError(f"unsupported cache tree node type {kind!r}")


def _cache_config(cache):
    if type(cache).__name__ != "PagedKVCache":
        return {}
    return {
        "page_size": int(cache.page_size),
        "top_k_pages": int(cache.top_k_pages),
        "pinned_pages": int(cache.pinned_pages),
        "recent_pages": int(cache.recent_pages),
        "refresh_every": int(cache.refresh_every),
        "min_context": int(cache.min_context),
        "resident_pages": int(cache.resident_pages),
        "gather_decode": bool(cache.gather_decode),
    }


def _cache_state(cache):
    try:
        return cache.state
    except (AttributeError, IndexError, TypeError):
        if hasattr(cache, "empty") and cache.empty():
            return None
        raise ValueError(f"could not read {type(cache).__name__} state")


@dataclass
class Stats:
    finish: str
    tokens: int
    rate: float
    seconds: float
    prompt_tokens: int
    prompt_rate: float
    prefill_seconds: float
    host_free_gb: float | None
    swap_gb: float | None


class FrankensteinBackend:
    """Adapt the Qwen3.8-27B model runtime to the shared chat."""

    def __init__(
        self,
        model_path: str,
        *,
        prefill_step_size: int = 512,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 8192,
        temperature: float = 0.0,
        repetition_penalty: float | None = 1.12,
        repetition_context_size: int = 512,
        backtrack_bias: float = 0.0,
        paged: bool = False,
        page_size: int = 256,
        top_k_pages: int = 16,
        resident_pages: int = 24,
        spill_dir: str | None = "/tmp/frankenstein_pages",
        min_context: int = 16384,
        lm_head_last: bool = False,
        wired_limit_gb: float | None = None,
        layer_indices: str | None = None,
        bf16_ends: bool = False,
        shortlist_k: int = 1024,
        session_dir: str = "~/.frankenstein/sessions",
    ):
        from models.qwen27b.frankenstein_engine import FrankensteinEngine

        self._model_lock = open(
            os.environ.get("MACQWEN_MODEL_LOCK", "/tmp/macqwen_model.lock"), "w"
        )
        try:
            fcntl.flock(self._model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another MACQWEN chat already has the model loaded") from exc

        self._cache_options = {
            "paged": paged,
            "page_size": page_size,
            "top_k_pages": top_k_pages,
            "resident_pages": resident_pages,
            "spill_dir": spill_dir,
            "min_context": min_context,
        }
        self._session_dir = Path(session_dir).expanduser()
        self.thinking_enabled = False
        self._interactive_budgets = None
        self._startup_settings = {
            "prefill-step-size": prefill_step_size,
            "kv-bits": kv_bits if kv_bits is not None else "off",
            "kv-group-size": kv_group_size,
            "quantized-kv-start": quantized_kv_start,
            "paged": paged,
            "page-size": page_size,
            "top-k-pages": top_k_pages,
            "resident-pages": resident_pages,
            "min-context": min_context,
            "temperature": temperature,
            "repetition-penalty": repetition_penalty,
            "repetition-context-size": repetition_context_size,
            "backtrack-bias": backtrack_bias,
            "shortlist-k": shortlist_k,
            "spill-dir": spill_dir,
            "wired-limit-gb": wired_limit_gb,
            "bf16-ends": bf16_ends,
            "lm-head-opt": lm_head_last,
            "layer-indices": layer_indices,
            "session-dir": str(self._session_dir),
        }
        self._setting_sources = {}
        self.engine = FrankensteinEngine(
            os.path.expanduser(model_path),
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            backtrack_bias=backtrack_bias,
            paged=paged,
            page_size=page_size,
            top_k_pages=top_k_pages,
            resident_pages=resident_pages,
            spill_dir=spill_dir,
            min_context=min_context,
            lm_head_last=lm_head_last,
            wired_limit_gb=wired_limit_gb,
            layer_indices=layer_indices,
            bf16_ends=bf16_ends,
            shortlist_k=shortlist_k,
        )

    def configure(self, argument: str) -> str:
        return get_registry().configure(self, argument, "qwen27b")

    def settings_registry(self):
        return get_registry()

    def settings_state(self, include_research: bool = False) -> str:
        return self.settings_registry().render(self, "qwen27b", include_research)

    @property
    def tape(self):
        return self.engine.tape

    @property
    def pending(self):
        return self.engine.pending

    @property
    def cache(self):
        return self.engine.cache

    @property
    def cache_tokens(self) -> int:
        return self.engine.cache_tokens

    def open_conversation(self, *args, **kwargs):
        return self.engine.open_conversation(*args, **kwargs)

    def append_user(self, *args, **kwargs):
        return self.engine.append_user(*args, **kwargs)

    def append_text(self, *args, **kwargs):
        return self.engine.append_text(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.engine.encode(*args, **kwargs)

    def append_tokens(self, *args, **kwargs):
        return self.engine.append_tokens(*args, **kwargs)

    def common_prefix(self, ids) -> int:
        """How many leading tokens of `ids` the tape already holds.

        The server uses this to keep the cache when a request extends the
        conversation it already prefilled.
        """
        tape = self.engine.tape
        limit = min(len(ids), len(tape))
        index = 0
        while index < limit and tape[index] == ids[index]:
            index += 1
        return index

    def append_tool_results(self, *args, **kwargs):
        return self.engine.append_tool_results(*args, **kwargs)

    def check_invariant(self) -> bool:
        return self.engine.check_invariant()

    def generate(self, max_tokens: int, out=None, on_prefilled=None,
                 on_prefill_progress=None):
        fired = False

        def prefilled():
            nonlocal fired
            if not fired and on_prefilled is not None:
                fired = True
                on_prefilled()

        def progress(done, total):
            if on_prefill_progress is not None:
                on_prefill_progress(done, total)
            if done >= total:
                prefilled()

        def on_token(_count, result):
            prefilled()
            if out is not None and result.text:
                out(result.text)

        old_budgets = getattr(self.engine, "_interactive_budgets", None)
        old_thinking = getattr(self.engine, "_thinking_enabled", None)
        self.engine._interactive_budgets = getattr(self, "_interactive_budgets", None)
        self.engine._thinking_enabled = getattr(self, "thinking_enabled", False)
        try:
            text, old = self.engine.generate(
                max_tokens=max_tokens,
                echo=False,
                progress=progress,
                on_token=on_token,
            )
        finally:
            self.engine._interactive_budgets = old_budgets
            self.engine._thinking_enabled = old_thinking
        prefilled()
        prompt_seconds = (
            old.new_prompt_tokens / old.prompt_tps if old.prompt_tps else 0.0
        )
        decode_seconds = old.gen_tokens / old.gen_tps if old.gen_tps else 0.0
        return text, Stats(
            finish=old.finish,
            tokens=old.gen_tokens,
            rate=old.gen_tps,
            seconds=decode_seconds,
            prompt_tokens=old.new_prompt_tokens,
            prompt_rate=old.prompt_tps,
            prefill_seconds=prompt_seconds,
            host_free_gb=old.host_free_gb,
            swap_gb=old.swap_gb,
        )

    def _path(self, name: str) -> Path:
        if not _SESSION_NAME.fullmatch(name):
            raise ValueError(
                "invalid name; use 1 to 64 letters, numbers, dots, _ or -"
            )
        return self._session_dir / name

    def _cache_records(self, tensors):
        records = []
        for index, cache in enumerate(self.cache):
            state = _cache_state(cache)
            try:
                meta_state = cache.meta_state
            except AttributeError:
                meta_state = ""
            records.append({
                "class": type(cache).__name__,
                "state": _encode_tree(state, tensors, (index,)),
                "meta_state": _encode_tree(meta_state, tensors, ("meta", index)),
                "config": _cache_config(cache),
            })
        return records

    @staticmethod
    def _atomic_json(path, content):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False,
                prefix=f".{path.name}.", suffix=".tmp"
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_safetensors(path, tensors, metadata):
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            import mlx.core as mx
            mx.save_safetensors(str(temporary), tensors, metadata=metadata)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _new_cache(self, record, live_cache, tensors):
        """Construct one supported cache and restore its complete tree."""
        name = record.get("class")
        live_name = type(live_cache).__name__
        compatible = {name, live_name} <= {"KVCache", "QuantizedKVCache"}
        if name != live_name and not compatible:
            raise ValueError(
                f"saved cache type {name!r} does not match {live_name!r}"
            )
        state = _decode_tree(record.get("state"), tensors)
        meta_state = _decode_tree(record.get("meta_state"), tensors)
        config = record.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"invalid configuration for {name}")

        from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache

        if name == "ArraysCache":
            if not isinstance(state, list):
                raise ValueError("ArraysCache state must be a list")
            cache = ArraysCache(size=len(state))
        elif name == "KVCache":
            cache = KVCache()
        elif name == "QuantizedKVCache":
            if not isinstance(meta_state, (list, tuple)) or len(meta_state) != 3:
                raise ValueError("QuantizedKVCache metadata is incomplete")
            try:
                group_size, bits = int(meta_state[1]), int(meta_state[2])
            except (TypeError, ValueError) as exc:
                raise ValueError("QuantizedKVCache metadata is invalid") from exc
            cache = QuantizedKVCache(group_size=group_size, bits=bits)
        elif name == "PagedKVCache":
            from models.qwen27b.paged_kv import PagedKVCache

            allowed = (
                "page_size", "top_k_pages", "pinned_pages", "recent_pages",
                "refresh_every", "min_context", "resident_pages", "gather_decode",
            )
            if any(key not in config for key in allowed):
                raise ValueError("PagedKVCache configuration is incomplete")
            parent = self._cache_options.get("spill_dir")
            if parent is None:
                parent = getattr(live_cache, "_spill_parent", None)
            cache = PagedKVCache(
                page_size=int(config["page_size"]),
                top_k_pages=int(config["top_k_pages"]),
                pinned_pages=int(config["pinned_pages"]),
                recent_pages=int(config["recent_pages"]),
                refresh_every=int(config["refresh_every"]),
                min_context=int(config["min_context"]),
                resident_pages=int(config["resident_pages"]),
                spill_dir=parent,
            )
            cache.gather_decode = bool(config["gather_decode"])
        else:
            raise ValueError(f"unsupported cache type {name!r}")

        if state is not None:
            cache.state = state
            if name == "PagedKVCache":
                cache._update_bounds()
                cache._enforce_budget()
        if meta_state not in (None, "", []):
            cache.meta_state = meta_state
        return cache

    def _legacy_records(self, tensors, tape):
        """Read the old one-level format when its structure remains safe."""
        records = []
        for index, live_cache in enumerate(self.cache):
            prefix = f"c{index}_"
            indexed = sorted(
                (int(key[len(prefix):]), key) for key in tensors
                if key.startswith(prefix) and key[len(prefix):].isdigit()
            )
            values = {part: tensors[key] for part, key in indexed}
            name = type(live_cache).__name__
            if name == "ArraysCache":
                expected = len(live_cache.state)
                if any(part >= expected for part in values):
                    raise ValueError("legacy ArraysCache state has invalid slots")
                state = [values.get(part) for part in range(expected)]
            elif name == "KVCache":
                if sorted(values) != [0, 1]:
                    raise ValueError(f"legacy {name} state is incomplete")
                state = (values[0], values[1])
            elif name == "QuantizedKVCache":
                raise ValueError(
                    "legacy QuantizedKVCache state lacks nested quantization metadata"
                )
            elif name == "PagedKVCache":
                if len(values) % 2 or sorted(values) != list(range(len(values))):
                    raise ValueError("legacy PagedKVCache state is incomplete")
                state = [values[part] for part in range(len(values))]
            else:
                raise ValueError(f"unsupported legacy cache type {name!r}")
            meta_state = getattr(live_cache, "meta_state", "")
            if name == "QuantizedKVCache" and isinstance(meta_state, (list, tuple)):
                meta_state = (str(len(tape)), str(meta_state[1]), str(meta_state[2]))
            records.append({
                "class": name,
                "state": _encode_tree(state, tensors, ("legacy", index)),
                "meta_state": _encode_tree(meta_state, tensors, ("legacy-meta", index)),
                "config": _cache_config(live_cache),
            })
        return records

    @staticmethod
    def _validate_cache(cache, tape):
        offsets = [int(item.offset) for item in cache if hasattr(item, "offset")]
        if not offsets:
            if tape:
                raise ValueError("saved cache has no token offset")
            return
        if any(offset != len(tape) for offset in offsets):
            raise ValueError("saved cache and token tape disagree")

    def save_session(self, name: str) -> str:
        tensors = {}
        try:
            import mlx.core as mx

            directory = self._path(name)
            directory.mkdir(parents=True, exist_ok=True)
            metadata = {
                "format": _SESSION_FORMAT,
                "tape": self.tape,
                "started": bool(self.tape),
                "turn_closed": self.engine.turn_closed,
                "thinking": self.thinking_enabled,
            }
            metadata["caches"] = self._cache_records(tensors)
            encoded = json.dumps(metadata, separators=(",", ":"))
            self._atomic_safetensors(
                directory / "cache.safetensors", tensors,
                {"macqwen_session": encoded},
            )
            # Keep meta.json for listing and compatibility. Both files use
            # replacement, so a failed write cannot truncate a good session.
            self._atomic_json(directory / "meta.json", encoded)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return f"could not save session: {exc}"
        size_mb = sum(value.nbytes for value in tensors.values()) / 1e6
        return f"saved {name}  {len(self.tape)} tokens, {size_mb:.0f} MB"

    def load_session(self, name: str) -> str:
        temporary = []
        try:
            import mlx.core as mx

            directory = self._path(name)
            loaded = mx.load(str(directory / "cache.safetensors"), return_metadata=True)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                tensors, file_metadata = loaded
            else:
                tensors, file_metadata = loaded, {}
            # Force the file payload through MLX before touching live caches.
            # A corrupt or unreadable tensor must leave the current session intact.
            mx.eval(list(tensors.values()))
            embedded = file_metadata.get("macqwen_session") if isinstance(file_metadata, dict) else None
            if isinstance(embedded, bytes):
                embedded = embedded.decode("utf-8")
            if embedded:
                try:
                    metadata = json.loads(embedded)
                except (TypeError, ValueError) as exc:
                    raise ValueError("embedded session metadata is invalid") from exc
            else:
                try:
                    metadata = json.loads((directory / "meta.json").read_text())
                except (OSError, TypeError, ValueError) as exc:
                    raise ValueError("session metadata is missing or invalid") from exc
            if not isinstance(metadata, dict):
                raise ValueError("session metadata is not an object")
            tape_data = metadata.get("tape")
            if not isinstance(tape_data, list):
                raise ValueError("session metadata has no token tape")
            tape = [int(token) for token in tape_data]
            if metadata.get("format") == _SESSION_FORMAT:
                records = metadata.get("caches")
                if not isinstance(records, list) or len(records) != len(self.cache):
                    raise ValueError("saved cache list does not match this model")
            elif "format" not in metadata:
                records = self._legacy_records(tensors, tape)
            else:
                raise ValueError(f"unsupported session format {metadata.get('format')!r}")

            for record, live_cache in zip(records, self.cache):
                temporary.append(self._new_cache(record, live_cache, tensors))
            self._validate_cache(temporary, tape)

            old_cache = self.engine.cache
            self.engine.cache = temporary
            temporary = []
            for cache in old_cache:
                if hasattr(cache, "close"):
                    cache.close()
            self.engine.tape = tape
            self.engine.pending = []
            self.engine.turn_closed = bool(metadata.get("turn_closed", True))
            self.thinking_enabled = bool(metadata.get("thinking", False))
            self.engine.stats.clear()
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            for cache in temporary:
                if hasattr(cache, "close"):
                    cache.close()
            return f"could not load session: {exc}"
        return f"loaded {name}  {len(self.tape)} tokens restored, no old prefill"

    def list_sessions(self) -> str:
        try:
            directories = sorted(path for path in self._session_dir.iterdir() if path.is_dir())
        except FileNotFoundError:
            return "no saved sessions"
        except OSError as exc:
            return f"could not list sessions: {exc}"
        rows = []
        for directory in directories:
            try:
                count = len(json.loads((directory / "meta.json").read_text())["tape"])
                rows.append(f"  {directory.name:<24} {count:>7} tok")
            except (OSError, KeyError, TypeError, ValueError):
                rows.append(f"  {directory.name:<24} invalid")
        return "\n".join(rows) if rows else "no saved sessions"

    def delete_session(self, name: str) -> str:
        try:
            directory = self._path(name)
            if not directory.exists():
                return f"{name} not found"
            (directory / "cache.safetensors").unlink(missing_ok=True)
            (directory / "meta.json").unlink(missing_ok=True)
            directory.rmdir()
        except (OSError, ValueError) as exc:
            return f"could not delete session: {exc}"
        return f"{name} deleted"

    def reset(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        for cache in self.cache:
            if hasattr(cache, "close"):
                cache.close()
        options = self._cache_options
        if options["paged"]:
            from models.qwen27b.paged_kv import make_paged_cache

            self.engine.cache = make_paged_cache(
                self.engine.model,
                options["page_size"],
                top_k_pages=options["top_k_pages"],
                pinned_pages=1,
                recent_pages=2,
                refresh_every=16,
                min_context=options["min_context"],
                spill_dir=options["spill_dir"],
                resident_pages=options["resident_pages"],
            )
        else:
            self.engine.cache = make_prompt_cache(self.engine.model)
        self.engine.tape = []
        self.engine.pending = []
        self.engine.turn_closed = True
        self.engine.turn = 0
        self.engine.stats.clear()
        mx.clear_cache()
