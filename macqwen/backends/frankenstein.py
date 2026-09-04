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

from models.qwen27b.settings import get_registry


_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


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

        text, old = self.engine.generate(
            max_tokens=max_tokens,
            echo=False,
            progress=progress,
            on_token=on_token,
        )
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

    def save_session(self, name: str) -> str:
        try:
            import mlx.core as mx

            directory = self._path(name)
            directory.mkdir(parents=True, exist_ok=True)
            tensors = {}
            for index, cache in enumerate(self.cache):
                state = cache.state
                arrays = state if isinstance(state, (list, tuple)) else [state]
                for part, array in enumerate(arrays):
                    if array is not None:
                        tensors[f"c{index}_{part}"] = array
            mx.save_safetensors(str(directory / "cache.safetensors"), tensors)
            metadata = {
                "tape": self.tape,
                "started": bool(self.tape),
                "turn_closed": self.engine.turn_closed,
                "thinking": self.thinking_enabled,
            }
            (directory / "meta.json").write_text(json.dumps(metadata))
        except (OSError, TypeError, ValueError) as exc:
            return f"could not save session: {exc}"
        size_mb = sum(value.nbytes for value in tensors.values()) / 1e6
        return f"saved {name}  {len(self.tape)} tokens, {size_mb:.0f} MB"

    def load_session(self, name: str) -> str:
        try:
            import mlx.core as mx

            directory = self._path(name)
            metadata = json.loads((directory / "meta.json").read_text())
            tensors = mx.load(str(directory / "cache.safetensors"))
            self.reset()
            for index, cache in enumerate(self.cache):
                arrays = []
                part = 0
                while f"c{index}_{part}" in tensors:
                    arrays.append(tensors[f"c{index}_{part}"])
                    part += 1
                if arrays:
                    cache.state = arrays if len(arrays) > 1 else arrays[0]
            self.engine.tape = [int(token) for token in metadata["tape"]]
            self.engine.pending = []
            self.engine.turn_closed = bool(metadata.get("turn_closed", True))
            self.thinking_enabled = bool(metadata.get("thinking", False))
            if not self.check_invariant():
                raise ValueError("saved cache and token tape disagree")
        except (OSError, KeyError, TypeError, ValueError) as exc:
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
