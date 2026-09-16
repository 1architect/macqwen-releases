"""Runtime routing profiles for the Flash-Next backend."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os


PROFILES = (
    "standard", "fast", "fast-quality", "exact-quality", "cache-aware",
    "fused-quality",
)
# The profile picks the read path. `fast` and `fast-quality` were measured on
# `shared_mmap` and keep it. Every other profile takes the store's default, so
# FLASHNEXT_READ reaches the chat instead of being overwritten here.
DEFAULT_READ_MODE = os.environ.get("FLASHNEXT_READ", "pread")
READ_MODES = ("pread", "preadv", "resident", "shared_mmap", "hybrid")
# Turn one decodes on a cold page cache and is the slowest turn of a session.
# The expert set a session pins is stable, so recording it and reading those
# rows once at load puts them in the cache before the user types. The mlock
# is not what matters here: the cached pages survive `unpin_all`, so this
# stays independent of the per-turn pin cycle.
def pin_cache_path() -> str:
    return os.path.expanduser(
        os.environ.get("FLASHNEXT_PIN_CACHE", "~/.cache/flashnext/pins.json")
    )
# A pinned expert costs 3000 KB per layer: 2400 of quantised weight and 600
# of scales and biases. Pinning whole experts therefore exhausts a 6 GB budget
# at about 32 of 512, which reaches roughly 70% of accesses. Pinning only the
# scales and biases costs 600 KB, so the same budget reaches about 128 experts
# and roughly 93% of accesses, for the 20% of bytes those tensors represent.
# Whether that trade pays is unmeasured: set FLASHNEXT_PIN_PARTS=scales and
# compare with bench_production.
PIN_PARTS = {
    "all": ("weight", "scales", "biases"),
    "scales": ("scales", "biases"),
}


def pin_parts() -> tuple:
    """Which tensors of an expert get locked into memory."""
    choice = os.environ.get("FLASHNEXT_PIN_PARTS", "all")
    return PIN_PARTS.get(choice, PIN_PARTS["all"])


# Cache-aware routing. Off unless FLASHNEXT_SWAP_RESIDENT is set, because it
# changes what the model computes and is gated by the reasoning quality gate
# rather than by token identity.
def swap_epsilon() -> float:
    return float(os.environ.get("FLASHNEXT_SWAP_EPSILON", "0.02"))


def swap_enabled() -> bool:
    return os.environ.get("FLASHNEXT_SWAP_RESIDENT") == "1"


def swap_max_rows() -> int:
    """Largest batch the swap runs on. Prefill routes exactly above it."""
    return int(os.environ.get("FLASHNEXT_SWAP_MAX_ROWS", "4"))


def prewarm_enabled() -> bool:
    """Read at call time. A module-level constant cannot be flipped by a
    benchmark, because Python caches the module after the first import."""
    return os.environ.get("FLASHNEXT_PREWARM") == "1"


def checkpoint_identity_for_store(store) -> str | None:
    """Return and memoize the identity of ``store``'s checkpoint.

    Pin profiles contain expert row IDs, so they are only safe to reuse when
    they came from this exact checkpoint.  This is deliberately independent
    of the experimental G64 flag: the normal MLX reference path also writes
    and consumes pin profiles.
    """
    identity = getattr(store, "_flashnext_checkpoint_identity", None)
    if identity:
        return str(identity)
    model_dir = getattr(store, "dir", None)
    if not model_dir:
        return None
    try:
        from .slab_pack import checkpoint_identity

        identity = checkpoint_identity(model_dir)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    try:
        store._flashnext_checkpoint_identity = identity
    except AttributeError:
        pass
    return str(identity)


def _switch_prefixes(store) -> dict[str, str]:
    """Return the loader's streamed expert prefixes keyed by layer.

    ``SafeTensorStore.refs`` is the source of truth for which expert tensors
    exist.  In particular, do not infer a layout from ``config.json`` here:
    mixed-layout exports are valid and the loader inspects each layer's
    tensors when it replaces that layer.
    """
    refs = getattr(store, "refs", None)
    if refs is None:
        return {}
    suffix = ".mlp.switch_mlp.gate_proj.weight"
    result = {}
    for key in refs:
        if not isinstance(key, str) or not key.endswith(suffix):
            continue
        prefix = key[: -len(".gate_proj.weight")]
        marker = "language_model.model.layers."
        if not prefix.startswith(marker):
            continue
        layer_and_rest = prefix[len(marker) :]
        layer, separator, rest = layer_and_rest.partition(".")
        if not separator or rest != "mlp.switch_mlp":
            continue
        try:
            layer_id = int(layer)
        except (TypeError, ValueError):
            continue
        if layer_id >= 0:
            result[str(layer_id)] = prefix
    return result


def _quantization_layouts(store) -> dict[str, dict] | None:
    """Infer every streamed expert layout using the loader's shape logic."""
    from .loader import infer_switch_quantization

    prefixes = _switch_prefixes(store)
    # Small test stores and older callers may expose shape() without refs.
    # Keep that compatibility fallback, but still use the exact loader helper.
    if not prefixes:
        prefixes = {
            "0": "language_model.model.layers.0.mlp.switch_mlp",
        }
    layouts = {}
    for layer, prefix in prefixes.items():
        try:
            group_size, bits = infer_switch_quantization(store, prefix)
        except (
            AttributeError, IndexError, KeyError, TypeError, ValueError,
            ZeroDivisionError,
        ):
            # A declared streamed layer with incomplete metadata is not safe
            # provenance.  Fail closed rather than silently dropping it.
            if _switch_prefixes(store):
                return None
            continue
        layouts[str(layer)] = {
            "group_size": int(group_size),
            "bits": int(bits),
        }
    return layouts or None


def quantization_for_store(store) -> dict | None:
    """Describe all expert layouts from the tensors the loader will use."""
    cached = getattr(store, "_flashnext_pin_quantization", None)
    if cached:
        return dict(cached)
    layouts = _quantization_layouts(store)
    if not layouts:
        return None
    group_sizes = {item["group_size"] for item in layouts.values()}
    bits = {item["bits"] for item in layouts.values()}
    result = {"layouts": layouts}
    # Retain the flat fields for single-layout profile readers.  A mixed
    # export deliberately has no singular group size, so compatibility checks
    # must use the complete per-layer map.
    if len(group_sizes) == 1:
        result["group_size"] = next(iter(group_sizes))
    if len(bits) == 1:
        result["bits"] = next(iter(bits))
    try:
        store._flashnext_pin_quantization = dict(result)
    except AttributeError:
        pass
    return result


def _normalized_layouts(value) -> dict[str, tuple[int, int]] | None:
    if not isinstance(value, dict):
        return None
    normalized = {}
    for layer, metadata in value.items():
        if not isinstance(metadata, dict):
            return None
        try:
            layer_id = str(int(layer))
            group_size = int(metadata["group_size"])
            bits = int(metadata["bits"])
        except (KeyError, TypeError, ValueError):
            return None
        if int(layer_id) < 0 or group_size <= 0 or bits <= 0:
            return None
        normalized[layer_id] = (group_size, bits)
    return normalized


def pin_profile_compatible(
    store,
    profile: dict | None,
    expected_group_size: int | None = None,
    layer_ids=None,
) -> tuple[bool, str | None]:
    """Check provenance before a persisted expert set can be used.

    ``layer_ids`` narrows validation for callers that consume one known layer
    (or a known subset of layers).  Without it, all recorded layouts must
    agree with the current store, which remains the safe default for global
    packed allocations.
    """
    if not isinstance(profile, dict):
        return False, "pin history is missing"
    recorded_identity = profile.get(
        "checkpoint_identity", profile.get("model_identity")
    )
    if not recorded_identity:
        return False, "pin history has no checkpoint identity"
    current_identity = checkpoint_identity_for_store(store)
    if not current_identity:
        return False, "cannot verify checkpoint identity"
    if str(recorded_identity) != str(current_identity):
        return False, "pin history belongs to another checkpoint"

    recorded_quantization = profile.get("quantization") or {}
    if not isinstance(recorded_quantization, dict):
        return False, "pin history has invalid quantization metadata"
    recorded_group_size = None
    if "layouts" not in recorded_quantization:
        recorded_group_size = recorded_quantization.get(
            "group_size",
            profile.get("group_size", profile.get("quantization_group_size")),
        )
        try:
            recorded_group_size = int(recorded_group_size)
        except (TypeError, ValueError):
            return False, "pin history has no quantization group size"
    current_quantization = quantization_for_store(store)
    if not current_quantization:
        return False, "cannot verify checkpoint quantization"
    current_layouts = _normalized_layouts(current_quantization.get("layouts"))
    if current_layouts is None:
        return False, "cannot verify checkpoint quantization"
    if layer_ids is not None:
        try:
            selected_layers = {str(int(layer)) for layer in layer_ids}
        except (TypeError, ValueError):
            return False, "pin history has invalid layer metadata"
        if not selected_layers or not selected_layers.issubset(current_layouts):
            return False, "pin history has incomplete quantization metadata"
        checked_layouts = {
            layer: current_layouts[layer] for layer in selected_layers
        }
    else:
        checked_layouts = current_layouts
    current_groups = {group for group, _bits in checked_layouts.values()}
    if expected_group_size is not None:
        if current_groups != {int(expected_group_size)}:
            return False, "pin history has incompatible quantization"

    recorded_layouts = _normalized_layouts(recorded_quantization.get("layouts"))
    if recorded_layouts is not None:
        if layer_ids is not None:
            if not selected_layers.issubset(recorded_layouts):
                return False, "pin history has incomplete quantization metadata"
            recorded_checked = {
                layer: recorded_layouts[layer] for layer in selected_layers
            }
        else:
            recorded_checked = recorded_layouts
        if recorded_checked != checked_layouts:
            return False, "pin history has incompatible quantization"
    else:
        # Flat metadata is safe only for a genuinely uniform current export.
        # A legacy profile cannot prove that its per-layer IDs fit a mixed
        # layout, even when its aggregate group size happens to match.  When a
        # caller names one layer, only that layer is consumed and can be
        # checked against the legacy flat value.
        if len(current_groups) != 1:
            return False, "pin history has incompatible quantization"
        current_group_size = next(iter(current_groups))
        if recorded_group_size != current_group_size:
            return False, "pin history has incompatible quantization"
        for field in ("bits", "mode"):
            if field in recorded_quantization:
                if current_quantization.get(field) != recorded_quantization[field]:
                    return False, "pin history has incompatible quantization"
    return True, None
NEXT_TURN_THINK = (
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
NEXT_TURN_DIRECT = (
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


class RoutingProfile:
    def __init__(
        self,
        mode,
        store,
        language,
        threshold=0.85,
        resident_experts=32,
        tail_experts=6,
        warmup=8,
        pin_budget_gb=6.0,
        swap_epsilon_value=0.02,
    ):
        if mode not in PROFILES:
            raise ValueError(f"unknown Flash-Next routing profile: {mode}")
        self.mode = mode
        self.store = store
        self.language = language
        self.threshold = 0.20 if mode == "fast" else float(threshold)
        self.resident_experts = int(resident_experts)
        self.tail_experts = int(tail_experts)
        self.warmup = int(warmup)
        self.pin_budget = float(pin_budget_gb) * 1e9
        self.swap_epsilon = float(swap_epsilon_value)
        self.default_renorm = float(os.environ.get(
            "FLASHNEXT_RENORM_BLEND",
            "1" if os.environ.get("FLASHNEXT_RENORM", "1") == "1" else "0",
        ))
        self.candidates = {}
        self.observed = {}
        self.pinned = {}
        self.pinned_bytes = 0
        self.pinned_signature = ""
        self._saved_signature = ""
        self._prefixes: dict = {}
        # Collect provenance for every runtime, including the normal MLX
        # reference path.  Restricting this to experimental G64 made the
        # reference profile unsafe to reuse after a checkpoint switch.
        self._checkpoint_identity = checkpoint_identity_for_store(self.store)
        saved_read_mode = getattr(
            self.store, "_flashnext_requested_read_mode", None
        )
        self._requested_read_mode = (
            saved_read_mode
            if saved_read_mode in READ_MODES
            else self._read_mode()
        )
        self._read_mode_was_forced = getattr(
            self.store, "_flashnext_read_mode_forced", None
        )
        self._effective_read_mode = None
        self.reset()

    def _read_mode(self) -> str:
        value = getattr(self.store, "_read_mode", DEFAULT_READ_MODE)
        if value in READ_MODES:
            return value
        return DEFAULT_READ_MODE if DEFAULT_READ_MODE in READ_MODES else "pread"

    @property
    def quality(self):
        return self.mode in (
            "fast-quality", "exact-quality", "cache-aware", "fused-quality",
        )

    @property
    def cache_aware(self):
        """Whether routing can select a close resident expert."""
        return self.mode == "cache-aware" or swap_enabled()

    def reset(self):
        from models.flashnext.adaptive_topk import (
            set_fast_profile,
            set_layer_thresholds,
            set_min_keep,
            set_renorm_blend,
            set_resident_experts,
            set_route_observer,
            set_threshold,
        )

        self.store.unpin_all()
        set_resident_experts(None)
        set_route_observer(None)
        set_min_keep(1)
        set_layer_thresholds({})
        set_threshold(self.threshold)
        set_renorm_blend(
            self.default_renorm if self.mode == "standard" else 1.0
        )
        current_read_mode = self._read_mode()
        if self.mode != "fast":
            # Keep benchmark/live writes made directly on the store. A
            # profile-forced effective mode must not become the next request.
            if (
                self._effective_read_mode is None
                and (
                    self._read_mode_was_forced != "shared_mmap"
                    or current_read_mode != "shared_mmap"
                )
            ) or (
                self._effective_read_mode is not None
                and current_read_mode != self._effective_read_mode
            ):
                self._requested_read_mode = current_read_mode
            self.store._read_mode = self._requested_read_mode
            self._read_mode_was_forced = None
        else:
            self.store._read_mode = "shared_mmap"
            self._read_mode_was_forced = "shared_mmap"
        self._effective_read_mode = self.store._read_mode
        self.store._flashnext_requested_read_mode = self._requested_read_mode
        self.store._flashnext_read_mode_forced = self._read_mode_was_forced
        if self.mode == "fast":
            set_fast_profile()
        self.candidates = {layer: Counter() for layer in range(48)}
        self.route_counts = {layer: Counter() for layer in range(48)}
        self.observed = {layer: 0 for layer in range(48)}
        self.pinned.clear()
        self.pinned_bytes = 0
        self.pinned_signature = ""

    def session_profile(self, stops):
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "stop_ids": sorted(int(value) for value in stops if value is not None),
            "prompt_protocol": {
                "first_turn": "tokenizer.apply_chat_template",
                "next_turn_direct": NEXT_TURN_DIRECT,
                "next_turn_think": NEXT_TURN_THINK,
            },
            "renorm": (
                {"warmup": 1.0, "tail": 0.1}
                if self.mode == "fast-quality"
                else 0.0 if self.mode == "fast" else self.default_renorm
                if self.mode == "standard" else 1.0
            ),
            "tail_warmup": self.warmup if self.quality else None,
            "tail_experts": self.tail_experts if self.mode == "fast-quality" else None,
            "resident_experts": (
                self.resident_experts
                if self.mode in ("exact-quality", "cache-aware", "fused-quality")
                else None
            ),
            "swap_epsilon": self.swap_epsilon if self.cache_aware else None,
            "pin_budget_gb": self.pin_budget / 1e9 if self.quality else None,
            "mtp_depth": 0,
            "speculative_fast": False,
            "draft_depth": None,
            "draft_model": None,
            "fused_quality": False,
            "fusion_block": None,
            "fusion_alpha": None,
            "fusion_min_margin": None,
            "fusion_min_block": None,
            "fusion_margin_tokens": None,
            "fusion_max_prompt": None,
        }

    def _expert_resident(self, layer: int, expert: int) -> bool:
        """Whether every tensor of this expert is already in memory."""
        prefix = self._prefixes.get(layer)
        if prefix is None:
            return False
        believed = self.store.believed_resident
        return (
            believed(f"{prefix}.gate_proj.weight", expert)
            and believed(f"{prefix}.up_proj.weight", expert)
            and believed(f"{prefix}.down_proj.weight", expert)
        )

    def begin_decode(self):
        from models.flashnext.adaptive_topk import (
            set_route_observer,
            set_swap_resident,
        )

        if self.cache_aware:
            if not self._prefixes:
                for index, layer in enumerate(self.language.model.layers):
                    block = getattr(layer.mlp, "switch_mlp", None)
                    if block is not None:
                        self._prefixes[index] = (
                            block.gate_proj.cache.prefix.rsplit(".", 1)[0]
                        )
            epsilon = swap_epsilon() if swap_enabled() else self.swap_epsilon
            set_swap_resident(
                self._expert_resident, epsilon, swap_max_rows()
            )
        if not self.quality:
            return
        # `_observe` stops at `warmup` rows per layer, so a prefill batch of
        # one row per prompt token gives it nothing extra.
        set_route_observer(self._observe, self.warmup)

    def _fast_keep(self, scores, threshold):
        total = sum(scores)
        accumulated = 0.0
        for position, score in enumerate(scores):
            accumulated += score / total
            if accumulated >= threshold:
                return position + 1
        return len(scores)

    def _observe(self, layer, experts, scores, keeps):
        from models.flashnext.adaptive_topk import FAST_LAYERS

        threshold = 0.40 if layer in FAST_LAYERS else 0.20
        for expert_row, score_row, normal_keep in zip(experts, scores, keeps):
            if self.observed[layer] >= self.warmup:
                break
            self.observed[layer] += 1
            mass = sum(score_row)
            if self.mode in ("exact-quality", "cache-aware", "fused-quality"):
                selected = zip(expert_row[:normal_keep], score_row[:normal_keep])
            else:
                keep = self._fast_keep(score_row, threshold)
                selected = zip(expert_row[keep:], score_row[keep:])
            for expert, score in selected:
                self.candidates[layer][expert] += score / mass
                if hasattr(self, "route_counts"):
                    self.route_counts[layer][expert] += 1

    def after_token(self, count, generation_limit):
        # The first output token comes from prefill. Output N+1 therefore
        # arrives after N decode routes have been observed. Pin after output
        # warmup+1 so the candidate pool contains the requested observations.
        if (
            not self.quality
            or count != self.warmup + 1
            or count >= generation_limit
        ):
            return False
        from models.flashnext.adaptive_topk import (
            FAST_LAYERS,
            set_layer_thresholds,
            set_renorm_blend,
            set_resident_experts,
            set_route_observer,
            set_threshold,
        )

        set_route_observer(None)
        resident = self._pin_candidates()
        self.save_pins()
        self._apply_pinned_settings(resident)
        return True

    def _apply_pinned_settings(self, resident):
        """The routing settings that belong with a pinned expert set."""
        from models.flashnext.adaptive_topk import (
            FAST_LAYERS,
            set_layer_thresholds,
            set_renorm_blend,
            set_resident_experts,
            set_threshold,
        )

        if self.mode == "fast-quality":
            set_resident_experts(resident)
            self.store._read_mode = "shared_mmap"
            self._read_mode_was_forced = "shared_mmap"
            self._effective_read_mode = "shared_mmap"
            self.store._flashnext_read_mode_forced = "shared_mmap"
            set_threshold(0.20)
            set_layer_thresholds({layer: 0.40 for layer in FAST_LAYERS})
            set_renorm_blend(0.10)

    def save_pins(self) -> None:
        """Record the pinned set so the next session can warm it.

        This runs inside a decode. The set is identical on almost every turn,
        so write only when the signature moves, and never spend a disk write
        on the hot path for a file that already says the same thing.
        """
        if not self.pinned or self.pinned_signature == self._saved_signature:
            return
        cache_file = pin_cache_path()
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            ranked_scores = {}
            ranked_counts = {}
            if self.candidates:
                for layer, counts in self.candidates.items():
                    ranked_scores[str(layer)] = [
                        (int(e), round(float(s), 4))
                        for e, s in counts.most_common(32)
                    ]
            if hasattr(self, "route_counts") and self.route_counts:
                for layer, counts in self.route_counts.items():
                    ranked_counts[str(layer)] = [
                        (int(e), int(c))
                        for e, c in counts.most_common(32)
                    ]
            payload = {
                "mode": self.mode,
                "resident_experts": self.resident_experts,
                "layers": {
                    str(layer): [
                        e for e, _ in ranked_scores.get(str(layer), []) if e in experts
                    ] or sorted(experts)
                    for layer, experts in self.pinned.items()
                },
                "ranked_scores": ranked_scores,
                "ranked_counts": ranked_counts,
            }
            # Pin IDs are only reusable when their checkpoint and quantization
            # layout are known. Do not write an unprovenanceable profile: an
            # old ID-only file is precisely the cross-checkpoint bug this
            # metadata prevents.
            quantization = quantization_for_store(self.store)
            if not self._checkpoint_identity or not quantization:
                return
            payload["checkpoint_identity"] = self._checkpoint_identity
            payload["quantization"] = quantization
            # Keep the flat key for readers of the existing profile format.
            if "group_size" in quantization:
                payload["group_size"] = int(quantization["group_size"])
            with open(cache_file, "w") as handle:
                json.dump(payload, handle)
            self._saved_signature = self.pinned_signature
        except OSError:
            pass

    def prewarm(self) -> int:
        """Read last session's expert set so turn one is not cold.

        Returns bytes paged into the cache, zero when no cache exists.
        """
        try:
            with open(pin_cache_path(), "r") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return 0
        compatible, _reason = pin_profile_compatible(self.store, payload)
        if not compatible:
            return 0
        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            return 0
        # Build and budget the complete plan before calling pin_rows. This
        # prevents a profile that exceeds the current process budget from
        # consuming rows from the first layers and only failing later.
        plans = []
        pinned = 0
        for layer_str, experts in layers.items():
            try:
                layer = int(layer_str)
            except (TypeError, ValueError):
                continue
            if layer >= len(self.language.model.layers):
                continue
            if layer < 0 or not isinstance(experts, list):
                continue
            block = getattr(
                getattr(self.language.model.layers[layer], "mlp", None),
                "switch_mlp", None,
            )
            if block is None:
                continue
            try:
                prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
            except AttributeError:
                continue
            names = [
                f"{prefix}.{projection}.{part}"
                for projection in ("gate_proj", "up_proj", "down_proj")
                for part in pin_parts()
            ]
            try:
                expert_count = int(self.store.shape(f"{prefix}.gate_proj.weight")[0])
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                continue
            valid_experts = []
            seen = set()
            for value in experts:
                try:
                    expert = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= expert < expert_count and expert not in seen:
                    seen.add(expert)
                    valid_experts.append(expert)
            allowed = []
            selected_bytes = 0
            try:
                for expert in valid_experts:
                    size = sum(self.store.pin_size(name, [expert]) for name in names)
                    if pinned + selected_bytes + size > self.pin_budget:
                        break
                    allowed.append(expert)
                    selected_bytes += size
            except (AttributeError, KeyError, TypeError, ValueError, OSError):
                return 0
            if allowed:
                plans.append((layer, names, allowed))
                pinned += selected_bytes

        if pinned > self.pin_budget:
            return 0
        actual_pinned = 0
        try:
            for layer, names, experts in plans:
                for name in names:
                    actual_pinned += self.store.pin_rows(name, experts)
                self.pinned.setdefault(layer, set()).update(experts)
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            self.store.unpin_all()
            self.pinned.clear()
            self.pinned_bytes = 0
            return 0
        self.pinned_bytes = pinned
        return actual_pinned

    def _pin_candidates(self):
        count = (
            self.resident_experts
            if self.mode in ("exact-quality", "cache-aware", "fused-quality")
            else self.tail_experts
        )
        ranked = {
            layer: [expert for expert, _score in values.most_common(count)]
            for layer, values in self.candidates.items()
        }
        pinned = 0
        try:
            for layer_number, experts in ranked.items():
                fresh = [
                    expert for expert in experts
                    if expert not in self.pinned.get(layer_number, set())
                ]
                if not fresh:
                    continue
                block = self.language.model.layers[layer_number].mlp.switch_mlp
                prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
                names = [
                    f"{prefix}.{projection}.{part}"
                    for projection in ("gate_proj", "up_proj", "down_proj")
                    for part in pin_parts()
                ]
                allowed = []
                selected_bytes = 0
                for expert in fresh:
                    size = sum(
                        self.store.pin_size(name, [expert]) for name in names
                    )
                    if pinned + selected_bytes + size > self.pin_budget:
                        break
                    allowed.append(expert)
                    selected_bytes += size
                if not allowed:
                    continue
                for name in names:
                    pinned += self.store.pin_rows(name, allowed)
                self.pinned.setdefault(layer_number, set()).update(allowed)
        except OSError:
            self.store.unpin_all()
            self.pinned.clear()
            raise
        encoded = repr(
            [(layer, sorted(experts)) for layer, experts in sorted(self.pinned.items())]
        ).encode()
        self.pinned_signature = hashlib.sha256(encoded).hexdigest()[:16]
        self.pinned_bytes = pinned
        return {layer: set(experts) for layer, experts in self.pinned.items()}

    def finish_decode(self):
        from models.flashnext.adaptive_topk import (
            set_route_observer,
            set_swap_resident,
        )

        set_route_observer(None)
        set_swap_resident(None)
