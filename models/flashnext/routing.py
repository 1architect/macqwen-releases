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
# Turn one decodes on a cold page cache and is the slowest turn of a session.
# The expert set a session pins is stable, so recording it and reading those
# rows once at load puts them in the cache before the user types. The mlock
# is not what matters here: the cached pages survive `unpin_all`, so this
# stays independent of the per-turn pin cycle.
PIN_CACHE = os.path.expanduser(
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


def prewarm_enabled() -> bool:
    """Read at call time. A module-level constant cannot be flipped by a
    benchmark, because Python caches the module after the first import."""
    return os.environ.get("FLASHNEXT_PREWARM") == "1"
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
        self.reset()

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
        self.store._read_mode = DEFAULT_READ_MODE
        if self.mode == "fast":
            set_fast_profile()
            self.store._read_mode = "shared_mmap"
        self.candidates = {layer: Counter() for layer in range(48)}
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
            set_swap_resident(self._expert_resident, epsilon)
        if not self.quality:
            return
        set_route_observer(self._observe)

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
        try:
            os.makedirs(os.path.dirname(PIN_CACHE), exist_ok=True)
            payload = {
                "mode": self.mode,
                "resident_experts": self.resident_experts,
                "layers": {
                    str(layer): sorted(experts)
                    for layer, experts in self.pinned.items()
                },
            }
            with open(PIN_CACHE, "w") as handle:
                json.dump(payload, handle)
            self._saved_signature = self.pinned_signature
        except OSError:
            pass

    def prewarm(self) -> int:
        """Read last session's expert set so turn one is not cold.

        Returns the number of rows touched. Results are discarded, so this
        cannot change what the model computes.
        """
        try:
            with open(PIN_CACHE) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return 0
        if payload.get("mode") != self.mode:
            return 0
        touched = 0
        for key, experts in payload.get("layers", {}).items():
            try:
                layer = int(key)
                block = self.language.model.layers[layer].mlp.switch_mlp
            except (ValueError, AttributeError, IndexError):
                continue
            prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    try:
                        self.store.rows_np(
                            f"{prefix}.{projection}.{part}", list(experts)
                        )
                        touched += len(experts)
                    except (KeyError, OSError):
                        pass
        return touched

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
