#!/usr/bin/env python3
"""Measure how much expert selection can improve with perfect information.

This is an oracle study, not a speed benchmark. During measured decode steps,
it evaluates every routed top-k expert before choosing a subset. It compares:

* the current cumulative router-score threshold;
* ranking by each expert's leave-one-out output change;
* the best possible subset, found by exhaustive search over at most 2^10 sets.

The reconstruction target is the full top-k routed mixture. Shared-expert output
is excluded because every policy keeps it unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("FLASHNEXT_RENORM", "1")

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from macqwen.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import (
    set_layer_thresholds,
    set_renorm_blend,
    set_threshold,
)
from models.flashnext.loader import load_streaming


MODEL = str(resolve_flashnext())
PROMPT = (
    "Explique em cerca de 200 palavras como a fotossintese transforma luz "
    "solar em energia quimica."
)


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def _percentile(rows, key, percentile):
    if not rows:
        return 0.0
    return float(np.percentile([row[key] for row in rows], percentile))


def _rank_correlation(left, right):
    count = len(left)
    if count < 2:
        return 1.0
    left_rank = np.empty(count, dtype=np.float64)
    right_rank = np.empty(count, dtype=np.float64)
    left_rank[np.argsort(left, kind="stable")] = np.arange(count)
    right_rank[np.argsort(right, kind="stable")] = np.arange(count)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


class ContributionOracle:
    """Analyze routed mixtures after their expert outputs are available."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.rows = []
        self.by_layer = defaultdict(list)
        self._masks = {}

    def _subset_masks(self, count):
        cached = self._masks.get(count)
        if cached is not None:
            return cached
        values = np.arange(1, 1 << count, dtype=np.uint16)[:, None]
        masks = ((values >> np.arange(count)) & 1).astype(np.float64)
        counts = masks.sum(axis=1).astype(np.int16)
        self._masks[count] = (masks, counts)
        return masks, counts

    def observe(self, layer, expert_ids, scores, outputs):
        width = scores.shape[-1]
        ids = expert_ids.reshape(-1, width)
        score_rows = scores.reshape(-1, width)
        output_rows = outputs.reshape(-1, width, outputs.shape[-1])
        for row_ids, row_scores, row_outputs in zip(ids, score_rows, output_rows):
            result = self._analyze_row(layer, row_ids, row_scores, row_outputs)
            self.rows.append(result)
            self.by_layer[layer].append(result)

    def _analyze_row(self, layer, expert_ids, scores, outputs):
        count = len(scores)
        weights = scores.astype(np.float64)
        weights /= weights.sum()
        vectors = outputs.astype(np.float64)
        gram = vectors @ vectors.T
        full_norm_sq = max(float(weights @ gram @ weights), 1e-24)

        masks, counts = self._subset_masks(count)
        subset_mass = masks @ weights
        coefficients = masks * weights[None, :] / subset_mass[:, None]
        differences = coefficients - weights[None, :]
        error_sq = np.einsum(
            "bi,ij,bj->b", differences, gram, differences, optimize=True
        )
        errors = np.sqrt(np.maximum(error_sq, 0.0) / full_norm_sq)

        score_k = int(
            np.searchsorted(np.cumsum(weights), self.threshold, side="left") + 1
        )
        score_mask = np.zeros(count, dtype=np.float64)
        score_mask[:score_k] = 1.0
        score_index = int(np.flatnonzero(np.all(masks == score_mask, axis=1))[0])
        score_error = float(errors[score_index])

        # This is the real local effect of one expert. Remove it, renormalize
        # the remaining router weights, then measure the routed-output change.
        contribution = np.empty(count, dtype=np.float64)
        full_mask = np.ones(count, dtype=np.float64)
        for position in range(count):
            without = full_mask.copy()
            without[position] = 0.0
            without_index = int(
                np.flatnonzero(np.all(masks == without, axis=1))[0]
            )
            contribution[position] = errors[without_index]
        contribution_order = np.argsort(-contribution, kind="stable")

        same_k_mask = np.zeros(count, dtype=np.float64)
        same_k_mask[contribution_order[:score_k]] = 1.0
        same_k_index = int(
            np.flatnonzero(np.all(masks == same_k_mask, axis=1))[0]
        )

        contribution_total = max(float(contribution.sum()), 1e-24)
        ordered_contribution = contribution[contribution_order]
        contribution_k = int(
            np.searchsorted(
                np.cumsum(ordered_contribution) / contribution_total,
                self.threshold,
                side="left",
            )
            + 1
        )
        contribution_mask = np.zeros(count, dtype=np.float64)
        contribution_mask[contribution_order[:contribution_k]] = 1.0
        contribution_index = int(
            np.flatnonzero(np.all(masks == contribution_mask, axis=1))[0]
        )

        best_by_count = {
            size: float(errors[counts == size].min())
            for size in range(1, count + 1)
        }
        allowance = score_error * (1.0 + 1e-6) + 1e-9
        oracle_k = next(
            size for size in range(1, count + 1)
            if best_by_count[size] <= allowance
        )

        return {
            "layer": int(layer),
            "score_k": score_k,
            "score_error": score_error,
            "contribution_k": contribution_k,
            "contribution_error": float(errors[contribution_index]),
            "contribution_same_k_error": float(errors[same_k_index]),
            "oracle_same_k_error": best_by_count[score_k],
            "oracle_k": oracle_k,
            "saved": score_k - oracle_k,
            "rank_correlation": _rank_correlation(weights, contribution),
            "top1_match": int(contribution_order[0] == 0),
            "score_contribution_mass": float(
                contribution[:score_k].sum() / contribution_total
            ),
            "expert_ids": [int(value) for value in expert_ids],
            "router_weights": [float(value) for value in weights],
            "ablation_contribution": [float(value) for value in contribution],
        }

    def aggregate(self, rows=None):
        rows = self.rows if rows is None else rows
        score_slots = sum(row["score_k"] for row in rows)
        saved = sum(row["saved"] for row in rows)
        return {
            "samples": len(rows),
            "score_k_mean": _mean(rows, "score_k"),
            "score_error_mean": _mean(rows, "score_error"),
            "score_error_p90": _percentile(rows, "score_error", 90),
            "contribution_k_mean": _mean(rows, "contribution_k"),
            "contribution_error_mean": _mean(rows, "contribution_error"),
            "contribution_same_k_error_mean": _mean(
                rows, "contribution_same_k_error"
            ),
            "oracle_same_k_error_mean": _mean(rows, "oracle_same_k_error"),
            "oracle_k_mean": _mean(rows, "oracle_k"),
            "saved_mean": _mean(rows, "saved"),
            "saved_percent": 100.0 * saved / score_slots if score_slots else 0.0,
            "rank_correlation_mean": _mean(rows, "rank_correlation"),
            "top1_match_percent": 100.0 * _mean(rows, "top1_match"),
            "score_contribution_mass_mean": _mean(
                rows, "score_contribution_mass"
            ),
        }

    def report(self, measured_tokens, generated_text, per_layer=False):
        summary = self.aggregate()
        print("\nContribution oracle", flush=True)
        print(
            f"measured: {measured_tokens} decode steps, "
            f"{summary['samples']} layer-token rows",
            flush=True,
        )
        print(f"router threshold: {self.threshold:.2f}", flush=True)
        print(
            "target: full top-k routed output; this does not test token identity",
            flush=True,
        )
        print(
            "router score: "
            f"{summary['score_k_mean']:.2f} experts, "
            f"{100 * summary['score_error_mean']:.2f}% mean output error, "
            f"{100 * summary['score_error_p90']:.2f}% p90",
            flush=True,
        )
        print(
            "contribution threshold: "
            f"{summary['contribution_k_mean']:.2f} experts, "
            f"{100 * summary['contribution_error_mean']:.2f}% mean output error",
            flush=True,
        )
        print(
            "contribution rank at router k: "
            f"{100 * summary['contribution_same_k_error_mean']:.2f}% mean error",
            flush=True,
        )
        print(
            "exact subset at router k: "
            f"{100 * summary['oracle_same_k_error_mean']:.2f}% mean error",
            flush=True,
        )
        print(
            "exact subset matching router error: "
            f"{summary['oracle_k_mean']:.2f} experts, "
            f"{summary['saved_percent']:.1f}% fewer slots",
            flush=True,
        )
        print(
            "router score versus contribution: "
            f"rho {summary['rank_correlation_mean']:.3f}, "
            f"top-1 match {summary['top1_match_percent']:.1f}%",
            flush=True,
        )
        print(f"generated: {generated_text!r}", flush=True)

        if per_layer:
            print(
                "\nlayer rows score_k oracle_k save% rho score_err% oracle_err%",
                flush=True,
            )
            for layer in sorted(self.by_layer):
                layer_summary = self.aggregate(self.by_layer[layer])
                print(
                    f"{layer:>5} {layer_summary['samples']:>4} "
                    f"{layer_summary['score_k_mean']:>7.2f} "
                    f"{layer_summary['oracle_k_mean']:>8.2f} "
                    f"{layer_summary['saved_percent']:>5.1f} "
                    f"{layer_summary['rank_correlation_mean']:>5.2f} "
                    f"{100 * layer_summary['score_error_mean']:>10.2f} "
                    f"{100 * layer_summary['oracle_same_k_error_mean']:>11.2f}",
                    flush=True,
                )

    def json_report(self, measured_tokens, generated_text):
        return {
            "threshold": self.threshold,
            "measured_tokens": measured_tokens,
            "generated_text": generated_text,
            "summary": self.aggregate(),
            "layers": {
                str(layer): self.aggregate(rows)
                for layer, rows in sorted(self.by_layer.items())
            },
            "rows": self.rows,
        }


def _keeps(score_rows, threshold):
    keeps = []
    for weights in score_rows:
        total = sum(weights)
        accumulated = 0.0
        keep = len(weights)
        for position, weight in enumerate(weights):
            accumulated += weight / total
            if accumulated >= threshold:
                keep = position + 1
                break
        keeps.append(keep)
    return keeps


def make_instrumented_call(oracle, threshold):
    """Build a MoE call that observes top-k, then emits threshold output."""

    def instrumented_call(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        width = self.top_k
        indices = mx.argpartition(gates, kth=-width, axis=-1)[..., -width:]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        order = mx.argsort(-scores, axis=-1)
        indices = mx.take_along_axis(indices, order, axis=-1)
        scores = mx.take_along_axis(scores, order, axis=-1)

        score_copy = scores.astype(mx.float32)
        mx.eval(score_copy)
        keeps = _keeps(score_copy.reshape(-1, width).tolist(), threshold)

        shared = self.shared_expert(x)
        shared_gate = mx.sigmoid(self.shared_expert_gate(x))
        if os.environ.get("FLASHNEXT_OVERLAP", "1") == "1":
            mx.async_eval(shared, shared_gate)

        expert_outputs = self.switch_mlp(x, indices)
        output_copy = expert_outputs.astype(mx.float32)
        mx.eval(indices, output_copy)
        oracle.observe(
            getattr(self, "_flashnext_layer_id", -1),
            np.array(indices, copy=True),
            np.array(score_copy, copy=True),
            np.array(output_copy, copy=True),
        )

        active_width = max(keeps)
        selected_scores = scores[..., :active_width]
        selected_outputs = expert_outputs[..., :active_width, :]
        keep_shape = (*selected_scores.shape[:-1], 1)
        keep_array = mx.array(keeps, dtype=mx.int32).reshape(keep_shape)
        active = mx.arange(active_width) < keep_array
        selected_scores = mx.where(active, selected_scores, 0)
        selected_scores /= selected_scores.sum(axis=-1, keepdims=True)
        routed = (selected_outputs * selected_scores[..., None]).sum(axis=-2)
        return routed + shared_gate * shared

    return instrumented_call


def main():
    parser = argparse.ArgumentParser(
        description="Compare router score with actual expert-output contribution."
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    model_path = os.path.expanduser(args.model)
    model, _, store = load_streaming(
        model_path,
        expert_capacity=0,
        verbose=True,
        keep_vision=False,
        use_mtp=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    language = model.language_model
    store._read_mode = "pread"
    store.set_mmap_advice("random")
    set_threshold(args.threshold)
    set_layer_thresholds({})
    set_renorm_blend(1.0)

    text = args.prompt if args.raw else tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = mx.array(tokenizer(text)["input_ids"])[None]
    cache = language.make_cache()
    logits = language(input_ids, cache=cache).logits
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    oracle = ContributionOracle(args.threshold)
    original_call = Qwen3_5MoeSparseMoeBlock.__call__
    Qwen3_5MoeSparseMoeBlock.__call__ = make_instrumented_call(
        oracle, args.threshold
    )
    generated = [int(token.item())]
    measured = 0
    stops = {tokenizer.eos_token_id, 248044, 248046}
    try:
        for _ in range(args.tokens):
            if int(token.item()) in stops:
                break
            logits = language(token[None], cache=cache).logits
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            generated.append(int(token.item()))
            measured += 1
    finally:
        Qwen3_5MoeSparseMoeBlock.__call__ = original_call

    generated_text = tokenizer.decode(generated)
    oracle.report(measured, generated_text, args.per_layer)
    if args.json:
        args.json.write_text(
            json.dumps(
                oracle.json_report(measured, generated_text),
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"json: {args.json}", flush=True)


if __name__ == "__main__":
    main()
