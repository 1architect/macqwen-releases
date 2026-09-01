"""Singleton-equivalent block verification for Qwen4Exp."""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update_with_states
from mlx_vlm.models.qwen3_5.speculative_verifier import (
    Qwen3_5ExactSpeculativeVerifier,
)

from . import adaptive_topk as routing


class Qwen4ExactSpeculativeVerifier(Qwen3_5ExactSpeculativeVerifier):
    """Extend the MLX singleton verifier for Qwen4 hyper-connections and PLE."""

    def _hyper(self, module, hidden):
        normed = module.hc_norm(hidden)
        mix = nn.silu(self._linear(module.input_mix_weight_down, normed) / module.hc_count)
        mix = mx.sigmoid(self._linear(module.input_mix_weight_up, mix))
        mix = mix.reshape(*mix.shape[:-1], module.hc_count, module.hidden_size)
        streams = normed.reshape(
            *normed.shape[:-1], module.hc_count, module.hidden_size
        )
        mixed = mx.mean(mix * streams, axis=-2)
        if "block_inject_weight" not in module:
            return mixed
        weights = 2 * mx.sigmoid(
            self._linear(module.block_inject_weight, normed) / module.hc_count
        )
        return mixed, hidden, weights

    def _indexer(self, indexer, hidden, cache, position_ids):
        batch, length, _ = hidden.shape
        past = cache.offset if cache is not None else 0
        if position_ids is None:
            position_ids = indexer._default_position_ids(batch, past, length)

        qk = self._linear(indexer.index_qk_proj, hidden).reshape(
            batch,
            length,
            indexer.n_heads + indexer.kv_heads,
            indexer.head_dim,
        )
        query = qk[:, :, : indexer.n_heads]
        raw_keys = qk[:, :, indexer.n_heads :].squeeze(2)
        query = indexer.q_layernorm(query).transpose(0, 2, 1, 3)
        if cache is not None:
            raw_keys, full_positions = cache.update_indexer(raw_keys, position_ids)
        else:
            full_positions = position_ids

        key_length = raw_keys.shape[1]
        complete_blocks = key_length // indexer.compress_ratio
        if complete_blocks <= indexer.block_topk:
            return None

        query = indexer._apply_rope(query, position_ids)
        complete_length = complete_blocks * indexer.compress_ratio
        pooled = raw_keys[:, :complete_length].reshape(
            batch,
            complete_blocks,
            indexer.compress_ratio,
            indexer.head_dim,
        )
        pooled = mx.expand_dims(
            indexer.k_layernorm(
                mx.mean(pooled.astype(mx.float32), axis=2).astype(raw_keys.dtype)
            ),
            axis=1,
        )
        block_starts = mx.arange(complete_blocks) * indexer.compress_ratio
        pooled = indexer._apply_rope(pooled, full_positions[..., block_starts])
        scores = query @ pooled.transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1)
        scores = scores / math.sqrt(indexer.head_dim)

        query_ends = past + mx.arange(length) + 1
        complete_counts = query_ends // indexer.compress_ratio
        valid = (
            mx.arange(complete_blocks)[None, None, :]
            < complete_counts[None, :, None]
        )
        scores = mx.where(valid, scores, -mx.inf)
        selected = mx.argpartition(
            scores, kth=-indexer.block_topk, axis=-1
        )[..., -indexer.block_topk :]
        token_indices = mx.arange(key_length)
        token_blocks = token_indices // indexer.compress_ratio
        selected_tokens = mx.any(
            token_blocks[None, None, None, :] == selected[..., None], axis=2
        )
        tail_starts = complete_counts * indexer.compress_ratio
        tail = (token_indices[None, None, :] >= tail_starts[None, :, None]) & (
            token_indices[None, None, :] < query_ends[None, :, None]
        )
        causal = token_indices[None, None, :] < query_ends[None, :, None]
        use_sparse = complete_counts > indexer.block_topk
        return mx.where(
            use_sparse[None, :, None], selected_tokens | tail, causal
        )[:, None]

    def _qwen4_attention(self, attention, hidden, mask, cache, position_ids):
        qsa_mask = self._indexer(attention.indexer, hidden, cache, position_ids)
        if qsa_mask is not None:
            if mask is None or (isinstance(mask, str) and mask == "causal"):
                mask = qsa_mask
            elif isinstance(mask, mx.array):
                if mask.dtype == mx.bool_:
                    mask = mask & qsa_mask
                else:
                    bias = mx.where(qsa_mask, 0.0, -mx.inf).astype(mask.dtype)
                    mask = mask + bias
        return self._attention(attention, hidden, mask, cache, position_ids, None)

    def _gated_delta(self, layer, inputs, mask, cache, gdn_sink):
        helpers = self._helpers()
        batch, length, _ = inputs.shape
        mixed_qkv, z, b, a = self._linears(
            (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a),
            inputs,
        )
        z = z.reshape(batch, length, -1, layer.head_v_dim)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, layer.conv_kernel_size - 1, layer.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, layer.conv_kernel_size - 1, layer.conv_dim),
                dtype=inputs.dtype,
            )
        if mask is not None:
            if mask.shape[0] != batch:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            keep = layer.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])

        conv_output = nn.silu(layer.conv1d(conv_input))
        q, k, v = [
            value.reshape(batch, length, heads, width)
            for value, heads, width in zip(
                mx.split(conv_output, [layer.key_dim, 2 * layer.key_dim], -1),
                [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
                [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        if state is not None and state.shape[0] != batch:
            state = None
        q, k = layer._normalize_qk(q, k)
        initial_state = state
        output, state, intermediate = gated_delta_update_with_states(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            state,
            mask,
            use_kernel=not layer.training,
            state_steps=length - 1,
        )
        gdn_sink.append(
            (
                q,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                initial_state,
                mask,
                conv_input,
                layer.conv_kernel_size,
                intermediate,
            )
        )
        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(length)
                helpers._qwen3_5_advance_left_padding_info(cache, length)
                helpers._qwen3_5_advance_lengths_info(cache, length)
        output = layer.norm(output, z)
        return self._linear(layer.out_proj, output.reshape(batch, length, -1))

    def _ple(self, ple, hidden, input_ids, cache, mask):
        batch = input_ids.shape[0]
        embedding = ple.ple_embedding
        if cache is not None and cache[3] is not None:
            previous_context = cache[3]
        else:
            previous_context = mx.full(
                (batch, embedding.context_len),
                embedding.eos_token_id,
                dtype=mx.int64,
            )
        token_history = mx.concatenate(
            [previous_context, input_ids.astype(mx.int64)], axis=-1
        )
        embeddings = ple.ple_embedding(input_ids, cache)
        keys = ple.norm_key(self._linear(ple.key_proj, embeddings)).reshape(
            *hidden.shape[:-1], ple.hc_count, ple.hidden_size
        )
        values = self._linear(ple.value_proj, embeddings)
        queries = ple.norm_query(hidden).reshape(
            *hidden.shape[:-1], ple.hc_count, ple.hidden_size
        )
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(
            ple.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * values[..., None, :]).reshape(*hidden.shape)
        normed = ple.norm_conv(gated)
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            gated = mx.where(mask[..., None], gated, 0)
            normed = mx.where(mask[..., None], normed, 0)
        if cache is not None and cache[2] is not None:
            conv_state = cache[2]
        else:
            conv_state = mx.zeros(
                (batch, ple.short_conv_state_len, normed.shape[-1]),
                dtype=normed.dtype,
            )
        conv_input = mx.concatenate([conv_state, normed], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(
                conv_input[:, -ple.short_conv_state_len :]
            )
        output = gated + nn.silu(ple.conv1d(conv_input))
        rollback_state = (
            token_history,
            embedding.context_len,
            conv_input,
            ple.short_conv_state_len,
        )
        return output, rollback_state

    def _adaptive_moe(self, moe, hidden):
        gates = mx.softmax(self._linear(moe.gate, hidden), axis=-1, precise=True)
        width = moe.top_k
        indices = mx.argpartition(gates, kth=-width, axis=-1)[..., -width:]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        topk_mass = scores.sum(axis=-1, keepdims=True)
        layer_id = getattr(moe, "_flashnext_layer_id", None)
        threshold = routing._LAYER_THRESHOLDS.get(
            layer_id, routing._THRESHOLD[0]
        )
        if threshold < 1.0:
            order = mx.argsort(-scores, axis=-1)
            indices = mx.take_along_axis(indices, order, axis=-1)
            scores = mx.take_along_axis(scores, order, axis=-1)
            mx.eval(scores)
            keeps = []
            for weights in scores.reshape(-1, width).tolist():
                total = sum(weights)
                accumulated = 0.0
                keep = width
                for position, weight in enumerate(weights):
                    accumulated += weight / total
                    if accumulated >= threshold:
                        keep = position + 1
                        break
                keeps.append(keep)
            observer = routing._ROUTE_OBSERVER[0]
            if observer is not None and layer_id is not None:
                observer(
                    layer_id,
                    indices.reshape(-1, width).tolist(),
                    scores.reshape(-1, width).tolist(),
                    keeps,
                )
            active_width = max(keeps)
            indices = indices[..., :active_width]
            scores = scores[..., :active_width]
            shape = (*scores.shape[:-1], 1)
            active = mx.arange(active_width) < mx.array(keeps).reshape(shape)
            indices = mx.where(active, indices, indices[..., :1])
            scores = mx.where(active, scores, 0)
        selected_mass = scores.sum(axis=-1, keepdims=True)
        normalizer = topk_mass + routing._RENORM_BLEND[0] * (
            selected_mass - topk_mass
        )
        scores = scores / normalizer
        if hasattr(moe.switch_mlp, "_one_pass"):
            switched = moe.switch_mlp(hidden, indices, allow_sort=False)
        elif indices.size >= 64:
            rows = max(1, 63 // int(indices.shape[-1]))
            switched = mx.concatenate(
                [
                    moe.switch_mlp(
                        hidden[:, start : start + rows],
                        indices[:, start : start + rows],
                    )
                    for start in range(0, int(hidden.shape[1]), rows)
                ],
                axis=1,
            )
        else:
            switched = moe.switch_mlp(hidden, indices)
        switched = (switched * scores[..., None]).sum(axis=-2)
        shared = self._feed_forward(moe.shared_expert, hidden)
        shared_gate = mx.sigmoid(self._linear(moe.shared_expert_gate, hidden))
        return switched + shared_gate * shared

    def _layer(self, layer, hidden, input_ids, mask, cache, position_ids, gdn_sink):
        ple_state = None
        if "ple" in layer:
            ple_output, ple_state = self._ple(
                layer.ple, hidden, input_ids, cache, mask
            )
            hidden = hidden + ple_output
        mixed, hyper_input, weights = self._hyper(
            layer.attn_hyper_connection, hidden
        )
        if layer.is_linear:
            branch = self._gated_delta(
                layer.linear_attn, mixed, mask, cache, gdn_sink
            )
            if ple_state is not None:
                gdn_sink[-1] = (*gdn_sink[-1], ple_state)
        else:
            branch = self._qwen4_attention(
                layer.self_attn, mixed, mask, cache, position_ids
            )
        hidden = hyper_input + (
            branch[..., None, :] * weights[..., None]
        ).reshape(*hyper_input.shape)
        mixed, hyper_input, weights = self._hyper(
            layer.mlp_hyper_connection, hidden
        )
        branch = self._adaptive_moe(layer.mlp, mixed)
        return hyper_input + (
            branch[..., None, :] * weights[..., None]
        ).reshape(*hyper_input.shape)

    def _model(
        self,
        model,
        inputs,
        cache,
        inputs_embeds,
        position_ids,
        capture_layer_ids,
        hidden_sink,
        gdn_sink,
    ):
        helpers = self._helpers()
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
        if cache is None:
            cache = [None] * len(model.layers)
        fa_mask = helpers._create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        ssm_mask = helpers._create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
        capture = set(capture_layer_ids or [])
        for index, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(
                layer,
                hidden,
                inputs,
                layer_mask,
                layer_cache,
                position_ids,
                gdn_sink,
            )
            if hidden_sink is not None and index in capture:
                hidden_sink.append(self._hyper(model.hyper_connection_mixer, hidden))
        return self._hyper(model.hyper_connection_mixer, hidden)
