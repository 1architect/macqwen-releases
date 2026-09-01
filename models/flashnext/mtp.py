"""Native Qwen4 one-layer MTP head with streamed routed experts."""
from __future__ import annotations

from dataclasses import replace
import os

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen4_exp.language import (
    QSAKVCache,
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    Qwen4ExpModel,
    Qwen4ExpRMSNorm,
)
from mlx_vlm.models.qwen3_5.language import (
    _create_qwen3_5_attention_mask,
    _create_qwen3_5_ssm_mask,
)

from .expert_cache import StreamingSwitchGLU


PREFIX = "language_model.mtp"


def _model_call_with_mtp_capture(
    self,
    inputs,
    inputs_embeds=None,
    mask=None,
    cache=None,
    position_ids=None,
    capture_layer_ids=None,
    hidden_sink=None,
    **kwargs,
):
    """Expose all residual streams before the final mixer for Lightning MTP."""
    del kwargs
    hidden_states = (
        self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
    )
    hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
    if cache is None:
        cache = [None] * len(self.layers)

    fa_mask = _create_qwen3_5_attention_mask(
        hidden_states, cache[self.fa_idx]
    )
    ssm_mask = _create_qwen3_5_ssm_mask(
        hidden_states, cache[self.ssm_idx]
    )
    if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
        ssm_mask = mask

    capture = set(capture_layer_ids or [])
    for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
        layer_mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(
            hidden_states,
            inputs,
            mask=layer_mask,
            cache=layer_cache,
            position_ids=position_ids,
        )
        if hidden_sink is not None and index in capture:
            hidden_sink.append(self.hyper_connection_mixer(hidden_states))

    if hidden_sink is not None and capture_layer_ids == []:
        hidden_sink.append(hidden_states)
    return self.hyper_connection_mixer(hidden_states)


class Qwen4ExpMTPModule(nn.Module):
    """The checkpoint's native speculative draft block."""

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden_size = self.hc_count * self.hidden_size
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(
            self.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(
            hc_hidden_size, eps=args.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        layer_config = replace(
            args,
            num_hidden_layers=1,
            layer_types=["qwen_sparse_attention"],
            full_attention_interval=1,
            ple_layer_ids=[],
        )
        self.layers = [Qwen4ExpDecoderLayer(layer_config, layer_idx=0)]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            layer_config, use_combine=False
        )

    def fuse_inputs(self, token_embeddings, hidden_states):
        expected_width = self.hc_count * self.hidden_size
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.reshape(
                *hidden_states.shape[:-2], expected_width
            )
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != expected_width:
            raise ValueError(
                "MTP expects hidden shape [batch, tokens, hc_count * hidden_size]"
            )

        projected_embedding = self.fc_embedding(
            self.pre_fc_norm_embedding(token_embeddings)
        )
        hidden_streams = self.pre_fc_norm_hidden(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        projected_hidden = self.fc_hidden(hidden_streams)
        return (projected_embedding[..., None, :] + projected_hidden).reshape(
            hidden_states.shape
        )

    def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
        hidden_states = self.fuse_inputs(
            embed_tokens(next_token_ids), hidden_states
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = _create_qwen3_5_attention_mask(
            hidden_states, cache[0] if cache else None
        )
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                next_token_ids,
                mask=mask,
                cache=layer_cache,
                position_ids=None,
            )
        return self.hyper_connection_mixer(hidden_states), hidden_states


def attach(language) -> None:
    """Add the MTP module before quantization and weight loading."""
    if Qwen4ExpModel.__call__ is not _model_call_with_mtp_capture:
        Qwen4ExpModel.__call__ = _model_call_with_mtp_capture
    language.mtp = Qwen4ExpMTPModule(language.args)
    language.mtp.layers[0].mlp._flashnext_threshold = float(
        os.environ.get(
            "FLASHNEXT_MTP_THRESHOLD",
            os.environ.get("FLASHNEXT_TOPK_THRESHOLD", "0.85"),
        )
    )


def swap_streaming(language, store, mode: str, capacity: int = 0) -> None:
    """Keep the MTP expert pool on SSD like the backbone expert pools."""
    prefix = f"{PREFIX}.layers.0.mlp.switch_mlp"
    block = language.mtp.layers[0].mlp
    old = block.switch_mlp
    packed = store.shape(f"{prefix}.gate_proj.weight")[-1]
    groups = store.shape(f"{prefix}.gate_proj.scales")[-1]
    down_out = store.shape(f"{prefix}.down_proj.weight")[1]
    bits = packed * 32 // down_out
    group_size = down_out // groups
    block.switch_mlp = StreamingSwitchGLU(
        store,
        prefix,
        group_size,
        bits,
        mode,
        capacity,
        old.activation,
        layer_id=0,
        next_prefix=prefix,
    )


def forward(language, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
    """Run one native draft step and reuse the target embedding and LM head."""
    mtp_output, hc_hidden = language.mtp(
        hidden_states,
        next_token_ids,
        language.model.embed_tokens,
        mtp_cache,
    )
    if language.args.tie_word_embeddings:
        logits = language.model.embed_tokens.as_linear(mtp_output)
    else:
        logits = language.lm_head(mtp_output)
    return (logits, hc_hidden) if return_hidden else logits


def make_cache(language):
    return [QSAKVCache() for _ in language.mtp.layers]
