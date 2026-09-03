"""Flash-Next prompt ingestion without the interactive chat."""
from __future__ import annotations

import os

import mlx.core as mx

from models.flashnext.qsa_chunk import QSA_CHUNK_THRESHOLD


PREFILL_FULL_LOGITS_MAX_TOKENS = int(
    os.environ.get("FLASHNEXT_PREFILL_FULL_LOGITS_MAX_TOKENS", "8192")
)
PREFILL_RELEASE_BYTES = int(
    os.environ.get("FLASHNEXT_PREFILL_RELEASE_BYTES", "256000000")
)
PREFILL_CLEAR_CACHE = os.environ.get("FLASHNEXT_PREFILL_CLEAR_CACHE", "1") != "0"


def prefill_target(
    language,
    ids,
    cache,
    want_logits: bool = False,
    want_hidden: bool = False,
    sampler=None,
):
    """Push one target prompt through the model and seed its next token.

    The returned tuple is ``(logits, hidden_states, token)``. Requested
    tensors are retained for callers that need them. The token always comes
    from the exact target path, using ``sampler`` when supplied.

    Short prompts keep the normal full-logit call. Large prompts skip
    full-sequence logits unless the caller explicitly requests them.
    """
    def pick(row):
        if sampler is None:
            return mx.argmax(row, axis=-1).astype(mx.uint32)
        return sampler(row).astype(mx.uint32)

    full_logits_limit = min(
        QSA_CHUNK_THRESHOLD,
        PREFILL_FULL_LOGITS_MAX_TOKENS,
    )
    short_prompt = int(ids.shape[1]) <= full_logits_limit
    full_logits = short_prompt or want_logits
    call = {
        "cache": cache,
        "return_hidden": want_hidden or not full_logits,
        "skip_logits": not full_logits,
    }
    # MTP relies on the model's explicit hidden-sink path. Keep this
    # keyword absent for ordinary large prefills, which use the fast path.
    if want_hidden:
        call["capture_layer_ids"] = []
    output = language(ids, **call)
    logits = output.logits if full_logits else None
    hidden_states = output.hidden_states if (want_hidden or not full_logits) else None

    if full_logits:
        mx.eval(logits)
        if hidden_states:
            mx.eval(hidden_states)
        token = pick(logits[:, -1, :])
        mx.eval(token)
        release_cache = int(logits.nbytes) >= PREFILL_RELEASE_BYTES
        if release_cache or (not short_prompt and PREFILL_CLEAR_CACHE):
            mx.clear_cache()
    else:
        last_hidden = hidden_states[-1][:, -1:]
        if sampler is None:
            # The fused head skips materialising logits, which is why this
            # path exists. Only a sampler needs them, and only for the final
            # row.
            token = language.speculative_argmax_from_hidden(
                last_hidden
            ).reshape(-1).astype(mx.uint32)
        else:
            token = pick(language.speculative_logits_from_hidden(last_hidden))
        mx.eval(token, [entry.state for entry in cache])
        if PREFILL_CLEAR_CACHE:
            mx.clear_cache()

    # Drop the model output wrapper while retaining only what the caller
    # requested. This keeps the allocator release effective for large calls.
    output = None
    if want_logits:
        retained_logits = logits
    else:
        retained_logits = None
    retained_hidden = hidden_states if want_hidden else None
    return retained_logits, retained_hidden, token


def prefill_language(language, ids, cache, sampler=None):
    """Compatibility wrapper for the standard target prefill entry point."""
    _logits, _hidden, token = prefill_target(
        language, ids, cache, sampler=sampler
    )
    return None, token
