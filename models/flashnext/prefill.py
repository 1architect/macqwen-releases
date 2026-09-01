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


def prefill_language(language, ids, cache):
    """Keep large MoE prefills whole while avoiding full-sequence logits."""
    full_logits_limit = min(
        QSA_CHUNK_THRESHOLD,
        PREFILL_FULL_LOGITS_MAX_TOKENS,
    )
    if int(ids.shape[1]) <= full_logits_limit:
        output = language(ids, cache=cache)
        logits = output.logits
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(token)
        release_cache = int(logits.nbytes) >= PREFILL_RELEASE_BYTES
        output = None
        logits = None
        if release_cache:
            mx.clear_cache()
        return None, token

    output = language(
        ids,
        cache=cache,
        return_hidden=True,
        skip_logits=True,
    )
    last_hidden = output.hidden_states[-1][:, -1:]
    token = language.speculative_argmax_from_hidden(
        last_hidden
    ).reshape(-1).astype(mx.uint32)
    mx.eval(token, [entry.state for entry in cache])
    output = None
    last_hidden = None
    if PREFILL_CLEAR_CACHE:
        mx.clear_cache()
    return None, token
