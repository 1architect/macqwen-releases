# Archived V3.7 context-image result

Archive status: completed experiment. The measurements apply to V3.1-Compact on 2026-08-24.
2026-08-24, V3.1-Compact, M4 Air. One model load, paired in-process comparison, so no thermal drift between runs.

## Result

```text
restore        0.27 s   2377 tokens
cold prefill  52.38 s   2377 tokens   46.8 tok/s
speedup       190.8x
```

Restoring is equivalent to about 8,800 tok/s of ingestion.
The saved file contains the KV state. Restored context is bit-identical to normal prefill.

## Comparison with other project measurements

```text
State Codec target        1.75x   requires training
layer pruning (v36)       1.89x   2 of 24 tokens matched, broken
sparse MLP (v32)          1.88x   11.7% RMSE, failed the gate
ANE hybrid (v34)          1.45x   collapses above 542 MB
context image             190x    exact
```

## Cost and limits

```text
image size        188.6 MiB for 1208 source tokens
per-token cost    about 156 KB, dominated by the constant GDN state
floor             about 160 MB per image regardless of file size
```

The 48 Gated DeltaNet layers hold a fixed-size recurrent state that must be saved whole, so a small file costs nearly as
much as a large one.
Limits:

```text
1. only helps content already ingested once
2. images do not compose. The recurrent state is a sequential scan, so two
   independently built images cannot be merged. One image is one ordered
   prefix.
3. an image is tied to the model, the system prompt, the tool contract and
   the effort setting. Change any of them and it is invalid.
```

Fifty files would cost about 9.5 GB on disk, which is affordable, but they cannot be combined into one context.

## Status before this measurement

`repo_context_image.py` was already written and already imported by `frankenstein_chat.py`. No image had ever been
built. The feature existed and had never run.

## Operational effect

Repository work often reads the same files again. The first read costs 46.8 tok/s. Each later read costs 0.27 seconds.
Prefill kernel speed stops being the interesting number. The interesting number is the cache hit rate.

## Next

```text
1. build images for the whole working set, measure hit rate in real sessions
2. decide the eviction policy for 160 MB per image
3. warm the images in the background instead of on first use
```
