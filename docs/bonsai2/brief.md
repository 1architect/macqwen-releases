# Bonsai-2 ternary 27B overview

## Purpose

Bonsai-2 is our ternary-weight 27B alternative for low-memory Apple Silicon.
It runs text-only through its bundled Hadamard runtime while keeping all
model-specific loading, protocol translation, and settings inside
`models/bonsai2/`.

## Model and checkpoint

We support `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, a 2-bit/group-128 ternary
pack that occupies about 8.6 GB on disk (7.67 GB language model plus 0.92 GB
FP16 vision tower). The `b2` alias resolves to
`~/models/Ternary-Bonsai-2-27B-mlx-2bit` under the configured model root.

The checkpoint declares `model_type: prism_hadamard_qwen35` and requires its
bundled `runtime/` loader. Stock loaders skip the activation transform and
return wrong output silently. We load it only from a source we trust, and our
backend refuses to run when the runtime files are absent.

Milestone 1 is text-only. The vision tower stays unloaded.

## Main retained results

- The 256-token product baseline measured median 5.5 tok/s over 256 tokens
  after a 3,282-token prompt, with MLX peak 10.43 GB.
- The 32-token control measured about 5.7 tok/s. We keep it separate from
  sustained product throughput.
- Prefill chunk 512 stays the default: larger chunks add peak memory up to
  15.47 GB with no speed gain.
- Allocator cap, wired limit, and post-generation clear change no default
  yet; the cap saves about 470 MB free at 2k and awaits 16k confirmation.
- Greedy comparison arms retained identical token digests.
- Exact-only context reaches about 32k on 16 GB; 262K needs 16.8 GB of KV
  payload alone and stays out of scope without a quality-gated design.

## Current status

Bonsai-2 is experimental research code on this branch. Our retained defaults
are prefill step 512 and text-only generation. No experimental optimization
changed model precision, context retention, or output semantics.

Read [`handoff.md`](handoff.md) before operating or changing the runtime and
[`research.md`](research.md) before proposing another experiment.
