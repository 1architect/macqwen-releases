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

No promoted results yet. This branch establishes discovery, backend, protocol,
benchmark harness, and validation. The complete table, commands, and raw
records will live in [`measurements/`](measurements/).

## Current status

Bonsai-2 is experimental research code on this branch. Our retained defaults
are prefill step 512 and text-only generation. No experimental optimization
changed model precision, context retention, or output semantics.

Read [`handoff.md`](handoff.md) before operating or changing the runtime and
[`research.md`](research.md) before proposing another experiment.
