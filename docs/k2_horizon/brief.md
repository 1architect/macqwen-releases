# K2-Horizon 7B overview

## Purpose

K2-Horizon is our resident dense-model alternative to the SSD-streamed
Flash-Next runtime. It runs the checkpoint through MLX-LM while keeping all
model-specific loading, protocol translation, and settings inside
`models/k2_horizon/`.

## Model and checkpoint

We support `abenzerps/K2-Horizon-7B-MLX-8bit`, a Q8/G64 checkpoint that
occupies about 9.6 GB on disk. The `k2` alias resolves to
`~/models/K2-Horizon-7B-MLX-8bit` under the configured model root.

The checkpoint supplies executable `model.py` code. We load it only from a
source we trust.

## Main retained results

- The 256-token product baseline measured median 9.680 tok/s overall and
  9.676 tok/s over tokens 33–256 after a 2,731-token prompt.
- The 32-token control measured median 11.752 tok/s. We keep it separate from
  sustained product throughput.
- Prefill step 256 did not improve the 8K memory peak and was not promoted.
- MLX `wired_limit` lost all three paired comparisons and remains disabled.
- Greedy comparison arms retained identical token digests.

The complete table, commands, and raw records are in
[`measurements/`](measurements/).

## Current status

K2-Horizon is supported for local chat. Our retained defaults are prefill step
512, native BF16 KV cache growth, allocator cap and post-generation cache clear
off, and `wired_limit` off. No experimental optimization from the current
research changed model precision, context retention, or output semantics.

Read [`handoff.md`](handoff.md) before operating or changing the runtime and
[`research.md`](research.md) before proposing another experiment.
