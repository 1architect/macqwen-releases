# Qwen3.8-27B brief

## Purpose

This runtime explores dense and hybrid Qwen inference on a 16 GB Apple Silicon
Mac. It established the original MACQWEN memory and context techniques.

## Model

The runtime targets Qwen3.8-27B checkpoints with vocabulary size `248320`.
Local V4 builds use measured heterogeneous affine quantization.

The last documented V4-flat build used 13.05 GB on disk. Its lean loader kept
about 12.65 GB resident by moving the embedding outside resident memory.

## Main retained results

- Append-only conversation state avoids repeated old prefill.
- Exact context images restored repository state about 190 times faster.
- Paged KV held 256K logical tokens in about 0.81 GB resident KV memory.
- A 256-token prefill step reduced peak memory without a speed loss.
- Measured bit allocation replaced fixed manual quantization rules.
- API documentation and code checks reduced invented library calls.

## Limits

The 27B runtime needs a custom MLX environment. Builds above about 12.6 GB
resident memory caused swap and large generation losses on the tested machine.

The most recent local verification found no installed V4 checkpoint. Unit and
import tests still pass.

## Status

Qwen27B is supported as a secondary research runtime. Flash-Next is the current
project focus.
