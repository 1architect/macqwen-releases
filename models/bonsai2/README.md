# Bonsai-2 runtime

This package owns Bonsai-2 ternary 27B checkpoint discovery, MLX loading,
generation state, session defaults, protocol translation, benchmarking, and
tests. The shared chat keeps its model-independent `<think>` and `<tool_call>`
contract; we translate the native protocol at this boundary.

Milestone 1 is text-only. The checkpoint pack includes a 0.92 GB FP16 vision
tower that stays unloaded.

## Documentation

| Document | Purpose |
|---|---|
| [`docs/bonsai2/brief.md`](../../docs/bonsai2/brief.md) | Scope, support status, and main results |
| [`docs/bonsai2/research.md`](../../docs/bonsai2/research.md) | Measurements, rejected ideas, and decisions |
| [`docs/bonsai2/handoff.md`](../../docs/bonsai2/handoff.md) | Operation, validation, constraints, and next work |
| [`results/bonsai2/`](../../results/bonsai2/) | Exact commands, results table, and raw records |

Read the handoff and research record before changing or measuring this runtime.

## Run

```bash
./chat.sh --model bonsai2 --checkpoint b2
```

The checkpoint declares `model_type: prism_hadamard_qwen35` and requires its
bundled `runtime/` loader. Stock loaders skip the activation transform and
return wrong output silently. We only load a compatible checkpoint from a
source we trust, and the backend refuses to run when the runtime files are
absent.

## Test

```bash
.venv/bin/python -m unittest discover \
  -s models/bonsai2/tests/unit -t . -p 'test_*.py' -q
```

## Package map

| Path | Responsibility |
|---|---|
| `backend.py` | Resident generation, cache lifecycle, session replay, and opt-in probes |
| `checkpoint.py` | Checkpoint validation, discovery, and aliases |
| `protocol.py` | Reasoning and tool-call translation |
| `settings.py` | Model-owned environment and session defaults |
| `tests/bench/bench.py` | Fresh-process paired benchmark harness |
| `cache.py` | Instance-level KV growth test support |
| `q2_kernel.py` | Opt-in packed-Q2/G128 MPP prefill probe and runtime hook |
| `q4_attention_kernel.py` | Opt-in affine-Q4/G64 fused-attention probe |
| `qmm_metadata.py` | Opt-in one-time FP32 QMM metadata preparation probe |
| `tests/unit/test_*.py` | Checkpoint-free runtime and harness coverage |

We keep live prefill screens on the bounded `context-1k` fixture and two
rounds to limit exposure on the fanless reference Mac while retaining paging
evidence. Historical larger-context records are preserved but are not rerun
without explicit authorization.
