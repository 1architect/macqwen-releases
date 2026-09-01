# Archived V3.2 Sparse MLP result

Archive status: rejected. Sparse conversion failed the quality gate and remains disabled.
Layout tested: 40 shared groups and 8 conditional experts of 12 groups. The router selects 2 experts. Each group
contains 128 neurons. The active MLP width is 64/136 groups, or 47.1%.

## Local kernel result

| Workload | Dense MLP | Sparse MLP | MLP speedup |
| --- | ---: | ---: | ---: |
| Generation, 1 token | 1.621 ms | 1.192 ms | 1.360x |
| Prefill, 256 tokens | 54.988 ms | 29.214 ms | 1.882x |

Command: `v32_sparse_mlp_benchmark.py`, 11 repetitions.

## Quality gate

Calibration uses 256 tokens and all 64 layers. The first linear router preserves 54.2% mean held-out activation energy.
This result does not pass the E2 quality gate.
The test did not change the model. Deployment requires distillation or sparse continued training.

## Distillation result

Calibration grew to 1,024 tokens from MACBAT source files.
Layer 0 results:

| Route | Active width | Best held-out relative RMSE | Cosine |
| --- | ---: | ---: | ---: |
| top-4 with Sparse LoRA | 64.7% | 15.5% | 0.9900 |
| top-6 with Sparse LoRA | 82.4% | 11.7% | 0.9945 |

An oracle scan measured all 64 layers. Most layers still have 17% to 25% error when only one expert is skipped.
The SwiGLU model has insufficient natural contextual sparsity. Direct conversion misses the E2 quality target.
Deployment requires large sparse continued training.

## Prefill-only end-to-end probe

The runtime used Sparse MLP only when the input contained multiple tokens. Decode remained dense.
Top-4 passed 2/3 established cases. It repeated the complexity answer until the 600-token limit.
Top-6 passed 2/3 established cases. It repeated the lock-order answer and failed to close reasoning.
Sparse prefill remains disabled in the default terminal chat.
Calibration data lives outside the project: `~/Library/Caches/MACQWEN/v32_sparse_mlp`
