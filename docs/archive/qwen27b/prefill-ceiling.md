# V3.8 Prefill accounting: where every millisecond goes

2026-08-24, V3.1-Compact, M4 Air, 256-token chunk. Measured, not estimated.

## Layer costs

```text
GDN layer including MLP     83.47 ms   x48
attention layer inc. MLP    83.68 ms   x16
MLP alone                   55.90 ms   x64
```

## Inside one GDN layer

```text
projections (matmul)   17.67 ms   65%
norms, gating, shapes  ~6.5 ms    24%
recurrent scan          2.84 ms   10%
conv1d                  0.39 ms    1%
```

Scan cost per token settles at about 10.3 us and stops falling, which confirms
it is a sequential scan. It is also too small to matter.

```text
T      scan_ms   scan/T_us
64        1.01       15.70
128       1.59       12.43
256       2.84       11.10
512       5.35       10.44
1024     10.57       10.33
```

## Whole prefill, 5346 ms per chunk

```text
MLP matmuls           3578 ms   67%
GDN projection matmul  848 ms   16%
attention              445 ms    8%
recurrent scan         136 ms    2.5%
norms, gating, shapes  ~340 ms   6%
```

Quantised matrix multiplication accounts for about 87% of prefill.

## Resulting limit

The quantised matmul was measured at 2.44 TFLOPS:

```text
q2, q3, q4, q6, q8, group 32 and 64   all 2.39 - 2.45 TFLOPS
dense bf16                            2.68 TFLOPS
fp32                                  2.16 TFLOPS
```

Tiles are already optimal. The sweep, shuffled over 9 repeats:

```text
BM BK BN  median_ms  speedup
64 32 64      17.25    1.070   <- current setting
64 32 32      17.58    1.049
32 32 32      18.45    1.000
```

The custom `mlx-qwen38-kernel-lab` build gives the same 17.25 ms as the stock
`mlx-qwen38-apple` build for this projection. No difference.

Ceiling:

```text
27B params x 2 FLOP = 54 GFLOP per token
2.6 TFLOPS / 54 GFLOP = about 48 tok/s
measured 46.8 tok/s
```

## Verdict

The measured prefill rate is 46.8 tok/s. Prefill is closed as a compute problem.

Raising it requires one of three things, all of which cost something already
ruled out:

```text
fewer active parameters   smaller model or MoE, costs quality
fewer layers in prefill   measured 1.89x with 2 of 24 tokens correct, broken
fewer tokens              cache or selection, rejected as the answer
```

Optimising the recurrent scan perfectly would return 2.5%.

## Closed in this project

```text
prefill kernels        93% of ceiling
quantisation width     every width identical
compute dtype          fp16 equals bf16, fp32 slower
QMM tiles              64/32/64 already best of 8
custom MLX build       same speed as stock
lm_head                MLX laziness already elides it
MLX buffer pool        holds 0.03 to 0.29 GB
layer pruning          1.89x, output broken
sparse MLP             1.88x, 11.7% RMSE, failed
A8W2 TensorOps         0.70x, rejected
ANE hybrid             1.45x, collapses above 542 MB
recurrent scan         2.5% of prefill, not worth it
```
