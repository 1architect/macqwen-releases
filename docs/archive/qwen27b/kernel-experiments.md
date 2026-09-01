# Kernel experiments (V3.1)

Dead-end kernel work, kept for the record. All rejected: the quantised matmul is already at the chip ceiling.


---

# V3.1 experimental prefill mode benchmark

This protocol compares a candidate kernel with the current `64/32/64` QMM.
It measures raw neural prefill. Prefix-cache load speed is a separate metric.

## Fixed baseline

```text
model                 Qwen3.8-27B-Apple-MLX-V3.1-Compact
tokens per chunk      256
baseline QMM tiles    BM=64, BK=32, BN=64
sampling              greedy
wired limit           disabled for the primary server test
KV                    FP16 below 8192, Q4 from 8192
```

Run a secondary terminal test with the 8 GB wired limit. Do not mix its result
with the server result.

## Measurement rules

1. Load one model copy.
2. Compile and warm every candidate twice before timing.
3. Use the same token IDs and initial state for both paths.
4. Alternate paths as `A B B A`, then `B A A B`.
5. Record each paired ratio `baseline_seconds / candidate_seconds`.
6. Use the median paired ratio and a bootstrap 95% confidence interval.
7. Invalidate a sample when swapout occurs or free memory falls below 1 GB.
8. Do not compare two independent runs. The M4 Air changes speed with heat.
9. Save raw samples. A summary without samples is insufficient.

Use at least 11 pairs for microbenchmarks and 9 pairs for the full model.

## Stage 0: numerical gate

Test these representative projections:

```text
layer 0   gate Q2, up Q2, down Q2
layer 26  gate Q2, up Q2, down Q3
layer 63  gate Q2, up Q2, down Q3
M         64, 128, 256, 512, 1024
```

Use fixed random BF16 inputs and real repository activations.

An exact mode must meet all conditions:

```text
projection max absolute error     0
MLP max absolute error            0
full-body max absolute error      0
cache-state max absolute error    0
128 greedy tokens                 identical
```

A fused epilogue must preserve the baseline BF16 rounding points. Otherwise,
classify it as an approximate mode and use the separate quality gate below.

## Stage 1: component speed

Measure each projection and the complete MLP. Report median milliseconds,
paired speedup, active memory, peak memory, and cache memory.

Component continuation gates:

```text
Q2 LUT                     at least 1.05x on Q2 projections
gate+up+SwiGLU fusion      at least 1.15x on the affected subgraph
new matrix/Tensor path     at least 1.25x on the complete MLP
```

Stop a branch that misses its continuation gate. Small component gains cannot
produce a large full-model gain.

## Stage 2: cold full-model speed

Place the Mac on AC power and a cool hard surface. Leave it idle first.
Warm only kernel compilation. Then run nine interleaved pairs with 256 real
repository tokens and independent, equal caches.

Report:

```text
median prompt tok/s
median paired speedup
95% confidence interval
peak MLX memory
pageins and swapouts
```

Current reference is approximately 50 tok/s before thermal throttling.

## Stage 3: hot sustained speed

Advance two equal prompt caches, one per path. Process the same 256-token
chunks in alternating order for 15 minutes. Do not reset the thermal state.

Report one-minute windows. Define hot speed as the median of the last five
minutes. Also report:

```text
hot retention = hot tok/s / cold tok/s
worst one-minute tok/s
time until speed first falls below 90% of cold speed
```

Current reference is approximately 40 tok/s when hot.

Run context checkpoints at 0, 4096, 8192, and 16384 cached tokens. This finds
an improvement that helps short MLP work but loses to long-context attention.

After the paired test, run each path alone for 15 minutes. Randomize their
order across four cooled starts. Start again only when the baseline probe is
within 2% of its original cold rate. This detects extra candidate heat that
can slow both paths during an interleaved test.

## Stage 4: acceptance targets

Use paired speedup as the primary result. Absolute rates are diagnostic.

```text
                    cold target          hot target
useful component    >= 1.05x             >= 1.05x
production mode     >= 1.20x             >= 1.20x
experimental mode   >= 1.30x             >= 1.30x
stretch             >= 1.50x             >= 1.50x
```

With the current 50/40 tok/s reference:

```text
production mode       60 cold, 48 hot tok/s
experimental mode     65 cold, 52 hot tok/s
stretch               75 cold, 60 hot tok/s
```

For acceptance, the lower 95% confidence bound must exceed the selected
target. Candidate hot retention must not trail baseline retention by more
than two percentage points.

Peak MLX memory can increase by at most 256 MB. Swapouts must remain zero.

## Amdahl estimates

Measured short-prefill shares are:

```text
MLP                     75.0%
other model work        25.0%
Q2 projections          84.375% of MLP
gate plus up             66.667% of MLP
```

For an MLP-wide speedup `s`:

```text
model speedup = 1 / (0.25 + 0.75 / s)
```

```text
MLP speedup       full-model estimate
1.10x             1.073x
1.20x             1.143x
1.30x             1.209x
1.50x             1.333x
2.00x             1.600x
3.00x             2.000x
```

### Q2 LUT

The old 256-token result was 169.05 ms for QMM and 150.01 ms for persistent
BF16 GEMM. The measured dense comparator was only 1.127x faster. Use this as
a conservative LUT-only estimate, not as a physical kernel ceiling. A Q2-only
optimization covers 162 of 192 equal-size MLP projections.

```text
Q2 speedup        full-model estimate
1.05x             1.031x
1.10x             1.061x
1.127x            1.077x
1.20x             1.118x
1.50x             1.267x
```

The new `64/32/64` baseline can reduce this remaining headroom. Re-measure it
before implementing a LUT. Treat LUT Q2 as a supporting optimization.

### Gate, up, and SwiGLU fusion

Gate and up contain two thirds of MLP matrix work. If the fusion accelerates
only that subgraph by `s`:

```text
model speedup = 1 / (0.50 + 0.50 / s)
```

```text
subgraph speedup  full-model estimate
1.10x             1.048x
1.20x             1.091x
1.30x             1.130x
1.50x             1.200x
2.00x             1.333x
```

For 256 tokens, fusion avoids at least 35.65 MB of gate/up intermediate
traffic per layer. Across 64 layers, this is 2.28 GB per chunk.

### Tensor operations

The current Metal QMM already uses Steel `BlockMMA`. Turning on matrix
operations again cannot add a speedup. MLX NAX requires a later architecture
generation than this M4.

A different matrix path must beat the complete MLP by 1.25x before full-model
testing. CPU and GPU co-processing must include synchronization, dequantizing,
shared-memory traffic, and sustained thermal cost in its timing.

Do not use isolated CPU TFLOPS as the projected gain.

## Approximate-mode quality gate

Run this only when numerical equality fails.

```text
20 fixed prompts              greedy output identity report
EvalPlus                      pass@1 delta no worse than -0.5 points
BFCL                          accuracy delta no worse than -0.5 points
RepoQA 4K and 8K              delta no worse than -0.5 points
local tool suite              zero new malformed tool calls
MacBat review fixtures        zero new false severe findings
```

Use the same tokenizer, prompt, sampling, and token limit. Pin benchmark
commits. Report paired task outcomes, not only aggregate scores.

Reject an approximate mode when any safety or tool-format regression appears.

## Required result record

Save each experiment under `v31_work/` with a `prefill_mode_` prefix.
Include:

```text
runtime commit or source hash
model fingerprint
candidate name and settings
raw paired times
cold and hot windows
memory and swap data
numerical errors
quality results
final accept or reject reason
```

---

# V3.1 A8W2 stop result

Device: Apple M4 GPU. Projection weights: real V3.1 layer-0 affine Q2.
Activation estimate: symmetric A8 per token and group of 128 values.

| Projection | Shape | Current BF16 QMM | Raw U8 x U2 | Raw speed | A8 pack | A8 output NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| gate | 256x5120x17408 | 17.246 ms | 24.366 ms | 0.708x | 1.325 ms | 1.061% |
| down | 256x17408x5120 | 17.303 ms | 24.642 ms | 0.702x | 4.639 ms | 2.075% |

The raw integer result matches the reference exactly. It omits affine weight
scales, weight biases, A8 scales, and group corrections. Those operations only
increase its final time.

Stop result: reject A8W2 TensorOps on M4. The raw arithmetic is about 1.42x
slower than the current QMM before required epilogue work.

---

# V3.1 Metal LUT/Q2 prototype

Date: 2026-08-21

The benchmark uses the complete layer 0 `mlp.gate_proj` from V3.1 Compact:

- input width: 5120
- output width: 17408
- affine weight quantization: 2 bits
- group size: 128
- activation and output: BF16

It loads only `model-layer-00.safetensors`. It does not patch MLX, the model,
the runtime, or a launcher.

## Method

Each packed byte contains four Q2 codes. The kernel creates a 16-entry subset
sum table from the corresponding four BF16 activations. It stores only eight
entries and reconstructs the other eight with the complement identity.

For `W = scale * q + bias`, the affine kernel evaluates:

```text
XW = scale * Xq + bias * sum(X)
```

The codebook variant also preserves the four BF16-rounded dequantization
levels before it reorganizes the sum.

## Complete projection result

Command:

```text
/Users/gioma/mlx-qwen38-kernel-lab/bin/python \
  v31_lut_q2_benchmark.py \
  --tokens 64,128,256 --outputs 17408 --warmup 1 --repeats 3
```

| M | Kernel | Median | Relative speed | Max absolute error | Exact BF16 outputs |
|---:|---|---:|---:|---:|---:|
| 64 | MLX QMM | 11.97 ms | 1.000x | 0 | 100% |
| 64 | LUT affine | 81.12 ms | 0.148x | 0.0078125 | 99.97% |
| 64 | LUT codebook | 103.59 ms | 0.116x | 0.0078125 | 99.97% |
| 128 | MLX QMM | 10.41 ms | 1.000x | 0 | 100% |
| 128 | LUT affine | 159.75 ms | 0.065x | 0.0078125 | 99.97% |
| 128 | LUT codebook | 220.93 ms | 0.047x | 0.0078125 | 99.97% |
| 256 | MLX QMM | 20.48 ms | 1.000x | 0 | 100% |
| 256 | LUT affine | 335.86 ms | 0.061x | 0.0078125 | 99.97% |
| 256 | LUT codebook | 428.23 ms | 0.048x | 0.0078125 | 99.97% |

The affine kernel is mathematically exact for the stored affine parameters.
It is not bitwise identical because it changes the floating-point reduction
order. The projection-level error is small.

## Decision

This scalar Metal LUT design is not viable for prefill. It is 6.8x slower at
64 tokens and 16.4x slower at 256 tokens.

The M4 SIMD matrix unit makes dequantize-plus-MMA much faster than scalar
threadgroup lookups. LUT methods that help CPU decode do not transfer directly
to a large GPU prefill batch.

Do not integrate this kernel into V3.1.

A future LUT attempt needs one of these changes before another full run:

1. ARM CPU `TBL` kernels with a persistent thread pool.
2. A tensor primitive that performs low-bit lookup inside matrix hardware.
3. A GPU design that vectorizes many LUT results per instruction.
