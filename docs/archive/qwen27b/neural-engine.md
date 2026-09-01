# V3.4 Neural Engine: result

Measured 2026-08-24 on the M4 Air, model V3.1-Compact.

The measurements provide a positive component-level result.

## ANE rationale

Every other optimisation tried in this project reduces work because the GPU is
saturated. Measured: prefill runs at 93% of the GPU compute ceiling, and the
MLP is 67% of prefill time. The 16-core Neural Engine sits completely idle.

Using it raises the ceiling instead of doing less work.

## Engine overlap

`v34_dual_engine_benchmark.py`:

```text
gpu_ms             53.257
ane_ms              5.454
sequential_ms      58.711
parallel_ms        53.867
ideal_parallel_ms  53.257
engine_overlap     88.8%
result             DUAL_ENGINE_WORKS
```

The ANE work disappears almost entirely inside the GPU work.

## Qwen MLP accuracy on the ANE

`v34_qwen_ane_mlp.py`, layer 0, 256 tokens, int4:

```text
relative_rmse  0.00708
cosine         0.99998
result         MLP_MATCH
package_mb     135.5
```

Isolated speed is not the point and is unimpressive:

```text
CPU_ONLY     46.9 ms
CPU_AND_NE   44.7 ms
ane_vs_cpu   1.05x
```

The ANE does not need to beat the GPU. It needs to run at the same time.

## Two real Qwen layers, hybrid prefill

`v34_two_layer_pipeline_benchmark.py --layers 2`

Without residency:

```text
layer 00  load_ms 840.9  process_ms 450.7
layer 01  load_ms 811.6  process_ms 446.1
baseline_ms  1310.1
pipeline_ms  2598.7
speedup      0.504x
```

Loading each Core ML model costs about 840 ms, roughly twice the processing
time. That alone makes the hybrid twice as slow.

With `--resident`:

```text
layer 00  load_ms 0.0  process_ms 480.4
layer 01  load_ms 0.0  process_ms 432.2
baseline_ms       1332.7
pipeline_ms        912.8
speedup           1.460x
relative_rmse     0.00507
resident_load_ms  1613.7
result            MULTILAYER_WORKS
```

The measured result is 1.46x with 0.5% relative RMSE.

## Scaling: separate models collapse, multifunction holds

Four layers as four separate Core ML models:

```text
layer 00  process_ms 1548.8
layer 01  process_ms 1537.2
layer 02  process_ms 1870.8
layer 03  process_ms 1955.9
speedup   0.377x
```

Per-layer processing rose from about 450 ms to about 1700 ms. The engines were
not the problem: four separate models contend for the ANE and it swaps them.

The same four layers exported as one multifunction model:

```text
layer 00  process_ms 475.6
layer 01  process_ms 437.8
layer 02  process_ms 448.9
layer 03  process_ms 460.9
baseline_ms       2644.3
pipeline_ms       1823.5
speedup           1.450x
resident_load_ms  3940.0
relative_rmse     0.00636
```

Per-layer time returns to the two-layer level and the gain holds:

```text
2 layers, separate       1.460x
4 layers, separate       0.377x
4 layers, multifunction  1.450x
```

The curve is flat once the packaging is right. Scale from the 4-layer model:

```text
542 MB for 4 layers   ->  about 8.7 GB for 64
3.9 s resident load   ->  about 63 s for 64, once per session
```

Eight layers in one multifunction model, 1.1 GB:

```text
process_ms per layer   1624 - 2491
baseline_ms            5293.5
pipeline_ms           14528.5
speedup                0.364x
resident_load_ms      10151.6
```

Multifunction packaging fixes four layers and does not fix eight. The ANE
resident working-set limit lies between 542 MB and 1.1 GB.

```text
2 layers, separate       271 MB   ~450 ms   1.460x
4 layers, separate       542 MB  ~1700 ms   0.377x
4 layers, multifunction  542 MB   ~455 ms   1.450x
8 layers, multifunction  1.1 GB  ~1700 ms   0.364x
```

## Verdict

The ANE cannot hold this model's complete MLP.

Sixty-four layers of MLP are about 8.7 GB in Core ML int4, eight to sixteen
times what the ANE sustains. Dropping to int2 would give about 4.4 GB, still
far beyond it. Keeping four resident and streaming the rest fails too: each
load costs about 840 ms against about 450 ms of processing.

Accelerating four layers out of sixty-four is worth nothing overall.

Retained results:

```text
the two engines overlap by 88.8%
the exact Qwen MLP runs on the ANE at 0.5 to 0.8% RMSE, which passes
within the resident limit the hybrid is genuinely 1.45x faster
```

The idea is sound. The hardware does not have the memory to apply it at this
model's scale. This is a capacity limit, not a correctness or design failure.

Reopen this line if any of these change:

```text
a model whose MLP fits in about 500 MB total
an ANE with a larger resident working set
a Core ML path that streams weights without the 840 ms load
```

Rules learned, both of which invert the result if ignored:

```text
never benchmark the ANE without --resident
never use separate packages where one multifunction model fits
```

## Comparison with prior experiments

```text
layer pruning (v36)   1.89x   2 of 24 tokens matched   broken
sparse MLP (v32)      1.88x   11.7% RMSE               failed the gate
A8W2 TensorOps (v31)  0.70x                            rejected
ANE MLP resident      1.46x   0.5% RMSE                passes
```

Only the ANE approach improves speed and passes the quality gate in these
measurements. It uses the same weights and arithmetic on different hardware.

## Projection and costs

```text
48 t/s x 1.46 = about 70 t/s
```

Startup: residency costs about 807 ms per layer, so roughly 52 s once per
session for 64 layers.

Memory: 135.5 MB per layer, 8.7 GB for 64. This only fits if the MLP leaves
the MLX model and lives solely in Core ML, leaving MLX with the mixing weights
at about 2.6 GB. Total about 11.3 GB, close to today's footprint.

## Open

```text
1. does 1.46x hold across all 64 layers, or only the first two
2. build the MLX model without MLP weights, so nothing is duplicated
3. measure decode, which is bandwidth bound and may not benefit at all
4. amortise or hide the 52 s residency load
```

Rule that came out of this: never benchmark the ANE without `--resident`.
The load cost inverts the result.
