# Archived Flash-Next streaming study

This study records results from 2026-08-30. Current runtime instructions can
differ.

`flashnext/` ran Qwen3.8-Flash-Next-MLX-oQ4 on a 16 GB M4. The runtime kept
routed experts and hashed n-grams on the SSD.

```text
checkpoint on disk   111.70 GB
resident              4.19 GB
load                     2.1 s
decode                  1.00 tok/s
prefill                 0.77 tok/s
```

Stock mlx-vlm materialized every weight. Swap grew by 24 GB in two minutes.
RSS stayed below 30 MB, and the model produced no token.

## Streaming components

| File | Role |
|---|---|
| `store.py` | Memory-mapped tensors, `MADV_RANDOM`, and C row gathers |
| `expert_cache.py` | Nine parallel expert reads per layer |
| `ngram.py` | Row-level n-gram reads instead of a 19 GB allocation |
| `loader.py` | Module replacement before weight materialization |
| `patch_rmsnorm.py` | Upstream correctness fix |

The streaming output matched the dense path exactly. Tests from 1 through 800
tokens produced `max_diff = 0.0`.

## Decode measurements

Each token read the routed experts for 48 layers.

```text
per token          1475 MB
gather throughput  1068 MB/s
pure I/O            1381 ms
measured            1390 ms
```

The 48 routing host synchronizations used 8 ms total.

| Change | ms/token | tok/s |
|---|---:|---:|
| Starting point | 7960 | 0.126 |
| `np.memmap` gathers | 5298 | 0.189 |
| Batched reads without LRU | 2735 | 0.366 |
| Nine parallel layer reads | 1840 | 0.543 |
| 16 I/O workers | 1788 | 0.559 |
| Sorted indices | 1671 | 0.598 |
| `MADV_RANDOM` and chunked gathers | 1390 | 0.719 |
| Adaptive `top_k` at 0.85 | 1000 | 1.00 |

## Adaptive routing decision

Adaptive routing selected experts until their cumulative probability reached a
threshold. Alternating runs produced these results:

```text
threshold 0.85    981 / 1017 ms/token    1.00 tok/s
threshold 1.00   1832 / 1831 ms/token    0.55 tok/s
```

The 1.86x gain came from 15% fewer bytes and a page-cache threshold. Threshold
0.90 measured 1675 ms. Thresholds 0.80 and 0.70 added little speed.

At threshold 0.70, 7 of 10 prompts matched token for token. The other three
produced different correct continuations. Every threshold returned Au for
gold. Fixed `top_k=6` measured 587 ms against 2056 ms, but arithmetic failed
below `top_k=10`. The retained default was adaptive threshold 0.85.

## Rejected changes

| Change | Result |
|---|---|
| Expert cache, capacity 96 | 76.5% hit rate; 4257 versus 1390 ms/token |
| Cache warming | 1870 versus 1840 ms/token |
| Bulk prefill reads | 208 versus 124.6 seconds |
| 32 I/O workers | 212.9 versus 120.5 seconds |
| Checkpoint repacking | Apparent 1.84x gain vanished under a controlled test |
| Bias removal | Bias standard deviation 0.035; zero entries 0% |
| `MADV_WILLNEED` | 1.5x slower |

Sixteen readers reached 907 MB/s. Thirty-two reached 615 MB/s. The repacked
file reached 21 to 31 GB/s in cached tests. The original layout reached 43 to
48 GB/s because NumPy used one vectorized gather.

`FLASHNEXT_WARM` and `FLASHNEXT_BULK` preserved the rejected paths.

## Prefill result

A 93-token prompt took about 120 seconds. Prefill selected about 290 distinct
experts per layer and read 25.35 GB. Sixteen-token chunks would read 69 GB.
`gather_qmm` used 3.2 seconds of a 121-second prefill. The GPU stayed idle.

## Upstream RMSNorm defect

`Qwen4ExpRMSNorm` applied `y * (1.0 + weight)`. The implementation claimed
zero-centered checkpoint weights. Measurements showed positive minima:

```text
norm_conv       mean +0.8906   min +0.543
norm_key        mean +0.8933   min +0.672
q/k_layernorm   mean +0.9306   min +0.500
q/k_norm        mean +1.3576   min +0.098
hc_norm         mean +3.7498   min +0.633
```

The extra `1.0` doubled each norm gain. The broken model returned punctuation
at 3% for `The capital of France is`. The fixed model returned ` Paris` at
44.4%. The recorded decision was to report the defect to `Blaizzy/mlx-vlm`.

## Requantization result

| Recipe | Effective bits | MB/token | Gate error | h error | Output error |
|---|---:|---:|---:|---:|---:|
| 4-bit/group-128 | 4.25 | 1254 | 9.6% | 14.5% | 15.7% |
| 3-bit/group-64 | 3.50 | 1032 | 20.6% | 30.6% | 33.0% |
| 2-bit/group-64 | 2.50 | 738 | 40.8% | 60.9% | 64.7% |
| 2-bit/group-128 | 2.25 | 664 | 44.4% | 67.1% | 71.9% |

SiLU increased gate errors. Requantization also stacked new error on the oQ4
source. The BF16 source did not fit on disk. The 2-bit rewrite needed 45 GB,
but only 11 GB was free. The study rejected this path.

## Measurement controls

Unchanged runs varied from 1390 to 2920 ms/token. Page-cache state invalidated
five comparisons. Valid results needed more than about 20% improvement and a
reversed or interleaved control.
