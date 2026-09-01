# Running a 111.7 GB MoE from disk on 16 GB

`flashnext/` runs Qwen3.8-Flash-Next-MLX-oQ4 (176B: 121B routed experts, 51B
hashed n-gram, 4B remaining tensors) on a 16 GB M4 by keeping the two large tensor
families on the SSD.

```
checkpoint on disk   111.70 GB
resident              4.19 GB
load                     2.1 s
decode                  1.00 tok/s
prefill                 0.77 tok/s
```

Stock mlx-vlm materializes every weight, which on this machine means macOS grows
swap until the boot volume fills. Measured before this work: 24 GB of swap in two
minutes, RSS never above 30 MB, no token produced.

## Streaming layer

| File | Role |
|---|---|
| `store.py` | `np.memmap` per tensor, `MADV_RANDOM`, row gathers in C |
| `expert_cache.py` | reads the routed experts per layer, 9 reads in parallel |
| `ngram.py` | n-gram rows read per lookup instead of holding 19 GB |
| `loader.py` | swaps modules in before anything materializes |
| `patch_rmsnorm.py` | fixes an upstream correctness bug (below) |

Output is bit-identical to the dense path (`max_diff = 0.0`) at every token count
tested, from 1 to 800.

## Decode is at the drive's limit

A token needs the routed experts of all 48 layers:

```
per token          1475 MB
gather throughput  1068 MB/s
                   --------
pure I/O            1381 ms
measured            1390 ms
```

The 48 host syncs that routing forces cost **8 ms in total**, not the hundreds of
milliseconds assumed early on. There is no remaining slack in the code. Reading
fewer bytes is the only lever, and that means a different checkpoint.

## Optimizations that worked

| Change | ms/token | tok/s |
|---|---|---|
| starting point | 7960 | 0.126 |
| `np.memmap` gathers instead of per-row `np.frombuffer` | 5298 | 0.189 |
| batched reads, LRU cache removed | 2735 | 0.366 |
| nine parallel reads per layer | 1840 | 0.543 |
| 16 IO workers | 1788 | 0.559 |
| restore index sorting | 1671 | 0.598 |
| `MADV_RANDOM` on the numpy view, chunked gathers | 1390 | 0.719 |
| adaptive `top_k`, threshold 0.85 (default) | 1000 | 1.00 |

## Rejected optimizations

Three expert-cache designs lost. At capacity 96, the hit rate
is 76.5% and decode runs 4257 ms/token against 1390 with no cache at all.
Merging hits with `mx.stack` leaves a lazy node per call for the next `mx.eval`;
merging with `np.stack` instead still loses because cached rows are views that
pin the whole chunk they were read in. Routed sets never repeat between
tokens (0% exact over 480 samples, 35.7% overlap), so no whole-result cache can
hit.

Cache-warming prefetch lost at 1870 versus 1840 ms/token. NVMe bandwidth is the
scarce resource, not parallelism: 907 MB/s at 16 concurrent readers, 615 MB/s at
32. Warm threads steal from real reads.

Bulk tensor reads lost on prefill at 208 s against 124.6 s. A prefill routes
about 400 of 512 experts, so bulk reads 28% more than needed, and 943 MB per
layer evicts the page cache later layers want.

Sixteen IO workers gave the best result. Thirty-two workers took 212.9 s against 120.5 s
on the same prefill.

Both rejected paths remain behind `FLASHNEXT_WARM` and `FLASHNEXT_BULK`.

## Accepted adaptive top_k change

Taking experts until the router's cumulative probability passes a threshold,
rather than always taking ten. Alternating runs, default against shipped
routing:

```
threshold 0.85    981 / 1017 ms/token    1.00 tok/s
threshold 1.00   1832 / 1831 ms/token    0.55 tok/s
```

1.86x, and the cut in bytes is only ~15%. The rest is the page cache: below 0.85
the routed working set crosses what the machine holds. That is why 0.90 gives
almost nothing (1675 ms) and 0.85 nearly doubles, while 0.80 and 0.70 add little
more, the cache benefit having already been taken.

Cost: at 0.85 facts and code stay right, but the greedy path can differ. Over ten
prompts at ten tokens each, threshold 0.70 reproduces 7/10 token-for-token and
the three that differ pick an equally correct continuation. Gold is Au at every
threshold.

A fixed lower `top_k` removes the last experts when the router is undecided.
The removed experts carry enough weight to change the result. Adaptive selection
keeps them on these tokens.

## Additional measurements

Fixed lower `top_k` reduced quality. Cutting 10 to 6 roughly triples decode
(587 ms against 2056), confirmed in both sweep directions so it is not a cache
artifact. But quality goes with it: `2 + 2 =` answers `4` only at `top_k=10`, and
breaks at 8. Facts survive longer than arithmetic. The failure mode is unsafe
for coding tasks.

Adaptive `top_k` kept quality but saved little. Taking experts until the
router's cumulative probability passes a threshold holds `2 + 2 = 4` down to 0.90,
where fixed cutting had already broken it. The catch: mean k only falls from 10.0
to 8.8, because the router does not concentrate its mass. There is no negligible
tail to trim.

Checkpoint repacking did not help. Storing each expert's nine arrays
together looked like a 1.84x win (660 MB/s against 359), but that compared a
4.7 GB repacked file to a 45 GB pool, so it measured page-cache residency, not
layout. Reading both layouts over the same three layers, interleaved in one
process, the repacked path is *slower* (21-31 GB/s against 43-48), because
per-expert slicing needs a Python loop where the original uses one vectorised
numpy gather. The sequential/scattered gap (2292 MB/s against 1068 MB/s) is real
but a repack does not capture it.

Biases cannot be skipped. They are not a symmetric-quantization artifact:
std 0.035, zero entries 0%.

`MADV_WILLNEED` costs 1.5x. `MADV_NORMAL` is slightly faster than `MADV_RANDOM`.

## Measurement variance

Run-to-run variance on an unchanged configuration reached 2x (1390 to 2920
ms/token). The page cache falsified five separate measurements during this work:
the repack comparison, the sequential-read test, two layout benchmarks, and a
`top_k` sweep that had to be re-run in reverse order to prove the effect was
real. With a 45 GB expert pool against roughly 8 GB of usable cache, no naive
timing survives. Only changes above ~20%, reproduced under a control, are
trustworthy.

## Prefill is the open problem

93 tokens take about 120 s. A MoE prefill routes ~290 distinct experts per layer
against decode's 10, so it reads 25.35 GB to process 93 tokens. Chunking the
prompt makes it worse, not better: 16-token chunks would read 69 GB in total
because each chunk re-reads most of the pool.

The GPU is idle throughout. `gather_qmm` accounts for 3.2 s of a 121 s prefill.

## The upstream bug

`mlx_vlm/models/qwen4_exp/language.py` applies `y * (1.0 + weight)` in
`Qwen4ExpRMSNorm`, documented as "checkpoint weights are centered at zero". They
are not. Measured across every norm family in the checkpoint:

```
norm_conv       mean +0.8906   min +0.543
norm_key        mean +0.8933   min +0.672
q/k_layernorm   mean +0.9306   min +0.500
q/k_norm        mean +1.3576   min +0.098
hc_norm         mean +3.7498   min +0.633
```

Every minimum is positive; a zero-centered tensor would be half negative. The
`1.0 +` doubles the gain of every norm in the model. It produces no NaN and no
overflow, so hidden states look healthy layer by layer and the failure reads as
fluent nonsense. With the fix, "The capital of France is" returns " Paris" at
44.4%; without it, punctuation at 3%.

Report this issue to `Blaizzy/mlx-vlm`.

## Requantization quality failure

The 2.2x from a 2-bit expert pool was the centrepiece of a plan to reach 2 tok/s.
It does not survive measurement. Relative error in an expert's output, taking
the current 4-bit/group-32 weights as the reference:

```
recipe        eff. bits   MB/token   err gate   err h   err output
4bit/g128          4.25       1254       9.6%   14.5%        15.7%
3bit/g64           3.50       1032      20.6%   30.6%        33.0%
2bit/g64           2.50        738      40.8%   60.9%        64.7%
2bit/g128          2.25        664      44.4%   67.1%        71.9%
```

SiLU gating amplifies: 44% error on `gate` becomes 67% on `h`. Even the mildest
recipe injects 15.7% for a 1.18x gain, past the 11.7% RMSE gate this project
used to reject the sparse MLP. Two causes: requantizing an already-4-bit tensor
re-grids a coarse grid, and this model has no compression slack anyway. Vontra's
own oQ2, quantized straight from BF16 with no stacking, was declared incoherent
by its author.

Taken with the rest of this file, the pattern is consistent: experts are
full-rank, usage across them is uniform by training design, co-activation is
weak, and intra-expert sparsity is real but unpredictable. A 512-expert MoE with
auxiliary-loss balancing is built to have no exploitable redundancy.

## Remaining speed mechanism

Requantizing the experts from 4-bit group-32 (5 effective bits) to 2-bit
group-64 would cut bytes per token from 1475 MB to 737 MB, so roughly 1.45
tok/s. It needs disk room this machine does not have (11 GB free against a 45 GB
expert pool), so it would have to rewrite in place with no backup, and
requantizing from oQ4 stacks error on error rather than starting from the BF16
source, which does not fit either.
