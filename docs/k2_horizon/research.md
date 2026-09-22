# K2-Horizon 7B research record

This is our active record of K2-Horizon measurements, rejected ideas, and
design decisions. Current operation belongs in [`handoff.md`](handoff.md).
Exact commands, the compact results table, and retained JSONL arms are in
[`results/k2_horizon/`](../../results/k2_horizon/).

## 2026-09-16 — Baseline and memory study

### Scope and environment

We measured the installed `K2-Horizon-7B-MLX-8bit` checkpoint on an Apple M4
`Mac16,12` with 16 GiB unified memory, using MLX 0.32.2 and MLX-LM 0.31.3.
The checkpoint contains Q8/G64 weights and 36 ordinary BF16 `KVCache` layers.
Its recorded weight files occupy 9,561,907,200 bytes.

The retained harness uses a fresh child process for every arm, writes raw data
before validation, reverses condition order between rounds, and records the
checkpoint and imported source fingerprints. Greedy comparisons require exact
token digests. We keep sampled diagnostics separate from speed promotion.

### Baselines

| Run | Prompt → output | Median throughput | Correctness |
|---|---:|---:|---|
| Product baseline | 2,731 → 256 | 9.680 tok/s overall; 9.676 tail | 3/3 digest matches |
| Short greedy | 2,731 → 32 | 11.752 tok/s | 3/3 digest matches |
| Sampled diagnostic, seed 7 | 2,730 → 32 | 11.682 tok/s | 3/3 digest matches |

The product run is our sustained baseline. The short and sampled rates are
diagnostics, not substitutes for the product horizon.

### Memory floor

For batch size one and BF16 KV, useful cache payload is:

```text
2 (K and V) × 36 layers × 8 KV heads × 128 dimensions × 2 bytes
= 147,456 bytes/token = 144 KiB/token
```

| Retained tokens | Useful BF16 KV payload |
|---:|---:|
| 8,192 | 1.125 GiB |
| 16,384 | 2.250 GiB |
| 32,768 | 4.500 GiB |
| 65,536 | 9.000 GiB |

At 32K, weights plus useful KV already approach 13.4 GiB before temporary
buffers, allocator pools, macOS, and other applications. The advertised 512K
context would require 72 GiB of BF16 KV alone. We therefore measure overhead
above the useful payload instead of promising a multiple-fold context gain on
the 16 GiB reference machine.

The native cache grows in 256-token blocks. At the 2K control its padding was
about 7.8 MB, so padding did not establish a premise for a replacement cache.

### Prefill comparison

We compared prefill step 256 with the 512 control using three reversed pairs.

- At 2K, 256 lost 0.483% decode throughput (2SE 0.268). MLX peak fell from
  10.338 to 10.199 GB, but process footprint did not fall.
- At 8K, both conditions peaked at 11.087 GB. The 256 candidate's prefill was
  6.11% slower (2SE 9.18); decode was 1.891% faster (2SE 4.737), unresolved.
- All six arms at each context retained the required greedy digest.

Decision: retain prefill step 512. We did not run 128, 16K, or 32K because the
8K comparison showed no peak benefit and the 16 GiB machine had no useful
headroom premise.

### Residency comparison

We wrapped actual lazy-generator consumption with MLX-LM `wired_limit`, while
holding checkpoint, BF16 KV, prompt, sampler, prefill, and allocator policy
constant. Wired residency lost all three pairs: mean decode delta -0.557%
(2SE 0.527). Tokens remained identical.

Decision: reject `wired_limit` and keep it off by default.

### Allocator and cache-growth decisions

The short control's MLX allocator pool was about 3.1 MB. We therefore skipped
allocator-cap and post-generation-clear comparisons: no material pool existed
to remove. The backend retains these as guarded opt-in diagnostics, not normal
chat defaults.

We also skipped cache-step and reserve-cache benchmarks. The native cache
padding was small and no trace showed costly growth. `models/k2_horizon/cache.py`
keeps instance-level 256/1024 step selection testable without mutating the
dependency's global class, but normal generation retains the native setting.

### Compute work not started

No complete-runtime trace established a compute or submission bottleneck large
enough to justify `mx.compile`, grouped RMSNorm fusion, projection packing, or
a custom kernel. We did not create an optimization layer or modify checkpoint
and dependency sources.

If profiling supplies a premise, try one bounded candidate in this order:

1. compile one stateless pure operation;
2. fuse grouped RMSNorm and its scale multiplication while preserving FP32
   reduction and BF16 rounding boundaries;
3. pack one compatible Q/K/V or gate/up projection family while retaining
   Q8/G64 arithmetic.

A microbenchmark is insufficient. Promotion requires a resolved complete-
backend gain and exact candidate values and token digests.

## Retained runtime work

The research implementation added a reproducible benchmark harness and made
runtime lifecycle checks explicit without changing defaults:

- append-before-validation JSONL records preserve failed arms and tokens;
- checkpoint, runtime, and dependency fingerprints identify every run;
- token-arrival windows separate short startup effects from product tail;
- all cache-layer offsets are checked against the token tape;
- generator cleanup covers normal completion, stop, cancellation, and callback
  failure;
- manual cancellation can continue when the cache/tape invariant is valid;
  partial prefill and failed turns require reset/replay before reuse;
- tiny real-MLX tests cover native cache boundaries without loading weights.

## Promotion rules

Future comparisons must freeze checkpoint, prompt IDs, template, sampler,
reasoning effort, token limits, interpreter, and source fingerprints. Use one
fresh process per arm, at least three arms per condition, and
forward/reverse/forward ordering. Avoid arbitrary warmups on the fanless Mac.

Report paired effects and the two-standard-error resolution band alongside
the raw arms. You decide what promotes based on that evidence. Different
matrix shapes can change rounding, so exact operations require intermediate
equality and whole-run greedy digest checks. Output-changing work also
requires the sampled quality gate in `CONTRIBUTING.md`.

We keep Q8/G64 weights, BF16 KV, every context token, all layers, and current
model semantics. Lower precision, alternate checkpoints, layer skipping,
sliding windows, context truncation, vocabulary shortlists, changed reasoning
budgets, disk offload, KV recomputation, and speculative decoding are outside
this research scope.

## Product-path profiling on k2-research

We added an opt-in `profile` comparison without changing runtime defaults.
We ran six fresh-process arms in AB/BA/AB order with 2,731 prompt tokens
and 256 greedy output tokens. All six arms retain digest
`e7031fcb1943d25d3269e8e384c40577d9f0dfda24b3f247afe7668137f65048`.
The raw record is `results/k2_horizon/k2-profile-product.jsonl`.

Our control median is 11.314 tok/s. This is not an improvement over the
historical 9.680 tok/s baseline: environment and run conditions differ.
The profiled condition differs by +0.620% on paired rates, with a 0.096%
two-SE band. This measures instrumentation effects, not an optimization.
Three pairs do not establish sign-test significance (two-sided p=0.25).

Across the three decode profiles, `generate_step` self time accounts for
97.757% of recorded wall time. This includes native execution and waits.
Text decoding plus protocol translation takes 67.845 ms across 67.541 seconds,
about 0.100%. We have no premise for a 5% gain from removing this text work.
The profiles exclude checkpoint loading and terminal UI.

We also captured one Metal System Trace diagnostic. The process exits normally.
Our exported intervals identify Python PID 62240 explicitly. Merged active
compute intervals cover 23.515 of 23.530 seconds in the decode snapshot window
(99.936%). These intervals identify command buffers, not individual kernels.
They cannot separate arithmetic, memory stalls, or synchronization.

The prefill snapshot window contains 16.672 seconds of active intervals across
25.628 seconds. The remaining 8.956 seconds has no attributed cause here.
We must identify that cause before claiming removable prefill work.
Snapshot correlation uses the trace wall-clock origin; it is not a kernel marker.
Lookahead submission crosses the prefill boundary, and decode includes cleanup.

We retain the Metal child record in `results/k2_horizon/k2-metal-diagnostic.jsonl`.
The full trace and interval export remain temporary local artifacts, as detailed
in the measurement index. This single diagnostic does not establish a speed gain.

Our next step is kernel-level GPU attribution and prefill gap attribution.
We do not select compilation, normalization fusion, or projection packing yet.
No model, dependency, weight, cache, or sampling implementation changes result
from these measurements.

## Decision

No candidate produced a promotable gain. We retain prefill 512, native BF16 KV
growth, allocator cap and post-generation clear off, and `wired_limit` off.
The [measurement index](../../results/k2_horizon/LEGACY.md) is authoritative for exact
commands, byte counters, timings, digests, and raw artifacts.
