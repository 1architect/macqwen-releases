# Flash-Next research

This file is the single active research record for Flash-Next. It preserves the chronological experiments below. Current operation belongs
in [`handoff.md`](handoff.md). The original streaming architecture study remains archived at
[`docs/archive/flashnext/streaming-architecture.md`](../archive/flashnext/streaming-architecture.md).

## Research summary

- Streaming selected expert and n-gram rows makes checkpoints larger than RAM
run. The current oQ3-MTP checkpoint has 86.2 GiB of model weights.
- Adaptive routing crossed a page-cache working-set boundary.
- Exact-quality residency stabilizes the generation tail when RAM is free.
- Large-prompt allocator release restored decode performance after prefill.
- Prefill throughput increases with prompt length. A prompt near 5,000 tokens
may reach about 40 to 50 tok/s under favorable conditions. Controlled synthetic runs reached higher rates, including 62.19 tok/s at 5,002
tokens.
- Repacking, prefetch, weight caches, and in-process overlap failed controls.
- Exact MTP and speculative paths did not improve the complete runtime.

## Original chronological record

Every measured experiment, in the order it was run. Read [`handoff.md`](handoff.md) first: it holds the current state, how to run the model,
and the rules. This file holds the evidence behind them. Nothing here is a proposal. Each section records what was tested, what it measured,
and why it was kept or rejected. Before proposing an optimisation, search this file for it. The test registry in the handoff is the index.

## Historical prefill bottleneck

This section records the first implementation state. Later sections supersede its rates and proposed next steps. Prefill runs at 0.77 tok/s.
93 tokens take about 120 seconds. A MoE prefill routes about 290 distinct experts per layer against decode's 10, so it reads 25.35 GB to
process 93 tokens. Chunking the prompt is worse, not better: 16-token chunks would read 69 GB total. The GPU is idle throughout.
`gather_qmm` is 3.2 s of a 121 s prefill. This is where the remaining work is. MACQWEN integration: `macqwen/session.py` now appends only
the new turn to its live cache. Context images remain separate work.

## Native MTP tests on 2026-08-27

- Downloaded only the 76 native MTP tensors: 1.63 GB.
- Added mixed-precision inference for MTP modules missing from `config.json`.
- Fixed adaptive top-k for multi-token forwards.
- Five-token prefill improved from 10.17 s to 7.23 s, 29% faster.
- Exact MTP depth 4 reached 0.37 tok/s with 34% draft acceptance.
- Exact MTP depth 1 reached 0.72 tok/s with 89% draft acceptance.
- Target-only threshold 0.30 reached 1.98 tok/s, but factual output failed.
- Target-only top-1 reached 3.08 tok/s, but output became unusable.
- MTP-only reached 44.3 tok/s, then entered repetition and lost coherence.

Keep MTP disabled by default. Exact MTP adds more streamed work than it saves on this 16 GB machine. Later I/O work raises top-1 above this
result. See the final continuation.

## Non-quantized path limits

Later tests changed the physical result, but not the quality conclusion.

- Positioned `pread` replaces NumPy fancy-indexed `mmap` for normal routing.
Reversed tests measure 1.64 to 2.09 tok/s, versus 0.84 to 1.02.
- DLPack removes the NumPy-to-MLX copy. Shared-expert overlap adds about 3%.
- At 1.4 experts per layer, shared-shard `mmap` beats `pread` by about 34%.
- One shared `mmap` per shard adds about 3% over one map per tensor.
- Disabling reasoning cuts the first prompt from 68 to 28 tokens.
- `--fast` keeps threshold 0.20 globally and 0.40 on 12 sensitive layers.
It does not renormalize scores after removing experts.
- Warm raw-prompt tests cross 5 tok/s. A real 40-token Portuguese chat answer
measures 3.21 tok/s and shows lexical quality loss.
- Top-1 averages 4.75 tok/s and sometimes reaches 5.68 tok/s. It outputs
control tokens instead of an answer for the seasons prompt.
- Disabling PLE averages 5.36 tok/s. It outputs code fences and control-token
repetitions. PLE is required. Rejected in this continuation:

- `preadv` ties `pread` normally and loses to `mmap` at low routing.
- Partial score renormalization changes correct answers.
- `MADV_NORMAL`, `MADV_SEQUENTIAL`, and `MADV_WILLNEED` lose.
- A last-result cache hits 17% to 22%, but gives no speed gain.
- Grouping nine reads into three tasks loses about 12%.
- Eight and twelve I/O workers do not beat sixteen.
- Parallel n-gram reads give no measurable gain.
- Running PLE every two or four tokens gives no speed gain.
- BF16 router softmax loses to FP32 accumulation.
- Softmax over only ten selected router values is slower.

The non-quantized quality ceiling remains below 5 tok/s. Every tested configuration above 5 tok/s destroys useful output.

## Exact-path tests

More non-quantized paths were measured after the final continuation.

- Oracle speculative verification reaches 1.53 tok/s at block 1, 2.21 at
block 8, and 3.17 at block 24. These numbers assume a free, perfect draft. A real draft must be slower and has lower acceptance, so it
cannot reach 5.
- Lossless Zstandard compression reduces packed expert weights by 4%.
Scales and biases compress better, but the combined row shrinks only 10%.
- The normal profile requests 37.60 GB for 32 tokens. macOS reads 16.16 GB
physically, so the page-cache fill rate is 57%. It runs at 2.06 tok/s.
- The fast profile requests 8.77 GB and reads 2.82 GB physically. Its fill rate
is 68% and it runs at 3.30 tok/s, but the Portuguese output is broken.
- Pinning the six experts selected during the first eight generation tokens
uses 0.93 GB and improves 1.61-1.62 to 1.65-1.68 tok/s, about 3%. This is below measured variance. Pinning 1.85 GB is slower. Updating the
pinned set every eight tokens is also slower.
- Loading those six experts as reusable MLX slabs ties page pinning at
1.23 versus 1.20 tok/s. The extra merge path has no useful gain.
- Pinning two additional recurring fast experts uses 1.24 GB. Its repeated
gain is only 9%, below variance. Four extras use 1.54 GB and fall to 1.40 tok/s against a 1.49 tok/s control.
- Allocating 288 tail slots globally improves future mass coverage from
28.2% to 30.2%, but increases active experts and falls from 1.63 to 1.47 tok/s. A 192-slot global set covers 24.2%, keeps coherent output,
and measures 1.61 tok/s. Cost-aware allocation loses the same tail mass.
- `F_NOCACHE` for n-gram rows ties the control. Using it for experts is 8%
slower. Disabling cache only during prefill also gives no gain.
- Sorting expert reads, clearing MLX's 26-87 MB temporary cache, and forcing
router-first scheduling all tie or lose against the control.
- Selective layer protection improves the fast response but still emits
lexical errors, while speed falls from 2.64-2.85 to 2.19-2.43 tok/s.
- Post-cut score renormalization makes the low-I/O profiles incoherent.
- `F_RDAHEAD=0` is about 13% slower than the default.
- `F_RDADVISEV` averages 0.89 tok/s against 0.98 for direct `pread`, 10%
slower. Its `NOAGE` flag returns `EPERM` on this checkpoint filesystem.
- A 1024-row cache per n-gram shard has only a 4.9% hit rate. It averages
0.75 tok/s against 0.99 without the cache, while token IDs stay equal.
- Metal I/O returns byte-identical rows. On 15.7 GB of cold random ranges,
batch 8 improves only 3%. Batches 2 and 4 lose 2% to 3% against `pread`. Broad RAM pinning confirms the kernel already uses free RAM
efficiently. A narrow eight-expert normal-route set is the useful exception below.

## Contribution-oracle and fixed-overhead tests

Two remaining non-quantized hypotheses were tested and rejected. The expert-contribution oracle evaluates all top-10 expert outputs, then
checks every possible subset. Over eight decode tokens and 384 layer-token rows:

- threshold 0.85 selects 7.93 experts per layer;
- the best possible subset at the same local output error selects 7.92;
- perfect information saves only 0.1% of expert slots;
- router-score and measured-contribution ranks have rho 0.877;
- their top expert matches on 75.5% of rows.

The router score is already nearly optimal for choosing a smaller subset. An output-based controller cannot recover the quality lost by
`--fast`. It also must compute an expert before observing its output, so it cannot save runtime I/O directly. The retained diagnostic is:

```bash
./models/flashnext/bench_contribution_oracle.py --tokens 8
```

A proposed hyper-connection profile inserted `mx.eval` after every component. The profile added 192 GPU synchronizations per token and mixed
work between timing buckets. Its reported 89.3 ms per token is not a valid component cost. A smaller algebraic test concatenated
`input_mix_weight_down` with `block_inject_weight`. It removed 96 quantized-matmul calls per token. Paired A/B/B/A testing measured 1.22
tok/s against 1.23 for the control. Logits also diverged by up to 1.588. The code was removed. `mx.fast.rms_norm` was also removed.
End-to-end testing measured 1.68 tok/s against 1.75 for the original implementation, and generated tokens diverged. The results close the
remaining software-only subset and fixed-dispatch paths. The next material lever must reduce stored expert bytes or add physical storage
bandwidth.

## Rank-selective precision tests

A final in-memory probe kept the highest-score experts in the current Q4 and requantized only lower-score routed experts. It did not write a
checkpoint or change the Q4 decode trajectory. Four decode tokens across seven distributed layers produced 28 layer-token samples. The local
error target was the selected threshold-0.85 Q4 MoE output:

| Tail format | Exact Q4 ranks | Expert bytes | Mean local error | P90 |
|---|---:|---:|---:|---:|
| Q2/G32 | 2 | 70.0% | 31.49% | 44.77% |
| Q2/G32 | 4 | 80.0% | 22.55% | 31.22% |
| Q3/G32 | 2 | 85.0% | 16.48% | 23.83% |
| Q3/G32 | 4 | 90.0% | 11.70% | 16.55% |
| Q3/G32 | 5 | 92.5% | 9.73% | 15.01% |

The only profile below 10% mean error saves 7.5% of expert bytes. This is too small to justify a second runtime path. Lower-byte profiles
add large error to the already approximate threshold-0.85 output. The probe code was removed. Storage also blocks a safer sidecar build. The
machine has 8.66 GiB free. The current expert pool is 70.31 GiB. A complete Q2/G128 sidecar needs 31.64 GiB; Q3/G128 needs 45.70 GiB. Use at
least 40 GiB of external free space for Q2 or 55 GiB for Q3. Keep the original checkpoint immutable. Q4 codes can be split into nested Q2 or
Q3 bitplanes without the BF16 source. Avoiding the extra copy requires replacing the original representation. Recovering Q4 then needs a
residual matmul or a custom kernel. The measured quality and speed ceilings do not justify that destructive format change.

## Fast-draft speculative experiment

`--speculative-fast` uses `Qwen3.5-0.8B-MLX-4bit` as a draft model. It occupies 627 MB and has the same 248,077-token ID map as the target.
A custom Qwen4Exp block verifier commits only tokens selected by exact-quality routing. It keeps separate target and draft caches. Saved
sessions store the target cache only. The experiment preserves all compared greedy tokens. It does not improve speed on this machine:

| Draft block | Exact | Speculative | Acceptance |
|---:|---:|---:|---:|
| 8 | 1.65 tok/s | 1.33 tok/s | 100% |
| 16 | 1.51 tok/s | 0.32 tok/s | 44% |

The external 0.8B draft also stayed exact, but did not pass the target:

| Draft block | Exact | Speculative | Acceptance |
|---:|---:|---:|---:|
| 8 | 1.25 tok/s | 0.93 tok/s | 67% |
| 4 | 1.39 tok/s | 0.97 tok/s | 71% |
| 2 | 1.35 tok/s | 1.29 tok/s | 82% |

The block-16 result replayed 34 rejected or partial tokens. Keep this mode optional. The default remains `exact-quality`.

## Whallm MXFP4 evaluation on 16 GB

The pinned Whallm checkpoint was downloaded and verified:

```text
model     ~/models/Qwen3.8-Flash-Next-MXFP4
source    Yanun/Qwen3.8-Flash-Next-MXFP4
revision  753d0aa57059fad70a5f7e6cc249f25df56bbd34
files     57 manifest files, all sizes verified
storage   125,291,490,955 bytes in manifest, about 117 GiB on disk
runtime   Whallm c86532e6585a9a4d0d6d36222049b0721f926701
```

This checkpoint is already text-only. Its manifest contains 1,069 language tensors and no visual, image, video, audio, or MTP tensors.
`vision_config` is only unused metadata. The large files are all required:

| Data | Bytes | Purpose |
|---|---:|---|
| `common.bin` | 9,895,409,152 | text model common tensors |
| `ngram.bin` | 51,200,245,760 | text n-gram store |
| 48 expert files | 64,172,851,200 | routed text experts |

During the evaluation, the Python 3.12 environment used Whallm's pinned runtime: MLX 0.32.0, MLX-LM 0.31.3 from Blaizzy commit `5c105381`,
NumPy 2.3.5, Transformers 5.12.1, and tiktoken 0.13.0. This conflicts with the installed `mlx-audio` and `mlx-vlm`, which require
Transformers 5.14 or newer. The public runtime listed at the top of this document is now restored. `pip check` passes.

### Measured results

All valid greedy tests used the same prompt. Their first 16 generated tokens had SHA-256
`f929948c24b0f9feb5eebcaa2ab7ac456e92b2ae85bb52f31fe94a3e4c8dc898`. Cache size and scheduling did not change those tokens.

| Profile | Decode | Short prefill | Peak active memory | Result |
|---|---:|---:|---:|---|
| 64 slots, 4 readers, FP8 KV, ready order | 1.14 tok/s | 1.52 tok/s | 10.20 GB | baseline |
| 128 slots, 4 readers, FP8 KV | 0.88 tok/s | 1.20 tok/s | 10.37 GB | reject |
| 512 slots, 4 readers, FP8 KV | 0.28 tok/s | 0.64 tok/s | 11.38 GB | swap, reject |
| 64 slots, 8 readers, FP8 KV | 0.82 tok/s | 1.45 tok/s | 10.20 GB | reject |
| 64 slots, 2 readers, FP8 KV | 1.14 tok/s | 1.64 tok/s | 10.20 GB | tie |
| 64 slots, 2 readers, FP8 KV, no ready order | 1.17 tok/s | 1.49 tok/s | 10.20 GB | small gain |
| 64 slots, 2 readers, BF16 KV, no ready order | 1.45 tok/s | 1.69 tok/s | 10.20 GB | best short run |
| previous row, 58 output tokens | 1.19 tok/s | 1.76 tok/s | 10.20 GB | sustained result |
| previous row without FP4 index cache | 1.11 tok/s | 1.42 tok/s | 10.20 GB | reject |

The sustained run read 78.71 GB of logical expert data and 51.67 GB from the process disk counter. It reached 3.09 GB/s and used 1.19 GB of
logical expert data per generated token. Exact energy was not recorded because macOS requires administrator access for `powermetrics`. A
5,001-token prefill used 512 slots, BF16 KV, and a 14 GiB MLX limit. It did not produce a token after more than nine minutes. Swap reached
6.7 GB, free disk space fell below 3 GiB, and the machine required a reboot. Do not repeat this profile on a 16 GB Mac.

### Whallm evaluation result

Whallm needs a large expert cache to reach its published speed. On this 16 GB Mac, a large cache causes swap and severe slowdown. A 64-slot
cache avoids swap, but its hit rate is effectively zero. It must stream about 1.2 GB per token. The best sustained Whallm result is 1.19
tok/s. The original exact-quality backend reached about 1.9 tok/s while using 4.19 GB resident memory and less model storage. Whallm
therefore uses more RAM and disk while running about 37% slower on this machine. Stop this path. Resume the original FlashNext backend.

## Exact token-prediction results after restore

Tests used exact greedy token IDs. Each candidate had to match the baseline. Timing varied with the macOS page cache, so small gains were
rejected.

| Candidate | Baseline | Candidate | Result |
|---|---:|---:|---|
| fused argmax decode | 1.41 tok/s | 1.45 tok/s | +2.8%, below measured variance |
| fused argmax prefill, 256 tokens | 13.93 tok/s | 13.73 tok/s | -1.4% |
| self draft, depth 8 | 1.55 tok/s | 1.00 tok/s | -35% |
| lookup draft, depths 4 to 24 | 1.50 to 1.68 tok/s | 1.31 to 1.46 tok/s | -9% to -20% |
| external 0.8B draft, depths 2 and 4 | 1.57 to 1.58 tok/s | 1.10 to 1.18 tok/s | -25% to -30% |

The exact oracle used free, perfect draft tokens. Its best block-24 mean was 2.12 tok/s. Individual runs ranged from 0.96 to 2.48 tok/s
because of cache state. This result is the practical upper bound for the current block verifier. All candidates preserved token IDs. None
produced a reliable speed gain. Remove these runtime paths. Keep only the verifier and rollback correctness test. Keep `exact-quality` as
the default. The temporary 0.8B draft is removed.

## Three-token target tests

More exact, non-quantized tests used controlled A/B/B/A orders:

| Candidate | Result |
|---|---|
| verifier without rollback-state capture | 2.50 to 2.58 tok/s, about 2% gain |
| one 24-token MoE read group | 2.70 to 2.75 tok/s versus 2.53 to 2.64 |
| `pread` chunks 1, 2, 4, and 8 | 2.70 to 2.88 tok/s, no stable winner |
| shared-expert overlap | 2.59 to 2.63 tok/s versus 2.68 to 2.79 |
| compact copied cache, eight rows | 38% hits, 1.43 versus 1.73 tok/s |
| native MTP experts fully resident | 1.25 versus 1.31 tok/s at depth 1 |

The resident MTP path used 4.93 GB for the model and kept its 1.57 GB expert pool in RAM. Depth 1 accepted 88.9% of drafts. Depth 2 reached
0.93 tok/s and depth 4 reached 0.65 tok/s. All generated token IDs stayed exact. The temporary 1.63 GB MTP sidecar is removed. The best
controlled perfect-draft oracle result is now about 2.8 tok/s. This includes no draft cost and already verifies 24 correct future tokens per
block. It is below the 3 tok/s target. Exact speculation cannot reach the target on this storage path, even with an impossible perfect
predictor. On 2026-08-31, `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` replaced the local oQ4 checkpoint. It is quantized directly from official
BF16 weights. Its sensitivity-guided layout uses 3-bit base weights and 746 protected higher-precision modules. It contains 86.152 GiB of
model weights. The oQ4 measurements remain comparison controls.

## Exact expert-major layout experiment

Whallm's 3.09 GB/s peak motivated a complete fixed-stride expert layout test. The oQ4 weights were unchanged. Each layer packed all nine
expert components into one 3,072,000-byte record per expert. `preadv` scattered each record into the same final contiguous arrays used by
`gather_qmm`. The isolated physical-reader A/B was positive and byte-exact:

- current safetensor ranges: 1.97 GB/s;
- fixed-stride blob: 2.84 GB/s;
- reader-only gain: 43.8%.

Eight integrated blob layers tied the control at 1.413 versus 1.404 tok/s. A restart-safe APFS conversion then tested all 48 layers without
duplicating the 75.50 GB expert payload. All 48 first-token sequences kept SHA-256
`5d1feddf896864b566350f66daff3335beb8d01e22436c4646da56a9c1d2c2be`. The full model rejected every blob scheduling variant:

| Blob path | Decode result |
|---|---:|
| one read per expert | 1.124, 1.127 tok/s |
| three reads per expert | 1.179, 1.218 tok/s |
| nine reads per expert | 1.243, 1.178 tok/s |
| three reads, read-ahead disabled | 1.202, 1.206 tok/s |
| cache bypass | 1.037 tok/s in the first 24-token interval |

A 120-token cached run slowed from 1.306 to 1.070 tok/s across five intervals and averaged 1.165 tok/s. The isolated SSD gain did not
survive the complete model's page-cache, N-gram, and compute workload. A 3 GB/s reader peak is not a decode-throughput result. The
conversion was fully restored. Rewriting the 15 affected shards sequentially removed restoration fragmentation. The original path then
measured 1.34 tok/s cold and 1.77 tok/s warm with unchanged output. All blob runtime, conversion, and defragmentation code was removed. Do
not repeat this layout.

## Exact RAM and prediction sweep

All tests kept the same greedy token hashes. Apparent gains from separate processes disappeared under A/B/B/A tests in one loaded model.

| Candidate | Controlled result | Decision |
|---|---:|---|
| previous-token expert prefetch | 1.15 versus 1.63 tok/s | reject |
| overlap gate/up compute with down reads | 1.57 versus 1.72 tok/s | reject |
| `mmap` weights, `pread` metadata | 0.90 versus 1.95 tok/s | reject |
| `pread` weights, `mmap` metadata | 1.48 to 1.58 versus 1.78 to 1.98 | reject |
| 12 pinned experts per layer | 1.82 versus 1.84 tok/s mean at eight | reject |
| persistent pins between turns | 1.81 versus 1.81 tok/s mean | tie |
| self-draft depth 1 with rollback | 1.31 versus 1.50 tok/s | reject |
| top-1 self-draft with rollback | 1.40 versus 1.93 tok/s | reject |
| resident MTP with rollback | 1.28 versus 1.43 tok/s | reject |
| 256-expert approximate MTP slab | 1.43 versus 1.48 tok/s | reject |

The exact rollback removed all replay work and preserved token IDs, but draft and verification costs still exceeded the saved target calls.
Selective metadata pinning briefly measured 1.94 tok/s, then tied the control at 1.27 tok/s when repeated. It was a page-cache artifact. The
affine `biases` cannot be reconstructed losslessly from `scales` and an integer zero point. A complete check found 314,161,932 mismatches
among 3,774,873,600 values. Do not remove or synthesize these tensors. The perfect-draft verifier tops out near 2.8 tok/s before any
predictor cost. The failed MTP sidecar and all failed runtime paths are removed.

## Direct n-gram shard dispatch, isolated result only

The final isolated benchmark gain came from the PLE n-gram table, not the experts. The model has one PLE layer, 16 n-gram IDs per token, and
128 table shards. The stock `ShardedEmbedding` evaluates every requested ID against every shard. With streamed shards, that caused up to
6,144 positioned reads per token: 128 shards, 16 rows, and three quantization tensors. `StreamingShardedEmbedding` now resolves each ID to
its owning shard on the host. It reads only those local rows, dequantizes them normally, and restores the original position order. Weights
and arithmetic are unchanged. Set `FLASHNEXT_NGRAM_DIRECT=0` only to reproduce the old all-shard path. Controlled A/B/B/A results in one
loaded model:

| Metric | Legacy | Direct | Gain |
|---|---:|---:|---:|
| 24-token decode | 1.492 tok/s | 1.712 tok/s | 14.8% |
| 16-token decode | 1.463 tok/s | 1.628 tok/s | 11.3% |
| short prefill | 2.876 tok/s | 3.062 tok/s | 6.5% |

Two additional factual and Portuguese prompts improved decode from 1.284 t5o 1.499 tok/s and from 1.468 to 1.704 tok/s. Their paired token
hashes matched. The photosynthesis A/B/B/A hashes also matched in every run. This path is on by default for evaluation. It did not improve
the established complete-chat baseline of about 1.9 tok/s. Its accepted end-to-end gain is zero.

## Non-quantized path closure on 2026-08-29

Four additional paths were tested after direct n-gram dispatch. None met the 3 tok/s target with exact-quality output.

### Lossless affine-bias sidecar

All 7.031 GiB of expert BF16 biases have a four-bit integer zero point. Their reconstruction differs from the stored BF16 value by only -1,
0, or +1 bit. A one-byte lossless code therefore replaces each two-byte bias. A 3.515 GiB sidecar covered all 144 expert projection tensors.
The sidecar was bit-exact on 288 distributed rows. All generated-token hashes also matched. A fused Metal kernel reduced reconstruction from
144 dispatches per token to 48. Results still rejected the path:

- plain decode tied or lost by 1% to 7%;
- short prefill improved by about 4% to 7%;
- exact-quality pinning improved decode by only about 2%;
- using the saved RAM for a ninth expert lost about 2%.

The sidecar and its runtime were removed. This recovered 3.515 GiB of storage.

### Native C positioned-I/O loop

A native `pread` loop read directly into final NumPy buffers. It removed the Python bytes allocation and row-copy path. One row per native
task tied the Python path at about +2%, below variance. Two-row and ten-row tasks lost about 10% because they reduced SSD queue depth. The
native runtime was removed.

### Parallel and overlapped n-gram reads

Sixteen workers for the remaining 48 direct n-gram reads measured 6% to 7.5% above serial reads across reversed orders. This stayed below
the 20% system variance. Thirty-two workers lost 4.5%. Starting the reads before layer 0 and overlapping them with Metal gained only 1.5%.
These paths were removed. Three later exact variants also lost:

- one fused dequantization for all active n-gram shards lost 1.8% on decode;
- DLPack for the small n-gram buffers lost 4.5% on decode;
- POSIX AIO kept all token hashes but lost 12% against Python futures.

The AIO backend exposed the macOS limit directly: `kern.aiothreads` is four. Submitting 90 native reads therefore reduced useful
concurrency. All three prototypes and their compiled libraries were removed. The exact perfect-draft oracle was then repeated with direct
n-gram dispatch. A 48-token test with block 24 reached only 2.47 tok/s. Parallel n-gram reads reached 2.49 tok/s. The verifier remains below
3 tok/s before predictor cost.

### Online tail predictor

The final approximation fitted omitted MoE output during exact warmup. It used the existing shared expert as a scalar or per-channel basis.
It also pinned two to six prompt-selected experts per layer. This path used no new quantization. Several runs crossed the speed target:

- code generation reached 3.45 tok/s through EOS with a correct function;
- short Portuguese runs reached 3.08 to 3.90 tok/s;
- aggressive long runs reached 4.22 to 4.87 tok/s.

Long Portuguese output exposed real quality loss. It emitted `clorofa`, mixed English biochemical names, repeated words, and occasionally
stated a wrong energy-transfer explanation. Longer exact warmups delayed the divergence but did not prevent it. The exact 311-token control
reached 2.186 tok/s and stayed clean. The predictor and its benchmark were removed. Conclusion: 3 tok/s is reachable only after changing the
threshold-0.85 trajectory. Every tested oQ4 profile that preserves that trajectory remains below 3 tok/s. The sensitivity-guided oQ3-MTP
checkpoint now replaces oQ4. Its throughput and quality need new measurements on the shared chat path.

### Whallm-style record and selective-RAM continuation

The accepted baseline stayed at about 1.9 tok/s in the complete `chat.sh --exact-quality` path. These tests preserved generated token IDs.
Selective RAM used prompt or warmup routes to retain expert rows. Two to eight experts per layer used 0.29 to 1.18 GB. Decode stayed between
1.55 and 1.80 tok/s in paired tests. Prompt-selected experts performed worse than warmup experts. This rejects selective expert retention
for the current router. A fused Metal kernel selected resident or streamed rows inside one quantized matrix-vector operation. Its isolated
gate operation improved by about 31%. The complete 48-layer decode stayed near 1.7 tok/s. Kernel savings did not offset data preparation and
dispatch costs. A Whallm-style record stored each expert as one contiguous 3,072,000-byte row. The record joined gate, up, down, scale, and
bias data. Cached reads reached 8.67 GB/s. A fused gate-up kernel and original MLX down projection kept tokens exact. Four hot sidecars
appeared 2.7% faster, but reversed cache-controlled tests removed that gain. Later-layer records tied or lost against normal reads. The
final no-sidecar variant scattered checkpoint reads directly into one record buffer for all 48 layers. It measured 1.99 tok/s against 2.23
tok/s in the paired benchmark, a 10.9% loss. All record, hybrid, and sidecar code was removed. All generated sidecars were deleted. The
original checkpoint remains complete. The direct n-gram path plus full exact-quality warmup measured 1.858 tok/s against 1.486 tok/s for the
legacy n-gram path. This still does not exceed the accepted 1.9 tok/s complete-chat baseline. Its accepted gain remains zero.

### Full-block exact verifier

The block verifier previously split MoE work below 64 route indices. Larger groups enabled expert sorting, which changed arithmetic order
and token IDs. The streaming verifier now sends one unsorted block through each MoE layer. Normal decode and prefill keep their existing
sorted path. The 48-token perfect-draft oracle improved from 2.52 to 2.90 tok/s, about 15%. A 24-token run reached 3.04 tok/s. Both runs
matched exact greedy token IDs. The result applies to verifier progress, not complete-chat progress. The oracle supplies future tokens
without computing them. A real self-draft still lost:

| Self-draft depth | Exact decode | Speculative decode | Acceptance |
|---:|---:|---:|---:|
| 1 | 1.33 tok/s | 1.09 tok/s | 92% |
| 2 | 1.98 tok/s | 1.60 tok/s | 94% |

The draft and verifier run sequentially. Their combined cost remains above one exact decode. The accepted complete-chat gain therefore
remains zero.

### Corrected native MTP continuation

The native MTP sidecar was restored from `Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP`. Only its 76 tensors were fetched:

```text
file    model-mtp.safetensors
size    1,633,340,568 bytes
sha256  072a9ef44b06c3112d508d521da19ddc5d514eeda24a9a5c9ff792511df8e69b
```

The old MTP path did not install the Qwen4 verifier. It also used the normal multi-token forward and replayed target layers after rejected
drafts. The new path installs the Qwen4 verifier, uses its unsorted exact block, returns fused argmax IDs, and rolls cache state back
without target replay. The MTP sidecar now uses `F_NOCACHE`. Its 1.63 GB expert pool no longer evicts the target model's useful page-cache
pages. The target checkpoint keeps normal macOS caching. This changes no weights or arithmetic. Controlled exact-quality tests used eight
warmup tokens and eight pinned target experts per layer. Every generated-token hash matched:

| Test | Exact control | MTP depth 1 | Difference |
|---|---:|---:|---:|
| 24 tokens, A/B/B/A mean | 1.766 tok/s | 1.909 tok/s | +8.1% |
| 48 tokens, A/B/B/A mean | 1.585 tok/s | 1.691 tok/s | +6.7% |
| 48 tokens, warm cache | 1.879 tok/s | 1.909 tok/s | +1.6% |

The MTP acceptance rate was 92.3% for 24 tokens and 84.6% for 48 tokens. Depth 2 lost because acceptance fell to about 80%. Confidence-based
depth 2 also lost. Lower and higher MTP routing thresholds did not improve the result. The final test used the real terminal chat with the
same 21-token prompt and 48-token reply limit:

| Complete chat | Prefill | Decode | Tail |
|---|---:|---:|---:|
| exact-quality control | 3.1 tok/s | 1.9 tok/s | 1.9 tok/s |
| corrected MTP depth 1 | 2.9 tok/s | 1.6 tok/s | 1.7 tok/s |

This complete-chat result rejects MTP. The benchmark gain came from cache order and did not transfer to the product path. The corrected MTP
prototype, its benchmark, and the 1.633 GB sidecar were removed. The accepted 1.9 tok/s product baseline and accepted end-to-end gain of
zero remain unchanged.

### Resumed exact-speculation tests

The external draft was retested after the full-block verifier and exact rollback fixes. The original algorithm verified the draft's first
uncertain token. A rejection then required a separate exact target call. The revised algorithm anchors every block with the target token
already known. Corrections become the next anchor. This removed all target replay calls and kept hashes identical. Controlled 24-token
A/B/B/A results with Qwen3.5 0.8B:

| External draft | Exact control | Speculative | Acceptance | Result |
|---|---:|---:|---:|---|
| old depth 2 | 1.772 tok/s | 1.576 tok/s | 81% | reject |
| anchored depth 2 | 1.749 tok/s | 1.747 tok/s | 92% | tie |
| anchored depth 3 | 1.866 tok/s | 1.571 tok/s | 80% | reject |
| depth 3, confidence margin 1.5 | 1.720 tok/s | 1.432 tok/s | 100% | reject |

The confidence gate removed wrong drafts but created expensive short target blocks. It did not improve throughput. A larger
`mlx-community/Qwen3.5-2B-MLX-4bit` draft was tested at revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`. Its 1.6 GiB local checkpoint
improved prediction but added too much compute and reduced page-cache RAM:

| 2B draft | Exact control | Speculative | Acceptance | Result |
|---|---:|---:|---:|---|
| anchored depth 4 | 1.916 tok/s | 1.391 tok/s | 90% | reject |
| anchored depth 8 | 1.841 tok/s | 1.139 tok/s | 60% | reject |

The exact oracle then established the required block length on the current path. Across one 24-token sequence, block 24 reached 2.90 tok/s,
block 16 reached 2.20, block 8 reached 1.87, block 4 reached 1.31, and block 2 reached 1.25. Small-block order variance affected block 1 and
block 12. No draft can reach 3 tok/s while its verifier block stays at eight tokens or fewer. Two Whallm-inspired verifier changes also
failed with identical token IDs:

- sorting physical expert reads reduced 48-token block-24 verification from
2.64 tok/s to 1.18 and 1.06 tok/s;
- reusable fixed NumPy buffers with direct `preadv` measured 2.75 and 2.72
tok/s against 2.86 and 2.80 tok/s controls.
- an exact Metal kernel fused gate and up Q4 projections. Its A/B/B/A mean was
2.670 tok/s against 2.715 tok/s for separate MLX projections.
- separate Metal streams for gate and up measured 2.505 tok/s against 2.735
tok/s on one stream.
- forcing the dense full-token kernel measured about 2.545 tok/s against
2.680 tok/s for the default token-tiled kernel;
- forcing the 6-to-8-token streamed dense kernel at 24 tokens changed the
first verified token. The identity gate rejected it immediately.
- changing the exact dense token tile from two to four lost about 5%; changing
it from two to three lost about 6.6%. The fixed-buffer and fused-kernel runtimes were removed. The sorted-read switch remains only as an
explicit benchmark option. Do not enable these paths in the chat.

## Worktree and validation state on 2026-08-29

The remaining evaluation changes include direct n-gram dispatch and exact block-verifier support:

| File | State |
|---|---|
| `models/flashnext/ngram.py` | adds `StreamingShardedEmbedding`; direct mode defaults on |
| `models/flashnext/loader.py` | installs one dispatcher around all 128 streamed shards |
| `models/flashnext/bench_ngram_direct.py` | reproduces legacy/direct A/B tests |
| `models/flashnext/test_ngram.py` | checks exact output and unused-shard avoidance |
| `models/flashnext/expert_cache.py` | permits unsorted full blocks only for verification |
| `models/flashnext/qwen4_verifier.py` | verifies full streamed MoE blocks without sorting |

The speculative verifier and rollback files remain modified as research and correctness support. They do not make speculative mode the
default:

```text
models/flashnext/bench_oracle_spec.py
models/flashnext/qwen4_verifier.py
models/flashnext/speculative.py
models/flashnext/test_speculative_tiny.py
```

Validation command and result:

```bash
~/models/.venv-qwen4exp/bin/python -m unittest \
  flashnext.test_terminal flashnext.test_sessions \
  flashnext.test_ngram flashnext.test_speculative_tiny
# Ran 8 tests: OK

git diff --check
# clean
```

The real-model session snapshot test is not part of this short unit run. Its earlier measured round trip produced identical restored logits
and cache state. The Apple Terminal chat was opened after validation. It reported:

```text
ready in 1.9s   profile=exact-quality  threshold=0.85  RSS 3.43 GB
```


## Different-trajectory tests on 2026-08-30

The test allowed a different greedy trajectory. The exact-quality default did not change.

### Routing and logit fusion

Minimum expert floors and sensitivity tiers did not solve the lexical damage. A 2.15-expert profile measured 2.45 tok/s and still mixed
Portuguese and English. Fixed floors, protected layers, periodic rich-route anchors, and draft top-k filters were rejected. Resident Qwen3.5
models were then fused with the fast target logits:

- 0.8B crossed 3 tok/s, but mixed biochemical terms or lost factual detail;
- 2B reached 3.13 tok/s with good science, but retained grammar errors;
- the 0.8B plus 2B ensemble reached 3.03 tok/s, but still wrote incorrect
noun genders;
- a prompt-fitted three-feature residual estimated omitted MoE output;
- per-channel fitting caused repetition and lexical drift;
- scalar fitting was more stable, but it did not restore exact reasoning.

The retained diagnostic is `bench_logit_fusion.py`. These token-by-token profiles are not product profiles.

### Dense block fusion

This subsection records the earlier approximate implementation. It is no longer the implementation behind `--fused-quality`. The current
exact path is documented below. `mlx-community/Qwen3.5-4B-MLX-4bit` was downloaded from Hugging Face. It uses 2.9 GB on disk. Its token IDs
match the target tokenizer. The useful changed-trajectory design drafts a block with the resident 4B model. FlashNext consumes that block in
one fast multi-token forward. The target and draft logits choose the next block anchor. The exact prompt prefill also fits a scalar residual
for omitted target experts. The complete `chat.py` photosynthesis run with block 16 measured:

- prompt: 34 tokens at 2.53 tok/s;
- decode: 196 tokens at 3.21 tok/s;
- target routing: 1.87 experts per layer;
- output: six correct and fluent Portuguese sentences.

Other block-16 checks:

- Python normalization function: correct, 82 tokens at 3.08 tok/s;
- JSON facts: correct, 46 tokens at 2.41 tok/s;
- exact four-line format: correct, 19 tokens at 2.43 tok/s.

Short replies stay below 3 tok/s because one target block is a large fixed cost. Sustained replies cross 3 tok/s. The former Flash-Next chat
exposed this profile as `--fused-quality`. The shared chat does not expose this rejected experimental profile. It is experimental. It
preserves the 4B draft quality on tested explanation, format, fact, and code prompts. It does not preserve FlashNext reasoning quality. A
probability test returned 10/21 instead of 5/9. Exact-quality returned the correct 5/9 at 1.65 tok/s. Eight exact prefix tokens did not fix
the block draft. The exact 4B speculative verifier exceeded two minutes for 48 tokens and was interrupted. Qwen3.5 9B MLX-4bit was also
tested. Block 32 measured 0.45 tok/s and returned 11/21 on the same probability test. Its rejected 5.6 GB directory was deleted. Keep
`--exact-quality` as the default. Do not present `--fused-quality` as quality-equivalent until a reasoning gate or stronger draft fixes this
case. Test the terminal profile before a commit.

### Large exact-quality prefill

A pasted prompt contained 105,760 tokens. QSA tried to allocate a dense 105,760 by 105,760 BF16 selection buffer. Its 22,370,355,200 bytes
exceeded Metal's 9,534,836,240-byte single-buffer limit. The later full-vocabulary logits would also have exceeded the limit. Whole-model
prefill chunks were rejected because they repeat streamed expert reads and would remove the measured large-batch scaling. The retained path
keeps one full-sequence MoE call. Above 2,048 tokens, only QSA query rows are processed in 2,048-row internal chunks. The language head
reads only the final hidden row, so it does not materialize full-sequence prompt logits. Forced small-prompt validation compared the
original path with both safeguards active. The next token ID matched. A separate attention-only comparison had maximum logit difference 0.
Unit tests cover the QSA mask and the one-call large-prefill contract. Prompts at or below 2,048 tokens retain the original QSA code path.
The original 105,760-token paste still needs an interactive end-to-end interactive test. Do not commit before that test. The complete `gen N
@ X` metric includes the first eight unpinned tokens and the 1.24 GB expert pin operation. Compare its `tail` value with steady-state
decode. Recorded exact-quality runs range from 1.65 to 2.19 tok/s because macOS page-cache state and the routed token trajectory change disk
reads. A 50-token result at 1.7 tok/s does not establish a profile regression. A controlled warm 50-token run after this change measured
2.46 tok/s complete and 2.79 tok/s for the 42-token pinned tail. Its output remained coherent.

## Exact fused and QSA tests on 2026-08-30

### `--fused-quality` status

The current path loads `Qwen3.5-4B-MLX-4bit` only for the first short prompt. It drafts at most 23 tokens, releases the 2.9 GB draft, and
verifies the block with the exact FlashNext target. It commits only target token IDs. A weak early margin or a prompt above 512 tokens uses
the normal exact path. The 48-token photosynthesis reference keeps SHA-256
`b44b44890ba3db93c2ba5e0b55f2ae3d55a68191eba438cd30848d752af4bed5`. The draft proposed 22 tokens and all 22 matched. Warm complete-chat runs
ranged from 2.16 to 2.32 tok/s. The matching exact-quality control measured 2.07 tok/s. This remains below the 3 tok/s target and below the
20% acceptance threshold for macOS cache variance. At 96 tokens, six and eight pinned experts per layer tied at 2.272 and 2.268 tok/s. Their
token hash was `f17982cc655080f4318033af20708171c4c29547c2da9776332a77ed22c8bcd7`. Keep the existing eight-expert exact default. Rejected
exact draft variants:

- target and 4B prefill overlap improved prompt rate but reduced decode;
- 0.8B, 2B, and their ensembles did not beat the 4B prefix of 22 tokens;
- teacher-forced 4B agreement was 87.2%, with the exact token in local top-4;
- beams 2 through 32 still lost the exact trajectory by position 28;
- explicit Metal wiring at 4 and 6 GB measured 1.754 and 1.631 tok/s against
a 1.710 tok/s control;
- uncached 4B reads measured 2.036 and 2.183 tok/s against buffered controls
of 2.113 and 2.253 tok/s. Reject `F_GLOBAL_NOCACHE` for the draft. The transient LoRA distillation route remains untested. Verification
would preserve target IDs, but a perfect draft remains bounded by target block verification and cannot establish 3 tok/s alone.

### BigMoeOnEdge comparison

The public 3.48 tok/s result uses an 80.4 GB Q2_K GGUF. Its dense side is about 2.4 GB, and the shown profile enables `drop 100%`. This
changes expert routing and is not an exact-quality comparison. BigMoeOnEdge reports about 2 tok/s for its larger lossless UD-IQ3_XXS profile
with roughly 4.3 GB pinned dense data. Transferable exact techniques were checked:

- input-embedding row streaming would release 397,312,000 bytes, but three
earlier implementations already lost complete-chat speed;
- the n-gram table already streams;
- macOS `F_NOCACHE` already exists and loses on target expert reads;
- Metal wiring does not repeat Android `AHardwareBuffer` pinning gains;
- BigMoeOnEdge's canonical sparse expert address space works with CPU ggml.
MLX materialized the equivalent mapping and made one layer 22 times slower;
- a compact hit/miss Metal kernel was already tested. Its isolated gate gained
about 31%, but complete decode stayed near 1.7 tok/s. Do not use the public 3.48 tok/s number as evidence of a missing exact MLX flag. Its
main gains come from lower-bit weights and dropped cold experts.

### Retained QSA result

The old 65,536-token threshold was unsafe. A 5,002-token synthetic prompt made the original QSA graph request 12,810,242,048 bytes. The
intermediate shape was effectively 5,002 by 512 by 5,002 bytes. The retained threshold is 2,048 tokens. Only QSA query rows split into 2,048
row chunks. MoE expert reads still run once over the complete prompt.

| Prompt | Previous | Chunked QSA | Token result |
|---:|---:|---:|---|
| 3,002 tokens | 48.27 tok/s | 66.33 tok/s | first two IDs identical |
| 5,002 tokens | Metal allocation failure | 62.19 tok/s | generation starts |

The 3,002-token gain is 37%. Query chunks of 2,048 and 4,096 tied. Keep 2,048 because it bounds memory better near the 262,144-token context
limit. At 1,002 tokens, forced chunking gained only 2%, so the original path remains below the threshold. Full prompt logits are released
before decode. `mx.clear_cache()` remains for logit buffers above 256 MB. Removing that cleanup reduced the two-token cold decode from 0.114
to 0.071 tok/s in the synthetic 3,002-token test. Rejected recovery attempts after a large prefill:

- summing every resident parameter reduced prefill to 40.37 tok/s and decode
reached only 0.213 tok/s;
- pinning experts from the last prompt routes reduced prefill from 66.33 to
63.79 tok/s and decode from 0.187 to 0.148 tok/s.
- pinning eight experts after two output tokens averaged 1.04 tok/s against
0.95 tok/s after eight tokens, with identical IDs. The 10% gain stays below the acceptance threshold, so the default remains eight tokens;
- a two-stage four-expert then eight-expert pin measured 0.91 tok/s. Reject it.

Large-prefill decode remains cold because the full prompt touches most of the 70 GB expert bank. The retained change fixes the allocation
failure and speeds prefill without changing target routes or verified token IDs.

## flash-moe comparison on 2026-08-30, closed

`github.com/danveloper/flash-moe` reports 4.36 tok/s on a 397B MoE. Its techniques do not transfer to this machine. The reason is measured
below. Do not re-investigate this repository.

### Hardware constraint

| | flash-moe | this machine |
|---|---|---|
| Chip | M3 Max, 40-core GPU | M4 |
| RAM | 48 GB, about 35 GB page cache | 16 GB |
| SSD sequential | 17.5 GB/s measured | 2.29 GB/s |
| SSD on the expert gather | not stated | about 1.0 GB/s |
| Active experts | K=4 fixed | about 8 at threshold 0.85 |
| Page-cache hit rate | about 71% | 57% |

Both engines stream a similar volume per token, about 1.6 GB against MACQWEN's 1.475 GB. Their disk is about 7.6 times faster and their page
cache is about three times larger. That gap alone covers the difference in tok/s.

### Measured cost split on this machine

Three greedy decode passes ran over the same 24-token reply in one process. Physical disk bytes came from `proc_pid_rusage` with
`RUSAGE_INFO_V4`. Note that `getrusage` `ru_inblock` returns 0 on Darwin and cannot be used.

| Pass | Time | Physical read |
|---|---:|---:|
| 1 | 459.7 ms/token | 389.3 MB/token |
| 2 | 414.8 ms/token | 383.3 MB/token |
| 3 | 416.8 ms/token | 383.2 MB/token |

All three passes produced identical token IDs. The page cache does not warm between passes. Every pass reads the same 383 MB. Twenty-four
tokens need about 9.2 GB of distinct expert rows, so a pass evicts its own head before its tail runs. Repeating a prompt reuses nothing. Any
future test that assumes a warm expert working set on this machine is invalid. 383.2 MB in 416.8 ms is 919 MB/s, which matches the
established gather rate. Physical reads therefore account for about 86% of a decode token. All compute, dispatch, and host synchronization
fit in the remaining 58 ms.

### Transfer limits of the retained techniques

| Their win | Their gain | Value here |
|---|---|---|
| FMA dequant kernel | +12% GPU compute | about 3.6 ms of a 417 ms token, under 1% |
| Trust the OS page cache | +38% | already the foundation here |
| GPU combine and norm fused | pipeline | fusion tests already lost end to end |
| BLAS delta-net, Accelerate | +64% attention | they replaced scalar C; GDN already runs on Metal |
| `F_NOCACHE` for 2-bit | +3% | 2-bit only; measured 8% slower on MACQWEN's 4-bit experts |
| Fused RoPE attention | +2% | below the 20% acceptance threshold |
| C BPE tokenizer | 20x startup | MACQWEN's complete load is 2.1 s |
| Deferred expert submit | pipeline | 48 host syncs cost 8 ms per token, see `expert_cache.py:80` |

Their entire kept list targets compute. Compute is at most 14% of a token here. The same code meets the opposite bottleneck on this machine.

### Reusable repository components

Two findings remain useful. First, independent confirmation. Their 58 discarded experiments reproduce eight MACQWEN rejections in a separate
C and Metal implementation: LZ4 compression -13%, `F_RDADVISE` prefetch net 0%, temporal expert prediction -18%, an MLP routing predictor at
31% accuracy, `mmap` expert files -5x, expert file clustering 0%, spin-poll GPU wait -23%, and MTP speculative decoding at break-even.
Second, the mechanism this document lacked. They measured it directly:

> SSD DMA and GPU compute share the same memory controller and cannot be
> profitably overlapped. Even small background SSD DMA causes disproportionate
> GPU latency spikes.

Background SSD DMA cost their GPU up to 73%. That explains every failed overlap, prefetch, and route-prediction result recorded in this
document. The serial order of GPU, then SSD, then GPU is hardware-optimal on Apple Silicon. Treat any future proposal to overlap expert
reads with GPU work as rejected by this mechanism and the local timings.

### Corrected 3 tok/s requirement

An earlier version priced the token at a 927 MB/s gather rate. It concluded that only fewer bytes could reach 3 tok/s. That
conclusion was incorrect. 927 MB/s is physical bytes divided by *total token time*, so it charges the drive for every millisecond it sits
idle. It is not a bandwidth measurement. `FLASHNEXT_PROFILE_IO=1` times the blocking waits directly. Three passes, 20 tokens each, identical
token IDs:

| Phase | ms/token | Share |
|---|---:|---:|
| blocked on expert reads | 223.0 | 49% |
| blocked on n-gram reads | 6.2 | 1% |
| router host sync | 20.5 | 5% |
| everything else | 206.0 | 45% |
| **total** | **455.7** | |

The SSD delivers 1763 MB/s during reads, close to the 1.97 GB/s this document already recorded for the isolated reader. The drive is near
its practical rate whenever it is used. It is idle for about half a token. The result retires the claim in `expert_cache.py:80` that decode
is I/O bound at the drive's limit and that reading fewer bytes is the only lever. That was true when written. It does not describe the
current code.

### Requirements for 3 tok/s

3.0 tok/s is 333 ms/token against 455.7 ms measured. The gap is 123 ms.

| Lever | Mechanism | Result if achieved |
|---|---|---:|
| overlap reads with compute | the two halves are serialized today | up to 4.4 tok/s at perfect overlap |
| cut the 206 ms non-disk half | unattributed; see below | 3.0 tok/s at -60%, 4.0 tok/s at -100% |
| cut physical bytes | 393 MB at 1763 MB/s | 3.0 tok/s needs -55% of bytes |
| raise gather rate | already 1763 MB/s | closed; near the isolated ceiling |

Byte reduction is now the *weakest* of the three open levers, not the only one. A lower-bit checkpoint must cut 55% of bytes to reach the
target alone. The 206 ms non-disk half is not yet attributed. Candidates, in order of suspected size: the `to_mx` host copy of 393 MB per
token, the `gather_qmm` expert matmuls, attention and GDN, and Python dispatch across 48 layers. Measure it before proposing a fix. Note on
overlap: flash-moe measured that background SSD DMA costs their GPU up to 73% and concluded the serial pipeline is hardware-optimal. Their
drive runs at 17.5 GB/s, so their idle window is small and their DMA bursts are short. This machine idles the drive for half a token.
Re-measure overlap here rather than inheriting that conclusion.

### Retained diagnostic

The cost-split probe lives at `models/flashnext/bench_decode_split.py`. It reports time and physical bytes per decode token and asserts
identical token IDs.


## Route 1 closed by hardware on 2026-08-30

Overlapping expert reads with GPU work loses on this machine. The cause is memory-controller contention, not the dependency chain and not
prediction accuracy. Do not attempt overlap, prefetch, or read-ahead again without new hardware.

### Implementation

`_moe_call` learns the routed experts at `mx.eval(scores)`, but the reads were not issued until `switch_mlp` ran its own host sync in
`build_plan`. Between those points sit the top-k loop, the mask and normalizer graph, the shared expert, and a 22 ms GPU drain, all with the
drive idle. `switch_mlp.prefetch()` now issues the reads at the earlier point. `_one_pass` reuses those futures when the routed list matches
and falls back to a normal submit otherwise. It reads the same experts, the same bytes, in the same order. Only the moment of issue changes.

### Functional result and performance loss

A/B/B/A, one fresh process per run, identical token IDs throughout:

| Run | ms/token | Reads blocked | Read rate | Score sync |
|---|---:|---:|---:|---:|
| baseline | 499.3 | 233 ms | 1750 MB/s | 173 ms |
| early submit | 548.7 | 201 ms | 2053 MB/s | 253 ms |

Read blocking fell 32 ms and the drive reached its best measured rate. The token still lost 49 ms, because the GPU drain grew about 80 ms.

### Control results

The change did two things: it submits reads early, and it adds `mx.eval(scores, inds)` plus a `tolist()` per layer. Mode `2` pays the extra
eval and skips the submit.

| Mode | ms/token | Score sync | Top-k Python | Shared expert |
|---|---:|---:|---:|---:|
| eval only, no submit | 498.5 | 150 | 6.0 | 2.0 |
| submit early | 539.3 | 201 | 17.4 | 3.8 |

The extra eval is free: 498.5 ms against a 499.3 ms baseline. Submitting the reads early costs 8.2% on its own. With reads in flight, GPU
work grew 51 ms, the pure-Python top-k loop grew 11 ms, and the shared expert grew 1.8 ms. The Python number is the important one. A
host-side loop touching a few hundred floats has no GPU dependency. It still slowed by a factor of three while the SSD streamed.
Both the CPU and the GPU lose bandwidth to SSD DMA on this part.

### Closed paths

The controlled A/B reproduces flash-moe's finding on this machine and it explains, with one mechanism, why every one of these already lost:

- `Cache-warming prefetch`
- `Expert prefetch`, previous-token and router-first
- `Pre-attention route prefetch`, 59% expert hit rate
- `shared-expert overlap`, 2.59 against 2.68 tok/s
- `F_RDADVISE` and `MADV_WILLNEED`

Route 1 required no prediction to test, and it still lost. Better prediction cannot rescue it. Overlap is worth less than zero here.
`FLASHNEXT_EARLY_SUBMIT` defaults to `0`. Keep the code as a reproducible diagnostic: `1` submits early, `2` is the eval-only control.
Remaining routes: cut the 42 ms `to_mx` copy, cut the 186 ms GPU, cut the 246 ms of bytes. All three reduce work rather than overlap it, so
contention does not apply to them.


## Route 4 measured on 2026-08-30: copy removed, no gain

The `to_mx` bucket was not the DLPack wrap. `to_mx` already calls `mx.from_dlpack(out, copy=False)`. The cost was one line above it in
`ExpertLRU.to_mx`:

```python
block = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
```

`FLASHNEXT_PREAD_CHUNK` defaults to 1, which keeps 16 workers busy on the NVMe queue but gives every expert its own single-row allocation.
The main thread then concatenated them, copying the whole layer a second time after the read had already finished.

### Implementation

`store.rows_into()` gathers straight into a caller-owned slice. `ExpertLRU._submit_shared()` allocates one destination per part and hands
each chunk a disjoint slice of it. Reads stay as parallel as before, every chunk lands in its final position, and the concatenate
disappears. Same bytes, same reads, same order, one fewer copy.

### Copy removal result

A/B/B/A, one fresh process per run, identical token IDs throughout:

| Bucket | concatenate | shared buffer | Change |
|---|---:|---:|---:|
| `to_mx` | 37.85 ms | 3.05 ms | **-34.8 ms** |
| GPU drain at score sync | 139.5 ms | 176.7 ms | **+37.2 ms** |
| blocked on expert reads | 231.1 ms | 229.8 ms | unchanged |
| **token** | **461.9 ms** | **464.9 ms** | -0.7% tok/s |

The copy is gone, verified: 3.05 ms against 37.85 ms. The saving reappeared almost exactly in the GPU drain. Net result is a tie inside
noise. This measurement provides a second case where another stage absorbs a local improvement. Route 1 moved 32 ms off the read path and
lost 80 ms of GPU. Route 4 removed 35 ms of host copy and gained 37 ms of GPU. Both are consistent with a machine whose memory controller is
the shared constraint, where the winner is total traffic rather than which unit performs it. One untested explanation for the GPU increase:
the shared buffer is written by 16 worker threads scattered across it, while the concatenate produced the same bytes through one sequential
copy on the main thread. The GPU then reads memory in a different coherency state. This was not measured. Do not treat it as established.
`FLASHNEXT_SHARED_READ_BUFFER` defaults to `0`. The path is kept because it is strictly less work and may matter if the GPU side ever stops
absorbing it.

### Standing implication for routes 2 and 3

Route 2 cuts GPU work and route 3 cuts bytes. Both reduce total memory traffic rather than moving it between units, so neither is refuted by
these two results. But both must be measured end to end in the complete chat, because this system has now twice returned a local win as a
global tie or loss.


## Route 2 measured on 2026-08-30: the GPU work is real

An earlier note in this session claimed roughly 150 of the 186 ms of GPU time was kernel-launch overhead. That conclusion was incorrect. It
came from comparing each component against its memory-bandwidth floor, which assumes a decode kernel can saturate bandwidth. At batch 1 it
cannot: these kernels are latency-bound on small matrices, so the ratio proves nothing.

### Measured MLX sync floor

| Measurement | Result |
|---|---:|
| empty `mx.eval` round trip | 0.189 ms |
| 36 chained `rms_norm`, one eval | 0.466 ms, or 0.013 ms per kernel |
| 36 chained `rms_norm`, one eval each | 4.546 ms, or 0.126 ms per op |
| implied sync overhead | **0.113 ms per eval** |

A sync costs about ten times the kernel it waits for. But decode runs two evals per layer, `mx.eval(scores)` and `mx.eval(flat)`, so 96
evals per token is about **11 ms**, not 150. The remaining GPU time is genuine kernel execution. The measurement also corrects earlier
isolated figures: the router at 0.331 ms and GDN at 2.916 ms each include one 0.189 ms round trip.

### First `mx.compile` test in this project

`mx.compile` appears nowhere in `models/flashnext/` and in none of the earlier experiments. It works and it is bit-exact, verified with
`mx.array_equal` on every output:

| Chain | Plain | Compiled | Gain |
|---|---:|---:|---:|
| router top-k chain | 0.387 ms | 0.377 ms | 2.6% |
| `_normalize_qk` | 0.229 ms | 0.216 ms | 5.7% |
| PLE-style gate chain | 0.263 ms | 0.190 ms | 28% |

It helps pure elementwise chains and barely helps chains built around sorts, reductions, and matmuls. Those are the majority of the 186 ms.
Applied everywhere it plausibly returns 10 to 20 ms, or 3 to 4% of a token. Keep `mx.compile` in mind for any future elementwise-heavy code.
Do not expect it to move the product baseline.

### Route 2 verdict

The GPU half is real work spread over 48 layers, dominated by GatedDeltaNet at about 57 ms per token. There is no hotspot and no overhead to
reclaim. Route 2 cannot reach 3 tok/s.

## GDN projection fusion retested on 2026-08-30: confirmed reject

`Fused GDN input projections` was retested after the token profile suggested the projections were GDN's whole cost. The earlier rejection
stands, and the reason is now known. Group size 32 runs along the input dimension, so every output row is quantized independently and
concatenating rows is a layout change rather than an arithmetic one. The fused result is bit-identical, verified with `mx.array_equal` on
all four outputs.

| Path | Net time | Bandwidth |
|---|---:|---:|
| four separate matvecs, one eval | 0.392 ms | 67.2 GB/s |
| one fused matvec plus split | 0.373 ms | 70.8 GB/s |

Saved 0.020 ms per layer, 0.7 ms per token, **0.2% of a 462 ms token**. Launch overhead is not what these projections cost. They already run
at about two thirds of the 105 GB/s this machine reaches, and fusing four launches into one moves 67 to 71 GB/s. There is no 2x to recover.

### Correction to the GDN stage table above

The per-stage figures earlier in this document were measured with one `mx.eval` per stage and a fixed round trip subtracted from each. That
inflates any sum across stages, because each stage carries its own launch and drain. Measured in a single eval, GDN's four input projections
cost 0.392 ms, not the 0.674 ms implied by summing them separately. The reliable GDN figures are the whole-layer ones: **0.948 ms net per
layer, 34 ms per token across 36 layers** in isolation, against about 57 ms in situ. The claim that its projections run at 38 to 52 GB/s was
an artifact of that subtraction and is withdrawn. They run at about 67 GB/s. The recurrence result stands: `gated_delta_kernel` is a
hand-written `mx.fast.metal_kernel` and costs essentially nothing at decode. Do not optimize it.

### Measured bandwidth hierarchy

| Path | Achieved |
|---|---:|
| streaming read, 256 MB | 98 GB/s |
| bf16 matvec 16384 | 98 GB/s |
| Q4 matvec 2560, chained, one eval | 105 GB/s |
| GDN input projections | 67 GB/s |
| **SSD expert gather** | **1.79 GB/s** |

RAM bandwidth is not the constraint and never was. This machine reaches its rated bandwidth. The constraint is RAM **capacity**: 16 GB in
front of a 70 GB expert bank forces 405 MB per token across an SSD that is 59 times slower than RAM. Those bytes would take 3.9 ms resident;
they take 246 ms. `mx.compile` is also not an unexplored lever. `mlx_vlm/models/qwen3_5/ gated_delta.py` already applies
`@partial(mx.compile, shapeless=True)`, and the recurrence is already a fused Metal kernel. An earlier note in this session claiming it had
never been tried referred only to `models/flashnext/` and was misleading.


## Parallel instances and the prefetch retest on 2026-08-30

### Two instances reach 1.52x aggregate

`models/flashnext/bench_parallel.py` runs N chat instances at once and asserts they all produce the solo run's token IDs. Two instances, 40
tokens, no pinning so both fit in RAM:

| | decode | tail |
|---|---:|---:|
| solo | 1.936 t/s | 2.001 t/s |
| parallel, instance 1 | 1.473 t/s | 1.581 t/s |
| parallel, instance 2 | 1.474 t/s | 1.581 t/s |
| **aggregate** | **2.948 t/s** | |

Total throughput is 1.52 times the solo rate. Each instance runs at 76% of solo. End to end, including load and prefill, 1.062 to 1.490 t/s.
Token IDs identical. Parallel execution provides the overlap that route 1 could not achieve inside one process. It uses letting the kernel
schedule two of them. It gives the exchange rate directly: Concurrent SSD traffic costs the GPU about 31% and returns 52% more work.
Parallelism does not help a single conversation, and it competes with pinning for RAM. Two instances at the default 32 resident experts need
15.8 GB before the OS gets anything. The script estimates the footprint and refuses to start rather than swap; `--force` overrides. Pick
one: pin for a fast single conversation, or run unpinned instances for aggregate throughput.

### Byte-exact warm-prefetch result

The 1.52x result suggested filling the solo idle window with predicted reads. Projected gain at the 59% hit rate this document records was
+23%. `warm_layer()` reads the previous token's experts for this layer onto a separate pool at the top of `_moe_call`, so they proceed while
the main thread blocks on `mx.eval(scores)`. Results are discarded, so no prediction error can change output. That is the difference from
`Pre-attention route prefetch`, which substituted predicted experts for real ones. Six runs at 32 resident experts, ordered `1 0 0 1 1 0`:

| Warm | Tail mean | Runs |
|---|---:|---|
| off | 2.365 t/s | 2.221, 2.501, 2.375 |
| on | 2.002 t/s | 1.869, 2.018, 2.120 |

The tail lost 15.4 percent. Full decode lost 33 percent. Token IDs remained identical. `FLASHNEXT_WARM` stays default off. The code remains
as a reproducible diagnostic.

### In-process overlap is closed

Three mechanisms, three losses, all byte-exact and all measured today:

| Mechanism | Result |
|---|---:|
| early submit, no prediction needed | -8.2% |
| shared read buffer, removes a copy | neutral, absorbed by the GPU |
| warm prefetch, discardable reads | -15.4% |

Cross-process overlap wins 52% while every in-process attempt loses. The difference is whose critical path the overlapped work sits on.
Inside one process, I/O slows the GPU work on the main thread. The process makes no independent progress in exchange.
Across processes both sides advance. Do not propose further prefetch, read-ahead, or overlap work for a single instance. The mechanism is
understood and it is a hardware property.



## Prompt-dependent throughput, explained and closed on 2026-08-30

Throughput depends on what the model generates, by about 23%, at an identical expert count per layer. This section records the mechanism and
the failed attempts to exploit it.

### Measurement

Serial, one instance, guarded, thinking off, 32 resident experts:

| Prompt style | Tail | Experts/layer |
|---|---:|---:|
| focused technical explanation | 2.435 t/s | 7.92 |
| open-ended, "just keep talking" | 1.984 t/s | 7.95 |

### Routing locality, `bench_route_locality.py`

| Style | Reuse | Distinct/layer | Top-32 coverage |
|---|---:|---:|---:|
| focused | 42.5% | 154.5 | 69.6% |
| open-ended | 34.9% | 150.5 | 62.7% |

The working set is the same size. Open-ended output reuses fewer of the previous token's experts, so its access pattern is less clustered.
Top-32 coverage predicts throughput exactly:

```text
implied miss rate   focused 30.4%   open-ended 37.3%
predicted slowdown  22.7%
measured slowdown   22.7%
```

Both styles give the same constant: `rate = 0.740 / (1 - coverage)`.

### Coverage is a descriptor, not a lever

The model fails everywhere except at 32 experts. Extrapolated to 8 experts it predicts 1.26 tok/s; the sweep measured 2.474. Coverage curve,
from `bench_route_locality.py`:

```text
focused      8:41.2% 16:55.5% 32:69.6% 48:77.8% 64:83.6% 96:91.2%
open-ended   8:32.0% 16:45.7% 32:62.7% 48:73.8% 64:81.5% 96:91.1%
pinned GB    8: 1.15 16: 2.30 32: 4.60 48: 6.91 64: 9.21 96:13.81
```

3.0 tok/s implies 75.3% coverage, which the curve reaches at 48 experts. Tested on the open-ended prompt, alternating, 160 tokens:

| Resident | Tail mean | Pinned |
|---:|---:|---:|
| 32 | 1.798 t/s | 4.83 GB |
| 48 | 1.803 t/s | 5.65 GB |

The measured change was +0.3 percent against a predicted +42 percent. Pinning ranks 33 to 48 changes nothing because those experts are used
rarely enough that the page cache already retains them between uses. Pinning does not add residency, it locks pages that were resident
anyway. This matches every pinning result here: 8 to 32 gives +4.4%, 32 to 48 gives +0.3%, and 12 and 16 were rejected earlier. Coverage
measures how clustered the access pattern is, and clustering is what the page cache rewards. It describes the difference between prompts. It
cannot be bought.

### Periodic re-pinning

`--repin-interval N` re-selects the resident set every N tokens, since the 8-token warmup goes stale when output drifts. It confirmed drift
is real: the resident set grew 4.83 to 6.98 GB with genuinely new experts. It returned +5.2% with overlapping ranges, inside noise. Default
off.

### Prompt-locality result

The 23% is a property of what the model is asked to generate. Open-ended output routes less repetitively, so the page cache hits less often.
No residency policy addresses it, because the cost is the access order rather than the set size. Do not attempt to close this gap with
pinning.

## CPU measurement and interpretation, 2026-08-31

### Correction to the device-bound assumption

A decode token is 426 ms. Of that, 395 ms is CPU time, and cores busy averages **0.93 on a ten core machine**. Ninety one percent of the
machine is idle while a token is produced. This result suggested unused CPU capacity. The following measurements reject that interpretation.

### Existing copy overlap

`pread` copies twice: kernel to a Python `bytes`, then into the destination. `preadv` writes straight into the destination with the GIL
released.

| Read mode | ms/token | CPU ms | Cores |
|---|---:|---:|---:|
| pread | 426.9 | 388 | 0.90 |
| preadv | 426.9 | 335 | 0.79 |

CPU time fell 13 percent, but wall time stayed constant. The copying happens in worker threads outside the critical path. Removing it
reduces active cores without reducing wall time. Removing the second copy as well, with the shared read buffer, is worse:

| Read | Buffer | ms/token | CPU ms |
|---|---|---:|---:|
| pread | off | 424.3 | 388 |
| preadv | off | 417.7 | 338 |
| pread | on | 436.9 | 339 |
| preadv | on | 462.2 | 321 |

Least CPU, worst wall time. The shared buffer moves allocation and coordination onto the critical path in exchange for copying that was
free.

### GIL exclusion

Early submit lost 8.2% when first measured with `pread`, and the copy holding the GIL was a possible cause. With `preadv`, which never holds
it:

| Read | Early submit | ms/token |
|---|---|---:|
| pread | off | 424.3 |
| pread | on | 501.8 |
| preadv | off | 422.1 |
| preadv | on | 517.3 |

Overlap loses more without the GIL involved. Three mechanisms and two read paths now agree: concurrent SSD traffic and GPU work do not mix
on this machine.

### Token time allocation

From `cProfile` rather than derived buckets:

```text
posix.pread             168 ms/token   device and copy, in worker threads
_moe_call               152 ms/token   blocked at mx.eval on the router
to_mx                    31 ms/token   concatenate and convert
expert_cache.__call__    22 ms/token   Python bookkeeping
_rows_pread              25 ms/token
```

Roughly 168 ms of reads, 152 ms of GPU, and about 100 ms of single threaded Python, alternating 48 times. The CPU is busy, but it is busy
*waiting*: the same core blocks on a read, then blocks on a GPU drain, and neither wait can use the other nine cores. More cores cannot help
a dependency chain. This interpretation follows from 0.93 cores busy.

### Detected regression

The early submit experiment left three things running 48 times per token even with the feature off: `mx.eval(scores, inds)` materialising
`inds`, a `tolist()` host transfer, and a list comprehension over the routed set. All three are now gated on `_EARLY_SUBMIT or _WARM_ON`.
Correct either way, though not separately measurable against the noise floor.

### Phone comparison scope

BigMoeOnEdge reports 3.48 tok/s on a phone. Projecting this machine's measured constants onto a 2-bit checkpoint also gives 3.48 tok/s. The
checkpoint bit width explains the rate comparison: BigMoeOnEdge uses 2-bit weights, while this project uses 4-bit weights. Projections from
the measured constants, disk time scaling with bytes and GPU traffic with them:

| Checkpoint | bytes/weight | MB/token | disk | gpu | total | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| oQ4, today | 0.625 | 390 | 246 ms | 186 ms | 460 ms | 2.17 |
| oQ3 | 0.500 | 312 | 197 ms | 149 ms | 374 ms | 2.68 |
| oQ2 | 0.375 | 234 | 148 ms | 112 ms | 287 ms | 3.48 |

oQ3 reaches 2.91 tok/s at a 72% hit rate and 3.18 at 77%. Both results are plausible with the smaller working set and unchanged cache.
The uncertainty is whether GPU time falls with bit width. If it does not, oQ3 gives 2.43. The measured paths require a smaller
checkpoint to reach 3 tok/s on one stream.

## Drive rate correction, 2026-08-31

An earlier version claimed 1.6 GB/s against a drive capable of 2.99 GB/s. It described the 1.87x gap as an optimization target.
The result contained an arithmetic error. The 1.6 figure divided physical bytes by a window that also served 790 MB from cache. The
corrected result follows.

### Uncached drive rate on model shards

`F_NOCACHE`, real device reads, on `model-00012-of-00022.safetensors`:

| Pattern | GB/s |
|---|---:|
| sequential, 1 MB blocks | 1.51 |
| sequential, 4 MB blocks | 2.15 |
| scattered 340 KB, 4 workers | 2.82 |
| scattered 340 KB, 16 workers | 2.69 |
| scattered 800 KB, 16 workers | 3.03 |
| scattered 2 MB, 16 workers | 3.35 |

The **exact** expert pattern, three 800 KB weight rows and six 100 KB scale and bias rows per expert, 3456 reads, 16 workers:

| Pattern | Reads | GB/s | ms for 390 MB |
|---|---:|---:|---:|
| real: 3 weight + 6 scale/bias | 3456 | **2.99** | 130 |
| scales and biases merged | 1536 | 3.32 | 118 |
| one contiguous range per expert | 384 | 3.52 | 111 |

The store's read path reaches 17.3 GB/s with `pread` and 24.8 GB/s with `preadv` on cached rows.

### Model read rate of 1.6 GB/s

The model reads 390 MB in 246 ms. The identical isolated pattern reaches 2.99 GB/s, a 1.87x gap. A 130 ms read would put a token at 344 ms,
or 2.9 tok/s.

### Excluded causes of the gap

Each was tested and each reduced CPU without reducing wall time:

| Change | CPU | Wall |
|---|---|---|
| `preadv`, one copy instead of two | -13% | unchanged |
| shared read buffer, no concatenate | -17% | 3% worse |
| both together | -21% | 9% worse |
| chunk 8, nine futures per layer instead of 72 | -24% | 8% worse |
| early submit, with `preadv` so the GIL is free | - | 18% worse |

So it is not the double copy, not the concatenate, not thread-pool overhead, and not the GIL. Every one of those is real work removed, and
none of it was on the critical path.

### Untested primary explanation

The isolated benchmark keeps 16 workers reading continuously, so the NVMe queue stays full. The model reads in bursts: about 24 MB per
layer, then 3 ms of GPU work with the pool idle, 48 times per token. The queue drains between bursts and never reaches steady state. If that
is right, the fix must keep the queue full across the GPU window. That method requires overlap, which loses on this machine for a measured reason.
Those two facts together may be the real ceiling, but the burst hypothesis itself has not been tested. A
direct test would replay the model's exact read schedule, with and without the 3 ms gaps, and compare achieved bandwidth.

### Corrected phone comparison

BigMoeOnEdge's 3.48 tok/s uses `drop 100%`, which discards cold experts. Its lossless profile is about 2 tok/s, which is where this project
already is. The phone result does not use equal quality.

### Correction: there is no 1.87x gap

The 246 ms read window serves two different things:

```text
device reads 390 MB at 2.43 GB/s = 160 ms
cached reads 790 MB at 17.3 GB/s =  46 ms
                                    206 ms, against 246 measured
```

2.43 GB/s is exactly what the bursty pattern benchmarks at. The device is being driven at the rate this access pattern allows. Dividing 390
MB by the whole 246 ms window produced 1.6 GB/s. This is not a bandwidth measurement. Two thirds of those bytes never touched the device.

### Burst cost

Reading the model's exact schedule with `F_NOCACHE`, 16 workers:

| Shape | GB/s |
|---|---:|
| continuous, one long stream | 2.87 |
| 48 bursts of one layer, back to back | 2.86 |
| 48 bursts with a 1 ms gap | 2.71 |
| 48 bursts with a 3 ms gap | 2.43 |
| 48 bursts with a 5 ms gap | 2.23 |

Bursting itself adds no measured cost. Back-to-back bursts match a continuous stream. What costs is the gap: the GPU window drains the
queue, and a realistic 3 ms gap takes 2.87 down to 2.43, about 15%. Larger bursts help slightly: 8 experts 2.93, 16 experts 2.96, 32 experts
3.00, 64 experts 3.05 GB/s. A layer only routes 8, so this is not reachable without knowing the next layer's experts.

### Remaining options and estimates

| Change | Saves | Note |
|---|---:|---|
| contiguous range per expert | 50 ms | the blob layout, previously rejected |
| `preadv` on cached rows | 14 ms | measured neutral in the full model |
| removing the burst gaps | 25 ms | requires overlap, which loses |

Best case without overlap is **362 ms, or 2.76 tok/s**. The GPU's 152 ms and the bytes themselves are the rest. 3 tok/s on one stream still
needs a smaller checkpoint. oQ3 projects to 2.68 at today's hit rate and 2.91 to 3.18 as the smaller working set improves it.


## Mapping pinned rows instead of copying them, 2026-08-31

`pread` copies every row twice: the kernel fills a `bytes` object, then numpy copies it into the destination. A row the page cache already
holds does not need either copy. `mlock` guarantees residency, so the pinned rows are the one set that can take a map with no guessing.
Isolated, on real shards, 72 rows totalling 25.1 MB:

| rows | pread | mmap |
|---|---:|---:|
| cold | 13.62 ms | 21.89 ms |
| resident | 3.26 ms | **0.99 ms** |

Cold rows must use `pread`: mmap faults serialise and cost 1.6x. Resident rows favour the map by 3.3x. Predicted saving was 1.5 ms per
layer, 72 ms per token.

### Gate cost and accuracy requirements

An initial gate used previous-read status and ran 1.78 against 2.21 tok/s. Physical reads fell from 428.9 to 385.9 MB per token, so the
mechanism worked. The guess did not. The asymmetry is 3.6 to 1 against: a correct guess saves 2.27 ms, a wrong one costs 8.27 ms, so the
gate needs 78% accuracy to break even. On a 111 GB model with 16 GB of RAM, "read once" does not predict "still resident". `mincore` answers
exactly but costs 0.518 ms per layer against the 0.59 ms it saves. Only `mlock` gives residency for free, and `pin_rows` already holds it.

### End-to-end result

`FLASHNEXT_READ=resident`, `--exact --hot 32 --tokens 120`, ten arms across two runs with the order reversed between them:

| Mode | Runs | Mean | SD |
|---|---|---:|---:|
| pread | 2.42, 2.49, 2.63, 2.71, 2.71 | 2.592 | 0.132 |
| resident | 2.73, 2.47, 2.44, 2.62, 2.68 | 2.588 | 0.128 |

The measured difference is -0.2 percent. Output is byte-identical, expert counts match at 7.96, pinned bytes match at 4.66 GB. One initial
run indicated +12.8 percent and gave the wrong result. Both runs show the first arm slowest, 2.42 and 2.44, with free RAM at 4310 MB before
run 2 arm 1 and 800 to 1500 MB after. The page cache warms across arms and moves the rate more than the change does. **Alternate at least
three times per condition and reverse the order between runs. Two arms cannot resolve 5% on this machine.**

### Repeated absorption pattern

Early submit moved reads earlier and lost 8.2%. The shared read buffer removed 35 ms of copy and the GPU took back 37 ms. This removes a
kernel copy from the resident two thirds and returns nothing. Removing host work from the read path does not make this model faster,
whichever copy is removed. `FLASHNEXT_READ` keeps `pread` as its default. The `resident` mode stays as a measured option, correct if the
drive or the residency budget ever changes.


## Synthetic all-RAM and all-disk bounds, 2026-08-31

`bench_read_ceiling.py` holds the GPU work and the read count fixed and varies only where the bytes come from. `ram` pins one fixed expert
set and reuses it every token. `disk` picks a fresh random set out of 512 per layer per token. Production sits between them. The text is not
the model's real reply; this measures time.

| Arm | ms/token | Rate | io_wait |
|---|---:|---:|---:|
| disk, every read cold | 920.7 | 1.09 | 590.6 |
| fixed-prompt pinned tail, 32 requested, pread | 413 | 2.42 | 192.4 |
| ram, zero disk, pread | 235.2 | 4.25 | 69.7 |
| ram, zero disk, mmap | 187.7 | **5.33** | 36.7 |

The synthetic fixed-route arm reaches 5.33 tok/s with resident reads. Its GPU and Python work costs 188 ms per token. This proves that
compute can exceed 3 tok/s under this artificial route pattern. It does not generate a real model reply and does not establish a production
rate.

### Map-path result without drive contention

With zero disk, `resident` takes io_wait from 69.7 to 36.7 ms and the token from 235.2 to 187.7 ms, **+25%**. `score_sync` does not absorb
it: 91.0 to 83.0 ms. In production the same change gives nothing. io_wait falls 9.7 ms and score_sync rises 17.3 ms. The map copies at 25
GB/s against pread's 7.7. Under drive pressure, it takes a larger share of the memory controller. The GPU cost exceeds the read-path gain.
Absorption is a symptom of contention, not a property of the change.

### Invalidated assumptions

"Pinning beyond 32 experts does nothing" was measured with `pread`, where a pinned row and a page-cached row cost exactly the same, so
pinning could only prevent a fault it was already avoiding. With the read path gated on `_pinned_rows`, pinning changes a row's cost by
3.3x. Pinning depth and read mode were each tested alone and never together. That pair is now open again.

### Pinning depth reopened, and closed again

The coverage curve predicted 77.8% at 48 experts, a 276 MB cold budget and 3.09 tok/s. That prediction was written down before the run and
it failed. The harness flaw was real. `bench_resident_tail` and `RoutingProfile` both stop observing a layer after `warmup` rows, 8 by
default. A layer then holds fewer distinct candidates than the pin depth asks for:

| warmup | candidates/layer | pinned/layer of 32 | pinned | rate |
|---:|---|---:|---:|---:|
| 8 | min 23, mean 32.3, max 43 | 30.2 | 4.66 GB | 2.42 |
| 40 | min 31, mean 81.4, max 145 | 32.0 | 4.94 GB | 2.68 |

At `warmup=8` the model pins 30.2 experts when asked for 32, and could never have pinned 48. Every earlier "pin more" test was measuring a
pool that did not exist. `pin_budget_gb` is 6.0 and was never reached. Fixing the pool does not fix the rate. With a 81.4-candidate pool,
`hot=40` pins 39.6 experts and 6.12 GB, 24% more bytes than `hot=32`. It returns the same 2.68 tok/s with io_wait 171.2 against 174.0.
Ranks 33 to 40 are experts the page cache already held. Pinning relabels them as unevictable; it does not make them resident, because they
were. The cold 386 MB per token comes from the long tail of about 154 distinct experts per layer. That tail is 21.7 GB. No pin budget on a
16 GB machine reaches it. **Coverage of accesses is not the same as page-cache residency. The curve describes routing, not I/O, and must not
be used to predict throughput.**

### Longer warmup shows no measured benefit

A/B/A/B, `hot=32`, 120 tokens, `warmup=40` first to counter the order effect:

| warmup | Runs | Mean | pinned/layer |
|---:|---|---:|---:|
| 8 | 2.88, 2.78 | **2.830** | 30.2 |
| 40 | 2.71, 2.74 | 2.725 | 32.0 |

The warmup-40 pair measured 3.7% lower. The sample has only two arms per condition, so it cannot resolve a change this small under the rule
below. Keep `warmup=8` because warmup 40 adds startup work with no measured tail gain. The order effect appeared in every multi-arm run that
day. The first arm was always slowest. A two-arm A/B cannot resolve the change on this machine.

### Standing summary

| Lever | Result |
|---|---|
| map resident rows instead of copying | +25% with no disk, nothing in production |
| pin deeper, 32 to 40, with a real pool | nothing |
| longer warmup, 8 to 40 | no measured gain; two-arm means differ by -3.7% |
| synthetic fixed-route all-resident reads | **5.33 tok/s ceiling** |

The tested byte-layout changes did not move the pinned tail. A smaller checkpoint remains an open direction. Do not use 2.83 tok/s as its
baseline. That number is the mean of two warmup-eight arms. A complete-chat projection must also include warmup and pinning time.


## Residency tracking, measured 2026-08-31

The `resident` read mode maps rows it believes the page cache holds. Only mlocked rows qualified, about 32 experts of 512, so most cached
rows still took the slower copy. `FLASHNEXT_TRACK_RESIDENT` adds a bounded LRU of rows the process has read, which reaches far more of them.

### Gate accuracy

`bench_residency.py` compares each decision against `mincore`, over 238,812 gated rows:

| | rows |
|---|---:|
| claimed resident | 150,026 (62.8% of reads) |
| correct | 146,427 |
| wrong, confirmed cold | 3,599 |
| cached but not claimed | 14,093 |

Precision is 97.6 percent against a 78.5 percent break-even. Gate accuracy exceeds the requirement.

### End-to-end regression

`bench_production.py --compare track-resident`, eight arms per condition, interleaved:

| Condition | gen median | SD | MB/token | drift |
|---|---:|---:|---:|---:|
| pinned-only | 2.611 | 0.046 | 432.4 | +0.61 |
| tracked | **2.458** | 0.119 | **418.7** | **-0.85** |

The model ran 5.9 percent slower while reading 3.2 percent fewer bytes.

### Regression by mapped share

Arm rates in the order they ran:

```text
pinned-only   2.60 2.80 2.55 2.59 2.63 2.64 2.57 2.67
tracked       2.61 2.45 2.50 2.53 2.48 2.42 2.43 2.20
```

The conditions are interleaved, so both cover the same wall clock. Heat would reach both. Only the tracked arm slides, and it slides
monotonically as its LRU fills and more rows become eligible to map. The three scales give these results:

| Mapped share of reads | Result |
|---|---|
| mlocked rows only, ~32 experts | neutral |
| 62.8% of all reads | -5.9% |
| rising within one run | worsens monotonically |

Mapping a resident row is faster in isolation, at 0.99 ms against 3.26. In the model it copies at 25 GB/s where pread copies at 7.7, takes a
larger share of the memory controller, and the GPU pays more than the read path saves. The harm is proportional to the mapped fraction,
which is why a better gate made it worse rather than better. Do not retry any variant of mapping resident expert rows. The mechanism is
understood, it is a hardware property, and a more accurate gate makes it worse. `FLASHNEXT_TRACK_RESIDENT` stays off.


## Contiguous expert layout, retried and withdrawn 2026-08-31

`bench_layout.py` measured an isolated reader A/B on this drive. It used 72 reads per layer against 8. It reported 2.73 GB/s against 3.22.
It was written up as the day's one gain and put into the handoff as next work. The conclusion was incorrect. The same experiment appears
above under "Exact expert-major layout experiment". "Repacking the complete checkpoint" was already in the handoff's do-not-retry list. The
earlier work went far further than this retry:

| | 2026-08-30 | this retry |
|---|---|---|
| record per expert | 3,072,000 bytes | 3,072,000 bytes |
| isolated reader gain | 1.97 to 2.84 GB/s, +43.8% | 2.73 to 3.22 GB/s, +18% |
| byte-exact across 48 layers | SHA-256 verified | never run |
| conversion without a second copy | restart-safe APFS | never run |
| integrated, 8 layers | tied, 1.413 against 1.404 | never run |
| full model | 1.124 to 1.243 tok/s, every variant rejected | never run |

The earlier isolated gain was more than twice this one and still did not survive integration. The archived brief names this benchmark among
the tests that "inflated false gains (repack, sequential reads, two layout benchmarks)". An isolated reader A/B cannot support a layout
claim. The reader is not what the model waits on once the page cache, the n-gram stream and the compute share one process.
`bench_layout.py`, `repack.py` and the blob read path were removed rather than left to be found and rebuilt a third time. Check the
do-not-retry list before building. This retry cost a day's credibility on the one result that looked like a win.


## Sweep of 2026-08-31: four changes, none resolved

Every comparison prints the band it can resolve, two standard errors of the difference between the medians. A reading inside that band is
unresolved, not absent.

| Change | gen median | Band | MB/token |
|---|---:|---:|---|
| sorted reads | +1.1% | 5.9% | 389 to 395 |
| pin scales and biases only | +3.5% | 5.6% | 417 to 404 |
| prewarm last session's experts | +2.7% | 7.1% | 404 to 407 |

None stands alone. Two of them are cheaper to settle together than to re-run separately: if `pin-parts` and `prewarm` were each real, the
pair would be near 6% and clear of the band.

| Stacked | gen median | Band | MB/token |
|---|---:|---:|---|
| whole experts, no prewarm | 2.57 | | 434.6 |
| scales only, prewarm on | 2.53 | 3.5% | 470.0 |

The pair lost 1.4 percent and read 8 percent more physical bytes. The pair does not add, and the bytes move the wrong way, so the individual
readings were noise. Stacking settled two weak results for the cost of one comparison. `sorted reads` moved no bytes in either direction, so
it had no mechanism behind its number either.


## Prefill amortisation unavailable to decode, measured 2026-08-31

Prefill reaches 40 to 60 tok/s where decode reaches 2.7, on the same weights and the same drive. `bench_prefill_scaling.py` measures where
that comes from.

| Tokens | tok/s | Read GB | MB/token | Drive | Experts/layer |
|---:|---:|---:|---:|---:|---:|
| 128 | 8.72 | 20.56 | 160.6 | 1.40 GB/s | 135 |
| 512 | 20.66 | 30.08 | 58.8 | 1.21 GB/s | 197 |
| 1024 | 30.75 | 34.93 | 34.1 | 1.05 GB/s | 227 |
| 2048 | 41.97 | 39.95 | 19.5 | 0.82 GB/s | 255 |

The drive rate falls as prefill speeds up. Decode sustains 1.06 GB/s, more than a 2048-token prefill. Prefill does not read faster, use less
memory more cleverly, or schedule I/O better. Sixteen times the tokens cost 1.94 times the bytes, because distinct experts per layer
saturates toward 512 while the token count keeps rising. It is amortisation and nothing else, so there is no mechanism in that path for
decode to copy.

### Amortisation curve and speculation result

Decode routes about 8 experts per layer for one token, and a batch shares them. Measured round-robin over four rounds, the first discarded,
so no length warms another:

| Batch | Experts/layer | Per token | tok/s | Physical MB/token |
|---:|---:|---:|---:|---:|
| 2 | 13 | 6.5 | 1.38 | 808 |
| 4 | 22 | 5.5 | 2.68 | 340 |
| 8 | 37 | 4.6 | 2.55 | 515 |
| 16 | 58 | 3.6 | 2.65 | 557 |
| 32 | 79 | 2.5 | 3.84 | 375 |
| 64 | 104 | 1.6 | 5.84 | 247 |

Decode is 2.71 tok/s at 390 MB/token. **No batch below 32 tokens beats it.** The expert counts promise savings that the drive does not
deliver, and the gap is the page cache:

| | Experts/token | Gathered | Physical | Hit rate |
|---|---:|---:|---:|---:|
| decode | 8 | 1152 MB | 390 MB | 66% |
| batch of 8 | 4.6 | 666 MB | 515 MB | 23% |

A batch of 8 gathers 42% fewer bytes and reads 32% more from the drive. Decode's eight experts are mostly resident already, because
consecutive tokens reuse 42.5% of them. A batch of 8 touches 37 distinct experts per layer, most of them first-touch, so it pays close to
full price. Batching widens the distinct working set and reduces page-cache effectiveness. This behavior explains the failed speculation
results more than draft cost. Self-draft accepted 92% at depth 1 and still lost, 1.33 against 1.09 tok/s; depth 2 accepted 94% and lost,
1.98 against 1.60. A verify pass over a few tokens reads more physical bytes than the same tokens decoded one at a time, whatever the draft
costs and however often it is right. Any future scheme that verifies several tokens per pass has to beat 390 MB per token on physical reads,
not on gathered bytes. Below 32 tokens nothing here does.

### Measurement note

An earlier version of the small-batch table was produced in ascending order, and the runs warmed one another: 4 tokens read less than 2. It
showed a batch of 8 at 3.85 tok/s and 295 MB/token, beating decode, and a crossover at 8 tokens. Round-robin measurement put that batch at
2.55 tok/s and 515 MB/token, below decode, and moved the crossover to about 32. The expert counts were unaffected, because they do not
depend on the cache. Interleave the lengths. A sweep that walks them in order measures its own warm-up.


## Cache-aware routing, opportunity measured 2026-08-31

Adaptive top-k scores ten experts per layer and keeps about eight, so two are already scored and discarded. When a kept expert has to come
off the drive. A discarded expert can already be resident with an almost equal score. Taking it changes a cold read into a cached read.
`bench_route_swap.py` counted how often that situation arises. It observed and swapped nothing, so the tokens were the model's own.

| | Value |
|---|---:|
| routing decisions | 3,984 |
| experts kept | 31,462 |
| of those, cold | 14,436 (45.9%) |

| Epsilon | Swaps | Of cold reads | Bytes saved | Mass given up |
|---:|---:|---:|---:|---:|
| 0.005 | 1,715 | 11.9% | 46 MB/token | 0.0014 per swap |
| 0.010 | 1,919 | 13.3% | 52 MB/token | 0.0020 |
| 0.020 | 2,007 | 13.9% | 54 MB/token | 0.0025 |
| 0.050 | 2,028 | 14.0% | 55 MB/token | 0.0027 |

The opportunity saturates by epsilon 0.02, and a tight epsilon of 0.005 already captures most of it. At epsilon 0.02 this is one swap every
two decisions, 54 MB off a 390 MB token, about 19 ms, projecting 2.86 tok/s against 2.713. The cost is 0.15 percent of the kept routed mass.
Adaptive top-k already discards about 15 percent of that mass by design, so this adds a hundredth of what the router already gives up.

### Limits of this result

The projection assumes the saving reaches the clock. Nothing else measured today did: every host-side reduction returned to the GPU wait.
This one differs in kind, because it removes physical reads rather than moving them, and fewer cold reads is the only quantity that has
never been tried and failed here. `believed_resident` is 97.6 percent precise, not exact, so some spares counted as resident are cold and
the realised saving is lower than the count. The projected gain is about 5 percent, which sits at the resolution floor of the standard
harness. It needs the arm count raised to be told from noise. No earlier change in this set alters the model's output. Token identity stops
being the check, and the retained reasoning gate becomes mandatory rather than optional. A second-order effect is unmeasured in both
directions. Preferring resident experts may shrink the working set and compound the gain, or it may settle routing onto a subset of experts
and degrade long generations.


## Cache-aware routing, built and measured 2026-08-31

Adaptive top-k scores ten experts per layer and keeps about eight, so two are already scored and discarded. When a kept expert has to come
off the drive and a discarded one is already in memory within `epsilon` of its score, the resident one is taken instead.
`FLASHNEXT_SWAP_RESIDENT=1` enables it, `FLASHNEXT_SWAP_EPSILON` sets the tolerance, default 0.02. This initial benchmark switch was off by
default. The environment flags were the initial benchmark interface. The runtime now exposes the mechanism as the `cache-aware` routing
profile. `/config model routing cache-aware` enables it live. `/config model swap-epsilon VALUE` controls the tolerance. `exact-quality` remains the
default.

### Quality

Cache-aware routing is the first accepted change here that alters model computation, so token identity no longer checks it.
`bench_swap_quality.py` runs the same prompts with the swap off and on and compares. Exact routing is the reference.

| Checkable prompt | Exact | Cache-aware |
|---|---|---|
| 17 x 23 | correct | correct |
| probability, 3/10 | correct | correct |
| days in three years, 1095 | correct | correct |
| coin, 3/8 | truncated | truncated the same way |

No answer that exact routing reached was lost. Three of seven replies were byte-identical and four differed in wording while saying the same
thing. The initial gate was invalid. Two checkable prompts ran past the token budget before reaching their answer, so both conditions failed
for the same irrelevant reason. A checkable prompt has to reach its answer inside the budget, so each one now demands the answer first.

### Rate

`--compare swap-resident --arms 10`:

| | gen median | tail | MB/token |
|---|---:|---:|---:|
| exact | 2.54 | 2.50 | 417.8 |
| cache-aware | 2.79 | 2.77 | 347.6 |

Physical reads fell 16.8 percent, above the 13.9 percent the opportunity analysis predicted. The machine was hot and both conditions
drifted, so the medians are depressed. Arms alternate, so pairing reduces the effect of machine-state drift:

| Exact | Cache-aware | Difference |
|---:|---:|---:|
| 2.28 | 2.50 | +9.8% |
| 2.32 | 2.68 | +15.5% |
| 2.78 | 2.96 | +6.3% |
| 2.82 | 3.01 | +6.6% |
| 2.58 | 2.82 | +9.7% |
| 2.42 | 2.58 | +6.5% |
| 2.41 | 2.76 | +14.5% |
| 2.50 | 2.45 | -2.2% |

**The mean gain is 8.3 percent. Seven of eight pairs lead, with p = 0.035. All pairs read fewer bytes.** Exact takes the earlier and
cooler slot of every pair, so heat works against cache-aware and the figure is conservative. Applied to the 2.713 baseline this is about
2.94 tok/s, which is unconfirmed: the run that produced 8.3 percent never saw a cool machine.

### Cause of the measured gain

Every other change measured today moved bytes around: mapping instead of copying, reordering reads, pinning differently, warming ahead. Each
returned its saving to the GPU wait. This one removes reads. Physical bytes per token fell in every arm, which no other change achieved.

### Still open

- Repeat on a cool machine. The medians here are depressed and only the paired
comparison controls machine-state drift.
- The quality gate is four checkable prompts. Widen it before this becomes a
default.
- The second-order effect is unmeasured. Preferring resident experts may
shrink the working set and compound the gain, or settle routing onto a subset and degrade long generations.


## Cache-aware routing, product option and long-context check 2026-08-31

The measured mechanism is now a normal routing profile. It no longer needs environment flags or a separate implementation path.

```text
/config model routing cache-aware
/config model swap-epsilon 0.02
/config model routing exact-quality
```

The CLI supports `--cache-aware` and `--swap-epsilon`. Selecting the profile enables bounded residency
tracking in the open tensor store. This also works after a live `/config model` change. The initial four-prompt factual gate did not lose a
correct exact answer. It was too small to decide product quality. A later test compared Portuguese dream analysis at about 4,600 to 5,700
context tokens. Both profiles stayed coherent. The exact-quality answer was better because it had better structure, continuity, and detail
selection. The long-context rates were directional only. The prompts had different context lengths and the machine state was uncontrolled.
Across five aligned segments, cache-aware had a 2.2 tok/s median `gen` rate against 2.1 for exact. Its median `tail` rate was 2.4 against
2.2. Do not publish these values as a controlled gain. Decision:

- Keep `exact-quality` as the default.
- Offer `cache-aware` as an explicit speed option.
- Show a quality warning in `/config model`.
- Keep the retained speed claim at 2.79 versus 2.54 tok/s in the hot
interleaved benchmark, with a paired mean gain of 8.3 percent.
- Require a wider quality gate before any default change.

## oQ3-MTP failed the SketchUp Ruby test on 2026-09-01

The gate requested a SketchUp extension that extrudes selected faces to a user-specified height. The reply had to be a complete `.rb` file.
Each file was loaded in SketchUp.

```text
checkpoint          effort    result
oQ4                 xhigh     worked
oQ3-MTP             low       broken
oQ3-MTP             xhigh     broken
```

### What each one wrote

oQ4 called `face.pushpull(distance)`, which is the real method. oQ3-MTP at `xhigh` called `face.extrude(vector, true)`. There is no
`Sketchup::Face#extrude`. `Face` has `pushpull` and `followme`. The script raises `NoMethodError` on the first face. oQ3-MTP at `low` called
`face.pushpull(height, true)`. The method is right and the second argument is wrong: it's `copy`, not a direction flag, so this makes a copy
and leaves the original face in place. Its own comment says `# true = flip if needed`. The same file calls `JSON.parse` and
`JSON.pretty_generate` without `require 'json'`, and writes a settings file into the Plugins folder, which is often read-only.

### More thinking made oQ3 worse

Method-name counts across each run's reasoning:

```text
run                 face.extrude(   .pushpull(   reasoning chars
oQ4                       0              8            29,876
oQ3-MTP low               0              2             7,713
oQ3-MTP xhigh            13              0            34,203
```

oQ3-MTP at `xhigh` states the wrong method in its first reasoning sentence:

> em SketchUp Ruby API, para extrudar faces, podemos usar
> face.extrude(vector, [allow_collapse])

It never mentions `pushpull` again across 34,203 characters, and builds everything on that first claim. oQ4 doubts the same method about
twenty times. It writes "I'm not sure" and "No, that's not right" while trying several signatures, then settles it:

> Let me think about what I know for sure: `Face#pushpull(distance)` works.

oQ4 gets there because it treats the signature as uncertain. oQ3-MTP at `xhigh` treats it as settled and never checks. oQ3-MTP at `low` does
get the method name right, so more effort moved it away from the answer rather than toward it.

### Reading

Extra reasoning can't rebuild a fact the weights no longer hold, and an API method name is that kind of fact. The 3-bit layout damages that
recall while leaving prose alone, which is why dream interpretation hides it and code that has to name a real API doesn't.

### Decision

- oQ4 is the quality baseline. oQ3-MTP doesn't replace it.
- Don't judge a checkpoint on prose alone. Keep a code task that names a real
external API in every checkpoint gate.
- The generation cap didn't cause any of the three failures. All three closed
their code blocks.

## Cache-aware prefill regression, fixed 2026-09-01

One cache-aware chat turn prefilled 4,406 tokens at 35.1 tok/s. An exact turn of 4,439 tokens prefilled at 45.0. That's 22% slower. The swap
runs once per row of the batch. Decode gives it one row; prefill gives it one row per prompt token. So a 4,400-token prompt across 48 layers
means about 211,000 `swap_row` calls, two host lists of shape (4400, 10) built per layer, and two MLX arrays rebuilt per layer from Python
lists. None of that work buys anything during prefill, because a prefill batch already shares one expert read across all its tokens. The
section on prefill amortisation measures it: sixteen times the tokens cost 1.94 times the bytes. The swap pays the full per-token cost to
remove a read that's already shared. The swap now stands down above `FLASHNEXT_SWAP_MAX_ROWS` rows in one call, default 4, which covers
decode and a short speculative batch. That also drops the forced `inds` host sync during prefill. Decode is unchanged. The prefill recovery
remains unmeasured.

## The route observer was getting the whole prefill batch, fixed 2026-09-01

`RoutingProfile._observe` stops after 8 rows per layer. It was being handed one row per prompt token, 48 times, and building two Python
lists of them each time. A 4,400-token prefill created about 4.2 million Python floats and threw nearly all of them away.
`set_route_observer` now takes a row cap and the quality profile passes its warmup value. The observations are identical, because the
observer only ever read the first rows. Benchmarks that need every token pass no cap. This one affects `exact-quality`, which is the default
profile.

## Re-reading the dream analyses, 2026-09-01

The three long-context dream replies use the same dream text. An earlier section marks the exact-quality answer as better. Reading the
transcripts in full doesn't support that for the Lacanian section. The cache-aware reply tracks the object across both dreams, moving from
the flight in the first to the kitten and then to the designer clothes. The exact reply makes one claim and stops. The oQ3-MTP reply is
still listing concepts where the transcript ends. Two things limit the comparison. The exact conversation had an earlier Jung frame, so its
Lacan turn had to pivot. All three runs hit the 120-token cap and were continued by hand, so coverage depends on how often the user pressed
continue. None of the three showed incoherence, factual drift, a repetition loop, or language slippage. The cache-aware run repeated one
heading after a continue, which is stitching. Missing spaces show up in both oQ4 profiles at what look like chunk joins: `o focodeixa` in
exact, `resgatarou` in cache-aware. It appears under both profiles, so routing isn't the cause. The word animator probably is. Still open.

## oQ3-MTP rates, chat turns only, 2026-09-01

No oQ3-MTP run went through `bench_production`. Everything below is a single chat turn read off the terminal stat line, under conditions
that differ between runs. Treat it as directional. The three runs analyse the same Portuguese dream text at a context near 5,000 tokens.
Each turn generates up to 120 tokens and the user continues by hand.

```text
oQ3-MTP, exact-quality
  4407 new tok @ 45.2 t/s | gen 2.2 | tail 2.7 | ctx 4661 | 153.0s
    15 new tok @  3.9     | gen 2.6 | tail 2.6 | ctx 4796 |  50.3s
    15 new tok @  3.9     | gen 2.5 | tail 2.5 | ctx 4931 |  52.0s
    15 new tok @  3.7     | gen 2.5 | tail 2.6 | ctx 5066 |  51.6s
    24 new tok @  3.9     | gen 2.6 | tail 2.7 | ctx 5210 |  52.2s
    15 new tok @  3.7     | gen 2.7 | tail 2.9 | ctx 5345 |  48.0s

oQ4, exact-quality
  4439 new tok @ 45.0 t/s | gen 2.0 | tail 2.3 | ctx 4559 | 157.4s
    15 new tok @  3.8     | gen 2.2 | tail 2.2 | ctx 4694 |  59.3s
    15 new tok @  3.7     | gen 2.1 | tail 2.1 | ctx 4829 |  61.4s
    23 new tok @  3.6     | gen 2.4 | tail 2.5 | ctx 4972 |  56.1s
    15 new tok @  3.5     | gen 1.9 | tail 1.9 | ctx 5107 |  68.0s

oQ4, cache-aware
  4406 new tok @ 35.1 t/s | gen 2.0 | tail 2.5 | ctx 4863 | 184.5s
    15 new tok @  3.8     | gen 2.2 | tail 2.3 | ctx 4998 |  59.4s
    15 new tok @  3.9     | gen 2.3 | tail 2.4 | ctx 5133 |  55.5s
    15 new tok @  3.3     | gen 2.2 | tail 2.4 | ctx 5241 |  46.1s
    23 new tok @  3.4     | gen 2.4 | tail 2.5 | ctx 5384 |  57.3s
    15 new tok @  3.7     | gen 2.1 | tail 2.2 | ctx 5519 |  60.7s
    15 new tok @  3.6     | gen 2.2 | tail 2.2 | ctx 5654 |  59.6s
```

Medians over the turns of each run:

```text
run                       gen    tail   long prefill
oQ3-MTP exact-quality     2.55   2.65      45.2
oQ4 cache-aware           2.2    2.4       35.1
oQ4 exact-quality         2.1    2.2       45.0
```

oQ3-MTP leads oQ4 exact by 21% on `gen` and 20% on `tail`. Its weights are 23% smaller, so the direction and rough size both fit a
read-bound runtime. Don't quote this as a measured gain. The runs are unpaired, they're different conversations, the context lengths differ,
and nothing controlled the machine state. The rules in `handoff.md` want three arms per condition and a printed resolution band, and none of
that applies here. Two other things fell out of it:

- Short follow-up prefills of 15 to 24 tokens run at 3.3 to 3.9 tok/s on every
checkpoint and profile, while the long prompt reaches 45. That's the amortisation result again, from a different direction.
- The cache-aware long prefill at 35.1 against 45.0 is the deficit that
`FLASHNEXT_SWAP_MAX_ROWS` addresses. The short prefills show no such gap, which fits a cost that scales with batch rows. Published
Flash-Next rates stay on oQ4, which has a harness baseline and passes the checkpoint gate. Quoting an oQ3-MTP rate needs a
`bench_production` run first.

## Compressing the checkpoint doesn't work, measured 2026-09-01

The question was whether oQ4 can be made smaller on disk. It can't, by any route that keeps the runtime working. The exact-path tests
already recorded this: "Lossless Zstandard compression reduces packed expert weights by 4%. Scales and biases compress better, but the
combined row shrinks only 10%." The 2026-09-01 measurement below agrees with it and was run without finding it first. Search this file
before measuring. Generic compression on 4 MB of real oQ4 expert data, sampled from the middle of `model-00010-of-00022.safetensors`:

```text
method     saves     throughput
zlib -6     3.59%      43 MB/s
zlib -9     3.59%      55 MB/s
lzma        3.02%      10 MB/s
bz2         0.80%      18 MB/s
```

Best case removes 4.0 GB from 111.7 GB. The weights are already 4-bit quantized with scales and biases, so there's little redundancy left.
Throughput rules it out on its own. Decode reads about 390 MB per token and the drive sustains 1.06 GB/s; decompression runs at a third of
that. The reader also uses positioned `pread` at exact byte offsets, and a compressed stream has no such offsets, so serving one
3,072,000-byte expert row would mean decompressing a whole block. That's the standing result about host work on the read path. APFS
transparent compression gives the same ratio and the same broken random reads. oQ4 does carry dead weight. Its index holds 324
`vision_tower` tensors the text runtime never loads. The shard 1 header puts them at 0.90 GB, 0.8% of the checkpoint.
The tensors span a 2.05 GB region with 93 language tensors interleaved. Removing them requires rewriting shard 1 and every index offset.
Compression and vision removal together save under 5 GB. They require a shard rewrite and break the read path. The machine cleanup list in
`handoff.md` is worth 13 GB at no risk. The only remaining way to gain space is external storage, which trades against the 1.06 GB/s read
rate the design depends on.

## What an off-process draft costs the target, measured 2026-09-01

Four speculative decoding tests reached the same result: the draft and verifier run sequentially, so together they cost more than one exact
decode. The anchored depth-2 external draft hit 92% acceptance and tied its control at 1.747 against 1.749 tok/s. It paid for exactly what
it saved. That's a scheduling result. Two independent processes reach 1.52x aggregate on this machine, at roughly 31% of the GPU for 52%
more work. A draft running beside the target is independent work by that definition, so its cost moves into the window that exchange rate
prices. `bench_draft_contention.py` measures how much of its solo rate the target keeps while a draft process runs. It doesn't draft or
verify anything; it measures contention.

### Result

Three arms per condition, alternating so drift reaches both equally, 48 tokens each, `Qwen3.5-0.8B-MLX-4bit` as the load:

```text
duty   solo median   beside median   retention   paired   band
100%      1.857         1.708          92.0%     -9.5%    4.5%
 10%      1.862         1.838          98.7%     -1.5%    1.4%
```

Both losses clear their band and both went the same way in all three pairs. With three pairs a sign test can't report below p = 0.125, so 3
of 3 is as strong as the arm count allows. The target's token IDs were identical across all twelve arms of both runs. A real drafter emits
two tokens per target block. At the draft's measured 70.5 tok/s solo that's 28 ms against a target block near 370 ms, so it's busy about
7.6% of the time. The 100% row is a 13x worst case and the 10% row is close to real use. The draft never starves the target. At 10% duty it
produced 4.1 tok/s while a depth-2 block at the production rate needs about 2.8.

### Where the loss comes from

The beside arms read about 3% more per token at both duty cycles. They read 1038 against 1005 MB, and 1026 against 996 MB. The draft's
627 MB evicts page cache whether it computes or idles, so that part doesn't scale with duty. At a 60% read share it accounts for roughly 2
points. The rest is contention, and that's what duty cycle removes.

### Machine state

Both runs sat near 1013 MB/token against a 390 MB production baseline. The recent 111.7 GB oQ4 download had evicted the page cache.
The benchmark's premise gate refused to print an absolute projection for that reason. Retention is a ratio between alternating arms so drift
cancels out of it, but confirm it warm before quoting it. Two effects are open and unresolved. A more drive-bound target waits on the SSD
longer, so a compute-only draft steals less, which would make 98.7% optimistic. Scarcer page cache makes the draft's eviction cost more,
which would make it pessimistic.

### This doesn't rescue speculation

The projection is `committed / (block_cost / retention)`. Retention is measured here and acceptance of 92% was measured earlier. The block
cost was measured on 2026-08-31 in the amortisation curve above, and it kills the idea. A batch of two runs at 1.38 tok/s and 808 MB/token,
against decode at 2.71 and
390. Two tokens take 1.45 seconds where one decode takes 0.37, so the block
costs 3.9 times one decode, not the 1.15 this benchmark assumed when it was written. That gives 0.48 times solo, which is half the speed of
decoding one token at a time. Longer blocks don't help. At 92% per-token acceptance the expected accepted prefix is about 11.5 tokens, and a
32-token verify pass takes 8.3 seconds, which is 1.38 tok/s. Every block length in the curve loses. Draft cost and scheduling were never the
binding constraint. Batching widens the distinct working set and defeats the page cache. Decode routes 8 experts per layer at a 66% hit rate
because consecutive tokens reuse 42.5% of them; a batch of two touches 13 distinct experts per layer, mostly first touch. This section was
written with the block cost recorded as unmeasured. It was measured 16 lines below where the file had been read. Read the amortisation curve
before proposing any multi-token scheme.

### What the contention measurement is still worth

An off-process draft at realistic duty costs the target 1.5%. That was unknown and now isn't, and it removes the scheduling objection from
any future scheme that wants a second process doing independent work. It isn't the objection that killed speculation.
`bench_draft_contention.py` and `draft_worker.py` stay as the harness for measuring what a second process costs this one.

## Cache-aware routing on the production harness, 2026-09-01

Every cache-aware number until now came from one hot interleaved comparison. The harness had never run it. This is that run.
`bench_production.py --compare swap-resident --arms 8`, conditions alternating, 60 tokens per arm, first two arms of each condition dropped
as cold:

```text
condition      gen median   tail   MB/token   sd      n
exact             2.73      2.70     430.0   0.012    4
cache-aware       2.91      2.92     360.4   0.013    4
+6.5 percent gen median, band 0.6 percent, so this one stands
paired over 6 arms: mean +10.0 percent, median +7.0
cache-aware ahead in 6 of 6 pairs, sign test p = 0.016
fewer bytes in 6 of 6 pairs
```

The exact arm measured 2.73 against the recorded 2.713 baseline, at 430 MB/token against 390. The machine matched normal production warmth.
The byte reduction transferred: 430.0 to 360.4 is 16.2%, against the 16.8% from the earlier
hot comparison. Whether the opportunity would survive a different residency state was open until now. Standard deviations of 0.012 and 0.013
are the tightest recorded arms. They give a 0.6% band against a 6.5% effect.

### The drift flag isn't a result

The harness reported cache-aware falling with elapsed time at r = -0.71 while exact didn't. The arms in order:

```text
cache-aware  2.905  2.883  2.911  2.919  2.913  2.890
exact        2.290  2.683  2.722  2.729  2.748  2.739
```

Cache-aware is flat across a 1.2% range and isn't monotonic. The correlation runs over the four kept points inside that range. Exact rises
because it's still warming: its reads fall from 437 to 428 MB/token across the run while cache-aware starts at its floor near 360. That
doesn't settle the second-order question. Preferring resident experts might shrink the working set over a long generation, or settle routing
onto a subset. Sixty-token arms can't see either. Measure it on a long generation.

### Position

Cache-aware measures 2.91 gen and 2.92 tail against a 3.0 target. The gap is 3.0% on gen and 2.7% on tail. Closing it through bytes needs
about 5% fewer, from 360 to near 341. All these rates come from greedy decoding, which the benchmarks use so token IDs stay comparable.
Quality is what gates a default, not speed. Cache-aware changes expert choices and its factual gate is four prompts wide.

## Swap epsilon is done at 0.02, measured 2026-09-01

Cache-aware measured 2.91 gen on the harness at epsilon 0.02, so 3.0 needed about 5% fewer bytes. Epsilon was the one lever with a mechanism
and no measurement against it. Two comparisons close it.

```text
comparison            gen median   tail    MB/token   band     pairs
e=0.02 vs e=0.05         2.79        2.83     376.6    1.4%    4 of 6
                         2.81        2.82     371.2            p=0.344
e=0.02 vs e=0.10         2.82        2.84     334.3    6.6%    5 of 7
                         2.94        2.92     329.5            p=0.227
```

Neither resolves on rate. The byte column is the result. Going from 0.02 to 0.05 removed 1.4% of physical bytes. Going from 0.02 to 0.10,
five times the reach, removed the same 1.4%. The first 0.02 captures the whole opportunity, which is the 16.2% measured against exact
routing that morning. Expert score gaps look bimodal. A discarded expert is either within 0.02 of a kept one or well past 0.10, with nothing
in between to harvest. That also fits the opportunity study, which measured 11.9% of cold reads replaced at 0.005 and 13.9% at 0.02 before
flattening. The quality cost isn't unresolved. Every arm at 0.10 produced different tokens, 7 of 7, and 5 of 6 did at 0.05. A certain change
in output for a gain that clears neither band settles it. Keep `swap-epsilon` at 0.02.

### Run conditions

The wide run was much noisier than the earlier swap comparison. Standard deviations were 0.159 and 0.132, against 0.010 and 0.013.
The resolution band was 6.6%, against
0.6. Machine warmth moved between runs too, with e=0.02 reading 334.3 MB/token
here, 376.6 in the 0.05 comparison, and 360.4 that morning. Compare arms within a run, never rates across runs. Individual arms crossed the
target: e=0.02 reached 3.06 gen and e=0.10 reached 3.05. The median doesn't hold there, but 3.0 is inside the machine's range on a warm
cache at the shipped epsilon.

## Cache-aware and the trajectory gate, 2026-09-01

The gate exists because oQ3-MTP was adopted on speed and failed a code task. Cache-aware was in the same position: a rate and a byte count,
no code test. This is that test. The prompt, verbatim, is the one oQ4 exact-quality passed earlier:

```text
crie uma extensao para sketchup que extrude varias faces ao mesmo tempo ate
uma altura definida pelo usuario. produza o codigo para eu salvar em um
arquivo .rb
```

Both runs used oQ4, `effort xhigh`, thinking on, `think_budget` off, same machine, same day, and greedy decoding.

### Result

Cache-aware never produced a file. It named `Face#pushpull` correctly and reasoned about it sensibly, then got stuck on whether
`Sketchup::Length.new` exists. It asked that about forty times in near-identical words, never wrote a declarative sentence, and used up the
budget. It had already written the correct idiom, `input[0].to_l`, and lost it. Exact-quality reached the code. It hit the same spot, spent
two paragraphs there, then decided:

```text
Safer: parse_length returns Length? ... Could use
height = Sketchup.parse_length(height_str) and then face.pushpull(height).
```

It then produced a module with `Sketchup.parse_length`, `sel.grep`, `model.start_operation`, and `next unless face.valid?`; caught the flaw
in its own sign fix on vertical faces where `normal.z` is zero; and correctly rejected `Selection#pushpull` and `Entities#pushpull` as
non-existent.

### Exact loops too, and gets out

A first reading of this run said exact-quality reached the code without looping. That was wrong. It got stuck three times: on whether
`Sketchup::Length.new` exists, on whether `require 'sketchup.rb'` is needed, and on whether the menu is `UI.menu("Plugins")` or
`"Extensions"`. Each time it repeated the question for a paragraph or two. It got out every time by deciding. "Safer: parse_length returns
Length ... Could use height = Sketchup.parse_length(height_str)". "Maybe safe to not require." Cache-aware got out of its first one and
never got out of the second. So it isn't knowledge and it isn't avoidance. Both named the right APIs and both got stuck. One of them ends it
by picking a branch.

### Every loop sat on a choice that didn't matter

Each one, on both profiles, was a question where both branches are correct:

```text
Sketchup::Length.new or String#to_l   both parse a length
require 'sketchup.rb' or omit it      both load
UI.menu("Plugins") or "Extensions"    aliased to the same menu since 2015
```

The model doesn't loop when it's missing a fact. It loops when it has enough and nothing tells it which to pick. oQ3-MTP failed the other
way round, confidently wrong about `Face#extrude`, which does have a right answer. That points at a mechanism. `swap_row` exchanges the
expert index and its weight together, so the kept mass falls by up to epsilon per swapped slot. `scores / selected_mass` then scales every
kept expert up to refill that mass, including the substitute, at about 1.4 swaps per layer across 48 layers. Inflating the kept set flattens
the differences between candidates, and ties are where these loops live. That predicts cache-aware loops harder than exact on ties, which is
what these two runs show. It's a hypothesis with a code location and a two-line test: swap the expert index and leave the weight alone, so
the router's mass for the slot survives and renormalization doesn't move.

### The prose degradation is effort, not routing

An earlier reading of this session blamed cache-aware for telegraphic thinking text, comparing against the oQ4 run in the test PDF, which
writes full English. That comparison doesn't hold. The PDF's oQ4 section carries no effort label while its oQ3 sections are labelled, so it
probably ran at a different effort. Exact-quality at `xhigh` produces the same telegraphese: "Need produce code", "Need be precise", "Need
ensure". That belongs to `xhigh`. Anything built on top of it, including a feedback loop between resident preference and narrowing output,
is withdrawn.

### What exact-quality produced

A complete module using `model.selection.grep(Sketchup::Face)`, `Sketchup.parse_length` with a nil check, `start_operation` with
`commit_operation` and `abort_operation` in a rescue, `next unless face.valid?` for faces an adjacent extrusion invalidates, and
single-argument `face.pushpull`, which avoids the `copy` parameter that broke oQ3-MTP at `low`. Its direction logic is right. A top face at
`normal.z = 1` takes `+height` and a bottom face at `-1` takes `-height`, so horizontal faces always extrude upward and other faces extrude
along the normal.

### Limits

One run per profile. A routing change cascades, so any single trajectory can go strange. The two runs shared prompt, checkpoint, effort,
machine, and hour, and differed only in routing, but n=1 can't separate routing from chance. Both runs decoded greedily, which the model's
authors advise against and which causes repetition by itself. Both also ran with `think_budget` off, so reasoning came out of the answer
allowance with no room of its own.

### Decision

Keep `exact-quality` as the default. Cache-aware stays an opt-in speed profile. Its 2.91 gen and 2.92 tail stand as measured. Repeat this
task with the recommended sampler and a think budget before treating the collapse as a property of either routing profile.

## Reasoning effort is one sentence, and medium is empty, 2026-09-01

The chat template turns `reasoning_effort` into a single line of system text:

```text
xhigh   Reasoning effort is set to xhigh. Please think carefully through the
        task, validate key assumptions, consider plausible alternatives, and
        prioritize correctness, consistency, and clarity in the final answer.
low     Reasoning effort is set to low. Keep your thinking brief and focused,
        moving directly to the conclusion without unnecessary elaboration.
medium  ''
```

`medium` isn't a middle setting, it's no instruction, and there's nothing between asking for validation and alternatives and asking for
nothing. The same SketchUp Ruby task at both levels, exact-quality, same checkpoint:

```text
xhigh   telegraphic thinking text, three repetition loops, and the better
        code: Sketchup.parse_length with a nil check, and a
        `next unless face.valid?` guard for faces an adjacent extrusion
        invalidates.
medium  full prose, no loops, and weaker code: String#to_l, which raises
        rather than returning nil, so its own nil check is dead, and no
        valid? guard at all.
```

Both behaviours follow from the instruction. `xhigh` was told to consider plausible alternatives and did, with nothing telling it when to
stop, so it looped on questions where both branches were correct. `medium` was told nothing, so it neither validated nor looped. `xhigh`
is not consistently worse. It caught a real defect that `medium` shipped.

### The fix

`high` keeps the validation and adds the missing stopping rule:

```text
Think carefully through the task and validate key assumptions before you
answer. When two options are both correct, choose one and move on. Do not
re-ask a question you have already answered.
```

The template raises on unknown effort names, so `high` passes `medium` to the template, which renders nothing, and prepends its own text to
the system turn. Checked against the real tokenizer: `medium` renders the system prompt alone, `high` renders the instruction above it,
`xhigh` renders the template's. `high` is project-specific. Qwen supports `low`, `medium`, and `xhigh` only. Qwen defaults to `xhigh`; this
project defaults to `medium`.

### What this means for every gate

Keep effort constant in any quality comparison. The oQ4 run in the test PDF that passed the SketchUp task has no effort label while the oQ3
sections below it are labelled. Its full prose suggests `medium`. Comparing it with `xhigh` runs produced two wrong answers in one session,
first blaming routing for telegraphic prose and then for loops. Run the trajectory gate at `medium` as well as `high`. `medium` is the
schema default and is where most output lives.

## Greedy decoding correction, 2026-09-01

Qwen's model card recommends thinking mode at `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, and `presence_penalty=0.0`.
It recommends a presence penalty from 0 to 2 for repetition. Higher values can mix languages and reduce quality.

The runtime used `mx.argmax` in `macqwen/backends/flashnext.py` and `models/flashnext/prefill.py`. This sets temperature 0 and omits the
rest. Greedy breaks a tie the same way every time, so a model that lands in a state where two continuations are near-equal can stay there.
Every observed SketchUp Ruby loops sat on such ties: `Sketchup::Length.new` against `String#to_l`, `require 'sketchup.rb'` against omitting
it, `UI.menu("Plugins")` against `"Extensions"`. Both branches are correct in each case, so nothing separated them. The loops showed up on
two routing profiles and two effort levels. Greedy isn't an oversight in the benchmarks. Every comparison here proves a change left the
trajectory alone by matching token IDs across arms, and sampling makes that impossible. So it's greedy for measurement and Qwen's sampler
for the chat.

### What shipped

`macqwen/sampling.py` holds `Sampling` and `Sampler`, with temperature, top-k, top-p, min-p, and presence penalty, defaulting to the card's
thinking-mode values. The backend defaults to greedy so no benchmark can sample by accident; the chat sets it from preferences.
`run_benchmark` forces greedy whatever the preferences say. `/config sampling` reads and writes it live, and `/config model` shows it alongside effort,
thinking, and the token budget. Checked against the real model, same prompt, thinking off:

```text
sampled  O mar sussurra segredos eternos à areia.
sampled  O mar respira e as ondas sussurram segredos antigos.
greedy   O mar sussurra segredos antigos à areia.
greedy   O mar sussurra segredos antigos à areia.
```

Two benchmark runs kept an identical token hash, so the measurement path is unchanged. At the real 248,320-wide vocabulary the sampler costs
about 3 ms per token, which is what the `argmax` it replaced already cost. One bug worth remembering: greedy first returned a scalar where
the decode loop needs shape `(1,)`. The unit tests used `.item()`, which hid it, and the real model caught it. There's a shape test across
all five sampler paths now.

### What this does to the gate result

Cache-aware was judged on a decode path the model's authors advise against. Repeat the SketchUp task with the recommended sampler before
treating the repetition collapse as a property of either routing profile.

### Also from the card

- Supported effort levels are `xhigh`, `medium`, and `low`, with `xhigh` the
default. This project defaults to `medium`, a step below.
- Lower effort is warned against for agentic work: it "can also lead to
insufficient analysis, more failures, and repeated retries". That matches `medium` shipping without the `Face#valid?` guard.
- Reasoning and final output are meant to have separate allowances, 262,144 and
131,072 tokens. Running with `think_budget` off puts reasoning inside the answer allowance, which is the opposite of the intent.

## Weight-preserving cache-aware swap rejected, 2026-09-01

Issue #10 proposed swapping the expert index while keeping the selected
expert's weight. This keeps selected route mass unchanged. It still changes
the expert function, so exact token identity remained the acceptance check.

The prototype used a separate environment switch. It did not change the
current cache-aware implementation. Six paired production-harness arms used
epsilon 0.02 and 60 tokens. The first two arms per condition were dropped:

| Condition | Gen median | Tail | Physical MB/token |
|---|---:|---:|---:|
| current cache-aware | 2.211 | 2.213 | 386.5 |
| weight-preserving | 2.142 | 2.103 | 379.2 |

The weight-preserving median was 3.1 percent lower. The run resolved only
differences above 11.1 percent, so the speed result is unresolved. It led in
three of six pairs. Physical reads fell in five of six pairs and by 1.9
percent at the median.

The seven-prompt exactness gate compared exact routing with the
weight-preserving variant. Three replies were identical. Four diverged at
generated tokens 43, 23, 37, and 31. No checkable answer was lost, but the
variant was not bit-perfect.

Decision: reject this variant for `exact-quality`. Keeping a route weight does
not preserve output when the expert index changes. The prototype was removed,
and the current cache-aware implementation remains unchanged.

## Open pathways from exact-quality to 3 tok/s, 2026-09-01

The retained production baseline is 2.713 tok/s, or 369 ms/token. Reaching
3 tok/s needs about 36 ms/token. The following paths remain open. Treat every
number below as a projection until the production harness measures it.

### Path 2: deschedule host work, not DMA ([#21](https://github.com/1architect/macqwen-releases/issues/21))

Every rejected overlap experiment moved SSD DMA into GPU work. That invokes
the measured unified-memory contention. Moving small host bookkeeping can be
different because it does not start storage traffic.

`cProfile` reports these current costs:

| Function | Reported time/token |
|---|---:|
| `to_mx` | 31 ms |
| `expert_cache.__call__` | 22 ms |
| `_rows_pread` | 25 ms |

These timings overlap through caller and callee stacks, so they are not an
additive 78 ms budget. `to_mx` and `_rows_pread` also move bulk data. Only the
bookkeeping part qualifies as low-bandwidth host work.

The early-submit control added `mx.eval(scores, inds)`, `tolist()`, and host
list work without DMA. It measured 498.5 ms/token against a 499.3 ms/token
baseline. This is evidence that small host work can fit outside the critical
path.

First instrument each interval with three states: NVMe requests pending, GPU
work pending, and host function active. Count only host intervals where both
devices are idle. If that exclusive window holds 15 to 20 ms/token, move one
dependency-safe bookkeeping block into an existing device wait. Do not move
DMA. Token IDs must match. This path does not need a quality gate.

### Path 4: measure `gather_qmm` in situ ([#22](https://github.com/1architect/macqwen-releases/issues/22))

The bandwidth record prices GDN at 67 GB/s and a small Q4 matvec at 105 GB/s.
It does not isolate the routed-expert gather inside the complete model. The
model gathers about 1,152 MB of expert data per token. At 105 GB/s this costs
about 11 ms. At 20 GB/s it costs about 58 ms.

Use fixed resident expert sets and the real gate, up, and down projection
shapes. Prebuild the exact arrays and routed indices. Time the three
`gather_qmm` calls, activation, gather sorting, and scatter restoration with
explicit evaluation boundaries. Verify every compared output with
`mx.array_equal`.

If the gather stays near 105 GB/s, close this GPU block. If it stays near
20 GB/s, about 30 to 45 ms of real GPU work becomes a focused optimization
target. This benchmark changes no model behavior and needs no quality gate.

### Exact compile sweep ([#23](https://github.com/1architect/macqwen-releases/issues/23))

`mx.compile` is bit-exact in the retained probes. It improved the PLE gate
chain by 28%, the router chain by 2.6%, and `_normalize_qk` by 5.7%. The
research estimate is 10 to 20 ms/token when applied to every suitable
elementwise chain. That estimate has not been tested in the complete runtime.

Add one experimental switch that compiles the router, QK normalization, PLE
gate, and other suitable chains. Do not duplicate compilation already present
inside upstream GDN. Exclude first-call compilation from steady-state timing,
but report its interactive cost separately. Require `mx.array_equal` at each
compiled boundary and identical greedy token IDs. Then use paired production
arms. This path needs no quality gate.

### Routed-expert Q4 group-size sweep ([#24](https://github.com/1architect/macqwen-releases/issues/24))

The oQ4 base uses four-bit affine quantization with group size 32. Its 228
protected modules are resident `shared_expert` and `linear_attn.out_proj`
modules. The streamed routed experts use plain Q4/G32.

One routed expert record is 3,072,000 bytes:

| Part | Bytes | Share |
|---|---:|---:|
| four-bit weights | 2,457,600 | 80% |
| BF16 scales and biases | 614,400 | 20% |

Larger groups keep four-bit codes and reduce only metadata:

| Group size | Bytes/expert | Projected physical MB/token | Saved |
|---:|---:|---:|---:|
| 32, current | 3,072,000 | 390 | 0% |
| 64 | 2,764,800 | 351 | 10% |
| 128 | 2,611,200 | 332 | 15% |

The physical-byte projection assumes the current cache hit rate stays fixed.
At about 185 ms of expert-read wait, G128 projects to 28 ms saved. That gives
about 341 ms/token, or 2.93 tok/s, before any compile gain.

This is lower risk than changing the base bit width. The routed weights stay
at four bits. All protected dense modules remain unchanged. The checkpoint
already uses group size 128 for some protected five-bit and eight-bit modules.
However, requantization changes weight values. This path is not bit-perfect
and must pass the complete quality gate.

Run a distributed in-memory G64 and G128 error probe before a full build. If
the local MoE error is acceptable, dequantize one Q4/G32 tensor at a time to
BF16 and requantize it to Q4/G128. Keep the installed oQ4 checkpoint immutable.
The projected output is about 101 GB, so the build needs external storage or
a second machine. Use paired production arms, then run the quality gate at
`medium` and `high` with sampling enabled.

The most plausible stack is Q4/G128 plus the exact compile sweep. Their
projected savings total 38 to 48 ms/token, enough to cross 3 tok/s if both
effects survive the complete runtime.

## Compatible checkpoint survey and REAP decision, 2026-09-01

A shard-header scan found several MLX checkpoints with the Flash-Next text
architecture. Each reports `model_type=qwen4_exp_text`, 48 layers, hidden size
2560, top-10 routing, and the `language_model.model.` tensor prefix. Sizes
below come from the real shard headers, not model-card summaries.

| Checkpoint | Disk | Experts | Expert quant | Bytes/expert | Expert bank | Gathered/token |
|---|---:|---:|---:|---:|---:|---:|
| [`Vontra/Qwen3.8-Flash-Next-MLX-oQ4`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4), installed | 111.7 GB | 512 | Q4/G32 | 3,072,000 | 75.5 GB | 1,169 MB |
| [`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) | **73.5 GB** | **288** | **Q4/G64** | **2,764,800** | **38.2 GB** | **1,052 MB** |
| [`sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit`](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit) | 86.3 GB | 384 | Q4/G64 | 2,764,800 | 51.0 GB | 1,052 MB |
| [`Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit`](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit) | 73.1 GB | 512 | Q2/G128 | 1,382,400 | 34.0 GB | 526 MB |
| [`Vontra/Qwen3.8-Flash-Next-MLX-oQ2`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ2) | 67.7 GB | 512 | Q2/G32 | 1,843,200 | 45.3 GB | 702 MB |
| `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` | 92.5 GB | 512 | Q3 mixed | about 2,457,600 | about 60 GB | about 935 MB |

The header method reproduces oQ3-MTP at 92.5 GB, equal to the recorded 86.2
GiB. This cross-check supports the table's decimal sizes.

### Primary candidate: REAP-288 ([#25](https://github.com/1architect/macqwen-releases/issues/25))

REAP-288 cuts the routed-expert bank from 75.5 GB to 38.2 GB. The measured oQ4
cold tail is about 154 distinct experts per layer and 21.7 GB. Proportional
scaling puts the 288-expert tail near 87 experts per layer and about 11 GB.
That could fit in the available page cache, but the earlier coverage mistake
shows that this remains a hypothesis until `mincore` and physical reads prove
it.

Q4/G64 also reduces each routed record by 10% while keeping four-bit weight
codes. The n-gram table remains Q4/G32. Two honest speed bounds follow:

- With no hit-rate change, physical reads project from 390 to 351 MB/token,
  or about 2.85 tok/s.
- At a 75% hit rate on 1,052 MB gathered, physical reads project near 263
  MB/token. That can put decode near 3.2 tok/s.

The second bound is the reason to test REAP-288. It is not a prediction. The
production harness must measure the new hit rate, physical bytes, and rate.

The model card reports HumanEval pass@1 at 91.5% for REAP-288 against 93.9%
for its stock Q4 control. It reports 92.1% for REAP-384. These are single runs
without confidence intervals, and calibration used agentic-coding traffic.
They do not replace this project's quality gate.

## Exact-quality performance sweep closed, 2026-09-01

The three investigations opened from the 3 tok/s gap are now measured. Issues
[#21](https://github.com/1architect/macqwen-releases/issues/21),
[#22](https://github.com/1architect/macqwen-releases/issues/22), and
[#23](https://github.com/1architect/macqwen-releases/issues/23) are closed.

### GPU, SSD, and dependency timing

The complete runtime was measured for 30 generated tokens on a cold machine.
The token took 492.5 ms and read 381.2 MB physically:

| State | ms/token | Share |
|---|---:|---:|
| GPU running | 197.4 | 40.1% |
| SSD reading | 226.7 | 46.0% |
| host only | 44.8 | 9.1% |
| unaccounted | 23.5 | 4.8% |

The GPU and SSD windows were disjoint. During a warm 2.713 tok/s run, the
split moves to about 53% GPU, 41% SSD, and 6% host because fixed host work
occupies less of the token. Neither device has spare capacity during the
other device's required work.

The dependency chain repeats 48 times per token:

```text
GPU scores -> host route -> SSD rows -> host assembly -> GPU expert matmul
```

The n-gram reads are independent because input IDs are known at token start.
They take 6.2 ms. Starting them before layer 0 measured +1.5%, inside noise
and close to the arithmetic ceiling. The shared expert also depends only on
the layer input, but running it during SSD reads measured 2.59 against 2.68
tok/s. Route prediction reached 59% accuracy and lost for the same reason.

### Path 2 closed: host-only time is too small

The host-window instrument measured 38.37 ms/token of exclusive host time:

| Window | Exclusive ms/token | Meaning |
|---|---:|---|
| `to_mx_host` | 34.21 | bulk copy of the gathered layer |
| `keep_loop` | 2.77 | Python threshold loop |
| `moe_issue_host` | 0.79 | host-side issue work |
| `plan_host` | 0.51 | route plan construction |
| `route_tolist` | 0.09 | route conversion |
| `io_await` | 0.00 | shared SSD wait, not exclusive |

The low-bandwidth bookkeeping totals 4.16 ms/token. It cannot recover the
15 to 20 ms required by Path 2. The 34.21 ms `to_mx_host` window copies the
whole gathered layer and remains the largest addressable host item, but it
belongs to bulk movement and stays excluded from this path. The independent
read counter and the `io_await` control agree that the window classification
is valid.

### Path 4 closed: routed `gather_qmm` is not the missing block

`models/flashnext/bench_gather_qmm.py` uses resident arrays, real checkpoint
shapes, chained repetitions, and one evaluation per chain. It verifies output
identity with `mx.array_equal`.

| Slots | Gate | Up | Down | Three projections | Time/token |
|---:|---:|---:|---:|---:|---:|
| 8 | 102.8 GB/s | 112.0 | 73.4 | 92.4 GB/s | 12.8 ms |
| 10 | 87.2 GB/s | 102.8 | 76.7 | 92.2 GB/s | 16.0 ms |

The result follows the 105 GB/s branch, not the 20 GB/s branch. Expert gather
costs 13 to 16 ms/token. It cannot explain a hidden 30 to 45 ms block.

The test also reproduces the short-chain timing error. At eight repetitions,
separate projection sums took 0.742 ms against 0.316 ms as one chain, a factor
of 2.3. At 64 repetitions they measured 0.264 and 0.266 ms. Separate stages
carry a synchronization cost and their times must not be added.

### Compile sweep closed: bit-exact and too small

`models/flashnext/compiled.py` compiles the retained router, QK normalization,
PLE gate, and related chains. The complete benchmark compares four kept arms:

```text
plain      2.68 tok/s   559.1 MB/token
compiled   2.66 tok/s   564.7 MB/token
```

Token IDs matched in every arm. The compiled median was 0.6% lower, and the
paired sign test gave p = 0.812. The three useful chain savings total about
1.0 ms/token, or 0.3%. Compilation stays available as a diagnostic and is
not installed by default.

### Layer attribution correction

The first component split produced an invalid residual because independently
timed chains allowed overlap. A dependency link fixed that error, but the
parts still did not add to the whole. A whole-layer control shows the scale:

| Measurement | ms/token |
|---|---:|
| individually timed component parts | 41.00 |
| whole decoder layers, expert pages hot | 255.93 |
| unexplained difference | 214.93 |

The 41 ms includes GDN, attention, both hyper-connections, PLE, the router,
and the shared expert. The remaining cost is MoE plumbing, dispatch, and
evaluation behavior that the current split does not isolate. The chained split
measures GDN at 18.25 ms/token. The earlier 57 ms in-situ figure used a
different synchronization pattern and needs a new control before use.

### Shared buffer plus chunk 2 is the first positive result

The earlier shared-buffer test removed the 35 ms concatenate but returned the
saving as GPU time. It used 16 workers that scattered writes across one
destination. The crossed test keeps the shared buffer but uses chunk 2, so
each worker writes a contiguous range while most of the SSD queue remains
active.

The final 12-arm comparison used 10 kept arms per condition:

```text
concat-chunk1  2.67 tok/s   467.7 MB/token
buffer-chunk2  2.83 tok/s   457.7 MB/token
```

`buffer-chunk2` won 10 of 12 paired arms, used fewer physical bytes in 10 of
12, and improved the generation median by 6.3%. The result resolves above a
4.4% band. Token IDs matched in every arm. This is the first positive result
from this sweep, and the 12-arm run started after a clean boot.
Issue [#26](https://github.com/1architect/macqwen-releases/issues/26) tracks
that change.

The current 36 ms gap remains open. The new split shows that the missing cost
is not routed expert matmul or small host bookkeeping. Issue
[#27](https://github.com/1architect/macqwen-releases/issues/27) tracks a valid
whole-layer GPU attribution method. Issue [#24](https://github.com/1architect/macqwen-releases/issues/24) remains open for Q4 group sizes, and issue
[#25](https://github.com/1architect/macqwen-releases/issues/25) remains open
for the REAP-288 gate.

## Quality warning from the REAP-288 discussion

In [discussion #5](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit/discussions/5), DarkJoney reported that one hard prompt
entered reasoning loops under oMLX 0.6.4. The behavior remained with the MTP
drafter off and with n-grams moved from SSD to memory.

The author later said the problem starts near `xhigh` effort. He described it
as an inference-time REAP side effect. His current workaround is effort
`high` or lower, or repetition penalty 1.05 to 1.10. He reported 1.08 as
verified and plans a light default penalty in pMLX.

This report is confounded, but it is not cleared. This project has reproduced
the same loop shape on unpruned oQ4 at `xhigh` and under greedy decoding. Every
recorded loop sat on a choice where both branches were correct. The external
report does not state its sampler and has no controlled unpruned Q4 arm. At
the same time, the author's response explicitly says REAP makes the behavior
worse at high reasoning effort.

A repetition penalty cannot serve as the first quality result. The oQ4
SketchUp answer reached correct code by doubting and correcting itself. A
penalty can hide repeated doubt without restoring that resolution. Qwen also
warns that larger penalties can mix languages and reduce quality.

### Required REAP-288 gate

Run the checkpoint with Qwen's sampler and no repetition penalty at `medium`,
`high`, and `xhigh`. `xhigh` is mandatory because the open report names that
regime. Use the recorded oQ4 `xhigh` SketchUp transcript as the control. Then
repeat REAP-288 at penalty 1.08 and judge the answer, not only loop removal.

The gate must verify the complete SketchUp file, factual prompts, API recall,
and long-context trajectory. Only after it passes should
`bench_production.py` measure rate and physical reads. REAP-384 is the fallback
when REAP-288 fails. It keeps Q4/G64 and still cuts the expert bank by 32%.

The two-bit checkpoints are rejected as first candidates. The retained
Q2/G32 probe measured 31.49% mean local error, and two-bit expert
requantization is already closed. `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`
also stays excluded until its repository file count and stored size reconcile
with a complete expert bank and n-gram table.

### Loader compatibility and storage cost

The current `_swap_ngram` loader builds prefixes as `shard_N`. REAP names the
tensors `ngram_embedding.shards.N.*`. The loader needs a small compatibility
change that accepts both forms. All routed `switch_mlp` tensor names already
match. Add a focused loader test before any checkpoint download.

Testing REAP-288 on this 256 GB machine still needs a storage decision. The
standing rule keeps oQ4 immutable, and local free space cannot hold both.
Deleting oQ4 makes a failed gate cost another 111.7 GB download. Do not start
the checkpoint transfer until external storage is available or that restore
cost is explicitly accepted.

## External oQ4-MTP repetition report

The [Vontra oQ4-MTP discussion](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4/discussions/2) reports a failure during long agentic and coding turns. The model entered a degenerate repetition loop, reached `max_tokens`, and ended with an unclosed tool-call envelope. The report describes two failures on an M5 Max with oMLX and matched sampling settings. One run reached 28,383 generated tokens after 12,487 MTP cycles. Another reached the limit after 14,684 tokens.

The report compares oQ4-MTP with oQ3-MTP under the same server settings. It reports two truncated oQ4-MTP requests out of eight and no truncated oQ3-MTP requests across 177 requests. The author identifies the quantized checkpoint as the main variable, but the environment and MTP runtime differ from this project.

The same report mentions a secondary `QSAKVCache` failure on oQ4-MTP. The cache object lacked `extend`, and recovery failed. The author marks this as a possible oMLX or `mlx-vlm` defect rather than a quantization defect. The checkpoint maintainer is testing longer runs and has not yet posted a fix.

This report does not measure standard oQ4, and it does not replace our local
quality gate. It supports the existing decision to keep MTP disabled in the
production backend. It also adds a long-agentic-run failure mode to the
checkpoint review record.
