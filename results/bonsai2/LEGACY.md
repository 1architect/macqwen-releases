# Bonsai-2 measurements

This directory supports the active
[`research.md`](../../docs/bonsai2/research.md); operational instructions remain in
[`handoff.md`](../../docs/bonsai2/handoff.md).

All runs use greedy decoding with exact digests, one fresh child process per
arm, and forward/reverse/forward rounds unless noted. Throughput shows median
with arm range where present. Every arm carries its token digest; a mismatched
digest rejects the comparison before any speed reading.

The current low-context verification is retained separately from the
historical table below. It is diagnostic only: system VM activity occurred,
and schema-1 physical-read values span prefill plus decode.

Current check: [`20260919-low-context-short-check.jsonl`](20260919-low-context-short-check.jsonl)
measured 3,283 → 32 tokens at 7.375, 7.339, and 7.328 tok/s (median 7.339),
with three matching greedy digests and 10.43 GB peak MLX memory.

New live prefill screens are deliberately bounded: they use the `context-1k`
fixture (32 records; under 1,000 rendered prompt tokens) and two
reverse-interleaved rounds, four total arms. The historical 3,283-token,
8K, and 16K records remain append-only evidence; we do not rerun them merely
to fill a table, and a larger context requires explicit user authorization.

## Master results table

| # | Run (raw artifact) | Prompt → output | Decode, tok/s by arm | Prefill, s by arm | Memory | Digests | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Short 2k ([JSONL](20260918-baseline-short.jsonl)) | 3,282 → 32 | 4.97, 5.84, 5.74 | 103, 130, 124 | peak 10.43 GB | 3/3 `d2004e2ef089` | Historical reference |
| 2 | Product 2k ([JSONL](20260918-baseline-product.jsonl)) | 3,282 → 256 | 5.44, 5.55, 5.54 | 117, 126, 124 | peak 10.43 GB | 3/3 `cbd9b4e29f07` | Historical; revalidate |
| 3 | Prefill chunk 512/1024/2048 at 8k ([JSONL](20260918-prefill-wide-8k.jsonl), 7 of 9 arms) | 9,752 → 32 | 4.7–5.0, flat | 400–546, flat | peak 11.63 / 13.0 / 15.47 GB | all `161f886164bb` | Keep 512 |
| 4 | Cache step 256/1024 at 16k product ([JSONL](20260918-cache-step-16k.jsonl), 2 of 6 arms) | 19,458 → 256 | 3.94 vs 3.99 | 1477 vs 1006 | KV 2,774 MB both | both `a6bbfb944b17` | Directional only |
| 5 | Allocator cap at 2k product ([JSONL](20260918-allocator-2k.jsonl)) | 3,282 → 256 | control 5.56–5.58, capped 5.55–5.56 | tied ~124 | pool 774 → ~300 MB | all `cbd9b4e29f07` | Historical promotion; revalidate |
| 6 | Wired limit at 2k product ([JSONL](20260918-wired-2k.jsonl)) | 3,282 → 256 | tied ~5.56; one 5.90 outlier tracks page warmth | tied | — | all match | Keep off |
| 7 | Post-generation clear at 2k product ([JSONL](20260918-clear-cache-2k.jsonl)) | 3,282 → 256 | tied ~5.57 | tied | pool → ~1 MB | all match | Opt-in diagnostic |
| 8 | Fused FWHT at 2k short ([JSONL](20260919-fused-fwht-short.jsonl)) | 3,282 → 32 | control 7.27, 5.96, 6.00 vs fused 6.30, 6.29, 6.08 | tied | peak 10.54 fused | all `d2004e2ef089` | Default-on on prefill evidence; decode unresolved |
| 9 | Locked baseline 2k ([JSONL](20260919-phase0-2k.jsonl)) | 3,282 → 32 | control 8.21, 6.46, 6.04 vs fused 7.40, 7.19, 6.15 | 85–120 | — | all `d2004e2ef089` | Decode tied within noise |
| 10 | Locked baseline 8k ([JSONL](20260919-phase0-8k.jsonl)) | 9,752 → 32 | control 4.21, 4.43 vs fused 4.62, 4.41 | 459–600, first-arm cold | peak ~11.7 GB | all `161f886164bb` | Decode ties; prefill confounded |
| 11 | Quantized KV 8-bit at 2k short ([JSONL](20260919-quant-kv8-short.jsonl)) | 3,282 → 32 | control 7.76, 7.51, 5.12 vs q8 5.77, 7.01, 5.47 | tied | KV 593 → 280 MB | all `d2004e2ef089` | Opt-in; speed unresolved |
| 11b | Quantized KV 4-bit at 2k short ([JSONL](20260919-quant-kv4-short.jsonl)) | 3,282 → 32 | control 4.70, 5.72, 5.34 vs q4 5.17, 5.16, 5.66 | tied | KV 593 → 225 MB | all `108ee966ba92` | Opt-in; speed unresolved |
| 12 | Shared transforms screen ([JSONL](20260919-share-fwht-screen.jsonl)) | 51 → 32 | control 8.40, 8.35 vs shared 8.42, 8.41 | ~2–8 | peak 8.6 GB | all `730c92bf` | Screen only |
| 12b | Shared transforms full ([JSONL](20260919-share-fwht-full.jsonl)) | 3,282 → 32 | control 8.25, 5.39, 5.66 vs shared 5.83, 5.73, 5.95 | tied | — | all `108ee966ba92` | Digest-equal; paired −29%, +6.3%, +5.1%; stays opt-in |
| 13 | Shared transforms short ([JSONL](20260919-share-fwht-short.jsonl), 3 arms) | 3,282 → 32 | control 5.54 vs shared 5.26, 4.68 | 130–172 | — | all `d2004e2ef089` | Incomplete; rerun |
| 14 | Decode 3k tg128 ([JSONL](20260919-decode-3k-tg128.jsonl)) | 3,282 → 128 | 7.02, 6.08, 5.98 | — | — | all match | Locked reference |
| 15 | Decode 3k tg256 ([JSONL](20260919-decode-3k-tg256.jsonl)) | 3,282 → 256 | 5.57, 5.79, 5.75 | — | — | all match | Locked reference |
| 16 | Custom GEMV at tg128 ([JSONL](20260919-gemv-decode.jsonl)) | 3,282 → 128 | control 6.45, 5.32, 5.27 vs gemv 5.50, 5.37, 5.21 | tied | — | all match | Rejected for default |
| 17 | Allocator cap at 16k product ([JSONL](20260919-allocator-16k.jsonl), 3 of 6 arms) | 19,458 → 256 | control 4.29 vs capped 3.82, 3.92 | control 1327 vs capped 895, 870 | pool 2,890 → ~270 MB | all match | Directional; fast-first pattern |

## Incomplete diagnostics

These records are retained for failure and provenance, not promotion:

- [`20260919-q4-attention-8k.jsonl`](20260919-q4-attention-8k.jsonl) contains
  an interrupted arm.
- [`20260919-q4-attention-smoke.jsonl`](20260919-q4-attention-smoke.jsonl) is a
  one-arm smoke check.
- [`20260919-low-context-short-check.jsonl`](20260919-low-context-short-check.jsonl)
  is a current diagnostic check, not a sustained baseline.

## Paired effects

Decode effect is positive when the candidate is faster; prefill effect is
positive when the candidate is shorter. Machine noise runs about ±10% on
decode and wider on prefill; first-arm outliers track page-cache coldness,
not the candidate.

| Run | Decode effect by round | Prefill effect by round |
|---|---|---|
| 3, prefill chunk | 1024: −2.9%, −3.8%; 2048: +3.5%, −0.7% | 1024: −9.9%, +0.0%; 2048: −7.7%, +3.3% |
| 4, cache step | +1.4% | +31.9% once, confounded cold |
| 5, allocator cap | −0.1%, −0.4%, −0.5% | +1.5%, −0.2%, +0.3% |
| 6, wired limit | −0.9%, −0.4%, +5.8% | −0.3%, −1.1%, +11.4% |
| 7, post-generation clear | −7.3%, +0.2%, +0.3% | −17.6%, −0.1%, +0.3% |
| 8, fused FWHT | −13.3%, +5.5%, +1.4% | −24.8%, +3.7%, −0.3% |
| 9, locked 2k | −9.9%, +11.3%, +1.9% | −2.9%, +6.9%, +0.9% |
| 10, locked 8k | +9.9%, −0.6% | +16.0%, −6.4% |
| 11, quantized KV 8-bit | −25.6%, −6.7%, +6.8% | −3.9%, +16.6%, +16.9% |
| 12, shared screen | +0.1%, +0.7% | +77.4% once (cold), +0.9% |
| 16, custom GEMV | −14.8%, +1.0%, −1.1% | −42.5%, +3.9%, −0.5% |
| 17, allocator 16k | −11.0% once | +32.6% once, confounded |
| 12b, shared full | −29.3%, +6.3%, +5.2% | −39.0%, +1.1%, −0.3% |

## Cross-engine spot checks (single runs, directional)
| # | Setup | Prefill | Decode | Note |
|---|---|---|---|---|
| 14 | GGUF Prism Metal, 512-prompt | 41.3 tok/s | 7.64 tok/s tg128 | Published methodology; holds only at short prompts |
| 15 | GGUF Prism Metal, 3,282-prompt | 26.9 tok/s | 5.82 tok/s tg32 | Ties MLX at 3K; GGUF path closed after this |
| 16 | MLX stock, ~776-prompt | 24.6 tok/s | 7.75 tok/s tg128 | Matched short context |
| 17 | MLX stock, 1,308-prompt | 21.6 tok/s | 5.88 tok/s tg128 | Longer context reads slower on both engines |
| 18 | MLX fused FWHT, 776-prompt | 33.4 tok/s | 7.70 tok/s, digest matches | Directional prefill probe behind the default |

## Microbenchmarks and probes (no promotion weight)

| # | Probe | Result |
|---|---|---|
| 19 | Prefill subsystem split, 2,476 tokens | QMM 90.8%, rotation 2.4%, attention 2.4%, recurrence 2.3%, norm 1.1%, conv 1.0% |
| 20 | QMM gate projection, batch 512 | 2.1 TFLOPS vs ~4 dense roof; fused 43.0 ms vs dequantize 7.7 + dense 38.9 ms |
| 21 | Greedy logsumexp | 0.5 ms of ~190 ms/token; rejected |
| 22 | Chat sampler full vs survivor path | 3.5 ms vs 2.1 ms; shipped |
| 23 | GDN general-path conv per layer | 0.17 ms, 8.1 ms/token ceiling; no specialization |
| 24 | ANE single ternary layer | 535 MB package, 6.1 ms vs 4.4 ms Metal, wrong output; line closed |
| 25 | PLD n-gram acceptance, code transcript | 35% of tokens repeat context at n≥3; implementation deferred |

## Long-context allocator notes

The 16k allocator file ([JSONL](20260919-allocator-16k.jsonl)) holds 3 of 6
arms: control 4.29 tok/s with pool 2,890 MB against capped 3.82 and
3.92 tok/s with pool ~270 MB, all digests matching. The capped arms match
historical 16k levels while the control repeats the fast-first pattern, so
no regression is established either way. The chat default cap still rests
on the complete 2k evidence.

## Decisions and scope

Current evidence:

- We record the current short low-context control at 7.34 tok/s as a
  diagnostic rate only; it is not our sustained baseline.
- We keep prefill chunk 512 and wired off.

Historical decisions requiring current revalidation:

- The allocator cap remains the chat default on complete historical 2k
  evidence; the current short check does not validate its memory behavior.
- The historical 5.5 tok/s product baseline remains unconfirmed on the
  current runtime.
- We do not promote any 16k cache-step result; only 2 of 6 arms completed.
- We reject greedy `logsumexp` skipping (0.5 ms of ~190 ms/token),
  `mx.compile` of the decode step (Python dispatch is ~5% of the token),
  and `lm_head` prefill skipping (~5% of a paging-bound prefill) as
  vendored-loop work without a surviving complete-runtime gain.
- 262K exact-only context is unreachable on 16 GB (16.8 GB KV payload
  alone). The honest ceiling is about 32k; see the research record.
