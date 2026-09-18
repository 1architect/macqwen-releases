# Bonsai-2 measurements

This directory supports the active
[`research.md`](../research.md); operational instructions remain in
[`handoff.md`](../handoff.md).

All runs use greedy decoding with exact digests, one fresh child process per
arm, and forward/reverse/forward rounds. Throughput shows median with arm
range where present.

## Baselines (2026-09-19)

| Run (raw artifact) | Prompt → output | Decode median | Prefill | MLX active / peak | Correctness |
|---|---|---:|---|---|---|
| Short 2k ([JSONL](20260918-baseline-short.jsonl)) | 3,282 → 32 | 5.7 tok/s | ~124 s | 8.28 / 10.43 GB | 3/3 `d2004e2ef089` matches |
| Product 2k ([JSONL](20260918-baseline-product.jsonl)) | 3,282 → 256 | 5.5 tok/s | ~124 s | 8.31 / 10.43 GB | 3/3 `cbd9b4e29f07` matches |

## Comparisons (2026-09-19)

| Run (raw artifact) | Result |
|---|---|
| Prefill chunk 512/1024/2048 at 8k ([JSONL](20260918-prefill-wide-8k.jsonl), 7 of 9 arms) | Prefill flat within noise; peak 11.63 / 13.0 / 15.47 GB. Keep 512. All digests `161f886164bb` match. |
| Cache step 256/1024 at 16k product ([JSONL](20260918-cache-step-16k.jsonl), 2 of 6 arms) | Directional only: decode tie ~4.0 tok/s, identical KV allocation. Not a promotion result. |
| Allocator cap at 2k product ([JSONL](20260918-allocator-2k.jsonl)) | Pool 774 → ~300 MB, decode medians identical. Promote to default after 16k confirmation. |
| Wired limit at 2k product ([JSONL](20260918-wired-2k.jsonl)) | No resolved benefit; keep off. |
| Post-generation clear at 2k product ([JSONL](20260918-clear-cache-2k.jsonl)) | Pool → ~1 MB, steady-state rates identical. Opt-in diagnostic. |

## Decisions and scope

Accepted:

- We accept the product 2k control at about 5.5 tok/s with MLX peak
  10.43 GB as our sustained baseline.
- We keep prefill chunk 512 and wired off.

Not promoted:

- We do not promote the allocator cap yet despite zero-cost 470 MB savings
  at 2k; it needs the 16k confirmation run.
- We do not promote any 16k cache-step result; only 2 of 6 arms completed.
- We reject greedy `logsumexp` skipping (0.5 ms of ~190 ms/token),
  `mx.compile` of the decode step (Python dispatch is ~5% of the token),
  and `lm_head` prefill skipping (~5% of a paging-bound prefill) as
  vendored-loop work without a surviving complete-runtime gain.
- 262K exact-only context is unreachable on 16 GB (16.8 GB KV payload
  alone). The honest ceiling is about 32k; see the research record.
