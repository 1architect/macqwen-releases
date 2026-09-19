# Bonsai-2 ternary 27B research record

This is our active record of Bonsai-2 measurements, rejected ideas, and
design decisions. Current operation belongs in [`handoff.md`](handoff.md).
Exact commands, the compact results table, and retained JSONL arms will live
in [`measurements/`](measurements/).

## 2026-09-18 — Branch setup

We created branch `bonsai-2-research` and added `models/bonsai2/` following
the K2-Horizon package pattern: settings, checkpoint discovery with
`prism_hadamard_qwen35` compatibility, resident backend, protocol translator,
KV cache helper, fresh-process bench harness, and checkpoint-free tests.

Key loader finding from the downloaded checkpoint: the pack stores
`language_model.*` plus `vision_tower.*` keys in one `model.safetensors`.
The text-only path must use the bundled `runtime/vision_artifact.py`
`load_vl_model` entry point and run its `language_model`. The bare
`runtime/artifact.py` `load_model` path targets text-only packs and does not
match this pack's key namespace. Stock `mlx_lm.load` skips the Hadamard
activation transform and returns wrong output silently, so our backend
refuses to run when `runtime/` files are absent.

We downloaded `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (about 8.6 GB) to
`~/models/Ternary-Bonsai-2-27B-mlx-2bit`. Milestone 1 stays text-only with
the vision tower unloaded.

## 2026-09-18 — Stop rewind fix (Phase 0)

`_rewind_stop_token` decremented only KV offsets. The 48 GDN linear states
have no offset and already absorbed the stop token through the one-ahead
lookahead, so the next turn continued from contaminated state. A live
two-turn probe (stop at a reference token, continue, compare against a fresh
replay of the identical tape) diverged at generated position 10 with equal
lengths, confirming the bug.

The fix drops the whole cache on stop and replays the tape on the next turn
via the existing `_mark_replay_needed` path. The dead rewind helper is
removed. Unit tests pin the replay marking with fake caches. The live probe
now reports MATCH over 267 continuation tokens. Replay costs a prefill on
the turn after a stop; we measure that cost as its own arm in Phase 1.

## 2026-09-18 — Live smoke test

We ran the resident backend against the downloaded checkpoint in `.venv`
(MLX 0.32.2, MLX-LM 0.31.3, mlx-vlm 0.6.17). Both arms used the bundled
`runtime/vision_artifact.py` loader and kept the vision tower unloaded.

- No-thinking hello prompt, 64 prompt tokens, 32-token limit: finish `stop`,
  3 tokens, visible `Hello!`, cache invariant holds.
- Thinking `medium` arithmetic prompt (17 times 23), 72 prompt tokens,
  64-token limit: finish `stop`, 52 tokens at about 8.5 tok/s, correct
  answer 391 with reasoning closed, cache invariant holds.

Three loader facts follow from this run. The text path must use
`load_vl_model` with `load_processor=False` and run its `language_model`;
the bare `artifact.load_model` path does not match this pack's
`language_model.*` key namespace. The language model answers
`LanguageModelOutput`, so our backend unwraps `.logits` before the shared
`generate_step` helper. Its prompt cache mixes 48 `ArraysCache` and 16
`KVCache` layers, so validation and the invariant accept both kinds instead
of requiring every layer to carry an offset.

## 2026-09-18 — Truncated turns after tool use

Interactive agent chat ended turns with no visible answer right after the
model reached for a tool. The cause was our protocol translator, not the
model. Bonsai-2 emits native Qwen XML calls
(`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`),
which already match the shared contract. The first translator tried to
JSON-parse every tool block and replaced non-JSON content with a bare close
tag, swallowing the function name. The tool filter then found no function,
nothing executed, and the turn closed with an empty answer.

The fix passes non-JSON tool blocks through untouched and converts JSON
blocks to the shared XML form as before. A forced `list_dir` probe now
parses to `[('list_dir', {'path': '.'})]`, and a regression test pins the
passthrough across split-marker chunkings.

## 2026-09-19 — Speed and memory passes (Phases 1–4)

All runs use `.venv` on the reference M4/16GB machine, greedy decoding with
exact digests, fresh child process per arm, forward/reverse/forward rounds.
Raw arms live in [`measurements/`](measurements/). Digests match within
every comparison.

### Baselines

| Run | Prompt → output | Decode median | Prefill | MLX active / peak |
|---|---|---:|---|---|
| Short 2k | 3,282 → 32 | 5.7 tok/s | ~124 s | 8.28 / 10.43 GB |
| Product 2k | 3,282 → 256 | 5.5 tok/s | ~124 s | 8.31 / 10.43 GB |
| 16k probe (cache-step file, 2 arms) | 19,458 → 256 | 4.0 tok/s | ~1,240 s | — / 13.42 GB |

Decode falls from 5.5 tok/s at 2k to 4.0 tok/s at 19.5k context. Prefill runs
about 26 tok/s at 2k and 16 tok/s at 19.5k. Round-1 arms run slower
throughout; weight paging from the page cache dominates long-prefill timing
more than chunk compute does.

### Prefill chunk sweep (512 vs 1024 vs 2048, 8k fixture, 7 of 9 arms)

Prefill seconds are flat across chunk sizes within machine noise, while peak
memory climbs steeply: 11.63 GB at 512, 13.0 GB at 1024, 15.47 GB at 2048.
Chunk 2048 nearly exhausts the 16 GB machine for no speed gain, so 512 stays
the default. Digests match across all three sizes.

### Allocator, wired limit, post-generation clear (2k product, 6 arms each)

- Allocator cap 256 MB: pool drops from ~774 MB to ~300 MB with identical
  decode medians (5.56 vs 5.57 tok/s) and matching digests. The cap is free
  memory at 2k; promote it to the default after a 16k confirmation run.
- Wired limit: five of six arms tie near 5.56 tok/s; the sixth (5.90 tok/s
  with a faster prefill) tracks page-cache warmth, not wiring. No resolved
  benefit; keep off. This matches the K2 wired rejection.
- Post-generation clear: pool drops to ~1 MB with identical steady-state
  rates. Keep as an opt-in diagnostic; combine with the allocator cap only
  after the 16k confirmation.

### Fused FWHT kernel (exact, speed-unresolved on decode)

`models/bonsai2/ternary_kernel.py` folds the per-module sign multiply,
Hadamard transform, and downcast into one Metal dispatch. Two bugs fell out
during validation: the launch grid counts total threads, not threadgroups,
and signs index by last-dim position, not flat offset (3D grouped inputs
read out of bounds otherwise). Both are fixed with regression tests.

The kernel is bit-exact against the stock path on production shapes and the
full-module level with real weights. The six-arm `fused-fwht` comparison
keeps digest `d2004e2e...` on all arms: exactness gate passes. Decode speed
is unresolved: paired effects are −13.3%, +5.6%, +1.5% (mean −2.1%, median
+1.5%), inside any honest band on this machine's ±15% arm noise.

A directional prefill probe (same 776-token prompt, one run each side)
shows 29.1 tok/s control versus 33.4 tok/s fused with matching digests
`0e6dfb5ba42ce281`, consistent with prefill multiplying every launch by
batch width. Directional only, not promotion evidence. Promoted to default
on this prefill evidence with decode digests matching on all arms; the
stock path remains available via `fused_fwht=False`.

### NPU re-probe (single ternary layer, closed)

We installed coremltools 9.0 and exported one layer-0 MLP to Core ML,
dequantized to fp32 program with fp16 precision. Results close the line:

- Single-layer package is 534.8 MB, projecting to about 34 GB for 64
  layers against the archived 0.5–1.1 GB ANE resident limit. Capacity
  alone rejects full-model offload again, by an order of magnitude.
- Isolated ANE resident median is 6.1 ms against 4.4 ms for the Metal
  packed path. No per-layer win even before correctness.
- The ANE output mismatches completely (relative RMSE 1.25, cosine −0.03)
  because the weights live rotated and the activation transform has no
  in-graph equivalent. Correctness would need the FWHT staged in MIL ops,
  more work for a path already rejected twice over.

The 2026-08-24 verdict stands for Bonsai-2: sound idea, hardware cannot
hold it. We spend no further machine time here.

### Full FWHT-into-matmul fusion (assessed, not built)

The wheel ships the Q2 `qmv` vector helpers, so the extraction pattern
applies. But the fusable traffic is only the transformed activation,
about 3–4 MB per token, worth about 0.04 ms against a 190 ms token
(0.02%). The single-dispatch kernel already removed the launches and
measured neutral end to end. Building the full fusion can only chase that
0.02%, so we do not build it. The launch-level kernel stays as a
bit-exact diagnostic.

### KV quantization to 8-bit (opt-in, digest-equal)

All cache state runs fp32: 48 GDN layers hold about 157 MB constant and the
16 full-attention layers cost 131 KiB per token. The backend accepts
`quantized_kv=(bits, group_size)` and converts the 16 KVCache layers at
construction and on every replay; GDN state stays fp32. The six-arm
`quant-kv8` comparison keeps digest `d2004e2e` on all arms: the greedy
trajectory survives. KV allocation halves from 593 MB to 280 MB at 2k plus
32 tokens. Speed is unresolved inside machine noise (paired −26%, −7%,
+7% while the control itself swings 5.1–7.8 tok/s across rounds). Q8 stays
opt-in; it is the leading candidate for long context, where halved KV
directly extends the fitting ceiling. Quality validation stays manual.

### Further speed avenues (measured)

- Vision weights: the backend keeps only `language_model` and drops the
  parent VL model after load, so the 0.92 GB tower never stays resident.
  Verified by construction; no saving left to take.
- `lm_head` prefill skip: full-vocab logits cost about 1.6 GB of traffic
  over a 124 s prefill, around 0.1%. Rejected; it needs a vendored
  generate loop for nothing measurable.
- Sampled sampler: the chat path ran full-vocab softmax, argsort, and
  reductions per token (3.5 ms). The survivor path gathers top-k first and
  runs the identical mask sequence over 20 elements (2.1 ms), saving about
  1.4 ms per token. Masked entries are exactly `-inf`, so the nucleus
  matches up to fp summation order; greedy decoding is untouched and
  benchmarks stay greedy. Unit-pinned for shape, determinism under seed,
  and top-k membership.

### Shared transforms (memoized, digest-equal on screen)

Same-width sign vectors are byte-identical per loaded checkpoint (3 distinct
vectors across 402 modules, verified by hash), and the backend proves it at
construction before arming the memo. An instrumented forward shows 144 of
402 transforms eliminated, matching the predicted table. A four-arm smoke
screen keeps digest `730c92bf` on all arms with rates tied (+0.1%, +0.7%).
Screening never promotes; the full comparison remains future work.

### Stop retention (kept, with a recorded boundary difference)

Retained `<|im_end|>` continuations do not match replay references
token-for-token: the first generated token flips between the two turn
boundary markers while the following prose is identical, deterministically
on both sides. Prefill-path and decode-path batching round differently at
the boundary race. The prose equality is verified by inspection; the
trajectory gate stays manual. Other stops and interruptions keep the replay
recovery path.

## Status: landed versus open
Landed on this branch: gated stop-token retention, shared transforms per
projection group with per-backend enablement, cache-growth repair targeting
the real classes, allocator cap as chat default, opt-in 8-bit KV with
digest-equal short evidence, fused FWHT kernel default-on with matching
decode digests, chat sampler over top-k survivors, text-only materialization
with strict language-subtree loading, hardened runtime entry-point checks,
runtime fingerprinting in benchmark identity, sharded weight loading, locked
3K/8K baselines with a closed GGUF comparison, and a measured prefill
profile that reordered the plan toward quantized GEMM.

Open: the PLD acceptance probe, the 16k allocator confirmation, and the
full shared-transform comparison.

### Runtime ownership (Phase 10 recommendation)
Keep the native MLX backend and import ideas selectively, as done
throughout this record. Adopting mlx-serve as the engine is rejected on
evidence: decode ties GGUF at matched context, the prefill gap sits inside
one kernel family at 2.3 versus 4 TFLOPS, and every portable mlx-serve idea
either measured small or duplicated landed work.

### PLD acceptance probe (positive signal, not implemented)

Static n-gram analysis on a 256-token greedy code-explanation transcript
(1,806 prompt tokens) simulates prompt-lookup drafting without running a
draft model. About 35% of output tokens repeat context n-grams of length 3
or more, projecting roughly 1.3 to 1.5x effective throughput at block 8.
Greedy transcript overstates chat acceptance under sampling, and this is
one transcript. The signal justifies implementing PLD behind a flag with
acceptance, effective rate, and target-call metrics kept separate from
base-runtime tables.

### Text-only materialization (shipped)

The backend now mirrors the bundled construction with vision tensors
filtered out before loading and the tower module dropped afterward, keeping
all schema, duplicate, shape, and sign validation. Load peak falls from
8.6 GB to 7.68 GB with identical hello-smoke behavior and a true cache
invariant. Load wall time is unchanged at about 4.8 s: the file read
dominates. This is a loading-memory improvement, not a decode gain.

A first version loaded with `strict=False`, which could silently retain
initialized values for missing auxiliary weights. The loader now drops the
tower module first and loads the language subtree with `strict=True`, so
every remaining parameter is validated. A unit test pins the strict call
and the live strict load reproduces the 7.68 GB peak.

### Stop retention (experimental, off by default)

Retained `<|im_end|>` continuations do not match replay references
token-for-token: the first generated token flips between the two turn
boundary markers while the following prose is identical, deterministically
on both sides. Identical prose in one probe does not clear the no-loss
requirement for boundary control tokens, so retention ships behind
`retain_stop=False` until user-turn and tool-turn continuation checks pass.
Other stops and interruptions keep the replay recovery path regardless.

### Shared transforms (per-backend enablement)

The memo hook replaces a process-global function, so enablement now belongs
to the backend instance: the wrapper arms the memo only for backends
constructed with sharing on, after those backends verify same-width signs.
A backend without the flag never consults the memo even when another
backend installed the hook. Pinned by a two-backend unit test.

### GDN convolution profile (closed without specialization)

The general-path depthwise convolution measures 0.17 ms per layer, or
8.1 ms per token across 48 GDN layers: about 4% of a token. An infinitely
fast fp32 specialization could save at most that, so no custom kernel.
The fp32 weights stay untouched per the no-loss requirement.

### Native GGUF comparison (closed)

Matched runs on M4/16GB with the Prism Metal fork (build 9a9394a) against
PQ2_0 show both engines tied at 3K context: prefill 26.9 tok/s GGUF versus
about 26 to 30 tok/s MLX, decode 5.82 tok/s GGUF versus 6 to 8 tok/s MLX,
with MLX digests matching across arms. Decode is hardware-bound on both
engines. The 41.3 tok/s GGUF figure holds only for 512-token prompts and
does not transfer to 3K contexts. No further GGUF work: the comparison
answered its question and the GGUF path stays out of the project. The
7.2 GB GGUF file and the Prism fork build tree are deleted.

### Quantized GEMM dissection (Phase 5 answer)

Per gate-projection matmul at batch 512: fused `quantized_matmul` 43.0 ms,
separate dequantize 7.7 ms plus dense fp16 38.9 ms. MLX already fuses;
the dense path itself runs 2.3 TFLOPS against a roughly 4 TFLOPS roof.
A perfect single-pass kernel recovers at most the 7.7 ms temporary, about
1.2x on the matmul and far less end to end after Amdahl. The remaining
prefill gap is kernel-tuning depth, not a missing algorithm. We do not
build a custom QMM: disproportionate effort for a fractional gain, with
tuning risk against Metal's own dense path.

### Prefill subsystem profile (measured)

One 2,476-token prefill, 84.4 s wall, each subsystem wrapped with its own
eval boundary for attribution:

| Subsystem | Seconds | Share of wall |
|---|---|---:|
| Quantized matmul | 76.7 | 90.8% |
| Hadamard rotation | 2.1 | 2.4% |
| Attention | 2.0 | 2.4% |
| GDN recurrence | 1.9 | 2.3% |
| Normalization | 1.0 | 1.1% |
| Convolution | 0.8 | 1.0% |

The QMM microbenchmark on the gate projection reaches 2.1 TFLOPS at batch
512 against a roughly 4 TFLOPS dense roof: unpack-bound, not
bandwidth-bound. This reorders the optimization plan. Blocked GDN (2.3%),
SIMD rotation (2.4%), and residual-norm-rotation fusion (under 4%
combined) are deprioritized. Quantized GEMM at 91% goes first, and the
41.3 versus 33.4 gap reads as a QMM efficiency gap, not launch overhead.

### Allocator cap at 16k (not run)

A first 16k attempt was killed twice by tool timeouts (each 16k arm needs
15–25 minutes of prefill) and then aborted on request. Its 3 observed arms
are cited here, not as results: round-1 control 3.46 tok/s with 1,458 s
prefill against round-1 capped 4.16 tok/s with 801 s prefill, and round-2
capped 4.16 tok/s with 823 s prefill, all digests matching. The gap
confounds the cap effect with first-arm page-cache coldness, so it proves
nothing alone. The 16k confirmation remains future work; until it lands, the
chat default cap rests on the complete 2k evidence only.

Two kernel bugs fell out during validation, both fixed with regression
tests. The `metal_kernel` grid counts total threads, not threadgroups: our
first launch ran one element per group and returned mostly zeros. Signs
index by last-dim position, not flat offset: 3D grouped inputs read out of
bounds otherwise, which surfaced as full-forward NaNs while 2D probes
stayed exact. The full-forward probe now matches to 0.0036 maxabs with
identical greedy digests.

### Allocator cap promotion

The 2k comparison showed pool memory falling from about 774 MB to about
300 MB with identical decode medians and matching digests, so `session.py`
now constructs the chat backend with a 256 MB cap. The backend library
default stays off so bench controls remain valid references. A construction
test pins the 256 MB value. The 16k confirmation run below decides whether
the cap survives long-context prefill pressure.

### Decode micro-probes (no model runs beyond unit scope)

- Greedy `logsumexp` costs 0.5 ms against ~190 ms/token (0.3%). Skipping it
  needs a vendored generate loop, so we reject the change with data.
- `cProfile` over a 16-token turn: model-forward Python is 0.3 s of 6.1 s
  wall; the rest is GPU execution plus sync waits inside `generate_step`.
  Python dispatch is about 5% of the token, so `mx.compile` has almost
  nothing to reclaim. Rejected with data.
- FWHT stays the primary structural lever: about 400 sign-plus-Hadamard
  launches precede 400 quantized matmuls per token, and fusion needs a
  custom Metal kernel. That kernel is the recommended next work item, gated
  on greedy digest equality plus profiled decode windows.
- `lm_head` skip during prefill saves about 4–5% of prefill traffic plus
  254 MB peak per 512-token chunk, but also needs a vendored generate loop.
  Deferred: prefill is paging-bound here, so the gain would not survive the
  complete runtime until the cache story improves.

### Context ceiling toward 262K (exact-only)

Useful full-attention KV payload is 64 KiB/token (2 × 16 layers × 4 heads ×
256 dims × 2 bytes). Measured MLX peak grows about 185 KiB/token from 2k to
19.5k including capacities rounding, GDN state, and allocator overhead.
Weights hold about 8.3 GB resident with peak headroom near 13.4 GB at 19.5k.

| Context | Weights + state | Useful KV | Measured MLX peak | Fits 16 GB |
|---|---|---:|---:|---|
| 2k | ~8.4 GB | 0.1 GB | 10.43 GB | yes |
| 8k | ~8.4 GB | 0.5 GB | 11.63 GB | yes |
| 19.5k | ~8.4 GB | 1.2 GB | 13.42 GB | yes |
| ~32k (projected) | ~8.4 GB | 2.0 GB | ~14.9 GB | borderline |
| 262k | ~8.4 GB | 16.8 GB | — | no |

262K needs 16.8 GB of KV payload alone before weights, state, allocator, or
macOS. Exact-only retention cannot reach it on this machine; the honest
ceiling is about 32k. Anything beyond needs KV quantization, GDN state
precision reduction, disk offload, or recomputation, each with its own
quality gate. The 16k cache-step comparison (2 of 6 arms: decode tie at
~4.0 tok/s, identical 2,774 MB KV allocation) stays directional, not a
promotion result.

## Promotion rules

Future comparisons must freeze checkpoint, prompt IDs, template, sampler,
reasoning effort, token limits, interpreter, and source fingerprints. Use one
fresh process per arm, at least three arms per condition, and
forward/reverse/forward ordering. Avoid arbitrary warmups on the fanless Mac.

Report paired effects and the two-standard-error resolution band alongside
the raw arms. You decide what promotes based on that evidence. Exact
operations require intermediate equality and whole-run greedy digest checks.
Output-changing work also requires the sampled quality gate in
`CONTRIBUTING.md`.

We keep ternary weights, every context token, all layers, and current model
semantics for milestone 1. Vision input, changed reasoning budgets, disk
offload, KV recomputation, and speculative decoding are outside this scope.
