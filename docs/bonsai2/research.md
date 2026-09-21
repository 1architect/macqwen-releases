# Bonsai-2 ternary 27B research record

This is our active record of Bonsai-2 measurements, rejected ideas, and
design decisions. Current operation belongs in [`handoff.md`](handoff.md).
Exact commands, the compact results table, and retained JSONL arms are indexed
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

The rows below are historical reference artifacts from the earlier harness
and runtime. We keep them for comparison, but do not treat them as current
promotion evidence until they are rerun on a clean revision.

| Run | Prompt → output | Decode median | Prefill | MLX active / peak |
|---|---|---:|---|---|
| Short 2k | 3,282 → 32 | 5.7 tok/s | ~124 s | 8.28 / 10.43 GB |
| Product 2k | 3,282 → 256 | 5.5 tok/s | ~124 s | 8.31 / 10.43 GB |
| 16k probe (cache-step file, 2 arms) | 19,458 → 256 | 4.0 tok/s | ~1,240 s | — / 13.42 GB |

### Current low-context verification

On 2026-09-19 we ran three fresh full-precision control children with the
current benchmark structure and the bundled runtime. The `context-2k`
fixture currently tokenizes to 3,283 prompt tokens and generated 32 tokens
per arm. Decode measured 7.375, 7.339, and 7.328 tok/s, for a 7.339 tok/s
median and a 0.65% arm spread. All arms completed, matched greedy digest
`108ee966ba929`, and reached about 10.43 GB peak MLX memory. Prefill was
83.0–87.2 seconds, with an 84.3-second median.

This is a current diagnostic rate, not a sustained product baseline. System
swap activity occurred, and schema-1 physical-read accounting spans prefill
plus decode. We therefore promoted only
the decode-rate observation from this run. Raw evidence is retained at
[`20260919-low-context-short-check.jsonl`](measurements/20260919-low-context-short-check.jsonl).

The historical artifacts show decode falling from 5.5 tok/s at 2k to 4.0
tok/s at 19.5k context. Their prefill runs about 26 tok/s at 2k and 16 tok/s
at 19.5k. Round-1 arms run slower throughout; weight paging from the page
cache dominates long-prefill timing more than chunk compute does. The current
short check above is not a replacement for these sustained comparisons.

### Prefill chunk sweep (512 vs 1024 vs 2048, 8k fixture, 7 of 9 arms)

Prefill seconds are flat across chunk sizes within machine noise, while peak
memory climbs steeply: 11.63 GB at 512, 13.0 GB at 1024, 15.47 GB at 2048.
Chunk 2048 nearly exhausts the 16 GB machine for no speed gain, so 512 stays
the default. Digests match across all three sizes.

### Allocator, wired limit, post-generation clear (2k product, 6 arms each)

- Allocator cap 256 MB: pool drops from ~774 MB to ~300 MB with identical
  decode medians (5.56 vs 5.57 tok/s) and matching digests. The cap remains
  the chat default from this historical 2k evidence; current sustained and
  16k confirmation runs are still required.
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
hold it. We spent no further machine time here.

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

### KV quantization to 4-bit (opt-in, digest-equal)

The same six-arm shape with 4-bit groups keeps digest `108ee966ba92` on
all arms. KV allocation falls from 593 MB to 225 MB at 2k plus 32 tokens.
Paired decode effects are +10.0%, −9.8%, +6.0%: unresolved inside machine
noise, same as 8-bit. The validator initially rejected the mlx-vlm
quantized class; recognition now covers both stacks. Quality validation
stays manual.

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
The full six-arm comparison keeps all digests equal with paired effects of −29%, +6.3%, and +5.1%, dominated by an anomalous fast round-1 control seen across files. Unresolved; the memo stays opt-in.

### Stop retention (kept, with a recorded boundary difference)

Retained `<|im_end|>` continuations do not match replay references
token-for-token: the first generated token flips between the two turn
boundary markers while the following prose is identical, deterministically
on both sides. Prefill-path and decode-path batching round differently at
the boundary race. The prose equality is verified by inspection; the
trajectory gate stays manual. We keep replay recovery whenever cancellation
leaves the cache invariant broken; a manual stop with a valid live cache
continues into the next user turn without replaying the tape.

## 2026-09-19 — Steps 1-2: sampler-forced closure and answer replay

Post-hoc forced-close substitution corrupted GDN state: generate_step
consumes each yielded token into cache before yielding the next, so the
tape recorded </think> while the cache held an unrelated token. The
answer-budget cutoff had the same shape: break before a stop leaves one
lookahead token consumed but never taped.

Fix: `_ForcingSampler` wraps the decode sampler and emits the close token
at the think-budget boundary, so the forced token is genuinely consumed
and tape matches cache. Phase switch keys on `forced_last` (exact id
match), keeping natural `</think>` decode-sniffing for model closes.
`end_thinking()` disarms on natural close. Answer-cap break now marks
replay.

Narrow documented race: the one-ahead lookahead can sample a stale forced
close after observing a natural close, doubling </think>. Cache and tape
stay consistent; one extra close token enters the transcript.

Tests: fakes now route candidates through the passed sampler (mirroring
generate_step); new `_ForcingSampler` unit tests; answer-cap test asserts
replay. Full bonsai2 suite 67 tests OK, macqwen 253 OK. Live two-turn
smoke (`7*8=56` correct, finish stop both turns) on M4.

## 2026-09-19 — Step 3: turn joins match the template

`Conversation._separator` skipped the newline after the close when
generated text ended with newlines, producing
`<|im_end|><|im_start|>` where the template (chat_template.jinja line 109:
every message ends `<|im_end|>\n`) and Flash-Next always emit
`<|im_end|>\n<|im_start|>`. Content trailing newlines belong to the
message, not the join. When this side appends the close itself
(truncated turn) the newline is now unconditional; the already-closed
case keeps its check. Verified against the real template: assistant
content `56\n\n` renders `56<|im_end|>\n<|im_start|>user`. Shared by k2
and bonsai2; k2 suite still green (41 tests).

## 2026-09-19 — Step 4: canonical FWHT hook composition

Hook result depended on construction order: fused-after-share silently
dropped the share layer, and `BONSAI2_FUSED_FWHT` / `BONSAI2_SHARE_FWHT`
leaked through the environment into later backends. `apply_runtime_hooks`
rebuilds canonically (restore to stock, fused innermost, share outermost),
owns both env vars in both directions, and treats repeat construction as
a verified no-op keyed on module identity plus flags. Backend delegates
fully and verifies signs before applying. New 2x2 matrix test plus a
composition-switch test; bonsai2 suite 87 tests OK.

## 2026-09-19 — Step 5: native-XML-only tool protocol

The translator no longer converts JSON object payloads: native XML
(`<function=name>`) is the only accepted tool-call syntax. Block-end
scanning is structural (a `</tool_call>` candidate must follow a complete
function element) instead of JSON-parsing, so value-embedded closes no
longer truncate. Schema-known calls re-render through the shared parser
with value-aware delimiter escaping; unknown blocks pass through raw so
hallucinated calls stay visible; non-call blocks drop. EOF recovery
synthesizes only structure (never content): truncated JSON and
unterminated values stay dropped. Hostile round-trip (`</tool_call>`,
`</function>`, `</parameter>` in values) verified. Residual: a
value-embedded closer followed by structural-looking text can still
misfire; that is the model's escaping duty.

## 2026-09-19 — Step 6: required params and session budgets

`run_agent` fails closed before dispatch when required parameters are
missing (`REQUIRED_PARAMS`), naming the missing keys so the model can
retry; previously the tool ran with absent arguments. Sessions now persist
`_interactive_budgets` with strict shape validation (bools rejected as
token counts), so a restored conversation keeps its reasoning contract
instead of inheriting a later chat's budgets. Macqwen suite 257 OK.
Live two-turn smoke after all steps: `17*23=391`, `7*8=56`, both stop.

## 2026-09-19 — Review reconciliation batch

An external re-review checked a stale snapshot and re-raised items already
fixed on this branch (sampler-forced close, newline join, canonical hook
composition, native-only protocol, dispatch-level required params). Five
remaining gaps were valid and are now closed:

- Answer cap breaks after accepting its last token instead of pulling one
  more and replaying. Saves a forward and a full replay per capped turn;
  the one-ahead invariant is now a code comment plus a test asserting
  every pulled token is taped.
- `append_user` / `append_tool_results` route content through
  `build_user_encoder`. Marker-free text keeps byte-identical joint
  encoding; pasted `</think>`, `<|im_end|>`, `</tool_call>` split at
  marker boundaries and never become control tokens. Verified against the
  real checkpoint tokenizer: joint encoding emits the `</think>` id, safe
  encoding does not, decoded text identical.
- The fused hook captures its own stock module instead of reading the
  `_STOCK_FWHT` global; hook restore verifies checkpoint provenance like
  the install path. Both pinned by tests.
- Session fingerprint hashes `tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja`, and bundled runtime
  sources in addition to `config.json`. Old sessions mismatch cleanly.
- Removed the dead `gemv` constructor switch and its no-op bench arm.
  `gemv_kernel.py` and its tests stay as research artifacts.

Suites: bonsai2 90 OK, macqwen 261 OK, k2 41 OK. Live two-turn smoke:
`17*23=391`, `7*8=56`, both stop.

## Status: what stays and what does not

Stays on by default: fused FWHT kernel, allocator cap at 256 MB in chat,
chat sampler over top-k survivors.
Stays opt-in: 8-bit and 4-bit KV with digest-equal short evidence,
shared-transform memo, post-generation cache clear, fused FWHT rollback
via `fused_fwht=False`.
Stays off: stop retention behind `retain_stop=False`, wired limit,
custom GEMV kernel, prefill chunk above 512, cache step above 256.
Shipped infrastructure: text-only strict loading with sharded support,
runtime entry-point checks, benchmark runtime fingerprints, repaired
cache-growth helper, smoke screening path, tg128/tg256 horizons with p95
and ms-per-token fields.
Closed without building or promoting: ANE offload, full FWHT-matmul
fusion, PLD, MTP and small-M dispatch, greedy logsumexp skip, mx.compile,
lm_head prefill skip.
Open: sustained-baseline revalidation, the 16k allocator confirmation, the
full shared-transform comparison, stop-retention continuation checks, and
8-bit KV quality validation.

### Q2/G128 GEMV experiment (rejected for default)

A custom M=1 kernel calling the wheel's `qmv_fast_impl` directly is
bit-exact on gate, down, and head shapes. Isolated, it beats stock by 34%
on the wide gate projection and ties on the down projection. The six-arm
tg128 comparison keeps digest `0c739afabdec` on all arms but measures
−14.7%, +0.9%, −1.1% paired: no reproducible full decode gain. The module
stays as an opt-in diagnostic. Lesson repeated: isolated matmul wins do
not survive the complete runtime on this hardware.

### Closed gates: D3, D4, D9, D10, D11

- D3 fused single-token GDN: recurrence is 3.6% of the decode token,
  below the 5% implementation bar in the mission. Closed without building.
- D4 decode norm plus rotation fusion: launch-only savings against the
  documented absorption pattern. Closed without building.
- D9 MTP: no Bonsai-compatible draft weights exist locally. Deferred until
  exact target performance stabilizes and weights appear.
- D10 small-M dispatch: no consumer without MTP or PLD. Deferred.
- D11 sampling fast path: already shipped as the top-k survivor path.

### Runtime ownership (current recommendation)
Keep the native MLX backend and import ideas selectively, as done
throughout this record. Adopting mlx-serve as the engine is rejected on
evidence: decode ties GGUF at matched context, the prefill gap sits inside
one kernel family at 2.3 versus 4 TFLOPS, and every portable mlx-serve idea
either measured small or duplicated landed work.

### PLD block simulation (rejected, acceptance near zero)

Static n-gram overlap overstated the opportunity: 35% of output tokens
repeat context, but block-level simulation over the same transcript shows
72 proposed drafts with zero first-token hits, projecting 1.00x target
forwards. The model rarely continues a repeated 4-gram the same way, so
there is nothing to verify. No implementation. The pure draft utilities in
`models/bonsai2/pld.py` remain for reuse if a richer draft source appears.

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
We keep replay recovery whenever cancellation leaves the cache invariant
broken; a valid live cache can continue after a manual stop.

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

### Allocator cap at 16k (directional, 3 of 6 arms)

Control decodes 4.29 tok/s with a 2,890 MB pool; capped arms decode 3.82
and 3.92 tok/s with ~270 MB pools and matching digests. The capped arms
match historical 16k levels while the control arm repeats the fast-first
pattern, so no regression is established either way. The default still
rests on the complete 2k evidence.

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
the raw arms. We decide what promotes based on that evidence. Exact
operations require intermediate equality and whole-run greedy digest checks.
Output-changing work also requires the sampled quality gate in
`CONTRIBUTING.md`.

### Long-context arm authorization

Long-context arms are conditional research work, not automatic follow-ups.
We run an 8k, 16k, or other materially long arm only when it is relevant to
the current research question and explicit authorization was given for that
run. If either condition is missing, we do not launch or wait on the arm;
we keep the shorter evidence and record the deferral. If an authorized arm
is interrupted, we retain its append-only raw records as incomplete evidence
and do not restart it without a new explicit authorization.

We keep ternary weights, every context token, all layers, and current model
semantics for milestone 1. Vision input, changed reasoning budgets, disk
offload, KV recomputation, and speculative decoding are outside this scope.

## 2026-09-19 — Boundary and protocol regression closure

We closedd the remaining correctness gaps from the previous handoff without
changing model defaults. The forcing sampler now inspects a naturally sampled
`</think>` before the next sampler call, so a lookahead immediately before the
think-budget boundary cannot add a second close. The backend routes token
observation through the wrapper, and the regression covers one close, the full
answer allowance, and an aligned tape with no replay marker.

Server content substitution now protects `tool` history as well as user and
system content. If a template drops or repeats a sentinel, we fail the
request closed instead of feeding placeholders or unprotected content into
the model. Tests cover complete multi-result tool histories and both malformed
sentinel cases.

The shared tool parser now keeps quoted and short-form string payload bytes,
leaving only structural boundary newlines and typed scalar conversion to
normalize whitespace. Whole-body protocol-tag stripping is gone; the Bonsai
translator-to-parser hostile-content tests continue to preserve literal tags.

The cached-tool benchmark fixture passes its explicit thinking mode through
tool-result framing and its test verifies the generated-cache state before the
append. Greedy digest validation chooses its reference only from a clean first
arm, so a failed raw control cannot poison later filtering. Bonsai and K2
protocol doubles now accept and assert the propagated thinking setting.

The shared suite passed locally under the available Python 3.12 MLX runtime
(298 tests), the complete Bonsai suite passes 113 tests, and compile checks
pass after installing the declared `mlx-vlm==0.6.17` dependency and its
required Pillow runtime. The current low-context diagnostic is recorded above;
sustained and long-context validation remain outstanding, so no promotion
claim is made from the diagnostic run.

## 2026-09-20 — Packed-Q2 and fused-Q4 probe benchmark status

The completed fixture contained 3,283 prompt tokens and 32 generated tokens.
The paired fused-attention run kept tracing off in both arms and matched the
complete greedy token digest in all six fresh processes. The configured fused
option lost all three decode pairs: mean throughput delta was −24.25%, with
a 5.47 percentage-point two-standard-error band. It therefore remains
opt-in and is not promoted. These older records contain no execution-path
counters: their empty `attention_events` arrays prove only that tracing was
off, not that the fused kernel ran. We do not reconstruct that provenance.
New runs record per-backend selection and fallback counters and reject arms
that do not meet the explicit fused-coverage gate. The retained records are
[`20260920-015510-q4-attention-fused-2k.jsonl`](measurements/20260920-015510-q4-attention-fused-2k.jsonl)
and the raw arm log
[`20260920-015510-q4-attention-fused-2k-arms.jsonl`](measurements/20260920-015510-q4-attention-fused-2k-arms.jsonl).

Neither experimental kernel has established general numerical equivalence.
Small FP32 Q4 checks differed from stock by up to about 1.7e-6. A nonzero
synthetic Q2 check failed exactness with a maximum absolute difference of
0.03125. We keep the intermediate-equality and complete greedy-digest gates
unchanged; neither kernel is promoted or used as a production default.

An 8k run was relevant to the long-context question and explicitly
authorized, but was interrupted after its first stock arm while the fused
arm was running, so it has no completed comparison. Its started and partial
arm records are retained at
[`20260920-021144-q4-attention-fused-8k.jsonl`](measurements/20260920-021144-q4-attention-fused-8k.jsonl)
and
[`20260920-021144-q4-attention-fused-8k-arms.jsonl`](measurements/20260920-021144-q4-attention-fused-8k-arms.jsonl).
They do not establish an 8k performance result, and we do not restart that
or any other long-context arm without explicit user authorization.

## 2026-09-20 — FlashNext-style question-only speed arm

We addeded a diagnostic `question-only` fixture to isolate the lowest-context
live workload. It reuses FlashNext's reference question,
`Explique a fotossintese em duas frases.`, with an empty system message and
no records, tools, or prior turn. Bonsai's chat framing made the resulting
prefill 23 tokens. We used three fresh control processes, greedy decoding,
32 generated tokens, and no extra warmup; with one condition there was no
paired arm order to reverse.

| Arm | Prefill (s) | Decode rate (tok/s) | Greedy digest |
|---|---:|---:|---|
| Round 1 | 1.378 | 7.226 | `cdb7ac70…295ff5` |
| Round 2 | 1.335 | 7.196 | `cdb7ac70…295ff5` |
| Round 3 | 1.229 | 7.173 | `cdb7ac70…295ff5` |

The median was 7.196 tok/s (mean 7.198; range 7.173–7.226). All arms
completed and matched the greedy digest. Each arm made 544 stock attention
calls, with zero fused attempts or selections. This is a short-workload
diagnostic, not a replacement for the 3,283-token retained baseline and not
a promotion claim. The append-only records are
[`20260920-102501-baseline-question-short.jsonl`](measurements/20260920-102501-baseline-question-short.jsonl)
and its raw arms
[`20260920-102501-baseline-question-short-arms.jsonl`](measurements/20260920-102501-baseline-question-short-arms.jsonl).

## 2026-09-20 — One-time QMM metadata preparation probe

We integrated an opt-in runtime candidate that prepares the existing FP16
Q2/G128 scales and biases as FP32 once per resident model. Packed `uint32`
weights, Hadamard transforms, MLX's stock quantized-matmul arithmetic, and
embedding metadata remain unchanged. The candidate prepared 401 non-embedding
projections and added 799,948,800 bytes (about 0.745 GiB) of persistent
metadata; preparation itself took 0.134–0.267 seconds in the question-only
screen. Stock and candidate calls were recorded separately, and both completed
candidate arms made 13,634 prepared FP32 calls with zero unsupported calls.

The question-only screen used 23 rendered prompt tokens, greedy 32-token
output, two reverse-interleaved rounds, fresh child processes, and matching
complete digests. Candidate decode was 9.298 and 9.290 tok/s against stock
7.586 and 7.332 tok/s: mean paired improvement +24.64% with a ±4.13 percentage
point two-standard-error band. This is screening evidence only; it is not the
three-arm promotion gate. Question-only prefill was 1.184 and 1.212 seconds
against 1.338 and 1.199 seconds: +5.27% mean with a ±12.60 point band, so it
was unresolved.

We then started the one authorized comparison on the existing 3,283-token
fixture and stopped it when the machine-load cost became clear. The first
complete stock arm took 135.190 seconds to prefill and the first complete
candidate arm took 254.755 seconds, a −88.44% prefill change. Their 32-token
greedy digests matched, and the candidate recorded 16,040 prepared calls;
the one-pair decode difference was +4.91% and is not a promotion result. A
second candidate arm was interrupted during decode after partial evidence was
written. The run therefore has no complete three-arm comparison. We keep the
candidate opt-in for diagnosis, do not promote it, and stop this branch rather
than hammering the fanless machine with repeats. Records are
[`20260920-121033-prepared-qmm-metadata-question.jsonl`](measurements/20260920-121033-prepared-qmm-metadata-question.jsonl),
[`20260920-121033-prepared-qmm-metadata-question-arms.jsonl`](measurements/20260920-121033-prepared-qmm-metadata-question-arms.jsonl),
[`20260920-121312-prepared-qmm-metadata-2k.jsonl`](measurements/20260920-121312-prepared-qmm-metadata-2k.jsonl),
and
[`20260920-121312-prepared-qmm-metadata-2k-arms.jsonl`](measurements/20260920-121312-prepared-qmm-metadata-2k-arms.jsonl).

The benchmark harness now streams bounded arm progress, writes raw and
partial evidence before validation, records candidate-path counters, rejects
invalid execution coverage from paired statistics, and stops scheduling after
cancellation or a decisive child/path/digest failure. The project runner
propagates interrupted outcomes and retains the partial canonical record.
These changes are diagnostic infrastructure; they do not change Bonsai
defaults or reinterpret older records that lack path counters.

## 2026-09-20 — Prefill test safety cap

The first 3,283-token QMM prefill attempt entered paging and raised yellow
memory pressure on the fanless machine. We therefore changed the runnable
prefill screens to the new `context-1k` fixture (32 records; under 1,000
rendered prompt tokens) and two reverse-interleaved rounds, four total arms.
The benchmark CLI now defaults to that bounded fixture as well. The older
3,283-token, 8K, and 16K records remain append-only historical evidence; we do
not rerun or extend them without a relevant question and explicit user
authorization. The deferred 8K fused-attention catalog entry is not runnable.

## 2026-09-20 — QMM metadata default promotion

We promoted `prepared_qmm_metadata` to default on.

Evidence on record includes question-only `+17.77%` and 1k
`+21.07% ±2.24` decode, with 1k prefill `+1.28% ±3.52`.
All checked arms show matched provenance, complete 401/401 coverage, and
matching greedy digests. The candidate adds about 763 MB residency and
raises MLX peak from 8.08 to 8.75 GB with swap activity present.

This promotion accepts the 1k evidence for the default. A 3,283-token
probe showed prefill `-12.86%` (95.29 s stock vs 107.54 s prepared,
matching digests), so the decode win stands at short context while the
long-context cost stays open.

We retain `prepared_qmm_metadata=False` as our explicit stock rollback.

## 2026-09-20 — Chat second-turn stream fix

We fixed the `There is no Stream(gpu, 3) in current thread` crash on the
second chat turn. MLX binds materialized cache state to the generating
thread's stream, and we ran each turn on a fresh worker thread. We now
run every turn on one persistent worker thread, so the live cache
stays valid and continuations never repay a full prefill. As a safety
net we also rebuild the cache from the tape when generation arrives
on a thread with no cache affinity, tracked with thread-local storage
so recycled thread idents cannot fool the check. We pinned this with a
worker-reuse test, a two-thread replay test, and a live two-turn check
with no replay and a holding cache invariant.

## 2026-09-20 — Decode host vs MLX split

We measured host time outside MLX/Metal per decode token.

We added two perf counters in our decode loop. `next_s` covers
`next(steps_iter)`, including model forward, sampler graph build, and
all MLX eval and sync waits. `host_s` covers tape append, sampler
observe, `stream_decode`, protocol feed, and callbacks, excluding
terminal emit through `DecodeTimer`. Our change adds no `mx.eval`,
no warmup, no sleep, and no cache flush. Our bench exports the trace
as `decode_trace` alongside `stop_token_sync`.

We ran three fresh baseline arms. Our fixture is `question-only`
with 23 prompt tokens and 32 greedy output tokens. Our seed is 7.
Our defaults hold, including prepared QMM metadata on. All arms
complete with digest `cdb7ac707f12`. Decode rates are 9.178, 9.388,
and 9.302 tok/s. Our record is
[`20260920-decode-outside-split.jsonl`](measurements/20260920-decode-outside-split.jsonl).

We reported steady tokens only, excluding token 1. Token 1 carries the
prefill boundary and measures 1,141 to 2,353 ms.

| Metric | Result |
|---|---|
| Steady total, median | 106.9 ms per token |
| Host post, median | 0.1207 ms per token |
| Host post, mean | 0.1245 ms per token |
| Median host share | 0.1139% of the token |
| Per-arm host medians | 0.1371, 0.1390, 0.1175 ms |

Our bench harness adds its own spikes. Resource sampling and progress
pipe added about 6.4 to 6.7 ms on every third token. We excluded those
spikes from our production claim. Our base set holds 63 tokens under
1.0 ms host time.

Our `stop_token_sync` reports 20.31 to 24.16 ms per token over 32
calls. This `.item()` waits for prior GPU sampler output inside the
model call. We counted it as Metal wait, not host compute. It lives
inside `next_s`.

Our prior `cProfile` bounds total Python at about 5% of wall. At
107 ms that is about 5.4 ms per token. Our direct host is 0.12 ms of
that total. The remainder is Python graph build inside `next()`.
About 95% of the token remains in MLX/Metal execution and sync waits.

We keep this as diagnostic evidence. It changes no default and
promotes no optimization. Our `decode_trace` stays as diagnostic
infrastructure with the existing attention, QMM, Q2, FWHT, and stop
sync counters.

## 2026-09-20 — Necessary GPU work vs MLX bubbles

We split our steady 106.9 ms decode token into necessary work and
bubbles. Our scope stays question-only with no context-1k arm.

We parsed our safetensors header only, loading no weights. Our
language-model pack holds 7,673,714,688 bytes: 6.256 GiB packed
`uint32` weights plus 0.391 GiB scales, 0.391 GiB biases, 0.109 GiB
F32 signs and metadata, and 0.782 GiB F16 tables. Each decode token
touches all 401 non-embedding projections plus `lm_head`, so weight
traffic is about 7.7 GB per token. GDN state adds about 0.3 GB
read plus write per token. KV payload is 7.2 MB at this context.
Total necessary traffic is about 8.2 GB per token.

We probed our achievable device roof with a resident 180 MB `float16`
sum. Cold run measures 41.8 GB/s and warm run measures 90.7 GB/s.
At 90 GB/s, 8.2 GB needs about 91 ms. Our measured token is
106.9 ms. Necessary memory traffic therefore covers about 80 to 85%
of the token. The 15 to 20 ms remainder holds FWHT math, attention
softmax, GDN recurrence, norms, sampler, Python build, and any
barrier or fence wait.

We ruled four bubble sources near zero in steady state. Disk reads
are 507,904 bytes on warm arms with zero pageout. MLX cache holds
steady at 52 MB against 8.67 GB active. Wired limit stays off, so
residency commits nothing. Host post-processing measures 0.1207 ms
median, or 0.11% of the token. One-ahead overlap hides most Python
build behind GPU execution because GPU work exceeds host work by
about 20 to 1.

We attempted one single-token `.gputrace` capture with
`MTL_CAPTURE_ENABLED=1`. Without the flag, capture fails closed
with `Capture layer is not inserted`. With the flag, capture opens
and closes cleanly around our token 4 to 5 window. The bundle
measures 8.3 GB on disk, matching the expected single-token size.
Capture stop perturbs token 5 by 2,932 ms host, so the captured
window is diagnostic only and never a production timing. Our derived
record is
[`20260920-decode-gpu-split.jsonl`](measurements/20260920-decode-gpu-split.jsonl),
holding both attempts, token walls, counters, and the size note.
The 8 GB bundle itself stays out of our repo. Sum versus union
inspection needs Xcode GPU tools and stays open. Even if our full
15 to 20 ms remainder were bubbles, bubbles cap at about 15 to 20%
and necessary work floors at about 80%.

We ran no further capture or comparison on this question. Our
machine stays cool and our evidence stands on the header parse,
the roofline probe, the retained split arms, and one gated capture.

We recaptured one single-token bundle for Xcode analysis. It
lives at `~/Downloads/bonsai-decode-1tok.gputrace` with 8.3 GB on
disk. Our derived record is
[`20260920-decode-gpu-xcode.jsonl`](measurements/20260920-decode-gpu-xcode.jsonl)
with the same 6 greedy tokens, token walls, counters, and size note.
Token 5 carries the capture stop cost and stays diagnostic only.

## 2026-09-20 — One-token GPU census and simplification ranking

We built our one-token GPU census from the MLX side because our 8.3 GB
`.gputrace` bundle has no Xcode-free decoder. Its `capture` stream
holds 3,784 opaque ID runs across 135 IDs, but no kernel names, so we
do not quote dispatch counts from it. Our MLX-side census is exact at
the lazy-graph level. Our defaults are prepared QMM plus fused FWHT.

Our per-token census counts 401 `quantized_matmul` calls, about 401
fused FWHT launches, 16 single-row stock attention calls, and about
900 elementwise launches (norms, adds, SiLU, multiplies, splits,
reshapes, RoPE, softmax). Total dispatches run about 1,700 per token.
Encoders run about 40 to 45 per token under the 40-ops buffer cap,
consistent with our trace screenshot. Unique kernel types run 15 to
20. Weight traffic runs about 8.2 GB per token against our 90.7 GB/s
roofline probe, needing about 91 ms of our 106.9 ms wall. Barriers
separate most dependent dispatches by construction and stay required
for correctness.

Our top repeated chains are FWHT plus QMM times 401, GDN
5-projection plus conv plus norm times 48, MLP gate plus up plus
swiglu plus down times 64, residual plus norm plus projection, and
attention QKV plus norm plus RoPE plus SDPA plus out times 16.

We ranked five simplification candidates. Shared FWHT memo removes up
to 144 launches with zero math change. GDN conv specialization caps
at 8.1 ms or 4% with custom-kernel cost. QMM metadata slimming has
negative expected value because it reverses our +21% prepared win.
Swiglu, norm, and residual micro-fusion caps near 0.2 ms of traffic.
Attention-prep fusion caps near 0.5 ms at short context where KV is
7.2 MB.

We tested shared FWHT first at question-only with two reversed rounds.
Control rates are 9.440 and 8.957 tok/s against shared 9.448 and
9.479 tok/s. All digests match `cdb7ac707f12`. Paired effect is
+2.96% with a ±5.75 band, so our result is unresolved and we do not
promote it. Our record is
[`20260920-share-fwht-question.jsonl`](measurements/20260920-share-fwht-question.jsonl).

We implemented nothing further. No candidate shows a resolved positive
result, and every remaining ceiling sits at or below 4% against
custom-kernel cost. Our runtime code is unchanged.

## 2026-09-20 — FWHT contradiction, GDN finding, chain split

We resolved our FWHT contradiction from recorded counters. Shared arms
show 8,738 FWHT attempts against 13,634 on control arms. The
difference is 4,896 over 34 forwards, or exactly 144 per forward, so
the memo removes 144 launches per token as designed. QMM stays at
13,634 on both sides and attention stays at 544 calls. All digests
match `cdb7ac707f12`. Speed stays flat at +2.96% inside our ±5.75
band. Launch count is therefore not our bottleneck: 144 launches
carry about 2.6 ms of kernel work, matching our point estimate but
sitting below resolution.

We corrected our GDN ceiling. The 4% label was stale against a 190 ms
token. General-path conv costs 0.17 ms per layer, so 48 layers cost
8.16 ms against our 106.9 ms token, or 7.6%. We found the cause in
source: decode takes the compiled `_causal_conv1d_decode` branch only
when conv weights are fp16 or bf16, but our checkpoint stores all 48
`conv1d.weight` tensors as F32 with shape 10240 by 4 by 1. Every
decode token therefore runs the general `nn.Conv1d` path. The
recurrence itself already runs a fused metal kernel, so conv plus its
silu and state bookkeeping is our largest single fusion target.

We split our 401 chain with batched in-graph probes, 64 calls per
graph with one eval. FWHT costs 18 us per call in-graph, or 7.2 ms
per token. Gate-size QMM is memory-bound near 0.21 ms per call in
production weights, totaling about 88 ms. Elementwise rms plus add
costs 8.9 us per pair, or about 4 ms across our ~900 launches. Glue
between FWHT and QMM is zero launches on our prepared FP32 path.

We attributed our ~19 ms gap to FlashNext as Hadamard tax plus
linear-attention tax: FWHT 7.2 ms, GDN general conv 8.2 ms, and extra
norm and bookkeeping across 64 layers. FlashNext pays about 16 ms of
sparse MoE against our dense 401 plus GDN extras. We dropped QMM
metadata slimming per instruction and keep our +21% prepared win.

| Class | Launches/token | GPU ms/token | Share | Bytes/token | Removable ms |
|---|---|---:|---:|---|---:|
| QMM | 401 | ~88 | ~79% | ~7.7 GB | 0 without fewer bytes |
| FWHT fused | ~401 | ~7.2 | ~6% | small | ~2.6 via share memo, n.s. |
| GDN conv general | 48 | ~8.2 | ~7% | state traffic | up to ~6 with fast path or kernel |
| Elementwise | ~900 | ~4 | ~4% | ~15 MB | ~1-2 max |
| Attention | 16 | ~2 | ~2% | 7.2 MB KV | ~0.5 max |
| Sampler | 1 | ~1 | ~1% | vocab row | 0 |
| Total | ~1,767 | ~110 | ~100% | ~8.2 GB | GDN only material target |

The Xcode read showed 135.286 ms of GPU span for our token 4 to 5
window. That window holds about 1.2 to 1.3 steady tokens of GPU work
plus one-ahead lookahead, so it correctly exceeds our 106.9 ms
steady wall. The reported counters confirmed our bound: active cores near
100%, bandwidth peak 85.5 GiB/s against our 90.7 GB/s roofline probe,
and ALU utilization 46.53% at the cursor. That is a memory-bound
signature. Necessary traffic near 8.2 GB per token explains the wall,
and MLX bubbles stay capped at about 15 to 20%.

## 2026-09-20 — GDN conv probe stops the hook

We probed our GDN decode conv with synthetic shapes and no checkpoint
load. Isolated per-call timing is sync-floor dominated: general
0.359 ms against fused fp16 0.4 ms and fused F32 0.567 ms per call.
Those numbers cannot attribute production cost, so we probe 48 calls
in one graph with one eval. General wins at 1.11 ms per 48 against
1.46 ms fused F32 and 1.4 ms fused fp16, or 23.0 us against 30.4 and
29.3 us per call. Our 2 ms gate fails in the wrong direction, so we
build no hook and run no full-token arms. Numeric distance is
maxabs 0.0096 for fused F32 and 0.0054 for fp16 cast, so the digest
gate would likely reject as well. Our 8.16 ms figure measured
surrounding GDN bookkeeping or sync floors, not the conv kernel,
which costs about 1.1 ms in-graph. Our record is
[`20260920-gdn-conv-probe.jsonl`](measurements/20260920-gdn-conv-probe.jsonl).

## 2026-09-20 — GDN bookkeeping census and reconciliation

We profiled GDN bookkeeping apart from conv with synthetic shapes and
no checkpoint load, 48 calls in one graph with one eval. Concat of
65 KB costs 0.776 ms per 48. State take of 49 KB costs 0.107 ms.
Split views cost 0.087 ms. One q/k norm costs 0.663 ms, doubled to
about 1.3 ms since production runs q plus k. State write of 2 MB
fp32 costs 2.87 ms, with a similar reread next token, so the
round-trip is about 5.7 ms and is recurrence-mandated. Residual adds
cost 0.426 ms per 48, or about 1.1 ms for our 128 production adds.
Gated norm costs 0.451 ms. Our record is
[`20260920-gdn-bookkeeping.jsonl`](measurements/20260920-gdn-bookkeeping.jsonl).

We reconciled our token as QMM ~88, FWHT 7.2, conv 1.1, bookkeeping
traffic ~2.4, state round-trip ~5.7, attention ~2, sampler ~1,
totaling about 107 ms against our 106.9 ms wall within probe error.

## 2026-09-20 — Xcode SIMD-group census corrects two claims

The reported table held 27 unique kernels by SIMD groups over our token 4 to 5
window, about 34,609 groups for 1.2 tokens. Cost percent is
unavailable, so groups ranked work, not time. Our record holds the reported
full table in
[`20260920-decode-gpu-xcode-r2.jsonl`](measurements/20260920-decode-gpu-xcode-r2.jsonl).

We corrected our first claim: `affine_qmv_fast_float_gs_128_b_2`
exists with 15,078 groups, so our prepared path already runs a
qmv-structured kernel. Our narrow kernel mirrors it, which explains
our bit-identical result. Our second correction concerns copies:
`gg1`, `g1`, `g2`, `gg2`, and two `v_copy` casts total about 2,800
groups near 8% of all groups. Every copy moves its tensor twice
with no arithmetic, so copy elimination outranks further QMM tuning.
Norms total about 2,100 groups across `rms` variants, and
`depthwise_conv_1d_float32` holds 792 groups. Gated delta holds
3,997 groups and fused FWHT holds 4,032 across widths. Small
sigmoid, multiply, add, subtract, exp, rope, gather, and conversion
kernels make up our measured elementwise tail.
No multi-ms removable chain remains: concat feeds both conv and
cache, takes write required cache, norms feed the fused kernel, and
state traffic is genuine dependency. Views stay views. Runtime code
is unchanged.

## 2026-09-20 — Copy attribution closes the family

The Cost percent read on our 178 MB attribution capture showed about 44%
as script RNG artifact (`rbitsc`, `Erfinv`, `Minimum`,
`copyuint32`, `all_reduce_sum` from an in-capture random plus sum,
`arange` from take positions), so we excluded it. Our confirmed
mapping is `gg1_copy` from scatter-style in-place strided writes,
`gather_axis` 8.25% from `take_along_axis`, Greater plus Select
from `where`, and `rmsfloat16` 2.63% from strided norm. Our record
is
[`20260920-copy-attr.jsonl`](measurements/20260920-copy-attr.jsonl).

## 2026-09-20 — Q2 weights are ternary, packing audit passes

We streamed every packed tensor once and count codes per class. Code
11 measures exactly 0.000% in all eleven projection classes, with
00, 01, and 10 near 33.6, 32.7, and 33.6%. Entropy 1.585 equals
log2(3). Only non-decode tensors outside our classes use code 11.
Our record is
[`20260920-ternary-audit.jsonl`](measurements/20260920-ternary-audit.jsonl).

Five trits per byte (3^5 = 243) encodes 1.6 bits per weight against
2.0 now. Packed traffic drops 6.40 to 5.12 GB per token, saving
1.28 GB for a ceiling near 14 ms. Gate and up alone save 0.57 GB
for a ceiling near 6.3 ms. Super-blocks of 640 weights keep
per-128 scale grouping intact because packing order never affects
scale application. Decode stays branch-free with fixed unroll of 5
per byte. The prototype stands queued for gate and up scope.

## 2026-09-20 — Ternary decoder loses to cooperative qmv

We built table and divmod base-243 decoders for gate and up with a
2.32 GB sidecar that round-trips 128 of 128 modules bit-exactly.
Both variants reach maxabs 1.4e-06 against prepared QMM, so our
math is right at single-ULP level. Both lose decisively on speed:
prepared 40.28 ms against table 143.34 ms and divmod 110.91 ms over
128 real calls. Our per-thread GEMV cannot match the wheel
simdgroup-cooperative qmv with vectorized loads and quad
accumulations; lost vectorization dwarfs our 20% traffic saving.
Divmod beats table because device-memory table loads cost more than
ALU division here. We ran no full-token arms and expanded to no
further shapes. Our 1.28 GB hypothesis is falsified as a kernel win:
traffic saved, execution lost. Code stays default-off. Our record
is
[`20260920-ternary-decoder.jsonl`](measurements/20260920-ternary-decoder.jsonl).

## 2026-09-20 — Ternary qmv-structured attempt closed permanently

We rebuilt ternary decode inside qmv traversal: 640-weight steps
with integer strides, dual-scale arithmetic selects for straddling
lanes, unrolled divmod unpack, same grid and per-group order. Our
first attempt indexed the wrong group bases and read maxabs 0.65;
we fix lane-local groups and reach maxabs 1.1e-06, so our math is
right at single-ULP level. Speed is not close: prepared 43.22 ms
against ternary 908.41 ms over 128 real calls. Divmod chains plus
selects expand ALU about 15 times, and no table variant closes a
900 ms gap, so we attempt no second redesign. Lossless ternary
compression is closed. Code stays default-off. Our record is
[`20260920-ternary-qmv.jsonl`](measurements/20260920-ternary-qmv.jsonl).

## 2026-09-20 — Exact decode saturated for this checkpoint

We marked exact Bonsai decode optimization saturated. No family
below reopens without genuinely new evidence.

| Family | Measured result | Why it failed |
|---|---|---|
| QMM tuning | At bandwidth floor ~85 GB/s | Large shapes bandwidth-bound, small shapes sub-ms ceilings |
| Narrow F16 metadata | −0.44% ±0.20 resolved | Gate/up QMM is dequant-ALU-bound, not bandwidth-bound |
| Ternary packing | −865 ms component | qmv cooperation loss dwarfs 20% traffic win |
| FWHT fusion/share | +2.96% ±5.75 n.s. | 144 launches carry ~2.6 ms, below resolution |
| GDN conv | General wins 1.11 vs 1.46 ms | Compiled path slower; 8.16 ms figure was bookkeeping |
| Copies | Mandated cache writes | gg1 appends and fills required; copiable ~0.3 ms |
| Speculative | −22% with perfect draft | 5-wide pass costs 6.2 tokens, yields at most 5 |
| Python/C++ | 0.12 ms host, ~5% Python | Nothing material outside GPU |

Our exact baseline holds 106.9 ms per token with 9.3 to 9.8 tok/s
question-only and digest `cdb7ac707f12`. Prefill runs 1.1 to 1.2 s
for 23 tokens. MLX peak holds 8.92 GB with 8.67 GB active and a
52 MB cache. Steady physical reads hold near 0.5 MB with zero
pageout. Narrow arms confirm 8.66 GB peak with 273 prepared
modules. Exact Bonsai stays our control. New representations must
report decode speed, prefill, memory, and quality, never kernel
microbenchmarks alone.

We ranked representation-level directions by bytes per token with
quality risk. Smaller ternary sidecars with qmv-structured lanes
keep 5.12 GB packed traffic with low trajectory risk but need a
cooperative decoder we could not build cheaply. Fewer resident
projections through routing sparsity cut traffic proportionally
with high trajectory risk. Lower-precision KV extends context, not
decode speed, with medium quality risk. Distilled small-draft
speculation needs a draft model that does not exist locally, with
high trajectory risk. Larger-group quantization regroups scales
with medium numerical risk. We implemented none of these in this
round.

## 2026-09-20 — Speculative verifier closed on arithmetic

We decomposed our existing K=4 oracle arms with no new runs. Seven
blocks propose 25 draft tokens and accept all 25, with 32 target
tokens and zero rollback or commit over 32 output tokens. Accepted
per pass is 3.57, or 4.57 with bonus, which clears every acceptance
bar. Verification seconds equal total seconds, so one 5-wide pass
costs about 820 ms or 6.2 stock tokens while perfect acceptance
yields at most 5. A perfect draft still loses 22%, and narrower or
wider K keeps the same per-row activation overhead against fewer or
capped bonus tokens, so no width reverses the ratio. Our 128-token
attempt already showed the oracle needs a complete transcript
upfront. We closed this idea and run no fresh arms. Reopening needs
a verifier that costs about one sweep, not new draft weights.

We closed the copy family without code change. Production `gg1`
copies are cache appends with offsets, which are mandated writes,
not removable materialization. Scalar `s_copy` fills are required
initialization. Our 27 MB non-state copiable ceiling stands, and no
single source reaches our 1 ms gate. Norms stay untouched per our
ordering rule.

## 2026-09-20 — FWHT-QMM fusion gate stops the family

We verified gate and up match exactly: 17408 by 5120, block 1024,
sign width 5120, Q2/G128, F16 scales, 64 plus 64 calls. Down,
attention, and GDN shapes differ and stay on stock fallback. Our
existing `gemv_kernel.py` calls wheel `qmv_fast_impl` and is
bit-exact on gate and down at M equals 1.

We measured the removable margin in one graph with one eval over 128
gate/up pairs. Stock FWHT plus QMM costs 43.39 ms against 42.32 ms
for QMM on precached transforms, so our FWHT marginal is 1.08 ms
for the highest-value shape. A fused kernel keeps the arithmetic and
removes only intermediate traffic near 2.5 MB plus launch overhead,
projecting under 0.5 ms against our 2 ms gate. We built no kernel
and wire nothing into decode. Our record is
[`20260920-fwht-qmm-gate.jsonl`](measurements/20260920-fwht-qmm-gate.jsonl).

We stopped this optimization family. Our exact runtime sits near its
mathematical bandwidth floor near 87.5 GB/s against a 90.7 GB/s
roof. Further large gains need a representation or model-level
change that reduces our 7.7 GB of projection traffic per token.

## 2026-09-20 — QMM bytes census and representation decision

We scanned every packed-weight tensor once, offline, with no runtime
change. Packed Q2 codes carry entropy near 1.585 of 2.0 bits with a
zstd ratio near 80% and zero zero-words. They are essentially
incompressible, so we reject GPU-friendly decompression on measured
data. Scales peak at 0.4 and biases peak at 0.4 with zero non-finite
values, so F16 narrow-load is safe everywhere.

Our per-token reads total 8.01 GB prepared: 6.40 GB packed weights,
1.60 GB FP32 metadata, and 0.0116 GB signs. Narrow-load reads F16
metadata instead, saving 0.80 GB per token for a ceiling near
8.8 ms. Reconstruction is bit-exact because F16 finite values extend
exactly to F32. Stock QMM with F16 scales is excluded as a vehicle
because it restores our measured-slow promotion path, so the
prototype needs a custom kernel. Signs narrowing saves 0.006 GB and
we ignore it.

| Candidate | Now | Proposed | Saved | Exact | Ceiling | ALU | Complexity | Call |
|---|---|---:|---|---:|---|---|---|---|
| Narrow F16 metadata kernel | 8.01 GB | 7.21 GB | 0.80 GB | yes | ~8.8 ms | convert per group | custom kernel | prototype |
| Packed recompression | 6.40 GB | 6.40 GB | 0 | n/a | 0 | n/a | n/a | reject |
| Signs F16 | 0.0116 GB | 0.0058 GB | 0.006 GB | yes | ~0.06 ms | trivial | low | ignore |
| Prepared-path removal | 8.01 GB | 7.21 GB | 0.80 GB | yes | negative | n/a | n/a | reject |

Our record is
[`20260920-qmm-bytes-census.jsonl`](measurements/20260920-qmm-bytes-census.jsonl).
We built no kernel this round. The narrow-metadata prototype clears
our 2 ms gate on paper and stands as our high-priority experiment.

## 2026-09-20 — Per-shape QMM profile corrects the bound

We timed every projection class with its real per-token tensor set in
one graph with one eval, FP32 activations with prepared metadata.
FWHT plus QMM per token runs gate 20.54, up 21.04, down 20.83,
in_qkv 10.42, in_z 10.45, out 8.40, q 6.79, k 0.97, v 1.11, o 5.52,
and head 6.36 ms, totaling 111.5 ms against our 106.9 ms wall. Less
about 7.2 ms of FWHT, QMM runs about 85 ms. Our record is
[`20260920-qmm-shape-profile.jsonl`](measurements/20260920-qmm-shape-profile.jsonl).

We refined our bound verdict per shape. Large shapes run near 80
GB/s and are bandwidth-bound with nothing removable. Small shapes
fall to 30 to 55 GB/s and are launch-bound, but their total is
about 15 ms and launch fusion across 16 calls caps below 1 ms. No
repeated dequant sequence shows a concrete 2 ms opportunity, so we
propose no kernel. The repeating unit stays one `load_vector` plus
four `qdot` calls with `scale*accum + sum*bias` per 512-block step.

## 2026-09-20 — Narrow-F16 gate/up prototype stops on dtype

We implemented our narrow branch by reusing wheel `qmv_fast_impl`
behind a shape gate for 17408 by 5120 with stock F16 metadata. Our
prepared install excludes gate/up by weight shape, and our progress
log confirms 273 prepared modules. Our component check proves
narrow output equals stock F16 output bit-identically, while our
promoted prepared FP32 path already differs from stock in low bits
with matching digests.

Our full-token arm fails closed and loud. Production Packed inputs
are FP32, but `qmv_fast_impl<half,128,2>` accepts only half input,
so our lazy Metal build fails at eval with no matching overload.
All 128 decode gate/up calls fall back or fail, and our gate
requires zero fallbacks. Casting activations to half would change
math and rebuild the promotion we remove, so we stop before the
remaining 273 projections. A float-capable tail would reopen this,
but that is a new kernel, not a reuse. Our routing, exclusion, and
counter machinery stays behind its default-off flag with passing
unit tests. Our record is
[`20260920-narrow-f16-gateup.jsonl`](measurements/20260920-narrow-f16-gateup.jsonl).

## 2026-09-20 — Narrow-F16 custom kernel loses on gate/up

We built our float-input narrow kernel in
`models/bonsai2/metal/qmv_f32_narrow.metal` with dispatch in
`gemv_kernel.py`. It loads packed weights plus F16 scales and
biases, promotes metadata exactly in registers, and follows stock
per-group order. Our component check is bit-identical against
prepared-F32 QMM across all 128 gate/up projections with maxabs
0.0. Our 128-population in-cache delta is 0.32 ms, which hides
traffic by cache residence.

Our full-token comparison resolves against us. Two reversed rounds
at question-only give control 9.773 and 9.786 tok/s against narrow
9.720 and 9.753 tok/s. All digests match `cdb7ac707f12`. Narrow
takes all 4,224 decode gate/up calls with legitimate fallbacks only
on prefill multi-row and non-gate shapes. Paired effect is −0.44%
mean with a ±0.20 band: a resolved slight regression. Saving
0.32 GB of metadata traffic buys nothing because gate/up QMM is
dequant-ALU-bound, not bandwidth-bound. We expanded to no further
shapes. Our code stays default-off. Records are
[`20260920-narrow-f16-gateup-r3.jsonl`](measurements/20260920-narrow-f16-gateup-r3.jsonl)
with the valid comparison plus earlier attempt files.
