# Changelog

## MACQWEN 0.5.0 - 2026-09-23

### Added

- Support the installed `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`
  checkpoint: alias `vontra-mtp`, content-identity policy with 8 resident
  experts, frozen slab profiles with a rolling opt-in, and boot-stable
  slab identity.
- Add GPU keep-warm as the chat default with a
  `/config model gpu-keepwarm` toggle, after resolving the GPU
  clock-collapse finding.
- Promote the exact opt-in bundle (streamed embedding, compiled glue and
  norm, cached norm gain, interactive QoS, parallel n-gram prefill, QSA
  flags, stream-pack) to the chat default with a resolved +2.4%
  measurement.
- Add an opt-in locked expert working set (`FLASHNEXT_EXPERT_LOCK_GB`,
  off, no measured gain) and retained benchmarks for long states,
  prefill I/O, route traces, cache simulation, footprints, and slab drift.

### Changed

- Make Flash-Next the documented primary runtime: the README leads with
  SSD-streamed MoE on the installed `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`
  checkpoint (alias `vontra-mtp`, 22 shards, 105.4 GiB) with its download,
  current chat defaults, and current 128-token results; K2-Horizon, Bonsai-2,
  and Qwen3.8-27B stay supported as secondary runtimes. oQ4 and REAP-288
  numbers remain as labeled history.
- Remove the `fast` routing profile and its `--fast` flag. It dropped
  experts without renormalizing and never passed the quality gate; use
  `fast-quality` for approximate routing with quality recovery.
- Move every model's tests into `models/<model>/tests/`: checkpoint-free
  tests in `unit/`, benchmark scripts and their helpers in `bench/`, and
  test-terminal cards in `cases/`. Shared unit tests move to `macqwen/tests/`,
  and Qwen3.8-27B offline scripts move to `models/qwen27b/tools/`.
- Discover cases for every model with one loader in `macqwen/testsuite`, and
  remove the four per-model catalogs and the old Flash-Next terminal copy.
- Write every test and benchmark run to `results/<model>/<stamp>-<test-id>/`
  with `record.jsonl` and `output.log`. `macqwen.results` refuses other
  destinations, and a policy test enforces the layout. Existing records move
  from `docs/<model>/measurements/` to `results/<model>/`.
- Start benchmarks as modules, so commands no longer depend on the working
  directory.

### Fixed

- Stop `fast-quality` decode from crashing with "top_k cannot exceed
  expert_count" when adaptive top-k pads dropped slots with a repeated expert.
- Apply a case's `environment()` hook in the shared terminal. Cases such as
  `chunk-after-workers` lost their selected worker count after the terminal
  migration.
- Stop writing benchmark output to `/tmp`, to the working directory, or to
  fixed paths.

### Tests

- Pass 425 checkpoint-free shared tests, 417 Flash-Next tests, 22 Qwen27B
  tests, 58 K2-Horizon tests, and 175 Bonsai-2 tests (5 skipped), plus
  Python bytecode compilation.

## MACQWEN 0.4.6 - 2026-09-20

### Added

- Add `--prepared-qmm on|off` startup control for the Bonsai-2 default,
  with source reporting in `/config model` and fail-closed live writes.
- Add K2-Horizon and Qwen27B hostile-marker regression coverage, Qwen27B
  snapshot fingerprint tests, and 31 synthetic Qwen27B unit tests.
- Add server tests for SSE failure events, Anthropic stop mapping, model
  cards, and scalar coercion tables.

### Changed

- Route every chat turn through one persistent worker thread, so the
  live cache stays valid across turns with incremental prefill only.
- Return backend-derived display names and context windows from
  `/v1/models`, with 32768 as fallback.
- Centralize scalar coercion: strict integers, explicit boolean set,
  no silent truncation or false coercion.
- Run timed-out commands in their own process group and kill the whole
  group; report code-check gaps as unavailable instead of passing.

### Fixed

- Fix K2-Horizon and Qwen27B user/tool content handling so pasted
  control markers stay literal text and never become structural tokens.
- Fix SSE streaming to emit in-stream error events instead of a second
  HTTP response after headers are sent.
- Map Anthropic stop reasons correctly: tool calls give `tool_use`,
  budget exhaustion gives `max_tokens`, natural stops give `end_turn`.
- Bind K2 saved sessions and Qwen27B cache snapshots to
  checkpoint/tokenizer/runtime fingerprints and validate before
  touching live state.

### Tests

- Pass 374 checkpoint-free shared tests, 172 Bonsai-2 tests, 58
  K2-Horizon tests, 352 Flash-Next tests (2 skipped), and 60 Qwen27B
  tests, plus Python bytecode compilation.

## MACQWEN 0.4.5 - 2026-09-20

### Added

- Promote prepared QMM metadata to the Bonsai-2 default: one-time FP32
  scales/biases preparation for the 401 non-embedding packed projections
  (about 763 MB residency). The bounded 1k screen shows decode
  `+21.07% ±2.24` with matching provenance, complete coverage, and
  matching greedy digests; prefill is unresolved at `+1.28% ±3.52`.
  `prepared_qmm_metadata=False` rolls back to the stock path.
- Add opt-in Bonsai-2 diagnostics with execution-path gates: Q2/G128
  prefill MPP, fused Q4 attention with tiling controls, and exact
  speculative-oracle ceiling probes. None is promoted.
- Add bounded live-test cases for the question-only and context-1k
  prepared-metadata screens through the project-owned test terminal.

### Changed

- Run every chat turn on one persistent worker thread, so the live
  cache stays valid across turns with incremental prefill only.
- Make prefill cancellation responsive across the Flash-Next, Qwen27B,
  K2-Horizon, and Bonsai-2 runtimes.
- Cut the agent-profile system prompt from 943 to 450 Bonsai tokens
  while keeping the api_docs-first rule and the measured environment
  facts.

### Fixed

- Fix Bonsai-2 provenance self-match: confirmed optional-file absence
  no longer reports unknown, namespace packages resolve to their root
  with the real binary fingerprint, and provenance failures increment
  validation counts, fail the comparison, and return a nonzero exit.
- Fix the second-turn chat crash (`There is no Stream(gpu, 3) in
  current thread`) caused by evaluating one thread's cache state on a
  fresh worker thread.

### Tests

- Pass 169 checkpoint-free Bonsai-2 tests and 303 checkpoint-free
  shared tests, plus Python bytecode compilation.

## MACQWEN 0.4.3 - 2026-09-19

### Added

- Add an isolated `models/bonsai2` package for Bonsai-2 ternary 27B
  checkpoint discovery, resident text-only generation, sessions, protocol
  adaptation, settings, benchmark harness, and tests.
- Add the `b2` checkpoint alias and `--model bonsai2` launcher path.
- Add an opt-in fused sign+Hadamard+downcast Metal kernel for Bonsai-2
  ternary projections. It is bit-exact on production shapes and keeps
  greedy digests on all comparison arms. Promoted to default on prefill
  evidence; `fused_fwht=False` rolls back to the stock path.
- Add one project-owned live-test terminal with runtime case discovery and
  append-only JSONL measurement records.

### Changed

- Cap the Bonsai-2 chat allocator cache at 256 MB. The six-arm comparison
  shows pool memory falling from about 774 MB to about 300 MB with
  identical decode rates and matching digests at 2k context.
- Use the managed repository `.venv` for all runtimes by default.
- Make the REAP-288 G64 Metal executor the Flash-Next default under the
  documented short exact-digest promotion exception; long-turn quality stays
  unverified.

## MACQWEN 0.4.2 - 2026-09-17

### Added

- Add an isolated `models/k2_horizon` package for K2-Horizon 7B MLX
  checkpoint discovery, resident generation, sessions, protocol adaptation,
  settings, and tests.
- Add the `k2` checkpoint alias and `--model k2-horizon` launcher path.

### Tests

- Pass 251 checkpoint-free MACQWEN tests, 352 checkpoint-free FlashNext
  tests (2 skipped), 41 K2-Horizon tests, and 1 Qwen27B test, plus
  Python bytecode compilation and whitespace validation.

## MACQWEN 0.4.1 - 2026-09-16

### Added

- Add checkpoint compatibility for the REAP-288 export, including Q4/G64
  experts, Q4/G32 n-gram weights, and both supported n-gram naming schemes.
- Add opt-in G64 Metal execution, slab tooling, diagnostics, and
  checkpoint-bound pin profiles without enabling experimental paths in normal
  chat.
- Add explicit sampling seeds and stronger benchmark artifacts with checkpoint,
  runtime-source, harness, token, and failed-arm evidence.

### Changed

- Default `chat.sh` to the MLX-backed Metal runtime while keeping REAP expert
  execution on generic MLX and preserving an explicit runtime opt-out.
- Keep G64 execution, G64 slabs, and expert-major stream packing disabled by
  default until the retained quality and performance gates pass.
- Bound REAP `xhigh` reasoning inside the existing generation allowance instead
  of adding tokens to the shared total.
- Correct the FlashNext documentation to distinguish historical Q4/G32
  controls, REAP sanity observations, incomplete quality attempts, and future
  work.

### Fixed

- Guard QSA mask allocation for large projected masks and correct G64 streamed
  down-projection offsets.
- Keep custom QMV helpers compatible with Metal compilers that reject
  unqualified reference parameters.
- Infer mixed quantization layouts from tensor metadata consistently across the
  loader, MTP path, runtime checks, and pin profiles.
- Preserve failed benchmark arms before validation, report paired regressions
  correctly, and bind ordinary reference artifacts to their checkpoint and
  source identity.
- Enforce shared pin budgets and reject stale or mismatched prewarm history
  before pinning rows.

### Tests

- Pass 251 checkpoint-free MACQWEN tests and 323 checkpoint-free FlashNext
  tests, plus Python bytecode compilation and whitespace validation.

## MACQWEN 0.4.0 - 2026-09-04

### Added

- Add the SIMD Q4/G32 Metal MoE executor with fused down-projection and router
  score combination.
- Add the native Objective-C++ Metal scheduler and DMA contention probes.
- Add file-backed, page-aligned expert slabs with skew-aware allocation and
  direct expert-major Metal addressing.
- Add the FlashNext settings registry and source reporting for runtime options.
- Add physical-miss, I/O scheduling, score-sync, chat-parity, long-answer,
  slab, kernel, and wired-limit benchmark tools.
- Add the interactive FlashNext research test catalog with runnable cases and
  result tracking.

### Changed

- Use 60 skew-selected slots, file-backed slabs, chunk 2, and the current
  fused runtime controls as the engineering profile.
- Strengthen benchmark controls with reversed ordering, private pin snapshots,
  physical-byte checks, digest checks, and corrected post-prefill counters.
- Replace strict kernel bit identity as the only kernel criterion with layer
  tolerance, fixed-route performance, and end-to-end trajectory checks.
- Rewrite the FlashNext research record and reference documentation for the
  current runtime, measurements, commands, and third-party acknowledgements.

### Fixed

- Remove stale runtime setting paths and expose current options through the
  shared settings registry.
- Correct fresh-arm isolation, slab cache state, read accounting, and source
  fingerprint checks in the production harness.
- Keep experimental Frontier 8B, physical-miss, expert-major stream, and
  cache-aware paths disabled when their controls remain unresolved.

### Measured

- The current 60-slot Frontier 8A profile with Up-QMV/SwiGLU measures 3.08
  tok/s generation, 3.00 tok/s tail, and 279.7 MB/token.
- The corrected 16-worker decode-only control measures 3.13 tok/s generation,
  3.04 tok/s tail, and 262.0 MB/token.
- The isolated custom Q4 kernel is bit-identical and 3.5% to 4.4% faster on
  production shapes. The complete-model comparison remains unresolved inside a
  7.8% resolution band.

## MACQWEN 0.3.6 - 2026-09-03

### Fixed

- Pause live agent-tool progress before approval prompts and resume it only
  when approved execution starts.
- Prevent progress rendering from racing with terminal cleanup during
  interruption.
- Keep unterminated tool-call protocol chunks hidden across streamed chunks.
- Keep agent approval and denial states from restarting progress incorrectly.

### Tests

- Add regression coverage for agent approval and denial, progress cleanup,
  streamed tool-call boundaries, and UI progress state.

## MACQWEN 0.3.5 - 2026-09-02

### Added

- Add FlashNext diagnostics for context decay, evaluation cost, glue work,
  layer locality, GPU utilization, Metal spans, and Xcode GPU captures.
- Add FlashNext tests for imports, expert-cache behavior, GPU reporting, and
  prefill contracts.
- Add MLX Metal source notes, trace graphics, residual plots, and session
  records to the documentation.

### Changed

- Make FlashNext prefill and speculative paths use the shared prefill contract.
- Remove the unreachable row-level expert LRU path from the active reader.
- Mark IOKit GPU utilization as a relative signal. Use Metal trace for absolute
  GPU timing.
- Update FlashNext research documentation with Session 4 findings, revised
  work fronts, and current issue links.

### Fixed

- Keep JSON benchmark runs within their answer-token limit instead of adding a
  saved reasoning budget to the decode ceiling.
- Restore the FlashNext tokenizer import after limiting the Transformers
  advisory environment to the import itself.
- Correct host-window and evaluation-cost reports that converted relative IOKit
  utilization into false GPU milliseconds.
- Keep session, CLI, loader, and benchmark behavior covered by the new tests.

## MACQWEN 0.3.4 - 2026-09-02

### Added

- Add the six-command shared chat surface: `/help`, `/new`, `/session`, `/config`, `/status`, and `/quit`.
- Add grouped session and configuration commands while keeping the existing commands as compatibility aliases.
- Add `/help all` with profile-aware compatibility command details.
- Add effective-value and inactive-setting reporting for Flash-Next routing.
- Add shared command metadata for web-terminal shortcut buttons.
- Add branch synchronization warnings when a checkout does not contain `origin/main`.
- Add Flash-Next host-window, layer-split, routed-gather, compile, and production comparison benchmarks.
- Add optional bit-exact compiled router, normalization, gate, renormalization, and combine chains.

### Changed

- Replace the Plain profile prompt with: `Answer precisely. Never invent an API, a name, or a result. Ask for what you need, and say when you are unsure.`
- Make shared-buffer chunk 2 the default for the `pread`, `preadv`, and `resident` read modes.
- Keep `fast` and `fast-quality` on `shared_mmap` until their shared-buffer behavior is measured.
- Instrument expert-read futures and host intervals so GPU, SSD, and host-only time can be separated.
- Keep `/settings`, `/thinking`, `/save`, `/load`, `/reset`, and other former commands accepted through compatibility routing.
- Add oQ3-MTP to the checkpoint notice and record the external oQ4-MTP repetition warning. MTP stays disabled in production.
- Update the shared chat, Flash-Next, Qwen27B, and release documentation with the current command surface, measurements, and issue links.

### Fixed

- Show routing values that apply to the active profile and mark inactive or ignored settings.
- Keep web-terminal shortcuts aligned with the shared command table.
- Preserve token IDs while changing the Flash-Next read-buffer layout.
- Keep Plain mode free of tools while allowing it to request missing information.

### Measured

- `buffer-chunk2` reaches 2.83 gen, 2.70 tail, and 457.7 MB/token in the clean-boot 12-arm comparison. It wins 10 of 12 pairs, uses fewer bytes in 10 of 12, and preserves token IDs.
- Host-only bookkeeping contributes 4.16 ms/token after bulk movement is excluded. The routed `gather_qmm` path runs at 92.2 to 92.4 GB/s and costs 13 to 16 ms/token.
- The compiled path remains bit-exact but changes the complete result by -0.6%. Its approximately 1 ms/token saving stays diagnostic.
- Device duty is 172.4 ms GPU, 234.8 ms drive, 42.5 ms host-only, and 18.8 ms unaccounted per token in the final cold run.
- Whole decoder layers cost 255.93 ms/token while individually timed neural components total 41.00 ms/token. The remaining attribution stays open.
- The external oQ4-MTP report records repetition loops that reach `max_tokens` and truncate tool calls. It does not measure standard oQ4 or this runtime.

### Quality

- The recorded oQ3-MTP SketchUp failure called `Sketchup::Face#extrude` across 34,203 reasoning characters. oQ4 questioned `pushpull` before settling on the correct API.
- Keep oQ4 as the quality baseline. REAP-288 remains gated until its reasoning-loop report passes the complete quality check.

## MACQWEN 0.3.3 - 2026-09-01

### Added

- Add Qwen's recommended thinking-mode sampler and the `/sampling` command.
- Keep benchmark backends greedy and make `run_benchmark` enforce that mode.
- Show sampling, effort, thinking, and token budgets in `/settings`.
- Add `high` reasoning effort between `medium` and `xhigh`.
- Give `high` a validation instruction with an explicit stopping rule.

### Measured

- Cache-aware measured 2.91 tok/s against 2.73 for exact routing.
- Cache-aware read 360.4 MB per token against 430.0 MB for exact routing.
- The 6.5 percent gain exceeded the 0.6 percent resolution band in all six pairs.
- An off-process draft at realistic duty reduced target speed by 1.5 percent.
- A speculative batch of two read 808 MB per token against 390 MB for decode.
- Cache-aware failed the greedy `xhigh` trajectory gate through repetition.
- Exact-quality completed the same SketchUp task and remains the default.
- Telegraphic thinking comes from `xhigh` effort instead of routing.
- oQ4 produced a working SketchUp extension in the checkpoint gate.
- oQ3-MTP produced invalid extensions at `low` and `xhigh` effort.
- oQ4 remains the quality baseline for tasks that require real API names.

### Fixed

- Limit cache-aware swaps to batches of four rows.
- Stop cache-aware routing from slowing large prefill batches.
- Add a row cap to `set_route_observer` for normal chat prefill.
- Keep uncapped observation available for benchmarks.

### Changed

- Rewrite the complete documentation set for shorter, direct instructions.
- Keep performance results near the top of the README.
- Replace the model-specific contributor guide with `CONTRIBUTING.md`.
- Replace personal absolute paths with portable home and workspace defaults.

### Removed

- Remove the non-portable `chat-swap.sh` convenience launcher.

## MACQWEN 0.3.2 - 2026-09-01

### Distribution

- Use the public `macqwen-releases` URL in Quick Start and package metadata.
- Add a Python package, the `macqwen` command, and `macqwen setup`.
- Discover project environments and compatible checkpoints without personal paths.
- Add Apple Silicon CI for shared and Flash-Next tests.
- Publish version tags as GitHub Releases after CI passes.
- Add dependency checks and repository identity tests.

## MACQWEN 0.3.1 - 2026-08-31

### Changed

- Add `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` as the current research
  checkpoint.
- Resolve Flash-Next checkpoints from a saved choice, `--checkpoint`, the
  environment, or the sole compatible local installation.
- Keep oQ4 available as a supported checkpoint choice.
- Keep MTP disabled in the production backend until local tests support it.

### Added

- Add `cache-aware` as a live Flash-Next routing profile. It starts from
  `exact-quality` and can select a near-equal resident expert instead of a
  cold selected expert.
- Add `/settings swap-epsilon VALUE`, `--cache-aware`, and `--swap-epsilon`.
- Enable residency tracking when cache-aware routing starts or becomes active
  through `/settings`.

### Performance

- Measure cache-aware routing at 2.79 tok/s against 2.54 for exact routing in
  one hot interleaved run. Paired arms improved by 8.3 percent. Seven of eight
  pairs were faster.
- Reduce physical reads by 16.8 percent in that run, from 417.8 to 347.6 MB
  per token. All eight cache-aware arms read fewer bytes.
- Keep 2.713 tok/s as the exact-quality production baseline. The cache-aware
  run used a different machine state.

### Quality

- Keep `exact-quality` as the default. The small cache-aware factual gate lost
  no correct answer. A later long-context comparison favored exact-quality.
- Show this quality warning when `cache-aware` is active under `/settings`.

### Correctness

- Add `qsa_chunk.py` and `prefill.py` to the session engine fingerprint. Both
  change what lands in the cache, so a session saved before a change to either
  restored against different code and the payload checksum could not tell.
- Fix the streamed n-gram row cache. A request larger than the cache evicted
  rows the same call still had to return, raising `KeyError`.
- Honour `FLASHNEXT_READ` on the fused path, which hardcoded `pread`.
- Report the measured acceptance rate for fused block decoding instead of a
  constant zero.
- Close the tensor store in tests and between benchmark conditions. Each store
  maps 22 shards, and a leaked map held page-cache references into the next
  measurement.

### Removed

- Delete 1,161 lines of unreachable code: `block_fusion.py` and its test, which
  nothing imported, and four benchmarks whose draft checkpoints no longer exist.

### Measurement

- Add `bench_production.py`, the standard protocol for any published number. It
  alternates conditions, stops when the median settles rather than after a fixed
  count, reports median and range with physical MB per token, flags a run whose
  rate falls with elapsed time as thermal, and refuses to report a comparison
  whose setting never changed.
- Add `diskio.py` for physical bytes read, which distinguishes a cold run from
  a warm one.
- Print a token digest from the pinned-tail benchmark so two runs can be
  compared without diffing prose.

## MACQWEN 0.3.0 - 2026-08-31

This release expands FlashNext into MACQWEN, a shared low-memory LLM runtime
for Apple Silicon.

### Performance highlights

Measurements use the reference M4 Mac with 16 GB of unified memory.

- Measure 2.713 tok/s for a complete `exact-quality` decode and 2.650 for the
  pinned tail, over ten kept arms at 390 MB of physical reads per token. An
  older harness that reloads the model per arm measured a 2.59 tok/s tail mean
  across ten arms, range 2.42 to 2.73; the gap between the two is page-cache
  state rather than a code change.
- Measure prefill across prompt lengths. It is faster than decode because it
  amortises: sixteen times the tokens cost 1.94 times the bytes, and the drive
  rate falls from 1.40 to 0.82 GB/s as the rate rises from 8.72 to 41.97
  tok/s. Decode sustains 1.06 GB/s, more than a 2048-token prefill.
- Record 2.83 tok/s only as the mean of two warmup-eight arms in a separate
  four-arm sweep. Do not use it as a production baseline.
- Keep complete-chat and pinned-tail rates separate because warmup and expert
  pinning occur before the tail timer.
- Reduce the first plain `hi` prompt from 253 tokens to 47 tokens, an 81
  percent reduction.
- Reduce the measured first `hi` turn from 33.5 seconds to 21.2 seconds.
- Reuse server cache state and prefill only 18 new tokens on a measured
  follow-up request.
- Run terminal animation outside model generation, so word fading does not
  reduce decode throughput.
- Pin up to 32 recurring experts within a configurable 6 GB memory budget.
- Measure 4.66 GB of pinned expert data and about 4.19 GB of baseline resident
  memory.
- Reach a synthetic 5.33 tok/s read ceiling with fixed routes and resident
  expert reads. This benchmark does not generate a real model reply.

### Project structure

- Reframe the project as MACQWEN, a low-memory Apple Silicon LLM runtime.
- Keep Qwen3.8-Flash-Next as the primary supported runtime.
- Add Qwen3.8-27B as a research runtime.
- Move model runtimes under `models/`.
- Move shared chat code under `macqwen/`.
- Add one launcher for all supported models and profiles.
- Organize active documentation into briefs, research records, and handoffs.

### Shared chat

- Add one chat interface for all models.
- Add plain chat and repository-tool profiles.
- Add one shared command table.
- Add all nine tools to the repository-tool profile.
- Require approval before tools change files or system state by default.
- Add persistent preferences across chat runs.
- Add live `/settings` controls for Flash-Next runtime values.
- Add persistent `/animate on|off` control for the word fade.
- Add `/stream on|off` and `/effort low|medium|xhigh`.
- Add live profile changes with conversation reset and toolbox rebuild.
- Store one editable system prompt file for each profile.
- Migrate the legacy saved system prompt into the active profile file.
- Set the repository-tool answer allowance to 2,048 tokens.
- Keep the plain answer allowance at 4,096 tokens.
- Add a separate 512-token reasoning capacity and `/think-budget` control.
- Add secure terminal management for Tavily and Context7 API keys.
- Store managed API keys outside the repository with private permissions.
- Hide compatibility aliases from `/help` while keeping them accepted.

### Terminal experience

- Stream complete words without showing partial token fragments.
- Fade complete words through four shades of grey.
- Run word animation on an output worker.
- Add the `MACQWEN_FADE_MS` animation budget.
- Keep prefill animation outside redirected output.
- Align help columns and show routing and thinking status on the ready line.
- Handle Ctrl+C during Flash-Next prefill without a traceback.
- Reset an interrupted conversation while keeping the model loaded.
- Keep machine environment data out of plain chat prompts.
- Start visible reasoning directly below the input prompt.
- Keep one blank line between reasoning and answer text.
- Give visible reasoning a darker version of the word fade.
- Replace predicted prefill progress with backend work callbacks.
- Render the fill bar with `█` and `░` cells and no border characters.
- Show pending tool activity while the model generates hidden protocol.
- Replace raw tool protocol with action descriptions.
- Remove success icons, emoji, and green status words from tool results.
- Measure tool execution separately from its minimum display interval.

### Token and performance reporting

- Count generated model token IDs instead of rendered words.
- Calculate decode speed from model time.
- Exclude terminal writes and word animation from decode timing.
- Use monotonic performance counters for interval measurements.
- Report prompt tokens, generated tokens, context size, and complete turn time.
- Aggregate all model segments in the final tool-request statistics.
- Show measured live prefill rate and remaining time.
- Add regression tests that keep terminal speed independent from reported
  model speed.

### Local API server

- Add local OpenAI-compatible and Anthropic-compatible APIs.
- Stream server responses and process one request at a time.
- Reset model state when server mode starts and stops.
- Reuse model cache state when a request extends the cached conversation.
- Prefill only new request tokens when cache reuse succeeds.
- Rebuild the cache when a request diverges from the cached conversation.
- Require exact assistant-turn replay for cache reuse.
- Add optional Bearer and `x-api-key` authentication.
- Block browser origins unless the operator explicitly allows them.
- Add configurable CORS origin rules.

### Flash-Next runtime

- Keep `exact-quality` as the default routing profile.
- Pin recurring experts after the eight-token warmup.
- Expose pinned expert counts and memory use.
- Add a configurable expert pinning budget.
- Let `FLASHNEXT_READ` reach supported chat profiles.
- Preserve `shared_mmap` for the measured `fast` profiles.
- Add the experimental `resident` read mode and keep it disabled by default.
- Measure `resident` as 25 percent faster without drive traffic, with no
  production gain under normal drive pressure.
- Add bounded reads for large prompts.
- Release temporary allocator state after standard, speculative, and fused
  prefill.
- Fix streamed decoding for characters that span multiple tokens.
- Fix settings that were previously discarded.
- Load saved sessions across compatible profile schema revisions.
- Enforce the 262,144-token context boundary.
- Translate Flash-Next saved-session validation errors into English.
- Report a clear error when `fused-quality` has no draft model.
- Keep `fused-quality`, speculative decoding, and MTP as research-only paths.
- Confirm that longer routing warmup and more than 32 pinned experts do not
  improve production throughput.

### Qwen3.8-27B research runtime

- Integrate the Qwen3.8-27B V4 runtime with the shared chat.
- Add shared session and generation statistics.
- Add server cache reuse and append-only conversation handling.
- Exclude output callback delays from generation timing.
- Keep the runtime research-only and require a compatible local V4 checkpoint.

### Documentation and security

- Add a repository security policy and third-party notices.
- Document supported models and their support status.
- Document retained measurements and their conditions.
- Document the tested Flash-Next checkpoint and verification steps.
- Document routing quality differences.
- Document local API access and browser-origin restrictions.
- Document persistent settings and session storage.
- Record unsuccessful optimization experiments to prevent repeated work.

### Compatibility notes

- The old `flashnext/` package layout changes to the shared MACQWEN layout.
- Use `./chat.sh` as the main launcher.
- Use `./chat.sh --model flashnext` to select Flash-Next explicitly.
- Model weights remain separate from the source repository.
- The tested public checkpoint remains
  `Vontra/Qwen3.8-Flash-Next-MLX-oQ4`.
- Existing Flash-Next sessions remain supported when their runtime profiles
  are compatible.
- A legacy `think_budget` value of `0` now selects the 512-token default.
- Use `/think-budget off` or `--think-budget -1` to disable extra capacity.

## Flash-Next 0.2.1

- Preserve complete multiline terminal pastes.
- Require Enter after a paste before generation starts.
- Enforce the model's 262,144-token context boundary.

## Flash-Next 0.2.0

- Make `--exact-quality` the default routing profile.
- Add the animated prefill indicator.
- Persist thinking and reply-limit settings.
- Add English session-command aliases.

## Flash-Next 0.1.0

- Add SSD-streamed MoE experts and n-gram tables.
- Add adaptive routing profiles.
- Add persistent exact sessions.
- Add optional MTP research code.
