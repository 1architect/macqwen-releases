# Changelog

## MACQWEN 0.3.2 - 2026-09-01

### Distribution

- Use the public `macqwen-releases` URL in Quick Start and package metadata.
- Add a Python package, the `macqwen` command, and `macqwen setup`.
- Discover project environments and compatible checkpoints without user-specific paths.
- Add Apple Silicon CI for shared and Flash-Next tests.
- Publish future version tags as GitHub Releases after CI passes.
- Add dependency update checks and repository identity regression tests.

## MACQWEN 0.3.1 - 2026-08-31

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
  no correct answer, but we preferred exact-quality in a later
  long-context comparison.
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
- Add `plain` and repository `agent` profiles.
- Add one shared command table.
- Add all nine repository tools to the agent profile.
- Require approval before tools change files or system state by default.
- Add persistent preferences across chat runs.
- Add live `/settings` controls for Flash-Next runtime values.
- Add persistent `/animate on|off` control for the word fade.
- Add `/stream on|off` and `/effort low|medium|xhigh`.
- Add `/profile plain|agent` with conversation reset and toolbox rebuild.
- Store one editable system prompt file for each profile.
- Migrate the legacy saved system prompt into the active profile file.
- Raise the default agent answer allowance to 2,048 tokens.
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
- Aggregate all model segments in the final agent statistics.
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
