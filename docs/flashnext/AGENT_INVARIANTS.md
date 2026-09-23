# Flash-Next agent invariants

This file records hard constraints for agents working on Flash-Next. Read it
with [`handoff.md`](handoff.md) and [`research.md`](research.md) before code
changes or experiments.
The shared measurement rules are in [`../measurement-standard.md`](../measurement-standard.md).

## Runtime invariants

- Keep the canonical control path unchanged unless we explicitly approve
  promotion with recorded evidence.
- Make every optimization opt-in until it passes the promotion rules below.
  Our requested G64 default promotion on 2026-09-17 is a specific exception:
  we accept short speed and exact-digest evidence and skip long-turn quality
  validation at our request. We do not treat that gate as passed or general
  quality as proven. This exception does not authorize other promotions.
- `chat.sh` defaults to the MLX-backed Metal runtime. For REAP Q4/G64, we use
  the G64 Metal executor by default with `FLASHNEXT_METAL_G64=1`.
- Preserve generic MLX Q4/G64 as our comparison reference and explicit rollback
  with `FLASHNEXT_METAL_G64=0`. The 60-slot Q4/G32 Frontier 8A profiles remain
  checkpoint-specific historical controls, not the current REAP runtime.
- Keep G64 packed residency and Frontier 8B disabled by default. G64 slabs
  stay off with `FLASHNEXT_SLAB_G64=0`. Streamed expert-major records
  (`FLASHNEXT_STREAM_PACK=1`, chunk 2) are part of the exact opt-in bundle we
  made the chat default on 2026-09-23.
- For normal Vontra Q4/G32 chat, keep a frozen per-checkpoint slab profile as
  the requested default. Live pin history still updates for routing; use
  `FLASHNEXT_SLAB_PROFILE=rolling` for the previous slab behavior.
- The exact opt-in bundle is the chat default since 2026-09-23 at our request:
  streamed embedding, compiled injections and norm, cached norm gain, QoS
  user-interactive, parallel n-gram prefill, both QSA flags, overlap off,
  read-ahead off, 8 I/O workers and stream-pack. Since 2026-09-23 the
  chat also defaults to one sync per decode layer, compiled GDN q/k
  normalization and gated norm, and decode n-gram rows on the read pool
  (+6.6% inside an 8.0% band over three pairs, identical digest). With the slab on it measured
  +2.4% inside a 1.0% two-SE band over three pairs, identical digest; the list
  is in `settings/launch.py`. Any member rolls back with an explicit environment
  value at launch. The existing QSA allocation guard remains active.
- Preserve the exact token digest. Any digest change rejects the optimization.
- Preserve BF16 rounding boundaries. A small numerical difference is not an
  acceptable quality result.
- Do not add `mx.eval` or warmup work to the decode path without a controlled
  measurement and explicit approval.
- Do not merge buffers only to reduce object count. Buffer layout can change
  graph dependencies, alignment, and storage concurrency.
- Do not increase worker or thread counts without a controlled measurement.

Every disabled diagnostic feature must reduce to a branch on the decode hot
path. It must allocate no closures, dictionaries, timers, counters, strings,
callbacks, or context managers when disabled.

## Interpretation invariants

- Treat measured submission-to-worker-start delay as queue residence. It does
  not identify worker saturation, storage limits, queue scheduling, GIL or lock
  contention, or any combination of them.
- Treat the historical 211.79 ms/token value as prefill-contaminated. It is not
  a decode-only queue-residence result. Use only corrected post-prefill counters.
- Do not describe the 211.79 ms/token result as Python queue-lock or GIL
  thrashing. Worker-side non-read overhead was only 2.95 ms/token.
- Treat logical route hits as a weak proxy for physical-I/O savings. Slab
  selection must use physical-miss evidence when that evidence is available.
- Do not call 60 slots mathematically optimal. The 56/60/64 sweep selected 60
  as an engineering default, but did not resolve a meaningful rate difference.
- Do not attribute a performance cliff to compressor or VM pressure without
  direct compressor, pageout, reclaim, or swap measurements.
- Do not rank separate same-boot boundary probes by absolute time when their
  physical-read states differ.
- Do not infer heat, application interference, or immunity from drift from an
  elapsed-time correlation. Interleaving mitigates order bias but does not
  establish a cause or eliminate confounding.
- Limit negative performance conclusions to the checkpoint, hardware, and
  controls actually measured. A rejected sweep is not a universal
  impossibility result.

## Experiment invariants

- Do not require a reboot. Use file-cache purge and the VM quiescence gate only
  as explicit diagnostics. They are disabled by default.
- Never stop `dynamic_pager`, delete swapfiles, or invoke `memory_pressure` as
  benchmark preparation.
- Performance work uses greedy decoding and exact digests. We perform
  final quality evaluation through `chat.sh` with sampling and `xhigh` effort.
- The future paired G64 quality comparison predeclares seeds 7, 19, and 73,
  alternates G64-off and G64-on order, and keeps slabs and stream-pack off.
  Require completed outputs and score the complete SketchUp `.rb` artifact
  blind to arm labels. Seed 42 is a known regression case, not a
  representative quality seed. An interrupted generation is an incomplete
  gate, not a quality failure.

Every optimization experiment must:

1. Keep the current control path available.
2. Use interleaved arms with reversed ordering.
3. Keep destinations, reads, requested bytes, worker count, slab capacity, and
   quality settings fixed when testing task topology.
4. Pass the exact token-digest gate.
5. Report the resolution band and the raw arms; you decide what the
   evidence supports.
6. Report physical MB/token and active RAM.
7. Report generation and tail rate.
8. Report queue residence, positioned-read wall time, layer completion time,
   and total I/O wait when testing I/O scheduling.

Add new terminal tests as separate `models/flashnext/tests/cases/case_*.py`
files. Put benchmark scripts in `models/flashnext/tests/bench/`, unit tests in
`models/flashnext/tests/unit/`, and send every output through
`macqwen.results`; see [docs/testing.md](../testing.md).
Do not add case-specific commands to the terminal. Each runnable file must
provide its explanation, proposal reason, controls, metrics, source, and
executable script through the test plugin API.

Preserve the established 32-token arm when comparing against the current
Flash-Next baseline. A longer horizon changes route locality, page-cache state,
memory pressure, and GPU utilization. Treat any duration change as a separate
experiment with its own baseline. Report token-level or block-level metrics
when the harness provides them.

Treat every 256-token product test as a separate horizon. It never replaces
or redefines either 32-token control.
Run one long answer arm per selected path. Long runs are directional validation
only. They never supply promotion statistics or repeat the short-arm protocol.

Keep the losing full `physical-miss` replacement unavailable. The guarded
`physical-miss-hybrid` must preserve the canonical 48-slot core, pass its
20 MB/token offline premise gate, and change only the 12 extension slots.

Wide resolution bands describe noisy conditions, not absent effects. Report
the evidence with its band and let us decide enablement, promotion, and
defaults.

## Current next-work order

1. Keep G64 Metal as our requested default and generic MLX as the explicit
   rollback and comparison reference.
2. Leave long and cached behavior validation open until we authorize it.
3. The QSA flags are on as part of the bundle. Complete the cached
   tool-result QSA/prefill check and a 16K-32K context comparison only with
   our permission.
4. Retain the predeclared-seed, completed-output G64 quality protocol as pending
   work. Our default promotion does not clear this gate or authorize a run.
5. Revisit expert-pin counts only with a new premise and unchanged routes
   or arithmetic. The Vontra 8-pin policy is set.
6. Evaluate last-row-only prefill below 2,048 tokens as a TTFT/memory change.
7. Profile the current reference path before deciding whether another
   checkpoint-specific G64 kernel experiment is worthwhile.

The completed worker and topology experiments remain historical evidence. Do
not prescribe or repeat them without a new, checkpoint-specific premise and
our explicit approval.

Do not run benchmarks, sweeps, or tests without our explicit permission.
