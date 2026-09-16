# FlashNext Agent Invariants

This file records hard constraints for agents working on FlashNext. Read it
with [`handoff.md`](handoff.md) and [`research.md`](research.md) before code
changes or experiments.

## Runtime invariants

- Keep the canonical control path unchanged unless a measured result promotes
  an opt-in path.
- Make every optimization opt-in until it passes the promotion rules below.
- `chat.sh` defaults to the MLX-backed Metal runtime. For REAP Q4/G64, keep
  generic MLX expert execution unless `FLASHNEXT_METAL_G64=1` explicitly opts
  into the experimental executor.
- Preserve the generic MLX Q4/G64 path as the current REAP control. The
  60-slot Q4/G32 Frontier 8A profiles are checkpoint-specific historical
  controls and must not be presented as the current REAP runtime.
- Keep the G64 Metal executor, G64 packed residency, Frontier 8B, and streamed
  expert-major records disabled by default. G64 slabs and stream-pack stay off
  with `FLASHNEXT_SLAB_G64=0` and `FLASHNEXT_STREAM_PACK=0`.
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
5. Report the resolution band and reject small effects inside that band.
6. Report physical MB/token and active RAM.
7. Report generation and tail rate.
8. Report queue residence, positioned-read wall time, layer completion time,
   and total I/O wait when testing I/O scheduling.

Add new terminal tests as separate `models/flashnext/tests/case_*.py` files.
Do not add case-specific commands to the terminal. Each runnable file must
provide its explanation, proposal reason, controls, metrics, source, and
executable script through the test plugin API.

Preserve the established 32-token arm when comparing against the current
FlashNext baseline. A longer horizon changes route locality, page-cache state,
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

Resolution bands above 8–10% are environmentally unresolved for small-effect
decisions. A 17%, 28.6%, or 32.4% band cannot establish a 1–5% gain. Do not
claim such a result as resolved. Report the evidence and let us decide
enablement, promotion, and defaults.

## Current next-work order

1. Keep the generic MLX Q4/G64 path as the REAP control.
2. Validate long and cached behavior on that control.
3. Complete the cached tool-result QSA/prefill check, then test completed-block
   QSA key caching and the existing scatter mask separately; claim no speed
   gain until measured.
4. Run the existing paired G64 comparison under the predeclared-seed,
   completed-output protocol before considering executor promotion.
5. Compare 32 versus 8 REAP expert pins without changing routes or arithmetic.
6. Evaluate last-row-only prefill below 2,048 tokens as a TTFT/memory change.
7. Profile the current reference path before deciding whether another
   checkpoint-specific G64 kernel experiment is worthwhile.

The completed worker and topology experiments remain historical evidence. Do
not prescribe or repeat them without a new, checkpoint-specific premise and
our explicit approval.

Do not run benchmarks, sweeps, or tests without our explicit permission.
