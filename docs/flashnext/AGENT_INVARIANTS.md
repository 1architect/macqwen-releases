# FlashNext Agent Invariants

This file records hard constraints for agents working on FlashNext. Read it
with [`handoff.md`](handoff.md) and [`research.md`](research.md) before code
changes or experiments.

## Runtime invariants

- Keep the canonical control path unchanged unless a measured result promotes
  an opt-in path.
- Make every optimization opt-in until it passes the promotion rules below.
- Preserve two controls. The historical control is 60 skew slots with
  Frontier 8A and Up-QMV-to-SwiGLU off. The current runtime control uses the
  same settings with Up-QMV-to-SwiGLU on, following the user's decision.
- Keep Frontier 8B and streamed expert-major records disabled by default.
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

## Experiment invariants

- Do not require a reboot. Use file-cache purge and the VM quiescence gate only
  as explicit diagnostics. They are disabled by default.
- Never stop `dynamic_pager`, delete swapfiles, or invoke `memory_pressure` as
  benchmark preparation.
- Performance work uses greedy decoding and exact digests. The user performs
  final quality evaluation through `chat.sh` with sampling and `xhigh` effort.

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
claim such a result as resolved. Report the evidence and let the user decide
enablement, promotion, and defaults.

## Current next-work order

1. With user approval, rerun the corrected decode-only Section 17 control.
2. Then sweep worker-pool width using the Section 17
   instrumentation. Use the same reads and report queue residence,
   positioned-read time, total I/O wait, physical MB/token, and generation.
3. Test one task per expert only if the worker sweep supports a scheduling or
   queue hypothesis. Keep all other topology variables fixed.
4. Calibrate the 48-slot core, then run the offline physical-miss ceiling gate.
   Do not run the hybrid model comparison below 20 MB/token predicted saving.
5. Keep both 60-slot Frontier 8A controls frozen. Historical has Up fusion
   off. Current runtime has Up fusion on.

Do not run benchmarks, sweeps, or tests without explicit user permission.
