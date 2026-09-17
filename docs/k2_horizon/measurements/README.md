# K2-Horizon measurements

This directory supports the active
[`research.md`](../research.md); operational instructions remain in
[`handoff.md`](../handoff.md).

This is our results index for the 2026-09-16 K2-Horizon 7B memory and decode
check, plus our product profiling diagnostic and Metal diagnostic. We used
the installed Q8/G64 checkpoint, BF16 KV, one fresh child
process per arm, three forward/reverse/forward rounds (`AB, BA, AB` for the
two-condition comparisons), and 32-token arrival windows. Greedy arms required exact token
digests. The retained artifacts record the complete source/checkpoint
fingerprints, cache offsets and dtypes, MLX/process memory, physical reads,
VM counters, timings, and raw token arrays.

The operational defaults remain prefill step 512, allocator cap/clear off, and
`wired_limit` off. We made no model, precision, context-retention, or output
contract change from these measurements.

## Reproduction commands

We ran the benchmark from `/Users/gioma/Developer/MACQWEN` with the recorded
model interpreter and checkpoint alias. These are the exact argv for the
retained run shapes (the greedy seed does not affect the required digest):

```bash
cd /Users/gioma/Developer/MACQWEN

/Users/gioma/models/.venv-qwen4exp/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare baseline --fixture context-2k --horizon product --window 32 --rounds 3 --sampling greedy --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/20260916-baseline-product.jsonl

/Users/gioma/models/.venv-qwen4exp/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare baseline --fixture context-2k --horizon short --window 32 --rounds 3 --sampling greedy --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/20260916-baseline-short.jsonl

/Users/gioma/models/.venv-qwen4exp/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare prefill --fixture context-2k --horizon short --window 32 --rounds 3 --sampling greedy --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/20260916-prefill-short.jsonl

/Users/gioma/models/.venv-qwen4exp/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare prefill --fixture context-8k --horizon short --window 32 --rounds 3 --sampling greedy --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/20260916-prefill-8k.jsonl

/Users/gioma/models/.venv-qwen4exp/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare wired --fixture context-2k --horizon short --window 32 --rounds 3 --sampling greedy --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/20260916-wired-short.jsonl

/Users/gioma/models/.venv-qwen4exp/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare baseline --fixture context-2k --horizon short --window 32 --rounds 3 --sampling sampled --thinking --effort medium --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/20260916-sampled-short.jsonl
```

The older greedy control rows for the short prefill and wired runs carry a
null seed in their metadata because that harness revision did not persist the
seed field; their required greedy outputs are still digest-checked. The
explicit seed above is the reproducible command for the current harness.

## Compact final table

Memory is shown in decimal GB; KV values are useful payload / allocated bytes.
`active/peak/pool` are MLX values; footprint is the process physical
footprint. VM entries are median deltas per arm (`pagein`, `swapin`,
`swapout`). A dash means the metric is not applicable to that horizon.
Throughput shows median with arm range in parentheses where present.

| Run (raw artifact) | Prompt → output; chunk | Allocator / cache-growth / residency | Useful / allocated KV | MLX active / peak / pool; footprint | Prefill / first token | Overall / window / tail tok/s | Reads; VM deltas | Paired result; correctness |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| Product baseline ([JSONL](20260916-baseline-product.jsonl)) | 2,731 → 256; 512 | default / native 256 / off | 440,451,072 / 452,984,832 | 10.015 / 10.338 / 0.428; 10.799 | 20.810 s / 0.625 ms | 9.680 / 9.677 / 9.676 | 0.938 GB; 11,997 / 67,165 / 0 | 3/3 `e7031fcb1943` matches; accepted baseline |
| Short greedy ([JSONL](20260916-baseline-short.jsonl)) | 2,731 → 32; 512 | default / native 256 / off | 407,420,928 / 415,236,096 | 9.977 / 10.338 / 0.003; 10.335 | 23.700 s / 1.163 ms | 11.752 (9.769–11.797) / 11.865 / — | 1.927 GB; 30,745 / 163,119 / 53,352 | 3/3 `b380a1bc594c` matches; accepted control |
| Prefill 2K ([JSONL](20260916-prefill-short.jsonl)) | 2,731 → 32; 256 vs 512 | default / native 256 / off | 407,420,928 / 415,236,096 | peak 10.199 vs 10.338; footprint unchanged | candidate slower within same ~20.5 s prefill | decode -0.483%, 2SE 0.268 | 0.695 GB median; VM retained in JSONL | 6/6 digest matches; reject 256 default |
| Prefill 8K ([JSONL](20260916-prefill-8k.jsonl)) | 8,107 → 32; 256 vs 512 | default / native 256 / off | 1,200,144,384 / 1,207,959,552 | 10.770 / identical 11.087 / 0.006; 11.133 | candidate prefill -6.11%, 2SE 9.18 (slower) | decode +1.891%, 2SE 4.737 (unresolved) | 1.760 GB median; 18,297 / 169,703 / 120,246 | 6/6 digest matches; reject 256/default 512 |
| Wired comparison ([JSONL](20260916-wired-short.jsonl)) | 2,731 → 32; 512 | default / native 256 / off vs on | 407,420,928 / 415,236,096 | 9.977 / 10.338 / 0.003; 10.339 | 20.402 s / 1.023 ms | mean decode -0.557%, 2SE 0.527 | 0.648 GB median; 2,521 / 48,229 / 0 | 3 losses, digest identical; reject wired |
| Sampled diagnostic ([JSONL](20260916-sampled-short.jsonl)) | 2,730 → 32; 512 | default / native 256 / off; seed 7 | 407,273,472 / 415,236,096 | 9.977 / 10.338 / 0.022; 10.358 | 16.653 s / 0.519 ms | 11.682 / 11.792 / — | 0.038 GB median; 2,059 / 14,691 / 0 | 3/3 `3540215cd7bd` matches; diagnostic only |

The table rounds display values; the JSONL artifacts remain authoritative for
every byte, counter, timing, token, and full digest. Product tail means tokens
33–256. We do not apply FlashNext's expert-warmup definition to K2.

## Decisions and scope

Accepted:

- We accept the current native BF16 cache and prefill 512 control. The product
  control reproduces median 9.680 tok/s overall and 9.676 tok/s on the tail.
- We accept the benchmark's fresh-process, append-before-validation records
  and the separate sampled diagnostic. The sampled median is 11.682 tok/s on
  a 2,730-token prompt, but it is not a speed promotion or quality gate.

Rejected or not promoted:

- We reject `wired_limit`: mean decode delta -0.557%, 2SE 0.527, and three
  losses despite identical greedy tokens.
- We reject prefill 256 as a default. At 2K it loses 0.483% decode (2SE
  0.268) and leaves footprint unchanged despite a lower MLX peak; at 8K the
  11.087 GB peak is identical, prefill is slower (-6.11%, 2SE 9.18), and the
  +1.891% decode result (2SE 4.737) is unresolved. We retain 512.
- We skipped allocator-cap and post-generation-clear comparisons because the
  short control's allocator pool was only about 3.1 MB, so there was no
  material premise. No allocator default is changed.
- We skipped `cache-step` and `reserve(n)`: 2K cache padding was about 7.8 MB
  and no trace established costly growth. The native cache remains unchanged.
- We skipped Task 5 compute optimizations: no trace established a candidate,
  the product already reproduced about 9.68 tok/s, and no speculative kernel
  was justified.
- We skipped 16K and 32K. At 8K, prefill 256 provided no peak benefit, and
  this machine has 16 GB unified memory; at 32K the useful BF16 KV floor plus
  weights leaves no comfortable headroom.

The initial failed or incomplete harness artifacts are not retained
conclusions and are omitted from this index. We did not use them to compute
the table or decisions. We did not run model inference while preparing this
documentation handoff.

## Product profiling diagnostic

We retain six fresh-process arms in [`k2-profile-product.jsonl`](k2-profile-product.jsonl), with
AB/BA/AB ordering, 2,731 prompt tokens, and 256 greedy output tokens.
All arms preserve the product digest. We use the project `.venv` interpreter.
The following command reproduces the run configuration:

```bash
.venv/bin/python -m models.k2_horizon.bench --checkpoint k2 --compare profile --fixture context-2k --horizon product --window 32 --rounds 3 --sampling greedy --prefill-step-size 512 --seed 7 --jsonl docs/k2_horizon/measurements/k2-profile-product.jsonl
```

The original invocation calls `run_comparison` with these same options.
Each profile record references separate prefill and decode `.pstats` files.
We retain all six files in this directory:
`k2-profile-product-round-1-cprofile-*.pstats`,
`k2-profile-product-round-2-cprofile-*.pstats`, and
`k2-profile-product-round-3-cprofile-*.pstats` (prefill and decode each).
Our control median is 11.314 tok/s. The paired +0.620% difference measures
instrumentation effects, not a runtime improvement. Text decoding and protocol
translation account for 0.100% of pooled profiled decode time.
We do not compare these rates directly with the historical baseline.

Our separate [`k2-metal-diagnostic.jsonl`](k2-metal-diagnostic.jsonl) records one unprofiled backend arm
under Xcode-beta's Metal System Trace. We used this command:

```bash
/Applications/Xcode-beta.app/Contents/Developer/usr/bin/xctrace record --template "Metal System Trace" --output "/var/folders/58/86027c3j0tvc1gkm43k28c180000gn/T/opencode/k2-product-metal.trace" --time-limit 100s --launch -- /Users/gioma/Developer/MACQWEN/.venv/bin/python -m models.k2_horizon.bench --child --checkpoint k2 --arm-id metal-diagnostic --condition control --round 0 --options-json '{}' --record docs/k2_horizon/measurements/k2-metal-diagnostic.jsonl --fixture context-2k --horizon 256 --window 32 --effort medium --sampling greedy --prefill-step-size 512 --seed 7
```

The trace starts at `2026-09-16T20:22:28.578-03:00` and ends after 57.636162 seconds.
We exported `metal-gpu-intervals` to `gpu-intervals.xml` in that same temporary directory.
These temporary artifacts are not part of the repository and may expire.
The export contains 13,553 intervals attributed to Python PID 62240.
We merge overlapping intervals before measuring coverage.

| Snapshot window | Wall seconds | Active GPU interval union | Coverage |
|---|---:|---:|---:|
| Generation start to prefill | 25.627784 | 16.672142 | 65.055% |
| Prefill to decode end | 23.530236 | 23.515268 | 99.936% |

These command-buffer intervals do not identify individual kernel costs or
separate arithmetic from memory stalls. Snapshot alignment uses wall-clock
correlation. We have no attributed cause for the 8.956-second prefill gap.
No kernel candidate or performance change earns promotion from this diagnostic.

## Retained constraints

We keep the installed Q8/G64 weights, BF16 KV, every context token, all
layers, and current model semantics. Lower weight/KV/activation quantization,
alternate checkpoints, layer skipping, sliding windows, context truncation,
vocabulary shortlists, changed reasoning budgets, disk offload, KV
recomputation, and speculative decoding remain out of scope.

Any future comparison must keep the same checkpoint and imported source
fingerprints, freeze prompt/template/sampler/effort/token limits, use fresh
processes, preserve forward/reverse/forward ordering, avoid arbitrary thermal
warmups, and publish each raw arm before validation. It must report paired
effects with the two-standard-error band and exact greedy token digests.
Different matrix shapes can change rounding, so candidate operations require
intermediate and whole-run equality. We never infer physical memory from a
sliced view's `nbytes`, sum overlapping counters, add per-layer `mx.eval`,
clear/synchronize per token, or claim a memory reduction below the useful KV
payload floor. Generator lookahead, stop rewind, cache offsets, cleanup,
reset, session replay, cached appends, JSON tool calls, and long-context
retrieval remain required correctness checks if we reopen implementation work.
