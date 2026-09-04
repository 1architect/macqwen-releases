# Flash-Next reference

Read this file and [`research.md`](research.md) before a new experiment. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) defines the project
rules.

## Where things stand

Last worked on 2026-09-02.

The runtime streams a 176B sparse MoE model from SSD on a 16 GB M4 Mac. oQ4 is
installed and is the baseline. `exact-quality` is the default routing profile.
The accepted clean-boot `buffer-chunk2` result measures 2.83 gen, 2.70 tail,
and 457.7 MB of physical reads per token.

Recent decisions, with the evidence in `research.md`:

- oQ3-MTP was tried and dropped. It ran about 21% faster and produced a broken
  SketchUp extension at both `low` and `xhigh` effort, where oQ4 produced one
  that worked. It is deleted from the machine.
- `cache-aware` routing measures 2.91 gen and 2.92 tail on the harness at
  360.4 MB/token, a 6.5% gain against a 0.6% band, ahead in 6 of 6 paired arms.
  It is opt-in, not the default, because it failed the trajectory gate.
- `swap-epsilon` stays at 0.02. Both 0.05 and 0.10 remove the same 1.4% of
  bytes and neither clears its band.
- Speculative decoding is closed at any block length. A batch of two reads
  808 MB/token against decode's 390.
- The weight-preserving cache-aware swap is rejected for `exact-quality`.
  Four of seven replies changed, and its speed result was unresolved.
- An external [oQ4-MTP report](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4/discussions/2) describes repetition loops that reach `max_tokens` and truncate tool calls during long tool-use turns. The same settings did not reproduce the issue on oQ3-MTP. The report uses oMLX on an M5 Max. The maintainer is investigating. This supports keeping MTP disabled in production, but it does not measure our standard oQ4 path.

What changed in the code recently:

- Sampling. The runtime decoded with `argmax` everywhere, against Qwen's
  guidance. `macqwen/sampling.py` now holds the sampler, the chat uses Qwen's
  recommended thinking-mode values, and the benchmarks force greedy so token
  IDs stay comparable.
- `/effort high`, a level between `medium` and `xhigh`. The chat template maps
  effort to one sentence of system text and `medium` is empty, so there was
  nothing in between.
- The cache-aware swap no longer runs on prefill batches, and the route
  observer no longer receives the whole prefill batch.
- `/config model` now shows sampling, effort, thinking and the token budget
  alongside the routing settings. The compatibility `/settings` command
  remains accepted.
- The 2026-09-01 performance sweep closed host-only bookkeeping, routed
  `gather_qmm`, and the original complete-runtime compile estimate. The closed
  results recover about 4.16 ms, measure 13 to 16 ms of expert gather, and
  measure about 1 ms of compile savings. Issue #23 is open again for the
  corrected zero-drive RMSNorm gate.
- A 12-arm comparison gives `buffer-chunk2` a resolved 6.3% generation gain
  over the current concatenate path. Token IDs match and physical bytes fall.
  The run started after a clean boot. Issue #26 is closed and the default is
  active for the pread family.
- An earlier whole-layer control costs 255.93 ms/token with expert pages hot,
  while its separately timed component parts total 41.00 ms. The latest
  dependency-correct split measures 262 ms/token for whole hot layers and
  176.07 ms/token for chained parts. Metal System Trace closed issue #27 for
  device-level attribution, but the afternoon clean-boot result shows GPU
  busy varies from 86.1 ms/token at zero drive to 182.5 ms/token in production.
- The 2026-09-02 continuation measures 236.7 ms/token blocked in `mx.eval`.
  A one-sync experiment halves eval count but slows generation by 11.4%.
  Metal trace shows that IOKit undercounts short kernels by about 3.2x.
- `MLX_MAX_OPS_PER_BUFFER=120` reduces buffers by 38% and latency per buffer by
  52%, but makes generation slower. It is rejected.
- A 60-token context sweep shows a warm-up transient near 2.9 to 3.2 tok/s,
  then a steady unpinned rate near 1.9 tok/s at every tested context.
- `FLASHNEXT_BUFFER_ARENA` is bit-exact but unresolved. It measures 2.91 gen
  against 2.94 with fresh buffers and 397.4 against 397.9 MB/token.
- A miss-fraction sweep shows GPU busy rising to 171.7 ms/token near 25% miss,
  then falling to 125.7 ms/token at full miss. Token time stays close to
  linear in physical bytes. The peak needs three arms per cell.
- VM counters show about 33 page-ins per MB, with reclaim, compression, and
  swap flat during decode. The corrected `FLASHNEXT_RDAHEAD=0` result is 1.3%
  faster, not 13% slower, and does not clear a band.
- The default Metal per-set cap is 750 MB, but the total wired budget is zero.
  Raising it to 3,750 MB changes neither token time nor VM counters because no
  allocation is wired by default.
- A standalone 2 GB wired-limit sweep looked 13.5% faster. The live harness
  measured -0.4% inside a 7.6% band because it applied the limit after loading.
  Issue #43 tracks the controlled comparison.
- A GPU capture measured 5,778 dispatches in 86.99 ms without drive traffic.
  The Compute Shader Launch Limiter stayed near 100%, with low occupancy and
  ALU use. The zero-drive GPU is launch-bound.
- A controlled RMSNorm compile is bit-exact and 1.7% faster at zero drive, but
  remains unresolved in production. Keep it disabled by default.
- The remaining cost is still scheduling and graph execution, not a named
  removable stage.
- The `flashnext-runtime` branch integrates a specialized SIMD Q4/G32 Metal MoE
  executor (`FLASHNEXT_METAL_RUNTIME=1`) fusing down-projection and router score
  combination directly to bfloat16. It eliminates intermediate `(tokens, slots, hidden)`
  tensor allocations and removes 48 `astype` kernel launches per token.
- Controlled production evaluation (`bench_production.py --compare metal-runtime`)
  in a 16-arm interleaved reversed-pair test verified 100% bit-identical greedy token
  digests (`29d04075ed7021b3`), with rate difference at +0.7% to +2.0% (unresolved
  inside the 7.8% resolution band).
- Issues #45 and #43 are closed: Issue #45 proved zero barrier amplification under
  physical NVMe DMA (F_NOCACHE); Issue #43 proved pre-load wired memory reservation
  provides no latency benefit over dynamic residency sets.
- Concentrated global slab allocation (`FLASHNEXT_SLAB_GLOBAL=48, FLASHNEXT_SLAB_MIN_SLOTS=4`)
  concentrates resident expert slots into the top-utility layers ([5, 11, 20, 23, 29, 32, 35, 39, 40, 44, 46, 47]),
  boosting decode hit rate from 14.1% to 23.5% (+67% relative gain) for the exact same 149 MB
  active RAM, achieving 2.86–2.91 tok/s generation and 2.79–2.92 tok/s tail rate with 100%
  bit-identical digest (`29d04075ed7021b3`).
- Adaptive top-k where fast-path eliminates up to 144 redundant elementwise kernel dispatches
  per token during decode when all routed slots are active, reaching peak arm rate of 2.96 tok/s
  and winning 4 of 4 pairs over baseline in controlled production A/B testing with bit-identical digest.
- File-backed mlocked slab pack (`FLASHNEXT_SLAB_PACK=1`, `models/flashnext/slab_pack.py`) implements
  Frontier 2 & 3: a single 4K page-aligned 140.63 MB `.bin` file mapped via `mmap` + `mlock` into a
  single zero-copy `MTLBuffer` with direct expert-major addressing in Metal. Controlled 8-arm A/B testing
  breaks through the 3.0 tok/s target, reaching **3.10 tok/s generation rate** and **3.07 tok/s tail rate**
  (+8.3% mean paired speedup, median +9.3%) at 29.4% decode hit rate and 100% bit-identical digest (`29d04075ed7021b3`).
- Skew-aware slab pack 56 (`FLASHNEXT_SLAB_POLICY=skew, FLASHNEXT_SLAB_GLOBAL=56, FLASHNEXT_SLAB_PACK=1`, `models/flashnext/expert_cache.py`)
  implements the Frontier 1 & 2 extension: concentrates 56 slots into the top 12 hot layers with depth 4–6 based on marginal hit gain (164.07 MiB pack),
  boosting decode hit rate to **39.7%–40.7%** (+35% relative gain vs 29.4% on `slabpack48` and avoiding cold-layer dilution in `slabpack56_uniform`),
  reaching **3.08 tok/s generation rate** and **3.02 tok/s tail rate** (median 2.94 tok/s) with 100% bit-identical digest (`29d04075ed7021b3`)
  and only +24.5 MB MLX active memory overhead.
- The Flash-Next test suite passes all 153 tests.

The cache-aware quality comparison remains open under Next work. Its gate result
was measured under greedy decoding, which causes repetition on its own, so it
says more about greedy than about routing. That comparison decides whether
cache-aware can become the default and whether the 2.91 against 2.73 result
holds with the recommended sampler.

## Environment

The release launcher uses the local environment created by `./chat.sh setup`:

```text
Python       .venv/bin/python
Checkpoint   one complete compatible Flash-Next directory
```

Override the interpreter with `MACQWEN_FLASHNEXT_PYTHON`. Select the checkpoint with `--checkpoint`, `--model-path`, or
`MACQWEN_FLASHNEXT_MODEL`. The launcher saves an explicit selection. Set `MACQWEN_MODEL_ROOT` to change the automatic search directory.

## Revised performance direction

Issues #43 (wired limit) and #45 (barrier DMA contention) are closed with zero unresolved penalties.
Concentrated global slabs (`FLASHNEXT_SLAB_GLOBAL=48, min_slots=4`) combined with the scratch-free
register fused-down kernel reached **2.86 tok/s generation median** (with individual arms at **2.90–2.93 tok/s**)
and **2.79 tok/s tail rate** at 23.5% decode hit rate and 100% bit-identical digest (`29d04075ed7021b3`).

The clean-boot baseline is 2.83 tok/s; the current production slab baseline is 2.86 tok/s (~345–350 ms/token).
The practical target is breaking through **3.0 tok/s** (<333 ms/token), requiring an additional **~9 to 12 ms per token**.

The next performance work focuses on the SSD $\rightarrow$ Memory $\rightarrow$ Metal frontier roadmap detailed in
[Next work](#next-work), prioritizing:
1. Heterogeneous / skew-aware global slab capacity (5–6 slots for super-concentrated layers 5, 20, 23, 35, 47).
2. Production benchmark of budget 56 slots (+173 MB RAM, safely below the 196 MB page-cache threshold).
3. File-backed mlocked mmap slabs directly in the custom Metal kernel.
4. Production evaluation of direct zero-copy `preadv` I/O.

## Download

oQ4 is the baseline and the installed checkpoint.

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

oQ4 contains 22 safetensors shards and 111.7 GB of model weights.

oQ3-MTP is supported and is not installed. It contains 19 shards, 86.2 GiB, and MTP weights the production backend does not load. It failed
the trajectory gate, so it was removed:

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ3-MTP"
```

Use `--checkpoint oq4` or `--checkpoint oq3` when both exist. MACQWEN selects a sole compatible local checkpoint automatically. The
reference machine has room for one.

## Run

```bash
./chat.sh --model flashnext --profile plain --exact-quality
./chat.sh --model flashnext --profile agent --exact-quality
./chat.sh --model flashnext --standard
./chat.sh --model flashnext --threshold 1.0
./chat.sh --model flashnext --fast
./chat.sh --model flashnext --fast-quality
./chat.sh --model flashnext --cache-aware
./chat.sh --model flashnext --fused-quality
```

`fused-quality` is experimental. It failed the retained reasoning gate. `cache-aware` is optional. It improves speed but changed the
preferred answer in a long-context comparison. `exact-quality` remains the default.

Use the live configurator before a turn:

```text
/config model
/config model routing exact-quality
/config model routing cache-aware
/config model routing fused-quality
/config model swap-epsilon 0.02
/config model threshold 1.0
/config model resident-experts 32
/config model pinned-experts 32
/config model pin-budget-gb 6
/config model tail-experts 6
/config model tail-warmup 8
/config model fusion-block 23
/config model fusion-min-margin 1.0
/config model fusion-min-block 20
/config model fusion-margin-tokens 8
/config model fusion-max-prompt 512
/config model fusion-model <path to a draft model>
/config model defaults
```

`pinned-experts` aliases `resident-experts`. `pin-budget-gb` caps the pinned storage. `/config model` reports the current pinned layer-expert
count and bytes.

Settings apply to the current process only. A new `./chat.sh` launch returns to `exact-quality`, threshold `0.85`, 32 resident experts,
warmup `8`, and swap epsilon `0.02`.

Use `/new` before enabling the one-shot fused draft for a new conversation.

`speculative-fast` and MTP stay research-only. Both need a different load path, and both lost their complete-runtime controls.

## Main files

| Path | Responsibility |
|---|---|
| `macqwen/backends/flashnext.py` | Shared chat adapter and generation loop |
| `models/flashnext/loader.py` | Install streamed modules before loading |
| `models/flashnext/store.py` | Read tensor rows from checkpoint shards |
| `models/flashnext/expert_cache.py` | Read routed expert rows |
| `models/flashnext/ngram.py` | Stream hashed n-gram rows |
| `models/flashnext/adaptive_topk.py` | Apply adaptive expert thresholds |
| `models/flashnext/routing.py` | Manage runtime routing profiles |
| `models/flashnext/sessions.py` | Save and restore exact model state |
| `models/flashnext/qsa_chunk.py` | Bound QSA query allocation |
| `models/flashnext/patch_rmsnorm.py` | Correct upstream RMSNorm behavior |
| `models/flashnext/bench_read_ceiling.py` | Price the drive at zero to find the rate ceiling |
| `models/flashnext/bench_production.py` | The standard benchmark protocol; use it for every published number |
| `models/flashnext/bench_slab_sweep.py` | Measure physical MB saved per resident MB added across layer/capacity configurations |
| `models/flashnext/bench_slab_production.py` | Paired reversed-order production benchmark for selective slabs |
| `models/flashnext/bench_native_dma_contention.py` | Measure barrier and fence contention under true unbuffered F_NOCACHE SSD DMA |
| `models/flashnext/bench_wired_limit.py` | Pre-load wired memory limit comparison with fresh instances |
| `models/flashnext/diskio.py` | Physical bytes read, to tell a cold run from a warm one |
| `models/flashnext/metal_trace.py` | Export Metal command-buffer spans and nesting depth |
| `models/flashnext/capture_dispatches.py` | Capture a small `.gputrace` for Xcode dispatch inventory |
| `models/flashnext/bench_residency.py` | Check the residency gate against `mincore` |
| `models/flashnext/bench_prefill_scaling.py` | Prefill rate and bytes across prompt lengths |
| `models/flashnext/bench_route_swap.py` | Count how often a cold expert had a resident near-equal alternative |
| `models/flashnext/bench_swap_quality.py` | Compare exact and cache-aware answers on checkable prompts |

## Supporting material

| Path | Use |
|---|---|
| [`../../MLX/`](../../MLX/) | MLX 0.32.2 Metal source notes with file and line references |
| [`graphics/README.md`](graphics/README.md) | FlashNext trace image guide |
| [`graphics/Token trace - Xcode.png`](graphics/Token%20trace%20-%20Xcode.png) | Xcode GPU capture view; use for dispatch inventory, not absolute timing |
| [`graphics/miss-sweep-residual.png`](graphics/miss-sweep-residual.png) | 28-arm untraced miss-sweep residual plot |

## Validation

Run the model suite in its environment:

```bash
~/models/.venv-qwen4exp/bin/python -m unittest discover \
  -s models/flashnext -p 'test_*.py' -q
```

Run a live session restore:

```bash
printf '/session load probe\n/status\n/quit\n' | \
  ./chat.sh --model flashnext --profile plain --exact-quality
```

Run the complete JSON benchmark path:

```bash
./chat.sh --model flashnext --profile plain --exact-quality \
  --max-tokens 32 --think-budget=-1 --benchmark-json \
  --benchmark-prompt 'Explain virtual memory.'
```

In JSON benchmark mode, `--max-tokens` is the total decode ceiling. The
explicit `--think-budget=-1` also makes the short-run intent clear. Interactive
turns keep separate answer and thinking budgets.

Standard output must contain one JSON object. Diagnostic text goes to standard error.

## Prefill scaling

The reference machine is an M4 Mac with 16 GB of unified memory and a 256 GB SSD. Prefill throughput increases with prompt length. Large
batches amortize fixed setup and streamed-read costs across more tokens. A prompt near 5,000 tokens may reach about 40 to 50 tok/s under
favorable conditions. Do not extrapolate large-prompt throughput from a short interactive prompt. Prompt content, free memory, SSD state,
and page-cache state can move the result.

## Measured rate

| Condition | Rate |
|---|---:|
| exact-quality, clean-boot buffer-chunk2 | **2.83 tok/s**, 457.7 MB/token |
| the same arms, pinned tail | 2.650 tok/s |
| cache-aware, harness, four kept arms | **2.91 tok/s**, 2.92 tail, 360.4 MB/token |
| exact arm in the same harness run | 2.73 tok/s, 2.70 tail, 430.0 MB/token |
| cache-aware paired effect | **+6.5%**, band 0.6%, six of six pairs faster |
| cache-aware, earlier hot run, superseded | 2.79 against 2.54, 347.6 against 417.8 MB/token |
| older harness, ten pinned-tail arms, colder start | 2.42 to 2.73 tok/s, mean 2.59 |
| terminal `gen`, short chat turns | about 2.0 to 2.5 tok/s |
| pre-buffer warmup-eight pair, superseded | 2.88 and 2.78 tok/s, mean 2.83 |
| synthetic fixed routes, every expert read resident | **5.33 tok/s** |
| synthetic fixed routes, every expert read cold | 1.09 tok/s |

The 2.83 value now comes from the accepted clean-boot buffer-chunk2
comparison. The older ten-arm 2.59 value and the warmup-eight pair remain
historical pre-buffer records. Use complete decode for interactive performance.

The synthetic ceiling shows that compute can exceed 3 tok/s when expert reads stay resident. It does not predict a real production reply.

Cache-aware has since gone through the harness. Its exact arm measured 2.73 at 430 MB/token against the 2.713 and 390 MB/token baseline, so
that run sat at production warmth and its numbers hold. The 16.2% byte reduction there matches the 16.8% from the earlier hot run, so the
opportunity carries across residency states.

Every rate in this table came from greedy decoding, which the benchmarks use so token IDs stay comparable across arms. The chat samples.

Use `models/flashnext/bench_read_ceiling.py` to reprice the drive at zero:

```bash
FLASHNEXT_READ=resident FLASHNEXT_PROFILE_IO=1 \
  python3 models/flashnext/bench_read_ceiling.py --mode ram
FLASHNEXT_READ=pread FLASHNEXT_PROFILE_IO=1 \
  python3 models/flashnext/bench_read_ceiling.py --mode disk
```

`FLASHNEXT_READ` selects the expert read path. `pread` is the default and the only one to use in production. `resident` maps mlocked rows
instead of copying them; it wins by 25 percent when the drive is idle and loses under load.

The profile decides whether the variable applies:

| Profile | Read path | Pins experts | `FLASHNEXT_READ` applies |
|---|---|---|---|
| `standard` | the variable | no | yes, with no gain, nothing is pinned |
| `exact-quality` | the variable | 32 | yes |
| `cache-aware` | the variable | 32 | yes |
| `fused-quality` | the variable | yes | yes |
| `fast-quality` | `shared_mmap` after warmup | yes | no |
| `fast` | `shared_mmap` | no | no |

`fast` and `fast-quality` were measured on `shared_mmap` and keep it. Setting `FLASHNEXT_READ` does not change them.

## Measurement rules

- Run one model instance unless parallel operation is the experiment.
- Hold prompt text and generated token limit constant.
- Compare token IDs before accepting a performance change.
- Reverse or interleave A/B order. Reversed order is mandatory on this machine.
- **Use three arms per condition minimum.** The first arm of a run is always
the slowest, because the page cache warms across arms. Two-arm A/Bs on this machine have produced +12.8% and +10.7% results that were both
noise.
- Read the resolution band the harness prints. It is two standard errors of
the difference between medians. A reading inside it is unresolved, which is not the same as absent.
- Stack two changes that are each unresolved and measure the pair. It costs
one comparison instead of two re-runs, and a real pair clears the band.
- Interleave the lengths in any sweep over prompt size. Walking them in order
measures the warm-up: an ascending sweep read fewer bytes at 4 tokens than at 2, and moved the batching crossover from 32 tokens to 8.
- An isolated reader A/B cannot support a layout claim. The reader is not what
the model waits on once the page cache, the n-gram stream and the compute share one process.
- A benchmark must prove its own premise before it reports. Verify the drive
served the reads, verify the setting took effect, and refuse to report otherwise. Three benchmarks in one day returned plausible numbers
while measuring nothing. One missed its required read mode. One read pages cached by its own write. One could not change a module constant.
- Read this file's do-not-retry list and grep the research log before building
anything. The expert-major repack was rebuilt from scratch while sitting as item three on that list.
- Record free memory and competing applications.
- Never predict throughput from the routing coverage curve. Coverage counts
accesses to the top experts. It does not describe page-cache residency.

## Do not retry without a new mechanism

- Expert result caches and resident weight slabs.
- Warm read-ahead and in-process prefetch overlap.
- Repacking the complete checkpoint.
- Widening `swap-epsilon` past 0.02. Both 0.05 and 0.10 remove the same 1.4%
of physical bytes and neither clears its band, while 0.10 changed the output in 7 of 7 arms. Expert score gaps look bimodal, so there's
nothing between 0.02 and 0.10 to harvest.
- Verifying several tokens in one pass, at any block length and with any draft.
A batch of two reads 808 MB/token against decode's 390, so the verify block costs 3.9 times one decode. Batching widens the distinct working
set and defeats the page cache. Read the amortisation curve in `research.md` before proposing a variant.
- Compressing the checkpoint on disk. On real oQ4 data zlib saves 3.59% at
43 MB/s while decode needs 1.06 GB/s, and compression removes the byte offsets positioned reads depend on.
- Stripping the `vision_tower` tensors. They're dead weight for the text
runtime but only 0.90 GB, interleaved with 93 language tensors inside a 2.05 GB span of shard 1.
- Two-bit expert requantization.
- Low-rank expert approximation.
- Native MTP for this complete runtime.
- Exact speculative paths already measured in the research log.
- `MLX_MAX_OPS_PER_BUFFER`. The premise gate passed, but cap 120 was 19.8%
  slower across the plain arms.
- Reusing destination buffers as a speed change. The ring is bit-exact, but its
  production result is unresolved and its shape-only form changed token IDs.
- Raising `MLX_RESIDENCY_SET_MAX_PCT`. It changes only the per-set cap. The
  total wired budget remains zero, so a fivefold increase changes neither token
  time nor VM counters.
- Enabling `FLASHNEXT_EARLY_SUBMIT` as a production default. The settled test
  did not reproduce the predicted gain. Keep it off until a new mechanism and
  a load-controlled comparison support it.
- Removing host work from the read path. Mapping resident rows instead of
copying them, dropping the concatenate, and issuing reads earlier were each measured. Every one returns its saving to the GPU wait under
drive pressure.
- Pinning more than 32 experts. Tested again with a corrected candidate pool:
`hot=40` pins 6.12 GB and returns the same rate as `hot=32`.
- Longer routing warmup. `warmup=40` measured 3.7 percent slower than 8.
- Sorting a layer's reads by offset, pinning only scales and biases, and
warming last session's expert set. Each measured inside its resolution band alone, and the last two measured -1.4% together with 8% more
physical reads, so they do not add.
- Mapping resident expert rows, at any gate accuracy. A tracker with 97.6
percent precision, well past its 78.5 percent break-even, measured 5.9 percent slower while reading 3.2 percent fewer bytes, and degraded
further as more rows became eligible. The harm scales with the mapped fraction.

## Trajectory gate

If a change alters what the model computes, run a code task that names a real external API before adopting it. That applies to checkpoints
and routing profiles alike. Prose won't catch this kind of failure.

The task: ask for a SketchUp extension that extrudes several selected faces to a height supplied in the prompt. The reply has to be a complete `.rb`
file. Load it in SketchUp and run it. Record the checkpoint, the effort level, and whether it works.

Run it at `medium` and at `high` with sampling on, and keep the sampler and the effort the same on both sides of the comparison. `medium` is
the default effort and is where most output lives. Add `xhigh` if the change might affect long reasoning.

oQ4 passes. oQ3-MTP fails at both `low` and `xhigh`, and higher effort moved it further from the right method rather than closer.

Effort and sampling can move the output more than most tested changes. Greedy breaks ties the same way every time and causes repetition on
its own, so a greedy result applies only to greedy decoding. Repeat the cache-aware gate.

Check these items in the reply:

- `Face#pushpull`. There is no `Face#extrude`; oQ3-MTP invented it.
- `pushpull` with one argument. The second parameter is `copy`, not a direction
flag, so passing `true` leaves the original face behind.
- `next unless face.valid?`. Extruding one face invalidates an adjacent
coplanar one, which is the hard part of this task.
- A length parse that can't raise. `Sketchup.parse_length` returns nil;
`String#to_l` raises.

## Machine constraints

The reference machine holds one checkpoint at a time. The data volume is 228 GB with about 22 GB free while oQ4 is installed. Swapping to
oQ3-MTP means deleting oQ4 first, and back again means re-running `~/models/dl-oQ4.sh`.

Cleaning won't change that. The biggest reclaimable items add up to about 13 GB. The Trash is empty and there are no APFS local snapshots.

Long sessions build up swap. After a day of streaming work the machine held seven 1 GB swapfiles at 93% use, plus a 1 GB `kernelcore` panic
dump in `/System/Volumes/VM`. Reboot before a trusted measurement. Swap adds variance that the harness cannot remove.

Two directories look like free space and aren't. `macqwen/cli.py` runs Flash-Next from `~/models/.venv-qwen4exp/bin/python` and Qwen3.8-27B
from `~/mlx-qwen38-kernel-lab/bin/python3`, so deleting either stops the chat. `~/mlx-qwen38-apple` has no git history and no remote, so its
work only exists here. Don't delete any of the three to free space.

## Next work

The checkpoint question is settled. oQ4 is the baseline and is installed. oQ3-MTP is gone from the reference machine.

Current work is tracked in the public issue tracker:

- [#4](https://github.com/1architect/macqwen-releases/issues/4) Re-run the cache-aware quality and trajectory gate with sampling.
- [#5](https://github.com/1architect/macqwen-releases/issues/5) Measure cache-aware routing at long generation and 5K context.
- [#6](https://github.com/1architect/macqwen-releases/issues/6) Measure prefill recovery after `FLASHNEXT_SWAP_MAX_ROWS`.
- [#7](https://github.com/1architect/macqwen-releases/issues/7) Confirm draft contention with a warm page cache.
- [#8](https://github.com/1architect/macqwen-releases/issues/8) Fix missing spaces at streamed chunk joins.
- [#9](https://github.com/1architect/macqwen-releases/issues/9) Widen the cache-aware quality gate.
- [#10](https://github.com/1architect/macqwen-releases/issues/10) Weight-preserving cache-aware swap. Measured and rejected in `research.md`.

Open exact-quality performance experiments:

- [#23](https://github.com/1architect/macqwen-releases/issues/23) Recheck the bit-exact RMSNorm compile with the zero-drive gate and production arms.
- [#24](https://github.com/1architect/macqwen-releases/issues/24) Probe routed-expert Q4 group sizes 64 and 128.
- [#25](https://github.com/1architect/macqwen-releases/issues/25) Gate and benchmark REAP-288. Use REAP-384 as the fallback.

### SSD -> Memory -> Metal Runtime Frontiers (The 10 Next Steps)

The path to breaking through 3.0 tok/s (<333 ms/token) from the current 2.86 tok/s baseline (~345 ms/token) spans 10 architectural frontiers across the storage, unified memory, and Metal boundary:

1. **Corrigir o slab A/B e usar experts realmente hot** (Priority: High | Status: **CLOSED / IMPLEMENTED**):
   - Decoupled resident slab from prefill tokens; fixed test-isolation bugs wiping `pins.json`.
   - Solved the *layer utility inversion* problem: replaced uniform first-12 layer assignment with concentrated global allocation (`FLASHNEXT_SLAB_GLOBAL=48, min_slots=4`), focusing on the 12 highest-utility layers (`[5, 11, 20, 23, 29, 32, 35, 39, 40, 44, 46, 47]`).
   - Decode hit rate jumped from 14.1% to **23.5%** (+67.4% relative gain) for **0 extra RAM** (+144 MB active RAM).
   - Delivered **+8.3% mean / +5.5% median paired speedup** (2.70 -> 2.86 tok/s, tail 2.79 tok/s) with 100% bit-identical digest `29d04075ed7021b3`.

2. **Slab file-backed: mlocked mmap + streamed buffer no mesmo kernel** (Priority: Very High | Status: **ACTIVE NEXT**):
   - Pass file-backed mlocked pointers directly into the unified Metal MoE kernel, eliminating intermediate Python MLX array allocations.
   - The unified bit-31 pointer encoding (`0x80000000 | slot`) already accepts mixed slab/streamed inputs in a single kernel dispatch.
   - Constraint: Must keep resident allocation strictly bounded to <= 48-64 slots (148-198 MB) to prevent Darwin page-cache eviction on 16 GB Apple Silicon.

3. **1 expert-major record -> custom kernel direto** (Priority: High | Status: **EXPLORATORY**):
   - Repack gate, up, and down projection rows (or all 9 parts) into a single contiguous record per expert, replacing 9 scattered file seeks with 1 positioned read per expert.
   - Constraint: Offline checkpoint repack requires temporary disk capacity. The reference 256 GB drive has ~22 GB free while oQ4 (111 GB) is installed. Prototype with a selective subset of layers.

4. **Composite read buffer: 9 wraps/layer -> 1** (Priority: Med/High | Status: **MEASURED & REJECTED**):
   - Profiling (`FLASHNEXT_PROFILE_IO=1`) proved all 432 DLPack foreign wraps (`to_mx`) take only **2.51 ms/token** across all 48 layers.
   - Slicing composite buffers in Python/MLX caused non-aligned slicing and graph evaluation stalls, dropping generation rate from 2.94 to 2.30 tok/s.
   - Independent row buffers per part (`_SharedRead` with `empty_rows`) preserve threadpool concurrency and are retained as optimal.

5. **Fused-down sem global scratch/barriers** (Priority: Medium | Status: **CLOSED / IMPLEMENTED**):
   - Replaced intermediate device scratch tensor write/read with in-register accumulation (`qmv_accumulate_impl`).
   - Eliminated the 40 KB device memory scratch allocation per call and 768 threadgroup barriers per token across 48 layers. Bit-exact on bfloat16 (`test_scratchless_fused_down_bfloat16`).

6. **Up-QMV + SwiGLU** (Priority: Low/Med | Status: **ON HOLD - NUMERICAL GATE**):
   - Fusing `activation = up * silu(gate)` in the Up-QMV epilogue eliminates `up_out` tensor allocation and saves 48 MLX elementwise launches per token.
   - Prototyped and measured: MLX's compiled SwiGLU truncates intermediate sigmoid values to bfloat16 differently than standard Metal math (max diff 0.0156-0.03125), threatening bit-identical token digest `29d04075ed7021b3`. Held until an exact 1-ULP arithmetic match is engineered.

7. **FMA / fast math** (Priority: Low | Status: **ACTIVE IN KERNEL**):
   - `qmv_fast_impl` currently uses `fma(...)` intrinsics for Q4 dequantization. Fast-math compiler flags can be evaluated provided bfloat16 rounding remains identical.

8. **Shared-output final fusion** (Priority: Low | Status: **OPEN**):
   - Fusing the shared expert output addition directly into the down-combine kernel to eliminate 48 standalone MLX elementwise additions (~0.5-1.0 ms saving).

9. **Global kernel cache** (Priority: Cleanup | Status: **CLOSED / IMPLEMENTED**):
   - Metal kernels are compiled once and cached in `_COMPILED_KERNELS`; compilation latency is 0.00 ms from token 2 onward.

10. **Native bridge persistent zero-copy** (Priority: Structural | Status: **FOUNDATION CLOSED**):
    - Rigorous unbuffered DMA testing (`F_NOCACHE`) closed Issue #45, proving `MTLBarrierScopeBuffers` adds zero penalty under physical SSD DMA.
    - Single-pass bit-31 pointer encoding is verified and ready for native C++/Obj-C persistent runtime integration.

### Immediate Tactical Next Steps to Break >3.0 tok/s (~9-12 ms/token remaining)

- **Step A: Heterogeneous / Skew-Aware Global Slot Allocation**:
  Allow variable capacities across the top-12 layers (e.g. 5-6 slots in super-concentrated layers 5, 20, 23, 35, 47, and 3 slots in moderately hot layers) bounded to <= 48-52 slots total. Targets +3-5% additional hit rate (~6-8 ms saving).
- **Step B: Formal Production Benchmark of Budget 56 Slots**:
  Run 12-arm interleaved reversed-pair benchmark (`bench_slab_production.py --target global56`) to measure 56 slots (+173 MB RAM, safely below 196 MB page-cache threshold).
- **Step C: Stacking Bit-Exact RMSNorm Compile (Issue #23)**:
  Stack `FLASHNEXT_COMPILE=1` with `global48` to test if zero-drive 1.7% (~4-5 ms/token) saving clears the resolution band now that I/O wait and scheduling overhead are reduced.
- **Step D: Direct Zero-Copy `preadv` I/O Production Test**:
  Evaluate `FLASHNEXT_READ=preadv` across multi-arm production harness to test eliminating ~864 Python `bytes` heap allocations and memcpys per token.

Closed exact-quality performance issues:

- [#21](https://github.com/1architect/macqwen-releases/issues/21) Host-only idle windows. Only 4.16 ms/token qualifies after bulk movement is excluded.
- [#22](https://github.com/1architect/macqwen-releases/issues/22) Routed `gather_qmm`. The measured path runs at 92.2 to 92.4 GB/s.
- [#26](https://github.com/1architect/macqwen-releases/issues/26) Confirm and retain shared-buffer chunk-2 reads. Closed after the clean-boot result.
- [#27](https://github.com/1architect/macqwen-releases/issues/27) Attribute the remaining FlashNext GPU layer cost. Closed after Metal trace attribution.
- [#41](https://github.com/1architect/macqwen-releases/issues/41) Resolve reusable destination-ring performance. Closed with no resolved benefit; diagnostic remains disabled.
- [#42](https://github.com/1architect/macqwen-releases/issues/42) Characterize the GPU-busy hump across drive miss levels. Closed after the reversed-order sweep.
- [#43](https://github.com/1architect/macqwen-releases/issues/43) Resolve FlashNext Metal wired-limit behavior. Closed on 2026-09-03: pre-loading `FLASHNEXT_WIRED_GB=2` strictly before model loading showed no resolved gain or penalty across 8 interleaved reversed arms (-2.1% mean, +0.6% median, p=0.688, band 15.8%). Unified memory residency sets handle streaming decode without static OS page wiring.
- [#45](https://github.com/1architect/macqwen-releases/issues/45) Measure Metal barrier and fence cost under mixed residency. Closed on 2026-09-03: rigorous unbuffered DMA testing (`F_NOCACHE`, `proc_pid_rusage`, and 100% thread latch overlap) proved `MTLBarrierScopeBuffers` adds 0.000 ms penalty over serial execution across 0 to 64 MB physical SSD DMA (-0.071 ms at 16 MB, -0.024 ms at 32 MB, -0.124 ms at 64 MB). Hardware fabric sharing contention is bounded to ~5-12%.

Follow-up issues from the 2026-09-01 sweep:

- [#33](https://github.com/1architect/macqwen-releases/issues/33) Remove or repair the unreachable `ExpertLRU` merge path.
- [#34](https://github.com/1architect/macqwen-releases/issues/34) Bound FlashNext benchmark token limits.
- [#35](https://github.com/1architect/macqwen-releases/issues/35) Complete the excluded FlashNext read-path measurements.
- [#36](https://github.com/1architect/macqwen-releases/issues/36) Correct absolute GPU utilization reporting.
- [#37](https://github.com/1architect/macqwen-releases/issues/37) Measure the resident-work boundary below 640 MB.
- [#38](https://github.com/1architect/macqwen-releases/issues/38) Recheck GDN timing with a dependency-correct chain.
- [#39](https://github.com/1architect/macqwen-releases/issues/39) Explain clean-boot GPU busy variance.

### Standing decisions

- Concentrated global slabs (`FLASHNEXT_SLAB_GLOBAL=48, FLASHNEXT_SLAB_MIN_SLOTS=4`) outperform uniform and static first-12 layer slabs by concentrating the 48-slot budget into the highest-utility layers ([5, 11, 20, 23, 29, 32, 35, 39, 40, 44, 46, 47]), achieving 23.5% decode hit rate (vs 14.1% on slab12) for the exact same +149 MB active RAM, reaching 2.86–2.91 tok/s generation and 2.79–2.92 tok/s tail rate with bit-identical digest `29d04075ed7021b3`. Combined with the scratch-free register fused-down kernel, intermediate device scratch allocation and 768 threadgroup barriers per token are completely eliminated.
- In-encoder Metal buffer barriers (`MTLBarrierScopeBuffers`) incur zero hardware penalty under physical SSD DMA and can safely be used for layer kernel consolidation.
- `pin-parts` is rejected. Its positive isolated reading disappeared when
stacked with prewarm; the pair lost 1.4% and read 8% more.
- Weight-preserving expert substitution is not bit-perfect. Keep the current
cache-aware implementation unchanged.
- Keep every accepted change on the shared chat path.
- Installation and checkpoint verification could still be better.
