# Flash-Next handoff

## Environment

The launcher expects:

```text
Python       ~/models/.venv-qwen4exp/bin/python
Checkpoint   ~/models/Qwen3.8-Flash-Next-MLX-oQ4
```

Override the interpreter with `MACQWEN_FLASHNEXT_PYTHON`.
Override the checkpoint with `--model-path`.

## Download

```bash
hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ4 \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-oQ4"
```

The checkpoint should contain 22 safetensors shards and use about 112 GB.

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
./chat-swap.sh
```

`fused-quality` is experimental. It failed the retained reasoning gate.
`cache-aware` is optional. It improves speed but changed the preferred answer
in a long-context comparison. `exact-quality` remains the default.

Use the live configurator before a turn:

```text
/settings
/settings routing exact-quality
/settings routing cache-aware
/settings routing fused-quality
/settings swap-epsilon 0.02
/settings threshold 1.0
/settings resident-experts 32
/settings pinned-experts 32
/settings pin-budget-gb 6
/settings tail-experts 6
/settings tail-warmup 8
/settings fusion-block 23
/settings fusion-min-margin 1.0
/settings fusion-min-block 20
/settings fusion-margin-tokens 8
/settings fusion-max-prompt 512
/settings fusion-model <path to a draft model>
/settings defaults
```

`pinned-experts` aliases `resident-experts`. `pin-budget-gb` caps the pinned
storage. `/settings` reports the current pinned layer-expert count and bytes.

Settings apply to the current process only. A new `./chat.sh` launch returns to
`exact-quality`, threshold `0.85`, 32 resident experts, warmup `8`, and
swap epsilon `0.02`.

Use `/reset` before enabling the one-shot fused draft for a new conversation.

`speculative-fast` and MTP stay research-only. Both need a different load path,
and both lost their complete-runtime controls.

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
| `models/flashnext/diskio.py` | Physical bytes read, to tell a cold run from a warm one |
| `models/flashnext/bench_residency.py` | Check the residency gate against `mincore` |
| `models/flashnext/bench_prefill_scaling.py` | Prefill rate and bytes across prompt lengths |
| `models/flashnext/bench_route_swap.py` | Count how often a cold expert had a resident near-equal alternative |
| `models/flashnext/bench_swap_quality.py` | Compare exact and cache-aware answers on checkable prompts |

## Validation

Run the model suite in its environment:

```bash
~/models/.venv-qwen4exp/bin/python -m unittest discover \
  -s models/flashnext -p 'test_*.py' -q
```

Run a live session restore:

```bash
printf '/load probe\n/status\n/quit\n' | \
  ./chat.sh --model flashnext --profile plain --exact-quality
```

Run the complete JSON benchmark path:

```bash
./chat.sh --model flashnext --profile plain --exact-quality \
  --max-tokens 32 --benchmark-json --benchmark-prompt 'Explain virtual memory.'
```

Standard output must contain one JSON object. Diagnostic text goes to standard
error.

## Prefill scaling

The reference machine is an M4 Mac with 16 GB of unified memory and a 256 GB
SSD. Prefill throughput increases with prompt length. Large batches amortize
fixed setup and streamed-read costs across more tokens. A prompt near 5,000
tokens may reach about 40 to 50 tok/s under favorable conditions. Do not
extrapolate large-prompt throughput from a short interactive prompt. Prompt
content, free memory, SSD state, and page-cache state can move the result.

## Measured rate

| Condition | Rate |
|---|---:|
| exact-quality, complete decode, ten kept arms | **2.713 tok/s**, 390 MB/token |
| the same arms, pinned tail | 2.650 tok/s |
| cache-aware hot-run median | **2.79 tok/s**, 347.6 MB/token |
| exact arm in the same hot run | 2.54 tok/s, 417.8 MB/token |
| cache-aware paired effect | **+8.3%**, seven of eight pairs faster |
| older harness, ten pinned-tail arms, colder start | 2.42 to 2.73 tok/s, mean 2.59 |
| terminal `gen`, short chat turns | about 2.0 to 2.5 tok/s |
| separate warmup-eight pair | 2.88 and 2.78 tok/s, mean 2.83 |
| synthetic fixed routes, every expert read resident | **5.33 tok/s** |
| synthetic fixed routes, every expert read cold | 1.09 tok/s |

The 2.83 value is not a production baseline. The ten-arm mean is 2.59, and
prompt-locality tests ranged from 1.98 to 2.44 tok/s. All these values measure
the pinned tail. Use complete decode for interactive performance.

The synthetic ceiling shows that compute can exceed 3 tok/s when expert reads
stay resident. It does not predict a real production reply.

The cache-aware comparison ran on a hot machine. Its alternating arms make the
paired effect useful, but its absolute 2.79 rate does not replace the 2.713
exact-quality baseline. All eight cache-aware arms read fewer bytes.

Use `models/flashnext/bench_read_ceiling.py` to reprice the drive at zero:

```bash
FLASHNEXT_READ=resident FLASHNEXT_PROFILE_IO=1 \
  python3 models/flashnext/bench_read_ceiling.py --mode ram
FLASHNEXT_READ=pread FLASHNEXT_PROFILE_IO=1 \
  python3 models/flashnext/bench_read_ceiling.py --mode disk
```

`FLASHNEXT_READ` selects the expert read path. `pread` is the default and the
only one to use in production. `resident` maps mlocked rows instead of copying
them; it wins by 25 percent when the drive is idle and loses under load.

The profile decides whether the variable applies:

| Profile | Read path | Pins experts | `FLASHNEXT_READ` applies |
|---|---|---|---|
| `standard` | the variable | no | yes, with no gain, nothing is pinned |
| `exact-quality` | the variable | 32 | yes |
| `cache-aware` | the variable | 32 | yes |
| `fused-quality` | the variable | yes | yes |
| `fast-quality` | `shared_mmap` after warmup | yes | no |
| `fast` | `shared_mmap` | no | no |

`fast` and `fast-quality` were measured on `shared_mmap` and keep it. Setting
`FLASHNEXT_READ` does not change them.

## Measurement rules

- Run one model instance unless parallel operation is the experiment.
- Hold prompt text and generated token limit constant.
- Compare token IDs before accepting a performance change.
- Reverse or interleave A/B order.
- **Use three arms per condition minimum.** The first arm of a run is always
  the slowest, because the page cache warms across arms. Two-arm A/Bs on this
  machine have produced +12.8% and +10.7% results that were both noise.
- Read the resolution band the harness prints. It is two standard errors of
  the difference between medians. A reading inside it is unresolved, which is
  not the same as absent.
- Stack two changes that are each unresolved and measure the pair. It costs
  one comparison instead of two re-runs, and a real pair clears the band.
- Interleave the lengths in any sweep over prompt size. Walking them in order
  measures the warm-up: an ascending sweep read fewer bytes at 4 tokens than
  at 2, and moved the batching crossover from 32 tokens to 8.
- An isolated reader A/B cannot support a layout claim. The reader is not what
  the model waits on once the page cache, the n-gram stream and the compute
  share one process.
- A benchmark must prove its own premise before it reports. Verify the drive
  served the reads, verify the setting took effect, and refuse to report
  otherwise. Three benchmarks in one day returned plausible numbers while
  measuring nothing: one never set the read mode its gate needed, one read
  back pages its own write had cached, and one could not change a setting held
  in a module constant.
- Read this file's do-not-retry list and grep the research log before building
  anything. The expert-major repack was rebuilt from scratch while sitting as
  item three on that list.
- Record free memory and competing applications.
- Never predict throughput from the routing coverage curve. Coverage counts
  accesses to the top experts. It does not describe page-cache residency.

## Do not retry without a new mechanism

- Expert result caches and resident weight slabs.
- Warm read-ahead and in-process prefetch overlap.
- Repacking the complete checkpoint.
- Two-bit expert requantization.
- Low-rank expert approximation.
- Native MTP for this complete runtime.
- Exact speculative paths already measured in the research log.
- Removing host work from the read path. Mapping resident rows instead of
  copying them, dropping the concatenate, and issuing reads earlier were each
  measured. Every one returns its saving to the GPU wait under drive pressure.
- Pinning more than 32 experts. Tested again with a corrected candidate pool:
  `hot=40` pins 6.12 GB and returns the same rate as `hot=32`.
- Longer routing warmup. `warmup=40` measured 3.7 percent slower than 8.
- Sorting a layer's reads by offset, pinning only scales and biases, and
  warming last session's expert set. Each measured inside its resolution band
  alone, and the last two measured -1.4% together with 8% more physical reads,
  so they do not add.
- Mapping resident expert rows, at any gate accuracy. A tracker with 97.6
  percent precision, well past its 78.5 percent break-even, measured 5.9
  percent slower while reading 3.2 percent fewer bytes, and degraded further
  as more rows became eligible. The harm scales with the mapped fraction.

## Next work

- Repeat cache-aware routing on a cool machine. Confirm its absolute rate.
- Expand the cache-aware quality gate. Include long generations, reasoning,
  Portuguese, code, and factual prompts.
- Test `swap-epsilon 0.005` as the conservative profile. The opportunity study
  found 11.9 percent fewer cold reads at this value, versus 13.9 percent at
  `0.02`, with less routed mass changed.
- Keep exact-quality as the default. Promote cache-aware only if the wider
  quality gate supports the change.
- Treat `pin-parts` as rejected. Its positive isolated reading disappeared
  when stacked with prewarm. The pair lost 1.4 percent and read 8 percent more.
- Lower-bit checkpoints such as oQ3 are an option nobody has chosen. The
  projection is 2.95 to 3.10 tok/s from about 17 percent fewer bytes, but it
  means deleting oQ4, which is not reversible without a 92 GB download, and it
  trades quality for rate. Do not treat it as planned work.
- Improve installation and checkpoint verification.
- Explore lower-bit published checkpoints with explicit quality gates.
- Keep all accepted changes on the complete shared chat path.
