# Flash-Next operational handoff

Read this file, [`research.md`](research.md), and
[`AGENT_INVARIANTS.md`](AGENT_INVARIANTS.md) before changing code or starting
an experiment. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) defines the project
rules. Release 0.4.1 closes the corrective REAP integration work; inspect the
worktree before every change instead of relying on a recorded clean/dirty
state.

## Decision and current state

Our canonical backend is the MLX-backed `FlashNextBackend`. `chat.sh` defaults
to the MLX-backed Metal runtime (`FLASHNEXT_METAL_RUNTIME=1`). For REAP Q4/G64,
we now default to the G64 Metal executor (`FLASHNEXT_METAL_G64=1`).
We retain generic MLX expert execution as the explicit rollback with
`FLASHNEXT_METAL_G64=0`.

We promote this default on 2026-09-17 at our explicit request, using the
controlled 32-token speed and exact-digest evidence below. We skip the
long-turn quality gate at our request for this decision. This is a specific
promotion exception, not proof of general quality or long-turn equivalence.

The current research checkpoint is:

```text
sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit
```

It uses generic MLX Q4/G64 expert weights and Q4/G32 n-gram weights. We keep
the generic MLX Q4/G64 reference path available for REAP comparisons and rollback.
Only the G64 executor default changes. G64 slabs remain off with
`FLASHNEXT_SLAB_G64=0`, and stream-pack remains off with
`FLASHNEXT_STREAM_PACK=0`. We reject and remove the native prototype because
it does not implement the complete model. MLX remains canonical; no native
runtime flag is needed.

The 60-slot Frontier 8A and corrected decode-only Q4/G32 controls are
historical compatibility evidence. They are not the current REAP runtime and
must not be presented as REAP results.

REAP general quality and long-turn equivalence remain open. Our short controlled
speed result supports this requested default change only within its measured
conditions. A matching short digest does not clear the long-turn quality gate.
We retain the normal promotion protocol below for other optimizations.

## Default configuration and rollback

We use the MLX-backed Metal runtime with the G64 executor enabled:

```text
FLASHNEXT_METAL_RUNTIME=1
FLASHNEXT_METAL_G64=1
FLASHNEXT_SLAB_G64=0
FLASHNEXT_STREAM_PACK=0
FLASHNEXT_QSA_CACHE_POOLED_KEYS=0
FLASHNEXT_QSA_SCATTER_DECODE=0
```

We roll back to generic MLX Q4/G64 execution with an explicit override:

```bash
FLASHNEXT_METAL_G64=0 ./chat.sh --checkpoint reap
```

The runtime flag selects the MLX-backed Metal path. The separate G64 flag
selects the executor, and we preserve explicit `0` overrides. Generic Q4/G32
slab settings remain available for compatible historical checkpoints. They do
not activate G64 slabs while `FLASHNEXT_SLAB_G64=0`. QSA optimization flags
remain off; the existing allocation guard remains active.

`FLASHNEXT_NATIVE_PIPELINE` is obsolete because the native integration was
removed. Do not add it to a launcher or use it as a control.

For compatible Q4/G32 checkpoints, the historical engineering control is
`exact-quality` with threshold `0.85`, 32 resident experts, warmup 8, swap
epsilon `0.02`, chunk 2, and 16 I/O workers. Do not copy that profile onto
REAP and call it a REAP measurement.

The normal routing profile is `exact-quality`. Research-only profiles include
`cache-aware`, `speculative-fast`, and MTP variants. They require their own
quality and trajectory gates and are not normal defaults.

REAP `xhigh` keeps its requested reasoning mode but caps reasoning within the
existing total generation allowance at 4,096 tokens by default. Historical
`think_budget=-1` means reasoning shares that total allowance; the guard must
not add a new 4,096-token allowance to it. We can set a positive
`MACQWEN_REAP_XHIGH_THINK_BUDGET` for an explicit cap or set
`MACQWEN_ALLOW_REAP_XHIGH=1` for an explicit diagnostic without the cap.
`/status` reports requested effort and effective budget.

The QSA guard selects the existing bounded-mask path when the projected
upstream mask exceeds 512 MiB. Its estimate includes cached tokens and batch
size. At 32K decode context, the logical 512-by-context mask is about 16 MiB per
QSA layer and remains on the original path. Caching completed pooled indexer
keys and reusing the scatter-based decode mask are unmeasured, exactness-bound
opportunities. Manual validation of the cached tool-result prefill path remains
a follow-up item.

## Legitimate validation on record

The following are the validation records we may use while resuming work:

- Current non-model validation: 251 `macqwen` tests and 323 FlashNext tests.
  The FlashNext suite includes 56 focused Metal/runtime and slab tests; none
  of these checks load a checkpoint or generate model tokens.
- Our terminal sanity check on 2026-09-12 used 3 arms and 32 tokens. The baseline
  median was 3.74 tok/s, range 3.23–3.78, tail 3.45 tok/s, and 193.3 MB/token;
  the token digest was `1a9abb4b5fdc523a`. This is a quick sanity check, not a
  statistical promotion of REAP throughput.
- A fixed-seed sampled `xhigh` check used one 32-token arm per path. Reference
  and G64 produced the same digest, `956631b20cf2b294e4ddb07cee1ada16aa12b576c5d4779de521cc2446568cd0`.
  Seed 42 is a known regression case, not a representative quality seed. Both
  stayed inside reasoning and made the same language-identification error;
  this is short trajectory equality, not a completed quality or speed gate.
- Our 2026-09-17 G64 comparison uses six reversed interleaved pairs of 32
  greedy tokens. Reference median is 2.374 tok/s versus 2.665 tok/s for Metal.
  The paired gain is +13.4% mean and +15.1% median, above the 5.0% resolution
  band, with 6/6 wins and two-sided sign-test `p=0.031`. All digests match
  `1a9abb4b5fdc523a7a2986fb62f2b63570af515a98cecde6f201e682580fa65d`.
  Physical-read medians are 191.2 versus 191.1 MB/token. We retain the JSON
  and log at `~/.cache/flashnext/exact-speed-20260917/g64.json` and `g64.log`
  in that directory. This supports our requested executor default, not general
  quality. We skip long-turn quality validation at our request.
- oQ4 is the recorded quality baseline only. It is historical and is not the
  installed REAP checkpoint.

We do not use any removed native-engine, zero-drive, synthetic fixed-route, or
uncommitted scratch result as a production claim. Research history and rejected
experiments belong in `research.md`, not in this operational summary.

## Required benchmark protocol

Use `models/flashnext/bench_production.py` for published decode numbers and
`models/flashnext/bench_slab_production.py` for selective slab comparisons.
Before any run, we must record:

- checkpoint identity and complete source/configuration fingerprints;
- prompt, chat template, sampling settings, token limit, and seed policy;
- arm definitions, environment variables, worker count, and resident policy;
- exact generated token arrays or a reproducible digest;
- wall time, generation and tail rates, physical MB/token, RSS, swap, and I/O
  counters.

Use multi-arm interleaving with reverse ordering on alternate rounds. This
mitigates ordering drift; it does not prove that drift cannot affect the
comparison. Pair the same prompt and controls, report paired means/medians and
the resolution band, and do not infer a gain from a single run or from logical
hit rate. Preserve a 32-token exact-digest arm. On a fanless machine, avoid
unnecessary warmup loops that can thermally throttle the device.

Do not insert `mx.eval`, warmup, cache flushes, or artificial sleeps into a
timed path unless that operation is the measured variable. Do not use a
zero-drive estimate as a production result. A synthetic route or uninitialized
weight buffer is a diagnostic, never a model benchmark. Reject any arm that
does not load the same real checkpoint and produce the same required outputs.
For exact performance comparisons, the required output is the same token
digest. A sampled quality pair may branch, but both arms must complete and are
scored blind against the functional criteria below.

Construct and publish the raw arm immediately after generation, before any
post-generation path validation. If a digest, startup, or path check fails,
write the failed arm and available evidence to the result artifact before
returning a failing status. Every artifact must bind the checkpoint identity,
runtime-source fingerprint, and benchmark-harness fingerprint.

Our normal G64 or REAP experiment protocol first requires a short exact-quality
gate, then long-turn quality with the same agent prompt and explicit sampling
seed. A short matching digest is necessary but not sufficient. Our requested
2026-09-17 G64 executor promotion is a specific exception: we skip long-turn
validation without marking it passed. G64 slabs and stream-pack remain disabled
until their separate quality and controlled performance gates are clear.

Our pending long-turn G64 comparison retains Astra's recommendation and requires
our permission before execution. We predeclare exactly three seeds, 7, 19,
and 73, and pair G64 off versus on with `FLASHNEXT_METAL_RUNTIME=1`. We keep all
slabs and stream-pack off in both arms (`FLASHNEXT_SLAB=0`,
`FLASHNEXT_SLAB_GLOBAL=0`, `FLASHNEXT_SLAB_PACK=0`,
`FLASHNEXT_SLAB_G64=0`, and `FLASHNEXT_STREAM_PACK=0`). Alternate arm order
between rounds, hold the checkpoint, prompt, template, sampler, effort, and
token allowance fixed, and require completed outputs. We score the complete
SketchUp `.rb` artifact blind to arm labels against functional criteria.
Interrupted generations are incomplete gates, not quality failures.

## Essential files and commands

| Path | Role |
|---|---|
| `macqwen/backends/flashnext.py` | Backend adapter and generation loop |
| `macqwen/session.py` | Session state and chat integration |
| `models/flashnext/loader.py` | Checkpoint and streamed-module loading |
| `models/flashnext/store.py` | Tensor-row reads from checkpoint shards |
| `models/flashnext/expert_cache.py` | Routed expert residency and reads |
| `models/flashnext/routing.py` | Routing profiles and token transitions |
| `models/flashnext/adaptive_topk.py` | Adaptive expert thresholding |
| `models/flashnext/qsa_chunk.py` | Bounded QSA allocation |
| `models/flashnext/patch_rmsnorm.py` | Checkpoint-specific RMSNorm behavior |
| `models/flashnext/metal_runtime.py` | Compatible MLX Metal Q4/G32 paths |
| `models/flashnext/slab_pack.py` | File-backed compatible slab storage |
| `models/flashnext/settings/` | Settings registry and safe launch defaults |
| `models/flashnext/tests/` | Interactive research test catalog |
| `models/flashnext/bench_production.py` | Standard production benchmark |
| `models/flashnext/bench_slab_production.py` | Paired slab benchmark |
| `models/flashnext/diskio.py` | Physical-read accounting |

Set up the local environment with:

```bash
./chat.sh setup
./chat.sh --checkpoint reap
```

Use `--model-path` or `MACQWEN_FLASHNEXT_MODEL` for an explicit complete
checkpoint. The REAP directory is normally under
`$HOME/models/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`.

Before a code change, inspect the worktree and the last commit. Keep unrelated
user changes intact and do not reset or discard them implicitly.

## Open risks and bugs

- REAP long-turn quality and trajectory are unverified; hard reasoning turns
  can loop or produce incomplete output.
- We enable the custom Q4/G64 executor at our request despite the open
  long-turn quality gate. The earlier interrupted attempt also used an oversized
  `xhigh` allowance. It remains incomplete, not a quality failure. The new short
  speed evidence does not establish general quality or long-turn equivalence.
- REAP checkpoint-specific RMSNorm and mixed Conv1d layout handling must remain
  shape-checked and fingerprint-aware; do not rewrite checkpoint tensors.
- Issue #23 tracks the bit-exact RMSNorm compile gate.
- Issue #24 tracks routed-expert Q4/G64 and Q4/G128 investigation.
- Issue #25 tracks the REAP-288 quality/performance gate, with REAP-384 as a
  fallback checkpoint.
- Issue #43 tracks the pre-load wired-memory comparison.
- Issue #45 tracks interval-valid Metal/SSD DMA contention evidence.
- Issue #48 tracks SSD DMA and GPU contention outside FlashNext.
- Issue #49 tracks FlashNext prefill when opened for plain and agent profiles.

We keep a candidate's speed result unresolved when it falls inside its measured
resolution band. Our 2026-09-17 G64 short result clears its reported band;
our requested promotion does not resolve the separate long-turn quality gate.

## Next steps

1. Keep our requested G64 Metal default and the explicit generic MLX rollback.
   Keep G64 slabs, stream-pack, and QSA optimization flags off. Do not
   reintroduce the removed native prototype.
2. Keep the corrected shared-total reasoning semantics, benchmark evidence,
   and checkpoint-bound pin profiles covered by non-model regression tests.
3. Test QSA completed-block key caching and the scatter-based decode mask as
   separate exact changes, including partial blocks, trimming, restored
   sessions, and positions.
4. Complete the pending manual QSA/prefill validation without changing the
   benchmark protocol.
5. Retain the pending long-turn G64 quality comparison with packed residency off
   in both arms. Run it only with our permission; do not start a fused-down rewrite.
6. Run a REAP-specific 32-versus-8 expert-pin comparison only with unchanged
   routes, arithmetic, and packed-residency policy.
7. Treat last-row-only logits below the 2,048-token prefill threshold as a
   time-to-first-token and memory candidate, not a decode optimization.
8. If investigating G64 or REAP performance, run the short digest gate first,
   then the long-turn quality gate, then the reversed interleaved benchmark.
9. Record new evidence in `research.md`, update this handoff only with the
   resulting operational decision, and state clearly whether the result is
   accepted, unresolved, or rejected.

Our immediate objective is reproducible, exact-quality MLX behavior. Speed
claims follow evidence; they do not define the control.
