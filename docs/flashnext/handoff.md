# Flash-Next operational handoff

Read this file, [`research.md`](research.md), and
[`AGENT_INVARIANTS.md`](AGENT_INVARIANTS.md) before changing code or starting
an experiment. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) defines the project
rules. The current worktree changes are uncommitted and require review before
we create a commit.

## Decision and current state

Our canonical backend is the MLX-backed `FlashNextBackend`.

The current research checkpoint is:

```text
sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit
```

It uses generic MLX Q4/G64 expert weights and Q4/G32 n-gram weights. We keep
the generic MLX Q4/G64 reference path for REAP. The custom G64 kernel, G64
slab pack, and expert-major stream-pack path are off for normal chat. The
native prototype was rejected and removed because it did not implement the
complete model; MLX is canonical and no native runtime flag is needed.

The 60-slot Frontier 8A and corrected decode-only Q4/G32 controls are
historical compatibility evidence. They are not the current REAP runtime and
must not be presented as REAP results.

REAP quality and throughput remain open. A short equality or sanity check does
not clear the long-turn quality gate. We do not promote an optimization until
the same checkpoint, prompt, sampling policy, token digest, and physical-I/O
accounting support it.

## Safe configuration

For the current REAP reference path, keep these settings off:

```text
FLASHNEXT_METAL_G64=0
FLASHNEXT_SLAB_G64=0
FLASHNEXT_STREAM_PACK=0
```

`FLASHNEXT_NATIVE_PIPELINE` is obsolete because the native integration was
removed. Do not add it to a launcher or use it as a control.

For compatible Q4/G32 checkpoints, the historical engineering control is
`exact-quality` with threshold `0.85`, 32 resident experts, warmup 8, swap
epsilon `0.02`, chunk 2, and 16 I/O workers. Do not copy that profile onto
REAP and call it a REAP measurement.

The normal routing profile is `exact-quality`. Research-only profiles include
`cache-aware`, `speculative-fast`, and MTP variants. They require their own
quality and trajectory gates and are not normal defaults.

REAP `xhigh` keeps its requested reasoning mode but caps the effective
reasoning budget at 4,096 tokens by default. We can set a positive
`MACQWEN_REAP_XHIGH_THINK_BUDGET` for an explicit cap or set
`MACQWEN_ALLOW_REAP_XHIGH=1` for an explicit diagnostic without the cap.
`/status` reports requested effort and effective budget.

The QSA guard selects the existing bounded-mask path when the projected
upstream mask exceeds 512 MiB. Its estimate includes cached tokens and batch
size. Manual validation of the cached tool-result prefill path remains a
follow-up item.

## Legitimate validation on record

The following are the validation records we may use while resuming work:

- Recorded test coverage: 247 `macqwen` tests and 295 FlashNext tests.
- The focused 48-test check covers reasoning policy, session behavior, the
  FlashNext settings registry, and the G64 safety guard.
- Our terminal sanity check on 2026-09-12 used 3 arms and 32 tokens. The baseline
  median was 3.74 tok/s, range 3.23–3.78, tail 3.45 tok/s, and 193.3 MB/token;
  the token digest was `1a9abb4b5fdc523a`. This is a quick sanity check, not a
  statistical promotion of REAP throughput.
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

Use multi-arm interleaving with reverse ordering on alternate rounds. Pair the
same prompt and controls, report paired means/medians and the resolution band,
and do not infer a gain from a single run or from logical hit rate. Preserve a
32-token exact-digest arm. On a fanless machine, avoid unnecessary warmup
loops that can thermally throttle the device.

Do not insert `mx.eval`, warmup, cache flushes, or artificial sleeps into a
timed path unless that operation is the measured variable. Do not use a
zero-drive estimate as a production result. A synthetic route or uninitialized
weight buffer is a diagnostic, never a model benchmark. Reject any arm that
does not load the same real checkpoint and produce the same required outputs.

For G64 or REAP experiments, first pass a short exact-quality gate, then the
long-turn quality gate with the same agent prompt. A short matching digest is
necessary but not sufficient. Keep the G64 kernel, G64 slab pack, and
stream-pack disabled unless both quality and controlled performance gates are
clear.

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
- The custom Q4/G64 kernel remains held out after a long-turn quality failure.
  Its short equality evidence does not establish quality or speed.
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

The complete-model comparison remains unresolved when a candidate falls inside
the measured resolution band. We keep the current control unchanged until a
new result clears both the statistical and exact-quality gates.

## Next steps

1. Keep REAP on generic MLX Q4/G64 with custom G64, G64 slabs, and stream-pack
   off. Do not reintroduce the removed native prototype.
2. Review the uncommitted code changes against the current control and split
   safe compatibility work from rejected experiments before committing.
3. Complete the pending manual QSA/prefill validation without changing the
   benchmark protocol.
4. If investigating G64 or REAP performance, run the short digest gate first,
   then the long-turn quality gate, then the reversed interleaved benchmark.
5. Record new evidence in `research.md`, update this handoff only with the
   resulting operational decision, and state clearly whether the result is
   accepted, unresolved, or rejected.

Our immediate objective is reproducible, exact-quality MLX behavior. Speed
claims follow evidence; they do not define the control.
