# Flash-Next overview

## Purpose

Flash-Next runs a sparse 176B Qwen model on a 16 GB Apple Silicon Mac. Large sparse tensor families stay on SSD. Flash-Next is the primary
MACQWEN runtime.

## Model and checkpoint

The installed checkpoint is `Vontra/Qwen3.8-Flash-Next-MLX-oQ4`. It has 111.7 GB of model weights across 22 safetensors shards, quantised
from the official BF16 weights with a 4-bit base and 228 protected modules at 5 and 8 bit. It is the quality baseline.

`Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` is supported and is not installed. It has 86.2 GiB across 19 shards with a 3-bit base and 746
protected modules. It runs faster and fails a code task that oQ4 passes, so it was removed from the reference machine. See the checkpoint
quality section below.

The runtime saves an explicit `--checkpoint` choice and otherwise selects the sole complete compatible local checkpoint.

The model has 512 experts per routed layer. Each token uses a small expert set. The large hashed n-gram table is also sparse at lookup time.

## Runtime structure

The loader replaces large tensors before MLX materializes them. It streams selected expert rows and n-gram rows from the original shards.

The dense model core stays resident.

## oQ4 results

These are the current measurements. The oQ3-MTP sweep finished: it failed the trajectory gate, so oQ4 stayed.

- Baseline resident memory was about 4.19 GB on the tested M4 Mac.
- Load time was about 2.1 seconds.
- Prefill speed increases with prompt length. A prompt near 5,000 tokens may
reach about 40 to 50 tokens per second under favorable conditions.
- Ten fixed-prompt exact-quality arms measured a 2.59 tokens-per-second
  pinned-tail mean before the buffer-chunk2 change. Their range was 2.42 to
  2.73. The older harness reloads the model for each arm and starts colder.
- The accepted clean-boot `buffer-chunk2` comparison measures 2.83 tokens per
  second for `gen`, 2.70 for the pinned tail, and 457.7 MB of physical reads
  per token. It wins 10 of 12 pairs and preserves token IDs.
- Prefill is faster only because it amortises. A layer reads each distinct
expert once and serves every token in the prompt with it, so sixteen times the tokens cost 1.94 times the bytes. The drive rate falls as
prefill speeds up, from 1.40 to 0.82 GB/s, while decode sustains 1.06 GB/s.
- A synthetic fixed-route test measured a 5.33 tokens-per-second all-resident
read ceiling. It does not generate the model's real reply.
- Cache-aware routing measures 2.91 tokens per second and a 2.92 pinned tail
on the standard harness, against 2.73 and 2.70 for exact routing. Physical reads fall 16.2 percent, from 430.0 to 360.4 MB per token. The
gain is 6.5 percent against a 0.6 percent resolution band. It leads in all six paired arms and reads fewer bytes in all six. This is the fastest
measured exact-weight result on this machine. Measured under greedy decoding, as every benchmark here is.
- Cache-aware failed the trajectory gate under greedy decoding. Asked for a
SketchUp extension at `xhigh` it got stuck repeating one question and never produced a file, where exact-quality on the same prompt and
settings reached working code. Greedy causes repetition on its own and Qwen advises against it, so that comparison needs redoing with the
recommended sampler. Cache-aware stays opt-in until it does.
- Cache-aware routing changes expert choices. A small factual gate passed. A
long-context comparison stayed coherent, but the exact-quality answer was better. Exact-quality remains the default.
- Exact sessions restore recurrent and attention state without replay.
- QSA chunking bounds large temporary masks.
- Selective expert residency reduces runtime variance when RAM is available.
- A local RMSNorm patch corrects an upstream checkpoint interpretation error.
- The 2026-09-01 timing sweep closes host-only bookkeeping, routed
  `gather_qmm`, and the complete-runtime compile estimate. They recover about
  4.16 ms, measure 13 to 16 ms, and save about 1 ms per token respectively.
- A 12-arm test gives `buffer-chunk2` a resolved 6.3% generation gain over the
  concatenate path. Token IDs match and physical bytes fall. Issue #26 is
  closed.
- The Metal trace measured about 149 ms of GPU execution and 257 ms of drive
  reading in one state. A later clean-boot comparison measured 182.5 ms GPU
  busy in production and 86.1 ms with zero drive, so GPU busy is not a fixed
  model cost. IOKit values are relative only.
- A 60-token context sweep found no decode-rate decay with context. Short
  prompts include a warm-up transient: about 40 tokens near 2.9 to 3.2 tok/s,
  then about 1.9 tok/s as the working set widens.
- A reusable destination ring is bit-exact but unresolved on speed. It reads
  397.4 MB/token against 397.9 with fresh buffers and stays disabled.
- A traced miss sweep finds a GPU-busy hump. It peaks near 25% to 50% missing
  reads, while token time stays close to linear in physical bytes.
- VM counters show page-ins at about 33 pages/MB, with flat reclaim,
  compression, and swap during decode. Read-ahead off is 1.3% faster, but the
  result does not clear a band.
- The default Metal residency set is 750 MB. Raising it fivefold changes
  neither token time nor VM counters.

The reference machine is an M4 Mac with 16 GB of unified memory and a 256 GB SSD. Prefill and decode measurements are separate. Short
prompts have lower prefill rates because fixed setup and streamed-read costs apply to fewer tokens. Performance also depends on prompt type,
RAM pressure, SSD state, and page cache. Free memory matters most: a run that starts with a cold page cache is slower than the next one on
the same code.

## Checkpoint quality

oQ3-MTP has no harness baseline. Its rates come from single chat turns. At a context near 5,000 tokens it reaches a 2.55 tok/s median `gen`
and 2.65 `tail`, against 2.1 and 2.2 for oQ4 exact-quality on comparable turns. Long prefill is 45.2 against 45.0. The runs are unpaired and
nothing controlled the machine state, so treat the gap as directional. The per-turn series is in the research log. Publishing an oQ3-MTP
rate needs a `bench_production` run first.

oQ3-MTP has lost some recall of external APIs. Asked for a SketchUp extension, oQ4 produced a file that worked; oQ3-MTP produced broken
files at both `low` and `xhigh`. At `xhigh` it called `Sketchup::Face#extrude`, which doesn't exist, and never mentioned the real method
across 34,203 characters of reasoning. At `low` it used the right method with the wrong second argument.

An external [oQ4-MTP report](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ4/discussions/2) describes a different failure on long agentic and coding turns. The model entered a repetition loop, reached `max_tokens`, and cut off an open tool call. The report did not reproduce the failure with oQ3-MTP under matching settings. It used oMLX on an M5 Max, so it does not replace our local gate. The maintainer is investigating. MACQWEN keeps MTP disabled in production.

Prose did not expose this loss. Long Portuguese analysis stayed coherent on both checkpoints. A code task that requires a real API exposed
it.

Use a similar code task for each checkpoint gate.

## Routing profiles

| Profile | Purpose |
|---|---|
| `standard` | Threshold `0.85` without selective pinning |
| `exact-quality` | Same trajectory with selective file-backed residency |
| `cache-aware` | Exact-quality base with near-equal resident substitution |
| `fast` | Aggressive approximate routing |
| `fast-quality` | Approximate routing with quality recovery |
| `fused-quality` | Experimental one-shot draft; failed its reasoning gate |

Threshold `1.0` keeps the shipped router selection.

## Status

The text runtime, six routing profiles, exact sessions, and shared chat integration are active. Cache-aware is optional. The production
backend keeps the included MTP weights disabled. Current performance work is
tracked by issues #24, #25, and #33 through #39, plus #41 and #42. Issues #21 through #23 and
#26 through #27 are closed.
