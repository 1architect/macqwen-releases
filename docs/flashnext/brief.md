# Flash-Next brief

## Purpose

Flash-Next runs a 176B sparse Qwen model on a 16 GB Apple Silicon Mac.
The runtime leaves large sparse tensor families on SSD.

Flash-Next is the current MACQWEN model focus.

## Model and checkpoint

The tested checkpoint is `Vontra/Qwen3.8-Flash-Next-MLX-oQ4`.
It uses about 111.7 GB on disk and contains 22 safetensors shards.

The model has 512 experts per routed layer. Each token uses a small expert set.
The large hashed n-gram table is also sparse at lookup time.

## Runtime structure

The loader replaces large tensors before MLX materializes them.
It streams selected expert rows and n-gram rows from the original shards.

The dense model core stays resident. The measured baseline resident memory was
about 4.19 GB on the tested M4 Mac.

## Retained results

- Load time was about 2.1 seconds.
- Prefill speed increases with prompt length. A prompt near 5,000 tokens may
  reach about 40 to 50 tokens per second under favorable conditions.
- Ten fixed-prompt exact-quality arms measured a 2.59 tokens-per-second
  pinned-tail mean. Their range was 2.42 to 2.73. A separate two-run subset
  averaged 2.83, so 2.83 is not a ten-run baseline.
- The standard harness measures 2.713 tokens per second for a complete decode
  and 2.650 for the pinned tail, over ten kept arms at 390 MB of physical
  reads per token. Terminal `gen` on short chat turns runs near 2.0 to 2.5.
  The older 2.59 figure comes from a harness that reloads the model for each
  arm and therefore starts colder. The gap is page-cache state, not code.
- Prefill is faster only because it amortises. A layer reads each distinct
  expert once and serves every token in the prompt with it, so sixteen times
  the tokens cost 1.94 times the bytes. The drive rate falls as prefill speeds
  up, from 1.40 to 0.82 GB/s, while decode sustains 1.06 GB/s.
- A synthetic fixed-route test measured a 5.33 tokens-per-second all-resident
  read ceiling. It does not generate the model's real reply.
- Cache-aware routing measured 2.79 tokens per second against 2.54 for exact
  routing in one hot interleaved comparison. Paired arms improved by 8.3
  percent. Physical reads fell 16.8 percent, from 417.8 to 347.6 MB per token.
- Cache-aware routing changes expert choices. A small factual gate passed. A
  long-context comparison stayed coherent, but we preferred the
  exact-quality answer. Exact-quality remains the default.
- Exact sessions restore recurrent and attention state without replay.
- QSA chunking bounds large temporary masks.
- Selective expert residency reduces runtime variance when RAM is available.
- A local RMSNorm patch corrects an upstream checkpoint interpretation error.

The reference machine is an M4 Mac with 16 GB of unified memory and a 256 GB
SSD. Prefill and decode measurements are separate. Short prompts have lower
prefill rates because fixed setup and streamed-read costs apply to fewer
tokens. Performance also depends on prompt type, RAM pressure, SSD state, and
page cache. Free memory matters most: a run that starts with a cold page cache
is slower than the next one on the same code.

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

The text runtime, six routing profiles, exact sessions, and shared chat
integration are active. Cache-aware is optional. MTP remains research code.
