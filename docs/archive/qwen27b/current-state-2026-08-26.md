# Archived state: 2026-08-26

Updated: 2026-08-26
Archive status: historical snapshot. It does not describe the current runtime. `RESEARCH-LOG.md` records the related
experiments.

## Installed model

One model remained on this date. It was Qwen3.8-27B with the section 57 bit allocator.

```text
~/models/Qwen3.8-27B-Apple-MLX-V4-flat   13.05 GB
```

It averages 3.88 bits per parameter over 402 tensors. The lean loader keeps `embed_tokens` on SSD, so resident weights
are about 12.65 GB.
V4-flat holds every FFN tensor at 3 bits or more, and pays for that with less attention precision than the allocator
would otherwise buy:

```text
down_proj  3:61 4:3      gate_proj  3:36 4:28     up_proj  3:35 4:29
q_proj     4:14 5:2      o_proj     5:16          out_proj 5:48
k_proj     5:1  6:15     v_proj     5:4 6:7 8:5
```

V4-out held the same budget with 6-bit and 8-bit attention, paid for by dropping 38 FFN tensors to 2 bits. It was
deleted on 2026-08-26 before any A/B test ran, so the comparison is still unmade.

## Launch

```bash
./chat.sh
```

With no arguments the launcher lists the builds it finds on disk, then exits. It had no default. The user selected a
build by its directory suffix:

```bash
./chat.sh flat
```

The list is read from disk at every run, so a deleted build stops being offered instead of becoming a switch that fails
at load. `MACQWEN_MODEL=<path>` still overrides everything, and the vocabulary guard still refuses any model that is not
Qwen3.8-27B.

## Resident-memory limit

12.65 GB of resident weights runs. 13.05 GB swaps.
V5 at 13.05 GB reached 1.0 tokens per second with 1.81 GB of swap. V4 at 12.65 GB runs at 4 to 6 tokens per second with
no swap. Do not build past 12.6 GB on this machine.

## Required settings

`chat.sh` sets these and they are load-bearing:

```text
MLX_QMM_BK=32      never raise it, see research log 59
KV_BITS=4          quantize the KV cache from the first token
KV_START=0
PREFILL_STEP=256   smaller chunks, smaller prefill spike
--bf16-ends        SSD embedding and shortlist head
```

`mx.set_wired_limit` runs before `load()`. It did not, until 2026-08-26, so no earlier `WIRED=` measurement is valid.

## Phone access

```bash
~/mlx-qwen38-kernel-lab/bin/python3 -u web_terminal.py -- flat
```

`web_terminal.py` forks the chat under a PTY and mirrors it to the browser over SSE. The phone gets the same interface,
including approvals. The URL carries a token. Anything on the network that reaches this port can run shell commands on
the Mac, so never remove the token.

## Code-generation support

These checks ran outside the model and prevented invented APIs.

```text
context7.py    loads real API docs before generation, plus verified FACTS
api_guard.py   checks methods, constants, arity and argument types
code_check.py  compiles the file before the model answers
free_search.py Tavily first, then docs, ddgs, StackExchange, Wikipedia
```

`TAVILY_API_KEY` lives in `~/.frankenstein/env`, mode 600, outside the repo.

## Remaining model errors

Invented APIs are gone. Unit semantics are not. The last working run produced a SketchUp extension that loaded and ran,
and was 25.4 times too tall. The code was valid Ruby and every method existed, so no checker caught it.
The overnight run of 2026-08-26 produced no file. It ran on V5, which was swapping, and it looped on
`api_docs('extrude')` while asserting that `Face#extrude` exists. The correct method is `pushpull`.

## Removed from the repository

The llama.cpp backend is gone: `llama_cpp_engine.py`, `chat_whittle.sh`, `--backend llama-cpp` and every conditional
that served it. The Whittle GGUF it loaded was deleted, so the backend had nothing to run.
The Sparse MLP flags are gone: `--sparse-prefill`, `--sparse-generation`, `--sparse-plan`, `--sparse-top-k`. They
imported `v32_prefill_sparse_probe`, which is not in this repository, so passing any of them raised ImportError. Sparse
MLP had already failed its quality gate at 11.7 percent RMSE.
Also removed: `f16_moe_plan.py`, `moe_bit_plan.py` and `docs/V4-F16-MOE.md` (planning for the direction closed in
section 53), `dwq_corpus.py` (blocked in section 60), `overnight.sh` (built models that no longer exist), and one stale
eval output.
A copy of everything deleted is at `~/.frankenstein/attic/stripped-2026-08-26.tar.gz`.
`bench_decode.py` and `profile_prefill.py` stay although nothing imports them. They are held-out corpus text for
`eval_models.py` and `bits_vs_quality.py`. Perplexity is comparable across builds only while that corpus is
byte-identical, so do not delete or edit them.

## Closed directions

Do not retry these. Each has a measurement in the research log.

```text
f16 MoE streaming       section 53   model breaks at 25% of neurons kept
grafting stock weights  section 55   NLL 1.78 -> 5.75
low-rank selector head  section 56   73% recall at rank 256
DWQ on this machine     section 60   optimizer needs 19.8 GB, artifacts deleted
MLX_QMM_BK=64           section 59   breaks the Metal kernel
lazy mmap weights       section 58   GPU watchdog timeout mid-answer
```

## Next

1. A/B the bit floors against a pure-knapsack build of the same size.
2. Sensitivity calibration, to replace the activation-RMS proxy.
3. MTP speculative decoding. Weights are on disk. About 1.5x generation.
4. Add `ruby -w` to the guard for undefined-variable typos.
5. Rebuild a second V4 variant, so there is something to A/B against.
