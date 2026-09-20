# Bonsai-2 ternary 27B reference

Read this file, [`research.md`](research.md), and [`brief.md`](brief.md) before
changing or measuring Bonsai-2.

## Environment and launch

The launcher uses the selected environment, then the project `.venv`. Override
the interpreter with `MACQWEN_BONSAI2_PYTHON`.

Start the installed checkpoint with:

```bash
./chat.sh --model bonsai2 --checkpoint b2
```

The `b2` alias resolves to `~/models/Ternary-Bonsai-2-27B-mlx-2bit` under
`MACQWEN_MODEL_ROOT`. The checkpoint requires its bundled `runtime/` loader;
use only a source we trust. Milestone 1 is text-only.

Two startup messages are expected advisories, not errors. Transformers logs
that `prism_hadamard_qwen35` has no registered model class; we never
instantiate a transformers model, only its tokenizer, so this changes
nothing. We pass `fix_mistral_regex=True` when loading the tokenizer, which
silences the regex-pattern advisory and produces identical token IDs to the
default on our English, code, Portuguese, and tag probes.

## Current defaults

| Setting | Retained value |
|---|---|
| Prefill step | 512 |
| Allocator cache cap | 256 MB in chat (backend default off; bench control is uncapped) |
| Fused FWHT | On by default; `fused_fwht=False` is the explicit stock rollback |
| Prepared QMM metadata | On by default; `prepared_qmm_metadata=False` is the explicit stock rollback |
| Generation | Text-only, vision unloaded |
| Context retention | Full |

## Main files

| Path | Responsibility |
|---|---|
| `models/bonsai2/backend.py` | Resident generation, cache tape, cleanup, and sessions |
| `models/bonsai2/checkpoint.py` | Compatible checkpoint discovery and aliases |
| `models/bonsai2/protocol.py` | Reasoning and tool-call protocol translation |
| `models/bonsai2/settings.py` | Model-owned environment and session defaults |
| `models/bonsai2/bench.py` | Fresh-process paired benchmark harness |
| `models/bonsai2/cache.py` | Testable instance-level KV growth selection |
| `docs/bonsai2/research.md` | Measurements, decisions, and rejected work |
| `docs/bonsai2/measurements/` | Commands, table, and retained raw records |

## Validation

Run checkpoint-free Bonsai-2 tests:

```bash
.venv/bin/python -m unittest discover \
  -s models/bonsai2 -p 'test_*.py' -q
```

Run affected shared tests and compile checks:

```bash
.venv/bin/python -m unittest discover -s macqwen -p 'test_*.py'
.venv/bin/python -m compileall -q macqwen models/bonsai2
```

A live model comparison must use the interpreter, checkpoint, prompts, token
limits, sampler, reasoning effort, and source fingerprints recorded by the
harness. Use fresh child processes, at least three arms per condition, and
forward/reverse/forward ordering. Do not add thermal warmup loops on the
fanless reference Mac.

Do not hammer the fanless machine with tests. We run only the smallest
controlled set relevant to the active question, stop when a correctness,
execution-path, performance-resolution, or compiler-feasibility gate is
decisively closed, and do not repeat a benchmark without a new premise or
explicit user authorization. Interrupted records remain evidence of what was
actually attempted; they are not a reason to immediately restart the same
work.

Screening runs may use `context-1k` with two reverse-interleaved rounds
(four total arms) for directional evidence only. This is the maximum context
for the current prefill screens because larger prompts caused paging and yellow
memory pressure on the fanless machine.

The 0.4.5 default accepts the 1k prepared-QMM evidence: decode `+21.07%
±2.24` with matched provenance, complete 401/401 coverage, and matching
greedy digests. Prefill was unresolved at 1k (`+1.28% ±3.52`), and a
3,283-token probe showed prefill `-12.86%` with higher peak memory. The
decode win stands at short context; the long-context cost stays open.
Rollback is `prepared_qmm_metadata=False`.

Long-context arms are conditional. Run an 8k, 16k, or other materially long
fixture only when it is relevant to the current research question and the
user has explicitly authorized that run. Otherwise do not launch or wait for
it; record the deferral and stop with the shorter evidence. Preserve
append-only records from interrupted authorized arms, but do not restart
them without new explicit authorization.

## Correctness invariants

- The bundled `runtime/` loader must be active on every arm; stock-loader runs
  are invalid because they skip the activation transform silently.
- Every completed cache layer offset must match the conversation token tape.
- Stop-token handling, EOS, token-limit completion, cancellation, callback
  failure, reset, and replay must leave explicit recoverable state. Stop
  retention stays experimental and off by default.
- Exact candidates must retain intermediate values and complete greedy token
  digests. Output-changing candidates require the sampled quality gate from
  `CONTRIBUTING.md`.
- Fused-attention benchmark arms must record per-backend execution-path
  counters before validation. The fused candidate must select at least two
  calls, attempt every attention call, and keep fallback coverage at or below
  10%; the stock control must show zero fused selections. Older records
  without these counters are not reinterpreted.
- One model runs per process. The transform hooks patch process-global
  runtime state, so multi-model residency in one process is unsupported.

## Closed directions

Do not retry these without a new measured premise:

- loading through stock `mlx_lm.load` without the bundled runtime;
- vision input before the text-only baseline passes its quality gate.

## Current handoff — 2026-09-19

We completed the pending P1/P2 correctness batch: natural reasoning closure
now disarms before lookahead can force a duplicate close; tool-result content
uses the safe encoder and malformed sentinel renders fail closed; accepted
tool formats preserve payload bytes and literal protocol tags; benchmark
fixtures and digest references are filtered by clean-arm state; and the Bonsai
and K2 protocol doubles accept `enable_thinking`.

The checkpoint and budget fixes are covered by the current tests. The shared
suite passes 300 tests, the Bonsai suite passes 114 tests, and compile checks
pass in the managed Python 3.12 MLX runtime with the declared
`mlx-vlm==0.6.17` and Pillow dependencies.

We also completed a low-context check with three fresh
control children: 3,283 prompt tokens, 32 output tokens, and 7.33, 7.34,
and 7.38 tok/s with matching greedy digests. The raw record is
[`20260919-low-context-short-check.jsonl`](measurements/20260919-low-context-short-check.jsonl).
This is diagnostic evidence only: VM swap activity occurred, and the current
physical-read field includes prefill and decode. Revalidate the sustained and
long-context baselines before making a promotion claim.

The new question-only diagnostic follows FlashNext's short speed-test prompt,
`Explique a fotossintese em duas frases.`, with no records, tools, or prior
turn. Its rendered prefill is 23 tokens including Bonsai's chat framing. Three
fresh greedy 32-token control arms measured 7.226, 7.196, and 7.173 tok/s
(median 7.196), with identical complete digests and no validation failures.
This is a minimum-context workload check only; it does not replace the
3,283-token baseline or promote either experimental kernel. Evidence is in
[`20260920-102501-baseline-question-short.jsonl`](measurements/20260920-102501-baseline-question-short.jsonl)
and
[`20260920-102501-baseline-question-short-arms.jsonl`](measurements/20260920-102501-baseline-question-short-arms.jsonl).

We do not launch or wait for long-context arms unless they are relevant to the
active question and the user explicitly authorizes them.

## Next work

The 5.5 tok/s 2k product and 4.0 tok/s 16k figures remain historical
references. The current 7.34 tok/s short check does not replace the sustained
baseline. Revalidate the 256-token fixture and other short evidence on a
clean revision as needed. Treat the 19.5k/16k work, full shared-transform
comparison, stop-retention continuation checks, and 8-bit KV quality
validation as conditional research: launch a long-context arm only when it
is relevant to the current question and explicitly authorized by the user.
Do not wait on deferred long-context work. Record new evidence in
[`research.md`](research.md) and raw arms under [`measurements/`](measurements/).

## Current handoff — 2026-09-20 QMM preparation result

The opt-in `prepared_qmm_metadata` candidate prepares FP32 scales and biases
once for the 401 non-embedding packed projections, adding about 0.745 GiB of
resident metadata while retaining packed weights and stock QMM arithmetic.
It passed the short question-only execution-path and greedy-digest checks:
13,634 prepared calls per completed candidate arm, zero unsupported calls, and
9.298/9.290 tok/s versus 7.586/7.332 tok/s for stock over two screening
pairs (+24.64%, ±4.13 percentage points). This remains screening evidence,
not a three-arm promotion result.

The authorized 3,283-token prefill check failed the useful prefill hypothesis
on its first complete pair: stock prefill was 135.190 s and the candidate was
254.755 s (−88.44%). The greedy digest still matched, but the extra metadata
did not produce a complete-runtime win; the candidate is not promoted. The
second candidate arm was interrupted during decode and its partial evidence
was preserved. We do not repeat this load on the fanless machine without a
new premise and explicit authorization.

The harness now exposes bounded arm phase progress, candidate/fallback
counters, immediate execution-path validation, and explicit cancellation
propagation. Historical measurement files remain append-only, and old fused
records without path counters remain provenance-unknown rather than being
reinterpreted.

## Prefill test safety limit — 2026-09-20

The runnable Bonsai prefill probes now default to `context-1k` and two
reverse-interleaved rounds. The 3,283-token QMM attempt remains a preserved
interrupted measurement, not a template for future runs. We do not launch
`context-2k`, 8K, or 16K prefill arms from the test catalog without a new,
relevant question and explicit user authorization.
