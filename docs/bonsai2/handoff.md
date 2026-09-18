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

Screening runs may use the `smoke` fixture with two rounds for directional
evidence only. Screens never promote; promotion keeps the three-arm rule
with a production fixture.

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
- One model runs per process. The transform hooks patch process-global
  runtime state, so multi-model residency in one process is unsupported.

## Closed directions

Do not retry these without a new measured premise:

- loading through stock `mlx_lm.load` without the bundled runtime;
- vision input before the text-only baseline passes its quality gate.

## Next work

Baselines are established: 5.5 tok/s product at 2k with 10.43 GB peak, 4.0
tok/s at 16k with 13.42 GB peak. Open items are the 16k allocator
confirmation, the full shared-transform comparison, stop-retention
continuation checks, and 8-bit KV quality validation. Record evidence in
[`research.md`](research.md) and raw arms under
[`measurements/`](measurements/).
