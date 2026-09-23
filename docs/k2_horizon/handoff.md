# K2-Horizon 7B reference

Read this file, [`research.md`](research.md), and [`brief.md`](brief.md) before
changing or measuring K2-Horizon.

## Environment and launch

The launcher uses the selected environment, then the project `.venv`. Override
the interpreter with `MACQWEN_K2_HORIZON_PYTHON`.

Start the installed checkpoint with:

```bash
./chat.sh --model k2-horizon --checkpoint k2
```

The `k2` alias resolves to `~/models/K2-Horizon-7B-MLX-8bit` under
`MACQWEN_MODEL_ROOT`. We can also pass a full compatible checkpoint path. The
checkpoint contains executable `model.py`; use only a source we trust.

## Current defaults

| Setting | Retained value |
|---|---|
| Prefill step | 512 |
| KV cache | Native BF16 `KVCache`, 256-token growth |
| Allocator cache cap | Off |
| Post-generation `mx.clear_cache()` | Off |
| MLX `wired_limit` | Off |
| Context retention | Full |

K2 maps the shared `xhigh` reasoning request to the checkpoint's native
`high` effort. Model-specific JSON tool calls are translated to the shared
`<tool_call>` protocol at the backend boundary.

## Main files

| Path | Responsibility |
|---|---|
| `models/k2_horizon/backend.py` | Resident generation, cache tape, cleanup, and sessions |
| `models/k2_horizon/checkpoint.py` | Compatible checkpoint discovery and aliases |
| `models/k2_horizon/protocol.py` | Reasoning and tool-call protocol translation |
| `models/k2_horizon/settings.py` | Model-owned environment and session defaults |
| `models/k2_horizon/tests/bench/bench.py` | Fresh-process paired benchmark harness |
| `models/k2_horizon/cache.py` | Testable instance-level KV growth selection |
| `docs/k2_horizon/research.md` | Measurements, decisions, and rejected work |
| `results/k2_horizon/` | Commands, table, and retained raw records |

## Validation

Run checkpoint-free K2 tests:

```bash
.venv/bin/python -m unittest discover \
  -s models/k2_horizon/tests/unit -t . -p 'test_*.py' -q
```

Run affected shared tests and compile checks:

```bash
.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'
.venv/bin/python -m compileall -q macqwen models/k2_horizon
```

A live model comparison must use the interpreter, checkpoint, prompts, token
limits, sampler, reasoning effort, and source fingerprints recorded by the
harness. Use fresh child processes, at least three arms per condition, and
forward/reverse/forward ordering. Do not add thermal warmup loops on the
fanless reference Mac.

## Correctness invariants

- Every completed cache layer offset must match the conversation token tape.
- Stop-token rewind, EOS, token-limit completion, and valid manual
  cancellation must leave the cache aligned with the tape; partial prefill,
  callback failure, reset, and replay must leave explicit recoverable state.
- Cached user turns, JSON tool results, session restore, and prefix divergence
  must preserve the shared chat contract.
- Exact candidates must retain intermediate values and complete greedy token
  digests. Output-changing candidates require the sampled quality gate from
  `CONTRIBUTING.md`.
- Long-context checks must retain facts near the beginning, middle, and end.

## Closed directions

Do not retry these without a new measured premise:

- prefill step 256 as the default;
- MLX `wired_limit`;
- allocator cap or post-generation clear while the pool remains negligible;
- larger cache steps or a reserve cache without a cache-growth trace;
- compilation, custom normalization, projection packing, or another kernel
  without a compute/submission trace.

Lower precision, layer skipping, sliding windows, context truncation,
vocabulary shortlists, changed reasoning budgets, disk-offloaded KV, KV
recomputation, and speculative decoding are separate quality or scope changes.

## Next work

Our backend profiling finds about 0.100% of decode time in text decoding and
protocol translation. One Metal diagnostic shows nearly continuous active GPU
intervals during decode, but does not identify individual kernel costs.

Next, attribute kernel costs and the unexplained prefill interval gap before
selecting an optimization. These profiles exclude terminal UI. Reopen one
bounded candidate at a time and record the evidence with its raw arms; you
decide what promotes. Record evidence in [`research.md`](research.md) and
raw arms under [`results/k2_horizon/`](../../results/k2_horizon/).
