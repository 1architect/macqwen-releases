# Contributing to MACQWEN

## Required reading

Read the current documents for the component before changing or measuring it.

For Flash-Next, use this order:

1. `docs/flashnext/handoff.md`
2. `docs/flashnext/research.md`
3. `docs/flashnext/brief.md`
4. `README.md`
5. `docs/README.md`

The research record contains rejected approaches and prior measurements.
Search it before starting an experiment.

For K2-Horizon, use the equivalent active set:

1. `docs/k2_horizon/handoff.md`
2. `docs/k2_horizon/research.md`
3. `docs/k2_horizon/brief.md`

For Bonsai-2, use the equivalent active set:

1. `docs/bonsai2/handoff.md`
2. `docs/bonsai2/research.md`
3. `docs/bonsai2/brief.md`

## Documentation structure

Each active component has three documents:

| Document | Content |
|---|---|
| `brief.md` | Purpose, scope, status, and main results |
| `research.md` | Measurements, decisions, and rejected approaches |
| `handoff.md` | Commands, constraints, validation, and next work |

Update an existing document instead of adding a session report.
Move superseded material to `docs/archive/` only when the active record stays complete.

## Measurement rules

- Use at least three arms for each condition.
- Keep prompts, token limits, sampling, and reasoning effort constant.
- Read the resolution band before interpreting a difference.
- Confirm that each tested setting took effect.
- Confirm that the SSD served the measured reads.
- Use the complete runtime path for layout and throughput claims.
- Publish rates only from a retained benchmark harness.
- Record memory pressure, swap state, and cache conditions.
- Do not require a reboot for measurements. Close unrelated workloads, use a
  file-cache purge only when the experiment needs a cold cache, and require a
  clean VM-counter and load window before measurement.
- Run only one model during a benchmark.

Do not calculate drive bandwidth from complete token time.
The drive stays idle during other parts of each token.

Do not retry overlap, prefetch, or read-ahead without new evidence.
These methods lost to memory-controller contention on the reference Mac.

Benchmarks use greedy decoding to compare token IDs.
Chat uses Qwen's recommended sampler.
Do not use greedy benchmark output for chat-quality conclusions.

## Quality gate

Use a quality gate when a change can alter model output.
This includes checkpoints, routing, quantization, speculation, and approximations.

Use this exact prompt for every retained quality-gate arm:

```text
crie uma extensão para sketchup que extrude várias faces ao mesmo tempo até uma altura definida pelo usuário. produza o código para eu salvar em um arquivo .rb
```

Run the gate at `medium` and `high` effort with sampling enabled.
Use identical settings and the same explicit `--seed` for both conditions.
Add `xhigh` when the change can affect long reasoning.

Check the complete file, not only the named API method.
The recorded oQ3-MTP test used `pushpull` with an invalid second argument.
The oQ4 checkpoint produced a working file in the same test.

For the pending REAP G64 comparison, we retain Astra's recommendation: we
predeclare seeds 7, 19, and 73, pair the MLX-backed Metal runtime G64-off
versus G64-on, and alternate arm order. We keep slabs and stream-pack off in
both arms. We score the completed SketchUp `.rb` outputs blind to arm labels
using functional criteria: the file must load, use the correct SketchUp API,
extrude multiple faces, and honor the user-defined height. Seed 42 is a known
regression case, not a representative quality seed. An interrupted generation
is an incomplete gate, not a quality failure, and cannot be scored as a
completed answer.

## Repository rules

- Commit to `main` unless the repository policy changes.
- Do not add automated tools as contributors.
- Do not add generated `Co-Authored-By` trailers.
- Push only validated release changes to the public repository.

## Machine rules

- Run tests with `~/models/.venv-qwen4exp/bin/python`.
- Keep `~/models/.venv-qwen4exp` intact.
- Keep `~/mlx-qwen38-kernel-lab` intact.
- Do not benchmark while another workload uses the SSD or unified memory.
