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

The retained gate requests a SketchUp extension that extrudes selected faces.
The extension must request a height and load as a `.rb` file.

Run the gate at `medium` and `high` effort with sampling enabled.
Use identical settings for both conditions.
Add `xhigh` when the change can affect long reasoning.

Check the complete file, not only the named API method.
The recorded oQ3-MTP test used `pushpull` with an invalid second argument.
The oQ4 checkpoint produced a working file in the same test.

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
