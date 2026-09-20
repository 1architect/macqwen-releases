# Contributing to MACQWEN

## Before changing code

Read the active handoff and research record for the component you will touch.
Then read its brief for scope and support status.

For Flash-Next, also read [`AGENT_INVARIANTS.md`](docs/flashnext/AGENT_INVARIANTS.md).
For live model work, read the [measurement standard](docs/measurement-standard.md).

Each active component uses the same documents:

| Document | Content |
|---|---|
| `brief.md` | Purpose, scope, status, and main results |
| `research.md` | Dated measurements, decisions, and rejected approaches |
| `handoff.md` | Commands, constraints, validation, and next work |

Update the existing active record instead of adding a session report. Move
superseded material to `docs/archive/` only when the active record remains
complete. Keep historical measurements and raw records unchanged.

## Documentation style

- Use plain language, present tense, and the project voice: “we”, “us”, and
  “our”.
- Put current instructions in handoffs, not in research history.
- Date historical findings and label diagnostic, incomplete, rejected, and
  unverified results explicitly.
- Use the same runtime names everywhere: Flash-Next, K2-Horizon, Bonsai-2,
  and Qwen3.8-27B. Keep lowercase identifiers only in code and paths.
- Prefer one short paragraph or table over repeated explanations.

## Measurement rules

- Start the project terminal with `./tests/run.sh` for retained live-model tests.
- Use at least three arms per condition and alternate arm order in reverse
  rounds to reduce ordering and cache effects.
- Keep the checkpoint, prompt, template, sampler, reasoning effort, and token
  limits fixed across a comparison.
- Use a fresh child process per arm. Avoid unnecessary warmups on the fanless
  reference machine.
- Confirm that every setting took effect and that the measured reads came from
  the SSD when the claim depends on physical I/O.
- Publish raw arms before validation. Preserve failures and interruptions;
  never infer missing provenance from filenames or assumptions.
- Report paired effects with the measured resolution band. A result inside the
  band is unresolved, not evidence of no effect.
- Use greedy decoding and exact token digests for exact comparisons. Use the
  recommended sampler and explicit seeds for sampled quality comparisons.
- Do not use greedy benchmark output as a chat-quality conclusion.
- Do not calculate drive bandwidth from total token time; the drive is idle
  during other parts of a token.
- Do not repeat overlap, prefetch, or read-ahead work without a new mechanism
  or evidence; those approaches lost to memory-controller contention here.

The measurement standard defines the JSONL schema, filename rules, and how to
label complete, diagnostic, failed, and interrupted records.

## Quality gate

Use a quality gate when a change can alter model output. This includes
checkpoints, routing, quantization, speculation, and approximations.

Use this prompt for every retained quality-gate arm:

```text
crie uma extensão para sketchup que extrude várias faces ao mesmo tempo até uma altura definida pelo usuário. produza o código para eu salvar em um arquivo .rb
```

Run it at `medium` and `high` effort with sampling enabled. Add `xhigh` when
the change can affect long reasoning. Use the same seed and settings across
conditions. Score the complete `.rb` file, not only the named API method.

The current Flash-Next G64 comparison has an explicit short exact-digest
promotion exception; its long-turn quality gate remains open. Follow the
[Flash-Next handoff](docs/flashnext/handoff.md) rather than copying its
checkpoint-specific controls to another runtime.

## Repository rules

- Keep unrelated user changes intact. Do not reset or discard them implicitly.
- Run unit tests with `.venv/bin/python` after `./chat.sh setup`.
- Use `./tests/run.sh` for the project live-test terminal; do not start a
  runtime-owned terminal for retained evidence.
- Keep raw measurement records append-only under `docs/<runtime>/measurements/`.
- Do not add generated `Co-Authored-By` trailers.
- Use plain commit titles that start with a capital letter; do not use
  conventional-commit prefixes.
- Push only validated release changes to the public repository.
- Do not benchmark while another workload uses the SSD or unified memory.
