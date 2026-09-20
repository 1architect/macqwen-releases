# Live model measurement standard

We use one project-level test terminal and one append-only JSONL record format
for live model tests. Runtime folders provide test cases and metric adapters;
they do not own terminal UI, result placement, or lifecycle recording.

## Running a test

Start the project terminal with:

```bash
./tests/run.sh
```

The terminal discovers compatible checkpoints using the same selection rules as
`chat.sh`, asks which model/checkpoint to test when necessary, and then loads
tests from `models/<runtime>/tests/`.

Retained artifacts belong under:

```text
docs/<runtime>/measurements/YYYYMMDD-HHMMSS-<experiment>.jsonl
```

Scratch logs are not published evidence.

## Record rules

Every run uses schema 1 and records a `run`, each raw `arm`, validation/failure
records, and a final `summary`. Raw arms are durable before validation. Greedy
comparisons require exact token digests; sampled quality runs record their seed,
sampler, completion status, and scoring information.

Common measurements live under `metrics.common`. Runtime-specific values live
under `metrics.<runtime>`. Missing historical provenance is recorded as
`unknown`; agents must not reconstruct it from filenames or assumptions.

Comparative tests use fresh child processes and reverse-interleaved arms. A
failed or interrupted run remains useful evidence about what happened but is
not a promotion result.
