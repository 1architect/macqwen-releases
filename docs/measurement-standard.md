# Live model measurement standard

We use one project-owned test terminal and one append-only JSONL format for
live model evidence. Runtime folders provide cases and metric adapters; they
do not own terminal UI, result placement, or run lifecycle.

## Run a test

Start the terminal from the repository root:

```bash
./tests/run.sh
```

It discovers compatible checkpoints using the same rules as `chat.sh`, asks
which runtime and checkpoint to use when necessary, and loads cases from
`models/<runtime>/tests/cases/`.

Every run writes to its own folder:

```text
results/<runtime>/YYYYMMDD-HHMMSS-<experiment>/record.jsonl
results/<runtime>/YYYYMMDD-HHMMSS-<experiment>/output.log
```

[docs/testing.md](testing.md) describes the folder, the `macqwen.results` API
and the policy test that enforces it.

Scratch logs and output directories do not support published claims. Existing
records with older names remain valid historical evidence when their metadata
and provenance are clear.

## Record rules

- Use schema 1 and the shared measurement engine.
- Write the `run` record before arms, then persist every raw `arm` before
  validation or interpretation.
- Preserve validation failures, interruptions, and partial runs. Mark them as
  incomplete or diagnostic; never present them as promotion results.
- Complete comparisons end with a `summary`. An interrupted or diagnostic file
  may end earlier, but its `run` metadata must explain why.
- Record the checkpoint, source fingerprints, harness, prompt, template,
  sampler, effort, token limits, ordering, environment, and cache conditions.
- Greedy comparisons require exact token digests. Sampled quality comparisons
  record seeds, sampler, completion status, and scoring information.
- Use fresh child processes and at least three arms per condition. Alternate
  forward and reverse arm order to reduce ordering and file-cache effects.
- Report paired effects and the measured resolution band. A result inside the
  band is unresolved.
- Do not hammer the machine with tests, especially on fanless hardware. Use
  the smallest relevant controlled run, avoid redundant reruns and warmups,
  and stop a branch once its correctness, execution-path, or feasibility gate
  decisively fails. Repeat only when a new premise or explicit user
  authorization justifies the cost.

Missing historical provenance is `unknown`. We do not reconstruct it from
filenames, timestamps, or assumptions.
