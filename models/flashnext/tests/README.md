# Flash-Next live-test cases

Start the project-owned terminal from the repository root:

```bash
./tests/run.sh --model flashnext --checkpoint PATH
```

This directory supplies Flash-Next `case_*.py` providers. The shared terminal
owns discovery, prompts, confirmation, execution, display, interruptions, and
JSONL result storage. The compatibility launcher
`./models/flashnext/tests/run.sh` forwards to the same terminal.

The interactive commands are:

```text
/help
/list
/show CASE
/run CASE
/results
/status
/quit
```

Do not add a second result directory or a runtime-owned terminal. Follow the
[measurement standard](../../../docs/measurement-standard.md) for retained
evidence.

## Case contract

The catalog discovers every `case_*.py` file; no central registry change is
needed. A case file must provide `TEST`, `TESTS`, or `get_tests()`. Each
returned `TestSpec` needs:

- a unique ID and title;
- a plain explanation and proposal rationale;
- metrics and controls;
- a source reference; and
- an executable function for runnable tests.

Cases may add environment controls, a live-metric parser, or a custom
interpreter. The catalog keeps three evidence levels: runnable retained
benchmarks, verification/manual quality entries, and historical entries from
the research record. Removed prototypes remain visible as non-runnable
history.

Every runnable case displays its purpose, rationale, controls, expected
metrics, command, per-arm results, interpretation, and JSONL record.

## Flash-Next controls

Historical Q4/G32 cases retain their original controls for provenance. Current
REAP Q4/G64 uses the G64 Metal executor by default; set
`FLASHNEXT_METAL_G64=0` for the generic-MLX rollback. G64 slabs and stream-pack
remain off. New comparisons use greedy decoding and exact digests. Quality
checks use `chat.sh`, normal sampling, and explicit effort settings.

The future paired G64 quality comparison predeclares seeds 7, 19, and 73,
alternates arm order, keeps slabs and stream-pack off, and requires completed
outputs plus blind scoring of the complete SketchUp `.rb` artifact. Seed 42 is
a known regression case, not a representative quality seed. An interrupted
generation is incomplete evidence, not a quality failure.

Run trusted performance tests in Apple Terminal. The suite never requires a
reboot; VM quiescence and file-cache purge remain optional diagnostics.
