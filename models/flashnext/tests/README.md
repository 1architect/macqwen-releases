# Flash-Next tests

```text
unit/     checkpoint-free unit tests; CI runs them
bench/    harness scripts that load a checkpoint, and their helpers
cases/    test-terminal cards, one runnable test per file
inputs/   operator-supplied inputs, such as chat-workload.txt
```

Start the project test terminal from the repository root:

```bash
./tests/run.sh --model flashnext --checkpoint PATH
```

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

Every run writes to `results/flashnext/<YYYYMMDD-HHMMSS>-<test-id>/`. Do not add
another results directory or a runtime-owned terminal. The general rules are
in [docs/testing.md](../../../docs/testing.md); retained evidence follows the
[measurement standard](../../../docs/measurement-standard.md).

## Case contract

The shared catalog discovers every `cases/case_*.py` file; no central registry change is
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

Historical Q4/G32 cases retain their original controls for provenance. The
installed Vontra Q4/G32 checkpoint runs the Metal runtime with the chat
defaults in `settings/launch.py` (slab pack, stream-pack, keep-warm, and the
exact bundle on); set an explicit environment value for the rollback of any
member. G64 slabs remain off. New comparisons use greedy decoding and exact
digests. Quality checks use `chat.sh`, normal sampling, and explicit effort settings.

The future paired G64 quality comparison predeclares seeds 7, 19, and 73,
alternates arm order, keeps slabs and stream-pack off, and requires completed
outputs plus blind scoring of the complete SketchUp `.rb` artifact. Seed 42 is
a known regression case, not a representative quality seed. An interrupted
generation is incomplete evidence, not a quality failure.

Run trusted performance tests in Apple Terminal. The suite never requires a
reboot; VM quiescence and file-cache purge remain optional diagnostics.
