# Results

Every test and benchmark run writes here, and nowhere else.

```text
results/<model>/<YYYYMMDD-HHMMSS>-<test-id>/
  record.jsonl    canonical measurement record
  output.log      the complete stdout and stderr of the run
  ...             every file the benchmark writes
```

Runs started by hand are named `<stamp>-<script>-manual`. Files directly under
`results/<model>/` are older records moved from `docs/<model>/measurements/`;
each model's `LEGACY.md` describes them.

Do not edit or delete a record that a document cites. See
[docs/testing.md](../docs/testing.md) for the rules and the API, and the
[measurement standard](../docs/measurement-standard.md) for what a retained
record must contain.
