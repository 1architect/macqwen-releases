# Project live-test terminal

Start the shared terminal from the repository root:

```bash
./tests/run.sh
```

The terminal uses the same checkpoint discovery and model selection as
`chat.sh`, then discovers cases from `models/<model>/tests/cases/`. It owns
prompts, confirmation, execution, display, interruptions and canonical JSONL
records. Model folders provide only cases, benchmarks and an optional
environment hook.

Use `--model` and `--checkpoint` to skip interactive selection. Each run writes
`record.jsonl`, `output.log` and its artifacts to
`results/<model>/<YYYYMMDD-HHMMSS>-<test-id>/`. See
[docs/testing.md](../docs/testing.md) and the
[measurement standard](../docs/measurement-standard.md).
