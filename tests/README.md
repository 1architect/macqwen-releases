# Project live-test terminal

Start the shared terminal from the repository root:

```bash
./tests/run.sh
```

The terminal uses the same checkpoint discovery and model selection as
`chat.sh`, then discovers cases from `models/<runtime>/tests/`. The project
terminal owns prompts, confirmation, execution, display, interruptions, and
canonical JSONL records. Runtime folders provide only the cases and metric
adapters.

Use `--model` and `--checkpoint` to skip interactive selection. Retained
records belong under `docs/<runtime>/measurements/`. See the
[measurement standard](../docs/measurement-standard.md) before creating or
retaining a live run.
