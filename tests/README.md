# Project live-test terminal

Start the shared test terminal with ./tests/run.sh.

It uses the same installed-checkpoint discovery and model selection behavior as
chat.sh, then discovers live-test providers from models/<runtime>/tests/.
Runtime folders provide case_*.py definitions; the project terminal owns the
prompt, confirmation, execution, display, and canonical JSONL measurement
records.

Use --model and --checkpoint to select a model without the interactive choice.
Retained records are written under docs/<runtime>/measurements/.
