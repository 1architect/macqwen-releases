# K2-Horizon runtime

This package owns K2-Horizon 7B checkpoint discovery, MLX-LM loading,
generation state, session defaults, protocol translation, benchmarking, and
tests. The shared chat keeps its model-independent `<think>` and `<tool_call>`
contract; we translate K2's native protocol at this boundary.

## Documentation

| Document | Purpose |
|---|---|
| [`docs/k2_horizon/brief.md`](../../docs/k2_horizon/brief.md) | Scope, support status, and main results |
| [`docs/k2_horizon/research.md`](../../docs/k2_horizon/research.md) | Measurements, rejected ideas, and decisions |
| [`docs/k2_horizon/handoff.md`](../../docs/k2_horizon/handoff.md) | Operation, validation, constraints, and next work |
| [`results/k2_horizon/`](../../results/k2_horizon/) | Exact commands, results table, and raw records |

Read the handoff and research record before changing or measuring this runtime.

## Run

```bash
./chat.sh --model k2-horizon --checkpoint k2
```

The checkpoint supplies executable `model.py` code. We only load a compatible
checkpoint from a source we trust.

## Test

```bash
.venv/bin/python -m unittest discover \
  -s models/k2_horizon/tests/unit -t . -p 'test_*.py' -q
```

## Package map

| Path | Responsibility |
|---|---|
| `backend.py` | Resident generation, cache lifecycle, and session replay |
| `checkpoint.py` | Checkpoint validation, discovery, and aliases |
| `protocol.py` | Reasoning and JSON tool-call translation |
| `settings.py` | Model-owned environment and session defaults |
| `tests/bench/bench.py` | Fresh-process paired benchmark harness |
| `tests/unit/test_*.py` | Checkpoint-free runtime and harness coverage |
