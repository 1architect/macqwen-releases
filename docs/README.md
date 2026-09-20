# Documentation

Use the component handoff before changing code or running a benchmark. Read
the research record when you need prior measurements or rejected approaches.
[`CONTRIBUTING.md`](../CONTRIBUTING.md) defines the shared workflow.

## Document structure

Every active component uses the same three documents:

| Document | Purpose |
|---|---|
| `brief.md` | Purpose, scope, support status, and main results |
| `research.md` | Dated measurements, decisions, and rejected approaches |
| `handoff.md` | Current operation, validation, constraints, and next work |

Measurement directories index raw evidence. The project-wide
[measurement standard](measurement-standard.md) defines the shared test and
record rules.

## Components

| Component | Brief | Research | Handoff |
|---|---|---|---|
| Shared chat | [Brief](chat/brief.md) | [Research](chat/research.md) | [Handoff](chat/handoff.md) |
| Flash-Next | [Brief](flashnext/brief.md) | [Research](flashnext/research.md) | [Handoff](flashnext/handoff.md) |
| K2-Horizon 7B | [Brief](k2_horizon/brief.md) | [Research](k2_horizon/research.md) | [Handoff](k2_horizon/handoff.md) |
| Bonsai-2 27B | [Brief](bonsai2/brief.md) | [Research](bonsai2/research.md) | [Handoff](bonsai2/handoff.md) |
| Qwen3.8-27B | [Brief](qwen27b/brief.md) | [Research](qwen27b/research.md) | [Handoff](qwen27b/handoff.md) |

Flash-Next also has [agent invariants](flashnext/AGENT_INVARIANTS.md), which
collect runtime-specific correctness and experiment constraints.

## Supporting material

| Path | Purpose |
|---|---|
| [`measurement-standard.md`](measurement-standard.md) | Shared live-test and JSONL evidence rules |
| [`flashnext/measurements/`](flashnext/measurements/) | Retained Flash-Next records |
| [`k2_horizon/measurements/`](k2_horizon/measurements/) | Retained K2-Horizon records and diagnostics |
| [`bonsai2/measurements/`](bonsai2/measurements/) | Retained Bonsai-2 records and diagnostics |
| [`flashnext/graphics/`](flashnext/graphics/) | Flash-Next trace graphics and plots |
| [`MLX/`](MLX/) | MLX Metal backend source notes; reference material, not workflow instructions |

The [archive](archive/README.md) preserves superseded runbooks and historical
records. Archived documents may contain obsolete paths or commands; active
documents are the source of current instructions.
