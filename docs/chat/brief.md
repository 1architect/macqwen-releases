# Shared chat overview

## Purpose

The shared chat provides one interface for each MACQWEN runtime. Model-specific MLX code stays outside the shared package. The reference
system has 16 GB of unified memory.

## Current scope

The chat supports two runtimes:

- `flashnext` runs Qwen3.8-Flash-Next with SSD-streamed sparse tensors.
- `qwen27b` runs the Qwen3.8-27B V4 runtime with a custom MLX build.

Each runtime uses a separate Python environment. The launcher selects it before model loading.

## Profiles

- `plain` provides direct conversation without tools.
- The repository-tool profile provides tools, approval controls, and workspace context.

Profile and model selection are independent.

### Plain system prompt

The Plain profile uses this system prompt:

```text
Answer precisely. Never invent an API, a name, or a result. Ask for what you need, and say when you are unsure.
```

| Clause | Purpose | Evidence or scope |
|---|---|---|
| Answer precisely. | Preserve the original direct-answer rule. | Prompt text. |
| Never invent an API, a name, or a result. | Block unsupported claims. | The recorded oQ3-MTP SketchUp failure called `Sketchup::Face#extrude` across 34,203 reasoning characters without questioning it. |
| Ask for what you need. | Plain has no tools, so asking is the only way to get more information. | Prompt text and profile scope. |
| Say when you are unsure. | Keep uncertainty explicit. | The recorded oQ4 response questioned `pushpull` about twenty times before settling on it. |

## Shared behavior

The chat manages input, commands, preferences, prompts, streaming, tools, and sessions. Both models use one command table. Each backend owns
its session format.

The ready line shows the model, profile, and `/help` hint. `/status` shows
thinking, routing, context, and memory diagnostics. `/help` lists six primary
commands. `/help all` lists compatibility commands.

Flash-Next routing changes apply through `/config model`. This includes
`cache-aware` routing and its `swap-epsilon` value.

The terminal shows measured prefill progress. It streams complete words with separate answer and reasoning colors. Tool protocol stays
hidden. Tool states and execution time remain visible.

## FlashNext research boundary

The next FlashNext performance work stays in the FlashNext runtime. It does
not change shared chat behavior. The active fronts are the pre-load wired-limit
test (#43), Metal barrier and fence instrumentation (#45), and expert working
set tests (#24 and #25).

Answer allowance and reasoning capacity use separate settings. Request statistics combine all generation segments.

## Local data

| Data | Location |
|---|---|
| Preferences | `~/.macqwen/preferences.json` |
| Plain system prompt | `~/.macqwen/system-prompt-plain.txt` |
| Repository-tool system prompt | `~/.macqwen/system-prompt-agent.txt` |
| API keys | `~/Library/Application Support/MACQWEN/api_keys.json` |
| Flash-Next sessions | `~/.cache/flashnext/sessions/` |
| Qwen27B sessions | `~/.frankenstein/sessions/` |

The API key directory uses mode `0700`. The key file uses mode `0600`.

## Status

The shared package is active. Flash-Next is the primary runtime. Qwen27B remains available for research.
