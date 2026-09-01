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

## Shared behavior

The chat manages input, commands, preferences, prompts, streaming, tools, and sessions. Both models use one command table. Each backend owns
its session format.

The ready line shows the model, profile, thinking state, routing profile, and resident memory. `/help` lists primary command names.

Flash-Next routing changes apply through `/settings`. This includes `cache-aware` routing and its `swap-epsilon` value.

The terminal shows measured prefill progress. It streams complete words with separate answer and reasoning colors. Tool protocol stays
hidden. Tool states and execution time remain visible.

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
