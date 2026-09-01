# Shared chat brief

## Purpose

The shared chat is the stable interface for every MACQWEN model runtime.
It keeps model-specific MLX code outside the shared package.

MACQWEN targets local LLM execution on low-memory Apple Silicon systems.
The tested machine has 16 GB of unified memory.

## Current scope

The chat supports two Qwen runtimes:

- `flashnext` runs Qwen3.8-Flash-Next with SSD-streamed sparse tensors.
- `qwen27b` runs the Qwen3.8-27B V4 runtime with a custom MLX build.

Each runtime uses its own Python environment. The launcher selects the correct
environment before the model loads.

## Profiles

- `plain` provides direct conversation without tools.
- `agent` provides repository tools, approval controls, and workspace context.

Profile and model selection are independent.

## Shared behavior

The chat owns terminal input, commands, preferences, prompt construction,
streaming, tool execution, and session controls.

Both models receive the same command table. Model-specific session formats stay
inside their backends.

The ready line reports the active model, chat profile, thinking status,
routing profile, and resident memory. Help lists primary command names only.

Flash-Next routing changes apply live through `/settings`. This includes the
optional `cache-aware` profile and its `swap-epsilon` tolerance.

The terminal shows measured prefill progress with an animated fill bar. It
streams complete words with separate answer and reasoning colours.

The agent hides model tool protocol. It shows pending and active tool states,
then reports the completed action and its execution time.

Answer allowance and reasoning capacity use separate settings. Final agent
statistics combine every model segment from one request.

## Local data

| Data | Location |
|---|---|
| Preferences | `~/.macqwen/preferences.json` |
| Plain system prompt | `~/.macqwen/system-prompt-plain.txt` |
| Agent system prompt | `~/.macqwen/system-prompt-agent.txt` |
| API keys | `~/Library/Application Support/MACQWEN/api_keys.json` |
| Flash-Next sessions | `~/.cache/flashnext/sessions/` |
| Qwen27B sessions | `~/.frankenstein/sessions/` |

The API key directory uses mode `0700`. The key file uses mode `0600`.

## Status

The shared package is active. Flash-Next is the current model focus.
Qwen27B remains supported as a secondary research runtime.
