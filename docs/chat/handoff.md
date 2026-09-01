# Shared chat handoff

## Entry points

```bash
./chat.sh --model flashnext --profile plain
./chat.sh --model flashnext --profile agent
./chat.sh BUILD --profile plain
./chat.sh BUILD --profile agent
./chat.sh /server
```

`BUILD` is a Qwen27B V4 directory suffix.

## Main files

| Path | Responsibility |
|---|---|
| `macqwen/cli.py` | Select model, model path, and Python environment |
| `macqwen/session.py` | Read prompts, run commands, and generate replies |
| `macqwen/server.py` | Provide local OpenAI and Anthropic API endpoints |
| `macqwen/commands.py` | Define commands and aliases once |
| `macqwen/preferences.py` | Validate and save shared preferences |
| `macqwen/text.py` | Filter reasoning and hidden tool protocol streams |
| `macqwen/ui.py` | Render word, prefill, and tool animations |
| `macqwen/agent.py` | Run model and tool segments with separate timing |
| `macqwen/profiles/` | Define plain and agent behavior |
| `macqwen/tools/` | Provide repository, API, code, and search tools |
| `macqwen/backends/` | Adapt each model runtime to the session loop |

## Commands

Run `/help` for the current command list. Important management commands are:

```text
/thinking on|off|show|hide
/max-tokens N|off
/think-budget N|off
/profile plain|agent
/workspace PATH
/prompt [text|edit|default]
/keys [list|set SERVICE|delete SERVICE]
/settings [NAME VALUE|defaults]
/server
/save [name]
/load [name]
/sessions
/reset
```

`/workspace` and approval controls apply to the agent profile.
`/help` shows primary command names only. Compatibility aliases still work.

For Flash-Next, use `/settings routing cache-aware` to select the optional
cache-aware profile. Use `/settings swap-epsilon 0.02` to set its tolerance.
Exact-quality remains the default.

## Token settings

| Setting | Plain default | Agent default | Meaning |
|---|---:|---:|---|
| Answer allowance | 4,096 | 2,048 | Normal answer capacity |
| Thinking capacity | 512 | 512 | Extra capacity when thinking is enabled |

`/max-tokens off` selects the profile default. `/think-budget off` removes
extra reasoning capacity. `/status` shows the resolved values.

The command-line forms are `--max-tokens N` and `--think-budget N`. A legacy
saved `think_budget` value of `0` resolves to the 512-token default. Use `-1`
to keep extra reasoning capacity disabled.

The backend receives the sum as one ceiling. It does not enforce separate
reasoning and answer counters.

## Terminal state flow

Prefill starts at zero and waits for backend progress callbacks. Flash-Next
uses completed streamed MoE layers. Qwen27B forwards native chunk progress.

The agent starts tool activity when streamed protocol begins. The label changes
when the function name and arguments become available. Actual timing begins
immediately before `repo.call()` and ends immediately after it returns.

The 200 ms minimum tool display time is visual only. It is not part of the
reported tool duration.

The final agent statistics aggregate every generation segment from one
request. This includes generations before and after tool results.

The `gen` rate covers the complete decode. The `tail` rate starts after the
eight-token routing warmup and expert pin operation. The standard harness
measures 2.713 tok/s for `gen` and 2.650 for `tail` over ten kept arms. An
older harness that reloads the model per arm measured a 2.59 tok/s tail mean;
the gap is page-cache state. The separate 2.83 mean contains only two
warmup-eight arms and is not a production baseline.

Cache-aware routing measured 2.79 tok/s against 2.54 in one hot interleaved
run. Pairing adjacent arms gave an 8.3 percent mean gain. This mode changes
expert choices, so its answer can differ from exact-quality.

The ready line shows model, chat profile, thinking status, routing profile,
and resident memory. `/profile` shows the active profile. Changing it resets
the conversation and rebuilds the toolbox.

## Output rules

- Do not show `<tool_call>` or function protocol markup.
- Do not show a success emoji, icon, or green status word.
- Show an explicit label for tool failure.
- Use only `█` and `░` inside the progress bar.
- Start visible reasoning below the input prompt.
- Keep one blank line between reasoning and answer text.
- Keep terminal animation outside model and tool timing.

## System prompt files

The default preferences path creates these profile files:

```text
~/.macqwen/system-prompt-plain.txt
~/.macqwen/system-prompt-agent.txt
```

`/prompt` prints the active prompt and path. `/prompt edit` opens that file.
`/prompt default` removes the custom file. Use `/reset` after an external edit.

## Validation

Run the shared suite:

```bash
python3 -m unittest discover -s macqwen -p 'test_*.py'
python3 -m compileall -q macqwen models/flashnext
git diff --check
```

Run model-specific tests in the matching environment. See each model handoff.

## Invariants

- Shared modules do not import MLX at module load time.
- Model-exclusive code stays under `models/`.
- Every persistent chat setting has one schema entry.
- Model runtime defaults live in `macqwen/model_settings.py`.
- Every command has one table entry.
- Tool subprocesses do not receive API keys.
- The agent requires approval for modifying tools unless configured otherwise.
- Server mode resets model state when it starts and when it stops.
- A server request reuses the cache when the prompt extends it, and rebuilds
  the cache when the prompt diverges. Reuse needs the client to echo the
  previous assistant turn token for token, so it fires with thinking off and
  usually does not fire with thinking on.
- Server mode processes one request at a time.
- Chat output releases complete words only, and fades each one in. The fade
  runs on an output worker. Its budget is capped by `MACQWEN_FADE_MS` and by
  half the model's own word time.
- Prefill progress comes from backend work callbacks, not prompt-size timing
  prediction.
- Tool protocol stays hidden while its pending state remains visible.
- Tool duration measures repository execution only.
- Agent request statistics include every model segment around tool calls.
- Profile changes reset the conversation and keep the model loaded.

## Troubleshooting

| Symptom | Meaning or check |
|---|---|
| `stopped: truncated` | Increase `/max-tokens` or `/think-budget` |
| Bar stays at zero | Confirm the backend sends progress callbacks |
| Tool cursor appears blank | Check `ToolCallStreamFilter` event handling |
| Tool duration is too large | Keep visual delay outside execution timing |
| Missing agent token totals | Confirm every segment reaches `on_stats` |

Flash-Next session validation errors use English. Their schema and payload
format remain unchanged.

## Adding another model

1. Put its runtime under `models/MODEL/`.
2. Add one lazy backend under `macqwen/backends/`.
3. Add model selection and environment resolution to `macqwen/cli.py`.
4. Reuse the shared session, profiles, commands, and tools.
5. Add backend tests and one live smoke test.

## Current next work

- Replace model-specific launcher assumptions with documented configuration.
- Add a supported installation flow for both environments.
- Keep Flash-Next as the primary optimization target. Its decode rate is bound
  by expert bytes read from the drive, not by the GPU or the host. See the
  [Flash-Next research](../flashnext/research.md).
