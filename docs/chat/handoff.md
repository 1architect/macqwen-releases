# Shared chat reference

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
| `macqwen/profiles/` | Define plain and repository-tool behavior |
| `macqwen/tools/` | Provide repository, API, code, and search tools |
| `macqwen/backends/` | Adapt each model runtime to the session loop |

## Commands

Run `/help` for the six primary commands. Use `/help all` for the complete
reference, including compatibility commands.

```text
/help [all]
/new
/session save|load|list|delete [name]
/config [section] ...
/status
/quit
```

`/session` groups save, load, list, and delete actions. `/config` groups
thinking, token, sampling, display, model, prompt, profile, and tool settings.
Use `/config keys` for API key settings in either profile. Use `/config tools`
for approval and workspace settings.
Approval and workspace controls require the `agent` profile.
Compatibility commands and aliases still work.

For Flash-Next, use `/config model routing cache-aware` to select the optional cache-aware profile. Use `/config model swap-epsilon 0.02` to set its
tolerance. Exact-quality remains the default.

## Token settings

| Setting | Plain default | Repository-tool default | Meaning |
|---|---:|---:|---|
| Answer allowance | 4,096 | 2,048 | Answer capacity |
| Thinking capacity | 512 | 512 | Extra capacity when thinking is enabled |

`/config tokens off` selects the profile default. `/config think-tokens off` removes extra reasoning capacity. `/status` shows the resolved values.

The command-line forms are `--max-tokens N` and `--think-budget N`. A legacy saved `think_budget` value of `0` resolves to the 512-token
default. Use `-1` to keep extra reasoning capacity disabled.

The backend receives the sum as one ceiling. It does not enforce separate reasoning and answer counters.

## Terminal state flow

Prefill starts at zero and waits for backend progress callbacks. Flash-Next uses completed streamed MoE layers. Qwen27B forwards native
chunk progress.

The tool UI starts when streamed protocol begins. The label changes when the function name and arguments become available. Actual timing
begins immediately before `repo.call()` and ends immediately after it returns.

The 200 ms minimum tool display time is visual only. It is not part of the reported tool duration.

The final statistics aggregate every generation segment. This includes generation before and after tool results.

The `gen` rate covers the complete decode. The `tail` rate starts after the eight-token routing warmup and expert pin operation. The
accepted clean-boot `buffer-chunk2` comparison measures 2.83 tok/s for `gen`,
2.70 for `tail`, and 457.7 MB/token. The older 2.713 and 2.59 figures are
pre-buffer records.

Cache-aware routing measured 2.79 tok/s against 2.54 in one hot interleaved run. Pairing adjacent arms gave an 8.3 percent mean gain. This
mode changes expert choices, so its answer can differ from exact-quality.

The ready line shows model, chat profile, and a `/help` hint. `/status` shows thinking status, routing profile, context, and memory.
Changing the profile resets the conversation and rebuilds the toolbox.

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

`/config prompt` prints the active prompt and path. `/config prompt edit` opens that file. `/config prompt default` removes the custom file. Use `/new` after
an external edit.

The default Plain prompt is:

```text
Answer precisely. Never invent an API, a name, or a result. Ask for what you need, and say when you are unsure.
```

Its clauses keep direct answers, reject invented API names and results, request missing information, and state uncertainty. Plain has no tools, so it can ask for information but cannot retrieve it. The wording reflects the recorded SketchUp comparison: oQ3-MTP called `Sketchup::Face#extrude` across 34,203 reasoning characters without questioning the method, while oQ4 questioned `pushpull` about twenty times before settling on it.

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
- The repository-tool profile requires approval for modifying tools unless configured otherwise.
- Server mode resets model state when it starts and when it stops.
- A server request reuses the cache when the prompt extends it, and rebuilds
the cache when the prompt diverges. Reuse needs the client to echo the previous assistant turn token for token, so it fires with thinking
off and usually does not fire with thinking on.
- Server mode processes one request at a time.
- Chat output releases complete words only, and fades each one in. The fade
runs on an output worker. Its budget is capped by `MACQWEN_FADE_MS` and by half the model's own word time.
- Prefill progress comes from backend work callbacks, not prompt-size timing
prediction.
- Tool protocol stays hidden while its pending state remains visible.
- Tool duration measures repository execution only.
- Request statistics include every model segment around tool calls.
- Profile changes reset the conversation and keep the model loaded.

## Troubleshooting

| Symptom | Meaning or check |
|---|---|
| `stopped: truncated` | Increase `/config tokens` or `/config think-tokens` |
| Bar stays at zero | Confirm the backend sends progress callbacks |
| Tool cursor appears blank | Check `ToolCallStreamFilter` event handling |
| Tool duration is too large | Keep visual delay outside execution timing |
| Missing token totals | Confirm every segment reaches `on_stats` |

Flash-Next session validation errors use English. Their schema and payload format remain unchanged.

## Adding another model

1. Put its runtime under `models/MODEL/`.
2. Add one lazy backend under `macqwen/backends/`.
3. Add model selection and environment resolution to `macqwen/cli.py`.
4. Reuse the shared session, profiles, commands, and tools.
5. Add backend tests and one live smoke test.

## Current next work

Current setup work is tracked in [#11](https://github.com/1architect/macqwen-releases/issues/11).

Flash-Next remains the primary optimization target. Expert reads set its decode rate. See [Flash-Next research](../flashnext/research.md).
