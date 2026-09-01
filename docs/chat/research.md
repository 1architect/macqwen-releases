# Shared chat research

## Problem

The project originally had one chat per model. Features and fixes diverged.
The models also require incompatible MLX environments.

A single process cannot safely load both runtimes. A single code interface can
still serve both through a model-specific launcher.

## Main design decisions

| Decision | Reason |
|---|---|
| One shared session loop | A chat feature reaches every model |
| One command table | Aliases and settings cannot diverge |
| One preferences schema | Each setting has one default and validator |
| Separate model environments | The two MLX stacks are incompatible |
| Lazy backend imports | The shared package stays pure Python |
| Profiles independent from models | Tools are behavior, not model capability |
| Backend-owned persistence | Each model has a different exact cache format |
| Backend progress callback | The UI uses measured work instead of prediction |
| Separate reasoning capacity | Long reasoning has extra generation space |
| Streaming protocol filter | Tool activity appears before tool execution |
| Execution-only tool clock | Display delays do not corrupt tool duration |
| Backend-owned live routing | Model-specific routing changes without reload |

## Package boundary

`macqwen/` contains the shared interface. `models/` contains model-exclusive
runtime code. Backends are the only shared modules that import model code.

The current backend interface follows the operations used by `session.py`:
conversation setup, append operations, generation, reset, status, and session
persistence.

## Feature union

The shared chat combines the useful features from both former chats:

- Thinking activation and display.
- Reply token limits.
- Reasoning effort and answer streaming.
- Routing profiles and exact sessions.
- Workspace awareness and editable prompts.
- Agent tools with approval controls.
- Pure JSON benchmark output.

## Generation limits

`max_tokens` is the answer allowance. A saved value of `-1` selects the
profile default. The defaults are 4,096 tokens for `plain` and 2,048 tokens
for `agent`.

`think_budget` adds 512 tokens by default when thinking is enabled. Generation
uses the answer allowance plus this reasoning capacity. Thinking-disabled
turns add nothing.

The old schema accepted `0` as no reasoning budget. The new schema uses `-1`.
A saved `0` fails validation and resolves to the new 512-token default.

The model backend still receives one generation ceiling. The runtime does not
count reasoning and answer phases independently. The extra capacity protects
the usual answer space when reasoning stays within its expected budget.

## Prefill progress

The old display predicted progress from prompt size. That prediction could
remain at zero or show an inaccurate rate.

The backend interface now accepts `on_prefill_progress(done, total)`. The
Qwen27B adapter forwards its native chunk progress.

Flash-Next reports each completed streamed MoE layer. The adapter maps the
completed-layer ratio to prompt-token-equivalent progress. It reserves the
last step until prefill finishes. This prevents an early 100 percent display.

The UI smooths rates from consecutive progress callbacks. Its moving colour
animation continues between callbacks. The final statistics calculate true
prefill throughput from prompt tokens divided by complete prefill time.

The progress bar uses filled `█` and unfilled `░` cells. It omits border
characters. Its width adjusts to the terminal width.

## Agent tool lifecycle

`ToolCallStreamFilter` detects `<tool_call>` during model streaming. It hides
all protocol markup and emits a pending event immediately. It emits the tool
name after `<function=NAME>` arrives.

`AgentUI` uses these states:

| State | Display | Timer |
|---|---|---|
| Protocol starts | `Preparing tool` | Not started |
| Function name arrives | Action label | Not started |
| Repository call starts | Action label | Starts |
| Repository call ends | Completed action and duration | Stops |

The UI keeps very fast tools visible for at least 200 ms. It captures elapsed
time before this display delay. A successful line has no icon, emoji, or green
status word. Failed tools keep an explicit failure label.

## Output layout and statistics

The ready line reports model, chat profile, thinking status, routing profile,
and RSS. Help output hides compatibility aliases, but dispatch still accepts
them.

Visible reasoning starts directly below the input prompt. The stream removes
leading tag newlines. It leaves one blank line before the answer.

Reasoning words use a darker four-step fade. Answer words use the normal fade.
Both animations run outside model timing.

Plain mode reports one generation segment. Agent mode can generate several
segments around tool calls. Agent statistics sum prompt tokens, output tokens,
prefill time, and decode time across all segments.

The final line reports prompt rate, decode rate, context size, and wall time.
The stop reason follows on the next line. `truncated` means generation reached
its ceiling before a final answer or tool call completed.

Flash-Next owns live routing configuration. The shared `/settings` command
passes `cache-aware` and `swap-epsilon` to that backend. This keeps routing
state outside the shared chat package.

## Profile prompts

Each profile stores its custom system prompt in a separate file beside the
preferences file. Profile changes reset conversation state and rebuild the
toolbox. The loaded model remains in memory.

A one-time migration moves the legacy `system_prompt` preference into the
active profile file. The preference field remains empty after migration.

## Message language

Flash-Next saved-session validation errors now use English. This change covers
names, headers, checksums, compatibility, tensors, cache state, and corruption.
It does not change the saved-session schema or payload format.

## Terminal input

Bracketed paste keeps all pasted newlines in one input message. Pressing Enter
after the paste submits it.

Input text keeps control-marker strings as text. It does not inject chat control
tokens into the transcript.

## Private API keys

The repository contained no hard-coded API keys during the 2026-08-31 audit.
The audit also checked high-confidence secret patterns in Git history.

The `/keys` interface stores Tavily and Context7 keys outside the repository.
Key input does not echo. Inline key arguments are rejected.

Secret-like environment variables are removed from tool and checker child
processes.

## Validation baseline

The restructure completed with these checks:

- Shared unit tests pass in the system Python environment.
- Backend tests pass in each model environment.
- Flash-Next restores an existing exact session after relocation.
- Flash-Next benchmark mode emits one valid JSON line on standard output.

The completed restructure plan remains in
[`docs/archive/chat/restructure-plan.md`](../archive/chat/restructure-plan.md).
