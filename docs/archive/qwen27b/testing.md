# TEST_INSTRUCTIONS.md

## Frankenstein E2: quality, agent, memory, and ingest test runbook

This document describes how to reproduce the current tests for Frankenstein E2.

Read `PROJECT.md` first for architecture, training history, and acceptance criteria.

Read `SERVER_INSTRUCTIONS.md` for server startup commands.

---

# 1. Test philosophy

The acceptance reference is the GGUF model:

```text
/Users/gioma/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf
```

Current target model:

```text
/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1
```

Original MLX baseline:

```text
/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-v2
```

Core rule:

> Do not accept a runtime optimization that makes E2 materially worse at reasoning, coding, tool use, or instruction following.

Judge in this order:

```text
1. autonomous generation correctness
2. real coding-agent usefulness
3. closure / looping behavior
4. tool navigation
5. long-context recall
6. memory
7. ingest speed
8. generation speed
```

---

# 2. Before every test

Kill old server/model processes:

```bash
pkill -f "mlx_lm.*server" 2>/dev/null || true
pkill -f "macbat_readonly_agent" 2>/dev/null || true
sleep 1
```

Verify:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

No output should remain.

Check system state:

```bash
sysctl vm.swapusage
memory_pressure | tail -1
```

Do not compare runs when the machine is already under radically different memory pressure.

---

# 3. Server smoke test

Start the desired E2 server using `SERVER_INSTRUCTIONS.md`.

Then:

```bash
curl -s http://127.0.0.1:8080/health
```

Expected:

```json
{"status":"ok"}
```

Minimal completion:

```bash
MODEL="/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"

curl -s \
http://127.0.0.1:8080/v1/chat/completions \
-H 'Content-Type: application/json' \
-d "$(python3 - <<PY
import json
print(json.dumps({
    "model": "$MODEL",
    "messages": [
        {"role": "user", "content": "Answer with exactly TEST_OK"}
    ],
    "temperature": 0,
    "max_tokens": 100,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": True}
}))
PY
)" | python3 -m json.tool
```

Pass:

```text
request succeeds
model generates normally
no server traceback
final answer contains TEST_OK
```

---

# 4. Real MacBat agent benchmark

Repository:

```text
/Users/gioma/Developer/MACBAT
```

Primary task:

> Review `main.swift` in the context of the rest of the project and give concrete improvement suggestions.

The MacBat run is the primary practical test.

The agent must be allowed to inspect related files.

It must not receive the contents of `main.swift` manually in the initial prompt.

---

# 5. Install the latest read-only agent harness

Current bounded-context harness:

```text
macbat_readonly_agent_v3.py
```

If it is still in Downloads:

```bash
cp ~/Downloads/macbat_readonly_agent_v3.py \
   ~/macbat_readonly_agent_v3.py

chmod +x ~/macbat_readonly_agent_v3.py

"$HOME/mlx-qwen38-apple/bin/python3" \
-m py_compile ~/macbat_readonly_agent_v3.py \
&& echo "AGENT OK"
```

If a newer `frankenstein_engine.py` / ContextVM agent exists, prefer it and update this document.

---

# 6. Run the E2 MacBat benchmark

Start the recommended E2 server first.

Then, in another Terminal:

```bash
cd /Users/gioma/Developer/MACBAT

MODEL="/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"

caffeinate -i \
"$HOME/mlx-qwen38-apple/bin/python3" \
~/macbat_readonly_agent_v3.py \
--root "/Users/gioma/Developer/MACBAT" \
--model "$MODEL" \
--max-turns 30 \
--max-tokens 1600 \
--compact-at 6000 \
--temperature 0 \
--log ~/macbat_frankenstein_e2.log
```

The agent is read-only.

Expected tool families:

```text
find_files
list_dir
read_file
search
```

---

# 7. MacBat run measurements

For every turn capture:

```text
turn number
prompt tokens
cached tokens
completion tokens
wall-clock time
tool selected
file / region inspected
whether a file was unnecessarily re-read
whether reasoning closed
whether final answer was produced
```

Also capture server memory:

```bash
PID=$(pgrep -f "mlx_lm.*server" | head -1)

ps -o pid,rss,vsz,%cpu,command -p "$PID"

vmmap -summary "$PID" | grep -Ei \
"Physical footprint|Physical footprint \(peak\)|resident|Metal"

sysctl vm.swapusage

memory_pressure | tail -1
```

Run this after several significant tool turns, not only at the end.

---

# 8. MacBat quality grading

Grade the final review on:

```text
A. Files inspected

B. Correct cross-file relationships

C. Real bugs found

D. False-positive / invented bugs

E. Swift/macOS correctness

F. Concurrency reasoning

G. Lifecycle reasoning

H. Performance suggestions justified by code

I. Architectural insight

J. Refactoring usefulness

K. Tool efficiency

L. Repeated or irrelevant reads

M. Reasoning loops

N. Final prioritization quality
```

Important:

A longer answer is not automatically better.

A model that invents issues to appear thorough fails.

---

# 9. Known original-v2 baseline

Historical baseline using the first agent harness:

```text
model:
/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-v2
```

Result:

```text
TURN 5

prompt:      4637
completion:  1800
time:        509.38 s

failure:
no TOOL or FINAL after thinking
```

The model consumed its full generation budget without committing to an action.

E2 previously progressed farther and made better agent decisions.

Do not interpret the old v2 failure as a server failure.

---

# 10. E2 known behavior

E2 previously demonstrated strong held-out autonomous closure.

Known held-out transition result summary:

```text
e_percentage
GGUF: 130 tokens, closed
v2:   227 tokens, closed
E2:   171 tokens, closed

e_lock_order
GGUF: 555 tokens, closed
v2:   900-token cap, did not close
E2:   485 tokens, closed and correct

e_complexity
GGUF: 500 tokens, closed
v2:   890 tokens, closed
E2:   306 tokens, closed and correct
```

Totals:

```text
GGUF:          1185
original v2: >=2017
E2:             962
```

These prior results do not replace current real-world tests.

---

# 11. Known invalid regression prompt

Do not use the previous Python `asyncio.TaskGroup` prompt as a termination regression.

Both E2 and the GGUF teacher looped on it.

Historical result:

```text
E2:
  generated 1800
  did not close

GGUF:
  generated 2400
  did not close
```

Therefore:

> If both reference and candidate fail the same pathological prompt, it is not useful as a candidate regression test.

---

# 12. Prompt-ingestion benchmark

Goal:

Measure true sustained prefill speed independently of agent quality.

Test the same prompt/context at approximately:

```text
1K
4K
8K
16K
32K
```

For each size record:

```text
prompt tokens
prefill-step-size
prompt-processing wall time
prompt tok/s
peak process footprint
MLX peak memory if available
swap
memory pressure
```

Do not compare different prompt contents unless necessary.

Current known approximate long-prompt result:

```text
~10.3K prompt tokens
~307 seconds prefill
~33.5 prompt tokens/sec
```

The run used conservative settings.

---

# 13. Prefill-step sweep

Test exactly one step size at a time.

Recommended order:

```text
512
1024
1536
2048
```

For each:

1. restart the server fresh;
2. use the exact same prompt;
3. record prefill speed;
4. record memory;
5. record output quality;
6. stop the server before the next configuration.

Reject any setting that destabilizes the Mac, regardless of speed.

---

# 14. Prompt-cache reuse test

This test answers:

> Are later agent turns processing only the new suffix, or re-ingesting the whole conversation?

Start server with prompt caching enabled:

```text
--prompt-cache-size 1
```

Run at least three related turns.

Watch:

```text
TURN STATS:
prompt=...
cached=...
completion=...
```

Pass:

```text
cached tokens become substantial on later turns
new prompt-processing work is much smaller than total logical prompt
```

Fail:

```text
cached=0 on every turn
```

unless intentionally running a cold benchmark.

Also watch server logs for:

```text
Prompt Cache: N sequences, X.XX GB
```

---

# 15. Memory-growth test

Goal:

Find whether memory scales uncontrollably as the model reads more files.

Start from a clean server.

Run the MacBat agent.

After each significant read/tool turn:

```bash
PID=$(pgrep -f "mlx_lm.*server" | head -1)

printf "\n=== %s ===\n" "$(date)"
ps -o pid,rss,vsz,%cpu -p "$PID"
vmmap -summary "$PID" | grep -Ei \
"Physical footprint|Physical footprint \(peak\)"
sysctl vm.swapusage
memory_pressure | tail -1
```

Pass for current interim architecture:

```text
Mac remains responsive
memory growth is controlled
prompt cache remains inside configured budget
agent can complete the review
```

The long-term ContextVM requirement is stronger:

> Logical context may grow while resident memory remains approximately bounded.

---

# 16. Q4-KV regression test

Compare:

```text
A. no KV quantization
B. --kv-bits 4 --kv-group-size 64
```

Keep everything else identical.

Compare:

```text
MacBat findings
tool decisions
closure
reasoning quality
final answer
memory
speed
```

Pass only if quality remains materially unchanged.

Do not judge only by token count.

---

# 17. Q2-KV regression test

Only run after Q4 is stable.

Compare:

```text
Q4
vs
Q2
```

Suggested Q2:

```text
--kv-bits 2
--kv-group-size 64
--quantized-kv-start 0
```

Required quality tests:

```text
early instruction recall
code-symbol recall
multi-file reasoning
distractor resistance
long-distance dependency
tool selection
```

If Q2 loses quality:

Do not modify E2 weights.

Prefer:

```text
mixed KV precision
hot Q4
cold Q2
```

---

# 18. Long-context ladder

Do not jump directly to 256K.

Progress:

```text
32K
64K
128K
192K
256K
```

At every stage test:

```text
needle retrieval
multi-needle retrieval
early instruction retention
code lookup
distractor resistance
long-distance reasoning
memory
swap
stability
```

A context length only counts as supported if the model remains useful at that length.

---

# 19. Fresh blind coding benchmark

Once runtime is stable, run a fresh external benchmark.

Use prompts never used during training/distillation.

Suggested domains:

```text
Python debugging
Swift/macOS code review
Swift concurrency
SQL / transactions
algorithms
API design
distributed retries / idempotency
cache correctness
requirements conflict
complexity reasoning
```

Aim for roughly 15 high-quality prompts.

Run:

```text
GGUF reference first
original MLX v2 second
E2 third
```

Use the same:

```text
prompt
temperature
max output
tool contract
repository state
```

Do not load multiple 27B models simultaneously.

---

# 20. Blind-benchmark grading

For every prompt record:

```text
correctness
important omissions
hallucinations
closure
reasoning length
final answer length
repetition
time
generation tok/s
memory
```

Primary question:

> Is E2 at least as useful as the GGUF teacher/reference for real coding work?

---

# 21. ContextVM V0 test

After direct single-process engine implementation, this becomes the primary runtime test.

Required invariant:

```text
Turn 1:
process initial prompt

Tool call:
execute tool

Turn 2:
process ONLY newly appended tool result / control tokens

Turn 3:
process ONLY newly appended tokens
```

The engine must not reconstruct and re-prefill the full conversation.

Telemetry required per turn:

```text
logical sequence tokens
new tokens processed
reused tokens/state
attention KV bytes
recurrent-state bytes
MLX active memory
MLX peak memory
system swap
prompt tok/s
generation tok/s
```

Pass:

> New-token work is proportional to the appended suffix, not total conversation length.

---

# 22. ContextVM V1 test: quantized persistent KV

Quantize only cache objects that support:

```python
to_quantized()
```

For Qwen3.8 this should target full-attention KV while recurrent `ArraysCache` remains fixed-state.

Verify:

```text
same output behavior
lower cache bytes
no corruption after repeated tool turns
```

---

# 23. ContextVM V2 test: paged KV equivalence

Before SSD paging, prove paged attention correctness entirely in RAM.

Compare:

```text
contiguous KV
vs
paged KV
```

For identical model state and query, compare:

```text
attention output
next-token logits
greedy next token
multi-token generation
```

Use a strict numerical tolerance.

Do not continue to SSD spill until paged and contiguous inference agree.

---

# 24. ContextVM V3 test: SSD spill

After paged equivalence works:

1. create many KV pages;
2. mark old pages cold;
3. spill them to SSD;
4. remove them from resident memory;
5. reload selected pages;
6. verify output against the all-RAM version.

Measure:

```text
resident memory
SSD read latency
page load latency
generation latency
correctness
```

Goal:

> Logical history grows without proportional unified-memory growth.

---

# 25. ContextVM retrieval test

The router should start simple.

Test:

```text
recent hot pages
+ metadata-matched source pages
+ pinned instruction pages
```

Create cases where required information lives:

```text
recently
far in the past
in another source file
behind distractor pages
```

The system should retrieve the correct exact pages rather than rely only on summaries.

---

# 26. Regression rules

If an optimization changes output:

Ask:

```text
Is it a harmless wording difference?
or
did correctness/tool behavior change?
```

Never accept:

```text
faster but wrong
smaller but forgetful
shorter but incomplete
more context but weaker instruction retention
```

---

# 27. Test logs

Use distinct files.

Suggested names:

```text
~/macbat_e2_q4.log
~/macbat_e2_no_kv_quant.log
~/macbat_e2_prefill_512.log
~/macbat_e2_prefill_1024.log
~/macbat_e2_prefill_1536.log
~/macbat_e2_prefill_2048.log
~/contextvm_v0.log
```

Never overwrite a useful baseline log.

---

# 28. Quick current test sequence

For the current project state, run in this order:

```text
1. server smoke test
2. E2 + Q4 KV MacBat run
3. monitor memory after every few file reads
4. verify prompt-cache reuse
5. compare Q4 against unquantized KV if behavior looks suspicious
6. sweep prefill 512 / 1024 / 1536 / 2048
7. begin ContextVM V0
```

Do not spend more time retraining E2 unless fresh external quality evidence demonstrates a real deficiency.

---

# 29. Current success criterion

The immediate milestone is:

```text
Frankenstein E2
+ real MacBat agent
+ stable 16 GB operation
+ no repeated full-history ingest
+ controlled KV memory
+ >= current quality
+ materially faster iterative turns
```

The next milestone is ContextVM:

```text
large logical context
bounded physical memory
exact old-context retrieval
read/process information once
reuse it thereafter
```

---

# 30. ContextVM V0 engine tests

The engine replaces the HTTP server for agent work. No server is needed.

```text
/Users/gioma/Developer/MACQWEN/frankenstein_engine.py
```

## 30.1 Template selftest

Run this first. It loads the tokenizer only. It takes about 2 seconds.

```bash
"$HOME/mlx-qwen38-apple/bin/python3" \
~/Developer/MACQWEN/frankenstein_engine.py --mode selftest
```

Expected:

```text
SELFTEST: PASS
```

A FAIL means the append-only segment builder no longer matches the chat
template. Fix that before any model run.

## 30.2 Three-turn engine demo

```bash
caffeinate -i "$HOME/mlx-qwen38-apple/bin/python3" \
~/Developer/MACQWEN/frankenstein_engine.py \
--mode demo --reasoning-effort low --max-tokens 250
```

Check:

```text
turn 2 and turn 3 process only a few new tokens
invariant cache==tape: True
```

## 30.3 MacBat agent benchmark

```bash
caffeinate -i "$HOME/mlx-qwen38-apple/bin/python3" \
~/Developer/MACQWEN/frankenstein_engine.py \
--mode agent \
--kv-bits 4 --quantized-kv-start 1024 \
--prefill-step-size 1024 \
--max-tokens 1600 --max-turns 24 \
--repetition-penalty 1.05 --repetition-context-size 128
```

Record per turn:

```text
new prompt tokens, prompt tok/s
generation tokens, generation tok/s
logical context, kv bytes, attention kv bytes
mlx active, mlx peak
host free memory, swap
```

The engine prints all of this and writes a log file.

## 30.4 Safety guards

The run stops itself on:

```text
host free memory below --min-free-gb        (default 0.35)
swap growth above --max-swap-growth-gb      (default 3.0)
cache/tape invariant broken
```

## 30.5 Known good result

See PROJECT.md section 45. Reference numbers:

```text
6065 logical tokens
0.27 GB total KV
13.14 GB mlx peak
no swap growth
no crash
```
