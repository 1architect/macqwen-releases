# Archived Frankenstein E2 MLX server runbook

Archive status: historical fallback instructions for the Qwen 27B runtime. Paths, versions, and recommended settings can
be obsolete. `PROJECT.md` contains the related architecture.

## 1. Environment

Python environment:

```bash
VENV="$HOME/mlx-qwen38-apple"
PY="$VENV/bin/python3"
```

Current best model:

```bash
MODEL="~/models/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"
```

Original MLX baseline model:

```bash
BASELINE="~/models/Qwen3.8-27B-Apple-MLX-v2"
```

Reference GGUF model:

```bash
GGUF="~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf"
```

Default server address:

```text
http://127.0.0.1:8080
```

## 2. Verify the Python environment

```bash
"$HOME/mlx-qwen38-apple/bin/python3" - <<'PY'
import mlx
import mlx_lm
print("MLX:", mlx.__version__)
print("MLX-LM:", getattr(mlx_lm, "__version__", "unknown"))
PY
```

Expected environment used during development:

```text
MLX     0.32.1
MLX-LM  0.32.0
```

This project modified the local MLX-LM install. Check `PROJECT.md` before an upgrade or reinstall.

## 3. Check whether the KV-server patch is installed

The stock MLX-LM server did not expose the KV-quantization arguments used by the normal generation path.
Check:

```bash
"$HOME/mlx-qwen38-apple/bin/python3" \
-m mlx_lm server --help \
| grep -E "kv-bits|kv-group-size|quantized-kv-start"
```

If all three options appear, the server is already patched.
Expected options:

```text
--kv-bits
--kv-group-size
--quantized-kv-start
```

If they do not appear, apply the patch script once:

```bash
"$HOME/mlx-qwen38-apple/bin/python3" \
~/patch_mlx_server_kv.py
```

The patch script should create a backup of `mlx_lm/server.py` before changing it.
Verify again:

```bash
"$HOME/mlx-qwen38-apple/bin/python3" \
-m mlx_lm server --help \
| grep -E "kv-bits|kv-group-size|quantized-kv-start"
```

## 4. Kill old servers before starting

Never leave multiple 27B MLX servers alive on a 16 GB Mac.

```bash
pkill -f "mlx_lm.*server" 2>/dev/null || true
sleep 1
```

Verify port 8080 is free:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

No output is expected before starting the new server.

## 5. Recommended current server

This configuration was recommended for repeated tool tests.
It enables:

- E2 model
- thinking
- prompt-prefix caching
- Q4 full-attention KV where supported
- 1024-token prefill chunks
- bounded prompt-cache storage

```bash
MODEL="~/models/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"
caffeinate -i \
"$HOME/mlx-qwen38-apple/bin/python3" \
-m mlx_lm server \
--model "$MODEL" \
--port 8080 \
--chat-template-args '{"enable_thinking":true}' \
--prompt-cache-size 1 \
--prompt-cache-bytes 512MB \
--prefill-step-size 1024 \
--kv-bits 4 \
--kv-group-size 64 \
--quantized-kv-start 1024
```

Keep this Terminal visible during tests.

## 6. Conservative / known-safe server

If the Q4-KV server crashes or appears incorrect, use this configuration to isolate the problem.

```bash
MODEL="~/models/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"
caffeinate -i \
"$HOME/mlx-qwen38-apple/bin/python3" \
-m mlx_lm server \
--model "$MODEL" \
--port 8080 \
--chat-template-args '{"enable_thinking":true}' \
--prompt-cache-size 1 \
--prompt-cache-bytes 512MB \
--prefill-step-size 512
```

This disables KV quantization but keeps prefix caching.
This configuration isolates these failure sources:

```text
Q4 KV
vs
the model or tool workflow
```

## 7. Cold-quality server

Use this only for controlled quality comparisons where prompt-cache reuse must not affect measurements.

```bash
MODEL="~/models/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"
caffeinate -i \
"$HOME/mlx-qwen38-apple/bin/python3" \
-m mlx_lm server \
--model "$MODEL" \
--port 8080 \
--chat-template-args '{"enable_thinking":true}' \
--prompt-cache-size 0 \
--prefill-step-size 512
```

Do not use this for normal multi-turn work.
With `--prompt-cache-size 0`, every turn can reprocess the entire conversation.
Previous runs grew from a few hundred prompt tokens to about 15K tokens and became slow.

## 8. Original-v2 baseline server

Only use this when reproducing the old baseline.

```bash
MODEL="~/models/Qwen3.8-27B-Apple-MLX-v2"
caffeinate -i \
"$HOME/mlx-qwen38-apple/bin/python3" \
-m mlx_lm server \
--model "$MODEL" \
--port 8080 \
--chat-template-args '{"enable_thinking":true}' \
--prompt-cache-size 0 \
--prefill-step-size 512
```

Known historical result with the first MacBat harness:

```text
Turn 5
prompt:     4637 tokens
completion: 1800 tokens
time:       509.38 s
result:     failure: reasoning hit the token cap without TOOL or FINAL
```

The current work does not target this model.

## 9. Verify the server is healthy

In another Terminal:

```bash
curl -s http://127.0.0.1:8080/health
```

Expected:

```json
{"status":"ok"}
```

List models:

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

Check listener:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

## 10. Minimal API smoke test

```bash
MODEL="~/models/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1"
curl -N \
http://127.0.0.1:8080/v1/chat/completions \
-H 'Content-Type: application/json' \
-d "$(python3 - <<PY
import json
print(json.dumps({
    "model": "$MODEL",
    "messages": [
        {"role": "user", "content": "Return exactly: SERVER_OK"}
    ],
    "temperature": 0,
    "max_tokens": 100,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": True}
}))
PY
)"
```

Success means:

- HTTP request succeeds
- generation streams
- no Python exception appears in the server Terminal
- final answer contains `SERVER_OK`

## 11. Monitor memory while the server is running

Find the process:

```bash
PID=$(pgrep -f "mlx_lm.*server" | head -1)
echo "$PID"
```

Basic process stats:

```bash
ps -o pid,rss,vsz,%cpu,command -p "$PID"
```

Memory footprint:

```bash
vmmap -summary "$PID" | grep -Ei \
"Physical footprint|Physical footprint \(peak\)|resident|mapped file|Metal"
```

System swap:

```bash
sysctl vm.swapusage
```

System memory pressure:

```bash
memory_pressure | tail -1
```

Useful combined command:

```bash
PID=$(pgrep -f "mlx_lm.*server" | head -1)
echo "=== PROCESS ==="
ps -o pid,rss,vsz,%cpu,command -p "$PID"
echo
echo "=== VM ==="
vmmap -summary "$PID" | grep -Ei \
"Physical footprint|Physical footprint \(peak\)|resident|Metal"
echo
echo "=== SWAP ==="
sysctl vm.swapusage
echo
echo "=== MEMORY PRESSURE ==="
memory_pressure | tail -1
```

## 12. Watch prompt-cache behavior

The server should log lines similar to:

```text
Prompt Cache: N sequences, X.XX GB
```

For a normal multi-turn run, `N` should become non-zero.
Bad sign:

```text
Prompt Cache: 0 sequences, 0.00 GB
```

on every single turn when caching was expected.
The old benchmark intentionally used `--prompt-cache-size 0`.

## 13. Prefill-step-size experiments

Do not change multiple variables at once.
Test:

```text
512
1024
1536
2048
```

Example:

```bash
--prefill-step-size 1536
```

For every setting record:

```text
prompt token count
prompt-processing time
prompt tok/s
generation tok/s
peak model memory
swap
whether the Mac remains responsive
quality / output differences
```

Current observed long-prompt prefill was roughly:

```text
~33 tok/s
```

with conservative settings.
Do not assume a larger prefill step is faster until measured.

## 14. Q2 KV experiments

Do not start here.
First prove Q4 is stable and quality-preserving.
When ready:

```bash
--kv-bits 2 \
--kv-group-size 64 \
--quantized-kv-start 0
```

Run long-context quality tests before adopting Q2.
The target is memory reduction without degrading:

```text
reasoning
code correctness
instruction recall
tool selection
long-distance retrieval
```

## 15. Stop the server

Preferred:

```text
Ctrl-C
```

If necessary:

```bash
pkill -f "mlx_lm.*server"
```

Then verify:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

## 16. If the Mac starts swapping heavily

Stop the test before the machine becomes unusable.
Check:

```bash
sysctl vm.swapusage
memory_pressure | tail -1
```

Then kill the server:

```bash
pkill -f "mlx_lm.*server"
```

Do not start a second large model until the first one is completely gone.

## 17. ContextVM memory rule

The HTTP server is an interim compatibility layer.
The planned runtime was a **single-process persistent engine** that:

```text
loads E2 once
keeps recurrent state once
keeps attention KV once
processes only newly appended tokens
executes tools in-process
virtualizes old KV pages
```

When `frankenstein_engine.py` / ContextVM V0 becomes functional, these server instructions should remain as a
fallback/reference implementation rather than the primary runtime.

## 18. Server for QWENUI

QWENUI is the SwiftUI chat client at `~/Developer/QWENUI`. It expects `http://127.0.0.1:8080` and the model
name `default_model`. The MLX server accepts that name and serves the loaded model.
Start the server:

```bash
~/Developer/MACQWEN/start_server.sh
```

KV settings in that script apply the section 47/48 findings in PROJECT.md:

```text
--quantized-kv-start 8192   fp16 KV below 8K, fused kernel, fastest decode
--kv-bits 4                 Q4 above 8K, bounded memory
--prefill-step-size 512     keeps the transient peak under the swap threshold
```

Build and launch the client:

```bash
~/Developer/QWENUI/bundle.sh
open ~/Developer/QWENUI/QWENUI.app
```

`bundle.sh` wraps the SwiftPM executable in a `.app`. A bare executable runs as a background process: no window, no menu
bar, and NSOpenPanel security-scoped bookmarks misbehave.
Verified working end to end:

```text
default_model name accepted
delta.reasoning streamed for the Thinking disclosure
delta.tool_calls returned in OpenAI format
/health polled by the client every 15s
```
