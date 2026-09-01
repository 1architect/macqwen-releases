# Qwen3.8-27B research

This file is the single active research record for the 27B runtime. It keeps
the original chronological log below. Current operation belongs in
[`handoff.md`](handoff.md).

Detailed dated findings and superseded runbooks remain in
[`docs/archive/qwen27b/`](../archive/qwen27b/).

## Research summary

- Append-only state removed repeated old prefill.
- Exact context images produced the largest accepted reuse gain.
- Paged KV made long logical context fit in bounded resident memory.
- Dense prefill reached a measured quantized matrix multiplication ceiling.
- Sparse MLP, Neural Engine offload, and dense FFN streaming failed their gates.
- Measured heterogeneous quantization produced the retained V4 direction.

## Original chronological record

_Last updated: 2026-08-20_

## 0. Purpose of this document

> Session of 2026-08-20: see `SESSION-2026-08-20.md` for the full record of
> that day, including the measured performance ceilings, the ideas that were
> tested and rejected, and the open list. Sections 45 to 52 below carry the
> detail.


The following record provides handoff context for continued development.

The project is to build a **high-quality local coding/agent model on a 16 GB Apple-Silicon MacBook Air M4**, using a custom MLX version of **Qwen3.8-27B**.

Requirements:

1. **Quality must reach or exceed the reference GGUF model.**
2. **Do not sacrifice reasoning/coding quality for speed or context size.**
3. The model must eventually support **large practical context**, with a target of **256K true/logical context** and potentially larger effective project memory.
4. Prompt ingestion must become much faster.
5. The runtime must remain stable on a **16 GB unified-memory Mac**.
6. The intended workload is a **real coding agent**, not synthetic benchmarks only.
7. The model itself is currently behaving well; focus should now shift from training to **runtime architecture, memory, context virtualization, and ingestion speed**.

The current best model is referred to as **Frankenstein E2**.


---

# 1. Hardware / environment

Machine:

- MacBook Air M4
- 16 GB unified memory
- fanless
- Apple Silicon / Metal

Python / MLX environment:

```text
venv:
~/mlx-qwen38-apple

python:
/Users/gioma/mlx-qwen38-apple/bin/python3

MLX:
0.32.1

MLX-LM:
0.32.0

mlx-lm commit used during investigation:
d06c5374a12e1f9384aad5fece583d7be9d2619d
```

llama.cpp build:

```text
$HOME/llama.cpp-apple

binaries:
$HOME/llama.cpp-apple/build/bin
```

Primary real-world coding repository used for testing:

```text
/Users/gioma/Developer/MACBAT
```

MacBat is a real macOS application and is the main coding-agent benchmark.


---

# 2. Models

## 2.1 GGUF reference / teacher

The GGUF model provides the quality reference.

```text
/Users/gioma/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf
```

This Unsloth GGUF uses highly heterogeneous IQ quantization and an importance matrix.

Important approximate tensor classes observed:

```text
MLP gate/up/down       IQ2_S
GDN qkv                IQ2_XS
GDN gate/z             IQ2_XXS
GDN alpha/beta         IQ1_M
GDN out                IQ3_XXS
GDN conv/norm          F32
full attention Q/K     IQ3_XXS
full attention V       IQ4_XS
full attention O       IQ2_S
Q/K norms              F32
embedding               Q2_K
output/lm_head          Q3_K
final norm              F32
```

The GGUF model remains the acceptance reference.

The scientific rule is:

> Autonomous generation quality compared with GGUF is the final acceptance criterion, not KL alone.


## 2.2 Original MLX v2

Base MLX model:

```text
/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-v2
```

Quantization strategy:

```text
MLP:
  gate/up  2-bit
  down     3-bit

GDN:
  qkv      3-bit
  z/a/b    4-bit
  out      4-bit

full attention:
  Q/K/V/O  4-bit

embedding:
  4-bit

lm_head:
  4-bit
```

The original MLX v2 was the strongest pre-distillation MLX model.

It was faster than some heavier alternatives, but still made semantic mistakes and had a tendency to enter long reasoning loops.


## 2.3 Current best model: Frankenstein E2

Frankenstein E2 is the model to preserve.

```text
/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1
```

Selected GGUF behavior was distilled into the original MLX v2 to produce this model.

Keep training disabled by default.

Current decision:

> Freeze E2. No epoch 3 and no additional trainable layers unless a new external benchmark demonstrates a real quality deficiency.

Fusion source:

```text
base:
  /Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-v2

delta:
  /Users/gioma/dwq-multisample/gguf_l63_head_multisample_epoch2.safetensors

fusion script:
  /tmp/fuse_dwq_multisample_epoch2.py
```

Fused target:

```text
/Users/gioma/.lmstudio/models/gioma/Qwen3.8-27B-Apple-MLX-GGUF-Distill-Multisample-E2-v1
```


---

# 3. Model architecture

Qwen3.8-27B is a hybrid architecture.

Important properties:

```text
64 text layers
hidden size: 5120
intermediate size: 17408

full-attention interval: every 4th layer

therefore approximately:
48 Gated DeltaNet / recurrent layers
16 full-attention layers
```

This distinction controls long-context memory design.

The recurrent/Gated DeltaNet layers use fixed-size state.

The full-attention layers are the part whose KV cache grows with sequence length.

Only 16 layers have growing Transformer KV state.

We only need to solve growing KV for the 16 full-attention layers.

Other useful architecture details:

```text
24 attention heads
4 KV heads
key/value head length: 256
maximum native context target: 262144
```

The MLX implementation currently returns:

```python
ArraysCache(size=2)
```

for linear/recurrent layers and:

```python
KVCache()
```

for full-attention layers.


---

# 4. Distillation/training work completed

## 4.1 Training objective

Goal:

Approximate the GGUF teacher's behavior while keeping the MLX quantized model format and speed advantages.

Only a tiny subset of model parameters was trained:

- quantization affine scales/biases in layer 63
- sparse lm_head scales/biases for teacher-relevant vocabulary rows

Packed quantized weights were not modified.


## 4.2 Trainable parameters

Exact trainables:

```text
layer63.self_attn.q_proj.scales  (12288,80) FP32
layer63.self_attn.q_proj.biases

layer63.self_attn.k_proj.scales  (1024,80)
layer63.self_attn.k_proj.biases

layer63.self_attn.v_proj.scales  (1024,80)
layer63.self_attn.v_proj.biases

layer63.self_attn.o_proj.scales  (5120,96)
layer63.self_attn.o_proj.biases

layer63.mlp.gate_proj.scales     (17408,80)
layer63.mlp.gate_proj.biases

layer63.mlp.down_proj.scales     (5120,272)
layer63.mlp.down_proj.biases

layer63.mlp.up_proj.scales       (17408,80)
layer63.mlp.up_proj.biases

head_scales                      (179953,80)
head_biases                      (179953,80)
```

No packed `.weight` tensor was trainable.


## 4.3 Prefix cache used during training

Frozen hidden states after layers 0 to 62 were cached:

```text
/Users/gioma/dwq-prefix63-v2/
```

Files:

```text
long_*.safetensors
trans_*.safetensors
```

Shape:

```text
(n_tokens, 5120)
```

dtype:

```text
bfloat16
```

The cache was created with `model.eval()` so the Gated DeltaNet layers used the exact inference path.


## 4.4 Trainer

Trainer:

```text
/tmp/train_dwq_multisample.py
```

Configuration:

```text
MODEL:
  original MLX v2

TEMP:
  2

LR:
  2e-6

EPOCHS:
  2

MAX_TRAIN_ROWS:
  256

ROW_CHUNK:
  16

SEED:
  20260820
```

Training samples:

```text
32 long train
12 transition train
8 clean transition examples additionally weighted x2

60 updates per epoch
```

Loss:

- top-1024 teacher KL at temperature 2
- sparse teacher vocabulary support
- exact sequence context through layer 63
- assistant rows only


---

# 5. Distillation results

## Baseline

Long evaluation:

```text
mean KL: .239298
top1:    75.66%
rows:    7690
```

Transition evaluation:

```text
mean KL: .212488
top1:    79.25%
rows:    1957
```

Combined score:

```text
.4517865
```


## Epoch 1

Long:

```text
KL:   .218905
top1: 76.57%
```

Transition:

```text
KL:   .185671
top1: 80.68%
```

Combined:

```text
.4045761
```


## Epoch 2

Long:

```text
mean KL: .216568
top1:    76.70%
```

Transition:

```text
mean KL: .180798
top1:    81.45%
```

Combined:

```text
.3973663
```

Relative improvement over baseline:

```text
long KL reduction:        ~9.5%
transition KL reduction: ~14.9%
combined improvement:    ~12%
```

All 10 held-out long trajectories improved KL.

All 4 held-out transition trajectories improved KL.

Epoch 2 slightly outperformed epoch 1 with no obvious held-out overfit.


---

# 6. Autonomous behavior evidence

A useful test used eval-only transition prompts that the optimizer never directly trained on.

Results:

## e_percentage

```text
GGUF:
  generated 130
  closed correctly

original MLX v2:
  generated 227
  closed correctly

E2:
  generated 171
  closed correctly
```


## e_lock_order

```text
GGUF:
  generated 555
  closed correctly

original MLX v2:
  generated 900
  HIT TOKEN CAP
  DID NOT CLOSE

E2:
  generated 485
  closed correctly
  answer correct
```


## e_complexity

```text
GGUF:
  generated 500
  closed correctly

original MLX v2:
  generated 890
  closed correctly

E2:
  generated 306
  closed correctly
  answer correct
```


Totals:

```text
GGUF:            1185 tokens
original v2:   >=2017 tokens
E2:               962 tokens
```

E2 closed all 3 cases.

Original v2 closed 2/3.

E2 was about 19% shorter than GGUF while remaining correct in these examples.

These results first showed behavioral generalization beyond teacher-forced imitation.


---

# 7. Invalid benchmark

The following Python asyncio prompt is **not a valid termination benchmark** because both E2 and the GGUF teacher looped.

Prompt:

```text
You are reviewing a production Python asyncio service.

async def process_all(items):
    results = {}
    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(process(item, results))
    return results

async def process(item, results):
    value = await remote_call(item)
    results[item.id] = value

The remote call can fail, duplicate item IDs may occur, cancellation is important,
and the caller requires output in the same order as the input.

Treat this implementation as untrusted.

Identify the concurrency, error-handling, cancellation, ordering, and API-design
problems. Then propose a production-quality Python 3.13 design and explain the
tradeoffs.
```

Results:

```text
E2:
  prompt 209
  generation 1800
  ~7.7 tok/s
  peak ~11.94 GB
  did not close

GGUF:
  generation 2400 / 2400
  did not close
```

Therefore do not interpret failure on this prompt as an E2 regression.


---

# 8. Real-world MacBat coding-agent benchmark

The real application repository is:

```text
/Users/gioma/Developer/MACBAT
```

The benchmark task:

> Review `main.swift` in the context of the rest of the MacBat project and produce concrete improvement suggestions.

The model is expected to use repository tools rather than receive the file manually.


## Original v2 failure

With the first terminal-agent harness:

```text
TURN 5
prompt=4637
completion=1800
time=509.38s
```

It hit the generation limit entirely inside reasoning and never emitted a tool call or final answer.

The run established a baseline failure:

```text
original v2
survived: turns 1 to 4
failed:   turn 5
failure:  endless internal reasoning / no action
```


## E2 behavior

Frankenstein E2 made better decisions.

It kept choosing useful actions and continued inspecting the repository well beyond the point where v2 failed.

We observed:

> "our frankestein was doing really well logic wise"

The run was terminated because memory eventually grew enough to crash the Mac after reading multiple files.

Therefore:

E2 logic meets the preservation gate. Runtime memory and context architecture are the next bottlenecks.


---

# 9. Terminal agent work

Several generations of the agent harness were created.

## v1

File previously created:

```text
macbat_readonly_agent.py
```

Problems:

- `stream=False`
- no live generation
- replayed full prior `<think>` contents into later turns
- caused context explosion
- server prompt cache was explicitly disabled


## v2

File:

```text
macbat_readonly_agent_v2.py
```

Improvements:

- native MLX/Qwen tool calls
- SSE streaming
- visible reasoning
- reasoning omitted from conversation history
- read-only filesystem tools

Read-only tools:

```text
find_files
list_dir
read_file
search
```

Repository root restricted to:

```text
/Users/gioma/Developer/MACBAT
```


## v3 / evidence-ledger experiment

File created:

```text
macbat_readonly_agent_v3.py
```

Concept:

- maintain a bounded active context
- when estimated context becomes too large, generate a compact evidence ledger
- discard raw tool history
- continue from the ledger
- source files remain on disk and can be re-read

This provides a fallback and a proof of bounded agent memory.

However, the current direction is to go beyond simple summarization.


---

# 10. Prompt ingestion observations

During one earlier agent run, large prompts were repeatedly reprocessed because prompt caching was disabled.

Observed prompt sizes:

```text
392
988
1062
2956
3595
5241
8382
10047
10283
14986
...
```

The server repeatedly reported:

```text
Prompt Cache: 0 sequences, 0.00 GB
```

because it had been launched with:

```text
--prompt-cache-size 0
```

At approximately 10,283 prompt tokens, prefill took about 307 seconds.

Approximate sustained ingest:

```text
~33.5 tokens/sec
```

The result exceeded the estimate.

Conclusion:

> Raw ingest speed is not the only problem. Reprocessing the same context repeatedly was wasting most of the available throughput.

Future architecture should ensure:

> A token that has already been neurally processed should normally never be processed again.


---

# 11. Current MLX server behavior

MLX-LM server already supports an LRU prompt cache.

Relevant behavior in the exact version used:

- server tokenizes prompts into segments
- it finds the nearest cached prefix
- cached prefix tokens can be reused
- completed assistant cache can be inserted back into the prompt cache
- default server prompt-cache size is 10 unless overridden

The earlier benchmark disabled this functionality intentionally.

For a real coding agent, caching should stay enabled.


---

# 12. KV cache facts

MLX currently includes:

```text
KVCache
QuantizedKVCache
RotatingKVCache
ChunkedKVCache
ArraysCache
```

`KVCache.to_quantized()` exists.

The generation path supports:

```text
--kv-bits
--kv-group-size
--quantized-kv-start
```

in the normal `mlx_lm.generate` implementation.

The standard HTTP server did not originally expose these CLI options through its sequential generation call.

A small patch script was created:

```text
patch_mlx_server_kv.py
```

Its job was to expose:

```text
--kv-bits
--kv-group-size
--quantized-kv-start
```

through the server.

The patch script makes a backup of `server.py` first.


---

# 13. Approximate KV memory math

Only the 16 full-attention layers have linearly growing KV.

Rough BF16/F16 attention-KV cost:

```text
~64 KB per context token
```

Approximate full KV:

```text
8K    ~0.5 GB
16K   ~1 GB
32K   ~2 GB
64K   ~4 GB
128K  ~8 GB
256K  ~16 GB
```

This makes native full-precision 256K impossible on a 16 GB machine.


## Q4 rough estimate

Including scale/bias metadata:

```text
~18 KB/token
```

Approximate:

```text
32K   ~0.56 GB
64K   ~1.1 GB
128K  ~2.25 GB
256K  ~4.5 GB
```


## Q2 rough estimate

```text
~10 KB/token
```

Approximate:

```text
32K   ~0.31 GB
64K   ~0.63 GB
128K  ~1.25 GB
256K  ~2.5 GB
```

This makes genuine 256K much more realistic, though quality must be tested.


---

# 14. Existing long-context roadmap

Before the ContextVM idea, the planned true-context progression was:

```text
32K
→ 64K
→ 128K
→ 192K
→ 256K
```

Target final KV settings were roughly:

```text
--kv-bits 2
--kv-group-size 64
--quantized-kv-start 0
```

with no rotating-cache truncation.

Quality tests required at each stage:

```text
needle retrieval
multi-needle retrieval
distractor resistance
code lookup
early instruction recall
long-distance reasoning
```

If Q2 KV loses too much quality, use mixed precision rather than sacrificing model weights.


---

# 15. New strategic direction: ContextVM

Simple compaction/evidence ledgers are useful, but the desired architecture is more ambitious.

The new goal is to build a **virtual-memory system for model context**.

Working name:

```text
ContextVM
```

Core idea:

> Give the model a large logical context while keeping only a bounded physical working set in unified memory.


## 15.1 Hybrid architecture properties

Qwen3.8-27B is hybrid:

```text
48 recurrent / Gated DeltaNet layers
16 full-attention layers
```

The 48 recurrent layers already summarize arbitrarily long history in fixed-size recurrent state.

Only the 16 attention layers need a growing KV history.

Therefore ContextVM only needs to virtualize the attention KV.


---

# 16. ContextVM concept

Break attention KV into pages.

Possible page size:

```text
256 tokens
```

Logical token tape:

```text
page 0000 = tokens 0 to 255
page 0001 = tokens 256 to 511
page 0002 = tokens 512 to 767
...
```

Each page contains attention KV for all 16 full-attention layers.


## Memory tiers

### Tier 0: pinned

Always resident:

```text
system prompt
tool contract
important instructions
critical request constraints
critical project facts
```

These must never be evicted.


### Tier 1: hot

Recent context.

Example:

```text
last 4K to 8K tokens
```

Potentially Q4 or higher precision.


### Tier 2: retrieved

Old context relevant to the current reasoning turn.

Example:

```text
another 4K to 16K tokens
```

Loaded only when relevant.


### Tier 3: cold

All other historical KV.

Stored as compressed pages on SSD.

Potentially Q2.


---

# 17. ContextVM target behavior

Physical memory should depend on:

```text
pinned + hot + retrieved
```

not total session length.

Example:

```text
2K pinned
8K hot
8K retrieved

physical attention context:
~18K tokens
```

while logical history might contain:

```text
256K
1M
or more tokens
```

Cold history remains available for exact retrieval.


---

# 18. Difference from summarization

Do NOT make the primary memory system:

```text
raw context
→ summary
→ delete original
```

Instead:

```text
raw context
→ compute KV once
→ page KV to SSD
→ index page
→ reload exact old page when needed
```

The principle is:

> Read something once. Never neural-process it again.

An evidence ledger may still exist as a cheap metadata layer, but it should not be the only memory of old code.


---

# 19. Page metadata / retrieval

Each KV page should have metadata.

Potential fields:

```text
token_start
token_end

source type:
  conversation
  tool output
  repository source

file path
line range
symbol names

semantic embedding/signature
attention-key signature
importance score
last-access time
pinning state
precision tier
```

For MacBat, example metadata:

```text
file:
  MacBat/main.swift

lines:
  420 to 610

symbols:
  BatteryMonitor
  updateEstimate
  SentinelManager
```

A page router should select relevant old pages before a reasoning/tool turn.


---

# 20. Turn-level retrieval

Do not perform SSD/page routing for every generated token.

That would destroy generation speed.

Instead page selection should happen:

```text
once per request turn
once per tool result
once per high-level reasoning phase
or when task/symbol focus changes
```

Selected pages remain resident for the duration of the current reasoning phase.


---

# 21. Stateful single-process engine

The OpenAI-compatible HTTP server is useful for compatibility, but it is not the ideal final runtime.

The desired architecture is a direct single-process engine:

```text
FrankensteinEngine

- model loaded once
- tokenizer loaded once
- token tape persists
- GDN recurrent state persists
- attention KV pages persist
- tool controller runs in same process
- repository index lives in same process
- page router lives in same process
- memory manager owns the physical context budget
```

This avoids:

```text
HTTP serialization
chat-history reconstruction
re-tokenization
prefix lookup
cache duplication/extraction
repeated prefill
```

For a tool result containing 600 new tokens:

Current conceptual behavior:

```text
reconstruct conversation
find/replay context
process a large prompt
```

Desired behavior:

```text
existing model state
+ 600 new tokens
→ process only the 600 new tokens
```


---

# 22. Swift-aware repository memory

For MacBat and other Swift projects, the model should not spend expensive model tokens discovering basic source structure.

Add a code index using one of:

```text
SourceKit
SwiftSyntax
sourcekit-lsp
indexstore-db
```

Desired agent tools:

```text
definition(symbol)
references(symbol)
callers(symbol)
callees(symbol)
read_symbol(symbol)
related_symbols(symbol)
file_outline(path)
```

This allows the agent to retrieve the smallest relevant source region.

Potential benefit:

> 10x less source-code ingestion for many repository tasks.


---

# 23. Ingestion speed strategy

Ingestion speed has three layers.

## A. Avoid irrelevant ingestion

Use:

```text
symbol graph
source index
retrieval
targeted file reads
```

Do not send whole files if only one type/function is relevant.


## B. Never ingest the same tokens twice

Use:

```text
persistent model state
persistent KV
append-only token tape
```

This direction likely gives a larger practical gain than kernel tuning.


## C. Optimize genuine new-token prefill

Once A and B work, tune:

```text
prefill-step-size:
512
1024
1536
2048
possibly 4096
```

Measure:

```text
prompt tok/s
peak RAM
swap
generation tok/s
quality
```


---

# 24. Current prefill performance target

Measured sustained ingest is already around:

```text
~33 tok/s
```

on a long prompt with conservative settings.

Desired milestones:

```text
first milestone:
>= 30 tok/s sustained while stable

next:
40 to 60+ tok/s if kernels and hardware allow it
```

But perceived agent performance should mostly come from reducing the amount of actual re-prefill.


---

# 25. Mixed-precision memory hierarchy idea

Do not force one KV precision globally.

Possible hierarchy:

```text
last 256 to 512 tokens:
  FP16/BF16 or Q8

hot recent context:
  Q4

retrieved old pages:
  Q4 or Q2

cold SSD pages:
  Q2
```

Precision becomes a property of **memory temperature**.

Further work requires quality validation.


---

# 26. Potential ContextVM difficulty

Technical issue:

The current attention implementation expects a conventional contiguous KV cache.

ContextVM may require modification of the attention path so it can attend over:

```text
pinned pages
+ selected cold pages
+ recent hot pages
```

without reconstructing the entire historical KV into one large contiguous array.

Potential implementation approaches:

1. concatenate selected page tensors before an attention call
2. create a paged KV-cache abstraction that presents a logical concatenation
3. implement block/page attention in Metal/MLX
4. compute attention per page and combine softmax statistics exactly

Option 4 is particularly interesting.

For exact attention over pages, compute per-page:

```text
max logit
exp-sum
weighted value sum
```

Then merge page-level softmax statistics using numerically stable online softmax.

This can avoid physically concatenating all K/V tensors.

The method is a candidate for the main paged-attention kernel.


---

# 27. Exact paged-attention idea

For one query `q`, attention is:

```text
softmax(qK^T) V
```

Suppose KV is split into pages.

For every selected page j compute:

```text
m_j = max(logits_j)
l_j = sum(exp(logits_j - m_j))
o_j = sum(exp(logits_j - m_j) * V_j)
```

Merge pages incrementally:

```text
m = max(m_old, m_new)

l = exp(m_old-m)*l_old
  + exp(m_new-m)*l_new

o = exp(m_old-m)*o_old
  + exp(m_new-m)*o_new
```

Final output:

```text
o / l
```

The method is mathematically equivalent to attention over concatenated KV, modulo numerical precision.

Therefore:

> ContextVM can potentially attend over discontiguous memory pages without ever creating one giant contiguous KV tensor.

This direction has high prototype value.


---

# 28. SSD-backed page store

Cold pages should live on SSD.

Possible layout:

```text
~/.frankenstein/contextvm/<session>/

manifest.json
pages/
  000000.kv
  000001.kv
  ...
```

Each page could use:

```text
safetensors
custom mmap format
or directly packed MLX quantized buffers
```

For speed, prefer:

```text
memory-mapped binary pages
```

over deserializing generic objects each time.

Possible requirements:

```text
asynchronous prefetch
LRU resident page pool
strict physical-memory budget
dirty-page handling
page pinning
page precision conversion
```


---

# 29. ContextVM router

Start simple.

V0 router:

```text
recent pages
+ pages whose metadata matches current file/symbol
```

Later:

```text
semantic vector retrieval
query hidden-state similarity
attention-key centroid similarity
hybrid lexical + semantic scoring
```

Do not begin with a complex learned router.

First prove the memory system itself.


---

# 30. ContextVM development stages

## V0: Stateful direct engine

Goal:

Remove HTTP and prove persistent append-only inference.

Requirements:

```text
load E2 once
maintain one mutable prompt cache
append new messages/tool outputs
process only new tokens
stream generation
execute read-only tools
```

Success:

```text
multiple MacBat tool turns
without re-prefilling prior conversation
stable memory
```


## V1: Memory-budgeted quantized KV

Add:

```text
Q4 full-attention KV
strict resident-memory budget
telemetry
```

Measure:

```text
resident cache bytes
peak MLX memory
system memory
swap
```


## V2: Paged full-attention KV

Split full-attention cache into 256-token pages.

Initially keep all pages in RAM.

Goal:

Prove paged attention produces output matching contiguous attention.


## V3: SSD spill

Cold pages move to disk.

Keep:

```text
pinned
hot
retrieved
```

resident.

Goal:

Logical context grows without proportional RAM growth.


## V4: Page router

Retrieve relevant historical pages.

Start with metadata/symbol-based routing.


## V5: Swift code index

Integrate SourceKit/SwiftSyntax/indexstore.

Agent reads symbols, not arbitrary giant source chunks.


## V6: Mixed precision

Use:

```text
hot Q4
cold Q2
possibly latest FP16/Q8
```

Validate quality.


## V7: True 256K benchmark

Test genuine long-context behavior.

Compare against GGUF where possible.


---

# 31. ContextVM implementation sequence

The next agent should NOT immediately continue model training.

Start here:

### Step 1

Inspect the local MLX-LM implementation corresponding to:

```text
~/mlx-qwen38-apple
```

Find the exact files for:

```text
qwen3_5.py
cache.py
generate.py
server.py
gated_delta.py
attention helpers
```

Do not assume upstream code exactly matches the locally patched environment.


### Step 2

Build a small direct inference prototype:

Suggested file:

```text
frankenstein_engine.py
```

Requirements:

```text
load E2 once
apply Qwen chat template
maintain persistent model cache
append one new request/tool segment at a time
stream output
show:
  new prompt tokens processed
  generation tokens
  prompt tok/s
  generation tok/s
  peak MLX memory
  cache bytes
```


### Step 3

Prove the direct engine can do:

```text
request
→ tool call
→ tool result
→ second reasoning turn
```

while the second turn processes only newly appended tokens.

Do not build ContextVM until this invariant is verified.


### Step 4

Add full-attention Q4 KV cache.

Because Qwen's `make_cache()` returns mixed:

```text
ArraysCache
KVCache
```

quantize only cache entries supporting:

```text
to_quantized()
```


### Step 5

Measure MacBat agent memory growth.

Use the real repository:

```text
/Users/gioma/Developer/MACBAT
```

Task:

```text
Review main.swift in context of the project.
```

Record after every tool turn:

```text
active sequence length
new tokens processed
cache nbytes
MLX active memory
MLX peak memory
system free memory
swap usage
```


### Step 6

Prototype `PagedKVCache`.

Do not start with SSD.

First prove:

```text
contiguous KV attention
vs
paged KV attention
```

produces near-identical logits.


### Step 7

Implement online-softmax page merging if necessary.

The target is exact/discontiguous attention without materializing one large concatenated K/V tensor.


---

# 32. Quality acceptance hierarchy

Always evaluate in this order:

```text
1. autonomous generation quality against GGUF
2. real coding-agent behavior
3. closure / looping behavior
4. correctness of tool navigation
5. long-context recall
6. memory
7. ingest speed
8. generation speed
```

Do not optimize speed by silently degrading items 1 to 5.


---

# 33. Coding benchmark criteria

For MacBat or another real repo, evaluate:

```text
files inspected
correct cross-file relationships
correctness of findings
false-positive bugs
Swift/macOS knowledge
architectural insight
usefulness of recommendations
reasoning loops
tool efficiency
repeated reads
final answer quality
prompt tokens processed
cached/reused tokens
generation tokens
time
memory
swap
```


---

# 34. Swift/macOS review expectations

When reviewing code, the model should understand things like:

```text
@MainActor
Swift concurrency
actors
TaskGroup
Sendable
cancellation
actor reentrancy
macOS lifecycle
SwiftUI/AppKit ownership
Observable state
process/background work
```

Avoid superficial style-only suggestions.


---

# 35. Scientific rules

1. **Do not claim success from KL alone.**

2. **Do not change multiple major variables at once.**

3. When comparing model quality:
   - same prompt
   - same max output
   - same temperature/sampling
   - same tool contract
   - same repository state

4. Fresh process per giant model when comparing separate models.

5. Do not load two 27B models simultaneously on the 16 GB machine.

6. When a prompt causes both GGUF and E2 to loop, it is not a valid E2 regression test.

7. Never treat faster output as success if correctness drops.

8. Keep E2 frozen unless external evidence shows a real model deficiency.


---

# 36. Current model-quality conclusion

The current working conclusion is:

> Frankenstein E2 is good enough logically to stop touching the weights for now.

It demonstrated:

```text
better held-out KL
better held-out transition closure
shorter reasoning than original v2
correct autonomous generalization
much better behavior than v2 during the real MacBat agent run
```

The MacBat run failed because of **runtime memory growth**, not because E2's reasoning failed.

Therefore the project has moved from:

```text
MODEL TRAINING
```

to:

```text
RUNTIME / MEMORY SYSTEMS ENGINEERING
```


---

# 37. Long-term goal

The final system should feel like:

```text
A strong 27B local coding agent
running on a 16 GB fanless MacBook Air
with GGUF-class reasoning quality
large project memory
low-latency iterative tool use
and large effective context.
```

Desired operating behavior:

```text
First repository inspection:
  expensive but manageable

Later turns:
  process only new information
  reuse already-computed memory
  retrieve exact old code/context when needed

Large project:
  does not cause RAM to grow indefinitely

Long session:
  does not require re-ingesting everything
```

The ambitious form is:

> A virtual-memory operating system for the context of a hybrid LLM on Apple Silicon.


---

# 38. Useful existing local artifacts

Potentially relevant files:

```text
/tmp/train_dwq_multisample.py
/tmp/cache_dwq_prefix63.py
/tmp/fuse_dwq_multisample_epoch2.py

/Users/gioma/dwq-multisample/
  gguf_l63_head_multisample_epoch1.safetensors
  gguf_l63_head_multisample_epoch2.safetensors

/Users/gioma/dwq-prefix63-v2/

/Users/gioma/dwq42-teacher
/Users/gioma/dwq16-transition

/tmp/dwq42_prompts.bin
/tmp/dwq42_manifest.json

/tmp/dwq_transition16_prompts.bin
/tmp/dwq_transition16_manifest.json
```

Teacher utility:

```text
source:
/tmp/gguf_dwq_teacher42.cpp

binary:
$HOME/llama.cpp-apple/build/bin/gguf-dwq-teacher42
```


---

# 39. Teacher corpus

Long corpus:

```text
42 total
32 train
10 eval
```

Transition corpus:

```text
16 total
12 train
4 eval
```

Held-out long evaluation names:

```text
dev_swift
dev_python_asyncio
eval_kotlin
eval_ledger
eval_logic
eval_math
eval_api
eval_bug
eval_reasoning
eval_requirements
```

Held-out transition names:

```text
e_percentage
e_idempotency
e_lock_order
e_complexity
```


---

# 40. Model saving patch

The local MLX-LM save path was previously patched to avoid large transient save memory.

Installed `mlx_lm/utils.py` contains behavior equivalent to:

```python
for value in shard.values():
    mx.eval(value)
    mx.synchronize()
```

and model shards were limited to roughly 1 GB.

Be careful not to overwrite local MLX modifications without checking.


---

# 41. Memory safety rules

This machine has only 16 GB unified memory.

Avoid:

```text
two 27B models loaded simultaneously
large duplicated prompt caches
large unquantized 64K+ KV
large prefill chunks without measurement
multiple retained LRU caches for giant prompts
```

When experimenting, monitor:

```bash
sysctl vm.swapusage
memory_pressure | tail -1
```

and model process memory with:

```bash
ps
vmmap -summary <PID>
```


---

# 42. Development interaction requirements

The owner prefers:

- direct technical answers
- pasteable Terminal commands
- measurable experiments
- no GUI detours
- no vague "try this" speculation
- preserve scientific comparability
- optimize for actual usefulness, not benchmark vanity

When implementing something, provide:

```text
exact file
exact command
what success looks like
what telemetry to capture
what result would change the next decision
```


---

# 43. Development handoff prompt

A good starting instruction is:

```text
Read PROJECT.md completely.

We have a trained Qwen3.8-27B MLX model called Frankenstein E2 that is already
reasoning well. Do not retrain it.

Your first task is to inspect the local mlx-lm implementation in
~/mlx-qwen38-apple and build ContextVM V0: a direct single-process agent engine
that loads E2 once, maintains persistent recurrent/KV state, processes only newly
appended tokens after each tool call, streams output, and reports exact cache and
memory telemetry.

Use /Users/gioma/Developer/MACBAT as the real benchmark repository.

Do not implement SSD paging yet. First prove persistent append-only inference
works and memory does not duplicate on every turn.
```


---

# 44. Recorded project state

> Sections 45 and 46 supersede the runtime plan below.
> Section 45 records the completed ContextVM V0.
> Section 46 revises the stage order in section 30.


### Keep

```text
Frankenstein E2
GGUF teacher
training corpora
epoch2 delta
prefix cache if disk permits
```

### Freeze

```text
model weights
```

### Focus now

```text
single-process persistent inference
KV quantization
bounded resident memory
paged attention
SSD-backed KV
code-aware retrieval
prompt ingestion efficiency
256K context
```

### Core principle

> Preserve model quality through runtime memory design.


# 45. ContextVM V0: completed 2026-08-20

Steps 1 to 5 of section 31 are done.

File:

```text
/Users/gioma/Developer/MACQWEN/frankenstein_engine.py
```

Modes:

```text
--mode selftest   tokenizer-only, no model load
--mode demo       three-turn tool cycle, synthetic tool result
--mode agent      real MacBat read-only review benchmark
```


## 45.1 Local MLX-LM facts verified

Confirmed in the installed environment, not assumed from upstream:

```text
qwen3_5.py make_cache():
  ArraysCache(size=2)  for linear/GDN layers
  KVCache()            for full-attention layers
  is_linear = (layer_idx + 1) % 4 != 0

generate.py maybe_quantize_kv_cache():
  skips any cache without to_quantized()
```

Therefore `--kv-bits 4` works on the mixed cache list with no patch.
Section 31 step 4 needs no extra code.

Every token that `stream_generate` yields is already inside the cache.
The cache length always equals prompt length plus tokens yielded.


## 45.2 Chat-template facts

The model has no `chat_template` in `tokenizer_config.json`.
The template is in `chat_template.jinja`.

The tool-call format is XML, not JSON:

```text
<tool_call>
<function=read_file>
<parameter=path>
Sources/MacBat/main.swift
</parameter>
</function>
</tool_call>
```

Tool results are a **user** turn:

```text
<|im_start|>user
<tool_response>
...
</tool_response><|im_end|>
```

Default `reasoning_effort` is `xhigh`.


## 45.3 Append-only segment builder

The engine never re-renders history. It appends one segment per turn.

After `<|im_end|>` the template writes `\n`, so each new segment starts with `\n`.
If a turn stops on `length`, the cache has no `<|im_end|>`, so the next
segment must start with `<|im_end|>` first.

The `selftest` mode proves the incremental tape is token-identical to
`apply_chat_template` on the same conversation:

```text
incremental tokens: 930
template tokens   : 930
SELFTEST: PASS
```


## 45.4 Invariant result

`cache_tokens == len(tape)` was true after every turn of every run.

Demo, 3 turns:

```text
turn 1: new  57 tok, ctx  89
turn 2: new  23 tok, ctx 362
turn 3: new  27 tok, ctx 519
prefill saved: 80.8%
```

The stateless path would have re-processed 470 tokens. The engine
processed 107.


## 45.5 MacBat agent run: memory

E2 + Q4 KV, prefill 1024, temp 0:

```text
turn  ctx    kv total  attn kv  mlx peak  host free  swap
1      884   0.22 GB   0.07 GB  12.90 GB  1.02 GB    1.99 GB
2     1328   0.18 GB   0.03 GB  12.90 GB  1.32 GB    1.93 GB
3     1460   0.18 GB   0.03 GB  12.90 GB  1.34 GB    1.92 GB
4     2542   0.20 GB   0.05 GB  13.07 GB  1.45 GB    2.11 GB
5     5931   0.27 GB   0.11 GB  13.14 GB  0.83 GB    1.92 GB
6     6065   0.27 GB   0.11 GB  13.14 GB  0.75 GB    1.81 GB
```

New facts:

```text
the 48 GDN layers cost a constant ~0.16 GB
only the 16 full-attention layers grow
Q4 conversion is visible at turn 2 (0.22 -> 0.18 GB)
swap stayed flat
no crash
```

The documented MacBat crash did not reproduce. That failure came from
HTTP history replay, not from the model and not from KV growth.

The run was stopped by hand at turn 7. It did not fail.


## 45.6 Generation-speed bottleneck

Prefill is healthy:

```text
22 - 42 tok/s
```

Generation falls as context grows:

```text
ctx   884 -> 8.6 tok/s
ctx  2542 -> 6.5 tok/s
ctx  5931 -> 4.0 tok/s
```

Memory is no longer the limit at this context size. Decode speed is.
A 1600-token reasoning turn costs about 8 minutes at 6K context.


## 45.7 Corrected harness failures

### Greedy repetition loop

Run 1 used temp 0 with no penalty. In turn 5 the model repeated its own
tail verbatim while reasoning about a Swift `contains` overload. It burned
all 1600 tokens and never emitted `</think>`.

Fixes added:

```text
--repetition-penalty 1.05 --repetition-context-size 128
loop guard: stop the turn when the tail repeats 3 times
```

Run 2 with the penalty showed no loop.

### Truncated reasoning treated as a final answer

A turn that ends on `length` with no tool call is **not** an answer.

The harness now forces closure:

```text
append "\n</think>\n\n"   (3 tokens)
generate again
```

Result in run 2, turn 6:

```text
new prompt tokens: 3
the model immediately produced two correct read_file calls
```

This converts the original v2 killer failure into a 3-token recovery,
and it keeps the earlier reasoning in the KV cache.


## 45.8 Next steps

```text
1. test --reasoning-effort medium for tool-navigation turns
   xhigh is right for the final review, wasteful for "read this file next"
2. attack decode speed at long context, not memory
3. V1: strict resident-memory budget (telemetry already in the engine)
4. V2: PagedKVCache, prove paged == contiguous logits
```

Do not re-open training. The model reasoned correctly, verified its own
hypotheses against the source, and navigated the repository without help.

# 46. Sparse paged attention direction

_Added 2026-08-20, after the V0 measurements in section 45._

This section revises the stage order in section 30. It does not replace the
exact-merge mathematics in section 27, which stays correct.

## 46.1 V0-V7 plan gap

Exact paged attention solves memory **layout**. It does not solve **speed**.

If the merge runs over every page, it reads the same bytes as a contiguous
cache. Decode costs the same. Speed comes only from reading fewer pages.

Section 29 puts the router at V4 and calls it a later refinement. That order
delivers speed last, and the system is unusable before it arrives.

Correction:

> The router is the speed mechanism, not a convenience.
> V2 and V4 are one component: sparse top-k page attention with exact merge.

## 46.2 Sparse-recall properties

The 48 GDN layers are recurrent. They have already integrated every token,
at fixed size, for free. Only the 16 attention layers need explicit recall.

```text
GDN layers        global gist of everything, zero marginal cost
attention layers  exact lookup of selected pages only
```

A pure transformer that drops unselected KV loses that content completely.
Here the content survives in the recurrent state.

The hybrid model can retain unselected content in recurrent state.
This property makes the 256K target feasible on 16 GB.

## 46.3 Memory arithmetic

Measured in section 45:

```text
attention KV at Q4   19 KB per token
GDN state            0.16 GB constant
```

With top-k selection at k = 4096 tokens:

```text
GDN state                       0.16 GB
selected pages (k = 4096)       0.078 GB
page centroids for 256K         0.033 GB
------------------------------------------
total resident context cost     ~0.27 GB
```

Resident context cost does not depend on logical context length.
256K costs the same resident memory as 6K costs today.

On SSD:

```text
256K image at Q4    4.8 GB
256K image at Q2    2.5 GB
```

Decode reads 10 GB of weights plus about 78 MB of KV. That is nearly the
same as 1K context today.

Target:

> decode speed stays near 8 tok/s at 256K logical context

## 46.4 Persistent repository image

256K tokens at 30 tok/s is 2.4 hours. This alone blocks large context.

Solution:

```text
prefill the repository once, in a fixed canonical order
persist GDN state + all KV pages to SSD
every later session mmaps the image and appends after it
```

Ingestion then amortizes to zero.

The image works as a fixed-order prefix, as required by the recurrent layers.

Constraint to respect:

> Two independently prefilled documents cannot be composed for the GDN path.
> Build one ordered image. Do not try to merge separate images.

## 46.5 Design sketch

```text
page size            256 tokens
page store           mmap'd binary, Q4 hot / Q2 cold
per page per layer   key centroid (or min/max bounds) held in RAM
selection            every 32-64 decode tokens, not per token
always selected      pinned pages + recency window
attention            gather selected pages, fused kernel, exact online merge
```

Centroid cost for 256K:

```text
1000 pages x 16 layers x 4 kv heads x 256 dim x 2 bytes = 33 MB
```

## 46.6 Risks

```text
selection misses cause wrong answers
  mitigate: pinned pages, recency window, generous k, quality gate

the kernel is the hard part
  the current quantized attention path is composed, not fused
  a paged version written the same way will be slower, not faster
  this needs real Metal work

fanless thermal throttling on sustained load
```

Quality gate before accepting any sparse configuration:

```text
needle retrieval
multi-needle retrieval
distractor resistance
code lookup
MacBat agent benchmark
```

## 46.7 Decisive first measurement

One cheap experiment decides whether a Metal kernel is on the critical path.

Run on a **quiet machine**, no other heavy processes:

```bash
"$HOME/mlx-qwen38-apple/bin/python3" \
~/Developer/MACQWEN/bench_decode.py --rungs 1024,2048,4096,8192

"$HOME/mlx-qwen38-apple/bin/python3" \
~/Developer/MACQWEN/bench_decode.py --kv-bits 4 --rungs 1024,2048,4096,8192
```

Interpretation:

```text
fp16 holds ~8 tok/s and Q4 collapses
  -> the composed quantized path is the bottleneck
  -> the fused paged kernel is the critical path

both collapse
  -> memory pressure / paging is the bottleneck
  -> reduce resident memory first
```

Relevant code fact, verified in section 45:

```text
mlx_lm/models/base.py
  fp16 KV -> mx.fast.scaled_dot_product_attention   (fused)
  Q4 KV   -> quantized_matmul, softmax, quantized_matmul  (composed)
```

## 46.8 Revised stage order

```text
S0  DONE   stateful append-only engine (section 45)
S1         decisive measurement of 46.7
S2         Swift symbol index; stop ingesting tokens that are not needed
S3  DONE   PagedKVCache with exact online-softmax merge (section 47)
           acceptance: logits match contiguous attention
S4  DONE   page centroids + top-k selection (section 48)
           acceptance: quality gate of 46.6, decode speed flat to 32K
S5         fused Metal paged attention kernel if 46.7 requires it
S6  PART   SSD page store done (section 50); repository image pending
S7         true 256K benchmark against GGUF where possible
```

Section 30 stages V1 and V3 are absorbed. The engine already reports the
V1 telemetry. SSD spill without selection is not worth building alone.

# 47. S3 complete: paged KV with exact merge

_2026-08-20._

File:

```text
/Users/gioma/Developer/MACQWEN/paged_kv.py
```

Contents:

```text
PagedKVCache        full-attention KV stored as fixed-size pages
paged_attention     exact online-softmax merge from section 27
install()           patches mlx_lm.models.qwen3_next.scaled_dot_product_attention
make_paged_cache()  ArraysCache for GDN layers, PagedKVCache for attention
--selftest          unit test, no model load
--model-test        generation equivalence on the real model
```

## 47.1 Acceptance result

```text
prompt 233 tokens, page_size 256, 2 pages
identical tokens: 32/32, no divergence
MODEL TEST: PASS
```

The section 27 mathematics is correct and works on the real model.

## 47.2 Precision requirement

Softmax statistics **must** accumulate in float32.

Measured relative error against a float32 reference:

```text
bfloat16 accumulation   6.3e-3
float32 accumulation    4.9e-3
```

A 6e-3 error per attention layer compounds through 64 layers and produced a
48% logit difference in the first attempt, even though argmax still matched.

## 47.3 Float32 acceptance reference

In bfloat16 `mx.fast.scaled_dot_product_attention` is itself an approximation.

Scored against a float32 reference:

```text
                bfloat16    float16
fused kernel    6.06e-3     7.07e-4
paged           4.88e-3     6.77e-4
```

The paged implementation is **more accurate than the fused kernel**.

Therefore:

> Never accept or reject paged attention by comparing it with the fused
> kernel in bfloat16. That measures rounding, not correctness.
> Score both against float32, and require identical generated tokens.

## 47.4 wired_limit is mandatory

A plain `model(prompt, cache=cache)` call **stalls** on this machine. The
process enters uninterruptible disk wait, RSS collapses to a few hundred MB,
and CPU goes to nearly zero.

Cause: without a wired limit the OS pages the weights in and out.

Fix:

```python
from mlx_lm.generate import wired_limit
with wired_limit(model):
    ...
```

`stream_generate` already does this. Any direct forward pass must do it too.

## 47.5 Known inefficiency

`PagedKVCache.update_and_fetch` grows the last partial page with
`mx.concatenate`. That copies the page on every decode step.

Fix later by preallocating a full page and tracking the fill level.
Correctness first, then this.

## 47.6 Memory knob measured

Reducing the prefill chunk lowers the transient peak:

```text
prefill-step-size 1024   MLX peak 12.93 GB   swapouts 39124
prefill-step-size  256   MLX peak 12.10 GB   swapouts     0
```

0.83 GB of headroom recovered. That is about 13K more tokens of fp16 KV, or
44K at Q4. The saving comes from the prefill chunk size, not from
`mx.set_cache_limit`, which only added allocator churn.

## 47.7 Measurement warning: thermal drift

On the fanless M4 Air, decode speed at a **fixed** context fell across
consecutive runs:

```text
run 1   8.20 tok/s
run 2   7.20 tok/s
run 3   5.19 tok/s
```

Same context, same KV format. `prefill-step-size` cannot affect decode.

Therefore:

> Timing comparisons across separate runs are not valid on this machine.
> Interleave configurations inside one process, alternate them, and take
> medians. Deterministic values such as MLX peak memory remain reliable.

An fp16-versus-Q4 decode comparison made across runs is **not** trustworthy.
That question is still open.

## 47.8 Next

```text
S4  page centroids + top-k selection      <- the speed mechanism
S5  fused Metal paged attention kernel
S6  SSD page store and repository image
```

# 48. S4: top-k page selection model test

_2026-08-20._

## 48.1 Mechanism

Each page keeps elementwise key bounds per KV head:

```text
k_min[page]  [n_kv, D]
k_max[page]  [n_kv, D]
```

The upper bound of q.k for a page is:

```text
sum_d ( q_d > 0 ? q_d * max_d : q_d * min_d )
```

This never ranks a page below its true relevance, so it is admissible.

Computed as two matmuls, not a large elementwise tensor:

```text
upper = relu(q) @ k_max.T + (q - relu(q)) @ k_min.T
```

Selection runs at decode only, refreshed every N tokens. Pinned pages and a
recency window are always included.

## 48.2 Real-model result

```text
prompt 800 tokens of MacBat Swift source
page_size 128, 7 pages, top_k 4, refresh every 8 tokens

selection: 4/7 pages read (43% of KV skipped)
identical tokens: 12/12, no divergence
MLX peak unchanged at 12.09 GB
```

Skipping 43% of the attention KV produced identical output.

## 48.3 Bound behavior on noise and real attention

A synthetic test with isotropic random keys **fails**:

```text
needle gain 32   bound rank 39 of 40   selection misses the needle
needle gain 64   bound rank  0 of 40   rel error 8.5e-3, 85% of KV skipped
needle gain 128  bound rank  0 of 40   rel error 3.2e-6
```

Reason: for a page of noise the per-dimension independent maxima give a very
loose upper bound, which can beat the true score of a genuinely relevant page.

Consequence:

> Do not validate page selection on synthetic random keys. The bound only
> discriminates when the relevant key clears the noise-page bound, which real
> attention distributions do and isotropic noise does not.

Looseness grows with page size. Smaller pages give tighter bounds and better
selection, at the cost of more pages to score.

## 48.4 Corrected memory failures

Both were found the hard way. Do not reintroduce them.

### Paged attention during prefill

The per-page loop at L = 3000 with 24 pages builds 24 score tensors of

```text
[1, 4, 6, 3000, page_size] in float32
```

inside one lazy graph, per layer. Memory exhausted, machine nearly died.

Fix: prefill uses the fused kernel over the concatenated pages. The page loop
runs only at decode, where L = 1 makes the tensors trivial.

### Unchunked prefill

A single `model(prompt)` call with 1600 tokens keeps the activations of all
64 layers in one lazy graph:

```text
unchunked   prefill 180.8s   MLX peak 14.32 GB   machine swapped to death
chunked 256 prefill  27.8s   MLX peak 12.09 GB   stable
```

Fix: chunk the prefill and call `mx.eval` on the cache state between chunks.
`stream_generate` already does this. Any direct forward pass must do it too,
in addition to `wired_limit` from section 47.4.

## 48.5 Verification status

Proven:

```text
bounds -> selection -> exact merge -> identical tokens
memory stays flat while pages are skipped
```

Not yet proven:

```text
behavior at hundreds of pages, for example 4096 of 256K tokens
quality on needle retrieval and the MacBat benchmark
any speed benefit
```

Speed was not measurable here. The machine was memory constrained and
`mx.clear_cache()` between prefill chunks drops the MLX buffer pool, so the
decode rates in this test are not meaningful.

## 48.6 Next

```text
1. scale the page count: 8K-32K context, hundreds of pages, top_k about 32
2. quality gate: needle retrieval, multi-needle, MacBat benchmark
3. preallocate pages, remove the concatenate in update_and_fetch
4. only then chase speed, and measure it interleaved (section 47.7)
```

# 49. S4 scale test and sparse-path overhead

_2026-08-20, end of session._

## 49.1 8K result

```text
prompt 8000 tokens of MacBat Swift source
page_size 256 -> 32 pages, top_k 8, refresh every 8 tokens

contiguous  prefill 273.0s (29 tok/s)  peak 12.62 GB
sparse k=8  prefill 293.4s             peak 12.64 GB
selection: 8/32 pages read (75% of KV skipped)
identical tokens: 8/8, no divergence
```

Correctness holds at 75% skip on the real model, at 8K, with real source.

## 49.2 Sparse and contiguous comparison

Decode, after removing prefill from the timing:

```text
contiguous  7.3 tok/s   32 pages, one fused kernel call
sparse k=8  5.0 tok/s    8 pages, eight separate calls
```

Reading 75% less KV was still slower.

Cause: the per-page loop pays a kernel launch per page. At 32 pages that
overhead exceeds the bandwidth saved.

## 49.3 Gather-path correction

Section 46.8 assumed a fused Metal paged kernel was required. It is not, at
least not first.

Better approach for decode:

```text
1. select pages by the min/max bound
2. gather the selected pages into ONE contiguous K/V buffer
3. one mx.fast.scaled_dot_product_attention call
```

The gather copies only the selected pages, which is exactly the data that had
to be read anyway. One fused call replaces k launches.

Implemented as `PagedKVCache.gather_decode`, default on. The online-softmax
merge path stays for correctness testing and for the future case where the
gathered buffer must not be materialised.

Measurement of the gather path was interrupted. **Still unmeasured.**

## 49.4 Model-test timing correction

The decode timer started before prefill, so reported decode rates included
the whole prefill:

```text
reported 0.03 tok/s   ->  actual 7.3 tok/s at 8K
reported 0.36 tok/s   ->  actual 8.6 tok/s at 800
```

Decode was never slow in these tests. Any decode number recorded before this
fix is wrong.

## 49.5 Open items

```text
1. measure the gather path against contiguous at 8K and at 32K
2. find the page count where sparse beats contiguous
3. quality gate: needle retrieval, multi-needle, MacBat benchmark
4. preallocate pages, remove the concatenate in update_and_fetch
5. Q4 pages: the gather path can quantize the gathered buffer
```

# 50. S6: SSD page store. 256K in 0.81 GB

_2026-08-20._

## 50.1 Result

Component measurement, 1024 pages of 256 tokens, top_k 32, resident budget 48:

```text
logical context            262144 tokens
resident pages             48 of 1024
resident KV, 1 layer       50.3 MB   (full would be 1074 MB)
resident KV, 16 layers     0.81 GB   (full would be 17.18 GB)
decode attention           2.37 ms per layer
per token, 16 layers       38 ms
```

17.18 GB is the whole machine. 256K goes from impossible to 0.81 GB.

With about 125 ms per token of weight reads, 38 ms of attention projects to
roughly 6 tok/s at 256K.

Output is bit-identical to keeping every page resident:

```text
attention vs resident-only: rel = 0.000e+00
```

## 50.2 Page-store design

```text
cold pages          safetensors files on SSD, dropped from RAM
resident set        pinned + recent + selected, capped by resident_pages
min/max bounds      always in RAM, 33 MB for 256K, needed to score every page
restore             on demand when a page enters the selection
```

Bounds are computed before a page can be spilled and never recomputed, so a
spilled page is still scoreable without touching the disk.

## 50.3 Decode path

Three changes made the sparse path fast:

```text
1. preallocated pages
   update_and_fetch writes in place instead of mx.concatenate, which copied
   the whole last page on every decode step

2. cached stable gather
   the selected full pages are concatenated once per selection refresh, not
   once per token; the growing last page is merged separately

3. two-chunk merge
   decode merges stable gather + last page, not one chunk per page
```

Component benchmark, per attention layer:

```text
ctx    8192   32 pages  k=8    dense  3.78 ms   sparse+gather 1.02 ms
ctx   32768  128 pages  k=16   dense  8.43 ms   sparse+gather 1.02 ms
ctx  131072  512 pages  k=32   dense 33.10 ms   sparse+gather 1.79 ms
ctx  262144 1024 pages  k=32                    sparse+gather 2.37 ms
```

Dense grows with context. Sparse stays nearly flat. This is the section 46
claim, measured.

## 50.4 Write amplification

A restored page is unchanged, so its file stays valid. Rewriting it on the
next spill doubled SSD writes:

```text
before  2.05 GB written for one layer at 256K
after   1.02 GB, exactly one write per page
```

## 50.5 Remaining validation

```text
quality at 32-of-1024 selection      <- the real remaining risk
full model run at large context
needle retrieval, multi-needle, MacBat benchmark
```

The measurements verify the mechanism and memory use. Aggressive-selection
quality remains unverified and needs the next test.

# 51. Transient memory: measurement and one closed direction

_2026-08-20._

## 51.1 prefill-step-size 256 result

```text
step 1024   MLX peak 12.93 GB   transients 1.50 GB   prefill 46.3 tok/s
step  256   MLX peak 12.10 GB   transients 0.63 GB   prefill 47.2 tok/s
```

0.87 GB recovered at no throughput cost. Applied in `start_server.sh`.

Correction: an earlier note in this session claimed step 256 cost about 19%
of prefill throughput. That was thermal drift between runs, not the setting.
See section 47.7.

## 51.2 MLX buffer-pool measurement

```text
ctx 1056   pool 0.03 GB
ctx 3879   pool 0.29 GB
```

`mx.set_cache_limit` is not worth using. There is at most 0.3 GB to reclaim
and capping it low measurably slows decode by forcing reallocation.

Do not revisit this.

## 51.3 Remaining floor

```text
weights            10.74 GB
runtime overhead    0.55 GB
transients          0.63 GB
--------------------------------
resident            about 11.9 GB peak at short context
```

With quality fixed, this is the floor. Further reduction needs weight
changes, which the freeze rule forbids.

# 52. Revised 32-64K target

_2026-08-20._

## 52.1 System-memory constraint

Weights are 11.25 GB on a 17.18 GB machine. macOS plus normal apps take
3-4 GB. That leaves about 2 GB of slack **while the machine is in use**.
Every transient spike above that freezes the Mac.

256K is arithmetically reachable:

```text
weights      11.25 GB
KV paged      0.81 GB
transients    0.63 GB
-----------------------
             12.70 GB   leaves 4.5 GB for macOS and apps
```

But that assumes the Mac is doing nothing else, and 256K prefill still costs
about 2.4 hours at 30 tok/s. 256K is a dedicated-machine target, not a
workflow on a daily driver.

Decision: **target 32-64K.** Comfortable margin, usable while working, no
repository image required.

## 52.2 Settings for the 32-64K target

`frankenstein_engine.py --paged` defaults:

```text
page_size        256
top_k_pages      16
resident_pages   24
refresh_every    16
min_context      16384   (selection off below this)
pages on SSD     fp16, so attention uses the fused kernel
```

Resident KV at 64K:

```text
per page per layer   1.05 MB
24 pages x 16 layers 0.40 GB
```

Budget:

```text
weights      11.25 GB
KV resident   0.40 GB
transients    0.63 GB
-----------------------
             12.28 GB   leaves 4.9 GB for macOS and apps
```

Unpaged fp16 at 64K would need 4.2 GB of KV, giving 16.1 GB total. That does
not fit. Paging is what makes 64K possible at fp16 quality and fused-kernel
speed.

## 52.3 Needle retrieval: first quality evidence

Context 2523 tokens, 20 pages, secret planted at 50% depth:

```text
dense (control)   FOUND  74391
sparse k=4        FOUND  74391    4/20 pages read, 80% skipped
```

The sparse run also cited where it found the fact. Selection preserves recall
at 80% skip. k=2 was not completed.

## 52.4 Operational rule

Every freeze in this session came from the harness, not from paging:

```text
paged loop running during prefill        fixed
unchunked prefill, 64 layers of graph    fixed
missing mx.clear_cache in prefill loop   fixed
launching with 3.48 GB free              guarded
```

Both `paged_kv.py` and `frankenstein_engine.py` now abort unless 8 GB is
free. The model needs 11.25 GB. Starting below that guarantees swap, and swap
here does not mean slow, it means the Mac stops responding.

# 53. Work from 2026-08-21 to 2026-08-24

The next work period focused on ingestion speed and terminal reliability.

Completed work included:

- persistent workspace and terminal settings
- safe approval parsing and repeated-call controls
- small Tavily search results
- model-off repository token caching
- exact MLX context images
- dense prefill profiling
- sparse MLP and Neural Engine experiments
- direct in-process Whittle GGUF support

The measurements established a dense MLX prefill ceiling near 47 tokens per second.
Exact context restoration reached 190.8 times speed for unchanged prior content.
Sparse MLP and Neural Engine paths failed their full-model quality or scale gates.

See [the complete work log](../archive/qwen27b/session-2026-08-21-to-24.md).
See [the current operating state](../archive/qwen27b/current-state-2026-08-26.md).

---

# 53. f16 MoE streaming: measured and rejected (2026-08-24)

## 53.1 FFN streaming proposal

Run the full BF16 27B model. Keep the attention stack resident. Cut each dense
SwiGLU FFN into expert blocks. Stream only the active blocks from SSD.

The cut itself is exact. A SwiGLU FFN is a sum of independent rank-1 terms:

```text
y = sum_i  down[:, i] * ( silu(gate[i] @ x) * up[i] @ x )
```

Loss comes only from skipping blocks, never from the split.

## 53.2 Machine limits, measured

Cold SSD read, F_NOCACHE, 21.5 GB file larger than the page cache:

```text
chunk   threads   GB/s
256 KB        4   1.72
  1 MB        2   2.91   <- best
  4 MB        4   2.80
 16 MB        2   2.81
```

Sustained uncached write is 0.73 GB/s. Design number for reads is 2.8 GB/s.

RAM is 17.18 GB. `iogpu.wired_limit_mb` is already 15000.

## 53.3 Model shape, exact

```text
group            params  share   bf16 GB
full-attn         1.68B    6.2%     3.36
linear-attn       5.56B   20.7%    11.12
FFN                17.11B  63.6%    34.23
embed (lookup)    1.27B    4.7%     2.54
lm_head           1.27B    4.7%     2.54
TOTAL LM         26.89B  100.0%    53.79
```

The attention stack is read every token. At BF16 it needs 14.5 GB resident.
That alone does not fit. It also cannot be streamed.

## 53.4 Decisive measurement

`ffn_oracle_bound.py` masks every FFN to its top-k neurons by |activation|,
with perfect hindsight. No router, permutation, or neuron bundling can beat
this. 768 real code tokens, all 64 layers:

```text
keep    GB/tok   tok/s ceiling   perplexity   top-1 match
100%     34.23           0.08         61.79        100.0%
 50%     17.11           0.16         75.00         83.6%
 25%      8.56           0.33        211.55         63.3%
12.5%     4.28           0.65       2398.22         35.4%
6.25%     2.14           1.31      29455.76         13.8%
```

3 tok/s needs keep at or below 3%. The model breaks at 25%.

`ffn_sparsity_probe.py` explains why. Only 43% to 62% of neurons hold 99% of
the activation energy. Contiguous blocks average this out further: top-16 of
128 blocks repeat across consecutive tokens only 20% to 30% of the time,
against 12.5% for chance.

## 53.5 Conclusion

SwiGLU is not ReLU. Deja Vu, PowerInfer, and Apple's LLM-in-a-flash all rely on
ReLU FFNs, where most neurons are exactly zero. This model has no such
structure. Streaming a BF16 FFN as a MoE is closed. Do not retry it without
retraining the FFN for sparsity.

## 53.6 Feasible uses of the same arithmetic

The iron law is `tok/s * bytes per token <= bandwidth`.

RAM gives 120 GB/s but holds 13.5 GB. SSD gives 2.8 GB/s and holds 50 GB.
At 6 tok/s the SSD affords about 0.5 GB per token. Spend it where BF16 is free:

1. `embed_tokens` to SSD at BF16. It is a row lookup, 10 KB per token.
2. `lm_head` to SSD at BF16, behind a resident shortlist head. Read only the
   top 2048 candidate rows, 21 MB per token, about 7 ms.

Both become exact BF16 and give back about 1.4 GB of RAM.

Measured resident weights are 9.66 GB. KV grows 64 KB per token, plus a
constant 0.155 GB for the 48 linear-attention layers.

```text
ctx      KV      free now   body after moving embed+head   bits/param
8000    0.68 GB    2.36 GB                       12.02 GB         3.95
32000   2.25 GB    0.79 GB                       10.45 GB         3.43
```

The body runs at 2.71 bits/param today. The budget allows 3.95 at 8K context.
That gain is larger than anything the streaming plan could return.

---

# 54. Exact BF16 model endpoints (2026-08-25)

## 54.1 Result

The shipped 4-bit `lm_head` disagrees with the exact BF16 head on **6.2% of
tokens**. A shortlist fixes this completely.

Method: a cheap resident head ranks the vocabulary. The top `k` rows come off
the SSD at BF16 and get exact logits. Everything else is -inf, which every
sampler already handles.

128 real code tokens, ground truth = full BF16 head:

```text
selector    RAM      k   recall   top-1        mass  MB/tok
 4-bit    0.70G      -        -    93.8%           -      0   <- today
    2b    0.40G    256  100.00%  100.0%      93.59%    2.6
    2b    0.40G   1024  100.00%  100.0%      97.13%   10.5
    3b    0.56G   1024  100.00%  100.0%      97.41%   10.5
    4b    0.72G   1024  100.00%  100.0%      97.46%   10.5
```

Recall is 100% at every selector width. Even 2 bits ranks the vocabulary well
enough to never miss the true argmax. So take the cheapest selector.

`mass` is the exact probability the shortlist covers. The rest is tail, which
top-p sampling discards anyway.

## 54.2 Decision

```text
embed_tokens   dropped from the model, BF16 row lookup on SSD, 10 KB/token
lm_head        2-bit resident selector, k=1024, exact BF16 rows on SSD
```

Frees 1.00 GB of RAM. Both ends become bit-exact. Cost is about 10 MB/token,
under 4 ms, and the page cache absorbs most of it.

Do not raise the selector width. It buys nothing.

---

# 55. E2 and stock Qwen incompatibility (2026-08-25)

## 55.1 Experiment

The plan was to predict the gain from a higher bit budget cheaply: take layers
already converted from the stock BF16 source, drop them into the shipped model
at higher precision, and read off the loss change.

It failed, and the failure is informative.

```text
                                          NLL      ppl
shipped V3.1 Compact                   1.7837    5.951
+ layer 0 from stock, 8-bit            1.7855    5.962   fine
+ 14 layers from stock, 8-bit          5.7493  313.976   destroyed
restored                               1.7837    5.951
```

Per layer, at 8 bits, which is near lossless:

```text
layer   delta NLL
    0      0.0017
    1      0.8622
    2      0.8892
    3      0.9505
    4      0.3303
    5      1.9566
    6      5.3010
    7      0.6382
   10      0.3714
   11      0.8286
   12      1.0887
   13      0.4176
   14      0.2844
```

Every layer except 0 rejects its own stock weights. The weights correlate with
stock at 0.91 to 0.97 but differ by 43% to 54%.

## 55.2 Cause

E2 carries DWQ-trained weights. See the `dwq-multisample`, `dwq-prefix63-v2`,
and `dwq42-teacher` work in sections above. DWQ optimises the quantised values
so the whole network matches a teacher. The layers are co-adapted. A stock
layer is a different layer, not a cleaner copy of the same one.

## 55.3 Consequence

Two rules follow.

1. Never mix stock tensors into E2. Not for measurement, not for repair.
2. E2 cannot be re-quantised at a higher bit budget. It exists only at 2, 3,
   and 4 bits, and dequantising does not recover what DWQ trained away.

A higher bit budget therefore needs a **new build from the stock BF16 source**,
compared against E2 head to head. That comparison is the open question:
a plain 4-bit build against a DWQ-trained 2.7-bit build.

## 55.4 Measured bit-plan difference from v3

First allocator output, activation-weighted, 3.81 bits/param:

```text
tensor          v3 rule   measured
in_proj_qkv         3b        5.8
in_proj_z           3b        6.1
q_proj              3b        6.0
k_proj              3b        8.0
v_proj              4b        8.0
gate_proj           3b        3.1
up_proj             2b        3.0
down_proj           3b        2.0
o_proj              3b        2.0
out_proj            3b        2.0
```

Attention and recurrent **inputs** are much more fragile than the hand rules
assumed. **Output** projections are much tougher. Total distortion is 69x below
a flat-minimum assignment.

---

# 56. Low-rank selector head: rejected (2026-08-25)

The resident head only ranks the vocabulary; exact logits come from BF16 rows
on SSD. A 2-bit head already gives 100% recall, so ranking looked like an easy
job worth compressing further. A rank-256 factorisation costs 0.13 GB against
0.40 GB for the 2-bit copy.

It does not work. Ground truth is the exact BF16 head, on real hidden states:

```text
selector      RAM      k   recall     mass
rank-256    0.13G    256   48.13%   44.49%
rank-256    0.13G   1024   73.12%   67.15%
rank-256    0.13G   2048   82.50%   76.22%
2-bit head  0.40G    256  100.00%   95.72%
2-bit head  0.40G   1024  100.00%   98.07%
2-bit head  0.40G   2048  100.00%   98.83%
```

At 73% recall roughly a quarter of tokens take the wrong argmax, silently. The
head maps 5120 dimensions onto 248320 tokens, so it is close to full rank and
has no low-dimensional subspace to exploit. Rank 1024 would restore recall at
0.52 GB, which is worse than the 2-bit head it replaces.

Quantisation beats factorisation for this matrix. Keep the 2-bit head.
`lowrank_head.py` stays for the streamed randomised SVD, which is reusable.

---

# 57. Bit allocator: measurement, plan, and build (2026-08-25)

Section 55.4 shows the hand-picked V3 bit plan disagrees with measurement.
This section replaces hand-picking with a solver.

## 57.1 Three stages

`bit_allocator.py calibrate` runs a code corpus through the model. It records
the activation RMS of every input channel for 497 linear tensors.

`quantize_v4.py score` quantizes each tensor at each of 18 options. The options
are bits 2, 3, 4, 5, 6, 8 crossed with group 32, 64, 128. It weights the
reconstruction error by the activation RMS of the matching input channel. The
score is therefore the expected error at the output, not the error in the
weights.

`quantize_v4.py plan` runs a greedy knapsack. It sorts every candidate upgrade
by error reduction per byte. It stops at the byte budget.

`quantize_v4.py build` writes one safetensors file per layer.

The cost model is exact, including the scale and bias overhead:

```python
def nbytes(out, inp, bits, group):
    return out * inp * bits / 8 + (out * inp / group) * 4
```

## 57.2 MLX calibration failure

The calibration must evaluate the output and every accumulator in ONE call:

```python
mx.eval(out, *accumulators)   # correct
```

Two separate `mx.eval` calls make MLX free the intermediates. It then recomputes
the whole forward pass for the second call. The first run took hours instead of
minutes and looked like a hardware problem.

## 57.3 The group-128 bug

The first allocator offered group 128 only. Group 64 and group 32 cost 4 more
bytes per group. On some tensors they cut error more than a whole extra bit
does. Adding the smaller groups gave about 7 percent more quality for the same
bytes. Never restrict the group size.

## 57.4 Allocator output

V4-flat, 402 planned tensors, 13.05 GB on disk, 3.88 bits per parameter:

```text
family          bit distribution
down_proj       3:61  4:3
gate_proj       3:36  4:28
up_proj         3:35  4:29
in_proj_qkv     3:3   4:28  5:17
in_proj_z       3:1   4:19  5:28
q_proj          4:14  5:2
o_proj          5:16
out_proj        5:48
k_proj          5:1   6:15
v_proj          5:4   6:7   8:5
embed_tokens    2:1
lm_head         2:1
```

Three readings:

- `down_proj` gets the fewest bits. It has 17408 input channels, so it is the
  cheapest tensor to cut.
- `v_proj` and `k_proj` get the most. They are small and they steer retrieval.
- The allocator never spends 8 bits inside an FFN.

## 57.5 Bit floors and validation status

`--floor FAMILY:BITS` and `--floor-layers LO-HI:BITS` override the knapsack.
They exist to test opinions. The builds:

```text
big     3.32 bits   pure knapsack
max     3.68 bits   pure knapsack
ultra   4.01 bits   pure knapsack
attn    o_proj and out_proj floored to 5 bits
out     every output path floored: o_proj, out_proj, down_proj
flat    outputs floored AND layers 0-15 floored
```

The floors change real decisions. V4-out pays for 6-bit and 8-bit attention by
dropping 19 `gate_proj` and 19 `up_proj` tensors to 2 bits. V4-flat refuses to
go below 3 bits in the FFN and buys less attention.

WARNING: no A/B test compares a floored build against a pure-knapsack build of
the same size. The floors are untested opinions. Test them before trusting them.

## 57.6 Cost of one cycle

Score takes about 50 minutes. Plan takes seconds. Build takes about 12 minutes.
A full new bit plan costs under 90 minutes.

---

# 58. Measured resident-memory ceiling (2026-08-26)

12.65 GB of weights loads and runs. 13.05 GB loads and then swaps.

Evidence. V5 at 13.05 GB reached 1.0 tokens per second with 1.81 GB of swap.
V4 at 12.65 GB runs at 4 to 6 tokens per second with no swap. The machine has
17.18 GB of RAM and `iogpu.wired_limit_mb` is 15000.

Rule: the resident weight budget on this 16 GB M4 is 12.6 GB. Do not build past
it. A model that swaps is slower than a smaller model that does not.

Two supporting results:

- `mx.set_wired_limit` must run BEFORE `load()`. It ran after `load()` in every
  earlier version, so the `WIRED=` setting had never taken effect on any model.
- Lazy mmap weights remove the load-time memory spike. They also cause
  `kIOGPUCommandBufferCallbackErrorTimeout` in the middle of an answer. A clean
  out-of-memory failure at load is better than a GPU watchdog crash. Reverted,
  with a comment that says why.

## 58.1 Moving the two ends to SSD

`bf16_ends.py` provides `load_v4_lean`. It copies `mlx_lm.load_model`, drops
`embed_tokens` from both the quantization plan and the weight list, and installs
`SSDEmbedding` BEFORE `load_weights`. Installing it after still pays the peak.
The saving is 0.40 GB of peak memory.

---

# 59. MLX_QMM_BK measurement corrections

A claimed 21 percent prefill speedup from `MLX_QMM_BK=64` was wrong three ways:

1. The benchmark ran in `mlx-qwen38-apple` (0.32.1), not in
   `mlx-qwen38-kernel-lab` (0.32.1.dev) where the kernel change lives.
2. It ran at batch 1, which is decode. Prefill is batch 256.
3. It omitted group 32, which 45 tensors in the current plan use.

`BK=64` fails `static_assert BK <= group_size` and breaks the model completely.
The real gain is about 5 percent, and it is mostly a cold-start artifact.

`MLX_QMM_BK` stays at 32. `chat.sh` carries a comment that says never raise it.

---

# 60. DWQ is blocked on this machine

Distilled weight quantization trains the scales and biases only. This model has
773M trainable scales. Adam keeps three copies of optimizer state. That needs
19.8 GB against 14 GB available.

Chunking the training by layer does not help. Tuning an early layer still needs
backpropagation through every later layer, so the full graph stays resident.

DWQ needs a larger machine or an optimizer with less state. The V3.1-Compact
weights were DWQ-trained, which is why section 55 forbids grafting stock tensors
into them.

Every DWQ artifact was deleted on 2026-08-26: the teacher logit caches
(`dwq42-teacher`, `dwq16-transition`, `dwq-python-control`) and the trained
outputs (`dwq-prefix63-v2`, `dwq-multisample`,
`gguf_l63_head_joint_128.safetensors`). Nothing from this line of work remains
on disk. Restarting it means starting from the teacher pass.

---

# 61. Code-generation support

Four parts, all outside the weights. None of them changes a single parameter.

## 61.1 context7.py, documentation before generation

`prefetch()` reads the request, matches it against `TRIGGERS`, and loads the
real API documentation into the system context before generation starts. The
model does not have to remember an API it never learned.

`FACTS` holds hand-verified rules that the vendor documentation states badly.
The SketchUp entry:

```text
- The internal unit is INCHES, not millimetres and not metres.
- Do NOT hand-roll unit conversion. Use String#to_l.
- Numeric#mm, #cm, #m, #feet convert the other way.
- String#to_f does NOT return NaN on bad input. It returns 0.0.
- vertex.position is in the LOCAL coordinates of its container.
```

## 61.2 api_guard.py, API call validation

`check_ruby` checks methods, namespaced constants, bare constants, and legacy
globals. `check_arity` and `check_arg_types` check call shape.
`vendor_param_types` returns every overload form, richest first.

One false positive on `MultiExtrude.run` made the model rewrite code that already
worked. The fix has three parts: a workspace-wide `def` scan, a skip for the
file's own constants, and a 0.72 similarity cutoff before suggesting a
correction. A guard that fires wrongly is worse than no guard.

## 61.3 code_check.py, compilation checks

```text
.swift        swiftc -typecheck
.rb           ruby -c, then api_guard
.py           compile(), then pyflakes, then stdlib attribute check
.js           node --check
.sh           bash -n
.json .yaml   parsers
```

## 61.4 free_search.py, search routing

The DuckDuckGo HTML scrape returned HTTP 202 anti-bot pages for every query, so
`web_search` had been silently returning nothing. The replacement routes by a
`CODE_HINT` regex across five backends: vendor docs, the `ddgs` package,
StackExchange, Wikipedia, and the DuckDuckGo instant-answer API. Tavily runs
first when `TAVILY_API_KEY` exists, and falls back per call.

## 61.5 Parser fixes in the engine

The model emits several closing-tag styles. The parser now accepts all of them:

```python
FUNCTION_RE = re.compile(r"\s*<function=([^>\s]+)>(.*?)(?:</function>|</>)?\s*$", re.S)
STRAY_TAG_RE = re.compile(r"</?(?:tool_call|function)\s*>")
```

`detect_loop` now returns a reason string, `loop` or `loop:intent`.
`write_file` creates parent directories.

---

# 62. Measured model behavior across eight runs

One fixed task: write a SketchUp extension. Each line is one full run.

```text
1  invents add_glue and delete_me
2  verifies pushpull, then over-engineers the solution
3  invents $sketchup and Sketchup::Length
4  correct method, wrong UI.inputbox arity
5  correct arity, wrong argument order
6  correct API, wrong unit constant (1/39.37)
7  correct API, claims mm is the internal unit (it is inches)
8  a working extension that loads and runs in SketchUp, 25.4x too tall
```

The tooling removed the invented-API failure class. Runs 6 to 8 fail on unit
semantics only. No checker catches that failure, because the code is valid Ruby
and every call it makes exists.

The overnight run of 2026-08-26 produced no file at all. Two causes, both
visible in the transcript. V5 was swapping at 1.0 tokens per second. The model
looped on `api_docs('extrude')` while asserting that `Face#extrude` exists. It
does not exist. The method is `pushpull`, which the same model verified
correctly in run 2.

---

# 63. Web terminal on the LAN (2026-08-26)

`web_terminal.py` forks the chat under a PTY, mirrors the bytes to the browser
over SSE, and converts ANSI to HTML. The phone gets the same interface as the
terminal, including approvals.

Access needs a token in the URL. Anything on the network that reaches this port
can run shell commands on the Mac, so an open port is not an option.

One bug took several attempts. The PTY converts `\n` to `\r\n`. A naive
carriage-return handler clears the completed line, so every banner
line disappeared and the page rendered 57 blank lines. The fix holds a trailing
`\r` across chunk boundaries, then collapses `\r\n` to `\n` before the
per-character pass:

```python
_pending_cr = [False]
def feed(data: bytes):
    text = data.decode("utf-8", "replace")
    if _pending_cr[0]:
        text = "\r" + text; _pending_cr[0] = False
    if text.endswith("\r"):
        _pending_cr[0] = True; text = text[:-1]
    text = text.replace("\r\n", "\n")
```

A child process that dies now prints a red message into the buffer. Before this,
V5 failed to load and the page stopped without an explanation.

---

# 64. Open items

1. No A/B test proves that any bit floor helps. Build a pure-knapsack model and
   a floored model at the same byte size, then score both. Until then, treat
   `attn`, `out`, and `flat` as untested.
2. Sensitivity calibration is the principled fix for the allocator proxy. It
   replaces activation RMS with a measured output-loss gradient. Two attempts
   ended in the noise floor. It needs about 40 minutes on a quiet machine.
3. MTP speculative decoding. The head weights are already on disk. Estimated
   1.5x generation speed, 4 to 6 hours of work, about 0.3 GB of RAM.
4. Add `ruby -w` to the guard. It catches undefined-variable typos.
5. Add receiver-aware method checking. It catches `Face#center`.
6. `chat.sh` defaults point at deleted models. See the archived
   [`current-state-2026-08-26.md`](../archive/qwen27b/current-state-2026-08-26.md).

---

# 65. Repository strip (2026-08-26)

Dead code is not free. It gets imported, it gets read, and it gets believed.
Two examples in this repository were actively harmful.

## 65.1 Stale backend import

`frankenstein_chat.py` imported `LlamaCppEngine` at module level. The GGUF it
served was deleted days earlier. The import still succeeded, so nothing
reported the problem, and the backend stayed in the help text, the status
panel, and six conditionals.

## 65.2 Invalid sparse flags

`--sparse-prefill` and `--sparse-generation` imported `v32_prefill_sparse_probe`.
That module is not in this repository and has not been for some time. Anyone who
passed the flag got a traceback. Sparse MLP had already failed its quality gate
at 11.7 percent RMSE, so the flags advertised a rejected result as an option.

## 65.3 Removed

```text
llama_cpp_engine.py     the Whittle GGUF it loaded is deleted
chat_whittle.sh         same
f16_moe_plan.py         planning for the direction closed in section 53
moe_bit_plan.py         same
docs/V4-F16-MOE.md      same
dwq_corpus.py           DWQ is blocked, section 60
overnight.sh            orchestrates builds of V3.1, V4-full and V4-match, all gone
watch-model-download.zsh  the download finished
eval__..._arc_easy      eval output for a deleted model
__pycache__/ plans/ calibration/ .DS_Store
```

Also removed from code: the `--backend`, `--llama-*` and `--sparse-*` arguments,
and every conditional that served them.

A copy of everything deleted is at
`~/.frankenstein/attic/stripped-2026-08-26.tar.gz`.

## 65.4 Held-out corpus files

`bench_decode.py` and `profile_prefill.py` are imported by nothing. They are
held-out corpus text for `eval_models.py` and `bits_vs_quality.py`.

Perplexity numbers compare across builds only while the corpus is byte
identical. Deleting either file silently changes the corpus, and every later
number drifts against every earlier one with no visible cause. Both `HELD_OUT`
declarations now carry that warning.

The strip deleted `bench_decode.py`. A reference search detected the error.

## 65.5 chat.sh no longer carries a model list

The switch table named nine builds. Seven of them had been deleted, so seven
switches failed at load while still appearing in the help text.

The launcher now reads the model directory at every run. A build that is
deleted stops being offered. There is still no default: it lists what exists and
exits. Reproducible benchmarks require an available model.

The hardcoded model list went stale twice within one week.
Dynamic discovery prevents the same failure.
