# FlashNext / MLX Metal Backend Findings

## Executive summary

This report reviews the MLX 0.32.2 Metal backend against the FlashNext decode loop used by **MACQWEN**, a 176B sparse MoE running on an M4 Mac with 16 GB unified memory and streaming expert weights from SSD.

The central observation is:

> **The MLX source does not reveal a single mechanism that cleanly explains the ~127 ms/token of additional GPU-span time seen under SSD load.**

The strongest source-level candidates are **GPU memory barriers, command-buffer synchronization, and Metal/driver work around foreign host buffers**. However, the unexplained non-monotonic GPU-busy curve may also be partly or entirely a **measurement artifact caused by how overlapping Metal command-buffer intervals are unioned**.

The highest-value next step is therefore **not another benchmark**. Re-analyze the existing traces using the **sum of command-buffer durations and overlap count**, rather than only the union of intervals.

---

## Workload

Per decode token, FlashNext runs 48 layers. Each layer roughly does:

1. GPU computes router scores.
2. `mx.eval(scores)` blocks the host.
3. Host selects ~8 of 512 experts.
4. 16 worker threads `pread` expert rows from SSD into fresh NumPy buffers.
5. Buffers are imported through `mx.from_dlpack(..., copy=False)`.
6. Three `mx.gather_qmm` operations consume the imported weights.
7. `mx.eval(flat)` blocks again.

Approximate totals per token:

- **98 `mx.eval` calls**
- **~201–250+ Metal command buffers**
- **~3,027 `mx.*` calls**
- **~432 fresh host allocations**
- **~1.18 GB transient host memory**
- Production: **~390 MB/token physically read from SSD**
- Decode speed: **~2.8 tok/s**

### Measured anomaly

| SSD miss fraction | Physical MB/token | ms/token | GPU busy/token |
|---:|---:|---:|---:|
| 0 | 0.2 | 217 | 69.5 ms |
| 0.125 | 152 | 278 | 93.8 ms |
| 0.25 | 300 | 440 | 171.7 ms |
| 0.5 | 603 | 596 | 170.7 ms |
| 1.0 | 1230 | 860 | 125.7 ms |

Token time grows roughly with bytes read, but **GPU busy is non-monotonic**: it peaks around 25–50% misses and falls again at 100%.

Known kernels account for only ~55 ms/token, leaving roughly **127 ms of GPU interval time** under production load without a corresponding named kernel.

---

# Findings, ranked by explanatory power

## 1. MLX inserts full buffer-scope memory barriers between dependent dispatches

**Status: proven by source; performance impact inferred.**

MLX creates Metal buffers with:

```text
MTL::ResourceHazardTrackingModeUntracked
```

so MLX performs hazard tracking itself.

When a dispatch reads data written by a previous dispatch, or writes data previously read, MLX sets `needs_barrier_`. Before the next dispatch it emits:

```text
memoryBarrier(MTL::BarrierScopeBuffers)
```

Relevant source:

- `mlx/backend/metal/allocator.cpp:15-16`
- `mlx/backend/metal/device.cpp:354-355`
- `mlx/backend/metal/device.cpp:377-378`
- `mlx/backend/metal/device.cpp:395`
- `mlx/backend/metal/device.cpp:411`
- `mlx/backend/metal/device.cpp:419`

The compute encoder uses `MTL::DispatchTypeConcurrent` (`device.cpp:580`), making these barriers necessary to serialize dependent work.

### Why it matters

FlashNext decode is effectively a long dependent chain. A large fraction of dispatches may therefore be separated by full buffer barriers.

A barrier can contribute GPU time without appearing as a named compute kernel. Under simultaneous NVMe DMA / unified-memory traffic, post-barrier memory access may become more expensive.

This is the strongest structural source-level candidate for the unexplained GPU span.

### Confirmation experiment

Use a local MLX build with:

```text
-DMLX_METAL_DEBUG=ON
```

and compare:

- the normal 48-layer dependent chain;
- a synthetic chain with the same number of dispatches but independent buffers.

If GPU time collapses in the independent version, barrier cost is significant.

---

## 2. `metal-gpu-intervals` command-buffer spans contain more than kernels

**Status: proven by source.**

A Metal command buffer may contain:

- compute dispatches;
- `memoryBarrier(...)`;
- `waitForFence(...)`;
- `updateFence(...)`;
- `encodeSignalEvent(...)`.

Relevant source:

- `device.cpp:395`
- `device.cpp:408-425`
- `device.cpp:456-470`
- `device.cpp:501`

Therefore:

> **GPU interval duration − named kernel duration is not equivalent to “unknown kernel work.”**

The difference can include barriers, GPU-side fence waits, synchronization, event signaling, and driver-managed resource work.

Completion handlers also exist, but execute after the command buffer completes and are host-side:

- `device.cpp:471`
- `device.cpp:520`
- `mlx/backend/metal/eval.cpp:66-67`

### Best diagnostic

Capture a `.gputrace` with:

```python
mx.metal.start_capture(path)
...
mx.metal.stop_capture()
```

and inspect per-dispatch timing in Xcode GPU tools.

Source:

- `python/src/metal.cpp:77-93`
- `mlx/backend/metal/metal.cpp:20-53`

---

## 3. Every `mx.eval` creates a shared event, breaks the encoder, commits, and waits

**Status: proven by source.**

Each `eval_impl` creates a fresh Metal shared event.

At the end of evaluation:

1. the event is signaled;
2. signaling ends the active encoder;
3. `gpu::finalize()` ends encoding and commits;
4. the host waits on the event.

Relevant source:

- `mlx/transforms.cpp:104`
- `mlx/transforms.cpp:340-343`
- `mlx/backend/metal/event.cpp:16`
- `mlx/backend/metal/event.cpp:28-30`
- `mlx/backend/metal/event.cpp:69-70`
- `mlx/backend/metal/device.cpp:500-501`
- `mlx/backend/metal/eval.cpp:71-77`
- `mlx/array.cpp:145-153`

This directly explains stack samples containing:

```text
IOSurfaceSharedEvent waitUntilSignaledValue
```

With ~98 evals/token, FlashNext pays roughly:

- 98 shared events;
- ≥98 fresh fences;
- ≥98 encoder breaks;
- ≥98 commits;
- 98 host waits.

### Potential optimization

The router-side `mx.eval(scores)` is a genuine CPU dependency.

The trailing `mx.eval(flat)` may not need a host wait. `mx.async_eval(flat)` performs evaluation and commit without the trailing host block.

Source:

- `mlx/transforms.cpp:350-362`

### Confirmation experiment

Replace only:

```python
mx.eval(flat)
```

with:

```python
mx.async_eval(flat)
```

while keeping `mx.eval(scores)` unchanged.

This is different from merging both graph boundaries into one eval.

---

## 4. Residency with `wired_limit == 0` is effectively inactive at commit time

**Status: proven by source.**

The wired-memory limit defaults to:

```text
0
```

Source:

- `mlx/backend/metal/allocator.h:71`

Allocations are still inserted into residency bookkeeping, but with zero capacity the insertion returns before assigning them to a residency set.

Relevant source:

- `allocator.cpp:161-163`
- `allocator.cpp:213`
- `resident.cpp:136-155`

`attach_new_sets()` is called before commits, but when no new sets exist it returns after essentially one atomic/acquire check:

- `device.cpp:519`
- `resident.cpp:222-226`

### Conclusion

With `wired_limit == 0`:

> **MLX residency bookkeeping is not a credible explanation for ~127 ms/token of additional GPU time.**

There is allocation/free bookkeeping, but no per-commit work proportional to the hundreds of imported buffers.

### Caveat for `mx.set_wired_limit(...)`

A non-zero wired limit changes behavior substantially. Buffers that fit may trigger `MTLResidencySet::commit()` during insertion and removal.

Source:

- `allocator.cpp:100-105`
- `resident.cpp:153-154`
- `resident.cpp:167-171`
- `resident.cpp:189-200`

So enabling residency is not guaranteed to be free, especially with hundreds of foreign buffers per token.

---

## 5. Command-buffer splitting has two explicit caps plus a load-sensitive throttle

**Status: proven by source.**

`needs_commit()` triggers on either:

```text
buffer_ops_ > max_ops
```

or:

```text
(buffer_sizes_ >> 20) > max_mb
```

Source:

- `device.cpp:511-514`

Defaults depend on architecture and are overridable with:

- `MLX_MAX_OPS_PER_BUFFER`
- `MLX_MAX_MB_PER_BUFFER`

Source:

- `mlx/utils.h:182-192`
- `device.cpp:602-624`

Typical architecture defaults are around **40 ops / 40 MB**.

### Important bug / accounting issue

`buffer_sizes_` accumulates `array.data_size()`, which is in **elements**, not bytes.

Source:

- `mlx/array.h:346`
- `device.cpp:349-351`

The code later interprets it like bytes by shifting by 20.

For packed Q4 weights stored as `uint32`, the nominal 40 MB cap behaves closer to an effective **~160 MB byte threshold**.

Therefore the ops cap is likely the main normal split trigger for this workload.

### Load-sensitive third path

MLX limits active GPU tasks:

```text
MAX_ACTIVE_TASKS = 10
```

Source:

- `mlx/transforms.cpp:25`

When too many buffers are in flight, MLX can:

- finalize open streams;
- create extra commits;
- block on a condition variable.

Source:

- `mlx/transforms.cpp:271-280`
- `mlx/scheduler.h:56-64`

This can make buffer counts change when the GPU falls behind.

It is a plausible explanation for some **203 vs. 250 vs. 269 command-buffer** variation, though the controlled miss sweep reportedly held buffer count nearly constant.

---

## 6. `from_dlpack(copy=False)` uses `newBufferWithBytesNoCopy`

**Status: proven by source; driver cost remains unknown.**

CPU NumPy arrays imported without copy ultimately become Metal buffers via:

```text
newBufferWithBytesNoCopy
```

Source:

- `python/src/convert.cpp:190-224`
- `python/src/convert.cpp:285-293`
- `mlx/backend/metal/allocator.cpp:207-218`

The buffer uses:

```text
StorageModeShared | HazardTrackingModeUntracked
```

### Important properties

- MLX performs no explicit alignment fix-up.
- If Metal rejects the pointer, `copy=False` fails rather than silently copying.
- Imported buffers are tracked by the allocator.
- With `wired_limit == 0`, they are not placed into active residency sets.
- They are handled like normal input buffers at command encoding.
- They are **not recycled through MLX's buffer cache**.

Relevant source:

- `allocator.cpp:208-232`
- `device.cpp:345-358`
- `convert.cpp:203-221`

Thus FlashNext pays hundreds of fresh Metal host-memory wraps and releases every token.

### What remains unknown

MLX itself performs only constant bookkeeping per wrap.

Any size-dependent cost would live inside Metal / the driver while mapping the host range into the GPU address space.

That is one of the few plausible places where cost could scale with imported byte volume without appearing in MLX source.

### Confirmation experiment

Measure `mx.from_dlpack(..., copy=False)` alone across buffer sizes such as:

- 100 KB
- 800 KB
- 3 MB
- 24 MB

If call latency is flat, the wrap itself is not byte-scaled. If it grows with size, the hidden term is likely driver-side mapping work.

---

## 7. Imported buffers can be released on Metal completion threads

**Status: proven by source.**

`gpu::eval` retains input `Data` objects until a command buffer completes:

- `mlx/backend/metal/eval.cpp:47-67`

The DLPack wrapper carries a custom deleter that eventually calls `MetalAllocator::release`.

Therefore large NumPy allocations may be released from a Metal completion-handler thread rather than the main thread.

For this workload that means roughly **1.18 GB/token of transient host blocks** can be destroyed asynchronously.

This is real CPU/VM work, although it does not by itself explain GPU interval duration.

---

## 8. The MLX buffer cache is narrow but does not appear to be thrashing

**Status: proven by source.**

The cache reuses the smallest allocation of sufficient size, but rejects buffers that are too oversized:

```text
cached_size < min(2 * requested_size, requested_size + 2 * page_size)
```

Source:

- `mlx/backend/common/buffer_cache.h:30-46`

For large allocations, reuse therefore requires sizes within approximately **32 KB** on Apple Silicon.

Eviction is driven by memory/resource thresholds:

- `buffer_cache.h:58-85`
- `allocator.cpp:132-138`
- `allocator.cpp:169-173`

Given the observed workload, those limits do not appear to be reached.

The measured ~45.7 MB cache plateau is therefore more consistent with a small steady-state pool of reusable intermediates than with eviction thrashing.

---

# The non-monotonic GPU-busy curve

## Source conclusion: MLX does not explain it

No inspected MLX code path naturally:

1. grows as SSD miss fraction rises from 0 → 25–50%, **and then**
2. becomes cheaper again at 100% miss,

while graph shape, dispatch count, and allocation behavior remain fixed.

The allocator, residency system, cache, command-buffer splitting, and hazard tracking are all either monotonic, flat, or inactive under these conditions.

### Strong alternative hypothesis: interval-union artifact

The current metric is the **union of `metal-gpu-intervals` spans**.

If command-buffer spans overlap differently depending on host submission timing, union duration can change even when total GPU work behaves differently.

At 100% SSD misses, the host spends longer blocked in `pread`, potentially submitting GPU work in tighter bursts. At intermediate miss rates, submissions may be more spread out.

That can alter the union of intervals without requiring a new MLX execution path.

This cannot be proven from the MLX source because Instruments defines the interval semantics.

## Highest-value next experiment

Re-analyze the existing Metal traces and compute, for every miss-fraction arm:

1. **sum of all command-buffer durations**
2. **union of command-buffer durations**
3. **number/fraction of overlapping command-buffer pairs**
4. **mean command-buffer span**
5. **command-buffer count**

Interpretation:

- **If sum is monotonic but union has the hump:** the anomaly is largely a span-overlap measurement artifact.
- **If sum also has the hump:** the extra cost is real GPU-side execution/synchronization and should be investigated with `.gputrace`.

This is the cheapest and most discriminating next step.

---

# Relevant knobs and APIs

| Knob / API | Default | Relevance |
|---|---:|---|
| `MLX_MAX_OPS_PER_BUFFER` | architecture-dependent, often 40 | Primary command-buffer split threshold |
| `MLX_MAX_MB_PER_BUFFER` | architecture-dependent, often 40 | Secondary split threshold; accounting appears element-based rather than byte-based |
| `MLX_BFS_MAX_WIDTH` | 20 | Changes tape construction / primitive grouping |
| `MLX_METAL_FAST_SYNCH` | 0 | Alternative cross-stream fence implementation; not `mx.eval` synchronization |
| `MLX_RESIDENCY_DEBUG` | 0 | Diagnostics for residency-set creation |
| `MLX_SDPA_BLOCKS` | 0 / auto | Attention tuning; low expected value here |
| `MLX_DISABLE_COMPILE` | unset | Low relevance; compilation is already small |
| `MLX_METAL_GPU_ARCH` | empty / auto | Overrides architecture and therefore several backend choices; risky |
| `MLX_METAL_DEBUG` | build-time OFF | Adds primitive names to command-buffer labels; very useful diagnostically |
| `mx.metal.start_capture()` | n/a | Generates `.gputrace`; highest-value detailed profiler |
| `mx.async_eval(x)` | n/a | Keeps commit/event but removes host wait |
| `mx.synchronize()` | n/a | Waits on command-buffer completion without creating a new eval shared event |
| `mx.set_memory_limit()` | platform-derived | Can affect GC / throttling; currently not binding |
| `mx.set_wired_limit()` | 0 | Enables residency sets; potentially useful but introduces residency-set commit work |
| `mx.device_info()` | n/a | Confirms architecture and recommended working-set assumptions |

---

# What the source rules out

The MLX 0.32.2 source does **not** support these explanations for the missing ~127 ms/token:

### Residency bookkeeping at `wired_limit == 0`
There is effectively no per-commit residency-set work beyond a trivial check.

### Buffer-cache eviction thrashing
The observed workload does not reach the allocator thresholds that trigger eviction.

### A byte-scaled cost inside MLX's DLPack wrapper
The MLX-side wrapper performs constant bookkeeping. Any byte-proportional cost would have to live inside Metal/driver mapping.

### A native MLX mechanism matching the non-monotonic miss curve
No inspected path has the required rise-then-fall behavior.

---

# Recommended investigation order

1. **Re-analyze existing traces using sum vs. union of command-buffer intervals.**
2. **Capture production and zero-drive runs with `.gputrace`.**
3. **Enable `MLX_METAL_DEBUG` to label command buffers by primitive.**
4. **Test `mx.async_eval(flat)` while retaining `mx.eval(scores)`.**
5. **Isolate `from_dlpack(copy=False)` cost vs. buffer size.**
6. **Measure the impact of dependent vs. independent dispatch chains to price barriers.**
7. Only then revisit residency / wired-limit tuning.

---

# Bottom line

The source narrows the problem considerably:

- MLX performs **aggressive explicit GPU synchronization** through barriers, fences, events, and frequent command-buffer commits.
- `mx.eval` is expensive structurally and forces synchronization boundaries.
- Foreign NumPy buffers are wrapped directly into Metal with `newBufferWithBytesNoCopy`, but any byte-scaled mapping cost is hidden in the driver.
- Residency bookkeeping with a zero wired limit is not the missing term.
- The buffer cache is not obviously thrashing.
- Most importantly, **nothing in MLX explains the non-monotonic SSD-miss curve**.

Before treating the ~127 ms as real unpriced GPU work, verify whether the current **union-of-command-buffer-spans metric is distorting the result**.

If the summed spans also show the same hump, the next target is no longer generic MLX bookkeeping: it is the **GPU/driver behavior inside an unchanged command stream**, especially barriers, fence waits, and host-memory mapping under unified-memory/NVMe contention.
