Read the MLX C++ source at /Users/gioma/Downloads/mlx-main and answer specific questions about its Metal backend. This is a source-reading task. Do not run benchmarks, do not modify anything, do not write files outside your report. Cite every claim as file:line.

The system you are reasoning about

MACQWEN runs a 176B sparse MoE model (Qwen3.8-Flash-Next) on an M4 Mac with 16 GB unified memory, streaming expert weights from SSD. It uses MLX 0.32.2 (the installed wheel; this source tree is the matching upstream). Decode runs at about 2.8 tokens/s.

Per decode token the runtime does this, 48 times (once per layer):

GPU computes router scores. Host calls mx.eval(scores) and blocks.
Host reads the routed expert list, picks about 8 experts of 512.
Sixteen worker threads pread about 24 MB of expert rows from SSD into freshly allocated numpy buffers. Roughly 432 fresh host allocations per token, about 1.18 GB of transient host memory.
Each numpy buffer is wrapped with mx.from_dlpack(array, copy=False) and, for bf16, .view(mx.bfloat16).
Three mx.gather_qmm calls (gate, up, down) consume those wrapped buffers. Group size 32, 4-bit.
mx.eval(flat) blocks again.

So 98 mx.eval calls per token, about 250 Metal command buffers per token, and about 3,027 mx.* calls per token.

What we measured, including things we cannot explain

Using Instruments' Metal System Trace, taking the union of metal-gpu-intervals spans for the process:

Zero drive (all expert reads served from mlocked RAM): GPU busy about 70-95 ms/token.
Production (about 390 MB/token physically read from SSD): GPU busy about 182 ms/token.
Identical dispatch count in both, about 201-250 command buffers per token. Same kernels, same shapes.

A controlled sweep varying only the fraction of expert reads that miss a pinned pool and reach the SSD, holding route width and dispatch count fixed:

miss fraction	physical MB/token	ms/token	GPU busy/token
0	0.2	217	69.5
0.125	152	278	93.8
0.25	300	440	171.7
0.5	603	596	170.7
1.0	1230	860	125.7

GPU busy is NON-MONOTONIC. It peaks around 25-50% miss and FALLS at 100% miss, while physical bytes keep rising. Token time is roughly linear in bytes. This shape reproduced forward and reverse.

Priced kernels only account for about 55 ms: gather_qmm 13-16 ms, GatedDeltaNet 18.5, hyper-connections 10.8, attention 5.3, shared expert 2.7, PLE 1.3, router 0.9. So at zero drive the books roughly close, and under drive load about 127 ms of GPU span time has no named kernel behind it.

We have measured and EXCLUDED all of these as causes: VM reclaim/compression/swap (flat, page-ins perfectly linear at 33 pages per MB), kernel read-ahead (F_RDAHEAD, flat), MLX_RESIDENCY_SET_MAX_PCT (no effect), reusing destination buffers instead of allocating fresh (no effect on bytes or rate), MLX_MAX_OPS_PER_BUFFER raised to 120 (19.8% SLOWER, and it did cut command buffers from 250 to 155 and per-buffer CPU-to-GPU latency from 1294 to 625 us), and merging the two evals per layer into one (11.4% slower).

We recently found that MLX's wired limit defaults to 0, so nothing is GPU-resident, and a standalone sweep suggested mx.set_wired_limit(2e9) might be ~13% faster. That is currently being verified.

Questions to answer from the source

Rank your findings by how much they could explain the 127 ms.

What does a metal-gpu-intervals span actually contain? Trace how a command buffer is built, committed and completed in mlx/backend/metal/device.cpp and eval.cpp. Does the GPU-side interval include anything other than kernel execution: residency establishment, fence or event waits, encoder setup, completion handlers? Is there anything that would make a command buffer's GPU span longer when the host is concurrently doing heavy I/O?
Residency. Read mlx/backend/metal/resident.cpp and .h and the allocator. With wired_limit == 0, what actually happens on every allocation and every commit? Does attach_new_sets still run? Is there per-commit work proportional to the number of allocations or sets even when nothing is wired? What is the cost path when an allocation is not in any set?
Command buffer splitting. Find max_ops_per_buffer and max_mb_per_buffer, their defaults, and exactly what triggers needs_commit(). We observe the buffer count per token varying between 203, 250 and 269 across runs with identical work. Explain what could vary it.
Foreign host memory. What happens in from_dlpack(copy=False)? Find where an external host pointer becomes an MTL::Buffer. Is newBufferWithBytesNoCopy used? What are the alignment requirements and what happens if they are not met? Is such a buffer tracked by the allocator, inserted into residency sets, or treated differently at commit? Is there a per-wrap cost that scales with buffer size?
mx.eval synchronization. What does eval do: commit and wait, shared events, fences? We measured about 143 microseconds of CPU-side scheduling per eval call and stack samples show IOSurfaceSharedEvent waitUntilSignaledValue. Where does that come from and is any of it avoidable?
Anything non-monotonic. Is there any code path whose cost would rise and then FALL as the fraction of freshly-allocated-versus-reused buffers increases? That is the shape we cannot explain. Look at the buffer cache in allocator.cpp (MLX settles at about 45.7 MB of cache regardless of the limit) and its eviction policy.
Untried knobs. List every environment variable and public API in this source that could plausibly matter for this workload, with its default, especially any we have not mentioned above.
Output

A concise report, findings ranked by explanatory power, each with file:line citations and a one-line statement of what experiment would confirm it. Explicitly separate what the code proves from what you are inferring. If you find nothing that explains the 127 ms, say so plainly rather than manufacturing a theory.

MLX 0.32.2 Metal backend, read against the Flash-Next decode loop

Read first, as required: docs/flashnext/handoff.md, docs/flashnext/research.md (all 3,351 lines), docs/flashnext/brief.md, README.md. Source read in /Users/gioma/Downloads/mlx-main: mlx/backend/metal/{device,eval,resident,allocator,event,fence,metal,quantized,device_info}.{cpp,h}, mlx/{transforms,array,scheduler,utils}.{cpp,h}, mlx/backend/common/buffer_cache.h, python/src/convert.cpp.

Findings ranked by explanatory power
1. Every dependent dispatch is preceded by a full buffer-scope memory barrier

Proven by code. MLX allocates every buffer with MTL::ResourceHazardTrackingModeUntracked (mlx/backend/metal/allocator.cpp:15-16), so it tracks hazards itself. set_input_array sets needs_barrier_ when this dispatch reads a buffer a previous dispatch in the same encoder wrote (mlx/backend/metal/device.cpp:354-355); register_output_array sets it for the write-after-read case (device.cpp:377-378). dispatch_threadgroups and dispatch_threads both call maybeInsertBarrier() first (device.cpp:411, 419), which emits memoryBarrier(MTL::BarrierScopeBuffers) (device.cpp:395). The encoder is created with MTL::DispatchTypeConcurrent (device.cpp:580), so without those barriers dispatches would overlap.

A 48-layer decode is a strict serial chain. Nearly every dispatch reads the previous dispatch's output, so nearly every dispatch carries a barrier.

Inferred, not proven here. A BarrierScopeBuffers barrier drains the GPU pipeline and invalidates buffer caches. Each post-barrier dispatch then takes its first reads from DRAM. Under 390 MB/token of NVMe DMA the DRAM path is contended, so that first-touch cost rises. The cost lands inside the command-buffer GPU span and belongs to no named kernel. This is the largest structural candidate in this source for the 127 ms, and it is consistent with the mechanism research.md already records from flash-moe.

Confirming experiment. Build MLX from this tree with -DMLX_METAL_DEBUG=ON (CMakeLists.txt:39, 181-182), which labels each command buffer with its primitive names (mlx/backend/metal/utils.h:35-46), then compare barrier-heavy against barrier-free by running one synthetic arm whose 48 layers are independent (no data dependency between layers) at fixed dispatch count and fixed bytes. If GPU busy per dispatch falls sharply, the barrier is the term.

2. A metal-gpu-intervals row is a command buffer, and it contains four things besides kernels

Proven by code. A committed command buffer can hold:

Compute dispatches, one per dispatch_threadgroups / dispatch_threads (device.cpp:408-422).
memoryBarrier calls, per finding 1 (device.cpp:395, 425).
waitForFence at encoder start, when any input of this encoder is an output of an earlier encoder (device.cpp:456-463). This is a GPU-side stall inside the span, not kernel execution.
updateFence at encoder end (device.cpp:470).
encodeSignalEvent for the shared event that ends each mx.eval (device.cpp:501).

Completion handlers are added at device.cpp:471 (per encoder), device.cpp:520 (per commit) and mlx/backend/metal/eval.cpp:66-67 (per evaluated array that does not trigger a commit). Those run after the buffer completes, on a driver thread, so they are outside the GPU span but are real host cost.

Residency establishment is not in this source. MLX never calls anything per commit except attach_new_sets (finding 3), so whatever the driver does to make resources resident is invisible here and is charged wherever Instruments charges it.

Consequence for your books. You cannot subtract priced kernels from a command-buffer span and call the remainder unnamed kernel work. The remainder also contains barriers, fence waits, encoder boundaries and event signals.

Confirming experiment. mx.metal.start_capture(path) (python/src/metal.cpp:77-87, mlx/backend/metal/metal.cpp:20-47) writes a .gputrace. Xcode's GPU pipeline profiler gives per-dispatch timing inside each command buffer, which metal-shader-profiler-intervals did not. Capture 3 decode tokens in production and 3 at zero drive and diff the per-dispatch times.

3. Every mx.eval creates a new MTLSharedEvent, forces an encoder break, and forces a commit

Proven by code. eval_impl creates one Event{stream} per call (mlx/transforms.cpp:104), which calls newSharedEvent() (mlx/backend/metal/event.cpp:16). At the end it signals that event (transforms.cpp:340) and then calls gpu::finalize(s) (transforms.cpp:343). Event::signal on a GPU stream calls encoder.signal_event (event.cpp:69-70), which forces end_encoding() before encoding the signal (device.cpp:500-501). gpu::finalize calls end_encoding() then commit() unconditionally (mlx/backend/metal/eval.cpp:71-77). Every new encoder allocates a fresh MTL::Fence (device.cpp:581).

The host block you see is array::wait() (mlx/array.cpp:145-153) into EventImpl::wait into waitUntilSignaledValue(value, -1) (event.cpp:28-30). That is exactly the IOSurfaceSharedEvent waitUntilSignaledValue frame in your stack samples, and the infinite timeout is why it lands in iokit_user_client_trap.

So per token at 98 evals: 98 new shared events, at least 98 new fences, at least 98 forced encoder breaks, at least 98 commits, and 98 host blocks. The 143 microseconds per eval you measured covers all of it.

What is avoidable. mx.eval(scores) is a real host dependency and cannot go. mx.eval(flat) at the end of a layer is a graph boundary, not a host read. mx.async_eval(flat) (transforms.cpp:350-362) runs the same eval_impl without the trailing .wait(), so it keeps the commit and the event signal but drops the host block. It still creates one shared event per call. Alternatively mx.synchronize() uses waitUntilCompleted() on the command buffer with no event at all (mlx/scheduler.cpp:14-24, mlx/backend/metal/eval.cpp:79-81, device.cpp:564-574).

Confirming experiment. Replace the per-layer mx.eval(flat) with mx.async_eval(flat) and keep mx.eval(scores). Token IDs must match. Three arms per condition, order reversed. This is not the FLASHNEXT_ONE_SYNC experiment: that merged two graphs into one eval, this keeps both boundaries and removes one host wait.

4. Residency: with wired_limit == 0 nothing is wired, and the per-commit cost is one atomic load

Proven by code. wired_limit_ defaults to 0 (mlx/backend/metal/allocator.h:71). ResidencySets::insert is called on every non-heap allocation (allocator.cpp:161-163) and on every foreign wrap (allocator.cpp:213). It takes mtx_, reads allocatedSize(), inserts into buf_to_set_, then returns early because total_wired_ + bytes > capacity_ with capacity_ == 0 (resident.cpp:136-155, early return at 150-152). No set is touched, no commit() is issued, choose_set_locked is never reached.

attach_new_sets runs at construction (device.cpp:321) and before every commit (device.cpp:519), but returns without taking the lock when num_sets_ == attached (resident.cpp:222-226). Set 0 is the only set that exists when nothing is wired, so per commit this is one acquire load. There is no per-commit work proportional to allocations or sets.

Cost path when an allocation is not in any set: two mutexes and one hash insert on allocate, the same on free. At 432 wraps per token this is well under 1 ms. Not your 127 ms.

Warning about mx.set_wired_limit(2e9). set_wired_limit calls residency_sets_.resize() (allocator.cpp:100-105), which walks buf_to_set_ once and adds what fits (resident.cpp:189-200). After that, every allocation that fits under the budget takes add_to_set_locked plus sets_[idx].set->commit() (resident.cpp:153-154), and every free takes remove_from_set_locked plus another commit() (resident.cpp:167-171). If the dlpack wraps land inside the budget, you buy roughly 864 MTLResidencySet::commit() calls per token. If the dense weights fill the budget first, the wraps hit the early return and you get today's behaviour on the wraps plus real wiring on the dense side. Which happens depends on allocation order, so the outcome is not predictable from the code. Also note MLX_RESIDENCY_SET_MAX_PCT defaults to 5 (mlx/utils.h:199), giving your measured 750 MB per set, and capacity_ is not affected by it (resident.h:25-26).

Confirming experiment. Run with MLX_RESIDENCY_DEBUG=1 and a wired limit, and count MTLResidencySet commits with a dtrace/Instruments symbol probe, or simply compare mx.get_active_memory() against wired size while sweeping the limit from 1 GB to 6 GB. Three arms, reversed order.

5. Command-buffer splitting: two caps, and one load-dependent third path you have not accounted for

Proven by code. needs_commit() returns (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb) (device.cpp:511-514). Defaults are set from the last character of the architecture string (device.cpp:602-624): 'p' 20/40, 'g' 40/40, 's' 50/50, 'd' 50/50, default 40/40. A Mac base or Pro part is 'g', so max_ops 40, max_mb 40. Both are overridable, MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER (mlx/utils.h:182-192).

buffer_ops_ counts only dispatches (device.cpp:412, 420) and resets at commit (device.cpp:560). buffer_sizes_ accumulates a.data_size() for each input array new to this encoder (device.cpp:349-351), resets at commit (device.cpp:561), and all_inputs_ clears at each end_encoding() (device.cpp:496) so an array re-bound in a later encoder of the same buffer counts again.

A defect worth knowing: data_size() is documented as being in units of item_size, not bytes (mlx/array.h:346), but needs_commit compares buffer_sizes_ >> 20 against a megabyte figure. For your Q4 weights, stored as packed uint32, the byte cap therefore counts 1 per 4 bytes, so the effective cap is about 160 MB, not 40 MB. Your per-layer gather inputs are roughly 7.4 M elements, so the byte cap never fires. Your splits come from the 40-dispatch cap and from evals.

Buffer count arithmetic. buffers/token is at least evals/token (98, one forced commit each, transforms.cpp:343), plus roughly ceil(dispatches / 40). Your 250 implies about 6,000 dispatches per token from the ops path.

Three sources of run-to-run variance in 203 / 250 / 269:

needs_commit() is checked after each primitive (mlx/backend/metal/eval.cpp:59), not after each dispatch. A primitive that emits several dispatches overshoots the cap, so the split point drifts with graph shape.
Adaptive top-k keeps 7.85 to 7.95 experts per layer. The routing graph's op count and the distinct-input set both vary with that.
MAX_ACTIVE_TASKS. transforms.cpp:25 sets it to 10. scheduler::notify_new_task increments only on the needs_commit commit path (eval.cpp:61), and the completion handler decrements (eval.cpp:63). When more than 10 command buffers are in flight, eval_impl calls gpu::finalize(s) on every open stream mid-tape (transforms.cpp:271-279), which is an extra commit, and then blocks in scheduler::wait_for_one() (transforms.cpp:280), a condition-variable wait (mlx/scheduler.h:56-64). That wait appears in stack samples as _pthread_cond_wait -> __psynch_cvwait, indistinguishable from your read-pool futures. This path fires more often when the GPU falls behind, which is exactly what happens when spans stretch.

Confirming experiment. Log mx.get_active_memory() is not enough. Patch a counter into a local build, or simpler: run the miss sweep and record buffers/token per arm alongside the Metal-trace CPU-to-GPU latency. If the throttle is firing, buffers/token rises where GPU busy rises. Your sweep shows buffers flat at 201, so in that harness the throttle is probably not firing; in production at 250 it may be. Check by comparing a production arm's buffer count against 98 + ceil(dispatches/40).

6. Foreign host memory: newBufferWithBytesNoCopy, tracked by the allocator, never cached

Proven by code. mx.from_dlpack(a, copy=False) on a CPU array goes to cpu_nd_array_to_mlx_no_copy (python/src/convert.cpp:190-224, dispatched at 285-289), which calls mx::allocator::make_buffer (convert.cpp:200-202). That reaches MetalAllocator::make_buffer (mlx/backend/metal/allocator.cpp:207-218), which calls device_->newBuffer(ptr, size, resource_options, nullptr), the newBufferWithBytesNoCopy overload, with StorageModeShared | HazardTrackingModeUntracked.

Answers to your sub-questions:

Yes, newBufferWithBytesNoCopy is used, allocator.cpp:208.
MLX does no alignment check. It relies on Metal returning nil, and convert.cpp:203-207 falls through to std::nullopt, after which copy=False raises [convert] Cannot import a CPU array without a copy (convert.cpp:290-293). Since your run does not raise, every wrap is succeeding, so your numpy pointers are page-aligned. That is worth stating: the platform requirement is page alignment, and a np.empty of 800 KB gets it from macOS malloc's large path. A smaller or offset buffer would raise, not silently copy.
Tracked by the allocator: yes. make_buffer increments active_memory_, peak_memory_ and num_resources_ (allocator.cpp:214-216), and inserts into residency tracking (allocator.cpp:213). Note this insert is unconditional, unlike malloc's which skips heap buffers (allocator.cpp:161-163).
Inserted into residency sets: no, while wired_limit == 0, per finding 4.
Treated differently at commit: no. set_input_array (device.cpp:345-358) makes no distinction.
Treated differently at free: yes. The array carries a custom deleter that calls allocator::release (convert.cpp:216-221), which goes to MetalAllocator::release (allocator.cpp:220-232), not free. So a wrapped buffer is never recycled into the buffer cache. Every token pays 432 fresh newBufferWithBytesNoCopy calls and 432 release calls.
Per-wrap cost that scales with size: not in MLX. make_buffer does constant host work. The scaling, if any, is inside newBufferWithBytesNoCopy, which must map the host range into the GPU address space. Apple GPUs do not take demand page faults, so the range must be resident before the GPU runs. That is driver work MLX does not see, and it is the one place in this whole path where a per-byte cost could hide.

One more consequence. gpu::eval retains every input array's Data shared_ptr until the command buffer completes (mlx/backend/metal/eval.cpp:47-57, 62-67). Your 24 MB numpy blocks are therefore freed on a Metal completion-handler thread, not on the main thread. About 1.18 GB per token of large-block frees run on a driver thread. Instruments attributes that to neither the GPU nor the main thread.

Also: .view(mx.bfloat16) on a same-width dtype takes the copy_shared_buffer branch with no dispatch (mlx/backend/gpu/primitives.cpp:240-250, first branch at 249-251). But it is still a primitive, so gpu::eval still runs and still adds a completion handler (eval.cpp:66-67). Several hundred extra completion handlers per token for zero GPU work.

Confirming experiment. Time mx.from_dlpack(a, copy=False) alone, in isolation, against buffer size: 100 KB, 800 KB, 3 MB, 24 MB, 1,000 iterations each, no GPU work. If the per-call time is flat, the wrap has no per-byte cost and this is closed. If it scales, you have found a term proportional to bytes that sits on the main thread.

7. The buffer cache explains the 45.7 MB plateau and the 18% cost of disabling it

Proven by code. reuse_from_cache(size) finds the smallest cached buffer of at least size but rejects it when it->first >= std::min(2 * size, size + 2 * page_size_) (mlx/backend/common/buffer_cache.h:30-46), with page_size_ = vm_page_size (allocator.cpp:49), 16 KB on Apple Silicon. For a large allocation the acceptance window is only [size, size + 32 KB). So the cache only helps when the exact size recurs.

Eviction: release_cached_buffers(n) clears the whole cache when n >= 0.9 * pool_size_, otherwise evicts LRU from the tail (buffer_cache.h:58-85). It is called from malloc when get_active_memory() + get_cache_memory() + size >= gc_limit_ or num_resources_ >= resource_limit_ (allocator.cpp:132-138), and again when get_cache_memory() > max_pool_size_ (allocator.cpp:169-173).

On your machine gc_limit_ = min(0.95 * recommendedMaxWorkingSetSize, block_limit_) and max_pool_size_ = block_limit_ = min(1.5 * max_rec_ws, 0.95 * memsize) (allocator.cpp:63-65). With max_rec_ws near 15 GB, gc_limit_ is about 14.25 GB and max_pool_size_ about 16.3 GB. Neither is ever reached. So no eviction path fires, and the 45.7 MB is simply the steady-state set of freed-but-not-yet-reused intermediates, about two layers' worth. Raising the limit does nothing because the limit was never binding. That is the explanation your log did not have.

Setting the limit to 0 costs 18% because every intermediate then goes to a real device_->newBuffer plus a residency-tracking insert plus a real release (allocator.cpp:151-163, 190-200). A fresh shared-storage Metal buffer is also lazily backed, so its pages are committed on first touch. When the GPU is the first toucher, that fault time lands inside the GPU span.

Confirming experiment. Instrument mx.get_cache_memory() per layer during one token. If it oscillates rather than sitting flat at 45.7 MB, allocation sizes are varying and the tight reuse window is costing you real cache misses. If it sits flat, the cache is doing its job and this is closed.

Question 6, the non-monotonic shape: I found nothing

There is no code path in this source whose cost rises and then falls with the fraction of freshly-allocated versus reused buffers. I looked at the allocator, the buffer cache, the eviction policy, residency, command-buffer splitting and the encoder's hazard bookkeeping. Every one of them is monotonic, or flat, or does not fire at all on your machine.

I will not manufacture a theory. Two things the code does establish that constrain the search:

The sweep holds allocation behaviour fixed. Route width, dispatch count and buffer count are constant across the miss sweep (your 201.2 to 201.6). Nothing in MLX's allocation or splitting logic reads the miss fraction. So the hump lives in the GPU-side execution of an unchanged command stream, not in MLX's host-side bookkeeping.
GPU busy is buffers × mean span, and buffers are flat, so the whole hump is mean span. With metal-gpu-intervals rows being command buffers, mean span is sensitive to how much command buffers overlap in the queue. If Instruments starts a span when a buffer is scheduled rather than when it begins executing, a submission pattern that queues buffers in tight bursts produces overlapping spans whose union collapses, while a spread-out pattern produces disjoint spans whose union is large. At miss 1.0 the host is blocked in pread for most of the token, so submissions arrive in one dense burst per layer. At intermediate miss they arrive spread out. That would produce exactly your shape. I cannot verify it from this source, because span semantics are Instruments' definition, not MLX's.

The experiment that settles it. Export metal-gpu-intervals for the miss sweep and, instead of the union, compute the plain sum of durations and the count of overlapping pairs. If the sum is monotonic in bytes while the union humps, the hump is a measurement artifact of overlap and there is no missing 127 ms of work. If the sum humps too, the hump is real and belongs to GPU execution. This costs one re-analysis of traces you already have and it is the cheapest discriminating test available.

Question 7: knobs in this source, with defaults

Never mentioned in your log:

Knob	Default	Where	Why it might matter
MLX_MAX_MB_PER_BUFFER	40, from arch	mlx/utils.h:188-192, device.cpp:606-626	The second commit trigger. Counts elements, not bytes, so on your Q4 weights it never fires. Lowering it would add byte-driven splits; that is the untested half of the MLX_MAX_OPS_PER_BUFFER experiment.
MLX_BFS_MAX_WIDTH	20	mlx/utils.h:177-180, used at transforms.cpp:181	Controls tape construction width in eval_impl. Changes which primitives land in which command buffer, and therefore the barrier count. Host-side only, bit-exact.
MLX_METAL_FAST_SYNCH	0	mlx/utils.h:209-212, mlx/backend/metal/fence.cpp:15	Swaps the cross-stream Fence from a shared event to a spin kernel. Does not affect mx.eval's event. Only useful if you ever run two streams. research.md already records spin-poll GPU wait losing 23% in flash-moe.
MLX_RESIDENCY_DEBUG	0	mlx/utils.h:204-207	Prints each residency set as created. You used it. Note it prints only on creation, so with wired limit 0 you see one line and nothing else.
MLX_SDPA_BLOCKS	0, auto	mlx/backend/metal/scaled_dot_product_attention.cpp:523	Attention is 5.3 ms. Low value.
MLX_DISABLE_COMPILE	unset	mlx/compile.cpp:220	You measured compile at about 1 ms. Low value.
MLX_METAL_GPU_ARCH	""	mlx/utils.h:224-227, device.cpp:589-591	Overrides the architecture string, which selects the max-ops and max-mb defaults and get_qmv_batch_limit (quantized.cpp:85-135) and NAX gating. Do not set it; listed because it silently changes kernel selection if anything else sets it.
MLX_METAL_DEBUG	OFF, build option	CMakeLists.txt:39, 181-182, mlx/backend/metal/utils.h:35-46	The direct answer to "127 ms with no named kernel". Labels each command buffer with its concatenated primitive names, which then appear in metal-application-command-buffer-submissions. Requires building this tree.
mx.metal.start_capture(path) / stop_capture()	n/a	python/src/metal.cpp:77-93, mlx/backend/metal/metal.cpp:20-53	Writes a .gputrace. Xcode's GPU pipeline profiler gives per-dispatch timing, which the Metal System Trace template could not. The single highest-value untried instrument here.
mx.async_eval(x)	n/a	mlx/transforms.cpp:350-362, binding at python/src/transforms.cpp:1196	Same commit and event signal, no host block. See finding 3.
mx.synchronize()	n/a	mlx/scheduler.cpp:14-24, device.cpp:564-574	Waits with waitUntilCompleted() instead of a shared event. No newSharedEvent() per call.
mx.set_memory_limit	~16.3 GB	mlx/memory.h:36-50, allocator.cpp:87-94	Feeds gc_limit_ and the mid-tape throttle at transforms.cpp:272. Currently never binds. Not a lever, listed so nobody lowers it by accident.
mx.device_info()	n/a	mlx/backend/metal/device_info.cpp:24-62	Cheap premise check. Report architecture to confirm whether your max-ops default is 40 or 50, and max_recommended_working_set_size to confirm the 750 MB residency figure.

MLX_METAL_NO_NAX is a build define, and is_nax_available() requires architecture generation 17 or higher (device.cpp:951-965), so on M4 the NAX kernel paths in quantized.cpp:1040, 1236, 1602 are dead. Nothing to gain there.

What the code does not support
Residency is not your 127 ms. With wired_limit == 0 there is no per-commit residency work at all beyond one atomic load.
The buffer cache is not thrashing. Its eviction thresholds are never reached on your machine.
The DLPack wrap has no per-byte cost inside MLX. If one exists it is inside newBufferWithBytesNoCopy, and finding 6's experiment measures it directly in about ten minutes.
Nothing here is non-monotonic in miss fraction. The cheapest next step is the union-versus-sum re-analysis of your existing traces, not new runs.