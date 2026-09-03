// Native scheduler probe for FlashNext.  This file is intentionally separate
// from MLX: it owns the Metal device, queue, command buffer, encoders, and
// memory barriers.  It does not yet implement the production Q4/G32 kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <mutex>

namespace {
constexpr uint32_t kMaxWidth = 16384;
constexpr uint32_t kMaxSteps = 64;

NSString *const kSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void flashnext_chain(device const float *src [[buffer(0)]],
                            device float *dst [[buffer(1)]],
                            constant uint &width [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
  if (id < width) dst[id] = src[id] + 1.0f;
}
)METAL";

std::once_flag g_resource_once;
int g_resource_status = -2;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pipeline = nil;

void initialize_resources() {
  std::call_once(g_resource_once, [] {
    g_device = MTLCreateSystemDefaultDevice();
    if (g_device == nil) {
      g_resource_status = -2;
      return;
    }
    NSError *error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:kSource
                                                     options:nil
                                                       error:&error];
    if (library == nil) {
      g_resource_status = -3;
      return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"flashnext_chain"];
    if (function == nil) {
      g_resource_status = -4;
      return;
    }
    g_pipeline = [g_device newComputePipelineStateWithFunction:function
                                                          error:&error];
    if (g_pipeline == nil) {
      g_resource_status = -5;
      return;
    }
    g_queue = [g_device newCommandQueue];
    g_resource_status = g_queue == nil ? -6 : 0;
  });
}
}

extern "C" int flashnext_native_available() {
  @autoreleasepool {
    initialize_resources();
    return g_resource_status == 0 ? 1 : 0;
  }
}

// Run a bounded dependency chain.  The result is input + steps.  `serial`
// uses one normal encoder.  `barrier` uses one concurrent-dispatch encoder and
// inserts a buffer barrier after every dispatch.  `fence` uses one encoder per
// dispatch with explicit MTLFence wait/update calls.  The host waits only
// after commit, so this measures native scheduling rather than MLX eval.
extern "C" int flashnext_native_chain(const float *input,
                                       float *output,
                                       uint32_t width,
                                       uint32_t steps,
                                       const char *strategy) {
  if (input == nullptr || output == nullptr || width == 0 ||
      width > kMaxWidth || steps == 0 || steps > kMaxSteps) {
    return -1;
  }
  if (strategy == nullptr || (std::strcmp(strategy, "serial") != 0 &&
                              std::strcmp(strategy, "barrier") != 0 &&
                              std::strcmp(strategy, "fence") != 0)) {
    return -12;
  }

  @autoreleasepool {
    initialize_resources();
    if (g_resource_status != 0) return g_resource_status;
    id<MTLCommandBuffer> command = [g_queue commandBuffer];
    if (command == nil) return -7;
    id<MTLFence> fence = nil;
    if (std::strcmp(strategy, "fence") == 0) {
      fence = [g_device newFence];
      if (fence == nil) return -8;
    }

    const NSUInteger bytes = static_cast<NSUInteger>(width) * sizeof(float);
    id<MTLBuffer> first = [g_device newBufferWithBytes:input
                                               length:bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> second = [g_device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (first == nil || second == nil) return -9;

    id<MTLBuffer> src = first;
    id<MTLBuffer> dst = second;
    id<MTLComputeCommandEncoder> shared_encoder = nil;
    if (std::strcmp(strategy, "serial") == 0) {
      shared_encoder = [command computeCommandEncoder];
    } else if (std::strcmp(strategy, "barrier") == 0) {
      shared_encoder = [command computeCommandEncoderWithDispatchType:
                                      MTLDispatchTypeConcurrent];
    }
    if (shared_encoder == nil && std::strcmp(strategy, "fence") != 0)
      return -10;

    for (uint32_t step = 0; step < steps; ++step) {
      id<MTLComputeCommandEncoder> encoder = shared_encoder;
      if (std::strcmp(strategy, "fence") == 0) {
        encoder = [command computeCommandEncoder];
        if (encoder == nil) return -10;
        if (step != 0) [encoder waitForFence:fence];
      }
      [encoder setComputePipelineState:g_pipeline];
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:dst offset:0 atIndex:1];
      [encoder setBytes:&width length:sizeof(width) atIndex:2];
      NSUInteger tg = MIN((NSUInteger)256, g_pipeline.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreads:MTLSizeMake(width, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      if (std::strcmp(strategy, "barrier") == 0)
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      if (std::strcmp(strategy, "fence") == 0) {
        [encoder updateFence:fence];
        [encoder endEncoding];
      }
      id<MTLBuffer> swap = src;
      src = dst;
      dst = swap;
    }
    if (shared_encoder != nil) [shared_encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return -11;

    memcpy(output, [src contents], bytes);
    return 0;
  }
}
