// Native scheduler probe for FlashNext.  This file is intentionally separate
// from MLX: it owns the Metal device, queue, command buffer, encoders, and
// memory barriers.  It provides both the simple synthetic chain probe and the
// full mixed-precision Q4/G32 MoE pipeline for Issue #45.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

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

// Issue #45 MoE pipelines and scratch buffers
id<MTLComputePipelineState> g_pipeline_qmv_proj = nil;
id<MTLComputePipelineState> g_pipeline_swiglu = nil;
id<MTLComputePipelineState> g_pipeline_down_combine = nil;

id<MTLBuffer> g_scratch_gate = nil;
id<MTLBuffer> g_scratch_up = nil;
id<MTLBuffer> g_scratch_act = nil;
id<MTLBuffer> g_scratch_down = nil;

std::mutex g_moe_mutex;
std::string g_last_error;

void set_error(const std::string &err) {
  std::lock_guard<std::mutex> lock(g_moe_mutex);
  g_last_error = err;
}

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

id<MTLBuffer> wrap_or_copy(id<MTLDevice> dev, const void *ptr, NSUInteger len, bool is_write) {
  if (ptr == nullptr || len == 0) return nil;
  if (((uintptr_t)ptr % 4096 == 0) && (len >= 4096)) {
    NSUInteger aligned_len = (len + 4095) & ~4095;
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:(void *)ptr
                                               length:aligned_len
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (buf != nil) return buf;
  }
  if (is_write) {
    return [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
  }
  return [dev newBufferWithBytes:ptr length:len options:MTLResourceStorageModeShared];
}

id<MTLBuffer> get_cached_buffer(id<MTLDevice> dev, __strong id<MTLBuffer> &cached, NSUInteger needed) {
  if (cached == nil || cached.length < needed) {
    cached = [dev newBufferWithLength:needed options:MTLResourceStorageModeShared];
  }
  return cached;
}

void encode_qmv(id<MTLComputeCommandEncoder> enc,
                id<MTLComputePipelineState> pipe,
                id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> s, id<MTLBuffer> b,
                id<MTLBuffer> routes, id<MTLBuffer> out,
                uint32_t tokens, uint32_t slots, uint32_t in_w, uint32_t out_w) {
  [enc setComputePipelineState:pipe];
  [enc setBuffer:x offset:0 atIndex:0];
  [enc setBuffer:w offset:0 atIndex:1];
  [enc setBuffer:s offset:0 atIndex:2];
  [enc setBuffer:b offset:0 atIndex:3];
  [enc setBuffer:routes offset:0 atIndex:4];
  [enc setBuffer:out offset:0 atIndex:5];
  [enc setBytes:&tokens length:sizeof(tokens) atIndex:6];
  [enc setBytes:&slots length:sizeof(slots) atIndex:7];
  [enc setBytes:&in_w length:sizeof(in_w) atIndex:8];
  [enc setBytes:&out_w length:sizeof(out_w) atIndex:9];
  MTLSize grid = MTLSizeMake(32, ((out_w + 7) / 8) * 2, tokens * slots);
  MTLSize tg = MTLSizeMake(32, 2, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void encode_swiglu(id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> pipe,
                   id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out,
                   uint32_t total_elements) {
  [enc setComputePipelineState:pipe];
  [enc setBuffer:gate offset:0 atIndex:0];
  [enc setBuffer:up offset:0 atIndex:1];
  [enc setBuffer:out offset:0 atIndex:2];
  [enc setBytes:&total_elements length:sizeof(total_elements) atIndex:3];
  NSUInteger max_tg = MIN((NSUInteger)256, pipe.maxTotalThreadsPerThreadgroup);
  MTLSize grid = MTLSizeMake(total_elements, 1, 1);
  MTLSize tg = MTLSizeMake(max_tg, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void encode_down_combine(id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pipe,
                         id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> s, id<MTLBuffer> b,
                         id<MTLBuffer> routes, id<MTLBuffer> scores,
                         id<MTLBuffer> scratch, id<MTLBuffer> out,
                         uint32_t tokens, uint32_t slots, uint32_t in_w, uint32_t out_w) {
  [enc setComputePipelineState:pipe];
  [enc setBuffer:x offset:0 atIndex:0];
  [enc setBuffer:w offset:0 atIndex:1];
  [enc setBuffer:s offset:0 atIndex:2];
  [enc setBuffer:b offset:0 atIndex:3];
  [enc setBuffer:routes offset:0 atIndex:4];
  [enc setBuffer:scores offset:0 atIndex:5];
  [enc setBuffer:scratch offset:0 atIndex:6];
  [enc setBuffer:out offset:0 atIndex:7];
  [enc setBytes:&tokens length:sizeof(tokens) atIndex:8];
  [enc setBytes:&slots length:sizeof(slots) atIndex:9];
  [enc setBytes:&in_w length:sizeof(in_w) atIndex:10];
  [enc setBytes:&out_w length:sizeof(out_w) atIndex:11];
  MTLSize grid = MTLSizeMake(((out_w + 7) / 8) * 64, 1, tokens);
  MTLSize tg = MTLSizeMake(64, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}
}  // namespace

extern "C" {

struct FlashNextMoEArgs {
  const float *x;
  const uint32_t *gate_weight;
  const void *gate_scales;
  const void *gate_biases;
  const uint32_t *up_weight;
  const void *up_scales;
  const void *up_biases;
  const uint32_t *down_weight;
  const void *down_scales;
  const void *down_biases;
  const uint32_t *routes;
  const float *scores;
  float *output;
  uint32_t tokens;
  uint32_t slots;
  uint32_t hidden_size;
  uint32_t inter_size;
  uint32_t expert_count;
};

int flashnext_native_available() {
  @autoreleasepool {
    initialize_resources();
    return g_resource_status == 0 ? 1 : 0;
  }
}

int flashnext_native_get_last_error(char *buf, uint32_t max_len) {
  if (buf == nullptr || max_len == 0) return -1;
  std::lock_guard<std::mutex> lock(g_moe_mutex);
  std::strncpy(buf, g_last_error.c_str(), max_len - 1);
  buf[max_len - 1] = '\0';
  return 0;
}

int flashnext_native_init_moe(const char *source) {
  if (source == nullptr) return -1;
  @autoreleasepool {
    initialize_resources();
    if (g_device == nil) {
      set_error("Metal device creation failed");
      return -2;
    }
    NSString *src_str = [NSString stringWithUTF8String:source];
    if (src_str == nil) {
      set_error("Failed to parse Metal source as UTF-8");
      return -3;
    }
    NSError *error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:src_str
                                                    options:nil
                                                      error:&error];
    if (library == nil) {
      set_error(error ? [error.localizedDescription UTF8String] : "Library compilation failed");
      return -4;
    }
    id<MTLFunction> fn_qmv = [library newFunctionWithName:@"flashnext_qmv_proj"];
    id<MTLFunction> fn_swiglu = [library newFunctionWithName:@"flashnext_swiglu"];
    id<MTLFunction> fn_down = [library newFunctionWithName:@"flashnext_fused_down_combine"];
    if (fn_qmv == nil || fn_swiglu == nil || fn_down == nil) {
      set_error("Failed to locate one or more MoE kernel functions in MTLLibrary");
      return -5;
    }
    g_pipeline_qmv_proj = [g_device newComputePipelineStateWithFunction:fn_qmv error:&error];
    if (g_pipeline_qmv_proj == nil) {
      set_error(error ? [error.localizedDescription UTF8String] : "Pipeline creation failed for qmv_proj");
      return -6;
    }
    g_pipeline_swiglu = [g_device newComputePipelineStateWithFunction:fn_swiglu error:&error];
    if (g_pipeline_swiglu == nil) {
      set_error(error ? [error.localizedDescription UTF8String] : "Pipeline creation failed for swiglu");
      return -7;
    }
    g_pipeline_down_combine = [g_device newComputePipelineStateWithFunction:fn_down error:&error];
    if (g_pipeline_down_combine == nil) {
      set_error(error ? [error.localizedDescription UTF8String] : "Pipeline creation failed for down_combine");
      return -8;
    }
    return 0;
  }
}

int flashnext_native_moe_execute(const FlashNextMoEArgs *args,
                                 const char *strategy,
                                 double *gpu_time_ms) {
  if (args == nullptr || strategy == nullptr) return -1;
  if (args->tokens == 0 || args->slots == 0 || args->hidden_size == 0 || args->inter_size == 0) {
    return -2;
  }
  if (std::strcmp(strategy, "serial") != 0 &&
      std::strcmp(strategy, "barrier") != 0 &&
      std::strcmp(strategy, "fence") != 0) {
    return -3;
  }
  if (g_pipeline_qmv_proj == nil || g_pipeline_swiglu == nil || g_pipeline_down_combine == nil) {
    set_error("MoE pipelines are not initialized; call flashnext_native_init_moe first");
    return -4;
  }

  @autoreleasepool {
    std::lock_guard<std::mutex> lock(g_moe_mutex);
    id<MTLCommandBuffer> command = [g_queue commandBuffer];
    if (command == nil) return -5;

    const NSUInteger x_bytes = static_cast<NSUInteger>(args->tokens) * args->hidden_size * sizeof(float);
    const NSUInteger gw_bytes = static_cast<NSUInteger>(args->expert_count) * args->inter_size * (args->hidden_size / 8) * sizeof(uint32_t);
    const NSUInteger gs_bytes = static_cast<NSUInteger>(args->expert_count) * args->inter_size * (args->hidden_size / 32) * 2; // bfloat16
    const NSUInteger dw_bytes = static_cast<NSUInteger>(args->expert_count) * args->hidden_size * (args->inter_size / 8) * sizeof(uint32_t);
    const NSUInteger ds_bytes = static_cast<NSUInteger>(args->expert_count) * args->hidden_size * (args->inter_size / 32) * 2;
    const NSUInteger route_bytes = static_cast<NSUInteger>(args->tokens) * args->slots * sizeof(uint32_t);
    const NSUInteger score_bytes = static_cast<NSUInteger>(args->tokens) * args->slots * sizeof(float);
    const NSUInteger out_bytes = static_cast<NSUInteger>(args->tokens) * args->hidden_size * sizeof(float);
    const NSUInteger inter_buf_bytes = static_cast<NSUInteger>(args->tokens) * args->slots * args->inter_size * sizeof(float);
    const NSUInteger scratch_down_bytes = static_cast<NSUInteger>(args->tokens) * args->slots * args->hidden_size * sizeof(float);

    id<MTLBuffer> b_x = wrap_or_copy(g_device, args->x, x_bytes, false);
    id<MTLBuffer> b_gw = wrap_or_copy(g_device, args->gate_weight, gw_bytes, false);
    id<MTLBuffer> b_gs = wrap_or_copy(g_device, args->gate_scales, gs_bytes, false);
    id<MTLBuffer> b_gb = wrap_or_copy(g_device, args->gate_biases, gs_bytes, false);
    id<MTLBuffer> b_uw = wrap_or_copy(g_device, args->up_weight, gw_bytes, false);
    id<MTLBuffer> b_us = wrap_or_copy(g_device, args->up_scales, gs_bytes, false);
    id<MTLBuffer> b_ub = wrap_or_copy(g_device, args->up_biases, gs_bytes, false);
    id<MTLBuffer> b_dw = wrap_or_copy(g_device, args->down_weight, dw_bytes, false);
    id<MTLBuffer> b_ds = wrap_or_copy(g_device, args->down_scales, ds_bytes, false);
    id<MTLBuffer> b_db = wrap_or_copy(g_device, args->down_biases, ds_bytes, false);
    id<MTLBuffer> b_routes = wrap_or_copy(g_device, args->routes, route_bytes, false);
    id<MTLBuffer> b_scores = wrap_or_copy(g_device, args->scores, score_bytes, false);
    id<MTLBuffer> b_out = wrap_or_copy(g_device, args->output, out_bytes, true);

    id<MTLBuffer> b_gate_out = get_cached_buffer(g_device, g_scratch_gate, inter_buf_bytes);
    id<MTLBuffer> b_up_out = get_cached_buffer(g_device, g_scratch_up, inter_buf_bytes);
    id<MTLBuffer> b_act = get_cached_buffer(g_device, g_scratch_act, inter_buf_bytes);
    id<MTLBuffer> b_scratch = get_cached_buffer(g_device, g_scratch_down, scratch_down_bytes);

    if (b_x == nil || b_gw == nil || b_uw == nil || b_dw == nil ||
        b_routes == nil || b_scores == nil || b_out == nil ||
        b_gate_out == nil || b_up_out == nil || b_act == nil || b_scratch == nil) {
      set_error("Failed to allocate one or more Metal buffers for MoE execution");
      return -6;
    }

    uint32_t total_swiglu = args->tokens * args->slots * args->inter_size;

    if (std::strcmp(strategy, "serial") == 0) {
      id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
      encode_qmv(enc, g_pipeline_qmv_proj, b_x, b_gw, b_gs, b_gb, b_routes, b_gate_out,
                 args->tokens, args->slots, args->hidden_size, args->inter_size);
      encode_qmv(enc, g_pipeline_qmv_proj, b_x, b_uw, b_us, b_ub, b_routes, b_up_out,
                 args->tokens, args->slots, args->hidden_size, args->inter_size);
      encode_swiglu(enc, g_pipeline_swiglu, b_gate_out, b_up_out, b_act, total_swiglu);
      encode_down_combine(enc, g_pipeline_down_combine, b_act, b_dw, b_ds, b_db,
                          b_routes, b_scores, b_scratch, b_out,
                          args->tokens, args->slots, args->inter_size, args->hidden_size);
      [enc endEncoding];
    } else if (std::strcmp(strategy, "barrier") == 0) {
      id<MTLComputeCommandEncoder> enc = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      encode_qmv(enc, g_pipeline_qmv_proj, b_x, b_gw, b_gs, b_gb, b_routes, b_gate_out,
                 args->tokens, args->slots, args->hidden_size, args->inter_size);
      encode_qmv(enc, g_pipeline_qmv_proj, b_x, b_uw, b_us, b_ub, b_routes, b_up_out,
                 args->tokens, args->slots, args->hidden_size, args->inter_size);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_swiglu(enc, g_pipeline_swiglu, b_gate_out, b_up_out, b_act, total_swiglu);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_down_combine(enc, g_pipeline_down_combine, b_act, b_dw, b_ds, b_db,
                          b_routes, b_scores, b_scratch, b_out,
                          args->tokens, args->slots, args->inter_size, args->hidden_size);
      [enc endEncoding];
    } else if (std::strcmp(strategy, "fence") == 0) {
      id<MTLFence> fence1 = [g_device newFence];
      id<MTLFence> fence2 = [g_device newFence];

      id<MTLComputeCommandEncoder> enc1 = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      encode_qmv(enc1, g_pipeline_qmv_proj, b_x, b_gw, b_gs, b_gb, b_routes, b_gate_out,
                 args->tokens, args->slots, args->hidden_size, args->inter_size);
      encode_qmv(enc1, g_pipeline_qmv_proj, b_x, b_uw, b_us, b_ub, b_routes, b_up_out,
                 args->tokens, args->slots, args->hidden_size, args->inter_size);
      [enc1 updateFence:fence1];
      [enc1 endEncoding];

      id<MTLComputeCommandEncoder> enc2 = [command computeCommandEncoder];
      [enc2 waitForFence:fence1];
      encode_swiglu(enc2, g_pipeline_swiglu, b_gate_out, b_up_out, b_act, total_swiglu);
      [enc2 updateFence:fence2];
      [enc2 endEncoding];

      id<MTLComputeCommandEncoder> enc3 = [command computeCommandEncoder];
      [enc3 waitForFence:fence2];
      encode_down_combine(enc3, g_pipeline_down_combine, b_act, b_dw, b_ds, b_db,
                          b_routes, b_scores, b_scratch, b_out,
                          args->tokens, args->slots, args->inter_size, args->hidden_size);
      [enc3 endEncoding];
    }

    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      set_error(command.error ? [command.error.localizedDescription UTF8String] : "Command buffer execution failed");
      return -7;
    }

    if (gpu_time_ms != nullptr) {
      if (command.GPUEndTime > command.GPUStartTime) {
        *gpu_time_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
      } else {
        *gpu_time_ms = 0.0;
      }
    }

    if (b_out.contents != args->output) {
      std::memcpy(args->output, b_out.contents, out_bytes);
    }
    return 0;
  }
}

// Simple dependency-chain benchmark probe (retained for backward compatibility)
int flashnext_native_chain(const float *input,
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

}  // extern "C"

