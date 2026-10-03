#pragma once

// Pure launch and route planning for segmented SDPA, shared by the Metal
// dispatch and the platform-independent tests. D=256 and the 32-lane
// reduction are algorithm shape constants; device limits come from the
// selected pipelines and MTLDevice at runtime.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlx::core::segmented_sdpa {

struct SegmentedSdpaCapabilities {
  size_t thread_execution_width;
  size_t max_threads_per_threadgroup;
  size_t static_threadgroup_memory;
  size_t max_threadgroup_memory;
};

struct SegmentedSdpaLaunchPlan {
  bool supported;
  bool two_pass;
  uint32_t partitions;
  uint32_t stage1_threads;
  uint32_t stage2_threads;
};

// MLX's vector-SDPA reduction policy (ScaledDotProductAttention::eval_gpu and
// sdpa_vector_2pass in mlx/backend/metal/scaled_dot_product_attention.cpp).
// Segmented SDPA equals concatenated K/V through MLX's vector SDPA only while
// both the route and the partition count match it, because either one changes
// the floating-point reduction order. `device_class` is the last character of
// the Metal architecture name; `blocks_override` is MLX_SDPA_BLOCKS.
inline bool sdpa_vector_uses_two_pass(char device_class, int sequence_length,
                                      int query_heads, int kv_heads) {
  return ((device_class == 'd' || device_class == 's') &&
          sequence_length >= 1024) ||
         (kv_heads < query_heads && sequence_length >= 4096);
}

inline int sdpa_vector_partition_count(char device_class, int sequence_length,
                                       int active_simdgroups,
                                       int blocks_override) {
  int partitions;
  if (device_class == 's') {
    partitions = 64;
    if (sequence_length > 1024 && active_simdgroups > 4) {
      if (sequence_length <= 8192) {
        partitions = 128;
      } else if (sequence_length <= 32768) {
        partitions = 256;
      } else if (sequence_length <= 65536) {
        partitions = 512;
      } else {
        partitions = 1024;
      }
    }
  } else if (device_class == 'd') {
    partitions = 128;
    if (active_simdgroups <= 2 && sequence_length > 8192) {
      partitions = 256;
    } else if (active_simdgroups >= 6) {
      if (sequence_length >= 16384 && sequence_length < 65536) {
        partitions = 512;
      } else if (sequence_length >= 65536) {
        partitions = 1024;
      }
    }
  } else {
    partitions = active_simdgroups >= 4 ? 64 : 32;
  }
  if (blocks_override > 0) {
    // The aggregation kernel consumes partitions in 32-wide chunks.
    partitions = ((blocks_override + 31) / 32) * 32;
  }
  return partitions;
}

inline SegmentedSdpaLaunchPlan
plan_segmented_sdpa_launch(int query_length, int gqa_factor, bool two_pass,
                           int partitions,
                           const SegmentedSdpaCapabilities &stage1,
                           const SegmentedSdpaCapabilities *stage2) {
  SegmentedSdpaLaunchPlan plan{
      false, two_pass, static_cast<uint32_t>(std::max(partitions, 0)), 0, 0};
  if (query_length < 1 || query_length > 8 || gqa_factor < 1 ||
      gqa_factor > 32 || partitions < 32 || (partitions % 32) != 0 ||
      stage1.thread_execution_width != 32 ||
      stage1.static_threadgroup_memory > stage1.max_threadgroup_memory) {
    return plan;
  }
  if (two_pass) {
    const uint64_t stage1_threads = uint64_t{32} * gqa_factor * query_length;
    if (stage1_threads > stage1.max_threads_per_threadgroup ||
        stage2 == nullptr || stage2->thread_execution_width != 32 ||
        stage2->max_threads_per_threadgroup < 1024 ||
        stage2->static_threadgroup_memory > stage2->max_threadgroup_memory) {
      return plan;
    }
    plan.stage1_threads = static_cast<uint32_t>(stage1_threads);
    plan.stage2_threads = 1024;
  } else {
    if (stage1.max_threads_per_threadgroup < 1024) {
      return plan;
    }
    plan.stage1_threads = 1024;
  }
  plan.supported = true;
  return plan;
}

struct SegmentedSdpaReductionPlan {
  bool two_pass;
  int partitions;
};

enum class SegmentedVerifyRoute : int {
  single = 0,
  one_pass = 1,
  unified = 2,
  split = 3,
};

// Rows of the leading chunk when a verify block exceeds the widest supported
// query chunk: the trailing chunk takes as many rows as the device allows.
// 0 = one chunk; -1 = two chunks cannot cover the block.
inline int segmented_verify_head_rows(int rows, int max_query_length) {
  if (rows < 1 || rows > 8 || max_query_length < 1) {
    return -1;
  }
  if (rows <= max_query_length) {
    return 0;
  }
  const int head = rows - std::min(max_query_length, rows - 1);
  return head > max_query_length ? -1 : head;
}

// First pass of the one-call verify kernel: two (head, row) pairs per
// simdgroup, all pairs of one KV head in one threadgroup.
inline SegmentedSdpaLaunchPlan
plan_segmented_verify_launch(int rows, int gqa_factor, int partitions,
                             const SegmentedSdpaCapabilities &stage1,
                             const SegmentedSdpaCapabilities &stage2) {
  SegmentedSdpaLaunchPlan plan{
      false, true, static_cast<uint32_t>(std::max(partitions, 0)), 0, 0};
  const int64_t pairs = int64_t{rows} * gqa_factor;
  if (rows < 2 || rows > 8 || gqa_factor < 1 || gqa_factor > 32 ||
      pairs % 2 != 0 || partitions < 32 || (partitions % 32) != 0 ||
      stage1.thread_execution_width != 32 ||
      stage1.static_threadgroup_memory > stage1.max_threadgroup_memory ||
      stage2.thread_execution_width != 32 ||
      stage2.max_threads_per_threadgroup < 1024 ||
      stage2.static_threadgroup_memory > stage2.max_threadgroup_memory) {
    return plan;
  }
  const uint64_t stage1_threads = uint64_t{32} * uint64_t(pairs / 2);
  if (stage1_threads > stage1.max_threads_per_threadgroup) {
    return plan;
  }
  plan.stage1_threads = static_cast<uint32_t>(stage1_threads);
  plan.stage2_threads = 1024;
  plan.supported = true;
  return plan;
}

// The reduction order of a (head, row) pair depends only on the route and
// partition count of the chunk that owns the row. One dispatch over the whole
// block is therefore exact only when both chunks reduce the same way.
// Requires head_rows >= 0.
inline SegmentedVerifyRoute
select_segmented_verify_route(int head_rows, SegmentedSdpaReductionPlan head,
                              SegmentedSdpaReductionPlan tail,
                              bool unified_supported) {
  if (head_rows == 0) {
    return SegmentedVerifyRoute::single;
  }
  if (!head.two_pass && !tail.two_pass) {
    return SegmentedVerifyRoute::one_pass;
  }
  if (head.two_pass && tail.two_pass && head.partitions == tail.partitions &&
      unified_supported) {
    return SegmentedVerifyRoute::unified;
  }
  return SegmentedVerifyRoute::split;
}

} // namespace mlx::core::segmented_sdpa
