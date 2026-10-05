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

// Simdgroup-matrix verify kernel (segmented_sdpa_verify_tile_2pass_1): keys
// per tile, threads and threadgroup bytes of one dispatch.
struct SegmentedSdpaTilePlan {
  bool supported;
  uint32_t tile_n;
  uint32_t stage1_threads;
  uint32_t stage2_threads;
  uint32_t threadgroup_bytes;
};

// Keys per tile from the device's threadgroup memory limit: the K/V tile is
// tile_n * (256 + 8) BF16 (one 16-byte pad per row against bank conflicts
// in the transposed K loads) followed by the fp32 score exchange
// (simdgroups * 32 lanes * tile_n / 4), and the kernel holds tile_n / 8
// score fragments per lane (16, 32 or 64 keys; 64 measured slower than 32
// from register pressure, so 32 is the ceiling). The largest tile whose
// tile part stays within three quarters of the limit and whose total fits
// wins, so a threadgroup never takes a whole limit's worth of the core's
// memory; the 32 KiB of Apple GPUs gives 32 keys (16.5 KiB + 12 KiB at 12
// simdgroups, 16 keys at 16 simdgroups).
inline uint32_t segmented_verify_tile_n(size_t max_threadgroup_memory,
                                        size_t static_threadgroup_memory,
                                        size_t simdgroups) {
  constexpr size_t kRowBytes = (256 + 8) * 2;
  if (max_threadgroup_memory <= static_threadgroup_memory) {
    return 0;
  }
  const size_t available = max_threadgroup_memory - static_threadgroup_memory;
  const size_t tile_budget = available / 4 * 3;
  for (uint32_t tile_n : {32u, 16u}) {
    const size_t tile_bytes = tile_n * kRowBytes;
    const size_t exchange_bytes = simdgroups * 32 * (tile_n / 4) * 4;
    if (tile_bytes <= tile_budget && tile_bytes + exchange_bytes <= available) {
      return tile_n;
    }
  }
  return 0;
}

// M = rows * gqa queries of one KV head per threadgroup, 8 per fragment row
// and 2 simdgroups (D halves) per 8 queries. The pipeline's own thread limit
// (register pressure) and the device's threadgroup memory decide; MLX's
// reduction kernel needs 1024 threads as everywhere else.
inline SegmentedSdpaTilePlan
plan_segmented_verify_tile_launch(int rows, int gqa_factor,
                                  const SegmentedSdpaCapabilities &stage1,
                                  const SegmentedSdpaCapabilities &stage2) {
  SegmentedSdpaTilePlan plan{false, 0, 0, 0, 0};
  const int64_t queries = int64_t{rows} * gqa_factor;
  if (rows < 2 || rows > 8 || gqa_factor < 1 || gqa_factor > 32 ||
      queries % 8 != 0 || stage1.thread_execution_width != 32 ||
      stage2.thread_execution_width != 32 ||
      stage2.max_threads_per_threadgroup < 1024 ||
      stage2.static_threadgroup_memory > stage2.max_threadgroup_memory) {
    return plan;
  }
  const uint64_t simdgroups = 2 * uint64_t(queries / 8);
  const uint64_t threads = 32 * simdgroups;
  const uint32_t tile_n =
      segmented_verify_tile_n(stage1.max_threadgroup_memory,
                              stage1.static_threadgroup_memory, simdgroups);
  if (tile_n == 0 || threads > stage1.max_threads_per_threadgroup) {
    return plan;
  }
  const uint64_t tile_bytes = uint64_t{tile_n} * (256 + 8) * 2;
  const uint64_t exchange_bytes = simdgroups * 32 * (tile_n / 4) * 4;
  plan.tile_n = tile_n;
  plan.stage1_threads = static_cast<uint32_t>(threads);
  plan.stage2_threads = 1024;
  plan.threadgroup_bytes = static_cast<uint32_t>(tile_bytes + exchange_bytes);
  plan.supported = true;
  return plan;
}

// Below some key count (prefix + new rows) the vector route is as fast as a
// block kernel: the block kernel's per-threadgroup prologue and 32-partition
// floor cost more than they save. The crossover differs per GPU, so each
// process measures it once (mlx_segmented_sdpa.cpp) on a verify block of
// the production shape at these key counts and selects with the function
// below.
constexpr int kSegmentedCalibrationKeys[] = {256, 512, 1024, 2048, 4096};
constexpr size_t kSegmentedCalibrationPoints =
    sizeof(kSegmentedCalibrationKeys) / sizeof(kSegmentedCalibrationKeys[0]);
constexpr int kSegmentedBlockMinKeysFloor = 256;
constexpr int kSegmentedBlockMinKeysCeiling = 8192;

// The block route must beat the vector route by this factor to take a key
// count: at a tie the bit-exact vector route keeps the step, and the
// crossover does not flip between processes on measurement noise.
constexpr double kSegmentedBlockMinGain = 1.05;

// The smallest measured key count from which the block route stays faster
// than the vector route by kSegmentedBlockMinGain through the largest
// measured count (a win followed by a loss is noise, not a crossover).
// None: one past the largest count. Non-positive or NaN seconds count as a
// loss. `keys` ascending; the result is clamped to [floor, ceiling].
inline int select_segmented_block_min_keys(const int *keys,
                                           const double *vector_seconds,
                                           const double *block_seconds,
                                           size_t count) {
  if (count == 0) {
    return kSegmentedBlockMinKeysCeiling;
  }
  int chosen = keys[count - 1] + 1;
  for (size_t i = count; i-- > 0;) {
    const double v = vector_seconds[i];
    const double b = block_seconds[i];
    if (!(v > 0.0) || !(b > 0.0) || !(b * kSegmentedBlockMinGain <= v)) {
      break;
    }
    chosen = keys[i];
  }
  return std::clamp(chosen, kSegmentedBlockMinKeysFloor,
                    kSegmentedBlockMinKeysCeiling);
}

// Contiguous key partitions of the tile kernel. MTLDevice does not report
// the core count, so the count follows the work: about 8 tiles per
// threadgroup so the Q prologue and the stage-2 partials stay small next to
// the K/V stream, at least 32 (and a multiple of 32, which the reduction
// kernel consumes) so batch x kv_heads x partitions threadgroups cover the
// GPU, at most 1024. `blocks_override` is MLX_SDPA_BLOCKS.
inline int segmented_verify_tile_partitions(int total_length, int tile_n,
                                            int blocks_override) {
  if (blocks_override > 0) {
    return std::min(1024, ((blocks_override + 31) / 32) * 32);
  }
  const int keys_per_partition = 8 * std::max(tile_n, 8);
  const int wanted =
      (total_length + keys_per_partition - 1) / keys_per_partition;
  return std::clamp(((wanted + 31) / 32) * 32, 32, 1024);
}

// Tensor-op verify kernel (segmented_sdpa_verify_nax_2pass_1): keys per
// tile, threads and threadgroup bytes of one dispatch. M = rows * gqa is
// compiled into the kernel; 256 threads (execution_simdgroups<8>) is the
// op's structural constant.
struct SegmentedSdpaNaxPlan {
  bool supported;
  uint32_t m;
  uint32_t tile_n;
  uint32_t stage1_threads;
  uint32_t stage2_threads;
  uint32_t threadgroup_bytes;
};

constexpr uint32_t kSegmentedNaxThreads = 256;

// The M values sdpa_segmented_nax.metal instantiates: multiples of 8 up to
// 64 (four softmax lanes per fused row within 256 threads).
inline bool segmented_nax_m_supported(int m) {
  return m >= 8 && m <= 64 && m % 8 == 0;
}

// fp32 scores [M][tile_n], BF16 probabilities [M][tile_n], fp32 previous
// scale [M], two rescale flags.
inline uint32_t segmented_nax_scratch_bytes(uint32_t m, uint32_t tile_n) {
  return m * tile_n * 4 + m * tile_n * 2 + m * 4 + 8;
}

// Keys per tile: K and V are tensor operands read from device memory, so
// only the softmax scratch bounds it. The largest of 64 and 32 whose scratch
// stays within three quarters of the device limit wins (32 KiB: 64 keys up
// to M = 48, 32 keys at M = 56 and 64).
inline uint32_t segmented_nax_tile_n(size_t max_threadgroup_memory,
                                     size_t static_threadgroup_memory,
                                     uint32_t m) {
  if (max_threadgroup_memory <= static_threadgroup_memory) {
    return 0;
  }
  const size_t budget =
      (max_threadgroup_memory - static_threadgroup_memory) / 4 * 3;
  for (uint32_t tile_n : {64u, 32u}) {
    if (segmented_nax_scratch_bytes(m, tile_n) <= budget) {
      return tile_n;
    }
  }
  return 0;
}

inline SegmentedSdpaNaxPlan
plan_segmented_verify_nax_launch(int rows, int gqa_factor,
                                 const SegmentedSdpaCapabilities &stage1,
                                 const SegmentedSdpaCapabilities &stage2) {
  SegmentedSdpaNaxPlan plan{false, 0, 0, 0, 0, 0};
  const int64_t queries = int64_t{rows} * gqa_factor;
  if (rows < 2 || rows > 8 || gqa_factor < 1 || gqa_factor > 32 ||
      !segmented_nax_m_supported(static_cast<int>(queries)) ||
      stage1.thread_execution_width != 32 ||
      stage1.max_threads_per_threadgroup < kSegmentedNaxThreads ||
      stage2.thread_execution_width != 32 ||
      stage2.max_threads_per_threadgroup < 1024 ||
      stage2.static_threadgroup_memory > stage2.max_threadgroup_memory) {
    return plan;
  }
  const uint32_t m = static_cast<uint32_t>(queries);
  const uint32_t tile_n = segmented_nax_tile_n(
      stage1.max_threadgroup_memory, stage1.static_threadgroup_memory, m);
  if (tile_n == 0) {
    return plan;
  }
  plan.m = m;
  plan.tile_n = tile_n;
  plan.stage1_threads = kSegmentedNaxThreads;
  plan.stage2_threads = 1024;
  plan.threadgroup_bytes = segmented_nax_scratch_bytes(m, tile_n);
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
