#pragma once

#ifdef MLX_NODE_METAL_ENABLED

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"

#include <string>
#include <vector>

namespace mlx::core::segmented_sdpa {

enum class SegmentedKernel {
  one_pass,
  two_pass_1,
  verify_two_pass_1,
  verify_tile_two_pass_1,
};

// One pipeline: a kernel and its function constants. `partitions` applies to
// the vector two-pass kernels, `gqa` / `rows` to the vector verify kernel,
// `tile_n` to the simdgroup-matrix verify kernel.
struct SegmentedSpecialization {
  SegmentedKernel kernel;
  bool causal;
  int partitions;
  int gqa;
  int rows;
  int tile_n;
};

// How the verify block (causal, new rows == query rows) is served: the
// simdgroup-matrix tile kernel when the device supports it, else the vector
// routes. `from_env` reads MLX_SDPA_VERIFY_TILE: 0 disables the tile kernel,
// N >= 1 takes it from N keys (unset: the measured crossover).
enum class SegmentedTileMode : int {
  from_env = -1,
  vector = 0,
  tile = 1,
};

// The prebuilt pipeline from paged_attn.metallib; throws when it is missing.
MTL::ComputePipelineState *
segmented_kernel(metal::Device &device,
                 const SegmentedSpecialization &specialization);

// Every kernel name the dispatcher can request.
std::vector<std::string> metal_kernel_names();

// Every function-constant shape the dispatcher builds, at each partition count
// the vector-SDPA policy returns (MLX_SDPA_BLOCKS can add others).
std::vector<SegmentedSpecialization> metal_kernel_specializations();

// BF16 D=256 attention of q over prefix K/V followed by new K/V, equal to
// MLX's SDPA over the concatenated K/V on the vector routes (the tile route
// reduces in another order). Without `require_segmented`, unsupported
// launches return that concatenated SDPA instead.
array segmented_sdpa(const array &q, const array &prefix_k,
                     const array &prefix_v, const array &new_k,
                     const array &new_v, float scale, bool causal,
                     bool require_segmented, SegmentedTileMode tile_mode);

// Widest query chunk both segmented launches support for `gqa_factor`.
int segmented_max_query_length(metal::Device &device, int gqa_factor);

} // namespace mlx::core::segmented_sdpa

#endif
