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
  verify_nax_two_pass_1,
};

// One pipeline: a kernel and its function constants or template shape.
// `partitions` applies to the vector two-pass kernels, `gqa` / `rows` to the
// vector verify kernel, `tile_n` to the simdgroup-matrix verify kernel (a
// function constant) and, with `m`, to the tensor-op verify kernel (both
// compiled into the function name).
struct SegmentedSpecialization {
  SegmentedKernel kernel;
  bool causal;
  int partitions;
  int gqa;
  int rows;
  int tile_n;
  int m;
};

// How the verify block (causal, new rows == query rows) is served. The
// block kernels are the tensor-op (NAX) kernel on gen-17+ GPUs, else the
// simdgroup-matrix tile kernel; below the key crossover, or where neither
// can launch, the vector routes. `from_env` reads MLX_SDPA_VERIFY_TILE (0
// disables both block kernels, N >= 1 takes them from N keys; unset: the
// crossover this process measured on first use, see
// calibrate_block_min_keys) and MLX_SDPA_VERIFY_NAX (0 keeps the tile kernel on
// NAX devices; unset or 1 prefers the tensor-op kernel). `tile` and `nax`
// force one block kernel (tests).
enum class SegmentedTileMode : int {
  from_env = -1,
  vector = 0,
  tile = 1,
  nax = 2,
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
