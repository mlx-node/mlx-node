#pragma once

#ifdef MLX_NODE_METAL_ENABLED

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::segmented_sdpa {

enum class SegmentedKernel {
  one_pass,
  two_pass_1,
  verify_two_pass_1,
};

// BF16 D=256 attention of q over prefix K/V followed by new K/V, equal to
// MLX's SDPA over the concatenated K/V. Without `require_segmented`,
// unsupported launches return that concatenated SDPA instead.
array segmented_sdpa(const array &q, const array &prefix_k,
                     const array &prefix_v, const array &new_k,
                     const array &new_v, float scale, bool causal,
                     bool require_segmented);

// Widest query chunk both segmented launches support for `gqa_factor`.
int segmented_max_query_length(metal::Device &device, int gqa_factor);

} // namespace mlx::core::segmented_sdpa

#endif
