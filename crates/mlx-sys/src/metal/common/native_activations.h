#pragma once

#include <cstdint>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"

#ifdef MLX_NODE_METAL_ENABLED
namespace mlx::core::metal {

// Activations::silu uses an uncompiled sigmoid followed by multiplication.
// Share its immutable BF16 lookup across prefill and decode, retaining both
// rounding boundaries. Compiled sigmoid_mul uses a different approximation.
inline const array& native_bf16_silu_table() {
  static const array table = [] {
    std::vector<uint16_t> bits(65536);
    for (size_t i = 0; i < bits.size(); ++i)
      bits[i] = uint16_t(i);
    auto values = view(array(bits.data(), {65536}, uint16), bfloat16);
    auto result = values * sigmoid(values);
    eval({result});
    return result;
  }();
  return table;
}

} // namespace mlx::core::metal
#endif
