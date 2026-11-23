#pragma once

#include <cstdint>

namespace mlx_node {

// Backend-specific helpers for `mlx_device_static_memory_info`.
// Caller initializes `*out_total = 0` and `*out_is_unified = true`.
// Returns 1 if the backend populated the outputs, 0 if the backend is inactive
// or not compiled in, or -1 on a caught C++ exception.
int32_t metal_device_static_memory_info(uint64_t* out_total, bool* out_is_unified);
int32_t cuda_device_static_memory_info(uint64_t* out_total, bool* out_is_unified);

}  // namespace mlx_node
