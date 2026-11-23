// Metal-specific device queries and D=256 SDPA capability probes.

#include "mlx_common.h"
#include "mlx_stream.h"

#include <variant>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"

namespace {
// Mirror of MLX's D=256 routing in ScaledDotProductAttention::use_fallback
// and has_fused_kernel (mlx/backend/metal/scaled_dot_product_attention.cpp)
// for inference (no logsumexp output, no sinks). Re-check on every MLX bump.
bool d256_full_sdpa_available(bool effective_dtype_is_float32) {
  return mlx::core::metal::is_nax_available() &&
      (mlx::core::env::enable_tf32() || !effective_dtype_is_float32);
}

bool d256_full_sdpa_would_use(
    bool effective_dtype_is_float32,
    int32_t query_head_dim,
    int32_t value_head_dim,
    int32_t query_length,
    int32_t key_length,
    bool do_causal,
    bool has_array_mask) {
  if (query_head_dim != 256 || value_head_dim != 256 || query_length <= 8) {
    return false;
  }
  // A causal mask needs q <= k; with no mask at all D=256 always falls back.
  if (!has_array_mask && !(do_causal && query_length <= key_length)) {
    return false;
  }
  if (query_length >= 1024 &&
      d256_full_sdpa_available(effective_dtype_is_float32)) {
    return true;
  }
  return !mlx::core::metal::is_nax_available() &&
      !effective_dtype_is_float32 && do_causal && !has_array_mask &&
      query_length >= 2048 && query_length == key_length;
}
}  // namespace
#endif

namespace mlx_node {

int32_t metal_device_static_memory_info(
    uint64_t* out_total,
    bool* out_is_unified) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (mlx::core::metal::is_available()) {
      const auto& info = mlx::core::gpu::device_info(0);
      auto it = info.find("max_recommended_working_set_size");
      if (it == info.end()) {
        it = info.find("memory_size");
      }
      if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
        if (out_total) {
          *out_total = static_cast<uint64_t>(std::get<size_t>(it->second));
        }
      }
      if (out_is_unified) *out_is_unified = true;
      return 1;
    }
  } catch (...) {
    return -1;
  }
#else
  (void)out_total;
  (void)out_is_unified;
#endif
  return 0;
}

}  // namespace mlx_node

extern "C" {

// Check whether the active Metal device can execute MLX's NAX kernels.
// Keep this fallible C boundary alongside mlx_metal_is_available(): device
// discovery is lazy and may throw when Metal is unavailable or degraded.
bool mlx_metal_is_nax_available() {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    return mlx::core::metal::is_nax_available();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

// Report whether MLX's NAX D=256 full-SDPA route exists on this device. The
// non-NAX square-causal route (q == k >= 2048) is not covered: callers that
// treat `false` as "scores materialize" stay conservative.
// `effective_dtype_is_float32` must describe the promoted dtype consumed by
// scaled_dot_product_attention, not merely the query's original dtype.
//
// Return 0 when the probe completed (including a supported non-Metal build,
// which conservatively reports false), or -1 for an invalid output pointer or
// a caught C++ exception. The output is initialized to false before probing so
// every failure is conservative across the FFI boundary.
int32_t mlx_metal_d256_full_sdpa_available(
    bool effective_dtype_is_float32,
    bool* out_available) {
  if (out_available == nullptr) {
    return -1;
  }
  *out_available = false;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    *out_available = d256_full_sdpa_available(effective_dtype_is_float32);
    return 0;
  } catch (...) {
    return -1;
  }
#else
  (void)effective_dtype_is_float32;
  return 0;
#endif
}

// D=256 eligibility probe for tests and planners: the mirror above, assuming a
// GPU stream and inference (training forces the fused forward for logsumexp).
int32_t mlx_metal_d256_full_sdpa_would_use(
    bool effective_dtype_is_float32,
    int32_t query_head_dim,
    int32_t value_head_dim,
    int32_t query_length,
    int32_t key_length,
    bool do_causal,
    bool has_array_mask,
    bool* out_would_use) {
  if (out_would_use == nullptr) {
    return -1;
  }
  *out_would_use = false;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    *out_would_use = d256_full_sdpa_would_use(
        effective_dtype_is_float32,
        query_head_dim,
        value_head_dim,
        query_length,
        key_length,
        do_causal,
        has_array_mask);
    return 0;
  } catch (...) {
    return -1;
  }
#else
  (void)effective_dtype_is_float32;
  (void)query_head_dim;
  (void)value_head_dim;
  (void)query_length;
  (void)key_length;
  (void)do_causal;
  (void)has_array_mask;
  return 0;
#endif
}

}  // extern "C"
