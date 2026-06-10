// =============================================================================
// Common weight storage FFI — shared by all compiled model forward passes.
//
// The weight map, mutex, and transpose cache are defined in mlx_qwen35_common.h.
// This file provides the extern "C" FFI functions for storing, clearing,
// querying, and setting model identity on those shared data structures.
//
// Auto-transpose: ALL 2D weights are pre-transposed on store. This covers
// any linear projection regardless of naming convention (q_proj, k_proj,
// router.proj, lm_head, etc.). 1D (norms/biases) and 3D (expert stacks)
// are excluded by the ndim==2 check.
// =============================================================================

#include "mlx_qwen35_common.h"

using namespace qwen35_common;

extern "C" {

void mlx_store_weight(const char* name, mlx_array* weight) {
  std::unique_lock<std::shared_mutex> lock(g_weights_mutex());
  auto& arr = *reinterpret_cast<array*>(weight);
  std::string key(name);
  g_weights().insert_or_assign(key, arr);
  // Auto-transpose all 2D weights for use by linear_proj() / get_weight_t().
  // Covers every linear projection regardless of naming convention.
  if (arr.ndim() == 2) {
    g_weight_transposes().insert_or_assign(key, transpose(arr));
  }
}

void mlx_store_quant_info(const char* prefix, const char* mode,
                          int bits, int group_size) {
  std::unique_lock<std::shared_mutex> lock(g_weights_mutex());
  g_quant_info().insert_or_assign(std::string(prefix),
                                  QuantInfo{std::string(mode), bits, group_size});
}

void mlx_clear_quant_info() {
  std::unique_lock<std::shared_mutex> lock(g_weights_mutex());
  g_quant_info().clear();
}

// Round-trip check: does the registry hold `prefix` with EXACTLY `mode`?
// Used by the Rust loader's load-time assertion that a sym8 layer's mode
// survived registration verbatim (a sym8 entry silently coerced/missing
// would make the compiled forward read the int8 operand as MXFP8/affine).
bool mlx_quant_info_mode_matches(const char* prefix, const char* mode) {
  if (prefix == nullptr || mode == nullptr) return false;
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  auto it = g_quant_info().find(std::string(prefix));
  if (it == g_quant_info().end()) return false;
  return it->second.mode == mode;
}

void mlx_clear_weights() {
  std::unique_lock<std::shared_mutex> lock(g_weights_mutex());
  g_weights().clear();
  g_weight_transposes().clear();
  // Prevent cross-model contamination: stale quant overrides referring to
  // the previous model's tensors would mis-dispatch dequant in the compiled
  // forward path. Clear the sidecar in the same critical section.
  g_quant_info().clear();
  g_active_model_id().store(0, std::memory_order_release);
}

size_t mlx_weight_count() {
  std::shared_lock<std::shared_mutex> lock(g_weights_mutex());
  return g_weights().size();
}

void mlx_set_model_id(uint64_t id) {
  g_active_model_id().store(id, std::memory_order_release);
}

}  // extern "C"
