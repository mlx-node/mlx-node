// In-place KV row store dispatcher for flat Qwen3.5 / Qwen4 caches.
// Backend implementations live in metal/mlx_kv_store_metal.cpp and
// cuda/mlx_kv_store_cuda.cpp.

#include "mlx_common.h"

namespace {
constexpr int kMaxTensors = 256;
}  // namespace

namespace mlx_node_internal {
#ifdef MLX_NODE_METAL_ENABLED
bool kv_store_rows_metal(int32_t tensors, mlx_array* const* dst,
                         mlx_array* const* src, const int32_t* offsets,
                         mlx_array** out);
#elif defined(MLX_NODE_CUDA_ENABLED)
bool kv_store_rows_cuda(int32_t tensors, mlx_array* const* dst,
                        mlx_array* const* src, const int32_t* offsets,
                        mlx_array** out);
#endif
}  // namespace mlx_node_internal

// Write `tensors` row blocks (`src[i]`, `[B, H, T, D_i]` or `[B, H, T]` for
// scales) into the matching cache buffers (`dst[i]`, `[B, H, cap, D_i]` /
// `[B, H, cap]`) at row `offsets[i]`, in place, in one dispatch. `out[i]`
// receives the cache array handles the caller must adopt (same buffers). Up
// to four tensors bind directly on Metal; more need MLX residency sets
// (macOS 15+). Returns false (message on stderr) without a supported GPU
// backend or on a contract violation; the caller keeps its slice_update path.
extern "C" bool mlx_kv_store_rows(int32_t tensors, mlx_array* const* dst,
                                  mlx_array* const* src,
                                  const int32_t* offsets, mlx_array** out) {
  for (int i = 0; out && i < tensors && i < kMaxTensors; ++i) out[i] = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  return mlx_node_internal::kv_store_rows_metal(tensors, dst, src, offsets, out);
#elif defined(MLX_NODE_CUDA_ENABLED)
  return mlx_node_internal::kv_store_rows_cuda(tensors, dst, src, offsets, out);
#else
  (void)tensors;
  (void)dst;
  (void)src;
  (void)offsets;
  return false;
#endif
}
