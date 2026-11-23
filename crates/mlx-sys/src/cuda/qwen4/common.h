#pragma once

#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace mlx_node::qwen4_cuda {

constexpr int kH = 2560;
constexpr int kW = 10240;
constexpr int kHc = 4;
constexpr int kRank = 320;
constexpr int kPleKeep = 9;
constexpr float kLog2E = 1.4426950408889634f;

inline mlx::core::array alloc_temp(
    mlx::core::Shape shape,
    mlx::core::Dtype dtype,
    mlx::core::cu::CommandEncoder& encoder) {
  mlx::core::array a(std::move(shape), dtype, nullptr, {});
  a.set_data(mlx::core::cu::malloc_async(a.nbytes(), encoder));
  encoder.add_temporary(a);
  return a;
}

inline mlx::core::array alloc_persistent(
    mlx::core::Shape shape,
    mlx::core::Dtype dtype,
    mlx::core::cu::CommandEncoder& encoder) {
  mlx::core::array a(std::move(shape), dtype, nullptr, {});
  a.set_data(mlx::core::cu::malloc_async(a.nbytes(), encoder));
  return a;
}

inline void ensure_dynamic_smem(CUfunction fn, int smem_bytes) {
  thread_local std::unordered_map<CUfunction, int> configured;
  auto it = configured.find(fn);
  if (it != configured.end() && it->second >= smem_bytes) {
    return;
  }
  CUresult rc = cuFuncSetAttribute(
      fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
  if (rc != CUDA_SUCCESS) {
    throw std::runtime_error(
        "cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) failed");
  }
  configured[fn] = smem_bytes;
}

// Implemented once in q8_prefill.cpp so cuda/qwen4_q8_prefill.cu.inc is never
// duplicated across translation units.
std::string_view qwen4_q8_prefill_source();

}  // namespace mlx_node::qwen4_cuda

#endif  // MLX_NODE_CUDA_ENABLED
