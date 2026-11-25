#pragma once

#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace mlx_node::inkling2_cuda {

constexpr int kHidden = 2304;
constexpr int kNumHeads = 24;
constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kGqa = 3;
constexpr int kRelDim = 16;
constexpr int kRelExtent = 1024;
constexpr int kSlidingWindow = 1024;
constexpr int kNumExperts = 128;
constexpr int kTopK = 8;
constexpr int kInter = 1152;
constexpr int kLoopRank = 64;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kRmsEps = 1e-6f;
constexpr float kQkClip = 22.627416997969522f;
constexpr float kRelClip = 8.0f;
constexpr float kStaticConvScale = 0.44721359549995793f;
constexpr float kDynConvScale = 0.3333333333333333f;

inline mlx::core::array alloc_temp(
    mlx::core::Shape shape,
    mlx::core::Dtype dtype,
    mlx::core::cu::CommandEncoder& encoder) {
  mlx::core::array a(std::move(shape), dtype, nullptr, {});
  a.set_data(mlx::core::cu::malloc_async(a.nbytes(), encoder));
  encoder.add_temporary(a);
  return a;
}

inline void ensure_dynamic_smem(CUfunction fn, int smem_bytes) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> configured;
  std::lock_guard<std::mutex> lock(mu);
  auto it = configured.find(fn);
  if (it != configured.end() && it->second >= smem_bytes) {
    return;
  }
  int static_smem = 0;
  cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
  if (smem_bytes > 49152 - static_smem) {
    CUresult rc = cuFuncSetAttribute(
        fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
    if (rc != CUDA_SUCCESS) {
      throw std::runtime_error(
          "cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) failed");
    }
  }
  configured[fn] = smem_bytes;
}

}  // namespace mlx_node::inkling2_cuda

#endif  // MLX_NODE_CUDA_ENABLED
