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

namespace mlx_node::hy4_cuda {

constexpr float kEngramEps = 1.1920928955078125e-7f;  // f32::EPSILON (2^-23)
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kMaxFloor = -1e30f;

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
          "hy4_cuda: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed");
    }
  }
  configured[fn] = smem_bytes;
}

inline void ensure_smem_carveout(CUfunction fn, int carveout_pct) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> configured;
  std::lock_guard<std::mutex> lock(mu);
  auto it = configured.find(fn);
  if (it != configured.end() && it->second == carveout_pct) {
    return;
  }
  (void)cuFuncSetAttribute(
      fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout_pct);
  configured[fn] = carveout_pct;
}

inline int device_sm_count(int dev_id) {
  static std::mutex mu;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(dev_id);
  if (it != cache.end()) {
    return it->second;
  }
  int sms = 48;
  cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_id);
  if (sms <= 0) sms = 48;
  cache[dev_id] = sms;
  return sms;
}

}  // namespace mlx_node::hy4_cuda

#endif  // MLX_NODE_CUDA_ENABLED
