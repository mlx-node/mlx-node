// CUDA-specific device queries, free-memory probe, and CUDA graph toggles.

#include "mlx_common.h"
#include "mlx_stream.h"

#include <variant>

#ifdef MLX_NODE_CUDA_ENABLED
#include <cuda_runtime_api.h>
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/cuda/device.h"
#endif

namespace mlx_node {

#ifdef MLX_NODE_CUDA_ENABLED
std::shared_mutex& mlx_cuda_graphs_eval_mutex() noexcept {
  // Intentionally leaked so background threads or late destructors during
  // process shutdown never lock a destroyed mutex.
  static auto* mu = new std::shared_mutex();
  return *mu;
}

uint32_t& mlx_cuda_graphs_eval_read_depth() noexcept {
  thread_local uint32_t depth = 0;
  return depth;
}
#endif

int32_t cuda_device_static_memory_info(
    uint64_t* out_total,
    bool* out_is_unified) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        if (out_total) *out_total = static_cast<uint64_t>(prop.totalGlobalMem);
        if (out_is_unified) *out_is_unified = (prop.integrated != 0);
        return 1;
      }
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

// Check if CUDA backend is compiled in AND at least one CUDA GPU is visible.
bool mlx_cuda_is_available() {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    return mlx::core::cu::is_available() && mlx::core::gpu::device_count() > 0;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

// Query CUDA device 0 compute capability without constructing a primary context.
bool mlx_cuda_compute_capability(int32_t* out_major, int32_t* out_minor) {
  if (out_major) *out_major = 0;
  if (out_minor) *out_minor = 0;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
      return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
      return false;
    }
    if (out_major) *out_major = static_cast<int32_t>(prop.major);
    if (out_minor) *out_minor = static_cast<int32_t>(prop.minor);
    return true;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

// Query dynamic free GPU memory (NVML-first on discrete CUDA via MLX device_info;
// falls back to cudaMemGetInfo when NVML only reports a WDDM dedicated carveout
// on integrated/unified GPUs like RTX Spark N1X). Returns 0 on success, -1 otherwise.
int32_t mlx_device_free_memory(uint64_t* out_free) {
  if (out_free) *out_free = 0;
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    if (mlx::core::gpu::device_count() <= 0) {
      return -1;
    }
    cudaDeviceProp prop{};
    const bool have_prop = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    const auto& info = mlx::core::gpu::device_info(0);
    auto free_it = info.find("free_memory");
    auto total_it = info.find("total_memory");
    const bool nvml_has_free =
        free_it != info.end() && std::holds_alternative<size_t>(free_it->second);
    const size_t nvml_total =
        (total_it != info.end() && std::holds_alternative<size_t>(total_it->second))
            ? std::get<size_t>(total_it->second)
            : 0;
    const bool nvml_is_wddm_carveout =
        have_prop && prop.integrated != 0 && prop.totalGlobalMem > 0 &&
        nvml_total < prop.totalGlobalMem / 2;
    if (nvml_has_free && !nvml_is_wddm_carveout) {
      if (out_free) *out_free = static_cast<uint64_t>(std::get<size_t>(free_it->second));
      return 0;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      if (out_free) *out_free = static_cast<uint64_t>(free_bytes);
      return 0;
    }
    return -1;
  } catch (...) {
    return -1;
  }
#else
  return -1;
#endif
}

// Toggle MLX's CUDA graph capture dynamically (returns previous value).
// Used to disable graph capture during large prefill passes while keeping it
// enabled for low-latency decode. No-op (returns false) on non-CUDA builds.
bool mlx_cuda_set_use_graphs(bool enable) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    if (mlx_node::mlx_cuda_graphs_eval_read_depth() > 0) {
      std::cerr << "[MLX] Refusing mlx_cuda_set_use_graphs("
                << (enable ? "true" : "false")
                << ") while CUDA eval read lock is held on current thread"
                << std::endl;
      return mlx::core::cu::get_use_cuda_graphs();
    }
    std::unique_lock<std::shared_mutex> lk(
        mlx_node::mlx_cuda_graphs_eval_mutex());
    return mlx::core::cu::set_use_cuda_graphs(enable);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in mlx_cuda_set_use_graphs: " << e.what()
              << std::endl;
    return false;
  } catch (...) {
    return false;
  }
#else
  (void)enable;
  return false;
#endif
}

bool mlx_cuda_get_use_graphs() {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    return mlx::core::cu::get_use_cuda_graphs();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

}  // extern "C"
