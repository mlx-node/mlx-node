// CUDA-specific device-memory array allocation and host-to-device promotion
// helpers.

#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include <cuda_runtime_api.h>
#include <mutex>
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/cuda/device.h"
#endif

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

#ifdef MLX_NODE_CUDA_ENABLED
using mlx::core::array;

bool known_dtype_code(int32_t code) {
  return code >= 0 && code <= 6;
}

mlx_array* wrap(array a) {
  return reinterpret_cast<mlx_array*>(new array(std::move(a)));
}

struct CudaAllocStream {
  std::mutex mu;
  cudaStream_t stream{nullptr};
  int dev{0};

  cudaStream_t ensure(int device_idx) {
    if (stream == nullptr || dev != device_idx) {
      mlx::core::cu::device(device_idx).make_current();
      if (stream != nullptr) {
        cudaStreamDestroy(stream);
        stream = nullptr;
      }
      cudaError_t err =
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaStreamCreateWithFlags failed: ") +
            cudaGetErrorString(err));
      }
      dev = device_idx;
    } else {
      mlx::core::cu::device(device_idx).make_current();
    }
    return stream;
  }
};

CudaAllocStream& cuda_alloc_stream() {
  static auto* s = new CudaAllocStream();
  return *s;
}
#endif

}  // namespace

extern "C" {

// Allocate a row-contiguous uninitialized leaf array directly in CUDA device
// memory (`device == 0`, `cudaMallocAsync`) so GPU kernels get full L2 cache
// and unified-memory bandwidth on Windows WDDM. Returns null on non-CUDA
// builds or allocation failure.
mlx_array* mlx_array_alloc_gpu_uninit(const int64_t* shape,
                                      size_t ndim,
                                      int32_t dtype_code) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    if (!mlx::core::cu::is_available() || (ndim > 0 && !shape) ||
        !known_dtype_code(dtype_code)) {
      return nullptr;
    }
    auto dtype = to_mlx_dtype(dtype_code);
    mlx::core::Shape target;
    target.reserve(ndim);
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
      if (shape[i] <= 0 ||
          shape[i] > std::numeric_limits<mlx::core::ShapeElem>::max()) {
        return nullptr;
      }
      const size_t d = static_cast<size_t>(shape[i]);
      if (numel > std::numeric_limits<size_t>::max() / d) return nullptr;
      numel *= d;
      target.push_back(static_cast<mlx::core::ShapeElem>(shape[i]));
    }
    if (numel > std::numeric_limits<size_t>::max() / dtype.size()) {
      return nullptr;
    }
    const size_t bytes = numel * dtype.size();
    const int dev_idx =
        mlx::core::default_stream(mlx::core::Device::gpu).device.index;
    auto& s = cuda_alloc_stream();
    std::lock_guard<std::mutex> lock(s.mu);
    cudaStream_t stream = s.ensure(dev_idx);
    auto buf = mlx::core::cu::allocator().malloc_async(bytes, dev_idx, stream);
    if (buf.ptr() == nullptr) return nullptr;
    mlx::core::allocator::Data data_owner(
        buf, [](mlx::core::allocator::Buffer b) {
          mlx::core::allocator::allocator().free(b);
        });
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaStreamSynchronize failed: ") +
          cudaGetErrorString(err));
    }
    array a(std::move(data_owner), std::move(target), dtype);
    return wrap(std::move(a));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] array_alloc_gpu_uninit: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  (void)shape;
  (void)ndim;
  (void)dtype_code;
  return nullptr;
#endif
}

// Promote a settled, row-contiguous CUDA pinned-host array (`device == -1`)
// into CUDA device memory (`device >= 0`) via async H2D copy. Returns a new
// `mlx_array*` if copied, or null if already on GPU device / unsupported /
// non-CUDA build (caller keeps the original array on null).
mlx_array* mlx_array_ensure_cuda_device(mlx_array* handle) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    if (!handle || !mlx::core::cu::is_available()) return nullptr;
    auto* arr = reinterpret_cast<array*>(handle);
    if (arr->status() != array::Status::available) {
      arr->eval();
    }
    if (!arr->flags().row_contiguous || arr->data_size() != arr->size() ||
        arr->nbytes() == 0 || arr->buffer().ptr() == nullptr) {
      return nullptr;
    }
    auto* src_cbuf =
        static_cast<mlx::core::cu::CudaBuffer*>(arr->buffer().ptr());
    if (src_cbuf == nullptr || src_cbuf->device >= 0 ||
        src_cbuf->data == nullptr) {
      return nullptr;
    }
    const size_t bytes = arr->nbytes();
    const int dev_idx =
        mlx::core::default_stream(mlx::core::Device::gpu).device.index;
    const void* src_ptr =
        static_cast<const char*>(src_cbuf->data) + arr->offset();
    auto& s = cuda_alloc_stream();
    std::lock_guard<std::mutex> lock(s.mu);
    cudaStream_t stream = s.ensure(dev_idx);
    auto buf = mlx::core::cu::allocator().malloc_async(bytes, dev_idx, stream);
    if (buf.ptr() == nullptr) return nullptr;
    mlx::core::allocator::Data data_owner(
        buf, [](mlx::core::allocator::Buffer b) {
          mlx::core::allocator::allocator().free(b);
        });
    auto* dst_cbuf = static_cast<mlx::core::cu::CudaBuffer*>(buf.ptr());
    cudaError_t err = cudaMemcpyAsync(
        dst_cbuf->data, src_ptr, bytes, cudaMemcpyDefault, stream);
    if (err != cudaSuccess) {
      cudaStreamSynchronize(stream);
      throw std::runtime_error(
          std::string("cudaMemcpyAsync H2D failed: ") +
          cudaGetErrorString(err));
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaStreamSynchronize H2D failed: ") +
          cudaGetErrorString(err));
    }
    array out(std::move(data_owner), arr->shape(), arr->dtype());
    return wrap(std::move(out));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] array_ensure_cuda_device: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  (void)handle;
  return nullptr;
#endif
}

}  // extern "C"
