// Uninitialized MLX-owned buffers the caller fills through a raw pointer, and
// numeric ops pinned to the CPU device. The CPU ops take the device per call,
// so they never touch the process-wide default device the way
// CpuConvertGuard does.

#include "mlx_common.h"
#include "mlx/primitives.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#endif
#ifdef MLX_NODE_CUDA_ENABLED
#include <cuda_runtime_api.h>
#include <mutex>
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/cuda/device.h"
#endif

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using mlx::core::array;
using mlx::core::Device;

bool known_dtype_code(int32_t code) {
  return code >= 0 && code <= 6;
}

mlx_array* wrap(array a) {
  return reinterpret_cast<mlx_array*>(new array(std::move(a)));
}

#ifdef MLX_NODE_CUDA_ENABLED
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

// Return settled storage without waiting. Do not inspect a pending event:
// doing so can clear a GPU error that the next wait must report.
const void* mlx_array_data_ptr(mlx_array* handle) {
  try {
    if (handle == nullptr) return nullptr;
    auto* arr = reinterpret_cast<array*>(handle);
    const bool settled = arr->status() == array::Status::available ||
        (arr->status() == array::Status::evaluated && !arr->event().valid());
    if (!settled || arr->buffer().ptr() == nullptr) return nullptr;
    return arr->data<void>();
  } catch (...) {
    return nullptr;
  }
}

// A row-contiguous array of `shape` / `dtype` over a fresh, uninitialized
// allocator buffer; `*out_data` receives its CPU-visible storage pointer.
// Null (and `*out_data` null) on a bad shape or dtype, or when the allocator
// fails. The bytes are garbage until the caller writes every one of them.
mlx_array* mlx_array_alloc_uninit(const int64_t* shape,
                                  size_t ndim,
                                  int32_t dtype_code,
                                  void** out_data) {
  if (out_data) *out_data = nullptr;
  try {
    if (!out_data || (ndim > 0 && !shape) || !known_dtype_code(dtype_code)) {
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
    auto data_owner = mlx::core::allocator::malloc(bytes);
    if (data_owner.buffer().ptr() == nullptr) return nullptr;
    // The (data, shape, dtype) constructor makes an `available` leaf; a
    // primitive-less array built with set_data stays `unscheduled` and eval
    // throws on it.
    array a(std::move(data_owner), std::move(target), dtype);
    void* data = a.data<void>();
    if (data == nullptr) return nullptr;
    *out_data = data;
    return wrap(std::move(a));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] array_alloc_uninit: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

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

// A native Metal allocation with a CPU-visible pointer and a lease invoked
// after the last array sharing it dies. Null on unsupported builds or failure;
// `release_fn(ctx)` is not invoked on a null return.
mlx_array* mlx_array_alloc_purgeable(const int64_t* shape,
                                     size_t ndim,
                                     int32_t dtype_code,
                                     void** out_data,
                                     void (*release_fn)(void*),
                                     void* ctx) {
  if (out_data) *out_data = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!out_data || (ndim > 0 && !shape) || !known_dtype_code(dtype_code)) {
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
    auto armed = std::make_shared<bool>(false);
    mlx::core::allocator::Deleter deleter =
        [release_fn, ctx, armed](mlx::core::allocator::Buffer b) {
          if (b.ptr()) {
            static_cast<MTL::Buffer*>(b.ptr())->setPurgeableState(
                MTL::PurgeableStateNonVolatile);
          }
          mlx::core::allocator::allocator().free(b);
          if (*armed && release_fn) release_fn(ctx);
        };
    auto buffer = mlx::core::allocator::allocator().malloc(bytes);
    if (buffer.ptr() == nullptr) return nullptr;
    mlx::core::allocator::Data data_owner(buffer, std::move(deleter));
    array a(std::move(data_owner), std::move(target), dtype);
    void* data = a.data<void>();
    if (data == nullptr) return nullptr;
    *out_data = data;
    auto handle = wrap(std::move(a));
    *armed = true;
    return handle;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] array_alloc_purgeable: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  (void)shape;
  (void)ndim;
  (void)dtype_code;
  (void)release_fn;
  (void)ctx;
  return nullptr;
#endif
}

// Set purgeability on the array's existing Metal allocation. Returns the
// previous state (2 non-volatile, 3 volatile, 4 empty), or -1 on unsupported
// builds and invalid handles/states. Never evaluates or replaces the array.
int32_t mlx_array_set_purgeable_state(mlx_array* handle, int32_t state) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!handle || state < MTL::PurgeableStateNonVolatile ||
        state > MTL::PurgeableStateEmpty) {
      return -1;
    }
    auto& a = *reinterpret_cast<array*>(handle);
    if (a.buffer().ptr() == nullptr) return -1;
    auto* buffer = static_cast<MTL::Buffer*>(a.buffer().ptr());
    return static_cast<int32_t>(buffer->setPurgeableState(
        static_cast<MTL::PurgeableState>(state)));
  } catch (...) {
    return -1;
  }
#else
  (void)handle;
  (void)state;
  return -1;
#endif
}

mlx_array* mlx_array_astype_cpu(mlx_array* handle, int32_t dtype_code) {
  try {
    if (!handle || !known_dtype_code(dtype_code)) return nullptr;
    auto& a = *reinterpret_cast<array*>(handle);
    return wrap(mlx::core::astype(a, to_mlx_dtype(dtype_code), Device::cpu));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] astype_cpu: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array* mlx_array_log_cpu(mlx_array* handle) {
  try {
    if (!handle) return nullptr;
    auto& a = *reinterpret_cast<array*>(handle);
    return wrap(mlx::core::log(a, Device::cpu));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] log_cpu: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

mlx_array* mlx_array_negative_cpu(mlx_array* handle) {
  try {
    if (!handle) return nullptr;
    auto& a = *reinterpret_cast<array*>(handle);
    return wrap(mlx::core::negative(a, Device::cpu));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] negative_cpu: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// The scalar takes the input's floating dtype (float32 for integer inputs),
// exactly as mlx_array_mul_scalar does, so the bytes match that op.
mlx_array* mlx_array_mul_scalar_cpu(mlx_array* handle, double value) {
  try {
    if (!handle) return nullptr;
    auto& a = *reinterpret_cast<array*>(handle);
    auto dt = mlx::core::issubdtype(a.dtype(), mlx::core::floating)
        ? a.dtype()
        : mlx::core::float32;
    array scalar(static_cast<float>(value), dt);
    return wrap(mlx::core::multiply(a, scalar, Device::cpu));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] mul_scalar_cpu: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// MLX's own dequantize (affine / mxfp4 / ... modes, output dtype follows the
// scales) on the CPU device. K-quant modes are not accepted here.
mlx_array* mlx_dequantize_cpu(mlx_array* w,
                              mlx_array* scales,
                              mlx_array* biases,
                              int32_t group_size,
                              int32_t bits,
                              const char* mode) {
  try {
    if (!w || !scales) return nullptr;
    auto& w_arr = *reinterpret_cast<array*>(w);
    auto& s_arr = *reinterpret_cast<array*>(scales);
    std::optional<array> b_opt = std::nullopt;
    if (biases) b_opt = *reinterpret_cast<array*>(biases);
    std::optional<int> gs =
        group_size > 0 ? std::optional<int>(group_size) : std::nullopt;
    std::optional<int> b = bits > 0 ? std::optional<int>(bits) : std::nullopt;
    std::string mode_str = (mode && mode[0]) ? std::string(mode) : "affine";
    return wrap(mlx::core::dequantize(w_arr, s_arr, b_opt, gs, b, mode_str,
                                      std::nullopt, std::nullopt,
                                      Device::cpu));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] dequantize_cpu: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// TEST ONLY: the device of the stream an unevaluated array's primitive runs
// on: 0 CPU, 1 GPU, -1 when the array has no primitive (already evaluated or
// a leaf).
int32_t mlx_array_stream_device_type(mlx_array* handle) {
  try {
    if (!handle) return -1;
    auto& a = *reinterpret_cast<array*>(handle);
    if (!a.has_primitive()) return -1;
    return a.primitive().stream().device.type == Device::DeviceType::cpu ? 0
                                                                          : 1;
  } catch (...) {
    return -1;
  }
}

}  // extern "C"
