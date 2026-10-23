// Uninitialized MLX-owned buffers the caller fills through a raw pointer, and
// numeric ops pinned to the CPU device. The CPU ops take the device per call,
// so they never touch the process-wide default device the way
// CpuConvertGuard does.

#include "mlx_common.h"
#include "mlx/primitives.h"

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

}  // namespace

extern "C" {

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
    array a(std::move(target), dtype, nullptr, {});
    auto data_owner = mlx::core::allocator::malloc(bytes);
    if (data_owner.buffer().ptr() == nullptr) return nullptr;
    a.set_data(std::move(data_owner));
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
