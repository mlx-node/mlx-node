// TEST-ONLY oracle: MLX's own K-quant path, reached by mode string. Only the
// MLX fork pinned at 053e43fec has it; delete this file with the pin move.

#include "mlx_common.h"

using mlx::core::array;

namespace {
std::optional<array> opt(mlx_array *a) {
  if (!a)
    return std::nullopt;
  return *reinterpret_cast<array *>(a);
}

const array &ref(mlx_array *a) { return *reinterpret_cast<array *>(a); }

mlx::core::Device device_of(int32_t code) {
  return code == 1 ? mlx::core::Device(mlx::core::Device::gpu)
                   : mlx::core::Device(mlx::core::Device::cpu);
}
} // namespace

extern "C" {

mlx_array *mlx_test_fork_kquant_quantized_matmul(mlx_array *x, mlx_array *w,
                                                 mlx_array *scales,
                                                 mlx_array *biases,
                                                 bool transpose, int group_size,
                                                 int bits, const char *mode,
                                                 int32_t device) {
  try {
    auto result = mlx::core::quantized_matmul(
        ref(x), ref(w), ref(scales), opt(biases), transpose, group_size, bits,
        mode, device_of(device));
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_kquant_quantized_matmul: " << e.what()
              << std::endl;
    return nullptr;
  }
}

mlx_array *
mlx_test_fork_kquant_gather_qmm(mlx_array *x, mlx_array *w, mlx_array *scales,
                                mlx_array *biases, mlx_array *lhs_indices,
                                mlx_array *rhs_indices, bool transpose,
                                int group_size, int bits, const char *mode,
                                bool sorted_indices, int32_t device) {
  try {
    auto result = mlx::core::gather_qmm(
        ref(x), ref(w), ref(scales), opt(biases), opt(lhs_indices),
        opt(rhs_indices), transpose, group_size, bits, mode, sorted_indices,
        device_of(device));
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_kquant_gather_qmm: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_test_fork_kquant_dequantize(mlx_array *w, mlx_array *scales,
                                           mlx_array *biases, int group_size,
                                           int bits, int32_t out_dtype,
                                           const char *mode, int32_t device) {
  try {
    std::optional<mlx::core::Dtype> dtype = std::nullopt;
    if (out_dtype >= 0)
      dtype = to_mlx_dtype(out_dtype);
    auto result = mlx::core::dequantize(ref(w), ref(scales), opt(biases),
                                        group_size, bits, mode, std::nullopt,
                                        dtype, device_of(device));
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_kquant_dequantize: " << e.what() << std::endl;
    return nullptr;
  }
}

} // extern "C"
