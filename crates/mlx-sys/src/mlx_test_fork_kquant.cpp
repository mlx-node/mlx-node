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
} // namespace

extern "C" {

mlx_array *mlx_test_fork_kquant_quantized_matmul(mlx_array *x, mlx_array *w,
                                                 mlx_array *scales,
                                                 mlx_array *biases,
                                                 bool transpose, int group_size,
                                                 int bits, const char *mode) {
  try {
    auto result = mlx::core::quantized_matmul(
        *reinterpret_cast<array *>(x), *reinterpret_cast<array *>(w),
        *reinterpret_cast<array *>(scales), opt(biases), transpose, group_size,
        bits, mode);
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_kquant_quantized_matmul: " << e.what()
              << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_test_fork_kquant_gather_qmm(
    mlx_array *x, mlx_array *w, mlx_array *scales, mlx_array *biases,
    mlx_array *lhs_indices, mlx_array *rhs_indices, bool transpose,
    int group_size, int bits, const char *mode, bool sorted_indices) {
  try {
    auto result = mlx::core::gather_qmm(
        *reinterpret_cast<array *>(x), *reinterpret_cast<array *>(w),
        *reinterpret_cast<array *>(scales), opt(biases), opt(lhs_indices),
        opt(rhs_indices), transpose, group_size, bits, mode, sorted_indices);
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_kquant_gather_qmm: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_test_fork_kquant_dequantize(mlx_array *w, mlx_array *scales,
                                           mlx_array *biases, int group_size,
                                           int bits, int32_t out_dtype,
                                           const char *mode) {
  try {
    std::optional<mlx::core::Dtype> dtype = std::nullopt;
    if (out_dtype >= 0)
      dtype = to_mlx_dtype(out_dtype);
    auto result = mlx::core::dequantize(
        *reinterpret_cast<array *>(w), *reinterpret_cast<array *>(scales),
        opt(biases), group_size, bits, mode, std::nullopt, dtype);
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_kquant_dequantize: " << e.what() << std::endl;
    return nullptr;
  }
}

} // extern "C"
