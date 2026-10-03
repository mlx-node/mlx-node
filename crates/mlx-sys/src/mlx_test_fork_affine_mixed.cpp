// TEST-ONLY oracle: the MLX fork's QuantizedMatmul built directly with BF16 x
// and F32 affine scales/biases (fork commit ce81e3b19). Only the fork pinned
// at 053e43fec computes this correctly; delete this file with the pin move.

#include "mlx/primitives.h"
#include "mlx_common.h"

using mlx::core::array;

extern "C" mlx_array *mlx_test_fork_affine_mixed_qmm(mlx_array *x, mlx_array *w,
                                                     mlx_array *scales,
                                                     mlx_array *biases,
                                                     int group_size, int bits) {
  try {
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    auto shape = xa.shape();
    shape.back() = wa.shape(0);
    return reinterpret_cast<mlx_array *>(new array(
        std::move(shape), mlx::core::bfloat16,
        std::make_shared<mlx::core::QuantizedMatmul>(
            mlx::core::default_stream(mlx::core::Device::gpu), group_size, bits,
            mlx::core::QuantizationMode::Affine, true),
        {xa, wa, *reinterpret_cast<array *>(scales),
         *reinterpret_cast<array *>(biases)}));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_fork_affine_mixed_qmm: " << e.what() << std::endl;
    return nullptr;
  }
}
