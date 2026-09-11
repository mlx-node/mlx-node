#include "mlx_common.h"
#include "mlx/transforms_impl.h"

namespace {
const char* kAffineQmvBf16 =
#include "metal/affine_qmv_bf16.metal.inc"
;

const mlx::core::fast::CustomKernelFunction& affine_qmv_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "affine_qmv_bf16",
      {"X", "W", "S", "B", "K"}, {"Y"},
      kAffineQmvBf16, "", true, false);
  return kernel;
}
}

// Narrow inference-only entry point; general QMM and autodiff keep MLX's
// existing primitive. Errors never silently substitute an incompatible pack.
extern "C" mlx_array* mlx_affine_qmv_bf16(
    mlx_array* x, mlx_array* w, mlx_array* scales, mlx_array* biases) {
  try {
    using namespace mlx::core;
    if (!x || !w || !scales) {
      throw std::invalid_argument("affine_qmv_bf16 requires activation, weight and scales");
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(w);
    const auto& sa = *reinterpret_cast<array*>(scales);
    const auto& ba = biases ? *reinterpret_cast<array*>(biases) : sa;
    if (xa.ndim() < 1 || wa.ndim() != 2 || sa.ndim() != 2 || ba.ndim() != 2) {
      throw std::invalid_argument("affine_qmv_bf16 invalid rank");
    }
    const int K = xa.shape(-1);
    const int N = wa.shape(0);
    if (K == 0 || N == 0 || K % 32 || N % 8 || xa.size() != K ||
        xa.dtype() != bfloat16 || wa.dtype() != uint32 || wa.shape(1) != K / 8 ||
        sa.shape() != Shape{N, K / 32} || ba.shape() != sa.shape() ||
        (sa.dtype() != float16 && sa.dtype() != float32) || ba.dtype() != sa.dtype()) {
      throw std::invalid_argument("affine_qmv_bf16 requires BF16/Q4 group32, M=1, K%32=0, N%8=0");
    }
    // Keep MLX's differentiation transforms on their existing
    // primitive; this custom kernel is for inference decode.
    if (detail::in_grad_tracing()) {
      auto bias = biases ? ba : multiply(sa, array(-8.0f, sa.dtype()));
      auto output = astype(quantized_matmul(xa, wa, sa, bias, true, 32, 4), bfloat16);
      return reinterpret_cast<mlx_array*>(new array(std::move(output)));
    }
    auto shape = xa.shape();
    shape.back() = N;
    auto outputs = affine_qmv_kernel()(
        {xa, wa, sa, ba, array(K, int32)}, {shape}, {bfloat16},
        {64 * (N / 8), 1, 1}, {64, 1, 1},
        {{"SYMMETRIC", biases == nullptr}, {"HALF_SCALES", sa.dtype() == float16}},
        std::nullopt, false,
        default_stream(Device::gpu));
    return reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
  } catch (const std::exception& e) {
    std::cerr << "mlx_affine_qmv_bf16 error: " << e.what() << std::endl;
    return nullptr;
  }
}
