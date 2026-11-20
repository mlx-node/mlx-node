#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"

#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace mlx::core::gemm_few {
namespace {

constexpr char affine_q8_few_source[] =
#include "cuda/affine_q8_few.cu.inc"
    ;

constexpr char bf16_gemm_few_source[] =
#include "cuda/bf16_gemm_few.cu.inc"
    ;

class AffineQ8Few : public UnaryPrimitive {
public:
  explicit AffineQ8Few(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("AffineQ8Few requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<8>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);
    array scales = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    array biases = cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_output_array(out);

    const int M = x.shape(-2);
    const int K = x.shape(-1);
    const int N = w.shape(0);
    if (M < 2 || M > 8 || N <= 0 || K < 32 || K % 32 != 0) {
      throw std::invalid_argument("AffineQ8Few: unsupported replay shape");
    }
    const bool use_8w = (N <= 128);
    const int num_warps = use_8w ? 8 : 4;
    static constexpr const char* kQ8Entries[2] = {
        "mlx_node_affine_q8_few<4>",
        "mlx_node_affine_q8_few<8>",
    };
    auto kernel = cu::jit_kernel(
        encoder.device(), affine_q8_few_source, kQ8Entries[use_8w ? 1 : 0]);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(scales);
    args.append(biases);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        kernel, dim3(N), dim3(32, num_warps), {}, 0, args.args());
  }

  DEFINE_NAME(AffineQ8Few)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const AffineQ8Few*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

class Bf16GemmFew : public UnaryPrimitive {
public:
  explicit Bf16GemmFew(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Bf16GemmFew requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_output_array(out);

    const int K = x.shape(-1);
    const int M = static_cast<int>(x.size() / K);
    const int N = w.shape(0);
    if (M < 1 || M > 8 || N <= 0 || K < 8 || K % 8 != 0) {
      throw std::invalid_argument("Bf16GemmFew: unsupported replay shape");
    }
    static constexpr const char* kBf16Entries[8] = {
        "mlx_node_bf16_gemm_few<1>",
        "mlx_node_bf16_gemm_few<2>",
        "mlx_node_bf16_gemm_few<3>",
        "mlx_node_bf16_gemm_few<4>",
        "mlx_node_bf16_gemm_few<5>",
        "mlx_node_bf16_gemm_few<6>",
        "mlx_node_bf16_gemm_few<7>",
        "mlx_node_bf16_gemm_few<8>",
    };
    auto kernel = cu::jit_kernel(
        encoder.device(), bf16_gemm_few_source, kBf16Entries[M - 1]);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(out);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        kernel, dim3((N + 7) / 8), dim3(32, 4), {}, 0, args.args());
  }

  DEFINE_NAME(Bf16GemmFew)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Bf16GemmFew*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

} // namespace
} // namespace mlx::core::gemm_few

extern "C" mlx_array* mlx_affine_q8_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || !scales_handle || !biases_handle ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(x_handle), *reinterpret_cast<array*>(w_handle),
        *reinterpret_cast<array*>(scales_handle),
        *reinterpret_cast<array*>(biases_handle)};
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];
    if (x.dtype() != bfloat16 || !(x.ndim() == 2 || x.ndim() == 3) ||
        (x.ndim() == 3 && x.shape(0) != 1)) {
      return nullptr;
    }
    const int M = x.shape(-2);
    const int K = x.shape(-1);
    if (M < 2 || M > 8 || K < 32 || K % 32 != 0 ||
        w.ndim() != 2 || w.shape(0) <= 0 || w.dtype() != uint32 ||
        w.shape(1) != K / 4 ||
        scales.ndim() != 2 || scales.dtype() != float32 ||
        scales.shape(0) != w.shape(0) || scales.shape(1) != K / 32 ||
        biases.ndim() != 2 || biases.dtype() != float32 ||
        biases.shape(0) != w.shape(0) || biases.shape(1) != K / 32) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = w.shape(0);
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<gemm_few::AffineQ8Few>(default_stream(Device::gpu)),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_affine_q8_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

extern "C" mlx_array* mlx_bf16_gemm_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle) {
  try {
    using namespace mlx::core;
    if (!x_handle || !w_handle || default_device() != Device::gpu) {
      return nullptr;
    }
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(x_handle),
        *reinterpret_cast<array*>(w_handle)};
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    if (x.dtype() != bfloat16 || w.dtype() != bfloat16 ||
        x.ndim() < 2 || w.ndim() != 2) {
      return nullptr;
    }
    const int K = x.shape(-1);
    if (K < 8 || K % 8 != 0 || w.shape(1) != K || w.shape(0) <= 0) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(x.size()) / K;
    if (rows < 1 || rows > 8) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = w.shape(0);
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<gemm_few::Bf16GemmFew>(default_stream(Device::gpu)),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_bf16_gemm_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

#else

extern "C" mlx_array* mlx_affine_q8_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle) {
  (void)x_handle;
  (void)w_handle;
  (void)scales_handle;
  (void)biases_handle;
  return nullptr;
}

extern "C" mlx_array* mlx_bf16_gemm_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle) {
  (void)x_handle;
  (void)w_handle;
  return nullptr;
}

#endif
