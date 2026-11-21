#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"

#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace mlx::core::affine_q4_few {
namespace {

constexpr char affine_q4_few_source[] =
#include "cuda/affine_q4_few.cu.inc"
    ;

class AffineQ4G64FewCuda : public UnaryPrimitive {
public:
  explicit AffineQ4G64FewCuda(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("AffineQ4G64FewCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
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
    if (M < 1 || M > 8 || N <= 0 || N % 16 != 0 || K < 1024 || K % 256 != 0) {
      throw std::invalid_argument("AffineQ4G64FewCuda: unsupported shape");
    }

    auto kernel = cu::jit_kernel(
        encoder.device(), affine_q4_few_source, "mlx_node_affine_q4_g64_few");
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
        kernel, dim3(N / 16), dim3(32, 8), {}, 0, args.args());
  }

  DEFINE_NAME(AffineQ4G64FewCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const AffineQ4G64FewCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

} // namespace
} // namespace mlx::core::affine_q4_few

extern "C" mlx_array* mlx_affine_q4_g64_few_cuda(
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
    const int N = (w.ndim() == 2) ? w.shape(0) : 0;
    if (M < 1 || M > 8 || K < 1024 || K % 256 != 0 ||
        w.ndim() != 2 || N <= 0 || N % 16 != 0 || w.dtype() != uint32 ||
        w.shape(1) != K / 8 ||
        scales.ndim() != 2 || scales.dtype() != bfloat16 ||
        scales.shape(0) != N || scales.shape(1) != K / 64 ||
        biases.ndim() != 2 || biases.dtype() != bfloat16 ||
        biases.shape(0) != N || biases.shape(1) != K / 64) {
      return nullptr;
    }
    auto shape = x.shape();
    shape.back() = N;
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<affine_q4_few::AffineQ4G64FewCuda>(
            default_stream(Device::gpu)),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_affine_q4_g64_few_cuda error: " << e.what() << std::endl;
  }
  return nullptr;
}

#else

extern "C" mlx_array* mlx_affine_q4_g64_few_cuda(
    mlx_array* x_handle, mlx_array* w_handle, mlx_array* scales_handle,
    mlx_array* biases_handle) {
  (void)x_handle;
  (void)w_handle;
  (void)scales_handle;
  (void)biases_handle;
  return nullptr;
}

#endif
