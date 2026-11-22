#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace {

constexpr char dflash2_conv_cuda_source[] =
#include "cuda/dflash2_conv.cu.inc"
    ;

class DFlash2ConvCuda final : public mlx::core::UnaryPrimitive {
 public:
  DFlash2ConvCuda(mlx::core::Stream stream, int32_t side)
      : mlx::core::UnaryPrimitive(stream), side_(side) {}

  void eval_cpu(const std::vector<mlx::core::array>&, mlx::core::array&) override {
    throw std::runtime_error("DFlash2ConvCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      mlx::core::array& out) override {
    using namespace mlx::core;
    auto& encoder = cu::get_command_encoder(stream());
    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    const auto& x = owned[0];
    const auto& dyn = owned[1];
    const auto& base = owned[2];
    const int B = x.shape(0);
    const int L = x.shape(1);
    const int H = x.shape(2);
    const int K = base.shape(1);
    const int G = dyn.shape(4);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_input_array(dyn);
    encoder.set_input_array(base);
    encoder.set_output_array(out);

    const char* entry = nullptr;
    if (x.dtype() == bfloat16) {
      entry = "mlx_dflash2_conv_cuda_kernel<nv_bfloat16>";
    } else if (x.dtype() == float16) {
      entry = "mlx_dflash2_conv_cuda_kernel<__half>";
    } else if (x.dtype() == float32) {
      entry = "mlx_dflash2_conv_cuda_kernel<float>";
    } else {
      throw std::invalid_argument("DFlash2ConvCuda: unsupported dtype");
    }
    auto kernel = cu::jit_kernel(encoder.device(), dflash2_conv_cuda_source, entry);
    cu::KernelArgs args;
    args.append(x);
    args.append(dyn);
    args.append(base);
    args.append(out);
    args.append<int32_t>(B);
    args.append<int32_t>(L);
    args.append<int32_t>(H);
    args.append<int32_t>(K);
    args.append<int32_t>(G);
    args.append<int32_t>(side_);
    encoder.add_kernel_node_raw(
        kernel, dim3((H + 255) / 256, L, B), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(DFlash2ConvCuda)
  bool is_equivalent(const mlx::core::Primitive& other) const override {
    const auto* rhs = dynamic_cast<const DFlash2ConvCuda*>(&other);
    return rhs && rhs->side_ == side_;
  }
  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int32_t side_;
};

}  // namespace
#endif

namespace mlx_node_internal {

#ifdef MLX_NODE_CUDA_ENABLED
std::optional<mlx::core::array> dflash2_conv_cuda(
    const mlx::core::array& x_arr,
    const mlx::core::array& dyn_arr,
    const mlx::core::array& base_arr,
    int32_t side) {
  if (mlx::core::default_device() == mlx::core::Device::gpu &&
      (x_arr.dtype() == mlx::core::bfloat16 ||
       x_arr.dtype() == mlx::core::float16 ||
       x_arr.dtype() == mlx::core::float32)) {
    return mlx::core::array(
        x_arr.shape(),
        x_arr.dtype(),
        std::make_shared<DFlash2ConvCuda>(
            mlx::core::default_stream(mlx::core::Device::gpu), side),
        {x_arr, dyn_arr, base_arr});
  }
  return std::nullopt;
}
#else
std::optional<mlx::core::array> dflash2_conv_cuda(
    const mlx::core::array&,
    const mlx::core::array&,
    const mlx::core::array&,
    int32_t) {
  return std::nullopt;
}
#endif

}  // namespace mlx_node_internal
