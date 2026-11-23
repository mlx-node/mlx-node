#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

constexpr char gdn_inverse_source[] =
#include "cuda/gdn_inverse.cu.inc"
    ;

class GdnInvert64 : public UnaryPrimitive {
public:
  explicit GdnInvert64(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("GdnInvert64 requires CUDA");
  }
  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    auto a = cu::contiguous_inputs(inputs, stream(), encoder)[0];
    encoder.set_input_array(a);
    encoder.set_output_array(out);

    auto kernel = cu::jit_kernel(
        encoder.device(), gdn_inverse_source, "mlx_node_gdn_invert64");
    cu::KernelArgs args;
    args.append(a);
    args.append(out);
    encoder.add_kernel_node_raw(
        kernel, dim3(a.size() / (64 * 64)),
        dim3(256), {}, 0, args.args());
  }

  DEFINE_NAME(GdnInvert64)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()
};

} // namespace
} // namespace mlx::core
#endif

extern "C" mlx_array* mlx_gdn_invert64(mlx_array* a_handle) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!a_handle || default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& a = *reinterpret_cast<array*>(a_handle);
    if (a.dtype() != float32 || a.ndim() < 2 || a.shape(-1) != 64 ||
        a.shape(-2) != 64 || a.size() == 0) {
      throw std::invalid_argument("mlx_gdn_invert64 expects nonempty [..., 64, 64] f32 input");
    }
    auto stream = default_stream(Device::gpu);
    auto result = array(
        a.shape(), float32, std::make_shared<GdnInvert64>(stream), {a});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_gdn_invert64 error: " << e.what() << std::endl;
  }
#else
  (void)a_handle;
#endif
  return nullptr;
}
