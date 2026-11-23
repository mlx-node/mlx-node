#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

constexpr char gdn_recurrent_source[] =
#include "cuda/gdn_recurrent.cu.inc"
    ;

constexpr char gdn_mma_source[] =
#include "cuda/gdn_mma.cu.inc"
    ;

constexpr int gdn_mma_shared_bytes = 38016;

class GdnRecurrent : public UnaryPrimitive {
public:
  explicit GdnRecurrent(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("GdnRecurrent requires CUDA");
  }
  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    auto owned = cu::contiguous_inputs(inputs, stream(), encoder);
    cu::KernelArgs args;
    for (const auto& input : owned) {
      encoder.set_input_array(input);
      args.append(input);
    }
    encoder.set_output_array(out);
    args.append(out);
    const auto& v = inputs[2];
    args.append<int32_t>(v.shape(1));
    args.append<int32_t>(v.shape(2));

    auto& device = encoder.device();
    const bool use_mma = v.shape(1) >= 64 &&
        device.compute_capability_major() >= 8 && v.shape(0) <= 65535 &&
        v.shape(2) <= 65535;
    if (use_mma) {
      auto kernel = cu::jit_kernel(
          device, gdn_mma_source, "mlx_node_gdn_recurrent_mma");
      encoder.add_kernel_node_raw(
          kernel,
          dim3(v.shape(3) / 32, v.shape(2), v.shape(0)), dim3(32, 8), {},
          gdn_mma_shared_bytes, args.args());
      return;
    }

    auto kernel = cu::jit_kernel(
        device, gdn_recurrent_source, "mlx_node_gdn_recurrent");
    encoder.add_kernel_node_raw(
        kernel,
        dim3(v.shape(2), v.shape(0), v.shape(3) / 16), dim3(32, 4), {}, 0,
        args.args());
  }

  DEFINE_NAME(GdnRecurrent)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{static_cast<int>(inputs[2].size() + inputs[5].size())}};
  }
};

} // namespace
} // namespace mlx::core
#endif

extern "C" mlx_array* mlx_gdn_recurrent(
    mlx_array* q_handle, mlx_array* k_handle, mlx_array* v_handle,
    mlx_array* g_handle, mlx_array* beta_handle, mlx_array* state_handle) {
#ifdef MLX_NODE_CUDA_ENABLED
  try {
    using namespace mlx::core;
    if (!q_handle || !k_handle || !v_handle || !g_handle || !beta_handle ||
        !state_handle || default_device() != Device::gpu) {
      return nullptr;
    }
    std::vector<array> inputs = {
        *reinterpret_cast<array*>(q_handle), *reinterpret_cast<array*>(k_handle),
        *reinterpret_cast<array*>(v_handle), *reinterpret_cast<array*>(g_handle),
        *reinterpret_cast<array*>(beta_handle), *reinterpret_cast<array*>(state_handle)};
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    const auto& g = inputs[3];
    const auto& beta = inputs[4];
    const auto& state = inputs[5];
    if (q.ndim() != 4 || k.shape() != q.shape() || v.ndim() != 4 ||
        g.ndim() != 3 || beta.shape() != g.shape() || state.ndim() != 4 ||
        q.shape(3) != 128 || v.shape(3) != 128 || v.shape(3) % 16 != 0 ||
        q.shape(0) != v.shape(0) || q.shape(1) != v.shape(1) ||
        q.shape(2) != v.shape(2) || g.shape(0) != v.shape(0) ||
        g.shape(1) != v.shape(1) || g.shape(2) != v.shape(2) ||
        state.shape(0) != v.shape(0) || state.shape(1) != v.shape(2) ||
        state.shape(2) != 128 || state.shape(3) != 128 || v.shape(1) <= 0) {
      throw std::invalid_argument("mlx_gdn_recurrent: incompatible GDN shapes");
    }
    for (const auto& input : inputs) {
      if (input.dtype() != float32) {
        throw std::invalid_argument("mlx_gdn_recurrent requires f32 inputs");
      }
    }
    auto stream = default_stream(Device::gpu);
    auto result = array(
        Shape{static_cast<int>(v.size() + state.size())}, float32,
        std::make_shared<GdnRecurrent>(stream), std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_gdn_recurrent error: " << e.what() << std::endl;
  }
#else
  (void)q_handle;
  (void)k_handle;
  (void)v_handle;
  (void)g_handle;
  (void)beta_handle;
  (void)state_handle;
#endif
  return nullptr;
}
