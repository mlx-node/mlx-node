#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include <memory>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx_test_counters.h"
#endif

namespace mlx::core::attention_sg8 {
bool supported(const std::vector<array>& a) {
  if (a.size() != 2 || a[0].shape() != Shape{1, 8, 4096} ||
      a[1].shape() != Shape{1, 16, 8, 256}) return false;
  return a[0].dtype() == bfloat16 && a[1].dtype() == bfloat16 &&
      a[0].strides(2) == 1 && a[1].strides(3) == 1 &&
      a[0].strides(1) >= 0 && a[1].strides(1) >= 0 && a[1].strides(2) >= 0;
}
class AttentionSg8Prepare : public fast::Custom {
 public:
  explicit AttentionSg8Prepare(Stream s)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("attention SG8 preparation requires Metal");
        }) {}
  DEFINE_NAME(AttentionSg8Prepare)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>&) override {
    return {{1, 8, 4096}, {16384}, {1024}};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("attention SG8 preparation requires Metal");
  }
  void eval_gpu(const std::vector<array>& a, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (!supported(a)) throw std::invalid_argument("attention SG8 replay changed its contract");
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto* library = device.get_library("mlx_node_attention_sg8_prepare", [] {
      return std::string(metal::utils()) + "\n" +
#include "metal/common/attention_sg8_prepare.metal.inc"
        ;
    });
    auto* pipeline = device.get_kernel("mlx_node_attention_sg8_prepare", library);
    if (pipeline->threadExecutionWidth() != 32 || pipeline->maxTotalThreadsPerThreadgroup() < 128)
      throw std::runtime_error("attention SG8 preparation launch unsupported");
    auto& enc = metal::get_command_encoder(s);
    for (auto& out : outputs) out.set_data(allocator::malloc(out.nbytes()));
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_array(a[0], 0);
    enc.set_input_array(a[1], 1);
    for (int i = 0; i < 3; ++i) enc.set_output_array(outputs[i], i + 2);
    enc.set_vector_bytes(std::vector<int64_t>{a[0].strides(1), a[1].strides(1), a[1].strides(2)}, 5);
    bridge_testing::record("attention_gate_sg8_prepare");
    enc.dispatch_threadgroups(MTL::Size(32, 1, 1), MTL::Size(128, 1, 1));
#else
    throw std::runtime_error("attention SG8 preparation requires Metal");
#endif
  }
};
}

// 1 constructed, 0 unsupported (caller uses native gate+projection), -1 error.
extern "C" int mlx_attention_sg8_prepare(mlx_array* gate, mlx_array* attention,
    mlx_array** values, mlx_array** packed, mlx_array** sums) {
  if (values) *values = nullptr;
  if (packed) *packed = nullptr;
  if (sums) *sums = nullptr;
  if (!gate || !attention || !values || !packed || !sums) return -1;
  try {
    using namespace mlx::core;
#ifdef MLX_NODE_METAL_ENABLED
    if (detail::in_grad_tracing() || !(default_device() == Device::gpu) ||
        !is_available(Device::gpu) || !metal::is_nax_available()) return 0;
    std::vector<array> a{*reinterpret_cast<array*>(gate), *reinterpret_cast<array*>(attention)};
    if (!attention_sg8::supported(a)) return 0;
    auto result = array::make_arrays({{1,8,4096},{16384},{1024}},
        {bfloat16,uint32,float32},
        std::make_shared<attention_sg8::AttentionSg8Prepare>(to_stream(Device::gpu)),std::move(a));
    auto v = std::make_unique<array>(std::move(result[0]));
    auto p = std::make_unique<array>(std::move(result[1]));
    auto s = std::make_unique<array>(std::move(result[2]));
    *values = reinterpret_cast<mlx_array*>(v.release());
    *packed = reinterpret_cast<mlx_array*>(p.release());
    *sums = reinterpret_cast<mlx_array*>(s.release());
    return 1;
#else
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "attention SG8 preparation: " << e.what() << '\n';
    return -1;
  }
}
