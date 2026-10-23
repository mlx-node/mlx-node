// Opt-in Q6 logits head: preserve the existing M8 multiplication and expose
// its FP32 accumulator before the BF16 output boundary. Splash's vocabulary
// projection uses the same output policy. See THIRD_PARTY_NOTICES.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_kquant.h"
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#endif
#include <memory>

namespace mlx::core::head_f32 {
bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  const char* flag = std::getenv("MLX_DFLASH2_HEAD_FP32");
  return flag && std::strcmp(flag, "1") == 0 &&
      !detail::in_grad_tracing() && default_device() == Device::gpu &&
      is_available(Device::gpu) && metal::is_nax_available();
#else
  return false;
#endif
}
bool supported(const std::vector<array>& a) {
  if (a.size() != 4 || a[0].ndim() < 2 || a[1].ndim() != 2) return false;
  const int K = a[0].shape(-1), N = a[1].shape(0);
  if (K < 256 || K > (1 << 20) || K % 256 || N < 64 ||
      N > (1 << 20) || N % 64 || a[0].size() != size_t(8) * K)
    return false;
  // Keep native's M8 dispatch contract; AR, tail rows and vector-tier shapes
  // retain the existing path and output dtype.
  if (8 >= kquant::tiled_qmv_vector_limit(K, N)) return false;
  return a[0].dtype() == bfloat16 && a[1].dtype() == uint32 &&
      a[1].shape() == Shape{N, K * 6 / 32} &&
      a[2].dtype() == int8 && a[2].shape() == Shape{N, K / 16} &&
      a[3].dtype() == float16 && a[3].shape() == Shape{N, K / 256};
}
class Q6HeadF32 : public fast::Custom {
 public:
  explicit Q6HeadF32(Stream s)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("Q6 FP32 head requires Metal");
        }) {}
  DEFINE_NAME(Q6HeadF32)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    auto shape = a[0].shape(); shape.back() = a[1].shape(0); return {shape};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Q6 FP32 head requires Metal");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (!supported(inputs)) throw std::invalid_argument("Q6 FP32 head replay changed contract");
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    std::vector<array> a;
    for (const auto& input : inputs) {
      if (input.flags().row_contiguous && input.offset() % 16 == 0) a.push_back(input);
      else {
        array copy(input.shape(), input.dtype(), nullptr, {});
        copy.set_data(allocator::malloc(copy.nbytes()));
        copy_gpu(input, copy, CopyType::General, s);
        enc.add_temporary(copy);
        a.push_back(std::move(copy));
      }
    }
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    const uint32_t K = a[0].shape(-1), N = a[1].shape(0), M = 8;
    const uint32_t splits = kquant::tiled_projection_splits(N, K);
    array partials = out, counters = out;
    if (splits > 1) {
      partials = array({int(splits * M * N)}, float32, nullptr, {});
      partials.set_data(allocator::malloc(partials.nbytes()));
      enc.add_temporary(partials);
      counters = kquant::tiled_projection_counters(s, N / 64);
    }
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        d, "kquant_head_f32", "m8", "q6k_head_f32_m8_nax_t64"));
    for (unsigned i = 0; i < 3; ++i) enc.set_input_array(a[1 + i], i);
    enc.set_input_array(a[0], 3);
    enc.set_output_array(out, 4);
    enc.set_output_array(partials, 5);
    enc.set_output_array(counters, 6);
    enc.set_bytes(K, 7); enc.set_bytes(N, 8); enc.set_bytes(splits, 9); enc.set_bytes(M, 10);
    const uint32_t spread = kquant::tiled_projection_spread(s);
    enc.set_bytes(spread, 11);
    enc.dispatch_threadgroups(MTL::Size(N / 64, splits, 1), MTL::Size(64, 1, 1));
#else
    throw std::runtime_error("Q6 FP32 head requires Metal");
#endif
  }
};
}
extern "C" int mlx_kquant_head_f32(mlx_array* x, mlx_array* w, mlx_array* scales,
    mlx_array* biases, mlx_array** output) {
  if (output) *output = nullptr;
  try {
    if (!x || !w || !scales || !biases || !output) return -1;
    using namespace mlx::core;
    if (!head_f32::available()) return 0;
    std::vector<array> a{*reinterpret_cast<array*>(x), *reinterpret_cast<array*>(w),
        *reinterpret_cast<array*>(scales), *reinterpret_cast<array*>(biases)};
    if (!head_f32::supported(a)) return 0;
    auto shape = a[0].shape(); shape.back() = a[1].shape(0);
    auto result = array(shape, float32,
        std::make_shared<head_f32::Q6HeadF32>(to_stream(Device::gpu)), std::move(a));
    *output = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_kquant_head_f32: %s\n", e.what()); return -1;
  }
}
