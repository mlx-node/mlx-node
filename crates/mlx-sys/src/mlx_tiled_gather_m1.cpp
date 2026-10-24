// A narrowly scoped T64 gather for the existing one-token MoE route.
// The GPU arithmetic/order is copied from baseline gather_qmv_fast.
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_common.h"
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx_paged_metallib.h"
#include "mlx_test_counters.h"
#endif
namespace mlx::core::tiled_gather_m1 {
bool supported(const std::vector<array>& a, int bits) {
  if (a.size() != 5 || (bits != 4 && bits != 5 && bits != 6)) return false;
  const auto& x = a[0];
  const auto& w = a[1];
  const auto& idx = a[4];
  if (x.ndim() != 4 || w.ndim() != 3 || idx.ndim() != 2 || x.shape(0) != 1 ||
      x.shape(2) != 1 || idx.shape(0) != 1 || idx.shape(1) < 1 || idx.shape(1) > 8 ||
      (x.shape(1) != 1 && x.shape(1) != idx.shape(1)))
    return false;
  const int K = x.shape(3), E = w.shape(0), N = w.shape(1);
  return K >= 512 && K % 512 == 0 && N >= 64 && N % 64 == 0 && E > 0 && E <= 256 &&
         x.dtype() == bfloat16 && w.dtype() == uint32 && w.shape(2) == K * bits / 32 &&
         a[2].shape() == Shape{E, N, K / 16} &&
         a[2].dtype() == (bits == 6 ? int8 : uint8) &&
         a[3].shape() == Shape{E, N, K / (bits == 6 ? 256 : 128)} &&
         a[3].dtype() == float16 && (idx.dtype() == int32 || idx.dtype() == uint32);
}
class TiledGatherM1 final : public fast::Custom {
 public:
  TiledGatherM1(Stream s, int bits)
      : Custom(s,
               [](std::vector<array>) -> std::vector<array> {
                 throw std::runtime_error("T64 M1 gather requires Metal");
               }),
        bits_(bits) {}
  DEFINE_NAME(TiledGatherM1)
  bool is_equivalent(const Primitive& other) const override {
    return bits_ == static_cast<const TiledGatherM1&>(other).bits_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    if (!supported(a, bits_)) throw std::invalid_argument("T64 M1 gather replay shape");
    return {{1, a[4].shape(1), 1, a[1].shape(1)}};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("T64 M1 gather requires Metal");
  }
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (!supported(inputs, bits_))
      throw std::invalid_argument("T64 M1 gather replay contract");
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    std::vector<array> a;
    a.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.flags().row_contiguous && input.offset() % 16 == 0)
        a.push_back(input);
      else {
        array copy(input.shape(), input.dtype(), nullptr, {});
        copy.set_data(allocator::malloc(copy.nbytes()));
        copy_gpu(input, copy, CopyType::General, s);
        enc.add_temporary(copy);
        a.push_back(std::move(copy));
      }
    }
    const std::string name = "q" + std::to_string(bits_) + "k_gather_t64_m1";
    auto* pipeline =
        fast::paged::get_prebuilt_kernel(d, "kquant", "T64 M1 gather", name);
    if (pipeline->threadExecutionWidth() != 32 ||
        pipeline->maxTotalThreadsPerThreadgroup() < 64)
      throw std::runtime_error("T64 M1 gather launch unsupported");
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_array(a[1], 0);
    enc.set_input_array(a[2], 1);
    enc.set_input_array(a[3], 2);
    enc.set_input_array(a[0], 3);
    enc.set_input_array(a[4], 4);
    enc.set_output_array(out, 5);
    enc.set_vector_bytes(
        std::vector<int>{a[0].shape(3), a[1].shape(1), a[1].shape(0), a[0].shape(1)},
        6);
    bridge_testing::record("gather_t64_m1");
    enc.dispatch_threadgroups(MTL::Size(1, a[1].shape(1) / 8, a[4].shape(1)),
                              MTL::Size(64, 1, 1));
#else
    throw std::runtime_error("T64 M1 gather requires Metal");
#endif
  }

 private:
  int bits_;
};
}  // namespace mlx::core::tiled_gather_m1
extern "C" int mlx_tiled_gather_qmv_m1(mlx_array* x, mlx_array* w, mlx_array* scales,
                                       mlx_array* biases, mlx_array* indices, int bits,
                                       mlx_array** output) {
  if (output) *output = nullptr;
  if (!x || !w || !scales || !biases || !indices || !output) return -1;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    using namespace mlx::core;
    if (detail::in_grad_tracing() || !(default_device() == Device::gpu) ||
        !is_available(Device::gpu))
      return 0;
    std::vector<array> a{*reinterpret_cast<array*>(x), *reinterpret_cast<array*>(w),
                         *reinterpret_cast<array*>(scales),
                         *reinterpret_cast<array*>(biases),
                         *reinterpret_cast<array*>(indices)};
    if (!tiled_gather_m1::supported(a, bits)) return 0;
    Shape shape{1, a[4].shape(1), 1, a[1].shape(1)};
    auto result = array(
        std::move(shape), bfloat16,
        std::make_shared<tiled_gather_m1::TiledGatherM1>(to_stream(Device::gpu), bits),
        std::move(a));
    *output = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "T64 M1 gather: " << e.what() << '\n';
    return -1;
  }
#else
  return 0;
#endif
}
