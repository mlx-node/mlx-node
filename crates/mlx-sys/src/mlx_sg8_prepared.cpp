// Native SG8 prepared operands. Producers may emit this exact packing from
// their rounded BF16 output, skipping the redundant read/packing dispatch.
// Initial production scope: Q4_K T64 GDN output at M8,K4096,N2048.
#include "mlx_common.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_kquant.h"
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#endif

namespace mlx::core::sg8_prepared {
bool supported(const std::vector<array>& a) {
  return a.size() == 6 && a[0].shape() == Shape{1, 8, 4096} &&
      a[0].dtype() == bfloat16 && a[1].shape() == Shape{2048, 512} &&
      a[1].dtype() == uint32 && a[2].shape() == Shape{2048, 256} &&
      a[2].dtype() == uint8 && a[3].shape() == Shape{2048, 32} &&
      a[3].dtype() == float16 && a[4].shape() == Shape{4096 * 4} &&
      a[4].dtype() == uint32 && a[5].shape() == Shape{4096 / 32 * 8} &&
      a[5].dtype() == float32;
}
class PreparedSg8 final : public fast::Custom {
 public:
  explicit PreparedSg8(Stream stream)
      : Custom(stream, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("prepared SG8 requires Metal");
        }) {}
  DEFINE_NAME(PreparedSg8)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    if (!supported(a)) throw std::invalid_argument("prepared SG8 replay shape");
    return {Shape{1, 8, 2048}};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("prepared SG8 requires Metal");
  }
  void eval_gpu(const std::vector<array>& a, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (!supported(a)) throw std::invalid_argument("prepared SG8 replay contract");
    kquant::prepared_sg8_matmul_eval(a, outputs[0], stream());
#else
    throw std::runtime_error("prepared SG8 requires Metal");
#endif
  }
};
}
extern "C" int mlx_kquant_sg8_prepared(mlx_array* x, mlx_array* w,
    mlx_array* scales, mlx_array* biases, mlx_array* packed, mlx_array* sums,
    mlx_array** output) {
  if (output) *output = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    using namespace mlx::core;
    if (!x || !w || !scales || !biases || !packed || !sums || !output) return -1;
    if (detail::in_grad_tracing() || !(default_device() == Device::gpu) ||
        !is_available(Device::gpu) || metal::device(Device::gpu).get_architecture_gen() < 17)
      return 0;
    std::vector<array> a{*reinterpret_cast<array*>(x), *reinterpret_cast<array*>(w),
        *reinterpret_cast<array*>(scales), *reinterpret_cast<array*>(biases),
        *reinterpret_cast<array*>(packed), *reinterpret_cast<array*>(sums)};
    if (!sg8_prepared::supported(a)) return 0;
    auto result = array(Shape{1, 8, 2048}, bfloat16,
        std::make_shared<sg8_prepared::PreparedSg8>(to_stream(Device::gpu)), std::move(a));
    *output = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_kquant_sg8_prepared: %s\n", e.what()); return -1;
  }
#else
  (void)x; (void)w; (void)scales; (void)biases; (void)packed; (void)sums; return 0;
#endif
}
