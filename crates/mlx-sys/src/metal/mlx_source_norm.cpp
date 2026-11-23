// Exact-width source arithmetic diagnostic; default models do not call this.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_test_counters.h"
#include <limits>
#include <memory>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#endif

namespace mlx::core::source_norm {
bool supported(const std::vector<array>& a, bool residual) {
  if (a.size() != (residual ? 3u : 2u) || a[0].ndim() < 1 ||
      a[0].shape(-1) != 2048 || a[0].dtype() != bfloat16 ||
      a[1].shape() != Shape{2048} || a[1].dtype() != float32 ||
      a[0].size() == 0 || a[0].size() / 2048 > std::numeric_limits<uint32_t>::max())
    return false;
  return !residual || (a[2].shape() == a[0].shape() && a[2].dtype() == bfloat16);
}
class SourceNorm : public fast::Custom {
 public:
  SourceNorm(Stream s, bool residual, bool staged)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("source RMS diagnostic does not support transforms");
        }), residual_(residual), staged_(staged) {}
  DEFINE_NAME(SourceNorm)
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const SourceNorm&>(other);
    return residual_ == rhs.residual_ && staged_ == rhs.staged_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    if (!supported(a, residual_)) throw std::invalid_argument("source RMS replay contract");
    return std::vector<Shape>(residual_ ? 2 : 1, a[0].shape());
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("source RMS diagnostic requires Metal");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (!supported(inputs, residual_)) throw std::invalid_argument("source RMS replay contract");
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    std::vector<array> a;
    a.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.flags().row_contiguous) a.push_back(input);
      else {
        array contiguous(input.shape(), input.dtype(), nullptr, {});
        contiguous.set_data(allocator::malloc(contiguous.nbytes()));
        copy_gpu(input, contiguous, CopyType::General, s);
        enc.add_temporary(contiguous);
        a.push_back(std::move(contiguous));
      }
    }
    for (auto& out : outputs) out.set_data(allocator::malloc(out.nbytes()));
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto* library = d.get_library("mlx_node_source_norm_2048_fast_v1", options, [] {
      return std::string(
#include "metal/common/source_norm.metal.inc"
      );
    });
    const size_t rows = a[0].size() / 2048;
    // Source chooses staged groups for <=64 rows. Current rows, not trace rows.
    const bool staged = staged_ && rows <= 64;
    const std::string name = std::string("source_norm_2048_") +
        (residual_ ? "add_" : "") + (staged ? "stage" : "stream");
    auto* kernel = d.get_kernel(name, library);
    const unsigned threads = staged ? 1024 : 256;
    if (kernel->threadExecutionWidth() != 32 || kernel->maxTotalThreadsPerThreadgroup() < threads)
      throw std::runtime_error("source RMS threadgroup unsupported");
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(a[0], 0);
    enc.set_input_array(a[1], 1);
    enc.set_input_array(residual_ ? a[2] : a[0], 2);
    enc.set_output_array(outputs[0], 3);
    // The standalone template does not access hidden; avoid allocating a dummy.
    enc.set_output_array(residual_ ? outputs[1] : outputs[0], 4);
    bridge_testing::record(residual_ ? "source_rms_add" : "source_rms");
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads, 1, 1));
#else
    throw std::runtime_error("source RMS diagnostic requires Metal");
#endif
  }
 private:
  bool residual_;
  bool staged_;
};
}

// 1 constructed; 0 declined (caller uses F32 reference); -1 invalid/error.
// A non-null residual requires a distinct out_hidden slot. No host readback.
extern "C" int mlx_source_norm_2048(mlx_array* input, mlx_array* weight,
    mlx_array* residual, bool staged, mlx_array** out_normalized, mlx_array** out_hidden) {
  if (out_normalized) *out_normalized = nullptr;
  if (out_hidden) *out_hidden = nullptr;
  try {
    if (!input || !weight || !out_normalized ||
        (out_hidden && out_hidden == out_normalized) || (residual && !out_hidden)) return -1;
    using namespace mlx::core;
#ifdef MLX_NODE_METAL_ENABLED
    if (detail::in_grad_tracing() || !(default_device() == Device::gpu) ||
        !is_available(Device::gpu) || metal::device(Device::gpu).get_architecture_gen() < 17)
      return 0;
    std::vector<array> a{*reinterpret_cast<array*>(input), *reinterpret_cast<array*>(weight)};
    if (residual) a.push_back(*reinterpret_cast<array*>(residual));
    if (!source_norm::supported(a, residual != nullptr)) return 0;
    auto shapes = std::vector<Shape>(residual ? 2 : 1, a[0].shape());
    auto values = array::make_arrays(shapes, std::vector<Dtype>(shapes.size(), bfloat16),
        std::make_shared<source_norm::SourceNorm>(to_stream(Device::gpu), residual != nullptr, staged),
        std::move(a));
    auto normalized = std::make_unique<array>(std::move(values[0]));
    auto hidden = residual ? std::make_unique<array>(std::move(values[1])) : nullptr;
    *out_normalized = reinterpret_cast<mlx_array*>(normalized.release());
    if (out_hidden) *out_hidden = reinterpret_cast<mlx_array*>(hidden.release());
    return 1;
#else
    (void)staged;
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "source RMS diagnostic: " << e.what() << '\n';
    return -1;
  }
}
