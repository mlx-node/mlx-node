#include "mlx_common.h"
#include "mlx_portable_sdpa.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::quantized_preamble {
const char* utils();
const char* steel_attention();
} // namespace mlx::core::quantized_preamble
#endif

namespace mlx::core::fast {
#ifdef MLX_NODE_METAL_ENABLED
namespace {
// The vendored Steel kernel shares K/V scratch and keeps softmax online.
// BF16/FP16: 32 * (256+8) * 2 + (16+8) * 256 * 2 = 29,184 bytes,
// below the 32 KiB threadgroup limit on earlier Apple GPUs. No NAX
// instructions, sequence-sized score matrix, or per-query partition output.
constexpr int BQ = 32, BK = 16, BD = 256, WM = 4;

MTL::ComputePipelineState* pipeline(Dtype dtype, bool align_q, bool align_k) {
  auto& device = metal::device(Device::gpu);
  const std::string type = dtype == bfloat16 ? "bfloat16_t" : "half";
  const std::string name = "mlx_node_portable_d256_" + type;
  auto* library = device.get_library(name, [&]() {
    // Expand the linked vendor header at build time: this works in the
    // precompiled-metallib distribution too, without installed source files.
    return std::string(quantized_preamble::utils()) + quantized_preamble::steel_attention() +
        "\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype(attention<" +
        type + ",32,16,256,4,1," + type + ",float>) attention<" +
        type + ",32,16,256,4,1," + type + ",float>;\n";
  });
  const bool no = false, causal = true;
  metal::MTLFCList constants = {
      {&align_q, MTL::DataType::DataTypeBool, 200},
      {&align_k, MTL::DataType::DataTypeBool, 201},
      {&no, MTL::DataType::DataTypeBool, 300},
      {&causal, MTL::DataType::DataTypeBool, 301},
      {&no, MTL::DataType::DataTypeBool, 302}};
  auto* result = device.get_kernel(
      name, library, name + (align_q ? "_aq" : "_rq") +
                         (align_k ? "_ak" : "_rk"), constants);
  // Check the compiled pipeline, not a chip name or assumed core count.
  if (result->threadExecutionWidth() != 32 ||
      result->maxTotalThreadsPerThreadgroup() < 32 * WM ||
      result->staticThreadgroupMemoryLength() >
          device.mtl_device()->maxThreadgroupMemoryLength()) {
    throw std::runtime_error("portable D256 SDPA exceeds device pipeline limits");
  }
  return result;
}

class PortableD256SDPA : public Custom {
 public:
  PortableD256SDPA(Stream stream, float scale)
      : Custom(stream, [stream, scale](std::vector<array> inputs) {
          return std::vector<array>{scaled_dot_product_attention(
              inputs[0], inputs[1], inputs[2], scale, "causal",
              std::nullopt, std::nullopt, stream)};
        }), scale_(scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("PortableD256SDPA requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    const int ql = q.shape(2), kl = k.shape(2);
    auto* kernel = pipeline(q.dtype(), ql % BQ == 0, kl % BK == 0);
    mlx::steel::AttnParams params{
        q.shape(0), q.shape(1), BD, ql, kl, q.shape(1) / k.shape(1), scale_,
        1 + (ql - 1) / BQ, 1 + (kl - 1) / BK, ql / BQ, kl / BK,
        ql % BQ, kl % BK, kl - ql,
        {q.strides(0), q.strides(1), q.strides(2)},
        {k.strides(0), k.strides(1), k.strides(2)},
        {v.strides(0), v.strides(1), v.strides(2)},
        {out.strides(0), out.strides(1), out.strides(2)}};
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(params.NQ, params.H, params.B), MTL::Size(32, WM, 1));
  }

  DEFINE_NAME(PortableD256SDPA)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const PortableD256SDPA&>(other).scale_;
  }
 private:
  float scale_;
};
} // namespace
#endif

bool portable_d256_sdpa_available() {
#ifdef MLX_NODE_METAL_ENABLED
  if (!metal::is_available() || default_device().type != Device::gpu) return false;
  static const bool available = [] {
    const char* global = std::getenv("MLX_ENABLE_D256_FULL_SDPA");
    if (global && std::string(global) == "0") return false;
    const char* override = std::getenv("MLX_PORTABLE_D256_SDPA");
    if (override && std::string(override) == "0") return false;
    // Preserve the established NAX route on M5. Explicit force is for
    // testing this exact portable kernel on any Apple GPU, including M5.
    if ((!override || std::string(override) != "1") && metal::is_nax_available())
      return false;
    try {
      for (auto dtype : {float16, bfloat16})
        for (bool aq : {false, true})
          for (bool ak : {false, true}) pipeline(dtype, aq, ak);
      return true;
    } catch (const std::exception& error) {
      std::cerr << "[mlx] portable D256 SDPA unavailable: " << error.what() << '\n';
      return false;
    }
  }();
  return available;
#else
  return false;
#endif
}

std::optional<array> portable_d256_sdpa(
    const array& q, const array& k, const array& v, float scale) {
#ifdef MLX_NODE_METAL_ENABLED
  // Restrict the bridge to the inference geometry whose memory contract we
  // expose to the paged planner. Other dtypes/masks/broadcasts use stock MLX.
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
      detail::in_grad_tracing() || q.shape(2) <= 8 || q.shape(3) != BD || k.shape(3) != BD ||
      k.shape() != v.shape() || q.shape(0) != k.shape(0) ||
      q.shape(0) < 1 || q.shape(1) < 1 || k.shape(1) < 1 ||
      q.shape(1) % k.shape(1) != 0 || k.shape(2) < q.shape(2) ||
      (q.dtype() != bfloat16 && q.dtype() != float16) ||
      q.dtype() != k.dtype() || q.dtype() != v.dtype() ||
      q.strides(3) != 1 || k.strides(3) != 1 || v.strides(3) != 1 ||
      !portable_d256_sdpa_available()) return std::nullopt;
  return array(q.shape(), q.dtype(),
               std::make_shared<PortableD256SDPA>(to_stream({}), scale), {q, k, v});
#else
  (void)q; (void)k; (void)v; (void)scale;
  return std::nullopt;
#endif
}
} // namespace mlx::core::fast

extern "C" bool mlx_metal_portable_d256_sdpa_available() {
  try { return mlx::core::fast::portable_d256_sdpa_available(); }
  catch (...) { return false; }
}
