#include "cuda/qwen4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::qwen4_ops {
namespace {

using namespace mlx_node::qwen4_cuda;

constexpr char qwen4_ple_source[] =
#include "cuda/qwen4_ple.cu.inc"
    ;

// ============================================================================
// 5. PLE Layer-1 Fused Gate + Dilated Causal Conv1d
// ============================================================================
class Qwen4PleGateConvCuda final : public Primitive {
 public:
  Qwen4PleGateConvCuda(
      Stream stream, bool has_history, bool has_taps, bool emit_hist,
      bool emit_conv_in, float eps, float weight_offset, bool fuse_residual)
      : Primitive(stream),
        has_history_(has_history),
        has_taps_(has_taps),
        emit_hist_(emit_hist),
        emit_conv_in_(emit_conv_in),
        eps_(eps),
        weight_offset_(weight_offset),
        fuse_residual_(fuse_residual) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4PleGateConvCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array key = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array value =
        cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array norm_key_w =
        cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array norm_query_w =
        cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
    array norm_conv_w =
        cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    const bool conv_fp32 = (inputs[6].dtype() == float32);
    array conv_w = conv_fp32
        ? cu::ensure_row_contiguous_aligned<16>(inputs[6], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<8>(inputs[6], stream(), encoder);

    const int T = static_cast<int>(x.size() / kW);
    if (T < 1 || T > 65535) {
      throw std::invalid_argument("Qwen4PleGateConvCuda: unsupported T");
    }

    array gated_val = alloc_temp({T, kW}, bfloat16, encoder);
    auto& out_x = outputs[0];
    out_x.set_data(cu::malloc_async(out_x.nbytes(), encoder));
    size_t next_out = 1;
    std::optional<size_t> hist_out_idx;
    if (emit_hist_) {
      hist_out_idx = next_out++;
      auto& out_hist = outputs[*hist_out_idx];
      out_hist.set_data(cu::malloc_async(out_hist.nbytes(), encoder));
    }
    array conv_in = emit_conv_in_
        ? [&]() -> array {
            auto& out_ci = outputs[next_out++];
            out_ci.set_data(cu::malloc_async(out_ci.nbytes(), encoder));
            return out_ci;
          }()
        : alloc_temp({T, kW}, bfloat16, encoder);

    // Kernel 1: 3x per-stream RMSNorm + signed-sqrt sigmoid gate + value broadcast
    {
      encoder.set_input_array(x);
      encoder.set_input_array(key);
      encoder.set_input_array(value);
      encoder.set_input_array(norm_key_w);
      encoder.set_input_array(norm_query_w);
      encoder.set_input_array(norm_conv_w);
      encoder.set_output_array(gated_val);
      encoder.set_output_array(conv_in);
      const bool norm_fp32 = (norm_key_w.dtype() == float32);
      auto k_gate = cu::jit_kernel(
          encoder.device(), qwen4_ple_source,
          norm_fp32 ? "qwen4_ple_gate_norm_cuda<true>"
                    : "qwen4_ple_gate_norm_cuda<false>");
      cu::KernelArgs args;
      args.append(x);
      args.append(key);
      args.append(value);
      args.append(norm_key_w);
      args.append(norm_query_w);
      args.append(norm_conv_w);
      args.append(gated_val);
      args.append(conv_in);
      args.append<int32_t>(T);
      args.append<float>(eps_);
      args.append<float>(weight_offset_);
      encoder.add_kernel_node_raw(
          k_gate, dim3(kHc, T, 1), dim3(320, 1, 1), {}, 0, args.args());
    }

    // Kernel 2: Dilated causal depthwise conv1d (k=4, d=3) + SiLU + add + history
    {
      encoder.set_input_array(conv_in);
      encoder.set_input_array(gated_val);
      if (fuse_residual_) {
        encoder.set_input_array(x);
      }
      cu::KernelArgs args;
      args.append(conv_in);
      args.append(gated_val);
      args.append(x);
      size_t next_in = 7;
      if (has_history_) {
        array hist =
            cu::ensure_row_contiguous_aligned<2>(inputs[next_in++], stream(), encoder);
        encoder.set_input_array(hist);
        args.append(hist);
      } else {
        args.append<CUdeviceptr>(0);
      }
      encoder.set_input_array(conv_w);
      args.append(conv_w);
      if (has_taps_) {
        array taps =
            cu::ensure_row_contiguous_aligned<16>(inputs[next_in++], stream(), encoder);
        encoder.set_input_array(taps);
        args.append(taps);
      } else {
        args.append<CUdeviceptr>(0);
      }
      encoder.set_output_array(out_x);
      args.append(out_x);
      if (hist_out_idx.has_value()) {
        auto& out_hist = outputs[*hist_out_idx];
        encoder.set_output_array(out_hist);
        args.append(out_hist);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(T);

      static constexpr const char* kConvEntries[2][2][2] = {
          {{"qwen4_ple_conv_silu_add_cuda<false, false, false>",
            "qwen4_ple_conv_silu_add_cuda<false, false, true>"},
           {"qwen4_ple_conv_silu_add_cuda<false, true, false>",
            "qwen4_ple_conv_silu_add_cuda<false, true, true>"}},
          {{"qwen4_ple_conv_silu_add_cuda<true, false, false>",
            "qwen4_ple_conv_silu_add_cuda<true, false, true>"},
           {"qwen4_ple_conv_silu_add_cuda<true, true, false>",
            "qwen4_ple_conv_silu_add_cuda<true, true, true>"}},
      };
      auto k_conv = cu::jit_kernel(
          encoder.device(), qwen4_ple_source,
          kConvEntries[conv_fp32 ? 1 : 0][fuse_residual_ ? 1 : 0]
                      [has_taps_ ? 1 : 0]);
      const uint32_t grid_y = static_cast<uint32_t>(std::min(T, 32));
      encoder.add_kernel_node_raw(
          k_conv, dim3(40, grid_y, 1), dim3(256, 1, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Qwen4PleGateConvCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4PleGateConvCuda*>(&other);
    return rhs && rhs->has_history_ == has_history_ &&
        rhs->has_taps_ == has_taps_ && rhs->emit_hist_ == emit_hist_ &&
        rhs->emit_conv_in_ == emit_conv_in_ && rhs->eps_ == eps_ &&
        rhs->weight_offset_ == weight_offset_ &&
        rhs->fuse_residual_ == fuse_residual_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> out{inputs[0].shape()};
    if (emit_hist_) {
      out.push_back(Shape{kPleKeep, kW});
    }
    if (emit_conv_in_) {
      const int t = static_cast<int>(inputs[0].size() / kW);
      out.push_back(Shape{t, kW});
    }
    return out;
  }

 private:
  bool has_history_;
  bool has_taps_;
  bool emit_hist_;
  bool emit_conv_in_;
  float eps_;
  float weight_offset_;
  bool fuse_residual_;
};

}  // namespace
}  // namespace mlx::core::qwen4_ops

extern "C" bool mlx_qwen4_ple_gate_conv_cuda(
    mlx_array* x_h, mlx_array* key_h, mlx_array* value_h,
    mlx_array* norm_key_w_h, mlx_array* norm_query_w_h,
    mlx_array* norm_conv_w_h, mlx_array* conv_w_h, mlx_array* history_h,
    mlx_array* conv_taps_h, float eps, float weight_offset, bool fuse_residual,
    mlx_array** out_x, mlx_array** out_hist, mlx_array** out_conv_in) {
  if (out_x) *out_x = nullptr;
  if (out_hist) *out_hist = nullptr;
  if (out_conv_in) *out_conv_in = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!x_h || !key_h || !value_h || !norm_key_w_h || !norm_query_w_h ||
        !norm_conv_w_h || !conv_w_h || !out_x || (!out_hist && !out_conv_in) ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& key = *reinterpret_cast<array*>(key_h);
    const auto& value = *reinterpret_cast<array*>(value_h);
    const auto& norm_key_w = *reinterpret_cast<array*>(norm_key_w_h);
    const auto& norm_query_w = *reinterpret_cast<array*>(norm_query_w_h);
    const auto& norm_conv_w = *reinterpret_cast<array*>(norm_conv_w_h);
    const auto& conv_w = *reinterpret_cast<array*>(conv_w_h);
    if (x.dtype() != bfloat16 || x.ndim() < 2 || x.shape(-1) != kW ||
        key.dtype() != bfloat16 || value.dtype() != bfloat16) {
      return false;
    }
    const int64_t t64 = static_cast<int64_t>(x.size()) / kW;
    if (t64 < 1 || t64 > 65535 ||
        key.size() != static_cast<size_t>(t64) * kW ||
        value.size() != static_cast<size_t>(t64) * kH) {
      return false;
    }
    const Dtype n_dt = norm_key_w.dtype();
    if ((n_dt != float32 && n_dt != bfloat16) ||
        norm_query_w.dtype() != n_dt || norm_conv_w.dtype() != n_dt ||
        norm_key_w.size() != static_cast<size_t>(kW) ||
        norm_query_w.size() != static_cast<size_t>(kW) ||
        norm_conv_w.size() != static_cast<size_t>(kW)) {
      return false;
    }
    if ((conv_w.dtype() != float32 && conv_w.dtype() != bfloat16) ||
        conv_w.size() != static_cast<size_t>(kW) * 4) {
      return false;
    }

    std::vector<array> inputs{
        x, key, value, norm_key_w, norm_query_w, norm_conv_w, conv_w};
    const bool has_history = (history_h != nullptr);
    if (has_history) {
      const auto& hist = *reinterpret_cast<array*>(history_h);
      if (hist.dtype() != bfloat16 ||
          hist.size() != static_cast<size_t>(kPleKeep) * kW) {
        return false;
      }
      inputs.push_back(hist);
    }
    const bool has_taps = (conv_taps_h != nullptr);
    if (has_taps) {
      const auto& taps = *reinterpret_cast<array*>(conv_taps_h);
      if (taps.dtype() != int32 ||
          taps.size() != static_cast<size_t>(t64) * 4) {
        return false;
      }
      inputs.push_back(taps);
    }

    const bool emit_hist = (out_hist != nullptr);
    const bool emit_conv_in = (out_conv_in != nullptr);
    std::vector<Shape> shapes{x.shape()};
    std::vector<Dtype> dtypes{bfloat16};
    if (emit_hist) {
      shapes.push_back(Shape{kPleKeep, kW});
      dtypes.push_back(bfloat16);
    }
    if (emit_conv_in) {
      shapes.push_back(Shape{static_cast<int>(t64), kW});
      dtypes.push_back(bfloat16);
    }

    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        std::move(shapes), std::move(dtypes),
        std::make_shared<Qwen4PleGateConvCuda>(
            stream, has_history, has_taps, emit_hist, emit_conv_in, eps,
            weight_offset, fuse_residual),
        std::move(inputs));
    size_t idx = 0;
    *out_x = reinterpret_cast<mlx_array*>(new array(std::move(results[idx++])));
    if (emit_hist) {
      *out_hist =
          reinterpret_cast<mlx_array*>(new array(std::move(results[idx++])));
    }
    if (emit_conv_in) {
      *out_conv_in =
          reinterpret_cast<mlx_array*>(new array(std::move(results[idx++])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_ple_gate_conv_cuda: %s\n", e.what());
    return false;
  }
}

#else

extern "C" bool mlx_qwen4_ple_gate_conv_cuda(
    mlx_array* x_h, mlx_array* key_h, mlx_array* value_h,
    mlx_array* norm_key_w_h, mlx_array* norm_query_w_h,
    mlx_array* norm_conv_w_h, mlx_array* conv_w_h, mlx_array* history_h,
    mlx_array* conv_taps_h, float eps, float weight_offset, bool fuse_residual,
    mlx_array** out_x, mlx_array** out_hist, mlx_array** out_conv_in) {
  (void)x_h; (void)key_h; (void)value_h; (void)norm_key_w_h;
  (void)norm_query_w_h; (void)norm_conv_w_h; (void)conv_w_h;
  (void)history_h; (void)conv_taps_h; (void)eps; (void)weight_offset;
  (void)fuse_residual;
  if (out_x) *out_x = nullptr;
  if (out_hist) *out_hist = nullptr;
  if (out_conv_in) *out_conv_in = nullptr;
  return false;
}

#endif
