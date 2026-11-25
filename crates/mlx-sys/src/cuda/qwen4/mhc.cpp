#include "cuda/qwen4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::qwen4_ops {
namespace {

using namespace mlx_node::qwen4_cuda;

constexpr char qwen4_mhc_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/qwen4_mhc.cu.inc"
    ;

// ============================================================================
// 1. mHC Inject + Per-Stream RMSNorm (or Inject-Only)
// ============================================================================
class Qwen4InjectNormCuda final : public Primitive {
 public:
  Qwen4InjectNormCuda(
      Stream stream, bool has_inject, bool has_norm, float eps,
      float weight_offset)
      : Primitive(stream),
        has_inject_(has_inject),
        has_norm_(has_norm),
        eps_(eps),
        weight_offset_(weight_offset) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4InjectNormCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    const int M = static_cast<int>(x.size() / kW);
    if (M < 1 || M > 65535) {
      throw std::invalid_argument("Qwen4InjectNormCuda: unsupported M");
    }

    if (has_inject_ && !has_norm_) {
      array branch =
          cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
      array gate =
          cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
      auto& out_x = outputs[0];
      out_x.set_data(cu::malloc_async(out_x.nbytes(), encoder));
      encoder.set_input_array(x);
      encoder.set_input_array(branch);
      encoder.set_input_array(gate);
      encoder.set_output_array(out_x);
      cu::KernelArgs args;
      args.append(x);
      args.append(branch);
      args.append(gate);
      args.append(out_x);
      args.append<int32_t>(M);
      auto fn = cu::jit_kernel(
          encoder.device(), qwen4_mhc_source, "qwen4_inject_only_cuda");
      encoder.add_kernel_node_raw(
          fn, dim3(kHc, M, 1), dim3(320, 1, 1), {}, 0, args.args());
      return;
    }

    cu::KernelArgs args;
    encoder.set_input_array(x);
    args.append(x);

    array norm_w = has_inject_
        ? cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);

    if (has_inject_) {
      array branch =
          cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
      array gate =
          cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
      encoder.set_input_array(branch);
      encoder.set_input_array(gate);
      args.append(branch);
      args.append(gate);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }

    encoder.set_input_array(norm_w);
    args.append(norm_w);

    if (has_inject_) {
      auto& out_x = outputs[0];
      auto& out_normed = outputs[1];
      out_x.set_data(cu::malloc_async(out_x.nbytes(), encoder));
      out_normed.set_data(cu::malloc_async(out_normed.nbytes(), encoder));
      encoder.set_output_array(out_x);
      encoder.set_output_array(out_normed);
      args.append(out_x);
      args.append(out_normed);
    } else {
      auto& out_normed = outputs[0];
      out_normed.set_data(cu::malloc_async(out_normed.nbytes(), encoder));
      encoder.set_output_array(out_normed);
      args.append<CUdeviceptr>(0);
      args.append(out_normed);
    }

    args.append<int32_t>(M);
    args.append<float>(eps_);
    args.append<float>(weight_offset_);

    static constexpr const char* kEntries[2][2] = {
        {"qwen4_inject_norm_cuda<false, false>",
         "qwen4_inject_norm_cuda<false, true>"},
        {"qwen4_inject_norm_cuda<true, false>",
         "qwen4_inject_norm_cuda<true, true>"},
    };
    const bool w_fp32 = (norm_w.dtype() == float32);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source, kEntries[has_inject_ ? 1 : 0][w_fp32 ? 1 : 0]);
    encoder.add_kernel_node_raw(
        fn, dim3(kHc, M, 1), dim3(320, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Qwen4InjectNormCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4InjectNormCuda*>(&other);
    return rhs && rhs->has_inject_ == has_inject_ &&
        rhs->has_norm_ == has_norm_ && rhs->eps_ == eps_ &&
        rhs->weight_offset_ == weight_offset_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    if (has_inject_ && has_norm_) {
      return {inputs[0].shape(), inputs[0].shape()};
    }
    return {inputs[0].shape()};
  }

 private:
  bool has_inject_;
  bool has_norm_;
  float eps_;
  float weight_offset_;
};

// ============================================================================
// 2. mHC Down + SiLU (+ Optional Inject Gate)
// ============================================================================
class Qwen4HcDownActCuda final : public Primitive {
 public:
  Qwen4HcDownActCuda(Stream stream, int inj_mode)
      : Primitive(stream), inj_mode_(inj_mode) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4HcDownActCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array normed =
        cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w_down =
        cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array s_down =
        cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array b_down =
        cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);

    const int M = static_cast<int>(normed.size() / kW);
    if (M < 1 || M > 65535 || inj_mode_ < 0 || inj_mode_ > 4) {
      throw std::invalid_argument("Qwen4HcDownActCuda: unsupported replay shape");
    }

    encoder.set_input_array(normed);
    encoder.set_input_array(w_down);
    encoder.set_input_array(s_down);
    if (M <= 16) {
      encoder.set_input_array(b_down);
    }

    cu::KernelArgs args;
    args.append(normed);
    args.append(w_down);
    args.append(s_down);
    if (M <= 16) {
      args.append(b_down);
    }

    if (inj_mode_ == 1) {
      array w_inj =
          cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
      array s_inj =
          cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
      array b_inj =
          cu::ensure_row_contiguous_aligned<16>(inputs[6], stream(), encoder);
      encoder.set_input_array(w_inj);
      encoder.set_input_array(s_inj);
      encoder.set_input_array(b_inj);
      args.append(w_inj);
      args.append(s_inj);
      args.append(b_inj);
    } else if (inj_mode_ >= 2) {
      array w_inj =
          cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
      encoder.set_input_array(w_inj);
      args.append(w_inj);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }

    auto& out_down = outputs[0];
    out_down.set_data(cu::malloc_async(out_down.nbytes(), encoder));
    encoder.set_output_array(out_down);
    args.append(out_down);

    if (inj_mode_ != 0) {
      auto& out_inj = outputs[1];
      out_inj.set_data(cu::malloc_async(out_inj.nbytes(), encoder));
      encoder.set_output_array(out_inj);
      args.append(out_inj);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(M);

    if (M == 1) {
      static constexpr const char* kU4Entries[5] = {
          "qwen4_hc_down_act_u4_cuda<1, 0>",
          "qwen4_hc_down_act_u4_cuda<1, 1>",
          "qwen4_hc_down_act_u4_cuda<1, 2>",
          "qwen4_hc_down_act_u4_cuda<1, 3>",
          "qwen4_hc_down_act_u4_cuda<1, 4>",
      };
      auto fn = cu::jit_kernel(
          encoder.device(), qwen4_mhc_source, kU4Entries[inj_mode_]);
      const uint32_t grid_x = (inj_mode_ != 0) ? 324u : 320u;
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), dim3(32, 4, 1), {}, 0, args.args());
    } else if (M <= 16) {
      static constexpr const char* kMmaEntries[5] = {
          "qwen4_hc_down_act_mma_n8_cuda<0>",
          "qwen4_hc_down_act_mma_n8_cuda<1>",
          "qwen4_hc_down_act_mma_n8_cuda<2>",
          "qwen4_hc_down_act_mma_n8_cuda<3>",
          "qwen4_hc_down_act_mma_n8_cuda<4>",
      };
      auto fn = cu::jit_kernel(
          encoder.device(), qwen4_mhc_source, kMmaEntries[inj_mode_]);
      const uint32_t grid_x = (inj_mode_ != 0) ? 41u : 40u;
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), dim3(32, 8, 1), {}, 0, args.args());
    } else {
      static constexpr const char* kPrefillEntries[5] = {
          "qwen4_hc_down_act_prefill<0>",
          "qwen4_hc_down_act_prefill<1>",
          "qwen4_hc_down_act_prefill<2>",
          "qwen4_hc_down_act_prefill<3>",
          "qwen4_hc_down_act_prefill<4>",
      };
      auto fn = cu::jit_kernel(
          encoder.device(), qwen4_q8_prefill_source(), kPrefillEntries[inj_mode_]);
      const uint32_t grid_x = static_cast<uint32_t>((M + 31) / 32);
      encoder.add_kernel_node_raw(
          fn, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Qwen4HcDownActCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4HcDownActCuda*>(&other);
    return rhs && rhs->inj_mode_ == inj_mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape down_shape = inputs[0].shape();
    down_shape.back() = kRank;
    if (inj_mode_ != 0) {
      Shape inj_shape = inputs[0].shape();
      inj_shape.back() = kHc;
      return {std::move(down_shape), std::move(inj_shape)};
    }
    return {std::move(down_shape)};
  }

 private:
  int inj_mode_;
};

// ============================================================================
// 3. mHC Up-Projection + Sigmoid * Normed + 4-Stream Mean
// ============================================================================
class Qwen4HcUpMixCuda final : public UnaryPrimitive {
 public:
  explicit Qwen4HcUpMixCuda(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Qwen4HcUpMixCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array down_act =
        cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array normed =
        cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array w_up =
        cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array s_up =
        cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array b_up =
        cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);

    const int M = static_cast<int>(down_act.size() / kRank);
    if (M < 1 || (M + 15) / 16 > 65535) {
      throw std::invalid_argument("Qwen4HcUpMixCuda: unsupported M");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(down_act);
    encoder.set_input_array(normed);
    encoder.set_input_array(w_up);
    encoder.set_input_array(s_up);
    encoder.set_input_array(b_up);
    encoder.set_output_array(out);

    if (M == 1) {
      cu::KernelArgs args;
      args.append(down_act);
      args.append(w_up);
      args.append(s_up);
      args.append(b_up);
      args.append(normed);
      args.append(out);
      auto fn = cu::jit_kernel(
          encoder.device(), qwen4_mhc_source, "qwen4_hc_up_mix_u4_m1_cuda");
      encoder.add_kernel_node_raw(
          fn, dim3(320, 1, 1), dim3(32, 4, 1), {}, 0, args.args());
      return;
    }

    // Note kernel parameter order: (down_act, w_up, sc_up, bi_up, normed, mixed, M)
    cu::KernelArgs args;
    args.append(down_act);
    args.append(w_up);
    args.append(s_up);
    args.append(b_up);
    args.append(normed);
    args.append(out);
    args.append<int32_t>(M);

    const bool two_tiles = (M > 8);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source,
        two_tiles ? "qwen4_hc_up_mix_mma_cuda<2>" : "qwen4_hc_up_mix_mma_cuda<1>");
    const uint32_t grid_y = two_tiles ? static_cast<uint32_t>((M + 15) / 16) : 1u;
    encoder.add_kernel_node_raw(
        fn, dim3(160, grid_y, 1), dim3(32, 4, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Qwen4HcUpMixCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Qwen4HcUpMixCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = kH;
    return {std::move(s)};
  }
};

// ============================================================================
// 6. Router Softmax + Top-K + Normalize (1-launch, E <= 512, top_k <= 16)
// ============================================================================
class Qwen4RouterTopkCuda final : public Primitive {
 public:
  Qwen4RouterTopkCuda(Stream stream, int num_experts, int top_k)
      : Primitive(stream), num_experts_(num_experts), top_k_(top_k) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4RouterTopkCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array logits =
        cu::ensure_row_contiguous_aligned<4>(inputs[0], stream(), encoder);
    const int M = static_cast<int>(logits.size() / static_cast<size_t>(num_experts_));
    if (M < 1 || M > 65535 || num_experts_ < 1 || num_experts_ > 512 ||
        top_k_ < 1 || top_k_ > 16 || top_k_ > num_experts_) {
      throw std::invalid_argument("Qwen4RouterTopkCuda: unsupported replay shape");
    }

    auto& out_ids = outputs[0];
    auto& out_scores = outputs[1];
    out_ids.set_data(cu::malloc_async(out_ids.nbytes(), encoder));
    out_scores.set_data(cu::malloc_async(out_scores.nbytes(), encoder));

    encoder.set_input_array(logits);
    encoder.set_output_array(out_ids);
    encoder.set_output_array(out_scores);

    const bool is_fp32 = (logits.dtype() == float32);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source,
        is_fp32 ? "qwen4_router_topk_cuda<true>"
                : "qwen4_router_topk_cuda<false>");
    cu::KernelArgs args;
    args.append(logits);
    args.append(out_ids);
    args.append(out_scores);
    args.append<int32_t>(M);
    args.append<int32_t>(num_experts_);
    args.append<int32_t>(top_k_);
    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(M), 1, 1), dim3(256, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Qwen4RouterTopkCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4RouterTopkCuda*>(&other);
    return rhs && rhs->num_experts_ == num_experts_ && rhs->top_k_ == top_k_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = top_k_;
    return {s, s};
  }

 private:
  int num_experts_;
  int top_k_;
};

// ============================================================================
// 7. Prefill mHC Mix (`sigmoid(up) * normed` + 4-stream mean, W=10240 -> H=2560)
// ============================================================================
class Qwen4PrefillHcMixCuda final : public Primitive {
 public:
  explicit Qwen4PrefillHcMixCuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4PrefillHcMixCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array up = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array normed =
        cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    const int M = static_cast<int>(up.size() / kW);
    if (M < 1 || M > 65535 || normed.size() != up.size()) {
      throw std::invalid_argument("Qwen4PrefillHcMixCuda: unsupported shape");
    }
    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(up);
    encoder.set_input_array(normed);
    encoder.set_output_array(out);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source, "qwen4_prefill_hc_mix_cuda");
    cu::KernelArgs args;
    args.append(up);
    args.append(normed);
    args.append(out);
    args.append<int32_t>(M);
    encoder.add_kernel_node_raw(
        fn, dim3(1, M, 1), dim3(320, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Qwen4PrefillHcMixCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Qwen4PrefillHcMixCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = kH;
    return {std::move(s)};
  }
};

class Qwen4StreamMeanCuda final : public Primitive {
 public:
  explicit Qwen4StreamMeanCuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4StreamMeanCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    const int M = static_cast<int>(x.size() / kW);
    if (M < 1 || M > 65535) {
      throw std::invalid_argument("Qwen4StreamMeanCuda: unsupported shape");
    }
    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_output_array(out);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source, "qwen4_stream_mean_cuda");
    cu::KernelArgs args;
    args.append(x);
    args.append(out);
    args.append<int32_t>(M);
    encoder.add_kernel_node_raw(
        fn, dim3(1, M, 1), dim3(320, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Qwen4StreamMeanCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Qwen4StreamMeanCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = kH;
    return {std::move(s)};
  }
};

class Qwen4DequantQ8Bf16Cuda final : public Primitive {
 public:
  Qwen4DequantQ8Bf16Cuda(Stream stream, bool symmetric)
      : Primitive(stream), symmetric_(symmetric) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4DequantQ8Bf16Cuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    if ((symmetric_ && inputs.size() < 2) || (!symmetric_ && inputs.size() < 3)) {
      throw std::invalid_argument("Qwen4DequantQ8Bf16Cuda: missing inputs");
    }
    const int sb_dtype =
        (inputs[1].dtype() == float16) ? 0
                                       : (inputs[1].dtype() == bfloat16 ? 1 : 2);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array sc = (sb_dtype == 2)
        ? cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<2>(inputs[1], stream(), encoder);
    std::optional<array> bi;
    if (!symmetric_) {
      bi = (sb_dtype == 2)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    }
    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    const uint32_t grid_k = static_cast<uint32_t>(((K >> 3) + 255) / 256);
    if (N <= 0 || K < 32 || (K & 31) != 0 || grid_k == 0 || grid_k > 65535 ||
        (!symmetric_ && sc.dtype() != bi->dtype())) {
      throw std::invalid_argument("Qwen4DequantQ8Bf16Cuda: unsupported shape");
    }
    static constexpr const char* kEntries[2][3] = {
        {
            "qwen4_dequant_q8_bf16_cuda<0, false>",
            "qwen4_dequant_q8_bf16_cuda<1, false>",
            "qwen4_dequant_q8_bf16_cuda<2, false>",
        },
        {
            "qwen4_dequant_q8_bf16_cuda<0, true>",
            "qwen4_dequant_q8_bf16_cuda<1, true>",
            "qwen4_dequant_q8_bf16_cuda<2, true>",
        },
    };
    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(w);
    encoder.set_input_array(sc);
    if (!symmetric_) {
      encoder.set_input_array(*bi);
    }
    encoder.set_output_array(out);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source,
        kEntries[symmetric_ ? 1 : 0][sb_dtype]);
    cu::KernelArgs args;
    args.append(w);
    args.append(sc);
    if (!symmetric_) {
      args.append(*bi);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(N), grid_k, 1), dim3(256, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Qwen4DequantQ8Bf16Cuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4DequantQ8Bf16Cuda*>(&other);
    return rhs && rhs->symmetric_ == symmetric_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), inputs[0].shape(1) * 4}};
  }

 private:
  bool symmetric_;
};

}  // namespace
}  // namespace mlx::core::qwen4_ops

extern "C" bool mlx_qwen4_inject_norm_cuda(
    mlx_array* x_h, mlx_array* branch_h, mlx_array* gate_h,
    mlx_array* norm_w_h, float eps, float weight_offset,
    mlx_array** out_x, mlx_array** out_normed) {
  if (out_x) *out_x = nullptr;
  if (out_normed) *out_normed = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!x_h || default_device() != Device::gpu) {
      return false;
    }
    const bool has_inject = (branch_h != nullptr) || (gate_h != nullptr);
    const bool has_norm = (norm_w_h != nullptr) || (out_normed != nullptr);
    if (!has_inject && !has_norm) {
      return false;
    }
    if (has_inject && (!branch_h || !gate_h || !out_x)) {
      return false;
    }
    if (has_norm && (!norm_w_h || !out_normed)) {
      return false;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    if (x.dtype() != bfloat16 || x.ndim() < 2 || x.shape(-1) != kW) {
      return false;
    }
    const int64_t m64 = static_cast<int64_t>(x.size()) / kW;
    if (m64 < 1 || m64 > 65535) {
      return false;
    }
    auto stream = default_stream(Device::gpu);
    if (has_inject && !has_norm) {
      const auto& branch = *reinterpret_cast<array*>(branch_h);
      const auto& gate = *reinterpret_cast<array*>(gate_h);
      if (branch.dtype() != bfloat16 ||
          branch.size() != static_cast<size_t>(m64) * kH ||
          gate.dtype() != bfloat16 ||
          gate.size() != static_cast<size_t>(m64) * kHc) {
        return false;
      }
      auto results = array::make_arrays(
          {x.shape()}, {bfloat16},
          std::make_shared<Qwen4InjectNormCuda>(stream, true, false, 0.0f, 0.0f),
          {x, branch, gate});
      *out_x = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      return true;
    }
    const auto& norm_w = *reinterpret_cast<array*>(norm_w_h);
    if ((norm_w.dtype() != float32 && norm_w.dtype() != bfloat16) ||
        norm_w.size() != static_cast<size_t>(kW)) {
      return false;
    }
    if (has_inject) {
      const auto& branch = *reinterpret_cast<array*>(branch_h);
      const auto& gate = *reinterpret_cast<array*>(gate_h);
      if (branch.dtype() != bfloat16 ||
          branch.size() != static_cast<size_t>(m64) * kH ||
          gate.dtype() != bfloat16 ||
          gate.size() != static_cast<size_t>(m64) * kHc) {
        return false;
      }
      auto results = array::make_arrays(
          {x.shape(), x.shape()}, {bfloat16, bfloat16},
          std::make_shared<Qwen4InjectNormCuda>(
              stream, true, true, eps, weight_offset),
          {x, branch, gate, norm_w});
      *out_x = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_normed = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    } else {
      auto results = array::make_arrays(
          {x.shape()}, {bfloat16},
          std::make_shared<Qwen4InjectNormCuda>(
              stream, false, true, eps, weight_offset),
          {x, norm_w});
      if (out_x) {
        *out_x = reinterpret_cast<mlx_array*>(new array(x));
      }
      *out_normed = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_inject_norm_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" bool mlx_qwen4_hc_down_act_cuda(
    mlx_array* normed_h, mlx_array* w_down_h, mlx_array* s_down_h,
    mlx_array* b_down_h, mlx_array* w_inj_h, mlx_array* s_inj_h,
    mlx_array* b_inj_h, int32_t inj_mode,
    mlx_array** out_down, mlx_array** out_inj) {
  if (out_down) *out_down = nullptr;
  if (out_inj) *out_inj = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!normed_h || !w_down_h || !s_down_h || !b_down_h || !out_down ||
        inj_mode < 0 || inj_mode > 4 || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    if (inj_mode != 0 && !out_inj) {
      return false;
    }
    const auto& normed = *reinterpret_cast<array*>(normed_h);
    const auto& w_down = *reinterpret_cast<array*>(w_down_h);
    const auto& s_down = *reinterpret_cast<array*>(s_down_h);
    const auto& b_down = *reinterpret_cast<array*>(b_down_h);
    if (normed.dtype() != bfloat16 || normed.ndim() < 2 ||
        normed.shape(-1) != kW) {
      return false;
    }
    const int64_t m64 = static_cast<int64_t>(normed.size()) / kW;
    if (m64 < 1 || m64 > 65535) {
      return false;
    }
    if (w_down.dtype() != uint32 || w_down.ndim() != 2 ||
        w_down.shape(0) != kRank || w_down.shape(1) != kW / 4 ||
        s_down.dtype() != float16 || s_down.ndim() != 2 ||
        s_down.shape(0) != kRank || s_down.shape(1) != kW / 32 ||
        b_down.dtype() != float16 || b_down.ndim() != 2 ||
        b_down.shape(0) != kRank || b_down.shape(1) != kW / 32) {
      return false;
    }

    std::vector<array> inputs{normed, w_down, s_down, b_down};
    if (inj_mode == 0) {
      if (w_inj_h || s_inj_h || b_inj_h) {
        return false;
      }
    } else if (inj_mode == 1) {
      if (!w_inj_h || !s_inj_h || !b_inj_h) {
        return false;
      }
      const auto& w_inj = *reinterpret_cast<array*>(w_inj_h);
      const auto& s_inj = *reinterpret_cast<array*>(s_inj_h);
      const auto& b_inj = *reinterpret_cast<array*>(b_inj_h);
      if (w_inj.dtype() != uint32 || w_inj.ndim() != 2 ||
          w_inj.shape(0) != kHc || w_inj.shape(1) != kW / 4 ||
          s_inj.dtype() != float16 || s_inj.ndim() != 2 ||
          s_inj.shape(0) != kHc || s_inj.shape(1) != kW / 32 ||
          b_inj.dtype() != float16 || b_inj.ndim() != 2 ||
          b_inj.shape(0) != kHc || b_inj.shape(1) != kW / 32) {
        return false;
      }
      inputs.push_back(w_inj);
      inputs.push_back(s_inj);
      inputs.push_back(b_inj);
    } else {
      if (!w_inj_h || s_inj_h || b_inj_h) {
        return false;
      }
      const auto& w_inj = *reinterpret_cast<array*>(w_inj_h);
      const Dtype want_dt =
          (inj_mode == 2) ? bfloat16 : ((inj_mode == 3) ? float16 : float32);
      if (w_inj.dtype() != want_dt || w_inj.ndim() != 2 ||
          w_inj.shape(0) != kHc || w_inj.shape(1) != kW) {
        return false;
      }
      inputs.push_back(w_inj);
    }

    Shape down_shape = normed.shape();
    down_shape.back() = kRank;
    auto stream = default_stream(Device::gpu);
    if (inj_mode != 0) {
      Shape inj_shape = normed.shape();
      inj_shape.back() = kHc;
      auto results = array::make_arrays(
          {std::move(down_shape), std::move(inj_shape)}, {bfloat16, bfloat16},
          std::make_shared<Qwen4HcDownActCuda>(stream, inj_mode),
          std::move(inputs));
      *out_down = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_inj = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    } else {
      auto results = array::make_arrays(
          {std::move(down_shape)}, {bfloat16},
          std::make_shared<Qwen4HcDownActCuda>(stream, 0),
          std::move(inputs));
      *out_down = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_hc_down_act_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" mlx_array* mlx_qwen4_hc_up_mix_cuda(
    mlx_array* down_act_h, mlx_array* normed_h,
    mlx_array* w_up_h, mlx_array* s_up_h, mlx_array* b_up_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!down_act_h || !normed_h || !w_up_h || !s_up_h || !b_up_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& down_act = *reinterpret_cast<array*>(down_act_h);
    const auto& normed = *reinterpret_cast<array*>(normed_h);
    const auto& w_up = *reinterpret_cast<array*>(w_up_h);
    const auto& s_up = *reinterpret_cast<array*>(s_up_h);
    const auto& b_up = *reinterpret_cast<array*>(b_up_h);
    if (down_act.dtype() != bfloat16 || down_act.ndim() < 2 ||
        down_act.shape(-1) != kRank || normed.dtype() != bfloat16 ||
        normed.ndim() < 2 || normed.shape(-1) != kW) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(down_act.size()) / kRank;
    if (m64 < 1 || (m64 + 15) / 16 > 65535 ||
        normed.size() != static_cast<size_t>(m64) * kW) {
      return nullptr;
    }
    if (w_up.dtype() != uint32 || w_up.ndim() != 2 ||
        w_up.shape(0) != kW || w_up.shape(1) != kRank / 4 ||
        s_up.dtype() != float16 || s_up.ndim() != 2 ||
        s_up.shape(0) != kW || s_up.shape(1) != kRank / 32 ||
        b_up.dtype() != float16 || b_up.ndim() != 2 ||
        b_up.shape(0) != kW || b_up.shape(1) != kRank / 32) {
      return nullptr;
    }
    Shape out_shape = down_act.shape();
    out_shape.back() = kH;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4HcUpMixCuda>(default_stream(Device::gpu)),
        {down_act, normed, w_up, s_up, b_up});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_hc_up_mix_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" bool mlx_qwen4_router_topk_cuda(
    mlx_array* logits_h, int32_t top_k,
    mlx_array** out_ids, mlx_array** out_scores) {
  if (out_ids) *out_ids = nullptr;
  if (out_scores) *out_scores = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!logits_h || !out_ids || !out_scores ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }
    const auto& logits = *reinterpret_cast<array*>(logits_h);
    if ((logits.dtype() != bfloat16 && logits.dtype() != float32) ||
        logits.ndim() < 2) {
      return false;
    }
    const int E = logits.shape(-1);
    if (E < 1 || E > 512 || top_k < 1 || top_k > 16 || top_k > E) {
      return false;
    }
    const int64_t m64 = static_cast<int64_t>(logits.size()) / E;
    if (m64 < 1 || m64 > 65535) {
      return false;
    }
    Shape out_shape = logits.shape();
    out_shape.back() = top_k;
    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        {out_shape, out_shape}, {uint32, float32},
        std::make_shared<Qwen4RouterTopkCuda>(stream, E, top_k),
        {logits});
    *out_ids = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_scores = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_router_topk_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" mlx_array* mlx_qwen4_prefill_hc_mix_cuda(
    mlx_array* up_h, mlx_array* normed_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!up_h || !normed_h || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& up = *reinterpret_cast<array*>(up_h);
    const auto& normed = *reinterpret_cast<array*>(normed_h);
    if (up.dtype() != bfloat16 || normed.dtype() != bfloat16 ||
        up.ndim() < 2 || up.shape() != normed.shape() || up.shape(-1) != kW) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(up.size()) / kW;
    if (m64 < 1 || m64 > 65535) {
      return nullptr;
    }
    Shape out_shape = up.shape();
    out_shape.back() = kH;
    auto stream = default_stream(Device::gpu);
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4PrefillHcMixCuda>(stream),
        {up, normed});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_prefill_hc_mix_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_qwen4_stream_mean_cuda(mlx_array* x_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    using namespace mlx_node::qwen4_cuda;
    if (!x_h || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    if (x.dtype() != bfloat16 || x.ndim() < 2 || x.shape(-1) != kW) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(x.size()) / kW;
    if (m64 < 1 || m64 > 65535) {
      return nullptr;
    }
    Shape out_shape = x.shape();
    out_shape.back() = kH;
    auto stream = default_stream(Device::gpu);
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4StreamMeanCuda>(stream),
        {x});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_stream_mean_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_qwen4_dequant_q8_bf16_cuda(
    mlx_array* w_h, mlx_array* scales_h, mlx_array* biases_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!w_h || !scales_h || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const bool symmetric = (biases_h == nullptr);
    const auto& w = *reinterpret_cast<array*>(w_h);
    const auto& sc = *reinterpret_cast<array*>(scales_h);
    if (w.dtype() != uint32 || w.ndim() != 2 || sc.ndim() != 2 ||
        (sc.dtype() != float16 && sc.dtype() != bfloat16 &&
         sc.dtype() != float32)) {
      return nullptr;
    }
    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    const uint32_t grid_k = static_cast<uint32_t>(((K >> 3) + 255) / 256);
    if (N <= 0 || K < 32 || (K & 31) != 0 || grid_k == 0 || grid_k > 65535 ||
        sc.shape(0) != N || sc.shape(1) != (K >> 5)) {
      return nullptr;
    }
    std::vector<array> inputs;
    inputs.reserve(symmetric ? 2 : 3);
    inputs.push_back(w);
    inputs.push_back(sc);
    if (!symmetric) {
      const auto& bi = *reinterpret_cast<array*>(biases_h);
      if (bi.ndim() != 2 || bi.dtype() != sc.dtype() ||
          bi.shape(0) != N || bi.shape(1) != (K >> 5)) {
        return nullptr;
      }
      inputs.push_back(bi);
    }
    auto stream = default_stream(Device::gpu);
    auto result = array(
        Shape{N, K}, bfloat16,
        std::make_shared<Qwen4DequantQ8Bf16Cuda>(stream, symmetric),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_dequant_q8_bf16_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" bool mlx_qwen4_inject_norm_cuda(
    mlx_array* x_h, mlx_array* branch_h, mlx_array* gate_h,
    mlx_array* norm_w_h, float eps, float weight_offset,
    mlx_array** out_x, mlx_array** out_normed) {
  (void)x_h; (void)branch_h; (void)gate_h; (void)norm_w_h;
  (void)eps; (void)weight_offset;
  if (out_x) *out_x = nullptr;
  if (out_normed) *out_normed = nullptr;
  return false;
}

extern "C" bool mlx_qwen4_hc_down_act_cuda(
    mlx_array* normed_h, mlx_array* w_down_h, mlx_array* s_down_h,
    mlx_array* b_down_h, mlx_array* w_inj_h, mlx_array* s_inj_h,
    mlx_array* b_inj_h, int32_t inj_mode,
    mlx_array** out_down, mlx_array** out_inj) {
  (void)normed_h; (void)w_down_h; (void)s_down_h; (void)b_down_h;
  (void)w_inj_h; (void)s_inj_h; (void)b_inj_h; (void)inj_mode;
  if (out_down) *out_down = nullptr;
  if (out_inj) *out_inj = nullptr;
  return false;
}

extern "C" mlx_array* mlx_qwen4_hc_up_mix_cuda(
    mlx_array* down_act_h, mlx_array* normed_h,
    mlx_array* w_up_h, mlx_array* s_up_h, mlx_array* b_up_h) {
  (void)down_act_h; (void)normed_h; (void)w_up_h; (void)s_up_h; (void)b_up_h;
  return nullptr;
}

extern "C" bool mlx_qwen4_router_topk_cuda(
    mlx_array* logits_h, int32_t top_k,
    mlx_array** out_ids, mlx_array** out_scores) {
  (void)logits_h; (void)top_k;
  if (out_ids) *out_ids = nullptr;
  if (out_scores) *out_scores = nullptr;
  return false;
}

extern "C" mlx_array* mlx_qwen4_prefill_hc_mix_cuda(
    mlx_array* up_h, mlx_array* normed_h) {
  (void)up_h; (void)normed_h;
  return nullptr;
}

extern "C" mlx_array* mlx_qwen4_stream_mean_cuda(mlx_array* x_h) {
  (void)x_h;
  return nullptr;
}

extern "C" mlx_array* mlx_qwen4_dequant_q8_bf16_cuda(
    mlx_array* w_h, mlx_array* scales_h, mlx_array* biases_h) {
  (void)w_h; (void)scales_h; (void)biases_h;
  return nullptr;
}

#endif
