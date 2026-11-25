#include "cuda/inkling2/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mlx::core::inkling2_moe_cuda {
namespace {

using namespace mlx_node::inkling2_cuda;

constexpr char inkling2_moe_few_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/inkling2_moe_few.cu.inc"
    ;

constexpr char inkling2_moe_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/inkling2_moe_prefill.cu.inc"
    ;

// Best-effort SMEM/L1 carveout hint (50% = 64 KB SMEM / 64 KB L1 on Blackwell
// sm_121a so 3 CTAs/SM x 21 KB SMEM fit inside 64 KB while doubling L1 for x/mid).
void ensure_smem_carveout(CUfunction fn, int carveout_pct) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, int> configured;
  std::lock_guard<std::mutex> lock(mu);
  auto it = configured.find(fn);
  if (it != configured.end() && it->second == carveout_pct) {
    return;
  }
  (void)cuFuncSetAttribute(
      fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout_pct);
  configured[fn] = carveout_pct;
}

inline int nonq5k_mode_index(int mode) {
  switch (mode) {
    case 0: return 0; // NVFP4
    case 1: return 1; // SymQ8
    case 2: return 2; // AffineQ8
    case 4: return 3; // MXFP8
    default: return -1;
  }
}

inline bool check_expert_bank(
    const array& a, Dtype expected_dt, int rows_per_exp, int expected_cols) {
  if (a.dtype() != expected_dt) return false;
  if (a.ndim() == 3) {
    return a.shape(0) == kNumExperts && a.shape(1) == rows_per_exp &&
        a.shape(2) == expected_cols;
  }
  if (a.ndim() == 2) {
    return a.shape(0) == kNumExperts * rows_per_exp &&
        a.shape(1) == expected_cols;
  }
  return false;
}

inline bool is_f16_or_bf16(Dtype dt) {
  return dt == float16 || dt == bfloat16;
}

struct Inkling2MoeConfig {
  int gu_mode;
  int dn_mode;
  bool gu_interleaved;
  float route_scale;
  bool has_router_bias;
  bool has_gu_biases;
  bool has_gu_outer_scale;
  bool has_dn_biases;
  bool has_dn_outer_scale;
  bool has_post_conv;
  bool has_residual;
  bool has_taps;
  bool update_hist;
  bool emit_hist;
  bool emit_pre_conv;

  bool operator==(const Inkling2MoeConfig& o) const noexcept {
    return gu_mode == o.gu_mode && dn_mode == o.dn_mode &&
        gu_interleaved == o.gu_interleaved && route_scale == o.route_scale &&
        has_router_bias == o.has_router_bias &&
        has_gu_biases == o.has_gu_biases &&
        has_gu_outer_scale == o.has_gu_outer_scale &&
        has_dn_biases == o.has_dn_biases &&
        has_dn_outer_scale == o.has_dn_outer_scale &&
        has_post_conv == o.has_post_conv && has_residual == o.has_residual &&
        has_taps == o.has_taps && update_hist == o.update_hist &&
        emit_hist == o.emit_hist && emit_pre_conv == o.emit_pre_conv;
  }
};

struct UnpackedInputs {
  array x;
  array router_w;
  array gu_w;
  array gu_scales;
  array dn_w;
  array dn_scales;
  std::optional<array> router_bias;
  std::optional<array> gu_biases;
  std::optional<array> gu_outer_scale;
  std::optional<array> dn_biases;
  std::optional<array> dn_outer_scale;
  std::optional<array> post_conv_w;
  std::optional<array> hist_in;
  std::optional<array> residual;
  std::optional<array> conv_taps;
};

class Inkling2MoeCuda final : public Primitive {
 public:
  Inkling2MoeCuda(Stream stream, Inkling2MoeConfig cfg)
      : Primitive(stream), cfg_(cfg) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2MoeCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    size_t idx = 0;
    UnpackedInputs in{
        cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder),
        cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder),
        cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder),
        cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder),
        cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder),
        cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder),
    };
    if (cfg_.has_router_bias) {
      in.router_bias =
          cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_gu_biases) {
      in.gu_biases =
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_gu_outer_scale) {
      in.gu_outer_scale =
          cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_dn_biases) {
      in.dn_biases =
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_dn_outer_scale) {
      in.dn_outer_scale =
          cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_post_conv) {
      in.post_conv_w =
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
      in.hist_in =
          cu::ensure_row_contiguous_aligned<2>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_residual) {
      in.residual =
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }
    if (cfg_.has_taps) {
      in.conv_taps =
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    }

    const int M = static_cast<int>(in.x.size() / kHidden);
    if (M < 1 || M > 65535) {
      throw std::invalid_argument("Inkling2MoeCuda: unsupported M");
    }

    size_t out_idx = 0;
    array& out_y = outputs[out_idx++];
    out_y.set_data(cu::malloc_async(out_y.nbytes(), encoder));
    array* out_hist = nullptr;
    if (cfg_.emit_hist) {
      out_hist = &outputs[out_idx++];
      out_hist->set_data(cu::malloc_async(out_hist->nbytes(), encoder));
    }
    array* out_pre_conv = nullptr;
    if (cfg_.emit_pre_conv) {
      out_pre_conv = &outputs[out_idx++];
      out_pre_conv->set_data(cu::malloc_async(out_pre_conv->nbytes(), encoder));
    }

    if (M <= 16) {
      eval_few(in, M, out_y, out_hist, out_pre_conv, encoder);
    } else {
      eval_prefill(in, M, out_y, out_hist, out_pre_conv, encoder);
    }
  }

  DEFINE_NAME(Inkling2MoeCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2MoeCuda*>(&other);
    return rhs && (cfg_ == rhs->cfg_);
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes{inputs[0].shape()};
    if (cfg_.emit_hist) {
      shapes.push_back(Shape{3, kHidden});
    }
    if (cfg_.emit_pre_conv) {
      const int m = static_cast<int>(inputs[0].size() / kHidden);
      shapes.push_back(Shape{m, kHidden});
    }
    return shapes;
  }

 private:
  void eval_few(
      const UnpackedInputs& in, int M, array& out_y, array* out_hist,
      array* out_pre_conv, cu::CommandEncoder& encoder) {
    const int P = M * kTopK;
    const bool is_q5k = (cfg_.gu_mode == 3);
    // Default to unified Tensor Core MMA across M in 1..=16 (m_bucket 1 for M<=8, 2 for M<=16)
    // so M=1 AR and M=2..=16 speculative verification share 0-ULP batch-invariant reduction order.
    static const bool force_gemv_m2 = [] {
      const char* v = std::getenv("MLX_INKLING2_MOE_GEMV");
      return v && v[0] == '1';
    }();
    const int m_bucket = (force_gemv_m2 && M <= 2) ? 0 : (M <= 8 ? 1 : 2);
    const bool allow_hi = (M > 8);
    const int gu_sb_bf16 = (in.gu_scales.dtype() == bfloat16) ? 1 : 0;
    const int dn_sb_bf16 = (in.dn_scales.dtype() == bfloat16) ? 1 : 0;

    array logits = alloc_temp({kNumExperts, 16}, float32, encoder);
    array gamma = alloc_temp({P}, float32, encoder);
    array unique_experts = alloc_temp({P}, int32, encoder);
    array bucket_offsets = alloc_temp({P + 1}, int32, encoder);
    array pair_indices = alloc_temp({P}, int32, encoder);
    array num_unique = alloc_temp({1}, int32, encoder);
    // 64B (>= small_block_size) so MLX places the arrival counter in device VRAM.
    array arrive_counter = alloc_temp({16}, uint32, encoder);
    array mid = alloc_temp({P, kInter}, bfloat16, encoder);
    array down_pairs = alloc_temp({P, kHidden}, bfloat16, encoder);

    std::optional<array> xq, x_scales, x_sums;
    if (is_q5k) {
      xq = alloc_temp({M * kHidden}, uint8, encoder);
      x_scales = alloc_temp({M * (kHidden / 32)}, float32, encoder);
      x_sums = alloc_temp({M * (kHidden / 32)}, float32, encoder);
    }

    // 1A. Zero CTA arrival counter (no inputs -> scheduled off critical path in CUDA graphs).
    {
      encoder.set_output_array(arrive_counter);
      auto k_zero = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source, "inkling2_moe_zero_u32");
      cu::KernelArgs args;
      args.append(arrive_counter);
      encoder.add_kernel_node_raw(
          k_zero, dim3(1, 1, 1), dim3(1, 1, 1), {}, 0, args.args());
    }

    // 1B. Fused Exact-FP32 Router GEMV + Top-8 Routing + Grouping (+ Q8_1 X quant for Q5_K)
    {
      encoder.set_input_array(in.x);
      encoder.set_input_array(in.router_w);
      if (in.router_bias) encoder.set_input_array(*in.router_bias);
      if (in.dn_outer_scale) encoder.set_input_array(*in.dn_outer_scale);
      encoder.set_output_array(logits);
      encoder.set_output_array(gamma);
      encoder.set_output_array(unique_experts);
      encoder.set_output_array(bucket_offsets);
      encoder.set_output_array(pair_indices);
      encoder.set_output_array(num_unique);
      // set_output_array alone adds the dependency edge from k_zero above.
      encoder.set_output_array(arrive_counter);
      if (is_q5k) {
        encoder.set_output_array(*xq);
        encoder.set_output_array(*x_scales);
        encoder.set_output_array(*x_sums);
      }
      auto k_rp = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          is_q5k ? "inkling2_moe_router_plan_few<true>"
                 : "inkling2_moe_router_plan_few<false>");
      cu::KernelArgs args;
      args.append(in.x);
      args.append(in.router_w);
      if (in.router_bias) args.append(*in.router_bias);
      else args.append<CUdeviceptr>(0);
      if (in.dn_outer_scale) args.append(*in.dn_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(logits);
      args.append(gamma);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(arrive_counter);
      if (is_q5k) {
        args.append(*xq);
        args.append(*x_scales);
        args.append(*x_sums);
      } else {
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(0);
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<float>(cfg_.route_scale);
      encoder.add_kernel_node_raw(
          k_rp, dim3(kNumExperts, 1, 1), dim3(288, 1, 1), {}, 0, args.args());
    }

    // 2. Hybrid Grouped Gate + Up + SiTU(4, 25) -> mid[P, 1152]
    if (is_q5k && m_bucket > 0) {
      encoder.set_input_array(in.gu_w);
      encoder.set_input_array(in.gu_scales);
      encoder.set_input_array(*in.gu_biases);
      if (in.gu_outer_scale) encoder.set_input_array(*in.gu_outer_scale);
      encoder.set_input_array(*xq);
      encoder.set_input_array(*x_scales);
      encoder.set_input_array(*x_sums);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(mid);
      static constexpr const char* kQ5kLinEntries[2][2] = {
          {"inkling2_moe_gate_up_situ_q5k_linear_few<false, false>",
           "inkling2_moe_gate_up_situ_q5k_linear_few<false, true>"},
          {"inkling2_moe_gate_up_situ_q5k_linear_few<true, false>",
           "inkling2_moe_gate_up_situ_q5k_linear_few<true, true>"},
      };
      auto k_gu = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          kQ5kLinEntries[cfg_.gu_interleaved ? 1 : 0][allow_hi ? 1 : 0]);
      ensure_smem_carveout(k_gu, 50);
      cu::KernelArgs args;
      args.append(in.gu_w);
      args.append(in.gu_scales);
      args.append(*in.gu_biases);
      if (in.gu_outer_scale) args.append(*in.gu_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(*xq);
      args.append(*x_scales);
      args.append(*x_sums);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(mid);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kInter / 16, P, 1), dim3(32, 9, 1), {}, 0, args.args());
    } else if (is_q5k) {
      encoder.set_input_array(in.gu_w);
      encoder.set_input_array(in.gu_scales);
      encoder.set_input_array(*in.gu_biases);
      if (in.gu_outer_scale) encoder.set_input_array(*in.gu_outer_scale);
      encoder.set_input_array(*xq);
      encoder.set_input_array(*x_scales);
      encoder.set_input_array(*x_sums);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(mid);
      static constexpr const char* kQ5kGuEntries[2][3] = {
          {"inkling2_moe_gate_up_situ_q5k_few<false, 0>",
           "inkling2_moe_gate_up_situ_q5k_few<false, 1>",
           "inkling2_moe_gate_up_situ_q5k_few<false, 2>"},
          {"inkling2_moe_gate_up_situ_q5k_few<true, 0>",
           "inkling2_moe_gate_up_situ_q5k_few<true, 1>",
           "inkling2_moe_gate_up_situ_q5k_few<true, 2>"},
      };
      auto k_gu = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          kQ5kGuEntries[cfg_.gu_interleaved ? 1 : 0][m_bucket]);
      ensure_smem_carveout(k_gu, 50);
      cu::KernelArgs args;
      args.append(in.gu_w);
      args.append(in.gu_scales);
      args.append(*in.gu_biases);
      if (in.gu_outer_scale) args.append(*in.gu_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(*xq);
      args.append(*x_scales);
      args.append(*x_sums);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(mid);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kInter / 16, P, 1), dim3(32, 4, 1), {}, 0, args.args());
    } else if (cfg_.gu_mode == 0 && m_bucket > 0) {
      encoder.set_input_array(in.gu_w);
      encoder.set_input_array(in.gu_scales);
      if (in.gu_outer_scale) encoder.set_input_array(*in.gu_outer_scale);
      encoder.set_input_array(in.x);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(mid);
      static constexpr const char* kNvfp4GuEntries[2][2] = {
          {"inkling2_moe_gate_up_situ_nvfp4_few<false, false>",
           "inkling2_moe_gate_up_situ_nvfp4_few<false, true>"},
          {"inkling2_moe_gate_up_situ_nvfp4_few<true, false>",
           "inkling2_moe_gate_up_situ_nvfp4_few<true, true>"},
      };
      auto k_gu = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          kNvfp4GuEntries[cfg_.gu_interleaved ? 1 : 0][allow_hi ? 1 : 0]);
      ensure_smem_carveout(k_gu, 50);
      cu::KernelArgs args;
      args.append(in.gu_w);
      args.append(in.gu_scales);
      if (in.gu_outer_scale) args.append(*in.gu_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(in.x);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(mid);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kInter / 16, P, 1), dim3(32, 9, 1), {}, 0, args.args());
    } else {
      encoder.set_input_array(in.gu_w);
      encoder.set_input_array(in.gu_scales);
      if (in.gu_biases) encoder.set_input_array(*in.gu_biases);
      if (in.gu_outer_scale) encoder.set_input_array(*in.gu_outer_scale);
      encoder.set_input_array(in.x);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(mid);
      static constexpr const char* kGuEntries[4][2][3] = {
          {{"inkling2_moe_gate_up_situ_few<0, false, 0>",
            "inkling2_moe_gate_up_situ_few<0, false, 1>",
            "inkling2_moe_gate_up_situ_few<0, false, 2>"},
           {"inkling2_moe_gate_up_situ_few<0, true, 0>",
            "inkling2_moe_gate_up_situ_few<0, true, 1>",
            "inkling2_moe_gate_up_situ_few<0, true, 2>"}},
          {{"inkling2_moe_gate_up_situ_few<1, false, 0>",
            "inkling2_moe_gate_up_situ_few<1, false, 1>",
            "inkling2_moe_gate_up_situ_few<1, false, 2>"},
           {"inkling2_moe_gate_up_situ_few<1, true, 0>",
            "inkling2_moe_gate_up_situ_few<1, true, 1>",
            "inkling2_moe_gate_up_situ_few<1, true, 2>"}},
          {{"inkling2_moe_gate_up_situ_few<2, false, 0>",
            "inkling2_moe_gate_up_situ_few<2, false, 1>",
            "inkling2_moe_gate_up_situ_few<2, false, 2>"},
           {"inkling2_moe_gate_up_situ_few<2, true, 0>",
            "inkling2_moe_gate_up_situ_few<2, true, 1>",
            "inkling2_moe_gate_up_situ_few<2, true, 2>"}},
          {{"inkling2_moe_gate_up_situ_few<4, false, 0>",
            "inkling2_moe_gate_up_situ_few<4, false, 1>",
            "inkling2_moe_gate_up_situ_few<4, false, 2>"},
           {"inkling2_moe_gate_up_situ_few<4, true, 0>",
            "inkling2_moe_gate_up_situ_few<4, true, 1>",
            "inkling2_moe_gate_up_situ_few<4, true, 2>"}},
      };
      const int gm = nonq5k_mode_index(cfg_.gu_mode);
      auto k_gu = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          kGuEntries[gm][cfg_.gu_interleaved ? 1 : 0][m_bucket]);
      ensure_smem_carveout(k_gu, 50);
      cu::KernelArgs args;
      args.append(in.gu_w);
      args.append(in.gu_scales);
      if (in.gu_biases) args.append(*in.gu_biases);
      else args.append<CUdeviceptr>(0);
      if (in.gu_outer_scale) args.append(*in.gu_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(in.x);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(mid);
      args.append<int32_t>(gu_sb_bf16);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kInter / 16, P, 1), dim3(32, 4, 1), {}, 0, args.args());
    }

    // 3. Hybrid Grouped Down Projection -> down_pairs[P, 2304]
    if (cfg_.dn_mode == 0 && m_bucket > 0) {
      encoder.set_input_array(in.dn_w);
      encoder.set_input_array(in.dn_scales);
      encoder.set_input_array(mid);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(down_pairs);
      const char* dn_entry = allow_hi
          ? "inkling2_moe_down_nvfp4_few<true>"
          : "inkling2_moe_down_nvfp4_few<false>";
      auto k_dn = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source, dn_entry);
      ensure_smem_carveout(k_dn, 50);
      cu::KernelArgs args;
      args.append(in.dn_w);
      args.append(in.dn_scales);
      args.append(mid);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(down_pairs);
      encoder.add_kernel_node_raw(
          k_dn, dim3(kHidden / 32, P, 1), dim3(32, 9, 1), {}, 0, args.args());
    } else if (cfg_.dn_mode == 1 && m_bucket > 0) {
      encoder.set_input_array(in.dn_w);
      encoder.set_input_array(in.dn_scales);
      encoder.set_input_array(mid);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(down_pairs);
      const char* dn_entry = allow_hi
          ? "inkling2_moe_down_symq8_few<true>"
          : "inkling2_moe_down_symq8_few<false>";
      auto k_dn = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source, dn_entry);
      ensure_smem_carveout(k_dn, 50);
      cu::KernelArgs args;
      args.append(in.dn_w);
      args.append(in.dn_scales);
      args.append(mid);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(down_pairs);
      args.append<int32_t>(dn_sb_bf16);
      encoder.add_kernel_node_raw(
          k_dn, dim3(kHidden / 32, P, 1), dim3(32, 9, 1), {}, 0, args.args());
    } else {
      encoder.set_input_array(in.dn_w);
      encoder.set_input_array(in.dn_scales);
      if (in.dn_biases) encoder.set_input_array(*in.dn_biases);
      encoder.set_input_array(mid);
      encoder.set_input_array(unique_experts);
      encoder.set_input_array(bucket_offsets);
      encoder.set_input_array(pair_indices);
      encoder.set_input_array(num_unique);
      encoder.set_output_array(down_pairs);
      static constexpr const char* kDnEntries[4][3] = {
          {"inkling2_moe_down_few<0, 0>", "inkling2_moe_down_few<0, 1>",
           "inkling2_moe_down_few<0, 2>"},
          {"inkling2_moe_down_few<1, 0>", "inkling2_moe_down_few<1, 1>",
           "inkling2_moe_down_few<1, 2>"},
          {"inkling2_moe_down_few<2, 0>", "inkling2_moe_down_few<2, 1>",
           "inkling2_moe_down_few<2, 2>"},
          {"inkling2_moe_down_few<4, 0>", "inkling2_moe_down_few<4, 1>",
           "inkling2_moe_down_few<4, 2>"},
      };
      const int dm = nonq5k_mode_index(cfg_.dn_mode);
      auto k_dn = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source, kDnEntries[dm][m_bucket]);
      ensure_smem_carveout(k_dn, 50);
      cu::KernelArgs args;
      args.append(in.dn_w);
      args.append(in.dn_scales);
      if (in.dn_biases) args.append(*in.dn_biases);
      else args.append<CUdeviceptr>(0);
      args.append(mid);
      args.append(unique_experts);
      args.append(bucket_offsets);
      args.append(pair_indices);
      args.append(num_unique);
      args.append(down_pairs);
      args.append<int32_t>(dn_sb_bf16);
      encoder.add_kernel_node_raw(
          k_dn, dim3(kHidden / 64, P, 1), dim3(32, 4, 1), {}, 0, args.args());
    }

    // 4. Fused Combine (+ Optional Static ShortConv k=4 Linear/Tree + Residual + Tape + Hist)
    if (!cfg_.has_post_conv) {
      encoder.set_input_array(down_pairs);
      encoder.set_input_array(gamma);
      encoder.set_output_array(out_y);
      if (out_pre_conv) encoder.set_output_array(*out_pre_conv);
      auto k_comb = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          cfg_.emit_pre_conv ? "inkling2_moe_combine_only_few<true>"
                             : "inkling2_moe_combine_only_few<false>");
      cu::KernelArgs args;
      args.append(down_pairs);
      args.append(gamma);
      args.append(out_y);
      if (out_pre_conv) args.append(*out_pre_conv);
      else args.append<CUdeviceptr>(0);
      args.append<int32_t>(M);
      encoder.add_kernel_node_raw(
          k_comb, dim3(1, M, 1), dim3(288, 1, 1), {}, 0, args.args());
    } else {
      encoder.set_input_array(down_pairs);
      encoder.set_input_array(gamma);
      encoder.set_input_array(*in.post_conv_w);
      encoder.set_input_array(*in.hist_in);
      if (in.residual) encoder.set_input_array(*in.residual);
      if (in.conv_taps) encoder.set_input_array(*in.conv_taps);
      encoder.set_output_array(out_y);
      if (out_hist) encoder.set_output_array(*out_hist);
      if (out_pre_conv) encoder.set_output_array(*out_pre_conv);
      static constexpr const char* kCombConvEntries[2][2][2] = {
          {{"inkling2_moe_combine_sconv_res_few<false, false, false>",
            "inkling2_moe_combine_sconv_res_few<false, false, true>"},
           {"inkling2_moe_combine_sconv_res_few<false, true, false>",
            "inkling2_moe_combine_sconv_res_few<false, true, true>"}},
          {{"inkling2_moe_combine_sconv_res_few<true, false, false>",
            "inkling2_moe_combine_sconv_res_few<true, false, true>"},
           {"inkling2_moe_combine_sconv_res_few<true, true, false>",
            "inkling2_moe_combine_sconv_res_few<true, true, true>"}},
      };
      auto k_cc = cu::jit_kernel(
          encoder.device(), inkling2_moe_few_source,
          kCombConvEntries[cfg_.has_residual ? 1 : 0][cfg_.has_taps ? 1 : 0]
                          [cfg_.emit_pre_conv ? 1 : 0]);
      cu::KernelArgs args;
      args.append(down_pairs);
      args.append(gamma);
      args.append(*in.post_conv_w);
      args.append(*in.hist_in);
      if (in.residual) args.append(*in.residual);
      else args.append<CUdeviceptr>(0);
      if (in.conv_taps) args.append(*in.conv_taps);
      else args.append<CUdeviceptr>(0);
      args.append(out_y);
      if (out_hist) args.append(*out_hist);
      else args.append<CUdeviceptr>(0);
      if (out_pre_conv) args.append(*out_pre_conv);
      else args.append<CUdeviceptr>(0);
      args.append<int32_t>(M);
      args.append<int32_t>(cfg_.update_hist ? 1 : 0);
      encoder.add_kernel_node_raw(
          k_cc, dim3(kHidden / 16, 1, 1), dim3(128, 1, 1), {}, 0, args.args());
    }
  }

  void eval_prefill(
      const UnpackedInputs& in, int M, array& out_y, array* out_hist,
      array* out_pre_conv, cu::CommandEncoder& encoder) {
    const int P = M * kTopK;
    const int max_tiles = (P + 31) / 32 + kNumExperts;
    if (max_tiles > 65535) {
      throw std::invalid_argument("Inkling2MoeCuda: prefill max_tiles > 65535");
    }
    const bool is_q5k = (cfg_.gu_mode == 3);
    const int gu_sb_bf16 = (in.gu_scales.dtype() == bfloat16) ? 1 : 0;
    const int dn_sb_bf16 = (in.dn_scales.dtype() == bfloat16) ? 1 : 0;

    // Allocate exact-sized per-invocation scratch via MLX's pooled BufferCache
    // (zero process-lifetime VRAM retention and exception-safe on OOM).
    array logits = alloc_temp({M, kNumExperts}, float32, encoder);
    array expert_ids = alloc_temp({P}, int32, encoder);
    array gamma = alloc_temp({P}, float32, encoder);
    array sorted_pairs = alloc_temp({P}, int32, encoder);
    array inv_perm = alloc_temp({P}, int32, encoder);
    array tile_expert = alloc_temp({max_tiles}, int32, encoder);
    array tile_begin = alloc_temp({max_tiles}, int32, encoder);
    array tile_end = alloc_temp({max_tiles}, int32, encoder);
    array num_tiles = alloc_temp({1}, int32, encoder);
    array mid = alloc_temp({P, kInter}, bfloat16, encoder);
    array down_out = alloc_temp({P, kHidden}, bfloat16, encoder);
    std::optional<array> moe_out;
    if (cfg_.has_post_conv && !out_pre_conv) {
      moe_out = alloc_temp({M, kHidden}, bfloat16, encoder);
    }
    std::optional<array> xq, x_scales, x_qsums;
    if (is_q5k) {
      xq = alloc_temp({M * kHidden}, uint8, encoder);
      x_scales = alloc_temp({M * (kHidden / 256)}, float32, encoder);
      x_qsums = alloc_temp({M * (kHidden / 32)}, int16, encoder);
    }

    // 1. Exact-FP32 Router GEMM
    {
      encoder.set_input_array(in.x);
      encoder.set_input_array(in.router_w);
      encoder.set_output_array(logits);
      auto k_rg = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          "inkling2_moe_prefill_router_gemm");
      cu::KernelArgs args;
      args.append(in.x);
      args.append(in.router_w);
      args.append(logits);
      args.append<int32_t>(M);
      encoder.add_kernel_node_raw(
          k_rg, dim3(kNumExperts / 16, (M + 15) / 16, 1), dim3(16, 16, 1),
          {}, 0, args.args());
    }

    // 2A. Exact Top-8 Routing
    {
      encoder.set_input_array(logits);
      if (in.router_bias) encoder.set_input_array(*in.router_bias);
      if (in.dn_outer_scale) encoder.set_input_array(*in.dn_outer_scale);
      encoder.set_output_array(expert_ids);
      encoder.set_output_array(gamma);
      auto k_tk = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          "inkling2_moe_prefill_topk");
      cu::KernelArgs args;
      args.append(logits);
      if (in.router_bias) args.append(*in.router_bias);
      else args.append<CUdeviceptr>(0);
      if (in.dn_outer_scale) args.append(*in.dn_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(expert_ids);
      args.append(gamma);
      args.append<int32_t>(M);
      args.append<float>(cfg_.route_scale);
      encoder.add_kernel_node_raw(
          k_tk, dim3((M + 7) / 8, 1, 1), dim3(32, 8, 1), {}, 0, args.args());
    }

    // 2B. Counting Sort + BM=32 Tile Table
    {
      encoder.set_input_array(expert_ids);
      encoder.set_output_array(sorted_pairs);
      encoder.set_output_array(inv_perm);
      encoder.set_output_array(tile_expert);
      encoder.set_output_array(tile_begin);
      encoder.set_output_array(tile_end);
      encoder.set_output_array(num_tiles);
      auto k_rs = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          "inkling2_moe_prefill_route_sort");
      cu::KernelArgs args;
      args.append(expert_ids);
      args.append(sorted_pairs);
      args.append(inv_perm);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append<int32_t>(P);
      encoder.add_kernel_node_raw(
          k_rs, dim3(1, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }

    // 2C. Optional GS=256 X Quantizer (for Q5_K gate_up)
    if (is_q5k) {
      encoder.set_input_array(in.x);
      encoder.set_output_array(*xq);
      encoder.set_output_array(*x_scales);
      encoder.set_output_array(*x_qsums);
      auto k_qx = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          "inkling2_moe_prefill_quantize_x_q5k");
      cu::KernelArgs args;
      args.append(in.x);
      args.append(*xq);
      args.append(*x_scales);
      args.append(*x_qsums);
      args.append<int32_t>(M);
      const uint32_t grid_x =
          static_cast<uint32_t>((M * (kHidden / 256) + 7) / 8);
      encoder.add_kernel_node_raw(
          k_qx, dim3(grid_x, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }

    // 3. Grouped Prefill BM=32 Gate + Up + SiTU(4, 25) -> mid[P, 1152] BF16
    if (is_q5k) {
      encoder.set_input_array(in.gu_w);
      encoder.set_input_array(in.gu_scales);
      encoder.set_input_array(*in.gu_biases);
      if (in.gu_outer_scale) encoder.set_input_array(*in.gu_outer_scale);
      encoder.set_input_array(*xq);
      encoder.set_input_array(*x_scales);
      encoder.set_input_array(*x_qsums);
      encoder.set_input_array(sorted_pairs);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(mid);
      auto k_gu = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          cfg_.gu_interleaved
              ? "inkling2_moe_prefill_gate_up_situ_q5k_bm32<true>"
              : "inkling2_moe_prefill_gate_up_situ_q5k_bm32<false>");
      cu::KernelArgs args;
      args.append(in.gu_w);
      args.append(in.gu_scales);
      args.append(*in.gu_biases);
      if (in.gu_outer_scale) args.append(*in.gu_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(*xq);
      args.append(*x_scales);
      args.append(*x_qsums);
      args.append(sorted_pairs);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(mid);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kInter / 64, max_tiles, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    } else {
      encoder.set_input_array(in.gu_w);
      encoder.set_input_array(in.gu_scales);
      if (in.gu_biases) encoder.set_input_array(*in.gu_biases);
      if (in.gu_outer_scale) encoder.set_input_array(*in.gu_outer_scale);
      encoder.set_input_array(in.x);
      encoder.set_input_array(sorted_pairs);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(mid);
      static constexpr const char* kPreGuEntries[4][2] = {
          {"inkling2_moe_prefill_gate_up_situ_bm32<0, false>",
           "inkling2_moe_prefill_gate_up_situ_bm32<0, true>"},
          {"inkling2_moe_prefill_gate_up_situ_bm32<1, false>",
           "inkling2_moe_prefill_gate_up_situ_bm32<1, true>"},
          {"inkling2_moe_prefill_gate_up_situ_bm32<2, false>",
           "inkling2_moe_prefill_gate_up_situ_bm32<2, true>"},
          {"inkling2_moe_prefill_gate_up_situ_bm32<4, false>",
           "inkling2_moe_prefill_gate_up_situ_bm32<4, true>"},
      };
      const int gm = nonq5k_mode_index(cfg_.gu_mode);
      auto k_gu = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          kPreGuEntries[gm][cfg_.gu_interleaved ? 1 : 0]);
      cu::KernelArgs args;
      args.append(in.gu_w);
      args.append(in.gu_scales);
      if (in.gu_biases) args.append(*in.gu_biases);
      else args.append<CUdeviceptr>(0);
      if (in.gu_outer_scale) args.append(*in.gu_outer_scale);
      else args.append<CUdeviceptr>(0);
      args.append(in.x);
      args.append(sorted_pairs);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(mid);
      args.append<int32_t>(gu_sb_bf16);
      encoder.add_kernel_node_raw(
          k_gu, dim3(kInter / 64, max_tiles, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 4. Grouped Prefill BM=32 Down Projection -> down_out[P, 2304] BF16
    {
      encoder.set_input_array(in.dn_w);
      encoder.set_input_array(in.dn_scales);
      if (in.dn_biases) encoder.set_input_array(*in.dn_biases);
      encoder.set_input_array(mid);
      encoder.set_input_array(tile_expert);
      encoder.set_input_array(tile_begin);
      encoder.set_input_array(tile_end);
      encoder.set_input_array(num_tiles);
      encoder.set_output_array(down_out);
      static constexpr const char* kPreDnEntries[4] = {
          "inkling2_moe_prefill_down_bm32<0>",
          "inkling2_moe_prefill_down_bm32<1>",
          "inkling2_moe_prefill_down_bm32<2>",
          "inkling2_moe_prefill_down_bm32<4>",
      };
      const int dm = nonq5k_mode_index(cfg_.dn_mode);
      auto k_dn = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source, kPreDnEntries[dm]);
      cu::KernelArgs args;
      args.append(in.dn_w);
      args.append(in.dn_scales);
      if (in.dn_biases) args.append(*in.dn_biases);
      else args.append<CUdeviceptr>(0);
      args.append(mid);
      args.append(tile_expert);
      args.append(tile_begin);
      args.append(tile_end);
      args.append(num_tiles);
      args.append(down_out);
      args.append<int32_t>(dn_sb_bf16);
      encoder.add_kernel_node_raw(
          k_dn, dim3(kHidden / 64, max_tiles, 1), dim3(32, 4, 1), {}, 0,
          args.args());
    }

    // 5A. Vectorized Unsort + Weighted Combine
    if (!cfg_.has_post_conv) {
      encoder.set_input_array(down_out);
      encoder.set_input_array(gamma);
      encoder.set_input_array(inv_perm);
      encoder.set_output_array(out_y);
      if (out_pre_conv) encoder.set_output_array(*out_pre_conv);
      auto k_comb = cu::jit_kernel(
          encoder.device(), inkling2_moe_prefill_source,
          cfg_.emit_pre_conv ? "inkling2_moe_prefill_combine<true>"
                             : "inkling2_moe_prefill_combine<false>");
      cu::KernelArgs args;
      args.append(down_out);
      args.append(gamma);
      args.append(inv_perm);
      args.append(out_y);
      if (out_pre_conv) args.append(*out_pre_conv);
      else args.append<CUdeviceptr>(0);
      args.append<int32_t>(M);
      encoder.add_kernel_node_raw(
          k_comb, dim3(1, M, 1), dim3(288, 1, 1), {}, 0, args.args());
    } else {
      const array& pre_conv_buf = out_pre_conv ? *out_pre_conv : *moe_out;
      {
        encoder.set_input_array(down_out);
        encoder.set_input_array(gamma);
        encoder.set_input_array(inv_perm);
        encoder.set_output_array(pre_conv_buf);
        auto k_comb = cu::jit_kernel(
            encoder.device(), inkling2_moe_prefill_source,
            "inkling2_moe_prefill_combine<false>");
        cu::KernelArgs args;
        args.append(down_out);
        args.append(gamma);
        args.append(inv_perm);
        args.append(pre_conv_buf);
        args.append<CUdeviceptr>(0);
        args.append<int32_t>(M);
        encoder.add_kernel_node_raw(
            k_comb, dim3(1, M, 1), dim3(288, 1, 1), {}, 0, args.args());
      }
      // 5B. Static ShortConv(k=4, linear OR tree) + Optional Residual + Hist
      {
        encoder.set_input_array(pre_conv_buf);
        encoder.set_input_array(*in.post_conv_w);
        encoder.set_input_array(*in.hist_in);
        if (in.residual) encoder.set_input_array(*in.residual);
        if (in.conv_taps) encoder.set_input_array(*in.conv_taps);
        encoder.set_output_array(out_y);
        if (out_hist) encoder.set_output_array(*out_hist);
        static constexpr const char* kPreConvEntries[2][2] = {
            {"inkling2_moe_prefill_sconv_res<false, false>",
             "inkling2_moe_prefill_sconv_res<false, true>"},
            {"inkling2_moe_prefill_sconv_res<true, false>",
             "inkling2_moe_prefill_sconv_res<true, true>"},
        };
        auto k_sc = cu::jit_kernel(
            encoder.device(), inkling2_moe_prefill_source,
            kPreConvEntries[cfg_.has_residual ? 1 : 0][cfg_.has_taps ? 1 : 0]);
        cu::KernelArgs args;
        args.append(pre_conv_buf);
        args.append(*in.post_conv_w);
        args.append(*in.hist_in);
        if (in.residual) args.append(*in.residual);
        else args.append<CUdeviceptr>(0);
        if (in.conv_taps) args.append(*in.conv_taps);
        else args.append<CUdeviceptr>(0);
        args.append(out_y);
        if (out_hist) args.append(*out_hist);
        else args.append<CUdeviceptr>(0);
        args.append<int32_t>(M);
        args.append<int32_t>(cfg_.update_hist ? 1 : 0);
        const uint32_t grid_y = static_cast<uint32_t>(std::min(M, 64));
        encoder.add_kernel_node_raw(
            k_sc, dim3(kHidden / 128, grid_y, 1), dim3(128, 1, 1), {}, 0,
            args.args());
      }
    }
  }

  Inkling2MoeConfig cfg_;
};

}  // namespace
}  // namespace mlx::core::inkling2_moe_cuda

extern "C" bool mlx_inkling2_moe_forward_cuda(
    mlx_array* x_h,
    mlx_array* router_w_h,
    mlx_array* router_bias_h,
    mlx_array* gu_w_h,
    mlx_array* gu_scales_h,
    mlx_array* gu_biases_h,
    mlx_array* gu_outer_scale_h,
    int gu_mode,
    bool gu_interleaved,
    mlx_array* dn_w_h,
    mlx_array* dn_scales_h,
    mlx_array* dn_biases_h,
    mlx_array* dn_outer_scale_h,
    int dn_mode,
    float route_scale,
    mlx_array* post_conv_w_h,
    mlx_array* hist_in_h,
    mlx_array* residual_h,
    mlx_array* conv_taps_h,
    bool update_hist,
    bool emit_pre_conv,
    mlx_array** out_y,
    mlx_array** out_hist,
    mlx_array** out_pre_conv) {
  if (out_y) *out_y = nullptr;
  if (out_hist) *out_hist = nullptr;
  if (out_pre_conv) *out_pre_conv = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_moe_cuda;
    using namespace mlx_node::inkling2_cuda;

    if (!x_h || !router_w_h || !gu_w_h || !gu_scales_h || !dn_w_h ||
        !dn_scales_h || !out_y || (emit_pre_conv && !out_pre_conv) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }

    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& router_w = *reinterpret_cast<array*>(router_w_h);
    const auto& gu_w = *reinterpret_cast<array*>(gu_w_h);
    const auto& gu_scales = *reinterpret_cast<array*>(gu_scales_h);
    const auto& dn_w = *reinterpret_cast<array*>(dn_w_h);
    const auto& dn_scales = *reinterpret_cast<array*>(dn_scales_h);

    if (x.dtype() != bfloat16 || x.ndim() < 2 || x.shape(-1) != kHidden) {
      return false;
    }
    const int64_t m64 = static_cast<int64_t>(x.size()) / kHidden;
    if (m64 < 1 || m64 > 65535) {
      return false;
    }
    const int M = static_cast<int>(m64);

    if (router_w.dtype() != float32 || router_w.ndim() != 2 ||
        router_w.shape(0) != kNumExperts || router_w.shape(1) != kHidden) {
      return false;
    }

    // Validate gate_up banks ([128, 2304, ...], K = 2304, N = 2304)
    constexpr int kN2 = 2 * kInter;
    if (gu_mode == 0) {
      if (gu_biases_h != nullptr ||
          !check_expert_bank(gu_w, uint32, kN2, kHidden / 8) ||
          !check_expert_bank(gu_scales, uint8, kN2, kHidden / 16)) {
        return false;
      }
    } else if (gu_mode == 1) {
      if (gu_biases_h != nullptr ||
          !check_expert_bank(gu_w, uint32, kN2, kHidden / 4) ||
          !is_f16_or_bf16(gu_scales.dtype()) ||
          !check_expert_bank(gu_scales, gu_scales.dtype(), kN2, kHidden / 32)) {
        return false;
      }
    } else if (gu_mode == 2) {
      if (gu_biases_h == nullptr ||
          !check_expert_bank(gu_w, uint32, kN2, kHidden / 4) ||
          !is_f16_or_bf16(gu_scales.dtype()) ||
          !check_expert_bank(gu_scales, gu_scales.dtype(), kN2, kHidden / 32)) {
        return false;
      }
      const auto& gu_b = *reinterpret_cast<array*>(gu_biases_h);
      if (!check_expert_bank(gu_b, gu_scales.dtype(), kN2, kHidden / 32)) {
        return false;
      }
    } else if (gu_mode == 3) {
      if (gu_biases_h == nullptr ||
          !check_expert_bank(gu_w, uint32, kN2, (kHidden / 32) * 5) ||
          !check_expert_bank(gu_scales, uint8, kN2, kHidden / 16)) {
        return false;
      }
      const auto& gu_b = *reinterpret_cast<array*>(gu_biases_h);
      if (!check_expert_bank(gu_b, float16, kN2, kHidden / 128)) {
        return false;
      }
    } else if (gu_mode == 4) {
      if (gu_biases_h != nullptr ||
          !check_expert_bank(gu_w, uint32, kN2, kHidden / 4) ||
          !check_expert_bank(gu_scales, uint8, kN2, kHidden / 32)) {
        return false;
      }
    } else {
      return false;
    }

    // Validate down banks ([128, 2304, ...], K = 1152, N = 2304)
    if (dn_mode == 0) {
      if (dn_biases_h != nullptr ||
          !check_expert_bank(dn_w, uint32, kHidden, kInter / 8) ||
          !check_expert_bank(dn_scales, uint8, kHidden, kInter / 16)) {
        return false;
      }
    } else if (dn_mode == 1) {
      if (dn_biases_h != nullptr ||
          !check_expert_bank(dn_w, uint32, kHidden, kInter / 4) ||
          !is_f16_or_bf16(dn_scales.dtype()) ||
          !check_expert_bank(dn_scales, dn_scales.dtype(), kHidden, kInter / 32)) {
        return false;
      }
    } else if (dn_mode == 2) {
      if (dn_biases_h == nullptr ||
          !check_expert_bank(dn_w, uint32, kHidden, kInter / 4) ||
          !is_f16_or_bf16(dn_scales.dtype()) ||
          !check_expert_bank(dn_scales, dn_scales.dtype(), kHidden, kInter / 32)) {
        return false;
      }
      const auto& dn_b = *reinterpret_cast<array*>(dn_biases_h);
      if (!check_expert_bank(dn_b, dn_scales.dtype(), kHidden, kInter / 32)) {
        return false;
      }
    } else if (dn_mode == 4) {
      if (dn_biases_h != nullptr ||
          !check_expert_bank(dn_w, uint32, kHidden, kInter / 4) ||
          !check_expert_bank(dn_scales, uint8, kHidden, kInter / 32)) {
        return false;
      }
    } else {
      return false;
    }

    std::vector<array> inputs{x, router_w, gu_w, gu_scales, dn_w, dn_scales};
    Inkling2MoeConfig cfg{};
    cfg.gu_mode = gu_mode;
    cfg.dn_mode = dn_mode;
    cfg.gu_interleaved = gu_interleaved;
    cfg.route_scale = route_scale;
    cfg.update_hist = update_hist;
    cfg.emit_pre_conv = emit_pre_conv;

    if (router_bias_h) {
      const auto& rb = *reinterpret_cast<array*>(router_bias_h);
      if (rb.dtype() != float32 || rb.size() != static_cast<size_t>(kNumExperts)) {
        return false;
      }
      cfg.has_router_bias = true;
      inputs.push_back(rb);
    }
    if (gu_biases_h) {
      cfg.has_gu_biases = true;
      inputs.push_back(*reinterpret_cast<array*>(gu_biases_h));
    }
    if (gu_outer_scale_h) {
      const auto& gos = *reinterpret_cast<array*>(gu_outer_scale_h);
      if (gos.dtype() != float32 || gos.size() != static_cast<size_t>(kNumExperts)) {
        return false;
      }
      cfg.has_gu_outer_scale = true;
      inputs.push_back(gos);
    }
    if (dn_biases_h) {
      cfg.has_dn_biases = true;
      inputs.push_back(*reinterpret_cast<array*>(dn_biases_h));
    }
    if (dn_outer_scale_h) {
      const auto& dos = *reinterpret_cast<array*>(dn_outer_scale_h);
      if (dos.dtype() != float32 || dos.size() != static_cast<size_t>(kNumExperts)) {
        return false;
      }
      cfg.has_dn_outer_scale = true;
      inputs.push_back(dos);
    }

    if (post_conv_w_h) {
      if (!hist_in_h) {
        return false;
      }
      const auto& pcw = *reinterpret_cast<array*>(post_conv_w_h);
      const auto& hin = *reinterpret_cast<array*>(hist_in_h);
      if (pcw.dtype() != float32 || pcw.size() != static_cast<size_t>(kHidden) * 4 ||
          hin.dtype() != bfloat16 || hin.size() != static_cast<size_t>(3) * kHidden) {
        return false;
      }
      cfg.has_post_conv = true;
      cfg.emit_hist = (out_hist != nullptr);
      inputs.push_back(pcw);
      inputs.push_back(hin);

      if (residual_h) {
        const auto& res = *reinterpret_cast<array*>(residual_h);
        if (res.dtype() != bfloat16 ||
            res.size() != static_cast<size_t>(M) * kHidden) {
          return false;
        }
        cfg.has_residual = true;
        inputs.push_back(res);
      }
      if (conv_taps_h) {
        const auto& taps = *reinterpret_cast<array*>(conv_taps_h);
        if (taps.dtype() != int32 ||
            taps.size() != static_cast<size_t>(M) * 4) {
          return false;
        }
        cfg.has_taps = true;
        inputs.push_back(taps);
      }
    }

    std::vector<Shape> out_shapes{x.shape()};
    std::vector<Dtype> out_dtypes{bfloat16};
    if (cfg.emit_hist) {
      out_shapes.push_back(Shape{3, kHidden});
      out_dtypes.push_back(bfloat16);
    }
    if (cfg.emit_pre_conv) {
      out_shapes.push_back(Shape{M, kHidden});
      out_dtypes.push_back(bfloat16);
    }

    auto stream = default_stream(Device::gpu);
    auto results = array::make_arrays(
        std::move(out_shapes), std::move(out_dtypes),
        std::make_shared<Inkling2MoeCuda>(stream, cfg),
        std::move(inputs));

    size_t r_idx = 0;
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(results[r_idx++])));
    if (cfg.emit_hist) {
      *out_hist =
          reinterpret_cast<mlx_array*>(new array(std::move(results[r_idx++])));
    }
    if (cfg.emit_pre_conv) {
      *out_pre_conv =
          reinterpret_cast<mlx_array*>(new array(std::move(results[r_idx++])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_moe_forward_cuda: %s\n", e.what());
    return false;
  }
}

#else

extern "C" bool mlx_inkling2_moe_forward_cuda(
    mlx_array* x_h,
    mlx_array* router_w_h,
    mlx_array* router_bias_h,
    mlx_array* gu_w_h,
    mlx_array* gu_scales_h,
    mlx_array* gu_biases_h,
    mlx_array* gu_outer_scale_h,
    int gu_mode,
    bool gu_interleaved,
    mlx_array* dn_w_h,
    mlx_array* dn_scales_h,
    mlx_array* dn_biases_h,
    mlx_array* dn_outer_scale_h,
    int dn_mode,
    float route_scale,
    mlx_array* post_conv_w_h,
    mlx_array* hist_in_h,
    mlx_array* residual_h,
    mlx_array* conv_taps_h,
    bool update_hist,
    bool emit_pre_conv,
    mlx_array** out_y,
    mlx_array** out_hist,
    mlx_array** out_pre_conv) {
  (void)x_h; (void)router_w_h; (void)router_bias_h;
  (void)gu_w_h; (void)gu_scales_h; (void)gu_biases_h; (void)gu_outer_scale_h;
  (void)gu_mode; (void)gu_interleaved;
  (void)dn_w_h; (void)dn_scales_h; (void)dn_biases_h; (void)dn_outer_scale_h;
  (void)dn_mode; (void)route_scale;
  (void)post_conv_w_h; (void)hist_in_h; (void)residual_h; (void)conv_taps_h;
  (void)update_hist; (void)emit_pre_conv;
  if (out_y) *out_y = nullptr;
  if (out_hist) *out_hist = nullptr;
  if (out_pre_conv) *out_pre_conv = nullptr;
  return false;
}

#endif
