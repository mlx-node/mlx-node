#include "mlx_common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include "mlx_cuda_ops.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mlx::core::qwen4_ops {
namespace {

constexpr int kH = 2560;
constexpr int kW = 10240;
constexpr int kHc = 4;
constexpr int kRank = 320;
constexpr int kPleKeep = 9;
constexpr float kLog2E = 1.4426950408889634f;

constexpr char qwen4_mhc_source[] =
#include "cuda/qwen4_mhc.cu.inc"
    ;

constexpr char qwen4_qsa_source[] =
#include "cuda/qwen4_qsa_cuda.cu.inc"
    ;

constexpr char qwen4_ple_source[] =
#include "cuda/qwen4_ple.cu.inc"
    ;

constexpr char qwen4_q8_prefill_source[] =
#include "cuda/qwen4_q8_prefill.cu.inc"
    ;

array alloc_temp(Shape shape, Dtype dtype, cu::CommandEncoder& encoder) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(cu::malloc_async(a.nbytes(), encoder));
  encoder.add_temporary(a);
  return a;
}

array alloc_persistent(Shape shape, Dtype dtype, cu::CommandEncoder& encoder) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(cu::malloc_async(a.nbytes(), encoder));
  return a;
}

void ensure_dynamic_smem(CUfunction fn, int smem_bytes) {
  thread_local std::unordered_map<CUfunction, int> configured;
  auto it = configured.find(fn);
  if (it != configured.end() && it->second >= smem_bytes) {
    return;
  }
  CUresult rc = cuFuncSetAttribute(
      fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
  if (rc != CUDA_SUCCESS) {
    throw std::runtime_error(
        "cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) failed");
  }
  configured[fn] = smem_bytes;
}

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
          encoder.device(), qwen4_q8_prefill_source, kPrefillEntries[inj_mode_]);
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
// 4. QSA Sparse Indexer + Zero-KV-Gather Sparse SDPA
// ============================================================================
class Qwen4QsaSparseSdpaCuda final : public UnaryPrimitive {
 public:
  Qwen4QsaSparseSdpaCuda(
      Stream stream, int kv_len, int num_blocks, float scale, int mask_mode)
      : UnaryPrimitive(stream),
        kv_len_(kv_len),
        num_blocks_(num_blocks),
        scale_(scale),
        mask_mode_(mask_mode) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Qwen4QsaSparseSdpaCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    auto ensure_kv_aligned = [&](const array& a) -> array {
      const bool ok = a.strides(3) == 1 && (a.strides(2) % 8 == 0) &&
          (a.strides(1) % 8 == 0) && (a.strides(0) % 8 == 0) &&
          (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
      if (ok) {
        return a;
      }
      auto copy = contiguous_copy_gpu(a, stream());
      encoder.add_temporary(copy);
      return copy;
    };

    array q = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array k = ensure_kv_aligned(inputs[1]);
    array v = ensure_kv_aligned(inputs[2]);
    array iq = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);

    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    const int gqa = Hq / Hkv;
    const int base_pos = kv_len_ - Q_len;
    if (B != 1 || D != 256 || gqa < 1 || gqa > 16 || Q_len < 1 ||
        Q_len > 65535 || base_pos < 0 || kv_len_ > k.shape(2) ||
        num_blocks_ < 0) {
      throw std::invalid_argument("Qwen4QsaSparseSdpaCuda: unsupported replay shape");
    }

    array block_ids = alloc_temp({Q_len, 512}, int32, encoder);
    array num_selected = alloc_temp({Q_len}, int32, encoder);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    CUdeviceptr tree_mask_ptr = 0;
    std::optional<array> tree_mask_owned;
    if (mask_mode_ == 1) {
      tree_mask_owned.emplace(
          cu::ensure_row_contiguous_aligned<4>(inputs[5], stream(), encoder));
      tree_mask_ptr =
          reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*tree_mask_owned));
    }

    const bool use_split_k = (Q_len <= 16);

    // Stage 1: Indexer Top-512 selection
    if (use_split_k) {
      if (num_blocks_ > 0) {
        array pooled_k =
            cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
        array scores = alloc_temp({Q_len, num_blocks_}, float32, encoder);

        encoder.set_input_array(iq);
        encoder.set_input_array(pooled_k);
        encoder.set_output_array(scores);
        auto k_score = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source, "qwen4_qsa_index_score_cuda");
        cu::KernelArgs sargs;
        sargs.append(iq);
        sargs.append(pooled_k);
        sargs.append(scores);
        sargs.append<int32_t>(Q_len);
        sargs.append<int32_t>(num_blocks_);
        encoder.add_kernel_node_raw(
            k_score, dim3((num_blocks_ + 31) / 32, 1, 1), dim3(256, 1, 1), {}, 0,
            sargs.args());

        encoder.set_input_array(scores);
        encoder.set_output_array(block_ids);
        encoder.set_output_array(num_selected);
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source, "qwen4_qsa_index_topk512_cuda");
        cu::KernelArgs targs;
        targs.append(scores);
        targs.append(block_ids);
        targs.append(num_selected);
        targs.append<int32_t>(num_blocks_);
        targs.append<int32_t>(base_pos);
        targs.append<int32_t>(mask_mode_);
        encoder.add_kernel_node_raw(
            k_topk, dim3(Q_len, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      } else {
        encoder.set_output_array(block_ids);
        encoder.set_output_array(num_selected);
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source, "qwen4_qsa_index_topk512_cuda");
        cu::KernelArgs targs;
        targs.append<CUdeviceptr>(0);
        targs.append(block_ids);
        targs.append(num_selected);
        targs.append<int32_t>(0);
        targs.append<int32_t>(base_pos);
        targs.append<int32_t>(mask_mode_);
        encoder.add_kernel_node_raw(
            k_topk, dim3(Q_len, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      }
    } else {
      if (num_blocks_ <= 512) {
        encoder.set_output_array(block_ids);
        encoder.set_output_array(num_selected);
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source,
            "qwen4_qsa_index_topk512_radix_cuda");
        cu::KernelArgs targs;
        targs.append<CUdeviceptr>(0);
        targs.append(block_ids);
        targs.append(num_selected);
        targs.append<int32_t>(0);
        targs.append<int32_t>(num_blocks_);
        targs.append<int32_t>(0);
        targs.append<int32_t>(base_pos);
        targs.append<int32_t>(mask_mode_);
        encoder.add_kernel_node_raw(
            k_topk, dim3(Q_len, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
      } else {
        array pooled_k =
            cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
        const int full_stride = (num_blocks_ + 7) & ~7;
        constexpr size_t kTargetScratchFloats = 2 * 1024 * 1024; // 8 MB L2-resident
        int q_chunk = static_cast<int>(
            kTargetScratchFloats / static_cast<size_t>(full_stride));
        q_chunk = (q_chunk / 64) * 64;
        if (q_chunk < 64) q_chunk = 64;
        if (q_chunk > Q_len) q_chunk = Q_len;

        array scores = alloc_temp({q_chunk, full_stride}, float32, encoder);
        auto k_score = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source,
            "qwen4_qsa_index_score_tc_cuda");
        auto k_topk = cu::jit_kernel(
            encoder.device(), qwen4_qsa_source,
            "qwen4_qsa_index_topk512_radix_cuda");

        for (int q0 = 0; q0 < Q_len; q0 += q_chunk) {
          const int q_cnt = std::min(q_chunk, Q_len - q0);
          int max_comp = (mask_mode_ != 0)
              ? (base_pos >> 2)
              : ((base_pos + q0 + q_cnt) >> 2);
          if (max_comp > num_blocks_) max_comp = num_blocks_;
          if (max_comp <= 512) {
            encoder.set_output_array(block_ids);
            encoder.set_output_array(num_selected);
            cu::KernelArgs targs;
            targs.append<CUdeviceptr>(0);
            targs.append(block_ids);
            targs.append(num_selected);
            targs.append<int32_t>(q0);
            targs.append<int32_t>(num_blocks_);
            targs.append<int32_t>(0);
            targs.append<int32_t>(base_pos);
            targs.append<int32_t>(mask_mode_);
            encoder.add_kernel_node_raw(
                k_topk, dim3(q_cnt, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
          } else {
            const int s_stride = (max_comp + 7) & ~7;
            encoder.set_input_array(iq);
            encoder.set_input_array(pooled_k);
            encoder.set_output_array(scores);
            cu::KernelArgs sargs;
            sargs.append(iq);
            sargs.append(pooled_k);
            sargs.append(scores);
            sargs.append<int32_t>(Q_len);
            sargs.append<int32_t>(q0);
            sargs.append<int32_t>(q_cnt);
            sargs.append<int32_t>(max_comp);
            sargs.append<int32_t>(s_stride);
            encoder.add_kernel_node_raw(
                k_score, dim3((max_comp + 63) / 64, (q_cnt + 63) / 64, 1),
                dim3(256, 1, 1), {}, 0, sargs.args());

            encoder.set_input_array(scores);
            encoder.set_output_array(scores);
            encoder.set_output_array(block_ids);
            encoder.set_output_array(num_selected);
            cu::KernelArgs targs;
            targs.append(scores);
            targs.append(block_ids);
            targs.append(num_selected);
            targs.append<int32_t>(q0);
            targs.append<int32_t>(num_blocks_);
            targs.append<int32_t>(s_stride);
            targs.append<int32_t>(base_pos);
            targs.append<int32_t>(mask_mode_);
            encoder.add_kernel_node_raw(
                k_topk, dim3(q_cnt, 1, 1), dim3(256, 1, 1), {}, 0, targs.args());
          }
        }
      }
    }

    // Stage 2: Zero-gather Sparse SDPA (+ Split-K reduce for Q_len <= 16)
    const float scale_log2 = scale_ * kLog2E;
    if (use_split_k) {
      const int target_splits = std::max(2, 24 / Q_len);
      const int max_tiles_bound = (std::min(num_blocks_, 512) + 7) / 8 + 1;
      const int num_splits = std::min(target_splits, std::max(1, max_tiles_bound));

      array partial_o =
          alloc_temp({B, Hkv, num_splits, Q_len, gqa, 256}, bfloat16, encoder);
      array partial_m =
          alloc_temp({B, Hkv, num_splits, Q_len, gqa}, float32, encoder);
      array partial_l =
          alloc_temp({B, Hkv, num_splits, Q_len, gqa}, float32, encoder);

      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      if (mask_mode_ == 1) {
        encoder.set_input_array(*tree_mask_owned);
      }
      encoder.set_output_array(partial_o);
      encoder.set_output_array(partial_m);
      encoder.set_output_array(partial_l);

      const char* entry = (mask_mode_ == 1)
          ? "qwen4_qsa_sparse_sdpa_cuda<true, 1>"
          : "qwen4_qsa_sparse_sdpa_cuda<true, 0>";
      auto k_sdpa = cu::jit_kernel(encoder.device(), qwen4_qsa_source, entry);
      cu::KernelArgs args;
      args.append(q);
      args.append(k);
      args.append(v);
      args.append(block_ids);
      args.append(num_selected);
      args.append<CUdeviceptr>(tree_mask_ptr);
      args.append(partial_o);
      args.append(partial_m);
      args.append(partial_l);
      args.append<int32_t>(Hkv);
      args.append<int32_t>(gqa);
      args.append<int32_t>(Q_len);
      args.append<int32_t>(kv_len_);
      args.append<int32_t>(512);
      args.append<int32_t>(base_pos);
      args.append<int64_t>(k.strides(0));
      args.append<int64_t>(k.strides(1));
      args.append<int64_t>(k.strides(2));
      args.append<int64_t>(v.strides(0));
      args.append<int64_t>(v.strides(1));
      args.append<int64_t>(v.strides(2));
      args.append<float>(scale_log2);
      args.append<int32_t>(num_splits);
      encoder.add_kernel_node_raw(
          k_sdpa, dim3(num_splits, Q_len, B * Hkv), dim3(32, 4, 1), {}, 0,
          args.args());

      encoder.set_input_array(partial_o);
      encoder.set_input_array(partial_m);
      encoder.set_input_array(partial_l);
      encoder.set_output_array(out);
      auto k_red = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source, "qwen4_qsa_sparse_reduce_cuda");
      cu::KernelArgs rargs;
      rargs.append(partial_o);
      rargs.append(partial_m);
      rargs.append(partial_l);
      rargs.append(out);
      rargs.append<int32_t>(Hkv);
      rargs.append<int32_t>(gqa);
      rargs.append<int32_t>(Q_len);
      rargs.append<int32_t>(num_splits);
      encoder.add_kernel_node_raw(
          k_red, dim3(Q_len * gqa, Hkv, B), dim3(32, 1, 1), {}, 0,
          rargs.args());
    } else if (gqa == 12 && mask_mode_ == 0) {
      constexpr int kMaxUnionTiles = 152; // 1216 blocks adaptive cutoff
      const int qt5 = (Q_len + 4) / 5;
      const int qt4 = (Q_len + 3) / 4;

      array tile_blks = alloc_temp({qt5, kMaxUnionTiles, 8}, int32, encoder);
      array tile_cmask = alloc_temp({qt5, kMaxUnionTiles, 5}, uint32, encoder);
      array tile_counts = alloc_temp({qt5}, int32, encoder);

      // 2A: Parallel 5-way block union + adaptive disjoint detection
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      encoder.set_output_array(tile_blks);
      encoder.set_output_array(tile_cmask);
      encoder.set_output_array(tile_counts);
      auto k_pm = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source, "qwen4_qsa_premerge_bq5_cuda");
      cu::KernelArgs pargs;
      pargs.append(block_ids);
      pargs.append(num_selected);
      pargs.append(tile_blks);
      pargs.append(tile_cmask);
      pargs.append(tile_counts);
      pargs.append<int32_t>(Q_len);
      pargs.append<int32_t>(512);
      pargs.append<int32_t>(kMaxUnionTiles);
      encoder.add_kernel_node_raw(
          k_pm, dim3(qt5, 1, 1), dim3(256, 1, 1), {}, 0, pargs.args());

      // 2B: BQ=5 Q-Tiled Sparse SDPA (60 Q-rows, 4w=128 thr, 8w/SM, 0 spills)
      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(tile_blks);
      encoder.set_input_array(tile_cmask);
      encoder.set_input_array(tile_counts);
      encoder.set_output_array(out);
      auto k_qt5 = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source,
          "qwen4_qsa_sdpa_prefill_qtiled5_cuda");
      cu::KernelArgs qargs;
      qargs.append(q);
      qargs.append(k);
      qargs.append(v);
      qargs.append(tile_blks);
      qargs.append(tile_cmask);
      qargs.append(tile_counts);
      qargs.append(out);
      qargs.append<int32_t>(Hkv);
      qargs.append<int32_t>(Q_len);
      qargs.append<int32_t>(kv_len_);
      qargs.append<int32_t>(base_pos);
      qargs.append<int32_t>(kMaxUnionTiles);
      qargs.append<int64_t>(k.strides(1));
      qargs.append<int64_t>(k.strides(2));
      qargs.append<int64_t>(v.strides(1));
      qargs.append<int64_t>(v.strides(2));
      qargs.append<float>(scale_log2);
      encoder.add_kernel_node_raw(
          k_qt5, dim3(qt5, Hkv, 1), dim3(128, 1, 1), {}, 0, qargs.args());

      // 2C: Disjoint Fallback (4 independent warps, BN=16, 0 barriers; instant return if tile_counts>=0)
      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      encoder.set_input_array(tile_counts);
      encoder.set_output_array(out);
      auto k_ind = cu::jit_kernel(
          encoder.device(), qwen4_qsa_source,
          "qwen4_qsa_sdpa_prefill_indep4w_cuda");
      cu::KernelArgs iargs;
      iargs.append(q);
      iargs.append(k);
      iargs.append(v);
      iargs.append(block_ids);
      iargs.append(num_selected);
      iargs.append(tile_counts);
      iargs.append(out);
      iargs.append<int32_t>(Hkv);
      iargs.append<int32_t>(gqa);
      iargs.append<int32_t>(Q_len);
      iargs.append<int32_t>(kv_len_);
      iargs.append<int32_t>(512);
      iargs.append<int32_t>(base_pos);
      iargs.append<int64_t>(k.strides(1));
      iargs.append<int64_t>(k.strides(2));
      iargs.append<int64_t>(v.strides(1));
      iargs.append<int64_t>(v.strides(2));
      iargs.append<float>(scale_log2);
      encoder.add_kernel_node_raw(
          k_ind, dim3(qt4, Hkv, 1), dim3(128, 1, 1), {}, 0, iargs.args());
    } else {
      encoder.set_input_array(q);
      encoder.set_input_array(k);
      encoder.set_input_array(v);
      encoder.set_input_array(block_ids);
      encoder.set_input_array(num_selected);
      if (mask_mode_ == 1) {
        encoder.set_input_array(*tree_mask_owned);
      }
      encoder.set_output_array(out);

      const char* entry = (mask_mode_ == 1)
          ? "qwen4_qsa_sparse_sdpa_cuda<false, 1>"
          : "qwen4_qsa_sparse_sdpa_cuda<false, 0>";
      auto k_sdpa = cu::jit_kernel(encoder.device(), qwen4_qsa_source, entry);
      cu::KernelArgs args;
      args.append(q);
      args.append(k);
      args.append(v);
      args.append(block_ids);
      args.append(num_selected);
      args.append<CUdeviceptr>(tree_mask_ptr);
      args.append(out);
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
      args.append<int32_t>(Hkv);
      args.append<int32_t>(gqa);
      args.append<int32_t>(Q_len);
      args.append<int32_t>(kv_len_);
      args.append<int32_t>(512);
      args.append<int32_t>(base_pos);
      args.append<int64_t>(k.strides(0));
      args.append<int64_t>(k.strides(1));
      args.append<int64_t>(k.strides(2));
      args.append<int64_t>(v.strides(0));
      args.append<int64_t>(v.strides(1));
      args.append<int64_t>(v.strides(2));
      args.append<float>(scale_log2);
      args.append<int32_t>(1);
      encoder.add_kernel_node_raw(
          k_sdpa, dim3(1, Q_len, B * Hkv), dim3(32, 4, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Qwen4QsaSparseSdpaCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4QsaSparseSdpaCuda*>(&other);
    return rhs && rhs->kv_len_ == kv_len_ && rhs->num_blocks_ == num_blocks_ &&
        rhs->scale_ == scale_ && rhs->mask_mode_ == mask_mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int kv_len_;
  int num_blocks_;
  float scale_;
  int mask_mode_;
};

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

extern "C" bool mlx_qwen4_inject_norm_cuda(
    mlx_array* x_h, mlx_array* branch_h, mlx_array* gate_h,
    mlx_array* norm_w_h, float eps, float weight_offset,
    mlx_array** out_x, mlx_array** out_normed) {
  if (out_x) *out_x = nullptr;
  if (out_normed) *out_normed = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
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

extern "C" mlx_array* mlx_qwen4_qsa_sparse_sdpa_cuda(
    mlx_array* q_h, mlx_array* k_h, mlx_array* v_h,
    mlx_array* iq_h, mlx_array* pooled_k_h, mlx_array* tree_mask_h,
    int32_t kv_len, int32_t num_blocks, float scale, float idx_scale,
    int32_t mask_mode) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!q_h || !k_h || !v_h || !iq_h || !pooled_k_h ||
        !(idx_scale > 0.0f) || (mask_mode != 0 && mask_mode != 1) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& q = *reinterpret_cast<array*>(q_h);
    const auto& k = *reinterpret_cast<array*>(k_h);
    const auto& v = *reinterpret_cast<array*>(v_h);
    const auto& iq = *reinterpret_cast<array*>(iq_h);
    const auto& pooled_k = *reinterpret_cast<array*>(pooled_k_h);
    if (q.dtype() != bfloat16 || k.dtype() != bfloat16 || v.dtype() != bfloat16 ||
        iq.dtype() != bfloat16 || pooled_k.dtype() != bfloat16 ||
        q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      return nullptr;
    }
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int Q_len = q.shape(2);
    const int D = q.shape(3);
    const int Hkv = k.shape(1);
    if (B != 1 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0 || D != 256 ||
        Q_len < 1 || Q_len > 65535 ||
        k.shape(0) != 1 || k.shape(3) != 256 || v.shape() != k.shape() ||
        kv_len < Q_len || kv_len > k.shape(2) || num_blocks < 0) {
      return nullptr;
    }
    const int gqa = Hq / Hkv;
    if (gqa < 1 || gqa > 16) {
      return nullptr;
    }
    const bool iq_ok =
        (iq.ndim() == 4 && iq.shape(0) == 1 && iq.shape(1) == 4 &&
         iq.shape(2) == Q_len && iq.shape(3) == 128) ||
        (iq.ndim() == 3 && iq.shape(0) == 4 && iq.shape(1) == Q_len &&
         iq.shape(2) == 128);
    const bool pk_ok =
        (pooled_k.ndim() == 2 && pooled_k.shape(0) >= num_blocks &&
         pooled_k.shape(1) == 128) ||
        (pooled_k.ndim() == 3 && pooled_k.shape(0) == 1 &&
         pooled_k.shape(1) >= num_blocks && pooled_k.shape(2) == 128);
    if (!iq_ok || !pk_ok) {
      return nullptr;
    }

    std::vector<array> inputs{q, k, v, iq, pooled_k};
    if (mask_mode == 1) {
      if (!tree_mask_h || Q_len > 16) {
        return nullptr;
      }
      const auto& tm = *reinterpret_cast<array*>(tree_mask_h);
      if (tm.dtype() != uint32 || tm.ndim() != 1 || tm.shape(0) != Q_len) {
        return nullptr;
      }
      inputs.push_back(tm);
    } else if (tree_mask_h != nullptr) {
      return nullptr;
    }

    auto result = array(
        q.shape(), bfloat16,
        std::make_shared<Qwen4QsaSparseSdpaCuda>(
            default_stream(Device::gpu), kv_len, num_blocks, scale, mask_mode),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_qsa_sparse_sdpa_cuda: %s\n", e.what());
    return nullptr;
  }
}

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

// ============================================================================
// 7. Prefill mHC Mix (`sigmoid(up) * normed` + 4-stream mean, W=10240 -> H=2560)
// ============================================================================
namespace mlx::core::qwen4_ops {

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
  explicit Qwen4DequantQ8Bf16Cuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4DequantQ8Bf16Cuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array sc = cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);
    array bi = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    if (N <= 0 || K < 32 || (K & 31) != 0 || sc.dtype() != bi.dtype()) {
      throw std::invalid_argument("Qwen4DequantQ8Bf16Cuda: unsupported shape");
    }
    const int sb_dtype =
        (sc.dtype() == float16) ? 0 : (sc.dtype() == bfloat16 ? 1 : 2);
    static constexpr const char* kEntries[3] = {
        "qwen4_dequant_q8_bf16_cuda<0>",
        "qwen4_dequant_q8_bf16_cuda<1>",
        "qwen4_dequant_q8_bf16_cuda<2>",
    };
    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(w);
    encoder.set_input_array(sc);
    encoder.set_input_array(bi);
    encoder.set_output_array(out);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_mhc_source, kEntries[sb_dtype]);
    cu::KernelArgs args;
    args.append(w);
    args.append(sc);
    args.append(bi);
    args.append(out);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    const uint32_t grid_x = static_cast<uint32_t>(((K >> 3) + 255) / 256);
    encoder.add_kernel_node_raw(
        fn, dim3(grid_x, static_cast<uint32_t>(N), 1), dim3(256, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Qwen4DequantQ8Bf16Cuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Qwen4DequantQ8Bf16Cuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), inputs[0].shape(1) * 4}};
  }
};

// Persistent per-(device, stream) workspace for direct Q8_0 prefill GEMM.
// Eliminates per-projection cudaMallocAsync churn across 325+ projections/chunk.
struct Qwen4Q8PrefillWorkspace {
  size_t cap_mk = 0;
  size_t cap_sc = 0;
  size_t cap_qs = 0;
  std::optional<array> xq;
  std::optional<array> x_sc;
  std::optional<array> x_qsum;

  static size_t bucket_size(
      size_t need, size_t b0, size_t b1, size_t b2, size_t b3, size_t step) {
    if (need <= b0) return b0;
    if (need <= b1) return b1;
    if (need <= b2) return b2;
    if (need <= b3) return b3;
    return ((need + step - 1) / step) * step;
  }

  void ensure(
      size_t need_mk, size_t need_sc, size_t need_qs,
      cu::CommandEncoder& encoder) {
    auto retire = [&](std::optional<array>& opt) {
      if (opt) {
        encoder.add_temporary(std::move(*opt));
        opt.reset();
      }
    };
    if (need_mk > cap_mk) {
      retire(xq);
      cap_mk = bucket_size(
          need_mk, 4u << 20, 16u << 20, 48u << 20, 96u << 20, 16u << 20);
      xq = alloc_persistent({static_cast<int>(cap_mk)}, int8, encoder);
    }
    if (need_sc > cap_sc) {
      retire(x_sc);
      cap_sc = bucket_size(need_sc, 65536, 262144, 524288, 1048576, 262144);
      x_sc = alloc_persistent({static_cast<int>(cap_sc)}, float32, encoder);
    }
    if (need_qs > cap_qs) {
      retire(x_qsum);
      cap_qs = bucket_size(need_qs, 262144, 1048576, 2097152, 4194304, 1048576);
      x_qsum = alloc_persistent({static_cast<int>(cap_qs)}, int16, encoder);
    }
  }
};

Qwen4Q8PrefillWorkspace& get_q8_prefill_workspace(int dev, int stream_idx) {
  static std::mutex mu;
  // Intentionally leaked so GPU buffers are not destroyed during teardown.
  static auto* map = new std::unordered_map<uint64_t, Qwen4Q8PrefillWorkspace>();
  uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(dev)) << 32) |
      static_cast<uint32_t>(stream_idx);
  std::lock_guard<std::mutex> lk(mu);
  return (*map)[key];
}

// ============================================================================
// Direct Q8_0 Prefill Tensor-Core GEMM (INT8 128x128 / 128x64 or Exact BF16)
// ============================================================================
class Qwen4Q8PrefillCuda final : public Primitive {
 public:
  Qwen4Q8PrefillCuda(Stream stream, int gs_shift, bool general_bias)
      : Primitive(stream), gs_shift_(gs_shift), general_bias_(general_bias) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4Q8PrefillCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array sc = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    std::optional<array> bi;
    if (general_bias_) {
      bi = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    }

    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    const int M = static_cast<int>(x.size() / static_cast<size_t>(K));
    if (M < 1 || M > 65535 || N < 2 || (N & 1) != 0 || K < 32 || (K & 31) != 0) {
      throw std::invalid_argument("Qwen4Q8PrefillCuda: unsupported shape");
    }

    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    if (gs_shift_ == -2) {
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(sc);
      if (general_bias_) {
        encoder.set_input_array(*bi);
      }
      encoder.set_output_array(out);
      const char* entry = general_bias_ ? "qwen4_q8_prefill_bf16<true>"
                                        : "qwen4_q8_prefill_bf16<false>";
      auto fn = cu::jit_kernel(encoder.device(), qwen4_q8_prefill_source, entry);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(sc);
      if (general_bias_) {
        args.append(*bi);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);
      const uint32_t g64 =
          static_cast<uint32_t>(((M + 127) / 128) * ((N + 63) / 64));
      encoder.add_kernel_node_raw(
          fn, dim3(g64, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
      return;
    }

    const size_t need_mk = static_cast<size_t>(M) * static_cast<size_t>(K);
    const size_t need_sc = (gs_shift_ == 0)
        ? static_cast<size_t>(M)
        : static_cast<size_t>(M) * static_cast<size_t>(K >> gs_shift_);
    const size_t need_qs = general_bias_
        ? static_cast<size_t>(M) * static_cast<size_t>(K >> 5)
        : 0;
    auto& ws = get_q8_prefill_workspace(
        encoder.device().cuda_device(), stream().index);
    ws.ensure(need_mk, need_sc, need_qs, encoder);
    const array& xq = *ws.xq;
    const array& x_sc = *ws.x_sc;

    const int gs_idx = (gs_shift_ == 0) ? 0 : (gs_shift_ == 7 ? 1 : 2);
    const int gb_idx = general_bias_ ? 1 : 0;

    // 1. Quantize BF16 activations -> INT8 + per-row / grouped scales (+ optional qsum)
    {
      static constexpr const char* kQuantEntries[3][2] = {
          {"qwen4_q8_quantize_x<0, false>", "qwen4_q8_quantize_x<0, true>"},
          {"qwen4_q8_quantize_x<7, false>", "qwen4_q8_quantize_x<7, true>"},
          {"qwen4_q8_quantize_x<8, false>", "qwen4_q8_quantize_x<8, true>"},
      };
      encoder.set_input_array(x);
      encoder.set_output_array(xq);
      encoder.set_output_array(x_sc);
      if (general_bias_) {
        encoder.set_output_array(*ws.x_qsum);
      }
      auto k_quant = cu::jit_kernel(
          encoder.device(), qwen4_q8_prefill_source,
          kQuantEntries[gs_idx][gb_idx]);
      cu::KernelArgs args;
      args.append(x);
      args.append(xq);
      args.append(x_sc);
      if (general_bias_) {
        args.append(*ws.x_qsum);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(K);
      uint32_t grid_q = 0;
      if (gs_shift_ == 0) {
        grid_q = static_cast<uint32_t>(M);
      } else if (gs_shift_ == 7) {
        grid_q = static_cast<uint32_t>(
            (static_cast<uint64_t>(M) * static_cast<uint64_t>(K >> 7) * 16 + 255) /
            256);
      } else {
        grid_q = static_cast<uint32_t>(
            (static_cast<uint64_t>(M) * static_cast<uint64_t>(K >> 8) * 32 + 255) /
            256);
      }
      encoder.add_kernel_node_raw(
          k_quant, dim3(grid_q, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
    }

    // 2. Direct Q8_0 INT8 Tensor-Core GEMM (128x128 dynamic SMEM vs 128x64 static SMEM)
    {
      const int m_tiles = (M + 127) / 128;
      const int n_tiles_128 = (N + 127) / 128;
      const bool use_128x128 = (K % 128 == 0) && (N >= 512) &&
          ((N & 127) == 0 || N >= 1024) &&
          (m_tiles * n_tiles_128 >= 96);

      static constexpr const char* kGemm128Entries[3][2] = {
          {"qwen4_q8_prefill_int8_128x128<0, false>",
           "qwen4_q8_prefill_int8_128x128<0, true>"},
          {"qwen4_q8_prefill_int8_128x128<7, false>",
           "qwen4_q8_prefill_int8_128x128<7, true>"},
          {"qwen4_q8_prefill_int8_128x128<8, false>",
           "qwen4_q8_prefill_int8_128x128<8, true>"},
      };
      static constexpr const char* kGemm64Entries[3][2] = {
          {"qwen4_q8_prefill_int8_128x64<0, false>",
           "qwen4_q8_prefill_int8_128x64<0, true>"},
          {"qwen4_q8_prefill_int8_128x64<7, false>",
           "qwen4_q8_prefill_int8_128x64<7, true>"},
          {"qwen4_q8_prefill_int8_128x64<8, false>",
           "qwen4_q8_prefill_int8_128x64<8, true>"},
      };

      encoder.set_input_array(xq);
      encoder.set_input_array(x_sc);
      if (general_bias_) {
        encoder.set_input_array(*ws.x_qsum);
      }
      encoder.set_input_array(w);
      encoder.set_input_array(sc);
      if (general_bias_) {
        encoder.set_input_array(*bi);
      }
      encoder.set_output_array(out);
      // Mark persistent workspace buffers as outputs of the GEMM so CUDA Graph
      // dependency tracking enforces strict happens-before before reuse.
      encoder.set_output_array(xq);
      encoder.set_output_array(x_sc);
      if (general_bias_) {
        encoder.set_output_array(*ws.x_qsum);
      }

      cu::KernelArgs args;
      args.append(xq);
      args.append(x_sc);
      if (general_bias_) {
        args.append(*ws.x_qsum);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(w);
      args.append(sc);
      if (general_bias_) {
        args.append(*bi);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(N);
      args.append<int32_t>(K);

      if (use_128x128) {
        constexpr int kSm128Sym =
            2 * (128 * 128 + 128 * 128 + 4 * 128 * 4 + 16);
        constexpr int kSm128Gen =
            2 * (128 * 128 + 128 * 128 + 8 * 128 * 4);
        const int smem_bytes = general_bias_ ? kSm128Gen : kSm128Sym;
        auto fn = cu::jit_kernel(
            encoder.device(), qwen4_q8_prefill_source,
            kGemm128Entries[gs_idx][gb_idx]);
        ensure_dynamic_smem(fn, smem_bytes);
        const uint32_t g128 = static_cast<uint32_t>(m_tiles * n_tiles_128);
        encoder.add_kernel_node_raw(
            fn, dim3(g128, 1, 1), dim3(256, 1, 1), {},
            static_cast<uint32_t>(smem_bytes), args.args());
      } else {
        auto fn = cu::jit_kernel(
            encoder.device(), qwen4_q8_prefill_source,
            kGemm64Entries[gs_idx][gb_idx]);
        const uint32_t g64 =
            static_cast<uint32_t>(m_tiles * ((N + 63) / 64));
        encoder.add_kernel_node_raw(
            fn, dim3(g64, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
      }
    }
  }

  DEFINE_NAME(Qwen4Q8PrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Qwen4Q8PrefillCuda*>(&other);
    return rhs && rhs->gs_shift_ == gs_shift_ &&
        rhs->general_bias_ == general_bias_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = inputs[1].shape(0);
    return {std::move(s)};
  }

 private:
  int gs_shift_;
  bool general_bias_;
};

// ============================================================================
// Fused Prefill mHC Up-Mix (BF16 MMA, Zero [M, 10240] DRAM Buffer!)
// ============================================================================
class Qwen4HcUpMixPrefillCuda final : public Primitive {
 public:
  explicit Qwen4HcUpMixPrefillCuda(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4HcUpMixPrefillCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array down_act =
        cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array normed =
        cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array w_up =
        cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array s_up =
        cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder);

    const int M = static_cast<int>(down_act.size() / kRank);
    if (M < 1 || M > 65535 ||
        normed.size() != static_cast<size_t>(M) * kW) {
      throw std::invalid_argument("Qwen4HcUpMixPrefillCuda: unsupported shape");
    }

    auto& out = outputs[0];
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(down_act);
    encoder.set_input_array(normed);
    encoder.set_input_array(w_up);
    encoder.set_input_array(s_up);
    encoder.set_output_array(out);

    const bool use_v2_i8 = (M > 512);
    auto fn = cu::jit_kernel(
        encoder.device(), qwen4_q8_prefill_source,
        use_v2_i8 ? "qwen4_hc_up_mix_prefill_v2"
                  : "qwen4_hc_up_mix_prefill<false>");
    cu::KernelArgs args;
    args.append(down_act);
    args.append(w_up);
    args.append(s_up);
    args.append(normed);
    args.append(out);
    args.append<int32_t>(M);
    const uint32_t g_hc = use_v2_i8
        ? static_cast<uint32_t>(((M + 63) / 64) * 80)
        : static_cast<uint32_t>(((M + 127) / 128) * 160);
    encoder.add_kernel_node_raw(
        fn, dim3(g_hc, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Qwen4HcUpMixPrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Qwen4HcUpMixPrefillCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    Shape s = inputs[0].shape();
    s.back() = kH;
    return {std::move(s)};
  }
};

} // namespace mlx::core::qwen4_ops

extern "C" mlx_array* mlx_qwen4_prefill_hc_mix_cuda(
    mlx_array* up_h, mlx_array* normed_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
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
    if (!w_h || !scales_h || !biases_h || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& w = *reinterpret_cast<array*>(w_h);
    const auto& sc = *reinterpret_cast<array*>(scales_h);
    const auto& bi = *reinterpret_cast<array*>(biases_h);
    if (w.dtype() != uint32 || w.ndim() != 2 || sc.ndim() != 2 || bi.ndim() != 2 ||
        sc.dtype() != bi.dtype() ||
        (sc.dtype() != float16 && sc.dtype() != bfloat16 && sc.dtype() != float32)) {
      return nullptr;
    }
    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    if (N <= 0 || N > 65535 || K < 32 || (K & 31) != 0 ||
        sc.shape(0) != N || sc.shape(1) != (K >> 5) ||
        bi.shape(0) != N || bi.shape(1) != (K >> 5)) {
      return nullptr;
    }
    auto stream = default_stream(Device::gpu);
    auto result = array(
        Shape{N, K}, bfloat16,
        std::make_shared<Qwen4DequantQ8Bf16Cuda>(stream),
        {w, sc, bi});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_dequant_q8_bf16_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_qwen4_q8_prefill_cuda(
    mlx_array* x_h, mlx_array* w_h, mlx_array* scales_h, mlx_array* biases_h,
    int32_t gs_shift, bool general_bias) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!x_h || !w_h || !scales_h || (general_bias && !biases_h) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& w = *reinterpret_cast<array*>(w_h);
    const auto& sc = *reinterpret_cast<array*>(scales_h);
    if (x.dtype() != bfloat16 || x.ndim() < 2 || w.dtype() != uint32 ||
        w.ndim() != 2 || sc.dtype() != float16 || sc.ndim() != 2) {
      return nullptr;
    }
    const int N = w.shape(0);
    const int K = w.shape(1) * 4;
    if (N < 2 || (N & 1) != 0 || K < 32 || (K & 31) != 0 ||
        x.shape(-1) != K || sc.shape(0) != N || sc.shape(1) != (K >> 5)) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(x.size()) / K;
    if (m64 < 1 || m64 > 65535 ||
        static_cast<uint64_t>(m64) * static_cast<uint64_t>(K) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return nullptr;
    }

    int eff_gs = gs_shift;
    if (eff_gs == -1) {
      if ((K % 256) == 0) {
        eff_gs = 8;
      } else if ((K % 128) == 0) {
        eff_gs = 7;
      } else if ((K % 64) == 0) {
        eff_gs = 0;
      } else {
        eff_gs = -2;
      }
    }
    if (eff_gs == 0) {
      if ((K % 64) != 0) return nullptr;
    } else if (eff_gs == 7) {
      if ((K % 128) != 0) return nullptr;
    } else if (eff_gs == 8) {
      if ((K % 256) != 0) return nullptr;
    } else if (eff_gs != -2) {
      return nullptr;
    }

    std::vector<array> inputs;
    inputs.reserve(general_bias ? 4 : 3);
    inputs.push_back(x);
    inputs.push_back(w);
    inputs.push_back(sc);
    if (biases_h) {
      const auto& bi = *reinterpret_cast<array*>(biases_h);
      if (bi.dtype() != float16 || bi.ndim() != 2 || bi.shape(0) != N ||
          bi.shape(1) != (K >> 5)) {
        return nullptr;
      }
      if (general_bias) {
        inputs.push_back(bi);
      }
    }

    Shape out_shape = x.shape();
    out_shape.back() = N;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4Q8PrefillCuda>(
            default_stream(Device::gpu), eff_gs, general_bias),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_q8_prefill_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_qwen4_hc_up_mix_prefill_cuda(
    mlx_array* down_act_h, mlx_array* normed_h,
    mlx_array* w_up_h, mlx_array* s_up_h, mlx_array* b_up_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::qwen4_ops;
    if (!down_act_h || !normed_h || !w_up_h || !s_up_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& down_act = *reinterpret_cast<array*>(down_act_h);
    const auto& normed = *reinterpret_cast<array*>(normed_h);
    const auto& w_up = *reinterpret_cast<array*>(w_up_h);
    const auto& s_up = *reinterpret_cast<array*>(s_up_h);
    if (down_act.dtype() != bfloat16 || down_act.ndim() < 2 ||
        down_act.shape(-1) != kRank || normed.dtype() != bfloat16 ||
        normed.ndim() < 2 || normed.shape(-1) != kW) {
      return nullptr;
    }
    const int64_t m64 = static_cast<int64_t>(down_act.size()) / kRank;
    if (m64 < 1 || m64 > 65535 ||
        normed.size() != static_cast<size_t>(m64) * kW) {
      return nullptr;
    }
    if (w_up.dtype() != uint32 || w_up.ndim() != 2 ||
        w_up.shape(0) != kW || w_up.shape(1) != kRank / 4 ||
        s_up.dtype() != float16 || s_up.ndim() != 2 ||
        s_up.shape(0) != kW || s_up.shape(1) != kRank / 32) {
      return nullptr;
    }
    if (b_up_h) {
      const auto& b_up = *reinterpret_cast<array*>(b_up_h);
      if (b_up.dtype() != float16 || b_up.ndim() != 2 ||
          b_up.shape(0) != kW || b_up.shape(1) != kRank / 32) {
        return nullptr;
      }
    }
    Shape out_shape = down_act.shape();
    out_shape.back() = kH;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Qwen4HcUpMixPrefillCuda>(default_stream(Device::gpu)),
        {down_act, normed, w_up, s_up});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qwen4_hc_up_mix_prefill_cuda: %s\n", e.what());
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

extern "C" mlx_array* mlx_qwen4_qsa_sparse_sdpa_cuda(
    mlx_array* q_h, mlx_array* k_h, mlx_array* v_h,
    mlx_array* iq_h, mlx_array* pooled_k_h, mlx_array* tree_mask_h,
    int32_t kv_len, int32_t num_blocks, float scale, float idx_scale,
    int32_t mask_mode) {
  (void)q_h; (void)k_h; (void)v_h; (void)iq_h; (void)pooled_k_h;
  (void)tree_mask_h; (void)kv_len; (void)num_blocks; (void)scale;
  (void)idx_scale; (void)mask_mode;
  return nullptr;
}

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

extern "C" mlx_array* mlx_qwen4_q8_prefill_cuda(
    mlx_array* x_h, mlx_array* w_h, mlx_array* scales_h, mlx_array* biases_h,
    int32_t gs_shift, bool general_bias) {
  (void)x_h; (void)w_h; (void)scales_h; (void)biases_h;
  (void)gs_shift; (void)general_bias;
  return nullptr;
}

extern "C" mlx_array* mlx_qwen4_hc_up_mix_prefill_cuda(
    mlx_array* down_act_h, mlx_array* normed_h,
    mlx_array* w_up_h, mlx_array* s_up_h, mlx_array* b_up_h) {
  (void)down_act_h; (void)normed_h; (void)w_up_h; (void)s_up_h; (void)b_up_h;
  return nullptr;
}

#endif
