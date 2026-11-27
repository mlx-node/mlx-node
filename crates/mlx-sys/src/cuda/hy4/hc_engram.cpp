#include "cuda/hy4/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::hy4_ops {
namespace {

using namespace mlx_node::hy4_cuda;

constexpr char hy4_hc_engram_source[] =
#include "cuda/hy4_fp4_helpers.cu.inc"
#include "cuda/hy4_hc_engram.cu.inc"
    ;

// ============================================================================
// 1. 4-Stream Hyper-Connections Pre / Head
// ============================================================================
class Hy4HcPreCuda final : public Primitive {
 public:
  Hy4HcPreCuda(
      Stream stream,
      int fn_mode,
      float norm_eps,
      float hc_eps,
      float hc_mag,
      bool head_only,
      bool has_norm)
      : Primitive(stream),
        fn_mode_(fn_mode),
        norm_eps_(norm_eps),
        hc_eps_(hc_eps),
        hc_mag_(hc_mag),
        head_only_(head_only),
        has_norm_(has_norm) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Hy4HcPreCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    size_t idx = 0;
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    array w_fn = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    std::optional<array> s_fn;
    if (fn_mode_ == 0) {
      s_fn.emplace(cu::ensure_row_contiguous_aligned<1>(inputs[idx++], stream(), encoder));
    } else if (fn_mode_ == 1) {
      s_fn.emplace(cu::ensure_row_contiguous_aligned<2>(inputs[idx++], stream(), encoder));
    }
    array scale = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    array base = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    std::optional<array> norm_w;
    if (has_norm_) {
      norm_w.emplace(
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder));
    }

    const int M = static_cast<int>(x.shape(0));
    const int HC = static_cast<int>(x.shape(1));
    const int D = static_cast<int>(x.shape(2));
    const int K = HC * D;
    const int n_gates = head_only_ ? HC : (2 * HC);

    auto& out_h = outputs[0];
    out_h.set_data(cu::malloc_async(out_h.nbytes(), encoder));
    if (!head_only_) {
      outputs[1].set_data(cu::malloc_async(outputs[1].nbytes(), encoder));
    }

    const bool use_tiled_prefill =
        (M >= 17 && K >= 1024 && HC <= 4 && (fn_mode_ == 2 || fn_mode_ == 3));
    const bool use_decode_splitk =
        (!use_tiled_prefill && M <= 16 && K >= 2048 && HC <= 8);

    if (use_tiled_prefill || use_decode_splitk) {
      int num_splits = 1;
      if (use_tiled_prefill) {
        num_splits = (K >= 4096) ? 8 : 4;
      } else if (fn_mode_ <= 1) {
        num_splits = std::min(24, std::max(1, (K >> 4) / 32));
      } else {
        const int n_c8 = K >> 3;
        if (M <= 2) {
          num_splits = std::min(48, std::max(1, n_c8 / 32));
        } else if (M <= 8) {
          num_splits = std::min(24, std::max(1, n_c8 / 64));
        } else {
          num_splits = std::min(16, std::max(1, n_c8 / 64));
        }
      }

      array partials = alloc_temp({M, 1 + n_gates, num_splits}, float32, encoder);

      encoder.set_input_array(x);
      encoder.set_input_array(w_fn);
      if (s_fn) encoder.set_input_array(*s_fn);
      encoder.set_output_array(partials);

      if (use_tiled_prefill) {
        cu::KernelArgs s1_args;
        s1_args.append(x);
        s1_args.append(w_fn);
        s1_args.append(partials);
        s1_args.append<int32_t>(M);
        s1_args.append<int32_t>(HC);
        s1_args.append<int32_t>(D);
        s1_args.append<int32_t>(num_splits);
        static constexpr const char* kTiledEntries[2][2] = {
            {"hy4_hc_pre_tiled_stage1<2, false>", "hy4_hc_pre_tiled_stage1<2, true>"},
            {"hy4_hc_pre_tiled_stage1<3, false>", "hy4_hc_pre_tiled_stage1<3, true>"},
        };
        auto fn1 = cu::jit_kernel(
            encoder.device(), hy4_hc_engram_source,
            kTiledEntries[fn_mode_ - 2][head_only_ ? 1 : 0]);
        encoder.add_kernel_node_raw(
            fn1, dim3((M + 31) / 32, num_splits, 1), dim3(256, 1, 1), {}, 0,
            s1_args.args());
      } else {
        cu::KernelArgs s1_args;
        s1_args.append(x);
        s1_args.append(w_fn);
        if (s_fn) {
          s1_args.append(*s_fn);
        } else {
          s1_args.append<CUdeviceptr>(0);
        }
        s1_args.append(partials);
        s1_args.append<int32_t>(M);
        s1_args.append<int32_t>(HC);
        s1_args.append<int32_t>(D);
        s1_args.append<int32_t>(num_splits);
        static constexpr const char* kSplitEntries[4][2] = {
            {"hy4_hc_pre_splitk_stage1<0, false>", "hy4_hc_pre_splitk_stage1<0, true>"},
            {"hy4_hc_pre_splitk_stage1<1, false>", "hy4_hc_pre_splitk_stage1<1, true>"},
            {"hy4_hc_pre_splitk_stage1<2, false>", "hy4_hc_pre_splitk_stage1<2, true>"},
            {"hy4_hc_pre_splitk_stage1<3, false>", "hy4_hc_pre_splitk_stage1<3, true>"},
        };
        auto fn1 = cu::jit_kernel(
            encoder.device(), hy4_hc_engram_source,
            kSplitEntries[fn_mode_][head_only_ ? 1 : 0]);
        encoder.add_kernel_node_raw(
            fn1, dim3(num_splits, M, 1), dim3(head_only_ ? 128 : 256, 1, 1), {},
            0, s1_args.args());
      }

      encoder.set_input_array(x);
      encoder.set_input_array(partials);
      encoder.set_input_array(scale);
      encoder.set_input_array(base);
      if (norm_w) encoder.set_input_array(*norm_w);
      encoder.set_output_array(out_h);
      if (!head_only_) encoder.set_output_array(outputs[1]);

      cu::KernelArgs s2_args;
      s2_args.append(x);
      s2_args.append(partials);
      s2_args.append(scale);
      s2_args.append(base);
      if (norm_w) {
        s2_args.append(*norm_w);
      } else {
        s2_args.append<CUdeviceptr>(0);
      }
      s2_args.append(out_h);
      if (!head_only_) {
        s2_args.append(outputs[1]);
      } else {
        s2_args.append<CUdeviceptr>(0);
      }
      s2_args.append<int32_t>(M);
      s2_args.append<int32_t>(HC);
      s2_args.append<int32_t>(D);
      s2_args.append<int32_t>(num_splits);
      s2_args.append<float>(norm_eps_);
      s2_args.append<float>(hc_eps_);
      s2_args.append<float>(hc_mag_);
      s2_args.append<int32_t>(has_norm_ ? 1 : 0);
      s2_args.append<int32_t>((has_norm_ && norm_w->dtype() == bfloat16) ? 1 : 0);

      auto fn2 = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source,
          head_only_ ? "hy4_hc_pre_splitk_stage2<true>"
                     : "hy4_hc_pre_splitk_stage2<false>");
      encoder.add_kernel_node_raw(
          fn2, dim3(M, 1, 1), dim3(384, 1, 1), {}, 0, s2_args.args());
      return;
    }

    encoder.set_input_array(x);
    encoder.set_input_array(w_fn);
    if (s_fn) encoder.set_input_array(*s_fn);
    encoder.set_input_array(scale);
    encoder.set_input_array(base);
    if (norm_w) encoder.set_input_array(*norm_w);
    encoder.set_output_array(out_h);

    cu::KernelArgs args;
    args.append(x);
    args.append(w_fn);
    if (s_fn) {
      args.append(*s_fn);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(scale);
    args.append(base);
    if (norm_w) {
      args.append(*norm_w);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_h);
    if (!head_only_) {
      auto& out_post = outputs[1];
      encoder.set_output_array(out_post);
      args.append(out_post);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(M);
    args.append<int32_t>(HC);
    args.append<int32_t>(D);
    args.append<float>(norm_eps_);
    args.append<float>(hc_eps_);
    args.append<float>(hc_mag_);
    args.append<int32_t>(has_norm_ ? 1 : 0);
    args.append<int32_t>((has_norm_ && norm_w->dtype() == bfloat16) ? 1 : 0);

    static constexpr const char* kEntries[4][2] = {
        {"hy4_hc_pre_cuda<0, false>", "hy4_hc_pre_cuda<0, true>"},
        {"hy4_hc_pre_cuda<1, false>", "hy4_hc_pre_cuda<1, true>"},
        {"hy4_hc_pre_cuda<2, false>", "hy4_hc_pre_cuda<2, true>"},
        {"hy4_hc_pre_cuda<3, false>", "hy4_hc_pre_cuda<3, true>"},
    };
    auto fn = cu::jit_kernel(
        encoder.device(), hy4_hc_engram_source,
        kEntries[fn_mode_][head_only_ ? 1 : 0]);
    encoder.add_kernel_node_raw(
        fn, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4HcPreCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4HcPreCuda*>(&other);
    return rhs && rhs->fn_mode_ == fn_mode_ && rhs->norm_eps_ == norm_eps_ &&
        rhs->hc_eps_ == hc_eps_ && rhs->hc_mag_ == hc_mag_ &&
        rhs->head_only_ == head_only_ && rhs->has_norm_ == has_norm_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int M = inputs[0].shape(0);
    const int HC = inputs[0].shape(1);
    const int D = inputs[0].shape(2);
    if (head_only_) {
      return {Shape{M, D}};
    }
    return {Shape{M, D}, Shape{M, HC}};
  }

 private:
  int fn_mode_;
  float norm_eps_;
  float hc_eps_;
  float hc_mag_;
  bool head_only_;
  bool has_norm_;
};

// ============================================================================
// 2. 4-Stream Hyper-Connections Post
// ============================================================================
class Hy4HcPostCuda final : public UnaryPrimitive {
 public:
  Hy4HcPostCuda(Stream stream, float norm_eps, bool has_norm)
      : UnaryPrimitive(stream), norm_eps_(norm_eps), has_norm_(has_norm) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4HcPostCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array y = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array post = cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder);
    std::optional<array> post_norm_w;
    if (has_norm_) {
      post_norm_w.emplace(
          cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder));
    }

    const int M = static_cast<int>(x.shape(0));
    const int HC = static_cast<int>(x.shape(1));
    const int D = static_cast<int>(x.shape(2));

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_input_array(y);
    encoder.set_input_array(post);
    if (post_norm_w) encoder.set_input_array(*post_norm_w);
    encoder.set_output_array(out);

    const bool use_multi_cta = (M <= 16 && D >= 1024);
    if (use_multi_cta) {
      const int num_ctas = std::min(12, std::max(1, (D >> 3) / 32));
      cu::KernelArgs args;
      args.append(x);
      args.append(y);
      args.append(post);
      if (post_norm_w) {
        args.append(*post_norm_w);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append(out);
      args.append<int32_t>(M);
      args.append<int32_t>(HC);
      args.append<int32_t>(D);
      args.append<int32_t>(num_ctas);
      args.append<float>(norm_eps_);
      args.append<int32_t>(has_norm_ ? 1 : 0);
      args.append<int32_t>(
          (has_norm_ && post_norm_w->dtype() == bfloat16) ? 1 : 0);

      auto fn = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source, "hy4_hc_post_multi_cuda");
      encoder.add_kernel_node_raw(
          fn, dim3(num_ctas, M, 1), dim3(128, 1, 1), {}, 0, args.args());
      return;
    }

    cu::KernelArgs args;
    args.append(x);
    args.append(y);
    args.append(post);
    if (post_norm_w) {
      args.append(*post_norm_w);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(HC);
    args.append<int32_t>(D);
    args.append<float>(norm_eps_);
    args.append<int32_t>(has_norm_ ? 1 : 0);
    args.append<int32_t>(
        (has_norm_ && post_norm_w->dtype() == bfloat16) ? 1 : 0);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_hc_engram_source, "hy4_hc_post_cuda");
    encoder.add_kernel_node_raw(
        fn, dim3(M, 1, 1), dim3(384, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4HcPostCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4HcPostCuda*>(&other);
    return rhs && rhs->norm_eps_ == norm_eps_ && rhs->has_norm_ == has_norm_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  float norm_eps_;
  bool has_norm_;
};

// ============================================================================
// 2b. Fused 4-Stream Hyper-Connections Post + Next Pre
// ============================================================================
class Hy4HcPostPreCuda final : public Primitive {
 public:
  Hy4HcPostPreCuda(
      Stream stream,
      int fn_mode,
      float norm_eps,
      float hc_eps,
      float hc_mag,
      bool has_prev_norm,
      bool has_next_norm)
      : Primitive(stream),
        fn_mode_(fn_mode),
        norm_eps_(norm_eps),
        hc_eps_(hc_eps),
        hc_mag_(hc_mag),
        has_prev_norm_(has_prev_norm),
        has_next_norm_(has_next_norm) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Hy4HcPostPreCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    size_t idx = 0;
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    array y = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    array prev_post =
        cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    std::optional<array> prev_norm_w;
    if (has_prev_norm_) {
      prev_norm_w.emplace(
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder));
    }
    array w_fn = cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder);
    std::optional<array> s_fn;
    if (fn_mode_ == 0) {
      s_fn.emplace(cu::ensure_row_contiguous_aligned<1>(inputs[idx++], stream(), encoder));
    } else if (fn_mode_ == 1) {
      s_fn.emplace(cu::ensure_row_contiguous_aligned<2>(inputs[idx++], stream(), encoder));
    }
    array scale = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    array base = cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
    std::optional<array> next_norm_w;
    if (has_next_norm_) {
      next_norm_w.emplace(
          cu::ensure_row_contiguous_aligned<16>(inputs[idx++], stream(), encoder));
    }

    const int M = static_cast<int>(x.shape(0));
    const int HC = static_cast<int>(x.shape(1));
    const int D = static_cast<int>(x.shape(2));
    const int K = HC * D;
    const int n_gates = 2 * HC;

    auto& out_x = outputs[0];
    auto& out_h = outputs[1];
    auto& out_post = outputs[2];
    out_x.set_data(cu::malloc_async(out_x.nbytes(), encoder));
    out_h.set_data(cu::malloc_async(out_h.nbytes(), encoder));
    out_post.set_data(cu::malloc_async(out_post.nbytes(), encoder));

    const bool use_fused_splitk =
        (M <= 16 && K >= 2048 && HC <= 8 && (fn_mode_ >= 2 || (D & 15) == 0));

    if (use_fused_splitk) {
      int num_splits = 1;
      if (fn_mode_ <= 1) {
        num_splits = std::min(24, std::max(1, (K >> 4) / 32));
      } else {
        const int n_c8 = K >> 3;
        if (M == 1) {
          num_splits = std::min(48, std::max(1, n_c8 / 32));
        } else {
          num_splits = std::min(24, std::max(1, n_c8 / 64));
        }
      }

      array partials = alloc_temp({M, 1 + n_gates, num_splits}, float32, encoder);

      encoder.set_input_array(x);
      encoder.set_input_array(y);
      encoder.set_input_array(prev_post);
      if (prev_norm_w) encoder.set_input_array(*prev_norm_w);
      encoder.set_input_array(w_fn);
      if (s_fn) encoder.set_input_array(*s_fn);
      encoder.set_output_array(out_x);
      encoder.set_output_array(partials);

      cu::KernelArgs s1_args;
      s1_args.append(x);
      s1_args.append(y);
      s1_args.append(prev_post);
      if (prev_norm_w) {
        s1_args.append(*prev_norm_w);
      } else {
        s1_args.append<CUdeviceptr>(0);
      }
      s1_args.append(w_fn);
      if (s_fn) {
        s1_args.append(*s_fn);
      } else {
        s1_args.append<CUdeviceptr>(0);
      }
      s1_args.append(out_x);
      s1_args.append(partials);
      s1_args.append<int32_t>(M);
      s1_args.append<int32_t>(HC);
      s1_args.append<int32_t>(D);
      s1_args.append<int32_t>(num_splits);
      s1_args.append<float>(norm_eps_);
      s1_args.append<int32_t>(has_prev_norm_ ? 1 : 0);
      s1_args.append<int32_t>(
          (has_prev_norm_ && prev_norm_w->dtype() == bfloat16) ? 1 : 0);

      static constexpr const char* kPostPreEntries[4] = {
          "hy4_hc_post_pre_stage1<0>",
          "hy4_hc_post_pre_stage1<1>",
          "hy4_hc_post_pre_stage1<2>",
          "hy4_hc_post_pre_stage1<3>",
      };
      auto fn1 = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source, kPostPreEntries[fn_mode_]);
      encoder.add_kernel_node_raw(
          fn1, dim3(num_splits, M, 1), dim3(256, 1, 1), {}, 0, s1_args.args());

      encoder.set_input_array(out_x);
      encoder.set_input_array(partials);
      encoder.set_input_array(scale);
      encoder.set_input_array(base);
      if (next_norm_w) encoder.set_input_array(*next_norm_w);
      encoder.set_output_array(out_h);
      encoder.set_output_array(out_post);

      cu::KernelArgs s2_args;
      s2_args.append(out_x);
      s2_args.append(partials);
      s2_args.append(scale);
      s2_args.append(base);
      if (next_norm_w) {
        s2_args.append(*next_norm_w);
      } else {
        s2_args.append<CUdeviceptr>(0);
      }
      s2_args.append(out_h);
      s2_args.append(out_post);
      s2_args.append<int32_t>(M);
      s2_args.append<int32_t>(HC);
      s2_args.append<int32_t>(D);
      s2_args.append<int32_t>(num_splits);
      s2_args.append<float>(norm_eps_);
      s2_args.append<float>(hc_eps_);
      s2_args.append<float>(hc_mag_);
      s2_args.append<int32_t>(has_next_norm_ ? 1 : 0);
      s2_args.append<int32_t>(
          (has_next_norm_ && next_norm_w->dtype() == bfloat16) ? 1 : 0);

      auto fn2 = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source, "hy4_hc_pre_splitk_stage2<false>");
      encoder.add_kernel_node_raw(
          fn2, dim3(M, 1, 1), dim3(384, 1, 1), {}, 0, s2_args.args());
      return;
    }

    // Fallback for prefill (M >= 17), small K < 2048, or unaligned D with FP4/Q8:
    // Step 1: Post-norm + residual scatter -> out_x
    encoder.set_input_array(x);
    encoder.set_input_array(y);
    encoder.set_input_array(prev_post);
    if (prev_norm_w) encoder.set_input_array(*prev_norm_w);
    encoder.set_output_array(out_x);

    if (M <= 16 && D >= 1024) {
      const int num_ctas = std::min(12, std::max(1, (D >> 3) / 32));
      cu::KernelArgs p_args;
      p_args.append(x);
      p_args.append(y);
      p_args.append(prev_post);
      if (prev_norm_w) {
        p_args.append(*prev_norm_w);
      } else {
        p_args.append<CUdeviceptr>(0);
      }
      p_args.append(out_x);
      p_args.append<int32_t>(M);
      p_args.append<int32_t>(HC);
      p_args.append<int32_t>(D);
      p_args.append<int32_t>(num_ctas);
      p_args.append<float>(norm_eps_);
      p_args.append<int32_t>(has_prev_norm_ ? 1 : 0);
      p_args.append<int32_t>(
          (has_prev_norm_ && prev_norm_w->dtype() == bfloat16) ? 1 : 0);
      auto fnp = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source, "hy4_hc_post_multi_cuda");
      encoder.add_kernel_node_raw(
          fnp, dim3(num_ctas, M, 1), dim3(128, 1, 1), {}, 0, p_args.args());
    } else {
      cu::KernelArgs p_args;
      p_args.append(x);
      p_args.append(y);
      p_args.append(prev_post);
      if (prev_norm_w) {
        p_args.append(*prev_norm_w);
      } else {
        p_args.append<CUdeviceptr>(0);
      }
      p_args.append(out_x);
      p_args.append<int32_t>(M);
      p_args.append<int32_t>(HC);
      p_args.append<int32_t>(D);
      p_args.append<float>(norm_eps_);
      p_args.append<int32_t>(has_prev_norm_ ? 1 : 0);
      p_args.append<int32_t>(
          (has_prev_norm_ && prev_norm_w->dtype() == bfloat16) ? 1 : 0);
      auto fnp = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source, "hy4_hc_post_cuda");
      encoder.add_kernel_node_raw(
          fnp, dim3(M, 1, 1), dim3(384, 1, 1), {}, 0, p_args.args());
    }

    // Step 2: Pre-gate + mix + next norm reading out_x -> (out_h, out_post)
    const bool use_tiled_prefill =
        (M >= 17 && K >= 1024 && HC <= 4 && (fn_mode_ == 2 || fn_mode_ == 3));
    const bool use_decode_splitk =
        (!use_tiled_prefill && M <= 16 && K >= 2048 && HC <= 8);

    if (use_tiled_prefill || use_decode_splitk) {
      int num_splits = 1;
      if (use_tiled_prefill) {
        num_splits = (K >= 4096) ? 8 : 4;
      } else if (fn_mode_ <= 1) {
        num_splits = std::min(24, std::max(1, (K >> 4) / 32));
      } else {
        const int n_c8 = K >> 3;
        num_splits = (M <= 2) ? std::min(48, std::max(1, n_c8 / 32))
                              : std::min(24, std::max(1, n_c8 / 64));
      }
      array partials = alloc_temp({M, 1 + n_gates, num_splits}, float32, encoder);

      encoder.set_input_array(out_x);
      encoder.set_input_array(w_fn);
      if (s_fn) encoder.set_input_array(*s_fn);
      encoder.set_output_array(partials);

      if (use_tiled_prefill) {
        cu::KernelArgs s1_args;
        s1_args.append(out_x);
        s1_args.append(w_fn);
        s1_args.append(partials);
        s1_args.append<int32_t>(M);
        s1_args.append<int32_t>(HC);
        s1_args.append<int32_t>(D);
        s1_args.append<int32_t>(num_splits);
        static constexpr const char* kTiledEntries[2] = {
            "hy4_hc_pre_tiled_stage1<2, false>",
            "hy4_hc_pre_tiled_stage1<3, false>",
        };
        auto fn1 = cu::jit_kernel(
            encoder.device(), hy4_hc_engram_source, kTiledEntries[fn_mode_ - 2]);
        encoder.add_kernel_node_raw(
            fn1, dim3((M + 31) / 32, num_splits, 1), dim3(256, 1, 1), {}, 0,
            s1_args.args());
      } else {
        cu::KernelArgs s1_args;
        s1_args.append(out_x);
        s1_args.append(w_fn);
        if (s_fn) {
          s1_args.append(*s_fn);
        } else {
          s1_args.append<CUdeviceptr>(0);
        }
        s1_args.append(partials);
        s1_args.append<int32_t>(M);
        s1_args.append<int32_t>(HC);
        s1_args.append<int32_t>(D);
        s1_args.append<int32_t>(num_splits);
        static constexpr const char* kSplitEntries[4] = {
            "hy4_hc_pre_splitk_stage1<0, false>",
            "hy4_hc_pre_splitk_stage1<1, false>",
            "hy4_hc_pre_splitk_stage1<2, false>",
            "hy4_hc_pre_splitk_stage1<3, false>",
        };
        auto fn1 = cu::jit_kernel(
            encoder.device(), hy4_hc_engram_source, kSplitEntries[fn_mode_]);
        encoder.add_kernel_node_raw(
            fn1, dim3(num_splits, M, 1), dim3(256, 1, 1), {}, 0, s1_args.args());
      }

      encoder.set_input_array(out_x);
      encoder.set_input_array(partials);
      encoder.set_input_array(scale);
      encoder.set_input_array(base);
      if (next_norm_w) encoder.set_input_array(*next_norm_w);
      encoder.set_output_array(out_h);
      encoder.set_output_array(out_post);

      cu::KernelArgs s2_args;
      s2_args.append(out_x);
      s2_args.append(partials);
      s2_args.append(scale);
      s2_args.append(base);
      if (next_norm_w) {
        s2_args.append(*next_norm_w);
      } else {
        s2_args.append<CUdeviceptr>(0);
      }
      s2_args.append(out_h);
      s2_args.append(out_post);
      s2_args.append<int32_t>(M);
      s2_args.append<int32_t>(HC);
      s2_args.append<int32_t>(D);
      s2_args.append<int32_t>(num_splits);
      s2_args.append<float>(norm_eps_);
      s2_args.append<float>(hc_eps_);
      s2_args.append<float>(hc_mag_);
      s2_args.append<int32_t>(has_next_norm_ ? 1 : 0);
      s2_args.append<int32_t>(
          (has_next_norm_ && next_norm_w->dtype() == bfloat16) ? 1 : 0);

      auto fn2 = cu::jit_kernel(
          encoder.device(), hy4_hc_engram_source, "hy4_hc_pre_splitk_stage2<false>");
      encoder.add_kernel_node_raw(
          fn2, dim3(M, 1, 1), dim3(384, 1, 1), {}, 0, s2_args.args());
      return;
    }

    encoder.set_input_array(out_x);
    encoder.set_input_array(w_fn);
    if (s_fn) encoder.set_input_array(*s_fn);
    encoder.set_input_array(scale);
    encoder.set_input_array(base);
    if (next_norm_w) encoder.set_input_array(*next_norm_w);
    encoder.set_output_array(out_h);
    encoder.set_output_array(out_post);

    cu::KernelArgs pre_args;
    pre_args.append(out_x);
    pre_args.append(w_fn);
    if (s_fn) {
      pre_args.append(*s_fn);
    } else {
      pre_args.append<CUdeviceptr>(0);
    }
    pre_args.append(scale);
    pre_args.append(base);
    if (next_norm_w) {
      pre_args.append(*next_norm_w);
    } else {
      pre_args.append<CUdeviceptr>(0);
    }
    pre_args.append(out_h);
    pre_args.append(out_post);
    pre_args.append<int32_t>(M);
    pre_args.append<int32_t>(HC);
    pre_args.append<int32_t>(D);
    pre_args.append<float>(norm_eps_);
    pre_args.append<float>(hc_eps_);
    pre_args.append<float>(hc_mag_);
    pre_args.append<int32_t>(has_next_norm_ ? 1 : 0);
    pre_args.append<int32_t>(
        (has_next_norm_ && next_norm_w->dtype() == bfloat16) ? 1 : 0);

    static constexpr const char* kPreEntries[4] = {
        "hy4_hc_pre_cuda<0, false>",
        "hy4_hc_pre_cuda<1, false>",
        "hy4_hc_pre_cuda<2, false>",
        "hy4_hc_pre_cuda<3, false>",
    };
    auto fn_pre = cu::jit_kernel(
        encoder.device(), hy4_hc_engram_source, kPreEntries[fn_mode_]);
    encoder.add_kernel_node_raw(
        fn_pre, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, pre_args.args());
  }

  DEFINE_NAME(Hy4HcPostPreCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4HcPostPreCuda*>(&other);
    return rhs && rhs->fn_mode_ == fn_mode_ && rhs->norm_eps_ == norm_eps_ &&
        rhs->hc_eps_ == hc_eps_ && rhs->hc_mag_ == hc_mag_ &&
        rhs->has_prev_norm_ == has_prev_norm_ &&
        rhs->has_next_norm_ == has_next_norm_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int M = inputs[0].shape(0);
    const int HC = inputs[0].shape(1);
    const int D = inputs[0].shape(2);
    return {Shape{M, HC, D}, Shape{M, D}, Shape{M, HC}};
  }

 private:
  int fn_mode_;
  float norm_eps_;
  float hc_eps_;
  float hc_mag_;
  bool has_prev_norm_;
  bool has_next_norm_;
};

// ============================================================================
// 3. Engram Lookup Norm
// ============================================================================
class Hy4EngramLookupNormCuda final : public UnaryPrimitive {
 public:
  explicit Hy4EngramLookupNormCuda(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4EngramLookupNormCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array raw_rows =
        cu::ensure_row_contiguous_aligned<4>(inputs[0], stream(), encoder);
    array lookup_norm =
        cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);

    const int M = static_cast<int>(raw_rows.shape(0));
    const int n_cols = static_cast<int>(raw_rows.shape(1));
    const int e_hd = static_cast<int>(raw_rows.shape(2));

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(raw_rows);
    encoder.set_input_array(lookup_norm);
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(raw_rows);
    args.append(lookup_norm);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(n_cols);
    args.append<int32_t>(e_hd);
    args.append<float>(kEngramEps);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_hc_engram_source, "hy4_engram_lookup_norm_cuda");
    encoder.add_kernel_node_raw(
        fn, dim3(n_cols, M, 1), dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4EngramLookupNormCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Hy4EngramLookupNormCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int M = inputs[0].shape(0);
    const int n_cols = inputs[0].shape(1);
    const int e_hd = inputs[0].shape(2);
    return {Shape{M, n_cols * e_hd}};
  }
};

// ============================================================================
// 4. Engram Gate + Residual Add
// ============================================================================
class Hy4EngramGateAddCuda final : public UnaryPrimitive {
 public:
  Hy4EngramGateAddCuda(Stream stream, bool has_mask)
      : UnaryPrimitive(stream), has_mask_(has_mask) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4EngramGateAddCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array kv = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array kv_b = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array gate_w = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array pre_gamma = cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
    array hc_scale = cu::ensure_row_contiguous_aligned<16>(inputs[5], stream(), encoder);
    std::optional<array> keep_mask;
    if (has_mask_) {
      keep_mask.emplace(
          cu::ensure_row_contiguous_aligned<1>(inputs[6], stream(), encoder));
    }

    const int M = static_cast<int>(x.shape(0));
    const int HC = static_cast<int>(x.shape(1));
    const int D = static_cast<int>(x.shape(2));

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_input_array(kv);
    encoder.set_input_array(kv_b);
    encoder.set_input_array(gate_w);
    encoder.set_input_array(pre_gamma);
    encoder.set_input_array(hc_scale);
    if (keep_mask) encoder.set_input_array(*keep_mask);
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(x);
    args.append(kv);
    args.append(kv_b);
    args.append(gate_w);
    args.append(pre_gamma);
    args.append(hc_scale);
    if (keep_mask) {
      args.append(*keep_mask);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(HC);
    args.append<int32_t>(D);
    args.append<float>(kEngramEps);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_hc_engram_source, "hy4_engram_gate_add_cuda");
    encoder.add_kernel_node_raw(
        fn, dim3(M, 1, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4EngramGateAddCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4EngramGateAddCuda*>(&other);
    return rhs && rhs->has_mask_ == has_mask_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  bool has_mask_;
};

// ============================================================================
// 5. MTP eh_norm_concat
// ============================================================================
class Hy4MtpEhNormConcatCuda final : public UnaryPrimitive {
 public:
  Hy4MtpEhNormConcatCuda(Stream stream, float norm_eps)
      : UnaryPrimitive(stream), norm_eps_(norm_eps) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Hy4MtpEhNormConcatCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array emb = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array hidden = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array enorm = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array hnorm = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);

    const int M = static_cast<int>(emb.shape(0));
    const int D = static_cast<int>(emb.shape(1));

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(emb);
    encoder.set_input_array(hidden);
    encoder.set_input_array(enorm);
    encoder.set_input_array(hnorm);
    encoder.set_output_array(out);

    cu::KernelArgs args;
    args.append(emb);
    args.append(hidden);
    args.append(enorm);
    args.append(hnorm);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(D);
    args.append<float>(norm_eps_);

    auto fn = cu::jit_kernel(
        encoder.device(), hy4_hc_engram_source, "hy4_mtp_eh_norm_concat_cuda");
    encoder.add_kernel_node_raw(
        fn, dim3(2, M, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Hy4MtpEhNormConcatCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Hy4MtpEhNormConcatCuda*>(&other);
    return rhs && rhs->norm_eps_ == norm_eps_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), inputs[0].shape(1) * 2}};
  }

 private:
  float norm_eps_;
};

}  // namespace
}  // namespace mlx::core::hy4_ops

extern "C" bool mlx_hy4_hc_pre_cuda(
    mlx_array* x_h,
    mlx_array* w_fn_h,
    mlx_array* s_fn_h,
    mlx_array* scale_h,
    mlx_array* base_h,
    mlx_array* norm_w_h,
    int32_t fn_mode,
    float norm_eps,
    float hc_eps,
    float hc_mag,
    bool head_only,
    mlx_array** out_h,
    mlx_array** out_post) {
  if (out_h) *out_h = nullptr;
  if (out_post) *out_post = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!x_h || !w_fn_h || !scale_h || !base_h || !out_h ||
        (!head_only && !out_post) ||
        fn_mode < 0 || fn_mode > 3 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }

    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& w_fn = *reinterpret_cast<array*>(w_fn_h);
    const auto& scale = *reinterpret_cast<array*>(scale_h);
    const auto& base = *reinterpret_cast<array*>(base_h);
    if (x.dtype() != bfloat16 || x.ndim() != 3 ||
        scale.dtype() != float32 || base.dtype() != float32) {
      return false;
    }

    const int64_t M = x.shape(0);
    const int64_t HC = x.shape(1);
    const int64_t D = x.shape(2);
    if (M < 1 || M > 65535 || HC < 1 || HC > 8 || D < 32 || (D & 7) != 0) {
      return false;
    }

    const int64_t K = HC * D;
    const int64_t n_gates = head_only ? HC : (2 * HC);
    if (static_cast<int64_t>(scale.size()) < (head_only ? 1 : 2) ||
        static_cast<int64_t>(base.size()) != n_gates) {
      return false;
    }

    std::vector<array> inputs;
    inputs.reserve(6);
    inputs.push_back(x);

    if (fn_mode == 0) {
      if (!s_fn_h || (K & 15) != 0) return false;
      const auto& s_fn = *reinterpret_cast<array*>(s_fn_h);
      if (w_fn.dtype() != uint32 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K / 8 ||
          s_fn.dtype() != uint8 || s_fn.ndim() != 2 ||
          s_fn.shape(0) != n_gates || s_fn.shape(1) != K / 16) {
        return false;
      }
      inputs.push_back(w_fn);
      inputs.push_back(s_fn);
    } else if (fn_mode == 1) {
      if (!s_fn_h || (K & 31) != 0) return false;
      const auto& s_fn = *reinterpret_cast<array*>(s_fn_h);
      if (w_fn.dtype() != uint32 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K / 4 ||
          s_fn.dtype() != float16 || s_fn.ndim() != 2 ||
          s_fn.shape(0) != n_gates || s_fn.shape(1) != K / 32) {
        return false;
      }
      inputs.push_back(w_fn);
      inputs.push_back(s_fn);
    } else if (fn_mode == 2) {
      if (s_fn_h != nullptr || w_fn.dtype() != float32 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K) {
        return false;
      }
      inputs.push_back(w_fn);
    } else {
      if (s_fn_h != nullptr || w_fn.dtype() != bfloat16 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K) {
        return false;
      }
      inputs.push_back(w_fn);
    }

    inputs.push_back(scale);
    inputs.push_back(base);

    const bool has_norm = (norm_w_h != nullptr);
    if (has_norm) {
      const auto& nw = *reinterpret_cast<array*>(norm_w_h);
      if ((nw.dtype() != float32 && nw.dtype() != bfloat16) ||
          nw.ndim() != 1 || nw.shape(0) != D) {
        return false;
      }
      inputs.push_back(nw);
    }

    auto stream = default_stream(Device::gpu);
    auto prim = std::make_shared<Hy4HcPreCuda>(
        stream, fn_mode, norm_eps, hc_eps, hc_mag, head_only, has_norm);
    if (head_only) {
      auto res = array(Shape{static_cast<int>(M), static_cast<int>(D)},
                       bfloat16, std::move(prim), std::move(inputs));
      *out_h = reinterpret_cast<mlx_array*>(new array(std::move(res)));
    } else {
      auto results = array::make_arrays(
          {Shape{static_cast<int>(M), static_cast<int>(D)},
           Shape{static_cast<int>(M), static_cast<int>(HC)}},
          {bfloat16, float32}, std::move(prim), std::move(inputs));
      *out_h = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
      *out_post = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_hc_pre_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" mlx_array* mlx_hy4_hc_head_cuda(
    mlx_array* x_h,
    mlx_array* w_fn_h,
    mlx_array* s_fn_h,
    mlx_array* scale_h,
    mlx_array* base_h,
    mlx_array* output_norm_h,
    int32_t fn_mode,
    float norm_eps,
    float hc_eps) {
  mlx_array* out_h = nullptr;
  if (!mlx_hy4_hc_pre_cuda(
          x_h, w_fn_h, s_fn_h, scale_h, base_h, output_norm_h,
          fn_mode, norm_eps, hc_eps, 0.0f, true, &out_h, nullptr)) {
    return nullptr;
  }
  return out_h;
}

extern "C" mlx_array* mlx_hy4_hc_post_cuda(
    mlx_array* x_h,
    mlx_array* y_h,
    mlx_array* post_h,
    mlx_array* post_norm_w_h,
    float norm_eps) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!x_h || !y_h || !post_h || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& y = *reinterpret_cast<array*>(y_h);
    const auto& post = *reinterpret_cast<array*>(post_h);
    if (x.dtype() != bfloat16 || y.dtype() != bfloat16 || post.dtype() != float32 ||
        x.ndim() != 3 || y.ndim() != 2 || post.ndim() != 2) {
      return nullptr;
    }
    const int64_t M = x.shape(0);
    const int64_t HC = x.shape(1);
    const int64_t D = x.shape(2);
    if (M < 1 || M > 65535 || HC < 1 || HC > 8 || D < 32 || (D & 7) != 0 ||
        y.shape(0) != M || y.shape(1) != D ||
        post.shape(0) != M || post.shape(1) != HC) {
      return nullptr;
    }
    std::vector<array> inputs{x, y, post};
    const bool has_norm = (post_norm_w_h != nullptr);
    if (has_norm) {
      const auto& pnw = *reinterpret_cast<array*>(post_norm_w_h);
      if ((pnw.dtype() != float32 && pnw.dtype() != bfloat16) ||
          pnw.ndim() != 1 || pnw.shape(0) != D) {
        return nullptr;
      }
      inputs.push_back(pnw);
    }
    auto res = array(
        x.shape(), bfloat16,
        std::make_shared<Hy4HcPostCuda>(
            default_stream(Device::gpu), norm_eps, has_norm),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_hc_post_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" bool mlx_hy4_hc_post_pre_cuda(
    mlx_array* x_h,
    mlx_array* y_h,
    mlx_array* prev_post_h,
    mlx_array* prev_post_norm_w_h,
    mlx_array* w_fn_h,
    mlx_array* s_fn_h,
    mlx_array* scale_h,
    mlx_array* base_h,
    mlx_array* next_norm_w_h,
    int32_t fn_mode,
    float norm_eps,
    float hc_eps,
    float hc_mag,
    mlx_array** out_x,
    mlx_array** out_h,
    mlx_array** out_post) {
  if (out_x) *out_x = nullptr;
  if (out_h) *out_h = nullptr;
  if (out_post) *out_post = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!x_h || !y_h || !prev_post_h || !w_fn_h || !scale_h || !base_h ||
        !out_x || !out_h || !out_post ||
        fn_mode < 0 || fn_mode > 3 ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return false;
    }

    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& y = *reinterpret_cast<array*>(y_h);
    const auto& prev_post = *reinterpret_cast<array*>(prev_post_h);
    const auto& w_fn = *reinterpret_cast<array*>(w_fn_h);
    const auto& scale = *reinterpret_cast<array*>(scale_h);
    const auto& base = *reinterpret_cast<array*>(base_h);
    if (x.dtype() != bfloat16 || y.dtype() != bfloat16 ||
        prev_post.dtype() != float32 || scale.dtype() != float32 ||
        base.dtype() != float32 ||
        x.ndim() != 3 || y.ndim() != 2 || prev_post.ndim() != 2) {
      return false;
    }

    const int64_t M = x.shape(0);
    const int64_t HC = x.shape(1);
    const int64_t D = x.shape(2);
    if (M < 1 || M > 65535 || HC < 1 || HC > 8 || D < 32 || (D & 7) != 0 ||
        y.shape(0) != M || y.shape(1) != D ||
        prev_post.shape(0) != M || prev_post.shape(1) != HC) {
      return false;
    }

    const int64_t K = HC * D;
    const int64_t n_gates = 2 * HC;
    if (static_cast<int64_t>(scale.size()) < 2 ||
        static_cast<int64_t>(base.size()) != n_gates) {
      return false;
    }

    std::vector<array> inputs;
    inputs.reserve(9);
    inputs.push_back(x);
    inputs.push_back(y);
    inputs.push_back(prev_post);

    const bool has_prev_norm = (prev_post_norm_w_h != nullptr);
    if (has_prev_norm) {
      const auto& pnw = *reinterpret_cast<array*>(prev_post_norm_w_h);
      if ((pnw.dtype() != float32 && pnw.dtype() != bfloat16) ||
          pnw.ndim() != 1 || pnw.shape(0) != D) {
        return false;
      }
      inputs.push_back(pnw);
    }

    if (fn_mode == 0) {
      if (!s_fn_h || (K & 15) != 0) return false;
      const auto& s_fn = *reinterpret_cast<array*>(s_fn_h);
      if (w_fn.dtype() != uint32 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K / 8 ||
          s_fn.dtype() != uint8 || s_fn.ndim() != 2 ||
          s_fn.shape(0) != n_gates || s_fn.shape(1) != K / 16) {
        return false;
      }
      inputs.push_back(w_fn);
      inputs.push_back(s_fn);
    } else if (fn_mode == 1) {
      if (!s_fn_h || (K & 31) != 0) return false;
      const auto& s_fn = *reinterpret_cast<array*>(s_fn_h);
      if (w_fn.dtype() != uint32 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K / 4 ||
          s_fn.dtype() != float16 || s_fn.ndim() != 2 ||
          s_fn.shape(0) != n_gates || s_fn.shape(1) != K / 32) {
        return false;
      }
      inputs.push_back(w_fn);
      inputs.push_back(s_fn);
    } else if (fn_mode == 2) {
      if (s_fn_h != nullptr || w_fn.dtype() != float32 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K) {
        return false;
      }
      inputs.push_back(w_fn);
    } else {
      if (s_fn_h != nullptr || w_fn.dtype() != bfloat16 || w_fn.ndim() != 2 ||
          w_fn.shape(0) != n_gates || w_fn.shape(1) != K) {
        return false;
      }
      inputs.push_back(w_fn);
    }

    inputs.push_back(scale);
    inputs.push_back(base);

    const bool has_next_norm = (next_norm_w_h != nullptr);
    if (has_next_norm) {
      const auto& nnw = *reinterpret_cast<array*>(next_norm_w_h);
      if ((nnw.dtype() != float32 && nnw.dtype() != bfloat16) ||
          nnw.ndim() != 1 || nnw.shape(0) != D) {
        return false;
      }
      inputs.push_back(nnw);
    }

    auto stream = default_stream(Device::gpu);
    auto prim = std::make_shared<Hy4HcPostPreCuda>(
        stream, fn_mode, norm_eps, hc_eps, hc_mag, has_prev_norm, has_next_norm);
    auto results = array::make_arrays(
        {Shape{static_cast<int>(M), static_cast<int>(HC), static_cast<int>(D)},
         Shape{static_cast<int>(M), static_cast<int>(D)},
         Shape{static_cast<int>(M), static_cast<int>(HC)}},
        {bfloat16, bfloat16, float32}, std::move(prim), std::move(inputs));
    *out_x = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_h = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
    *out_post = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_hc_post_pre_cuda: %s\n", e.what());
    return false;
  }
}

extern "C" mlx_array* mlx_hy4_engram_lookup_norm_cuda(
    mlx_array* raw_rows_h,
    mlx_array* lookup_norm_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!raw_rows_h || !lookup_norm_h || default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& raw = *reinterpret_cast<array*>(raw_rows_h);
    const auto& ln = *reinterpret_cast<array*>(lookup_norm_h);
    if (raw.dtype() != float32 || ln.dtype() != float32 ||
        raw.ndim() != 3 || ln.ndim() != 2) {
      return nullptr;
    }
    const int64_t M = raw.shape(0);
    const int64_t n_cols = raw.shape(1);
    const int64_t e_hd = raw.shape(2);
    if (M < 1 || M > 65535 || n_cols < 1 || n_cols > 65535 || e_hd < 1 ||
        ln.shape(0) != n_cols || ln.shape(1) != e_hd) {
      return nullptr;
    }
    auto res = array(
        Shape{static_cast<int>(M), static_cast<int>(n_cols * e_hd)}, bfloat16,
        std::make_shared<Hy4EngramLookupNormCuda>(default_stream(Device::gpu)),
        {raw, ln});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_engram_lookup_norm_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_engram_gate_add_cuda(
    mlx_array* x_h,
    mlx_array* kv_h,
    mlx_array* kv_b_h,
    mlx_array* gate_w_h,
    mlx_array* pre_gamma_h,
    mlx_array* hc_scale_h,
    mlx_array* keep_mask_h) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!x_h || !kv_h || !kv_b_h || !gate_w_h || !pre_gamma_h || !hc_scale_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& x = *reinterpret_cast<array*>(x_h);
    const auto& kv = *reinterpret_cast<array*>(kv_h);
    const auto& kv_b = *reinterpret_cast<array*>(kv_b_h);
    const auto& gate_w = *reinterpret_cast<array*>(gate_w_h);
    const auto& pre_gamma = *reinterpret_cast<array*>(pre_gamma_h);
    const auto& hc_scale = *reinterpret_cast<array*>(hc_scale_h);
    if (x.dtype() != bfloat16 || kv.dtype() != bfloat16 ||
        kv_b.dtype() != float32 || gate_w.dtype() != float32 ||
        pre_gamma.dtype() != float32 || hc_scale.dtype() != float32 ||
        x.ndim() != 3) {
      return nullptr;
    }
    const int64_t M = x.shape(0);
    const int64_t HC = x.shape(1);
    const int64_t D = x.shape(2);
    if (M < 1 || M > 65535 || HC < 1 || HC > 8 || D < 1) {
      return nullptr;
    }
    const bool kv_ok =
        (kv.ndim() == 2 && kv.shape(0) == M && kv.shape(1) == (HC + 1) * D) ||
        (kv.ndim() == 3 && kv.shape(0) == M && kv.shape(1) == HC + 1 && kv.shape(2) == D);
    const bool kvb_ok =
        (kv_b.ndim() == 2 && kv_b.shape(0) == HC + 1 && kv_b.shape(1) == D) ||
        (kv_b.ndim() == 1 && kv_b.shape(0) == (HC + 1) * D);
    if (!kv_ok || !kvb_ok ||
        gate_w.ndim() != 2 || gate_w.shape(0) != HC || gate_w.shape(1) != D ||
        pre_gamma.ndim() != 1 || pre_gamma.shape(0) != D ||
        hc_scale.ndim() != 2 || hc_scale.shape(0) != HC || hc_scale.shape(1) != D) {
      return nullptr;
    }
    std::vector<array> inputs{x, kv, kv_b, gate_w, pre_gamma, hc_scale};
    const bool has_mask = (keep_mask_h != nullptr);
    if (has_mask) {
      const auto& mask = *reinterpret_cast<array*>(keep_mask_h);
      if (mask.dtype() != uint8 || mask.ndim() != 1 || mask.shape(0) != M) {
        return nullptr;
      }
      inputs.push_back(mask);
    }
    auto res = array(
        x.shape(), bfloat16,
        std::make_shared<Hy4EngramGateAddCuda>(
            default_stream(Device::gpu), has_mask),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_engram_gate_add_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_hy4_mtp_eh_norm_concat_cuda(
    mlx_array* emb_h,
    mlx_array* hidden_h,
    mlx_array* enorm_h,
    mlx_array* hnorm_h,
    float norm_eps) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::hy4_ops;
    if (!emb_h || !hidden_h || !enorm_h || !hnorm_h ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& emb = *reinterpret_cast<array*>(emb_h);
    const auto& hidden = *reinterpret_cast<array*>(hidden_h);
    const auto& enorm = *reinterpret_cast<array*>(enorm_h);
    const auto& hnorm = *reinterpret_cast<array*>(hnorm_h);
    if (emb.dtype() != bfloat16 || hidden.dtype() != bfloat16 ||
        enorm.dtype() != float32 || hnorm.dtype() != float32 ||
        emb.ndim() != 2 || hidden.ndim() != 2 ||
        enorm.ndim() != 1 || hnorm.ndim() != 1) {
      return nullptr;
    }
    const int64_t M = emb.shape(0);
    const int64_t D = emb.shape(1);
    if (M < 1 || M > 65535 || D < 1 ||
        hidden.shape(0) != M || hidden.shape(1) != D ||
        enorm.shape(0) != D || hnorm.shape(0) != D) {
      return nullptr;
    }
    auto res = array(
        Shape{static_cast<int>(M), static_cast<int>(D * 2)}, bfloat16,
        std::make_shared<Hy4MtpEhNormConcatCuda>(
            default_stream(Device::gpu), norm_eps),
        {emb, hidden, enorm, hnorm});
    return reinterpret_cast<mlx_array*>(new array(std::move(res)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_hy4_mtp_eh_norm_concat_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" bool mlx_hy4_hc_pre_cuda(
    mlx_array* x_h,
    mlx_array* w_fn_h,
    mlx_array* s_fn_h,
    mlx_array* scale_h,
    mlx_array* base_h,
    mlx_array* norm_w_h,
    int32_t fn_mode,
    float norm_eps,
    float hc_eps,
    float hc_mag,
    bool head_only,
    mlx_array** out_h,
    mlx_array** out_post) {
  (void)x_h; (void)w_fn_h; (void)s_fn_h; (void)scale_h; (void)base_h;
  (void)norm_w_h; (void)fn_mode; (void)norm_eps; (void)hc_eps;
  (void)hc_mag; (void)head_only;
  if (out_h) *out_h = nullptr;
  if (out_post) *out_post = nullptr;
  return false;
}

extern "C" mlx_array* mlx_hy4_hc_head_cuda(
    mlx_array* x_h,
    mlx_array* w_fn_h,
    mlx_array* s_fn_h,
    mlx_array* scale_h,
    mlx_array* base_h,
    mlx_array* output_norm_h,
    int32_t fn_mode,
    float norm_eps,
    float hc_eps) {
  (void)x_h; (void)w_fn_h; (void)s_fn_h; (void)scale_h; (void)base_h;
  (void)output_norm_h; (void)fn_mode; (void)norm_eps; (void)hc_eps;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_hc_post_cuda(
    mlx_array* x_h,
    mlx_array* y_h,
    mlx_array* post_h,
    mlx_array* post_norm_w_h,
    float norm_eps) {
  (void)x_h; (void)y_h; (void)post_h; (void)post_norm_w_h; (void)norm_eps;
  return nullptr;
}

extern "C" bool mlx_hy4_hc_post_pre_cuda(
    mlx_array* x_h,
    mlx_array* y_h,
    mlx_array* prev_post_h,
    mlx_array* prev_post_norm_w_h,
    mlx_array* w_fn_h,
    mlx_array* s_fn_h,
    mlx_array* scale_h,
    mlx_array* base_h,
    mlx_array* next_norm_w_h,
    int32_t fn_mode,
    float norm_eps,
    float hc_eps,
    float hc_mag,
    mlx_array** out_x,
    mlx_array** out_h,
    mlx_array** out_post) {
  (void)x_h; (void)y_h; (void)prev_post_h; (void)prev_post_norm_w_h;
  (void)w_fn_h; (void)s_fn_h; (void)scale_h; (void)base_h;
  (void)next_norm_w_h; (void)fn_mode; (void)norm_eps; (void)hc_eps;
  (void)hc_mag;
  if (out_x) *out_x = nullptr;
  if (out_h) *out_h = nullptr;
  if (out_post) *out_post = nullptr;
  return false;
}

extern "C" mlx_array* mlx_hy4_engram_lookup_norm_cuda(
    mlx_array* raw_rows_h,
    mlx_array* lookup_norm_h) {
  (void)raw_rows_h; (void)lookup_norm_h;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_engram_gate_add_cuda(
    mlx_array* x_h,
    mlx_array* kv_h,
    mlx_array* kv_b_h,
    mlx_array* gate_w_h,
    mlx_array* pre_gamma_h,
    mlx_array* hc_scale_h,
    mlx_array* keep_mask_h) {
  (void)x_h; (void)kv_h; (void)kv_b_h; (void)gate_w_h;
  (void)pre_gamma_h; (void)hc_scale_h; (void)keep_mask_h;
  return nullptr;
}

extern "C" mlx_array* mlx_hy4_mtp_eh_norm_concat_cuda(
    mlx_array* emb_h,
    mlx_array* hidden_h,
    mlx_array* enorm_h,
    mlx_array* hnorm_h,
    float norm_eps) {
  (void)emb_h; (void)hidden_h; (void)enorm_h; (void)hnorm_h; (void)norm_eps;
  return nullptr;
}

#endif
