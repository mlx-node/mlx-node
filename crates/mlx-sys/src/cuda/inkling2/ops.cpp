#include "cuda/inkling2/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::inkling2_ops {
namespace {

using namespace mlx_node::inkling2_cuda;

constexpr char inkling2_ops_source[] =
#include "cuda/inkling2_ops.cu.inc"
    ;

constexpr char inkling2_lm_head_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/inkling2_lm_head.cu.inc"
    ;

int sidecar_dtype_code(Dtype dt) noexcept {
  if (dt == float32) {
    return 0;
  }
  if (dt == bfloat16) {
    return 1;
  }
  if (dt == float16) {
    return 2;
  }
  return -1;
}

array ensure_2d_last_contiguous(
    const array& x, const Stream& stream, cu::CommandEncoder& encoder) {
  if (x.ndim() == 2 && x.strides(1) == 1) {
    return x;
  }
  auto copy = contiguous_copy_gpu(x, stream);
  encoder.add_temporary(copy);
  return copy;
}

bool kv_cache_strides_ok(const array& c, int Hkv, int cap, int D) noexcept {
  return c.ndim() == 4 && c.shape(0) == 1 && c.shape(1) == Hkv &&
      c.shape(2) == cap && c.shape(3) == D && c.strides(3) == 1 &&
      (cap <= 1 || c.strides(2) >= D) &&
      (Hkv <= 1 ||
       (c.strides(1) >= static_cast<int64_t>(cap) * D &&
        (cap <= 1 || c.strides(1) / cap >= c.strides(2))));
}

bool kv_scale_strides_ok(const array& s, int Hkv, int cap) noexcept {
  const bool is_4d = s.ndim() == 4 && s.shape(0) == 1 && s.shape(1) == Hkv &&
      s.shape(2) == cap && s.shape(3) == 1;
  const bool is_3d =
      s.ndim() == 3 && s.shape(0) == 1 && s.shape(1) == Hkv && s.shape(2) == cap;
  if (!is_4d && !is_3d) {
    return false;
  }
  return (cap <= 1 || s.strides(2) >= 1) &&
      (Hkv <= 1 ||
       (s.strides(1) >= static_cast<int64_t>(cap) &&
        (cap <= 1 || s.strides(1) / cap >= s.strides(2))));
}

int kv_compact_elem_bytes(Dtype dt) noexcept {
  if (dt == int8 || dt == uint8) {
    return 1;
  }
  if (dt == bfloat16 || dt == float16) {
    return 2;
  }
  if (dt == float32) {
    return 4;
  }
  return 0;
}

bool kv_compact_cache_info(
    const array& c,
    int& H_kv,
    int& cap,
    int& D,
    int64_t& stride_h,
    int64_t& stride_t,
    int& elem_bytes) noexcept {
  elem_bytes = kv_compact_elem_bytes(c.dtype());
  if (elem_bytes == 0) {
    return false;
  }
  if (c.ndim() == 4) {
    if (c.shape(0) != 1) {
      return false;
    }
    H_kv = c.shape(1);
    cap = c.shape(2);
    D = c.shape(3);
    if (H_kv < 1 || H_kv > 65535 || cap < 1 || D < 1 || D > 65535) {
      return false;
    }
    if (D > 1 && c.strides(3) != 1) {
      return false;
    }
    stride_h = c.strides(1);
    stride_t = c.strides(2);
  } else if (c.ndim() == 3) {
    if (c.shape(0) != 1) {
      return false;
    }
    H_kv = c.shape(1);
    cap = c.shape(2);
    D = 1;
    if (H_kv < 1 || H_kv > 65535 || cap < 1) {
      return false;
    }
    stride_h = c.strides(1);
    stride_t = c.strides(2);
  } else {
    return false;
  }
  if (stride_h < 0 || stride_t < 0) {
    return false;
  }
  if (cap > 1 && stride_t < D) {
    return false;
  }
  if (H_kv > 1 &&
      (stride_h < static_cast<int64_t>(cap) * D ||
       (cap > 1 && stride_h / cap < stride_t))) {
    return false;
  }
  return true;
}

// ============================================================================
// 1. Block AttnRes over N in 1..=8 sources + optional fused sublayer RMSNorm
// ============================================================================
class Inkling2BlockAttnResNormCuda final : public UnaryPrimitive {
 public:
  Inkling2BlockAttnResNormCuda(
      Stream stream, int n_sources, bool has_norm, float eps, bool squeeze_q1)
      : UnaryPrimitive(stream),
        n_sources_(n_sources),
        has_norm_(has_norm),
        eps_(eps),
        squeeze_q1_(squeeze_q1) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2BlockAttnResNormCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    const size_t expected = static_cast<size_t>(n_sources_) + (has_norm_ ? 2 : 1);
    if (n_sources_ < 1 || n_sources_ > 8 || inputs.size() != expected) {
      throw std::invalid_argument("Inkling2BlockAttnResNormCuda: invalid inputs");
    }

    std::vector<array> srcs;
    srcs.reserve(static_cast<size_t>(n_sources_));
    uint32_t src_f32_mask = 0;
    for (int i = 0; i < n_sources_; ++i) {
      array s = cu::ensure_row_contiguous_aligned<16>(inputs[i], stream(), encoder);
      if (s.dtype() == float32) {
        src_f32_mask |= (1u << i);
      }
      srcs.push_back(std::move(s));
    }
    const int T = srcs[0].shape(0);
    const int D = srcs[0].shape(1);
    array queries =
        cu::ensure_row_contiguous_aligned<16>(inputs[n_sources_], stream(), encoder);
    const int Q = queries.shape(0);
    if (T < 1 || D < 1 || Q < 1 || Q > 65535 || queries.shape(1) != D) {
      throw std::invalid_argument("Inkling2BlockAttnResNormCuda: invalid shape");
    }

    std::optional<array> norm_w;
    int norm_w_bcast = 0;
    if (has_norm_) {
      norm_w = cu::ensure_row_contiguous_aligned<16>(
          inputs[n_sources_ + 1], stream(), encoder);
      norm_w_bcast = (norm_w->ndim() == 1 || norm_w->shape(0) == 1) ? 1 : 0;
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    for (const auto& s : srcs) {
      encoder.set_input_array(s);
    }
    encoder.set_input_array(queries);
    if (has_norm_) {
      encoder.set_input_array(*norm_w);
    }
    encoder.set_output_array(out);

    const bool vec4 = (D % 4) == 0;
    static constexpr const char* kEntries[2][2] = {
        {"ink2_block_attn_res_norm_cuda<false, false>",
         "ink2_block_attn_res_norm_cuda<false, true>"},
        {"ink2_block_attn_res_norm_cuda<true, false>",
         "ink2_block_attn_res_norm_cuda<true, true>"},
    };
    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source,
        kEntries[has_norm_ ? 1 : 0][vec4 ? 1 : 0]);

    cu::KernelArgs args;
    for (int i = 0; i < 8; ++i) {
      if (i < n_sources_) {
        args.append(srcs[i]);
      } else {
        args.append<CUdeviceptr>(0);
      }
    }
    args.append(queries);
    if (has_norm_) {
      args.append(*norm_w);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out);
    args.append<int32_t>(T);
    args.append<int32_t>(D);
    args.append<int32_t>(n_sources_);
    args.append<uint32_t>(src_f32_mask);
    args.append<int32_t>(norm_w_bcast);
    args.append<float>(eps_);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(T), static_cast<uint32_t>(Q), 1),
        dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Inkling2BlockAttnResNormCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2BlockAttnResNormCuda*>(&other);
    return rhs && rhs->n_sources_ == n_sources_ && rhs->has_norm_ == has_norm_ &&
        rhs->eps_ == eps_ && rhs->squeeze_q1_ == squeeze_q1_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int T = inputs[0].shape(0);
    const int D = inputs[0].shape(1);
    const int Q = inputs[n_sources_].shape(0);
    if (Q == 1 && squeeze_q1_) {
      return {Shape{T, D}};
    }
    return {Shape{Q, T, D}};
  }

 private:
  int n_sources_;
  bool has_norm_;
  float eps_;
  bool squeeze_q1_;
};

// ============================================================================
// 2. Fused LoopController + Sublayer RMSNorm (G=1 or G=3, rank=64)
// ============================================================================
class Inkling2LoopCtrlNormCuda final : public UnaryPrimitive {
 public:
  Inkling2LoopCtrlNormCuda(Stream stream, int g_count, float eps, bool squeeze_g1)
      : UnaryPrimitive(stream),
        g_count_(g_count),
        eps_(eps),
        squeeze_g1_(squeeze_g1) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2LoopCtrlNormCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array dest = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array msg = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array down_w = cu::ensure_row_contiguous_aligned<16>(inputs[2], stream(), encoder);
    array gf_w = cu::ensure_row_contiguous_aligned<16>(inputs[3], stream(), encoder);
    array norm_w = cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);

    const int T = (dest.ndim() == 2) ? dest.shape(0) : dest.shape(1);
    const int D = dest.shape(-1);
    if (T < 1 || D < 4 || (D & 3) != 0 || (g_count_ != 1 && g_count_ != 3)) {
      throw std::invalid_argument("Inkling2LoopCtrlNormCuda: invalid shape");
    }

    const int dest_bcast = (dest.ndim() == 2 || dest.shape(0) == 1) ? 1 : 0;
    const int msg_bcast = (msg.ndim() == 2 || msg.shape(0) == 1) ? 1 : 0;
    const int norm_w_bcast = (norm_w.ndim() == 1 || norm_w.shape(0) == 1) ? 1 : 0;
    const int dest_is_f32 = (dest.dtype() == float32) ? 1 : 0;
    const int msg_is_f32 = (msg.dtype() == float32) ? 1 : 0;
    const bool down_bf16 = (down_w.dtype() == bfloat16);
    const bool gf_bf16 = (gf_w.dtype() == bfloat16);

    array h_part = alloc_temp({2, g_count_, T, kLoopRank}, float32, encoder);
    array blended = alloc_temp({g_count_, T, D}, float32, encoder);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    // Step 1: Fused inv_rms + down projection -> h_part [2, G, T, 64] FP32
    {
      encoder.set_input_array(dest);
      encoder.set_input_array(msg);
      encoder.set_input_array(down_w);
      encoder.set_output_array(h_part);
      static constexpr const char* kDownEntries[2][2][3] = {
          {
              {"ink2_loop_down_fused_cuda<false, false, 1, 3>",
               "ink2_loop_down_fused_cuda<false, false, 4, 3>",
               "ink2_loop_down_fused_cuda<false, false, 8, 3>"},
              {"ink2_loop_down_fused_cuda<false, false, 1, 3>",
               "ink2_loop_down_fused_cuda<false, false, 4, 3>",
               "ink2_loop_down_fused_cuda<false, false, 8, 3>"},
          },
          {
              {"ink2_loop_down_fused_cuda<true, false, 1, 3>",
               "ink2_loop_down_fused_cuda<true, false, 4, 3>",
               "ink2_loop_down_fused_cuda<true, false, 8, 3>"},
              {"ink2_loop_down_fused_cuda<true, true, 1, 3>",
               "ink2_loop_down_fused_cuda<true, true, 4, 3>",
               "ink2_loop_down_fused_cuda<true, true, 8, 3>"},
          },
      };
      // Always use NWARPS=3 (96 threads/CTA, 8 blocks/SM) across all T buckets
      // (TILE_T=1 / 4 / 8) so M=1 AR, M=4..=16 speculative verify, and prefill
      // share identical 0-ULP FP32 warp-reduction order with minimal weight reads.
      const int bucket = (T == 1) ? 0 : (T <= 4 ? 1 : 2);
      const int tile_t = (bucket == 0) ? 1 : (bucket == 1 ? 4 : 8);
      const uint32_t block_threads = 96u;
      const bool vec8 = down_bf16 && ((D & 7) == 0);
      auto fn = cu::jit_kernel(
          encoder.device(), inkling2_ops_source,
          kDownEntries[down_bf16 ? 1 : 0][vec8 ? 1 : 0][bucket]);
      cu::KernelArgs args;
      args.append(dest);
      args.append(msg);
      args.append(down_w);
      args.append(h_part);
      args.append<int32_t>(T);
      args.append<int32_t>(D);
      args.append<int32_t>(g_count_);
      args.append<int32_t>(dest_bcast);
      args.append<int32_t>(msg_bcast);
      args.append<int32_t>(dest_is_f32);
      args.append<int32_t>(msg_is_f32);
      args.append<float>(eps_);
      const uint32_t grid_z = static_cast<uint32_t>((T + tile_t - 1) / tile_t);
      encoder.add_kernel_node_raw(
          fn, dim3(128, static_cast<uint32_t>(g_count_), grid_z),
          dim3(block_threads, 1, 1), {}, 0, args.args());
    }

    // Step 2: up projection (2-term BF16 TC MMA on sm_80+) + tanh gate + no-FMA blend -> blended [G, T, D] FP32
    {
      encoder.set_input_array(h_part);
      encoder.set_input_array(gf_w);
      encoder.set_input_array(dest);
      encoder.set_input_array(msg);
      encoder.set_output_array(blended);
      const bool use_mma =
          gf_bf16 && encoder.device().compute_capability_major() >= 8;
      cu::KernelArgs args;
      args.append(h_part);
      args.append(gf_w);
      args.append(dest);
      args.append(msg);
      args.append(blended);
      args.append<int32_t>(T);
      args.append<int32_t>(D);
      args.append<int32_t>(g_count_);
      args.append<int32_t>(dest_bcast);
      args.append<int32_t>(msg_bcast);
      args.append<int32_t>(dest_is_f32);
      args.append<int32_t>(msg_is_f32);
      if (use_mma) {
        auto fn = cu::jit_kernel(
            encoder.device(), inkling2_ops_source, "ink2_loop_up48_mma_cuda");
        const uint32_t grid_x = static_cast<uint32_t>((D + 47) / 48);
        const uint32_t grid_z = static_cast<uint32_t>((T + 7) / 8);
        encoder.add_kernel_node_raw(
            fn, dim3(grid_x, static_cast<uint32_t>(g_count_), grid_z),
            dim3(192, 1, 1), {}, 0, args.args());
      } else {
        auto fn = cu::jit_kernel(
            encoder.device(), inkling2_ops_source,
            gf_bf16 ? "ink2_loop_up_blend_f32_cuda<true>"
                    : "ink2_loop_up_blend_f32_cuda<false>");
        const uint32_t grid_x = static_cast<uint32_t>((D + 31) / 32);
        const uint32_t grid_z = static_cast<uint32_t>((T + 15) / 16);
        encoder.add_kernel_node_raw(
            fn, dim3(grid_x, static_cast<uint32_t>(g_count_), grid_z),
            dim3(128, 1, 1), {}, 0, args.args());
      }
    }

    // Step 3: sublayer RMSNorm -> out BF16
    {
      encoder.set_input_array(blended);
      encoder.set_input_array(norm_w);
      encoder.set_output_array(out);
      auto fn = cu::jit_kernel(
          encoder.device(), inkling2_ops_source, "ink2_rms_norm_f32_to_bf16_cuda");
      cu::KernelArgs args;
      args.append(blended);
      args.append(norm_w);
      args.append(out);
      args.append<int32_t>(T);
      args.append<int32_t>(D);
      args.append<int32_t>(norm_w_bcast);
      args.append<float>(eps_);
      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(T), static_cast<uint32_t>(g_count_), 1),
          dim3(256, 1, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Inkling2LoopCtrlNormCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2LoopCtrlNormCuda*>(&other);
    return rhs && rhs->g_count_ == g_count_ && rhs->eps_ == eps_ &&
        rhs->squeeze_g1_ == squeeze_g1_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int T = (inputs[0].ndim() == 2) ? inputs[0].shape(0) : inputs[0].shape(1);
    const int D = inputs[0].shape(-1);
    if (g_count_ == 1 && squeeze_g1_) {
      return {Shape{T, D}};
    }
    return {Shape{g_count_, T, D}};
  }

 private:
  int g_count_;
  float eps_;
  bool squeeze_g1_;
};

// ============================================================================
// 3. Fused Q/R L2-clip + K/V ShortConv + K L2-clip + V RMSNorm + Global tau
//    + In-Place KV Cache Store (BF16 or INT8, linear/ring, linear/tree)
// ============================================================================
class Inkling2KvConvPrepStoreCuda final : public Primitive {
 public:
  Inkling2KvConvPrepStoreCuda(
      Stream stream,
      bool is_dyn,
      bool is_int8,
      bool has_taps,
      bool has_depths,
      bool is_global,
      bool update_hist,
      int start_pos,
      int write_offset,
      int ring_cap,
      float log_n_floor,
      float log_alpha)
      : Primitive(stream),
        is_dyn_(is_dyn),
        is_int8_(is_int8),
        has_taps_(has_taps),
        has_depths_(has_depths),
        is_global_(is_global),
        update_hist_(update_hist),
        start_pos_(start_pos),
        write_offset_(write_offset),
        ring_cap_(ring_cap),
        log_n_floor_(log_n_floor),
        log_alpha_(log_alpha) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2KvConvPrepStoreCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array q_raw = ensure_2d_last_contiguous(inputs[0], stream(), encoder);
    array r_raw = ensure_2d_last_contiguous(inputs[1], stream(), encoder);
    array k_raw = ensure_2d_last_contiguous(inputs[2], stream(), encoder);
    array v_raw = ensure_2d_last_contiguous(inputs[3], stream(), encoder);
    array q_clip = cu::ensure_row_contiguous_aligned<4>(inputs[4], stream(), encoder);
    array rel_clip = cu::ensure_row_contiguous_aligned<4>(inputs[5], stream(), encoder);
    array k_conv_w = cu::ensure_row_contiguous_aligned<16>(inputs[6], stream(), encoder);
    array v_conv_w = cu::ensure_row_contiguous_aligned<16>(inputs[7], stream(), encoder);
    array k_hist_in = cu::ensure_row_contiguous_aligned<2>(inputs[8], stream(), encoder);
    array v_hist_in = cu::ensure_row_contiguous_aligned<2>(inputs[9], stream(), encoder);
    const array& kv_k_cache = inputs[10];
    const array& kv_v_cache = inputs[11];

    size_t next_in = 12;
    std::optional<array> dyn_k, dyn_v;
    if (is_dyn_) {
      dyn_k = (inputs[next_in].dtype() == float32)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[next_in], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[next_in], stream(), encoder);
      ++next_in;
      dyn_v = (inputs[next_in].dtype() == float32)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[next_in], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[next_in], stream(), encoder);
      ++next_in;
    }
    const array* kv_k_scales = nullptr;
    const array* kv_v_scales = nullptr;
    if (is_int8_) {
      kv_k_scales = &inputs[next_in++];
      kv_v_scales = &inputs[next_in++];
    }
    std::optional<array> conv_taps;
    if (has_taps_) {
      conv_taps =
          cu::ensure_row_contiguous_aligned<16>(inputs[next_in++], stream(), encoder);
    }
    std::optional<array> depths;
    if (has_depths_) {
      depths =
          cu::ensure_row_contiguous_aligned<4>(inputs[next_in++], stream(), encoder);
    }

    const int T = q_raw.shape(0);
    const int cap = kv_k_cache.shape(2);
    if (T < 1 || !kv_cache_strides_ok(kv_k_cache, kNumKvHeads, cap, kHeadDim) ||
        !kv_cache_strides_ok(kv_v_cache, kNumKvHeads, cap, kHeadDim) ||
        write_offset_ < 0 ||
        (ring_cap_ == 0 && static_cast<int64_t>(write_offset_) + T > cap) ||
        (ring_cap_ > 0 && ring_cap_ > cap)) {
      throw std::invalid_argument("Inkling2KvConvPrepStoreCuda: invalid cache bounds");
    }
    if (is_int8_ &&
        (!kv_scale_strides_ok(*kv_k_scales, kNumKvHeads, cap) ||
         !kv_scale_strides_ok(*kv_v_scales, kNumKvHeads, cap))) {
      throw std::invalid_argument("Inkling2KvConvPrepStoreCuda: invalid scale strides");
    }

    // Verify mutated cache/scale buffers do not alias each other or any read input.
    const void* kc_ptr = kv_k_cache.buffer().ptr();
    const void* vc_ptr = kv_v_cache.buffer().ptr();
    const void* ks_ptr = is_int8_ ? kv_k_scales->buffer().ptr() : nullptr;
    const void* vs_ptr = is_int8_ ? kv_v_scales->buffer().ptr() : nullptr;
    if (kc_ptr == vc_ptr ||
        (is_int8_ &&
         (ks_ptr == vs_ptr || ks_ptr == kc_ptr || ks_ptr == vc_ptr ||
          vs_ptr == kc_ptr || vs_ptr == vc_ptr))) {
      throw std::invalid_argument("Inkling2KvConvPrepStoreCuda: cache buffers alias");
    }
    auto check_no_alias = [&](const array& a) {
      const void* p = a.buffer().ptr();
      if (p == kc_ptr || p == vc_ptr ||
          (is_int8_ && (p == ks_ptr || p == vs_ptr))) {
        throw std::invalid_argument(
            "Inkling2KvConvPrepStoreCuda: cache buffer aliases input");
      }
    };
    check_no_alias(q_raw);
    check_no_alias(r_raw);
    check_no_alias(k_raw);
    check_no_alias(v_raw);
    check_no_alias(q_clip);
    check_no_alias(rel_clip);
    check_no_alias(k_conv_w);
    check_no_alias(v_conv_w);
    check_no_alias(k_hist_in);
    check_no_alias(v_hist_in);
    if (is_dyn_) {
      check_no_alias(*dyn_k);
      check_no_alias(*dyn_v);
    }
    if (has_taps_) check_no_alias(*conv_taps);
    if (has_depths_) check_no_alias(*depths);

    // Bind outputs: [out_q, out_r, out_kv_k, out_kv_v, (out_ks, out_vs), (out_k_hist, out_v_hist)]
    auto& out_q = outputs[0];
    auto& out_r = outputs[1];
    auto& out_kv_k = outputs[2];
    auto& out_kv_v = outputs[3];
    out_q.set_data(cu::malloc_async(out_q.nbytes(), encoder));
    out_r.set_data(cu::malloc_async(out_r.nbytes(), encoder));
    out_kv_k.copy_shared_buffer(kv_k_cache);
    out_kv_v.copy_shared_buffer(kv_v_cache);

    size_t next_out = 4;
    array* out_ks = nullptr;
    array* out_vs = nullptr;
    if (is_int8_) {
      out_ks = &outputs[next_out++];
      out_vs = &outputs[next_out++];
      out_ks->copy_shared_buffer(*kv_k_scales);
      out_vs->copy_shared_buffer(*kv_v_scales);
    }
    array* out_k_hist = nullptr;
    array* out_v_hist = nullptr;
    if (update_hist_) {
      out_k_hist = &outputs[next_out++];
      out_v_hist = &outputs[next_out++];
      out_k_hist->set_data(cu::malloc_async(out_k_hist->nbytes(), encoder));
      out_v_hist->set_data(cu::malloc_async(out_v_hist->nbytes(), encoder));
    }

    encoder.set_input_array(q_raw);
    encoder.set_input_array(r_raw);
    encoder.set_input_array(k_raw);
    encoder.set_input_array(v_raw);
    encoder.set_input_array(q_clip);
    encoder.set_input_array(rel_clip);
    encoder.set_input_array(k_conv_w);
    encoder.set_input_array(v_conv_w);
    encoder.set_input_array(k_hist_in);
    encoder.set_input_array(v_hist_in);
    if (is_dyn_) {
      encoder.set_input_array(*dyn_k);
      encoder.set_input_array(*dyn_v);
    }
    if (has_taps_) {
      encoder.set_input_array(*conv_taps);
    }
    if (has_depths_) {
      encoder.set_input_array(*depths);
    }
    // Do NOT call encoder.set_input_array(kv_k_cache/kv_v_cache/scales):
    // out_kv_* alias them and set_output_array already tracks dependencies
    // without inflating bytes_in_graph_ by the preallocated KV buffers.
    encoder.set_output_array(out_q);
    encoder.set_output_array(out_r);
    encoder.set_output_array(out_kv_k);
    encoder.set_output_array(out_kv_v);
    if (is_int8_) {
      encoder.set_output_array(*out_ks);
      encoder.set_output_array(*out_vs);
    }
    if (update_hist_) {
      encoder.set_output_array(*out_k_hist);
      encoder.set_output_array(*out_v_hist);
    }

    uint32_t flags = 0;
    if (is_dyn_) {
      flags |= 1u;
      if (dyn_k->dtype() == float32) flags |= (1u << 1);
      if (dyn_v->dtype() == float32) flags |= (1u << 2);
    }
    if (has_taps_) flags |= (1u << 3);
    if (has_depths_) flags |= (1u << 4);
    if (is_global_) flags |= (1u << 5);
    if (update_hist_) flags |= (1u << 6);

    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source,
        is_int8_ ? "ink2_kv_conv_prep_store_cuda<true>"
                 : "ink2_kv_conv_prep_store_cuda<false>");

    cu::KernelArgs args;
    args.append(q_raw);
    args.append(r_raw);
    args.append(k_raw);
    args.append(v_raw);
    args.append(q_clip);
    args.append(rel_clip);
    args.append(k_conv_w);
    args.append(v_conv_w);
    if (is_dyn_) {
      args.append(*dyn_k);
      args.append(*dyn_v);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }
    args.append(k_hist_in);
    args.append(v_hist_in);
    if (has_taps_) {
      args.append(*conv_taps);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (has_depths_) {
      args.append(*depths);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_q);
    args.append(out_r);
    if (update_hist_) {
      args.append(*out_k_hist);
      args.append(*out_v_hist);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }
    args.append(out_kv_k);
    args.append(out_kv_v);
    if (is_int8_) {
      args.append(*out_ks);
      args.append(*out_vs);
    } else {
      args.append<CUdeviceptr>(0);
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(T);
    args.append<int32_t>(start_pos_);
    args.append<int32_t>(write_offset_);
    args.append<int32_t>(ring_cap_);
    args.append<float>(log_n_floor_);
    args.append<float>(log_alpha_);
    args.append<uint32_t>(flags);
    args.append<int64_t>(q_raw.strides(0));
    args.append<int64_t>(r_raw.strides(0));
    args.append<int64_t>(k_raw.strides(0));
    args.append<int64_t>(v_raw.strides(0));
    args.append<int64_t>(kv_k_cache.strides(1));
    args.append<int64_t>(kv_k_cache.strides(2));
    args.append<int64_t>(kv_v_cache.strides(1));
    args.append<int64_t>(kv_v_cache.strides(2));
    args.append<int64_t>(is_int8_ ? kv_k_scales->strides(1) : 0);
    args.append<int64_t>(is_int8_ ? kv_k_scales->strides(2) : 0);
    args.append<int64_t>(is_int8_ ? kv_v_scales->strides(1) : 0);
    args.append<int64_t>(is_int8_ ? kv_v_scales->strides(2) : 0);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(T), 40, 1), dim3(32, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Inkling2KvConvPrepStoreCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2KvConvPrepStoreCuda*>(&other);
    return rhs && rhs->is_dyn_ == is_dyn_ && rhs->is_int8_ == is_int8_ &&
        rhs->has_taps_ == has_taps_ && rhs->has_depths_ == has_depths_ &&
        rhs->is_global_ == is_global_ && rhs->update_hist_ == update_hist_ &&
        rhs->start_pos_ == start_pos_ && rhs->write_offset_ == write_offset_ &&
        rhs->ring_cap_ == ring_cap_ && rhs->log_n_floor_ == log_n_floor_ &&
        rhs->log_alpha_ == log_alpha_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int T = inputs[0].shape(0);
    std::vector<Shape> shapes{
        Shape{1, kNumHeads, T, kHeadDim},
        Shape{T, kNumHeads, kRelDim},
        inputs[10].shape(),
        inputs[11].shape(),
    };
    if (is_int8_) {
      const size_t ks_idx = 12 + (is_dyn_ ? 2 : 0);
      shapes.push_back(inputs[ks_idx].shape());
      shapes.push_back(inputs[ks_idx + 1].shape());
    }
    if (update_hist_) {
      shapes.push_back(Shape{3, kNumKvHeads * kHeadDim});
      shapes.push_back(Shape{3, kNumKvHeads * kHeadDim});
    }
    return shapes;
  }

 private:
  bool is_dyn_;
  bool is_int8_;
  bool has_taps_;
  bool has_depths_;
  bool is_global_;
  bool update_hist_;
  int start_pos_;
  int write_offset_;
  int ring_cap_;
  float log_n_floor_;
  float log_alpha_;
};

// ============================================================================
// 4. Static ShortConv k=4 + optional fused FP32 residual add (linear OR tree)
// ============================================================================
class Inkling2SconvK4ResidualCuda final : public Primitive {
 public:
  Inkling2SconvK4ResidualCuda(
      Stream stream, bool has_res, bool has_taps, bool update_hist)
      : Primitive(stream),
        has_res_(has_res),
        has_taps_(has_taps),
        update_hist_(update_hist) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2SconvK4ResidualCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = ensure_2d_last_contiguous(inputs[0], stream(), encoder);
    array weight = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);
    array hist_in = cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
    size_t next_in = 3;
    std::optional<array> res;
    if (has_res_) {
      res = ensure_2d_last_contiguous(inputs[next_in++], stream(), encoder);
    }
    std::optional<array> conv_taps;
    if (has_taps_) {
      conv_taps =
          cu::ensure_row_contiguous_aligned<16>(inputs[next_in++], stream(), encoder);
    }

    const int T = x.shape(0);
    const int C = x.shape(1);
    if (T < 1 || C < 1) {
      throw std::invalid_argument("Inkling2SconvK4ResidualCuda: invalid shape");
    }

    auto& out_y = outputs[0];
    out_y.set_data(cu::malloc_async(out_y.nbytes(), encoder));
    array* out_hist = nullptr;
    if (update_hist_) {
      out_hist = &outputs[1];
      out_hist->set_data(cu::malloc_async(out_hist->nbytes(), encoder));
    }

    encoder.set_input_array(x);
    encoder.set_input_array(weight);
    encoder.set_input_array(hist_in);
    if (has_res_) encoder.set_input_array(*res);
    if (has_taps_) encoder.set_input_array(*conv_taps);
    encoder.set_output_array(out_y);
    if (update_hist_) encoder.set_output_array(*out_hist);

    uint32_t flags = 0;
    if (has_res_) {
      flags |= 1u;
      if (res->dtype() == float32) flags |= (1u << 1);
    }
    if (has_taps_) flags |= (1u << 2);
    if (update_hist_) flags |= (1u << 3);

    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source, "ink2_sconv_k4_residual_cuda");
    cu::KernelArgs args;
    args.append(x);
    args.append(weight);
    args.append(hist_in);
    if (has_res_) {
      args.append(*res);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (has_taps_) {
      args.append(*conv_taps);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(out_y);
    if (update_hist_) {
      args.append(*out_hist);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(T);
    args.append<int32_t>(C);
    args.append<int64_t>(x.strides(0));
    args.append<int64_t>(has_res_ ? res->strides(0) : 0);
    args.append<uint32_t>(flags);

    const uint32_t grid_x = static_cast<uint32_t>((C + 255) / 256);
    const uint32_t grid_y = static_cast<uint32_t>(std::min(T, 64));
    encoder.add_kernel_node_raw(
        fn, dim3(grid_x, grid_y, 1), dim3(256, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Inkling2SconvK4ResidualCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2SconvK4ResidualCuda*>(&other);
    return rhs && rhs->has_res_ == has_res_ && rhs->has_taps_ == has_taps_ &&
        rhs->update_hist_ == update_hist_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes{inputs[0].shape()};
    if (update_hist_) {
      shapes.push_back(Shape{3, inputs[0].shape(1)});
    }
    return shapes;
  }

 private:
  bool has_res_;
  bool has_taps_;
  bool update_hist_;
};

// ============================================================================
// 5. Fused Q8_0 (Sym / Affine) & BF16 LM Head + Softcap (+ Zero-DRAM Argmax)
// ============================================================================
class Inkling2LmHeadSoftcapCuda final : public UnaryPrimitive {
 public:
  Inkling2LmHeadSoftcapCuda(
      Stream stream,
      bool is_q8,
      bool symmetric,
      int unpadded_vocab,
      float softcap,
      int out_mode)
      : UnaryPrimitive(stream),
        is_q8_(is_q8),
        symmetric_(symmetric),
        unpadded_vocab_(unpadded_vocab),
        softcap_(softcap),
        out_mode_(out_mode) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2LmHeadSoftcapCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / static_cast<size_t>(K)) : 0;
    const int V = unpadded_vocab_;
    if (M < 1 || M > 16 || V < 1 || V > w.shape(0) ||
        (is_q8_ && (K < 32 || (K & 31) != 0)) ||
        (!is_q8_ && (K < 8 || (K & 7) != 0))) {
      throw std::invalid_argument("Inkling2LmHeadSoftcapCuda: unsupported shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    // Always use the 2-stage cp.async.cg + BF16 Tensor Core MMA kernel across M in 1..=16
    // so M=1 AR and M=2..=16 speculative verification share identical 0-ULP FP32 reduction order.
    static const bool force_u4 = [] {
      const char* v = std::getenv("MLX_INKLING2_LM_HEAD_U4");
      return v && v[0] == '1';
    }();
    const bool use_u4 = force_u4 && (M <= 2);
    const int num_ctas = (out_mode_ == 1)
        ? (use_u4 ? std::min(384, (V + 7) / 8) : std::min(384, (V + 15) / 16))
        : (use_u4 ? ((V + 7) / 8) : ((V + 15) / 16));

    std::optional<array> partial_vals;
    std::optional<array> partial_idxs;
    if (out_mode_ == 1) {
      partial_vals = alloc_temp({M, num_ctas}, float32, encoder);
      partial_idxs = alloc_temp({M, num_ctas}, int32, encoder);
    }

    if (is_q8_) {
      const int sb_dtype = sidecar_dtype_code(inputs[2].dtype());
      if (sb_dtype < 0 || (!symmetric_ && inputs[3].dtype() != inputs[2].dtype())) {
        throw std::invalid_argument("Inkling2LmHeadSoftcapCuda: unsupported sidecar dtype");
      }
      array scales = (sb_dtype == 0)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
      std::optional<array> biases;
      if (!symmetric_) {
        biases = (sb_dtype == 0)
            ? cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder)
            : cu::ensure_row_contiguous_aligned<2>(inputs[3], stream(), encoder);
      }

      encoder.set_input_array(x);
      encoder.set_input_array(w);
      encoder.set_input_array(scales);
      if (!symmetric_) encoder.set_input_array(*biases);
      if (out_mode_ == 1) {
        encoder.set_output_array(*partial_vals);
        encoder.set_output_array(*partial_idxs);
      } else {
        encoder.set_output_array(out);
      }

      const int sym_idx = symmetric_ ? 1 : 0;
      const char* entry = nullptr;
      if (use_u4) {
        static constexpr const char* kQ8U4[2][3][2][3] = {
            {
                {{"ink2_lm_head_q8_u4<0, 1, false, 0>", "ink2_lm_head_q8_u4<0, 1, false, 1>", "ink2_lm_head_q8_u4<0, 1, false, 2>"},
                 {"ink2_lm_head_q8_u4<0, 2, false, 0>", "ink2_lm_head_q8_u4<0, 2, false, 1>", "ink2_lm_head_q8_u4<0, 2, false, 2>"}},
                {{"ink2_lm_head_q8_u4<1, 1, false, 0>", "ink2_lm_head_q8_u4<1, 1, false, 1>", "ink2_lm_head_q8_u4<1, 1, false, 2>"},
                 {"ink2_lm_head_q8_u4<1, 2, false, 0>", "ink2_lm_head_q8_u4<1, 2, false, 1>", "ink2_lm_head_q8_u4<1, 2, false, 2>"}},
                {{"ink2_lm_head_q8_u4<2, 1, false, 0>", "ink2_lm_head_q8_u4<2, 1, false, 1>", "ink2_lm_head_q8_u4<2, 1, false, 2>"},
                 {"ink2_lm_head_q8_u4<2, 2, false, 0>", "ink2_lm_head_q8_u4<2, 2, false, 1>", "ink2_lm_head_q8_u4<2, 2, false, 2>"}},
            },
            {
                {{"ink2_lm_head_q8_u4<0, 1, true, 0>", "ink2_lm_head_q8_u4<0, 1, true, 1>", "ink2_lm_head_q8_u4<0, 1, true, 2>"},
                 {"ink2_lm_head_q8_u4<0, 2, true, 0>", "ink2_lm_head_q8_u4<0, 2, true, 1>", "ink2_lm_head_q8_u4<0, 2, true, 2>"}},
                {{"ink2_lm_head_q8_u4<1, 1, true, 0>", "ink2_lm_head_q8_u4<1, 1, true, 1>", "ink2_lm_head_q8_u4<1, 1, true, 2>"},
                 {"ink2_lm_head_q8_u4<1, 2, true, 0>", "ink2_lm_head_q8_u4<1, 2, true, 1>", "ink2_lm_head_q8_u4<1, 2, true, 2>"}},
                {{"ink2_lm_head_q8_u4<2, 1, true, 0>", "ink2_lm_head_q8_u4<2, 1, true, 1>", "ink2_lm_head_q8_u4<2, 1, true, 2>"},
                 {"ink2_lm_head_q8_u4<2, 2, true, 0>", "ink2_lm_head_q8_u4<2, 2, true, 1>", "ink2_lm_head_q8_u4<2, 2, true, 2>"}},
            },
        };
        entry = kQ8U4[sym_idx][sb_dtype][M == 1 ? 0 : 1][out_mode_];
      } else {
        static constexpr const char* kQ8Mma[2][3][2][3] = {
            {
                {{"ink2_lm_head_q8_mma<0, 1, false, 0>", "ink2_lm_head_q8_mma<0, 1, false, 1>", "ink2_lm_head_q8_mma<0, 1, false, 2>"},
                 {"ink2_lm_head_q8_mma<0, 2, false, 0>", "ink2_lm_head_q8_mma<0, 2, false, 1>", "ink2_lm_head_q8_mma<0, 2, false, 2>"}},
                {{"ink2_lm_head_q8_mma<1, 1, false, 0>", "ink2_lm_head_q8_mma<1, 1, false, 1>", "ink2_lm_head_q8_mma<1, 1, false, 2>"},
                 {"ink2_lm_head_q8_mma<1, 2, false, 0>", "ink2_lm_head_q8_mma<1, 2, false, 1>", "ink2_lm_head_q8_mma<1, 2, false, 2>"}},
                {{"ink2_lm_head_q8_mma<2, 1, false, 0>", "ink2_lm_head_q8_mma<2, 1, false, 1>", "ink2_lm_head_q8_mma<2, 1, false, 2>"},
                 {"ink2_lm_head_q8_mma<2, 2, false, 0>", "ink2_lm_head_q8_mma<2, 2, false, 1>", "ink2_lm_head_q8_mma<2, 2, false, 2>"}},
            },
            {
                {{"ink2_lm_head_q8_mma<0, 1, true, 0>", "ink2_lm_head_q8_mma<0, 1, true, 1>", "ink2_lm_head_q8_mma<0, 1, true, 2>"},
                 {"ink2_lm_head_q8_mma<0, 2, true, 0>", "ink2_lm_head_q8_mma<0, 2, true, 1>", "ink2_lm_head_q8_mma<0, 2, true, 2>"}},
                {{"ink2_lm_head_q8_mma<1, 1, true, 0>", "ink2_lm_head_q8_mma<1, 1, true, 1>", "ink2_lm_head_q8_mma<1, 1, true, 2>"},
                 {"ink2_lm_head_q8_mma<1, 2, true, 0>", "ink2_lm_head_q8_mma<1, 2, true, 1>", "ink2_lm_head_q8_mma<1, 2, true, 2>"}},
                {{"ink2_lm_head_q8_mma<2, 1, true, 0>", "ink2_lm_head_q8_mma<2, 1, true, 1>", "ink2_lm_head_q8_mma<2, 1, true, 2>"},
                 {"ink2_lm_head_q8_mma<2, 2, true, 0>", "ink2_lm_head_q8_mma<2, 2, true, 1>", "ink2_lm_head_q8_mma<2, 2, true, 2>"}},
            },
        };
        entry = kQ8Mma[sym_idx][sb_dtype][M <= 8 ? 0 : 1][out_mode_];
      }

      auto fn = cu::jit_kernel(encoder.device(), inkling2_lm_head_source, entry);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      args.append(scales);
      if (!symmetric_) {
        args.append(*biases);
      } else {
        args.append<CUdeviceptr>(0);
      }
      if (out_mode_ == 1) {
        args.append(*partial_vals);
        args.append(*partial_idxs);
      } else {
        args.append(out);
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(V);
      args.append<int32_t>(K);
      args.append<float>(softcap_);
      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(num_ctas), 1, 1), dim3(32, 8, 1), {},
          0, args.args());
    } else {
      encoder.set_input_array(x);
      encoder.set_input_array(w);
      if (out_mode_ == 1) {
        encoder.set_output_array(*partial_vals);
        encoder.set_output_array(*partial_idxs);
      } else {
        encoder.set_output_array(out);
      }

      const char* entry = nullptr;
      if (use_u4) {
        static constexpr const char* kBf16U4[2][3] = {
            {"ink2_lm_head_bf16_u4<1, 0>", "ink2_lm_head_bf16_u4<1, 1>", "ink2_lm_head_bf16_u4<1, 2>"},
            {"ink2_lm_head_bf16_u4<2, 0>", "ink2_lm_head_bf16_u4<2, 1>", "ink2_lm_head_bf16_u4<2, 2>"},
        };
        entry = kBf16U4[M == 1 ? 0 : 1][out_mode_];
      } else {
        static constexpr const char* kBf16Mma[2][3] = {
            {"ink2_lm_head_bf16_mma<1, 0>", "ink2_lm_head_bf16_mma<1, 1>", "ink2_lm_head_bf16_mma<1, 2>"},
            {"ink2_lm_head_bf16_mma<2, 0>", "ink2_lm_head_bf16_mma<2, 1>", "ink2_lm_head_bf16_mma<2, 2>"},
        };
        entry = kBf16Mma[M <= 8 ? 0 : 1][out_mode_];
      }

      auto fn = cu::jit_kernel(encoder.device(), inkling2_lm_head_source, entry);
      cu::KernelArgs args;
      args.append(x);
      args.append(w);
      if (out_mode_ == 1) {
        args.append(*partial_vals);
        args.append(*partial_idxs);
      } else {
        args.append(out);
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(M);
      args.append<int32_t>(V);
      args.append<int32_t>(K);
      args.append<float>(softcap_);
      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(num_ctas), 1, 1), dim3(32, 8, 1), {},
          0, args.args());
    }

    // Stage 2 argmax reduction when out_mode_ == 1: [M, num_ctas] -> [M] Int32
    if (out_mode_ == 1) {
      encoder.set_input_array(*partial_vals);
      encoder.set_input_array(*partial_idxs);
      encoder.set_output_array(out);
      auto fn2 = cu::jit_kernel(
          encoder.device(), inkling2_lm_head_source, "ink2_lm_head_argmax_stage2");
      cu::KernelArgs args2;
      args2.append(*partial_vals);
      args2.append(*partial_idxs);
      args2.append(out);
      args2.append<int32_t>(M);
      args2.append<int32_t>(num_ctas);
      encoder.add_kernel_node_raw(
          fn2, dim3(static_cast<uint32_t>(M), 1, 1), dim3(256, 1, 1), {}, 0,
          args2.args());
    }
  }

  DEFINE_NAME(Inkling2LmHeadSoftcapCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2LmHeadSoftcapCuda*>(&other);
    return rhs && rhs->is_q8_ == is_q8_ && rhs->symmetric_ == symmetric_ &&
        rhs->unpadded_vocab_ == unpadded_vocab_ && rhs->softcap_ == softcap_ &&
        rhs->out_mode_ == out_mode_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int K = inputs[0].shape(-1);
    const int M = (K > 0) ? static_cast<int>(inputs[0].size() / static_cast<size_t>(K)) : 0;
    if (out_mode_ == 1) {
      return {Shape{M}};
    }
    return {Shape{M, unpadded_vocab_}};
  }

 private:
  bool is_q8_;
  bool symmetric_;
  int unpadded_vocab_;
  float softcap_;
  int out_mode_;
};

// ============================================================================
// 6. High-Occupancy Small-N BF16 Projection (N in 1..=128, M in 1..=16, K%8==0)
// ============================================================================
class Inkling2SmallBf16ProjCuda final : public UnaryPrimitive {
 public:
  explicit Inkling2SmallBf16ProjCuda(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2SmallBf16ProjCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array x = cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    array w = cu::ensure_row_contiguous_aligned<16>(inputs[1], stream(), encoder);

    const int K = x.shape(-1);
    const int M = (K > 0) ? static_cast<int>(x.size() / static_cast<size_t>(K)) : 0;
    const int N = w.shape(0);
    if (M < 1 || M > 16 || N < 1 || N > 128 || K < 8 || (K & 7) != 0) {
      throw std::invalid_argument("Inkling2SmallBf16ProjCuda: unsupported shape");
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_output_array(out);

    static constexpr const char* kEntries[5][4] = {
        {"ink2_small_bf16_proj_cuda<1, 4>", "ink2_small_bf16_proj_cuda<1, 6>",
         "ink2_small_bf16_proj_cuda<1, 8>", "ink2_small_bf16_proj_cuda<1, 9>"},
        {"ink2_small_bf16_proj_cuda<2, 4>", "ink2_small_bf16_proj_cuda<2, 6>",
         "ink2_small_bf16_proj_cuda<2, 8>", "ink2_small_bf16_proj_cuda<2, 9>"},
        {"ink2_small_bf16_proj_cuda<4, 4>", "ink2_small_bf16_proj_cuda<4, 6>",
         "ink2_small_bf16_proj_cuda<4, 8>", "ink2_small_bf16_proj_cuda<4, 9>"},
        {"ink2_small_bf16_proj_cuda<8, 4>", "ink2_small_bf16_proj_cuda<8, 6>",
         "ink2_small_bf16_proj_cuda<8, 8>", "ink2_small_bf16_proj_cuda<8, 9>"},
        {"ink2_small_bf16_proj_cuda<16, 4>", "ink2_small_bf16_proj_cuda<16, 6>",
         "ink2_small_bf16_proj_cuda<16, 8>", "ink2_small_bf16_proj_cuda<16, 9>"},
    };
    const int m_bucket =
        (M == 1) ? 0 : (M == 2 ? 1 : (M <= 4 ? 2 : (M <= 8 ? 3 : 4)));
    const int kv_end = K >> 3;
    int nw_idx = 0;
    int num_warps = 4;
    // Keep num_warps independent of M so M=1 AR and M=2..=16 speculative verify
    // share identical 0-ULP FP32 warp-reduction order (at K=2304, kv_end=288=9*32
    // so 9 warps execute exactly 1 uint4 load per thread with zero loop overhead).
    if (kv_end >= 288 && (kv_end % 9) == 0) {
      nw_idx = 3;
      num_warps = 9;
    } else if (kv_end >= 256) {
      nw_idx = 2;
      num_warps = 8;
    } else if (kv_end >= 192 && (kv_end % 6) == 0) {
      nw_idx = 1;
      num_warps = 6;
    }

    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source, kEntries[m_bucket][nw_idx]);
    cu::KernelArgs args;
    args.append(x);
    args.append(w);
    args.append(out);
    args.append<int32_t>(M);
    args.append<int32_t>(N);
    args.append<int32_t>(K);
    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(N), 1, 1),
        dim3(32, static_cast<uint32_t>(num_warps), 1), {}, 0, args.args());
  }

  DEFINE_NAME(Inkling2SmallBf16ProjCuda)
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Inkling2SmallBf16ProjCuda*>(&other) != nullptr;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[1].shape(0);
    return {std::move(shape)};
  }
};

// ============================================================================
// 7. Batched ShortConv Rollback Tape Commit (1 launch for <= 96 streams)
// ============================================================================
struct Ink2CommitTapeBatchHost {
  uint64_t hists_in[96];
  uint64_t rows_in[96];
  uint64_t hists_out[96];
  int64_t hist_strides[96];
  int64_t row_strides[96];
};
static_assert(sizeof(Ink2CommitTapeBatchHost) == 96 * 40, "Ink2CommitTapeBatch layout mismatch");

class Inkling2CommitShortConvTapesCuda final : public Primitive {
 public:
  Inkling2CommitShortConvTapesCuda(
      Stream stream, size_t n_streams, bool has_nodes, int contiguous_keep)
      : Primitive(stream),
        n_streams_(n_streams),
        has_nodes_(has_nodes),
        contiguous_keep_(contiguous_keep) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2CommitShortConvTapesCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    const size_t expected_in = n_streams_ * 2 + (has_nodes_ ? 1 : 0);
    if (n_streams_ < 1 || n_streams_ > 128 || inputs.size() != expected_in ||
        outputs.size() != n_streams_) {
      throw std::invalid_argument("Inkling2CommitShortConvTapesCuda: invalid arity");
    }

    std::vector<array> hists;
    std::vector<array> rows;
    hists.reserve(n_streams_);
    rows.reserve(n_streams_);
    for (size_t s = 0; s < n_streams_; ++s) {
      hists.push_back(ensure_2d_last_contiguous(inputs[s], stream(), encoder));
      rows.push_back(
          ensure_2d_last_contiguous(inputs[n_streams_ + s], stream(), encoder));
    }
    std::optional<array> nodes;
    if (has_nodes_) {
      nodes = cu::ensure_row_contiguous_aligned<4>(
          inputs[2 * n_streams_], stream(), encoder);
    }

    const int C = hists[0].shape(1);
    const int T = rows[0].shape(0);
    const int keep = has_nodes_ ? static_cast<int>(nodes->shape(0)) : contiguous_keep_;
    if (C < 1 || T < 1 || keep < 0 || (!has_nodes_ && keep > T)) {
      throw std::invalid_argument("Inkling2CommitShortConvTapesCuda: invalid shape");
    }

    for (size_t s = 0; s < n_streams_; ++s) {
      outputs[s].set_data(cu::malloc_async(outputs[s].nbytes(), encoder));
    }

    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source, "ink2_commit_short_conv_tapes_cuda");
    const size_t num_chunks = (n_streams_ + 95) / 96;
    std::vector<Ink2CommitTapeBatchHost> batches(num_chunks);
    const uint32_t grid_y = static_cast<uint32_t>((C + 255) / 256);

    for (size_t ch = 0; ch < num_chunks; ++ch) {
      const size_t s0 = ch * 96;
      const size_t n_chunk = std::min<size_t>(96, n_streams_ - s0);
      auto& batch = batches[ch];
      for (size_t i = 0; i < n_chunk; ++i) {
        const size_t s = s0 + i;
        batch.hists_in[i] =
            reinterpret_cast<uint64_t>(gpu_ptr<void>(hists[s]));
        batch.rows_in[i] =
            reinterpret_cast<uint64_t>(gpu_ptr<void>(rows[s]));
        batch.hists_out[i] =
            reinterpret_cast<uint64_t>(gpu_ptr<void>(outputs[s]));
        batch.hist_strides[i] = hists[s].strides(0);
        batch.row_strides[i] = rows[s].strides(0);
      }
      for (size_t i = n_chunk; i < 96; ++i) {
        batch.hists_in[i] = 0;
        batch.rows_in[i] = 0;
        batch.hists_out[i] = 0;
        batch.hist_strides[i] = 0;
        batch.row_strides[i] = 0;
      }

      // Register CUDA Graph dependencies per launch chunk because
      // `add_kernel_node_raw` consumes and clears active inputs/outputs on each launch.
      for (size_t i = 0; i < n_chunk; ++i) {
        const size_t s = s0 + i;
        encoder.set_input_array(hists[s]);
        encoder.set_input_array(rows[s]);
        encoder.set_output_array(outputs[s]);
      }
      if (has_nodes_) {
        encoder.set_input_array(*nodes);
      }

      cu::KernelArgs args;
      args.append_ptr(&batch);
      if (has_nodes_) {
        args.append(*nodes);
      } else {
        args.append<CUdeviceptr>(0);
      }
      args.append<int32_t>(static_cast<int32_t>(n_chunk));
      args.append<int32_t>(C);
      args.append<int32_t>(T);
      args.append<int32_t>(keep);
      args.append<int32_t>(has_nodes_ ? 1 : 0);

      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(n_chunk), grid_y, 1), dim3(256, 1, 1),
          {}, 0, args.args());
    }
  }

  DEFINE_NAME(Inkling2CommitShortConvTapesCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs =
        dynamic_cast<const Inkling2CommitShortConvTapesCuda*>(&other);
    return rhs && rhs->n_streams_ == n_streams_ && rhs->has_nodes_ == has_nodes_ &&
        rhs->contiguous_keep_ == contiguous_keep_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int C = inputs[0].shape(1);
    return std::vector<Shape>(n_streams_, Shape{3, C});
  }

 private:
  size_t n_streams_;
  bool has_nodes_;
  int contiguous_keep_;
};

// ============================================================================
// 8. Fused Token Embedding Lookup + Precise FP32 RMSNorm -> BF16
// ============================================================================
class Inkling2EmbedRmsNormCuda final : public UnaryPrimitive {
 public:
  Inkling2EmbedRmsNormCuda(Stream stream, int mode, float eps)
      : UnaryPrimitive(stream), mode_(mode), eps_(eps) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2EmbedRmsNormCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());
    array token_ids =
        cu::ensure_row_contiguous_aligned<4>(inputs[0], stream(), encoder);
    const int D = inputs.back().shape(0);
    const int V = inputs[1].shape(0);
    const int T = static_cast<int>(token_ids.size());
    if (T < 1 || V < 1 || D < 1) {
      throw std::invalid_argument("Inkling2EmbedRmsNormCuda: invalid shape");
    }

    array weight = (mode_ == 0)
        ? (((D & 3) == 0)
               ? cu::ensure_row_contiguous_aligned<8>(inputs[1], stream(), encoder)
               : cu::ensure_row_contiguous_aligned<2>(inputs[1], stream(), encoder))
        : cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder);

    int sb_dtype = 0;
    std::optional<array> scales;
    std::optional<array> biases;
    if (mode_ >= 1) {
      sb_dtype = sidecar_dtype_code(inputs[2].dtype());
      if (sb_dtype < 0) {
        throw std::invalid_argument("Inkling2EmbedRmsNormCuda: unsupported scales dtype");
      }
      scales = (sb_dtype == 0)
          ? cu::ensure_row_contiguous_aligned<4>(inputs[2], stream(), encoder)
          : cu::ensure_row_contiguous_aligned<2>(inputs[2], stream(), encoder);
      if (mode_ == 2) {
        biases = (sb_dtype == 0)
            ? cu::ensure_row_contiguous_aligned<4>(inputs[3], stream(), encoder)
            : cu::ensure_row_contiguous_aligned<2>(inputs[3], stream(), encoder);
      }
    }
    const size_t nw_idx = (mode_ == 0) ? 2 : (mode_ == 1 ? 3 : 4);
    array norm_w = ((D & 3) == 0)
        ? cu::ensure_row_contiguous_aligned<16>(inputs[nw_idx], stream(), encoder)
        : cu::ensure_row_contiguous_aligned<4>(inputs[nw_idx], stream(), encoder);

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(token_ids);
    encoder.set_input_array(weight);
    if (scales.has_value()) encoder.set_input_array(*scales);
    if (biases.has_value()) encoder.set_input_array(*biases);
    encoder.set_input_array(norm_w);
    encoder.set_output_array(out);

    static constexpr const char* kEntries[3][3] = {
        {"ink2_embed_rmsnorm_cuda<0, 0>",
         "ink2_embed_rmsnorm_cuda<0, 0>",
         "ink2_embed_rmsnorm_cuda<0, 0>"},
        {"ink2_embed_rmsnorm_cuda<1, 0>",
         "ink2_embed_rmsnorm_cuda<1, 1>",
         "ink2_embed_rmsnorm_cuda<1, 2>"},
        {"ink2_embed_rmsnorm_cuda<2, 0>",
         "ink2_embed_rmsnorm_cuda<2, 1>",
         "ink2_embed_rmsnorm_cuda<2, 2>"},
    };
    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source, kEntries[mode_][sb_dtype]);
    cu::KernelArgs args;
    args.append(token_ids);
    args.append(weight);
    if (scales.has_value()) {
      args.append(*scales);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (biases.has_value()) {
      args.append(*biases);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append(norm_w);
    args.append(out);
    args.append<int32_t>(T);
    args.append<int32_t>(V);
    args.append<int32_t>(D);
    args.append<float>(eps_);

    encoder.add_kernel_node_raw(
        fn, dim3(static_cast<uint32_t>(T), 1, 1), dim3(256, 1, 1), {}, 0,
        args.args());
  }

  DEFINE_NAME(Inkling2EmbedRmsNormCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2EmbedRmsNormCuda*>(&other);
    return rhs && rhs->mode_ == mode_ && rhs->eps_ == eps_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int D = inputs.back().shape(0);
    auto shape = inputs[0].shape();
    shape.push_back(D);
    return {std::move(shape)};
  }

 private:
  int mode_;
  float eps_;
};

// ============================================================================
// 9. Batched In-Place KV Cache Compaction (1 launch for <= 96 caches)
// ============================================================================
struct Ink2CompactKvBatchHost {
  uint64_t ptrs[96];
  int64_t stride_h[96];
  int64_t stride_t[96];
  int32_t write_offsets[96];
  int32_t caps[96];
  uint32_t d_and_bytes[96];
};
static_assert(
    sizeof(Ink2CompactKvBatchHost) == 96 * 36,
    "Ink2CompactKvBatch layout mismatch");

class Inkling2CompactKvBatchCuda final : public Primitive {
 public:
  Inkling2CompactKvBatchCuda(
      Stream stream, size_t n_caches, std::vector<int> write_offsets)
      : Primitive(stream),
        n_caches_(n_caches),
        write_offsets_(std::move(write_offsets)) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Inkling2CompactKvBatchCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& encoder = cu::get_command_encoder(stream());
    if (n_caches_ < 1 || n_caches_ > 192 ||
        inputs.size() != n_caches_ + 1 || outputs.size() != n_caches_ ||
        write_offsets_.size() != n_caches_) {
      throw std::invalid_argument("Inkling2CompactKvBatchCuda: invalid arity");
    }

    array nodes = cu::ensure_row_contiguous_aligned<4>(
        inputs[n_caches_], stream(), encoder);
    const int K = (nodes.ndim() == 1) ? nodes.shape(0) : 0;
    if (nodes.dtype() != int32 || K < 1 || K > 16) {
      throw std::invalid_argument(
          "Inkling2CompactKvBatchCuda: invalid accepted_nodes");
    }

    struct CacheMeta {
      uint64_t ptr;
      uint64_t end;
      int64_t stride_h;
      int64_t stride_t;
      int cap;
      int D;
      int elem_bytes;
    };
    std::vector<CacheMeta> metas;
    metas.reserve(n_caches_);
    int H_kv = 0;
    int max_D = 1;
    const void* nodes_buf = nodes.buffer().ptr();

    for (size_t i = 0; i < n_caches_; ++i) {
      const array& c = inputs[i];
      if (c.buffer().ptr() == nodes_buf) {
        throw std::invalid_argument(
            "Inkling2CompactKvBatchCuda: cache aliases accepted_nodes");
      }
      int h_i = 0, cap_i = 0, d_i = 0, eb_i = 0;
      int64_t sh_i = 0, st_i = 0;
      if (!kv_compact_cache_info(c, h_i, cap_i, d_i, sh_i, st_i, eb_i)) {
        throw std::invalid_argument(
            "Inkling2CompactKvBatchCuda: invalid cache layout");
      }
      if (i == 0) {
        H_kv = h_i;
      } else if (h_i != H_kv) {
        throw std::invalid_argument(
            "Inkling2CompactKvBatchCuda: mismatched H_kv across batch");
      }
      if (write_offsets_[i] < 0 ||
          static_cast<int64_t>(write_offsets_[i]) + K > cap_i) {
        throw std::invalid_argument(
            "Inkling2CompactKvBatchCuda: write_offset out of bounds");
      }
      const uint64_t base_addr =
          reinterpret_cast<uint64_t>(gpu_ptr<void>(c));
      if (base_addr == 0) {
        throw std::invalid_argument(
            "Inkling2CompactKvBatchCuda: null cache device pointer");
      }
      const uint64_t max_off =
          static_cast<uint64_t>(H_kv - 1) * static_cast<uint64_t>(sh_i) +
          static_cast<uint64_t>(cap_i - 1) * static_cast<uint64_t>(st_i) +
          static_cast<uint64_t>(d_i - 1);
      const uint64_t span_bytes = (max_off + 1u) * static_cast<uint64_t>(eb_i);
      const uint64_t end_addr = base_addr + span_bytes;
      for (const auto& prev : metas) {
        if (base_addr < prev.end && prev.ptr < end_addr) {
          throw std::invalid_argument(
              "Inkling2CompactKvBatchCuda: cache buffers overlap");
        }
      }
      if (d_i > max_D) {
        max_D = d_i;
      }
      metas.push_back(
          CacheMeta{base_addr, end_addr, sh_i, st_i, cap_i, d_i, eb_i});
    }

    for (size_t i = 0; i < n_caches_; ++i) {
      outputs[i].copy_shared_buffer(inputs[i]);
    }

    auto fn = cu::jit_kernel(
        encoder.device(), inkling2_ops_source, "ink2_compact_kv_batch_cuda");
    const size_t num_chunks = (n_caches_ + 95u) / 96u;
    std::vector<Ink2CompactKvBatchHost> batches(num_chunks);
    const uint32_t block_threads = (max_D <= 32) ? 32u : 128u;

    for (size_t ch = 0; ch < num_chunks; ++ch) {
      const size_t c0 = ch * 96u;
      const size_t n_chunk = std::min<size_t>(96u, n_caches_ - c0);
      auto& batch = batches[ch];
      for (size_t i = 0; i < n_chunk; ++i) {
        const size_t idx = c0 + i;
        const auto& m = metas[idx];
        batch.ptrs[i] = m.ptr;
        batch.stride_h[i] = m.stride_h;
        batch.stride_t[i] = m.stride_t;
        batch.write_offsets[i] = write_offsets_[idx];
        batch.caps[i] = m.cap;
        batch.d_and_bytes[i] =
            (static_cast<uint32_t>(m.elem_bytes) << 16) |
            static_cast<uint32_t>(m.D);
      }
      for (size_t i = n_chunk; i < 96u; ++i) {
        batch.ptrs[i] = 0;
        batch.stride_h[i] = 0;
        batch.stride_t[i] = 0;
        batch.write_offsets[i] = 0;
        batch.caps[i] = 0;
        batch.d_and_bytes[i] = 0;
      }

      // Register CUDA Graph dependencies per launch chunk because
      // `add_kernel_node_raw` consumes and clears active inputs/outputs on each launch.
      encoder.set_input_array(nodes);
      for (size_t i = 0; i < n_chunk; ++i) {
        // CRITICAL CUDA Graph rule: do NOT call encoder.set_input_array(inputs[c0 + i]);
        // outputs[c0 + i] aliases inputs[c0 + i] and set_output_array tracks the dependency
        // without inflating bytes_in_graph_ by the preallocated KV buffers.
        encoder.set_output_array(outputs[c0 + i]);
      }

      cu::KernelArgs args;
      args.append_ptr(&batch);
      args.append(nodes);
      args.append<int32_t>(static_cast<int32_t>(n_chunk));
      args.append<int32_t>(H_kv);
      args.append<int32_t>(K);

      encoder.add_kernel_node_raw(
          fn, dim3(static_cast<uint32_t>(n_chunk), static_cast<uint32_t>(H_kv), 1),
          dim3(block_threads, 1, 1), {}, 0, args.args());
    }
  }

  DEFINE_NAME(Inkling2CompactKvBatchCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2CompactKvBatchCuda*>(&other);
    return rhs && rhs->n_caches_ == n_caches_ &&
        rhs->write_offsets_ == write_offsets_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes;
    shapes.reserve(n_caches_);
    for (size_t i = 0; i < n_caches_; ++i) {
      shapes.push_back(inputs[i].shape());
    }
    return shapes;
  }

 private:
  size_t n_caches_;
  std::vector<int> write_offsets_;
};

}  // namespace
}  // namespace mlx::core::inkling2_ops

// ============================================================================
// Extern "C" ABI Entry Points (CUDA Enabled)
// ============================================================================

extern "C" mlx_array* mlx_inkling2_block_attn_res_norm_cuda(
    mlx_array* const* sources,
    size_t n_sources,
    mlx_array* queries,
    mlx_array* norm_weights,
    float eps,
    bool squeeze_q1) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!sources || n_sources < 1 || n_sources > 8 || !queries ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    std::vector<array> inputs;
    inputs.reserve(n_sources + (norm_weights ? 2 : 1));
    for (size_t i = 0; i < n_sources; ++i) {
      if (!sources[i]) {
        return nullptr;
      }
      inputs.push_back(*reinterpret_cast<array*>(sources[i]));
    }
    const auto& s0 = inputs[0];
    if (s0.ndim() != 2 || (s0.dtype() != bfloat16 && s0.dtype() != float32)) {
      return nullptr;
    }
    const int T = s0.shape(0);
    const int D = s0.shape(1);
    if (T < 1 || D < 1) {
      return nullptr;
    }
    for (size_t i = 1; i < n_sources; ++i) {
      const auto& si = inputs[i];
      if (si.ndim() != 2 || si.shape(0) != T || si.shape(1) != D ||
          (si.dtype() != bfloat16 && si.dtype() != float32)) {
        return nullptr;
      }
    }
    const auto& q = *reinterpret_cast<array*>(queries);
    if (q.ndim() != 2 || q.dtype() != float32 || q.shape(1) != D ||
        q.shape(0) < 1 || q.shape(0) > 65535) {
      return nullptr;
    }
    const int Q = q.shape(0);
    inputs.push_back(q);

    const bool has_norm = (norm_weights != nullptr);
    if (has_norm) {
      const auto& nw = *reinterpret_cast<array*>(norm_weights);
      const bool valid_nw = nw.dtype() == float32 &&
          ((nw.ndim() == 1 && nw.shape(0) == D) ||
           (nw.ndim() == 2 && (nw.shape(0) == 1 || nw.shape(0) == Q) &&
            nw.shape(1) == D));
      if (!valid_nw) {
        return nullptr;
      }
      inputs.push_back(nw);
    }

    Shape out_shape = (Q == 1 && squeeze_q1) ? Shape{T, D} : Shape{Q, T, D};
    const Dtype out_dtype = has_norm ? bfloat16 : float32;
    auto result = array(
        std::move(out_shape), out_dtype,
        std::make_shared<Inkling2BlockAttnResNormCuda>(
            default_stream(Device::gpu), static_cast<int>(n_sources), has_norm,
            eps, squeeze_q1),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_block_attn_res_norm_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_block_attn_res_norm_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" mlx_array* mlx_inkling2_loop_ctrl_norm_cuda(
    mlx_array* dest,
    mlx_array* msg,
    mlx_array* down_w,
    mlx_array* gain_forget_w,
    mlx_array* norm_w,
    float eps,
    int g_count,
    bool squeeze_g1) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    using namespace mlx_node::inkling2_cuda;
    if (!dest || !msg || !down_w || !gain_forget_w || !norm_w ||
        (g_count != 1 && g_count != 3) || default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& da = *reinterpret_cast<array*>(dest);
    const auto& ma = *reinterpret_cast<array*>(msg);
    const auto& dw = *reinterpret_cast<array*>(down_w);
    const auto& gfw = *reinterpret_cast<array*>(gain_forget_w);
    const auto& nw = *reinterpret_cast<array*>(norm_w);

    if ((da.dtype() != float32 && da.dtype() != bfloat16) ||
        (ma.dtype() != float32 && ma.dtype() != bfloat16) ||
        (dw.dtype() != float32 && dw.dtype() != bfloat16) ||
        (gfw.dtype() != float32 && gfw.dtype() != bfloat16) ||
        nw.dtype() != float32) {
      return nullptr;
    }
    if ((da.ndim() != 2 && da.ndim() != 3) ||
        (ma.ndim() != 2 && ma.ndim() != 3)) {
      return nullptr;
    }
    const int T = (da.ndim() == 2) ? da.shape(0) : da.shape(1);
    const int D = da.shape(-1);
    if (T < 1 || D < 4 || (D & 3) != 0) {
      return nullptr;
    }
    if (da.ndim() == 3 && da.shape(0) != 1 && da.shape(0) != g_count) {
      return nullptr;
    }
    const int mT = (ma.ndim() == 2) ? ma.shape(0) : ma.shape(1);
    if (mT != T || ma.shape(-1) != D ||
        (ma.ndim() == 3 && ma.shape(0) != 1 && ma.shape(0) != g_count)) {
      return nullptr;
    }

    const bool valid_dw =
        (g_count == 1 && dw.ndim() == 2 && dw.shape(0) == kLoopRank &&
         dw.shape(1) == 2 * D) ||
        (dw.ndim() == 3 && dw.shape(0) == g_count &&
         dw.shape(1) == kLoopRank && dw.shape(2) == 2 * D);
    const bool valid_gfw =
        (g_count == 1 && gfw.ndim() == 2 && gfw.shape(0) == 2 * D &&
         gfw.shape(1) == kLoopRank) ||
        (gfw.ndim() == 3 && gfw.shape(0) == g_count &&
         gfw.shape(1) == 2 * D && gfw.shape(2) == kLoopRank);
    const bool valid_nw =
        (nw.ndim() == 1 && nw.shape(0) == D) ||
        (nw.ndim() == 2 && (nw.shape(0) == 1 || nw.shape(0) == g_count) &&
         nw.shape(1) == D);
    if (!valid_dw || !valid_gfw || !valid_nw) {
      return nullptr;
    }

    Shape out_shape = (g_count == 1 && squeeze_g1) ? Shape{T, D} : Shape{g_count, T, D};
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Inkling2LoopCtrlNormCuda>(
            default_stream(Device::gpu), g_count, eps, squeeze_g1),
        {da, ma, dw, gfw, nw});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_loop_ctrl_norm_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_loop_ctrl_norm_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" bool mlx_inkling2_kv_conv_prep_store_cuda(
    mlx_array* q_raw,
    mlx_array* r_raw,
    mlx_array* k_raw,
    mlx_array* v_raw,
    mlx_array* q_clip,
    mlx_array* rel_clip,
    mlx_array* k_conv_w,
    mlx_array* v_conv_w,
    mlx_array* dyn_k,
    mlx_array* dyn_v,
    mlx_array* k_hist_in,
    mlx_array* v_hist_in,
    mlx_array* kv_k_cache,
    mlx_array* kv_v_cache,
    mlx_array* kv_k_scales,
    mlx_array* kv_v_scales,
    mlx_array* conv_taps,
    mlx_array* depths,
    int start_pos,
    int write_offset,
    int ring_cap,
    bool is_global,
    float log_n_floor,
    float log_alpha,
    bool update_hist,
    mlx_array** out_q,
    mlx_array** out_r,
    mlx_array** out_k_hist,
    mlx_array** out_v_hist,
    mlx_array** out_kv_k_alias,
    mlx_array** out_kv_v_alias,
    mlx_array** out_ks_alias,
    mlx_array** out_vs_alias) {
  if (out_q) *out_q = nullptr;
  if (out_r) *out_r = nullptr;
  if (out_k_hist) *out_k_hist = nullptr;
  if (out_v_hist) *out_v_hist = nullptr;
  if (out_kv_k_alias) *out_kv_k_alias = nullptr;
  if (out_kv_v_alias) *out_kv_v_alias = nullptr;
  if (out_ks_alias) *out_ks_alias = nullptr;
  if (out_vs_alias) *out_vs_alias = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    using namespace mlx_node::inkling2_cuda;
    if (!q_raw || !r_raw || !k_raw || !v_raw || !q_clip || !rel_clip ||
        !k_conv_w || !v_conv_w || !k_hist_in || !v_hist_in ||
        !kv_k_cache || !kv_v_cache || !out_q || !out_r ||
        !out_kv_k_alias || !out_kv_v_alias ||
        default_device() != Device::gpu) {
      return false;
    }
    if (update_hist && (!out_k_hist || !out_v_hist)) {
      return false;
    }
    const bool is_dyn = (dyn_k != nullptr) || (dyn_v != nullptr);
    if (is_dyn && (!dyn_k || !dyn_v)) {
      return false;
    }
    const bool is_int8 = (kv_k_scales != nullptr) || (kv_v_scales != nullptr);
    if (is_int8 && (!kv_k_scales || !kv_v_scales || !out_ks_alias || !out_vs_alias)) {
      return false;
    }

    const auto& qa = *reinterpret_cast<array*>(q_raw);
    const auto& ra = *reinterpret_cast<array*>(r_raw);
    const auto& ka = *reinterpret_cast<array*>(k_raw);
    const auto& va = *reinterpret_cast<array*>(v_raw);
    const auto& qc = *reinterpret_cast<array*>(q_clip);
    const auto& rc = *reinterpret_cast<array*>(rel_clip);
    const auto& kcw = *reinterpret_cast<array*>(k_conv_w);
    const auto& vcw = *reinterpret_cast<array*>(v_conv_w);
    const auto& khi = *reinterpret_cast<array*>(k_hist_in);
    const auto& vhi = *reinterpret_cast<array*>(v_hist_in);
    const auto& kc = *reinterpret_cast<array*>(kv_k_cache);
    const auto& vc = *reinterpret_cast<array*>(kv_v_cache);

    constexpr int kQCols = kNumHeads * kHeadDim;       // 3072
    constexpr int kRCols = kNumHeads * kRelDim;        // 384
    constexpr int kKvCols = kNumKvHeads * kHeadDim;    // 1024

    if (qa.ndim() != 2 || qa.shape(1) != kQCols || qa.dtype() != bfloat16 ||
        ra.ndim() != 2 || ra.shape(1) != kRCols || ra.dtype() != bfloat16 ||
        ka.ndim() != 2 || ka.shape(1) != kKvCols || ka.dtype() != bfloat16 ||
        va.ndim() != 2 || va.shape(1) != kKvCols || va.dtype() != bfloat16) {
      return false;
    }
    const int T = qa.shape(0);
    if (T < 1 || ra.shape(0) != T || ka.shape(0) != T || va.shape(0) != T) {
      return false;
    }
    if (qc.dtype() != float32 || qc.size() != static_cast<size_t>(kQCols) ||
        rc.dtype() != float32 || rc.size() != static_cast<size_t>(kRCols) ||
        kcw.dtype() != float32 || kcw.shape() != Shape{kKvCols, 4} ||
        vcw.dtype() != float32 || vcw.shape() != Shape{kKvCols, 4} ||
        khi.dtype() != bfloat16 || khi.shape() != Shape{3, kKvCols} ||
        vhi.dtype() != bfloat16 || vhi.shape() != Shape{3, kKvCols}) {
      return false;
    }

    const Dtype kv_dt = is_int8 ? int8 : bfloat16;
    if (kc.ndim() != 4 || vc.ndim() != 4 || kc.dtype() != kv_dt || vc.dtype() != kv_dt) {
      return false;
    }
    const int cap = kc.shape(2);
    if (cap < 1 || !kv_cache_strides_ok(kc, kNumKvHeads, cap, kHeadDim) ||
        !kv_cache_strides_ok(vc, kNumKvHeads, cap, kHeadDim) ||
        write_offset < 0 || ring_cap < 0 ||
        (ring_cap == 0 && static_cast<int64_t>(write_offset) + T > cap) ||
        (ring_cap > 0 && ring_cap > cap)) {
      return false;
    }

    std::vector<array> inputs;
    inputs.reserve(18);
    inputs.push_back(qa);
    inputs.push_back(ra);
    inputs.push_back(ka);
    inputs.push_back(va);
    inputs.push_back(qc);
    inputs.push_back(rc);
    inputs.push_back(kcw);
    inputs.push_back(vcw);
    inputs.push_back(khi);
    inputs.push_back(vhi);
    inputs.push_back(kc);
    inputs.push_back(vc);

    if (is_dyn) {
      const auto& dk = *reinterpret_cast<array*>(dyn_k);
      const auto& dv = *reinterpret_cast<array*>(dyn_v);
      const bool dk_ok = (dk.dtype() == bfloat16 || dk.dtype() == float32) &&
          ((dk.ndim() == 3 && dk.shape() == Shape{T, kNumKvHeads, 4}) ||
           (dk.ndim() == 2 && dk.shape() == Shape{T, kNumKvHeads * 4}));
      const bool dv_ok = (dv.dtype() == bfloat16 || dv.dtype() == float32) &&
          ((dv.ndim() == 3 && dv.shape() == Shape{T, kNumKvHeads, 4}) ||
           (dv.ndim() == 2 && dv.shape() == Shape{T, kNumKvHeads * 4}));
      if (!dk_ok || !dv_ok) {
        return false;
      }
      inputs.push_back(dk);
      inputs.push_back(dv);
    }

    std::vector<Shape> out_shapes{
        Shape{1, kNumHeads, T, kHeadDim},
        Shape{T, kNumHeads, kRelDim},
        kc.shape(),
        vc.shape(),
    };
    std::vector<Dtype> out_dtypes{bfloat16, float32, kv_dt, kv_dt};

    if (is_int8) {
      const auto& ks = *reinterpret_cast<array*>(kv_k_scales);
      const auto& vs = *reinterpret_cast<array*>(kv_v_scales);
      if (ks.dtype() != float32 || vs.dtype() != float32 ||
          !kv_scale_strides_ok(ks, kNumKvHeads, cap) ||
          !kv_scale_strides_ok(vs, kNumKvHeads, cap)) {
        return false;
      }
      inputs.push_back(ks);
      inputs.push_back(vs);
      out_shapes.push_back(ks.shape());
      out_shapes.push_back(vs.shape());
      out_dtypes.push_back(float32);
      out_dtypes.push_back(float32);
    }

    const bool has_taps = (conv_taps != nullptr);
    if (has_taps) {
      const auto& ct = *reinterpret_cast<array*>(conv_taps);
      if (ct.dtype() != int32 || ct.ndim() != 2 || ct.shape() != Shape{T, 4}) {
        return false;
      }
      inputs.push_back(ct);
    }

    const bool has_depths = (depths != nullptr);
    if (has_depths) {
      const auto& dp = *reinterpret_cast<array*>(depths);
      if (dp.dtype() != int32 || dp.ndim() != 1 || dp.shape(0) != T) {
        return false;
      }
      inputs.push_back(dp);
    }

    if (update_hist) {
      out_shapes.push_back(Shape{3, kKvCols});
      out_shapes.push_back(Shape{3, kKvCols});
      out_dtypes.push_back(bfloat16);
      out_dtypes.push_back(bfloat16);
    }

    auto outputs = array::make_arrays(
        std::move(out_shapes), std::move(out_dtypes),
        std::make_shared<Inkling2KvConvPrepStoreCuda>(
            default_stream(Device::gpu),
            is_dyn,
            is_int8,
            has_taps,
            has_depths,
            is_global,
            update_hist,
            start_pos,
            write_offset,
            ring_cap,
            log_n_floor,
            log_alpha),
        std::move(inputs));

    size_t idx = 0;
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
    *out_r = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
    *out_kv_k_alias = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
    *out_kv_v_alias = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
    if (is_int8) {
      *out_ks_alias = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
      *out_vs_alias = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
    }
    if (update_hist) {
      *out_k_hist = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
      *out_v_hist = reinterpret_cast<mlx_array*>(new array(std::move(outputs[idx++])));
    } else {
      if (out_k_hist) *out_k_hist = reinterpret_cast<mlx_array*>(new array(khi));
      if (out_v_hist) *out_v_hist = reinterpret_cast<mlx_array*>(new array(vhi));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_kv_conv_prep_store_cuda: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_kv_conv_prep_store_cuda: unknown exception\n");
    return false;
  }
}

extern "C" bool mlx_inkling2_sconv_k4_residual_cuda(
    mlx_array* x,
    mlx_array* weight,
    mlx_array* hist_in,
    mlx_array* residual,
    mlx_array* conv_taps,
    bool update_hist,
    mlx_array** out_y,
    mlx_array** out_hist) {
  if (out_y) *out_y = nullptr;
  if (out_hist) *out_hist = nullptr;
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!x || !weight || !hist_in || !out_y || (update_hist && !out_hist) ||
        default_device() != Device::gpu) {
      return false;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(weight);
    const auto& ha = *reinterpret_cast<array*>(hist_in);
    if (xa.ndim() != 2 || xa.dtype() != bfloat16) {
      return false;
    }
    const int T = xa.shape(0);
    const int C = xa.shape(1);
    if (T < 1 || C < 1 ||
        wa.dtype() != float32 || wa.shape() != Shape{C, 4} ||
        ha.dtype() != bfloat16 || ha.shape() != Shape{3, C}) {
      return false;
    }

    std::vector<array> inputs{xa, wa, ha};
    const bool has_res = (residual != nullptr);
    if (has_res) {
      const auto& ra = *reinterpret_cast<array*>(residual);
      if (ra.ndim() != 2 || ra.shape() != Shape{T, C} ||
          (ra.dtype() != bfloat16 && ra.dtype() != float32)) {
        return false;
      }
      inputs.push_back(ra);
    }
    const bool has_taps = (conv_taps != nullptr);
    if (has_taps) {
      const auto& ct = *reinterpret_cast<array*>(conv_taps);
      if (ct.dtype() != int32 || ct.ndim() != 2 || ct.shape() != Shape{T, 4}) {
        return false;
      }
      inputs.push_back(ct);
    }

    std::vector<Shape> shapes{Shape{T, C}};
    std::vector<Dtype> dtypes{bfloat16};
    if (update_hist) {
      shapes.push_back(Shape{3, C});
      dtypes.push_back(bfloat16);
    }

    auto outputs = array::make_arrays(
        std::move(shapes), std::move(dtypes),
        std::make_shared<Inkling2SconvK4ResidualCuda>(
            default_stream(Device::gpu), has_res, has_taps, update_hist),
        std::move(inputs));
    *out_y = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    if (update_hist) {
      *out_hist = reinterpret_cast<mlx_array*>(new array(std::move(outputs[1])));
    } else if (out_hist) {
      *out_hist = reinterpret_cast<mlx_array*>(new array(ha));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_sconv_k4_residual_cuda: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_sconv_k4_residual_cuda: unknown exception\n");
    return false;
  }
}

extern "C" mlx_array* mlx_inkling2_lm_head_softcap_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int unpadded_vocab,
    float softcap,
    int out_mode) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!x || !w || out_mode < 0 || out_mode > 2 || std::isnan(softcap) ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(w);
    if (xa.dtype() != bfloat16 || xa.ndim() < 2 || wa.ndim() != 2) {
      return nullptr;
    }
    const int K = xa.shape(-1);
    if (K < 8) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(xa.size()) / K;
    const int V_pad = wa.shape(0);
    if (rows < 1 || rows > 16 || unpadded_vocab < 1 || unpadded_vocab > V_pad) {
      return nullptr;
    }

    const bool is_q8 = (scales != nullptr);
    const bool symmetric = (biases == nullptr);
    std::vector<array> inputs;
    inputs.reserve(is_q8 ? (symmetric ? 3 : 4) : 2);
    inputs.push_back(xa);
    inputs.push_back(wa);

    if (is_q8) {
      const auto& sc = *reinterpret_cast<array*>(scales);
      const int sb_dtype = sidecar_dtype_code(sc.dtype());
      if ((K & 31) != 0 || K < 32 || wa.dtype() != uint32 ||
          wa.shape(1) != K / 4 || sb_dtype < 0 || sc.ndim() != 2 ||
          sc.shape(0) != V_pad || sc.shape(1) != K / 32) {
        return nullptr;
      }
      inputs.push_back(sc);
      if (!symmetric) {
        const auto& bi = *reinterpret_cast<array*>(biases);
        if (bi.ndim() != 2 || bi.dtype() != sc.dtype() ||
            bi.shape(0) != V_pad || bi.shape(1) != K / 32) {
          return nullptr;
        }
        inputs.push_back(bi);
      }
    } else {
      if (biases != nullptr || (K & 7) != 0 || wa.dtype() != bfloat16 ||
          wa.shape(1) != K) {
        return nullptr;
      }
    }

    const int M = static_cast<int>(rows);
    Shape out_shape = (out_mode == 1) ? Shape{M} : Shape{M, unpadded_vocab};
    const Dtype out_dtype =
        (out_mode == 0) ? float32 : ((out_mode == 1) ? int32 : bfloat16);
    auto result = array(
        std::move(out_shape), out_dtype,
        std::make_shared<Inkling2LmHeadSoftcapCuda>(
            default_stream(Device::gpu), is_q8, symmetric, unpadded_vocab,
            softcap, out_mode),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_lm_head_softcap_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_lm_head_softcap_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" mlx_array* mlx_inkling2_small_bf16_proj_cuda(
    mlx_array* x,
    mlx_array* weight) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!x || !weight || default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& xa = *reinterpret_cast<array*>(x);
    const auto& wa = *reinterpret_cast<array*>(weight);
    if (xa.dtype() != bfloat16 || wa.dtype() != bfloat16 ||
        xa.ndim() < 2 || wa.ndim() != 2) {
      return nullptr;
    }
    const int K = xa.shape(-1);
    const int N = wa.shape(0);
    if (K < 8 || (K & 7) != 0 || wa.shape(1) != K || N < 1 || N > 128) {
      return nullptr;
    }
    const int64_t rows = static_cast<int64_t>(xa.size()) / K;
    if (rows < 1 || rows > 16) {
      return nullptr;
    }
    auto out_shape = xa.shape();
    out_shape.back() = N;
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Inkling2SmallBf16ProjCuda>(default_stream(Device::gpu)),
        {xa, wa});
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_small_bf16_proj_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_small_bf16_proj_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" bool mlx_inkling2_commit_short_conv_tapes_cuda(
    mlx_array* const* hists_in,
    mlx_array* const* rows_in,
    size_t n_streams,
    mlx_array* accepted_nodes,
    int contiguous_keep,
    mlx_array** hists_out) {
  if (hists_out) {
    for (size_t s = 0; s < n_streams; ++s) {
      hists_out[s] = nullptr;
    }
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!hists_in || !rows_in || !hists_out || n_streams < 1 || n_streams > 128 ||
        default_device() != Device::gpu) {
      return false;
    }
    const bool has_nodes = (accepted_nodes != nullptr);
    std::vector<array> inputs;
    inputs.reserve(n_streams * 2 + (has_nodes ? 1 : 0));

    int C = 0;
    int T = 0;
    for (size_t s = 0; s < n_streams; ++s) {
      if (!hists_in[s]) {
        return false;
      }
      const auto& h = *reinterpret_cast<array*>(hists_in[s]);
      if (h.ndim() != 2 || h.dtype() != bfloat16 || h.shape(0) != 3 || h.shape(1) < 1) {
        return false;
      }
      if (s == 0) {
        C = h.shape(1);
      } else if (h.shape(1) != C) {
        return false;
      }
      inputs.push_back(h);
    }
    for (size_t s = 0; s < n_streams; ++s) {
      if (!rows_in[s]) {
        return false;
      }
      const auto& r = *reinterpret_cast<array*>(rows_in[s]);
      if (r.ndim() != 2 || r.dtype() != bfloat16 || r.shape(0) < 1 || r.shape(1) != C) {
        return false;
      }
      if (s == 0) {
        T = r.shape(0);
      } else if (r.shape(0) != T) {
        return false;
      }
      inputs.push_back(r);
    }

    if (has_nodes) {
      const auto& na = *reinterpret_cast<array*>(accepted_nodes);
      if (na.ndim() != 1 || na.dtype() != int32 || na.shape(0) < 1) {
        return false;
      }
      inputs.push_back(na);
    } else if (contiguous_keep < 0 || contiguous_keep > T) {
      return false;
    }

    std::vector<Shape> out_shapes(n_streams, Shape{3, C});
    std::vector<Dtype> out_dtypes(n_streams, bfloat16);
    auto outputs = array::make_arrays(
        std::move(out_shapes), std::move(out_dtypes),
        std::make_shared<Inkling2CommitShortConvTapesCuda>(
            default_stream(Device::gpu), n_streams, has_nodes, contiguous_keep),
        std::move(inputs));
    for (size_t s = 0; s < n_streams; ++s) {
      hists_out[s] = reinterpret_cast<mlx_array*>(new array(std::move(outputs[s])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_commit_short_conv_tapes_cuda: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_commit_short_conv_tapes_cuda: unknown exception\n");
    return false;
  }
}

extern "C" mlx_array* mlx_inkling2_embed_rmsnorm_cuda(
    mlx_array* token_ids,
    mlx_array* weight,
    mlx_array* scales,
    mlx_array* biases,
    mlx_array* norm_weight,
    float eps) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!token_ids || !weight || !norm_weight || std::isnan(eps) ||
        default_device() != Device::gpu) {
      return nullptr;
    }
    const auto& tids = *reinterpret_cast<array*>(token_ids);
    const auto& wa = *reinterpret_cast<array*>(weight);
    const auto& nwa = *reinterpret_cast<array*>(norm_weight);

    if ((tids.dtype() != int32 && tids.dtype() != uint32) ||
        !((tids.ndim() == 1 && tids.shape(0) >= 1) ||
          (tids.ndim() == 2 && tids.shape(0) == 1 && tids.shape(1) >= 1))) {
      return nullptr;
    }
    if (wa.ndim() != 2 || wa.shape(0) < 1 ||
        nwa.ndim() != 1 || nwa.dtype() != float32 || nwa.shape(0) < 1) {
      return nullptr;
    }
    const int V = wa.shape(0);
    const int D = nwa.shape(0);

    int mode = 0;
    std::vector<array> inputs;
    inputs.reserve(5);
    inputs.push_back(tids);
    inputs.push_back(wa);

    if (scales == nullptr) {
      if (biases != nullptr || wa.dtype() != bfloat16 || wa.shape(1) != D) {
        return nullptr;
      }
      mode = 0;
    } else {
      const auto& sc = *reinterpret_cast<array*>(scales);
      const int sb_dtype = sidecar_dtype_code(sc.dtype());
      if (D < 32 || (D & 31) != 0 || wa.dtype() != uint32 ||
          wa.shape(1) != D / 4 || sb_dtype < 0 || sc.ndim() != 2 ||
          sc.shape(0) != V || sc.shape(1) != D / 32) {
        return nullptr;
      }
      inputs.push_back(sc);
      if (biases == nullptr) {
        mode = 1;
      } else {
        const auto& bi = *reinterpret_cast<array*>(biases);
        if (bi.ndim() != 2 || bi.dtype() != sc.dtype() ||
            bi.shape(0) != V || bi.shape(1) != D / 32) {
          return nullptr;
        }
        inputs.push_back(bi);
        mode = 2;
      }
    }
    inputs.push_back(nwa);

    auto out_shape = tids.shape();
    out_shape.push_back(D);
    auto result = array(
        std::move(out_shape), bfloat16,
        std::make_shared<Inkling2EmbedRmsNormCuda>(
            default_stream(Device::gpu), mode, eps),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_embed_rmsnorm_cuda: %s\n", e.what());
    return nullptr;
  } catch (...) {
    std::fprintf(stderr, "mlx_inkling2_embed_rmsnorm_cuda: unknown exception\n");
    return nullptr;
  }
}

extern "C" bool mlx_inkling2_compact_kv_batch_cuda(
    mlx_array* const* caches,
    size_t n_caches,
    const int* write_offsets,
    mlx_array* accepted_nodes,
    mlx_array** caches_out) {
  if (caches_out) {
    for (size_t i = 0; i < n_caches; ++i) {
      caches_out[i] = nullptr;
    }
  }
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    if (!caches || !write_offsets || !accepted_nodes || !caches_out ||
        n_caches < 1 || n_caches > 192 || default_device() != Device::gpu) {
      return false;
    }

    const auto& na = *reinterpret_cast<array*>(accepted_nodes);
    if (na.ndim() != 1 || na.dtype() != int32) {
      return false;
    }
    const int K = na.shape(0);
    if (K < 1 || K > 16) {
      return false;
    }

    std::vector<array> inputs;
    inputs.reserve(n_caches + 1);
    std::vector<Shape> out_shapes;
    out_shapes.reserve(n_caches);
    std::vector<Dtype> out_dtypes;
    out_dtypes.reserve(n_caches);
    std::vector<int> offsets(write_offsets, write_offsets + n_caches);

    int H_kv = 0;
    for (size_t i = 0; i < n_caches; ++i) {
      if (!caches[i]) {
        return false;
      }
      const auto& c = *reinterpret_cast<array*>(caches[i]);
      int h_i = 0, cap_i = 0, d_i = 0, eb_i = 0;
      int64_t sh_i = 0, st_i = 0;
      if (!kv_compact_cache_info(c, h_i, cap_i, d_i, sh_i, st_i, eb_i)) {
        return false;
      }
      if (i == 0) {
        H_kv = h_i;
      } else if (h_i != H_kv) {
        return false;
      }
      if (offsets[i] < 0 || static_cast<int64_t>(offsets[i]) + K > cap_i) {
        return false;
      }
      inputs.push_back(c);
      out_shapes.push_back(c.shape());
      out_dtypes.push_back(c.dtype());
    }
    inputs.push_back(na);

    auto outputs = array::make_arrays(
        std::move(out_shapes), std::move(out_dtypes),
        std::make_shared<Inkling2CompactKvBatchCuda>(
            default_stream(Device::gpu), n_caches, std::move(offsets)),
        std::move(inputs));
    for (size_t i = 0; i < n_caches; ++i) {
      caches_out[i] =
          reinterpret_cast<mlx_array*>(new array(std::move(outputs[i])));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_compact_kv_batch_cuda: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(
        stderr, "mlx_inkling2_compact_kv_batch_cuda: unknown exception\n");
    return false;
  }
}

#else

// ============================================================================
// Non-CUDA Stubs (macOS / CPU-only builds)
// ============================================================================

extern "C" mlx_array* mlx_inkling2_block_attn_res_norm_cuda(
    mlx_array* const* sources,
    size_t n_sources,
    mlx_array* queries,
    mlx_array* norm_weights,
    float eps,
    bool squeeze_q1) {
  (void)sources;
  (void)n_sources;
  (void)queries;
  (void)norm_weights;
  (void)eps;
  (void)squeeze_q1;
  return nullptr;
}

extern "C" mlx_array* mlx_inkling2_loop_ctrl_norm_cuda(
    mlx_array* dest,
    mlx_array* msg,
    mlx_array* down_w,
    mlx_array* gain_forget_w,
    mlx_array* norm_w,
    float eps,
    int g_count,
    bool squeeze_g1) {
  (void)dest;
  (void)msg;
  (void)down_w;
  (void)gain_forget_w;
  (void)norm_w;
  (void)eps;
  (void)g_count;
  (void)squeeze_g1;
  return nullptr;
}

extern "C" bool mlx_inkling2_kv_conv_prep_store_cuda(
    mlx_array* q_raw,
    mlx_array* r_raw,
    mlx_array* k_raw,
    mlx_array* v_raw,
    mlx_array* q_clip,
    mlx_array* rel_clip,
    mlx_array* k_conv_w,
    mlx_array* v_conv_w,
    mlx_array* dyn_k,
    mlx_array* dyn_v,
    mlx_array* k_hist_in,
    mlx_array* v_hist_in,
    mlx_array* kv_k_cache,
    mlx_array* kv_v_cache,
    mlx_array* kv_k_scales,
    mlx_array* kv_v_scales,
    mlx_array* conv_taps,
    mlx_array* depths,
    int start_pos,
    int write_offset,
    int ring_cap,
    bool is_global,
    float log_n_floor,
    float log_alpha,
    bool update_hist,
    mlx_array** out_q,
    mlx_array** out_r,
    mlx_array** out_k_hist,
    mlx_array** out_v_hist,
    mlx_array** out_kv_k_alias,
    mlx_array** out_kv_v_alias,
    mlx_array** out_ks_alias,
    mlx_array** out_vs_alias) {
  (void)q_raw;
  (void)r_raw;
  (void)k_raw;
  (void)v_raw;
  (void)q_clip;
  (void)rel_clip;
  (void)k_conv_w;
  (void)v_conv_w;
  (void)dyn_k;
  (void)dyn_v;
  (void)k_hist_in;
  (void)v_hist_in;
  (void)kv_k_cache;
  (void)kv_v_cache;
  (void)kv_k_scales;
  (void)kv_v_scales;
  (void)conv_taps;
  (void)depths;
  (void)start_pos;
  (void)write_offset;
  (void)ring_cap;
  (void)is_global;
  (void)log_n_floor;
  (void)log_alpha;
  (void)update_hist;
  if (out_q) *out_q = nullptr;
  if (out_r) *out_r = nullptr;
  if (out_k_hist) *out_k_hist = nullptr;
  if (out_v_hist) *out_v_hist = nullptr;
  if (out_kv_k_alias) *out_kv_k_alias = nullptr;
  if (out_kv_v_alias) *out_kv_v_alias = nullptr;
  if (out_ks_alias) *out_ks_alias = nullptr;
  if (out_vs_alias) *out_vs_alias = nullptr;
  return false;
}

extern "C" bool mlx_inkling2_sconv_k4_residual_cuda(
    mlx_array* x,
    mlx_array* weight,
    mlx_array* hist_in,
    mlx_array* residual,
    mlx_array* conv_taps,
    bool update_hist,
    mlx_array** out_y,
    mlx_array** out_hist) {
  (void)x;
  (void)weight;
  (void)hist_in;
  (void)residual;
  (void)conv_taps;
  (void)update_hist;
  if (out_y) *out_y = nullptr;
  if (out_hist) *out_hist = nullptr;
  return false;
}

extern "C" mlx_array* mlx_inkling2_lm_head_softcap_cuda(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int unpadded_vocab,
    float softcap,
    int out_mode) {
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  (void)unpadded_vocab;
  (void)softcap;
  (void)out_mode;
  return nullptr;
}

extern "C" mlx_array* mlx_inkling2_small_bf16_proj_cuda(
    mlx_array* x,
    mlx_array* weight) {
  (void)x;
  (void)weight;
  return nullptr;
}

extern "C" bool mlx_inkling2_commit_short_conv_tapes_cuda(
    mlx_array* const* hists_in,
    mlx_array* const* rows_in,
    size_t n_streams,
    mlx_array* accepted_nodes,
    int contiguous_keep,
    mlx_array** hists_out) {
  (void)hists_in;
  (void)rows_in;
  (void)accepted_nodes;
  (void)contiguous_keep;
  if (hists_out) {
    for (size_t s = 0; s < n_streams; ++s) {
      hists_out[s] = nullptr;
    }
  }
  return false;
}

extern "C" mlx_array* mlx_inkling2_embed_rmsnorm_cuda(
    mlx_array* token_ids,
    mlx_array* weight,
    mlx_array* scales,
    mlx_array* biases,
    mlx_array* norm_weight,
    float eps) {
  (void)token_ids;
  (void)weight;
  (void)scales;
  (void)biases;
  (void)norm_weight;
  (void)eps;
  return nullptr;
}

extern "C" bool mlx_inkling2_compact_kv_batch_cuda(
    mlx_array* const* caches,
    size_t n_caches,
    const int* write_offsets,
    mlx_array* accepted_nodes,
    mlx_array** caches_out) {
  (void)caches;
  (void)write_offsets;
  (void)accepted_nodes;
  if (caches_out) {
    for (size_t i = 0; i < n_caches; ++i) {
      caches_out[i] = nullptr;
    }
  }
  return false;
}

#endif  // MLX_NODE_CUDA_ENABLED
