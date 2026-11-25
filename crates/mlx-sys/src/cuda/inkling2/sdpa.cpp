#include "cuda/inkling2/common.h"

#ifdef MLX_NODE_CUDA_ENABLED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mlx::core::inkling2_ops {
namespace {

using namespace mlx_node::inkling2_cuda;

constexpr char inkling2_sdpa_few_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/inkling2_sdpa_few.cu.inc"
    ;

constexpr char inkling2_sdpa_prefill_source[] =
#include "cuda/ptx_helpers.cu.inc"
#include "cuda/inkling2_sdpa_prefill.cu.inc"
    ;

constexpr float kInvSqrt128 = 0.08838834764831845f;
constexpr float kScaleLog2 = kInvSqrt128 * kLog2E;

inline bool check_r_shape(const array& r, int B, int T, int Hq) {
  if (r.dtype() != float32) return false;
  if (r.ndim() == 3) {
    return r.shape(0) == static_cast<int64_t>(B) * T && r.shape(1) == Hq &&
        r.shape(2) == kRelDim;
  }
  if (r.ndim() == 4) {
    return r.shape(0) == B && r.shape(1) == T && r.shape(2) == Hq &&
        r.shape(3) == kRelDim;
  }
  return false;
}

inline bool check_gate_shape(const array& g, int B, int T, int Hq) {
  if (g.dtype() != bfloat16) return false;
  const int64_t hd = static_cast<int64_t>(Hq) * kHeadDim;
  if (g.ndim() == 2) {
    return g.shape(0) == static_cast<int64_t>(B) * T && g.shape(1) == hd;
  }
  if (g.ndim() == 3) {
    if (B == 1 && g.shape(0) == T && g.shape(1) == Hq && g.shape(2) == kHeadDim) {
      return true;
    }
    return g.shape(0) == B && g.shape(1) == T && g.shape(2) == hd;
  }
  if (g.ndim() == 4) {
    return g.shape(0) == B && g.shape(1) == T && g.shape(2) == Hq &&
        g.shape(3) == kHeadDim;
  }
  return false;
}

inline bool check_sink_shape(const array& s, int B, int T, int Hq) {
  if (s.dtype() != bfloat16 && s.dtype() != float32) return false;
  if (s.ndim() == 2) {
    return s.shape(0) == static_cast<int64_t>(B) * T && s.shape(1) == Hq;
  }
  if (s.ndim() == 3) {
    return s.shape(0) == B && s.shape(1) == T && s.shape(2) == Hq;
  }
  return false;
}

inline bool check_scales_shape(const array& sc, int B, int Hkv, int cap) {
  if (sc.dtype() != float32) return false;
  if (sc.ndim() == 4) {
    return sc.shape(0) == B && sc.shape(1) == Hkv && sc.shape(2) == cap &&
        sc.shape(3) == 1;
  }
  if (sc.ndim() == 3) {
    return sc.shape(0) == B && sc.shape(1) == Hkv && sc.shape(2) == cap;
  }
  return false;
}

inline array ensure_q_aligned(
    const array& a, const Stream& stream, cu::CommandEncoder& encoder) {
  const bool ok = a.strides(3) == 1 &&
      a.strides(2) >= 0 && (a.strides(2) % 8 == 0) &&
      a.strides(1) >= 0 && (a.strides(1) % 8 == 0) &&
      a.strides(0) >= 0 && (a.strides(0) % 8 == 0) &&
      (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
  if (ok) {
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  return copy;
}

inline array ensure_kv_aligned(
    const array& a, bool is_int8, const Stream& stream, cu::CommandEncoder& encoder) {
  const int64_t elem_align = is_int8 ? 16 : 8;
  const bool ok = a.strides(3) == 1 &&
      a.strides(2) >= 0 && (a.strides(2) % elem_align == 0) &&
      a.strides(1) >= 0 && (a.strides(1) % elem_align == 0) &&
      a.strides(0) >= 0 && (a.strides(0) % elem_align == 0) &&
      (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
  if (ok) {
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  return copy;
}

inline array ensure_r_layout(
    const array& a, int B, int T, int Hq, const Stream& stream,
    cu::CommandEncoder& encoder, int64_t& r_sb, int64_t& r_st) {
  bool ok = false;
  int64_t sb = 0;
  int64_t st = 0;
  if (a.ndim() == 3) {
    ok = a.strides(2) == 1 && a.strides(1) == kRelDim &&
        a.strides(0) >= 0 && (a.strides(0) % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
    st = a.strides(0);
    sb = (B == 1) ? 0LL : (static_cast<int64_t>(T) * st);
  } else if (a.ndim() == 4) {
    ok = a.strides(3) == 1 && a.strides(2) == kRelDim &&
        a.strides(1) >= 0 && (a.strides(1) % 4 == 0) &&
        (B == 1 || (a.strides(0) >= 0 && (a.strides(0) % 4 == 0))) &&
        (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16 == 0);
    st = a.strides(1);
    sb = (B == 1) ? 0LL : a.strides(0);
  }
  if (ok) {
    r_sb = sb;
    r_st = st;
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  r_st = static_cast<int64_t>(Hq) * kRelDim;
  r_sb = (B == 1) ? 0LL : (static_cast<int64_t>(T) * r_st);
  return copy;
}

inline array ensure_gate_layout(
    const array& a, int B, int T, int Hq, const Stream& stream,
    cu::CommandEncoder& encoder, int64_t& gate_sb, int64_t& gate_st) {
  bool ok = false;
  int64_t sb = 0;
  int64_t st = 0;
  const uintptr_t ptr = reinterpret_cast<uintptr_t>(gpu_ptr<void>(a));
  if (a.ndim() == 2) {
    ok = a.strides(1) == 1 && a.strides(0) >= 0 && (a.strides(0) % 4 == 0) &&
        (ptr % 8 == 0);
    st = a.strides(0);
    sb = (B == 1) ? 0LL : (static_cast<int64_t>(T) * st);
  } else if (a.ndim() == 3) {
    if (B == 1 && a.shape(0) == T && a.shape(1) == Hq && a.shape(2) == kHeadDim) {
      ok = a.strides(2) == 1 && a.strides(1) == kHeadDim &&
          a.strides(0) >= 0 && (a.strides(0) % 4 == 0) && (ptr % 8 == 0);
      st = a.strides(0);
      sb = 0LL;
    } else {
      ok = a.strides(2) == 1 && a.strides(1) >= 0 && (a.strides(1) % 4 == 0) &&
          (B == 1 || (a.strides(0) >= 0 && (a.strides(0) % 4 == 0))) &&
          (ptr % 8 == 0);
      st = a.strides(1);
      sb = (B == 1) ? 0LL : a.strides(0);
    }
  } else if (a.ndim() == 4) {
    ok = a.strides(3) == 1 && a.strides(2) == kHeadDim &&
        a.strides(1) >= 0 && (a.strides(1) % 4 == 0) &&
        (B == 1 || (a.strides(0) >= 0 && (a.strides(0) % 4 == 0))) &&
        (ptr % 8 == 0);
    st = a.strides(1);
    sb = (B == 1) ? 0LL : a.strides(0);
  }
  if (ok) {
    gate_sb = sb;
    gate_st = st;
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  gate_st = static_cast<int64_t>(Hq) * kHeadDim;
  gate_sb = (B == 1) ? 0LL : (static_cast<int64_t>(T) * gate_st);
  return copy;
}

inline array ensure_sink_layout(
    const array& a, int B, int T, int Hq, const Stream& stream,
    cu::CommandEncoder& encoder, int64_t& sink_sb, int64_t& sink_st) {
  bool ok = false;
  int64_t sb = 0;
  int64_t st = 0;
  if (a.ndim() == 2) {
    ok = a.strides(1) == 1 && a.strides(0) >= 0;
    st = a.strides(0);
    sb = (B == 1) ? 0LL : (static_cast<int64_t>(T) * st);
  } else if (a.ndim() == 3) {
    ok = a.strides(2) == 1 && a.strides(1) >= 0 && (B == 1 || a.strides(0) >= 0);
    st = a.strides(1);
    sb = (B == 1) ? 0LL : a.strides(0);
  }
  if (ok) {
    sink_sb = sb;
    sink_st = st;
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  sink_st = static_cast<int64_t>(Hq);
  sink_sb = (B == 1) ? 0LL : (static_cast<int64_t>(T) * sink_st);
  return copy;
}

inline array ensure_scales_layout(
    const array& a, const Stream& stream, cu::CommandEncoder& encoder,
    int64_t& sb, int64_t& sh, int64_t& ss) {
  const bool ok = a.strides(0) >= 0 && a.strides(1) >= 0 && a.strides(2) >= 0;
  if (ok) {
    sb = a.strides(0);
    sh = a.strides(1);
    ss = a.strides(2);
    return a;
  }
  auto copy = contiguous_copy_gpu(a, stream);
  encoder.add_temporary(copy);
  sb = copy.strides(0);
  sh = copy.strides(1);
  ss = copy.strides(2);
  return copy;
}

// ============================================================================
// 1. Inkling-2-Mini Decode / Speculative Verify SDPA (T in 1..=16)
// ============================================================================
class Inkling2SdpaFewCuda final : public UnaryPrimitive {
 public:
  Inkling2SdpaFewCuda(
      Stream stream,
      int start_pos,
      int cache_len,
      int kv_start_pos,
      int kv_phys_base,
      int write_offset,
      int ring_cap,
      bool is_local,
      int sliding_window,
      bool is_int8,
      bool is_tree,
      bool sink_is_f32)
      : UnaryPrimitive(stream),
        start_pos_(start_pos),
        cache_len_(cache_len),
        kv_start_pos_(kv_start_pos),
        kv_phys_base_(kv_phys_base),
        write_offset_(write_offset),
        ring_cap_(ring_cap),
        is_local_(is_local),
        sliding_window_(sliding_window),
        is_int8_(is_int8),
        is_tree_(is_tree),
        sink_is_f32_(sink_is_f32) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2SdpaFewCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());

    const size_t expected_inputs =
        7u + (is_int8_ ? 2u : 0u) + (is_tree_ ? 2u : 0u);
    if (inputs.size() < expected_inputs) {
      throw std::invalid_argument("Inkling2SdpaFewCuda: missing inputs");
    }

    array q = ensure_q_aligned(inputs[0], stream(), encoder);
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int T = q.shape(2);
    const int D = q.shape(3);
    array k_cache = ensure_kv_aligned(inputs[2], is_int8_, stream(), encoder);
    array v_cache = ensure_kv_aligned(inputs[3], is_int8_, stream(), encoder);
    const int Hkv = k_cache.shape(1);
    const int cap = k_cache.shape(2);
    if (B <= 0 || B > 65535 || Hkv <= 0 || Hkv > 65535 ||
        Hq < Hkv || Hq % Hkv != 0 || D != kHeadDim || T < 1 || T > 16 ||
        k_cache.shape(0) != B || k_cache.shape(3) != kHeadDim ||
        v_cache.shape() != k_cache.shape()) {
      throw std::invalid_argument("Inkling2SdpaFewCuda: unsupported replay shape");
    }
    const int gqa = Hq / Hkv;
    const int M = gqa * T;
    const int num_warps = (M + 15) / 16;
    if (num_warps < 1 || num_warps > 4) {
      throw std::invalid_argument("Inkling2SdpaFewCuda: M exceeds 64");
    }
    if ((ring_cap_ > 0 && ring_cap_ > cap) ||
        (ring_cap_ == 0 && (cache_len_ > cap || write_offset_ > cap - T))) {
      throw std::invalid_argument("Inkling2SdpaFewCuda: cache bounds exceeded");
    }

    int64_t r_sb = 0, r_st = 0;
    array r = ensure_r_layout(inputs[1], B, T, Hq, stream(), encoder, r_sb, r_st);
    array rel_bias_w =
        cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
    const int rel_extent = static_cast<int>(rel_bias_w.shape(0));

    int64_t gate_sb = 0, gate_st = 0;
    array gate =
        ensure_gate_layout(inputs[5], B, T, Hq, stream(), encoder, gate_sb, gate_st);
    int64_t sink_sb = 0, sink_st = 0;
    array sink =
        ensure_sink_layout(inputs[6], B, T, Hq, stream(), encoder, sink_sb, sink_st);

    size_t idx = 7u;
    std::optional<array> k_scales_arr;
    std::optional<array> v_scales_arr;
    CUdeviceptr k_scales_ptr = 0;
    CUdeviceptr v_scales_ptr = 0;
    int64_t ks_sb = 0, ks_sh = 0, ks_ss = 0;
    int64_t vs_sb = 0, vs_sh = 0, vs_ss = 0;
    if (is_int8_) {
      k_scales_arr = ensure_scales_layout(
          inputs[idx++], stream(), encoder, ks_sb, ks_sh, ks_ss);
      v_scales_arr = ensure_scales_layout(
          inputs[idx++], stream(), encoder, vs_sb, vs_sh, vs_ss);
      k_scales_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*k_scales_arr));
      v_scales_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*v_scales_arr));
    }

    std::optional<array> tree_depths_arr;
    std::optional<array> tree_mask_arr;
    CUdeviceptr tree_depths_ptr = 0;
    CUdeviceptr tree_mask_ptr = 0;
    if (is_tree_) {
      tree_depths_arr =
          cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
      tree_mask_arr =
          cu::ensure_row_contiguous_aligned<4>(inputs[idx++], stream(), encoder);
      tree_depths_ptr =
          reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*tree_depths_arr));
      tree_mask_ptr =
          reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*tree_mask_arr));
    }

    const int K_total = cache_len_ + T;
    const int k_align = kv_start_pos_ & 31;
    const int abs_tile_base = kv_start_pos_ >> 5;
    const int total_aligned_tiles = (K_total + k_align + 31) / 32;
    int host_tile_first = 0;
    if (is_local_) {
      int pref_skip = (start_pos_ - (sliding_window_ - 1)) - kv_start_pos_;
      pref_skip = std::clamp(pref_skip, 0, cache_len_);
      host_tile_first = (pref_skip + k_align) / 32;
    }
    const int abs_tile_first = abs_tile_base + host_tile_first;
    const int abs_tile_end = abs_tile_base + total_aligned_tiles;
    // Fixed absolute-tile-aligned Split-K buckets so M=1 AR and M=2..=16 tree verify
    // place every KV token in the exact same 32-key MMA tile and split boundary (0-ULP).
    int tiles_per_split = is_local_ ? 3 : 4;
    const int max_splits = (B * Hkv <= 8) ? 16 : 8;
    auto calc_splits = [&](int tps) -> int {
      const int base = (abs_tile_first / tps) * tps;
      return std::max(1, (abs_tile_end - base + tps - 1) / tps);
    };
    while (calc_splits(tiles_per_split) > max_splits) {
      tiles_per_split *= 4;
    }
    const int abs_split_base = (abs_tile_first / tiles_per_split) * tiles_per_split;
    const int num_splits = calc_splits(tiles_per_split);

    array partial_o =
        alloc_temp({B, Hkv, num_splits, M, kHeadDim}, float32, encoder);
    array partial_m = alloc_temp({B, Hkv, num_splits, M}, float32, encoder);
    array partial_l = alloc_temp({B, Hkv, num_splits, M}, float32, encoder);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    // Stage 1: Split-K Tensor Core Attention
    encoder.set_input_array(q);
    encoder.set_input_array(r);
    encoder.set_input_array(k_cache);
    encoder.set_input_array(v_cache);
    encoder.set_input_array(rel_bias_w);
    if (k_scales_arr) {
      encoder.set_input_array(*k_scales_arr);
      encoder.set_input_array(*v_scales_arr);
    }
    if (tree_depths_arr) {
      encoder.set_input_array(*tree_depths_arr);
      encoder.set_input_array(*tree_mask_arr);
    }
    encoder.set_output_array(partial_o);
    encoder.set_output_array(partial_m);
    encoder.set_output_array(partial_l);

    // [num_warps - 1][is_local][is_int8][is_tree]
    static constexpr const char* kFewSplitEntries[4][2][2][2] = {
        {
            {{"mlx_inkling2_sdpa_few_split<1,false,false,false>",
              "mlx_inkling2_sdpa_few_split<1,false,false,true>"},
             {"mlx_inkling2_sdpa_few_split<1,false,true,false>",
              "mlx_inkling2_sdpa_few_split<1,false,true,true>"}},
            {{"mlx_inkling2_sdpa_few_split<1,true,false,false>",
              "mlx_inkling2_sdpa_few_split<1,true,false,true>"},
             {"mlx_inkling2_sdpa_few_split<1,true,true,false>",
              "mlx_inkling2_sdpa_few_split<1,true,true,true>"}},
        },
        {
            {{"mlx_inkling2_sdpa_few_split<2,false,false,false>",
              "mlx_inkling2_sdpa_few_split<2,false,false,true>"},
             {"mlx_inkling2_sdpa_few_split<2,false,true,false>",
              "mlx_inkling2_sdpa_few_split<2,false,true,true>"}},
            {{"mlx_inkling2_sdpa_few_split<2,true,false,false>",
              "mlx_inkling2_sdpa_few_split<2,true,false,true>"},
             {"mlx_inkling2_sdpa_few_split<2,true,true,false>",
              "mlx_inkling2_sdpa_few_split<2,true,true,true>"}},
        },
        {
            {{"mlx_inkling2_sdpa_few_split<3,false,false,false>",
              "mlx_inkling2_sdpa_few_split<3,false,false,true>"},
             {"mlx_inkling2_sdpa_few_split<3,false,true,false>",
              "mlx_inkling2_sdpa_few_split<3,false,true,true>"}},
            {{"mlx_inkling2_sdpa_few_split<3,true,false,false>",
              "mlx_inkling2_sdpa_few_split<3,true,false,true>"},
             {"mlx_inkling2_sdpa_few_split<3,true,true,false>",
              "mlx_inkling2_sdpa_few_split<3,true,true,true>"}},
        },
        {
            {{"mlx_inkling2_sdpa_few_split<4,false,false,false>",
              "mlx_inkling2_sdpa_few_split<4,false,false,true>"},
             {"mlx_inkling2_sdpa_few_split<4,false,true,false>",
              "mlx_inkling2_sdpa_few_split<4,false,true,true>"}},
            {{"mlx_inkling2_sdpa_few_split<4,true,false,false>",
              "mlx_inkling2_sdpa_few_split<4,true,false,true>"},
             {"mlx_inkling2_sdpa_few_split<4,true,true,false>",
              "mlx_inkling2_sdpa_few_split<4,true,true,true>"}},
        },
    };
    auto split_kernel = cu::jit_kernel(
        encoder.device(),
        inkling2_sdpa_few_source,
        kFewSplitEntries[num_warps - 1][is_local_ ? 1 : 0][is_int8_ ? 1 : 0]
                        [is_tree_ ? 1 : 0]);
    cu::KernelArgs sargs;
    sargs.append(q);
    sargs.append(r);
    sargs.append(k_cache);
    sargs.append(v_cache);
    sargs.append<CUdeviceptr>(k_scales_ptr);
    sargs.append<CUdeviceptr>(v_scales_ptr);
    sargs.append(rel_bias_w);
    sargs.append<CUdeviceptr>(tree_depths_ptr);
    sargs.append<CUdeviceptr>(tree_mask_ptr);
    sargs.append(partial_o);
    sargs.append(partial_m);
    sargs.append(partial_l);
    sargs.append<int32_t>(Hkv);
    sargs.append<int32_t>(gqa);
    sargs.append<int32_t>(T);
    sargs.append<int32_t>(start_pos_);
    sargs.append<int32_t>(cache_len_);
    sargs.append<int32_t>(kv_start_pos_);
    sargs.append<int32_t>(kv_phys_base_);
    sargs.append<int32_t>(write_offset_);
    sargs.append<int32_t>(ring_cap_);
    sargs.append<int32_t>(sliding_window_);
    sargs.append<int32_t>(rel_extent);
    sargs.append<int64_t>(q.strides(0));
    sargs.append<int64_t>(q.strides(1));
    sargs.append<int64_t>(q.strides(2));
    sargs.append<int64_t>(r_sb);
    sargs.append<int64_t>(r_st);
    sargs.append<int64_t>(k_cache.strides(0));
    sargs.append<int64_t>(k_cache.strides(1));
    sargs.append<int64_t>(k_cache.strides(2));
    sargs.append<int64_t>(v_cache.strides(0));
    sargs.append<int64_t>(v_cache.strides(1));
    sargs.append<int64_t>(v_cache.strides(2));
    sargs.append<int64_t>(ks_sb);
    sargs.append<int64_t>(ks_sh);
    sargs.append<int64_t>(ks_ss);
    sargs.append<int64_t>(vs_sb);
    sargs.append<int64_t>(vs_sh);
    sargs.append<int64_t>(vs_ss);
    sargs.append<float>(kScaleLog2);
    sargs.append<float>(kLog2E);
    sargs.append<int32_t>(num_splits);
    sargs.append<int32_t>(tiles_per_split);
    sargs.append<int32_t>(abs_split_base);
    encoder.add_kernel_node_raw(
        split_kernel, dim3(num_splits, Hkv, B), dim3(32, num_warps, 1), {}, 0,
        sargs.args());

    // Stage 2: Fused Split-K Reduction + Gate + Sink Epilogue -> [T, Hq*128] BF16
    encoder.set_input_array(partial_o);
    encoder.set_input_array(partial_m);
    encoder.set_input_array(partial_l);
    encoder.set_input_array(gate);
    encoder.set_input_array(sink);
    encoder.set_output_array(out);
    static constexpr const char* kFewReduceEntries[2] = {
        "mlx_inkling2_sdpa_few_reduce<false>",
        "mlx_inkling2_sdpa_few_reduce<true>",
    };
    auto reduce_kernel = cu::jit_kernel(
        encoder.device(),
        inkling2_sdpa_few_source,
        kFewReduceEntries[sink_is_f32_ ? 1 : 0]);
    cu::KernelArgs rargs;
    rargs.append(partial_o);
    rargs.append(partial_m);
    rargs.append(partial_l);
    rargs.append(gate);
    rargs.append(sink);
    rargs.append(out);
    rargs.append<int32_t>(Hkv);
    rargs.append<int32_t>(gqa);
    rargs.append<int32_t>(T);
    rargs.append<int32_t>(num_splits);
    rargs.append<int64_t>(gate_sb);
    rargs.append<int64_t>(gate_st);
    rargs.append<int64_t>(sink_sb);
    rargs.append<int64_t>(sink_st);
    encoder.add_kernel_node_raw(
        reduce_kernel, dim3(M, Hkv, B), dim3(32, 1, 1), {}, 0, rargs.args());
  }

  DEFINE_NAME(Inkling2SdpaFewCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2SdpaFewCuda*>(&other);
    return rhs && rhs->start_pos_ == start_pos_ &&
        rhs->cache_len_ == cache_len_ && rhs->kv_start_pos_ == kv_start_pos_ &&
        rhs->kv_phys_base_ == kv_phys_base_ &&
        rhs->write_offset_ == write_offset_ && rhs->ring_cap_ == ring_cap_ &&
        rhs->is_local_ == is_local_ &&
        rhs->sliding_window_ == sliding_window_ && rhs->is_int8_ == is_int8_ &&
        rhs->is_tree_ == is_tree_ && rhs->sink_is_f32_ == sink_is_f32_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int T = q.shape(2);
    if (B == 1) {
      return {Shape{T, Hq * kHeadDim}};
    }
    return {Shape{B, T, Hq * kHeadDim}};
  }

 private:
  int start_pos_;
  int cache_len_;
  int kv_start_pos_;
  int kv_phys_base_;
  int write_offset_;
  int ring_cap_;
  bool is_local_;
  int sliding_window_;
  bool is_int8_;
  bool is_tree_;
  bool sink_is_f32_;
};

// ============================================================================
// 2. Inkling-2-Mini FlashAttention Prefill SDPA (T >= 1, optimized for T > 16)
// ============================================================================
class Inkling2SdpaPrefillCuda final : public UnaryPrimitive {
 public:
  Inkling2SdpaPrefillCuda(
      Stream stream,
      int start_pos,
      int cache_len,
      int kv_start_pos,
      int kv_phys_base,
      int write_offset,
      int ring_cap,
      bool is_local,
      int sliding_window,
      bool is_int8,
      bool sink_is_f32)
      : UnaryPrimitive(stream),
        start_pos_(start_pos),
        cache_len_(cache_len),
        kv_start_pos_(kv_start_pos),
        kv_phys_base_(kv_phys_base),
        write_offset_(write_offset),
        ring_cap_(ring_cap),
        is_local_(is_local),
        sliding_window_(sliding_window),
        is_int8_(is_int8),
        sink_is_f32_(sink_is_f32) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling2SdpaPrefillCuda requires CUDA");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    auto& encoder = cu::get_command_encoder(stream());

    const size_t expected_inputs = 7u + (is_int8_ ? 2u : 0u);
    if (inputs.size() < expected_inputs) {
      throw std::invalid_argument("Inkling2SdpaPrefillCuda: missing inputs");
    }

    array q = ensure_q_aligned(inputs[0], stream(), encoder);
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int T = q.shape(2);
    const int D = q.shape(3);
    array k_cache = ensure_kv_aligned(inputs[2], is_int8_, stream(), encoder);
    array v_cache = ensure_kv_aligned(inputs[3], is_int8_, stream(), encoder);
    const int Hkv = k_cache.shape(1);
    const int cap = k_cache.shape(2);
    if (B <= 0 || B > 65535 || Hkv <= 0 || Hkv > 65535 ||
        Hq < Hkv || Hq % Hkv != 0 || D != kHeadDim || T < 1 ||
        k_cache.shape(0) != B || k_cache.shape(3) != kHeadDim ||
        v_cache.shape() != k_cache.shape()) {
      throw std::invalid_argument(
          "Inkling2SdpaPrefillCuda: unsupported replay shape");
    }
    const int gqa = Hq / Hkv;
    const int64_t num_q_blocks = (static_cast<int64_t>(T) + 63) / 64;
    if (num_q_blocks * gqa > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument(
          "Inkling2SdpaPrefillCuda: grid size exceeds CUDA limits");
    }
    if ((ring_cap_ > 0 && ring_cap_ > cap) ||
        (ring_cap_ == 0 && (cache_len_ > cap || write_offset_ > cap - T))) {
      throw std::invalid_argument(
          "Inkling2SdpaPrefillCuda: cache bounds exceeded");
    }

    int64_t r_sb = 0, r_st = 0;
    array r = ensure_r_layout(inputs[1], B, T, Hq, stream(), encoder, r_sb, r_st);
    array rel_bias_w =
        cu::ensure_row_contiguous_aligned<16>(inputs[4], stream(), encoder);
    const int rel_extent = static_cast<int>(rel_bias_w.shape(0));

    int64_t gate_sb = 0, gate_st = 0;
    array gate =
        ensure_gate_layout(inputs[5], B, T, Hq, stream(), encoder, gate_sb, gate_st);
    int64_t sink_sb = 0, sink_st = 0;
    array sink =
        ensure_sink_layout(inputs[6], B, T, Hq, stream(), encoder, sink_sb, sink_st);

    std::optional<array> k_scales_arr;
    std::optional<array> v_scales_arr;
    CUdeviceptr k_scales_ptr = 0;
    CUdeviceptr v_scales_ptr = 0;
    int64_t ks_sb = 0, ks_sh = 0, ks_ss = 0;
    int64_t vs_sb = 0, vs_sh = 0, vs_ss = 0;
    if (is_int8_) {
      k_scales_arr = ensure_scales_layout(
          inputs[7], stream(), encoder, ks_sb, ks_sh, ks_ss);
      v_scales_arr = ensure_scales_layout(
          inputs[8], stream(), encoder, vs_sb, vs_sh, vs_ss);
      k_scales_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*k_scales_arr));
      v_scales_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(*v_scales_arr));
    }

    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    encoder.set_input_array(q);
    encoder.set_input_array(r);
    encoder.set_input_array(k_cache);
    encoder.set_input_array(v_cache);
    encoder.set_input_array(rel_bias_w);
    encoder.set_input_array(gate);
    encoder.set_input_array(sink);
    if (k_scales_arr) {
      encoder.set_input_array(*k_scales_arr);
      encoder.set_input_array(*v_scales_arr);
    }
    encoder.set_output_array(out);

    // [is_local][is_int8][sink_is_f32]
    static constexpr const char* kPrefillEntries[2][2][2] = {
        {{"mlx_inkling2_sdpa_prefill<false,false,false>",
          "mlx_inkling2_sdpa_prefill<false,false,true>"},
         {"mlx_inkling2_sdpa_prefill<false,true,false>",
          "mlx_inkling2_sdpa_prefill<false,true,true>"}},
        {{"mlx_inkling2_sdpa_prefill<true,false,false>",
          "mlx_inkling2_sdpa_prefill<true,false,true>"},
         {"mlx_inkling2_sdpa_prefill<true,true,false>",
          "mlx_inkling2_sdpa_prefill<true,true,true>"}},
    };
    auto kernel = cu::jit_kernel(
        encoder.device(),
        inkling2_sdpa_prefill_source,
        kPrefillEntries[is_local_ ? 1 : 0][is_int8_ ? 1 : 0]
                       [sink_is_f32_ ? 1 : 0]);
    cu::KernelArgs args;
    args.append(q);
    args.append(r);
    args.append(k_cache);
    args.append(v_cache);
    args.append<CUdeviceptr>(k_scales_ptr);
    args.append<CUdeviceptr>(v_scales_ptr);
    args.append(rel_bias_w);
    args.append(gate);
    args.append(sink);
    args.append(out);
    args.append<int32_t>(Hkv);
    args.append<int32_t>(gqa);
    args.append<int32_t>(T);
    args.append<int32_t>(start_pos_);
    args.append<int32_t>(cache_len_);
    args.append<int32_t>(kv_start_pos_);
    args.append<int32_t>(kv_phys_base_);
    args.append<int32_t>(write_offset_);
    args.append<int32_t>(ring_cap_);
    args.append<int32_t>(sliding_window_);
    args.append<int32_t>(rel_extent);
    args.append<int64_t>(q.strides(0));
    args.append<int64_t>(q.strides(1));
    args.append<int64_t>(q.strides(2));
    args.append<int64_t>(r_sb);
    args.append<int64_t>(r_st);
    args.append<int64_t>(k_cache.strides(0));
    args.append<int64_t>(k_cache.strides(1));
    args.append<int64_t>(k_cache.strides(2));
    args.append<int64_t>(v_cache.strides(0));
    args.append<int64_t>(v_cache.strides(1));
    args.append<int64_t>(v_cache.strides(2));
    args.append<int64_t>(ks_sb);
    args.append<int64_t>(ks_sh);
    args.append<int64_t>(ks_ss);
    args.append<int64_t>(vs_sb);
    args.append<int64_t>(vs_sh);
    args.append<int64_t>(vs_ss);
    args.append<int64_t>(gate_sb);
    args.append<int64_t>(gate_st);
    args.append<int64_t>(sink_sb);
    args.append<int64_t>(sink_st);
    args.append<float>(kScaleLog2);
    args.append<float>(kLog2E);

    encoder.add_kernel_node_raw(
        kernel,
        dim3(static_cast<uint32_t>(num_q_blocks * gqa), Hkv, B),
        dim3(128, 1, 1), {}, 0, args.args());
  }

  DEFINE_NAME(Inkling2SdpaPrefillCuda)
  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const Inkling2SdpaPrefillCuda*>(&other);
    return rhs && rhs->start_pos_ == start_pos_ &&
        rhs->cache_len_ == cache_len_ && rhs->kv_start_pos_ == kv_start_pos_ &&
        rhs->kv_phys_base_ == kv_phys_base_ &&
        rhs->write_offset_ == write_offset_ && rhs->ring_cap_ == ring_cap_ &&
        rhs->is_local_ == is_local_ &&
        rhs->sliding_window_ == sliding_window_ && rhs->is_int8_ == is_int8_ &&
        rhs->sink_is_f32_ == sink_is_f32_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const auto& q = inputs[0];
    const int B = q.shape(0);
    const int Hq = q.shape(1);
    const int T = q.shape(2);
    if (B == 1) {
      return {Shape{T, Hq * kHeadDim}};
    }
    return {Shape{B, T, Hq * kHeadDim}};
  }

 private:
  int start_pos_;
  int cache_len_;
  int kv_start_pos_;
  int kv_phys_base_;
  int write_offset_;
  int ring_cap_;
  bool is_local_;
  int sliding_window_;
  bool is_int8_;
  bool sink_is_f32_;
};

}  // namespace
}  // namespace mlx::core::inkling2_ops

extern "C" mlx_array* mlx_inkling2_sdpa_few_cuda(
    mlx_array* q,
    mlx_array* r,
    mlx_array* k_cache,
    mlx_array* v_cache,
    mlx_array* k_scales,
    mlx_array* v_scales,
    mlx_array* rel_bias_w,
    mlx_array* gate,
    mlx_array* sink,
    mlx_array* tree_depths,
    mlx_array* tree_mask,
    int start_pos,
    int cache_len,
    int kv_start_pos,
    int kv_phys_base,
    int write_offset,
    int ring_cap,
    bool is_local,
    int sliding_window) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    using namespace mlx_node::inkling2_cuda;

    if (!q || !r || !k_cache || !v_cache || !rel_bias_w || !gate || !sink ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const bool is_int8 = (k_scales != nullptr || v_scales != nullptr);
    if (is_int8 && (!k_scales || !v_scales)) {
      return nullptr;
    }
    const bool is_tree = (tree_depths != nullptr || tree_mask != nullptr);
    if (is_tree && (!tree_depths || !tree_mask)) {
      return nullptr;
    }
    if (start_pos < 0 || cache_len < 0 || kv_start_pos < 0 ||
        kv_phys_base < 0 || write_offset < 0 || ring_cap < 0 ||
        (is_local && sliding_window < 1)) {
      return nullptr;
    }

    const auto& q_arr = *reinterpret_cast<array*>(q);
    const auto& r_arr = *reinterpret_cast<array*>(r);
    const auto& k_arr = *reinterpret_cast<array*>(k_cache);
    const auto& v_arr = *reinterpret_cast<array*>(v_cache);
    const auto& rel_arr = *reinterpret_cast<array*>(rel_bias_w);
    const auto& gate_arr = *reinterpret_cast<array*>(gate);
    const auto& sink_arr = *reinterpret_cast<array*>(sink);

    const Dtype kv_dtype = is_int8 ? mlx::core::int8 : bfloat16;
    if (q_arr.dtype() != bfloat16 || k_arr.dtype() != kv_dtype ||
        v_arr.dtype() != kv_dtype || rel_arr.dtype() != float32 ||
        q_arr.ndim() != 4 || k_arr.ndim() != 4 || v_arr.ndim() != 4 ||
        rel_arr.ndim() != 2 || rel_arr.shape(1) != kRelDim ||
        rel_arr.shape(0) < 0) {
      return nullptr;
    }

    const int B = q_arr.shape(0);
    const int Hq = q_arr.shape(1);
    const int T = q_arr.shape(2);
    const int D = q_arr.shape(3);
    const int Hkv = k_arr.shape(1);
    const int cap = k_arr.shape(2);
    if (B <= 0 || B > 65535 || Hkv <= 0 || Hkv > 65535 ||
        Hq < Hkv || Hq % Hkv != 0 || D != kHeadDim || T < 1 || T > 16 ||
        k_arr.shape(0) != B || k_arr.shape(3) != kHeadDim ||
        v_arr.shape() != k_arr.shape() ||
        cache_len > std::numeric_limits<int>::max() - T) {
      return nullptr;
    }
    const int gqa = Hq / Hkv;
    const int M = gqa * T;
    if (M < 1 || M > 64) {
      return nullptr;
    }
    if (ring_cap > 0) {
      if (ring_cap > cap) return nullptr;
    } else {
      if (cache_len > cap || write_offset > cap - T) return nullptr;
    }
    if (!check_r_shape(r_arr, B, T, Hq) ||
        !check_gate_shape(gate_arr, B, T, Hq) ||
        !check_sink_shape(sink_arr, B, T, Hq)) {
      return nullptr;
    }

    std::vector<array> inputs;
    inputs.reserve(11);
    inputs.push_back(q_arr);
    inputs.push_back(r_arr);
    inputs.push_back(k_arr);
    inputs.push_back(v_arr);
    inputs.push_back(rel_arr);
    inputs.push_back(gate_arr);
    inputs.push_back(sink_arr);

    if (is_int8) {
      const auto& ks_arr = *reinterpret_cast<array*>(k_scales);
      const auto& vs_arr = *reinterpret_cast<array*>(v_scales);
      if (!check_scales_shape(ks_arr, B, Hkv, cap) ||
          !check_scales_shape(vs_arr, B, Hkv, cap)) {
        return nullptr;
      }
      inputs.push_back(ks_arr);
      inputs.push_back(vs_arr);
    }
    if (is_tree) {
      const auto& td_arr = *reinterpret_cast<array*>(tree_depths);
      const auto& tm_arr = *reinterpret_cast<array*>(tree_mask);
      if (td_arr.dtype() != int32 || td_arr.ndim() != 1 || td_arr.shape(0) != T ||
          tm_arr.dtype() != uint32 || tm_arr.ndim() != 1 || tm_arr.shape(0) != T) {
        return nullptr;
      }
      inputs.push_back(td_arr);
      inputs.push_back(tm_arr);
    }

    const int kv_base_norm = (ring_cap > 0) ? (kv_phys_base % ring_cap) : 0;
    const int write_off_norm =
        (ring_cap > 0) ? (write_offset % ring_cap) : write_offset;
    const bool sink_is_f32 = (sink_arr.dtype() == float32);
    const Shape out_shape = (B == 1)
        ? Shape{T, Hq * kHeadDim}
        : Shape{B, T, Hq * kHeadDim};

    auto result = array(
        out_shape,
        bfloat16,
        std::make_shared<Inkling2SdpaFewCuda>(
            default_stream(Device::gpu),
            start_pos,
            cache_len,
            kv_start_pos,
            kv_base_norm,
            write_off_norm,
            ring_cap,
            is_local,
            sliding_window,
            is_int8,
            is_tree,
            sink_is_f32),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_sdpa_few_cuda: %s\n", e.what());
    return nullptr;
  }
}

extern "C" mlx_array* mlx_inkling2_sdpa_prefill_cuda(
    mlx_array* q,
    mlx_array* r,
    mlx_array* k_cache,
    mlx_array* v_cache,
    mlx_array* k_scales,
    mlx_array* v_scales,
    mlx_array* rel_bias_w,
    mlx_array* gate,
    mlx_array* sink,
    int start_pos,
    int cache_len,
    int kv_start_pos,
    int kv_phys_base,
    int write_offset,
    int ring_cap,
    bool is_local,
    int sliding_window) {
  try {
    using namespace mlx::core;
    using namespace mlx::core::inkling2_ops;
    using namespace mlx_node::inkling2_cuda;

    if (!q || !r || !k_cache || !v_cache || !rel_bias_w || !gate || !sink ||
        default_device() != Device::gpu ||
        cu::device(Device::gpu).compute_capability_major() < 8) {
      return nullptr;
    }
    const bool is_int8 = (k_scales != nullptr || v_scales != nullptr);
    if (is_int8 && (!k_scales || !v_scales)) {
      return nullptr;
    }
    if (start_pos < 0 || cache_len < 0 || kv_start_pos < 0 ||
        kv_phys_base < 0 || write_offset < 0 || ring_cap < 0 ||
        (is_local && sliding_window < 1)) {
      return nullptr;
    }

    const auto& q_arr = *reinterpret_cast<array*>(q);
    const auto& r_arr = *reinterpret_cast<array*>(r);
    const auto& k_arr = *reinterpret_cast<array*>(k_cache);
    const auto& v_arr = *reinterpret_cast<array*>(v_cache);
    const auto& rel_arr = *reinterpret_cast<array*>(rel_bias_w);
    const auto& gate_arr = *reinterpret_cast<array*>(gate);
    const auto& sink_arr = *reinterpret_cast<array*>(sink);

    const Dtype kv_dtype = is_int8 ? mlx::core::int8 : bfloat16;
    if (q_arr.dtype() != bfloat16 || k_arr.dtype() != kv_dtype ||
        v_arr.dtype() != kv_dtype || rel_arr.dtype() != float32 ||
        q_arr.ndim() != 4 || k_arr.ndim() != 4 || v_arr.ndim() != 4 ||
        rel_arr.ndim() != 2 || rel_arr.shape(1) != kRelDim ||
        rel_arr.shape(0) < 0) {
      return nullptr;
    }

    const int B = q_arr.shape(0);
    const int Hq = q_arr.shape(1);
    const int T = q_arr.shape(2);
    const int D = q_arr.shape(3);
    const int Hkv = k_arr.shape(1);
    const int cap = k_arr.shape(2);
    if (B <= 0 || B > 65535 || Hkv <= 0 || Hkv > 65535 ||
        Hq < Hkv || Hq % Hkv != 0 || D != kHeadDim || T < 1 ||
        k_arr.shape(0) != B || k_arr.shape(3) != kHeadDim ||
        v_arr.shape() != k_arr.shape() ||
        cache_len > std::numeric_limits<int>::max() - T) {
      return nullptr;
    }
    const int gqa = Hq / Hkv;
    const int64_t num_q_blocks = (static_cast<int64_t>(T) + 63) / 64;
    if (num_q_blocks * gqa > std::numeric_limits<int32_t>::max()) {
      return nullptr;
    }
    if (ring_cap > 0) {
      if (ring_cap > cap) return nullptr;
    } else {
      if (cache_len > cap || write_offset > cap - T) return nullptr;
    }
    if (!check_r_shape(r_arr, B, T, Hq) ||
        !check_gate_shape(gate_arr, B, T, Hq) ||
        !check_sink_shape(sink_arr, B, T, Hq)) {
      return nullptr;
    }

    std::vector<array> inputs;
    inputs.reserve(9);
    inputs.push_back(q_arr);
    inputs.push_back(r_arr);
    inputs.push_back(k_arr);
    inputs.push_back(v_arr);
    inputs.push_back(rel_arr);
    inputs.push_back(gate_arr);
    inputs.push_back(sink_arr);

    if (is_int8) {
      const auto& ks_arr = *reinterpret_cast<array*>(k_scales);
      const auto& vs_arr = *reinterpret_cast<array*>(v_scales);
      if (!check_scales_shape(ks_arr, B, Hkv, cap) ||
          !check_scales_shape(vs_arr, B, Hkv, cap)) {
        return nullptr;
      }
      inputs.push_back(ks_arr);
      inputs.push_back(vs_arr);
    }

    const int kv_base_norm = (ring_cap > 0) ? (kv_phys_base % ring_cap) : 0;
    const int write_off_norm =
        (ring_cap > 0) ? (write_offset % ring_cap) : write_offset;
    const bool sink_is_f32 = (sink_arr.dtype() == float32);
    const Shape out_shape = (B == 1)
        ? Shape{T, Hq * kHeadDim}
        : Shape{B, T, Hq * kHeadDim};

    auto result = array(
        out_shape,
        bfloat16,
        std::make_shared<Inkling2SdpaPrefillCuda>(
            default_stream(Device::gpu),
            start_pos,
            cache_len,
            kv_start_pos,
            kv_base_norm,
            write_off_norm,
            ring_cap,
            is_local,
            sliding_window,
            is_int8,
            sink_is_f32),
        std::move(inputs));
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_inkling2_sdpa_prefill_cuda: %s\n", e.what());
    return nullptr;
  }
}

#else

extern "C" mlx_array* mlx_inkling2_sdpa_few_cuda(
    mlx_array* q,
    mlx_array* r,
    mlx_array* k_cache,
    mlx_array* v_cache,
    mlx_array* k_scales,
    mlx_array* v_scales,
    mlx_array* rel_bias_w,
    mlx_array* gate,
    mlx_array* sink,
    mlx_array* tree_depths,
    mlx_array* tree_mask,
    int start_pos,
    int cache_len,
    int kv_start_pos,
    int kv_phys_base,
    int write_offset,
    int ring_cap,
    bool is_local,
    int sliding_window) {
  (void)q;
  (void)r;
  (void)k_cache;
  (void)v_cache;
  (void)k_scales;
  (void)v_scales;
  (void)rel_bias_w;
  (void)gate;
  (void)sink;
  (void)tree_depths;
  (void)tree_mask;
  (void)start_pos;
  (void)cache_len;
  (void)kv_start_pos;
  (void)kv_phys_base;
  (void)write_offset;
  (void)ring_cap;
  (void)is_local;
  (void)sliding_window;
  return nullptr;
}

extern "C" mlx_array* mlx_inkling2_sdpa_prefill_cuda(
    mlx_array* q,
    mlx_array* r,
    mlx_array* k_cache,
    mlx_array* v_cache,
    mlx_array* k_scales,
    mlx_array* v_scales,
    mlx_array* rel_bias_w,
    mlx_array* gate,
    mlx_array* sink,
    int start_pos,
    int cache_len,
    int kv_start_pos,
    int kv_phys_base,
    int write_offset,
    int ring_cap,
    bool is_local,
    int sliding_window) {
  (void)q;
  (void)r;
  (void)k_cache;
  (void)v_cache;
  (void)k_scales;
  (void)v_scales;
  (void)rel_bias_w;
  (void)gate;
  (void)sink;
  (void)start_pos;
  (void)cache_len;
  (void)kv_start_pos;
  (void)kv_phys_base;
  (void)write_offset;
  (void)ring_cap;
  (void)is_local;
  (void)sliding_window;
  return nullptr;
}

#endif
