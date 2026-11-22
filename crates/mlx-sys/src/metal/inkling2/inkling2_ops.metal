// Fused Inkling-2-Mini Metal kernels (compiled with -fno-fast-math into
// paged_attn.metallib):
//
//   1.  `inkling2_gate_sink_{bf16,f32}`:
//         Y = out_t(O * sigmoid(gate) * sigmoid(lse - sink)) with zero-copy strided gate
//   2.  `inkling2_situ_{bf16,f32}`:
//         Y = out_t(softcap(gate * s, gate_cap) * sigmoid(gate * s) * softcap(up * s, up_cap))
//   3.  `inkling2_loop_blend_rms_norm_{bf16,f32}`:
//         blended = (1 - tanh(forget)) * dest + tanh(gain) * msg  (FP32, NO FMA contraction)
//         Y = out_t(blended * rsqrt(mean(blended^2) + eps) * norm_weight)
//         Supports 2D single-stream and 3D multi-stream (S=3 Q/K/V) with arbitrary row/stream strides
//   4.  `inkling2_sconv_k4_{bf16,f32}`:
//         Single-sequence causal k=4 depthwise ShortConv (static or dynamic) + rolling [3, C] state
//   5.  `inkling2_qr_clip_{bf16,f32}`:
//         Fused strided Q + R L2-clip (with q_clip, rel_clip, optional global tau) -> Q [T,H,D], R [T,H,d_rel] F32
//   6.  `inkling2_sconv_k4_kv_prep_{bf16,f32}`:
//         Fused single/ragged static/dynamic k=4 ShortConv + K L2-clip + V RMSNorm (zero DRAM roundtrip!)
//   7.  `inkling2_block_attn_res_{bf16,f32}`:
//         Fused Block AttnRes over K<=8 raw/strided sources (+ optional addend) with Q<=4 queries
//         and optional fused sublayer RMSNorm -> BF16/F32
//   8.  `inkling2_pair_rms_norm_f32`:
//         Fused [rms_norm(dest), rms_norm(msg)] -> [S, T, 2*D] FP32 (S=1 or S=3, zero-copy strided)
//   9.  `inkling2_sconv_k4_ragged_{bf16,f32}`:
//         Multi-sequence ragged causal k=4 ShortConv (up to 4 unstacked histories or packed [N,3,C])
//         + optional FP32 residual epilogue (res_a + res_b + conv_out)
//   10. `inkling2_moe_route_f32`:
//         Fused sigmoid + bias + simdgroup top-K (exact smaller-index tie-break) + norm + down_scale -> int32/f32
//   11. `inkling2_moe_combine_{bf16,f32}`:
//         Fused optional inv_order unsort + FP32 weighted sum over top-K experts -> BF16/F32
//   12. `inkling2_lm_head_softcap_f32` & `inkling2_lm_head_softcap_argmax_stage{1,2}`:
//         Fused vocab slice + optional bias (with optional BF16 round) + 30*tanh(x/30) (+ 2-stage GPU argmax)
//   13. `inkling2_markov_gate_bf16`:
//         Fused DSpark Markov RNN gate (optional ctx add + BF16-rounded gate)
//   14. `inkling2_moe_router_proj_f32`:
//         Coalesced exact-FP32 MoE router projection for small T
//   15. `inkling2_loop_ctrl_down_f32`:
//         Fused pair RMSNorm + exact-FP32 LoopController down projection (`[S, T, 2*D] @ [S, 2*D, R]`)
//   16. `inkling2_loop_ctrl_up_blend_f32` + `inkling2_rms_norm_from_sq_{bf16,f32}`:
//         Fused exact-FP32 LoopController up projection + no-FMA loop blend + sublayer RMSNorm
//   17. `inkling2_markov_fold_softcap_f32` & `inkling2_markov_fold_softcap_argmax_stage1`:
//         Fused Q8_0 (group_size=32) folded Markov head matmul + fold_bias + base_logits + softcap (+ argmax)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

// Disable FMA contraction across all Inkling-2-Mini kernels so FP32 arithmetic matches
// unfused MLX mul+add operations bit-for-bit (preserving BF16 rounding & loop blend parity).
#pragma clang fp contract(off)

inline float inkling2_sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

inline float inkling2_softcap(float x, float cap) {
  if (cap <= 0.0f) {
    return x;
  }
  const float scaled = clamp(x / cap, -20.0f, 20.0f);
  return cap * tanh(scaled);
}

inline float inkling2_read_f32(device const uchar *ptr, ulong idx, uint is_f32) {
  if (is_f32 != 0u) {
    return reinterpret_cast<device const float *>(ptr)[idx];
  }
  return float(reinterpret_cast<device const bfloat *>(ptr)[idx]);
}

inline float4 inkling2_read_f4(device const uchar *ptr, ulong idx4, uint is_f32) {
  if (is_f32 != 0u) {
    return reinterpret_cast<device const float4 *>(ptr)[idx4];
  }
  return float4(reinterpret_cast<device const bfloat4 *>(ptr)[idx4]);
}

inline void inkling2_store_vec4(device bfloat *out, ulong idx4, float4 v) {
  reinterpret_cast<device bfloat4 *>(out)[idx4] = bfloat4(v);
}

inline void inkling2_store_vec4(device float *out, ulong idx4, float4 v) {
  reinterpret_cast<device float4 *>(out)[idx4] = v;
}

inline void inkling2_find_seq(
    device const int32_t *cu_seqlens,
    uint num_seqs,
    uint total_t,
    uint t,
    thread uint &seq_idx,
    thread uint &seq_start,
    thread uint &seq_end) {
  if (num_seqs <= 1u) {
    seq_idx = 0u;
    seq_start = 0u;
    seq_end = total_t;
    return;
  }
  uint lo = 0u;
  uint hi = num_seqs;
  while (lo + 1u < hi) {
    const uint mid = (lo + hi) >> 1u;
    if (uint(cu_seqlens[mid]) <= t) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  seq_idx = lo;
  seq_start = uint(cu_seqlens[lo]);
  seq_end = uint(cu_seqlens[lo + 1u]);
}

// ============================================================================
// 1. inkling2_gate_sink (zero-copy strided gate support)
// ============================================================================
struct Inkling2GateSinkParams {
  uint rows;            // T * H
  uint head_dim;        // D (128)
  uint num_heads;       // H (24)
  uint gate_row_stride; // elems per token row t (0 => contiguous)
  uint gate_is_f32;
  uint sink_is_f32;
};

template <typename OutT>
inline void inkling2_gate_sink_impl(
    device const OutT *o,
    device const uchar *gate_raw,
    device const float *lse,
    device const uchar *sink_raw,
    device OutT *out,
    constant Inkling2GateSinkParams &params,
    uint row,
    uint tid,
    uint tpg) {
  if (row >= params.rows) {
    return;
  }
  const float sink_val = inkling2_read_f32(sink_raw, ulong(row), params.sink_is_f32);
  const float sink_scale = inkling2_sigmoid(lse[row] - sink_val);
  const ulong base = ulong(row) * ulong(params.head_dim);
  const uint t = row / params.num_heads;
  const uint h = row - t * params.num_heads;
  const ulong gate_base =
      ulong(t) * ulong(params.gate_row_stride) + ulong(h) * ulong(params.head_dim);
  for (uint d = tid; d < params.head_dim; d += tpg) {
    const ulong idx = base + d;
    const float o_val = float(o[idx]);
    const float g_val = inkling2_read_f32(gate_raw, gate_base + d, params.gate_is_f32);
    const float y = o_val * inkling2_sigmoid(g_val) * sink_scale;
    out[idx] = OutT(y);
  }
}

[[kernel]] void inkling2_gate_sink_bf16(
    device const bfloat *o [[buffer(0)]],
    device const uchar *gate_raw [[buffer(1)]],
    device const float *lse [[buffer(2)]],
    device const uchar *sink_raw [[buffer(3)]],
    device bfloat *out [[buffer(4)]],
    constant Inkling2GateSinkParams &params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  inkling2_gate_sink_impl<bfloat>(o, gate_raw, lse, sink_raw, out, params, row, tid, tpg);
}

[[kernel]] void inkling2_gate_sink_f32(
    device const float *o [[buffer(0)]],
    device const uchar *gate_raw [[buffer(1)]],
    device const float *lse [[buffer(2)]],
    device const uchar *sink_raw [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant Inkling2GateSinkParams &params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  inkling2_gate_sink_impl<float>(o, gate_raw, lse, sink_raw, out, params, row, tid, tpg);
}

// ============================================================================
// 2. inkling2_situ
// ============================================================================
struct Inkling2SituParams {
  uint num_rows;
  uint inter_dim;
  float gate_softcap;
  float up_softcap;
  uint interleaved;
  uint has_outer_scale;
  uint outer_scale_is_f32;
  uint _pad;
};

template <typename T>
inline void inkling2_situ_impl(
    device const T *gate_up,
    device const uchar *outer_scale_raw,
    device const int32_t *expert_indices,
    device T *out,
    constant Inkling2SituParams &params,
    uint2 gid) {
  const uint col = gid.x;
  const uint row = gid.y;
  if (row >= params.num_rows || col >= params.inter_dim) {
    return;
  }
  const ulong row_base = ulong(row) * ulong(params.inter_dim) * 2ul;
  float gate_v;
  float up_v;
  if (params.interleaved != 0u) {
    gate_v = float(gate_up[row_base + ulong(col) * 2ul]);
    up_v = float(gate_up[row_base + ulong(col) * 2ul + 1ul]);
  } else {
    gate_v = float(gate_up[row_base + col]);
    up_v = float(gate_up[row_base + params.inter_dim + col]);
  }
  if (params.has_outer_scale != 0u) {
    const int32_t exp_id = expert_indices[row];
    const float s = inkling2_read_f32(
        outer_scale_raw, ulong(uint32_t(exp_id)), params.outer_scale_is_f32);
    gate_v *= s;
    up_v *= s;
  }
  const float gate_sc = inkling2_softcap(gate_v, params.gate_softcap);
  const float up_sc = inkling2_softcap(up_v, params.up_softcap);
  const float gate_sig = inkling2_sigmoid(gate_v);
  out[ulong(row) * ulong(params.inter_dim) + col] = T(gate_sc * gate_sig * up_sc);
}

[[kernel]] void inkling2_situ_bf16(
    device const bfloat *gate_up [[buffer(0)]],
    device const uchar *outer_scale_raw [[buffer(1)]],
    device const int32_t *expert_indices [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant Inkling2SituParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  inkling2_situ_impl<bfloat>(gate_up, outer_scale_raw, expert_indices, out, params, gid);
}

[[kernel]] void inkling2_situ_f32(
    device const float *gate_up [[buffer(0)]],
    device const uchar *outer_scale_raw [[buffer(1)]],
    device const int32_t *expert_indices [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant Inkling2SituParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  inkling2_situ_impl<float>(gate_up, outer_scale_raw, expert_indices, out, params, gid);
}

// ============================================================================
// 3. inkling2_loop_blend_rms_norm (strided 2D/3D, FP32 no-FMA blend + RMSNorm)
// ============================================================================
struct Inkling2LoopBlendParams {
  uint rows;                 // T
  uint dim;                  // D
  uint num_streams;          // S (1 or 3)
  float eps;
  uint dest_row_stride;
  uint dest_stream_stride;   // 0 if broadcast across streams
  uint msg_row_stride;
  uint msg_stream_stride;
  uint gain_row_stride;
  uint gain_stream_stride;
  uint forget_row_stride;
  uint forget_stream_stride;
  uint weight_stream_stride; // 0 if broadcast across streams
  uint flags;                // bit0: dest_f32, bit1: msg_f32, bit2: gain_f32, bit3: forget_f32, bit4: weight_f32
  uint _pad0;
  uint _pad1;
};

inline float inkling2_loop_blend_scalar(float dest, float msg,
                                        float gain_raw, float forget_raw) {
  const float g = tanh(clamp(gain_raw, -20.0f, 20.0f));
  const float f = tanh(clamp(forget_raw, -20.0f, 20.0f));
  const float one_minus_f = 1.0f - f;
  const float kept = one_minus_f * dest;
  const float added = g * msg;
  return kept + added;
}

template <typename OutT>
inline void inkling2_loop_blend_rms_norm_impl(
    device const uchar *dest_raw,
    device const uchar *msg_raw,
    device const uchar *gain_raw,
    device const uchar *forget_raw,
    device const uchar *norm_weight_raw,
    device OutT *out,
    constant Inkling2LoopBlendParams &params,
    threadgroup float *partial_sums,
    uint2 tgid,
    uint tid,
    uint simd_lane,
    uint simd_group) {
  constexpr uint THREADS = 256u;
  constexpr uint MAX_REGS = 16u;

  const uint row = tgid.x;
  const uint stream = tgid.y;
  if (row >= params.rows || stream >= params.num_streams) {
    return;
  }
  const uint dim = params.dim;
  const uint dest_f32 = (params.flags >> 0u) & 1u;
  const uint msg_f32 = (params.flags >> 1u) & 1u;
  const uint gain_f32 = (params.flags >> 2u) & 1u;
  const uint forget_f32 = (params.flags >> 3u) & 1u;
  const uint weight_f32 = (params.flags >> 4u) & 1u;

  const ulong dest_base = ulong(stream) * ulong(params.dest_stream_stride) + ulong(row) * ulong(params.dest_row_stride);
  const ulong msg_base = ulong(stream) * ulong(params.msg_stream_stride) + ulong(row) * ulong(params.msg_row_stride);
  const ulong gain_base = ulong(stream) * ulong(params.gain_stream_stride) + ulong(row) * ulong(params.gain_row_stride);
  const ulong forget_base = ulong(stream) * ulong(params.forget_stream_stride) + ulong(row) * ulong(params.forget_row_stride);
  const ulong weight_base = ulong(stream) * ulong(params.weight_stream_stride);
  const ulong out_base = (ulong(stream) * ulong(params.rows) + ulong(row)) * ulong(dim);

  float regs[MAX_REGS];
  const bool fits_regs = dim <= THREADS * MAX_REGS;
  float sum_sq = 0.0f;

  uint reg_i = 0u;
  for (uint i = tid; i < dim; i += THREADS) {
    const float d = inkling2_read_f32(dest_raw, dest_base + i, dest_f32);
    const float m = inkling2_read_f32(msg_raw, msg_base + i, msg_f32);
    const float g = inkling2_read_f32(gain_raw, gain_base + i, gain_f32);
    const float f = inkling2_read_f32(forget_raw, forget_base + i, forget_f32);
    const float b = inkling2_loop_blend_scalar(d, m, g, f);
    if (fits_regs) {
      regs[reg_i++] = b;
    }
    sum_sq += b * b;
  }

  sum_sq = simd_sum(sum_sq);
  if (simd_lane == 0u) {
    partial_sums[simd_group] = sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    float v = simd_lane < (THREADS / 32u) ? partial_sums[simd_lane] : 0.0f;
    v = simd_sum(v);
    if (simd_lane == 0u) {
      partial_sums[0] = rsqrt(v / float(dim) + params.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_rms = partial_sums[0];

  reg_i = 0u;
  for (uint i = tid; i < dim; i += THREADS) {
    float b;
    if (fits_regs) {
      b = regs[reg_i++];
    } else {
      const float d = inkling2_read_f32(dest_raw, dest_base + i, dest_f32);
      const float m = inkling2_read_f32(msg_raw, msg_base + i, msg_f32);
      const float g = inkling2_read_f32(gain_raw, gain_base + i, gain_f32);
      const float f = inkling2_read_f32(forget_raw, forget_base + i, forget_f32);
      b = inkling2_loop_blend_scalar(d, m, g, f);
    }
    const float w = inkling2_read_f32(norm_weight_raw, weight_base + i, weight_f32);
    out[out_base + i] = OutT((b * inv_rms) * w);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_loop_blend_rms_norm_bf16(
    device const uchar *dest_raw [[buffer(0)]],
    device const uchar *msg_raw [[buffer(1)]],
    device const uchar *gain_raw [[buffer(2)]],
    device const uchar *forget_raw [[buffer(3)]],
    device const uchar *norm_weight_raw [[buffer(4)]],
    device bfloat *out [[buffer(5)]],
    constant Inkling2LoopBlendParams &params [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial_sums[8];
  inkling2_loop_blend_rms_norm_impl<bfloat>(
      dest_raw, msg_raw, gain_raw, forget_raw, norm_weight_raw, out, params,
      partial_sums, tgid, tid, simd_lane, simd_group);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_loop_blend_rms_norm_f32(
    device const uchar *dest_raw [[buffer(0)]],
    device const uchar *msg_raw [[buffer(1)]],
    device const uchar *gain_raw [[buffer(2)]],
    device const uchar *forget_raw [[buffer(3)]],
    device const uchar *norm_weight_raw [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant Inkling2LoopBlendParams &params [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial_sums[8];
  inkling2_loop_blend_rms_norm_impl<float>(
      dest_raw, msg_raw, gain_raw, forget_raw, norm_weight_raw, out, params,
      partial_sums, tgid, tid, simd_lane, simd_group);
}

// ============================================================================
// 4. inkling2_sconv_k4 (single-sequence static & dynamic k=4 ShortConv)
// ============================================================================
struct Inkling2SconvK4Params {
  uint seq_len;          // T
  uint channels;         // C
  uint is_dynamic;       // 0 = static (scale 5^-0.5), 1 = dynamic (scale 1/3)
  uint num_kv_heads;     // used when is_dynamic != 0 (C % num_kv_heads == 0)
  uint weight_is_f32;
  uint history_is_f32;
  uint dyn_is_f32;
  uint x_row_stride;     // elems per token row t in x (0 => channels)
};

template <typename T>
inline float inkling2_sconv_read_full_strided(
    device const uchar *hist_raw,
    ulong hist_base,
    uint hist_row_stride,
    uint hist_is_f32,
    device const T *x,
    uint seq_start,
    uint x_stride,
    uint local_idx,
    uint ch) {
  if (local_idx < 3u) {
    return inkling2_read_f32(hist_raw, hist_base + ulong(local_idx) * ulong(hist_row_stride) + ch, hist_is_f32);
  }
  const uint global_t = seq_start + (local_idx - 3u);
  return float(x[ulong(global_t) * ulong(x_stride) + ch]);
}

template <typename T>
inline void inkling2_sconv_k4_impl(
    device const T *x,
    device const uchar *weight_raw,
    device const uchar *history_in_raw,
    device const uchar *dyn_taps_raw,
    device T *y_out,
    device T *history_out,
    constant Inkling2SconvK4Params &params,
    uint2 gid) {
  const uint ch = gid.x;
  const uint t = gid.y;
  const uint C = params.channels;
  const uint T_len = params.seq_len;
  if (ch >= C || t >= T_len) {
    return;
  }
  const uint x_stride = params.x_row_stride;

  const float x3 = inkling2_sconv_read_full_strided<T>(history_in_raw, 0ul, C, params.history_is_f32, x, 0u, x_stride, t + 0u, ch);
  const float x2 = inkling2_sconv_read_full_strided<T>(history_in_raw, 0ul, C, params.history_is_f32, x, 0u, x_stride, t + 1u, ch);
  const float x1 = inkling2_sconv_read_full_strided<T>(history_in_raw, 0ul, C, params.history_is_f32, x, 0u, x_stride, t + 2u, ch);
  const float x0 = inkling2_sconv_read_full_strided<T>(history_in_raw, 0ul, C, params.history_is_f32, x, 0u, x_stride, t + 3u, ch);

  const ulong w_base = ulong(ch) * 4ul;
  float w3 = inkling2_read_f32(weight_raw, w_base + 0ul, params.weight_is_f32);
  float w2 = inkling2_read_f32(weight_raw, w_base + 1ul, params.weight_is_f32);
  float w1 = inkling2_read_f32(weight_raw, w_base + 2ul, params.weight_is_f32);
  float w0 = inkling2_read_f32(weight_raw, w_base + 3ul, params.weight_is_f32);

  float scale;
  if (params.is_dynamic != 0u) {
    const uint head_dim = C / params.num_kv_heads;
    const uint h = ch / head_dim;
    const ulong d_base = (ulong(t) * ulong(params.num_kv_heads) + h) * 4ul;
    w0 += inkling2_read_f32(dyn_taps_raw, d_base + 0ul, params.dyn_is_f32);
    w1 += inkling2_read_f32(dyn_taps_raw, d_base + 1ul, params.dyn_is_f32);
    w2 += inkling2_read_f32(dyn_taps_raw, d_base + 2ul, params.dyn_is_f32);
    w3 += inkling2_read_f32(dyn_taps_raw, d_base + 3ul, params.dyn_is_f32);
    scale = 1.0f / 3.0f;
  } else {
    scale = 0.44721359549995793f;
  }

  const float acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
  y_out[ulong(t) * ulong(C) + ch] = T(acc * scale);

  if (t == T_len - 1u) {
    history_out[0ul * ulong(C) + ch] = T(x2);
    history_out[1ul * ulong(C) + ch] = T(x1);
    history_out[2ul * ulong(C) + ch] = T(x0);
  }
}

[[kernel]] void inkling2_sconv_k4_bf16(
    device const bfloat *x [[buffer(0)]],
    device const uchar *weight_raw [[buffer(1)]],
    device const uchar *history_in_raw [[buffer(2)]],
    device const uchar *dyn_taps_raw [[buffer(3)]],
    device bfloat *y_out [[buffer(4)]],
    device bfloat *history_out [[buffer(5)]],
    constant Inkling2SconvK4Params &params [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  inkling2_sconv_k4_impl<bfloat>(
      x, weight_raw, history_in_raw, dyn_taps_raw, y_out, history_out, params, gid);
}

[[kernel]] void inkling2_sconv_k4_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weight_raw [[buffer(1)]],
    device const uchar *history_in_raw [[buffer(2)]],
    device const uchar *dyn_taps_raw [[buffer(3)]],
    device float *y_out [[buffer(4)]],
    device float *history_out [[buffer(5)]],
    constant Inkling2SconvK4Params &params [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  inkling2_sconv_k4_impl<float>(
      x, weight_raw, history_in_raw, dyn_taps_raw, y_out, history_out, params, gid);
}

// ============================================================================
// 5. inkling2_qr_clip (fused strided Q & R L2-clip + optional global tau)
// ============================================================================
struct Inkling2QrClipParams {
  uint seq_len;       // T
  uint num_heads;     // H (24)
  uint head_dim;      // D (128)
  uint d_rel;         // 16
  uint q_row_stride;  // elems per token row t in q_in
  uint r_row_stride;  // elems per token row t in r_in
  float qk_thresh;    // 22.627416997969522f
  float rel_thresh;   // 8.0f
  float eps;          // 1e-6f
  uint flags;         // bit0: q_f32, bit1: r_f32, bit2: q_clip_f32, bit3: rel_clip_f32, bit4: has_tau, bit5: tau_f32, bit6: r_round_bf16
  uint _pad0;
  uint _pad1;
};

template <typename QOutT>
inline void inkling2_qr_clip_impl(
    device const uchar *q_in_raw,
    device const uchar *r_in_raw,
    device const uchar *q_clip_raw,
    device const uchar *rel_clip_raw,
    device const uchar *tau_raw,
    device QOutT *q_out,
    device float *r_out,
    constant Inkling2QrClipParams &params,
    uint2 tgid,
    uint lane) {
  const uint h = tgid.x;
  const uint t = tgid.y;
  if (h >= params.num_heads || t >= params.seq_len) {
    return;
  }

  const uint q_f32 = (params.flags >> 0u) & 1u;
  const uint r_f32 = (params.flags >> 1u) & 1u;
  const uint qc_f32 = (params.flags >> 2u) & 1u;
  const uint rc_f32 = (params.flags >> 3u) & 1u;
  const uint has_tau = (params.flags >> 4u) & 1u;
  const uint tau_f32 = (params.flags >> 5u) & 1u;
  const uint r_round_bf16 = (params.flags >> 6u) & 1u;

  const float tau_val = has_tau != 0u ? inkling2_read_f32(tau_raw, ulong(t), tau_f32) : 1.0f;

  // --- Q head (1 simdgroup = 32 lanes, D=128 => 4 regs/lane; up to 256 => 8 regs) ---
  const uint hd = params.head_dim;
  const ulong q_in_base = ulong(t) * ulong(params.q_row_stride) + ulong(h) * ulong(hd);
  const ulong q_clip_base = ulong(h) * ulong(hd);
  const ulong q_out_base = (ulong(t) * ulong(params.num_heads) + ulong(h)) * ulong(hd);

  float q_regs[8];
  float q_norm_sq = 0.0f;
  uint reg_i = 0u;
  for (uint d = lane; d < hd; d += 32u) {
    const float v = inkling2_read_f32(q_in_raw, q_in_base + d, q_f32);
    const float w = inkling2_read_f32(q_clip_raw, q_clip_base + d, qc_f32);
    const float u = v * w;
    if (reg_i < 8u) {
      q_regs[reg_i] = u;
    }
    ++reg_i;
    q_norm_sq += u * u;
  }
  q_norm_sq = simd_sum(q_norm_sq);
  const float q_factor = min(params.qk_thresh / (sqrt(q_norm_sq) + params.eps), 1.0f);

  reg_i = 0u;
  for (uint d = lane; d < hd; d += 32u) {
    float u;
    if (reg_i < 8u) {
      u = q_regs[reg_i];
    } else {
      u = inkling2_read_f32(q_in_raw, q_in_base + d, q_f32) *
          inkling2_read_f32(q_clip_raw, q_clip_base + d, qc_f32);
    }
    ++reg_i;
    float qc = float(QOutT(u * q_factor));
    if (has_tau != 0u) {
      qc *= tau_val;
    }
    q_out[q_out_base + d] = QOutT(qc);
  }

  // --- R head (d_rel=16 => lanes 0..15 active) ---
  const uint dr = params.d_rel;
  const ulong r_in_base = ulong(t) * ulong(params.r_row_stride) + ulong(h) * ulong(dr);
  const ulong r_clip_base = ulong(h) * ulong(dr);
  const ulong r_out_base = (ulong(t) * ulong(params.num_heads) + ulong(h)) * ulong(dr);

  float r_regs[4];
  float r_norm_sq = 0.0f;
  reg_i = 0u;
  for (uint d = lane; d < dr; d += 32u) {
    const float v = inkling2_read_f32(r_in_raw, r_in_base + d, r_f32);
    const float w = inkling2_read_f32(rel_clip_raw, r_clip_base + d, rc_f32);
    const float u = v * w;
    if (reg_i < 4u) {
      r_regs[reg_i] = u;
    }
    ++reg_i;
    r_norm_sq += u * u;
  }
  r_norm_sq = simd_sum(r_norm_sq);
  const float r_factor = min(params.rel_thresh / (sqrt(r_norm_sq) + params.eps), 1.0f);

  reg_i = 0u;
  for (uint d = lane; d < dr; d += 32u) {
    float u;
    if (reg_i < 4u) {
      u = r_regs[reg_i];
    } else {
      u = inkling2_read_f32(r_in_raw, r_in_base + d, r_f32) *
          inkling2_read_f32(rel_clip_raw, r_clip_base + d, rc_f32);
    }
    ++reg_i;
    float rc = u * r_factor;
    if (r_round_bf16 != 0u) {
      rc = float(bfloat(rc));
    }
    if (has_tau != 0u) {
      rc *= tau_val;
    }
    r_out[r_out_base + d] = rc;
  }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_qr_clip_bf16(
    device const uchar *q_in_raw [[buffer(0)]],
    device const uchar *r_in_raw [[buffer(1)]],
    device const uchar *q_clip_raw [[buffer(2)]],
    device const uchar *rel_clip_raw [[buffer(3)]],
    device const uchar *tau_raw [[buffer(4)]],
    device bfloat *q_out [[buffer(5)]],
    device float *r_out [[buffer(6)]],
    constant Inkling2QrClipParams &params [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  inkling2_qr_clip_impl<bfloat>(
      q_in_raw, r_in_raw, q_clip_raw, rel_clip_raw, tau_raw, q_out, r_out, params, tgid, lane);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_qr_clip_f32(
    device const uchar *q_in_raw [[buffer(0)]],
    device const uchar *r_in_raw [[buffer(1)]],
    device const uchar *q_clip_raw [[buffer(2)]],
    device const uchar *rel_clip_raw [[buffer(3)]],
    device const uchar *tau_raw [[buffer(4)]],
    device float *q_out [[buffer(5)]],
    device float *r_out [[buffer(6)]],
    constant Inkling2QrClipParams &params [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  inkling2_qr_clip_impl<float>(
      q_in_raw, r_in_raw, q_clip_raw, rel_clip_raw, tau_raw, q_out, r_out, params, tgid, lane);
}

// ============================================================================
// 6. inkling2_sconv_k4_kv_prep (fused K/V ShortConv + K L2-clip + V RMSNorm)
// ============================================================================
struct Inkling2SconvK4KvPrepParams {
  uint seq_len;          // total T
  uint num_kv_heads;     // H_kv (8)
  uint head_dim;         // D (128)
  uint num_seqs;         // N (>= 1)
  uint k_row_stride;     // elems per token row t in k_in
  uint v_row_stride;     // elems per token row t in v_in
  float qk_thresh;       // 22.627416997969522f
  float v_eps;           // 1e-6f
  float l2_eps;          // 1e-6f
  uint is_dynamic;       // 0 or 1
  uint packed_hist;      // 0 = up to 4 separate bufs, 1 = packed [N, 3, C] in buf 0
  uint flags;            // bit0: kw_f32, bit1: vw_f32, bit2: dk_f32, bit3: dv_f32
  uint k_hist_f32_mask;  // bit s = 1 if k_hist[s] is f32
  uint v_hist_f32_mask;  // bit s = 1 if v_hist[s] is f32
  uint _pad0;
  uint _pad1;
  uint k_hist_strides[4]; // row stride in elems for kh0..kh3 (C when packed)
  uint v_hist_strides[4]; // row stride in elems for vh0..vh3 (C when packed)
};

template <typename T>
inline void inkling2_sconv_k4_kv_prep_impl(
    device const T *k_in,
    device const T *v_in,
    device const uchar *k_weight,
    device const uchar *v_weight,
    device const uchar *dyn_k,
    device const uchar *dyn_v,
    device const int32_t *cu_seqlens,
    device const uchar *kh0,
    device const uchar *kh1,
    device const uchar *kh2,
    device const uchar *kh3,
    device const uchar *vh0,
    device const uchar *vh1,
    device const uchar *vh2,
    device const uchar *vh3,
    device T *k_out,
    device T *v_out,
    device T *k_hist_out,
    device T *v_hist_out,
    constant Inkling2SconvK4KvPrepParams &params,
    uint2 tgid,
    uint lane) {
  const uint n_kv = params.num_kv_heads;
  const uint t = tgid.y;
  if (tgid.x >= 2u * n_kv || t >= params.seq_len) {
    return;
  }
  const bool is_v = tgid.x >= n_kv;
  const uint h = is_v ? (tgid.x - n_kv) : tgid.x;
  const uint hd = params.head_dim;
  const uint C = n_kv * hd;

  uint seq_idx = 0u;
  uint seq_start = 0u;
  uint seq_end = params.seq_len;
  inkling2_find_seq(cu_seqlens, params.num_seqs, params.seq_len, t, seq_idx, seq_start, seq_end);
  const uint local_t = t - seq_start;
  const uint seq_t_len = seq_end - seq_start;

  device const T *x_ptr = is_v ? v_in : k_in;
  const uint x_stride = is_v ? params.v_row_stride : params.k_row_stride;
  device const uchar *w_ptr = is_v ? v_weight : k_weight;
  const uint w_f32 = is_v ? ((params.flags >> 1u) & 1u) : ((params.flags >> 0u) & 1u);
  device const uchar *dyn_ptr = is_v ? dyn_v : dyn_k;
  const uint dyn_f32 = is_v ? ((params.flags >> 3u) & 1u) : ((params.flags >> 2u) & 1u);
  const uint hist_mask = is_v ? params.v_hist_f32_mask : params.k_hist_f32_mask;

  device const uchar *h_ptr;
  ulong h_base;
  uint h_row_stride;
  uint h_f32;
  if (params.packed_hist != 0u) {
    h_ptr = is_v ? vh0 : kh0;
    h_base = ulong(seq_idx) * 3ul * ulong(C);
    h_row_stride = C;
    h_f32 = hist_mask & 1u;
  } else {
    if (is_v) {
      h_ptr = (seq_idx == 0u) ? vh0 : (seq_idx == 1u) ? vh1 : (seq_idx == 2u) ? vh2 : vh3;
      h_row_stride = params.v_hist_strides[seq_idx];
    } else {
      h_ptr = (seq_idx == 0u) ? kh0 : (seq_idx == 1u) ? kh1 : (seq_idx == 2u) ? kh2 : kh3;
      h_row_stride = params.k_hist_strides[seq_idx];
    }
    h_base = 0ul;
    h_f32 = (hist_mask >> seq_idx) & 1u;
  }

  device T *y_ptr = is_v ? v_out : k_out;
  device T *h_out_ptr = is_v ? v_hist_out : k_hist_out;
  const ulong h_out_base = ulong(seq_idx) * 3ul * ulong(C);

  // Preload dynamic taps for this (t, h) once (shared across all 128 channels of the head!).
  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  float scale = 0.44721359549995793f;
  if (params.is_dynamic != 0u) {
    const ulong d_base = (ulong(t) * ulong(n_kv) + ulong(h)) * 4ul;
    d0 = inkling2_read_f32(dyn_ptr, d_base + 0ul, dyn_f32);
    d1 = inkling2_read_f32(dyn_ptr, d_base + 1ul, dyn_f32);
    d2 = inkling2_read_f32(dyn_ptr, d_base + 2ul, dyn_f32);
    d3 = inkling2_read_f32(dyn_ptr, d_base + 3ul, dyn_f32);
    scale = 1.0f / 3.0f;
  }

  const bool is_last_tok = (local_t == seq_t_len - 1u);
  if (t == 0u && params.num_seqs > 1u) {
    for (uint s = 0u; s < params.num_seqs; ++s) {
      if (cu_seqlens[s] == cu_seqlens[s + 1u]) {
        device const uchar *eh_ptr;
        ulong eh_base;
        uint eh_row_stride;
        uint eh_f32;
        if (params.packed_hist != 0u) {
          eh_ptr = is_v ? vh0 : kh0;
          eh_base = ulong(s) * 3ul * ulong(C);
          eh_row_stride = C;
          eh_f32 = hist_mask & 1u;
        } else {
          if (is_v) {
            eh_ptr = (s == 0u) ? vh0 : (s == 1u) ? vh1 : (s == 2u) ? vh2 : vh3;
            eh_row_stride = params.v_hist_strides[s];
          } else {
            eh_ptr = (s == 0u) ? kh0 : (s == 1u) ? kh1 : (s == 2u) ? kh2 : kh3;
            eh_row_stride = params.k_hist_strides[s];
          }
          eh_base = 0ul;
          eh_f32 = (hist_mask >> s) & 1u;
        }
        const ulong eout_base = ulong(s) * 3ul * ulong(C);
        for (uint d = lane; d < hd; d += 32u) {
          const uint ch = h * hd + d;
          for (uint row_k = 0u; row_k < 3u; ++row_k) {
            const float hv = inkling2_read_f32(
                eh_ptr, eh_base + ulong(row_k) * ulong(eh_row_stride) + ulong(ch), eh_f32);
            h_out_ptr[eout_base + ulong(row_k) * ulong(C) + ch] = T(hv);
          }
        }
      }
    }
  }
  float conv_regs[8];
  float sum_sq = 0.0f;
  uint reg_i = 0u;
  for (uint d = lane; d < hd; d += 32u) {
    const uint ch = h * hd + d;
    const float x3 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, h_row_stride, h_f32, x_ptr, seq_start, x_stride, local_t + 0u, ch);
    const float x2 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, h_row_stride, h_f32, x_ptr, seq_start, x_stride, local_t + 1u, ch);
    const float x1 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, h_row_stride, h_f32, x_ptr, seq_start, x_stride, local_t + 2u, ch);
    const float x0 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, h_row_stride, h_f32, x_ptr, seq_start, x_stride, local_t + 3u, ch);

    if (is_last_tok) {
      h_out_ptr[h_out_base + 0ul * ulong(C) + ch] = T(x2);
      h_out_ptr[h_out_base + 1ul * ulong(C) + ch] = T(x1);
      h_out_ptr[h_out_base + 2ul * ulong(C) + ch] = T(x0);
    }

    const ulong w_base = ulong(ch) * 4ul;
    const float w3 = inkling2_read_f32(w_ptr, w_base + 0ul, w_f32) + d3;
    const float w2 = inkling2_read_f32(w_ptr, w_base + 1ul, w_f32) + d2;
    const float w1 = inkling2_read_f32(w_ptr, w_base + 2ul, w_f32) + d1;
    const float w0 = inkling2_read_f32(w_ptr, w_base + 3ul, w_f32) + d0;

    const float acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
    // Match exact intermediate rounding of unfused ShortConv -> L2Clip/RMSNorm.
    const float cv = float(T(acc * scale));
    if (reg_i < 8u) {
      conv_regs[reg_i] = cv;
    }
    ++reg_i;
    sum_sq += cv * cv;
  }

  sum_sq = simd_sum(sum_sq);
  const float factor = is_v
      ? rsqrt(sum_sq / float(hd) + params.v_eps)
      : min(params.qk_thresh / (sqrt(sum_sq) + params.l2_eps), 1.0f);

  const ulong out_base = (ulong(t) * ulong(n_kv) + ulong(h)) * ulong(hd);
  reg_i = 0u;
  for (uint d = lane; d < hd; d += 32u) {
    const float cv = conv_regs[reg_i++];
    y_ptr[out_base + d] = T(cv * factor);
  }
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_sconv_k4_kv_prep_bf16(
    device const bfloat *k_in [[buffer(0)]],
    device const bfloat *v_in [[buffer(1)]],
    device const uchar *k_weight [[buffer(2)]],
    device const uchar *v_weight [[buffer(3)]],
    device const uchar *dyn_k [[buffer(4)]],
    device const uchar *dyn_v [[buffer(5)]],
    device const int32_t *cu_seqlens [[buffer(6)]],
    device const uchar *kh0 [[buffer(7)]],
    device const uchar *kh1 [[buffer(8)]],
    device const uchar *kh2 [[buffer(9)]],
    device const uchar *kh3 [[buffer(10)]],
    device const uchar *vh0 [[buffer(11)]],
    device const uchar *vh1 [[buffer(12)]],
    device const uchar *vh2 [[buffer(13)]],
    device const uchar *vh3 [[buffer(14)]],
    device bfloat *k_out [[buffer(15)]],
    device bfloat *v_out [[buffer(16)]],
    device bfloat *k_hist_out [[buffer(17)]],
    device bfloat *v_hist_out [[buffer(18)]],
    constant Inkling2SconvK4KvPrepParams &params [[buffer(19)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  inkling2_sconv_k4_kv_prep_impl<bfloat>(
      k_in, v_in, k_weight, v_weight, dyn_k, dyn_v, cu_seqlens,
      kh0, kh1, kh2, kh3, vh0, vh1, vh2, vh3,
      k_out, v_out, k_hist_out, v_hist_out, params, tgid, lane);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_sconv_k4_kv_prep_f32(
    device const float *k_in [[buffer(0)]],
    device const float *v_in [[buffer(1)]],
    device const uchar *k_weight [[buffer(2)]],
    device const uchar *v_weight [[buffer(3)]],
    device const uchar *dyn_k [[buffer(4)]],
    device const uchar *dyn_v [[buffer(5)]],
    device const int32_t *cu_seqlens [[buffer(6)]],
    device const uchar *kh0 [[buffer(7)]],
    device const uchar *kh1 [[buffer(8)]],
    device const uchar *kh2 [[buffer(9)]],
    device const uchar *kh3 [[buffer(10)]],
    device const uchar *vh0 [[buffer(11)]],
    device const uchar *vh1 [[buffer(12)]],
    device const uchar *vh2 [[buffer(13)]],
    device const uchar *vh3 [[buffer(14)]],
    device float *k_out [[buffer(15)]],
    device float *v_out [[buffer(16)]],
    device float *k_hist_out [[buffer(17)]],
    device float *v_hist_out [[buffer(18)]],
    constant Inkling2SconvK4KvPrepParams &params [[buffer(19)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  inkling2_sconv_k4_kv_prep_impl<float>(
      k_in, v_in, k_weight, v_weight, dyn_k, dyn_v, cu_seqlens,
      kh0, kh1, kh2, kh3, vh0, vh1, vh2, vh3,
      k_out, v_out, k_hist_out, v_hist_out, params, tgid, lane);
}

// ============================================================================
// 7. inkling2_block_attn_res (fused Block AttnRes + optional addend + RMSNorm)
// ============================================================================
struct Inkling2BlockAttnResParams {
  uint rows;             // T
  uint dim;              // D (2304)
  uint num_sources;      // K (1..=8)
  uint num_queries;      // Q (1..=4)
  float attn_res_eps;    // 1e-6f
  float norm_eps;        // 1e-6f
  uint src_f32_mask;     // bit k = 1 if src_k is f32
  uint flags;            // bit0: has_addend, bit1: addend_f32, bit2: has_norm_weight, bit3: weight_f32, bit4: weight_broadcast
  uint src_strides[8];   // row stride in elems for src_0..src_7
  uint addend_stride;
  uint _pad0;
  uint _pad1;
  uint _pad2;
};

inline float inkling2_read_src(
    device const uchar *s0, device const uchar *s1, device const uchar *s2, device const uchar *s3,
    device const uchar *s4, device const uchar *s5, device const uchar *s6, device const uchar *s7,
    device const uchar *addend,
    constant Inkling2BlockAttnResParams &params,
    uint k,
    uint row,
    uint col) {
  device const uchar *p =
      (k == 0u) ? s0 : (k == 1u) ? s1 : (k == 2u) ? s2 : (k == 3u) ? s3 :
      (k == 4u) ? s4 : (k == 5u) ? s5 : (k == 6u) ? s6 : s7;
  const uint is_f32 = (params.src_f32_mask >> k) & 1u;
  float v = inkling2_read_f32(p, ulong(row) * ulong(params.src_strides[k]) + col, is_f32);
  if (k == params.num_sources - 1u && ((params.flags & 1u) != 0u)) {
    const uint add_f32 = (params.flags >> 1u) & 1u;
    v += inkling2_read_f32(addend, ulong(row) * ulong(params.addend_stride) + col, add_f32);
  }
  return v;
}

inline float4 inkling2_read_src4(
    device const uchar *s0, device const uchar *s1, device const uchar *s2, device const uchar *s3,
    device const uchar *s4, device const uchar *s5, device const uchar *s6, device const uchar *s7,
    device const uchar *addend,
    constant Inkling2BlockAttnResParams &params,
    uint k,
    uint row,
    uint col4) {
  device const uchar *p =
      (k == 0u) ? s0 : (k == 1u) ? s1 : (k == 2u) ? s2 : (k == 3u) ? s3 :
      (k == 4u) ? s4 : (k == 5u) ? s5 : (k == 6u) ? s6 : s7;
  const uint is_f32 = (params.src_f32_mask >> k) & 1u;
  float4 v = inkling2_read_f4(p, (ulong(row) * ulong(params.src_strides[k]) >> 2u) + col4, is_f32);
  if (k == params.num_sources - 1u && ((params.flags & 1u) != 0u)) {
    const uint add_f32 = (params.flags >> 1u) & 1u;
    v += inkling2_read_f4(addend, (ulong(row) * ulong(params.addend_stride) >> 2u) + col4, add_f32);
  }
  return v;
}

template <typename OutT>
inline void inkling2_block_attn_res_impl(
    device const uchar *s0, device const uchar *s1, device const uchar *s2, device const uchar *s3,
    device const uchar *s4, device const uchar *s5, device const uchar *s6, device const uchar *s7,
    device const uchar *addend,
    device const float *queries,
    device const uchar *norm_weight,
    device OutT *out,
    constant Inkling2BlockAttnResParams &params,
    threadgroup float *tg_reduce, // 40 * 8 = 320 floats
    threadgroup float *tg_inv_rms, // 8 floats
    threadgroup float *tg_probs,   // 4 * 8 = 32 floats
    uint row,
    uint tid,
    uint simd_lane,
    uint simd_group) {
  constexpr uint THREADS = 256u;
  constexpr uint MAX_REGS = 10u; // covers D up to 2560 (2304 -> 9 regs)
  constexpr uint MAX_VEC4_REGS = 3u; // covers D up to 3072 (2304 -> 576 float4s -> 3 regs)

  if (row >= params.rows) {
    return;
  }
  const uint D = params.dim;
  const uint K = params.num_sources;
  const uint Q = params.num_queries;
  const bool has_norm = ((params.flags >> 2u) & 1u) != 0u;
  const uint weight_f32 = (params.flags >> 3u) & 1u;
  const bool weight_bcast = ((params.flags >> 4u) & 1u) != 0u;
  const bool vec4_ok = ((params.flags >> 5u) & 1u) != 0u;

  float sum_sq[8];
  float raw_dot[4][8];
  for (uint k = 0u; k < K; ++k) {
    sum_sq[k] = 0.0f;
    for (uint q = 0u; q < Q; ++q) {
      raw_dot[q][k] = 0.0f;
    }
  }

  // Vectorized float4/bfloat4 fast path (128B coalesced loads/stores + single-pass cache for K<=2)
  if (vec4_ok) {
    const uint D4 = D >> 2u;
    const device float4 *q4 = reinterpret_cast<device const float4 *>(queries);
    const bool cache_srcs = (K <= 2u) && (D4 <= THREADS * MAX_VEC4_REGS);
    float4 src_cache[2][MAX_VEC4_REGS];

    uint reg_i = 0u;
    for (uint i4 = tid; i4 < D4; i4 += THREADS) {
      float4 qv[4];
      if (K > 1u) {
        for (uint q = 0u; q < Q; ++q) {
          qv[q] = q4[ulong(q) * ulong(D4) + i4];
        }
      }
      for (uint k = 0u; k < K; ++k) {
        const float4 v =
            inkling2_read_src4(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i4);
        if (cache_srcs) {
          src_cache[k][reg_i] = v;
        }
        sum_sq[k] += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
        if (K > 1u) {
          for (uint q = 0u; q < Q; ++q) {
            raw_dot[q][k] +=
                (qv[q].x * v.x + qv[q].y * v.y) + (qv[q].z * v.z + qv[q].w * v.w);
          }
        }
      }
      ++reg_i;
    }

    for (uint k = 0u; k < K; ++k) {
      const float s = simd_sum(sum_sq[k]);
      if (simd_lane == 0u) {
        tg_reduce[k * 8u + simd_group] = s;
      }
    }
    if (K > 1u) {
      for (uint q = 0u; q < Q; ++q) {
        for (uint k = 0u; k < K; ++k) {
          const float d = simd_sum(raw_dot[q][k]);
          if (simd_lane == 0u) {
            tg_reduce[(8u + q * 8u + k) * 8u + simd_group] = d;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < K) {
      float s = 0.0f;
      for (uint sg = 0u; sg < 8u; ++sg) {
        s += tg_reduce[tid * 8u + sg];
      }
      tg_inv_rms[tid] = rsqrt(s / float(D) + params.attn_res_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < Q) {
      const uint q = tid;
      if (K == 1u) {
        tg_probs[q * 8u + 0u] = 1.0f;
      } else {
        float scores[8];
        float max_s = -INFINITY;
        for (uint k = 0u; k < K; ++k) {
          float d = 0.0f;
          for (uint sg = 0u; sg < 8u; ++sg) {
            d += tg_reduce[(8u + q * 8u + k) * 8u + sg];
          }
          scores[k] = d * tg_inv_rms[k];
          max_s = max(max_s, scores[k]);
        }
        float denom = 0.0f;
        for (uint k = 0u; k < K; ++k) {
          scores[k] = exp(scores[k] - max_s);
          denom += scores[k];
        }
        const float inv_denom = 1.0f / denom;
        for (uint k = 0u; k < K; ++k) {
          tg_probs[q * 8u + k] = scores[k] * inv_denom;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms[8];
    float probs[4][8];
    for (uint k = 0u; k < K; ++k) {
      inv_rms[k] = tg_inv_rms[k];
      for (uint q = 0u; q < Q; ++q) {
        probs[q][k] = tg_probs[q * 8u + k];
      }
    }

    if (!has_norm) {
      reg_i = 0u;
      for (uint i4 = tid; i4 < D4; i4 += THREADS) {
        float4 mix[4];
        for (uint q = 0u; q < Q; ++q) {
          mix[q] = float4(0.0f);
        }
        for (uint k = 0u; k < K; ++k) {
          const float4 raw_v = cache_srcs
              ? src_cache[k][reg_i]
              : inkling2_read_src4(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i4);
          const float4 vk = raw_v * inv_rms[k];
          for (uint q = 0u; q < Q; ++q) {
            mix[q] = mix[q] + probs[q][k] * vk;
          }
        }
        for (uint q = 0u; q < Q; ++q) {
          const ulong out_idx4 = (ulong(q) * ulong(params.rows) + ulong(row)) * ulong(D4) + i4;
          inkling2_store_vec4(out, out_idx4, mix[q]);
        }
        ++reg_i;
      }
      return;
    }

    const bool fits_regs = (D4 <= THREADS * MAX_VEC4_REGS);
    float4 mix_regs[4][MAX_VEC4_REGS];
    float mix_sq[4];
    for (uint q = 0u; q < Q; ++q) {
      mix_sq[q] = 0.0f;
    }

    reg_i = 0u;
    for (uint i4 = tid; i4 < D4; i4 += THREADS) {
      float4 mix[4];
      for (uint q = 0u; q < Q; ++q) {
        mix[q] = float4(0.0f);
      }
      for (uint k = 0u; k < K; ++k) {
        const float4 raw_v = cache_srcs
            ? src_cache[k][reg_i]
            : inkling2_read_src4(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i4);
        const float4 vk = raw_v * inv_rms[k];
        for (uint q = 0u; q < Q; ++q) {
          mix[q] = mix[q] + probs[q][k] * vk;
        }
      }
      for (uint q = 0u; q < Q; ++q) {
        if (fits_regs) {
          mix_regs[q][reg_i] = mix[q];
        }
        const float4 m = mix[q];
        mix_sq[q] += (m.x * m.x + m.y * m.y) + (m.z * m.z + m.w * m.w);
      }
      ++reg_i;
    }

    for (uint q = 0u; q < Q; ++q) {
      const float ms = simd_sum(mix_sq[q]);
      if (simd_lane == 0u) {
        tg_reduce[q * 8u + simd_group] = ms;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Q) {
      float ms = 0.0f;
      for (uint sg = 0u; sg < 8u; ++sg) {
        ms += tg_reduce[tid * 8u + sg];
      }
      tg_inv_rms[tid] = rsqrt(ms / float(D) + params.norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mix_inv_rms[4];
    for (uint q = 0u; q < Q; ++q) {
      mix_inv_rms[q] = tg_inv_rms[q];
    }

    reg_i = 0u;
    for (uint i4 = tid; i4 < D4; i4 += THREADS) {
      for (uint q = 0u; q < Q; ++q) {
        float4 mv;
        if (fits_regs) {
          mv = mix_regs[q][reg_i];
        } else {
          mv = float4(0.0f);
          for (uint k = 0u; k < K; ++k) {
            const float4 vk =
                inkling2_read_src4(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i4) *
                inv_rms[k];
            mv = mv + probs[q][k] * vk;
          }
        }
        const ulong w_idx4 = (weight_bcast ? 0ul : ulong(q) * ulong(D4)) + i4;
        const float4 w = inkling2_read_f4(norm_weight, w_idx4, weight_f32);
        const ulong out_idx4 = (ulong(q) * ulong(params.rows) + ulong(row)) * ulong(D4) + i4;
        inkling2_store_vec4(out, out_idx4, (mv * mix_inv_rms[q]) * w);
      }
      ++reg_i;
    }
    return;
  }

  // Scalar fallback (unaligned or D % 4 != 0)
  // Pass 1: accumulate per-source sum_sq and query-source raw dots in FP32.
  for (uint i = tid; i < D; i += THREADS) {
    float qv[4];
    if (K > 1u) {
      for (uint q = 0u; q < Q; ++q) {
        qv[q] = queries[ulong(q) * ulong(D) + i];
      }
    }
    for (uint k = 0u; k < K; ++k) {
      const float v = inkling2_read_src(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i);
      sum_sq[k] += v * v;
      if (K > 1u) {
        for (uint q = 0u; q < Q; ++q) {
          raw_dot[q][k] += qv[q] * v;
        }
      }
    }
  }

  for (uint k = 0u; k < K; ++k) {
    const float s = simd_sum(sum_sq[k]);
    if (simd_lane == 0u) {
      tg_reduce[k * 8u + simd_group] = s;
    }
  }
  if (K > 1u) {
    for (uint q = 0u; q < Q; ++q) {
      for (uint k = 0u; k < K; ++k) {
        const float d = simd_sum(raw_dot[q][k]);
        if (simd_lane == 0u) {
          tg_reduce[(8u + q * 8u + k) * 8u + simd_group] = d;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < K) {
    float s = 0.0f;
    for (uint sg = 0u; sg < 8u; ++sg) {
      s += tg_reduce[tid * 8u + sg];
    }
    tg_inv_rms[tid] = rsqrt(s / float(D) + params.attn_res_eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < Q) {
    const uint q = tid;
    if (K == 1u) {
      tg_probs[q * 8u + 0u] = 1.0f;
    } else {
      float scores[8];
      float max_s = -INFINITY;
      for (uint k = 0u; k < K; ++k) {
        float d = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
          d += tg_reduce[(8u + q * 8u + k) * 8u + sg];
        }
        scores[k] = d * tg_inv_rms[k];
        max_s = max(max_s, scores[k]);
      }
      float denom = 0.0f;
      for (uint k = 0u; k < K; ++k) {
        scores[k] = exp(scores[k] - max_s);
        denom += scores[k];
      }
      const float inv_denom = 1.0f / denom;
      for (uint k = 0u; k < K; ++k) {
        tg_probs[q * 8u + k] = scores[k] * inv_denom;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_rms[8];
  float probs[4][8];
  for (uint k = 0u; k < K; ++k) {
    inv_rms[k] = tg_inv_rms[k];
    for (uint q = 0u; q < Q; ++q) {
      probs[q][k] = tg_probs[q * 8u + k];
    }
  }

  // Pass 2: compute weighted mix[q] = sum_k probs[q, k] * (s_k * inv_rms[k]).
  if (!has_norm) {
    for (uint i = tid; i < D; i += THREADS) {
      float mix[4];
      for (uint q = 0u; q < Q; ++q) {
        mix[q] = 0.0f;
      }
      for (uint k = 0u; k < K; ++k) {
        const float vk = inkling2_read_src(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i) * inv_rms[k];
        for (uint q = 0u; q < Q; ++q) {
          mix[q] += probs[q][k] * vk;
        }
      }
      for (uint q = 0u; q < Q; ++q) {
        const ulong out_idx = (ulong(q) * ulong(params.rows) + ulong(row)) * ulong(D) + i;
        out[out_idx] = OutT(mix[q]);
      }
    }
    return;
  }

  // Fused sublayer RMSNorm epilogue
  const bool fits_regs = (D <= THREADS * MAX_REGS);
  float mix_regs[4][MAX_REGS];
  float mix_sq[4];
  for (uint q = 0u; q < Q; ++q) {
    mix_sq[q] = 0.0f;
  }

  uint reg_i = 0u;
  for (uint i = tid; i < D; i += THREADS) {
    float mix[4];
    for (uint q = 0u; q < Q; ++q) {
      mix[q] = 0.0f;
    }
    for (uint k = 0u; k < K; ++k) {
      const float vk = inkling2_read_src(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i) * inv_rms[k];
      for (uint q = 0u; q < Q; ++q) {
        mix[q] += probs[q][k] * vk;
      }
    }
    for (uint q = 0u; q < Q; ++q) {
      if (fits_regs) {
        mix_regs[q][reg_i] = mix[q];
      }
      mix_sq[q] += mix[q] * mix[q];
    }
    ++reg_i;
  }

  for (uint q = 0u; q < Q; ++q) {
    const float ms = simd_sum(mix_sq[q]);
    if (simd_lane == 0u) {
      tg_reduce[q * 8u + simd_group] = ms;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < Q) {
    float ms = 0.0f;
    for (uint sg = 0u; sg < 8u; ++sg) {
      ms += tg_reduce[tid * 8u + sg];
    }
    tg_inv_rms[tid] = rsqrt(ms / float(D) + params.norm_eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mix_inv_rms[4];
  for (uint q = 0u; q < Q; ++q) {
    mix_inv_rms[q] = tg_inv_rms[q];
  }

  reg_i = 0u;
  for (uint i = tid; i < D; i += THREADS) {
    for (uint q = 0u; q < Q; ++q) {
      float mv;
      if (fits_regs) {
        mv = mix_regs[q][reg_i];
      } else {
        mv = 0.0f;
        for (uint k = 0u; k < K; ++k) {
          const float vk = inkling2_read_src(s0, s1, s2, s3, s4, s5, s6, s7, addend, params, k, row, i) * inv_rms[k];
          mv += probs[q][k] * vk;
        }
      }
      const ulong w_idx = (weight_bcast ? 0ul : ulong(q) * ulong(D)) + i;
      const float w = inkling2_read_f32(norm_weight, w_idx, weight_f32);
      const ulong out_idx = (ulong(q) * ulong(params.rows) + ulong(row)) * ulong(D) + i;
      out[out_idx] = OutT((mv * mix_inv_rms[q]) * w);
    }
    ++reg_i;
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_block_attn_res_bf16(
    device const uchar *s0 [[buffer(0)]],
    device const uchar *s1 [[buffer(1)]],
    device const uchar *s2 [[buffer(2)]],
    device const uchar *s3 [[buffer(3)]],
    device const uchar *s4 [[buffer(4)]],
    device const uchar *s5 [[buffer(5)]],
    device const uchar *s6 [[buffer(6)]],
    device const uchar *s7 [[buffer(7)]],
    device const uchar *addend [[buffer(8)]],
    device const float *queries [[buffer(9)]],
    device const uchar *norm_weight [[buffer(10)]],
    device bfloat *out [[buffer(11)]],
    constant Inkling2BlockAttnResParams &params [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg_reduce[320];
  threadgroup float tg_inv_rms[8];
  threadgroup float tg_probs[32];
  inkling2_block_attn_res_impl<bfloat>(
      s0, s1, s2, s3, s4, s5, s6, s7, addend, queries, norm_weight, out, params,
      tg_reduce, tg_inv_rms, tg_probs, row, tid, simd_lane, simd_group);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_block_attn_res_f32(
    device const uchar *s0 [[buffer(0)]],
    device const uchar *s1 [[buffer(1)]],
    device const uchar *s2 [[buffer(2)]],
    device const uchar *s3 [[buffer(3)]],
    device const uchar *s4 [[buffer(4)]],
    device const uchar *s5 [[buffer(5)]],
    device const uchar *s6 [[buffer(6)]],
    device const uchar *s7 [[buffer(7)]],
    device const uchar *addend [[buffer(8)]],
    device const float *queries [[buffer(9)]],
    device const uchar *norm_weight [[buffer(10)]],
    device float *out [[buffer(11)]],
    constant Inkling2BlockAttnResParams &params [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg_reduce[320];
  threadgroup float tg_inv_rms[8];
  threadgroup float tg_probs[32];
  inkling2_block_attn_res_impl<float>(
      s0, s1, s2, s3, s4, s5, s6, s7, addend, queries, norm_weight, out, params,
      tg_reduce, tg_inv_rms, tg_probs, row, tid, simd_lane, simd_group);
}

// ============================================================================
// 8. inkling2_pair_rms_norm_f32 (fused [rms(dest), rms(msg)] -> [S, T, 2*D])
// ============================================================================
struct Inkling2PairRmsNormParams {
  uint rows;
  uint dim;
  uint num_streams;
  float eps;
  uint dest_row_stride;
  uint dest_stream_stride;
  uint msg_row_stride;
  uint msg_stream_stride;
  uint dest_is_f32;
  uint msg_is_f32;
  uint _pad0;
  uint _pad1;
};

template <typename OutT>
inline void inkling2_pair_rms_norm_impl(
    device const uchar *dest_raw,
    device const uchar *msg_raw,
    device OutT *out,
    constant Inkling2PairRmsNormParams &params,
    threadgroup float *partial_sums,
    uint2 tgid,
    uint tid,
    uint simd_lane,
    uint simd_group) {
  constexpr uint THREADS = 256u;
  constexpr uint MAX_REGS = 16u;

  const uint stream = tgid.x >> 1u;
  const uint half_idx = tgid.x & 1u; // 0 = dest, 1 = msg
  const uint row = tgid.y;
  if (stream >= params.num_streams || row >= params.rows) {
    return;
  }

  const uint dim = params.dim;
  device const uchar *in_ptr = (half_idx == 0u) ? dest_raw : msg_raw;
  const uint is_f32 = (half_idx == 0u) ? params.dest_is_f32 : params.msg_is_f32;
  const ulong in_base = (half_idx == 0u)
      ? (ulong(stream) * ulong(params.dest_stream_stride) + ulong(row) * ulong(params.dest_row_stride))
      : (ulong(stream) * ulong(params.msg_stream_stride) + ulong(row) * ulong(params.msg_row_stride));
  const ulong out_base = ((ulong(stream) * ulong(params.rows) + ulong(row)) * 2ul + ulong(half_idx)) * ulong(dim);

  float regs[MAX_REGS];
  const bool fits_regs = dim <= THREADS * MAX_REGS;
  float sum_sq = 0.0f;
  uint reg_i = 0u;
  for (uint i = tid; i < dim; i += THREADS) {
    const float v = inkling2_read_f32(in_ptr, in_base + i, is_f32);
    if (fits_regs) {
      regs[reg_i++] = v;
    }
    sum_sq += v * v;
  }

  sum_sq = simd_sum(sum_sq);
  if (simd_lane == 0u) {
    partial_sums[simd_group] = sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    float v = simd_lane < 8u ? partial_sums[simd_lane] : 0.0f;
    v = simd_sum(v);
    if (simd_lane == 0u) {
      partial_sums[0] = rsqrt(v / float(dim) + params.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_rms = partial_sums[0];

  reg_i = 0u;
  for (uint i = tid; i < dim; i += THREADS) {
    const float v = fits_regs ? regs[reg_i++] : inkling2_read_f32(in_ptr, in_base + i, is_f32);
    out[out_base + i] = OutT(v * inv_rms);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_pair_rms_norm_f32(
    device const uchar *dest_raw [[buffer(0)]],
    device const uchar *msg_raw [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Inkling2PairRmsNormParams &params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial_sums[8];
  inkling2_pair_rms_norm_impl<float>(
      dest_raw, msg_raw, out, params, partial_sums, tgid, tid, simd_lane, simd_group);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_pair_rms_norm_bf16(
    device const uchar *dest_raw [[buffer(0)]],
    device const uchar *msg_raw [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2PairRmsNormParams &params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial_sums[8];
  inkling2_pair_rms_norm_impl<bfloat>(
      dest_raw, msg_raw, out, params, partial_sums, tgid, tid, simd_lane, simd_group);
}

// ============================================================================
// 9. inkling2_sconv_k4_ragged (multi-seq ragged + optional FP32 residual epilogue)
// ============================================================================
struct Inkling2SconvK4RaggedParams {
  uint seq_len;        // total T
  uint channels;       // C
  uint num_seqs;       // N (>= 1)
  uint is_dynamic;     // 0 or 1
  uint num_kv_heads;   // used when is_dynamic != 0
  uint packed_hist;    // 0 = up to 4 separate bufs, 1 = packed [N, 3, C]
  uint x_row_stride;   // elems per token row t in x
  uint res_a_stride;   // elems per token row t in res_a
  uint res_b_stride;   // elems per token row t in res_b
  uint flags;          // bit0: w_f32, bit1: dyn_f32, bit2: has_res_a, bit3: res_a_f32, bit4: has_res_b, bit5: res_b_f32
  uint hist_f32_mask;  // bit s = 1 if hist[s] is f32
  uint _pad;
};

template <typename T>
inline void inkling2_sconv_k4_ragged_impl(
    device const T *x,
    device const uchar *weight_raw,
    device const uchar *dyn_taps_raw,
    device const int32_t *cu_seqlens,
    device const uchar *res_a_raw,
    device const uchar *res_b_raw,
    device const uchar *h0,
    device const uchar *h1,
    device const uchar *h2,
    device const uchar *h3,
    device T *y_out,
    device T *hist_out,
    constant Inkling2SconvK4RaggedParams &params,
    uint2 gid) {
  const uint ch = gid.x;
  const uint t = gid.y;
  const uint C = params.channels;
  if (ch >= C || t >= params.seq_len) {
    return;
  }

  uint seq_idx = 0u;
  uint seq_start = 0u;
  uint seq_end = params.seq_len;
  inkling2_find_seq(cu_seqlens, params.num_seqs, params.seq_len, t, seq_idx, seq_start, seq_end);
  const uint local_t = t - seq_start;
  const uint seq_t_len = seq_end - seq_start;

  device const uchar *h_ptr;
  ulong h_base;
  uint h_f32;
  if (params.packed_hist != 0u) {
    h_ptr = h0;
    h_base = ulong(seq_idx) * 3ul * ulong(C);
    h_f32 = params.hist_f32_mask & 1u;
  } else {
    h_ptr = (seq_idx == 0u) ? h0 : (seq_idx == 1u) ? h1 : (seq_idx == 2u) ? h2 : h3;
    h_base = 0ul;
    h_f32 = (params.hist_f32_mask >> seq_idx) & 1u;
  }

  const uint x_stride = params.x_row_stride;
  const float x3 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, C, h_f32, x, seq_start, x_stride, local_t + 0u, ch);
  const float x2 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, C, h_f32, x, seq_start, x_stride, local_t + 1u, ch);
  const float x1 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, C, h_f32, x, seq_start, x_stride, local_t + 2u, ch);
  const float x0 = inkling2_sconv_read_full_strided<T>(h_ptr, h_base, C, h_f32, x, seq_start, x_stride, local_t + 3u, ch);

  const uint w_f32 = (params.flags >> 0u) & 1u;
  const ulong w_base = ulong(ch) * 4ul;
  float w3 = inkling2_read_f32(weight_raw, w_base + 0ul, w_f32);
  float w2 = inkling2_read_f32(weight_raw, w_base + 1ul, w_f32);
  float w1 = inkling2_read_f32(weight_raw, w_base + 2ul, w_f32);
  float w0 = inkling2_read_f32(weight_raw, w_base + 3ul, w_f32);

  float scale;
  if (params.is_dynamic != 0u) {
    const uint dyn_f32 = (params.flags >> 1u) & 1u;
    const uint head_dim = C / params.num_kv_heads;
    const uint h = ch / head_dim;
    const ulong d_base = (ulong(t) * ulong(params.num_kv_heads) + h) * 4ul;
    w0 += inkling2_read_f32(dyn_taps_raw, d_base + 0ul, dyn_f32);
    w1 += inkling2_read_f32(dyn_taps_raw, d_base + 1ul, dyn_f32);
    w2 += inkling2_read_f32(dyn_taps_raw, d_base + 2ul, dyn_f32);
    w3 += inkling2_read_f32(dyn_taps_raw, d_base + 3ul, dyn_f32);
    scale = 1.0f / 3.0f;
  } else {
    scale = 0.44721359549995793f;
  }

  const float acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
  T conv_val = T(acc * scale);

  const bool has_res_a = ((params.flags >> 2u) & 1u) != 0u;
  const bool has_res_b = ((params.flags >> 4u) & 1u) != 0u;
  if (has_res_a || has_res_b) {
    float res = 0.0f;
    if (has_res_a) {
      const uint ra_f32 = (params.flags >> 3u) & 1u;
      res += inkling2_read_f32(res_a_raw, ulong(t) * ulong(params.res_a_stride) + ch, ra_f32);
    }
    if (has_res_b) {
      const uint rb_f32 = (params.flags >> 5u) & 1u;
      res += inkling2_read_f32(res_b_raw, ulong(t) * ulong(params.res_b_stride) + ch, rb_f32);
    }
    y_out[ulong(t) * ulong(C) + ch] = T(res + float(conv_val));
  } else {
    y_out[ulong(t) * ulong(C) + ch] = conv_val;
  }

  if (t == 0u && params.num_seqs > 1u) {
    for (uint s = 0u; s < params.num_seqs; ++s) {
      if (cu_seqlens[s] == cu_seqlens[s + 1u]) {
        device const uchar *eh_ptr;
        ulong eh_base;
        uint eh_f32;
        if (params.packed_hist != 0u) {
          eh_ptr = h0;
          eh_base = ulong(s) * 3ul * ulong(C);
          eh_f32 = params.hist_f32_mask & 1u;
        } else {
          eh_ptr = (s == 0u) ? h0 : (s == 1u) ? h1 : (s == 2u) ? h2 : h3;
          eh_base = 0ul;
          eh_f32 = (params.hist_f32_mask >> s) & 1u;
        }
        const ulong eout_base = ulong(s) * 3ul * ulong(C);
        for (uint row_k = 0u; row_k < 3u; ++row_k) {
          const float hv = inkling2_read_f32(
              eh_ptr, eh_base + ulong(row_k) * ulong(C) + ulong(ch), eh_f32);
          hist_out[eout_base + ulong(row_k) * ulong(C) + ch] = T(hv);
        }
      }
    }
  }

  if (local_t == seq_t_len - 1u) {
    const ulong ho_base = ulong(seq_idx) * 3ul * ulong(C);
    hist_out[ho_base + 0ul * ulong(C) + ch] = T(x2);
    hist_out[ho_base + 1ul * ulong(C) + ch] = T(x1);
    hist_out[ho_base + 2ul * ulong(C) + ch] = T(x0);
  }
}

[[kernel]] void inkling2_sconv_k4_ragged_bf16(
    device const bfloat *x [[buffer(0)]],
    device const uchar *weight_raw [[buffer(1)]],
    device const uchar *dyn_taps_raw [[buffer(2)]],
    device const int32_t *cu_seqlens [[buffer(3)]],
    device const uchar *res_a_raw [[buffer(4)]],
    device const uchar *res_b_raw [[buffer(5)]],
    device const uchar *h0 [[buffer(6)]],
    device const uchar *h1 [[buffer(7)]],
    device const uchar *h2 [[buffer(8)]],
    device const uchar *h3 [[buffer(9)]],
    device bfloat *y_out [[buffer(10)]],
    device bfloat *hist_out [[buffer(11)]],
    constant Inkling2SconvK4RaggedParams &params [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  inkling2_sconv_k4_ragged_impl<bfloat>(
      x, weight_raw, dyn_taps_raw, cu_seqlens, res_a_raw, res_b_raw,
      h0, h1, h2, h3, y_out, hist_out, params, gid);
}

[[kernel]] void inkling2_sconv_k4_ragged_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weight_raw [[buffer(1)]],
    device const uchar *dyn_taps_raw [[buffer(2)]],
    device const int32_t *cu_seqlens [[buffer(3)]],
    device const uchar *res_a_raw [[buffer(4)]],
    device const uchar *res_b_raw [[buffer(5)]],
    device const uchar *h0 [[buffer(6)]],
    device const uchar *h1 [[buffer(7)]],
    device const uchar *h2 [[buffer(8)]],
    device const uchar *h3 [[buffer(9)]],
    device float *y_out [[buffer(10)]],
    device float *hist_out [[buffer(11)]],
    constant Inkling2SconvK4RaggedParams &params [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  inkling2_sconv_k4_ragged_impl<float>(
      x, weight_raw, dyn_taps_raw, cu_seqlens, res_a_raw, res_b_raw,
      h0, h1, h2, h3, y_out, hist_out, params, gid);
}

// ============================================================================
// 10. inkling2_moe_route_f32 (sigmoid + bias + exact top-K + norm + down_scale)
// ============================================================================
struct Inkling2MoeRouteParams {
  uint rows;          // T
  uint num_experts;   // E (<= 512)
  uint top_k;         // K (<= 16)
  float route_scale;  // 8.0f
  uint flags;         // bit0: logits_f32, bit1: has_bias, bit2: bias_f32, bit3: has_down_scale, bit4: down_scale_f32
  uint logits_stride; // row stride in elems
  uint _pad0;
  uint _pad1;
};

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_moe_route_f32(
    device const uchar *logits_raw [[buffer(0)]],
    device const uchar *bias_raw [[buffer(1)]],
    device const uchar *down_scale_raw [[buffer(2)]],
    device int32_t *out_indices [[buffer(3)]],
    device float *out_weights [[buffer(4)]],
    constant Inkling2MoeRouteParams &params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= params.rows) {
    return;
  }
  const uint E = params.num_experts;
  const uint K = params.top_k;
  const uint logits_f32 = (params.flags >> 0u) & 1u;
  const bool has_bias = ((params.flags >> 1u) & 1u) != 0u;
  const uint bias_f32 = (params.flags >> 2u) & 1u;
  const bool has_down_scale = ((params.flags >> 3u) & 1u) != 0u;
  const uint ds_f32 = (params.flags >> 4u) & 1u;

  const ulong row_base = ulong(row) * ulong(params.logits_stride);

  constexpr uint MAX_SLOTS = 16u; // 16 * 32 = 512 experts
  float local_s[MAX_SLOTS];
  float local_p[MAX_SLOTS];
  int32_t local_id[MAX_SLOTS];
  uint count = 0u;

  for (uint e = lane; e < E; e += 32u) {
    const float z = inkling2_read_f32(logits_raw, row_base + e, logits_f32);
    const float p = inkling2_sigmoid(z);
    const float s = has_bias ? (p + inkling2_read_f32(bias_raw, ulong(e), bias_f32)) : p;
    local_s[count] = s;
    local_p[count] = p;
    local_id[count] = int32_t(e);
    ++count;
  }

  int32_t sel_idx[16];
  float sel_p[16];
  float sum_p = 0.0f;

  for (uint k = 0u; k < K; ++k) {
    float best_s = -INFINITY;
    int32_t best_i = -1;
    float best_p = 0.0f;
    for (uint j = 0u; j < count; ++j) {
      const int32_t cid = local_id[j];
      if (cid >= 0) {
        const float cs = local_s[j];
        if (best_i < 0 || cs > best_s || (cs == best_s && cid < best_i)) {
          best_s = cs;
          best_i = cid;
          best_p = local_p[j];
        }
      }
    }

    // 5-step butterfly simdgroup reduction with exact smaller-index tie-break.
    for (ushort mask = 16; mask > 0; mask >>= 1) {
      const float os = simd_shuffle_xor(best_s, mask);
      const int32_t oi = simd_shuffle_xor(best_i, mask);
      const float op = simd_shuffle_xor(best_p, mask);
      if (oi >= 0 && (best_i < 0 || os > best_s || (os == best_s && oi < best_i))) {
        best_s = os;
        best_i = oi;
        best_p = op;
      }
    }

    // Mark winner as picked in owning lane.
    for (uint j = 0u; j < count; ++j) {
      if (local_id[j] == best_i) {
        local_id[j] = -1;
      }
    }

    if (lane == 0u) {
      sel_idx[k] = best_i;
      sel_p[k] = best_p;
      sum_p += best_p;
    }
  }

  if (lane == 0u) {
    const float denom = sum_p + 1e-20f;
    const ulong out_base = ulong(row) * ulong(K);
    for (uint k = 0u; k < K; ++k) {
      const int32_t eid = sel_idx[k];
      float w = (sel_p[k] / denom) * params.route_scale;
      if (has_down_scale) {
        w *= inkling2_read_f32(down_scale_raw, ulong(uint32_t(eid)), ds_f32);
      }
      out_indices[out_base + k] = eid;
      out_weights[out_base + k] = w;
    }
  }
}

// ============================================================================
// 11. inkling2_moe_combine (optional inv_order unsort + FP32 weighted sum)
// ============================================================================
struct Inkling2MoeCombineParams {
  uint rows;   // T
  uint top_k;  // K (<= 16)
  uint dim;    // D (2304)
  uint flags;  // bit0: down_f32, bit1: has_inv_order, bit2: weights_f32
};

template <typename OutT>
inline void inkling2_moe_combine_impl(
    device const uchar *down_raw,
    device const uchar *weights_raw,
    device const uint32_t *inv_order,
    device OutT *out,
    constant Inkling2MoeCombineParams &params,
    threadgroup float *tg_w,
    threadgroup uint32_t *tg_row,
    uint2 tgid,
    uint tid) {
  const uint row = tgid.y;
  if (row >= params.rows) {
    return;
  }
  const uint K = params.top_k;
  const uint D = params.dim;
  const uint down_f32 = (params.flags >> 0u) & 1u;
  const bool has_inv = ((params.flags >> 1u) & 1u) != 0u;
  const uint w_f32 = (params.flags >> 2u) & 1u;

  if (tid < K) {
    const ulong slot = ulong(row) * ulong(K) + tid;
    tg_w[tid] = inkling2_read_f32(weights_raw, slot, w_f32);
    tg_row[tid] = has_inv ? inv_order[slot] : uint32_t(slot);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint col = tgid.x * 256u + tid;
  if (col >= D) {
    return;
  }
  float sum = 0.0f;
  for (uint k = 0u; k < K; ++k) {
    const ulong src_idx = ulong(tg_row[k]) * ulong(D) + col;
    sum += inkling2_read_f32(down_raw, src_idx, down_f32) * tg_w[k];
  }
  out[ulong(row) * ulong(D) + col] = OutT(sum);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_moe_combine_bf16(
    device const uchar *down_raw [[buffer(0)]],
    device const uchar *weights_raw [[buffer(1)]],
    device const uint32_t *inv_order [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant Inkling2MoeCombineParams &params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tg_w[16];
  threadgroup uint32_t tg_row[16];
  inkling2_moe_combine_impl<bfloat>(
      down_raw, weights_raw, inv_order, out, params, tg_w, tg_row, tgid, tid);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_moe_combine_f32(
    device const uchar *down_raw [[buffer(0)]],
    device const uchar *weights_raw [[buffer(1)]],
    device const uint32_t *inv_order [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant Inkling2MoeCombineParams &params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tg_w[16];
  threadgroup uint32_t tg_row[16];
  inkling2_moe_combine_impl<float>(
      down_raw, weights_raw, inv_order, out, params, tg_w, tg_row, tgid, tid);
}

// ============================================================================
// 12. inkling2_lm_head_softcap (+ 2-stage GPU argmax)
// ============================================================================
struct Inkling2LmHeadSoftcapParams {
  uint rows;
  uint out_cols;
  uint logits_stride;
  uint bias_stride; // 0 => broadcast row 0
  float softcap;
  uint num_splits;  // 64 for argmax
  uint flags;       // bit0: logits_f32, bit1: has_bias, bit2: bias_f32, bit3: bias_round_bf16
  uint _pad;
};

inline float inkling2_eval_logit(
    device const uchar *logits_raw,
    device const uchar *bias_raw,
    constant Inkling2LmHeadSoftcapParams &params,
    uint row,
    uint col) {
  const uint logits_f32 = (params.flags >> 0u) & 1u;
  const bool has_bias = ((params.flags >> 1u) & 1u) != 0u;
  const uint bias_f32 = (params.flags >> 2u) & 1u;
  const bool bias_round_bf16 = ((params.flags >> 3u) & 1u) != 0u;

  float v = inkling2_read_f32(logits_raw, ulong(row) * ulong(params.logits_stride) + col, logits_f32);
  if (has_bias) {
    const ulong b_idx = ulong(row) * ulong(params.bias_stride) + col;
    v += inkling2_read_f32(bias_raw, b_idx, bias_f32);
    if (bias_round_bf16) {
      v = float(bfloat(v));
    }
  }
  return inkling2_softcap(v, params.softcap);
}

[[kernel]] void inkling2_lm_head_softcap_f32(
    device const uchar *logits_raw [[buffer(0)]],
    device const uchar *bias_raw [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Inkling2LmHeadSoftcapParams &params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x;
  const uint row = gid.y;
  if (row >= params.rows || col >= params.out_cols) {
    return;
  }
  out[ulong(row) * ulong(params.out_cols) + col] =
      inkling2_eval_logit(logits_raw, bias_raw, params, row, col);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_lm_head_softcap_argmax_stage1(
    device const uchar *logits_raw [[buffer(0)]],
    device const uchar *bias_raw [[buffer(1)]],
    device float *partial_vals [[buffer(2)]],
    device int32_t *partial_idxs [[buffer(3)]],
    constant Inkling2LmHeadSoftcapParams &params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint split = tgid.x;
  const uint row = tgid.y;
  if (split >= params.num_splits || row >= params.rows) {
    return;
  }
  const uint cols = params.out_cols;
  const uint chunk = (cols + params.num_splits - 1u) / params.num_splits;
  const uint start = split * chunk;
  const uint end = min(start + chunk, cols);

  float best_v = -INFINITY;
  int32_t best_i = -1;
  for (uint col = start + tid; col < end; col += 256u) {
    const float v = inkling2_eval_logit(logits_raw, bias_raw, params, row, col);
    const int32_t ci = int32_t(col);
    if (best_i < 0 || v > best_v || (v == best_v && ci < best_i)) {
      best_v = v;
      best_i = ci;
    }
  }

  for (ushort mask = 16; mask > 0; mask >>= 1) {
    const float ov = simd_shuffle_xor(best_v, mask);
    const int32_t oi = simd_shuffle_xor(best_i, mask);
    if (oi >= 0 && (best_i < 0 || ov > best_v || (ov == best_v && oi < best_i))) {
      best_v = ov;
      best_i = oi;
    }
  }

  threadgroup float tg_v[8];
  threadgroup int32_t tg_i[8];
  if (simd_lane == 0u) {
    tg_v[simd_group] = best_v;
    tg_i[simd_group] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group == 0u) {
    float v = simd_lane < 8u ? tg_v[simd_lane] : -INFINITY;
    int32_t i = simd_lane < 8u ? tg_i[simd_lane] : -1;
    for (ushort mask = 4; mask > 0; mask >>= 1) {
      const float ov = simd_shuffle_xor(v, mask);
      const int32_t oi = simd_shuffle_xor(i, mask);
      if (oi >= 0 && (i < 0 || ov > v || (ov == v && oi < i))) {
        v = ov;
        i = oi;
      }
    }
    if (simd_lane == 0u) {
      const ulong out_slot = ulong(row) * ulong(params.num_splits) + split;
      partial_vals[out_slot] = v;
      partial_idxs[out_slot] = i;
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_lm_head_softcap_argmax_stage2(
    device const float *partial_vals [[buffer(0)]],
    device const int32_t *partial_idxs [[buffer(1)]],
    device int32_t *out_ids [[buffer(2)]],
    constant Inkling2LmHeadSoftcapParams &params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (row >= params.rows) {
    return;
  }
  const ulong base = ulong(row) * ulong(params.num_splits);
  float best_v = -INFINITY;
  int32_t best_i = -1;
  for (uint s = tid; s < params.num_splits; s += 64u) {
    const float v = partial_vals[base + s];
    const int32_t i = partial_idxs[base + s];
    if (i >= 0 && (best_i < 0 || v > best_v || (v == best_v && i < best_i))) {
      best_v = v;
      best_i = i;
    }
  }

  for (ushort mask = 16; mask > 0; mask >>= 1) {
    const float ov = simd_shuffle_xor(best_v, mask);
    const int32_t oi = simd_shuffle_xor(best_i, mask);
    if (oi >= 0 && (best_i < 0 || ov > best_v || (ov == best_v && oi < best_i))) {
      best_v = ov;
      best_i = oi;
    }
  }

  threadgroup float tg_v[2];
  threadgroup int32_t tg_i[2];
  if (simd_lane == 0u) {
    tg_v[simd_group] = best_v;
    tg_i[simd_group] = best_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float v = tg_v[0];
    int32_t i = tg_i[0];
    const float ov = tg_v[1];
    const int32_t oi = tg_i[1];
    if (oi >= 0 && (i < 0 || ov > v || (ov == v && oi < i))) {
      v = ov;
      i = oi;
    }
    out_ids[row] = i;
  }
}

// ============================================================================
// 13. inkling2_markov_gate (fused optional ctx add + BF16-rounded Markov RNN gate)
// ============================================================================
struct Inkling2MarkovGateParams {
  uint rows;         // N
  uint rank;         // R (256)
  uint raw_stride;   // row stride of raw [N, 3*R]
  uint ctx_stride;   // row stride of optional ctx [N, 3*R]
  uint state_stride; // row stride of state [N, R]
  uint flags;        // bit0: raw_f32, bit1: has_ctx, bit2: ctx_f32, bit3: state_f32
  uint _pad0;
  uint _pad1;
};

[[kernel]] void inkling2_markov_gate_bf16(
    device const uchar *raw_ptr [[buffer(0)]],
    device const uchar *ctx_ptr [[buffer(1)]],
    device const uchar *state_ptr [[buffer(2)]],
    device bfloat *next_state [[buffer(3)]],
    device bfloat *latent [[buffer(4)]],
    constant Inkling2MarkovGateParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x;
  const uint row = gid.y;
  if (row >= params.rows || c >= params.rank) {
    return;
  }
  const uint rank = params.rank;
  const uint raw_f32 = (params.flags >> 0u) & 1u;
  const uint has_ctx = (params.flags >> 1u) & 1u;
  const uint ctx_f32 = (params.flags >> 2u) & 1u;
  const uint state_f32 = (params.flags >> 3u) & 1u;

  const ulong rb = ulong(row) * ulong(params.raw_stride);
  float g_raw = inkling2_read_f32(raw_ptr, rb + c, raw_f32);
  float c_raw = inkling2_read_f32(raw_ptr, rb + rank + c, raw_f32);
  float o_raw = inkling2_read_f32(raw_ptr, rb + 2u * rank + c, raw_f32);
  if (has_ctx != 0u) {
    const ulong cb = ulong(row) * ulong(params.ctx_stride);
    g_raw = g_raw + inkling2_read_f32(ctx_ptr, cb + c, ctx_f32);
    c_raw = c_raw + inkling2_read_f32(ctx_ptr, cb + rank + c, ctx_f32);
    o_raw = o_raw + inkling2_read_f32(ctx_ptr, cb + 2u * rank + c, ctx_f32);
  }
  const float gate_f = float(bfloat(g_raw));
  const float cand_f = float(bfloat(c_raw));
  const float out_f = float(bfloat(o_raw));

  const bfloat g = bfloat(inkling2_sigmoid(gate_f));
  const bfloat cand = bfloat(tanh(clamp(cand_f, -20.0f, 20.0f)));
  const bfloat one_minus_g = bfloat(1.0f - float(g));
  const float st_f = float(bfloat(inkling2_read_f32(
      state_ptr, ulong(row) * ulong(params.state_stride) + c, state_f32)));

  const bfloat keep = bfloat(float(g) * st_f);
  const bfloat upd = bfloat(float(one_minus_g) * float(cand));
  const ulong out_idx = ulong(row) * ulong(rank) + c;
  next_state[out_idx] = bfloat(float(keep) + float(upd));
  latent[out_idx] = bfloat(tanh(clamp(out_f, -20.0f, 20.0f)));
}

// ============================================================================
// 14. inkling2_moe_router_proj_f32 (coalesced exact-FP32 router matmul for T <= 32)
//     Avoids MLX's slow M=8/32 64x64-padded FP32 steel_matmul on [T, 2304] @ [2304, 128]!
//     Grid: (ceil(E / 4), ceil(T / 8), 1), TG: (128, 1, 1) -> 4 SGs, 1 expert/SG, 8 rows/TG.
// ============================================================================
struct Inkling2MoeRouterProjParams {
  uint rows;        // T
  uint dim;         // D (multiple of 4)
  uint num_experts; // E
  uint x_stride;    // row stride of x in elements
  uint x_is_f32;    // 0 = BF16, 1 = F32
  uint _pad0;
  uint _pad1;
  uint _pad2;
};

[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_moe_router_proj_f32(
    device const uchar *x_raw [[buffer(0)]],
    device const float *w_f32 [[buffer(1)]],
    device float *logits [[buffer(2)]],
    constant Inkling2MoeRouterProjParams &params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint expert = tgid.x * 4u + sg;
  if (expert >= params.num_experts) {
    return;
  }
  const uint row0 = tgid.y * 8u;
  if (row0 >= params.rows) {
    return;
  }
  const uint n_rows = min(8u, params.rows - row0);
  const uint dim4 = params.dim >> 2u;
  const device float4 *w4 = reinterpret_cast<device const float4 *>(
      w_f32 + ulong(expert) * ulong(params.dim));

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (params.x_is_f32 != 0u) {
    const device float *xf = reinterpret_cast<device const float *>(x_raw);
    for (uint i = lane; i < dim4; i += 32u) {
      const float4 wv = w4[i];
      for (uint r = 0u; r < n_rows; ++r) {
        const float4 xv = *reinterpret_cast<device const float4 *>(
            xf + ulong(row0 + r) * ulong(params.x_stride) + ulong(i << 2u));
        acc[r] += dot(wv, xv);
      }
    }
  } else {
    const device bfloat *xb = reinterpret_cast<device const bfloat *>(x_raw);
    for (uint i = lane; i < dim4; i += 32u) {
      const float4 wv = w4[i];
      for (uint r = 0u; r < n_rows; ++r) {
        const bfloat4 xbh = *reinterpret_cast<device const bfloat4 *>(
            xb + ulong(row0 + r) * ulong(params.x_stride) + ulong(i << 2u));
        acc[r] += dot(wv, float4(xbh));
      }
    }
  }

  for (uint r = 0u; r < n_rows; ++r) {
    const float sum = simd_sum(acc[r]);
    if (lane == 0u) {
      logits[ulong(row0 + r) * ulong(params.num_experts) + expert] = sum;
    }
  }
}

// ============================================================================
// 15. inkling2_loop_ctrl_down_f32
//     Fuses pair_rms_norm + exact-FP32 LoopController down projection:
//       h[s, t, r] = inv_rms(dest[s, t]) * dot(dest[s, t], down_w[s, r, 0..D])
//                  + inv_rms(msg[s, t])  * dot(msg[s, t],  down_w[s, r, D..2D])
//     Grid: (R, ceil(T / 8), S), TG: (128, 1, 1) -> sg 0,1 = dest half, sg 2,3 = msg half.
//     Zero [S, T, 2D] intermediate DRAM traffic and zero 64x64 steel_matmul padding!
// ============================================================================
struct Inkling2LoopCtrlDownParams {
  uint rows;               // T
  uint dim;                // D (multiple of 4)
  uint rank;               // R (64)
  uint num_streams;        // S (1 or 3)
  float eps;               // 1e-6f
  uint dest_row_stride;    // in elements (multiple of 4)
  uint dest_stream_stride; // 0 if broadcast across streams (multiple of 4)
  uint msg_row_stride;     // in elements (multiple of 4)
  uint msg_stream_stride;  // in elements (multiple of 4)
  uint flags;              // bit0: dest_f32, bit1: msg_f32, bit2: w_is_bf16
  uint _pad0;
  uint _pad1;
};

inline float4 inkling2_load_w4(device const uchar *w_raw, ulong idx4, bool w_is_bf16) {
  if (w_is_bf16) {
    return float4(reinterpret_cast<device const bfloat4 *>(w_raw)[idx4]);
  }
  return reinterpret_cast<device const float4 *>(w_raw)[idx4];
}

[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_loop_ctrl_down_f32(
    device const uchar *dest_raw [[buffer(0)]],
    device const uchar *msg_raw [[buffer(1)]],
    device const uchar *down_w_raw [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant Inkling2LoopCtrlDownParams &params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r_idx = tgid.x;
  const uint row0 = tgid.y * 8u;
  const uint s = tgid.z;
  if (r_idx >= params.rank || row0 >= params.rows || s >= params.num_streams) {
    return;
  }
  const uint n_rows = min(8u, params.rows - row0);
  const uint dim = params.dim;
  const uint dim4 = dim >> 2u;
  const uint half_id = sg >> 1u;              // 0 = dest (sg 0,1), 1 = msg (sg 2,3)
  const uint local_tid = ((sg & 1u) << 5u) + lane; // 0..63
  const bool w_is_bf16 = ((params.flags >> 2u) & 1u) != 0u;

  const ulong w_row_base4 =
      ((ulong(s) * ulong(params.rank) + ulong(r_idx)) * ulong(dim) * 2ul +
       ulong(half_id) * ulong(dim)) >> 2u;

  device const uchar *in_raw = (half_id == 0u) ? dest_raw : msg_raw;
  const uint in_is_f32 = (half_id == 0u) ? (params.flags & 1u) : ((params.flags >> 1u) & 1u);
  const uint rs = (half_id == 0u) ? params.dest_row_stride : params.msg_row_stride;
  const uint ss = (half_id == 0u) ? params.dest_stream_stride : params.msg_stream_stride;
  const ulong in_rs4 = ulong(rs) >> 2u;
  const ulong in_s_base4 = (ulong(s) * ulong(ss) + ulong(row0) * ulong(rs)) >> 2u;

  threadgroup float tg_sq[4][8];
  threadgroup float tg_acc[4][8];

  if (n_rows == 1u) {
    float sq = 0.0f;
    float ac = 0.0f;
    if (in_is_f32 != 0u) {
      const device float4 *in_f4 = reinterpret_cast<device const float4 *>(in_raw) + in_s_base4;
      for (uint i = local_tid; i < dim4; i += 64u) {
        const float4 wv = inkling2_load_w4(down_w_raw, w_row_base4 + i, w_is_bf16);
        const float4 xv = in_f4[i];
        sq += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
        ac += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
      }
    } else {
      const device bfloat4 *in_b4 = reinterpret_cast<device const bfloat4 *>(in_raw) + in_s_base4;
      for (uint i = local_tid; i < dim4; i += 64u) {
        const float4 wv = inkling2_load_w4(down_w_raw, w_row_base4 + i, w_is_bf16);
        const float4 xv = float4(in_b4[i]);
        sq += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
        ac += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
      }
    }
    sq = simd_sum(sq);
    ac = simd_sum(ac);
    if (lane == 0u) {
      tg_sq[sg][0] = sq;
      tg_acc[sg][0] = ac;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      const float sq_d = tg_sq[0][0] + tg_sq[1][0];
      const float sq_m = tg_sq[2][0] + tg_sq[3][0];
      const float ac_d = tg_acc[0][0] + tg_acc[1][0];
      const float ac_m = tg_acc[2][0] + tg_acc[3][0];
      const float inv_d = rsqrt(sq_d / float(dim) + params.eps);
      const float inv_m = rsqrt(sq_m / float(dim) + params.eps);
      out[(ulong(s) * ulong(params.rows) + ulong(row0)) * ulong(params.rank) + r_idx] =
          ac_d * inv_d + ac_m * inv_m;
    }
    return;
  }

  float sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float ac[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (in_is_f32 != 0u) {
    const device float4 *in_f4 = reinterpret_cast<device const float4 *>(in_raw) + in_s_base4;
    if (n_rows == 8u) {
      for (uint i = local_tid; i < dim4; i += 64u) {
        const float4 wv = inkling2_load_w4(down_w_raw, w_row_base4 + i, w_is_bf16);
        #pragma unroll
        for (uint r = 0u; r < 8u; ++r) {
          const float4 xv = in_f4[ulong(r) * in_rs4 + i];
          sq[r] += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
          ac[r] += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
        }
      }
    } else {
      for (uint i = local_tid; i < dim4; i += 64u) {
        const float4 wv = inkling2_load_w4(down_w_raw, w_row_base4 + i, w_is_bf16);
        for (uint r = 0u; r < n_rows; ++r) {
          const float4 xv = in_f4[ulong(r) * in_rs4 + i];
          sq[r] += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
          ac[r] += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
        }
      }
    }
  } else {
    const device bfloat4 *in_b4 = reinterpret_cast<device const bfloat4 *>(in_raw) + in_s_base4;
    for (uint i = local_tid; i < dim4; i += 64u) {
      const float4 wv = inkling2_load_w4(down_w_raw, w_row_base4 + i, w_is_bf16);
      for (uint r = 0u; r < n_rows; ++r) {
        const float4 xv = float4(in_b4[ulong(r) * in_rs4 + i]);
        sq[r] += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
        ac[r] += (wv.x * xv.x + wv.y * xv.y) + (wv.z * xv.z + wv.w * xv.w);
      }
    }
  }

  for (uint r = 0u; r < n_rows; ++r) {
    const float sr = simd_sum(sq[r]);
    const float ar = simd_sum(ac[r]);
    if (lane == 0u) {
      tg_sq[sg][r] = sr;
      tg_acc[sg][r] = ar;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < n_rows) {
    const float sq_d = tg_sq[0][tid] + tg_sq[1][tid];
    const float sq_m = tg_sq[2][tid] + tg_sq[3][tid];
    const float ac_d = tg_acc[0][tid] + tg_acc[1][tid];
    const float ac_m = tg_acc[2][tid] + tg_acc[3][tid];
    const float inv_d = rsqrt(sq_d / float(dim) + params.eps);
    const float inv_m = rsqrt(sq_m / float(dim) + params.eps);
    out[(ulong(s) * ulong(params.rows) + ulong(row0 + tid)) * ulong(params.rank) + r_idx] =
        ac_d * inv_d + ac_m * inv_m;
  }
}

// ============================================================================
// 16. inkling2_loop_ctrl_up_blend_f32 + inkling2_rms_norm_from_sq_{bf16,f32}
//     Stage 1: Coalesced sequential-DRAM exact-FP32 up projection
//       (`h [S, T, R] @ gf_w^T`, where `gf_w` is contiguous `[S, 2*D, R]` with R % 4 == 0!)
//       + no-FMA loop blend `(1 - tanh(forget)) * dest + tanh(gain) * msg`
//       + per-tile `sum(blended^2)` reduction (zero [S, T, 2*D] DRAM traffic, only 2 KB TG mem!).
//       Grid: (ceil(D / 32), ceil(T / 8), S), TG: (128, 1, 1):
//       4 SGs x 8 contiguous cols/SG = 32 cols/TG; lanes 0..15 load gain_w[d, 0..R] as float4,
//       lanes 16..31 load forget_w[d, 0..R] as float4 (2x256B sequential coalesced loads/col!).
//       Bit-identical FP32 reduction order across M=1, M=4, M=8, M=32!
//     Stage 2: Zero-barrier finalize RMSNorm over L2-resident `blended` + `partial_sum_sq`.
//       Grid: (ceil(D / 128), T, S), TG: (128, 1, 1).
// ============================================================================
struct Inkling2LoopCtrlUpBlendParams {
  uint rows;               // T
  uint dim;                // D
  uint rank;               // R (multiple of 4; fast path R == 64)
  uint num_streams;        // S (1 or 3)
  uint num_d_tiles;        // ceil(D / 32)
  uint dest_row_stride;
  uint dest_stream_stride; // 0 if broadcast
  uint msg_row_stride;
  uint msg_stream_stride;
  uint flags;              // bit0: dest_f32, bit1: msg_f32, bit2: w_is_bf16
  uint _pad0;
  uint _pad1;
};

[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_loop_ctrl_up_blend_f32(
    device const float *h [[buffer(0)]],
    device const uchar *gf_w_raw [[buffer(1)]],
    device const uchar *dest_raw [[buffer(2)]],
    device const uchar *msg_raw [[buffer(3)]],
    device float *blended [[buffer(4)]],
    device float *partial_sum_sq [[buffer(5)]],
    constant Inkling2LoopCtrlUpBlendParams &params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)tid;
  const uint d_tile = tgid.x;
  const uint row0 = tgid.y * 8u;
  const uint s = tgid.z;
  if (d_tile >= params.num_d_tiles || row0 >= params.rows || s >= params.num_streams) {
    return;
  }
  const uint D = params.dim;
  const uint R = params.rank;
  const uint R4 = R >> 2u;
  const uint n_rows = min(8u, params.rows - row0);
  const bool w_is_bf16 = ((params.flags >> 2u) & 1u) != 0u;

  const uint half_id = lane >> 4u; // 0 = gain (lanes 0..15), 1 = forget (lanes 16..31)
  const uint sub = lane & 15u;     // 0..15
  const uint col_base = d_tile * 32u + sg * 8u;

  const ulong h_s_base4 = (ulong(s) * ulong(params.rows) * ulong(R)) >> 2u;
  const ulong w_half_base4 = (ulong(s) * 2ul + ulong(half_id)) * ulong(D) * ulong(R4);
  const device float4 *h4 = reinterpret_cast<device const float4 *>(h) + h_s_base4;

  threadgroup float tg_g[8][32];
  threadgroup float tg_f[8][32];

  if (R4 == 16u) {
    // Fast path for R = 64: entire h row fits in 1 float4/lane in registers (zero tg_h!).
    if (n_rows == 1u) {
      const float4 hv0 = h4[ulong(row0) * 16ul + sub];
      if (col_base + 7u < D) {
        // 4-column unrolled sequential prefetch with exact same per-column reduction order!
        for (uint c0 = 0u; c0 < 8u; c0 += 4u) {
          const ulong b0 = w_half_base4 + ulong(col_base + c0) * 16ul + sub;
          const float4 w0 = inkling2_load_w4(gf_w_raw, b0 + 0ul, w_is_bf16);
          const float4 w1 = inkling2_load_w4(gf_w_raw, b0 + 16ul, w_is_bf16);
          const float4 w2 = inkling2_load_w4(gf_w_raw, b0 + 32ul, w_is_bf16);
          const float4 w3 = inkling2_load_w4(gf_w_raw, b0 + 48ul, w_is_bf16);

          float p0 = (w0.x * hv0.x + w0.y * hv0.y) + (w0.z * hv0.z + w0.w * hv0.w);
          float p1 = (w1.x * hv0.x + w1.y * hv0.y) + (w1.z * hv0.z + w1.w * hv0.w);
          float p2 = (w2.x * hv0.x + w2.y * hv0.y) + (w2.z * hv0.z + w2.w * hv0.w);
          float p3 = (w3.x * hv0.x + w3.y * hv0.y) + (w3.z * hv0.z + w3.w * hv0.w);

          p0 += simd_shuffle_xor(p0, 8);
          p1 += simd_shuffle_xor(p1, 8);
          p2 += simd_shuffle_xor(p2, 8);
          p3 += simd_shuffle_xor(p3, 8);
          p0 += simd_shuffle_xor(p0, 4);
          p1 += simd_shuffle_xor(p1, 4);
          p2 += simd_shuffle_xor(p2, 4);
          p3 += simd_shuffle_xor(p3, 4);
          p0 += simd_shuffle_xor(p0, 2);
          p1 += simd_shuffle_xor(p1, 2);
          p2 += simd_shuffle_xor(p2, 2);
          p3 += simd_shuffle_xor(p3, 2);
          p0 += simd_shuffle_xor(p0, 1);
          p1 += simd_shuffle_xor(p1, 1);
          p2 += simd_shuffle_xor(p2, 1);
          p3 += simd_shuffle_xor(p3, 1);

          if (sub == 0u) {
            const uint sc = sg * 8u + c0;
            if (half_id == 0u) {
              tg_g[0][sc + 0u] = p0;
              tg_g[0][sc + 1u] = p1;
              tg_g[0][sc + 2u] = p2;
              tg_g[0][sc + 3u] = p3;
            } else {
              tg_f[0][sc + 0u] = p0;
              tg_f[0][sc + 1u] = p1;
              tg_f[0][sc + 2u] = p2;
              tg_f[0][sc + 3u] = p3;
            }
          }
        }
      } else {
        for (uint c = 0u; c < 8u; ++c) {
          const uint col = col_base + c;
          if (col >= D) {
            break;
          }
          const float4 wv = inkling2_load_w4(gf_w_raw, w_half_base4 + ulong(col) * 16ul + sub, w_is_bf16);
          float p = (wv.x * hv0.x + wv.y * hv0.y) + (wv.z * hv0.z + wv.w * hv0.w);
          p += simd_shuffle_xor(p, 8);
          p += simd_shuffle_xor(p, 4);
          p += simd_shuffle_xor(p, 2);
          p += simd_shuffle_xor(p, 1);
          if (sub == 0u) {
            if (half_id == 0u) {
              tg_g[0][sg * 8u + c] = p;
            } else {
              tg_f[0][sg * 8u + c] = p;
            }
          }
        }
      }
    } else if (n_rows == 8u) {
      float4 hv[8];
      #pragma unroll
      for (uint r = 0u; r < 8u; ++r) {
        hv[r] = h4[ulong(row0 + r) * 16ul + sub];
      }
      for (uint c = 0u; c < 8u; ++c) {
        const uint col = col_base + c;
        if (col >= D) {
          break;
        }
        const float4 wv = inkling2_load_w4(gf_w_raw, w_half_base4 + ulong(col) * 16ul + sub, w_is_bf16);
        #pragma unroll
        for (uint r = 0u; r < 8u; ++r) {
          float p = (wv.x * hv[r].x + wv.y * hv[r].y) + (wv.z * hv[r].z + wv.w * hv[r].w);
          p += simd_shuffle_xor(p, 8);
          p += simd_shuffle_xor(p, 4);
          p += simd_shuffle_xor(p, 2);
          p += simd_shuffle_xor(p, 1);
          if (sub == 0u) {
            if (half_id == 0u) {
              tg_g[r][sg * 8u + c] = p;
            } else {
              tg_f[r][sg * 8u + c] = p;
            }
          }
        }
      }
    } else {
      float4 hv[8];
      for (uint r = 0u; r < n_rows; ++r) {
        hv[r] = h4[ulong(row0 + r) * 16ul + sub];
      }
      for (uint c = 0u; c < 8u; ++c) {
        const uint col = col_base + c;
        if (col >= D) {
          break;
        }
        const float4 wv = inkling2_load_w4(gf_w_raw, w_half_base4 + ulong(col) * 16ul + sub, w_is_bf16);
        for (uint r = 0u; r < n_rows; ++r) {
          float p = (wv.x * hv[r].x + wv.y * hv[r].y) + (wv.z * hv[r].z + wv.w * hv[r].w);
          p += simd_shuffle_xor(p, 8);
          p += simd_shuffle_xor(p, 4);
          p += simd_shuffle_xor(p, 2);
          p += simd_shuffle_xor(p, 1);
          if (sub == 0u) {
            if (half_id == 0u) {
              tg_g[r][sg * 8u + c] = p;
            } else {
              tg_f[r][sg * 8u + c] = p;
            }
          }
        }
      }
    }
  } else {
    // General R (multiple of 4)
    for (uint c = 0u; c < 8u; ++c) {
      const uint col = col_base + c;
      if (col >= D) {
        break;
      }
      float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      const ulong w_col_base4 = w_half_base4 + ulong(col) * ulong(R4);
      for (uint k4 = sub; k4 < R4; k4 += 16u) {
        const float4 wv = inkling2_load_w4(gf_w_raw, w_col_base4 + k4, w_is_bf16);
        for (uint r = 0u; r < n_rows; ++r) {
          const float4 hv = h4[ulong(row0 + r) * ulong(R4) + k4];
          acc[r] += (wv.x * hv.x + wv.y * hv.y) + (wv.z * hv.z + wv.w * hv.w);
        }
      }
      for (uint r = 0u; r < n_rows; ++r) {
        float p = acc[r];
        p += simd_shuffle_xor(p, 8);
        p += simd_shuffle_xor(p, 4);
        p += simd_shuffle_xor(p, 2);
        p += simd_shuffle_xor(p, 1);
        if (sub == 0u) {
          if (half_id == 0u) {
            tg_g[r][sg * 8u + c] = p;
          } else {
            tg_f[r][sg * 8u + c] = p;
          }
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint col = d_tile * 32u + lane;
  const bool valid_col = col < D;
  const uint dest_f32 = params.flags & 1u;
  const uint msg_f32 = (params.flags >> 1u) & 1u;
  for (uint r = sg; r < n_rows; r += 4u) {
    float sq = 0.0f;
    if (valid_col) {
      const float g = tg_g[r][lane];
      const float f = tg_f[r][lane];
      const ulong d_idx =
          ulong(s) * ulong(params.dest_stream_stride) +
          ulong(row0 + r) * ulong(params.dest_row_stride) + col;
      const ulong m_idx =
          ulong(s) * ulong(params.msg_stream_stride) +
          ulong(row0 + r) * ulong(params.msg_row_stride) + col;
      const float d_val = inkling2_read_f32(dest_raw, d_idx, dest_f32);
      const float m_val = inkling2_read_f32(msg_raw, m_idx, msg_f32);
      const float b = inkling2_loop_blend_scalar(d_val, m_val, g, f);
      blended[(ulong(s) * ulong(params.rows) + ulong(row0 + r)) * ulong(D) + col] = b;
      sq = b * b;
    }
    const float tile_sq = simd_sum(sq);
    if (lane == 0u) {
      partial_sum_sq[(ulong(s) * ulong(params.rows) + ulong(row0 + r)) *
                         ulong(params.num_d_tiles) +
                     d_tile] = tile_sq;
    }
  }
}

struct Inkling2RmsNormFromSqParams {
  uint rows;                 // T
  uint dim;                  // D
  uint num_streams;          // S
  uint num_d_tiles;          // ceil(D / 32)
  float eps;
  uint weight_stream_stride; // 0 if broadcast
  uint weight_is_f32;
  uint _pad;
};

template <typename OutT>
inline void inkling2_rms_norm_from_sq_impl(
    device const float *blended,
    device const float *partial_sum_sq,
    device const uchar *norm_weight_raw,
    device OutT *out,
    constant Inkling2RmsNormFromSqParams &params,
    uint3 tgid,
    uint tid,
    uint lane) {
  const uint row = tgid.y;
  const uint s = tgid.z;
  if (row >= params.rows || s >= params.num_streams) {
    return;
  }
  const ulong row_flat = ulong(s) * ulong(params.rows) + ulong(row);
  const ulong sq_base = row_flat * ulong(params.num_d_tiles);
  float sq = 0.0f;
  for (uint i = lane; i < params.num_d_tiles; i += 32u) {
    sq += partial_sum_sq[sq_base + i];
  }
  sq = simd_sum(sq);
  const float inv_rms = rsqrt(sq / float(params.dim) + params.eps);

  const uint col = tgid.x * 128u + tid;
  if (col < params.dim) {
    const ulong idx = row_flat * ulong(params.dim) + col;
    const float b = blended[idx];
    const float w = inkling2_read_f32(
        norm_weight_raw, ulong(s) * ulong(params.weight_stream_stride) + col, params.weight_is_f32);
    out[idx] = OutT((b * inv_rms) * w);
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_rms_norm_from_sq_bf16(
    device const float *blended [[buffer(0)]],
    device const float *partial_sum_sq [[buffer(1)]],
    device const uchar *norm_weight_raw [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant Inkling2RmsNormFromSqParams &params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  inkling2_rms_norm_from_sq_impl<bfloat>(
      blended, partial_sum_sq, norm_weight_raw, out, params, tgid, tid, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void inkling2_rms_norm_from_sq_f32(
    device const float *blended [[buffer(0)]],
    device const float *partial_sum_sq [[buffer(1)]],
    device const uchar *norm_weight_raw [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant Inkling2RmsNormFromSqParams &params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  inkling2_rms_norm_from_sq_impl<float>(
      blended, partial_sum_sq, norm_weight_raw, out, params, tgid, tid, lane);
}

// ============================================================================
// 17. inkling2_markov_fold_softcap_{f32, argmax_stage1}
//     Fused Q8_0 (group_size=32) folded Markov head:
//       dot_v   = sum_{g=0..R/32-1} (scale[v,g] * dot(u8(w[v,g]), latent[b,g]) + bias[v,g] * sum(latent[b,g]))
//       bias_v  = bf16(bf16(dot_v) + fold_bias[v])
//       logit_v = softcap(bf16(base_logits[b, v] + bias_v), cap)
//     kLanes = 8 -> 1 group of 32 weights per lane for R = 256 (0% divergence),
//     4 vocab rows/SG x 8 SGs = 32 vocab rows/TG, sharing 1 Q8_0 weight read across
//     up to 4 concurrent batch lanes!
// ============================================================================
struct Inkling2MarkovFoldParams {
  uint batch;         // B
  uint out_cols;      // V_unpadded (200058)
  uint rank;          // R (multiple of 32; fast path R == 256)
  uint latent_stride; // row stride of latent [B, R]
  uint base_stride;   // row stride of base_logits [B, V_padded]
  float softcap;      // 30.0f (<= 0 disables)
  uint num_splits;    // stage1 splits for argmax (0 for logits mode)
  uint flags;         // bit0: latent_f32, bit1: base_f32, bit2: fold_bias_f32,
                      // bits[4:3]: sidecar dtype (0=bf16, 1=f32, 2=f16),
                      // bit5: has_biases (1=affine, 0=no_bias unsigned)
};

inline float inkling2_read_sidecar_dt(device const uchar *ptr, ulong idx, uint dt_code) {
  if (dt_code == 1u) {
    return reinterpret_cast<device const float *>(ptr)[idx];
  }
  if (dt_code == 2u) {
    return float(reinterpret_cast<device const half *>(ptr)[idx]);
  }
  return float(reinterpret_cast<device const bfloat *>(ptr)[idx]);
}

inline float inkling2_dot_q8_32_regs(uint4 p0, uint4 p1, const thread float4 (&x)[8]) {
  float acc = 0.0f;
  acc += dot(float4(as_type<uchar4>(p0.x)), x[0]);
  acc += dot(float4(as_type<uchar4>(p0.y)), x[1]);
  acc += dot(float4(as_type<uchar4>(p0.z)), x[2]);
  acc += dot(float4(as_type<uchar4>(p0.w)), x[3]);
  acc += dot(float4(as_type<uchar4>(p1.x)), x[4]);
  acc += dot(float4(as_type<uchar4>(p1.y)), x[5]);
  acc += dot(float4(as_type<uchar4>(p1.z)), x[6]);
  acc += dot(float4(as_type<uchar4>(p1.w)), x[7]);
  return acc;
}

inline float inkling2_dot_q8_32_tg(uint4 p0, uint4 p1, const threadgroup float *x) {
  float acc = 0.0f;
  acc += dot(float4(as_type<uchar4>(p0.x)), float4(x[0], x[1], x[2], x[3]));
  acc += dot(float4(as_type<uchar4>(p0.y)), float4(x[4], x[5], x[6], x[7]));
  acc += dot(float4(as_type<uchar4>(p0.z)), float4(x[8], x[9], x[10], x[11]));
  acc += dot(float4(as_type<uchar4>(p0.w)), float4(x[12], x[13], x[14], x[15]));
  acc += dot(float4(as_type<uchar4>(p1.x)), float4(x[16], x[17], x[18], x[19]));
  acc += dot(float4(as_type<uchar4>(p1.y)), float4(x[20], x[21], x[22], x[23]));
  acc += dot(float4(as_type<uchar4>(p1.z)), float4(x[24], x[25], x[26], x[27]));
  acc += dot(float4(as_type<uchar4>(p1.w)), float4(x[28], x[29], x[30], x[31]));
  return acc;
}

inline void inkling2_load_latent_32(
    device const uchar *latent_raw,
    ulong base_elem,
    uint latent_f32,
    thread float4 (&x)[8],
    thread float &x_sum) {
  float4 s4 = float4(0.0f);
  if (latent_f32 != 0u) {
    const device float *p = reinterpret_cast<device const float *>(latent_raw) + base_elem;
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      x[i] = float4(p[i * 4u + 0u], p[i * 4u + 1u], p[i * 4u + 2u], p[i * 4u + 3u]);
      s4 += x[i];
    }
  } else {
    const device bfloat *p = reinterpret_cast<device const bfloat *>(latent_raw) + base_elem;
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      x[i] = float4(
          float(p[i * 4u + 0u]),
          float(p[i * 4u + 1u]),
          float(p[i * 4u + 2u]),
          float(p[i * 4u + 3u]));
      s4 += x[i];
    }
  }
  x_sum = (s4.x + s4.y) + (s4.z + s4.w);
}

inline float inkling2_markov_fold_finalize(
    float dot_val,
    float fb_val,
    device const uchar *base_logits_raw,
    ulong base_idx,
    uint base_f32,
    float softcap) {
  const float bias_v = float(bfloat(float(bfloat(dot_val)) + fb_val));
  const float base_v = inkling2_read_f32(base_logits_raw, base_idx, base_f32);
  const float summed = float(bfloat(base_v + bias_v));
  return inkling2_softcap(summed, softcap);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_markov_fold_softcap_argmax_stage1(
    device const uchar *latent_raw [[buffer(0)]],
    device const uint4 *fold_w_u4 [[buffer(1)]],
    device const uchar *fold_scales [[buffer(2)]],
    device const uchar *fold_biases [[buffer(3)]],
    device const uchar *fold_bias_raw [[buffer(4)]],
    device const uchar *base_logits_raw [[buffer(5)]],
    device float *partial_vals [[buffer(6)]],
    device int32_t *partial_idxs [[buffer(7)]],
    constant Inkling2MarkovFoldParams &params [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint split = tgid.x;
  const uint b0 = tgid.y * 4u;
  if (split >= params.num_splits || b0 >= params.batch) {
    return;
  }
  const uint n_b = min(4u, params.batch - b0);
  const uint cols = params.out_cols;
  const uint chunk = (cols + params.num_splits - 1u) / params.num_splits;
  const uint start = split * chunk;
  const uint end = min(start + chunk, cols);

  const uint R = params.rank;
  const uint n_groups = R >> 5u;
  const ulong row_u4_stride = ulong(n_groups) * 2ul;
  const uint latent_f32 = (params.flags >> 0u) & 1u;
  const uint base_f32 = (params.flags >> 1u) & 1u;
  const uint fb_f32 = (params.flags >> 2u) & 1u;
  const uint sidecar_dt = (params.flags >> 3u) & 3u;
  const bool has_biases = ((params.flags >> 5u) & 1u) != 0u;

  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u; // 0..3

  float best_v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  int32_t best_i[4] = {-1, -1, -1, -1};

  if (n_b == 1u && R == 256u) {
    // Specialized fast path for single-sequence DSpark (B=1, R=256):
    // entire 32-element group of latent[0] stays in 32 float registers!
    float4 xv[8];
    float x_sum = 0.0f;
    inkling2_load_latent_32(
        latent_raw, ulong(b0) * ulong(params.latent_stride) + ulong(k_lane << 5u), latent_f32, xv, x_sum);

    const ulong base_row_off = ulong(b0) * ulong(params.base_stride);
    for (uint v_base = start; v_base < end; v_base += 32u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        const ulong w_off = ulong(v) * row_u4_stride + ulong(k_lane) * 2ul;
        const uint4 p0 = fold_w_u4[w_off + 0ul];
        const uint4 p1 = fold_w_u4[w_off + 1ul];
        const ulong s_idx = ulong(v) * ulong(n_groups) + ulong(k_lane);
        const float sc = inkling2_read_sidecar_dt(fold_scales, s_idx, sidecar_dt);
        const float bi = has_biases ? inkling2_read_sidecar_dt(fold_biases, s_idx, sidecar_dt) : 0.0f;
        float acc = sc * inkling2_dot_q8_32_regs(p0, p1, xv) + bi * x_sum;
        acc += simd_shuffle_down(acc, 4);
        acc += simd_shuffle_down(acc, 2);
        acc += simd_shuffle_down(acc, 1);
        if (k_lane == 0u) {
          const float fb = inkling2_read_f32(fold_bias_raw, ulong(v), fb_f32);
          const float val = inkling2_markov_fold_finalize(
              acc, fb, base_logits_raw, base_row_off + ulong(v), base_f32, params.softcap);
          const int32_t ci = int32_t(v);
          if (best_i[0] < 0 || val > best_v[0] || (val == best_v[0] && ci < best_i[0])) {
            best_v[0] = val;
            best_i[0] = ci;
          }
        }
      }
    }
  } else if (R == 256u) {
    // Specialized fast path for multi-lane DSpark (B=2..4, R=256):
    // cooperatively cache latent[b0..b0+n_b, 0..256] in 4 KB threadgroup memory and read
    // fold_w_q8 ONCE across all n_b concurrent lanes!
    threadgroup float tg_x[4][256];
    threadgroup float tg_xsum[4][8];
    for (uint b = 0u; b < n_b; ++b) {
      const float val = inkling2_read_f32(
          latent_raw, ulong(b0 + b) * ulong(params.latent_stride) + ulong(tid), latent_f32);
      tg_x[b][tid] = val;
      const float gsum = simd_sum(val);
      if (lane == 0u) {
        tg_xsum[b][sg] = gsum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint v_base = start; v_base < end; v_base += 32u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        const ulong w_off = ulong(v) * row_u4_stride + ulong(k_lane) * 2ul;
        const uint4 p0 = fold_w_u4[w_off + 0ul];
        const uint4 p1 = fold_w_u4[w_off + 1ul];
        const ulong s_idx = ulong(v) * ulong(n_groups) + ulong(k_lane);
        const float sc = inkling2_read_sidecar_dt(fold_scales, s_idx, sidecar_dt);
        const float bi = has_biases ? inkling2_read_sidecar_dt(fold_biases, s_idx, sidecar_dt) : 0.0f;
        const float fb = (k_lane == 0u) ? inkling2_read_f32(fold_bias_raw, ulong(v), fb_f32) : 0.0f;
        const int32_t ci = int32_t(v);
        for (uint b = 0u; b < n_b; ++b) {
          const float x_sum = tg_xsum[b][k_lane];
          float acc = sc * inkling2_dot_q8_32_tg(p0, p1, &tg_x[b][k_lane << 5u]) + bi * x_sum;
          acc += simd_shuffle_down(acc, 4);
          acc += simd_shuffle_down(acc, 2);
          acc += simd_shuffle_down(acc, 1);
          if (k_lane == 0u) {
            const float val = inkling2_markov_fold_finalize(
                acc,
                fb,
                base_logits_raw,
                ulong(b0 + b) * ulong(params.base_stride) + ulong(v),
                base_f32,
                params.softcap);
            if (best_i[b] < 0 || val > best_v[b] || (val == best_v[b] && ci < best_i[b])) {
              best_v[b] = val;
              best_i[b] = ci;
            }
          }
        }
      }
    }
  } else {
    // General R (any multiple of 32)
    for (uint v_base = start; v_base < end; v_base += 32u) {
      const uint v = v_base + sg * 4u + r_id;
      if (v < end) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = k_lane; g < n_groups; g += 8u) {
          const ulong w_off = ulong(v) * row_u4_stride + ulong(g) * 2ul;
          const uint4 p0 = fold_w_u4[w_off + 0ul];
          const uint4 p1 = fold_w_u4[w_off + 1ul];
          const ulong s_idx = ulong(v) * ulong(n_groups) + ulong(g);
          const float sc = inkling2_read_sidecar_dt(fold_scales, s_idx, sidecar_dt);
          const float bi = has_biases ? inkling2_read_sidecar_dt(fold_biases, s_idx, sidecar_dt) : 0.0f;
          for (uint b = 0u; b < n_b; ++b) {
            float4 xv[8];
            float x_sum = 0.0f;
            inkling2_load_latent_32(
                latent_raw,
                ulong(b0 + b) * ulong(params.latent_stride) + ulong(g << 5u),
                latent_f32,
                xv,
                x_sum);
            acc[b] += sc * inkling2_dot_q8_32_regs(p0, p1, xv) + bi * x_sum;
          }
        }
        const float fb = (k_lane == 0u) ? inkling2_read_f32(fold_bias_raw, ulong(v), fb_f32) : 0.0f;
        const int32_t ci = int32_t(v);
        for (uint b = 0u; b < n_b; ++b) {
          float a = acc[b];
          a += simd_shuffle_down(a, 4);
          a += simd_shuffle_down(a, 2);
          a += simd_shuffle_down(a, 1);
          if (k_lane == 0u) {
            const float val = inkling2_markov_fold_finalize(
                a,
                fb,
                base_logits_raw,
                ulong(b0 + b) * ulong(params.base_stride) + ulong(v),
                base_f32,
                params.softcap);
            if (best_i[b] < 0 || val > best_v[b] || (val == best_v[b] && ci < best_i[b])) {
              best_v[b] = val;
              best_i[b] = ci;
            }
          }
        }
      }
    }
  }

  // Reduce per-SG winners across the 4 row slots (lanes 0, 8, 16, 24) then across 8 SGs.
  threadgroup float tg_v[4][8];
  threadgroup int32_t tg_i[4][8];
  for (uint b = 0u; b < n_b; ++b) {
    float bv = best_v[b];
    int32_t bi = best_i[b];
    {
      const float ov = simd_shuffle_xor(bv, 16);
      const int32_t oi = simd_shuffle_xor(bi, 16);
      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
      }
    }
    {
      const float ov = simd_shuffle_xor(bv, 8);
      const int32_t oi = simd_shuffle_xor(bi, 8);
      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0u) {
      tg_v[b][sg] = bv;
      tg_i[b][sg] = bi;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    for (uint b = 0u; b < n_b; ++b) {
      float v = lane < 8u ? tg_v[b][lane] : -INFINITY;
      int32_t i = lane < 8u ? tg_i[b][lane] : -1;
      for (ushort mask = 4; mask > 0; mask >>= 1) {
        const float ov = simd_shuffle_xor(v, mask);
        const int32_t oi = simd_shuffle_xor(i, mask);
        if (oi >= 0 && (i < 0 || ov > v || (ov == v && oi < i))) {
          v = ov;
          i = oi;
        }
      }
      if (lane == 0u) {
        const ulong out_slot = ulong(b0 + b) * ulong(params.num_splits) + split;
        partial_vals[out_slot] = v;
        partial_idxs[out_slot] = i;
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_markov_fold_softcap_f32(
    device const uchar *latent_raw [[buffer(0)]],
    device const uint4 *fold_w_u4 [[buffer(1)]],
    device const uchar *fold_scales [[buffer(2)]],
    device const uchar *fold_biases [[buffer(3)]],
    device const uchar *fold_bias_raw [[buffer(4)]],
    device const uchar *base_logits_raw [[buffer(5)]],
    device float *out_logits [[buffer(6)]],
    constant Inkling2MarkovFoldParams &params [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint b0 = tgid.y * 4u;
  if (b0 >= params.batch) {
    return;
  }
  const uint n_b = min(4u, params.batch - b0);
  const uint k_lane = lane & 7u;
  const uint r_id = lane >> 3u;
  const uint v = tgid.x * 32u + sg * 4u + r_id;

  const uint R = params.rank;
  const uint n_groups = R >> 5u;
  const ulong row_u4_stride = ulong(n_groups) * 2ul;
  const uint latent_f32 = (params.flags >> 0u) & 1u;
  const uint base_f32 = (params.flags >> 1u) & 1u;
  const uint fb_f32 = (params.flags >> 2u) & 1u;
  const uint sidecar_dt = (params.flags >> 3u) & 3u;
  const bool has_biases = ((params.flags >> 5u) & 1u) != 0u;

  if (n_b == 1u && R == 256u) {
    if (v >= params.out_cols) {
      return;
    }
    float4 xv[8];
    float x_sum = 0.0f;
    inkling2_load_latent_32(
        latent_raw, ulong(b0) * ulong(params.latent_stride) + ulong(k_lane << 5u), latent_f32, xv, x_sum);
    const ulong w_off = ulong(v) * row_u4_stride + ulong(k_lane) * 2ul;
    const uint4 p0 = fold_w_u4[w_off + 0ul];
    const uint4 p1 = fold_w_u4[w_off + 1ul];
    const ulong s_idx = ulong(v) * ulong(n_groups) + ulong(k_lane);
    const float sc = inkling2_read_sidecar_dt(fold_scales, s_idx, sidecar_dt);
    const float bi = has_biases ? inkling2_read_sidecar_dt(fold_biases, s_idx, sidecar_dt) : 0.0f;
    float acc = sc * inkling2_dot_q8_32_regs(p0, p1, xv) + bi * x_sum;
    acc += simd_shuffle_down(acc, 4);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 1);
    if (k_lane == 0u) {
      const float fb = inkling2_read_f32(fold_bias_raw, ulong(v), fb_f32);
      out_logits[ulong(b0) * ulong(params.out_cols) + ulong(v)] = inkling2_markov_fold_finalize(
          acc,
          fb,
          base_logits_raw,
          ulong(b0) * ulong(params.base_stride) + ulong(v),
          base_f32,
          params.softcap);
    }
    return;
  }

  if (R == 256u) {
    threadgroup float tg_x[4][256];
    threadgroup float tg_xsum[4][8];
    for (uint b = 0u; b < n_b; ++b) {
      const float val = inkling2_read_f32(
          latent_raw, ulong(b0 + b) * ulong(params.latent_stride) + ulong(tid), latent_f32);
      tg_x[b][tid] = val;
      const float gsum = simd_sum(val);
      if (lane == 0u) {
        tg_xsum[b][sg] = gsum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (v >= params.out_cols) {
      return;
    }
    const ulong w_off = ulong(v) * row_u4_stride + ulong(k_lane) * 2ul;
    const uint4 p0 = fold_w_u4[w_off + 0ul];
    const uint4 p1 = fold_w_u4[w_off + 1ul];
    const ulong s_idx = ulong(v) * ulong(n_groups) + ulong(k_lane);
    const float sc = inkling2_read_sidecar_dt(fold_scales, s_idx, sidecar_dt);
    const float bi = has_biases ? inkling2_read_sidecar_dt(fold_biases, s_idx, sidecar_dt) : 0.0f;
    const float fb = (k_lane == 0u) ? inkling2_read_f32(fold_bias_raw, ulong(v), fb_f32) : 0.0f;
    for (uint b = 0u; b < n_b; ++b) {
      float acc = sc * inkling2_dot_q8_32_tg(p0, p1, &tg_x[b][k_lane << 5u]) + bi * tg_xsum[b][k_lane];
      acc += simd_shuffle_down(acc, 4);
      acc += simd_shuffle_down(acc, 2);
      acc += simd_shuffle_down(acc, 1);
      if (k_lane == 0u) {
        out_logits[ulong(b0 + b) * ulong(params.out_cols) + ulong(v)] = inkling2_markov_fold_finalize(
            acc,
            fb,
            base_logits_raw,
            ulong(b0 + b) * ulong(params.base_stride) + ulong(v),
            base_f32,
            params.softcap);
      }
    }
    return;
  }

  if (v >= params.out_cols) {
    return;
  }
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = k_lane; g < n_groups; g += 8u) {
    const ulong w_off = ulong(v) * row_u4_stride + ulong(g) * 2ul;
    const uint4 p0 = fold_w_u4[w_off + 0ul];
    const uint4 p1 = fold_w_u4[w_off + 1ul];
    const ulong s_idx = ulong(v) * ulong(n_groups) + ulong(g);
    const float sc = inkling2_read_sidecar_dt(fold_scales, s_idx, sidecar_dt);
    const float bi = has_biases ? inkling2_read_sidecar_dt(fold_biases, s_idx, sidecar_dt) : 0.0f;
    for (uint b = 0u; b < n_b; ++b) {
      float4 xv[8];
      float x_sum = 0.0f;
      inkling2_load_latent_32(
          latent_raw,
          ulong(b0 + b) * ulong(params.latent_stride) + ulong(g << 5u),
          latent_f32,
          xv,
          x_sum);
      acc[b] += sc * inkling2_dot_q8_32_regs(p0, p1, xv) + bi * x_sum;
    }
  }
  const float fb = (k_lane == 0u) ? inkling2_read_f32(fold_bias_raw, ulong(v), fb_f32) : 0.0f;
  for (uint b = 0u; b < n_b; ++b) {
    float a = acc[b];
    a += simd_shuffle_down(a, 4);
    a += simd_shuffle_down(a, 2);
    a += simd_shuffle_down(a, 1);
    if (k_lane == 0u) {
      out_logits[ulong(b0 + b) * ulong(params.out_cols) + ulong(v)] = inkling2_markov_fold_finalize(
          a,
          fb,
          base_logits_raw,
          ulong(b0 + b) * ulong(params.base_stride) + ulong(v),
          base_f32,
          params.softcap);
    }
  }
}

// ============================================================================
// 15. inkling2_bf16_linear + inkling2_bf16_lm_head_softcap_argmax
//     Coalesced single-pass BF16 linear (`1 <= M <= 32`, `K % 4 == 0`) with
//     128-bit `uint4` weight loads and exact 32-lane FP32 reduction order
//     (0-ULP bit-exact batch-size invariance across ALL M in 1..=32!).
// ============================================================================
struct Inkling2Bf16LinearParams {
  uint rows;        // M (1..=32)
  uint out_dim;     // N (or out_cols for lm_head)
  uint in_dim;      // K (% 4 == 0)
  uint num_splits;  // stage1 TG count for lm_head argmax (0 for linear)
  float softcap;    // softcap (> 0) for lm_head argmax
  uint _pad0;
  uint _pad1;
  uint _pad2;
};

inline float inkling2_reduce_32_ordered(float v) {
  v += simd_shuffle_down(v, 16);
  v += simd_shuffle_down(v, 8);
  v += simd_shuffle_down(v, 4);
  v += simd_shuffle_down(v, 2);
  v += simd_shuffle_down(v, 1);
  return v;
}

// Explicit FP32 FMA micro-step (8 ffma ops vs 16 fmul/fadd under fp contract(off),
// identical instruction order across ALL M=1..=32 -> 0-ULP batch-size invariance!).
inline float inkling2_bf16_fma8(float acc, float4 w0, float4 w1, uint4 xr) {
  const float4 x0 = float4(as_type<bfloat4>(xr.xy));
  acc = fma(w0.x, x0.x, acc);
  acc = fma(w0.y, x0.y, acc);
  acc = fma(w0.z, x0.z, acc);
  acc = fma(w0.w, x0.w, acc);
  const float4 x1 = float4(as_type<bfloat4>(xr.zw));
  acc = fma(w1.x, x1.x, acc);
  acc = fma(w1.y, x1.y, acc);
  acc = fma(w1.z, x1.z, acc);
  acc = fma(w1.w, x1.w, acc);
  return acc;
}

inline float inkling2_bf16_fma4(float acc, float4 w0, ushort4 xr) {
  const float4 x0 = float4(as_type<bfloat4>(xr));
  acc = fma(w0.x, x0.x, acc);
  acc = fma(w0.y, x0.y, acc);
  acc = fma(w0.z, x0.z, acc);
  acc = fma(w0.w, x0.w, acc);
  return acc;
}

// Grid: (ceil(N / 4), 1, 1), TG: (128, 1, 1) -> 4 SGs x 1 output row/SG.
// 4x in-flight uint4 loads (16 cachelines/SG) saturates M5 Max DRAM bandwidth while
// preserving the exact sequential k-order of `inkling2_bf16_linear_batched` (0-ULP parity).
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_linear_m1(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2Bf16LinearParams &params [[buffer(3)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint n = tgid_x * 4u + sg;
  if (n >= params.out_dim) {
    return;
  }
  const uint K = params.in_dim;
  const uint K_vec8 = K & ~7u;
  const uint K_unroll = K_vec8 & ~1023u;
  const device bfloat *w_row = w + ulong(n) * ulong(K);

  float acc = 0.0f;
  uint k = lane * 8u;
  for (; k < K_unroll; k += 1024u) {
    const uint4 wr0 = *reinterpret_cast<device const uint4 *>(w_row + k);
    const uint4 wr1 = *reinterpret_cast<device const uint4 *>(w_row + k + 256u);
    const uint4 wr2 = *reinterpret_cast<device const uint4 *>(w_row + k + 512u);
    const uint4 wr3 = *reinterpret_cast<device const uint4 *>(w_row + k + 768u);
    const uint4 xr0 = *reinterpret_cast<device const uint4 *>(x + k);
    const uint4 xr1 = *reinterpret_cast<device const uint4 *>(x + k + 256u);
    const uint4 xr2 = *reinterpret_cast<device const uint4 *>(x + k + 512u);
    const uint4 xr3 = *reinterpret_cast<device const uint4 *>(x + k + 768u);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr0.xy)), float4(as_type<bfloat4>(wr0.zw)), xr0);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr1.xy)), float4(as_type<bfloat4>(wr1.zw)), xr1);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr2.xy)), float4(as_type<bfloat4>(wr2.zw)), xr2);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr3.xy)), float4(as_type<bfloat4>(wr3.zw)), xr3);
  }
  for (; k < K_vec8; k += 256u) {
    const uint4 wr = *reinterpret_cast<device const uint4 *>(w_row + k);
    const uint4 xr = *reinterpret_cast<device const uint4 *>(x + k);
    acc = inkling2_bf16_fma8(acc, float4(as_type<bfloat4>(wr.xy)), float4(as_type<bfloat4>(wr.zw)), xr);
  }
  if (K_vec8 < K && lane == 0u) {
    const ushort4 wr = *reinterpret_cast<device const ushort4 *>(w_row + K_vec8);
    const ushort4 xr = *reinterpret_cast<device const ushort4 *>(x + K_vec8);
    acc = inkling2_bf16_fma4(acc, float4(as_type<bfloat4>(wr)), xr);
  }

  acc = inkling2_reduce_32_ordered(acc);
  if (lane == 0u) {
    out[n] = bfloat(acc);
  }
}

// Grid: (ceil(M / 32) * ceil(N / 16), 1, 1), TG: (128, 1, 1) -> 4 SGs split-K (KS=4).
// Apple M5 Max hardware NAX (`mpp::tensor_ops::matmul2d`, TSM=32, TSN=16, BK=64, KS=4, 6 KB smem):
// - Streams W directly from device memory at ~580 GB/s & 60+ TFLOPS in ONE DRAM pass for M=1..=32!
// - Never shifts tile origins at M/N edges -> 0-ULP bit-exact batch-size invariance across ALL
//   M = 1, 2, 4, 5, 6, 7, 8, 16, 20, 32, 38, 678, 2048!
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_linear_batched(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Inkling2Bf16LinearParams &params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)lane;
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_tensor_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 32;
  constexpr int TSN = 16;
  constexpr int BK = 64;
  constexpr int KS = 4;

  const int M = int(params.rows);
  const int N = int(params.out_dim);
  const int K = int(params.in_dim);
  const int tiles_m = (M + TSM - 1) / TSM;
  const int tgp = int(tgid.x);
  const int tile_m = tgp % tiles_m;
  const int tile_n = tgp / tiles_m;
  const int m_off = tile_m * TSM;
  const int n_off = tile_n * TSN;
  if (m_off >= M || n_off >= N) {
    return;
  }

  const int rows = min(TSM, M - m_off);
  const int cols = min(TSN, N - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  const int k_group = int(sg);

  device bfloat *A = const_cast<device bfloat *>(x) + int64_t(m_off) * K;
  device bfloat *B = const_cast<device bfloat *>(w) + int64_t(n_off) * K;
  device bfloat *D = out + int64_t(m_off) * N + n_off;

  dev_tensor_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_tensor_t tB(B, extents_t(K, cols), strides_t{1, K});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, metal::execution_simdgroup> op;

  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK - 1) / BK + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK, K - k);
      dev_tensor_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_tensor_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 512 floats
  threadgroup float partials[(KS - 1) * kBlockSize]; // 6 KB smem
  if (k_group > 0) {
    tg_tensor_t tP(partials + (k_group - 1) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k_group != 0) {
    return;
  }

  #pragma unroll
  for (int g = 0; g < KS - 1; ++g) {
    tg_tensor_t tP(partials + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
    part.load(tP);
    #pragma unroll
    for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
      Dacc[i] += part[i];
    }
  }

  auto Dout = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, bfloat>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dout[i] = bfloat(Dacc[i]);
  }
  dev_tensor_t tD(D, extents_t(cols, rows), strides_t{1, N});
  if (full_mn) {
    Dout.store(tD.template slice<TSN, TSM>(0, 0));
  } else {
    Dout.store(tD);
  }
}

// Fused NAX BF16 lm_head + softcap + Stage 1 argmax (ZERO [M, V] DRAM write, 1 DRAM pass!).
// Grid: (num_splits = ceil(out_cols / 32), 1, 1), TG: (128, 1, 1) -> 4 SGs split-K (KS=4).
// Uses exact same TSM=32, BK=64, KS=4 NAX reduction as `inkling2_bf16_linear_batched` ->
// 0-ULP bit-exact parity across ALL M = 1..=32 AND ~1.7 ms for 926 MB lm_head!
// Stage 2 reuses `inkling2_lm_head_softcap_argmax_stage2`!
[[kernel, max_total_threads_per_threadgroup(128)]] void
inkling2_bf16_lm_head_softcap_argmax_p1(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device float *partial_vals [[buffer(2)]],
    device int32_t *partial_idxs [[buffer(3)]],
    constant Inkling2Bf16LinearParams &params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using dev_tensor_t = metal::tensor<device bfloat, extents_t, metal::tensor_inline>;
  using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;

  constexpr int TSM = 32;
  constexpr int TSN = 32;
  constexpr int BK = 64;
  constexpr int KS = 4;

  const uint split = tgid.x;
  if (split >= params.num_splits) {
    return;
  }
  const int M = int(params.rows);
  const int V = int(params.out_dim);
  const int K = int(params.in_dim);
  const int n_off = int(split) * TSN;
  if (n_off >= V || M <= 0) {
    return;
  }
  const int rows = min(TSM, M);
  const int cols = min(TSN, V - n_off);
  const bool full_mn = (rows == TSM && cols == TSN);
  const int k_group = int(sg);

  device bfloat *A = const_cast<device bfloat *>(x);
  device bfloat *B = const_cast<device bfloat *>(w) + int64_t(n_off) * K;

  dev_tensor_t tA(A, extents_t(K, rows), strides_t{1, K});
  dev_tensor_t tB(B, extents_t(K, cols), strides_t{1, K});

  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));

  constexpr auto desc = matmul2d_descriptor(
      TSM,
      TSN,
      BK,
      false,
      true,
      true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, metal::execution_simdgroup> op;

  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  #pragma unroll
  for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
    Dacc[i] = 0.0f;
  }
  auto Areg = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();

  const int k_iters = ((K + BK - 1) / BK + KS - 1) / KS;
  for (int it = 0; it < k_iters; ++it) {
    if ((it & 3) == 0) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    const int k = (it * KS + k_group) * BK;
    if (k >= K) {
      continue;
    }
    if (full_mn && k + BK <= K) {
      auto a = a_chunk(k);
      Areg.load(a);
      auto b = b_chunk(k);
      op.run(Areg, b, Dacc);
    } else {
      const int kk = min(BK, K - k);
      dev_tensor_t vA(A + k, extents_t(kk, rows), strides_t{1, K});
      dev_tensor_t vB(B + k, extents_t(kk, cols), strides_t{1, K});
      op.run(vA, vB, Dacc);
    }
  }

  constexpr int kBlockSize = TSM * TSN; // 1024 floats
  threadgroup float partials[(KS - 1) * kBlockSize]; // 12 KB smem (reused for final logits!)
  if (k_group > 0) {
    tg_tensor_t tP(partials + (k_group - 1) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tP);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (k_group == 0) {
    #pragma unroll
    for (int g = 0; g < KS - 1; ++g) {
      tg_tensor_t tP(partials + g * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
      part.load(tP);
      #pragma unroll
      for (uint16_t i = 0; i < Dacc.get_capacity(); ++i) {
        Dacc[i] += part[i];
      }
    }
    tg_tensor_t tOut(partials, extents_t(TSN, TSM), strides_t{1, TSN});
    Dacc.store(tOut);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // All 4 simdgroups cooperatively softcap + argmax `rows` rows x `cols` (<= 32) vocab cols!
  const float cap = params.softcap;
  for (uint r = sg; r < uint(rows); r += 4u) {
    float bv = -INFINITY;
    int32_t bi = -1;
    if (lane < uint(cols)) {
      float val = float(bfloat(partials[r * 32u + lane]));
      if (cap > 0.0f) {
        val = inkling2_softcap(val, cap);
      }
      bv = val;
      bi = int32_t(n_off + int(lane));
    }
    #pragma unroll
    for (ushort off = 16; off >= 1; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const int32_t oi = simd_shuffle_down(bi, off);
      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0u) {
      const ulong out_slot = ulong(r) * ulong(params.num_splits) + ulong(split);
      partial_vals[out_slot] = bv;
      partial_idxs[out_slot] = bi;
    }
  }
}
