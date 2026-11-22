// Grouped & Fused NVFP4 / Q8_0 MoE Metal kernels for Inkling-2-Mini on Apple Silicon
// (M5 Max NAX tensor ops + coalesced SIMD multi-row QMV + zero-gather M=1 fused kernels).
// Compiled into paged_attn.metallib with -fno-fast-math.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include "inkling2_moe_abi.h"

using namespace metal;
using namespace mpp::tensor_ops;

// Exact FP4 E2M1 lookup table (matches mlx/backend/metal/kernels/fp4.h bit-for-bit).
constant float INKLING2_FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Exact FP8 E4M3FN decode (matches mlx/backend/metal/kernels/fp8.h bit-for-bit).
inline float inkling2_decode_fp8_e4m3(uint8_t bits) {
  const uint16_t v = bits & 127u;
  const uint16_t sign_bit = uint16_t((bits >> 7u) & 1u) << 15u;
  const uint16_t u = (v << 7u) | (((v + 1u) >> 7u) << 14u) | sign_bit;
  return float(as_type<half>(u) * 256.0h);
}

inline float inkling2_read_sidecar(device const uchar *ptr, ulong idx, bool is_f32) {
  if (is_f32) {
    return reinterpret_cast<device const float *>(ptr)[idx];
  }
  return float(reinterpret_cast<device const bfloat *>(ptr)[idx]);
}

inline float inkling2_moe_sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

inline float inkling2_moe_softcap(float x, float cap) {
  if (cap <= 0.0f) {
    return x;
  }
  return cap * tanh(clamp(x / cap, -20.0f, 20.0f));
}

inline bfloat inkling2_moe_situ_epilogue(
    float gate_v,
    float up_v,
    float outer_scale,
    float gate_cap,
    float up_cap,
    bool round_bf16) {
  if (round_bf16) {
    gate_v = float(bfloat(gate_v));
    up_v = float(bfloat(up_v));
  }
  gate_v = gate_v * outer_scale;
  up_v = up_v * outer_scale;
  const float g_sc = inkling2_moe_softcap(gate_v, gate_cap);
  const float u_sc = inkling2_moe_softcap(up_v, up_cap);
  const float g_sig = inkling2_moe_sigmoid(gate_v);
  return bfloat(g_sc * g_sig * u_sc);
}

// Load 16 BF16 activations into 4 float4 registers using 2 x 128-bit uint4 loads.
inline void load_bf16_16(device const bfloat *src, thread float4 (&x)[4]) {
  const device uint4 *p = reinterpret_cast<device const uint4 *>(src);
  const uint4 a = p[0];
  const uint4 b = p[1];
  x[0] = float4(as_type<bfloat4>(a.xy));
  x[1] = float4(as_type<bfloat4>(a.zw));
  x[2] = float4(as_type<bfloat4>(b.xy));
  x[3] = float4(as_type<bfloat4>(b.zw));
}

// Load 16 NVFP4 weights (8 bytes = uint2) into 4 float4 vectors.
inline void decode_nvfp4_16(uint2 packed, thread float4 (&w)[4]) {
  const uint w0 = packed.x;
  const uint w1 = packed.y;
  w[0] = float4(
      INKLING2_FP4_LUT[w0 & 15u],
      INKLING2_FP4_LUT[(w0 >> 4u) & 15u],
      INKLING2_FP4_LUT[(w0 >> 8u) & 15u],
      INKLING2_FP4_LUT[(w0 >> 12u) & 15u]);
  w[1] = float4(
      INKLING2_FP4_LUT[(w0 >> 16u) & 15u],
      INKLING2_FP4_LUT[(w0 >> 20u) & 15u],
      INKLING2_FP4_LUT[(w0 >> 24u) & 15u],
      INKLING2_FP4_LUT[(w0 >> 28u) & 15u]);
  w[2] = float4(
      INKLING2_FP4_LUT[w1 & 15u],
      INKLING2_FP4_LUT[(w1 >> 4u) & 15u],
      INKLING2_FP4_LUT[(w1 >> 8u) & 15u],
      INKLING2_FP4_LUT[(w1 >> 12u) & 15u]);
  w[3] = float4(
      INKLING2_FP4_LUT[(w1 >> 16u) & 15u],
      INKLING2_FP4_LUT[(w1 >> 20u) & 15u],
      INKLING2_FP4_LUT[(w1 >> 24u) & 15u],
      INKLING2_FP4_LUT[(w1 >> 28u) & 15u]);
}

inline float dot_4x4(const thread float4 (&w)[4], const thread float4 (&x)[4]) {
  return (dot(w[0], x[0]) + dot(w[1], x[1])) + (dot(w[2], x[2]) + dot(w[3], x[3]));
}

// Dot 32 uint8 Q8_0 weights against 8 float4 activations, returning raw uint8*x dot product.
inline float dot_q8_32(const device uint4 *w_u4, const thread float4 (&x)[8]) {
  const uint4 p0 = w_u4[0];
  const uint4 p1 = w_u4[1];
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

template <int KLanes>
inline float reduce_k_lanes(float v) {
  if constexpr (KLanes >= 32) v += simd_shuffle_down(v, 16);
  if constexpr (KLanes >= 16) v += simd_shuffle_down(v, 8);
  if constexpr (KLanes >= 8)  v += simd_shuffle_down(v, 4);
  if constexpr (KLanes >= 4)  v += simd_shuffle_down(v, 2);
  if constexpr (KLanes >= 2)  v += simd_shuffle_down(v, 1);
  return v;
}

// ============================================================================
// 1. M=1 Zero-Gather Fused Kernels (NVFP4 & Q8_0)
// ============================================================================

// Grid: (ceil(I / 4), top_k, 1), TG: (64, 1, 1) -> 2 SGs x 2 cols/SG = 4 cols/TG.
// k_lanes = 16 -> 16 consecutive lanes read 16 * 8 = 128 bytes (1 full cacheline) per load!
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_gate_up_situ_nvfp4(
    device const bfloat *x [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const uchar *gu_w [[buffer(2)]],
    device const uint8_t *gu_s [[buffer(3)]],
    device const float *gu_outer [[buffer(4)]],
    device bfloat *act_out [[buffer(5)]],
    constant Inkling2MoeParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 16u;
  const uint slot = group.y;
  if (slot >= p.top_k) return;

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 4u; // 0..1
  const uint col = group.x * 4u + sg * 2u + r_id;
  if (col >= p.intermediate) return;

  const uint expert = uint(indices[slot]);
  const uint K = p.hidden;
  const uint N2 = p.intermediate * 2u;
  const uint n_groups = K >> 4u;
  const ulong row_w_stride = ulong(K >> 1u);
  const ulong row_s_stride = ulong(n_groups);
  const ulong exp_row_base = ulong(expert) * ulong(N2);

  const device uchar *gw_row = gu_w + (exp_row_base + col) * row_w_stride;
  const device uint8_t *gs_row = gu_s + (exp_row_base + col) * row_s_stride;
  const device uchar *uw_row = gu_w + (exp_row_base + p.intermediate + col) * row_w_stride;
  const device uint8_t *us_row = gu_s + (exp_row_base + p.intermediate + col) * row_s_stride;

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    float4 xv[4];
    load_bf16_16(x + (g << 4u), xv);

    float4 wv[4];
    const uint2 gp = *reinterpret_cast<device const uint2 *>(gw_row + ulong(g) * 8ul);
    decode_nvfp4_16(gp, wv);
    acc_g += inkling2_decode_fp8_e4m3(gs_row[g]) * dot_4x4(wv, xv);

    const uint2 up = *reinterpret_cast<device const uint2 *>(uw_row + ulong(g) * 8ul);
    decode_nvfp4_16(up, wv);
    acc_u += inkling2_decode_fp8_e4m3(us_row[g]) * dot_4x4(wv, xv);
  }

  acc_g = reduce_k_lanes<16>(acc_g);
  acc_u = reduce_k_lanes<16>(acc_u);
  if (k_lane == 0u) {
    const float outer_s = (p.flags & 2u) ? gu_outer[expert] : 1.0f;
    const bool round_bf16 = (p.flags & 8u) != 0u;
    act_out[ulong(slot) * p.intermediate + col] = inkling2_moe_situ_epilogue(
        acc_g, acc_u, outer_s, p.gate_softcap, p.up_softcap, round_bf16);
  }
}

// Grid: (ceil(I / 8), top_k, 1), TG: (64, 1, 1) -> 2 SGs x 4 cols/SG = 8 cols/TG.
// k_lanes = 8 -> 8 consecutive lanes read 8 * 32 = 256 bytes (2 full cachelines) per load!
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_gate_up_situ_q8(
    device const bfloat *x [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const uchar *gu_w [[buffer(2)]],
    device const uchar *gu_s [[buffer(3)]],
    device const uchar *gu_b [[buffer(4)]],
    device bfloat *act_out [[buffer(5)]],
    constant Inkling2MoeParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 8u;
  const uint slot = group.y;
  if (slot >= p.top_k) return;

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group.x * 8u + sg * 4u + r_id;
  if (col >= p.intermediate) return;

  const uint expert = uint(indices[slot]);
  const uint K = p.hidden;
  const uint N2 = p.intermediate * 2u;
  const uint n_groups = K >> 5u;
  const bool sidecar_f32 = (p.flags & 4u) != 0u;
  const ulong exp_row_base = ulong(expert) * ulong(N2);

  const device uchar *gw_row = gu_w + (exp_row_base + col) * ulong(K);
  const ulong gs_base = (exp_row_base + col) * ulong(n_groups);
  const device uchar *uw_row = gu_w + (exp_row_base + p.intermediate + col) * ulong(K);
  const ulong us_base = (exp_row_base + p.intermediate + col) * ulong(n_groups);

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    float4 xv[8];
    load_bf16_16(x + (g << 5u), *reinterpret_cast<thread float4 (*)[4]>(&xv[0]));
    load_bf16_16(x + (g << 5u) + 16u, *reinterpret_cast<thread float4 (*)[4]>(&xv[4]));
    float4 s4 = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
    const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);

    const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
    const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
    const float gd = dot_q8_32(reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul), xv);
    acc_g += gs * gd + gb * x_sum;

    const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
    const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
    const float ud = dot_q8_32(reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul), xv);
    acc_u += us * ud + ub * x_sum;
  }

  acc_g = reduce_k_lanes<8>(acc_g);
  acc_u = reduce_k_lanes<8>(acc_u);
  if (k_lane == 0u) {
    const bool round_bf16 = (p.flags & 8u) != 0u;
    act_out[ulong(slot) * p.intermediate + col] = inkling2_moe_situ_epilogue(
        acc_g, acc_u, 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
  }
}

// Grid: (ceil(D / 4), 1, 1), TG: (64, 1, 1).
// Both SGs compute the same 4 output cols across interleaved top_k slots (sg=0: even, sg=1: odd),
// then sg=0 sums all slots in exact 0..top_k-1 order!
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_down_combine_nvfp4(
    device const bfloat *act [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const uchar *dn_w [[buffer(3)]],
    device const uint8_t *dn_s [[buffer(4)]],
    device bfloat *out [[buffer(5)]],
    constant Inkling2MoeParams &p [[buffer(6)]],
    uint group_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 8u;
  threadgroup float slot_vals[16][4];

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group_x * 4u + r_id;
  const bool valid_col = col < p.hidden;

  const uint K = p.intermediate;
  const uint n_groups = K >> 4u;
  const ulong row_w_stride = ulong(K >> 1u);
  const ulong row_s_stride = ulong(n_groups);

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(p.hidden) + col;
      const device uchar *w_row = dn_w + row_idx * row_w_stride;
      const device uint8_t *s_row = dn_s + row_idx * row_s_stride;
      const device bfloat *a_row = act + ulong(slot) * K;
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        float4 xv[4];
        load_bf16_16(a_row + (g << 4u), xv);
        float4 wv[4];
        const uint2 wp = *reinterpret_cast<device const uint2 *>(w_row + ulong(g) * 8ul);
        decode_nvfp4_16(wp, wv);
        acc += inkling2_decode_fp8_e4m3(s_row[g]) * dot_4x4(wv, xv);
      }
    }
    acc = reduce_k_lanes<8>(acc);
    if (k_lane == 0u) {
      // Preserve exact BF16 boundary of down_proj before gamma-weighted FP32 combine!
      slot_vals[slot][r_id] = float(bfloat(acc));
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && k_lane == 0u && valid_col) {
    float sum = 0.0f;
    for (uint slot = 0u; slot < p.top_k; ++slot) {
      sum += gamma[slot] * slot_vals[slot][r_id];
    }
    out[col] = bfloat(sum);
  }
}

// Grid: (ceil(D / 8), 1, 1), TG: (64, 1, 1).
// k_lanes = 4 (4 * 32 = 128B full cacheline!), 8 output cols per TG, 2 SGs split top_k slots.
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_down_combine_q8(
    device const bfloat *act [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const uchar *dn_w [[buffer(3)]],
    device const uchar *dn_s [[buffer(4)]],
    device const uchar *dn_b [[buffer(5)]],
    device bfloat *out [[buffer(6)]],
    constant Inkling2MoeParams &p [[buffer(7)]],
    uint group_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 4u;
  threadgroup float slot_vals[16][8];

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 2u; // 0..7
  const uint col = group_x * 8u + r_id;
  const bool valid_col = col < p.hidden;

  const uint K = p.intermediate;
  const uint n_groups = K >> 5u;
  const bool sidecar_f32 = (p.flags & 4u) != 0u;

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(p.hidden) + col;
      const device uchar *w_row = dn_w + row_idx * ulong(K);
      const ulong s_base = row_idx * ulong(n_groups);
      const device bfloat *a_row = act + ulong(slot) * K;
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        float4 xv[8];
        load_bf16_16(a_row + (g << 5u), *reinterpret_cast<thread float4 (*)[4]>(&xv[0]));
        load_bf16_16(a_row + (g << 5u) + 16u, *reinterpret_cast<thread float4 (*)[4]>(&xv[4]));
        float4 s4 = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
        const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);

        const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
        const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
        const float dp = dot_q8_32(reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul), xv);
        acc += sc * dp + bi * x_sum;
      }
    }
    acc = reduce_k_lanes<4>(acc);
    if (k_lane == 0u) {
      slot_vals[slot][r_id] = float(bfloat(acc));
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && k_lane == 0u && valid_col) {
    float sum = 0.0f;
    for (uint slot = 0u; slot < p.top_k; ++slot) {
      sum += gamma[slot] * slot_vals[slot][r_id];
    }
    out[col] = bfloat(sum);
  }
}

// ============================================================================
// 2. Grouped Indirect-Dispatch Infrastructure (T >= 2: M=4, 8, 32, Prefill)
// ============================================================================

inline uint inkling2_matmul_rows(uint rows, uint tile_rows) {
  return min(tile_rows, rows <= 8u ? 8u : rows <= 16u ? 16u : tile_rows);
}

// Single-threadgroup GPU histogram + prefix sum + tile table + route scatter + dual-bucket
// (0-smem SIMD vs 4-SG NAX) indirect dispatch grids. Zero host sync!
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_moe_group(
    device const int32_t *selected [[buffer(0)]],
    device Inkling2MoeTile *tiles [[buffer(1)]],
    device Inkling2MoeTileCount *tile_count [[buffer(2)]],
    device uint *routes [[buffer(3)]],
    device uint *inverse [[buffer(4)]],
    device uint *tile_lists [[buffer(5)]],
    constant Inkling2MoeParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[256];
  threadgroup atomic_uint cursors[256];
  threadgroup uint offsets[256];
  threadgroup uint totals[8];
  threadgroup uint16_t tg_sel[512];
  threadgroup atomic_uint simd_cnt;
  threadgroup atomic_uint nax_cnt;

  atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
  atomic_store_explicit(cursors + tid, 0u, memory_order_relaxed);
  if (tid == 0u) {
    atomic_store_explicit(&simd_cnt, 0u, memory_order_relaxed);
    atomic_store_explicit(&nax_cnt, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total_routes = p.rows * p.top_k;
  const bool deterministic_scatter = total_routes <= 512u;
  for (uint r = tid; r < total_routes; r += 256u) {
    const uint e = uint(selected[r]);
    if (deterministic_scatter) {
      tg_sel[r] = uint16_t(e);
    }
    if (e < p.experts) {
      atomic_fetch_add_explicit(counts + e, 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint count = tid < p.experts
      ? atomic_load_explicit(counts + tid, memory_order_relaxed)
      : 0u;
  const uint nt = (count + p.tile_rows - 1u) / p.tile_rows;
  uint start = simd_prefix_exclusive_sum(nt);
  if (lane == 31u) {
    totals[sg] = start + nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0u; i < sg; ++i) {
    start += totals[i];
  }
  offsets[tid] = start;

  // Decide SIMD vs NAX per expert (not per tile) so a multi-tile expert never splits
  // across SIMD and NAX kernels.
  const bool use_simd = (count <= p.col_cutoff);
  for (uint t = 0u; t < nt; ++t) {
    const uint tile_idx = start + t;
    const uint live = min(p.tile_rows, count - t * p.tile_rows);
    tiles[tile_idx] = {tid, live};
    if (use_simd) {
      const uint pos = atomic_fetch_add_explicit(&simd_cnt, 1u, memory_order_relaxed);
      tile_lists[pos] = tile_idx;
    } else {
      const uint pos = atomic_fetch_add_explicit(&nax_cnt, 1u, memory_order_relaxed);
      tile_lists[p.max_tiles + pos] = tile_idx;
    }
  }
  if (nt != 0u) {
    const uint last = start + nt - 1u;
    const uint live = count - (nt - 1u) * p.tile_rows;
    const uint pad_end = last * p.tile_rows + inkling2_matmul_rows(live, p.tile_rows);
    for (uint r = start * p.tile_rows + count; r < pad_end; ++r) {
      routes[r] = ~0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (deterministic_scatter) {
    // 100% deterministic in-order route scatter per expert (zero atomics, ~0.002 ms).
    if (tid < p.experts && count > 0u) {
      uint slot = 0u;
      const uint base_pos = start * p.tile_rows;
      for (uint r = 0u; r < total_routes; ++r) {
        if (uint(tg_sel[r]) == tid) {
          const uint pos = base_pos + slot;
          routes[pos] = r;
          inverse[r] = pos;
          ++slot;
        }
      }
    }
  } else {
    for (uint r = tid; r < total_routes; r += 256u) {
      const uint e = uint(selected[r]);
      if (e < p.experts) {
        const uint slot = atomic_fetch_add_explicit(cursors + e, 1u, memory_order_relaxed);
        const uint pos = offsets[e] * p.tile_rows + slot;
        routes[pos] = r;
        inverse[r] = pos;
      } else {
        inverse[r] = ~0u;
      }
    }
  }

  if (tid == 255u) {
    const uint live_tiles = start + nt;
    const uint s_tiles = atomic_load_explicit(&simd_cnt, memory_order_relaxed);
    const uint n_tiles = atomic_load_explicit(&nax_cnt, memory_order_relaxed);
    const bool is_q8 = (p.flags & 1u) != 0u;

    tile_count->total_tiles = live_tiles;
    tile_count->simd_tiles = s_tiles;
    tile_count->nax_tiles = n_tiles;
    tile_count->_pad0 = 0u;

    tile_count->gather_grid[0] = live_tiles;
    tile_count->gather_grid[1] = (p.hidden + 255u) / 256u;
    tile_count->gather_grid[2] = 1u;

    const uint simd_gu_x = is_q8 ? (p.intermediate / 8u) : (p.intermediate / 4u);
    const uint simd_dn_x = is_q8 ? (p.hidden / 16u) : (p.hidden / 8u);
    tile_count->simd_gu_grid[0] = s_tiles > 0u ? simd_gu_x : 0u;
    tile_count->simd_gu_grid[1] = s_tiles;
    tile_count->simd_gu_grid[2] = 1u;

    tile_count->simd_dn_grid[0] = s_tiles > 0u ? simd_dn_x : 0u;
    tile_count->simd_dn_grid[1] = s_tiles;
    tile_count->simd_dn_grid[2] = 1u;

    // 2-SG NAX (8 KB smem for max occupancy): gate_up computes 32 cols/TG, down computes 64 cols/TG.
    tile_count->nax_gu_grid[0] = n_tiles > 0u ? (p.intermediate / 32u) : 0u;
    tile_count->nax_gu_grid[1] = n_tiles;
    tile_count->nax_gu_grid[2] = 1u;

    tile_count->nax_dn_grid[0] = n_tiles > 0u ? (p.hidden / 64u) : 0u;
    tile_count->nax_dn_grid[1] = n_tiles;
    tile_count->nax_dn_grid[2] = 1u;
  }
}

// Indirectly dispatched 128-bit vectorized gather into grouped tile layout.
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_moe_gather(
    device const bfloat *x [[buffer(0)]],
    device const uint *routes [[buffer(1)]],
    device const Inkling2MoeTile *tiles [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device bfloat *out [[buffer(4)]],
    constant Inkling2MoeParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (group.x >= count->total_tiles) return;
  const uint live = tiles[group.x].rows;
  // Always zero-pad up to the NAX tile height so multi-tile tail tiles routed to NAX see clean zeros.
  const uint rows = inkling2_matmul_rows(live, p.tile_rows);
  const uint col = group.y * 256u + (tid & 31u) * 8u;
  if (col >= p.hidden) return;
  for (uint i = tid >> 5u; i < rows; i += 8u) {
    const uint dst_row = group.x * p.tile_rows + i;
    const uint route = routes[dst_row];
    *reinterpret_cast<device uint4 *>(out + ulong(dst_row) * p.hidden + col) =
        route == ~0u
            ? uint4(0u)
            : *reinterpret_cast<device const uint4 *>(
                  x + ulong(route / p.top_k) * p.hidden + col);
  }
}

// ============================================================================
// 3. Zero-Smem 1-Pass Coalesced SIMD Multi-Row Kernels (1..8 rows/expert)
//    Exact same FP32 reduction order as M=1 -> 490+ GB/s AND bit-exact M=1..8!
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_gate_up_situ_nvfp4(
    device const bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uchar *gu_w [[buffer(4)]],
    device const uint8_t *gu_s [[buffer(5)]],
    device const float *gu_outer [[buffer(6)]],
    device bfloat *hidden [[buffer(7)]],
    constant Inkling2MoeParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  constexpr uint kLanes = 16u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 4u; // 0..1
  const uint col = group.x * 4u + sg * 2u + r_id;
  if (col >= p.intermediate) return;

  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint N2 = I * 2u;
  const uint n_groups = K >> 4u;
  const ulong row_w_stride = ulong(K >> 1u);
  const ulong row_s_stride = ulong(n_groups);
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);

  const device uchar *gw_row = gu_w + (exp_row_base + col) * row_w_stride;
  const device uint8_t *gs_row = gu_s + (exp_row_base + col) * row_s_stride;
  const device uchar *uw_row = gu_w + (exp_row_base + I + col) * row_w_stride;
  const device uint8_t *us_row = gu_s + (exp_row_base + I + col) * row_s_stride;

  const device bfloat *x_tile = x + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * I;
  const float outer_s = (p.flags & 2u) ? gu_outer[t.expert] : 1.0f;
  const bool round_bf16 = (p.flags & 8u) != 0u;

  if (n_rows == 1u) {
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      float4 xv[4];
      load_bf16_16(x_tile + (g << 4u), xv);

      float4 wv[4];
      const uint2 gp = *reinterpret_cast<device const uint2 *>(gw_row + ulong(g) * 8ul);
      decode_nvfp4_16(gp, wv);
      acc_g += inkling2_decode_fp8_e4m3(gs_row[g]) * dot_4x4(wv, xv);

      const uint2 up = *reinterpret_cast<device const uint2 *>(uw_row + ulong(g) * 8ul);
      decode_nvfp4_16(up, wv);
      acc_u += inkling2_decode_fp8_e4m3(us_row[g]) * dot_4x4(wv, xv);
    }
    acc_g = reduce_k_lanes<16>(acc_g);
    acc_u = reduce_k_lanes<16>(acc_u);
    if (k_lane == 0u) {
      hidden[out_tile_base + col] = inkling2_moe_situ_epilogue(
          acc_g, acc_u, outer_s, p.gate_softcap, p.up_softcap, round_bf16);
    }
    return;
  }

  if (n_rows <= 4u) {
    float acc_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const uint2 gp = *reinterpret_cast<device const uint2 *>(gw_row + ulong(g) * 8ul);
      const float sg_val = inkling2_decode_fp8_e4m3(gs_row[g]);
      const uint2 up = *reinterpret_cast<device const uint2 *>(uw_row + ulong(g) * 8ul);
      const float su_val = inkling2_decode_fp8_e4m3(us_row[g]);
      float4 w_g[4], w_u[4];
      decode_nvfp4_16(gp, w_g);
      decode_nvfp4_16(up, w_u);

      const uint k0 = g << 4u;
      for (uint r = 0u; r < n_rows; ++r) {
        float4 xv[4];
        load_bf16_16(x_tile + ulong(r) * K + k0, xv);
        acc_g[r] += sg_val * dot_4x4(w_g, xv);
        acc_u[r] += su_val * dot_4x4(w_u, xv);
      }
    }
    for (uint r = 0u; r < n_rows; ++r) {
      acc_g[r] = reduce_k_lanes<16>(acc_g[r]);
      acc_u[r] = reduce_k_lanes<16>(acc_u[r]);
      if (k_lane == 0u) {
        hidden[out_tile_base + ulong(r) * I + col] = inkling2_moe_situ_epilogue(
            acc_g[r], acc_u[r], outer_s, p.gate_softcap, p.up_softcap, round_bf16);
      }
    }
    return;
  }

  float acc_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float acc_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const uint k0 = g << 4u;
    float4 wv[4];
    const uint2 gp = *reinterpret_cast<device const uint2 *>(gw_row + ulong(g) * 8ul);
    const float sg_val = inkling2_decode_fp8_e4m3(gs_row[g]);
    decode_nvfp4_16(gp, wv);
    for (uint r = 0u; r < n_rows; ++r) {
      float4 xv[4];
      load_bf16_16(x_tile + ulong(r) * K + k0, xv);
      acc_g[r] += sg_val * dot_4x4(wv, xv);
    }

    const uint2 up = *reinterpret_cast<device const uint2 *>(uw_row + ulong(g) * 8ul);
    const float su_val = inkling2_decode_fp8_e4m3(us_row[g]);
    decode_nvfp4_16(up, wv);
    for (uint r = 0u; r < n_rows; ++r) {
      float4 xv[4];
      load_bf16_16(x_tile + ulong(r) * K + k0, xv);
      acc_u[r] += su_val * dot_4x4(wv, xv);
    }
  }
  for (uint r = 0u; r < n_rows; ++r) {
    acc_g[r] = reduce_k_lanes<16>(acc_g[r]);
    acc_u[r] = reduce_k_lanes<16>(acc_u[r]);
    if (k_lane == 0u) {
      hidden[out_tile_base + ulong(r) * I + col] = inkling2_moe_situ_epilogue(
          acc_g[r], acc_u[r], outer_s, p.gate_softcap, p.up_softcap, round_bf16);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_gate_up_situ_q8(
    device const bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uchar *gu_w [[buffer(4)]],
    device const uchar *gu_s [[buffer(5)]],
    device const uchar *gu_b [[buffer(6)]],
    device bfloat *hidden [[buffer(7)]],
    constant Inkling2MoeParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  constexpr uint kLanes = 8u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group.x * 8u + sg * 4u + r_id;
  if (col >= p.intermediate) return;

  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint N2 = I * 2u;
  const uint n_groups = K >> 5u;
  const bool sidecar_f32 = (p.flags & 4u) != 0u;
  const bool round_bf16 = (p.flags & 8u) != 0u;
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);

  const device uchar *gw_row = gu_w + (exp_row_base + col) * ulong(K);
  const ulong gs_base = (exp_row_base + col) * ulong(n_groups);
  const device uchar *uw_row = gu_w + (exp_row_base + I + col) * ulong(K);
  const ulong us_base = (exp_row_base + I + col) * ulong(n_groups);

  const device bfloat *x_tile = x + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * I;

  if (n_rows <= 4u) {
    float acc_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
      const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
      const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
      const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
      const device uint4 *gw_ptr = reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul);
      const device uint4 *uw_ptr = reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul);
      const uint k0 = g << 5u;
      for (uint r = 0u; r < n_rows; ++r) {
        float4 xv[8];
        load_bf16_16(x_tile + ulong(r) * K + k0, *reinterpret_cast<thread float4 (*)[4]>(&xv[0]));
        load_bf16_16(x_tile + ulong(r) * K + k0 + 16u, *reinterpret_cast<thread float4 (*)[4]>(&xv[4]));
        float4 s4 = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
        const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
        acc_g[r] += gs * dot_q8_32(gw_ptr, xv) + gb * x_sum;
        acc_u[r] += us * dot_q8_32(uw_ptr, xv) + ub * x_sum;
      }
    }
    for (uint r = 0u; r < n_rows; ++r) {
      acc_g[r] = reduce_k_lanes<8>(acc_g[r]);
      acc_u[r] = reduce_k_lanes<8>(acc_u[r]);
      if (k_lane == 0u) {
        hidden[out_tile_base + ulong(r) * I + col] = inkling2_moe_situ_epilogue(
            acc_g[r], acc_u[r], 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
      }
    }
    return;
  }

  float acc_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float acc_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
    const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
    const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
    const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
    const device uint4 *gw_ptr = reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul);
    const device uint4 *uw_ptr = reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul);
    const uint k0 = g << 5u;
    for (uint r = 0u; r < n_rows; ++r) {
      float4 xv[8];
      load_bf16_16(x_tile + ulong(r) * K + k0, *reinterpret_cast<thread float4 (*)[4]>(&xv[0]));
      load_bf16_16(x_tile + ulong(r) * K + k0 + 16u, *reinterpret_cast<thread float4 (*)[4]>(&xv[4]));
      float4 s4 = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
      const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
      acc_g[r] += gs * dot_q8_32(gw_ptr, xv) + gb * x_sum;
      acc_u[r] += us * dot_q8_32(uw_ptr, xv) + ub * x_sum;
    }
  }
  for (uint r = 0u; r < n_rows; ++r) {
    acc_g[r] = reduce_k_lanes<8>(acc_g[r]);
    acc_u[r] = reduce_k_lanes<8>(acc_u[r]);
    if (k_lane == 0u) {
      hidden[out_tile_base + ulong(r) * I + col] = inkling2_moe_situ_epilogue(
          acc_g[r], acc_u[r], 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_down_nvfp4(
    device const bfloat *hidden [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uchar *dn_w [[buffer(4)]],
    device const uint8_t *dn_s [[buffer(5)]],
    device bfloat *projected [[buffer(6)]],
    constant Inkling2MoeParams &p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  constexpr uint kLanes = 8u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group.x * 8u + sg * 4u + r_id;
  if (col >= p.hidden) return;

  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.intermediate;
  const uint D = p.hidden;
  const uint n_groups = K >> 4u;
  const ulong row_idx = ulong(t.expert) * ulong(D) + col;
  const device uchar *w_row = dn_w + row_idx * ulong(K >> 1u);
  const device uint8_t *s_row = dn_s + row_idx * ulong(n_groups);

  const device bfloat *x_tile = hidden + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * D;

  if (n_rows == 1u) {
    float acc = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      float4 xv[4], wv[4];
      load_bf16_16(x_tile + (g << 4u), xv);
      const uint2 wp = *reinterpret_cast<device const uint2 *>(w_row + ulong(g) * 8ul);
      decode_nvfp4_16(wp, wv);
      acc += inkling2_decode_fp8_e4m3(s_row[g]) * dot_4x4(wv, xv);
    }
    acc = reduce_k_lanes<8>(acc);
    if (k_lane == 0u) {
      projected[out_tile_base + col] = bfloat(acc);
    }
    return;
  }

  if (n_rows <= 4u) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      float4 wv[4];
      const uint2 wp = *reinterpret_cast<device const uint2 *>(w_row + ulong(g) * 8ul);
      decode_nvfp4_16(wp, wv);
      const float sv = inkling2_decode_fp8_e4m3(s_row[g]);
      const uint k0 = g << 4u;
      for (uint r = 0u; r < n_rows; ++r) {
        float4 xv[4];
        load_bf16_16(x_tile + ulong(r) * K + k0, xv);
        acc[r] += sv * dot_4x4(wv, xv);
      }
    }
    for (uint r = 0u; r < n_rows; ++r) {
      acc[r] = reduce_k_lanes<8>(acc[r]);
      if (k_lane == 0u) {
        projected[out_tile_base + ulong(r) * D + col] = bfloat(acc[r]);
      }
    }
    return;
  }

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    float4 wv[4];
    const uint2 wp = *reinterpret_cast<device const uint2 *>(w_row + ulong(g) * 8ul);
    decode_nvfp4_16(wp, wv);
    const float sv = inkling2_decode_fp8_e4m3(s_row[g]);
    const uint k0 = g << 4u;
    for (uint r = 0u; r < n_rows; ++r) {
      float4 xv[4];
      load_bf16_16(x_tile + ulong(r) * K + k0, xv);
      acc[r] += sv * dot_4x4(wv, xv);
    }
  }
  for (uint r = 0u; r < n_rows; ++r) {
    acc[r] = reduce_k_lanes<8>(acc[r]);
    if (k_lane == 0u) {
      projected[out_tile_base + ulong(r) * D + col] = bfloat(acc[r]);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_down_q8(
    device const bfloat *hidden [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uchar *dn_w [[buffer(4)]],
    device const uchar *dn_s [[buffer(5)]],
    device const uchar *dn_b [[buffer(6)]],
    device bfloat *projected [[buffer(7)]],
    constant Inkling2MoeParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  constexpr uint kLanes = 4u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 2u; // 0..7
  const uint col = group.x * 16u + sg * 8u + r_id;
  if (col >= p.hidden) return;

  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.intermediate;
  const uint D = p.hidden;
  const uint n_groups = K >> 5u;
  const bool sidecar_f32 = (p.flags & 4u) != 0u;
  const ulong row_idx = ulong(t.expert) * ulong(D) + col;
  const device uchar *w_row = dn_w + row_idx * ulong(K);
  const ulong s_base = row_idx * ulong(n_groups);

  const device bfloat *x_tile = hidden + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * D;

  if (n_rows <= 4u) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
      const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
      const device uint4 *w_ptr = reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul);
      const uint k0 = g << 5u;
      for (uint r = 0u; r < n_rows; ++r) {
        float4 xv[8];
        load_bf16_16(x_tile + ulong(r) * K + k0, *reinterpret_cast<thread float4 (*)[4]>(&xv[0]));
        load_bf16_16(x_tile + ulong(r) * K + k0 + 16u, *reinterpret_cast<thread float4 (*)[4]>(&xv[4]));
        float4 s4 = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
        const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
        acc[r] += sc * dot_q8_32(w_ptr, xv) + bi * x_sum;
      }
    }
    for (uint r = 0u; r < n_rows; ++r) {
      acc[r] = reduce_k_lanes<4>(acc[r]);
      if (k_lane == 0u) {
        projected[out_tile_base + ulong(r) * D + col] = bfloat(acc[r]);
      }
    }
    return;
  }

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
    const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
    const device uint4 *w_ptr = reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul);
    const uint k0 = g << 5u;
    for (uint r = 0u; r < n_rows; ++r) {
      float4 xv[8];
      load_bf16_16(x_tile + ulong(r) * K + k0, *reinterpret_cast<thread float4 (*)[4]>(&xv[0]));
      load_bf16_16(x_tile + ulong(r) * K + k0 + 16u, *reinterpret_cast<thread float4 (*)[4]>(&xv[4]));
      float4 s4 = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
      const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
      acc[r] += sc * dot_q8_32(w_ptr, xv) + bi * x_sum;
    }
  }
  for (uint r = 0u; r < n_rows; ++r) {
    acc[r] = reduce_k_lanes<4>(acc[r]);
    if (k_lane == 0u) {
      projected[out_tile_base + ulong(r) * D + col] = bfloat(acc[r]);
    }
  }
}

// ============================================================================
// 4. M5 Max 4-SG Prefetched NAX Staged Matmul Helpers (BN=64 gate_up, BN=128 down)
// ============================================================================

template <ushort MaxRows, class Run>
inline void inkling2_live_rows(uint rows, Run run) {
  if constexpr (MaxRows > 8) {
    if (rows <= 8u) {
      run(integral_constant<ushort, 8>{});
      return;
    }
  }
  if constexpr (MaxRows > 16) {
    if (rows <= 16u) {
      run(integral_constant<ushort, 16>{});
      return;
    }
  }
  run(integral_constant<ushort, MaxRows>{});
}

inline void stage_nvfp4_32_regs(
    uint4 packed,
    float s0,
    float s1,
    threadgroup bfloat *dst) {
  threadgroup bfloat4 *dst4 = reinterpret_cast<threadgroup bfloat4 *>(dst);

  float4 w0[4];
  decode_nvfp4_16(uint2(packed.x, packed.y), w0);
  dst4[0] = bfloat4(w0[0] * s0);
  dst4[1] = bfloat4(w0[1] * s0);
  dst4[2] = bfloat4(w0[2] * s0);
  dst4[3] = bfloat4(w0[3] * s0);

  float4 w1[4];
  decode_nvfp4_16(uint2(packed.z, packed.w), w1);
  dst4[4] = bfloat4(w1[0] * s1);
  dst4[5] = bfloat4(w1[1] * s1);
  dst4[6] = bfloat4(w1[2] * s1);
  dst4[7] = bfloat4(w1[3] * s1);
}

inline void stage_q8_32_pair_regs(
    uint4 p0,
    uint4 p1,
    float sc,
    float bi,
    threadgroup bfloat *dst_hi,
    threadgroup bfloat *dst_lo) {
  threadgroup bfloat4 *hi4 = reinterpret_cast<threadgroup bfloat4 *>(dst_hi);
  threadgroup bfloat4 *lo4 = reinterpret_cast<threadgroup bfloat4 *>(dst_lo);

  const float4 v0 = fma(float4(as_type<uchar4>(p0.x)), float4(sc), float4(bi));
  const float4 v1 = fma(float4(as_type<uchar4>(p0.y)), float4(sc), float4(bi));
  const float4 v2 = fma(float4(as_type<uchar4>(p0.z)), float4(sc), float4(bi));
  const float4 v3 = fma(float4(as_type<uchar4>(p0.w)), float4(sc), float4(bi));
  const float4 v4 = fma(float4(as_type<uchar4>(p1.x)), float4(sc), float4(bi));
  const float4 v5 = fma(float4(as_type<uchar4>(p1.y)), float4(sc), float4(bi));
  const float4 v6 = fma(float4(as_type<uchar4>(p1.z)), float4(sc), float4(bi));
  const float4 v7 = fma(float4(as_type<uchar4>(p1.w)), float4(sc), float4(bi));

  const bfloat4 h0 = bfloat4(v0); hi4[0] = h0; lo4[0] = bfloat4(v0 - float4(h0));
  const bfloat4 h1 = bfloat4(v1); hi4[1] = h1; lo4[1] = bfloat4(v1 - float4(h1));
  const bfloat4 h2 = bfloat4(v2); hi4[2] = h2; lo4[2] = bfloat4(v2 - float4(h2));
  const bfloat4 h3 = bfloat4(v3); hi4[3] = h3; lo4[3] = bfloat4(v3 - float4(h3));
  const bfloat4 h4 = bfloat4(v4); hi4[4] = h4; lo4[4] = bfloat4(v4 - float4(h4));
  const bfloat4 h5 = bfloat4(v5); hi4[5] = h5; lo4[5] = bfloat4(v5 - float4(h5));
  const bfloat4 h6 = bfloat4(v6); hi4[6] = h6; lo4[6] = bfloat4(v6 - float4(h6));
  const bfloat4 h7 = bfloat4(v7); hi4[7] = h7; lo4[7] = bfloat4(v7 - float4(h7));
}

template <ushort Rows, bool IsQ8, class Finish>
inline void inkling2_nax_staged(
    device bfloat *x,
    uint K,
    device const uchar *w_row_base,
    device const uchar *s_raw,
    device const uchar *b_raw,
    ulong s_row_base,
    bool sidecar_f32,
    uint lane,
    threadgroup bfloat *stage,
    Finish finish) {
  constexpr auto desc = matmul2d_descriptor(
      Rows, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  tensor<device bfloat, dextents<int, 2>, tensor_inline> at(
      x, dextents<int, 2>{int(K), Rows}, array<int, 2>{1, int(K)});
  tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt0(
      stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt1(
      stage + 1024, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  auto a0 = at.template slice<32, Rows>(0, 0);
  auto b0 = bt0.template slice<32, 32>(0, 0);
  auto b1 = bt1.template slice<32, 32>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }

  const uint n_steps = K >> 5u;
  if constexpr (IsQ8) {
    const device uint4 *wp = reinterpret_cast<device const uint4 *>(w_row_base);
    for (uint step = 0u; step < n_steps; ++step) {
      const uint4 p0 = wp[step * 2u];
      const uint4 p1 = wp[step * 2u + 1u];
      const float sc = inkling2_read_sidecar(s_raw, s_row_base + step, sidecar_f32);
      const float bi = inkling2_read_sidecar(b_raw, s_row_base + step, sidecar_f32);
      stage_q8_32_pair_regs(
          p0, p1, sc, bi,
          stage + lane * 32u, stage + 1024u + lane * 32u);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto av = at.template slice<32, Rows>(int(step * 32u), 0);
      op.run(av, b0, acc);
      op.run(av, b1, acc);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    device const uint8_t *s_u8 = reinterpret_cast<device const uint8_t *>(s_raw) + s_row_base;
    const device uint4 *wp = reinterpret_cast<device const uint4 *>(w_row_base);
    for (uint step = 0u; step < n_steps; ++step) {
      threadgroup bfloat *dst = stage + (step & 1u) * 1024u + lane * 32u;
      stage_nvfp4_32_regs(
          wp[step],
          inkling2_decode_fp8_e4m3(s_u8[step * 2u]),
          inkling2_decode_fp8_e4m3(s_u8[step * 2u + 1u]),
          dst);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto av = at.template slice<32, Rows>(int(step * 32u), 0);
      if (step & 1u) {
        op.run(av, b1, acc);
      } else {
        op.run(av, b0, acc);
      }
    }
  }

  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      auto index = acc.get_multidimensional_index(i);
      finish(uint(index[1]), uint(index[0]), float(acc[i]));
    }
  }
}

// ============================================================================
// 5. Grouped 2-SG BN=32 Fused `gate_up + outer_scale + SiTU` NAX Kernel (8 KB smem)
// ============================================================================

template <ushort TileRows, bool IsQ8>
kernel void inkling2_moe_gate_up_situ_nax(
    device bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *nax_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uchar *gu_w [[buffer(4)]],
    device const uchar *gu_s [[buffer(5)]],
    device const uchar *gu_aux [[buffer(6)]],
    device bfloat *hidden [[buffer(7)]],
    constant Inkling2MoeParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  // 8 KB: 2 SGs x 2 buffers x 32x32 bfloat (reused as 32x32 FP32 gate buffer = 4 KB).
  threadgroup bfloat stage[4096];
  if (group.y >= count->nax_tiles) return;

  const uint tile_idx = nax_list[p.max_tiles + group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint N2 = I * 2u;
  const bool is_up = sg == 1u;
  const uint origin = group.x * 32u;
  device bfloat *x_tile = x + ulong(tile_idx) * TileRows * K;
  const ulong out_tile_base = ulong(tile_idx) * TileRows * I;
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);
  const bool sidecar_f32 = (p.flags & 4u) != 0u;
  const bool round_bf16 = (p.flags & 8u) != 0u;
  const float outer_s = (!IsQ8 && (p.flags & 2u))
      ? reinterpret_cast<device const float *>(gu_aux)[t.expert]
      : 1.0f;

  const uint row_in_expert = (is_up ? I : 0u) + origin + lane;
  const ulong full_row_idx = exp_row_base + row_in_expert;
  const device uchar *w_row_base = gu_w + full_row_idx * ulong(IsQ8 ? K : (K >> 1u));
  const ulong s_row_base = full_row_idx * ulong(IsQ8 ? (K >> 5u) : (K >> 4u));

  inkling2_live_rows<TileRows>(t.rows, [&](auto n_const) {
    constexpr ushort LiveR = decltype(n_const)::value;
    constexpr auto desc = matmul2d_descriptor(
        LiveR, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<1>> op;
    tensor<device bfloat, dextents<int, 2>, tensor_inline> at(
        x_tile, dextents<int, 2>{int(K), LiveR}, array<int, 2>{1, int(K)});
    threadgroup bfloat *sg_stage = stage + sg * 2048u;
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt0(
        sg_stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt1(
        sg_stage + 1024u, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto a0 = at.template slice<32, LiveR>(0, 0);
    auto b0 = bt0.template slice<32, 32>(0, 0);
    auto b1 = bt1.template slice<32, 32>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      acc[i] = 0.0f;
    }

    const uint n_steps = K >> 5u;
    if constexpr (IsQ8) {
      const device uint4 *wp = reinterpret_cast<device const uint4 *>(w_row_base);
      for (uint step = 0u; step < n_steps; ++step) {
        const uint4 p0 = wp[step * 2u];
        const uint4 p1 = wp[step * 2u + 1u];
        const float sc = inkling2_read_sidecar(gu_s, s_row_base + step, sidecar_f32);
        const float bi = inkling2_read_sidecar(gu_aux, s_row_base + step, sidecar_f32);
        stage_q8_32_pair_regs(
            p0, p1, sc, bi,
            sg_stage + lane * 32u, sg_stage + 1024u + lane * 32u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto av = at.template slice<32, LiveR>(int(step * 32u), 0);
        op.run(av, b0, acc);
        op.run(av, b1, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    } else {
      device const uint8_t *s_u8 = reinterpret_cast<device const uint8_t *>(gu_s) + s_row_base;
      const device uint4 *wp = reinterpret_cast<device const uint4 *>(w_row_base);
      for (uint step = 0u; step < n_steps; ++step) {
        threadgroup bfloat *dst = sg_stage + (step & 1u) * 1024u + lane * 32u;
        stage_nvfp4_32_regs(
            wp[step],
            inkling2_decode_fp8_e4m3(s_u8[step * 2u]),
            inkling2_decode_fp8_e4m3(s_u8[step * 2u + 1u]),
            dst);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto av = at.template slice<32, LiveR>(int(step * 32u), 0);
        if (step & 1u) {
          op.run(av, b1, acc);
        } else {
          op.run(av, b0, acc);
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *gate_f32 = reinterpret_cast<threadgroup float *>(stage);
    if (!is_up) {
      for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
          auto index = acc.get_multidimensional_index(i);
          gate_f32[uint(index[1]) * 32u + uint(index[0])] = float(acc[i]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_up) {
      for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
          auto index = acc.get_multidimensional_index(i);
          const uint r = uint(index[1]);
          const uint c = uint(index[0]);
          const ulong dst_idx = out_tile_base + ulong(r) * I + origin + c;
          if (r < t.rows) {
            const float g_v = gate_f32[r * 32u + c];
            const float u_v = float(acc[i]);
            hidden[dst_idx] = inkling2_moe_situ_epilogue(
                g_v, u_v, outer_s, p.gate_softcap, p.up_softcap, round_bf16);
          } else {
            hidden[dst_idx] = bfloat(0.0f);
          }
        }
      }
    }
  });
}

// ============================================================================
// 6. Grouped 2-SG BN=64 `down_proj` NAX Kernel (8 KB smem, TileRows = 8 / 32)
// ============================================================================

template <ushort TileRows, bool IsQ8>
kernel void inkling2_moe_down_nax(
    device bfloat *hidden [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *nax_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uchar *dn_w [[buffer(4)]],
    device const uchar *dn_s [[buffer(5)]],
    device const uchar *dn_b [[buffer(6)]],
    device bfloat *projected [[buffer(7)]],
    constant Inkling2MoeParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat stage[4096]; // 8 KB (2 SGs x 2 buffers x 32x32 bfloat)
  if (group.y >= count->nax_tiles) return;

  const uint tile_idx = nax_list[p.max_tiles + group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint K = p.intermediate;
  const uint D = p.hidden;
  const uint origin = group.x * 64u + sg * 32u;
  if (origin >= D) return;

  device bfloat *x_tile = hidden + ulong(tile_idx) * TileRows * K;
  const ulong out_tile_base = ulong(tile_idx) * TileRows * D;
  const ulong exp_row_base = ulong(t.expert) * ulong(D);
  const bool sidecar_f32 = (p.flags & 4u) != 0u;

  const ulong full_row_idx = exp_row_base + origin + lane;
  const device uchar *w_row_base = dn_w + full_row_idx * ulong(IsQ8 ? K : (K >> 1u));
  const ulong s_row_base = full_row_idx * ulong(IsQ8 ? (K >> 5u) : (K >> 4u));

  inkling2_live_rows<TileRows>(t.rows, [&](auto n_const) {
    auto finish = [&](uint r, uint c, float v) {
      if (r < t.rows) {
        projected[out_tile_base + ulong(r) * D + origin + c] = bfloat(v);
      }
    };
    inkling2_nax_staged<decltype(n_const)::value, IsQ8>(
        x_tile,
        K,
        w_row_base,
        dn_s,
        dn_b,
        s_row_base,
        sidecar_f32,
        lane,
        stage + sg * 2048u,
        finish);
  });
}

// ============================================================================
// 6. Grouped Weighted Combine Kernel (`projected[inverse[row, k]] * gamma[row, k]`)
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_moe_grouped_combine_bf16(
    device const bfloat *projected [[buffer(0)]],
    device const uint *inverse [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant Inkling2MoeParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x;
  const uint col = group.y * 256u + tid;
  if (row >= p.rows || col >= p.hidden) return;

  const uint base = row * p.top_k;
  float sum = 0.0f;
  for (uint k = 0u; k < p.top_k; ++k) {
    const uint pos = inverse[base + k];
    sum += gamma[base + k] * float(projected[ulong(pos) * p.hidden + col]);
  }
  out[ulong(row) * p.hidden + col] = bfloat(sum);
}

// Explicit kernel instantiations for prebuilt metallib lookup.
template [[host_name("inkling2_moe_gate_up_situ_nax_nvfp4_8")]]
kernel decltype(inkling2_moe_gate_up_situ_nax<8, false>) inkling2_moe_gate_up_situ_nax<8, false>;
template [[host_name("inkling2_moe_gate_up_situ_nax_nvfp4_32")]]
kernel decltype(inkling2_moe_gate_up_situ_nax<32, false>) inkling2_moe_gate_up_situ_nax<32, false>;
template [[host_name("inkling2_moe_gate_up_situ_nax_q8_8")]]
kernel decltype(inkling2_moe_gate_up_situ_nax<8, true>) inkling2_moe_gate_up_situ_nax<8, true>;
template [[host_name("inkling2_moe_gate_up_situ_nax_q8_32")]]
kernel decltype(inkling2_moe_gate_up_situ_nax<32, true>) inkling2_moe_gate_up_situ_nax<32, true>;

template [[host_name("inkling2_moe_down_nax_nvfp4_8")]]
kernel decltype(inkling2_moe_down_nax<8, false>) inkling2_moe_down_nax<8, false>;
template [[host_name("inkling2_moe_down_nax_nvfp4_32")]]
kernel decltype(inkling2_moe_down_nax<32, false>) inkling2_moe_down_nax<32, false>;
template [[host_name("inkling2_moe_down_nax_q8_8")]]
kernel decltype(inkling2_moe_down_nax<8, true>) inkling2_moe_down_nax<8, true>;
template [[host_name("inkling2_moe_down_nax_q8_32")]]
kernel decltype(inkling2_moe_down_nax<32, true>) inkling2_moe_down_nax<32, true>;
