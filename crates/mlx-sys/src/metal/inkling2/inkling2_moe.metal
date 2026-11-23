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

// Exact 0-ULP branchless FP4 E2M1 decode WITHOUT the 16384.0h factor (2^14 hoisted ONCE
// per 16-elem group into FP32 `scale * 16384.0f` -> 0-ULP bit-identical while eliminating
// 4 vector half4 multiplies per 16 weights!).
// Matches mlx/backend/metal/kernels/fp4.h bit-for-bit for ALL 16 nibble values 0..15:
//   nib=0..7  -> (+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0) * 2^-14
//   nib=8..15 -> (-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0) * 2^-14
inline half4 decode_nvfp4_4_half_unscaled(uint packed16) {
  const ushort4 sh = ushort4(
      ushort(packed16),
      ushort(packed16 >> 4u),
      ushort(packed16 >> 8u),
      ushort(packed16 >> 12u));
  const ushort4 bits = ((sh & ushort4(7u)) << 9u) | ((sh & ushort4(8u)) << 12u);
  return as_type<half4>(bits);
}

inline void decode_nvfp4_16_half_unscaled(uint2 packed, thread half4 (&w)[4]) {
  w[0] = decode_nvfp4_4_half_unscaled(packed.x);
  w[1] = decode_nvfp4_4_half_unscaled(packed.x >> 16u);
  w[2] = decode_nvfp4_4_half_unscaled(packed.y);
  w[3] = decode_nvfp4_4_half_unscaled(packed.y >> 16u);
}

inline float inkling2_decode_fp8_e4m3_x16384(uint8_t bits) {
  const uint16_t v = bits & 127u;
  const uint16_t sign_bit = uint16_t((bits >> 7u) & 1u) << 15u;
  const uint16_t u = (v << 7u) | (((v + 1u) >> 7u) << 14u) | sign_bit;
  return float(as_type<half>(u) * 256.0h) * 16384.0f;
}

inline float dot_4x4_half(const thread half4 (&w)[4], const thread float4 (&x)[4]) {
  return (dot(float4(w[0]), x[0]) + dot(float4(w[1]), x[1])) +
         (dot(float4(w[2]), x[2]) + dot(float4(w[3]), x[3]));
}

// Load 32 BF16 activations into two float4[4] packs using 4 parallel 128-bit uint4 loads.
inline void load_bf16_32(
    device const bfloat *src,
    thread float4 (&x0)[4],
    thread float4 (&x1)[4]) {
  const device uint4 *p = reinterpret_cast<device const uint4 *>(src);
  const uint4 a = p[0];
  const uint4 b = p[1];
  const uint4 c = p[2];
  const uint4 d = p[3];
  x0[0] = float4(as_type<bfloat4>(a.xy));
  x0[1] = float4(as_type<bfloat4>(a.zw));
  x0[2] = float4(as_type<bfloat4>(b.xy));
  x0[3] = float4(as_type<bfloat4>(b.zw));
  x1[0] = float4(as_type<bfloat4>(c.xy));
  x1[1] = float4(as_type<bfloat4>(c.zw));
  x1[2] = float4(as_type<bfloat4>(d.xy));
  x1[3] = float4(as_type<bfloat4>(d.zw));
}

// Exact FP8 E4M3FN decode (matches mlx/backend/metal/kernels/fp8.h bit-for-bit).
inline float inkling2_decode_fp8_e4m3(uint8_t bits) {
  const uint16_t v = bits & 127u;
  const uint16_t sign_bit = uint16_t((bits >> 7u) & 1u) << 15u;
  const uint16_t u = (v << 7u) | (((v + 1u) >> 7u) << 14u) | sign_bit;
  return float(as_type<half>(u) * 256.0h);
}

// Exact vectorized 4x FP8 E4M3FN decode WITHOUT the 256.0h factor (hoisted ONCE per 32-elem
// group in FP32: bit-identical to 4x inkling2_decode_fp8_e4m3 while saving 8 vector half muls!).
inline half4 decode_fp8_e4m3_4_half_unscaled(uint packed) {
  const ushort4 b = ushort4(as_type<uchar4>(packed));
  const ushort4 v = b & ushort4(127u);
  const ushort4 sign_bit = (b & ushort4(128u)) << 8u;
  const ushort4 u = (v << 7u) | (((v + 1u) >> 7u) << 14u) | sign_bit;
  return as_type<half4>(u);
}

inline void decode_mxfp8_32_half_unscaled(const device uint4 *w_u4, thread half4 (&w)[8]) {
  const uint4 p0 = w_u4[0];
  const uint4 p1 = w_u4[1];
  w[0] = decode_fp8_e4m3_4_half_unscaled(p0.x);
  w[1] = decode_fp8_e4m3_4_half_unscaled(p0.y);
  w[2] = decode_fp8_e4m3_4_half_unscaled(p0.z);
  w[3] = decode_fp8_e4m3_4_half_unscaled(p0.w);
  w[4] = decode_fp8_e4m3_4_half_unscaled(p1.x);
  w[5] = decode_fp8_e4m3_4_half_unscaled(p1.y);
  w[6] = decode_fp8_e4m3_4_half_unscaled(p1.z);
  w[7] = decode_fp8_e4m3_4_half_unscaled(p1.w);
}

inline float dot_mxfp8_32_regs(
    const thread half4 (&w)[8],
    const thread float4 (&x0)[4],
    const thread float4 (&x1)[4]) {
  float acc = dot(float4(w[0]), x0[0]);
  acc += dot(float4(w[1]), x0[1]);
  acc += dot(float4(w[2]), x0[2]);
  acc += dot(float4(w[3]), x0[3]);
  acc += dot(float4(w[4]), x1[0]);
  acc += dot(float4(w[5]), x1[1]);
  acc += dot(float4(w[6]), x1[2]);
  acc += dot(float4(w[7]), x1[3]);
  return acc * 256.0f;
}

// Exact FP8 UE8M0 scale decode (matches mlx/backend/metal/kernels/fp8.h bit-for-bit).
inline float inkling2_decode_fp8_e8m0(uint8_t bits) {
  const uint u = (bits == 0u) ? 0x400000u : (uint(bits) << 23u);
  return as_type<float>(u);
}

inline float inkling2_read_sidecar(device const uchar *ptr, ulong idx, bool is_f32) {
  if (is_f32) {
    return reinterpret_cast<device const float *>(ptr)[idx];
  }
  return float(reinterpret_cast<device const bfloat *>(ptr)[idx]);
}

inline float inkling2_read_f16_scale(device const uchar *ptr, ulong idx) {
  return float(reinterpret_cast<device const half *>(ptr)[idx]);
}

inline float inkling2_moe_read_f32(device const uchar *ptr, ulong idx, uint is_f32) {
  if (is_f32 != 0u) {
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

// Accumulate dot of 16 uint8 affine Q8_0 weights (preloaded in uint4 p) against 4 float4 activations.
inline float dot_q8_16_accum(float acc, uint4 p, const thread float4 (&x)[4]) {
  acc += dot(float4(as_type<uchar4>(p.x)), x[0]);
  acc += dot(float4(as_type<uchar4>(p.y)), x[1]);
  acc += dot(float4(as_type<uchar4>(p.z)), x[2]);
  acc += dot(float4(as_type<uchar4>(p.w)), x[3]);
  return acc;
}

// Accumulate dot of 16 symmetric Q8_0 weights (`int8(u8 ^ 0x80)`, exact 0-ULP to `u8 - 128`).
inline float dot_q8_sym_16_accum(float acc, uint4 p, const thread float4 (&x)[4]) {
  acc += dot(float4(as_type<char4>(p.x ^ 0x80808080u)), x[0]);
  acc += dot(float4(as_type<char4>(p.y ^ 0x80808080u)), x[1]);
  acc += dot(float4(as_type<char4>(p.z ^ 0x80808080u)), x[2]);
  acc += dot(float4(as_type<char4>(p.w ^ 0x80808080u)), x[3]);
  return acc;
}

// Unpack 4 consecutive 5-bit unsigned codes (0..31) from the low 20 bits of `bits20`.
inline uchar4 unpack_q5k_4(uint bits20) {
  return uchar4(
      uchar(bits20 & 31u),
      uchar((bits20 >> 5u) & 31u),
      uchar((bits20 >> 10u) & 31u),
      uchar((bits20 >> 15u) & 31u));
}

// Unpack one 32-element RowMajor Q5_K group (5 uint32 words = 160 bits, LSB-first) into 8 uchar4 regs.
inline void decode_q5k_32_regs(const device uint *w_ptr, thread uchar4 (&cw)[8]) {
  const uint w0 = w_ptr[0];
  const uint w1 = w_ptr[1];
  const uint w2 = w_ptr[2];
  const uint w3 = w_ptr[3];
  const uint w4 = w_ptr[4];
  cw[0] = unpack_q5k_4(w0);
  cw[1] = unpack_q5k_4((w0 >> 20u) | (w1 << 12u));
  cw[2] = unpack_q5k_4(w1 >> 8u);
  cw[3] = unpack_q5k_4((w1 >> 28u) | (w2 << 4u));
  cw[4] = unpack_q5k_4((w2 >> 16u) | (w3 << 16u));
  cw[5] = unpack_q5k_4(w3 >> 4u);
  cw[6] = unpack_q5k_4((w3 >> 24u) | (w4 << 8u));
  cw[7] = unpack_q5k_4(w4 >> 12u);
}

inline float dot_q5k_32_regs(
    const thread uchar4 (&cw)[8],
    const thread float4 (&x0)[4],
    const thread float4 (&x1)[4]) {
  float acc = dot(float4(cw[0]), x0[0]);
  acc += dot(float4(cw[1]), x0[1]);
  acc += dot(float4(cw[2]), x0[2]);
  acc += dot(float4(cw[3]), x0[3]);
  acc += dot(float4(cw[4]), x1[0]);
  acc += dot(float4(cw[5]), x1[1]);
  acc += dot(float4(cw[6]), x1[2]);
  acc += dot(float4(cw[7]), x1[3]);
  return acc;
}

inline float4 sum_bf16_16_v4(const thread float4 (&x)[4]) {
  return (x[0] + x[1]) + (x[2] + x[3]);
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
// k_lanes = 16 -> 16 consecutive lanes read 16 * 8 = 128 bytes (1 full cacheline) per uint2 load,
// 0-ULP unscaled branchless ALU FP4 decode, exact same FP32 reduction order as simd_gate_up_situ_nvfp4!
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
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (p.intermediate + col);

  const device uint2 *gw_u2 = reinterpret_cast<device const uint2 *>(
      gu_w + (exp_row_base + g_row) * row_w_stride);
  const device uint8_t *gs_row = gu_s + (exp_row_base + g_row) * row_s_stride;
  const device uint2 *uw_u2 = reinterpret_cast<device const uint2 *>(
      gu_w + (exp_row_base + u_row) * row_w_stride);
  const device uint8_t *us_row = gu_s + (exp_row_base + u_row) * row_s_stride;

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    float4 xv[4];
    load_bf16_16(x + (g << 4u), xv);

    half4 wv[4];
    decode_nvfp4_16_half_unscaled(gw_u2[g], wv);
    acc_g += inkling2_decode_fp8_e4m3_x16384(gs_row[g]) * dot_4x4_half(wv, xv);

    decode_nvfp4_16_half_unscaled(uw_u2[g], wv);
    acc_u += inkling2_decode_fp8_e4m3_x16384(us_row[g]) * dot_4x4_half(wv, xv);
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
  const bool sym_q8 = (p.flags & 128u) != 0u;
  const ulong exp_row_base = ulong(expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (p.intermediate + col);

  const device uint4 *gw_u4 = reinterpret_cast<device const uint4 *>(
      gu_w + (exp_row_base + g_row) * ulong(K));
  const ulong gs_base = (exp_row_base + g_row) * ulong(n_groups);
  const device uint4 *uw_u4 = reinterpret_cast<device const uint4 *>(
      gu_w + (exp_row_base + u_row) * ulong(K));
  const ulong us_base = (exp_row_base + u_row) * ulong(n_groups);

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  if (sym_q8) {
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_read_f16_scale(gu_s, gs_base + g);
      const float us = inkling2_read_f16_scale(gu_s, us_base + g);
      const uint4 gp0 = gw_u4[g * 2u];
      const uint4 gp1 = gw_u4[g * 2u + 1u];
      const uint4 up0 = uw_u4[g * 2u];
      const uint4 up1 = uw_u4[g * 2u + 1u];
      float4 x0[4], x1[4];
      load_bf16_32(x + (g << 5u), x0, x1);
      const float gd = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, gp0, x0), gp1, x1);
      const float ud = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, up0, x0), up1, x1);
      acc_g += gs * gd;
      acc_u += us * ud;
    }
  } else {
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
      const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
      const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
      const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
      const uint4 gp0 = gw_u4[g * 2u];
      const uint4 gp1 = gw_u4[g * 2u + 1u];
      const uint4 up0 = uw_u4[g * 2u];
      const uint4 up1 = uw_u4[g * 2u + 1u];
      float4 x0[4], x1[4];
      load_bf16_32(x + (g << 5u), x0, x1);
      const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
      const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
      const float gd = dot_q8_16_accum(dot_q8_16_accum(0.0f, gp0, x0), gp1, x1);
      const float ud = dot_q8_16_accum(dot_q8_16_accum(0.0f, up0, x0), up1, x1);
      acc_g += gs * gd + gb * x_sum;
      acc_u += us * ud + ub * x_sum;
    }
  }

  acc_g = reduce_k_lanes<8>(acc_g);
  acc_u = reduce_k_lanes<8>(acc_u);
  if (k_lane == 0u) {
    const bool round_bf16 = (p.flags & 8u) != 0u;
    act_out[ulong(slot) * p.intermediate + col] = inkling2_moe_situ_epilogue(
        acc_g, acc_u, 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
  }
}

// Grid: (ceil(I / 8), top_k, 1), TG: (64, 1, 1) -> 2 SGs x 4 cols/SG = 8 cols/TG.
// Native MXFP8 (E4M3FN weights + UE8M0 scales, group_size=32): k_lanes = 8 -> 256B coalesced loads!
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_gate_up_situ_mxfp8(
    device const bfloat *x [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const uchar *gu_w [[buffer(2)]],
    device const uint8_t *gu_s [[buffer(3)]],
    device bfloat *act_out [[buffer(4)]],
    constant Inkling2MoeParams &p [[buffer(5)]],
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
  const ulong exp_row_base = ulong(expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (p.intermediate + col);

  const device uchar *gw_row = gu_w + (exp_row_base + g_row) * ulong(K);
  const device uint8_t *gs_row = gu_s + (exp_row_base + g_row) * ulong(n_groups);
  const device uchar *uw_row = gu_w + (exp_row_base + u_row) * ulong(K);
  const device uint8_t *us_row = gu_s + (exp_row_base + u_row) * ulong(n_groups);

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const float gs = inkling2_decode_fp8_e8m0(gs_row[g]);
    const float us = inkling2_decode_fp8_e8m0(us_row[g]);
    half4 gw[8], uw[8];
    decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul), gw);
    decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul), uw);
    float4 x0[4], x1[4];
    load_bf16_32(x + (g << 5u), x0, x1);
    acc_g += gs * dot_mxfp8_32_regs(gw, x0, x1);
    acc_u += us * dot_mxfp8_32_regs(uw, x0, x1);
  }

  acc_g = reduce_k_lanes<8>(acc_g);
  acc_u = reduce_k_lanes<8>(acc_u);
  if (k_lane == 0u) {
    const bool round_bf16 = (p.flags & 8u) != 0u;
    act_out[ulong(slot) * p.intermediate + col] = inkling2_moe_situ_epilogue(
        acc_g, acc_u, 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
  }
}

// Grid: (ceil(I / 8), top_k, 1), TG: (64, 1, 1) -> 2 SGs x 4 cols/SG = 8 cols/TG.
// RowMajor Q5_K (256-elem super-blocks = 8 sub-blocks of 32): k_lanes = 8 maps the 8 lanes
// 1:1 onto the 8 sub-blocks of each super-block (shared uniform half2 (d, dmin), 16B coalesced
// uchar2 (sc, m), 160B contiguous 5-bit weights). Exact same FP32 reduction order as
// `inkling2_moe_simd_gate_up_situ_q5k` -> 0-ULP bit-exact M=1..32!
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_gate_up_situ_q5k(
    device const bfloat *x [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const uint *gu_w [[buffer(2)]],
    device const uchar *gu_s [[buffer(3)]],
    device const half *gu_b [[buffer(4)]],
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
  const uint n_super = K >> 8u;
  const ulong row_w_words = ulong(n_groups) * 5ul;
  const ulong exp_row_base = ulong(expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (p.intermediate + col);

  const device uint *gw_row = gu_w + (exp_row_base + g_row) * row_w_words;
  const device uchar2 *gs_row =
      reinterpret_cast<device const uchar2 *>(gu_s) + (exp_row_base + g_row) * ulong(n_groups);
  const device half2 *gb_row =
      reinterpret_cast<device const half2 *>(gu_b) + (exp_row_base + g_row) * ulong(n_super);

  const device uint *uw_row = gu_w + (exp_row_base + u_row) * row_w_words;
  const device uchar2 *us_row =
      reinterpret_cast<device const uchar2 *>(gu_s) + (exp_row_base + u_row) * ulong(n_groups);
  const device half2 *ub_row =
      reinterpret_cast<device const half2 *>(gu_b) + (exp_row_base + u_row) * ulong(n_super);

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  for (uint sb = 0u; sb < n_super; ++sb) {
    const uint g = (sb << 3u) + k_lane;
    const half2 gdm = gb_row[sb];
    const half2 udm = ub_row[sb];
    const uchar2 gsm = gs_row[g];
    const uchar2 usm = us_row[g];
    const float gs = float(gdm.x) * float(gsm.x);
    const float gb = -(float(gdm.y) * float(gsm.y));
    const float us = float(udm.x) * float(usm.x);
    const float ub = -(float(udm.y) * float(usm.y));
    uchar4 gw[8], uw[8];
    decode_q5k_32_regs(gw_row + ulong(g) * 5ul, gw);
    decode_q5k_32_regs(uw_row + ulong(g) * 5ul, uw);
    float4 x0[4], x1[4];
    load_bf16_32(x + (g << 5u), x0, x1);
    const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
    const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
    acc_g += gs * dot_q5k_32_regs(gw, x0, x1) + gb * x_sum;
    acc_u += us * dot_q5k_32_regs(uw, x0, x1) + ub * x_sum;
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
// k_lanes = 8 with uint2 loads -> 72 / 8 = 9 exact iters (0% divergence!), 0-ULP unscaled
// branchless ALU FP4 decode, exact same FP32 reduction order as simd_down_nvfp4 -> 0-ULP M=1..32!
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
      const device uint2 *w_u2 = reinterpret_cast<device const uint2 *>(
          dn_w + row_idx * row_w_stride);
      const device uint8_t *s_row = dn_s + row_idx * row_s_stride;
      const device bfloat *a_row = act + ulong(slot) * K;
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        float4 xv[4];
        half4 wv[4];
        load_bf16_16(a_row + (g << 4u), xv);
        decode_nvfp4_16_half_unscaled(w_u2[g], wv);
        acc += inkling2_decode_fp8_e4m3_x16384(s_row[g]) * dot_4x4_half(wv, xv);
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

// Fused M=1 NVFP4 down_proj + combine + static post_conv (k=4) + optional FP32 residual + hist_out.
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_down_combine_sconv_nvfp4(
    device const bfloat *act [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const uchar *dn_w [[buffer(3)]],
    device const uint8_t *dn_s [[buffer(4)]],
    device const uchar *post_w [[buffer(5)]],
    device const uchar *hist_in [[buffer(6)]],
    device const uchar *res_raw [[buffer(7)]],
    device bfloat *out [[buffer(8)]],
    device bfloat *hist_out [[buffer(9)]],
    device bfloat *tape_out [[buffer(10)]],
    constant Inkling2MoeParams &p [[buffer(11)]],
    constant Inkling2MoePostConvParams &pc [[buffer(12)]],
    uint group_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 8u;
  threadgroup float slot_vals[16][4];

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group_x * 4u + r_id;
  const uint D = p.hidden;
  const bool valid_col = col < D;

  const uint K = p.intermediate;
  const uint n_groups = K >> 4u;
  const ulong row_w_stride = ulong(K >> 1u);
  const ulong row_s_stride = ulong(n_groups);

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(D) + col;
      const device uint2 *w_u2 = reinterpret_cast<device const uint2 *>(
          dn_w + row_idx * row_w_stride);
      const device uint8_t *s_row = dn_s + row_idx * row_s_stride;
      const device bfloat *a_row = act + ulong(slot) * K;
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        float4 xv[4];
        half4 wv[4];
        load_bf16_16(a_row + (g << 4u), xv);
        decode_nvfp4_16_half_unscaled(w_u2[g], wv);
        acc += inkling2_decode_fp8_e4m3_x16384(s_row[g]) * dot_4x4_half(wv, xv);
      }
    }
    acc = reduce_k_lanes<8>(acc);
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
    const bfloat moe_b = bfloat(sum);
    if ((pc.flags & 8u) != 0u) {
      tape_out[col] = moe_b;
    }
    const float x0 = float(moe_b);
    const uint h_f32 = pc.hist_f32_mask & 1u;
    const float x3 = inkling2_moe_read_f32(hist_in, 0ul * ulong(D) + col, h_f32);
    const float x2 = inkling2_moe_read_f32(hist_in, 1ul * ulong(D) + col, h_f32);
    const float x1 = inkling2_moe_read_f32(hist_in, 2ul * ulong(D) + col, h_f32);

    const uint w_f32 = pc.flags & 1u;
    const ulong wb = ulong(col) * 4ul;
    const float w3 = inkling2_moe_read_f32(post_w, wb + 0ul, w_f32);
    const float w2 = inkling2_moe_read_f32(post_w, wb + 1ul, w_f32);
    const float w1 = inkling2_moe_read_f32(post_w, wb + 2ul, w_f32);
    const float w0 = inkling2_moe_read_f32(post_w, wb + 3ul, w_f32);

    const float scale = 0.44721359549995793f;
    const float conv_acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
    const bfloat conv_val = bfloat(conv_acc * scale);
    if ((pc.flags & 2u) != 0u) {
      const uint r_f32 = (pc.flags >> 2u) & 1u;
      const float rv = inkling2_moe_read_f32(res_raw, ulong(col), r_f32);
      out[col] = bfloat(rv + float(conv_val));
    } else {
      out[col] = conv_val;
    }
    hist_out[0ul * ulong(D) + col] = bfloat(x2);
    hist_out[1ul * ulong(D) + col] = bfloat(x1);
    hist_out[2ul * ulong(D) + col] = moe_b;
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
  const bool sym_q8 = (p.flags & 128u) != 0u;

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(p.hidden) + col;
      const device uint4 *w_u4 = reinterpret_cast<device const uint4 *>(dn_w + row_idx * ulong(K));
      const ulong s_base = row_idx * ulong(n_groups);
      const device bfloat *a_row = act + ulong(slot) * K;
      if (sym_q8) {
        for (uint g = k_lane; g < n_groups; g += kLanes) {
          const float sc = inkling2_read_f16_scale(dn_s, s_base + g);
          const uint4 p0 = w_u4[g * 2u];
          const uint4 p1 = w_u4[g * 2u + 1u];
          float4 x0[4], x1[4];
          load_bf16_32(a_row + (g << 5u), x0, x1);
          const float dp = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, p0, x0), p1, x1);
          acc += sc * dp;
        }
      } else {
        for (uint g = k_lane; g < n_groups; g += kLanes) {
          const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
          const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
          const uint4 p0 = w_u4[g * 2u];
          const uint4 p1 = w_u4[g * 2u + 1u];
          float4 x0[4], x1[4];
          load_bf16_32(a_row + (g << 5u), x0, x1);
          const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
          const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
          const float dp = dot_q8_16_accum(dot_q8_16_accum(0.0f, p0, x0), p1, x1);
          acc += sc * dp + bi * x_sum;
        }
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

// Fused M=1 Q8_0 down_proj + combine + static post_conv (k=4) + optional FP32 residual + hist_out.
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_down_combine_sconv_q8(
    device const bfloat *act [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const uchar *dn_w [[buffer(3)]],
    device const uchar *dn_s [[buffer(4)]],
    device const uchar *dn_b [[buffer(5)]],
    device const uchar *post_w [[buffer(6)]],
    device const uchar *hist_in [[buffer(7)]],
    device const uchar *res_raw [[buffer(8)]],
    device bfloat *out [[buffer(9)]],
    device bfloat *hist_out [[buffer(10)]],
    device bfloat *tape_out [[buffer(11)]],
    constant Inkling2MoeParams &p [[buffer(12)]],
    constant Inkling2MoePostConvParams &pc [[buffer(13)]],
    uint group_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 4u;
  threadgroup float slot_vals[16][8];

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 2u; // 0..7
  const uint col = group_x * 8u + r_id;
  const uint D = p.hidden;
  const bool valid_col = col < D;

  const uint K = p.intermediate;
  const uint n_groups = K >> 5u;
  const bool sidecar_f32 = (p.flags & 4u) != 0u;
  const bool sym_q8 = (p.flags & 128u) != 0u;

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(D) + col;
      const device uint4 *w_u4 = reinterpret_cast<device const uint4 *>(dn_w + row_idx * ulong(K));
      const ulong s_base = row_idx * ulong(n_groups);
      const device bfloat *a_row = act + ulong(slot) * K;
      if (sym_q8) {
        for (uint g = k_lane; g < n_groups; g += kLanes) {
          const float sc = inkling2_read_f16_scale(dn_s, s_base + g);
          const uint4 p0 = w_u4[g * 2u];
          const uint4 p1 = w_u4[g * 2u + 1u];
          float4 x0[4], x1[4];
          load_bf16_32(a_row + (g << 5u), x0, x1);
          const float dp = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, p0, x0), p1, x1);
          acc += sc * dp;
        }
      } else {
        for (uint g = k_lane; g < n_groups; g += kLanes) {
          const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
          const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
          const uint4 p0 = w_u4[g * 2u];
          const uint4 p1 = w_u4[g * 2u + 1u];
          float4 x0[4], x1[4];
          load_bf16_32(a_row + (g << 5u), x0, x1);
          const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
          const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
          const float dp = dot_q8_16_accum(dot_q8_16_accum(0.0f, p0, x0), p1, x1);
          acc += sc * dp + bi * x_sum;
        }
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
    const bfloat moe_b = bfloat(sum);
    if ((pc.flags & 8u) != 0u) {
      tape_out[col] = moe_b;
    }
    const float x0 = float(moe_b);
    const uint h_f32 = pc.hist_f32_mask & 1u;
    const float x3 = inkling2_moe_read_f32(hist_in, 0ul * ulong(D) + col, h_f32);
    const float x2 = inkling2_moe_read_f32(hist_in, 1ul * ulong(D) + col, h_f32);
    const float x1 = inkling2_moe_read_f32(hist_in, 2ul * ulong(D) + col, h_f32);

    const uint w_f32 = pc.flags & 1u;
    const ulong wb = ulong(col) * 4ul;
    const float w3 = inkling2_moe_read_f32(post_w, wb + 0ul, w_f32);
    const float w2 = inkling2_moe_read_f32(post_w, wb + 1ul, w_f32);
    const float w1 = inkling2_moe_read_f32(post_w, wb + 2ul, w_f32);
    const float w0 = inkling2_moe_read_f32(post_w, wb + 3ul, w_f32);

    const float scale = 0.44721359549995793f;
    const float conv_acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
    const bfloat conv_val = bfloat(conv_acc * scale);
    if ((pc.flags & 2u) != 0u) {
      const uint r_f32 = (pc.flags >> 2u) & 1u;
      const float rv = inkling2_moe_read_f32(res_raw, ulong(col), r_f32);
      out[col] = bfloat(rv + float(conv_val));
    } else {
      out[col] = conv_val;
    }
    hist_out[0ul * ulong(D) + col] = bfloat(x2);
    hist_out[1ul * ulong(D) + col] = bfloat(x1);
    hist_out[2ul * ulong(D) + col] = moe_b;
  }
}

// Grid: (ceil(D / 8), 1, 1), TG: (64, 1, 1).
// Native MXFP8 M=1 down_proj + weighted combine (k_lanes = 4 -> 128B full cacheline!).
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_down_combine_mxfp8(
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
  constexpr uint kLanes = 4u;
  threadgroup float slot_vals[16][8];

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 2u; // 0..7
  const uint col = group_x * 8u + r_id;
  const bool valid_col = col < p.hidden;

  const uint K = p.intermediate;
  const uint n_groups = K >> 5u;

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(p.hidden) + col;
      const device uchar *w_row = dn_w + row_idx * ulong(K);
      const device uint8_t *s_row = dn_s + row_idx * ulong(n_groups);
      const device bfloat *a_row = act + ulong(slot) * K;
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float sc = inkling2_decode_fp8_e8m0(s_row[g]);
        half4 w[8];
        decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul), w);
        float4 x0[4], x1[4];
        load_bf16_32(a_row + (g << 5u), x0, x1);
        acc += sc * dot_mxfp8_32_regs(w, x0, x1);
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

// Fused M=1 MXFP8 down_proj + combine + static post_conv (k=4) + optional FP32 residual + hist_out.
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_m1_down_combine_sconv_mxfp8(
    device const bfloat *act [[buffer(0)]],
    device const int32_t *indices [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const uchar *dn_w [[buffer(3)]],
    device const uint8_t *dn_s [[buffer(4)]],
    device const uchar *post_w [[buffer(5)]],
    device const uchar *hist_in [[buffer(6)]],
    device const uchar *res_raw [[buffer(7)]],
    device bfloat *out [[buffer(8)]],
    device bfloat *hist_out [[buffer(9)]],
    device bfloat *tape_out [[buffer(10)]],
    constant Inkling2MoeParams &p [[buffer(11)]],
    constant Inkling2MoePostConvParams &pc [[buffer(12)]],
    uint group_x [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLanes = 4u;
  threadgroup float slot_vals[16][8];

  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 2u; // 0..7
  const uint col = group_x * 8u + r_id;
  const uint D = p.hidden;
  const bool valid_col = col < D;

  const uint K = p.intermediate;
  const uint n_groups = K >> 5u;

  for (uint slot = sg; slot < p.top_k; slot += 2u) {
    float acc = 0.0f;
    if (valid_col) {
      const uint expert = uint(indices[slot]);
      const ulong row_idx = ulong(expert) * ulong(D) + col;
      const device uchar *w_row = dn_w + row_idx * ulong(K);
      const device uint8_t *s_row = dn_s + row_idx * ulong(n_groups);
      const device bfloat *a_row = act + ulong(slot) * K;
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float sc = inkling2_decode_fp8_e8m0(s_row[g]);
        half4 w[8];
        decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul), w);
        float4 x0[4], x1[4];
        load_bf16_32(a_row + (g << 5u), x0, x1);
        acc += sc * dot_mxfp8_32_regs(w, x0, x1);
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
    const bfloat moe_b = bfloat(sum);
    if ((pc.flags & 8u) != 0u) {
      tape_out[col] = moe_b;
    }
    const float x0 = float(moe_b);
    const uint h_f32 = pc.hist_f32_mask & 1u;
    const float x3 = inkling2_moe_read_f32(hist_in, 0ul * ulong(D) + col, h_f32);
    const float x2 = inkling2_moe_read_f32(hist_in, 1ul * ulong(D) + col, h_f32);
    const float x1 = inkling2_moe_read_f32(hist_in, 2ul * ulong(D) + col, h_f32);

    const uint w_f32 = pc.flags & 1u;
    const ulong wb = ulong(col) * 4ul;
    const float w3 = inkling2_moe_read_f32(post_w, wb + 0ul, w_f32);
    const float w2 = inkling2_moe_read_f32(post_w, wb + 1ul, w_f32);
    const float w1 = inkling2_moe_read_f32(post_w, wb + 2ul, w_f32);
    const float w0 = inkling2_moe_read_f32(post_w, wb + 3ul, w_f32);

    const float scale = 0.44721359549995793f;
    const float conv_acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
    const bfloat conv_val = bfloat(conv_acc * scale);
    if ((pc.flags & 2u) != 0u) {
      const uint r_f32 = (pc.flags >> 2u) & 1u;
      const float rv = inkling2_moe_read_f32(res_raw, ulong(col), r_f32);
      out[col] = bfloat(rv + float(conv_val));
    } else {
      out[col] = conv_val;
    }
    hist_out[0ul * ulong(D) + col] = bfloat(x2);
    hist_out[1ul * ulong(D) + col] = bfloat(x1);
    hist_out[2ul * ulong(D) + col] = moe_b;
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
  threadgroup uint totals[8];
  threadgroup uint16_t tg_sel[512];
  threadgroup atomic_uint simd_cnt;
  threadgroup atomic_uint nax_cnt;

  atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
  if (tid == 0u) {
    atomic_store_explicit(&simd_cnt, 0u, memory_order_relaxed);
    atomic_store_explicit(&nax_cnt, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total_routes = p.rows * p.top_k;
  const bool use_tg_sel = total_routes <= 512u;
  for (uint r = tid; r < total_routes; r += 256u) {
    const uint e = uint(selected[r]);
    if (use_tg_sel) {
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

  // 100% deterministic in-order route scatter per expert for ALL `total_routes` (zero atomics!).
  // Store token index (`r / p.top_k`) directly in `routes[pos]` so downstream gather / direct-gather
  // needs ZERO integer division (`udiv`)!
  if (tid < p.experts && count > 0u) {
    uint slot = 0u;
    const uint base_pos = start * p.tile_rows;
    if (use_tg_sel) {
      for (uint r = 0u; r < total_routes; ++r) {
        if (uint(tg_sel[r]) == tid) {
          const uint pos = base_pos + slot;
          routes[pos] = r / p.top_k;
          inverse[r] = pos;
          ++slot;
        }
      }
    } else {
      for (uint r = 0u; r < total_routes; ++r) {
        if (uint(selected[r]) == tid) {
          const uint pos = base_pos + slot;
          routes[pos] = r / p.top_k;
          inverse[r] = pos;
          ++slot;
        }
      }
    }
  }

  if (tid == 255u) {
    const uint live_tiles = start + nt;
    const uint s_tiles = atomic_load_explicit(&simd_cnt, memory_order_relaxed);
    const uint n_tiles = atomic_load_explicit(&nax_cnt, memory_order_relaxed);
    const bool gu_8col = (p.flags & 1u) != 0u;
    const bool dn_16col = (p.flags & 64u) != 0u;

    tile_count->total_tiles = live_tiles;
    tile_count->simd_tiles = s_tiles;
    tile_count->nax_tiles = n_tiles;

    tile_count->gather_grid[0] = live_tiles;
    tile_count->gather_grid[1] = (p.hidden + 255u) / 256u;
    tile_count->gather_grid[2] = 1u;

    const uint simd_gu_x = gu_8col ? (p.intermediate / 8u) : (p.intermediate / 4u);
    const uint simd_dn_x = dn_16col ? (p.hidden / 16u) : (p.hidden / 8u);
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

// Single-threadgroup fused `route + group` for T <= 32 (T*K <= 512, E <= 256).
// 8 SGs (256 threads) compute exact top-K sigmoid+bias routing for rows t = sg, sg+8, ...
// directly into threadgroup `tg_sel[512]` + histogram, then run prefix-sum, tile table,
// deterministic route scatter, and indirect-grid emit in 1 TG with ZERO inter-kernel barrier!
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_moe_route_group_small(
    device const uchar *logits_raw [[buffer(0)]],
    device const uchar *bias_raw [[buffer(1)]],
    device const uchar *down_scale_raw [[buffer(2)]],
    device float *out_gamma [[buffer(3)]],
    device Inkling2MoeTile *tiles [[buffer(4)]],
    device Inkling2MoeTileCount *tile_count [[buffer(5)]],
    device uint *routes [[buffer(6)]],
    device uint *inverse [[buffer(7)]],
    device uint *tile_lists [[buffer(8)]],
    constant Inkling2MoeRouteGroupParams &rp [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[256];
  threadgroup uint totals[8];
  threadgroup uint16_t tg_sel[512];
  threadgroup atomic_uint simd_cnt;
  threadgroup atomic_uint nax_cnt;

  atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
  if (tid == 0u) {
    atomic_store_explicit(&simd_cnt, 0u, memory_order_relaxed);
    atomic_store_explicit(&nax_cnt, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint E = rp.experts;
  const uint K = rp.top_k;
  const uint logits_f32 = (rp.flags >> 1u) & 1u;
  const bool has_bias = ((rp.flags >> 2u) & 1u) != 0u;
  const uint bias_f32 = (rp.flags >> 3u) & 1u;
  const bool has_down_scale = ((rp.flags >> 4u) & 1u) != 0u;
  const uint ds_f32 = (rp.flags >> 5u) & 1u;

  for (uint row = sg; row < rp.rows; row += 8u) {
    const ulong row_base = ulong(row) * ulong(rp.logits_stride);
    constexpr uint MAX_SLOTS = 8u; // 8 * 32 = 256 experts
    float local_s[MAX_SLOTS];
    float local_p[MAX_SLOTS];
    int32_t local_id[MAX_SLOTS];
    uint cnt = 0u;

    for (uint e = lane; e < E; e += 32u) {
      const float z = inkling2_moe_read_f32(logits_raw, row_base + e, logits_f32);
      const float prob = inkling2_moe_sigmoid(z);
      const float sc = has_bias
          ? (prob + inkling2_moe_read_f32(bias_raw, ulong(e), bias_f32))
          : prob;
      local_s[cnt] = sc;
      local_p[cnt] = prob;
      local_id[cnt] = int32_t(e);
      ++cnt;
    }

    int32_t sel_idx[16];
    float sel_p[16];
    float sum_p = 0.0f;

    for (uint k = 0u; k < K; ++k) {
      float best_s = -INFINITY;
      int32_t best_i = -1;
      float best_p = 0.0f;
      for (uint j = 0u; j < cnt; ++j) {
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

      for (uint j = 0u; j < cnt; ++j) {
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
      const uint out_base = row * K;
      for (uint k = 0u; k < K; ++k) {
        const uint eid = uint(sel_idx[k]);
        float w = (sel_p[k] / denom) * rp.route_scale;
        if (has_down_scale) {
          w *= inkling2_moe_read_f32(down_scale_raw, ulong(eid), ds_f32);
        }
        out_gamma[out_base + k] = w;
        tg_sel[out_base + k] = uint16_t(eid);
        if (eid < E) {
          atomic_fetch_add_explicit(counts + eid, 1u, memory_order_relaxed);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint count = tid < E
      ? atomic_load_explicit(counts + tid, memory_order_relaxed)
      : 0u;
  const uint nt = (count + rp.tile_rows - 1u) / rp.tile_rows;
  uint start = simd_prefix_exclusive_sum(nt);
  if (lane == 31u) {
    totals[sg] = start + nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0u; i < sg; ++i) {
    start += totals[i];
  }

  const bool use_simd = (count <= rp.col_cutoff);
  for (uint t = 0u; t < nt; ++t) {
    const uint tile_idx = start + t;
    const uint live = min(rp.tile_rows, count - t * rp.tile_rows);
    tiles[tile_idx] = {tid, live};
    if (use_simd) {
      const uint pos = atomic_fetch_add_explicit(&simd_cnt, 1u, memory_order_relaxed);
      tile_lists[pos] = tile_idx;
    } else {
      const uint pos = atomic_fetch_add_explicit(&nax_cnt, 1u, memory_order_relaxed);
      tile_lists[rp.max_tiles + pos] = tile_idx;
    }
  }
  if (nt != 0u) {
    const uint last = start + nt - 1u;
    const uint live = count - (nt - 1u) * rp.tile_rows;
    const uint pad_end = last * rp.tile_rows + inkling2_matmul_rows(live, rp.tile_rows);
    for (uint r = start * rp.tile_rows + count; r < pad_end; ++r) {
      routes[r] = ~0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total_routes = rp.rows * K;
  if (tid < E && count > 0u) {
    uint slot = 0u;
    const uint base_pos = start * rp.tile_rows;
    for (uint r = 0u; r < total_routes; ++r) {
      if (uint(tg_sel[r]) == tid) {
        const uint pos = base_pos + slot;
        routes[pos] = r / K;
        inverse[r] = pos;
        ++slot;
      }
    }
  }

  if (tid == 255u) {
    const uint live_tiles = start + nt;
    const uint s_tiles = atomic_load_explicit(&simd_cnt, memory_order_relaxed);
    const uint n_tiles = atomic_load_explicit(&nax_cnt, memory_order_relaxed);
    const bool gu_8col = (rp.flags & 1u) != 0u;
    const bool dn_16col = (rp.flags & 64u) != 0u;

    tile_count->total_tiles = live_tiles;
    tile_count->simd_tiles = s_tiles;
    tile_count->nax_tiles = n_tiles;

    tile_count->gather_grid[0] = live_tiles;
    tile_count->gather_grid[1] = (rp.hidden + 255u) / 256u;
    tile_count->gather_grid[2] = 1u;

    const uint simd_gu_x = gu_8col ? (rp.intermediate / 8u) : (rp.intermediate / 4u);
    const uint simd_dn_x = dn_16col ? (rp.hidden / 16u) : (rp.hidden / 8u);
    tile_count->simd_gu_grid[0] = s_tiles > 0u ? simd_gu_x : 0u;
    tile_count->simd_gu_grid[1] = s_tiles;
    tile_count->simd_gu_grid[2] = 1u;

    tile_count->simd_dn_grid[0] = s_tiles > 0u ? simd_dn_x : 0u;
    tile_count->simd_dn_grid[1] = s_tiles;
    tile_count->simd_dn_grid[2] = 1u;

    tile_count->nax_gu_grid[0] = n_tiles > 0u ? (rp.intermediate / 32u) : 0u;
    tile_count->nax_gu_grid[1] = n_tiles;
    tile_count->nax_gu_grid[2] = 1u;

    tile_count->nax_dn_grid[0] = n_tiles > 0u ? (rp.hidden / 64u) : 0u;
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
    const uint tok = routes[dst_row];
    *reinterpret_cast<device uint4 *>(out + ulong(dst_row) * p.hidden + col) =
        tok == ~0u
            ? uint4(0u)
            : *reinterpret_cast<device const uint4 *>(
                  x + ulong(tok) * p.hidden + col);
  }
}

// ============================================================================
// 3. Coalesced SIMD Multi-Row Kernels (1..8 rows/expert)
//    - Direct-gather from L1-resident `x` via `routes` (zero udiv, 32B tg_rx smem)
//    - 0-ULP branchless half4 ALU FP4 decode, single x-load for ALL n_rows=1..8
//    - Hoisted weight loads + parallel 4x uint4 activation loads for Q8_0 & MXFP8
//    - Exact same FP32 reduction order as M=1 -> 0-ULP bit-exact M=1..32!
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_gate_up_situ_nvfp4(
    device const bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uint *routes [[buffer(4)]],
    device const uchar *gu_w [[buffer(5)]],
    device const uint8_t *gu_s [[buffer(6)]],
    device const float *gu_outer [[buffer(7)]],
    device bfloat *hidden [[buffer(8)]],
    constant Inkling2MoeParams &p [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint tile_row_base = tile_idx * p.tile_rows;
  const bool direct_gather = (p.flags & 16u) != 0u;

  threadgroup uint tg_rx[8];
  if (n_rows > 1u) {
    const uint tid = sg * 32u + lane;
    if (tid < n_rows) {
      tg_rx[tid] = direct_gather ? routes[tile_row_base + tid] : (tile_row_base + tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr uint kLanes = 16u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 4u; // 0..1
  const uint col = group.x * 4u + sg * 2u + r_id;
  if (col >= I) return;

  const uint N2 = I * 2u;
  const uint n_groups = K >> 4u;
  const ulong row_w_stride = ulong(K >> 1u);
  const ulong row_s_stride = ulong(n_groups);
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (I + col);

  const device uint2 *gw_u2 = reinterpret_cast<device const uint2 *>(
      gu_w + (exp_row_base + g_row) * row_w_stride);
  const device uint8_t *gs_row = gu_s + (exp_row_base + g_row) * row_s_stride;
  const device uint2 *uw_u2 = reinterpret_cast<device const uint2 *>(
      gu_w + (exp_row_base + u_row) * row_w_stride);
  const device uint8_t *us_row = gu_s + (exp_row_base + u_row) * row_s_stride;

  const ulong out_tile_base = ulong(tile_row_base) * I;
  const float outer_s = (p.flags & 2u) ? gu_outer[t.expert] : 1.0f;
  const bool round_bf16 = (p.flags & 8u) != 0u;

  if (n_rows == 1u) {
    const uint rx0 = direct_gather ? routes[tile_row_base] : tile_row_base;
    const device bfloat *xp0 = x + ulong(rx0) * K;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      float4 xv[4];
      load_bf16_16(xp0 + (g << 4u), xv);

      half4 wv[4];
      decode_nvfp4_16_half_unscaled(gw_u2[g], wv);
      acc_g += inkling2_decode_fp8_e4m3_x16384(gs_row[g]) * dot_4x4_half(wv, xv);

      decode_nvfp4_16_half_unscaled(uw_u2[g], wv);
      acc_u += inkling2_decode_fp8_e4m3_x16384(us_row[g]) * dot_4x4_half(wv, xv);
    }
    acc_g = reduce_k_lanes<16>(acc_g);
    acc_u = reduce_k_lanes<16>(acc_u);
    if (k_lane == 0u) {
      hidden[out_tile_base + col] = inkling2_moe_situ_epilogue(
          acc_g, acc_u, outer_s, p.gate_softcap, p.up_softcap, round_bf16);
    }
    return;
  }

#define NVFP4_GU_ROW_UNCOND(IDX, AG, AU) \
  { \
    float4 xv[4]; \
    load_bf16_16(x + ulong(tg_rx[(IDX)]) * K + k0, xv); \
    (AG) += gs * dot_4x4_half(wg, xv); \
    (AU) += us * dot_4x4_half(wu, xv); \
  }

#define NVFP4_GU_ROW(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    NVFP4_GU_ROW_UNCOND((IDX), (AG), (AU)); \
  }

#define NVFP4_GU_STORE(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    const float g_sum = reduce_k_lanes<16>((AG)); \
    const float u_sum = reduce_k_lanes<16>((AU)); \
    if (k_lane == 0u) { \
      hidden[out_tile_base + ulong(IDX) * I + col] = inkling2_moe_situ_epilogue( \
          g_sum, u_sum, outer_s, p.gate_softcap, p.up_softcap, round_bf16); \
    } \
  }

  if (n_rows <= 4u) {
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_decode_fp8_e4m3_x16384(gs_row[g]);
      const float us = inkling2_decode_fp8_e4m3_x16384(us_row[g]);
      half4 wg[4], wu[4];
      decode_nvfp4_16_half_unscaled(gw_u2[g], wg);
      decode_nvfp4_16_half_unscaled(uw_u2[g], wu);
      const uint k0 = g << 4u;
      NVFP4_GU_ROW_UNCOND(0u, ag0, au0);
      NVFP4_GU_ROW_UNCOND(1u, ag1, au1);
      NVFP4_GU_ROW(2u, ag2, au2);
      NVFP4_GU_ROW(3u, ag3, au3);
    }
    NVFP4_GU_STORE(0u, ag0, au0);
    NVFP4_GU_STORE(1u, ag1, au1);
    NVFP4_GU_STORE(2u, ag2, au2);
    NVFP4_GU_STORE(3u, ag3, au3);
    return;
  }

  // 5..8 rows: half4 wg[4]+wu[4] = only 16 regs total -> single x-load per row & 0 alloca!
  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float ag4 = 0.0f, ag5 = 0.0f, ag6 = 0.0f, ag7 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
  float au4 = 0.0f, au5 = 0.0f, au6 = 0.0f, au7 = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const float gs = inkling2_decode_fp8_e4m3_x16384(gs_row[g]);
    const float us = inkling2_decode_fp8_e4m3_x16384(us_row[g]);
    half4 wg[4], wu[4];
    decode_nvfp4_16_half_unscaled(gw_u2[g], wg);
    decode_nvfp4_16_half_unscaled(uw_u2[g], wu);
    const uint k0 = g << 4u;
    NVFP4_GU_ROW_UNCOND(0u, ag0, au0);
    NVFP4_GU_ROW_UNCOND(1u, ag1, au1);
    NVFP4_GU_ROW_UNCOND(2u, ag2, au2);
    NVFP4_GU_ROW_UNCOND(3u, ag3, au3);
    NVFP4_GU_ROW_UNCOND(4u, ag4, au4);
    NVFP4_GU_ROW(5u, ag5, au5);
    NVFP4_GU_ROW(6u, ag6, au6);
    NVFP4_GU_ROW(7u, ag7, au7);
  }
  NVFP4_GU_STORE(0u, ag0, au0);
  NVFP4_GU_STORE(1u, ag1, au1);
  NVFP4_GU_STORE(2u, ag2, au2);
  NVFP4_GU_STORE(3u, ag3, au3);
  NVFP4_GU_STORE(4u, ag4, au4);
  NVFP4_GU_STORE(5u, ag5, au5);
  NVFP4_GU_STORE(6u, ag6, au6);
  NVFP4_GU_STORE(7u, ag7, au7);
#undef NVFP4_GU_ROW_UNCOND
#undef NVFP4_GU_ROW
#undef NVFP4_GU_STORE
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_gate_up_situ_q8(
    device const bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uint *routes [[buffer(4)]],
    device const uchar *gu_w [[buffer(5)]],
    device const uchar *gu_s [[buffer(6)]],
    device const uchar *gu_b [[buffer(7)]],
    device bfloat *hidden [[buffer(8)]],
    constant Inkling2MoeParams &p [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint tile_row_base = tile_idx * p.tile_rows;
  const bool direct_gather = (p.flags & 16u) != 0u;

  threadgroup uint tg_rx[8];
  if (n_rows > 1u) {
    const uint tid = sg * 32u + lane;
    if (tid < n_rows) {
      tg_rx[tid] = direct_gather ? routes[tile_row_base + tid] : (tile_row_base + tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr uint kLanes = 8u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group.x * 8u + sg * 4u + r_id;
  if (col >= I) return;

  const uint N2 = I * 2u;
  const uint n_groups = K >> 5u;
  const bool sidecar_f32 = (p.flags & 4u) != 0u;
  const bool round_bf16 = (p.flags & 8u) != 0u;
  const bool sym_q8 = (p.flags & 128u) != 0u;
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (I + col);

  const device uint4 *gw_u4 = reinterpret_cast<device const uint4 *>(
      gu_w + (exp_row_base + g_row) * ulong(K));
  const ulong gs_base = (exp_row_base + g_row) * ulong(n_groups);
  const device uint4 *uw_u4 = reinterpret_cast<device const uint4 *>(
      gu_w + (exp_row_base + u_row) * ulong(K));
  const ulong us_base = (exp_row_base + u_row) * ulong(n_groups);

  const ulong out_tile_base = ulong(tile_row_base) * I;

  if (n_rows == 1u) {
    const uint rx0 = direct_gather ? routes[tile_row_base] : tile_row_base;
    const device bfloat *xp0 = x + ulong(rx0) * K;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (sym_q8) {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float gs = inkling2_read_f16_scale(gu_s, gs_base + g);
        const float us = inkling2_read_f16_scale(gu_s, us_base + g);
        const uint4 gp0 = gw_u4[g * 2u];
        const uint4 gp1 = gw_u4[g * 2u + 1u];
        const uint4 up0 = uw_u4[g * 2u];
        const uint4 up1 = uw_u4[g * 2u + 1u];
        float4 x0[4], x1[4];
        load_bf16_32(xp0 + (g << 5u), x0, x1);
        float gd = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, gp0, x0), gp1, x1);
        float ud = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, up0, x0), up1, x1);
        acc_g += gs * gd;
        acc_u += us * ud;
      }
    } else {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
        const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
        const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
        const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
        const uint4 gp0 = gw_u4[g * 2u];
        const uint4 gp1 = gw_u4[g * 2u + 1u];
        const uint4 up0 = uw_u4[g * 2u];
        const uint4 up1 = uw_u4[g * 2u + 1u];
        float4 x0[4], x1[4];
        load_bf16_32(xp0 + (g << 5u), x0, x1);
        const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
        const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
        const float gd = dot_q8_16_accum(dot_q8_16_accum(0.0f, gp0, x0), gp1, x1);
        const float ud = dot_q8_16_accum(dot_q8_16_accum(0.0f, up0, x0), up1, x1);
        acc_g += gs * gd + gb * x_sum;
        acc_u += us * ud + ub * x_sum;
      }
    }
    acc_g = reduce_k_lanes<8>(acc_g);
    acc_u = reduce_k_lanes<8>(acc_u);
    if (k_lane == 0u) {
      hidden[out_tile_base + col] = inkling2_moe_situ_epilogue(
          acc_g, acc_u, 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
    }
    return;
  }

#define Q8_GU_ROW_SYM(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x + ulong(tg_rx[(IDX)]) * K + k0, x0, x1); \
    const float gd = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, gp0, x0), gp1, x1); \
    const float ud = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, up0, x0), up1, x1); \
    (AG) += gs * gd; \
    (AU) += us * ud; \
  }

#define Q8_GU_ROW_AFF(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x + ulong(tg_rx[(IDX)]) * K + k0, x0, x1); \
    const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1); \
    const float x_sum = (s4.x + s4.y) + (s4.z + s4.w); \
    const float gd = dot_q8_16_accum(dot_q8_16_accum(0.0f, gp0, x0), gp1, x1); \
    const float ud = dot_q8_16_accum(dot_q8_16_accum(0.0f, up0, x0), up1, x1); \
    (AG) += gs * gd + gb * x_sum; \
    (AU) += us * ud + ub * x_sum; \
  }

#define Q8_GU_STORE(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    const float g_sum = reduce_k_lanes<8>((AG)); \
    const float u_sum = reduce_k_lanes<8>((AU)); \
    if (k_lane == 0u) { \
      hidden[out_tile_base + ulong(IDX) * I + col] = inkling2_moe_situ_epilogue( \
          g_sum, u_sum, 1.0f, p.gate_softcap, p.up_softcap, round_bf16); \
    } \
  }

  // Hoisted weight loads (`gp0, gp1, up0, up1`) + parallel 4x uint4 activation loads + 0 alloca!
  if (n_rows <= 4u) {
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
    if (sym_q8) {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float gs = inkling2_read_f16_scale(gu_s, gs_base + g);
        const float us = inkling2_read_f16_scale(gu_s, us_base + g);
        const uint4 gp0 = gw_u4[g * 2u];
        const uint4 gp1 = gw_u4[g * 2u + 1u];
        const uint4 up0 = uw_u4[g * 2u];
        const uint4 up1 = uw_u4[g * 2u + 1u];
        const uint k0 = g << 5u;
        Q8_GU_ROW_SYM(0u, ag0, au0);
        Q8_GU_ROW_SYM(1u, ag1, au1);
        Q8_GU_ROW_SYM(2u, ag2, au2);
        Q8_GU_ROW_SYM(3u, ag3, au3);
      }
    } else {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
        const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
        const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
        const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
        const uint4 gp0 = gw_u4[g * 2u];
        const uint4 gp1 = gw_u4[g * 2u + 1u];
        const uint4 up0 = uw_u4[g * 2u];
        const uint4 up1 = uw_u4[g * 2u + 1u];
        const uint k0 = g << 5u;
        Q8_GU_ROW_AFF(0u, ag0, au0);
        Q8_GU_ROW_AFF(1u, ag1, au1);
        Q8_GU_ROW_AFF(2u, ag2, au2);
        Q8_GU_ROW_AFF(3u, ag3, au3);
      }
    }
    Q8_GU_STORE(0u, ag0, au0);
    Q8_GU_STORE(1u, ag1, au1);
    Q8_GU_STORE(2u, ag2, au2);
    Q8_GU_STORE(3u, ag3, au3);
    return;
  }

  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float ag4 = 0.0f, ag5 = 0.0f, ag6 = 0.0f, ag7 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
  float au4 = 0.0f, au5 = 0.0f, au6 = 0.0f, au7 = 0.0f;
  if (sym_q8) {
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_read_f16_scale(gu_s, gs_base + g);
      const float us = inkling2_read_f16_scale(gu_s, us_base + g);
      const uint4 gp0 = gw_u4[g * 2u];
      const uint4 gp1 = gw_u4[g * 2u + 1u];
      const uint4 up0 = uw_u4[g * 2u];
      const uint4 up1 = uw_u4[g * 2u + 1u];
      const uint k0 = g << 5u;
      Q8_GU_ROW_SYM(0u, ag0, au0);
      Q8_GU_ROW_SYM(1u, ag1, au1);
      Q8_GU_ROW_SYM(2u, ag2, au2);
      Q8_GU_ROW_SYM(3u, ag3, au3);
      Q8_GU_ROW_SYM(4u, ag4, au4);
      Q8_GU_ROW_SYM(5u, ag5, au5);
      Q8_GU_ROW_SYM(6u, ag6, au6);
      Q8_GU_ROW_SYM(7u, ag7, au7);
    }
  } else {
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_read_sidecar(gu_s, gs_base + g, sidecar_f32);
      const float gb = inkling2_read_sidecar(gu_b, gs_base + g, sidecar_f32);
      const float us = inkling2_read_sidecar(gu_s, us_base + g, sidecar_f32);
      const float ub = inkling2_read_sidecar(gu_b, us_base + g, sidecar_f32);
      const uint4 gp0 = gw_u4[g * 2u];
      const uint4 gp1 = gw_u4[g * 2u + 1u];
      const uint4 up0 = uw_u4[g * 2u];
      const uint4 up1 = uw_u4[g * 2u + 1u];
      const uint k0 = g << 5u;
      Q8_GU_ROW_AFF(0u, ag0, au0);
      Q8_GU_ROW_AFF(1u, ag1, au1);
      Q8_GU_ROW_AFF(2u, ag2, au2);
      Q8_GU_ROW_AFF(3u, ag3, au3);
      Q8_GU_ROW_AFF(4u, ag4, au4);
      Q8_GU_ROW_AFF(5u, ag5, au5);
      Q8_GU_ROW_AFF(6u, ag6, au6);
      Q8_GU_ROW_AFF(7u, ag7, au7);
    }
  }
  Q8_GU_STORE(0u, ag0, au0);
  Q8_GU_STORE(1u, ag1, au1);
  Q8_GU_STORE(2u, ag2, au2);
  Q8_GU_STORE(3u, ag3, au3);
  Q8_GU_STORE(4u, ag4, au4);
  Q8_GU_STORE(5u, ag5, au5);
  Q8_GU_STORE(6u, ag6, au6);
  Q8_GU_STORE(7u, ag7, au7);
#undef Q8_GU_ROW_SYM
#undef Q8_GU_ROW_AFF
#undef Q8_GU_STORE
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_gate_up_situ_mxfp8(
    device const bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uint *routes [[buffer(4)]],
    device const uchar *gu_w [[buffer(5)]],
    device const uint8_t *gu_s [[buffer(6)]],
    device bfloat *hidden [[buffer(7)]],
    constant Inkling2MoeParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint tile_row_base = tile_idx * p.tile_rows;
  const bool direct_gather = (p.flags & 16u) != 0u;

  threadgroup uint tg_rx[8];
  if (n_rows > 1u) {
    const uint tid = sg * 32u + lane;
    if (tid < n_rows) {
      tg_rx[tid] = direct_gather ? routes[tile_row_base + tid] : (tile_row_base + tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr uint kLanes = 8u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group.x * 8u + sg * 4u + r_id;
  if (col >= I) return;

  const uint N2 = I * 2u;
  const uint n_groups = K >> 5u;
  const bool round_bf16 = (p.flags & 8u) != 0u;
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (I + col);

  const device uchar *gw_row = gu_w + (exp_row_base + g_row) * ulong(K);
  const device uint8_t *gs_row = gu_s + (exp_row_base + g_row) * ulong(n_groups);
  const device uchar *uw_row = gu_w + (exp_row_base + u_row) * ulong(K);
  const device uint8_t *us_row = gu_s + (exp_row_base + u_row) * ulong(n_groups);

  const ulong out_tile_base = ulong(tile_row_base) * I;

  if (n_rows == 1u) {
    const uint rx0 = direct_gather ? routes[tile_row_base] : tile_row_base;
    const device bfloat *xp0 = x + ulong(rx0) * K;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_decode_fp8_e8m0(gs_row[g]);
      const float us = inkling2_decode_fp8_e8m0(us_row[g]);
      half4 gw[8], uw[8];
      decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul), gw);
      decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul), uw);
      float4 x0[4], x1[4];
      load_bf16_32(xp0 + (g << 5u), x0, x1);
      acc_g += gs * dot_mxfp8_32_regs(gw, x0, x1);
      acc_u += us * dot_mxfp8_32_regs(uw, x0, x1);
    }
    acc_g = reduce_k_lanes<8>(acc_g);
    acc_u = reduce_k_lanes<8>(acc_u);
    if (k_lane == 0u) {
      hidden[out_tile_base + col] = inkling2_moe_situ_epilogue(
          acc_g, acc_u, 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
    }
    return;
  }

#define MXFP8_GU_ROW4(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x + ulong(tg_rx[(IDX)]) * K + k0, x0, x1); \
    (AG) += gs * dot_mxfp8_32_regs(gw, x0, x1); \
    (AU) += us * dot_mxfp8_32_regs(uw, x0, x1); \
  }

#define MXFP8_GU_ONE(IDX, ACC, SC) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x + ulong(tg_rx[(IDX)]) * K + k0, x0, x1); \
    (ACC) += (SC) * dot_mxfp8_32_regs(w, x0, x1); \
  }

#define MXFP8_GU_STORE(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    const float g_sum = reduce_k_lanes<8>((AG)); \
    const float u_sum = reduce_k_lanes<8>((AU)); \
    if (k_lane == 0u) { \
      hidden[out_tile_base + ulong(IDX) * I + col] = inkling2_moe_situ_epilogue( \
          g_sum, u_sum, 1.0f, p.gate_softcap, p.up_softcap, round_bf16); \
    } \
  }

  if (n_rows <= 4u) {
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float gs = inkling2_decode_fp8_e8m0(gs_row[g]);
      const float us = inkling2_decode_fp8_e8m0(us_row[g]);
      half4 gw[8], uw[8];
      decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul), gw);
      decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul), uw);
      const uint k0 = g << 5u;
      MXFP8_GU_ROW4(0u, ag0, au0);
      MXFP8_GU_ROW4(1u, ag1, au1);
      MXFP8_GU_ROW4(2u, ag2, au2);
      MXFP8_GU_ROW4(3u, ag3, au3);
    }
    MXFP8_GU_STORE(0u, ag0, au0);
    MXFP8_GU_STORE(1u, ag1, au1);
    MXFP8_GU_STORE(2u, ag2, au2);
    MXFP8_GU_STORE(3u, ag3, au3);
    return;
  }

  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float ag4 = 0.0f, ag5 = 0.0f, ag6 = 0.0f, ag7 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
  float au4 = 0.0f, au5 = 0.0f, au6 = 0.0f, au7 = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const uint k0 = g << 5u;
    half4 w[8];
    const float gs = inkling2_decode_fp8_e8m0(gs_row[g]);
    decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(gw_row + ulong(g) * 32ul), w);
    MXFP8_GU_ONE(0u, ag0, gs);
    MXFP8_GU_ONE(1u, ag1, gs);
    MXFP8_GU_ONE(2u, ag2, gs);
    MXFP8_GU_ONE(3u, ag3, gs);
    MXFP8_GU_ONE(4u, ag4, gs);
    MXFP8_GU_ONE(5u, ag5, gs);
    MXFP8_GU_ONE(6u, ag6, gs);
    MXFP8_GU_ONE(7u, ag7, gs);

    const float us = inkling2_decode_fp8_e8m0(us_row[g]);
    decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(uw_row + ulong(g) * 32ul), w);
    MXFP8_GU_ONE(0u, au0, us);
    MXFP8_GU_ONE(1u, au1, us);
    MXFP8_GU_ONE(2u, au2, us);
    MXFP8_GU_ONE(3u, au3, us);
    MXFP8_GU_ONE(4u, au4, us);
    MXFP8_GU_ONE(5u, au5, us);
    MXFP8_GU_ONE(6u, au6, us);
    MXFP8_GU_ONE(7u, au7, us);
  }
  MXFP8_GU_STORE(0u, ag0, au0);
  MXFP8_GU_STORE(1u, ag1, au1);
  MXFP8_GU_STORE(2u, ag2, au2);
  MXFP8_GU_STORE(3u, ag3, au3);
  MXFP8_GU_STORE(4u, ag4, au4);
  MXFP8_GU_STORE(5u, ag5, au5);
  MXFP8_GU_STORE(6u, ag6, au6);
  MXFP8_GU_STORE(7u, ag7, au7);
#undef MXFP8_GU_ROW4
#undef MXFP8_GU_ONE
#undef MXFP8_GU_STORE
}

// Coalesced SIMD multi-row RowMajor Q5_K gate_up + SiTU (1..8 rows/expert, direct-gather).
// Each 256-elem super-block (8 sub-blocks across k_lane = 0..7) is loaded and unpacked ONCE
// per (expert, col_pair) OUTSIDE the row loop and dotted across all n_rows in registers!
[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_gate_up_situ_q5k(
    device const bfloat *x [[buffer(0)]],
    device const Inkling2MoeTile *tiles [[buffer(1)]],
    device const uint *simd_list [[buffer(2)]],
    device const Inkling2MoeTileCount *count [[buffer(3)]],
    device const uint *routes [[buffer(4)]],
    device const uint *gu_w [[buffer(5)]],
    device const uchar *gu_s [[buffer(6)]],
    device const half *gu_b [[buffer(7)]],
    device bfloat *hidden [[buffer(8)]],
    constant Inkling2MoeParams &p [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= count->simd_tiles) return;
  const uint tile_idx = simd_list[group.y];
  const Inkling2MoeTile t = tiles[tile_idx];
  const uint n_rows = t.rows;
  const uint K = p.hidden;
  const uint I = p.intermediate;
  const uint tile_row_base = tile_idx * p.tile_rows;
  const bool direct_gather = (p.flags & 16u) != 0u;

  threadgroup uint tg_rx[8];
  if (n_rows > 1u) {
    const uint tid = sg * 32u + lane;
    if (tid < n_rows) {
      tg_rx[tid] = direct_gather ? routes[tile_row_base + tid] : (tile_row_base + tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  constexpr uint kLanes = 8u;
  const uint k_lane = lane & (kLanes - 1u);
  const uint r_id = lane >> 3u; // 0..3
  const uint col = group.x * 8u + sg * 4u + r_id;
  if (col >= I) return;

  const uint N2 = I * 2u;
  const uint n_groups = K >> 5u;
  const uint n_super = K >> 8u;
  const ulong row_w_words = ulong(n_groups) * 5ul;
  const bool round_bf16 = (p.flags & 8u) != 0u;
  const ulong exp_row_base = ulong(t.expert) * ulong(N2);
  const bool interleaved = (p.flags & 32u) != 0u;
  const uint g_row = interleaved ? (2u * col) : col;
  const uint u_row = interleaved ? (2u * col + 1u) : (I + col);

  const device uint *gw_row = gu_w + (exp_row_base + g_row) * row_w_words;
  const device uchar2 *gs_row =
      reinterpret_cast<device const uchar2 *>(gu_s) + (exp_row_base + g_row) * ulong(n_groups);
  const device half2 *gb_row =
      reinterpret_cast<device const half2 *>(gu_b) + (exp_row_base + g_row) * ulong(n_super);

  const device uint *uw_row = gu_w + (exp_row_base + u_row) * row_w_words;
  const device uchar2 *us_row =
      reinterpret_cast<device const uchar2 *>(gu_s) + (exp_row_base + u_row) * ulong(n_groups);
  const device half2 *ub_row =
      reinterpret_cast<device const half2 *>(gu_b) + (exp_row_base + u_row) * ulong(n_super);

  const ulong out_tile_base = ulong(tile_row_base) * I;

  if (n_rows == 1u) {
    const uint rx0 = direct_gather ? routes[tile_row_base] : tile_row_base;
    const device bfloat *xp0 = x + ulong(rx0) * K;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint sb = 0u; sb < n_super; ++sb) {
      const uint g = (sb << 3u) + k_lane;
      const half2 gdm = gb_row[sb];
      const half2 udm = ub_row[sb];
      const uchar2 gsm = gs_row[g];
      const uchar2 usm = us_row[g];
      const float gs = float(gdm.x) * float(gsm.x);
      const float gb = -(float(gdm.y) * float(gsm.y));
      const float us = float(udm.x) * float(usm.x);
      const float ub = -(float(udm.y) * float(usm.y));
      uchar4 gw[8], uw[8];
      decode_q5k_32_regs(gw_row + ulong(g) * 5ul, gw);
      decode_q5k_32_regs(uw_row + ulong(g) * 5ul, uw);
      float4 x0[4], x1[4];
      load_bf16_32(xp0 + (g << 5u), x0, x1);
      const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
      const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
      acc_g += gs * dot_q5k_32_regs(gw, x0, x1) + gb * x_sum;
      acc_u += us * dot_q5k_32_regs(uw, x0, x1) + ub * x_sum;
    }
    acc_g = reduce_k_lanes<8>(acc_g);
    acc_u = reduce_k_lanes<8>(acc_u);
    if (k_lane == 0u) {
      hidden[out_tile_base + col] = inkling2_moe_situ_epilogue(
          acc_g, acc_u, 1.0f, p.gate_softcap, p.up_softcap, round_bf16);
    }
    return;
  }

#define Q5K_GU_ROW(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x + ulong(tg_rx[(IDX)]) * K + k0, x0, x1); \
    const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1); \
    const float x_sum = (s4.x + s4.y) + (s4.z + s4.w); \
    (AG) += gs * dot_q5k_32_regs(gw, x0, x1) + gb * x_sum; \
    (AU) += us * dot_q5k_32_regs(uw, x0, x1) + ub * x_sum; \
  }

#define Q5K_GU_STORE(IDX, AG, AU) \
  if ((IDX) < n_rows) { \
    const float g_sum = reduce_k_lanes<8>((AG)); \
    const float u_sum = reduce_k_lanes<8>((AU)); \
    if (k_lane == 0u) { \
      hidden[out_tile_base + ulong(IDX) * I + col] = inkling2_moe_situ_epilogue( \
          g_sum, u_sum, 1.0f, p.gate_softcap, p.up_softcap, round_bf16); \
    } \
  }

  if (n_rows <= 4u) {
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
    for (uint sb = 0u; sb < n_super; ++sb) {
      const uint g = (sb << 3u) + k_lane;
      const half2 gdm = gb_row[sb];
      const half2 udm = ub_row[sb];
      const uchar2 gsm = gs_row[g];
      const uchar2 usm = us_row[g];
      const float gs = float(gdm.x) * float(gsm.x);
      const float gb = -(float(gdm.y) * float(gsm.y));
      const float us = float(udm.x) * float(usm.x);
      const float ub = -(float(udm.y) * float(usm.y));
      uchar4 gw[8], uw[8];
      decode_q5k_32_regs(gw_row + ulong(g) * 5ul, gw);
      decode_q5k_32_regs(uw_row + ulong(g) * 5ul, uw);
      const uint k0 = g << 5u;
      Q5K_GU_ROW(0u, ag0, au0);
      Q5K_GU_ROW(1u, ag1, au1);
      Q5K_GU_ROW(2u, ag2, au2);
      Q5K_GU_ROW(3u, ag3, au3);
    }
    Q5K_GU_STORE(0u, ag0, au0);
    Q5K_GU_STORE(1u, ag1, au1);
    Q5K_GU_STORE(2u, ag2, au2);
    Q5K_GU_STORE(3u, ag3, au3);
    return;
  }

  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float ag4 = 0.0f, ag5 = 0.0f, ag6 = 0.0f, ag7 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
  float au4 = 0.0f, au5 = 0.0f, au6 = 0.0f, au7 = 0.0f;
  for (uint sb = 0u; sb < n_super; ++sb) {
    const uint g = (sb << 3u) + k_lane;
    const half2 gdm = gb_row[sb];
    const half2 udm = ub_row[sb];
    const uchar2 gsm = gs_row[g];
    const uchar2 usm = us_row[g];
    const float gs = float(gdm.x) * float(gsm.x);
    const float gb = -(float(gdm.y) * float(gsm.y));
    const float us = float(udm.x) * float(usm.x);
    const float ub = -(float(udm.y) * float(usm.y));
    uchar4 gw[8], uw[8];
    decode_q5k_32_regs(gw_row + ulong(g) * 5ul, gw);
    decode_q5k_32_regs(uw_row + ulong(g) * 5ul, uw);
    const uint k0 = g << 5u;
    Q5K_GU_ROW(0u, ag0, au0);
    Q5K_GU_ROW(1u, ag1, au1);
    Q5K_GU_ROW(2u, ag2, au2);
    Q5K_GU_ROW(3u, ag3, au3);
    Q5K_GU_ROW(4u, ag4, au4);
    Q5K_GU_ROW(5u, ag5, au5);
    Q5K_GU_ROW(6u, ag6, au6);
    Q5K_GU_ROW(7u, ag7, au7);
  }
  Q5K_GU_STORE(0u, ag0, au0);
  Q5K_GU_STORE(1u, ag1, au1);
  Q5K_GU_STORE(2u, ag2, au2);
  Q5K_GU_STORE(3u, ag3, au3);
  Q5K_GU_STORE(4u, ag4, au4);
  Q5K_GU_STORE(5u, ag5, au5);
  Q5K_GU_STORE(6u, ag6, au6);
  Q5K_GU_STORE(7u, ag7, au7);
#undef Q5K_GU_ROW
#undef Q5K_GU_STORE
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
  const device uint2 *w_u2 = reinterpret_cast<device const uint2 *>(
      dn_w + row_idx * ulong(K >> 1u));
  const device uint8_t *s_row = dn_s + row_idx * ulong(n_groups);

  const device bfloat *x_tile = hidden + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * D;

  if (n_rows == 1u) {
    float acc = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      float4 xv[4];
      half4 wv[4];
      load_bf16_16(x_tile + (g << 4u), xv);
      decode_nvfp4_16_half_unscaled(w_u2[g], wv);
      acc += inkling2_decode_fp8_e4m3_x16384(s_row[g]) * dot_4x4_half(wv, xv);
    }
    acc = reduce_k_lanes<8>(acc);
    if (k_lane == 0u) {
      projected[out_tile_base + col] = bfloat(acc);
    }
    return;
  }

#define NVFP4_DN_ROW_UNCOND(IDX, ACC) \
  { \
    float4 xv[4]; \
    load_bf16_16(x_tile + ulong(IDX) * K + k0, xv); \
    (ACC) += sv * dot_4x4_half(wv, xv); \
  }

#define NVFP4_DN_ROW(IDX, ACC) \
  if ((IDX) < n_rows) { \
    NVFP4_DN_ROW_UNCOND((IDX), (ACC)); \
  }

#define NVFP4_DN_STORE(IDX, ACC) \
  if ((IDX) < n_rows) { \
    const float sum = reduce_k_lanes<8>((ACC)); \
    if (k_lane == 0u) { \
      projected[out_tile_base + ulong(IDX) * D + col] = bfloat(sum); \
    } \
  }

  if (n_rows <= 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      half4 wv[4];
      decode_nvfp4_16_half_unscaled(w_u2[g], wv);
      const float sv = inkling2_decode_fp8_e4m3_x16384(s_row[g]);
      const uint k0 = g << 4u;
      NVFP4_DN_ROW_UNCOND(0u, a0);
      NVFP4_DN_ROW_UNCOND(1u, a1);
      NVFP4_DN_ROW(2u, a2);
      NVFP4_DN_ROW(3u, a3);
    }
    NVFP4_DN_STORE(0u, a0);
    NVFP4_DN_STORE(1u, a1);
    NVFP4_DN_STORE(2u, a2);
    NVFP4_DN_STORE(3u, a3);
    return;
  }

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    half4 wv[4];
    decode_nvfp4_16_half_unscaled(w_u2[g], wv);
    const float sv = inkling2_decode_fp8_e4m3_x16384(s_row[g]);
    const uint k0 = g << 4u;
    NVFP4_DN_ROW_UNCOND(0u, a0);
    NVFP4_DN_ROW_UNCOND(1u, a1);
    NVFP4_DN_ROW_UNCOND(2u, a2);
    NVFP4_DN_ROW_UNCOND(3u, a3);
    NVFP4_DN_ROW_UNCOND(4u, a4);
    NVFP4_DN_ROW(5u, a5);
    NVFP4_DN_ROW(6u, a6);
    NVFP4_DN_ROW(7u, a7);
  }
  NVFP4_DN_STORE(0u, a0);
  NVFP4_DN_STORE(1u, a1);
  NVFP4_DN_STORE(2u, a2);
  NVFP4_DN_STORE(3u, a3);
  NVFP4_DN_STORE(4u, a4);
  NVFP4_DN_STORE(5u, a5);
  NVFP4_DN_STORE(6u, a6);
  NVFP4_DN_STORE(7u, a7);
#undef NVFP4_DN_ROW_UNCOND
#undef NVFP4_DN_ROW
#undef NVFP4_DN_STORE
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
  const bool sym_q8 = (p.flags & 128u) != 0u;
  const ulong row_idx = ulong(t.expert) * ulong(D) + col;
  const device uint4 *w_u4 = reinterpret_cast<device const uint4 *>(dn_w + row_idx * ulong(K));
  const ulong s_base = row_idx * ulong(n_groups);

  const device bfloat *x_tile = hidden + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * D;

  if (n_rows == 1u) {
    float acc = 0.0f;
    if (sym_q8) {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float sc = inkling2_read_f16_scale(dn_s, s_base + g);
        const uint4 p0 = w_u4[g * 2u];
        const uint4 p1 = w_u4[g * 2u + 1u];
        float4 x0[4], x1[4];
        load_bf16_32(x_tile + (g << 5u), x0, x1);
        const float dp = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, p0, x0), p1, x1);
        acc += sc * dp;
      }
    } else {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
        const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
        const uint4 p0 = w_u4[g * 2u];
        const uint4 p1 = w_u4[g * 2u + 1u];
        float4 x0[4], x1[4];
        load_bf16_32(x_tile + (g << 5u), x0, x1);
        const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1);
        const float x_sum = (s4.x + s4.y) + (s4.z + s4.w);
        const float dp = dot_q8_16_accum(dot_q8_16_accum(0.0f, p0, x0), p1, x1);
        acc += sc * dp + bi * x_sum;
      }
    }
    acc = reduce_k_lanes<4>(acc);
    if (k_lane == 0u) {
      projected[out_tile_base + col] = bfloat(acc);
    }
    return;
  }

  // Hoisted weight loads (`p0, p1`) + parallel 4x uint4 activation loads + 0 alloca!
#define Q8_DN_ROW_SYM(IDX, ACC) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x_tile + ulong(IDX) * K + k0, x0, x1); \
    const float dp = dot_q8_sym_16_accum(dot_q8_sym_16_accum(0.0f, p0, x0), p1, x1); \
    (ACC) += sc * dp; \
  }

#define Q8_DN_ROW_AFF(IDX, ACC) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x_tile + ulong(IDX) * K + k0, x0, x1); \
    const float4 s4 = sum_bf16_16_v4(x0) + sum_bf16_16_v4(x1); \
    const float x_sum = (s4.x + s4.y) + (s4.z + s4.w); \
    const float dp = dot_q8_16_accum(dot_q8_16_accum(0.0f, p0, x0), p1, x1); \
    (ACC) += sc * dp + bi * x_sum; \
  }

#define Q8_DN_STORE(IDX, ACC) \
  if ((IDX) < n_rows) { \
    const float sum = reduce_k_lanes<4>((ACC)); \
    if (k_lane == 0u) { \
      projected[out_tile_base + ulong(IDX) * D + col] = bfloat(sum); \
    } \
  }

  if (n_rows <= 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (sym_q8) {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float sc = inkling2_read_f16_scale(dn_s, s_base + g);
        const uint4 p0 = w_u4[g * 2u];
        const uint4 p1 = w_u4[g * 2u + 1u];
        const uint k0 = g << 5u;
        Q8_DN_ROW_SYM(0u, a0);
        Q8_DN_ROW_SYM(1u, a1);
        Q8_DN_ROW_SYM(2u, a2);
        Q8_DN_ROW_SYM(3u, a3);
      }
    } else {
      for (uint g = k_lane; g < n_groups; g += kLanes) {
        const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
        const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
        const uint4 p0 = w_u4[g * 2u];
        const uint4 p1 = w_u4[g * 2u + 1u];
        const uint k0 = g << 5u;
        Q8_DN_ROW_AFF(0u, a0);
        Q8_DN_ROW_AFF(1u, a1);
        Q8_DN_ROW_AFF(2u, a2);
        Q8_DN_ROW_AFF(3u, a3);
      }
    }
    Q8_DN_STORE(0u, a0);
    Q8_DN_STORE(1u, a1);
    Q8_DN_STORE(2u, a2);
    Q8_DN_STORE(3u, a3);
    return;
  }

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  if (sym_q8) {
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float sc = inkling2_read_f16_scale(dn_s, s_base + g);
      const uint4 p0 = w_u4[g * 2u];
      const uint4 p1 = w_u4[g * 2u + 1u];
      const uint k0 = g << 5u;
      Q8_DN_ROW_SYM(0u, a0);
      Q8_DN_ROW_SYM(1u, a1);
      Q8_DN_ROW_SYM(2u, a2);
      Q8_DN_ROW_SYM(3u, a3);
      Q8_DN_ROW_SYM(4u, a4);
      Q8_DN_ROW_SYM(5u, a5);
      Q8_DN_ROW_SYM(6u, a6);
      Q8_DN_ROW_SYM(7u, a7);
    }
  } else {
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float sc = inkling2_read_sidecar(dn_s, s_base + g, sidecar_f32);
      const float bi = inkling2_read_sidecar(dn_b, s_base + g, sidecar_f32);
      const uint4 p0 = w_u4[g * 2u];
      const uint4 p1 = w_u4[g * 2u + 1u];
      const uint k0 = g << 5u;
      Q8_DN_ROW_AFF(0u, a0);
      Q8_DN_ROW_AFF(1u, a1);
      Q8_DN_ROW_AFF(2u, a2);
      Q8_DN_ROW_AFF(3u, a3);
      Q8_DN_ROW_AFF(4u, a4);
      Q8_DN_ROW_AFF(5u, a5);
      Q8_DN_ROW_AFF(6u, a6);
      Q8_DN_ROW_AFF(7u, a7);
    }
  }
  Q8_DN_STORE(0u, a0);
  Q8_DN_STORE(1u, a1);
  Q8_DN_STORE(2u, a2);
  Q8_DN_STORE(3u, a3);
  Q8_DN_STORE(4u, a4);
  Q8_DN_STORE(5u, a5);
  Q8_DN_STORE(6u, a6);
  Q8_DN_STORE(7u, a7);
#undef Q8_DN_ROW_SYM
#undef Q8_DN_ROW_AFF
#undef Q8_DN_STORE
}

[[kernel, max_total_threads_per_threadgroup(64)]] void
inkling2_moe_simd_down_mxfp8(
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
  const ulong row_idx = ulong(t.expert) * ulong(D) + col;
  const device uchar *w_row = dn_w + row_idx * ulong(K);
  const device uint8_t *s_row = dn_s + row_idx * ulong(n_groups);

  const device bfloat *x_tile = hidden + ulong(tile_idx) * p.tile_rows * K;
  const ulong out_tile_base = ulong(tile_idx) * p.tile_rows * D;

  if (n_rows == 1u) {
    float acc = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float sc = inkling2_decode_fp8_e8m0(s_row[g]);
      half4 w[8];
      decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul), w);
      float4 x0[4], x1[4];
      load_bf16_32(x_tile + (g << 5u), x0, x1);
      acc += sc * dot_mxfp8_32_regs(w, x0, x1);
    }
    acc = reduce_k_lanes<4>(acc);
    if (k_lane == 0u) {
      projected[out_tile_base + col] = bfloat(acc);
    }
    return;
  }

#define MXFP8_DN_ROW(IDX, ACC) \
  if ((IDX) < n_rows) { \
    float4 x0[4], x1[4]; \
    load_bf16_32(x_tile + ulong(IDX) * K + k0, x0, x1); \
    (ACC) += sc * dot_mxfp8_32_regs(w, x0, x1); \
  }

#define MXFP8_DN_STORE(IDX, ACC) \
  if ((IDX) < n_rows) { \
    const float sum = reduce_k_lanes<4>((ACC)); \
    if (k_lane == 0u) { \
      projected[out_tile_base + ulong(IDX) * D + col] = bfloat(sum); \
    } \
  }

  if (n_rows <= 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint g = k_lane; g < n_groups; g += kLanes) {
      const float sc = inkling2_decode_fp8_e8m0(s_row[g]);
      half4 w[8];
      decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul), w);
      const uint k0 = g << 5u;
      MXFP8_DN_ROW(0u, a0);
      MXFP8_DN_ROW(1u, a1);
      MXFP8_DN_ROW(2u, a2);
      MXFP8_DN_ROW(3u, a3);
    }
    MXFP8_DN_STORE(0u, a0);
    MXFP8_DN_STORE(1u, a1);
    MXFP8_DN_STORE(2u, a2);
    MXFP8_DN_STORE(3u, a3);
    return;
  }

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (uint g = k_lane; g < n_groups; g += kLanes) {
    const float sc = inkling2_decode_fp8_e8m0(s_row[g]);
    half4 w[8];
    decode_mxfp8_32_half_unscaled(reinterpret_cast<device const uint4 *>(w_row + ulong(g) * 32ul), w);
    const uint k0 = g << 5u;
    MXFP8_DN_ROW(0u, a0);
    MXFP8_DN_ROW(1u, a1);
    MXFP8_DN_ROW(2u, a2);
    MXFP8_DN_ROW(3u, a3);
    MXFP8_DN_ROW(4u, a4);
    MXFP8_DN_ROW(5u, a5);
    MXFP8_DN_ROW(6u, a6);
    MXFP8_DN_ROW(7u, a7);
  }
  MXFP8_DN_STORE(0u, a0);
  MXFP8_DN_STORE(1u, a1);
  MXFP8_DN_STORE(2u, a2);
  MXFP8_DN_STORE(3u, a3);
  MXFP8_DN_STORE(4u, a4);
  MXFP8_DN_STORE(5u, a5);
  MXFP8_DN_STORE(6u, a6);
  MXFP8_DN_STORE(7u, a7);
#undef MXFP8_DN_ROW
#undef MXFP8_DN_STORE
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

  const float s0_scaled = s0 * 16384.0f;
  half4 w0[4];
  decode_nvfp4_16_half_unscaled(uint2(packed.x, packed.y), w0);
  dst4[0] = bfloat4(float4(w0[0]) * s0_scaled);
  dst4[1] = bfloat4(float4(w0[1]) * s0_scaled);
  dst4[2] = bfloat4(float4(w0[2]) * s0_scaled);
  dst4[3] = bfloat4(float4(w0[3]) * s0_scaled);

  const float s1_scaled = s1 * 16384.0f;
  half4 w1[4];
  decode_nvfp4_16_half_unscaled(uint2(packed.z, packed.w), w1);
  dst4[4] = bfloat4(float4(w1[0]) * s1_scaled);
  dst4[5] = bfloat4(float4(w1[1]) * s1_scaled);
  dst4[6] = bfloat4(float4(w1[2]) * s1_scaled);
  dst4[7] = bfloat4(float4(w1[3]) * s1_scaled);
}

inline void stage_q8_32_pair_regs(
    uint4 p0,
    uint4 p1,
    float sc,
    float bi,
    bool sym_q8,
    threadgroup bfloat *dst_hi,
    threadgroup bfloat *dst_lo) {
  threadgroup bfloat4 *hi4 = reinterpret_cast<threadgroup bfloat4 *>(dst_hi);
  threadgroup bfloat4 *lo4 = reinterpret_cast<threadgroup bfloat4 *>(dst_lo);

  float4 v0, v1, v2, v3, v4, v5, v6, v7;
  if (sym_q8) {
    v0 = float4(as_type<char4>(p0.x ^ 0x80808080u)) * sc;
    v1 = float4(as_type<char4>(p0.y ^ 0x80808080u)) * sc;
    v2 = float4(as_type<char4>(p0.z ^ 0x80808080u)) * sc;
    v3 = float4(as_type<char4>(p0.w ^ 0x80808080u)) * sc;
    v4 = float4(as_type<char4>(p1.x ^ 0x80808080u)) * sc;
    v5 = float4(as_type<char4>(p1.y ^ 0x80808080u)) * sc;
    v6 = float4(as_type<char4>(p1.z ^ 0x80808080u)) * sc;
    v7 = float4(as_type<char4>(p1.w ^ 0x80808080u)) * sc;
  } else {
    v0 = fma(float4(as_type<uchar4>(p0.x)), float4(sc), float4(bi));
    v1 = fma(float4(as_type<uchar4>(p0.y)), float4(sc), float4(bi));
    v2 = fma(float4(as_type<uchar4>(p0.z)), float4(sc), float4(bi));
    v3 = fma(float4(as_type<uchar4>(p0.w)), float4(sc), float4(bi));
    v4 = fma(float4(as_type<uchar4>(p1.x)), float4(sc), float4(bi));
    v5 = fma(float4(as_type<uchar4>(p1.y)), float4(sc), float4(bi));
    v6 = fma(float4(as_type<uchar4>(p1.z)), float4(sc), float4(bi));
    v7 = fma(float4(as_type<uchar4>(p1.w)), float4(sc), float4(bi));
  }

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
    bool sym_q8,
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
      const float sc = sym_q8
          ? inkling2_read_f16_scale(s_raw, s_row_base + step)
          : inkling2_read_sidecar(s_raw, s_row_base + step, sidecar_f32);
      const float bi = sym_q8
          ? 0.0f
          : inkling2_read_sidecar(b_raw, s_row_base + step, sidecar_f32);
      stage_q8_32_pair_regs(
          p0, p1, sc, bi, sym_q8,
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
  const bool sym_q8 = (p.flags & 128u) != 0u;
  const float outer_s = (!IsQ8 && (p.flags & 2u))
      ? reinterpret_cast<device const float *>(gu_aux)[t.expert]
      : 1.0f;

  const bool interleaved = (p.flags & 32u) != 0u;
  const uint row_in_expert = interleaved
      ? (2u * (origin + lane) + (is_up ? 1u : 0u))
      : ((is_up ? I : 0u) + origin + lane);
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
        const float sc = sym_q8
            ? inkling2_read_f16_scale(gu_s, s_row_base + step)
            : inkling2_read_sidecar(gu_s, s_row_base + step, sidecar_f32);
        const float bi = sym_q8
            ? 0.0f
            : inkling2_read_sidecar(gu_aux, s_row_base + step, sidecar_f32);
        stage_q8_32_pair_regs(
            p0, p1, sc, bi, sym_q8,
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
  const bool sym_q8 = (p.flags & 128u) != 0u;

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
        sym_q8,
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

// Fused grouped weighted combine + static post_conv (k=4, ragged multi-seq) + optional FP32
// residual + hist_out + optional pre-conv tape write (for T <= 32 decode & speculative verify).
// Grid: (ceil(D / 32), num_seqs, 1), TG: (256, 1, 1) -> 8 SGs combine & convolve up to 8 tokens
// in parallel per chunk (72 TGs at M=8, 288 TGs at M=32; full M5 Max 40-core utilization!).
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling2_moe_grouped_combine_sconv_bf16(
    device const bfloat *projected [[buffer(0)]],
    device const uint *inverse [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const uchar *post_w [[buffer(3)]],
    device const int32_t *cu_seqlens [[buffer(4)]],
    device const uchar *res_raw [[buffer(5)]],
    device const uchar *h0 [[buffer(6)]],
    device const uchar *h1 [[buffer(7)]],
    device const uchar *h2 [[buffer(8)]],
    device const uchar *h3 [[buffer(9)]],
    device bfloat *y_out [[buffer(10)]],
    device bfloat *hist_out [[buffer(11)]],
    device bfloat *tape_out [[buffer(12)]],
    constant Inkling2MoePostConvParams &pc [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint seq_idx = group.y;
  if (seq_idx >= pc.num_seqs) return;

  const uint col = group.x * 32u + lane;
  const uint D = pc.hidden;
  const bool valid_col = col < D;

  const uint seq_start = (pc.num_seqs == 1u) ? 0u : uint(cu_seqlens[seq_idx]);
  const uint seq_end = (pc.num_seqs == 1u) ? pc.rows : uint(cu_seqlens[seq_idx + 1u]);

  device const uchar *h_ptr;
  ulong h_base;
  uint h_f32;
  if (pc.packed_hist != 0u) {
    h_ptr = h0;
    h_base = ulong(seq_idx) * 3ul * ulong(D);
    h_f32 = pc.hist_f32_mask & 1u;
  } else {
    h_ptr = (seq_idx == 0u) ? h0 : (seq_idx == 1u) ? h1 : (seq_idx == 2u) ? h2 : h3;
    h_base = 0ul;
    h_f32 = (pc.hist_f32_mask >> seq_idx) & 1u;
  }

  // Rows 0..2 hold the rolling 3-row carry history; rows 3..10 hold up to 8 combined MoE rows.
  threadgroup float tg_moe[11][32];
  if (sg < 3u && valid_col) {
    tg_moe[sg][lane] = inkling2_moe_read_f32(h_ptr, h_base + ulong(sg) * ulong(D) + col, h_f32);
  }

  const uint w_f32 = pc.flags & 1u;
  const ulong wb = ulong(col) * 4ul;
  const float w3 = valid_col ? inkling2_moe_read_f32(post_w, wb + 0ul, w_f32) : 0.0f;
  const float w2 = valid_col ? inkling2_moe_read_f32(post_w, wb + 1ul, w_f32) : 0.0f;
  const float w1 = valid_col ? inkling2_moe_read_f32(post_w, wb + 2ul, w_f32) : 0.0f;
  const float w0 = valid_col ? inkling2_moe_read_f32(post_w, wb + 3ul, w_f32) : 0.0f;

  const bool has_res = ((pc.flags >> 1u) & 1u) != 0u;
  const uint res_f32 = (pc.flags >> 2u) & 1u;
  const bool write_tape = ((pc.flags >> 3u) & 1u) != 0u;
  const float scale = 0.44721359549995793f;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint last_chunk_len = 0u;
  for (uint chunk_start = seq_start; chunk_start < seq_end; chunk_start += 8u) {
    const uint chunk_len = min(8u, seq_end - chunk_start);
    if (sg < chunk_len && valid_col) {
      const uint t = chunk_start + sg;
      const uint base = t * pc.top_k;
      float sum = 0.0f;
      for (uint k = 0u; k < pc.top_k; ++k) {
        const uint pos = inverse[base + k];
        sum += gamma[base + k] * float(projected[ulong(pos) * ulong(D) + col]);
      }
      const bfloat moe_b = bfloat(sum);
      if (write_tape) {
        tape_out[ulong(t) * ulong(D) + col] = moe_b;
      }
      tg_moe[3u + sg][lane] = float(moe_b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < chunk_len && valid_col) {
      const uint t = chunk_start + sg;
      const float x3 = tg_moe[sg + 0u][lane];
      const float x2 = tg_moe[sg + 1u][lane];
      const float x1 = tg_moe[sg + 2u][lane];
      const float x0 = tg_moe[sg + 3u][lane];
      const float conv_acc = x0 + w3 * x3 + w2 * x2 + w1 * x1 + w0 * x0;
      const bfloat conv_val = bfloat(conv_acc * scale);
      if (has_res) {
        const float rv = inkling2_moe_read_f32(
            res_raw, ulong(t) * ulong(pc.res_stride) + col, res_f32);
        y_out[ulong(t) * ulong(D) + col] = bfloat(rv + float(conv_val));
      } else {
        y_out[ulong(t) * ulong(D) + col] = conv_val;
      }
    }

    if (chunk_start + 8u < seq_end) {
      // Stage into register before barrier so sg=0..2 never race when chunk_len < 3.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float carry = 0.0f;
      if (sg < 3u && valid_col) {
        carry = tg_moe[chunk_len + sg][lane];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg < 3u && valid_col) {
        tg_moe[sg][lane] = carry;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      last_chunk_len = chunk_len;
    }
  }

  if (sg < 3u && valid_col) {
    const ulong ho_base = ulong(seq_idx) * 3ul * ulong(D);
    hist_out[ho_base + ulong(sg) * ulong(D) + col] = bfloat(tg_moe[last_chunk_len + sg][lane]);
  }
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
