// Standalone Metal 4 (`-std=metal4.0 -O3`) Tiled64 INT8-activation NAX prefill
// kernels (`int8 x int8 -> int32` with per-64 activation quantization).
//
// Zero MLX header dependencies, plain non-template `kernel void` entrypoints:
//   - `kquant_quantize_prefill_int8`: per-64 symmetric INT8 activation
//     quantizer producing:
//       * `xq [M_pad, K]` (`int8`) in unified sequential-pair slot order
//         matching `IQ4_XS`/`IQ4_NL`/`Q4_K` (`tl256`) and `Q5_K` (`tl1024`)
//       * `x_sa [K/64, M_pad]` (`float32`) per-64 scale (0x7fc00000 on NaN/Inf)
//       * `x_ysum [M_pad, U_pad]` (`float32`, `U_pad = ((K/32 + 15)/16)*16`)
//         per-32 `(scale * 2^-12) * sum32(qa)` for Q4_K/Q5_K Pass 2 m-term
//         (exact FP32 exponent shift: never overflows for any finite bf16
//         input and preserves bit-exact affine cancellation)
//   - 12 plain GEMM kernels `<mode>_qmm_t_nax_t64_i8{,_res,_swiglu}_bf16` for
//     `{iq4xs, iq4nl, q4k, q5k}` x `{plain, res, swiglu}`:
//     * `IQ4_XS` / `IQ4_NL`: v4 double-buffered `tg_s[2][8][64]` (4096 B) +
//       `tl256[256]` (512 B) = 4608 B TG, 1 barrier / 256 K, walk-16 schedule,
//       register-direct 32x32-per-SG `matmul2d(16, 32, 16)`.
//     * `Q4_K` / `Q5_K`: v6 Split-Pass — Pass 1 pure INT8 dot product (ZERO
//       m-term in hot loop; `Q4_K` uses 4608 B double-buffered `tg_s` + `tl256`;
//       `Q5_K` uses 4096 B single-buffered `tg_s` + `tl1024` + pure-32-bit shift
//       extraction), Pass 2 exact FP32 1/32-FLOP `matmul2d(32, 32, 16)` with
//       single-buffered `tg_mb[64][16]` (4096 B, handles odd `num_sb`, scaling
//       `(-dmin * mn) * 2^+12` to pair with `x_ysum`'s `2^-12`), combined in
//       two 16x64 float halves inside `raw_tg` with vectorized `bfloat4` stores
//       and fused `NONE` / `RESIDUAL` / `SWIGLU` epilogues.

#pragma clang fp reassociate(off)

#include "kquant_prefill_nax.h"

namespace kq_prefill_i8 {

using namespace metal;
using namespace mpp::tensor_ops;
using namespace kq_m8;
using namespace kq_prefill;

constant constexpr uint kI8FragM = 16;
constant constexpr uint kI8FragN = 32;
constant constexpr uint kI8FragK = 16;
constant constexpr uint kI8RowFrags = 2;
constant constexpr uint kI8Elements = 16;
constant constexpr uint kI8BM = kI8RowFrags * kI8FragM; // 32
constant constexpr uint kI8BN = 64;
constant constexpr uint kI8Walk = 16;

// Lane's (col, row) in Apple10 matmul2d int8 16x16 fragment layout.
METAL_FUNC short2 i8_coord(ushort lane) {
  const short qid = lane >> 2;
  return short2(
      ((qid & 2) | (lane & 1)) * 4,
      (qid & 4) | ((lane >> 1) & 3));
}

// Walk-16 L2 tile schedule: 16 consecutive threadgroups along grid.x process
// up to 16 row tiles of the same 64-column weight tile.
METAL_FUNC uint2 i8_walk(uint at, uint2 grid) {
  const uint block = at / (kI8Walk * grid.x);
  const uint height = min(kI8Walk, grid.y - block * kI8Walk);
  const uint i = at - block * kI8Walk * grid.x;
  return uint2(i / height, block * kI8Walk + i % height);
}

template <int epilogue>
METAL_FUNC void store_bf16_quad(
    device bfloat* y,
    const device bfloat* aux,
    ulong o0,
    float v0,
    float v1,
    float v2,
    float v3) {
  if constexpr (epilogue == KQ_EP_NONE) {
    *reinterpret_cast<device bfloat4*>(y + o0) =
        bfloat4(bfloat(v0), bfloat(v1), bfloat(v2), bfloat(v3));
  } else {
    const bfloat4 a4 = *reinterpret_cast<const device bfloat4*>(aux + o0);
    if constexpr (epilogue == KQ_EP_RESIDUAL) {
      *reinterpret_cast<device bfloat4*>(y + o0) = bfloat4(
          kq_bf16_residual(a4.x, bfloat(v0)),
          kq_bf16_residual(a4.y, bfloat(v1)),
          kq_bf16_residual(a4.z, bfloat(v2)),
          kq_bf16_residual(a4.w, bfloat(v3)));
    } else {
      static_assert(epilogue == KQ_EP_SWIGLU, "unknown store epilogue");
      *reinterpret_cast<device bfloat4*>(y + o0) = bfloat4(
          kq_bf16_swiglu(a4.x, bfloat(v0)),
          kq_bf16_swiglu(a4.y, bfloat(v1)),
          kq_bf16_swiglu(a4.z, bfloat(v2)),
          kq_bf16_swiglu(a4.w, bfloat(v3)));
    }
  }
}

// ============================================================================
// v4 IQ4_XS & IQ4_NL:
//   - Double-buffered tg_s[2][8][64] (4096 B) + tl256[256] (512 B) = 4608 B
//   - 1 threadgroup_barrier per 256-element super-block
//   - Per-64 activation scale x_sa[g64, m], pair LUT lookup via tl256
// ============================================================================
template <bool IsIQ4NL, int epilogue>
METAL_FUNC void iq4_i8_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device half* biases,
    const device int8_t* xq,
    const device float* x_sa,
    device bfloat* y,
    const device bfloat* aux,
    int K,
    int N,
    int M,
    int M_pad,
    threadgroup uchar* raw_tg,
    uint2 group,
    ushort lane,
    ushort sg) {
  const uint col_tiles = uint(N) / kI8BN;
  const uint row_tiles = (uint(M) + kI8BM - 1u) / kI8BM;
  const uint at = group.y * kI8Walk + group.x;
  if (at >= col_tiles * row_tiles) {
    return;
  }

  const uint lid = uint(sg) * 32u + uint(lane);
  threadgroup float (*tg_s)[8][64] =
      reinterpret_cast<threadgroup float (*)[8][64]>(raw_tg);
  threadgroup char2* tl256 =
      reinterpret_cast<threadgroup char2*>(raw_tg + 4096u);
  for (uint i = lid; i < 256u; i += 64u) {
    tl256[i] = char2(kIQ4[i & 15u], kIQ4[i >> 4u]);
  }

  const uint2 tile = i8_walk(at, uint2(col_tiles, row_tiles));
  const uint row_base = tile.y * kI8BM;
  const uint col_local = uint(sg) * kI8FragN;
  const short2 coord = i8_coord(lane);
  const uint cx = uint(coord.x) >> 2u;
  const uint m_base = row_base + uint(coord.y);
  const uint n_lane = col_local + uint(coord.y);
  const uint n_run_local = col_local + uint(coord.x);
  const uint n_run = tile.x * kI8BN + n_run_local;

  constexpr auto desc = matmul2d_descriptor(
      kI8FragM,
      kI8FragN,
      kI8FragK,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(
      kI8FragM,
      kI8FragN,
      kI8FragK,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a =
      op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();

  float cf[kI8RowFrags][kI8Elements];
#pragma unroll
  for (uint i = 0u; i < kI8RowFrags; ++i) {
#pragma unroll
    for (uint e = 0u; e < kI8Elements; ++e) {
      cf[i][e] = 0.0f;
    }
  }

  const uint units = uint(K) / 32u;
  const uint num_sb = uint(K) / 256u;
  const device uint32_t* w_base =
      w + (ulong(tile.x) * units * 64u + n_lane) * 4u + cx;
  const device int8_t* x_base = xq + ulong(m_base) * uint(K) + cx * 4u;
  const ulong x_s8 = ulong(8u) * uint(K);
  const ulong x_s16 = ulong(16u) * uint(K);
  const ulong sb_tile_base = ulong(tile.x) * num_sb * 64u + lid;
  const ulong u_tile_base = ulong(tile.x) * units * 64u + lid;

  for (uint sb = 0u; sb < num_sb; ++sb) {
    const uint buf = sb & 1u;
    if constexpr (IsIQ4NL) {
      const ulong u_sb = u_tile_base + ulong(sb) * 8u * 64u;
#pragma unroll
      for (uint u = 0u; u < 8u; ++u) {
        const ulong idx = u_sb + ulong(u) * 64u;
        tg_s[buf][u][lid] =
            float(biases[idx]) * float(as_type<int8_t>(scales[idx]));
      }
    } else {
      const ulong col_sb = sb_tile_base + ulong(sb) * 64u;
      const float d_col = float(biases[col_sb]);
      const uint2 sc2 =
          *reinterpret_cast<const device uint2*>(scales + col_sb * 8u);
      const char4 c0 = as_type<char4>(sc2.x);
      const char4 c1 = as_type<char4>(sc2.y);
      tg_s[buf][0][lid] = d_col * float(c0.x);
      tg_s[buf][1][lid] = d_col * float(c0.y);
      tg_s[buf][2][lid] = d_col * float(c0.z);
      tg_s[buf][3][lid] = d_col * float(c0.w);
      tg_s[buf][4][lid] = d_col * float(c1.x);
      tg_s[buf][5][lid] = d_col * float(c1.y);
      tg_s[buf][6][lid] = d_col * float(c1.z);
      tg_s[buf][7][lid] = d_col * float(c1.w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(disable)
    for (uint g = 0u; g < 4u; ++g) {
      const uint g64 = sb * 4u + g;
      const device float* xsg = x_sa + ulong(g64) * uint(M_pad) + m_base;
      const float sa00 = xsg[0];
      const float sa01 = xsg[8];
      const float sa10 = xsg[16];
      const float sa11 = xsg[24];
#pragma clang loop unroll(disable)
      for (uint h = 0u; h < 2u; ++h) {
        const uint u = g * 2u + h;
        const uint g32 = sb * 8u + u;
        const device uint32_t* w_u = w_base + ulong(g32) * 256u;
        uint wv[4];
#pragma unroll
        for (uint q = 0u; q < 4u; ++q) {
          wv[q] = w_u[q * 32u];
        }
#pragma unroll
        for (uint t = 0u; t < 2u; ++t) {
#pragma unroll
          for (uint q = 0u; q < 4u; ++q) {
            const uchar4 b = as_type<uchar4>(wv[q]);
            const char2 p0 = tl256[t ? b.z : b.x];
            const char2 p1 = tl256[t ? b.w : b.y];
            const uint base = q * 4u;
            ct_b[base + 0u] = p0.x;
            ct_b[base + 1u] = p0.y;
            ct_b[base + 2u] = p1.x;
            ct_b[base + 3u] = p1.y;
          }
          const ulong xo = ulong(g32) * 32u + t * 16u;
#pragma unroll
          for (uint hf = 0u; hf < kI8RowFrags; ++hf) {
            const device int8_t* xa = x_base + ulong(hf) * x_s16 + xo;
            const char4 a0 =
                as_type<char4>(*reinterpret_cast<const device uint*>(xa));
            const char4 a1 = as_type<char4>(
                *reinterpret_cast<const device uint*>(xa + x_s8));
            ct_a[0] = a0.x;
            ct_a[1] = a0.y;
            ct_a[2] = a0.z;
            ct_a[3] = a0.w;
            ct_a[4] = a1.x;
            ct_a[5] = a1.y;
            ct_a[6] = a1.z;
            ct_a[7] = a1.w;
            if (t == 0u) {
              if (hf == 0u) {
                op_set.run(ct_a, ct_b, acc0);
              } else {
                op_set.run(ct_a, ct_b, acc1);
              }
            } else {
              if (hf == 0u) {
                op.run(ct_a, ct_b, acc0);
              } else {
                op.run(ct_a, ct_b, acc1);
              }
            }
          }
        }
        const float4 s0 = *reinterpret_cast<const threadgroup float4*>(
            &tg_s[buf][u][n_run_local]);
        const float4 s1 = *reinterpret_cast<const threadgroup float4*>(
            &tg_s[buf][u][n_run_local + 16u]);
#pragma unroll
        for (uint e = 0u; e < kI8Elements; ++e) {
          const uint r = (e & 7u) >> 2u;
          const float s = (e >> 3u) ? s1[e & 3u] : s0[e & 3u];
          cf[0][e] = fma((r ? sa01 : sa00) * s, float(acc0[e]), cf[0][e]);
          cf[1][e] = fma((r ? sa11 : sa10) * s, float(acc1[e]), cf[1][e]);
        }
      }
    }
  }

#pragma unroll
  for (uint i = 0u; i < kI8RowFrags; ++i) {
#pragma unroll
    for (uint r = 0u; r < 2u; ++r) {
      const uint m = m_base + i * kI8FragM + r * 8u;
      if (m >= uint(M)) {
        continue;
      }
#pragma unroll
      for (uint h = 0u; h < 2u; ++h) {
        const uint eb = h * 8u + r * 4u;
        const ulong o0 = ulong(m) * uint(N) + n_run + h * 16u;
        store_bf16_quad<epilogue>(
            y,
            aux,
            o0,
            cf[i][eb + 0u],
            cf[i][eb + 1u],
            cf[i][eb + 2u],
            cf[i][eb + 3u]);
      }
    }
  }
}

// ============================================================================
// v6 Split-Pass Q4_K & Q5_K:
//   Pass 1 (Hot INT8 loop): ZERO m-term, ZERO tg_m!
//     - Q4_K: double-buffered tg_s[2][8][64] (4096 B) + tl256[256] (512 B) = 4608 B
//     - Q5_K: single-buffered tg_s[8][64] (2048 B) + tl1024[1024] (2048 B) = 4096 B
//   Pass 2 (After K-loop): exact FP32 1/32-FLOP matmul2d(32, 32, 16) for m-term!
//     - Single-buffered tg_mb[64][16] float (4096 B inside raw_tg)
//     - `x_ysum` stores `(x_sa * 2^-12) * sum32(qa)` and `tg_mb` stores
//       `(-dmin * mn) * 2^+12` (exact exponent shift, zero overflow for any
//       finite bf16 input, bit-exact affine cancellation when `d*sc*q == dmin*mn`)
//     - Handles odd num_sb cleanly (zero-pads second sb slot + U_pad tail cols)
//   Combine: 2 row halves x (16 x 64 float = 4096 B inside raw_tg), vectorized
//     bfloat4 store with fused NONE / RESIDUAL / SWIGLU epilogue.
// ============================================================================
template <bool IsQ5K, int epilogue>
METAL_FUNC void q45k_split_i8_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device half* biases,
    const device int8_t* xq,
    const device float* x_sa,
    const device float* x_ysum,
    device bfloat* y,
    const device bfloat* aux,
    int K,
    int N,
    int M,
    int M_pad,
    threadgroup uchar* raw_tg,
    uint2 group,
    ushort lane,
    ushort sg) {
  const uint col_tiles = uint(N) / kI8BN;
  const uint row_tiles = (uint(M) + kI8BM - 1u) / kI8BM;
  const uint at = group.y * kI8Walk + group.x;
  if (at >= col_tiles * row_tiles) {
    return;
  }

  const uint lid = uint(sg) * 32u + uint(lane);
  const uint2 tile = i8_walk(at, uint2(col_tiles, row_tiles));
  const uint row_base = tile.y * kI8BM;
  const uint col_local = uint(sg) * kI8FragN;
  const short2 coord = i8_coord(lane);
  const uint cx = uint(coord.x) >> 2u;
  const uint m_base = row_base + uint(coord.y);
  const uint n_lane = col_local + uint(coord.y);
  const uint n_run_local = col_local + uint(coord.x);
  const uint n_run = tile.x * kI8BN + n_run_local;

  // raw_tg layout in Pass 1:
  //   Q5K (4096 B): tg_s0 (2048 B @ 0) + tl1024 (2048 B @ 2048)
  //   Q4K (4608 B): tg_s0 (2048 B @ 0) + tg_s1 (2048 B @ 2048) + tl256 (512 B @ 4096)
  threadgroup float (*tg_s0)[64] =
      reinterpret_cast<threadgroup float (*)[64]>(raw_tg);
  threadgroup float (*tg_s1)[64] =
      reinterpret_cast<threadgroup float (*)[64]>(raw_tg + 2048u);
  threadgroup char2* tl =
      reinterpret_cast<threadgroup char2*>(raw_tg + (IsQ5K ? 2048u : 4096u));

  if constexpr (IsQ5K) {
    for (uint i = lid; i < 1024u; i += 64u) {
      tl[i] = char2(int8_t(i & 31u), int8_t(i >> 5u));
    }
  } else {
    for (uint i = lid; i < 256u; i += 64u) {
      tl[i] = char2(int8_t(i & 15u), int8_t(i >> 4u));
    }
  }

  constexpr auto desc = matmul2d_descriptor(
      kI8FragM,
      kI8FragN,
      kI8FragK,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(
      kI8FragM,
      kI8FragN,
      kI8FragK,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a =
      op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();

  float cf[kI8RowFrags][kI8Elements];
#pragma unroll
  for (uint i = 0u; i < kI8RowFrags; ++i) {
#pragma unroll
    for (uint e = 0u; e < kI8Elements; ++e) {
      cf[i][e] = 0.0f;
    }
  }

  const uint units = uint(K) / 32u;
  const uint u_pad = ((units + 15u) / 16u) * 16u;
  const uint num_sb = uint(K) / 256u;
  constexpr uint kBits = IsQ5K ? 5u : 4u;
  const device uint32_t* w_base =
      w + (ulong(tile.x) * units * 64u + n_lane) * kBits + cx;
  const device int8_t* x_base = xq + ulong(m_base) * uint(K) + cx * 4u;
  const ulong x_s8 = ulong(8u) * uint(K);
  const ulong x_s16 = ulong(16u) * uint(K);
  const ulong sb_tile_base = ulong(tile.x) * num_sb * 64u + lid;

  // Precomputed pure-32-bit shift constants for Q5_K (zero 64-bit shifts).
  const uint q5_t0_rsh = cx * 8u;
  const uint q5_t0_lsh = (32u - q5_t0_rsh) & 31u;
  const uint q5_t0_mask = (cx >= 2u) ? 0xFFFFFFFFu : 0u;
  const bool q5_t1_hi = (cx >= 2u);
  const uint q5_t1_ash = 20u + 8u * (cx & 1u);
  const uint q5_t1_bsh = 12u - 8u * (cx & 1u);
  const uint q5_t1_hrsh = 4u + 8u * (cx & 1u);

  // ---------------- PASS 1: Pure INT8 dot product (ZERO m-term) -------------
  for (uint sb = 0u; sb < num_sb; ++sb) {
    threadgroup float (*tg_s)[64] = IsQ5K ? tg_s0 : ((sb & 1u) ? tg_s1 : tg_s0);
    if constexpr (IsQ5K) {
      if (sb > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    const ulong col_sb = sb_tile_base + ulong(sb) * 64u;
    const float d_col = float(biases[col_sb * 2u]);
    const uint4 sc4 =
        *reinterpret_cast<const device uint4*>(scales + col_sb * 16u);
#pragma unroll
    for (uint p = 0u; p < 4u; ++p) {
      const uchar4 b = as_type<uchar4>(sc4[p]);
      tg_s[2u * p + 0u][lid] = d_col * float(b.x);
      tg_s[2u * p + 1u][lid] = d_col * float(b.z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(disable)
    for (uint g = 0u; g < 4u; ++g) {
      const uint g64 = sb * 4u + g;
      const device float* xsg = x_sa + ulong(g64) * uint(M_pad) + m_base;
      const float sa00 = xsg[0];
      const float sa01 = xsg[8];
      const float sa10 = xsg[16];
      const float sa11 = xsg[24];
#pragma clang loop unroll(disable)
      for (uint h = 0u; h < 2u; ++h) {
        const uint u = g * 2u + h;
        const uint g32 = sb * 8u + u;
        if constexpr (IsQ5K) {
          const device uint32_t* w_u = w_base + ulong(g32) * 320u;
          uint w0[4], w1[4];
#pragma unroll
          for (uint q = 0u; q < 4u; ++q) {
            const device uint32_t* wq = w_u + q * 40u;
            w0[q] = wq[0];
            w1[q] = wq[1];
          }
#pragma unroll
          for (uint t = 0u; t < 2u; ++t) {
#pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
              const uint v = (t == 0u)
                  ? ((w0[q] >> q5_t0_rsh) | ((w1[q] << q5_t0_lsh) & q5_t0_mask))
                  : (q5_t1_hi ? (w1[q] >> q5_t1_hrsh)
                              : ((w0[q] >> q5_t1_ash) | (w1[q] << q5_t1_bsh)));
              const char2 p0 = tl[v & 0x3FFu];
              const char2 p1 = tl[(v >> 10u) & 0x3FFu];
              const uint base = q * 4u;
              ct_b[base + 0u] = p0.x;
              ct_b[base + 1u] = p0.y;
              ct_b[base + 2u] = p1.x;
              ct_b[base + 3u] = p1.y;
            }
            const ulong xo = ulong(g32) * 32u + t * 16u;
#pragma unroll
            for (uint hf = 0u; hf < kI8RowFrags; ++hf) {
              const device int8_t* xa = x_base + ulong(hf) * x_s16 + xo;
              const char4 a0 =
                  as_type<char4>(*reinterpret_cast<const device uint*>(xa));
              const char4 a1 = as_type<char4>(
                  *reinterpret_cast<const device uint*>(xa + x_s8));
              ct_a[0] = a0.x;
              ct_a[1] = a0.y;
              ct_a[2] = a0.z;
              ct_a[3] = a0.w;
              ct_a[4] = a1.x;
              ct_a[5] = a1.y;
              ct_a[6] = a1.z;
              ct_a[7] = a1.w;
              if (t == 0u) {
                if (hf == 0u) {
                  op_set.run(ct_a, ct_b, acc0);
                } else {
                  op_set.run(ct_a, ct_b, acc1);
                }
              } else {
                if (hf == 0u) {
                  op.run(ct_a, ct_b, acc0);
                } else {
                  op.run(ct_a, ct_b, acc1);
                }
              }
            }
          }
        } else {
          const device uint32_t* w_u = w_base + ulong(g32) * 256u;
          uint wv[4];
#pragma unroll
          for (uint q = 0u; q < 4u; ++q) {
            wv[q] = w_u[q * 32u];
          }
#pragma unroll
          for (uint t = 0u; t < 2u; ++t) {
#pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
              const uchar4 b = as_type<uchar4>(wv[q]);
              const char2 p0 = tl[t ? b.z : b.x];
              const char2 p1 = tl[t ? b.w : b.y];
              const uint base = q * 4u;
              ct_b[base + 0u] = p0.x;
              ct_b[base + 1u] = p0.y;
              ct_b[base + 2u] = p1.x;
              ct_b[base + 3u] = p1.y;
            }
            const ulong xo = ulong(g32) * 32u + t * 16u;
#pragma unroll
            for (uint hf = 0u; hf < kI8RowFrags; ++hf) {
              const device int8_t* xa = x_base + ulong(hf) * x_s16 + xo;
              const char4 a0 =
                  as_type<char4>(*reinterpret_cast<const device uint*>(xa));
              const char4 a1 = as_type<char4>(
                  *reinterpret_cast<const device uint*>(xa + x_s8));
              ct_a[0] = a0.x;
              ct_a[1] = a0.y;
              ct_a[2] = a0.z;
              ct_a[3] = a0.w;
              ct_a[4] = a1.x;
              ct_a[5] = a1.y;
              ct_a[6] = a1.z;
              ct_a[7] = a1.w;
              if (t == 0u) {
                if (hf == 0u) {
                  op_set.run(ct_a, ct_b, acc0);
                } else {
                  op_set.run(ct_a, ct_b, acc1);
                }
              } else {
                if (hf == 0u) {
                  op.run(ct_a, ct_b, acc0);
                } else {
                  op.run(ct_a, ct_b, acc1);
                }
              }
            }
          }
        }
        const float4 s0 =
            *reinterpret_cast<const threadgroup float4*>(&tg_s[u][n_run_local]);
        const float4 s1 = *reinterpret_cast<const threadgroup float4*>(
            &tg_s[u][n_run_local + 16u]);
#pragma unroll
        for (uint e = 0u; e < kI8Elements; ++e) {
          const uint r = (e & 7u) >> 2u;
          const float s = (e >> 3u) ? s1[e & 3u] : s0[e & 3u];
          cf[0][e] = fma((r ? sa01 : sa00) * s, float(acc0[e]), cf[0][e]);
          cf[1][e] = fma((r ? sa11 : sa10) * s, float(acc1[e]), cf[1][e]);
        }
      }
    }
  }

  // ---------------- PASS 2: Exact FP32 1/32-FLOP matmul2d(32, 32, 16) -------
  // Reuse `raw_tg` (4096 B) as single-buffered `tg_mb[64][16]` float!
  // `x_ysum` stores `(x_sa * 2^-12) * sum32(qa)` in float32 (since
  // |sum32(qa)| <= 4064 < 2^12, it never overflows for any finite bf16 input),
  // and `tg_mb` stores `(-dmin * mn) * 2^+12` in float32 (exact exponent shift,
  // zero mantissa rounding, exact cancellation when `d * sc * q == dmin * mn`).
  threadgroup float* tg_mb = reinterpret_cast<threadgroup float*>(raw_tg);

  constexpr auto desc_m = matmul2d_descriptor(
      32,
      32,
      16,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_m, execution_simdgroups<1>> op_m;
  auto tA_m = tensor(
      const_cast<device float*>(x_ysum + ulong(row_base) * u_pad),
      dextents<int, 2>{int(u_pad), 32},
      array<int, 2>{1, int(u_pad)});
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> tB_m(
      tg_mb + col_local * 16u, dextents<int, 2>{16, 32}, array<int, 2>{1, 16});
  auto a_m0 = tA_m.template slice<16, 32>(0, 0);
  auto b_m0 = tB_m.template slice<16, 32>(0, 0);
  auto m_acc = op_m.template get_destination_cooperative_tensor<
      decltype(a_m0),
      decltype(b_m0),
      float>();
#pragma unroll
  for (ushort i = 0; i < m_acc.get_capacity(); ++i) {
    m_acc[i] = 0.0f;
  }

  const uint num_m_steps = (num_sb + 1u) / 2u;
  for (uint ms = 0u; ms < num_m_steps; ++ms) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint sbi = 0u; sbi < 2u; ++sbi) {
      threadgroup float* dst = tg_mb + lid * 16u + sbi * 8u;
      const uint sb_idx = ms * 2u + sbi;
      if (sb_idx < num_sb) {
        const ulong col_sb = sb_tile_base + ulong(sb_idx) * 64u;
        const float m_col = -(float(biases[col_sb * 2u + 1u]) * 4096.0f);
        const uint4 sc4 =
            *reinterpret_cast<const device uint4*>(scales + col_sb * 16u);
#pragma unroll
        for (uint p = 0u; p < 4u; ++p) {
          const uchar4 b = as_type<uchar4>(sc4[p]);
          dst[2u * p + 0u] = m_col * float(b.y);
          dst[2u * p + 1u] = m_col * float(b.w);
        }
      } else {
#pragma unroll
        for (uint p = 0u; p < 8u; ++p) {
          dst[p] = 0.0f;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a_sl = tA_m.template slice<16, 32>(int(ms * 16u), 0);
    op_m.run(a_sl, b_m0, m_acc);
  }

  // ---------------- Combine Pass 1 (`cf`) + Pass 2 (`m_acc`) ----------------
  // Reuse `raw_tg` (4096 B = 16 x 64 float) across 2 row halves.
  threadgroup float* tg_half = reinterpret_cast<threadgroup float*>(raw_tg);
#pragma unroll
  for (uint half_i = 0u; half_i < 2u; ++half_i) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort i = 0; i < m_acc.get_capacity(); ++i) {
      if (!m_acc.is_valid_element(i)) {
        continue;
      }
      const auto idx = m_acc.get_multidimensional_index(i);
      const uint r = uint(idx[1]);
      if ((r >> 4u) == half_i) {
        tg_half[(r & 15u) * 64u + col_local + uint(idx[0])] = float(m_acc[i]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint r = 0u; r < 2u; ++r) {
      const uint local_r = uint(coord.y) + r * 8u;
      const uint m = row_base + half_i * 16u + local_r;
      if (m >= uint(M)) {
        continue;
      }
#pragma unroll
      for (uint h = 0u; h < 2u; ++h) {
        const uint eb = h * 8u + r * 4u;
        const float4 mv = *reinterpret_cast<const threadgroup float4*>(
            tg_half + local_r * 64u + n_run_local + h * 16u);
        const ulong o0 = ulong(m) * uint(N) + n_run + h * 16u;
        store_bf16_quad<epilogue>(
            y,
            aux,
            o0,
            cf[half_i][eb + 0u] + mv.x,
            cf[half_i][eb + 1u] + mv.y,
            cf[half_i][eb + 2u] + mv.z,
            cf[half_i][eb + 3u] + mv.w);
      }
    }
  }
}

} // namespace kq_prefill_i8

// Per-64-element symmetric INT8 activation quantizer for Tiled64 NAX prefill.
// Grid: (K / 256, M_pad / 8, 1), Threadgroup: (32, 4, 1) = 4 simdgroups.
// Each simdgroup processes one 64-element group across 8 consecutive rows:
//   - `xq [M_pad, K]` (`int8`) written in unified sequential-pair slot order
//     `slot = ((p & 4) << 2) | (c << 2) | (p & 3)` per 32-element half
//   - `x_sa [K/64, M_pad]` (`float32`): `any_nf ? 0x7fc00000u : scale`
//   - `x_ysum [M_pad, U_pad]` (`float32`): per-32 `(scale * 2^-12) * sum32(qa)`
//     (0x7fc00000u on NaN/Inf, zero-padded in tail cols [units, U_pad) and tail
//     rows [M, M_pad)).
[[kernel, max_total_threads_per_threadgroup(128)]] void
kquant_quantize_prefill_int8(
    const device bfloat* x [[buffer(0)]],
    device int8_t* xq [[buffer(1)]],
    device float* x_sa [[buffer(2)]],
    device float* x_ysum [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    const constant int& M_pad [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint g64 = group.x * 4u + simd_gid;
  const uint groups64 = uint(K) / 64u;
  const uint units = uint(K) / 32u;
  const uint u_pad = ((units + 15u) / 16u) * 16u;
  const uint m0 = group.y * 8u;
  const uint c = simd_lid >> 3u;
  const uint p = simd_lid & 7u;
  const uint slot = ((p & 4u) << 2u) | (c << 2u) | (p & 3u);
  const ulong g_col = ulong(g64) * 64u;
  const uint u0 = g64 * 2u;
  const bool zero_tail_cols = (units < u_pad) && (g64 == groups64 - 1u) && (simd_lid < 8u);

  device uint* sa_out =
      reinterpret_cast<device uint*>(x_sa) + ulong(g64) * uint(M_pad) + m0;
  device uint* ysum_u32 = reinterpret_cast<device uint*>(x_ysum);

#pragma unroll
  for (uint r = 0u; r < 8u; ++r) {
    const uint m = m0 + r;
    const ulong row_base = ulong(m) * uint(K) + g_col;
    const ulong ysum_row = ulong(m) * uint(u_pad);
    if (zero_tail_cols) {
      ysum_u32[ysum_row + units + simd_lid] = 0u;
    }
    if (m >= uint(M)) {
      xq[row_base + slot] = int8_t(0);
      xq[row_base + 32u + slot] = int8_t(0);
      if (simd_lid == 0u) {
        sa_out[r] = 0u;
        ysum_u32[ysum_row + u0] = 0u;
        ysum_u32[ysum_row + u0 + 1u] = 0u;
      }
      continue;
    }

    const bfloat b0 = x[row_base + simd_lid];
    const bfloat b1 = x[row_base + 32u + simd_lid];
    const bool nf0 =
        (as_type<ushort>(b0) & ushort(0x7f80u)) == ushort(0x7f80u);
    const bool nf1 =
        (as_type<ushort>(b1) & ushort(0x7f80u)) == ushort(0x7f80u);
    const float f0 = float(b0);
    const float f1 = float(b1);
    const float ax = max(nf0 ? 0.0f : fabs(f0), nf1 ? 0.0f : fabs(f1));
    const float amax = simd_max(ax);
    const bool any_nf = simd_any(nf0 || nf1);
    const bool coded = (!any_nf) && (amax > 0x1p-100f);
    const float scale = coded ? metal::precise::divide(amax, 127.0f) : 0.0f;
    const float inv = coded ? metal::precise::divide(127.0f, amax) : 0.0f;
    const int q0 = coded ? int(clamp(rint(f0 * inv), -127.0f, 127.0f)) : 0;
    const int q1 = coded ? int(clamp(rint(f1 * inv), -127.0f, 127.0f)) : 0;
    xq[row_base + slot] = int8_t(q0);
    xq[row_base + 32u + slot] = int8_t(q1);
    const int sum0 = simd_sum(q0);
    const int sum1 = simd_sum(q1);
    if (simd_lid == 0u) {
      const float scale_lo = scale * 0x1p-12f;
      sa_out[r] = any_nf ? 0x7fc00000u : as_type<uint>(scale);
      ysum_u32[ysum_row + u0] =
          any_nf ? 0x7fc00000u : as_type<uint>(scale_lo * float(sum0));
      ysum_u32[ysum_row + u0 + 1u] =
          any_nf ? 0x7fc00000u : as_type<uint>(scale_lo * float(sum1));
    }
  }
}

// clang-format off
#define DEFINE_KQ_I8_IQ4_PLAIN(mode, IsNL) \
  kernel void mode##_qmm_t_nax_t64_i8_bf16( \
      const device uint32_t* w [[buffer(0)]], \
      const device uint8_t* scales [[buffer(1)]], \
      const device half* biases [[buffer(2)]], \
      const device int8_t* xq [[buffer(3)]], \
      const device float* x_sa [[buffer(4)]], \
      const device float* x_ysum [[buffer(5)]], \
      device bfloat* y [[buffer(6)]], \
      const constant int& K [[buffer(7)]], \
      const constant int& N [[buffer(8)]], \
      const constant int& M [[buffer(9)]], \
      const constant int& M_pad [[buffer(10)]], \
      uint2 group [[threadgroup_position_in_grid]], \
      ushort sg [[simdgroup_index_in_threadgroup]], \
      ushort lane [[thread_index_in_simdgroup]]) { \
    (void)x_ysum; \
    threadgroup uint4 raw_tg[4608 / 16]; \
    kq_prefill_i8::iq4_i8_impl<IsNL, kq_prefill::KQ_EP_NONE>( \
        w, scales, biases, xq, x_sa, y, nullptr, K, N, M, M_pad, \
        reinterpret_cast<threadgroup uchar*>(raw_tg), group, lane, sg); \
  }

#define DEFINE_KQ_I8_IQ4_EP(mode, ep_name, ep, IsNL) \
  kernel void mode##_qmm_t_nax_t64_i8_##ep_name##_bf16( \
      const device uint32_t* w [[buffer(0)]], \
      const device uint8_t* scales [[buffer(1)]], \
      const device half* biases [[buffer(2)]], \
      const device int8_t* xq [[buffer(3)]], \
      const device float* x_sa [[buffer(4)]], \
      const device float* x_ysum [[buffer(5)]], \
      device bfloat* y [[buffer(6)]], \
      const constant int& K [[buffer(7)]], \
      const constant int& N [[buffer(8)]], \
      const constant int& M [[buffer(9)]], \
      const constant int& M_pad [[buffer(10)]], \
      const device bfloat* aux [[buffer(11)]], \
      uint2 group [[threadgroup_position_in_grid]], \
      ushort sg [[simdgroup_index_in_threadgroup]], \
      ushort lane [[thread_index_in_simdgroup]]) { \
    (void)x_ysum; \
    threadgroup uint4 raw_tg[4608 / 16]; \
    kq_prefill_i8::iq4_i8_impl<IsNL, ep>( \
        w, scales, biases, xq, x_sa, y, aux, K, N, M, M_pad, \
        reinterpret_cast<threadgroup uchar*>(raw_tg), group, lane, sg); \
  }

#define DEFINE_KQ_I8_Q45_PLAIN(mode, IsQ5K, TgBytes) \
  kernel void mode##_qmm_t_nax_t64_i8_bf16( \
      const device uint32_t* w [[buffer(0)]], \
      const device uint8_t* scales [[buffer(1)]], \
      const device half* biases [[buffer(2)]], \
      const device int8_t* xq [[buffer(3)]], \
      const device float* x_sa [[buffer(4)]], \
      const device float* x_ysum [[buffer(5)]], \
      device bfloat* y [[buffer(6)]], \
      const constant int& K [[buffer(7)]], \
      const constant int& N [[buffer(8)]], \
      const constant int& M [[buffer(9)]], \
      const constant int& M_pad [[buffer(10)]], \
      uint2 group [[threadgroup_position_in_grid]], \
      ushort sg [[simdgroup_index_in_threadgroup]], \
      ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint4 raw_tg[(TgBytes) / 16]; \
    kq_prefill_i8::q45k_split_i8_impl<IsQ5K, kq_prefill::KQ_EP_NONE>( \
        w, scales, biases, xq, x_sa, x_ysum, y, nullptr, K, N, M, M_pad, \
        reinterpret_cast<threadgroup uchar*>(raw_tg), group, lane, sg); \
  }

#define DEFINE_KQ_I8_Q45_EP(mode, ep_name, ep, IsQ5K, TgBytes) \
  kernel void mode##_qmm_t_nax_t64_i8_##ep_name##_bf16( \
      const device uint32_t* w [[buffer(0)]], \
      const device uint8_t* scales [[buffer(1)]], \
      const device half* biases [[buffer(2)]], \
      const device int8_t* xq [[buffer(3)]], \
      const device float* x_sa [[buffer(4)]], \
      const device float* x_ysum [[buffer(5)]], \
      device bfloat* y [[buffer(6)]], \
      const constant int& K [[buffer(7)]], \
      const constant int& N [[buffer(8)]], \
      const constant int& M [[buffer(9)]], \
      const constant int& M_pad [[buffer(10)]], \
      const device bfloat* aux [[buffer(11)]], \
      uint2 group [[threadgroup_position_in_grid]], \
      ushort sg [[simdgroup_index_in_threadgroup]], \
      ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint4 raw_tg[(TgBytes) / 16]; \
    kq_prefill_i8::q45k_split_i8_impl<IsQ5K, ep>( \
        w, scales, biases, xq, x_sa, x_ysum, y, aux, K, N, M, M_pad, \
        reinterpret_cast<threadgroup uchar*>(raw_tg), group, lane, sg); \
  }

DEFINE_KQ_I8_IQ4_PLAIN(iq4xs, false)
DEFINE_KQ_I8_IQ4_EP(iq4xs, res,    kq_prefill::KQ_EP_RESIDUAL, false)
DEFINE_KQ_I8_IQ4_EP(iq4xs, swiglu, kq_prefill::KQ_EP_SWIGLU,   false)

DEFINE_KQ_I8_IQ4_PLAIN(iq4nl, true)
DEFINE_KQ_I8_IQ4_EP(iq4nl, res,    kq_prefill::KQ_EP_RESIDUAL, true)
DEFINE_KQ_I8_IQ4_EP(iq4nl, swiglu, kq_prefill::KQ_EP_SWIGLU,   true)

DEFINE_KQ_I8_Q45_PLAIN(q4k, false, 4608)
DEFINE_KQ_I8_Q45_EP(q4k, res,    kq_prefill::KQ_EP_RESIDUAL, false, 4608)
DEFINE_KQ_I8_Q45_EP(q4k, swiglu, kq_prefill::KQ_EP_SWIGLU,   false, 4608)

DEFINE_KQ_I8_Q45_PLAIN(q5k, true, 4096)
DEFINE_KQ_I8_Q45_EP(q5k, res,    kq_prefill::KQ_EP_RESIDUAL, true, 4096)
DEFINE_KQ_I8_Q45_EP(q5k, swiglu, kq_prefill::KQ_EP_SWIGLU,   true, 4096)
// clang-format on
