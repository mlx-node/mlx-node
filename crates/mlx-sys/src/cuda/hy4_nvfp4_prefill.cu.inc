R"cuda(
// ============================================================================
// Hy4 CUDA NVFP4 Dense Prefill Kernels
//   Requires ptx_helpers.cu.inc and hy4_fp4_helpers.cu.inc prepended.
//   1. hy4_nvfp4_quantize_x_swizzled:
//        Fused BF16 [M, K] -> NVFP4 (u32 [M, K/8]) + 128x4 cuBLASLt tiled E4M3
//        scales (u8 [pad_M, K/16]) with zero-filled padding rows in 1 launch!
//   2. hy4_nvfp4_prefill_bf16_bm64:
//        Exact BF16-activation x NVFP4-weight BM=64, BN=128, BK=64 8-warp
//        double-buffered cp.async.cg Tensor Core MMA GEMM (0 B spills).
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
#define HY4_HAS_HW_FP4_ENCODE 1
#else
#define HY4_HAS_HW_FP4_ENCODE 0
#endif

// Exact round-to-nearest-even non-negative FP32 -> unsigned FP8 E4M3FN byte (0x00..0x7E).
__device__ __forceinline__ unsigned int hy4_f32_to_ue4m3_rn(float x) {
  if (!(x > 0.0f)) {
    return 0u;
  }
  if (x >= 448.0f) {
    return 0x7Eu;
  }
#if HY4_HAS_HW_FP8_F16_CVT
  unsigned short packed;
  asm("{ cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1; }" : "=h"(packed) : "f"(x));
  unsigned int b = static_cast<unsigned int>(packed & 0x7Fu);
  return (b > 0x7Eu) ? 0x7Eu : b;
#else
  const unsigned int u = __float_as_uint(x);
  const int exp32 = static_cast<int>((u >> 23) & 0xFFu);
  const unsigned int mant32 = u & 0x7FFFFFu;
  int e8 = exp32 - 120;
  if (e8 > 0) {
    unsigned int m3 = mant32 >> 20;
    const unsigned int rem = mant32 & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u))) {
      ++m3;
      if (m3 == 8u) {
        m3 = 0u;
        ++e8;
      }
    }
    const unsigned int res = (static_cast<unsigned int>(e8) << 3) | m3;
    return (res > 0x7Eu) ? 0x7Eu : res;
  }
  int k = __float2int_rn(x * 512.0f);
  if (k < 0) k = 0;
  if (k > 8) k = 8;
  return static_cast<unsigned int>(k);
#endif
}

// Exact round-to-nearest-even FP32 -> 4-bit E2M1 nibble (software fallback).
__device__ __forceinline__ unsigned int hy4_f32_to_e2m1_sw(float x) {
  const unsigned int sign = (__float_as_uint(x) >> 28) & 8u;
  const float a = fabsf(x);
  unsigned int code;
  if (a <= 0.25f) {
    code = 0u;
  } else if (a < 0.75f) {
    code = 1u;
  } else if (a <= 1.25f) {
    code = 2u;
  } else if (a < 1.75f) {
    code = 3u;
  } else if (a <= 2.5f) {
    code = 4u;
  } else if (a < 3.5f) {
    code = 5u;
  } else if (a <= 5.0f) {
    code = 6u;
  } else {
    code = 7u;
  }
  return sign | code;
}

__device__ __forceinline__ unsigned short hy4_pack_f32x4_to_e2m1x4(
    float v0, float v1, float v2, float v3) {
#if HY4_HAS_HW_FP4_ENCODE
  unsigned short out;
  asm("{ .reg .b8 q0, q1;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q0, %2, %1;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q1, %4, %3;\n"
      "  mov.b16 %0, {q0, q1}; }\n"
      : "=h"(out)
      : "f"(v0), "f"(v1), "f"(v2), "f"(v3));
  return out;
#else
  const unsigned int c0 = hy4_f32_to_e2m1_sw(v0);
  const unsigned int c1 = hy4_f32_to_e2m1_sw(v1);
  const unsigned int c2 = hy4_f32_to_e2m1_sw(v2);
  const unsigned int c3 = hy4_f32_to_e2m1_sw(v3);
  return static_cast<unsigned short>(c0 | (c1 << 4) | (c2 << 8) | (c3 << 12));
#endif
}

// ============================================================================
// 1. Fused BF16 -> NVFP4 Activation Quantizer + Direct 128x4 Scale Swizzle
//    Input:  x [M, K] BF16 (K >= 64, K % 64 == 0)
//    Output: x_q [M, K/8] uint32 (packed 8x E2M1 per u32)
//            x_s_swizzled [pad_M, K/16] uint8 (128x4 cuBLASLt tiled, 0-padded)
//    Grid: dim3(K / 64, ceil(M / 128), 1), Block: dim3(256, 1, 1)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4) void hy4_nvfp4_quantize_x_swizzled(
    const nv_bfloat16* __restrict__ x,
    unsigned int* __restrict__ x_q,
    unsigned char* __restrict__ x_s_swizzled,
    int M, int K) {
  const int cb = static_cast<int>(blockIdx.x);
  const int rb = static_cast<int>(blockIdx.y);
  const int cbg_cnt = K >> 6;  // K / 64
  if (cb >= cbg_cnt) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int c_in = tid & 3;     // 0..3 (4 consecutive 16-elem groups = 64 cols)
  const int r_half = tid >> 2;  // 0..63
  const int grp_base = lane & ~3;

  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  uint2* xq_u2 = reinterpret_cast<uint2*>(x_q);
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;   // K / 8
  const size_t xq_stride_u2 = static_cast<size_t>(K) >> 4;  // K / 16

  __shared__ unsigned int s_tile[32][5]; // +1 padded for 0 bank conflicts

#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int r_in = pass * 64 + r_half; // 0..127
    const int r = rb * 128 + r_in;
    unsigned int q_s = 0u;

    if (r < M) {
      const uint4* xp = x_u4 + static_cast<size_t>(r) * x_stride_u4 +
          static_cast<size_t>(cb) * 8 + c_in * 2;
      const uint4 raw0 = xp[0];
      const uint4 raw1 = xp[1];
      const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&raw0);
      const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&raw1);

      const float2 f0 = __bfloat1622float2(p0[0]);
      const float2 f1 = __bfloat1622float2(p0[1]);
      const float2 f2 = __bfloat1622float2(p0[2]);
      const float2 f3 = __bfloat1622float2(p0[3]);
      const float2 f4 = __bfloat1622float2(p1[0]);
      const float2 f5 = __bfloat1622float2(p1[1]);
      const float2 f6 = __bfloat1622float2(p1[2]);
      const float2 f7 = __bfloat1622float2(p1[3]);

      float amax = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f4.x), fabsf(f4.y)), fmaxf(fabsf(f5.x), fabsf(f5.y))));
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f6.x), fabsf(f6.y)), fmaxf(fabsf(f7.x), fabsf(f7.y))));

      q_s = hy4_f32_to_ue4m3_rn(amax / 6.0f);
      const float s_f = hy4_fast_e4m3_to_f(q_s);
      uint2 out_q = make_uint2(0u, 0u);
      if (s_f > 0.0f) {
        const float inv = 1.0f / s_f;
        const unsigned short pk0 =
            hy4_pack_f32x4_to_e2m1x4(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv);
        const unsigned short pk1 =
            hy4_pack_f32x4_to_e2m1x4(f2.x * inv, f2.y * inv, f3.x * inv, f3.y * inv);
        const unsigned short pk2 =
            hy4_pack_f32x4_to_e2m1x4(f4.x * inv, f4.y * inv, f5.x * inv, f5.y * inv);
        const unsigned short pk3 =
            hy4_pack_f32x4_to_e2m1x4(f6.x * inv, f6.y * inv, f7.x * inv, f7.y * inv);
        out_q.x = static_cast<unsigned int>(pk0) | (static_cast<unsigned int>(pk1) << 16);
        out_q.y = static_cast<unsigned int>(pk2) | (static_cast<unsigned int>(pk3) << 16);
      }
      xq_u2[static_cast<size_t>(r) * xq_stride_u2 + static_cast<size_t>(cb) * 4 + c_in] = out_q;
    }

    const unsigned int s0 = __shfl_sync(0xFFFFFFFFu, q_s, grp_base + 0);
    const unsigned int s1 = __shfl_sync(0xFFFFFFFFu, q_s, grp_base + 1);
    const unsigned int s2 = __shfl_sync(0xFFFFFFFFu, q_s, grp_base + 2);
    const unsigned int s3 = __shfl_sync(0xFFFFFFFFu, q_s, grp_base + 3);
    if (c_in == 0) {
      s_tile[r_in & 31][r_in >> 5] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    }
  }
  __syncthreads();

  if (tid < 32) {
    const uint4 sv = make_uint4(
        s_tile[tid][0], s_tile[tid][1], s_tile[tid][2], s_tile[tid][3]);
    reinterpret_cast<uint4*>(x_s_swizzled)[
        (static_cast<size_t>(rb) * cbg_cnt + cb) * 32 + tid] = sv;
  }
}
)cuda"
R"cuda(
// ============================================================================
// 2. Exact BF16-Activation x NVFP4-Weight Prefill GEMM (BM=64, BN=128, BK=64)
//    8 warps (256 thr), 2-stage double-buffered cp.async.cg/ca, 0 bank conflicts,
//    0 B spills, arbitrary M >= 1, N >= 1, K >= 64 (K % 64 == 0).
//    SMEM: 25.0 KB (CUDA 13.2+) or 26.0 KB (CUDA 13.0 with 1 KB E2M1 LUT).
//    Grid: dim3(ceil(M / 64) * ceil(N / 128), 1, 1), Block: dim3(32, 8, 1)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 2) void hy4_nvfp4_prefill_bf16_bm64(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  constexpr int BM = 64;
  constexpr int BN = 128;

  const int num_m_tiles = (M + BM - 1) / BM;
  const int num_n_tiles = (N + BN - 1) / BN;
  constexpr int BAND_M = 4;
  const int lin = static_cast<int>(blockIdx.x);
  const int tiles_per_band = BAND_M * num_n_tiles;
  const int band = lin / tiles_per_band;
  const int in_band = lin - band * tiles_per_band;
  const int band_m0 = band * BAND_M;
  const int band_rows = min(BAND_M, num_m_tiles - band_m0);
  const int n_tile = in_band / band_rows;
  const int m_tile = band_m0 + (in_band - n_tile * band_rows);
  if (n_tile >= num_n_tiles || m_tile >= num_m_tiles) {
    return;
  }

  const int m0 = m_tile * BM;
  const int n0 = n_tile * BN;
  const int cnt = min(BM, M - m0);
  if (cnt <= 0) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  union SharedBuf {
    struct {
      alignas(16) uint4 w[2][BN][2];       // 8,192 B
      alignas(16) uint4 x[2][BM][8];       // 16,384 B
      alignas(16) unsigned int sc[2][BN];  // 1,024 B
    } pipe;
    alignas(16) nv_bfloat16 s_out[8][16][16]; // 4,096 B
  };
  __shared__ SharedBuf sh;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
#else
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int k_steps = K >> 6;  // K / 64
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 5;   // K / 32
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;   // K / 8
  const size_t sc_stride_u32 = static_cast<size_t>(K) >> 6; // K / 64

  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  const unsigned int* sc_u32 = reinterpret_cast<const unsigned int*>(scales);

  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_stage = [&](int st, int step) {
    {
      const int r = tid >> 1;
      const int c = tid & 1;
      const int gn = n0 + r;
      uint4* dst = &sh.pipe.w[st][r][c ^ ((r >> 2) & 1)];
      if (gn < N) {
        ptx_cp_async16_cg(dst, w_u4 + static_cast<size_t>(gn) * w_stride_u4 + step * 2 + c);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    if (tid < BN) {
      const int gn = n0 + tid;
      if (gn < N) {
        ptx_cp_async4_ca(&sh.pipe.sc[st][tid],
                         sc_u32 + static_cast<size_t>(gn) * sc_stride_u32 + step);
      } else {
        sh.pipe.sc[st][tid] = 0u;
      }
    }
    const int total_x = load_x_rows * 8;
    for (int idx = tid; idx < total_x; idx += 256) {
      const int r = idx >> 3;
      const int c = idx & 7;
      const int gm = m0 + r;
      uint4* dst = &sh.pipe.x[st][r][c ^ ((r & 3) << 1)];
      if (gm < M) {
        ptx_cp_async16_ca(dst, x_u4 + static_cast<size_t>(gm) * x_stride_u4 + step * 8 + c);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    ptx_cp_commit();
  };

  float tot[8][4];
#pragma unroll
  for (int ms = 0; ms < 8; ++ms) {
    tot[ms][0] = tot[ms][1] = tot[ms][2] = tot[ms][3] = 0.0f;
  }

  if (k_steps > 0) {
    load_stage(0, 0);
  }

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;
  const int sw0 = (r0 >> 2) & 1;
  const int sw1 = (r1 >> 2) & 1;

#pragma unroll 1
  for (int step = 0; step < k_steps; ++step) {
    const int cur = step & 1;
    const int nxt = step + 1;
    ptx_cp_wait<0>();
    __syncthreads();
    if (nxt < k_steps) {
      load_stage(nxt & 1, nxt);
    }

    const unsigned int sc_w0 = sh.pipe.sc[cur][r0];
    const unsigned int sc_w1 = sh.pipe.sc[cur][r1];

#pragma unroll 1
    for (int lg2 = 0; lg2 < 2; ++lg2) {
      const float2 sf0 = hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w0 >> (lg2 * 16)));
      const float2 sf1 = hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w1 >> (lg2 * 16)));
      const unsigned short* wp0 =
          reinterpret_cast<const unsigned short*>(&sh.pipe.w[cur][r0][lg2 ^ sw0]);
      const unsigned short* wp1 =
          reinterpret_cast<const unsigned short*>(&sh.pipe.w[cur][r1][lg2 ^ sw1]);
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int lg = lg2 * 2 + sub;
        const unsigned short w0 = wp0[sub * 4 + t];
        const unsigned short w1 = wp1[sub * 4 + t];
        const unsigned int a0 = hy4_lut_e2m1x2_to_bf16x2(w0 & 0xFFu, s_e2m1_lut);
        const unsigned int a2 = hy4_lut_e2m1x2_to_bf16x2(w0 >> 8, s_e2m1_lut);
        const unsigned int a1 = hy4_lut_e2m1x2_to_bf16x2(w1 & 0xFFu, s_e2m1_lut);
        const unsigned int a3 = hy4_lut_e2m1x2_to_bf16x2(w1 >> 8, s_e2m1_lut);
        const float sc0 = (sub == 0) ? sf0.x : sf0.y;
        const float sc1 = (sub == 0) ? sf1.x : sf1.y;
        const int x_col = (lg * 2 + (t >> 1)) ^ ((g & 3) << 1);
        const int x_hf = t & 1;

#pragma unroll
        for (int ms = 0; ms < 8; ++ms) {
          if (ms < n_slices) {
            const int rx = ms * 8 + g;
            const uint2 xv =
                reinterpret_cast<const uint2*>(&sh.pipe.x[cur][rx][x_col])[x_hf];
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv.x, xv.y);
            tot[ms][0] = fmaf(sc0, d0, tot[ms][0]);
            tot[ms][1] = fmaf(sc0, d1, tot[ms][1]);
            tot[ms][2] = fmaf(sc1, d2, tot[ms][2]);
            tot[ms][3] = fmaf(sc1, d3, tot[ms][3]);
          }
        }
      }
    }
  }
  __syncthreads();

  const int w_col0 = n0 + warp * 16;
  const bool full_n16 = (w_col0 + 16 <= N) && ((N & 7) == 0);
  const int n_passes = (cnt + 15) >> 4;
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      sh.s_out[warp][2 * t][g]         = __float2bfloat16_rn(tot[s0][0]);
      sh.s_out[warp][2 * t + 1][g]     = __float2bfloat16_rn(tot[s0][1]);
      sh.s_out[warp][2 * t][g + 8]     = __float2bfloat16_rn(tot[s0][2]);
      sh.s_out[warp][2 * t + 1][g + 8] = __float2bfloat16_rn(tot[s0][3]);
      if (s1 < n_slices) {
        sh.s_out[warp][8 + 2 * t][g]         = __float2bfloat16_rn(tot[s1][0]);
        sh.s_out[warp][8 + 2 * t + 1][g]     = __float2bfloat16_rn(tot[s1][1]);
        sh.s_out[warp][8 + 2 * t][g + 8]     = __float2bfloat16_rn(tot[s1][2]);
        sh.s_out[warp][8 + 2 * t + 1][g + 8] = __float2bfloat16_rn(tot[s1][3]);
      }
      __syncwarp();
      if (full_n16) {
        const int r_in = lane >> 1;
        const int half = lane & 1;
        const int r_loc = pass * 16 + r_in;
        if (r_loc < cnt) {
          ptx_st_cs_u4(
              reinterpret_cast<uint4*>(out + static_cast<size_t>(m0 + r_loc) * N + w_col0) + half,
              reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
        }
      } else {
        for (int i = lane; i < 256; i += 32) {
          const int r_in = i >> 4;
          const int c_in = i & 15;
          const int r_loc = pass * 16 + r_in;
          const int gn = w_col0 + c_in;
          if (r_loc < cnt && gn < N) {
            out[static_cast<size_t>(m0 + r_loc) * N + gn] = sh.s_out[warp][r_in][c_in];
          }
        }
      }
      __syncwarp();
    }
  }
}
)cuda"
