R"cuda(
// ============================================================================
// Hy4 CUDA NVFP4 Dense Few-Row (M = 1..16) & Fused LM Head Kernels
//   Requires ptx_helpers.cu.inc and hy4_fp4_helpers.cu.inc prepended.
//   Dual-target PTX:
//     - CUDA 13.2+ (sm_121a): hardware cvt.rn.bf16x2.{e2m1x2,e4m3x2}
//     - CUDA 13.0  (sm_121a): SMEM 256-entry E2M1 LUT + PTX 8.1 cvt.rn.f16x2.e4m3x2
//     - sm_80 fallback: pure bitwise decode
// ============================================================================

template <int M_CAP>
__device__ __forceinline__ void hy4_nvfp4_dot16(
    unsigned int wx,
    unsigned int wy,
    const uint4* __restrict__ x_u4,
    size_t x_stride_u4,
    size_t x_off_u4,
    int M,
    float sc,
    const unsigned int* __restrict__ lut,
    float acc[M_CAP]) {
  const unsigned int bf0 = hy4_lut_e2m1x2_to_bf16x2(wx, lut);
  const unsigned int bf1 = hy4_lut_e2m1x2_to_bf16x2(wx >> 8, lut);
  const unsigned int bf2 = hy4_lut_e2m1x2_to_bf16x2(wx >> 16, lut);
  const unsigned int bf3 = hy4_lut_e2m1x2_to_bf16x2(wx >> 24, lut);
  const unsigned int bf4 = hy4_lut_e2m1x2_to_bf16x2(wy, lut);
  const unsigned int bf5 = hy4_lut_e2m1x2_to_bf16x2(wy >> 8, lut);
  const unsigned int bf6 = hy4_lut_e2m1x2_to_bf16x2(wy >> 16, lut);
  const unsigned int bf7 = hy4_lut_e2m1x2_to_bf16x2(wy >> 24, lut);

  const float2 w0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf0));
  const float2 w1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf1));
  const float2 w2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
  const float2 w3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf3));
  const float2 w4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf4));
  const float2 w5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf5));
  const float2 w6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf6));
  const float2 w7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf7));

#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      const uint4* xp = x_u4 + static_cast<size_t>(m) * x_stride_u4 + x_off_u4;
      const uint4 x0 = xp[0];
      const uint4 x1 = xp[1];
      const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
      const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
      const float2 f0 = __bfloat1622float2(p0[0]);
      const float2 f1 = __bfloat1622float2(p0[1]);
      const float2 f2 = __bfloat1622float2(p0[2]);
      const float2 f3 = __bfloat1622float2(p0[3]);
      const float2 f4 = __bfloat1622float2(p1[0]);
      const float2 f5 = __bfloat1622float2(p1[1]);
      const float2 f6 = __bfloat1622float2(p1[2]);
      const float2 f7 = __bfloat1622float2(p1[3]);
      float d = 0.0f;
      d = fmaf(w0.x, f0.x, fmaf(w0.y, f0.y, d));
      d = fmaf(w1.x, f1.x, fmaf(w1.y, f1.y, d));
      d = fmaf(w2.x, f2.x, fmaf(w2.y, f2.y, d));
      d = fmaf(w3.x, f3.x, fmaf(w3.y, f3.y, d));
      d = fmaf(w4.x, f4.x, fmaf(w4.y, f4.y, d));
      d = fmaf(w5.x, f5.x, fmaf(w5.y, f5.y, d));
      d = fmaf(w6.x, f6.x, fmaf(w6.y, f6.y, d));
      d = fmaf(w7.x, f7.x, fmaf(w7.y, f7.y, d));
      acc[m] = fmaf(sc, d, acc[m]);
    }
  }
}

template <int M_CAP>
__device__ __forceinline__ void hy4_nvfp4_u4_step(
    const uint4& wv,
    unsigned short sc_pair,
    const uint4* __restrict__ x_u4,
    size_t x_stride_u4,
    int u,
    int M,
    const unsigned int* __restrict__ lut,
    float acc[M_CAP]) {
  const float2 sf = hy4_fast_e4m3x2_to_f2(sc_pair);
  const size_t x_off = static_cast<size_t>(u) * 4;
  hy4_nvfp4_dot16<M_CAP>(wv.x, wv.y, x_u4, x_stride_u4, x_off, M, sf.x, lut, acc);
  hy4_nvfp4_dot16<M_CAP>(wv.z, wv.w, x_u4, x_stride_u4, x_off + 2, M, sf.y, lut, acc);
}

// ============================================================================
// 1A0. NVFP4 Streaming uint4 2-Row/Warp GEMV for M = 1 (N >= 192)
//      Unpacks 1 uint4 of x at a time across 2 output rows (2x x reuse + 2x MLP,
//      zero SMEM in HW CVT mode). Uses NUM_WARPS=2 (4 rows/CTA) for K<=1536 and
//      NUM_WARPS=4 (8 rows/CTA) for K>1536 to eliminate wave-tail quantization.
//      Grid: dim3((N + NUM_WARPS*2 - 1) / (NUM_WARPS*2), 1, 1), Block: dim3(32, NUM_WARPS, 1)
// ============================================================================
__device__ __forceinline__ void hy4_nvfp4_dot8_f2(
    unsigned int wword,
    const float2 f0,
    const float2 f1,
    const float2 f2,
    const float2 f3,
    const unsigned int* __restrict__ lut,
    float& d0,
    float& d1) {
  const unsigned int bf0 = hy4_lut_e2m1x2_to_bf16x2(wword, lut);
  const unsigned int bf1 = hy4_lut_e2m1x2_to_bf16x2(wword >> 8, lut);
  const unsigned int bf2 = hy4_lut_e2m1x2_to_bf16x2(wword >> 16, lut);
  const unsigned int bf3 = hy4_lut_e2m1x2_to_bf16x2(wword >> 24, lut);
  const float2 w0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf0));
  const float2 w1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf1));
  const float2 w2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
  const float2 w3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf3));
  d0 = fmaf(w0.x, f0.x, fmaf(w1.x, f1.x, fmaf(w2.x, f2.x, fmaf(w3.x, f3.x, d0))));
  d1 = fmaf(w0.y, f0.y, fmaf(w1.y, f1.y, fmaf(w2.y, f2.y, fmaf(w3.y, f3.y, d1))));
}

template <int NUM_WARPS>
__global__ __launch_bounds__(NUM_WARPS * 32, (NUM_WARPS == 4) ? 8 : 12) void hy4_nvfp4_few_u4_2row_m1(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    nv_bfloat16* __restrict__ out,
    int N, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  for (int i = tid; i < 256; i += NUM_WARPS * 32) {
    s_e2m1_lut[i] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(i));
  }
  __syncthreads();
#else
  (void)tid;
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int nA = static_cast<int>(blockIdx.x) * (NUM_WARPS * 2) + warp * 2;
  const int nB = nA + 1;
  if (nA >= N) {
    return;
  }
  const bool validB = (nB < N);

  const int num_u4 = K >> 5;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const unsigned short* s_u16 = reinterpret_cast<const unsigned short*>(scales);
  const uint4* w_rowA = w_u4 + static_cast<size_t>(nA) * num_u4;
  const unsigned short* s_rowA = s_u16 + static_cast<size_t>(nA) * num_u4;
  const uint4* w_rowB = w_u4 + static_cast<size_t>(validB ? nB : nA) * num_u4;
  const unsigned short* s_rowB = s_u16 + static_cast<size_t>(validB ? nB : nA) * num_u4;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);

  float accA = 0.0f;
  float accB = 0.0f;

  for (int u = lane; u < num_u4; u += 32) {
    const uint4 wvA = ptx_ld_cs_u4(w_rowA + u);
    const unsigned short spA = s_rowA[u];
    const uint4 wvB = validB ? ptx_ld_cs_u4(w_rowB + u) : make_uint4(0u, 0u, 0u, 0u);
    const unsigned short spB = validB ? s_rowB[u] : 0;
    const float2 sfA = hy4_fast_e4m3x2_to_f2(spA);
    const float2 sfB = validB ? hy4_fast_e4m3x2_to_f2(spB) : make_float2(0.0f, 0.0f);
    const uint4* xp = x_u4 + static_cast<size_t>(u) * 4;

    float dA0 = 0.0f, dA1 = 0.0f, dB0 = 0.0f, dB1 = 0.0f;
    {
      const uint4 xv = xp[0];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(p[0]);
      const float2 f1 = __bfloat1622float2(p[1]);
      const float2 f2 = __bfloat1622float2(p[2]);
      const float2 f3 = __bfloat1622float2(p[3]);
      hy4_nvfp4_dot8_f2(wvA.x, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) {
        hy4_nvfp4_dot8_f2(wvB.x, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
      }
    }
    {
      const uint4 xv = xp[1];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(p[0]);
      const float2 f1 = __bfloat1622float2(p[1]);
      const float2 f2 = __bfloat1622float2(p[2]);
      const float2 f3 = __bfloat1622float2(p[3]);
      hy4_nvfp4_dot8_f2(wvA.y, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) {
        hy4_nvfp4_dot8_f2(wvB.y, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
      }
    }
    accA = fmaf(sfA.x, dA0 + dA1, accA);
    if (validB) {
      accB = fmaf(sfB.x, dB0 + dB1, accB);
    }

    dA0 = 0.0f; dA1 = 0.0f; dB0 = 0.0f; dB1 = 0.0f;
    {
      const uint4 xv = xp[2];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(p[0]);
      const float2 f1 = __bfloat1622float2(p[1]);
      const float2 f2 = __bfloat1622float2(p[2]);
      const float2 f3 = __bfloat1622float2(p[3]);
      hy4_nvfp4_dot8_f2(wvA.z, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) {
        hy4_nvfp4_dot8_f2(wvB.z, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
      }
    }
    {
      const uint4 xv = xp[3];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(p[0]);
      const float2 f1 = __bfloat1622float2(p[1]);
      const float2 f2 = __bfloat1622float2(p[2]);
      const float2 f3 = __bfloat1622float2(p[3]);
      hy4_nvfp4_dot8_f2(wvA.w, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) {
        hy4_nvfp4_dot8_f2(wvB.w, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
      }
    }
    accA = fmaf(sfA.y, dA0 + dA1, accA);
    if (validB) {
      accB = fmaf(sfB.y, dB0 + dB1, accB);
    }
  }

  float vA = hy4_warp_sum(accA);
  float vB = validB ? hy4_warp_sum(accB) : 0.0f;
  if (lane == 0) {
    out[nA] = __float2bfloat16_rn(vA);
    if (validB) {
      out[nB] = __float2bfloat16_rn(vB);
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 1A. NVFP4 Streaming uint4 GEMV (pre-Ampere fallback, 1 warp/row, 8 warps/CTA)
//     2x software-pipelined ld.global.cs.v4.u32 with dual accumulators, zero SMEM (HW CVT).
//     Grid: dim3((N + 7) / 8, 1, 1), Block: dim3(32, 8, 1)
// ============================================================================
template <int M_CAP>
__global__ __launch_bounds__(256, (M_CAP >= 8) ? 2 : ((M_CAP >= 4) ? 3 : 4)) void hy4_nvfp4_few_u4(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  __syncthreads();
#else
  (void)tid;
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int n = static_cast<int>(blockIdx.x) * 8 + warp;
  if (n >= N) {
    return;
  }

  const int num_u4 = K >> 5;  // K / 32 (each uint4 = 32 NVFP4 nibbles = 2 groups)
  const uint4* w_row =
      reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
  const unsigned short* s_row =
      reinterpret_cast<const unsigned short*>(scales) + static_cast<size_t>(n) * num_u4;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  float acc0[M_CAP];
  float acc1[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc0[m] = 0.0f;
    acc1[m] = 0.0f;
  }

  int u_base = 0;
  for (; u_base + 64 <= num_u4; u_base += 64) {
    const int u = u_base + lane;
    const uint4 wv0 = ptx_ld_cs_u4(w_row + u);
    const unsigned short sp0 = s_row[u];
    const uint4 wv1 = ptx_ld_cs_u4(w_row + u + 32);
    const unsigned short sp1 = s_row[u + 32];
    hy4_nvfp4_u4_step<M_CAP>(wv0, sp0, x_u4, x_stride_u4, u, M, s_e2m1_lut, acc0);
    hy4_nvfp4_u4_step<M_CAP>(wv1, sp1, x_u4, x_stride_u4, u + 32, M, s_e2m1_lut, acc1);
  }
  const int u = u_base + lane;
  if (u < num_u4) {
    const uint4 wv0 = ptx_ld_cs_u4(w_row + u);
    const unsigned short sp0 = s_row[u];
    const bool has1 = (u + 32 < num_u4);
    uint4 wv1 = make_uint4(0u, 0u, 0u, 0u);
    unsigned short sp1 = 0;
    if (has1) {
      wv1 = ptx_ld_cs_u4(w_row + u + 32);
      sp1 = s_row[u + 32];
    }
    hy4_nvfp4_u4_step<M_CAP>(wv0, sp0, x_u4, x_stride_u4, u, M, s_e2m1_lut, acc0);
    if (has1) {
      hy4_nvfp4_u4_step<M_CAP>(wv1, sp1, x_u4, x_stride_u4, u + 32, M, s_e2m1_lut, acc1);
    }
  }

#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      float v = hy4_warp_sum(acc0[m] + acc1[m]);
      if (lane == 0) {
        out[static_cast<size_t>(m) * N + n] = __float2bfloat16_rn(v);
      }
    }
  }
}

// ============================================================================
// 1B. NVFP4 Streaming uint4 1-Row/CTA GEMV (for narrow N < 192 or cc < 8)
//     Grid: dim3(N, 1, 1), Block: dim3(32, 4, 1)
// ============================================================================
template <int M_CAP>
__global__ __launch_bounds__(128, 4) void hy4_nvfp4_few_u4_1row(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  s_e2m1_lut[tid + 128] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid + 128));
  __syncthreads();
#else
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) {
    return;
  }

  const int num_u4 = K >> 5;
  const uint4* w_row =
      reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
  const unsigned short* s_row =
      reinterpret_cast<const unsigned short*>(scales) + static_cast<size_t>(n) * num_u4;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  for (int u = tid; u < num_u4; u += 128) {
    const uint4 wv = ptx_ld_cs_u4(w_row + u);
    const unsigned short sp = s_row[u];
    hy4_nvfp4_u4_step<M_CAP>(wv, sp, x_u4, x_stride_u4, u, M, s_e2m1_lut, acc);
  }

  __shared__ float s_red[4][M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      float v = hy4_warp_sum(acc[m]);
      if (lane == 0) {
        s_red[warp][m] = v;
      }
    }
  }
  __syncthreads();
  if (warp == 0 && lane < M) {
    float sum = s_red[0][lane] + s_red[1][lane] + s_red[2][lane] + s_red[3][lane];
    out[static_cast<size_t>(lane) * N + n] = __float2bfloat16_rn(sum);
  }
}
)cuda"
R"cuda(
// ============================================================================
// 2. NVFP4 2-Stage BK=512 cp.async.cg + BF16 Tensor Core MMA (M = 1..16, optional Split-K)
//    16 output rows/CTA, 512 elements (32 NVFP4 groups) per K-tile, 8 warps/CTA.
//    All 256 threads load 4 KB of weights via cp.async16_cg; 32 threads load 512 B
//    of scales via 16B cp.async16_cg (with 4B fallback when K%256!=0 or unaligned).
//    4-CTA group-staggered tile traversal preserves L1 x sharing while breaking
//    power-of-2 DRAM bank / L2-set camping (e.g. K=8192, K=1024).
//    Grid: dim3((N + 15) / 16, split_k, 1), Block: dim3(32, 8, 1)
// ============================================================================
template <int M_TILES, bool SPLIT_K>
__global__ __launch_bounds__(256, 2) void hy4_nvfp4_few_mma(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    void* __restrict__ out,
    int M, int N, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  __syncthreads();
#else
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int row0 = static_cast<int>(blockIdx.x) * 16;
  if (row0 >= N) {
    return;
  }

  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;

  const int num_groups = K >> 4;         // K / 16 (multiple of 4)
  const int num_tiles = num_groups >> 5; // full 512-elem (32-group) tiles
  const int tail_groups = num_groups & 31;
  const int tile_start = (num_tiles * split_idx) / num_splits;
  const int tile_end = (num_tiles * (split_idx + 1)) / num_splits;
  const int n_tiles_mine = tile_end - tile_start;

  const size_t w_stride_u4 = static_cast<size_t>(K) >> 5;   // K / 32
  const size_t sc_stride_u4 = static_cast<size_t>(K) >> 8;  // K / 256 (when K % 256 == 0)
  const size_t x_stride_u2 = static_cast<size_t>(K) >> 2;   // K / 4 (each uint2 = 4 BF16s)

  const bool sc_aligned16 =
      ((K & 255) == 0) && ((reinterpret_cast<unsigned long long>(scales) & 15u) == 0);

  union SharedBuf {
    struct {
      alignas(16) uint4 w[2][16][16]; // 8,192 B (XOR swizzled, 0 bank conflicts)
      alignas(16) uint4 sc[2][16][2]; // 1,024 B (32 u8 scales/row)
    } pipe;
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const uint4* sc_u4 = reinterpret_cast<const uint4*>(scales);

  const int ld_row = tid >> 4;  // 0..15
  const int ld_col = tid & 15;  // 0..15
  const int ld_swz = ld_col ^ ld_row;
  const int ld_r = (row0 + ld_row < N) ? (row0 + ld_row) : (N - 1);
  const uint4* w_load_base = w_u4 + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;

  const int sc16_row = tid >> 1; // 0..15 (for tid < 32)
  const int sc16_col = tid & 1;  // 0..1
  const int sc16_r = (row0 + sc16_row < N) ? (row0 + sc16_row) : (N - 1);
  const uint4* sc16_load_base =
      sc_u4 + static_cast<size_t>(sc16_r) * sc_stride_u4 + sc16_col;

  const bool valid_m0 = (g < M);
  const uint2* x0_row_u2 = reinterpret_cast<const uint2*>(x) +
      static_cast<size_t>(valid_m0 ? g : 0) * x_stride_u2 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  [[maybe_unused]] const uint2* x1_row_u2 = reinterpret_cast<const uint2*>(x) +
      static_cast<size_t>(valid_m1 ? (g + 8) : 0) * x_stride_u2 + t;

  float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int stagger = (n_tiles_mine > 1)
      ? ((static_cast<int>(blockIdx.x) >> 2) % n_tiles_mine)
      : 0;
  auto map_tile = [&](int idx) -> int {
    int loc = idx + stagger;
    if (loc >= n_tiles_mine) {
      loc -= n_tiles_mine;
    }
    return tile_start + loc;
  };

  auto load_tile = [&](int st, int tile) {
    ptx_cp_async16_cg(
        &sh.pipe.w[st][ld_row][ld_swz], w_load_base + static_cast<size_t>(tile) * 16);
    if (sc_aligned16) {
      if (tid < 32) {
        ptx_cp_async16_cg(
            &sh.pipe.sc[st][sc16_row][sc16_col],
            sc16_load_base + static_cast<size_t>(tile) * 2);
      }
    } else if (tid < 128) {
      const int sc4_row = tid >> 3;
      const int sc4_col = tid & 7;
      const int sc4_r = (row0 + sc4_row < N) ? (row0 + sc4_row) : (N - 1);
      const size_t sc_stride_u32 = static_cast<size_t>(K) >> 6;
      ptx_cp_async4_ca(
          reinterpret_cast<unsigned int*>(&sh.pipe.sc[st][sc4_row][0]) + sc4_col,
          reinterpret_cast<const unsigned int*>(scales) +
              static_cast<size_t>(sc4_r) * sc_stride_u32 +
              static_cast<size_t>(tile) * 8 + sc4_col);
    }
    ptx_cp_commit();
  };

  if (n_tiles_mine > 0) {
    load_tile(0, map_tile(0));
    ptx_cp_wait<0>();
    __syncthreads();

    for (int i = 0; i < n_tiles_mine; ++i) {
      const int cur = i & 1;
      const int nxt = cur ^ 1;
      if (i + 1 < n_tiles_mine) {
        load_tile(nxt, map_tile(i + 1));
      }

      const int tile = map_tile(i);
      const unsigned int sc_u32_0 =
          reinterpret_cast<const unsigned int*>(&sh.pipe.sc[cur][g][0])[warp];
      const unsigned int sc_u32_1 =
          reinterpret_cast<const unsigned int*>(&sh.pipe.sc[cur][g + 8][0])[warp];
      const float2 sf0_lo =
          hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_0 & 0xFFFFu));
      const float2 sf0_hi =
          hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_0 >> 16));
      const float2 sf1_lo =
          hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_1 & 0xFFFFu));
      const float2 sf1_hi =
          hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_1 >> 16));

#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = warp * 2 + h;
        const unsigned short* wp0 =
            reinterpret_cast<const unsigned short*>(&sh.pipe.w[cur][g][col ^ g]);
        const unsigned short* wp1 =
            reinterpret_cast<const unsigned short*>(&sh.pipe.w[cur][g + 8][col ^ (g + 8)]);
        const float2 sf0 = (h == 0) ? sf0_lo : sf0_hi;
        const float2 sf1 = (h == 0) ? sf1_lo : sf1_hi;

#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const unsigned short w0 = wp0[sub * 4 + t];
          const unsigned short w1 = wp1[sub * 4 + t];
          const unsigned int a0 = hy4_lut_e2m1x2_to_bf16x2(w0 & 0xFFu, s_e2m1_lut);
          const unsigned int a2 = hy4_lut_e2m1x2_to_bf16x2(w0 >> 8, s_e2m1_lut);
          const unsigned int a1 = hy4_lut_e2m1x2_to_bf16x2(w1 & 0xFFu, s_e2m1_lut);
          const unsigned int a3 = hy4_lut_e2m1x2_to_bf16x2(w1 >> 8, s_e2m1_lut);
          const float sc0 = (sub == 0) ? sf0.x : sf0.y;
          const float sc1 = (sub == 0) ? sf1.x : sf1.y;
          const size_t x_off = static_cast<size_t>(tile * 32 + col * 2 + sub) * 4;

          const uint2 xv0 = valid_m0 ? x0_row_u2[x_off] : make_uint2(0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
          total0[0] = fmaf(sc0, d0, total0[0]);
          total0[1] = fmaf(sc0, d1, total0[1]);
          total0[2] = fmaf(sc1, d2, total0[2]);
          total0[3] = fmaf(sc1, d3, total0[3]);

          if constexpr (M_TILES == 2) {
            const uint2 xv1 = valid_m1 ? x1_row_u2[x_off] : make_uint2(0u, 0u);
            float h0, h1, h2, h3;
            ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
            total1[0] = fmaf(sc0, h0, total1[0]);
            total1[1] = fmaf(sc0, h1, total1[1]);
            total1[2] = fmaf(sc1, h2, total1[2]);
            total1[3] = fmaf(sc1, h3, total1[3]);
          }
        }
      }

      if (i + 1 < n_tiles_mine) {
        ptx_cp_wait<0>();
      }
      __syncthreads();
    }
  }

  if (tail_groups > 0 && split_idx == num_splits - 1) {
    const int r0 = (row0 + g < N) ? (row0 + g) : (N - 1);
    const int r1 = (row0 + g + 8 < N) ? (row0 + g + 8) : (N - 1);
    const size_t w_stride_u16 = static_cast<size_t>(K) >> 2;
    const unsigned short* w_u16 = reinterpret_cast<const unsigned short*>(w);
    for (int lg = warp; lg < tail_groups; lg += 8) {
      const int grp = num_tiles * 32 + lg;
      const unsigned short w0 =
          w_u16[static_cast<size_t>(r0) * w_stride_u16 + static_cast<size_t>(grp) * 4 + t];
      const unsigned short w1 =
          w_u16[static_cast<size_t>(r1) * w_stride_u16 + static_cast<size_t>(grp) * 4 + t];
      const unsigned int a0 = hy4_lut_e2m1x2_to_bf16x2(w0 & 0xFFu, s_e2m1_lut);
      const unsigned int a2 = hy4_lut_e2m1x2_to_bf16x2(w0 >> 8, s_e2m1_lut);
      const unsigned int a1 = hy4_lut_e2m1x2_to_bf16x2(w1 & 0xFFu, s_e2m1_lut);
      const unsigned int a3 = hy4_lut_e2m1x2_to_bf16x2(w1 >> 8, s_e2m1_lut);
      const float sc0 = hy4_fast_e4m3_to_f(scales[static_cast<size_t>(r0) * num_groups + grp]);
      const float sc1 = hy4_fast_e4m3_to_f(scales[static_cast<size_t>(r1) * num_groups + grp]);
      const size_t x_off = static_cast<size_t>(grp) * 4;

      const uint2 xv0 = valid_m0 ? x0_row_u2[x_off] : make_uint2(0u, 0u);
      float d0, d1, d2, d3;
      ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
      total0[0] = fmaf(sc0, d0, total0[0]);
      total0[1] = fmaf(sc0, d1, total0[1]);
      total0[2] = fmaf(sc1, d2, total0[2]);
      total0[3] = fmaf(sc1, d3, total0[3]);

      if constexpr (M_TILES == 2) {
        const uint2 xv1 = valid_m1 ? x1_row_u2[x_off] : make_uint2(0u, 0u);
        float h0, h1, h2, h3;
        ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
        total1[0] = fmaf(sc0, h0, total1[0]);
        total1[1] = fmaf(sc0, h1, total1[1]);
        total1[2] = fmaf(sc1, h2, total1[2]);
        total1[3] = fmaf(sc1, h3, total1[3]);
      }
    }
  }
  __syncthreads();

  sh.reduce[warp][2 * t][g] = total0[0];
  sh.reduce[warp][2 * t + 1][g] = total0[1];
  sh.reduce[warp][2 * t][g + 8] = total0[2];
  sh.reduce[warp][2 * t + 1][g + 8] = total0[3];
  if constexpr (M_TILES == 2) {
    sh.reduce[warp][8 + 2 * t][g] = total1[0];
    sh.reduce[warp][8 + 2 * t + 1][g] = total1[1];
    sh.reduce[warp][8 + 2 * t][g + 8] = total1[2];
    sh.reduce[warp][8 + 2 * t + 1][g + 8] = total1[3];
  }
  __syncthreads();

  for (int idx = tid; idx < M * 16; idx += 256) {
    const int m = idx >> 4;
    const int n_loc = idx & 15;
    const int n_col = row0 + n_loc;
    if (n_col < N) {
      float sum = 0.0f;
#pragma unroll
      for (int ww = 0; ww < 8; ++ww) {
        sum += sh.reduce[ww][m][n_loc];
      }
      if constexpr (SPLIT_K) {
        reinterpret_cast<float*>(out)[
            (static_cast<size_t>(split_idx) * M + m) * N + n_col] = sum;
      } else {
        reinterpret_cast<nv_bfloat16*>(out)[static_cast<size_t>(m) * N + n_col] =
            __float2bfloat16_rn(sum);
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(64, 4) void hy4_splitk_reduce_f32_to_bf16(
    const float* __restrict__ partials,
    nv_bfloat16* __restrict__ out,
    int M, int N, int split_k) {
  const int total = M * N;
  const int tid = static_cast<int>(blockIdx.x) * 64 + static_cast<int>(threadIdx.x);
  if ((total & 3) == 0) {
    const int total4 = total >> 2;
    if (tid >= total4) {
      return;
    }
    const float4* p4 = reinterpret_cast<const float4*>(partials);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < split_k; ++s) {
      const float4 v = p4[static_cast<size_t>(s) * total4 + tid];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(acc.x, acc.y);
    const __nv_bfloat162 b1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 pk;
    pk.x = *reinterpret_cast<const unsigned int*>(&b0);
    pk.y = *reinterpret_cast<const unsigned int*>(&b1);
    reinterpret_cast<uint2*>(out)[tid] = pk;
    return;
  }
  if (tid >= total) {
    return;
  }
  float sum = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    sum += partials[static_cast<size_t>(s) * total + tid];
  }
  out[tid] = __float2bfloat16_rn(sum);
}
)cuda"
R"cuda(
// ============================================================================
// 3. Hy4 Fused LM Head Kernels (NVFP4, SymQ8, BF16) + Zero-DRAM 2-Stage Argmax
//    Supports arbitrary M >= 1 via grid.y = ceil(M / 16) (M <= 16 uses grid.y=1).
//    OUT_MODE 0: FP32 logits [M, V]
//    OUT_MODE 1: Persistent Stage-1 partials [M, num_ctas] -> Stage-2 (i32[M], f32[M])
// ============================================================================

__device__ __forceinline__ void hy4_lm_argmax_update(
    float& bv, int& bi, float v, int idx) {
  if (idx >= 0) {
    const bool v_nan = isnan(v);
    const bool b_nan = isnan(bv);
    if (bi < 0 || (!v_nan && (b_nan || v > bv || (v == bv && idx < bi))) ||
        (v_nan && b_nan && idx < bi)) {
      bv = v;
      bi = idx;
    }
  }
}

template <int SB_DTYPE>
__device__ __forceinline__ float hy4_lm_load_sb1(const void* ptr, size_t idx) {
  if constexpr (SB_DTYPE == 0) {
    return reinterpret_cast<const float*>(ptr)[idx];
  } else if constexpr (SB_DTYPE == 1) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  } else {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
  }
}

template <int PAIR>
__device__ __forceinline__ unsigned int hy4_lm_unpack_symq8_pair(unsigned int w) {
  constexpr int shift = PAIR * 16;
  const unsigned int sw = w ^ 0x80808080u;
  const float f0 = static_cast<float>(static_cast<signed char>((sw >> shift) & 0xFFu));
  const float f1 = static_cast<float>(static_cast<signed char>((sw >> (shift + 8)) & 0xFFu));
  const __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

)cuda"
R"cuda(
// ============================================================================
// 3A. LM Head NVFP4 2-Stage BK=512 cp.async.cg + BF16 TC MMA (M >= 1)
//     All 256 threads load 4 KB of weights via cp.async16_cg; 32 threads load 512 B
//     of scales via 16B cp.async16_cg (4B fallback when K%256!=0 or unaligned).
//     OUT_MODE=1 uses contiguous row-tile chunking per persistent CTA for DRAM
//     row-buffer and TLB locality.
//     Grid: dim3(num_ctas, ceil(M / 16), 1), Block: dim3(32, 8, 1)
// ============================================================================
template <int M_TILES, int OUT_MODE>
__global__ __launch_bounds__(256, 2) void hy4_lm_head_nvfp4_mma(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  __syncthreads();
#else
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int m_base = static_cast<int>(blockIdx.y) * 16;
  const int m_cnt = min(16, M - m_base);
  if (m_cnt <= 0) {
    return;
  }

  const int total_rt = (V + 15) / 16;
  int rt_start, rt_end;
  if constexpr (OUT_MODE == 1) {
    rt_start = (total_rt * static_cast<int>(blockIdx.x)) / static_cast<int>(gridDim.x);
    rt_end = (total_rt * static_cast<int>(blockIdx.x + 1)) / static_cast<int>(gridDim.x);
  } else {
    rt_start = static_cast<int>(blockIdx.x);
    rt_end = rt_start + 1;
    if (rt_start >= total_rt) {
      return;
    }
  }

  const int num_groups = K >> 4;
  const int num_tiles = num_groups >> 5; // 512-elem (32-group) tiles
  const int tail_groups = num_groups & 31;
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 5;
  const size_t sc_stride_u4 = static_cast<size_t>(K) >> 8;
  const size_t x_stride_u2 = static_cast<size_t>(K) >> 2;

  const bool sc_aligned16 =
      ((K & 255) == 0) && ((reinterpret_cast<unsigned long long>(scales) & 15u) == 0);

  union SharedBuf {
    struct {
      alignas(16) uint4 w[2][16][16];
      alignas(16) uint4 sc[2][16][2];
    } pipe;
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const uint4* sc_u4 = reinterpret_cast<const uint4*>(scales);
  const unsigned short* w_u16 = reinterpret_cast<const unsigned short*>(w);

  const int ld_row = tid >> 4;
  const int ld_col = tid & 15;
  const int ld_swz = ld_col ^ ld_row;
  const int sc16_row = tid >> 1;
  const int sc16_col = tid & 1;

  const bool valid_m0 = (g < m_cnt);
  const uint2* x0_row_u2 = reinterpret_cast<const uint2*>(x) +
      static_cast<size_t>(m_base + (valid_m0 ? g : 0)) * x_stride_u2 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < m_cnt);
  [[maybe_unused]] const uint2* x1_row_u2 = reinterpret_cast<const uint2*>(x) +
      static_cast<size_t>(m_base + (valid_m1 ? (g + 8) : 0)) * x_stride_u2 + t;

  [[maybe_unused]] float best_v0 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i0 = -1;
  [[maybe_unused]] float best_v1 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i1 = -1;

  for (int rt = rt_start; rt < rt_end; ++rt) {
    const int row0 = rt * 16;
    const int ld_r = (row0 + ld_row < V) ? (row0 + ld_row) : (V - 1);
    const uint4* w_load_base = w_u4 + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;
    const int sc16_r = (row0 + sc16_row < V) ? (row0 + sc16_row) : (V - 1);
    const uint4* sc16_load_base =
        sc_u4 + static_cast<size_t>(sc16_r) * sc_stride_u4 + sc16_col;

    float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (num_tiles > 0) {
      auto issue_lm_nvfp4 = [&](int st, size_t nt) {
        ptx_cp_async16_cg(&sh.pipe.w[st][ld_row][ld_swz], w_load_base + nt * 16);
        if (sc_aligned16) {
          if (tid < 32) {
            ptx_cp_async16_cg(&sh.pipe.sc[st][sc16_row][sc16_col], sc16_load_base + nt * 2);
          }
        } else if (tid < 128) {
          const int sc4_row = tid >> 3;
          const int sc4_col = tid & 7;
          const int sc4_r = (row0 + sc4_row < V) ? (row0 + sc4_row) : (V - 1);
          const size_t sc_stride_u32 = static_cast<size_t>(K) >> 6;
          ptx_cp_async4_ca(
              reinterpret_cast<unsigned int*>(&sh.pipe.sc[st][sc4_row][0]) + sc4_col,
              reinterpret_cast<const unsigned int*>(scales) +
                  static_cast<size_t>(sc4_r) * sc_stride_u32 + nt * 8 + sc4_col);
        }
        ptx_cp_commit();
      };
      issue_lm_nvfp4(0, 0);
      ptx_cp_wait<0>();
      __syncthreads();

      for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile & 1;
        const int nxt = cur ^ 1;
        if (tile + 1 < num_tiles) {
          issue_lm_nvfp4(nxt, static_cast<size_t>(tile + 1));
        }

        const unsigned int sc_u32_0 =
            reinterpret_cast<const unsigned int*>(&sh.pipe.sc[cur][g][0])[warp];
        const unsigned int sc_u32_1 =
            reinterpret_cast<const unsigned int*>(&sh.pipe.sc[cur][g + 8][0])[warp];
        const float2 sf0_lo =
            hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_0 & 0xFFFFu));
        const float2 sf0_hi =
            hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_0 >> 16));
        const float2 sf1_lo =
            hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_1 & 0xFFFFu));
        const float2 sf1_hi =
            hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_u32_1 >> 16));

#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int col = warp * 2 + h;
          const unsigned short* wp0 =
              reinterpret_cast<const unsigned short*>(&sh.pipe.w[cur][g][col ^ g]);
          const unsigned short* wp1 =
              reinterpret_cast<const unsigned short*>(&sh.pipe.w[cur][g + 8][col ^ (g + 8)]);
          const float2 sf0 = (h == 0) ? sf0_lo : sf0_hi;
          const float2 sf1 = (h == 0) ? sf1_lo : sf1_hi;

#pragma unroll
          for (int sub = 0; sub < 2; ++sub) {
            const unsigned short w0 = wp0[sub * 4 + t];
            const unsigned short w1 = wp1[sub * 4 + t];
            const unsigned int a0 = hy4_lut_e2m1x2_to_bf16x2(w0 & 0xFFu, s_e2m1_lut);
            const unsigned int a2 = hy4_lut_e2m1x2_to_bf16x2(w0 >> 8, s_e2m1_lut);
            const unsigned int a1 = hy4_lut_e2m1x2_to_bf16x2(w1 & 0xFFu, s_e2m1_lut);
            const unsigned int a3 = hy4_lut_e2m1x2_to_bf16x2(w1 >> 8, s_e2m1_lut);
            const float sc0 = (sub == 0) ? sf0.x : sf0.y;
            const float sc1 = (sub == 0) ? sf1.x : sf1.y;
            const size_t x_off = static_cast<size_t>(tile * 32 + col * 2 + sub) * 4;

            const uint2 xv0 = valid_m0 ? x0_row_u2[x_off] : make_uint2(0u, 0u);
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
            total0[0] = fmaf(sc0, d0, total0[0]);
            total0[1] = fmaf(sc0, d1, total0[1]);
            total0[2] = fmaf(sc1, d2, total0[2]);
            total0[3] = fmaf(sc1, d3, total0[3]);

            if constexpr (M_TILES == 2) {
              const uint2 xv1 = valid_m1 ? x1_row_u2[x_off] : make_uint2(0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
              total1[0] = fmaf(sc0, h0, total1[0]);
              total1[1] = fmaf(sc0, h1, total1[1]);
              total1[2] = fmaf(sc1, h2, total1[2]);
              total1[3] = fmaf(sc1, h3, total1[3]);
            }
          }
        }

        if (tile + 1 < num_tiles) {
          ptx_cp_wait<0>();
        }
        __syncthreads();
      }
    }
)cuda"
R"cuda(
    if (tail_groups > 0) {
      const int r0 = (row0 + g < V) ? (row0 + g) : (V - 1);
      const int r1 = (row0 + g + 8 < V) ? (row0 + g + 8) : (V - 1);
      const size_t w_stride_u16 = static_cast<size_t>(K) >> 2;
      for (int lg = warp; lg < tail_groups; lg += 8) {
        const int grp = num_tiles * 32 + lg;
        const unsigned short w0 =
            w_u16[static_cast<size_t>(r0) * w_stride_u16 + static_cast<size_t>(grp) * 4 + t];
        const unsigned short w1 =
            w_u16[static_cast<size_t>(r1) * w_stride_u16 + static_cast<size_t>(grp) * 4 + t];
        const unsigned int a0 = hy4_lut_e2m1x2_to_bf16x2(w0 & 0xFFu, s_e2m1_lut);
        const unsigned int a2 = hy4_lut_e2m1x2_to_bf16x2(w0 >> 8, s_e2m1_lut);
        const unsigned int a1 = hy4_lut_e2m1x2_to_bf16x2(w1 & 0xFFu, s_e2m1_lut);
        const unsigned int a3 = hy4_lut_e2m1x2_to_bf16x2(w1 >> 8, s_e2m1_lut);
        const float sc0 = hy4_fast_e4m3_to_f(scales[static_cast<size_t>(r0) * num_groups + grp]);
        const float sc1 = hy4_fast_e4m3_to_f(scales[static_cast<size_t>(r1) * num_groups + grp]);
        const size_t x_off = static_cast<size_t>(grp) * 4;

        const uint2 xv0 = valid_m0 ? x0_row_u2[x_off] : make_uint2(0u, 0u);
        float d0, d1, d2, d3;
        ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
        total0[0] = fmaf(sc0, d0, total0[0]);
        total0[1] = fmaf(sc0, d1, total0[1]);
        total0[2] = fmaf(sc1, d2, total0[2]);
        total0[3] = fmaf(sc1, d3, total0[3]);

        if constexpr (M_TILES == 2) {
          const uint2 xv1 = valid_m1 ? x1_row_u2[x_off] : make_uint2(0u, 0u);
          float h0, h1, h2, h3;
          ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
          total1[0] = fmaf(sc0, h0, total1[0]);
          total1[1] = fmaf(sc0, h1, total1[1]);
          total1[2] = fmaf(sc1, h2, total1[2]);
          total1[3] = fmaf(sc1, h3, total1[3]);
        }
      }
    }
    __syncthreads();

    sh.reduce[warp][2 * t][g] = total0[0];
    sh.reduce[warp][2 * t + 1][g] = total0[1];
    sh.reduce[warp][2 * t][g + 8] = total0[2];
    sh.reduce[warp][2 * t + 1][g + 8] = total0[3];
    if constexpr (M_TILES == 2) {
      sh.reduce[warp][8 + 2 * t][g] = total1[0];
      sh.reduce[warp][8 + 2 * t + 1][g] = total1[1];
      sh.reduce[warp][8 + 2 * t][g + 8] = total1[2];
      sh.reduce[warp][8 + 2 * t + 1][g + 8] = total1[3];
    }
    __syncthreads();

    if constexpr (OUT_MODE == 1) {
      if (warp < m_cnt) {
        float v = -__int_as_float(0x7f800000);
        int idx = -1;
        if (lane < 16 && row0 + lane < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][warp][lane];
          }
          v = sum;
          idx = row0 + lane;
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, v, off);
          const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
          hy4_lm_argmax_update(v, idx, ov, oi);
        }
        if (lane == 0) {
          hy4_lm_argmax_update(best_v0, best_i0, v, idx);
        }
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < m_cnt) {
          float v = -__int_as_float(0x7f800000);
          int idx = -1;
          if (lane < 16 && row0 + lane < V) {
            float sum = 0.0f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww) {
              sum += sh.reduce[ww][warp + 8][lane];
            }
            v = sum;
            idx = row0 + lane;
          }
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            hy4_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            hy4_lm_argmax_update(best_v1, best_i1, v, idx);
          }
        }
      }
      __syncthreads();
    } else {
      for (int idx = tid; idx < m_cnt * 16; idx += 256) {
        const int m_loc = idx >> 4;
        const int n_loc = idx & 15;
        const int n_col = row0 + n_loc;
        if (n_col < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][m_loc][n_loc];
          }
          out[static_cast<size_t>(m_base + m_loc) * V + n_col] = sum;
        }
      }
    }
  }

  if constexpr (OUT_MODE == 1) {
    if (lane == 0) {
      if (warp < m_cnt) {
        const size_t slot = static_cast<size_t>(m_base + warp) * gridDim.x + blockIdx.x;
        out[slot] = best_v0;
        partial_idxs[slot] = best_i0;
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < m_cnt) {
          const size_t slot = static_cast<size_t>(m_base + warp + 8) * gridDim.x + blockIdx.x;
          out[slot] = best_v1;
          partial_idxs[slot] = best_i1;
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3A2. LM Head NVFP4 2-Row/Warp Streaming uint4 for M = 1, OUT_MODE = 0 (FP32 logits)
//      4 warps/CTA (128 thr, 8 rows/CTA), zero SMEM in HW CVT mode, ~280 GB/s cold-L2.
//      Grid: dim3((V + 7) / 8, 1, 1), Block: dim3(32, 4, 1)
// ============================================================================
__global__ __launch_bounds__(128, 6) void hy4_lm_head_nvfp4_u4_2row_m1(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    int V, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  s_e2m1_lut[tid + 128] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid + 128));
  __syncthreads();
#else
  (void)tid;
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int nA = static_cast<int>(blockIdx.x) * 8 + warp * 2;
  const int nB = nA + 1;
  if (nA >= V) {
    return;
  }
  const bool validB = (nB < V);

  const int num_u4 = K >> 5;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const unsigned short* s_u16 = reinterpret_cast<const unsigned short*>(scales);
  const uint4* w_rowA = w_u4 + static_cast<size_t>(nA) * num_u4;
  const unsigned short* s_rowA = s_u16 + static_cast<size_t>(nA) * num_u4;
  const uint4* w_rowB = w_u4 + static_cast<size_t>(validB ? nB : nA) * num_u4;
  const unsigned short* s_rowB = s_u16 + static_cast<size_t>(validB ? nB : nA) * num_u4;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);

  float accA = 0.0f;
  float accB = 0.0f;
  for (int u = lane; u < num_u4; u += 32) {
    const uint4 wvA = ptx_ld_cs_u4(w_rowA + u);
    const unsigned short spA = s_rowA[u];
    const uint4 wvB = validB ? ptx_ld_cs_u4(w_rowB + u) : make_uint4(0u, 0u, 0u, 0u);
    const unsigned short spB = validB ? s_rowB[u] : 0;
    const float2 sfA = hy4_fast_e4m3x2_to_f2(spA);
    const float2 sfB = validB ? hy4_fast_e4m3x2_to_f2(spB) : make_float2(0.0f, 0.0f);
    const uint4* xp = x_u4 + static_cast<size_t>(u) * 4;

    float dA0 = 0.0f, dA1 = 0.0f, dB0 = 0.0f, dB1 = 0.0f;
    {
      const uint4 xv = xp[0];
      const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(q[0]), f1 = __bfloat1622float2(q[1]);
      const float2 f2 = __bfloat1622float2(q[2]), f3 = __bfloat1622float2(q[3]);
      hy4_nvfp4_dot8_f2(wvA.x, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) hy4_nvfp4_dot8_f2(wvB.x, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
    }
    {
      const uint4 xv = xp[1];
      const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(q[0]), f1 = __bfloat1622float2(q[1]);
      const float2 f2 = __bfloat1622float2(q[2]), f3 = __bfloat1622float2(q[3]);
      hy4_nvfp4_dot8_f2(wvA.y, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) hy4_nvfp4_dot8_f2(wvB.y, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
    }
    accA = fmaf(sfA.x, dA0 + dA1, accA);
    if (validB) accB = fmaf(sfB.x, dB0 + dB1, accB);

    dA0 = 0.0f; dA1 = 0.0f; dB0 = 0.0f; dB1 = 0.0f;
    {
      const uint4 xv = xp[2];
      const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(q[0]), f1 = __bfloat1622float2(q[1]);
      const float2 f2 = __bfloat1622float2(q[2]), f3 = __bfloat1622float2(q[3]);
      hy4_nvfp4_dot8_f2(wvA.z, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) hy4_nvfp4_dot8_f2(wvB.z, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
    }
    {
      const uint4 xv = xp[3];
      const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 f0 = __bfloat1622float2(q[0]), f1 = __bfloat1622float2(q[1]);
      const float2 f2 = __bfloat1622float2(q[2]), f3 = __bfloat1622float2(q[3]);
      hy4_nvfp4_dot8_f2(wvA.w, f0, f1, f2, f3, s_e2m1_lut, dA0, dA1);
      if (validB) hy4_nvfp4_dot8_f2(wvB.w, f0, f1, f2, f3, s_e2m1_lut, dB0, dB1);
    }
    accA = fmaf(sfA.y, dA0 + dA1, accA);
    if (validB) accB = fmaf(sfB.y, dB0 + dB1, accB);
  }

  float vA = hy4_warp_sum(accA);
  float vB = validB ? hy4_warp_sum(accB) : 0.0f;
  if (lane == 0) {
    out[nA] = vA;
    if (validB) out[nB] = vB;
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3B. LM Head NVFP4 Streaming uint4 (optional M<=2 / cc<8 fallback)
//     Grid: dim3(num_ctas, ceil(M / M_CAP), 1), Block: dim3(32, 8, 1)
// ============================================================================
template <int M_CAP, int OUT_MODE>
__global__ __launch_bounds__(256, (M_CAP >= 4 && OUT_MODE == 1) ? 3 : 4) void hy4_lm_head_nvfp4_u4(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  __syncthreads();
#else
  (void)tid;
  const unsigned int* s_e2m1_lut = nullptr;
#endif

  const int m_base = static_cast<int>(blockIdx.y) * M_CAP;
  const int m_cnt = min(M_CAP, M - m_base);
  if (m_cnt <= 0) {
    return;
  }

  const int n_start = static_cast<int>(blockIdx.x) * 8 + warp;
  if constexpr (OUT_MODE != 1) {
    if (n_start >= V) {
      return;
    }
  }
  const int n_step = static_cast<int>(gridDim.x) * 8;

  const int num_u4 = K >> 5;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base) * x_stride_u4;

  [[maybe_unused]] float best_v[M_CAP];
  [[maybe_unused]] int best_i[M_CAP];
  if constexpr (OUT_MODE == 1) {
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      best_v[m] = -__int_as_float(0x7f800000);
      best_i[m] = -1;
    }
  }

  for (int n = n_start; n < V; n += n_step) {
    const uint4* w_row =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
    const unsigned short* s_row =
        reinterpret_cast<const unsigned short*>(scales) + static_cast<size_t>(n) * num_u4;

    float acc0[M_CAP];
    float acc1[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      acc0[m] = 0.0f;
      acc1[m] = 0.0f;
    }

    int u = lane;
    for (; u + 32 < num_u4; u += 64) {
      const uint4 wv0 = ptx_ld_cs_u4(w_row + u);
      const unsigned short sp0 = s_row[u];
      const uint4 wv1 = ptx_ld_cs_u4(w_row + u + 32);
      const unsigned short sp1 = s_row[u + 32];
      hy4_nvfp4_u4_step<M_CAP>(wv0, sp0, x_u4, x_stride_u4, u, m_cnt, s_e2m1_lut, acc0);
      hy4_nvfp4_u4_step<M_CAP>(wv1, sp1, x_u4, x_stride_u4, u + 32, m_cnt, s_e2m1_lut, acc1);
    }
    if (u < num_u4) {
      const uint4 wv0 = ptx_ld_cs_u4(w_row + u);
      const unsigned short sp0 = s_row[u];
      hy4_nvfp4_u4_step<M_CAP>(wv0, sp0, x_u4, x_stride_u4, u, m_cnt, s_e2m1_lut, acc0);
    }

#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < m_cnt) {
        float v = hy4_warp_sum(acc0[m] + acc1[m]);
        if (lane == 0) {
          if constexpr (OUT_MODE == 0) {
            out[static_cast<size_t>(m_base + m) * V + n] = v;
          } else {
            hy4_lm_argmax_update(best_v[m], best_i[m], v, n);
          }
        }
      }
    }
    if constexpr (OUT_MODE != 1) {
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    __shared__ float s_v[M_CAP][8];
    __shared__ int s_i[M_CAP][8];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        s_v[m][warp] = best_v[m];
        s_i[m][warp] = best_i[m];
      }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < m_cnt) {
          float v = (lane < 8) ? s_v[m][lane] : -__int_as_float(0x7f800000);
          int idx = (lane < 8) ? s_i[m][lane] : -1;
#pragma unroll
          for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            hy4_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            const size_t slot = static_cast<size_t>(m_base + m) * gridDim.x + blockIdx.x;
            out[slot] = v;
            partial_idxs[slot] = idx;
          }
        }
      }
    }
  }
}

)cuda"
R"cuda(
// ============================================================================
// 3B2. LM Head SymQ8 Streaming uint4 (M <= 4 or cc < 8; zero SMEM in OUT_MODE=0)
//      1 warp/row, 8 warps/CTA, 2x unrolled ld.global.cs.v4.u32, shared weight decode.
//      Grid: dim3(num_ctas, ceil(M / M_CAP), 1), Block: dim3(32, 8, 1)
// ============================================================================
__device__ __forceinline__ void hy4_lm_symq8_unpack8(
    unsigned int u0, unsigned int u1, float wf[8]) {
  const unsigned int s0 = u0 ^ 0x80808080u;
  const unsigned int s1 = u1 ^ 0x80808080u;
  wf[0] = static_cast<float>(static_cast<signed char>(s0));
  wf[1] = static_cast<float>(static_cast<signed char>(s0 >> 8));
  wf[2] = static_cast<float>(static_cast<signed char>(s0 >> 16));
  wf[3] = static_cast<float>(static_cast<signed char>(s0 >> 24));
  wf[4] = static_cast<float>(static_cast<signed char>(s1));
  wf[5] = static_cast<float>(static_cast<signed char>(s1 >> 8));
  wf[6] = static_cast<float>(static_cast<signed char>(s1 >> 16));
  wf[7] = static_cast<float>(static_cast<signed char>(s1 >> 24));
}

__device__ __forceinline__ void hy4_lm_symq8_dot8(
    const float wf[8], const uint4& xv, float& d) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  d = fmaf(f0.x, wf[0], fmaf(f0.y, wf[1], d));
  d = fmaf(f1.x, wf[2], fmaf(f1.y, wf[3], d));
  d = fmaf(f2.x, wf[4], fmaf(f2.y, wf[5], d));
  d = fmaf(f3.x, wf[6], fmaf(f3.y, wf[7], d));
}

template <int M_CAP>
__device__ __forceinline__ void hy4_lm_symq8_u4_step(
    const uint4& wv,
    float sc,
    const uint4* __restrict__ x_u4,
    size_t x_stride_u4,
    int x_off,
    int m_cnt,
    float acc[M_CAP]) {
  const uint4* x_col = x_u4 + x_off;
  if constexpr (M_CAP == 1) {
    float wf[8];
    float d = 0.0f;
    hy4_lm_symq8_unpack8(wv.x, wv.y, wf);
    hy4_lm_symq8_dot8(wf, x_col[0], d);
    hy4_lm_symq8_unpack8(wv.z, wv.w, wf);
    hy4_lm_symq8_dot8(wf, x_col[1], d);
    acc[0] = fmaf(sc, d, acc[0]);
  } else {
    float d[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      d[m] = 0.0f;
    }
    float wf[8];
    hy4_lm_symq8_unpack8(wv.x, wv.y, wf);
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < m_cnt) {
        hy4_lm_symq8_dot8(wf, x_col[static_cast<size_t>(m) * x_stride_u4], d[m]);
      }
    }
    hy4_lm_symq8_unpack8(wv.z, wv.w, wf);
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < m_cnt) {
        hy4_lm_symq8_dot8(wf, x_col[static_cast<size_t>(m) * x_stride_u4 + 1], d[m]);
        acc[m] = fmaf(sc, d[m], acc[m]);
      }
    }
  }
}

template <int SB_DTYPE, int M_CAP, int OUT_MODE>
__global__ __launch_bounds__(256, 4) void hy4_lm_head_symq8_u4(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    float* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);

  const int m_base = static_cast<int>(blockIdx.y) * M_CAP;
  const int m_cnt = min(M_CAP, M - m_base);
  if (m_cnt <= 0) {
    return;
  }

  const int n_start = static_cast<int>(blockIdx.x) * 8 + warp;
  if constexpr (OUT_MODE != 1) {
    if (n_start >= V) {
      return;
    }
  }
  const int n_step = static_cast<int>(gridDim.x) * 8;

  const int num_u4 = K >> 4;
  const int groups = K >> 5;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base) * x_stride_u4;

  [[maybe_unused]] float best_v[M_CAP];
  [[maybe_unused]] int best_i[M_CAP];
  if constexpr (OUT_MODE == 1) {
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      best_v[m] = -__int_as_float(0x7f800000);
      best_i[m] = -1;
    }
  }

  for (int n = n_start; n < V; n += n_step) {
    const uint4* w_row =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
    const size_t sb_row = static_cast<size_t>(n) * groups;

    float acc[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      acc[m] = 0.0f;
    }

    int u = lane;
#pragma unroll 1
    for (; u + 32 < num_u4; u += 64) {
      const uint4 wv0 = ptx_ld_cs_u4(w_row + u);
      const uint4 wv1 = ptx_ld_cs_u4(w_row + u + 32);
      const float sc0 = hy4_lm_load_sb1<SB_DTYPE>(scales, sb_row + (u >> 1));
      hy4_lm_symq8_u4_step<M_CAP>(
          wv0, sc0, x_u4, x_stride_u4, u * 2, m_cnt, acc);
      const float sc1 = hy4_lm_load_sb1<SB_DTYPE>(scales, sb_row + ((u + 32) >> 1));
      hy4_lm_symq8_u4_step<M_CAP>(
          wv1, sc1, x_u4, x_stride_u4, (u + 32) * 2, m_cnt, acc);
    }
    if (u < num_u4) {
      const uint4 wv0 = ptx_ld_cs_u4(w_row + u);
      const float sc0 = hy4_lm_load_sb1<SB_DTYPE>(scales, sb_row + (u >> 1));
      hy4_lm_symq8_u4_step<M_CAP>(
          wv0, sc0, x_u4, x_stride_u4, u * 2, m_cnt, acc);
    }

#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < m_cnt) {
        float v = hy4_warp_sum(acc[m]);
        if (lane == 0) {
          if constexpr (OUT_MODE == 0) {
            out[static_cast<size_t>(m_base + m) * V + n] = v;
          } else {
            hy4_lm_argmax_update(best_v[m], best_i[m], v, n);
          }
        }
      }
    }
    if constexpr (OUT_MODE != 1) {
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    __shared__ float s_v[M_CAP][8];
    __shared__ int s_i[M_CAP][8];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        s_v[m][warp] = best_v[m];
        s_i[m][warp] = best_i[m];
      }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < m_cnt) {
          float v = (lane < 8) ? s_v[m][lane] : -__int_as_float(0x7f800000);
          int idx = (lane < 8) ? s_i[m][lane] : -1;
#pragma unroll
          for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            hy4_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            const size_t slot = static_cast<size_t>(m_base + m) * gridDim.x + blockIdx.x;
            out[slot] = v;
            partial_idxs[slot] = idx;
          }
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3C. LM Head SymQ8 2-Stage cp.async.cg + BF16 TC MMA (M >= 1)
//     16B cp.async.cg for w AND (when SB_DTYPE != 0 and 16B-aligned) scales;
//     contiguous row-tile chunking for OUT_MODE=1.
//     Grid: dim3(num_ctas, ceil(M / 16), 1), Block: dim3(32, 8, 1)
// ============================================================================
__device__ __forceinline__ unsigned int hy4_lm_s8x2_to_bf16x2(unsigned int p16) {
  const float f0 = static_cast<float>(static_cast<signed char>(p16));
  const float f1 = static_cast<float>(static_cast<signed char>(p16 >> 8));
  const __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

template <int SB_DTYPE, int M_TILES, int OUT_MODE>
__global__ __launch_bounds__(256, 2) void hy4_lm_head_symq8_mma(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    float* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int m_base = static_cast<int>(blockIdx.y) * 16;
  const int m_cnt = min(16, M - m_base);
  if (m_cnt <= 0) {
    return;
  }

  const int total_rt = (V + 15) / 16;
  int rt_start, rt_end;
  if constexpr (OUT_MODE == 1) {
    rt_start = (total_rt * static_cast<int>(blockIdx.x)) / static_cast<int>(gridDim.x);
    rt_end = (total_rt * static_cast<int>(blockIdx.x + 1)) / static_cast<int>(gridDim.x);
  } else {
    rt_start = static_cast<int>(blockIdx.x);
    rt_end = rt_start + 1;
    if (rt_start >= total_rt) {
      return;
    }
  }

  const int num_groups = K >> 5;
  const int num_tiles = num_groups >> 3;
  const int tail_groups = num_groups & 7;
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 4;
  const size_t sc_stride_u4 = static_cast<size_t>(K) >> 8;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  constexpr bool kSc16 = (SB_DTYPE != 0);
  const bool sc_aligned16 =
      kSc16 && ((K & 255) == 0) && ((reinterpret_cast<unsigned long long>(scales) & 15u) == 0);

  union SharedBuf {
    struct {
      alignas(16) uint4 w[2][16][16];
      alignas(16) uint4 sc16[kSc16 ? 2 : 1][kSc16 ? 16 : 1];
      float sc32[kSc16 ? 1 : 2][kSc16 ? 1 : 16][kSc16 ? 1 : 9];
    } pipe;
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const int ld_row = tid >> 4;
  const int ld_col = tid & 15;
  const int ld_swz = ld_col ^ ld_row;
  const int sb_row = (tid >> 3) & 15;
  const int sb_grp = tid & 7;

  const bool valid_m0 = (g < m_cnt);
  const uint4* x0_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base + (valid_m0 ? g : 0)) * x_stride_u4 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < m_cnt);
  [[maybe_unused]] const uint4* x1_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base + (valid_m1 ? (g + 8) : 0)) * x_stride_u4 + t;

  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ (g + 8);

  [[maybe_unused]] float best_v0 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i0 = -1;
  [[maybe_unused]] float best_v1 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i1 = -1;

  for (int rt = rt_start; rt < rt_end; ++rt) {
    const int row0 = rt * 16;
    const int ld_r = (row0 + ld_row < V) ? (row0 + ld_row) : (V - 1);
    const uint4* w_load_ptr =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;
    const int sc16_r = (row0 + tid < V) ? (row0 + tid) : (V - 1);
    [[maybe_unused]] const uint4* sc16_load_ptr =
        reinterpret_cast<const uint4*>(scales) + static_cast<size_t>(sc16_r) * sc_stride_u4;

    auto issue_stage = [&](int st, size_t tile_idx) {
      ptx_cp_async16_cg(&sh.pipe.w[st][ld_row][ld_swz], w_load_ptr + tile_idx * 16);
      if constexpr (kSc16) {
        if (sc_aligned16) {
          if (tid < 16) {
            ptx_cp_async16_cg(&sh.pipe.sc16[st][tid], sc16_load_ptr + tile_idx);
          }
          ptx_cp_commit();
        } else {
          ptx_cp_commit();
          if (tid < 128) {
            const int sb_r = (row0 + sb_row < V) ? (row0 + sb_row) : (V - 1);
            const size_t sb_idx =
                static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + tile_idx * 8 + sb_grp;
            reinterpret_cast<unsigned short*>(&sh.pipe.sc16[st][sb_row])[sb_grp] =
                reinterpret_cast<const unsigned short*>(scales)[sb_idx];
          }
        }
      } else {
        ptx_cp_commit();
        if (tid < 128) {
          const int sb_r = (row0 + sb_row < V) ? (row0 + sb_row) : (V - 1);
          const size_t sb_idx =
              static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + tile_idx * 8 + sb_grp;
          sh.pipe.sc32[st][sb_row][sb_grp] = hy4_lm_load_sb1<SB_DTYPE>(scales, sb_idx);
        }
      }
    };

    float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (num_tiles > 0) {
      issue_stage(0, 0);
      ptx_cp_wait<0>();
      __syncthreads();

      for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile & 1;
        const int nxt = cur ^ 1;
        if (tile + 1 < num_tiles) {
          issue_stage(nxt, static_cast<size_t>(tile + 1));
        }

        const uint2 w0 = reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g][swz_r0])[half_u2];
        const uint2 w1 = reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g + 8][swz_r1])[half_u2];
        const size_t x_off = static_cast<size_t>(tile * 8 + warp) * 4;
        const uint4 xv0 = valid_m0 ? x0_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);
        [[maybe_unused]] const uint4 xv1 =
            (M_TILES == 2 && valid_m1) ? x1_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);

        float d0[4];
        [[maybe_unused]] float d1[4];
        {
          const unsigned int s0 = w0.x ^ 0x80808080u;
          const unsigned int s1 = w1.x ^ 0x80808080u;
          const unsigned int a0 = hy4_lm_s8x2_to_bf16x2(s0);
          const unsigned int a1 = hy4_lm_s8x2_to_bf16x2(s1);
          const unsigned int a2 = hy4_lm_s8x2_to_bf16x2(s0 >> 16);
          const unsigned int a3 = hy4_lm_s8x2_to_bf16x2(s1 >> 16);
          ptx_mma_m16n8k16_bf16_z(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.x, xv0.y);
          if constexpr (M_TILES == 2) {
            ptx_mma_m16n8k16_bf16_z(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.x, xv1.y);
          }
        }
        {
          const unsigned int s0 = w0.y ^ 0x80808080u;
          const unsigned int s1 = w1.y ^ 0x80808080u;
          const unsigned int a0 = hy4_lm_s8x2_to_bf16x2(s0);
          const unsigned int a1 = hy4_lm_s8x2_to_bf16x2(s1);
          const unsigned int a2 = hy4_lm_s8x2_to_bf16x2(s0 >> 16);
          const unsigned int a3 = hy4_lm_s8x2_to_bf16x2(s1 >> 16);
          ptx_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.z, xv0.w);
          if constexpr (M_TILES == 2) {
            ptx_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.z, xv1.w);
          }
        }

        float sc0, sc1;
        if constexpr (SB_DTYPE == 2) {
          sc0 = __half2float(reinterpret_cast<const __half*>(&sh.pipe.sc16[cur][g])[warp]);
          sc1 = __half2float(reinterpret_cast<const __half*>(&sh.pipe.sc16[cur][g + 8])[warp]);
        } else if constexpr (SB_DTYPE == 1) {
          sc0 = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&sh.pipe.sc16[cur][g])[warp]);
          sc1 = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&sh.pipe.sc16[cur][g + 8])[warp]);
        } else {
          sc0 = sh.pipe.sc32[cur][g][warp];
          sc1 = sh.pipe.sc32[cur][g + 8][warp];
        }
        total0[0] = fmaf(sc0, d0[0], total0[0]);
        total0[1] = fmaf(sc0, d0[1], total0[1]);
        total0[2] = fmaf(sc1, d0[2], total0[2]);
        total0[3] = fmaf(sc1, d0[3], total0[3]);
        if constexpr (M_TILES == 2) {
          total1[0] = fmaf(sc0, d1[0], total1[0]);
          total1[1] = fmaf(sc0, d1[1], total1[1]);
          total1[2] = fmaf(sc1, d1[2], total1[2]);
          total1[3] = fmaf(sc1, d1[3], total1[3]);
        }

        if (tile + 1 < num_tiles) {
          ptx_cp_wait<0>();
        }
        __syncthreads();
      }
    }
)cuda"
R"cuda(
    if (tail_groups > 0 && warp < tail_groups) {
      const int grp_idx = num_tiles * 8 + warp;
      const int r0 = (row0 + g < V) ? (row0 + g) : (V - 1);
      const int r1 = (row0 + g + 8 < V) ? (row0 + g + 8) : (V - 1);
      const size_t w_stride_u2 = static_cast<size_t>(K) >> 3;
      const uint2 w0 = reinterpret_cast<const uint2*>(w)[
          static_cast<size_t>(r0) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];
      const uint2 w1 = reinterpret_cast<const uint2*>(w)[
          static_cast<size_t>(r1) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];
      const size_t x_off = static_cast<size_t>(grp_idx) * 4;
      const uint4 xv0 = valid_m0 ? x0_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);
      [[maybe_unused]] const uint4 xv1 =
          (M_TILES == 2 && valid_m1) ? x1_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);

      float d0[4];
      [[maybe_unused]] float d1[4];
      {
        const unsigned int s0 = w0.x ^ 0x80808080u;
        const unsigned int s1 = w1.x ^ 0x80808080u;
        const unsigned int a0 = hy4_lm_s8x2_to_bf16x2(s0);
        const unsigned int a1 = hy4_lm_s8x2_to_bf16x2(s1);
        const unsigned int a2 = hy4_lm_s8x2_to_bf16x2(s0 >> 16);
        const unsigned int a3 = hy4_lm_s8x2_to_bf16x2(s1 >> 16);
        ptx_mma_m16n8k16_bf16_z(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.x, xv0.y);
        if constexpr (M_TILES == 2) {
          ptx_mma_m16n8k16_bf16_z(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.x, xv1.y);
        }
      }
      {
        const unsigned int s0 = w0.y ^ 0x80808080u;
        const unsigned int s1 = w1.y ^ 0x80808080u;
        const unsigned int a0 = hy4_lm_s8x2_to_bf16x2(s0);
        const unsigned int a1 = hy4_lm_s8x2_to_bf16x2(s1);
        const unsigned int a2 = hy4_lm_s8x2_to_bf16x2(s0 >> 16);
        const unsigned int a3 = hy4_lm_s8x2_to_bf16x2(s1 >> 16);
        ptx_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.z, xv0.w);
        if constexpr (M_TILES == 2) {
          ptx_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.z, xv1.w);
        }
      }
      const float sc0 =
          hy4_lm_load_sb1<SB_DTYPE>(scales, static_cast<size_t>(r0) * num_groups + grp_idx);
      const float sc1 =
          hy4_lm_load_sb1<SB_DTYPE>(scales, static_cast<size_t>(r1) * num_groups + grp_idx);
      total0[0] = fmaf(sc0, d0[0], total0[0]);
      total0[1] = fmaf(sc0, d0[1], total0[1]);
      total0[2] = fmaf(sc1, d0[2], total0[2]);
      total0[3] = fmaf(sc1, d0[3], total0[3]);
      if constexpr (M_TILES == 2) {
        total1[0] = fmaf(sc0, d1[0], total1[0]);
        total1[1] = fmaf(sc0, d1[1], total1[1]);
        total1[2] = fmaf(sc1, d1[2], total1[2]);
        total1[3] = fmaf(sc1, d1[3], total1[3]);
      }
    }
    __syncthreads();

    sh.reduce[warp][2 * t][g] = total0[0];
    sh.reduce[warp][2 * t + 1][g] = total0[1];
    sh.reduce[warp][2 * t][g + 8] = total0[2];
    sh.reduce[warp][2 * t + 1][g + 8] = total0[3];
    if constexpr (M_TILES == 2) {
      sh.reduce[warp][8 + 2 * t][g] = total1[0];
      sh.reduce[warp][8 + 2 * t + 1][g] = total1[1];
      sh.reduce[warp][8 + 2 * t][g + 8] = total1[2];
      sh.reduce[warp][8 + 2 * t + 1][g + 8] = total1[3];
    }
    __syncthreads();

    if constexpr (OUT_MODE == 1) {
      if (warp < m_cnt) {
        float v = -__int_as_float(0x7f800000);
        int idx = -1;
        if (lane < 16 && row0 + lane < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][warp][lane];
          }
          v = sum;
          idx = row0 + lane;
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, v, off);
          const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
          hy4_lm_argmax_update(v, idx, ov, oi);
        }
        if (lane == 0) {
          hy4_lm_argmax_update(best_v0, best_i0, v, idx);
        }
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < m_cnt) {
          float v = -__int_as_float(0x7f800000);
          int idx = -1;
          if (lane < 16 && row0 + lane < V) {
            float sum = 0.0f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww) {
              sum += sh.reduce[ww][warp + 8][lane];
            }
            v = sum;
            idx = row0 + lane;
          }
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            hy4_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            hy4_lm_argmax_update(best_v1, best_i1, v, idx);
          }
        }
      }
      __syncthreads();
    } else {
      for (int idx = tid; idx < m_cnt * 16; idx += 256) {
        const int m_loc = idx >> 4;
        const int n_loc = idx & 15;
        const int n_col = row0 + n_loc;
        if (n_col < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][m_loc][n_loc];
          }
          out[static_cast<size_t>(m_base + m_loc) * V + n_col] = sum;
        }
      }
    }
  }

  if constexpr (OUT_MODE == 1) {
    if (lane == 0) {
      if (warp < m_cnt) {
        const size_t slot = static_cast<size_t>(m_base + warp) * gridDim.x + blockIdx.x;
        out[slot] = best_v0;
        partial_idxs[slot] = best_i0;
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < m_cnt) {
          const size_t slot = static_cast<size_t>(m_base + warp + 8) * gridDim.x + blockIdx.x;
          out[slot] = best_v1;
          partial_idxs[slot] = best_i1;
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3D. LM Head BF16 Dense 2-Stage cp.async.cg + Tensor Core MMA (M >= 1)
//     Grid: dim3(num_ctas, ceil(M / 16), 1), Block: dim3(32, 8, 1)
// ============================================================================
template <int M_TILES, int OUT_MODE>
__global__ __launch_bounds__(256, M_TILES == 1 ? 4 : 2) void hy4_lm_head_bf16_mma(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ w,
    float* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int m_base = static_cast<int>(blockIdx.y) * 16;
  const int m_cnt = min(16, M - m_base);
  if (m_cnt <= 0) {
    return;
  }

  const int row_start = static_cast<int>(blockIdx.x) * 16;
  if constexpr (OUT_MODE != 1) {
    if (row_start >= V) {
      return;
    }
  }
  const int row_step = static_cast<int>(gridDim.x) * 16;

  const int num_tiles = K >> 8;
  const int tail_k = K & 255;
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 3;

  union SharedBuf {
    alignas(16) uint4 w[2][16][32];
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const int ld_row = tid >> 4;
  const int ld_col = tid & 15;
  const int ld_rperm = ((ld_row & 1) << 2) | ((ld_row >> 1) & 3);
  const int ld_swz0 = ld_col ^ ld_rperm;
  const int ld_swz1 = (16 + ld_col) ^ ld_rperm;

  const bool valid_m0 = (g < m_cnt);
  [[maybe_unused]] const bool valid_m1 = (M_TILES == 2) && (g + 8 < m_cnt);
  const int col_u4 = warp * 4 + t;
  const int g_rperm = ((g & 1) << 2) | (g >> 1);
  const int swz_col = col_u4 ^ g_rperm;

  const uint4* x0_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base + (valid_m0 ? g : 0)) * w_stride_u4 + col_u4;
  [[maybe_unused]] const uint4* x1_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base + (valid_m1 ? (g + 8) : 0)) * w_stride_u4 + col_u4;

  [[maybe_unused]] float best_v0 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i0 = -1;
  [[maybe_unused]] float best_v1 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i1 = -1;

  for (int row0 = row_start; row0 < V; row0 += row_step) {
    const int ld_r = (row0 + ld_row < V) ? (row0 + ld_row) : (V - 1);
    const uint4* w_load_ptr =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;

    float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (num_tiles > 0) {
      ptx_cp_async16_cg(&sh.w[0][ld_row][ld_swz0], w_load_ptr);
      ptx_cp_async16_cg(&sh.w[0][ld_row][ld_swz1], w_load_ptr + 16);
      ptx_cp_commit();
      ptx_cp_wait<0>();
      __syncthreads();

      for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile & 1;
        const int nxt = cur ^ 1;
        if (tile + 1 < num_tiles) {
          const uint4* w_nxt = w_load_ptr + static_cast<size_t>(tile + 1) * 32;
          ptx_cp_async16_cg(&sh.w[nxt][ld_row][ld_swz0], w_nxt);
          ptx_cp_async16_cg(&sh.w[nxt][ld_row][ld_swz1], w_nxt + 16);
          ptx_cp_commit();
        }

        const uint4 w0 = sh.w[cur][g][swz_col];
        const uint4 w1 = sh.w[cur][g + 8][swz_col];
        const size_t x_tile_off = static_cast<size_t>(tile) * 32;
        const uint4 xv0 = valid_m0 ? x0_row_u4[x_tile_off] : make_uint4(0u, 0u, 0u, 0u);
        ptx_mma_m16n8k16_bf16_v(
            total0[0], total0[1], total0[2], total0[3],
            w0.x, w1.x, w0.y, w1.y, xv0.x, xv0.y);
        ptx_mma_m16n8k16_bf16_v(
            total0[0], total0[1], total0[2], total0[3],
            w0.z, w1.z, w0.w, w1.w, xv0.z, xv0.w);
        if constexpr (M_TILES == 2) {
          const uint4 xv1 = valid_m1 ? x1_row_u4[x_tile_off] : make_uint4(0u, 0u, 0u, 0u);
          ptx_mma_m16n8k16_bf16_v(
              total1[0], total1[1], total1[2], total1[3],
              w0.x, w1.x, w0.y, w1.y, xv1.x, xv1.y);
          ptx_mma_m16n8k16_bf16_v(
              total1[0], total1[1], total1[2], total1[3],
              w0.z, w1.z, w0.w, w1.w, xv1.z, xv1.w);
        }

        if (tile + 1 < num_tiles) {
          ptx_cp_wait<0>();
        }
        __syncthreads();
      }
    }

    if (tail_k > 0) {
      const int r0 = (row0 + g < V) ? (row0 + g) : (V - 1);
      const int r1 = (row0 + g + 8 < V) ? (row0 + g + 8) : (V - 1);
      for (int k_base = num_tiles * 256 + warp * 16; k_base < K; k_base += 128) {
        const int k_off = k_base + t * 4;
        const uint2 w0 = (k_off < K)
            ? *reinterpret_cast<const uint2*>(w + static_cast<size_t>(r0) * K + k_off)
            : make_uint2(0u, 0u);
        const uint2 w1 = (k_off < K)
            ? *reinterpret_cast<const uint2*>(w + static_cast<size_t>(r1) * K + k_off)
            : make_uint2(0u, 0u);
        const uint2 xb0 = (valid_m0 && k_off < K)
            ? *reinterpret_cast<const uint2*>(
                  x + static_cast<size_t>(m_base + g) * K + k_off)
            : make_uint2(0u, 0u);
        ptx_mma_m16n8k16_bf16_v(
            total0[0], total0[1], total0[2], total0[3],
            w0.x, w1.x, w0.y, w1.y, xb0.x, xb0.y);
        if constexpr (M_TILES == 2) {
          const uint2 xb1 = (valid_m1 && k_off < K)
              ? *reinterpret_cast<const uint2*>(
                    x + static_cast<size_t>(m_base + g + 8) * K + k_off)
              : make_uint2(0u, 0u);
          ptx_mma_m16n8k16_bf16_v(
              total1[0], total1[1], total1[2], total1[3],
              w0.x, w1.x, w0.y, w1.y, xb1.x, xb1.y);
        }
      }
    }
    __syncthreads();

    sh.reduce[warp][2 * t][g] = total0[0];
    sh.reduce[warp][2 * t + 1][g] = total0[1];
    sh.reduce[warp][2 * t][g + 8] = total0[2];
    sh.reduce[warp][2 * t + 1][g + 8] = total0[3];
    if constexpr (M_TILES == 2) {
      sh.reduce[warp][8 + 2 * t][g] = total1[0];
      sh.reduce[warp][8 + 2 * t + 1][g] = total1[1];
      sh.reduce[warp][8 + 2 * t][g + 8] = total1[2];
      sh.reduce[warp][8 + 2 * t + 1][g + 8] = total1[3];
    }
    __syncthreads();

    if constexpr (OUT_MODE == 1) {
      if (warp < m_cnt) {
        float v = -__int_as_float(0x7f800000);
        int idx = -1;
        if (lane < 16 && row0 + lane < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][warp][lane];
          }
          v = sum;
          idx = row0 + lane;
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, v, off);
          const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
          hy4_lm_argmax_update(v, idx, ov, oi);
        }
        if (lane == 0) {
          hy4_lm_argmax_update(best_v0, best_i0, v, idx);
        }
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < m_cnt) {
          float v = -__int_as_float(0x7f800000);
          int idx = -1;
          if (lane < 16 && row0 + lane < V) {
            float sum = 0.0f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww) {
              sum += sh.reduce[ww][warp + 8][lane];
            }
            v = sum;
            idx = row0 + lane;
          }
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            hy4_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            hy4_lm_argmax_update(best_v1, best_i1, v, idx);
          }
        }
      }
      __syncthreads();
    } else {
      for (int idx = tid; idx < m_cnt * 16; idx += 256) {
        const int m_loc = idx >> 4;
        const int n_loc = idx & 15;
        const int n_col = row0 + n_loc;
        if (n_col < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][m_loc][n_loc];
          }
          out[static_cast<size_t>(m_base + m_loc) * V + n_col] = sum;
        }
      }
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    if (lane == 0) {
      if (warp < m_cnt) {
        const size_t slot = static_cast<size_t>(m_base + warp) * gridDim.x + blockIdx.x;
        out[slot] = best_v0;
        partial_idxs[slot] = best_i0;
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < m_cnt) {
          const size_t slot = static_cast<size_t>(m_base + warp + 8) * gridDim.x + blockIdx.x;
          out[slot] = best_v1;
          partial_idxs[slot] = best_i1;
        }
      }
    }
  }
}

// ============================================================================
// 3E. Stage 2 Argmax Reduction: [M, num_ctas] -> (out_ids i32[M], out_vals f32[M])
//     Lowest-ID tie-break. Grid: dim3(min(M, 65535), 1, 1), Block: dim3(256, 1, 1)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4) void hy4_lm_head_argmax_stage2(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    int* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int M,
    int num_ctas) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float s_v[8];
  __shared__ int s_i[8];

  for (int m = static_cast<int>(blockIdx.x); m < M; m += static_cast<int>(gridDim.x)) {
    const size_t base = static_cast<size_t>(m) * static_cast<size_t>(num_ctas);
    float bv = -__int_as_float(0x7f800000);
    int bi = -1;
    for (int i = tid; i < num_ctas; i += 256) {
      hy4_lm_argmax_update(bv, bi, partial_vals[base + i], partial_idxs[base + i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      hy4_lm_argmax_update(bv, bi, ov, oi);
    }
    if (lane == 0) {
      s_v[warp] = bv;
      s_i[warp] = bi;
    }
    __syncthreads();

    if (warp == 0) {
      float v = (lane < 8) ? s_v[lane] : -__int_as_float(0x7f800000);
      int idx = (lane < 8) ? s_i[lane] : -1;
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
        hy4_lm_argmax_update(v, idx, ov, oi);
      }
      if (lane == 0) {
        out_ids[m] = idx;
        out_vals[m] = v;
      }
    }
    __syncthreads();
  }
}
)cuda"
