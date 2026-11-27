R"cuda(
// ============================================================================
// Hy4-flash CUDA MoE Router Kernels (GEMV for M<=16, BF16 TC GEMM for M>=17,
// and Branchless uint64-Key 512-Thread Tree-Bitonic Exact Top-K)
//   Requires ptx_helpers.cu.inc and hy4_fp4_helpers.cu.inc prepended.
//   Supports 1<=E<=512, 1<=top_k<=16 (top_k<=E), arbitrary D>=32, M in 1..65535,
//   both BF16 and FP32 router_w, optional FP32 bias, lowest-ID tie-breaking,
//   unbiased normalized weights, and ascending expert-ID output order.
// ============================================================================

__device__ __forceinline__ float4 hy4r_ld_cs_f4(const float4* ptr) {
  float4 v;
  asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];\n"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ void hy4r_f32x2_split_bf16x2(
    float a, float b, unsigned int& hi_out, unsigned int& lo_out) {
  const __nv_bfloat162 bh = __floats2bfloat162_rn(a, b);
  const float2 fh = __bfloat1622float2(bh);
  const __nv_bfloat162 bl = __floats2bfloat162_rn(
      isfinite(a) ? (a - fh.x) : 0.0f,
      isfinite(b) ? (b - fh.y) : 0.0f);
  hi_out = *reinterpret_cast<const unsigned int*>(&bh);
  lo_out = *reinterpret_cast<const unsigned int*>(&bl);
}

// ---------------------------------------------------------------------------
// 1A. High-Occupancy 16-Byte Software-Pipelined Router GEMV (M in 1..=4)
//     Grid: dim3(E, 1, 1), Block: dim3(32, 4, 1) (4 warps = 128 thr/expert).
//     Requires (W_IS_BF16 ? (D % 8 == 0) : (D % 4 == 0)).
//     At E=512, D=3072 BF16: D8=384 = 3x128 exact, 2048 warps -> ~256 GB/s.
// ---------------------------------------------------------------------------
template <bool W_IS_BF16, int M_CAP>
__global__ __launch_bounds__(128, 6) void hy4_moe_router_gemv_u4(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ router_w,
    float* __restrict__ logits,
    int M, int E, int D) {
  const int e = static_cast<int>(blockIdx.x);
  if (e >= E) return;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

  __shared__ float s_red[4][M_CAP];
  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) acc[m] = 0.0f;

  if constexpr (W_IS_BF16) {
    const int D8 = D >> 3;
    const uint4* w_row = reinterpret_cast<const uint4*>(
        reinterpret_cast<const nv_bfloat16*>(router_w) + static_cast<size_t>(e) * D);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
    if (tid < D8) {
      uint4 wc = ptx_ld_cs_u4(w_row + tid);
      for (int kv = tid; kv < D8; kv += 128) {
        uint4 wn;
        const int kn = kv + 128;
        if (kn < D8) wn = ptx_ld_cs_u4(w_row + kn);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wc);
        const float2 w0 = __bfloat1622float2(wb[0]);
        const float2 w1 = __bfloat1622float2(wb[1]);
        const float2 w2 = __bfloat1622float2(wb[2]);
        const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
          if (m < M) {
            const uint4 xv = x_u4[static_cast<size_t>(m) * D8 + kv];
            const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const float2 f0 = __bfloat1622float2(xb[0]);
            const float2 f1 = __bfloat1622float2(xb[1]);
            const float2 f2 = __bfloat1622float2(xb[2]);
            const float2 f3 = __bfloat1622float2(xb[3]);
            acc[m] = fmaf(f0.x, w0.x, fmaf(f0.y, w0.y, fmaf(f1.x, w1.x, fmaf(f1.y, w1.y, acc[m]))));
            acc[m] = fmaf(f2.x, w2.x, fmaf(f2.y, w2.y, fmaf(f3.x, w3.x, fmaf(f3.y, w3.y, acc[m]))));
          }
        }
        wc = wn;
      }
    }
  } else {
    const int D4 = D >> 2;
    const float4* w_row = reinterpret_cast<const float4*>(
        reinterpret_cast<const float*>(router_w) + static_cast<size_t>(e) * D);
    const uint2* x_u2 = reinterpret_cast<const uint2*>(x);
    if (tid < D4) {
      float4 wc = hy4r_ld_cs_f4(w_row + tid);
      for (int kv = tid; kv < D4; kv += 128) {
        float4 wn;
        const int kn = kv + 128;
        if (kn < D4) wn = hy4r_ld_cs_f4(w_row + kn);
#pragma unroll
        for (int m = 0; m < M_CAP; ++m) {
          if (m < M) {
            const uint2 xv = x_u2[static_cast<size_t>(m) * D4 + kv];
            const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const float2 f0 = __bfloat1622float2(xb[0]);
            const float2 f1 = __bfloat1622float2(xb[1]);
            acc[m] = fmaf(f0.x, wc.x, fmaf(f0.y, wc.y, fmaf(f1.x, wc.z, fmaf(f1.y, wc.w, acc[m]))));
          }
        }
        wc = wn;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      const float v = hy4_warp_sum(acc[m]);
      if (lane == 0) s_red[warp][m] = v;
    }
  }
  __syncthreads();

  if (tid < M) {
    logits[static_cast<size_t>(tid) * E + e] =
        (s_red[0][tid] + s_red[1][tid]) + (s_red[2][tid] + s_red[3][tid]);
  }
}

// ---------------------------------------------------------------------------
// 1A2. 16-Byte Pipelined Router GEMV for M in 5..=16 (4-row strips, 4 warps/expert)
//      Grid: dim3(E, 1, 1), Block: dim3(32, 4, 1).
// ---------------------------------------------------------------------------
template <bool W_IS_BF16>
__global__ __launch_bounds__(128, 6) void hy4_moe_router_gemv_u4_multi(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ router_w,
    float* __restrict__ logits,
    int M, int E, int D) {
  const int e = static_cast<int>(blockIdx.x);
  if (e >= E) return;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

  __shared__ float s_red[4][4];

  for (int m0 = 0; m0 < M; m0 += 4) {
    const int mc = (M - m0 < 4) ? (M - m0) : 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if constexpr (W_IS_BF16) {
      const int D8 = D >> 3;
      const uint4* w_row = reinterpret_cast<const uint4*>(
          reinterpret_cast<const nv_bfloat16*>(router_w) + static_cast<size_t>(e) * D);
      const uint4* x_u4 = reinterpret_cast<const uint4*>(x) + static_cast<size_t>(m0) * D8;
      if (tid < D8) {
        uint4 wc = ptx_ld_cs_u4(w_row + tid);
        for (int kv = tid; kv < D8; kv += 128) {
          uint4 wn;
          const int kn = kv + 128;
          if (kn < D8) wn = ptx_ld_cs_u4(w_row + kn);
          const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wc);
          const float2 w0 = __bfloat1622float2(wb[0]);
          const float2 w1 = __bfloat1622float2(wb[1]);
          const float2 w2 = __bfloat1622float2(wb[2]);
          const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            if (m < mc) {
              const uint4 xv = x_u4[static_cast<size_t>(m) * D8 + kv];
              const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
              const float2 f0 = __bfloat1622float2(xb[0]);
              const float2 f1 = __bfloat1622float2(xb[1]);
              const float2 f2 = __bfloat1622float2(xb[2]);
              const float2 f3 = __bfloat1622float2(xb[3]);
              acc[m] = fmaf(f0.x, w0.x, fmaf(f0.y, w0.y, fmaf(f1.x, w1.x, fmaf(f1.y, w1.y, acc[m]))));
              acc[m] = fmaf(f2.x, w2.x, fmaf(f2.y, w2.y, fmaf(f3.x, w3.x, fmaf(f3.y, w3.y, acc[m]))));
            }
          }
          wc = wn;
        }
      }
    } else {
      const int D4 = D >> 2;
      const float4* w_row = reinterpret_cast<const float4*>(
          reinterpret_cast<const float*>(router_w) + static_cast<size_t>(e) * D);
      const uint2* x_u2 = reinterpret_cast<const uint2*>(x) + static_cast<size_t>(m0) * D4;
      if (tid < D4) {
        float4 wc = hy4r_ld_cs_f4(w_row + tid);
        for (int kv = tid; kv < D4; kv += 128) {
          float4 wn;
          const int kn = kv + 128;
          if (kn < D4) wn = hy4r_ld_cs_f4(w_row + kn);
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            if (m < mc) {
              const uint2 xv = x_u2[static_cast<size_t>(m) * D4 + kv];
              const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
              const float2 f0 = __bfloat1622float2(xb[0]);
              const float2 f1 = __bfloat1622float2(xb[1]);
              acc[m] = fmaf(f0.x, wc.x, fmaf(f0.y, wc.y, fmaf(f1.x, wc.z, fmaf(f1.y, wc.w, acc[m]))));
            }
          }
          wc = wn;
        }
      }
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
      if (m < mc) {
        const float v = hy4_warp_sum(acc[m]);
        if (lane == 0) s_red[warp][m] = v;
      }
    }
    __syncthreads();
    if (tid < mc) {
      logits[static_cast<size_t>(m0 + tid) * E + e] =
          (s_red[0][tid] + s_red[1][tid]) + (s_red[2][tid] + s_red[3][tid]);
    }
    if (m0 + 4 < M) __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// 1A3. Unaligned-D Fallback Router GEMV for M in 1..=16 (arbitrary D >= 32)
//      Grid: dim3(E, 1, 1), Block: dim3(256, 1, 1).
// ---------------------------------------------------------------------------
template <bool W_IS_BF16>
__global__ __launch_bounds__(256, 4) void hy4_moe_router_gemv_unaligned(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ router_w,
    float* __restrict__ logits,
    int M, int E, int D) {
  const int e = static_cast<int>(blockIdx.x);
  if (e >= E) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float s_red[8][16];

  auto load_w1 = [&](int col) -> float {
    if constexpr (W_IS_BF16) {
      return __bfloat162float(
          reinterpret_cast<const nv_bfloat16*>(router_w)[static_cast<size_t>(e) * D + col]);
    } else {
      return reinterpret_cast<const float*>(router_w)[static_cast<size_t>(e) * D + col];
    }
  };

  for (int m = 0; m < M; ++m) {
    float v = 0.0f;
    const nv_bfloat16* xm = x + static_cast<size_t>(m) * D;
    for (int i = tid; i < D; i += 256) {
      v = fmaf(__bfloat162float(xm[i]), load_w1(i), v);
    }
    v = hy4_warp_sum(v);
    if (lane == 0) s_red[warp][m] = v;
  }
  __syncthreads();
  if (tid < M) {
    float tot = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) tot += s_red[ww][tid];
    logits[static_cast<size_t>(tid) * E + e] = tot;
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 1B. Router BF16 Tensor Core GEMM for M >= 17: BM=64, BN=64, 4 warps (128 thr)
//     - W_IS_BF16=true : BK=64, 2-stage cp.async.cg, XOR-swizzled ldmatrix.x4 + m16n8k16
//     - W_IS_BF16=false: BK=32, 2-stage cp.async.cg (x) + 2-pass FP32->(bf16_hi+bf16_lo)
//                        splitting for ~1e-6 FP32 accuracy & exact tie preservation
//     Grid: dim3((E + 63) / 64, (M + 63) / 64, 1), Block: dim3(32, 4, 1).
// ---------------------------------------------------------------------------
template <bool W_IS_BF16, bool D_ALIGN8>
__global__ __launch_bounds__(128, W_IS_BF16 ? 3 : 4) void hy4_moe_router_gemm_prefill(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ router_w,
    float* __restrict__ logits,
    int M, int E, int D) {
  constexpr int BK = W_IS_BF16 ? 64 : 32;
  constexpr int BK_U4 = BK / 8;
  constexpr int C_SHIFT = W_IS_BF16 ? 3 : 2;
  constexpr int SWIZ_SHIFT = W_IS_BF16 ? 0 : 1;
  constexpr int SWIZ_MASK = BK_U4 - 1;
  constexpr int LOAD_ITERS = (64 * BK_U4) / 128;
  constexpr int K_SUBS = BK / 16;

  const int n0 = static_cast<int>(blockIdx.x) * 64;
  const int m0 = static_cast<int>(blockIdx.y) * 64;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;

  __shared__ alignas(16) uint4 s_a[2][64][BK_U4];
  __shared__ alignas(16) uint4 s_b0[2][64][BK_U4];
  __shared__ alignas(16) uint4 s_b1[W_IS_BF16 ? 1 : 2][W_IS_BF16 ? 1 : 64][W_IS_BF16 ? 1 : BK_U4];
  if constexpr (W_IS_BF16) {
    (void)s_b1;
  }

  auto swiz = [](int r, int c) -> int {
    return c ^ ((r >> SWIZ_SHIFT) & SWIZ_MASK);
  };

  const int n_steps = (D + BK - 1) / BK;

  auto load_stage = [&](int st, int step) {
#pragma unroll
    for (int iter = 0; iter < LOAD_ITERS; ++iter) {
      const int idx = tid + iter * 128;
      const int r = idx >> C_SHIFT;
      const int c = idx & (BK_U4 - 1);
      const int row = m0 + r;
      uint4* dst_a = &s_a[st][r][swiz(r, c)];
      if constexpr (D_ALIGN8) {
        const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
        const int d_u4 = D >> 3;
        const int col_u4 = step * BK_U4 + c;
        const bool valid = (row < M) && (col_u4 < d_u4);
        const uint4* src = valid
            ? (x_u4 + static_cast<size_t>(row) * d_u4 + col_u4)
            : x_u4;
        ptx_cp_async16_cg_zfill(dst_a, src, valid);
      } else {
        nv_bfloat16 tmp[8];
        const int kc = step * BK + c * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          tmp[j] = (row < M && kc + j < D)
              ? x[static_cast<size_t>(row) * D + kc + j]
              : __float2bfloat16_rn(0.0f);
        }
        *dst_a = *reinterpret_cast<const uint4*>(tmp);
      }
    }

    if constexpr (W_IS_BF16) {
#pragma unroll
      for (int iter = 0; iter < LOAD_ITERS; ++iter) {
        const int idx = tid + iter * 128;
        const int r = idx >> C_SHIFT;
        const int c = idx & (BK_U4 - 1);
        const int e_idx = n0 + r;
        uint4* dst_b = &s_b0[st][r][swiz(r, c)];
        if constexpr (D_ALIGN8) {
          const uint4* w_u4 = reinterpret_cast<const uint4*>(router_w);
          const int d_u4 = D >> 3;
          const int col_u4 = step * BK_U4 + c;
          const bool valid = (e_idx < E) && (col_u4 < d_u4);
          const uint4* src = valid
              ? (w_u4 + static_cast<size_t>(e_idx) * d_u4 + col_u4)
              : w_u4;
          ptx_cp_async16_cg_zfill(dst_b, src, valid);
        } else {
          const nv_bfloat16* w_bf16 = reinterpret_cast<const nv_bfloat16*>(router_w);
          nv_bfloat16 tmp[8];
          const int kc = step * BK + c * 8;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            tmp[j] = (e_idx < E && kc + j < D)
                ? w_bf16[static_cast<size_t>(e_idx) * D + kc + j]
                : __float2bfloat16_rn(0.0f);
          }
          *dst_b = *reinterpret_cast<const uint4*>(tmp);
        }
      }
      if constexpr (D_ALIGN8) ptx_cp_commit();
    } else {
      if constexpr (D_ALIGN8) ptx_cp_commit();
#pragma unroll
      for (int iter = 0; iter < LOAD_ITERS; ++iter) {
        const int idx = tid + iter * 128;
        const int r = idx >> C_SHIFT;
        const int c = idx & (BK_U4 - 1);
        const int e_idx = n0 + r;
        const int kc = step * BK + c * 8;
        uint4 hi_v = make_uint4(0u, 0u, 0u, 0u);
        uint4 lo_v = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (D_ALIGN8) {
          if (e_idx < E && kc < D) {
            const float4* w_f4 = reinterpret_cast<const float4*>(router_w);
            const int d_f4 = D >> 2;
            const float4* wp = w_f4 + static_cast<size_t>(e_idx) * d_f4 + (kc >> 2);
            const float4 f0 = wp[0];
            const float4 f1 = wp[1];
            hy4r_f32x2_split_bf16x2(f0.x, f0.y, hi_v.x, lo_v.x);
            hy4r_f32x2_split_bf16x2(f0.z, f0.w, hi_v.y, lo_v.y);
            hy4r_f32x2_split_bf16x2(f1.x, f1.y, hi_v.z, lo_v.z);
            hy4r_f32x2_split_bf16x2(f1.z, f1.w, hi_v.w, lo_v.w);
          }
        } else {
          if (e_idx < E) {
            const float* w_f32 = reinterpret_cast<const float*>(router_w);
            float fv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              fv[j] = (kc + j < D) ? w_f32[static_cast<size_t>(e_idx) * D + kc + j] : 0.0f;
            }
            hy4r_f32x2_split_bf16x2(fv[0], fv[1], hi_v.x, lo_v.x);
            hy4r_f32x2_split_bf16x2(fv[2], fv[3], hi_v.y, lo_v.y);
            hy4r_f32x2_split_bf16x2(fv[4], fv[5], hi_v.z, lo_v.z);
            hy4r_f32x2_split_bf16x2(fv[6], fv[7], hi_v.w, lo_v.w);
          }
        }
        const int cs = swiz(r, c);
        s_b0[st][r][cs] = hi_v;
        s_b1[st][r][cs] = lo_v;
      }
    }
  };

  float acc[2][4][4];
#pragma unroll
  for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
    for (int wn = 0; wn < 4; ++wn) {
      acc[wm][wn][0] = 0.0f;
      acc[wm][wn][1] = 0.0f;
      acc[wm][wn][2] = 0.0f;
      acc[wm][wn][3] = 0.0f;
    }
  }

  load_stage(0, 0);
  if constexpr (D_ALIGN8) ptx_cp_wait<0>();
  __syncthreads();

  const int ra0 = warp_m * 32 + (lane & 15);
  const int ra1 = ra0 + 16;
  const int ca_off = lane >> 4;
  const int rb_base = warp_n * 32 + ((lane >> 4) << 3) + (lane & 7);
  const int cb_off = (lane >> 3) & 1;

#pragma unroll 1
  for (int step = 0; step < n_steps; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    const bool has_nxt = (step + 1 < n_steps);
    if (has_nxt) {
      load_stage(nxt, step + 1);
    }

#pragma unroll
    for (int ks = 0; ks < K_SUBS; ++ks) {
      const int ku = ks * 2;
      unsigned int a0[4], a1[4];
      ptx_ldmatrix_x4(a0[0], a0[1], a0[2], a0[3], &s_a[cur][ra0][swiz(ra0, ku + ca_off)]);
      ptx_ldmatrix_x4(a1[0], a1[1], a1[2], a1[3], &s_a[cur][ra1][swiz(ra1, ku + ca_off)]);
#pragma unroll
      for (int wn2 = 0; wn2 < 2; ++wn2) {
        const int rb = rb_base + wn2 * 16;
        const int cb = swiz(rb, ku + cb_off);
        unsigned int b0, b1, b2, b3;
        ptx_ldmatrix_x4(b0, b1, b2, b3, &s_b0[cur][rb][cb]);
        ptx_mma_m16n8k16_bf16(
            acc[0][wn2 * 2][0], acc[0][wn2 * 2][1], acc[0][wn2 * 2][2], acc[0][wn2 * 2][3],
            a0[0], a0[1], a0[2], a0[3], b0, b1);
        ptx_mma_m16n8k16_bf16(
            acc[0][wn2 * 2 + 1][0], acc[0][wn2 * 2 + 1][1], acc[0][wn2 * 2 + 1][2], acc[0][wn2 * 2 + 1][3],
            a0[0], a0[1], a0[2], a0[3], b2, b3);
        ptx_mma_m16n8k16_bf16(
            acc[1][wn2 * 2][0], acc[1][wn2 * 2][1], acc[1][wn2 * 2][2], acc[1][wn2 * 2][3],
            a1[0], a1[1], a1[2], a1[3], b0, b1);
        ptx_mma_m16n8k16_bf16(
            acc[1][wn2 * 2 + 1][0], acc[1][wn2 * 2 + 1][1], acc[1][wn2 * 2 + 1][2], acc[1][wn2 * 2 + 1][3],
            a1[0], a1[1], a1[2], a1[3], b2, b3);
        if constexpr (!W_IS_BF16) {
          ptx_ldmatrix_x4(b0, b1, b2, b3, &s_b1[cur][rb][cb]);
          ptx_mma_m16n8k16_bf16(
              acc[0][wn2 * 2][0], acc[0][wn2 * 2][1], acc[0][wn2 * 2][2], acc[0][wn2 * 2][3],
              a0[0], a0[1], a0[2], a0[3], b0, b1);
          ptx_mma_m16n8k16_bf16(
              acc[0][wn2 * 2 + 1][0], acc[0][wn2 * 2 + 1][1], acc[0][wn2 * 2 + 1][2], acc[0][wn2 * 2 + 1][3],
              a0[0], a0[1], a0[2], a0[3], b2, b3);
          ptx_mma_m16n8k16_bf16(
              acc[1][wn2 * 2][0], acc[1][wn2 * 2][1], acc[1][wn2 * 2][2], acc[1][wn2 * 2][3],
              a1[0], a1[1], a1[2], a1[3], b0, b1);
          ptx_mma_m16n8k16_bf16(
              acc[1][wn2 * 2 + 1][0], acc[1][wn2 * 2 + 1][1], acc[1][wn2 * 2 + 1][2], acc[1][wn2 * 2 + 1][3],
              a1[0], a1[1], a1[2], a1[3], b2, b3);
        }
      }
    }

    if (has_nxt) {
      if constexpr (D_ALIGN8) ptx_cp_wait<0>();
      __syncthreads();
    }
  }

  const int g = lane >> 2;
  const int t = lane & 3;
#pragma unroll
  for (int wm = 0; wm < 2; ++wm) {
    const int r0 = m0 + warp_m * 32 + wm * 16 + g;
    const int r1 = r0 + 8;
#pragma unroll
    for (int wn = 0; wn < 4; ++wn) {
      const int c0 = n0 + warp_n * 32 + wn * 8 + 2 * t;
      const int c1 = c0 + 1;
      if (r0 < M) {
        if (c0 < E) logits[static_cast<size_t>(r0) * E + c0] = acc[wm][wn][0];
        if (c1 < E) logits[static_cast<size_t>(r0) * E + c1] = acc[wm][wn][1];
      }
      if (r1 < M) {
        if (c0 < E) logits[static_cast<size_t>(r1) * E + c0] = acc[wm][wn][2];
        if (c1 < E) logits[static_cast<size_t>(r1) * E + c1] = acc[wm][wn][3];
      }
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 1C. Branchless uint64-Key 512-Thread Tree-Bitonic Exact Top-K
//     Order-preserving key: (ordered_u32(sel) << 32) | (~expert_id), sentinel=0.
//     Guarantees exact lowest-ID tie-breaking, NaN -> -inf, unbiased normalized
//     sigmoid weights, and ascending expert-ID output for 1<=E<=512, 1<=top_k<=16.
//     Grid: dim3(M, 1, 1), Block: dim3(512, 1, 1) (16 warps, 1 expert/thread).
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long hy4r_pack_key(float val, unsigned int id) {
  if (val == 0.0f) val = 0.0f;
  const unsigned int u = __float_as_uint(val);
  const unsigned int ord = u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
  return (static_cast<unsigned long long>(ord) << 32) | static_cast<unsigned long long>(~id);
}

__device__ __forceinline__ unsigned long long hy4r_min_u64(
    unsigned long long a, unsigned long long b) {
  return (a < b) ? a : b;
}

__device__ __forceinline__ unsigned long long hy4r_max_u64(
    unsigned long long a, unsigned long long b) {
  return (a > b) ? a : b;
}

__device__ __forceinline__ void hy4r_bitonic_sort32_u64(
    unsigned long long& key, unsigned int lane) {
#pragma unroll
  for (unsigned int size = 2u; size <= 32u; size <<= 1u) {
    const bool dir_asc = (size < 32u) && ((lane & size) != 0u);
#pragma unroll
    for (unsigned int stride = size >> 1u; stride >= 1u; stride >>= 1u) {
      const unsigned long long ok = __shfl_xor_sync(0xFFFFFFFFu, key, stride);
      const bool want_min = (((lane & stride) != 0u) != dir_asc);
      key = want_min ? hy4r_min_u64(key, ok) : hy4r_max_u64(key, ok);
    }
  }
}

__device__ __forceinline__ void hy4r_bitonic_merge16_u64(
    unsigned long long& key, unsigned int h_lane) {
#pragma unroll
  for (unsigned int s = 8u; s >= 1u; s >>= 1u) {
    const unsigned long long ok = __shfl_xor_sync(0xFFFFFFFFu, key, s);
    key = ((h_lane & s) != 0u) ? hy4r_min_u64(key, ok) : hy4r_max_u64(key, ok);
  }
}

extern "C" __global__ __launch_bounds__(512, 2) void hy4_moe_router_topk_rows(
    const float* __restrict__ logits,
    const float* __restrict__ exp_probs_b,
    unsigned int* __restrict__ out_ids,
    float* __restrict__ out_weights,
    int M, int E, int top_k, float route_scale, int norm_topk) {
  const int m = static_cast<int>(blockIdx.x);
  if (m >= M) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float neg_inf = -__int_as_float(0x7F800000);

  __shared__ unsigned long long s_k[256];
  __shared__ unsigned long long s_k2[128];

  const float* lrow = logits + static_cast<size_t>(m) * E;
  unsigned long long key = 0ULL;
  if (tid < E) {
    const float lx = lrow[tid];
    const float s = 1.0f / (1.0f + expf(-lx));
    float sel = exp_probs_b ? (s + exp_probs_b[tid]) : s;
    if (isnan(sel)) sel = neg_inf;
    key = hy4r_pack_key(sel, static_cast<unsigned int>(tid));
  }

  // 16 warps x 32 elements -> 16 sorted descending top-16 lists (256 candidates).
  hy4r_bitonic_sort32_u64(key, static_cast<unsigned int>(lane));
  if (lane < 16) {
    s_k[warp * 16 + lane] = key;
  }
  __syncthreads();

  // Tree merge 16 -> 8 sorted top-16 lists (8 half-warps) into ping-pong s_k2.
  if (tid < 128) {
    const int hw = tid >> 4;
    const unsigned int hl = static_cast<unsigned int>(tid & 15);
    unsigned long long mk = hy4r_max_u64(
        s_k[(hw * 2) * 16 + hl], s_k[(hw * 2 + 1) * 16 + (15u - hl)]);
    hy4r_bitonic_merge16_u64(mk, hl);
    s_k2[hw * 16 + hl] = mk;
  }
  __syncthreads();

  // Tree merge 8 -> 4 sorted top-16 lists (4 half-warps) back into s_k.
  if (tid < 64) {
    const int hw = tid >> 4;
    const unsigned int hl = static_cast<unsigned int>(tid & 15);
    unsigned long long mk = hy4r_max_u64(
        s_k2[(hw * 2) * 16 + hl], s_k2[(hw * 2 + 1) * 16 + (15u - hl)]);
    hy4r_bitonic_merge16_u64(mk, hl);
    s_k[hw * 16 + hl] = mk;
  }
  __syncthreads();

  // Final merge 4 -> 2 -> 1 in warp 0 + unbiased sigmoid norm + ascending-ID sort.
  if (warp == 0) {
    const int hw = lane >> 4;
    const unsigned int hl = static_cast<unsigned int>(lane & 15);
    const unsigned int rev = 15u - hl;
    unsigned long long mk = hy4r_max_u64(
        s_k[(hw * 2) * 16 + hl], s_k[(hw * 2 + 1) * 16 + rev]);
    hy4r_bitonic_merge16_u64(mk, hl);

    const unsigned long long bk =
        __shfl_sync(0xFFFFFFFFu, mk, static_cast<int>(16u + rev));
    unsigned long long fk = hy4r_max_u64(mk, bk);
    hy4r_bitonic_merge16_u64(fk, hl);

    unsigned int fi =
        (lane < top_k && fk != 0ULL) ? (~static_cast<unsigned int>(fk)) : 0xFFFFFFFFu;
    float my_s = (fi < static_cast<unsigned int>(E))
        ? (1.0f / (1.0f + expf(-lrow[fi])))
        : 0.0f;

    float sum_s = my_s;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
      sum_s += __shfl_xor_sync(0xFFFFFFFFu, sum_s, off);
    }

    if (top_k > 1) {
      const unsigned int ulane = static_cast<unsigned int>(lane);
#pragma unroll
      for (unsigned int size = 2u; size <= 16u; size <<= 1u) {
        const bool dir_desc = (ulane & size) != 0u;
#pragma unroll
        for (unsigned int stride = size >> 1u; stride >= 1u; stride >>= 1u) {
          const unsigned int oi = __shfl_xor_sync(0xFFFFFFFFu, fi, stride);
          const float os = __shfl_xor_sync(0xFFFFFFFFu, my_s, stride);
          const bool upper = (ulane & stride) != 0u;
          const bool swap = (upper != dir_desc) ? (oi > fi) : (oi < fi);
          if (swap) {
            fi = oi;
            my_s = os;
          }
        }
      }
    }

    if (lane < top_k) {
      const size_t out_off = static_cast<size_t>(m) * top_k + lane;
      out_ids[out_off] = fi;
      out_weights[out_off] = (norm_topk && top_k > 1)
          ? ((my_s / (sum_s + 1e-20f)) * route_scale)
          : (my_s * route_scale);
    }
  }
}
)cuda"
