R"cuda(
// ============================================================================
// Hy4-flash CUDA MoE Decode/Verify (M in 1..=16) Routed Expert Kernels
//   Supports real Hy4-flash (E=512, top_k=16, lat=1536, ff=1536) and arbitrary
//   smaller test shapes (1<=E<=512, 1<=top_k<=16, lat>=64 && lat%64==0,
//   ff>=64 && ff%64==0).
//   Modes: 0=NVFP4 (b4/g16, FP8 E4M3 scales), 1=SymQ8 (b8/g32, s8=byte^0x80, F16/BF16 scales).
//   Architecture:
//     - M == 1: Direct zero-SMEM streaming kernels reading topk_ids directly
//       (zero route_plan launch, zero route buffers, exact even with duplicate IDs).
//     - M in 2..=16: Split cold (cnt <= 2, zero-SMEM streaming) + hot (cnt >= 3,
//       2-stage cp.async.cg + BF16 TC MMA without dead GEMV SMEM/registers).
//   Dual-target PTX: Windows CUDA 13.4+ HW FP4/FP8 cvt, Linux DGX Spark CUDA 13.0
//   SMEM E2M1 LUT + PTX 8.1 FP8 cvt, sm_80 bitwise SW fallback.
// ============================================================================

template <bool B>
struct hy4f_bool_tag {
  static constexpr bool value = B;
};

__device__ __forceinline__ float hy4f_halfwarp_sum(float v) {
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  }
  return v;
}

__device__ __forceinline__ float hy4f_read_f16_or_bf16(unsigned short raw, int is_bf16) {
  return is_bf16 ? __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(&raw))
                 : __half2float(__ushort_as_half(raw));
}

__device__ __forceinline__ void hy4f_unpack8_symq8_bf16x2(
    const uint2& w, unsigned int out[4]) {
  const unsigned int u0 = w.x ^ 0x80808080u;
  const unsigned int u1 = w.y ^ 0x80808080u;
  out[0] = hy4_pack_bf16x2(float(static_cast<signed char>(u0)),
                           float(static_cast<signed char>(u0 >> 8)));
  out[1] = hy4_pack_bf16x2(float(static_cast<signed char>(u0 >> 16)),
                           float(static_cast<signed char>(u0 >> 24)));
  out[2] = hy4_pack_bf16x2(float(static_cast<signed char>(u1)),
                           float(static_cast<signed char>(u1 >> 8)));
  out[3] = hy4_pack_bf16x2(float(static_cast<signed char>(u1 >> 16)),
                           float(static_cast<signed char>(u1 >> 24)));
}

__device__ __forceinline__ float2 hy4f_bf16x2_to_f2(unsigned int bf2) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
}

template <bool HAS_T1>
__device__ __forceinline__ void hy4f_dot_nvfp4_8_u4(
    unsigned int w32, const unsigned int* __restrict__ lut,
    const uint4& x0v, const uint4& x1v,
    float& d0, float& d1) {
  const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0v);
  [[maybe_unused]] const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1v);
  {
    const float2 w = hy4f_bf16x2_to_f2(hy4_lut_e2m1x2_to_bf16x2(w32, lut));
    const float2 a = __bfloat1622float2(p0[0]);
    d0 = fmaf(w.x, a.x, fmaf(w.y, a.y, d0));
    if constexpr (HAS_T1) {
      const float2 b = __bfloat1622float2(p1[0]);
      d1 = fmaf(w.x, b.x, fmaf(w.y, b.y, d1));
    }
  }
  {
    const float2 w = hy4f_bf16x2_to_f2(hy4_lut_e2m1x2_to_bf16x2(w32 >> 8, lut));
    const float2 a = __bfloat1622float2(p0[1]);
    d0 = fmaf(w.x, a.x, fmaf(w.y, a.y, d0));
    if constexpr (HAS_T1) {
      const float2 b = __bfloat1622float2(p1[1]);
      d1 = fmaf(w.x, b.x, fmaf(w.y, b.y, d1));
    }
  }
  {
    const float2 w = hy4f_bf16x2_to_f2(hy4_lut_e2m1x2_to_bf16x2(w32 >> 16, lut));
    const float2 a = __bfloat1622float2(p0[2]);
    d0 = fmaf(w.x, a.x, fmaf(w.y, a.y, d0));
    if constexpr (HAS_T1) {
      const float2 b = __bfloat1622float2(p1[2]);
      d1 = fmaf(w.x, b.x, fmaf(w.y, b.y, d1));
    }
  }
  {
    const float2 w = hy4f_bf16x2_to_f2(hy4_lut_e2m1x2_to_bf16x2(w32 >> 24, lut));
    const float2 a = __bfloat1622float2(p0[3]);
    d0 = fmaf(w.x, a.x, fmaf(w.y, a.y, d0));
    if constexpr (HAS_T1) {
      const float2 b = __bfloat1622float2(p1[3]);
      d1 = fmaf(w.x, b.x, fmaf(w.y, b.y, d1));
    }
  }
}

template <bool HAS_T1>
__device__ __forceinline__ void hy4f_dot_symq8_8_u4(
    unsigned int w0, unsigned int w1,
    const uint4& x0v, const uint4& x1v,
    float& d0, float& d1) {
  const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0v);
  [[maybe_unused]] const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1v);
  const unsigned int words[2] = {w0, w1};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const unsigned int u = words[i] ^ 0x80808080u;
    const float f0 = float(static_cast<signed char>(u));
    const float f1 = float(static_cast<signed char>(u >> 8));
    const float f2 = float(static_cast<signed char>(u >> 16));
    const float f3 = float(static_cast<signed char>(u >> 24));
    const float2 a0 = __bfloat1622float2(p0[2 * i]);
    const float2 a1 = __bfloat1622float2(p0[2 * i + 1]);
    d0 = fmaf(f0, a0.x, fmaf(f1, a0.y, fmaf(f2, a1.x, fmaf(f3, a1.y, d0))));
    if constexpr (HAS_T1) {
      const float2 b0 = __bfloat1622float2(p1[2 * i]);
      const float2 b1 = __bfloat1622float2(p1[2 * i + 1]);
      d1 = fmaf(f0, b0.x, fmaf(f1, b0.y, fmaf(f2, b1.x, fmaf(f3, b1.y, d1))));
    }
  }
}

// ---------------------------------------------------------------------------
// 1. Split Cold (cnt <= 2) / Hot (cnt >= 3) Decode Route Plan (M in 2..=16)
//    1-CTA deterministic counting sort over P = M*top_k <= 256, E <= 512.
//    cold_meta[ci] = (first << 1) | (cnt - 1); hot chunks <= chunk_max (>= 3).
//    counts[0] = num_cold (<= min(E, P)), counts[1] = num_hot (<= P / 3).
//    Grid: dim3(1, 1, 1), Block: dim3(512, 1, 1).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(512) void hy4_moe_few_route_plan_split(
    const unsigned int* __restrict__ topk_ids,
    int* __restrict__ counts,
    int* __restrict__ cold_experts,
    int* __restrict__ cold_meta,
    int* __restrict__ hot_experts,
    int* __restrict__ hot_begin,
    int* __restrict__ hot_end,
    int* __restrict__ sorted_pairs,
    int* __restrict__ pair_to_sorted,
    int P, int E, int chunk_max) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ unsigned int s_ids[256];
  __shared__ int s_cnt[512];
  __shared__ int s_start[512];
  __shared__ int s_wc[16];
  __shared__ int s_wcold[16];
  __shared__ int s_whot[16];

  s_cnt[tid] = 0;
  if (tid < P) s_ids[tid] = topk_ids[tid];
  __syncthreads();

  if (tid < P) {
    const unsigned int e = s_ids[tid];
    if (e < static_cast<unsigned int>(E)) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();

  const int c = (tid < E) ? s_cnt[tid] : 0;
  const int u_cold = (c == 1 || c == 2) ? 1 : 0;
  const int u_hot = (c >= 3) ? ((c + chunk_max - 1) / chunk_max) : 0;

  int sc = c;
  int scold = u_cold;
  int shot = u_hot;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int nc = __shfl_up_sync(0xFFFFFFFFu, sc, off);
    const int ncold = __shfl_up_sync(0xFFFFFFFFu, scold, off);
    const int nhot = __shfl_up_sync(0xFFFFFFFFu, shot, off);
    if (lane >= off) {
      sc += nc;
      scold += ncold;
      shot += nhot;
    }
  }
  if (lane == 31) {
    s_wc[warp] = sc;
    s_wcold[warp] = scold;
    s_whot[warp] = shot;
  }
  __syncthreads();

  if (warp == 0) {
    int wc = (lane < 16) ? s_wc[lane] : 0;
    int wcold = (lane < 16) ? s_wcold[lane] : 0;
    int whot = (lane < 16) ? s_whot[lane] : 0;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
      const int nc = __shfl_up_sync(0xFFFFFFFFu, wc, off);
      const int ncold = __shfl_up_sync(0xFFFFFFFFu, wcold, off);
      const int nhot = __shfl_up_sync(0xFFFFFFFFu, whot, off);
      if (lane >= off) {
        wc += nc;
        wcold += ncold;
        whot += nhot;
      }
    }
    if (lane == 15) {
      counts[0] = wcold;
      counts[1] = whot;
    }
    if (lane < 16) {
      s_wc[lane] = wc - s_wc[lane];
      s_wcold[lane] = wcold - s_wcold[lane];
      s_whot[lane] = whot - s_whot[lane];
    }
  }
  __syncthreads();

  if (tid < E) {
    const int row_start = s_wc[warp] + (sc - c);
    s_start[tid] = row_start;
    if (u_cold) {
      const int ci = s_wcold[warp] + (scold - 1);
      cold_experts[ci] = tid;
      cold_meta[ci] = (row_start << 1) | (c - 1);
    } else if (u_hot > 0) {
      const int hi0 = s_whot[warp] + (shot - u_hot);
      const int row_end = row_start + c;
      for (int k = 0; k < u_hot; ++k) {
        const int b = row_start + k * chunk_max;
        const int ed = (b + chunk_max < row_end) ? (b + chunk_max) : row_end;
        hot_experts[hi0 + k] = tid;
        hot_begin[hi0 + k] = b;
        hot_end[hi0 + k] = ed;
      }
    }
  }
  __syncthreads();

  if (tid < P) {
    const unsigned int e = s_ids[tid];
    if (e < static_cast<unsigned int>(E)) {
      int rank = 0;
      if (s_cnt[e] > 1) {
        for (int j = 0; j < tid; ++j) rank += (s_ids[j] == e) ? 1 : 0;
      }
      const int dst = s_start[e] + rank;
      sorted_pairs[dst] = tid;
      pair_to_sorted[tid] = dst;
    } else {
      pair_to_sorted[tid] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Zero-SMEM Warp-Shuffle Ascending-k Weighted Combine (M in 1..=16)
//    When pair_to_sorted == nullptr (M == 1 direct path), uses slot = p directly.
//    Grid: dim3((lat/8 + 31) / 32, M, 1), Block: dim3(32, 1, 1).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(32, 16) void hy4_moe_combine_few(
    const nv_bfloat16* __restrict__ down_out,
    const float* __restrict__ topk_weights,
    const int* __restrict__ pair_to_sorted,
    nv_bfloat16* __restrict__ out,
    int M, int lat, int top_k) {
  const int tok = static_cast<int>(blockIdx.y);
  if (tok >= M) return;
  const int lane = static_cast<int>(threadIdx.x);
  const int lat8 = lat >> 3;
  const int col8 = static_cast<int>(blockIdx.x) * 32 + lane;
  const int p = tok * top_k + lane;
  const int my_slot = (lane < top_k) ? (pair_to_sorted ? pair_to_sorted[p] : p) : 0;
  const float my_w = (lane < top_k) ? topk_weights[p] : 0.0f;

  const uint4* dn_u4 = reinterpret_cast<const uint4*>(down_out);
  const bool valid = (col8 < lat8);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int k = 0; k < top_k; ++k) {
    const int slot = __shfl_sync(0xFFFFFFFFu, my_slot, k);
    const float w = __shfl_sync(0xFFFFFFFFu, my_w, k);
    if (valid) {
      const uint4 raw = dn_u4[static_cast<size_t>(slot) * lat8 + col8];
      const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(bp[j]);
        acc[2 * j] = fmaf(f.x, w, acc[2 * j]);
        acc[2 * j + 1] = fmaf(f.y, w, acc[2 * j + 1]);
      }
    }
  }
  if (valid) {
    uint4 ov;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      op[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    }
    reinterpret_cast<uint4*>(out)[static_cast<size_t>(tok) * lat8 + col8] = ov;
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2A. Zero-SMEM Cold Gate + Up + Asymmetric Clamped SwiGLU
//     DIRECT_M1=true : M==1 reading topk_ids[blockIdx.y] directly (cnt==1).
//     DIRECT_M1=false: M in 2..=16 reading cold_experts / cold_meta (cnt in {1,2}).
//     4 warps (128 thr) / CTA, 1 warp/row -> 4 output rows/CTA, 2-stage
//     software-pipelined ld.global.cs.v4.u32, 0 B SMEM when HW FP4 CVT=1.
//     Grid: dim3(ff / 4, max_items, 1), Block: dim3(32, 4, 1).
// ---------------------------------------------------------------------------
template <int MODE, bool DIRECT_M1>
__global__ __launch_bounds__(128, 6) void hy4_moe_gate_up_cold(
    const unsigned int* __restrict__ gate_w,
    const void* __restrict__ gate_s,
    const unsigned int* __restrict__ up_w,
    const void* __restrict__ up_s,
    const nv_bfloat16* __restrict__ z,
    const int* __restrict__ experts_or_ids,
    const int* __restrict__ cold_meta,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ counts,
    nv_bfloat16* __restrict__ mid,
    int lat, int ff, int top_k, int E, float swiglu_limit, int sb_is_bf16,
    int z_row_u4) {
  const int u_idx = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int col = static_cast<int>(blockIdx.x) * 4 + warp;

  int expert, first, cnt;
  if constexpr (DIRECT_M1) {
    expert = experts_or_ids[u_idx];
    first = u_idx;
    cnt = 1;
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(E)) {
      if (lane == 0 && col < ff) {
        mid[static_cast<size_t>(first) * ff + col] = __float2bfloat16_rn(0.0f);
      }
      return;
    }
  } else {
    if (u_idx >= counts[0]) return;
    expert = experts_or_ids[u_idx];
    const int meta = cold_meta[u_idx];
    first = meta >> 1;
    cnt = (meta & 1) + 1;
  }

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    const int tid = warp * 32 + lane;
    s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
    s_e2m1_lut[tid + 128] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid + 128));
    __syncthreads();
  } else {
    (void)s_e2m1_lut;
  }
  const unsigned int* lut = s_e2m1_lut;
#else
  const unsigned int* lut = nullptr;
#endif

  if (col >= ff) return;

  const int tok0 = DIRECT_M1 ? 0 : (sorted_pairs[first] / top_k);
  const bool has_t1 = (!DIRECT_M1) && (cnt >= 2);
  const int tok1 = has_t1 ? (sorted_pairs[first + 1] / top_k) : 0;

  const uint4* z0 = reinterpret_cast<const uint4*>(z) + static_cast<size_t>(tok0) * z_row_u4;
  const uint4* z1 = reinterpret_cast<const uint4*>(z) + static_cast<size_t>(tok1) * z_row_u4;
  const size_t r_idx = static_cast<size_t>(expert) * ff + col;
  const int n_grp32 = lat >> 5;

  float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;

  auto run_body = [&](auto ht1_tag) {
    constexpr bool HT1 = decltype(ht1_tag)::value;
    const uint4 dummy = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (MODE == 0) {
      const uint4* gw = reinterpret_cast<const uint4*>(gate_w) + r_idx * n_grp32;
      const uint4* uw = reinterpret_cast<const uint4*>(up_w) + r_idx * n_grp32;
      const unsigned short* gs = reinterpret_cast<const unsigned short*>(gate_s) + r_idx * n_grp32;
      const unsigned short* us = reinterpret_cast<const unsigned short*>(up_s) + r_idx * n_grp32;
      if (lane < n_grp32) {
        uint4 gw_c = ptx_ld_cs_u4(gw + lane);
        uint4 uw_c = ptx_ld_cs_u4(uw + lane);
        unsigned short gs_c = gs[lane];
        unsigned short us_c = us[lane];
        for (int u = lane; u < n_grp32; u += 32) {
          uint4 gw_n, uw_n;
          unsigned short gs_n = 0, us_n = 0;
          const int un = u + 32;
          if (un < n_grp32) {
            gw_n = ptx_ld_cs_u4(gw + un);
            uw_n = ptx_ld_cs_u4(uw + un);
            gs_n = gs[un];
            us_n = us[un];
          }
          const float2 gsf = hy4_fast_e4m3x2_to_f2(gs_c);
          const float2 usf = hy4_fast_e4m3x2_to_f2(us_c);

          float gd0 = 0.0f, ud0 = 0.0f, gd1 = 0.0f, ud1 = 0.0f;
          const uint4 x0a = z0[u * 4 + 0];
          const uint4 x1a = HT1 ? z1[u * 4 + 0] : dummy;
          hy4f_dot_nvfp4_8_u4<HT1>(gw_c.x, lut, x0a, x1a, gd0, gd1);
          hy4f_dot_nvfp4_8_u4<HT1>(uw_c.x, lut, x0a, x1a, ud0, ud1);
          const uint4 x0b = z0[u * 4 + 1];
          const uint4 x1b = HT1 ? z1[u * 4 + 1] : dummy;
          hy4f_dot_nvfp4_8_u4<HT1>(gw_c.y, lut, x0b, x1b, gd0, gd1);
          hy4f_dot_nvfp4_8_u4<HT1>(uw_c.y, lut, x0b, x1b, ud0, ud1);
          g0 = fmaf(gsf.x, gd0, g0);
          u0 = fmaf(usf.x, ud0, u0);
          if constexpr (HT1) {
            g1 = fmaf(gsf.x, gd1, g1);
            u1 = fmaf(usf.x, ud1, u1);
          }

          gd0 = ud0 = gd1 = ud1 = 0.0f;
          const uint4 x0c = z0[u * 4 + 2];
          const uint4 x1c = HT1 ? z1[u * 4 + 2] : dummy;
          hy4f_dot_nvfp4_8_u4<HT1>(gw_c.z, lut, x0c, x1c, gd0, gd1);
          hy4f_dot_nvfp4_8_u4<HT1>(uw_c.z, lut, x0c, x1c, ud0, ud1);
          const uint4 x0d = z0[u * 4 + 3];
          const uint4 x1d = HT1 ? z1[u * 4 + 3] : dummy;
          hy4f_dot_nvfp4_8_u4<HT1>(gw_c.w, lut, x0d, x1d, gd0, gd1);
          hy4f_dot_nvfp4_8_u4<HT1>(uw_c.w, lut, x0d, x1d, ud0, ud1);
          g0 = fmaf(gsf.y, gd0, g0);
          u0 = fmaf(usf.y, ud0, u0);
          if constexpr (HT1) {
            g1 = fmaf(gsf.y, gd1, g1);
            u1 = fmaf(usf.y, ud1, u1);
          }

          gw_c = gw_n; uw_c = uw_n;
          gs_c = gs_n; us_c = us_n;
        }
      }
    } else {
      const int n_u4 = lat >> 4;
      const uint4* gw = reinterpret_cast<const uint4*>(gate_w) + r_idx * n_u4;
      const uint4* uw = reinterpret_cast<const uint4*>(up_w) + r_idx * n_u4;
      const unsigned short* gs = reinterpret_cast<const unsigned short*>(gate_s) + r_idx * n_grp32;
      const unsigned short* us = reinterpret_cast<const unsigned short*>(up_s) + r_idx * n_grp32;
      if (lane < n_u4) {
        uint4 gw_c = ptx_ld_cs_u4(gw + lane);
        uint4 uw_c = ptx_ld_cs_u4(uw + lane);
        unsigned short gs_c = gs[lane >> 1];
        unsigned short us_c = us[lane >> 1];
        for (int u = lane; u < n_u4; u += 32) {
          uint4 gw_n, uw_n;
          unsigned short gs_n = 0, us_n = 0;
          const int un = u + 32;
          if (un < n_u4) {
            gw_n = ptx_ld_cs_u4(gw + un);
            uw_n = ptx_ld_cs_u4(uw + un);
            gs_n = gs[un >> 1];
            us_n = us[un >> 1];
          }
          const float gsc = hy4f_read_f16_or_bf16(gs_c, sb_is_bf16);
          const float usc = hy4f_read_f16_or_bf16(us_c, sb_is_bf16);
          float gd0 = 0.0f, ud0 = 0.0f, gd1 = 0.0f, ud1 = 0.0f;
          const uint4 x0a = z0[u * 2 + 0];
          const uint4 x1a = HT1 ? z1[u * 2 + 0] : dummy;
          hy4f_dot_symq8_8_u4<HT1>(gw_c.x, gw_c.y, x0a, x1a, gd0, gd1);
          hy4f_dot_symq8_8_u4<HT1>(uw_c.x, uw_c.y, x0a, x1a, ud0, ud1);
          const uint4 x0b = z0[u * 2 + 1];
          const uint4 x1b = HT1 ? z1[u * 2 + 1] : dummy;
          hy4f_dot_symq8_8_u4<HT1>(gw_c.z, gw_c.w, x0b, x1b, gd0, gd1);
          hy4f_dot_symq8_8_u4<HT1>(uw_c.z, uw_c.w, x0b, x1b, ud0, ud1);
          g0 = fmaf(gsc, gd0, g0);
          u0 = fmaf(usc, ud0, u0);
          if constexpr (HT1) {
            g1 = fmaf(gsc, gd1, g1);
            u1 = fmaf(usc, ud1, u1);
          }
          gw_c = gw_n; uw_c = uw_n;
          gs_c = gs_n; us_c = us_n;
        }
      }
    }
  };

  if constexpr (DIRECT_M1) {
    run_body(hy4f_bool_tag<false>{});
  } else {
    if (!has_t1) run_body(hy4f_bool_tag<false>{});
    else run_body(hy4f_bool_tag<true>{});
  }

  g0 = hy4_warp_sum(g0);
  u0 = hy4_warp_sum(u0);
  if (has_t1) {
    g1 = hy4_warp_sum(g1);
    u1 = hy4_warp_sum(u1);
  }
  if (lane == 0) {
    mid[static_cast<size_t>(first) * ff + col] =
        __float2bfloat16_rn(hy4_clamped_swiglu(g0, u0, swiglu_limit));
    if (has_t1) {
      mid[static_cast<size_t>(first + 1) * ff + col] =
          __float2bfloat16_rn(hy4_clamped_swiglu(g1, u1, swiglu_limit));
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3A. Zero-SMEM Cold Down Projection (16 rows/CTA, 8 warps = 256 thr/CTA)
//     MODE=0 (NVFP4): 16 lanes/row x 1 row/half-warp = 16 rows/CTA (0 SMEM HW CVT).
//     MODE=1 (SymQ8): 32 lanes/warp x 2 simultaneous rows/warp = 16 rows/CTA (0 SMEM).
//     Grid: dim3(lat / 16, max_items, 1), Block: dim3(32, 8, 1).
// ---------------------------------------------------------------------------
template <int MODE, bool DIRECT_M1>
__global__ __launch_bounds__(256, 4) void hy4_moe_down_cold(
    const unsigned int* __restrict__ down_w,
    const void* __restrict__ down_s,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ experts_or_ids,
    const int* __restrict__ cold_meta,
    const int* __restrict__ counts,
    nv_bfloat16* __restrict__ down_out,
    int lat, int ff, int E, int sb_is_bf16) {
  const int u_idx = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

  int expert, first, cnt;
  if constexpr (DIRECT_M1) {
    expert = experts_or_ids[u_idx];
    first = u_idx;
    cnt = 1;
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(E)) {
      if (tid < 16) {
        const int r = static_cast<int>(blockIdx.x) * 16 + tid;
        if (r < lat) down_out[static_cast<size_t>(first) * lat + r] = __float2bfloat16_rn(0.0f);
      }
      return;
    }
  } else {
    if (u_idx >= counts[0]) return;
    expert = experts_or_ids[u_idx];
    const int meta = cold_meta[u_idx];
    first = meta >> 1;
    cnt = (meta & 1) + 1;
  }

#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
    __syncthreads();
  } else {
    (void)s_e2m1_lut;
  }
  const unsigned int* lut = s_e2m1_lut;
#else
  (void)tid;
  const unsigned int* lut = nullptr;
#endif

  const bool has_t1 = (!DIRECT_M1) && (cnt >= 2);
  const int m_row_u4 = ff >> 3;
  const uint4* m0 = reinterpret_cast<const uint4*>(mid) + static_cast<size_t>(first) * m_row_u4;
  const uint4* m1 = reinterpret_cast<const uint4*>(mid) + static_cast<size_t>(first + 1) * m_row_u4;
  const int n_grp32 = ff >> 5;

  if constexpr (MODE == 0) {
    const int h_id = lane >> 4;
    const int h_lane = lane & 15;
    const int row = static_cast<int>(blockIdx.x) * 16 + warp * 2 + h_id;
    if (row >= lat) return;
    const size_t r_idx = static_cast<size_t>(expert) * lat + row;
    const uint4* wp = reinterpret_cast<const uint4*>(down_w) + r_idx * n_grp32;
    const unsigned short* sp = reinterpret_cast<const unsigned short*>(down_s) + r_idx * n_grp32;

    float a0 = 0.0f, a1 = 0.0f;
    auto run_nv = [&](auto ht1_tag) {
      constexpr bool HT1 = decltype(ht1_tag)::value;
      const uint4 dummy = make_uint4(0u, 0u, 0u, 0u);
      if (h_lane < n_grp32) {
        uint4 wc = ptx_ld_cs_u4(wp + h_lane);
        unsigned short sc = sp[h_lane];
        for (int u = h_lane; u < n_grp32; u += 16) {
          uint4 wn;
          unsigned short sn = 0;
          const int un = u + 16;
          if (un < n_grp32) {
            wn = ptx_ld_cs_u4(wp + un);
            sn = sp[un];
          }
          const float2 sf = hy4_fast_e4m3x2_to_f2(sc);
          float d0 = 0.0f, d1 = 0.0f;
          hy4f_dot_nvfp4_8_u4<HT1>(wc.x, lut, m0[u * 4 + 0], HT1 ? m1[u * 4 + 0] : dummy, d0, d1);
          hy4f_dot_nvfp4_8_u4<HT1>(wc.y, lut, m0[u * 4 + 1], HT1 ? m1[u * 4 + 1] : dummy, d0, d1);
          a0 = fmaf(sf.x, d0, a0);
          if constexpr (HT1) a1 = fmaf(sf.x, d1, a1);

          d0 = 0.0f; d1 = 0.0f;
          hy4f_dot_nvfp4_8_u4<HT1>(wc.z, lut, m0[u * 4 + 2], HT1 ? m1[u * 4 + 2] : dummy, d0, d1);
          hy4f_dot_nvfp4_8_u4<HT1>(wc.w, lut, m0[u * 4 + 3], HT1 ? m1[u * 4 + 3] : dummy, d0, d1);
          a0 = fmaf(sf.y, d0, a0);
          if constexpr (HT1) a1 = fmaf(sf.y, d1, a1);

          wc = wn;
          sc = sn;
        }
      }
    };
    if constexpr (DIRECT_M1) {
      run_nv(hy4f_bool_tag<false>{});
    } else {
      if (!has_t1) run_nv(hy4f_bool_tag<false>{});
      else run_nv(hy4f_bool_tag<true>{});
    }
    a0 = hy4f_halfwarp_sum(a0);
    if (has_t1) a1 = hy4f_halfwarp_sum(a1);
    if (h_lane == 0) {
      down_out[static_cast<size_t>(first) * lat + row] = __float2bfloat16_rn(a0);
      if (has_t1) {
        down_out[static_cast<size_t>(first + 1) * lat + row] = __float2bfloat16_rn(a1);
      }
    }
  } else {
    const int row0 = static_cast<int>(blockIdx.x) * 16 + warp * 2;
    if (row0 >= lat) return;
    const int n_u4 = ff >> 4;
    const size_t r0_idx = static_cast<size_t>(expert) * lat + row0;
    const uint4* w0 = reinterpret_cast<const uint4*>(down_w) + r0_idx * n_u4;
    const unsigned short* s0 = reinterpret_cast<const unsigned short*>(down_s) + r0_idx * n_grp32;
    const uint4* w1 = w0 + n_u4;
    const unsigned short* s1 = s0 + n_grp32;

    float a00 = 0.0f, a01 = 0.0f, a10 = 0.0f, a11 = 0.0f;
    auto run_q8 = [&](auto ht1_tag) {
      constexpr bool HT1 = decltype(ht1_tag)::value;
      const uint4 dummy = make_uint4(0u, 0u, 0u, 0u);
      if (lane < n_u4) {
        uint4 w0_c = ptx_ld_cs_u4(w0 + lane);
        uint4 w1_c = ptx_ld_cs_u4(w1 + lane);
        unsigned short s0_c = s0[lane >> 1];
        unsigned short s1_c = s1[lane >> 1];
        for (int u = lane; u < n_u4; u += 32) {
          uint4 w0_n, w1_n;
          unsigned short s0_n = 0, s1_n = 0;
          const int un = u + 32;
          if (un < n_u4) {
            w0_n = ptx_ld_cs_u4(w0 + un);
            w1_n = ptx_ld_cs_u4(w1 + un);
            s0_n = s0[un >> 1];
            s1_n = s1[un >> 1];
          }
          const float sc0 = hy4f_read_f16_or_bf16(s0_c, sb_is_bf16);
          const float sc1 = hy4f_read_f16_or_bf16(s1_c, sb_is_bf16);
          float d00 = 0.0f, d01 = 0.0f, d10 = 0.0f, d11 = 0.0f;
          const uint4 x0a = m0[u * 2 + 0];
          const uint4 x1a = HT1 ? m1[u * 2 + 0] : dummy;
          hy4f_dot_symq8_8_u4<HT1>(w0_c.x, w0_c.y, x0a, x1a, d00, d01);
          hy4f_dot_symq8_8_u4<HT1>(w1_c.x, w1_c.y, x0a, x1a, d10, d11);
          const uint4 x0b = m0[u * 2 + 1];
          const uint4 x1b = HT1 ? m1[u * 2 + 1] : dummy;
          hy4f_dot_symq8_8_u4<HT1>(w0_c.z, w0_c.w, x0b, x1b, d00, d01);
          hy4f_dot_symq8_8_u4<HT1>(w1_c.z, w1_c.w, x0b, x1b, d10, d11);
          a00 = fmaf(sc0, d00, a00);
          a10 = fmaf(sc1, d10, a10);
          if constexpr (HT1) {
            a01 = fmaf(sc0, d01, a01);
            a11 = fmaf(sc1, d11, a11);
          }
          w0_c = w0_n; w1_c = w1_n;
          s0_c = s0_n; s1_c = s1_n;
        }
      }
    };
    if constexpr (DIRECT_M1) {
      run_q8(hy4f_bool_tag<false>{});
    } else {
      if (!has_t1) run_q8(hy4f_bool_tag<false>{});
      else run_q8(hy4f_bool_tag<true>{});
    }
    a00 = hy4_warp_sum(a00);
    a10 = hy4_warp_sum(a10);
    if (has_t1) {
      a01 = hy4_warp_sum(a01);
      a11 = hy4_warp_sum(a11);
    }
    if (lane == 0) {
      down_out[static_cast<size_t>(first) * lat + row0] = __float2bfloat16_rn(a00);
      down_out[static_cast<size_t>(first) * lat + row0 + 1] = __float2bfloat16_rn(a10);
      if (has_t1) {
        down_out[static_cast<size_t>(first + 1) * lat + row0] = __float2bfloat16_rn(a01);
        down_out[static_cast<size_t>(first + 1) * lat + row0 + 1] = __float2bfloat16_rn(a11);
      }
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2B. Hot Gate + Up + Asymmetric Clamped SwiGLU (cnt >= 3 -> 2-stage cp.async.cg + BF16 TC MMA)
//     No dead GEMV SMEM or scalar registers -> half SMEM, higher occupancy!
//     Grid: dim3(ff / 16, max_hot, 1), Block: dim3(32, 4, 1).
// ---------------------------------------------------------------------------
template <int MODE, bool ALLOW_HI>
__global__ __launch_bounds__(128, ALLOW_HI ? 3 : 4) void hy4_moe_gate_up_hot(
    const unsigned int* __restrict__ gate_w,
    const void* __restrict__ gate_s,
    const unsigned int* __restrict__ up_w,
    const void* __restrict__ up_s,
    const nv_bfloat16* __restrict__ z,
    const int* __restrict__ hot_experts,
    const int* __restrict__ hot_begin,
    const int* __restrict__ hot_end,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ counts,
    nv_bfloat16* __restrict__ mid,
    int lat, int ff, int top_k, float swiglu_limit, int sb_is_bf16,
    int z_row_u4) {
  const int u_idx = static_cast<int>(blockIdx.y);
  if (u_idx >= counts[1]) return;
  const int expert = hot_experts[u_idx];
  const int first = hot_begin[u_idx];
  const int cnt = hot_end[u_idx] - first;
  if (cnt <= 0) return;

  const int row0 = static_cast<int>(blockIdx.x) * 16;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const size_t exp_base = static_cast<size_t>(expert) * ff;

  struct StageMma {
    alignas(16) uint4 w[16][MODE == 0 ? 8 : 16];
    alignas(16) unsigned int sc[16][4];
  };
  union SharedGuHot {
    StageMma pipe[2];
    struct {
      float gate[4][ALLOW_HI ? 16 : 8][17];
      float up[4][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedGuHot smem;
  __shared__ int s_tok[ALLOW_HI ? 16 : 8];
#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
    s_e2m1_lut[tid + 128] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid + 128));
  } else {
    (void)s_e2m1_lut;
  }
#endif

  constexpr int MAX_C = ALLOW_HI ? 16 : 8;
  if (tid < cnt) {
    s_tok[tid] = sorted_pairs[first + tid] / top_k;
  } else if (tid < MAX_C) {
    s_tok[tid] = -1;
  }
  __syncthreads();

  const int g = lane >> 2;
  const int t = lane & 3;
  const int tok_g_lo = s_tok[g];
  [[maybe_unused]] const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);

  const uint4* gw_u4 = reinterpret_cast<const uint4*>(gate_w);
  const uint4* uw_u4 = reinterpret_cast<const uint4*>(up_w);
  const uint4* z_u4 = reinterpret_cast<const uint4*>(z);
  const unsigned int* gs_u32 = reinterpret_cast<const unsigned int*>(gate_s);
  const unsigned int* us_u32 = reinterpret_cast<const unsigned int*>(up_s);

  const int row_w_u4 = (MODE == 0) ? (lat >> 5) : (lat >> 4);
  const int row_sc_u32 = lat >> 6;
  const int n_chunks = (lat + 255) >> 8;
  const int total_steps = n_chunks * 2;

  auto load_chunk = [&](int st, int chunk, bool is_up) {
    const uint4* w_src = is_up ? uw_u4 : gw_u4;
    const unsigned int* s_src = is_up ? us_u32 : gs_u32;
    if constexpr (MODE == 0) {
      const int r = tid >> 3;
      const int c = tid & 7;
      const size_t ri = exp_base + row0 + r;
      const int col = chunk * 8 + c;
      ptx_cp_async16_cg_zfill(
          &smem.pipe[st].w[r][c ^ (r & 7)], w_src + ri * row_w_u4 + col, col < row_w_u4);
    } else {
      const int r = tid >> 3;
      const int c0 = (tid & 7) * 2;
      const size_t ri = exp_base + row0 + r;
      const int col0 = chunk * 16 + c0;
      const uint4* wp = w_src + ri * row_w_u4 + col0;
      ptx_cp_async16_cg_zfill(&smem.pipe[st].w[r][c0 ^ r], wp, col0 < row_w_u4);
      ptx_cp_async16_cg_zfill(&smem.pipe[st].w[r][(c0 + 1) ^ r], wp + 1, (col0 + 1) < row_w_u4);
    }
    if (tid < 64) {
      const int r = tid >> 2;
      const int c = tid & 3;
      const size_t sr = exp_base + row0 + r;
      const int sc_col = chunk * 4 + c;
      if (sc_col < row_sc_u32) {
        ptx_cp_async4_ca(&smem.pipe[st].sc[r][c], s_src + sr * row_sc_u32 + sc_col);
      }
    }
    ptx_cp_commit();
  };

  float g_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float g_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float u_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  load_chunk(0, 0, false);
#pragma unroll 1
  for (int step = 0; step < total_steps; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < total_steps) {
      load_chunk(nxt, (step + 1) >> 1, ((step + 1) & 1) != 0);
      ptx_cp_wait<1>();
    } else {
      ptx_cp_wait<0>();
    }
    __syncthreads();

    const int chunk = step >> 1;
    const bool is_up = (step & 1) != 0;
    float* t_lo = is_up ? u_tot_lo : g_tot_lo;
    [[maybe_unused]] float* t_hi = is_up ? u_tot_hi : g_tot_hi;

    if constexpr (MODE == 0) {
      const int total_grp16 = lat >> 4;
      const unsigned short* sc_r0 = reinterpret_cast<const unsigned short*>(smem.pipe[cur].sc[g]);
      const unsigned short* sc_r1 = reinterpret_cast<const unsigned short*>(smem.pipe[cur].sc[g + 8]);
      const uint2* z_u2 = reinterpret_cast<const uint2*>(z);
      const int z_row_u2 = z_row_u4 << 1;
#pragma unroll
      for (int kg2 = 0; kg2 < 2; ++kg2) {
        const int lg0 = warp * 4 + kg2 * 2;
        if (chunk * 16 + lg0 < total_grp16) {
          const float2 sf0 = hy4_fast_e4m3x2_to_f2(sc_r0[warp * 2 + kg2]);
          const float2 sf1 = hy4_fast_e4m3x2_to_f2(sc_r1[warp * 2 + kg2]);
#pragma unroll
          for (int sub = 0; sub < 2; ++sub) {
            const int lg = lg0 + sub;
            const int grp16 = chunk * 16 + lg;
            const int col_u4 = lg >> 1;
            const int sub_u16 = (lg & 1) * 4 + t;
            const unsigned short w0 = reinterpret_cast<const unsigned short*>(
                &smem.pipe[cur].w[g][col_u4 ^ g])[sub_u16];
            const unsigned short w1 = reinterpret_cast<const unsigned short*>(
                &smem.pipe[cur].w[g + 8][col_u4 ^ g])[sub_u16];
#if HY4_HAS_HW_FP4_CVT
            const unsigned int a0 = hy4_fast_e2m1x2_to_bf16x2(w0 & 0xFFu);
            const unsigned int a2 = hy4_fast_e2m1x2_to_bf16x2(w0 >> 8);
            const unsigned int a1 = hy4_fast_e2m1x2_to_bf16x2(w1 & 0xFFu);
            const unsigned int a3 = hy4_fast_e2m1x2_to_bf16x2(w1 >> 8);
#else
            const unsigned int a0 = s_e2m1_lut[w0 & 0xFFu];
            const unsigned int a2 = s_e2m1_lut[w0 >> 8];
            const unsigned int a1 = s_e2m1_lut[w1 & 0xFFu];
            const unsigned int a3 = s_e2m1_lut[w1 >> 8];
#endif
            const float sc0 = (sub == 0) ? sf0.x : sf0.y;
            const float sc1 = (sub == 0) ? sf1.x : sf1.y;
            const uint2 xv0 = (tok_g_lo >= 0)
                ? z_u2[static_cast<size_t>(tok_g_lo) * z_row_u2 + grp16 * 4 + t]
                : make_uint2(0u, 0u);
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
            t_lo[0] = fmaf(sc0, d0, t_lo[0]);
            t_lo[1] = fmaf(sc0, d1, t_lo[1]);
            t_lo[2] = fmaf(sc1, d2, t_lo[2]);
            t_lo[3] = fmaf(sc1, d3, t_lo[3]);
            if constexpr (ALLOW_HI) {
              if (do_hi) {
                const uint2 xv1 = (tok_g_hi >= 0)
                    ? z_u2[static_cast<size_t>(tok_g_hi) * z_row_u2 + grp16 * 4 + t]
                    : make_uint2(0u, 0u);
                float h0, h1, h2, h3;
                ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
                t_hi[0] = fmaf(sc0, h0, t_hi[0]);
                t_hi[1] = fmaf(sc0, h1, t_hi[1]);
                t_hi[2] = fmaf(sc1, h2, t_hi[2]);
                t_hi[3] = fmaf(sc1, h3, t_hi[3]);
              }
            }
          }
        }
      }
    } else {
      const int total_grp32 = lat >> 5;
      const unsigned short* sc_r0 = reinterpret_cast<const unsigned short*>(smem.pipe[cur].sc[g]);
      const unsigned short* sc_r1 = reinterpret_cast<const unsigned short*>(smem.pipe[cur].sc[g + 8]);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int lg = warp * 2 + kg;
        const int grp32 = chunk * 8 + lg;
        if (grp32 < total_grp32) {
          const int col_u4 = lg * 2 + (t >> 1);
          const int half_u2 = t & 1;
          const uint2 w0 = reinterpret_cast<const uint2*>(
              &smem.pipe[cur].w[g][col_u4 ^ g])[half_u2];
          const uint2 w1 = reinterpret_cast<const uint2*>(
              &smem.pipe[cur].w[g + 8][col_u4 ^ (g + 8)])[half_u2];
          unsigned int wb0[4], wb1[4];
          hy4f_unpack8_symq8_bf16x2(w0, wb0);
          hy4f_unpack8_symq8_bf16x2(w1, wb1);

          const uint4 xv0 = (tok_g_lo >= 0)
              ? z_u4[static_cast<size_t>(tok_g_lo) * z_row_u4 + grp32 * 4 + t]
              : make_uint4(0u, 0u, 0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
          ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

          const float sc0 = hy4f_read_f16_or_bf16(sc_r0[lg], sb_is_bf16);
          const float sc1 = hy4f_read_f16_or_bf16(sc_r1[lg], sb_is_bf16);
          t_lo[0] = fmaf(sc0, d0, t_lo[0]);
          t_lo[1] = fmaf(sc0, d1, t_lo[1]);
          t_lo[2] = fmaf(sc1, d2, t_lo[2]);
          t_lo[3] = fmaf(sc1, d3, t_lo[3]);
          if constexpr (ALLOW_HI) {
            if (do_hi) {
              const uint4 xv1 = (tok_g_hi >= 0)
                  ? z_u4[static_cast<size_t>(tok_g_hi) * z_row_u4 + grp32 * 4 + t]
                  : make_uint4(0u, 0u, 0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
              ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
              t_hi[0] = fmaf(sc0, h0, t_hi[0]);
              t_hi[1] = fmaf(sc0, h1, t_hi[1]);
              t_hi[2] = fmaf(sc1, h2, t_hi[2]);
              t_hi[3] = fmaf(sc1, h3, t_hi[3]);
            }
          }
        }
      }
    }
    __syncthreads();
  }

  smem.red.gate[warp][2 * t][g] = g_tot_lo[0];
  smem.red.gate[warp][2 * t + 1][g] = g_tot_lo[1];
  smem.red.gate[warp][2 * t][g + 8] = g_tot_lo[2];
  smem.red.gate[warp][2 * t + 1][g + 8] = g_tot_lo[3];
  smem.red.up[warp][2 * t][g] = u_tot_lo[0];
  smem.red.up[warp][2 * t + 1][g] = u_tot_lo[1];
  smem.red.up[warp][2 * t][g + 8] = u_tot_lo[2];
  smem.red.up[warp][2 * t + 1][g + 8] = u_tot_lo[3];
  if constexpr (ALLOW_HI) {
    if (do_hi) {
      smem.red.gate[warp][8 + 2 * t][g] = g_tot_hi[0];
      smem.red.gate[warp][8 + 2 * t + 1][g] = g_tot_hi[1];
      smem.red.gate[warp][8 + 2 * t][g + 8] = g_tot_hi[2];
      smem.red.gate[warp][8 + 2 * t + 1][g + 8] = g_tot_hi[3];
      smem.red.up[warp][8 + 2 * t][g] = u_tot_hi[0];
      smem.red.up[warp][8 + 2 * t + 1][g] = u_tot_hi[1];
      smem.red.up[warp][8 + 2 * t][g + 8] = u_tot_hi[2];
      smem.red.up[warp][8 + 2 * t + 1][g + 8] = u_tot_hi[3];
    }
  }
  __syncthreads();

  const int total_out = cnt * 16;
  for (int idx = tid; idx < total_out; idx += 128) {
    const int c = idx >> 4;
    const int r = idx & 15;
    const float gv = (smem.red.gate[0][c][r] + smem.red.gate[1][c][r]) +
                     (smem.red.gate[2][c][r] + smem.red.gate[3][c][r]);
    const float uv = (smem.red.up[0][c][r] + smem.red.up[1][c][r]) +
                     (smem.red.up[2][c][r] + smem.red.up[3][c][r]);
    mid[static_cast<size_t>(first + c) * ff + row0 + r] =
        __float2bfloat16_rn(hy4_clamped_swiglu(gv, uv, swiglu_limit));
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3B. Hot Down Projection (cnt >= 3 -> 2x16-row tiles, 2-stage cp.async.cg + BF16 TC MMA)
//     No dead GEMV SMEM or scalar registers -> half SMEM, higher occupancy!
//     Grid: dim3(lat / 32, max_hot, 1), Block: dim3(32, 4, 1).
// ---------------------------------------------------------------------------
template <int MODE, bool ALLOW_HI>
__global__ __launch_bounds__(128, ALLOW_HI ? 3 : 4) void hy4_moe_down_hot(
    const unsigned int* __restrict__ down_w,
    const void* __restrict__ down_s,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ hot_experts,
    const int* __restrict__ hot_begin,
    const int* __restrict__ hot_end,
    const int* __restrict__ counts,
    nv_bfloat16* __restrict__ down_out,
    int lat, int ff, int sb_is_bf16) {
  const int u_idx = static_cast<int>(blockIdx.y);
  if (u_idx >= counts[1]) return;
  const int expert = hot_experts[u_idx];
  const int first = hot_begin[u_idx];
  const int cnt = hot_end[u_idx] - first;
  if (cnt <= 0) return;

  const int base_row = static_cast<int>(blockIdx.x) * 32;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const size_t exp_base = static_cast<size_t>(expert) * lat;

  struct StageMmaDn {
    alignas(16) uint4 w[16][MODE == 0 ? 8 : 16];
    alignas(16) unsigned int sc[16][4];
  };
  __shared__ StageMmaDn pipe[2];
  __shared__ float s_red[4][ALLOW_HI ? 16 : 8][17];
#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    s_e2m1_lut[tid] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
    s_e2m1_lut[tid + 128] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid + 128));
  } else {
    (void)s_e2m1_lut;
  }
#endif

  const int g = lane >> 2;
  const int t = lane & 3;
  const int slot_lo = (g < cnt) ? (first + g) : -1;
  [[maybe_unused]] const int slot_hi = (ALLOW_HI && (8 + g) < cnt) ? (first + 8 + g) : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);

  const uint4* w_u4 = reinterpret_cast<const uint4*>(down_w);
  const uint4* m_u4 = reinterpret_cast<const uint4*>(mid);
  const unsigned int* s_u32 = reinterpret_cast<const unsigned int*>(down_s);
  const int m_row_u4 = ff >> 3;
  const int row_w_u4 = (MODE == 0) ? (ff >> 5) : (ff >> 4);
  const int row_sc_u32 = ff >> 6;
  const int n_chunks = (ff + 255) >> 8;
  const int total_steps = 2 * n_chunks;

  auto load_chunk = [&](int st, int row_off, int chunk) {
    if constexpr (MODE == 0) {
      const int r = tid >> 3;
      const int c = tid & 7;
      const size_t ri = exp_base + row_off + r;
      const int col = chunk * 8 + c;
      ptx_cp_async16_cg_zfill(
          &pipe[st].w[r][c ^ (r & 7)], w_u4 + ri * row_w_u4 + col, col < row_w_u4);
    } else {
      const int r = tid >> 3;
      const int c0 = (tid & 7) * 2;
      const size_t ri = exp_base + row_off + r;
      const int col0 = chunk * 16 + c0;
      const uint4* wp = w_u4 + ri * row_w_u4 + col0;
      ptx_cp_async16_cg_zfill(&pipe[st].w[r][c0 ^ r], wp, col0 < row_w_u4);
      ptx_cp_async16_cg_zfill(&pipe[st].w[r][(c0 + 1) ^ r], wp + 1, (col0 + 1) < row_w_u4);
    }
    if (tid < 64) {
      const int r = tid >> 2;
      const int c = tid & 3;
      const size_t sr = exp_base + row_off + r;
      const int sc_col = chunk * 4 + c;
      if (sc_col < row_sc_u32) {
        ptx_cp_async4_ca(&pipe[st].sc[r][c], s_u32 + sr * row_sc_u32 + sc_col);
      }
    }
    ptx_cp_commit();
  };

  float tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  load_chunk(0, base_row, 0);
#pragma unroll 1
  for (int step = 0; step < total_steps; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < total_steps) {
      const int n_ti = (step + 1 >= n_chunks) ? 1 : 0;
      const int n_ch = (step + 1) - n_ti * n_chunks;
      load_chunk(nxt, base_row + n_ti * 16, n_ch);
      ptx_cp_wait<1>();
    } else {
      ptx_cp_wait<0>();
    }
    __syncthreads();

    const int ti = (step >= n_chunks) ? 1 : 0;
    const int ch = step - ti * n_chunks;
    const int row0 = base_row + ti * 16;

    if constexpr (MODE == 0) {
      const int total_grp16 = ff >> 4;
      const unsigned short* sc_r0 = reinterpret_cast<const unsigned short*>(pipe[cur].sc[g]);
      const unsigned short* sc_r1 = reinterpret_cast<const unsigned short*>(pipe[cur].sc[g + 8]);
      const uint2* m_u2 = reinterpret_cast<const uint2*>(mid);
      const int m_row_u2 = ff >> 2;
#pragma unroll
      for (int kg2 = 0; kg2 < 2; ++kg2) {
        const int lg0 = warp * 4 + kg2 * 2;
        if (ch * 16 + lg0 < total_grp16) {
          const float2 sf0 = hy4_fast_e4m3x2_to_f2(sc_r0[warp * 2 + kg2]);
          const float2 sf1 = hy4_fast_e4m3x2_to_f2(sc_r1[warp * 2 + kg2]);
#pragma unroll
          for (int sub = 0; sub < 2; ++sub) {
            const int lg = lg0 + sub;
            const int grp16 = ch * 16 + lg;
            const int col_u4 = lg >> 1;
            const int sub_u16 = (lg & 1) * 4 + t;
            const unsigned short w0 = reinterpret_cast<const unsigned short*>(
                &pipe[cur].w[g][col_u4 ^ g])[sub_u16];
            const unsigned short w1 = reinterpret_cast<const unsigned short*>(
                &pipe[cur].w[g + 8][col_u4 ^ g])[sub_u16];
#if HY4_HAS_HW_FP4_CVT
            const unsigned int a0 = hy4_fast_e2m1x2_to_bf16x2(w0 & 0xFFu);
            const unsigned int a2 = hy4_fast_e2m1x2_to_bf16x2(w0 >> 8);
            const unsigned int a1 = hy4_fast_e2m1x2_to_bf16x2(w1 & 0xFFu);
            const unsigned int a3 = hy4_fast_e2m1x2_to_bf16x2(w1 >> 8);
#else
            const unsigned int a0 = s_e2m1_lut[w0 & 0xFFu];
            const unsigned int a2 = s_e2m1_lut[w0 >> 8];
            const unsigned int a1 = s_e2m1_lut[w1 & 0xFFu];
            const unsigned int a3 = s_e2m1_lut[w1 >> 8];
#endif
            const float sc0 = (sub == 0) ? sf0.x : sf0.y;
            const float sc1 = (sub == 0) ? sf1.x : sf1.y;
            const uint2 xv0 = (slot_lo >= 0)
                ? m_u2[static_cast<size_t>(slot_lo) * m_row_u2 + grp16 * 4 + t]
                : make_uint2(0u, 0u);
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
            tot_lo[0] = fmaf(sc0, d0, tot_lo[0]);
            tot_lo[1] = fmaf(sc0, d1, tot_lo[1]);
            tot_lo[2] = fmaf(sc1, d2, tot_lo[2]);
            tot_lo[3] = fmaf(sc1, d3, tot_lo[3]);
            if constexpr (ALLOW_HI) {
              if (do_hi) {
                const uint2 xv1 = (slot_hi >= 0)
                    ? m_u2[static_cast<size_t>(slot_hi) * m_row_u2 + grp16 * 4 + t]
                    : make_uint2(0u, 0u);
                float h0, h1, h2, h3;
                ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
                tot_hi[0] = fmaf(sc0, h0, tot_hi[0]);
                tot_hi[1] = fmaf(sc0, h1, tot_hi[1]);
                tot_hi[2] = fmaf(sc1, h2, tot_hi[2]);
                tot_hi[3] = fmaf(sc1, h3, tot_hi[3]);
              }
            }
          }
        }
      }
    } else {
      const int total_grp32 = ff >> 5;
      const unsigned short* sc_r0 = reinterpret_cast<const unsigned short*>(pipe[cur].sc[g]);
      const unsigned short* sc_r1 = reinterpret_cast<const unsigned short*>(pipe[cur].sc[g + 8]);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int lg = warp * 2 + kg;
        const int grp32 = ch * 8 + lg;
        if (grp32 < total_grp32) {
          const int col_u4 = lg * 2 + (t >> 1);
          const int half_u2 = t & 1;
          const uint2 w0 = reinterpret_cast<const uint2*>(
              &pipe[cur].w[g][col_u4 ^ g])[half_u2];
          const uint2 w1 = reinterpret_cast<const uint2*>(
              &pipe[cur].w[g + 8][col_u4 ^ (g + 8)])[half_u2];
          unsigned int wb0[4], wb1[4];
          hy4f_unpack8_symq8_bf16x2(w0, wb0);
          hy4f_unpack8_symq8_bf16x2(w1, wb1);

          const uint4 xv0 = (slot_lo >= 0)
              ? m_u4[static_cast<size_t>(slot_lo) * m_row_u4 + grp32 * 4 + t]
              : make_uint4(0u, 0u, 0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
          ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

          const float sc0 = hy4f_read_f16_or_bf16(sc_r0[lg], sb_is_bf16);
          const float sc1 = hy4f_read_f16_or_bf16(sc_r1[lg], sb_is_bf16);
          tot_lo[0] = fmaf(sc0, d0, tot_lo[0]);
          tot_lo[1] = fmaf(sc0, d1, tot_lo[1]);
          tot_lo[2] = fmaf(sc1, d2, tot_lo[2]);
          tot_lo[3] = fmaf(sc1, d3, tot_lo[3]);
          if constexpr (ALLOW_HI) {
            if (do_hi) {
              const uint4 xv1 = (slot_hi >= 0)
                  ? m_u4[static_cast<size_t>(slot_hi) * m_row_u4 + grp32 * 4 + t]
                  : make_uint4(0u, 0u, 0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
              ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
              tot_hi[0] = fmaf(sc0, h0, tot_hi[0]);
              tot_hi[1] = fmaf(sc0, h1, tot_hi[1]);
              tot_hi[2] = fmaf(sc1, h2, tot_hi[2]);
              tot_hi[3] = fmaf(sc1, h3, tot_hi[3]);
            }
          }
        }
      }
    }

    if (ch == n_chunks - 1) {
      s_red[warp][2 * t][g] = tot_lo[0];
      s_red[warp][2 * t + 1][g] = tot_lo[1];
      s_red[warp][2 * t][g + 8] = tot_lo[2];
      s_red[warp][2 * t + 1][g + 8] = tot_lo[3];
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          s_red[warp][8 + 2 * t][g] = tot_hi[0];
          s_red[warp][8 + 2 * t + 1][g] = tot_hi[1];
          s_red[warp][8 + 2 * t][g + 8] = tot_hi[2];
          s_red[warp][8 + 2 * t + 1][g + 8] = tot_hi[3];
        }
      }
      __syncthreads();
      const int total_out = cnt * 16;
      for (int idx = tid; idx < total_out; idx += 128) {
        const int c = idx >> 4;
        const int r = idx & 15;
        const float val = (s_red[0][c][r] + s_red[1][c][r]) +
                          (s_red[2][c][r] + s_red[3][c][r]);
        down_out[static_cast<size_t>(first + c) * lat + row0 + r] = __float2bfloat16_rn(val);
      }
      tot_lo[0] = tot_lo[1] = tot_lo[2] = tot_lo[3] = 0.0f;
      if constexpr (ALLOW_HI) {
        tot_hi[0] = tot_hi[1] = tot_hi[2] = tot_hi[3] = 0.0f;
      }
    }
    __syncthreads();
  }
}
)cuda"
