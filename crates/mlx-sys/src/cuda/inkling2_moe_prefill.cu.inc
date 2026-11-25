R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Inkling-2-Mini Grouped MoE Prefill CUDA Kernels (M >= 17)
//   E = 128, top_k = 8, H = 2304, I = 1152, route_scale = 8.0, no shared expert
//   Self-contained NVRTC CUDA (zero CuTe headers; safe on Windows CUDA!)
//   Modes: 0=NVFP4, 1=SymQ8, 2=AffineQ8, 3=Q5_K (gate_up only), 4=MXFP8
//   BM = 64, BN = 128, BK = 64, 8 warps/CTA (Q5_K uses BM=32/BN=64), 2-stage
//   cp.async.cg/ca + HW FP4/FP8 PTX cvt + Tensor Core MMA, 0 B spills
// ============================================================================

__device__ __forceinline__ unsigned int i2p_pack_bf16x2(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float i2p_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

// PTX ISA 9.2 (CUDA 13.2+) Blackwell arch-specific direct FP4/FP8 -> bf16x2 conversions.
// Earlier toolchains (e.g. CUDA 13.0 on Blackwell) fall back to PTX 8.1 cvt.rn.f16x2.e4m3x2
// for FP8 and the SMEM E2M1 LUT for NVFP4 so NVRTC never emits unsupported PTX 9.2 ops.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define I2P_HAS_HW_FP4_CVT 1
#else
#define I2P_HAS_HW_FP4_CVT 0
#endif

// PTX ISA 8.1 (CUDA 12.1+) Ada/Hopper/Blackwell FP8 E4M3FN -> f16x2 conversion.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))
#define I2P_HAS_HW_FP8_F16_CVT 1
#else
#define I2P_HAS_HW_FP8_F16_CVT 0
#endif

__device__ __forceinline__ float2 i2p_decode_fp8_e4m3x2_unscaled(unsigned int pair16) {
  const unsigned int b0 = pair16 & 0xFFu;
  const unsigned int b1 = (pair16 >> 8) & 0xFFu;
  const unsigned int v0 = b0 & 127u;
  const unsigned int v1 = b1 & 127u;
  const unsigned int h0 = (v0 << 7) | (((v0 + 1u) >> 7) << 14) | ((b0 & 128u) << 8);
  const unsigned int h1 = (v1 << 7) | (((v1 + 1u) >> 7) << 14) | ((b1 & 128u) << 8);
  const unsigned int h2 = h0 | (h1 << 16);
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

// Exact 0-ULP 2x FP8 E4M3FN -> float2 (1-instr HW PTX on Ada/Hopper/Blackwell).
__device__ __forceinline__ float2 i2p_fast_e4m3x2_to_f2(unsigned short pair16) {
#if I2P_HAS_HW_FP4_CVT
  unsigned int bf2;
  asm("{ cvt.rn.bf16x2.e4m3x2 %0, %1; }" : "=r"(bf2) : "h"(pair16));
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
#elif I2P_HAS_HW_FP8_F16_CVT
  unsigned int h2;
  asm("{ cvt.rn.f16x2.e4m3x2 %0, %1; }" : "=r"(h2) : "h"(pair16));
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
#else
  const float2 f = i2p_decode_fp8_e4m3x2_unscaled(pair16);
  return make_float2(f.x * 256.0f, f.y * 256.0f);
#endif
}

// Exact 0-ULP 2x FP8 E4M3FN -> bf16x2 (scale 1.0) for BF16 Tensor Cores.
__device__ __forceinline__ unsigned int i2p_decode_fp8_e4m3x2_bf16x2(unsigned int pair16) {
#if I2P_HAS_HW_FP4_CVT
  unsigned int bf2;
  const unsigned short p16 = static_cast<unsigned short>(pair16);
  asm("{ cvt.rn.bf16x2.e4m3x2 %0, %1; }" : "=r"(bf2) : "h"(p16));
  return bf2;
#elif I2P_HAS_HW_FP8_F16_CVT
  unsigned int h2;
  const unsigned short p16 = static_cast<unsigned short>(pair16);
  asm("{ cvt.rn.f16x2.e4m3x2 %0, %1; }" : "=r"(h2) : "h"(p16));
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
  return i2p_pack_bf16x2(f.x, f.y);
#else
  const float2 f = i2p_decode_fp8_e4m3x2_unscaled(pair16);
  return i2p_pack_bf16x2(f.x * 256.0f, f.y * 256.0f);
#endif
}

__device__ __forceinline__ float i2p_decode_ue8m0(unsigned int b) {
  return __uint_as_float((b == 0u) ? 0x00400000u : (b << 23));
}

__device__ __forceinline__ unsigned int i2p_e2m1_to_bf16(unsigned int nib) {
  const unsigned int m = nib & 7u;
  const unsigned int s = (nib & 8u) << 12;
  const unsigned int mag = (m >= 2u) ? (0x3F00u + (m << 6)) : (m ? 0x3F00u : 0u);
  return s | mag;
}

__device__ __forceinline__ unsigned int i2p_e2m1x2_to_bf16x2(unsigned int byte8) {
  return i2p_e2m1_to_bf16(byte8 & 0xFu) | (i2p_e2m1_to_bf16((byte8 >> 4) & 0xFu) << 16);
}

// Bit-exact 2x E2M1 -> bf16x2 (1-instr hardware PTX on Blackwell sm_100a/120a/121a).
__device__ __forceinline__ unsigned int i2p_fast_e2m1x2_to_bf16x2(unsigned int byte8) {
#if I2P_HAS_HW_FP4_CVT
  unsigned int out;
  const unsigned short tmp = static_cast<unsigned short>(byte8);
  asm("{ .reg .b8 b0, b1; mov.b16 {b0, b1}, %1; cvt.rn.bf16x2.e2m1x2 %0, b0; }"
      : "=r"(out) : "h"(tmp));
  return out;
#else
  return i2p_e2m1x2_to_bf16x2(byte8);
#endif
}

__device__ __forceinline__ float i2p_read_f16_or_bf16(unsigned short raw, int is_bf16) {
  return is_bf16 ? __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(&raw))
                 : __half2float(__ushort_as_half(raw));
}

__device__ __forceinline__ unsigned int i2p_unpack4_q5k(unsigned int u) {
  return (u & 0x0000001fu) | ((u & 0x000003e0u) << 3) |
         ((u & 0x00007c00u) << 6) | ((u & 0x000f8000u) << 9);
}

__device__ __forceinline__ void i2p_load_pair_q5k(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  const int bit0 = t * 20;
  const int w0 = bit0 >> 5;
  const int sh0 = bit0 & 31;
  unsigned int u0 = seg[w0];
  if (sh0 > 12) u0 = __funnelshift_r(u0, seg[w0 + 1], sh0);
  else u0 >>= sh0;

  const int bit1 = 80 + t * 20;
  const int w1 = bit1 >> 5;
  const int sh1 = bit1 & 31;
  unsigned int u1 = seg[w1];
  if (sh1 > 12) u1 = __funnelshift_r(u1, seg[w1 + 1], sh1);
  else u1 >>= sh1;

  out0 = i2p_unpack4_q5k(u0);
  out1 = i2p_unpack4_q5k(u1);
}

__device__ __forceinline__ int i2p_min_dot8(const uint4& sc, const uint4& qsu4) {
  const short* q = reinterpret_cast<const short*>(&qsu4);
  int sum = static_cast<int>(q[0]) * static_cast<int>((sc.x >> 8) & 0xffu) +
            static_cast<int>(q[1]) * static_cast<int>(sc.x >> 24);
  sum += static_cast<int>(q[2]) * static_cast<int>((sc.y >> 8) & 0xffu) +
         static_cast<int>(q[3]) * static_cast<int>(sc.y >> 24);
  sum += static_cast<int>(q[4]) * static_cast<int>((sc.z >> 8) & 0xffu) +
         static_cast<int>(q[5]) * static_cast<int>(sc.z >> 24);
  sum += static_cast<int>(q[6]) * static_cast<int>((sc.w >> 8) & 0xffu) +
         static_cast<int>(q[7]) * static_cast<int>(sc.w >> 24);
  return sum;
}

__device__ __forceinline__ nv_bfloat16 i2p_situ_epilogue(
    float acc_g, float acc_u, float outer_scale) {
  const float g = __bfloat162float(__float2bfloat16_rn(acc_g)) * outer_scale;
  const float u = __bfloat162float(__float2bfloat16_rn(acc_u)) * outer_scale;
  const float g_sc = 4.0f * tanhf(fminf(fmaxf(g * 0.25f, -20.0f), 20.0f));
  const float u_sc = 25.0f * tanhf(fminf(fmaxf(u * 0.04f, -20.0f), 20.0f));
  const float g_sig = 1.0f / (1.0f + expf(-g));
  return __float2bfloat16_rn(g_sc * g_sig * u_sc);
}

template <int MODE>
__device__ __forceinline__ void i2p_unpack8_to_bf16x2(const uint2& w, unsigned int out[4]) {
  if constexpr (MODE == 1) {
    const unsigned int u0 = w.x ^ 0x80808080u;
    const unsigned int u1 = w.y ^ 0x80808080u;
    out[0] = i2p_pack_bf16x2(float(static_cast<signed char>(u0)), float(static_cast<signed char>(u0 >> 8)));
    out[1] = i2p_pack_bf16x2(float(static_cast<signed char>(u0 >> 16)), float(static_cast<signed char>(u0 >> 24)));
    out[2] = i2p_pack_bf16x2(float(static_cast<signed char>(u1)), float(static_cast<signed char>(u1 >> 8)));
    out[3] = i2p_pack_bf16x2(float(static_cast<signed char>(u1 >> 16)), float(static_cast<signed char>(u1 >> 24)));
  } else if constexpr (MODE == 2) {
    out[0] = i2p_pack_bf16x2(float(w.x & 0xFFu), float((w.x >> 8) & 0xFFu));
    out[1] = i2p_pack_bf16x2(float((w.x >> 16) & 0xFFu), float(w.x >> 24));
    out[2] = i2p_pack_bf16x2(float(w.y & 0xFFu), float((w.y >> 8) & 0xFFu));
    out[3] = i2p_pack_bf16x2(float((w.y >> 16) & 0xFFu), float(w.y >> 24));
  } else {
    out[0] = i2p_decode_fp8_e4m3x2_bf16x2(w.x);
    out[1] = i2p_decode_fp8_e4m3x2_bf16x2(w.x >> 16);
    out[2] = i2p_decode_fp8_e4m3x2_bf16x2(w.y);
    out[3] = i2p_decode_fp8_e4m3x2_bf16x2(w.y >> 16);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 1. Prefill Exact-FP32 Router GEMM: logits[M, 128] = F32(x[M, 2304]) @ router_w[128, 2304]^T
//    BM = 16, BN = 16, BK = 32, Block: dim3(16, 16), Grid: dim3(8, ceil(M/16), 1)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 4) void inkling2_moe_prefill_router_gemm(
    const nv_bfloat16* __restrict__ x,
    const float* __restrict__ router_w,
    float* __restrict__ logits,
    int M) {
  const int n0 = blockIdx.x * 16;
  const int m0 = blockIdx.y * 16;
  const int tx = threadIdx.x; // 0..15 (col in tile)
  const int ty = threadIdx.y; // 0..15 (row in tile)
  const int m = m0 + ty;
  const int e = n0 + tx;

  __shared__ float s_a[16][33];
  __shared__ float s_b[16][33];

  float acc = 0.0f;
#pragma unroll 1
  for (int k0 = 0; k0 < 2304; k0 += 32) {
    if (m < M) {
      s_a[ty][tx]      = __bfloat162float(x[static_cast<size_t>(m) * 2304 + k0 + tx]);
      s_a[ty][tx + 16] = __bfloat162float(x[static_cast<size_t>(m) * 2304 + k0 + tx + 16]);
    } else {
      s_a[ty][tx] = 0.0f;
      s_a[ty][tx + 16] = 0.0f;
    }
    const int e_ld = n0 + ty;
    s_b[ty][tx]      = router_w[static_cast<size_t>(e_ld) * 2304 + k0 + tx];
    s_b[ty][tx + 16] = router_w[static_cast<size_t>(e_ld) * 2304 + k0 + tx + 16];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < 32; ++k) {
      acc = fmaf(s_a[ty][k], s_b[tx][k], acc);
    }
    __syncthreads();
  }

  if (m < M) {
    logits[static_cast<size_t>(m) * 128 + e] = acc;
  }
}

// ---------------------------------------------------------------------------
// 2A. Prefill Exact Top-8 Routing (sigmoid + bias + smaller-ID tie-break + norm)
//     Grid: dim3((M + 7) / 8, 1, 1), Block: dim3(32, 8) (1 warp/token).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 4) void inkling2_moe_prefill_topk(
    const float* __restrict__ logits,
    const float* __restrict__ router_bias,
    const float* __restrict__ dn_outer_scale,
    int* __restrict__ expert_ids,
    float* __restrict__ gamma,
    int M,
    float route_scale) {
  const int t = blockIdx.x * 8 + threadIdx.y;
  if (t >= M) return;
  const int lane = threadIdx.x;

  float lp[4], ls[4];
  int lid[4];
  const float* lrow = logits + static_cast<size_t>(t) * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int e = lane + j * 32;
    const float p = 1.0f / (1.0f + expf(-lrow[e]));
    const float b = router_bias ? router_bias[e] : 0.0f;
    lp[j] = p;
    ls[j] = p + b;
    lid[j] = e;
  }

  int sel_id[8];
  float sel_p[8];
  float sum_p = 0.0f;
#pragma unroll 1
  for (int k = 0; k < 8; ++k) {
    float best_s = -1e38f;
    int best_i = -1;
    float best_p = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int cid = lid[j];
      if (cid >= 0) {
        const float cs = ls[j];
        if (best_i < 0 || cs > best_s || (cs == best_s && cid < best_i)) {
          best_s = cs;
          best_i = cid;
          best_p = lp[j];
        }
      }
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      const float os = __shfl_xor_sync(0xffffffffu, best_s, mask);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
      const float op = __shfl_xor_sync(0xffffffffu, best_p, mask);
      if (oi >= 0 && (best_i < 0 || os > best_s || (os == best_s && oi < best_i))) {
        best_s = os;
        best_i = oi;
        best_p = op;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (lid[j] == best_i) lid[j] = -1;
    }
    if (lane == 0) {
      sel_id[k] = best_i;
      sel_p[k] = best_p;
      sum_p += best_p;
    }
  }

  if (lane == 0) {
    const float denom = sum_p + 1e-20f;
    const size_t base = static_cast<size_t>(t) * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int eid = sel_id[k];
      float w = (sel_p[k] / denom) * route_scale;
      if (dn_outer_scale) w *= dn_outer_scale[eid];
      expert_ids[base + k] = eid;
      gamma[base + k] = w;
    }
  }
}

// ---------------------------------------------------------------------------
// 2B. Prefill Counting Sort + Variable BM (32 or 64) Tile Table (E = 128)
//     Grid: 1, Block: 256.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void inkling2_moe_prefill_route_sort(
    const int* __restrict__ expert_ids,
    int* __restrict__ sorted_pairs,
    int* __restrict__ inv_perm,
    int* __restrict__ tile_expert,
    int* __restrict__ tile_begin,
    int* __restrict__ tile_end,
    int* __restrict__ num_tiles,
    int P,
    int bm_shift) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bm = 1 << bm_shift;
  const int bm_mask = bm - 1;

  __shared__ int s_cnt[128];
  __shared__ int s_cursor[128];
  __shared__ int s_wc[4];
  __shared__ int s_wt[4];

  if (tid < 128) s_cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < P; i += 256) {
    const int e = expert_ids[i];
    if (static_cast<unsigned int>(e) < 128u) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();

  int c = 0, nt = 0, scan_c = 0, scan_t = 0;
  if (tid < 128) {
    c = s_cnt[tid];
    nt = (c + bm_mask) >> bm_shift;
    scan_c = c;
    scan_t = nt;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int nc = __shfl_up_sync(0xffffffffu, scan_c, off);
      const int n_t = __shfl_up_sync(0xffffffffu, scan_t, off);
      if (lane >= off) { scan_c += nc; scan_t += n_t; }
    }
    if (lane == 31) {
      s_wc[warp] = scan_c;
      s_wt[warp] = scan_t;
    }
  }
  __syncthreads();

  if (warp == 0) {
    int wc = (lane < 4) ? s_wc[lane] : 0;
    int wt = (lane < 4) ? s_wt[lane] : 0;
    const int nc1 = __shfl_up_sync(0xffffffffu, wc, 1);
    const int nt1 = __shfl_up_sync(0xffffffffu, wt, 1);
    if (lane >= 1) { wc += nc1; wt += nt1; }
    const int nc2 = __shfl_up_sync(0xffffffffu, wc, 2);
    const int nt2 = __shfl_up_sync(0xffffffffu, wt, 2);
    if (lane >= 2) { wc += nc2; wt += nt2; }
    if (lane == 3) num_tiles[0] = wt;
    if (lane < 4) {
      s_wc[lane] = wc - s_wc[lane];
      s_wt[lane] = wt - s_wt[lane];
    }
  }
  __syncthreads();

  if (tid < 128) {
    const int row_start = s_wc[warp] + (scan_c - c);
    s_cursor[tid] = row_start;
    if (c > 0) {
      const int ts = s_wt[warp] + (scan_t - nt);
      for (int k = 0; k < nt; ++k) {
        const int b = row_start + (k << bm_shift);
        const int en = (b + bm < row_start + c) ? (b + bm) : (row_start + c);
        tile_expert[ts + k] = tid;
        tile_begin[ts + k] = b;
        tile_end[ts + k] = en;
      }
    }
  }
  __syncthreads();

  for (int i = tid; i < P; i += 256) {
    const int e = expert_ids[i];
    if (static_cast<unsigned int>(e) < 128u) {
      const int dst = atomicAdd(&s_cursor[e], 1);
      sorted_pairs[dst] = i;
      inv_perm[i] = dst;
    }
  }
}

// ---------------------------------------------------------------------------
// 2C. Prefill Super-Block (GS=256) INT8 Activation Quantizer (for Q5_K gate_up)
//     Grid: ceil((M * 9) / 8), Block: 256 (1 warp = 1 super-block of 256).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void inkling2_moe_prefill_quantize_x_q5k(
    const nv_bfloat16* __restrict__ x,
    signed char* __restrict__ xq,
    float* __restrict__ x_scales,
    short* __restrict__ x_qsums,
    int M) {
  const int total_units = M * 9;
  const int warp_id = static_cast<int>((static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const int lane = threadIdx.x & 31;
  if (warp_id >= total_units) return;

  const uint4 raw = reinterpret_cast<const uint4*>(x)[static_cast<size_t>(warp_id) * 32 + lane];
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float v[8], amax = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(p[k]);
    v[2 * k] = f.x;
    v[2 * k + 1] = f.y;
    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const bool valid = (amax > 0.0f) && isfinite(amax);
  const float inv = valid ? (127.0f / amax) : 0.0f;
  int q[8], lsum = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    q[k] = valid ? static_cast<int>(rintf(v[k] * inv)) : 0;
    lsum += q[k];
  }
  uint2 pk;
  pk.x = (static_cast<unsigned int>(q[0]) & 0xffu) | ((static_cast<unsigned int>(q[1]) & 0xffu) << 8) |
         ((static_cast<unsigned int>(q[2]) & 0xffu) << 16) | ((static_cast<unsigned int>(q[3]) & 0xffu) << 24);
  pk.y = (static_cast<unsigned int>(q[4]) & 0xffu) | ((static_cast<unsigned int>(q[5]) & 0xffu) << 8) |
         ((static_cast<unsigned int>(q[6]) & 0xffu) << 16) | ((static_cast<unsigned int>(q[7]) & 0xffu) << 24);
  reinterpret_cast<uint2*>(xq)[static_cast<size_t>(warp_id) * 32 + lane] = pk;
  if (lane == 0) x_scales[warp_id] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 1, 4);
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 2, 4);
  if ((lane & 3) == 0) x_qsums[static_cast<size_t>(warp_id) * 8 + (lane >> 2)] = static_cast<short>(lsum);
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3A. Prefill Gate + Up + SiTU(4, 25) BM=64, BN=128, BK=64 (NVFP4/SymQ8/AffineQ8/MXFP8)
//     8 warps (256 thr), 2-stage double-buffered cp.async.cg/ca weights + x +
//     async MXFP8 scales, 1-instr HW FP4/FP8 PTX decode, 0 bank conflicts, 0 B spills!
//     NVFP4: 25.25 KB SMEM (2 CTAs/SM <= 64 KB); 8-bit: 33.3..34.3 KB (2 CTAs/SM <= 100 KB).
//     Grid: dim3(9, max_tiles, 1), Block: dim3(32, 8).
// ---------------------------------------------------------------------------
template <int MODE, bool INTERLEAVED>
__global__ __launch_bounds__(256, 2) void inkling2_moe_prefill_gate_up_situ_bm64(
    const unsigned int* __restrict__ gu_w,
    const void* __restrict__ gu_scales,
    const void* __restrict__ gu_biases,
    const float* __restrict__ gu_outer_scale,
    const nv_bfloat16* __restrict__ x,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    nv_bfloat16* __restrict__ mid,
    int sb_is_bf16) {
  const int tile_idx = blockIdx.y;
  if (tile_idx >= num_tiles[0]) return;
  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  constexpr int BN = 128;
  constexpr int W_U4 = (MODE == 0) ? 2 : 4;
  constexpr int ROW_W_U4 = (MODE == 0) ? 72 : 144;
  constexpr int X_SWIZ_SHIFT = (MODE == 0) ? 1 : 2;
  constexpr int X_SWIZ_MASK = (MODE == 0) ? 3 : 1;
  const int base_row = blockIdx.x * BN;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  union SharedGu64 {
    struct {
      alignas(16) uint4 w[2][BN][W_U4];
      alignas(16) uint4 x[2][64][8];
      alignas(16) unsigned int sc[2][BN];
      alignas(16) unsigned int bi[MODE == 2 ? 2 : 1][BN];
      alignas(16) int tok[64];
    } pipe;
    alignas(16) nv_bfloat16 s_out[8][16][16];
  };
  __shared__ SharedGu64 sh;
#if !I2P_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    s_e2m1_lut[tid] = i2p_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  } else {
    (void)s_e2m1_lut;
  }
#endif

  if (tid < 64) sh.pipe.tok[tid] = (tid < cnt) ? (sorted_pairs[begin + tid] >> 3) : -1;
  __syncthreads();

  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  auto row_idx_of = [&](int r, bool is_up) -> size_t {
    return exp_base + (INTERLEAVED ? (2 * (base_row + r) + (is_up ? 1 : 0)) : ((is_up ? I : 0) + base_row + r));
  };

  auto load_w = [&](int wst, bool is_up, int step) {
    if constexpr (MODE == 0) {
      const int r = tid >> 1;
      const int c = tid & 1;
      const size_t ri = row_idx_of(r, is_up);
      ptx_cp_async16_cg(&sh.pipe.w[wst][r][c ^ ((r >> 2) & 1)], w_u4 + ri * ROW_W_U4 + step * 2 + c);
      if (tid < BN) {
        const size_t sr = row_idx_of(tid, is_up);
        ptx_cp_async4_ca(&sh.pipe.sc[wst][tid],
                         reinterpret_cast<const unsigned int*>(gu_scales) + sr * 36 + step);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 2; ++iter) {
        const int idx = tid + iter * 256;
        const int r = idx >> 2;
        const int c = idx & 3;
        const size_t ri = row_idx_of(r, is_up);
        ptx_cp_async16_cg(&sh.pipe.w[wst][r][c ^ (r & 3)], w_u4 + ri * ROW_W_U4 + step * 4 + c);
      }
      if (tid < BN) {
        const size_t sr = row_idx_of(tid, is_up);
        if constexpr (MODE == 4) {
          if ((step & 1) == 0) {
            ptx_cp_async4_ca(&sh.pipe.sc[wst][tid],
                             reinterpret_cast<const unsigned int*>(gu_scales) + sr * 18 + (step >> 1));
          }
        } else {
          ptx_cp_async4_ca(&sh.pipe.sc[wst][tid],
                           reinterpret_cast<const unsigned int*>(gu_scales) + sr * 36 + step);
        }
      } else if constexpr (MODE == 2) {
        const int r = tid - BN;
        const size_t sr = row_idx_of(r, is_up);
        ptx_cp_async4_ca(&sh.pipe.bi[wst][r],
                         reinterpret_cast<const unsigned int*>(gu_biases) + sr * 36 + step);
      }
    }
  };

  auto load_x = [&](int xst, int step) {
    const int total_x = load_x_rows * 8;
    for (int idx = tid; idx < total_x; idx += 256) {
      const int r = idx >> 3;
      const int c = idx & 7;
      const int tok = sh.pipe.tok[r];
      uint4* dst = &sh.pipe.x[xst][r][c ^ ((r & X_SWIZ_MASK) << X_SWIZ_SHIFT)];
      if (tok >= 0) ptx_cp_async16_ca(dst, x_u4 + static_cast<size_t>(tok) * 288 + step * 8 + c);
      else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
  };

  load_w(0, false, 0);
  load_x(0, 0);
  ptx_cp_commit();
  ptx_cp_wait<0>();
  __syncthreads();

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;
  float g_tot[8][4];
  float u_tot[8][4];
#pragma unroll
  for (int ms = 0; ms < 8; ++ms) {
    g_tot[ms][0] = g_tot[ms][1] = g_tot[ms][2] = g_tot[ms][3] = 0.0f;
    u_tot[ms][0] = u_tot[ms][1] = u_tot[ms][2] = u_tot[ms][3] = 0.0f;
  }

  auto compute_mat = [&](int wst, int xst, int step, float tot[8][4]) {
    if constexpr (MODE == 0) {
      const unsigned int sc_w0 = sh.pipe.sc[wst][r0];
      const unsigned int sc_w1 = sh.pipe.sc[wst][r1];
      const int sw0 = (r0 >> 2) & 1;
      const int sw1 = (r1 >> 2) & 1;
#pragma unroll 1
      for (int lg2 = 0; lg2 < 2; ++lg2) {
        const float2 sf0 = i2p_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w0 >> (lg2 * 16)));
        const float2 sf1 = i2p_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w1 >> (lg2 * 16)));
        const unsigned short* wp0 = reinterpret_cast<const unsigned short*>(&sh.pipe.w[wst][r0][lg2 ^ sw0]);
        const unsigned short* wp1 = reinterpret_cast<const unsigned short*>(&sh.pipe.w[wst][r1][lg2 ^ sw1]);
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int lg = lg2 * 2 + sub;
          const unsigned short w0 = wp0[sub * 4 + t];
          const unsigned short w1 = wp1[sub * 4 + t];
#if I2P_HAS_HW_FP4_CVT
          const unsigned int a0 = i2p_fast_e2m1x2_to_bf16x2(w0 & 0xFFu);
          const unsigned int a2 = i2p_fast_e2m1x2_to_bf16x2(w0 >> 8);
          const unsigned int a1 = i2p_fast_e2m1x2_to_bf16x2(w1 & 0xFFu);
          const unsigned int a3 = i2p_fast_e2m1x2_to_bf16x2(w1 >> 8);
#else
          const unsigned int a0 = s_e2m1_lut[w0 & 0xFFu];
          const unsigned int a2 = s_e2m1_lut[w0 >> 8];
          const unsigned int a1 = s_e2m1_lut[w1 & 0xFFu];
          const unsigned int a3 = s_e2m1_lut[w1 >> 8];
#endif
          const float sc0 = (sub == 0) ? sf0.x : sf0.y;
          const float sc1 = (sub == 0) ? sf1.x : sf1.y;
          const int x_col = (lg * 2 + (t >> 1)) ^ ((g & 3) << 1);
          const int x_hf = t & 1;
#pragma unroll
          for (int ms = 0; ms < 8; ++ms) {
            if (ms < n_slices) {
              const int rx = ms * 8 + g;
              const uint2 xv = reinterpret_cast<const uint2*>(
                  &sh.pipe.x[xst][rx][x_col])[x_hf];
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
    } else {
      const unsigned int sc_w0 = sh.pipe.sc[wst][r0];
      const unsigned int sc_w1 = sh.pipe.sc[wst][r1];
      [[maybe_unused]] const unsigned int bi_w0 = (MODE == 2) ? sh.pipe.bi[wst][r0] : 0u;
      [[maybe_unused]] const unsigned int bi_w1 = (MODE == 2) ? sh.pipe.bi[wst][r1] : 0u;
      [[maybe_unused]] const int sc_half_sh = (step & 1) * 16;
#pragma unroll
      for (int lg = 0; lg < 2; ++lg) {
        const int col_u4 = lg * 2 + (t >> 1);
        const int hf = t & 1;
        const uint2 w0 = reinterpret_cast<const uint2*>(
            &sh.pipe.w[wst][r0][col_u4 ^ (r0 & 3)])[hf];
        const uint2 w1 = reinterpret_cast<const uint2*>(
            &sh.pipe.w[wst][r1][col_u4 ^ (r1 & 3)])[hf];
        unsigned int wb0[4], wb1[4];
        i2p_unpack8_to_bf16x2<MODE>(w0, wb0);
        i2p_unpack8_to_bf16x2<MODE>(w1, wb1);

        float sc0, sc1;
        [[maybe_unused]] float bi0 = 0.0f;
        [[maybe_unused]] float bi1 = 0.0f;
        if constexpr (MODE == 4) {
          sc0 = i2p_decode_ue8m0((sc_w0 >> (sc_half_sh + lg * 8)) & 0xFFu);
          sc1 = i2p_decode_ue8m0((sc_w1 >> (sc_half_sh + lg * 8)) & 0xFFu);
        } else {
          sc0 = i2p_read_f16_or_bf16(static_cast<unsigned short>(sc_w0 >> (lg * 16)), sb_is_bf16);
          sc1 = i2p_read_f16_or_bf16(static_cast<unsigned short>(sc_w1 >> (lg * 16)), sb_is_bf16);
          if constexpr (MODE == 2) {
            bi0 = i2p_read_f16_or_bf16(static_cast<unsigned short>(bi_w0 >> (lg * 16)), sb_is_bf16);
            bi1 = i2p_read_f16_or_bf16(static_cast<unsigned short>(bi_w1 >> (lg * 16)), sb_is_bf16);
          }
        }
        const int x_col = (lg * 4 + t) ^ ((g & 1) << 2);
#pragma unroll
        for (int ms = 0; ms < 8; ++ms) {
          if (ms < n_slices) {
            const int rx = ms * 8 + g;
            const uint4 xv = sh.pipe.x[xst][rx][x_col];
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv.x, xv.y);
            ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv.z, xv.w);
            if constexpr (MODE == 2) {
              float my_s = i2p_sum_bf16x8(xv);
              my_s += __shfl_xor_sync(0xffffffffu, my_s, 1);
              my_s += __shfl_xor_sync(0xffffffffu, my_s, 2);
              const float xs0 = __shfl_sync(0xffffffffu, my_s, (2 * t) * 4);
              const float xs1 = __shfl_sync(0xffffffffu, my_s, (2 * t + 1) * 4);
              tot[ms][0] = fmaf(sc0, d0, fmaf(bi0, xs0, tot[ms][0]));
              tot[ms][1] = fmaf(sc0, d1, fmaf(bi0, xs1, tot[ms][1]));
              tot[ms][2] = fmaf(sc1, d2, fmaf(bi1, xs0, tot[ms][2]));
              tot[ms][3] = fmaf(sc1, d3, fmaf(bi1, xs1, tot[ms][3]));
            } else {
              tot[ms][0] = fmaf(sc0, d0, tot[ms][0]);
              tot[ms][1] = fmaf(sc0, d1, tot[ms][1]);
              tot[ms][2] = fmaf(sc1, d2, tot[ms][2]);
              tot[ms][3] = fmaf(sc1, d3, tot[ms][3]);
            }
          }
        }
      }
    }
  };

#pragma unroll 1
  for (int step = 0; step < 36; ++step) {
    const int cur_x = step & 1;
    const int nxt_x = cur_x ^ 1;
    const bool has_nxt = (step + 1 < 36);

    load_w(1, true, step);
    ptx_cp_commit();
    if (has_nxt) {
      load_x(nxt_x, step + 1);
      ptx_cp_commit();
    }

    compute_mat(0, cur_x, step, g_tot);

    if (has_nxt) ptx_cp_wait<1>();
    else ptx_cp_wait<0>();
    __syncthreads();

    if (has_nxt) {
      load_w(0, false, step + 1);
      ptx_cp_commit();
    }

    compute_mat(1, cur_x, step, u_tot);

    if (has_nxt) {
      ptx_cp_wait<0>();
      __syncthreads();
    }
  }
  __syncthreads();

  const int w_col0 = base_row + warp * 16;
  const int n_passes = (cnt + 15) >> 4;
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      sh.s_out[warp][2 * t][g]         = i2p_situ_epilogue(g_tot[s0][0], u_tot[s0][0], outer_s);
      sh.s_out[warp][2 * t + 1][g]     = i2p_situ_epilogue(g_tot[s0][1], u_tot[s0][1], outer_s);
      sh.s_out[warp][2 * t][g + 8]     = i2p_situ_epilogue(g_tot[s0][2], u_tot[s0][2], outer_s);
      sh.s_out[warp][2 * t + 1][g + 8] = i2p_situ_epilogue(g_tot[s0][3], u_tot[s0][3], outer_s);
      if (s1 < n_slices) {
        sh.s_out[warp][8 + 2 * t][g]         = i2p_situ_epilogue(g_tot[s1][0], u_tot[s1][0], outer_s);
        sh.s_out[warp][8 + 2 * t + 1][g]     = i2p_situ_epilogue(g_tot[s1][1], u_tot[s1][1], outer_s);
        sh.s_out[warp][8 + 2 * t][g + 8]     = i2p_situ_epilogue(g_tot[s1][2], u_tot[s1][2], outer_s);
        sh.s_out[warp][8 + 2 * t + 1][g + 8] = i2p_situ_epilogue(g_tot[s1][3], u_tot[s1][3], outer_s);
      }
      __syncwarp();
      const int r_in = lane >> 1;
      const int half = lane & 1;
      const int r_loc = pass * 16 + r_in;
      if (r_loc < cnt) {
        ptx_st_cs_u4(
            reinterpret_cast<uint4*>(mid + static_cast<size_t>(begin + r_loc) * I + w_col0) + half,
            reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
      }
      __syncwarp();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3B. Prefill Q5_K Gate + Up + SiTU(4, 25) BM=32, BN=64, BK=256 (UNITS = 9)
//     INT8 Tensor Core mma.m16n8k32.s8, coalesced uint4 BF16 mid output, 0 B spills!
//     Grid: dim3(18, max_tiles, 1), Block: dim3(32, 4).
// ---------------------------------------------------------------------------
template <bool INTERLEAVED>
__global__ __launch_bounds__(128, 2) void inkling2_moe_prefill_gate_up_situ_q5k_bm32(
    const unsigned int* __restrict__ gu_w,
    const unsigned char* __restrict__ gu_scales,
    const unsigned short* __restrict__ gu_biases,
    const float* __restrict__ gu_outer_scale,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const short* __restrict__ x_qsums,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    nv_bfloat16* __restrict__ mid) {
  const int tile_idx = blockIdx.y;
  if (tile_idx >= num_tiles[0]) return;
  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  constexpr int UNITS = 9;
  constexpr int ROW_V4 = 90;
  const int base_row = blockIdx.x * 64;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  struct WeightStage {
    alignas(16) uint4 w[64][11];
    alignas(16) uint4 sc[64];
    alignas(16) unsigned int bi[64];
  };
  union SharedQ5k32 {
    struct {
      WeightStage wst[2];
      alignas(16) uint4 s_xq[32][16];
      alignas(16) uint4 s_xqs[32];
      alignas(16) float s_xsc[32];
      alignas(16) int s_tok[32];
    } pipe;
    alignas(16) nv_bfloat16 s_out[4][16][16];
  };
  __shared__ SharedQ5k32 sh;

  if (tid < 32) sh.pipe.s_tok[tid] = (tid < cnt) ? (sorted_pairs[begin + tid] >> 3) : -1;
  __syncthreads();

  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
  const uint4* sc_u4 = reinterpret_cast<const uint4*>(gu_scales);
  const unsigned int* bi_u32 = reinterpret_cast<const unsigned int*>(gu_biases);
  const uint4* xq_u4 = reinterpret_cast<const uint4*>(xq);
  const uint4* xqs_u4 = reinterpret_cast<const uint4*>(x_qsums);
  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  auto row_idx_of = [&](int r, bool is_up) -> size_t {
    return exp_base + (INTERLEAVED ? (2 * (base_row + r) + (is_up ? 1 : 0)) : ((is_up ? I : 0) + base_row + r));
  };

  auto load_w = [&](int st, bool is_up, int unit) {
#pragma unroll
    for (int iter = 0; iter < 5; ++iter) {
      const int idx = tid + iter * 128;
      const int r = idx / 10;
      const int c = idx - r * 10;
      const size_t ri = row_idx_of(r, is_up);
      ptx_cp_async16_cg(&sh.pipe.wst[st].w[r][c], w_u4 + ri * ROW_V4 + unit * 10 + c);
    }
    if (tid < 64) {
      const size_t sr = row_idx_of(tid, is_up);
      ptx_cp_async16_cg(&sh.pipe.wst[st].sc[tid], sc_u4 + sr * UNITS + unit);
    } else {
      const int r = tid - 64;
      const size_t sr = row_idx_of(r, is_up);
      sh.pipe.wst[st].bi[r] = bi_u32[sr * UNITS + unit];
    }
    ptx_cp_commit();
  };

  auto load_x_unit = [&](int unit) {
    const int total_u4 = load_x_rows * 16;
    for (int idx = tid; idx < total_u4; idx += 128) {
      const int r = idx >> 4;
      const int c = idx & 15;
      const int tok = sh.pipe.s_tok[r];
      uint4* dst = &sh.pipe.s_xq[r][c ^ (r & 7)];
      if (tok >= 0) ptx_cp_async16_ca(dst, xq_u4 + (static_cast<size_t>(tok) * UNITS + unit) * 16 + c);
      else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < load_x_rows) {
      const int tok = sh.pipe.s_tok[tid];
      if (tok >= 0) {
        sh.pipe.s_xsc[tid] = x_scales[static_cast<size_t>(tok) * UNITS + unit];
        sh.pipe.s_xqs[tid] = xqs_u4[static_cast<size_t>(tok) * UNITS + unit];
      } else {
        sh.pipe.s_xsc[tid] = 0.0f;
        sh.pipe.s_xqs[tid] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    ptx_cp_commit();
  };

  load_w(0, false, 0);
  load_x_unit(0);
  ptx_cp_wait<0>();
  __syncthreads();

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;
  float g_tot[4][4] = {{0.0f}};
  float u_tot[4][4] = {{0.0f}};

  auto compute_stage = [&](int st, float tot[4][4]) {
    const uint4 sc0 = sh.pipe.wst[st].sc[r0];
    const uint4 sc1 = sh.pipe.wst[st].sc[r1];
    int idot[4][4] = {{0}};
#pragma unroll 1
    for (int s = 0; s < 8; ++s) {
      unsigned int a0, a1, a2, a3;
      i2p_load_pair_q5k(reinterpret_cast<const unsigned int*>(sh.pipe.wst[st].w[r0]) + s * 5, t, a0, a2);
      i2p_load_pair_q5k(reinterpret_cast<const unsigned int*>(sh.pipe.wst[st].w[r1]) + s * 5, t, a1, a3);
      const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
      const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
      const int shft = (s & 1) * 16;
      const int ws0 = static_cast<int>((sw0 >> shft) & 0xffu);
      const int ws1 = static_cast<int>((sw1 >> shft) & 0xffu);
      const int col0 = (2 * s) ^ g;
      const int col1 = (2 * s + 1) ^ g;
#pragma unroll
      for (int ms = 0; ms < 4; ++ms) {
        if (ms < n_slices) {
          const int rx = ms * 8 + g;
          const unsigned int b0 = reinterpret_cast<const unsigned int*>(&sh.pipe.s_xq[rx][col0])[t];
          const unsigned int b1 = reinterpret_cast<const unsigned int*>(&sh.pipe.s_xq[rx][col1])[t];
          int c0, c1, c2, c3;
          ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
          idot[ms][0] += c0 * ws0;
          idot[ms][1] += c1 * ws0;
          idot[ms][2] += c2 * ws1;
          idot[ms][3] += c3 * ws1;
        }
      }
    }
    const unsigned int b0p = sh.pipe.wst[st].bi[r0];
    const unsigned int b1p = sh.pipe.wst[st].bi[r1];
    const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(&b0p));
    const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(&b1p));
#pragma unroll
    for (int ms = 0; ms < 4; ++ms) {
      if (ms < n_slices) {
        const int c0_loc = ms * 8 + 2 * t;
        const int c1_loc = c0_loc + 1;
        const float ad0 = sh.pipe.s_xsc[c0_loc];
        const float ad1 = sh.pipe.s_xsc[c1_loc];
        const uint4 qs0 = sh.pipe.s_xqs[c0_loc];
        const uint4 qs1 = sh.pipe.s_xqs[c1_loc];
        tot[ms][0] = fmaf(ad0, fmaf(dm0.x, float(idot[ms][0]), -dm0.y * float(i2p_min_dot8(sc0, qs0))), tot[ms][0]);
        tot[ms][1] = fmaf(ad1, fmaf(dm0.x, float(idot[ms][1]), -dm0.y * float(i2p_min_dot8(sc0, qs1))), tot[ms][1]);
        tot[ms][2] = fmaf(ad0, fmaf(dm1.x, float(idot[ms][2]), -dm1.y * float(i2p_min_dot8(sc1, qs0))), tot[ms][2]);
        tot[ms][3] = fmaf(ad1, fmaf(dm1.x, float(idot[ms][3]), -dm1.y * float(i2p_min_dot8(sc1, qs1))), tot[ms][3]);
      }
    }
  };

#pragma unroll 1
  for (int unit = 0; unit < UNITS; ++unit) {
    load_w(1, true, unit);
    compute_stage(0, g_tot);
    ptx_cp_wait<0>();
    __syncthreads();
    if (unit + 1 < UNITS) load_w(0, false, unit + 1);
    compute_stage(1, u_tot);
    __syncthreads();
    if (unit + 1 < UNITS) {
      load_x_unit(unit + 1);
      ptx_cp_wait<0>();
      __syncthreads();
    }
  }

  const int w_col0 = base_row + warp * 16;
  const int n_passes = (cnt > 16) ? 2 : 1;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      sh.s_out[warp][2 * t][g]         = i2p_situ_epilogue(g_tot[s0][0], u_tot[s0][0], outer_s);
      sh.s_out[warp][2 * t + 1][g]     = i2p_situ_epilogue(g_tot[s0][1], u_tot[s0][1], outer_s);
      sh.s_out[warp][2 * t][g + 8]     = i2p_situ_epilogue(g_tot[s0][2], u_tot[s0][2], outer_s);
      sh.s_out[warp][2 * t + 1][g + 8] = i2p_situ_epilogue(g_tot[s0][3], u_tot[s0][3], outer_s);
      if (s1 < n_slices) {
        sh.s_out[warp][8 + 2 * t][g]         = i2p_situ_epilogue(g_tot[s1][0], u_tot[s1][0], outer_s);
        sh.s_out[warp][8 + 2 * t + 1][g]     = i2p_situ_epilogue(g_tot[s1][1], u_tot[s1][1], outer_s);
        sh.s_out[warp][8 + 2 * t][g + 8]     = i2p_situ_epilogue(g_tot[s1][2], u_tot[s1][2], outer_s);
        sh.s_out[warp][8 + 2 * t + 1][g + 8] = i2p_situ_epilogue(g_tot[s1][3], u_tot[s1][3], outer_s);
      }
      __syncwarp();
      const int r_in = lane >> 1;
      const int half = lane & 1;
      const int r_loc = pass * 16 + r_in;
      if (r_loc < cnt) {
        ptx_st_cs_u4(
            reinterpret_cast<uint4*>(mid + static_cast<size_t>(begin + r_loc) * I + w_col0) + half,
            reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
      }
      __syncwarp();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 4. Prefill Down Projection BM=64, BN=128, BK=64 (NVFP4/SymQ8/AffineQ8/MXFP8)
//    K = 1152 (18 steps), N = 2304 (18 N-tiles), 8 warps (256 thr), 2-stage
//    cp.async + async MXFP8 scales + HW FP4/FP8 PTX decode + BF16 MMA, 0 spills!
//    Handles both BM=64 and BM=32 tiles (via dynamic n_slices / n_passes).
//    Grid: dim3(18, max_tiles, 1), Block: dim3(32, 8).
// ---------------------------------------------------------------------------
template <int MODE>
__global__ __launch_bounds__(256, MODE == 0 ? 3 : 2) void inkling2_moe_prefill_down_bm64(
    const unsigned int* __restrict__ dn_w,
    const void* __restrict__ dn_scales,
    const void* __restrict__ dn_biases,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    nv_bfloat16* __restrict__ down_out,
    int sb_is_bf16) {
  const int tile_idx = blockIdx.y;
  if (tile_idx >= num_tiles[0]) return;
  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;

  constexpr int N = 2304;
  constexpr int BN = 128;
  constexpr int W_U4 = (MODE == 0) ? 2 : 4;
  constexpr int ROW_W_U4 = (MODE == 0) ? 36 : 72;
  constexpr int X_SWIZ_SHIFT = (MODE == 0) ? 1 : 2;
  constexpr int X_SWIZ_MASK = (MODE == 0) ? 3 : 1;
  const int base_row = blockIdx.x * BN;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  struct StageDn64 {
    alignas(16) uint4 x[64][8];
    alignas(16) uint4 w[BN][W_U4];
    alignas(16) unsigned int sc[BN];
    alignas(16) unsigned int bi[MODE == 2 ? BN : 1];
  };
  union SharedDn64 {
    StageDn64 st[2];
    alignas(16) nv_bfloat16 s_out[8][16][16];
  };
  __shared__ SharedDn64 sh;
#if !I2P_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    s_e2m1_lut[tid] = i2p_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
  } else {
    (void)s_e2m1_lut;
  }
#endif

  const size_t exp_row0 = static_cast<size_t>(expert) * N + base_row;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(dn_w) + exp_row0 * ROW_W_U4;
  const uint4* mid_u4 = reinterpret_cast<const uint4*>(mid);
  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_stage = [&](int s_idx, int step) {
    if constexpr (MODE == 0) {
      const int r = tid >> 1;
      const int c = tid & 1;
      ptx_cp_async16_cg(&sh.st[s_idx].w[r][c ^ ((r >> 2) & 1)],
                        w_u4 + static_cast<size_t>(r) * ROW_W_U4 + step * 2 + c);
      if (tid < BN) {
        ptx_cp_async4_ca(&sh.st[s_idx].sc[tid],
                         reinterpret_cast<const unsigned int*>(dn_scales) + (exp_row0 + tid) * 18 + step);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 2; ++iter) {
        const int idx = tid + iter * 256;
        const int r = idx >> 2;
        const int c = idx & 3;
        ptx_cp_async16_cg(&sh.st[s_idx].w[r][c ^ (r & 3)],
                          w_u4 + static_cast<size_t>(r) * ROW_W_U4 + step * 4 + c);
      }
      if (tid < BN) {
        if constexpr (MODE == 4) {
          if ((step & 1) == 0) {
            ptx_cp_async4_ca(&sh.st[(step >> 1) & 1].sc[tid],
                             reinterpret_cast<const unsigned int*>(dn_scales) + (exp_row0 + tid) * 9 + (step >> 1));
          }
        } else {
          ptx_cp_async4_ca(&sh.st[s_idx].sc[tid],
                           reinterpret_cast<const unsigned int*>(dn_scales) + (exp_row0 + tid) * 18 + step);
        }
      } else if constexpr (MODE == 2) {
        const int r = tid - BN;
        ptx_cp_async4_ca(&sh.st[s_idx].bi[r],
                         reinterpret_cast<const unsigned int*>(dn_biases) + (exp_row0 + r) * 18 + step);
      }
    }
    const int total_x = load_x_rows * 8;
    for (int idx = tid; idx < total_x; idx += 256) {
      const int r = idx >> 3;
      const int c = idx & 7;
      uint4* dst = &sh.st[s_idx].x[r][c ^ ((r & X_SWIZ_MASK) << X_SWIZ_SHIFT)];
      if (r < cnt) ptx_cp_async16_ca(dst, mid_u4 + static_cast<size_t>(begin + r) * 144 + step * 8 + c);
      else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    ptx_cp_commit();
  };

  load_stage(0, 0);

  float tot[8][4];
#pragma unroll
  for (int ms = 0; ms < 8; ++ms) {
    tot[ms][0] = tot[ms][1] = tot[ms][2] = tot[ms][3] = 0.0f;
  }
  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;

#pragma unroll 1
  for (int step = 0; step < 18; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < 18) {
      load_stage(nxt, step + 1);
      ptx_cp_wait<1>();
    } else {
      ptx_cp_wait<0>();
    }
    __syncthreads();

    if constexpr (MODE == 0) {
      const unsigned int sc_w0 = sh.st[cur].sc[r0];
      const unsigned int sc_w1 = sh.st[cur].sc[r1];
      const int sw0 = (r0 >> 2) & 1;
      const int sw1 = (r1 >> 2) & 1;
#pragma unroll 1
      for (int lg2 = 0; lg2 < 2; ++lg2) {
        const float2 sf0 = i2p_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w0 >> (lg2 * 16)));
        const float2 sf1 = i2p_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w1 >> (lg2 * 16)));
        const unsigned short* wp0 = reinterpret_cast<const unsigned short*>(&sh.st[cur].w[r0][lg2 ^ sw0]);
        const unsigned short* wp1 = reinterpret_cast<const unsigned short*>(&sh.st[cur].w[r1][lg2 ^ sw1]);
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int lg = lg2 * 2 + sub;
          const unsigned short w0 = wp0[sub * 4 + t];
          const unsigned short w1 = wp1[sub * 4 + t];
#if I2P_HAS_HW_FP4_CVT
          const unsigned int a0 = i2p_fast_e2m1x2_to_bf16x2(w0 & 0xFFu);
          const unsigned int a2 = i2p_fast_e2m1x2_to_bf16x2(w0 >> 8);
          const unsigned int a1 = i2p_fast_e2m1x2_to_bf16x2(w1 & 0xFFu);
          const unsigned int a3 = i2p_fast_e2m1x2_to_bf16x2(w1 >> 8);
#else
          const unsigned int a0 = s_e2m1_lut[w0 & 0xFFu];
          const unsigned int a2 = s_e2m1_lut[w0 >> 8];
          const unsigned int a1 = s_e2m1_lut[w1 & 0xFFu];
          const unsigned int a3 = s_e2m1_lut[w1 >> 8];
#endif
          const float sc0 = (sub == 0) ? sf0.x : sf0.y;
          const float sc1 = (sub == 0) ? sf1.x : sf1.y;
          const int x_col = (lg * 2 + (t >> 1)) ^ ((g & 3) << 1);
          const int x_hf = t & 1;
#pragma unroll
          for (int ms = 0; ms < 8; ++ms) {
            if (ms < n_slices) {
              const int rx = ms * 8 + g;
              const uint2 xv = reinterpret_cast<const uint2*>(
                  &sh.st[cur].x[rx][x_col])[x_hf];
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
    } else {
      const int sc_st = (MODE == 4) ? ((step >> 1) & 1) : cur;
      const unsigned int sc_w0 = sh.st[sc_st].sc[r0];
      const unsigned int sc_w1 = sh.st[sc_st].sc[r1];
      [[maybe_unused]] const unsigned int bi_w0 = (MODE == 2) ? sh.st[cur].bi[r0] : 0u;
      [[maybe_unused]] const unsigned int bi_w1 = (MODE == 2) ? sh.st[cur].bi[r1] : 0u;
      [[maybe_unused]] const int sc_half_sh = (step & 1) * 16;
#pragma unroll
      for (int lg = 0; lg < 2; ++lg) {
        const int col_u4 = lg * 2 + (t >> 1);
        const int hf = t & 1;
        const uint2 w0 = reinterpret_cast<const uint2*>(
            &sh.st[cur].w[r0][col_u4 ^ (r0 & 3)])[hf];
        const uint2 w1 = reinterpret_cast<const uint2*>(
            &sh.st[cur].w[r1][col_u4 ^ (r1 & 3)])[hf];
        unsigned int wb0[4], wb1[4];
        i2p_unpack8_to_bf16x2<MODE>(w0, wb0);
        i2p_unpack8_to_bf16x2<MODE>(w1, wb1);

        float sc0, sc1;
        [[maybe_unused]] float bi0 = 0.0f;
        [[maybe_unused]] float bi1 = 0.0f;
        if constexpr (MODE == 4) {
          sc0 = i2p_decode_ue8m0((sc_w0 >> (sc_half_sh + lg * 8)) & 0xFFu);
          sc1 = i2p_decode_ue8m0((sc_w1 >> (sc_half_sh + lg * 8)) & 0xFFu);
        } else {
          sc0 = i2p_read_f16_or_bf16(static_cast<unsigned short>(sc_w0 >> (lg * 16)), sb_is_bf16);
          sc1 = i2p_read_f16_or_bf16(static_cast<unsigned short>(sc_w1 >> (lg * 16)), sb_is_bf16);
          if constexpr (MODE == 2) {
            bi0 = i2p_read_f16_or_bf16(static_cast<unsigned short>(bi_w0 >> (lg * 16)), sb_is_bf16);
            bi1 = i2p_read_f16_or_bf16(static_cast<unsigned short>(bi_w1 >> (lg * 16)), sb_is_bf16);
          }
        }
        const int x_col = (lg * 4 + t) ^ ((g & 1) << 2);
#pragma unroll
        for (int ms = 0; ms < 8; ++ms) {
          if (ms < n_slices) {
            const int rx = ms * 8 + g;
            const uint4 xv = sh.st[cur].x[rx][x_col];
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv.x, xv.y);
            ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv.z, xv.w);
            if constexpr (MODE == 2) {
              float my_s = i2p_sum_bf16x8(xv);
              my_s += __shfl_xor_sync(0xffffffffu, my_s, 1);
              my_s += __shfl_xor_sync(0xffffffffu, my_s, 2);
              const float xs0 = __shfl_sync(0xffffffffu, my_s, (2 * t) * 4);
              const float xs1 = __shfl_sync(0xffffffffu, my_s, (2 * t + 1) * 4);
              tot[ms][0] = fmaf(sc0, d0, fmaf(bi0, xs0, tot[ms][0]));
              tot[ms][1] = fmaf(sc0, d1, fmaf(bi0, xs1, tot[ms][1]));
              tot[ms][2] = fmaf(sc1, d2, fmaf(bi1, xs0, tot[ms][2]));
              tot[ms][3] = fmaf(sc1, d3, fmaf(bi1, xs1, tot[ms][3]));
            } else {
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
  }

  const int w_col0 = base_row + warp * 16;
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
      const int r_in = lane >> 1;
      const int half = lane & 1;
      const int r_loc = pass * 16 + r_in;
      if (r_loc < cnt) {
        ptx_st_cs_u4(
            reinterpret_cast<uint4*>(down_out + static_cast<size_t>(begin + r_loc) * N + w_col0) + half,
            reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
      }
      __syncwarp();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 5A. Prefill Vectorized Unsort + Weighted Combine (288 threads = 2304 cols exact!)
//     Grid: dim3(1, M, 1), Block: 288.
// ---------------------------------------------------------------------------
template <bool EMIT_COPY>
__global__ __launch_bounds__(288, 2) void inkling2_moe_prefill_combine(
    const nv_bfloat16* __restrict__ down_out,
    const float* __restrict__ gamma,
    const int* __restrict__ inv_perm,
    nv_bfloat16* __restrict__ out_a,
    nv_bfloat16* __restrict__ out_b,
    int M) {
  const int tok = blockIdx.y;
  const int col8 = threadIdx.x;
  if (tok >= M || col8 >= 288) return;

  __shared__ int s_srow[8];
  __shared__ float s_sc[8];
  if (col8 < 8) {
    const int bp = tok * 8 + col8;
    s_srow[col8] = inv_perm[bp];
    s_sc[col8] = gamma[bp];
  }
  __syncthreads();

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint4* dn_u4 = reinterpret_cast<const uint4*>(down_out);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float sc = s_sc[k];
    const uint4 raw = dn_u4[static_cast<size_t>(s_srow[k]) * 288 + col8];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      acc[2 * j] = fmaf(f.x, sc, acc[2 * j]);
      acc[2 * j + 1] = fmaf(f.y, sc, acc[2 * j + 1]);
    }
  }
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int j = 0; j < 4; ++j) op[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  reinterpret_cast<uint4*>(out_a)[static_cast<size_t>(tok) * 288 + col8] = ov;
  if constexpr (EMIT_COPY) {
    reinterpret_cast<uint4*>(out_b)[static_cast<size_t>(tok) * 288 + col8] = ov;
  }
}

// ---------------------------------------------------------------------------
// 5B. Prefill Static ShortConv(k=4, linear OR tree) + Optional Residual + Hist
//     Grid: dim3(18, min(M, 64), 1), Block: 128 (18 * 128 = 2304 exact!).
// ---------------------------------------------------------------------------
template <bool HAS_RESIDUAL, bool HAS_TAPS>
__global__ __launch_bounds__(128, 4) void inkling2_moe_prefill_sconv_res(
    const nv_bfloat16* __restrict__ moe_out,
    const float* __restrict__ post_conv_w,
    const nv_bfloat16* __restrict__ hist_in,
    const nv_bfloat16* __restrict__ residual,
    const int* __restrict__ conv_taps,
    nv_bfloat16* __restrict__ out_y,
    nv_bfloat16* __restrict__ out_hist,
    int M,
    int update_hist) {
  constexpr int D = 2304;
  constexpr float kScale = 0.44721359549995793f;
  const int col = blockIdx.x * 128 + threadIdx.x;
  if (col >= D) return;

  const float4 wf = reinterpret_cast<const float4*>(post_conv_w)[col];
  for (int t = blockIdx.y; t < M; t += gridDim.y) {
    float x3, x2, x1, x0;
    if constexpr (HAS_TAPS) {
      auto load_tap = [&](int idx) -> float {
        if (idx >= 0) {
          return (idx < M) ? __bfloat162float(moe_out[static_cast<size_t>(idx) * D + col]) : 0.0f;
        }
        const int hr = idx + 3;
        return (hr >= 0 && hr < 3) ? __bfloat162float(hist_in[static_cast<size_t>(hr) * D + col]) : 0.0f;
      };
      const int4 taps = reinterpret_cast<const int4*>(conv_taps)[t];
      x3 = load_tap(taps.x);
      x2 = load_tap(taps.y);
      x1 = load_tap(taps.z);
      x0 = load_tap(taps.w);
    } else {
      auto load_full = [&](int row) -> float {
        if (row < 3) return __bfloat162float(hist_in[static_cast<size_t>(row) * D + col]);
        return __bfloat162float(moe_out[static_cast<size_t>(row - 3) * D + col]);
      };
      x3 = load_full(t + 0);
      x2 = load_full(t + 1);
      x1 = load_full(t + 2);
      x0 = load_full(t + 3);
    }
    const float conv_acc = __fadd_rn(
        __fadd_rn(__fadd_rn(__fadd_rn(x0, __fmul_rn(wf.x, x3)), __fmul_rn(wf.y, x2)),
                  __fmul_rn(wf.z, x1)),
        __fmul_rn(wf.w, x0));
    const nv_bfloat16 conv_bf = __float2bfloat16_rn(__fmul_rn(conv_acc, kScale));
    const size_t off = static_cast<size_t>(t) * D + col;
    if constexpr (HAS_RESIDUAL) {
      out_y[off] = __float2bfloat16_rn(__bfloat162float(residual[off]) + __bfloat162float(conv_bf));
    } else {
      out_y[off] = conv_bf;
    }
  }

  if (out_hist != nullptr && blockIdx.y == 0) {
#pragma unroll
    for (int r = 0; r < 3; ++r) {
      const int src = (update_hist ? M : 0) + r;
      out_hist[static_cast<size_t>(r) * D + col] =
          (src < 3) ? hist_in[static_cast<size_t>(src) * D + col]
                    : moe_out[static_cast<size_t>(src - 3) * D + col];
    }
  }
}
)cuda"
