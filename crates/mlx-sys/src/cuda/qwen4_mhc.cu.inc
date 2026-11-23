R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Qwen3.8-Flash-Next (qwen4_exp) Fused mHC (Hyper-Connection) Mixer Kernels
// Geometry: HC = 4, H = 2560, W = 4*H = 10240, R = hc_lowrank = 320
// Quantization: Q8_0 (affine 8-bit, group_size=32, fp16 scales & biases)
// ============================================================================

__device__ __forceinline__ float q4mhc_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ __forceinline__ uint4 q4mhc_ld_cs_u4(const uint4* ptr) {
  return ptx_ld_cs_u4(ptr);
}

__device__ __forceinline__ void q4mhc_cp16(void* smem_ptr, const void* glob_ptr) {
  ptx_cp_async16_cg(smem_ptr, glob_ptr);
}

__device__ __forceinline__ void q4mhc_cp_commit() {
  ptx_cp_commit();
}

template <int N>
__device__ __forceinline__ void q4mhc_cp_wait() {
  ptx_cp_wait<N>();
}

__device__ __forceinline__ void q4mhc_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  ptx_mma_m16n8k16_bf16_v(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
}

template <int PAIR>
__device__ __forceinline__ unsigned int q4mhc_unpack_u8x2_bf16x2(unsigned int w) {
  constexpr int shift = PAIR * 16;
  const float f0 = static_cast<float>((w >> shift) & 0xFFu);
  const float f1 = static_cast<float>((w >> (shift + 8)) & 0xFFu);
  const __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float q4mhc_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

__device__ __forceinline__ float q4mhc_precise_sigmoid_f32(float x) {
  const float y = 1.0f / (1.0f + expf(fabsf(x)));
  return (x < 0.0f) ? y : (1.0f - y);
}

__device__ __forceinline__ nv_bfloat16 q4mhc_bf16_sigmoid(nv_bfloat16 x) {
  return __float2bfloat16(q4mhc_precise_sigmoid_f32(__bfloat162float(x)));
}

__device__ __forceinline__ nv_bfloat16 q4mhc_bf16_silu(nv_bfloat16 x) {
  const nv_bfloat16 sig = q4mhc_bf16_sigmoid(x);
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(sig));
}

// ----------------------------------------------------------------------------
// Kernel 1: qwen4_inject_norm_cuda
// Fuses optional residual stream injection + per-stream RMSNorm over H=2560:
//   If HAS_INJECT:
//     x_out[m, s, h] = bf16(x[m, s, h] + bf16(branch[m, h] * inject_gate[m, s]))
//   normed[m, s, h] = bf16((float(r) / sqrtf(mean(r^2) + eps)) * (w[s, h] + w_off))
// Works for ANY M >= 1 (decode M=1..16 AND prefill M=1024+).
// Grid: dim3(4, M, 1), Block: dim3(320, 1, 1) -> 320 threads * 8 BF16 = 2560!
// ----------------------------------------------------------------------------
template <bool HAS_INJECT, bool W_FP32>
__global__ __launch_bounds__(320, 4) void qwen4_inject_norm_cuda(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ branch,
    const nv_bfloat16* __restrict__ inject_gate,
    const void* __restrict__ norm_w,
    nv_bfloat16* __restrict__ x_out,
    nv_bfloat16* __restrict__ normed,
    int M,
    float eps,
    float weight_offset) {
  constexpr int H = 2560;
  constexpr int W = 10240;
  const int s = static_cast<int>(blockIdx.x);   // 0..3
  const int m = static_cast<int>(blockIdx.y);   // 0..M-1
  const int tid = static_cast<int>(threadIdx.x); // 0..319
  if (m >= M) {
    return;
  }

  const size_t stream_off = static_cast<size_t>(m) * W + static_cast<size_t>(s) * H;
  const uint4 xv = reinterpret_cast<const uint4*>(x + stream_off)[tid];
  uint4 rv;

  if constexpr (HAS_INJECT) {
    const uint4 bv = reinterpret_cast<const uint4*>(branch + static_cast<size_t>(m) * H)[tid];
    const float g = __bfloat162float(inject_gate[static_cast<size_t>(m) * 4 + s]);
    const nv_bfloat16* xe = reinterpret_cast<const nv_bfloat16*>(&xv);
    const nv_bfloat16* be = reinterpret_cast<const nv_bfloat16*>(&bv);
    nv_bfloat16* re = reinterpret_cast<nv_bfloat16*>(&rv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const nv_bfloat16 sp = __float2bfloat16(__bfloat162float(be[j]) * g);
      re[j] = __float2bfloat16(__bfloat162float(xe[j]) + __bfloat162float(sp));
    }
    reinterpret_cast<uint4*>(x_out + stream_off)[tid] = rv;
  } else {
    rv = xv;
  }

  const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
  float sq = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(r2[j]);
    sq += f.x * f.x + f.y * f.y;
  }

  __shared__ float warp_sums[10];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  sq = q4mhc_warp_sum(sq);
  if (lane == 0) {
    warp_sums[warp] = sq;
  }
  __syncthreads();

  if (warp == 0) {
    float tot = (lane < 10) ? warp_sums[lane] : 0.0f;
    tot = q4mhc_warp_sum(tot);
    if (lane == 0) {
      warp_sums[0] = sqrtf(tot * (1.0f / static_cast<float>(H)) + eps);
    }
  }
  __syncthreads();
  const float denom = warp_sums[0];

  const size_t w_base = static_cast<size_t>(s) * H + static_cast<size_t>(tid) * 8;
  uint4 nv;
  nv_bfloat16* ne = reinterpret_cast<nv_bfloat16*>(&nv);
  const nv_bfloat16* re = reinterpret_cast<const nv_bfloat16*>(&rv);

  if constexpr (W_FP32) {
    const float4 w0 = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(norm_w) + w_base);
    const float4 w1 = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(norm_w) + w_base + 4);
    const float wf[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float n = __bfloat162float(re[j]) / denom;
      ne[j] = __float2bfloat16(n * (wf[j] + weight_offset));
    }
  } else {
    const uint4 wv = *reinterpret_cast<const uint4*>(reinterpret_cast<const nv_bfloat16*>(norm_w) + w_base);
    const nv_bfloat16* we = reinterpret_cast<const nv_bfloat16*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float n = __bfloat162float(re[j]) / denom;
      ne[j] = __float2bfloat16(n * (__bfloat162float(we[j]) + weight_offset));
    }
  }
  reinterpret_cast<uint4*>(normed + stream_off)[tid] = nv;
}

// ----------------------------------------------------------------------------
// Kernel 1B: qwen4_inject_only_cuda
// Residual stream injection without RMSNorm (used on Layer 0 pre-PLE and Layer 47 tail):
//   x_out[m, s, h] = bf16(x[m, s, h] + bf16(branch[m, h] * inject_gate[m, s]))
// Grid: dim3(4, M, 1), Block: dim3(320, 1, 1) -> 320 threads * 8 BF16 = 2560!
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(320, 4) void qwen4_inject_only_cuda(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ branch,
    const nv_bfloat16* __restrict__ inject_gate,
    nv_bfloat16* __restrict__ x_out,
    int M) {
  constexpr int H = 2560;
  constexpr int W = 10240;
  const int s = static_cast<int>(blockIdx.x);   // 0..3
  const int m = static_cast<int>(blockIdx.y);   // 0..M-1
  const int tid = static_cast<int>(threadIdx.x); // 0..319
  if (m >= M) {
    return;
  }

  const size_t stream_off = static_cast<size_t>(m) * W + static_cast<size_t>(s) * H;
  const uint4 xv = reinterpret_cast<const uint4*>(x + stream_off)[tid];
  const uint4 bv = reinterpret_cast<const uint4*>(branch + static_cast<size_t>(m) * H)[tid];
  const float g = __bfloat162float(inject_gate[static_cast<size_t>(m) * 4 + s]);
  uint4 rv;
  const nv_bfloat16* xe = reinterpret_cast<const nv_bfloat16*>(&xv);
  const nv_bfloat16* be = reinterpret_cast<const nv_bfloat16*>(&bv);
  nv_bfloat16* re = reinterpret_cast<nv_bfloat16*>(&rv);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const nv_bfloat16 sp = __float2bfloat16(__bfloat162float(be[j]) * g);
    re[j] = __float2bfloat16(__bfloat162float(xe[j]) + __bfloat162float(sp));
  }
  reinterpret_cast<uint4*>(x_out + stream_off)[tid] = rv;
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2A: qwen4_hc_down_act_u4_cuda (M_CAP in {1, 4} for M in 1..=4)
// Fuses Q8_0 (affine 8b g32 fp16 s+b) matmul:
//   down_act[m, n] = SiLU(bf16((normed[m] . W_down[n]) * 0.25f))    (n in 0..319)
//   inject_gate[m, s] = bf16(2.0f * sigmoid((normed[m] . W_inj[s]) * 0.25f)) (s in 0..3)
// INJ_MODE: 0=none, 1=Q8_0 (g32 fp16 s+b), 2=dense BF16, 3=dense FP16, 4=dense FP32
// Grid: dim3(INJ_MODE ? 324 : 320, 1, 1), Block: dim3(32, 4, 1) -> 5 full uint4 iters/thread.
// ----------------------------------------------------------------------------
template <int M_CAP, int INJ_MODE>
__global__ __launch_bounds__(128, 4) void qwen4_hc_down_act_u4_cuda(
    const nv_bfloat16* __restrict__ normed,
    const unsigned int* __restrict__ w_down,
    const __half* __restrict__ sc_down,
    const __half* __restrict__ bi_down,
    const void* __restrict__ w_inj,
    const __half* __restrict__ sc_inj,
    const __half* __restrict__ bi_inj,
    nv_bfloat16* __restrict__ down_act,
    nv_bfloat16* __restrict__ inject_gate,
    int M) {
  constexpr int K = 10240;
  constexpr int N_DOWN = 320;
  constexpr int NUM_U4 = K >> 4;   // 640
  constexpr int GROUPS = K >> 5;   // 320
  constexpr int X_U4 = K >> 3;     // 1280

  const int bid = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const bool is_inj = (INJ_MODE != 0) && (bid >= N_DOWN);
  const int row = is_inj ? (bid - N_DOWN) : bid;

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  if (!is_inj || INJ_MODE == 1) {
    const uint4* w_row = is_inj
        ? (reinterpret_cast<const uint4*>(w_inj) + static_cast<size_t>(row) * NUM_U4)
        : (reinterpret_cast<const uint4*>(w_down) + static_cast<size_t>(row) * NUM_U4);
    const __half* sc_row = (is_inj ? sc_inj : sc_down) + static_cast<size_t>(row) * GROUPS;
    const __half* bi_row = (is_inj ? bi_inj : bi_down) + static_cast<size_t>(row) * GROUPS;

#pragma unroll
    for (int it = 0; it < 5; ++it) {
      const int u = it * 128 + warp * 32 + lane;
      const uint4 wv = q4mhc_ld_cs_u4(w_row + u);
      const int g = u >> 1;
      const float sc = __half2float(sc_row[g]);
      const float bi = __half2float(bi_row[g]);
      const size_t x_off = static_cast<size_t>(u) * 2;

#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (M_CAP == 1 || m < M) {
          const uint4* xp = reinterpret_cast<const uint4*>(normed) +
              static_cast<size_t>(m) * X_U4 + x_off;
          const uint4 x0 = xp[0];
          const uint4 x1 = xp[1];
          const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
          const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
          float d = 0.0f;
          float s = 0.0f;
          float2 f;
          unsigned int uu;

          uu = wv.x;
          f = __bfloat1622float2(p0[0]);
          d = fmaf(f.x, static_cast<float>(uu & 0xFFu),
                   fmaf(f.y, static_cast<float>((uu >> 8) & 0xFFu), d));
          s += f.x + f.y;
          f = __bfloat1622float2(p0[1]);
          d = fmaf(f.x, static_cast<float>((uu >> 16) & 0xFFu),
                   fmaf(f.y, static_cast<float>(uu >> 24), d));
          s += f.x + f.y;

          uu = wv.y;
          f = __bfloat1622float2(p0[2]);
          d = fmaf(f.x, static_cast<float>(uu & 0xFFu),
                   fmaf(f.y, static_cast<float>((uu >> 8) & 0xFFu), d));
          s += f.x + f.y;
          f = __bfloat1622float2(p0[3]);
          d = fmaf(f.x, static_cast<float>((uu >> 16) & 0xFFu),
                   fmaf(f.y, static_cast<float>(uu >> 24), d));
          s += f.x + f.y;

          uu = wv.z;
          f = __bfloat1622float2(p1[0]);
          d = fmaf(f.x, static_cast<float>(uu & 0xFFu),
                   fmaf(f.y, static_cast<float>((uu >> 8) & 0xFFu), d));
          s += f.x + f.y;
          f = __bfloat1622float2(p1[1]);
          d = fmaf(f.x, static_cast<float>((uu >> 16) & 0xFFu),
                   fmaf(f.y, static_cast<float>(uu >> 24), d));
          s += f.x + f.y;

          uu = wv.w;
          f = __bfloat1622float2(p1[2]);
          d = fmaf(f.x, static_cast<float>(uu & 0xFFu),
                   fmaf(f.y, static_cast<float>((uu >> 8) & 0xFFu), d));
          s += f.x + f.y;
          f = __bfloat1622float2(p1[3]);
          d = fmaf(f.x, static_cast<float>((uu >> 16) & 0xFFu),
                   fmaf(f.y, static_cast<float>(uu >> 24), d));
          s += f.x + f.y;

          acc[m] = fmaf(sc, d, fmaf(bi, s, acc[m]));
        }
      }
    }
  } else {
    for (int idx = warp * 32 + lane; idx < X_U4; idx += 128) {
      float wf[8];
      if constexpr (INJ_MODE == 2) {
        const uint4 wv = reinterpret_cast<const uint4*>(w_inj)[static_cast<size_t>(row) * X_U4 + idx];
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(wp[j]);
          wf[2 * j] = f.x;
          wf[2 * j + 1] = f.y;
        }
      } else if constexpr (INJ_MODE == 3) {
        const uint4 wv = reinterpret_cast<const uint4*>(w_inj)[static_cast<size_t>(row) * X_U4 + idx];
        const __half2* wp = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __half22float2(wp[j]);
          wf[2 * j] = f.x;
          wf[2 * j + 1] = f.y;
        }
      } else {
        const float4* wp = reinterpret_cast<const float4*>(w_inj) +
            (static_cast<size_t>(row) * X_U4 + idx) * 2;
        const float4 a = wp[0];
        const float4 b = wp[1];
        wf[0] = a.x; wf[1] = a.y; wf[2] = a.z; wf[3] = a.w;
        wf[4] = b.x; wf[5] = b.y; wf[6] = b.z; wf[7] = b.w;
      }
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (M_CAP == 1 || m < M) {
          const uint4 xv = reinterpret_cast<const uint4*>(normed)[static_cast<size_t>(m) * X_U4 + idx];
          const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
          float d = 0.0f;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(xp[j]);
            d = fmaf(f.x, wf[2 * j], fmaf(f.y, wf[2 * j + 1], d));
          }
          acc[m] += d;
        }
      }
    }
  }

  __shared__ float smem[4][M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (M_CAP == 1 || m < M) {
      const float v = q4mhc_warp_sum(acc[m]);
      if (lane == 0) {
        smem[warp][m] = v;
      }
    }
  }
  __syncthreads();

  if (warp == 0 && lane < M) {
    const float sum = smem[0][lane] + smem[1][lane] + smem[2][lane] + smem[3][lane];
    const nv_bfloat16 scaled = __float2bfloat16(__bfloat162float(__float2bfloat16(sum)) * 0.25f);
    if (is_inj) {
      const nv_bfloat16 sig = q4mhc_bf16_sigmoid(scaled);
      inject_gate[static_cast<size_t>(lane) * 4 + row] =
          __float2bfloat16(__bfloat162float(sig) * 2.0f);
    } else {
      down_act[static_cast<size_t>(lane) * N_DOWN + row] = q4mhc_bf16_silu(scaled);
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2B: qwen4_hc_down_act_mma_n8_cuda (M in 2..=16, 41 CTAs, 2-stage cp.async.cg!)
// 100% full-width mma.m16n8k16 (A = X[16], B = W[8]):
//   - 8 output rows/CTA -> 320/8 = 40 CTAs + 1 inject CTA = 41 CTAs (4x less X L2 traffic!)
//   - 256 threads (8 warps) load 8 rows x 32 groups (1024B/row = 512 uint4 = 2 cp.async/tid!)
//   - Only 10 tiles across K=10240 (18.4 KB SMEM, 0 spills), 4 groups/warp/tile!
// Grid: dim3(INJ_MODE ? 41 : 40, 1, 1), Block: dim3(32, 8, 1).
// ----------------------------------------------------------------------------
template <int INJ_MODE>
__global__ __launch_bounds__(256, 3) void qwen4_hc_down_act_mma_n8_cuda(
    const nv_bfloat16* __restrict__ normed,
    const unsigned int* __restrict__ w_down,
    const __half* __restrict__ sc_down,
    const __half* __restrict__ bi_down,
    const void* __restrict__ w_inj,
    const __half* __restrict__ sc_inj,
    const __half* __restrict__ bi_inj,
    nv_bfloat16* __restrict__ down_act,
    nv_bfloat16* __restrict__ inject_gate,
    int M) {
  constexpr int K = 10240;
  constexpr int N_DOWN = 320;
  constexpr int GROUPS = K >> 5;   // 320
  constexpr int NUM_TILES = 5;     // 5 tiles of 64 groups (2048 cols = 128 uint4/row)
  constexpr int W_U4 = K >> 4;     // 640
  constexpr int X_U4 = K >> 3;     // 1280

  const int bid = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  const bool is_inj = (INJ_MODE != 0) && (bid == 40);
  const int row0 = is_inj ? 0 : (bid * 8);
  const int n_valid = is_inj ? 4 : 8;

  if (is_inj && INJ_MODE > 1) {
    __shared__ float red_inj[2][4][16];
    const int r = warp >> 1;    // 0..3
    const int sub = warp & 1;   // 0..1
    float acc_m[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc_m[m] = 0.0f;
    for (int idx = sub * 32 + lane; idx < X_U4; idx += 64) {
      float wf[8];
      if constexpr (INJ_MODE == 2) {
        const uint4 wv = reinterpret_cast<const uint4*>(w_inj)[static_cast<size_t>(r) * X_U4 + idx];
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(wp[j]);
          wf[2 * j] = f.x; wf[2 * j + 1] = f.y;
        }
      } else if constexpr (INJ_MODE == 3) {
        const uint4 wv = reinterpret_cast<const uint4*>(w_inj)[static_cast<size_t>(r) * X_U4 + idx];
        const __half2* wp = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __half22float2(wp[j]);
          wf[2 * j] = f.x; wf[2 * j + 1] = f.y;
        }
      } else {
        const float4* wp = reinterpret_cast<const float4*>(w_inj) +
            (static_cast<size_t>(r) * X_U4 + idx) * 2;
        const float4 a = wp[0]; const float4 b = wp[1];
        wf[0] = a.x; wf[1] = a.y; wf[2] = a.z; wf[3] = a.w;
        wf[4] = b.x; wf[5] = b.y; wf[6] = b.z; wf[7] = b.w;
      }
      for (int m = 0; m < M; ++m) {
        const uint4 xv = reinterpret_cast<const uint4*>(normed)[static_cast<size_t>(m) * X_U4 + idx];
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        float d = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(xp[j]);
          d = fmaf(f.x, wf[2 * j], fmaf(f.y, wf[2 * j + 1], d));
        }
        acc_m[m] += d;
      }
    }
    for (int m = 0; m < M; ++m) {
      const float v = q4mhc_warp_sum(acc_m[m]);
      if (lane == 0) red_inj[sub][r][m] = v;
    }
    __syncthreads();
    if (tid < M * 4) {
      const int m = tid >> 2;
      const int rr = tid & 3;
      const float sum = red_inj[0][rr][m] + red_inj[1][rr][m];
      const nv_bfloat16 scaled = __float2bfloat16(__bfloat162float(__float2bfloat16(sum)) * 0.25f);
      inject_gate[static_cast<size_t>(m) * 4 + rr] =
          __float2bfloat16(__bfloat162float(q4mhc_bf16_sigmoid(scaled)) * 2.0f);
    }
    return;
  }

  // 2-stage cp.async.cg buffer: 2 * (8 rows * 128 uint4 = 16384 B + 2048 B s/b) = 36,864 B!
  union SharedDownN8 {
    struct {
      uint4 w[2][8][128];   // 32,768 B
      __half sc[2][8][64];  // 2,048 B
      __half bi[2][8][64];  // 2,048 B
    } pipe;
    float red[8][16][9];    // 4,608 B
  };
  __shared__ __align__(16) SharedDownN8 sh;

  const unsigned int* w_src = is_inj ? reinterpret_cast<const unsigned int*>(w_inj) : w_down;
  const __half* sc_src = is_inj ? sc_inj : sc_down;
  const __half* bi_src = is_inj ? bi_inj : bi_down;

  // All 256 threads load 8 rows x 128 uint4 = 1024 uint4 (4 cp.async/thread, 16KB in flight!)
  const int ld_row0 = tid >> 7;        // 0..1
  const int ld_col = tid & 127;        // 0..127
  const int rl0 = (ld_row0 < n_valid) ? (row0 + ld_row0) : (row0 + n_valid - 1);
  const int rl1 = (ld_row0 + 2 < n_valid) ? (row0 + ld_row0 + 2) : (row0 + n_valid - 1);
  const int rl2 = (ld_row0 + 4 < n_valid) ? (row0 + ld_row0 + 4) : (row0 + n_valid - 1);
  const int rl3 = (ld_row0 + 6 < n_valid) ? (row0 + ld_row0 + 6) : (row0 + n_valid - 1);
  const uint4* w_ptr0 = reinterpret_cast<const uint4*>(w_src) + static_cast<size_t>(rl0) * W_U4 + ld_col;
  const uint4* w_ptr1 = reinterpret_cast<const uint4*>(w_src) + static_cast<size_t>(rl1) * W_U4 + ld_col;
  const uint4* w_ptr2 = reinterpret_cast<const uint4*>(w_src) + static_cast<size_t>(rl2) * W_U4 + ld_col;
  const uint4* w_ptr3 = reinterpret_cast<const uint4*>(w_src) + static_cast<size_t>(rl3) * W_U4 + ld_col;

  // 128 threads load 8 rows x 64 fp16 scales (64 uint4) + 8 rows x 64 fp16 biases (64 uint4)!
  const bool is_sb = (tid < 128);
  const int sb_is_bi = (tid >> 6) & 1;
  const int sb_row = (tid >> 3) & 7;
  const int sb_c8 = tid & 7; // 0..7 (8 fp16s = 16B each!)
  const int r_sb = (sb_row < n_valid) ? (row0 + sb_row) : (row0 + n_valid - 1);
  const __half* sb_ptr = (sb_is_bi ? bi_src : sc_src) +
      static_cast<size_t>(r_sb) * GROUPS + sb_c8 * 8;

  auto issue_tile = [&](int stage, int tile) {
    const size_t off_u4 = static_cast<size_t>(tile) * 128;
    q4mhc_cp16(&sh.pipe.w[stage][ld_row0][ld_col], w_ptr0 + off_u4);
    q4mhc_cp16(&sh.pipe.w[stage][ld_row0 + 2][ld_col], w_ptr1 + off_u4);
    q4mhc_cp16(&sh.pipe.w[stage][ld_row0 + 4][ld_col], w_ptr2 + off_u4);
    q4mhc_cp16(&sh.pipe.w[stage][ld_row0 + 6][ld_col], w_ptr3 + off_u4);
    if (is_sb) {
      __half* dst = sb_is_bi ? &sh.pipe.bi[stage][sb_row][sb_c8 * 8]
                             : &sh.pipe.sc[stage][sb_row][sb_c8 * 8];
      q4mhc_cp16(dst, sb_ptr + static_cast<size_t>(tile) * 64);
    }
    q4mhc_cp_commit();
  };

  issue_tile(0, 0);

  const bool valid_m0 = (g < M);
  const bool valid_m1 = (g + 8 < M);
  const uint4* x0_u4 = reinterpret_cast<const uint4*>(normed) +
      static_cast<size_t>(valid_m0 ? g : 0) * X_U4 + t;
  const uint4* x1_u4 = reinterpret_cast<const uint4*>(normed) +
      static_cast<size_t>(valid_m1 ? (g + 8) : 0) * X_U4 + t;

  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile = 0; tile < NUM_TILES; ++tile) {
    const int cur = tile & 1;
    const int nxt = cur ^ 1;
    if (tile + 1 < NUM_TILES) {
      issue_tile(nxt, tile + 1);
      q4mhc_cp_wait<1>();
    } else {
      q4mhc_cp_wait<0>();
    }
    __syncthreads();

    // Each of the 8 warps processes 8 groups of this 64-group tile!
#pragma unroll 2
    for (int dg = 0; dg < 8; ++dg) {
      const int local_grp = warp * 8 + dg;
      const int global_grp = tile * 64 + local_grp;
      const size_t x_grp_u4 = static_cast<size_t>(global_grp) * 4;

      const uint4 xv0 = valid_m0 ? x0_u4[x_grp_u4] : make_uint4(0, 0, 0, 0);
      const uint4 xv1 = valid_m1 ? x1_u4[x_grp_u4] : make_uint4(0, 0, 0, 0);
      float s0 = valid_m0 ? q4mhc_sum_bf16x8(xv0) : 0.0f;
      float s1 = valid_m1 ? q4mhc_sum_bf16x8(xv1) : 0.0f;
      s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 2);

      const uint2 w2 = reinterpret_cast<const uint2*>(sh.pipe.w[cur][g])[local_grp * 4 + t];
      const unsigned int b0_0 = q4mhc_unpack_u8x2_bf16x2<0>(w2.x);
      const unsigned int b1_0 = q4mhc_unpack_u8x2_bf16x2<1>(w2.x);
      const unsigned int b0_1 = q4mhc_unpack_u8x2_bf16x2<0>(w2.y);
      const unsigned int b1_1 = q4mhc_unpack_u8x2_bf16x2<1>(w2.y);

      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      q4mhc_mma_m16n8k16_bf16(d0, d1, d2, d3, xv0.x, xv1.x, xv0.y, xv1.y, b0_0, b1_0);
      q4mhc_mma_m16n8k16_bf16(d0, d1, d2, d3, xv0.z, xv1.z, xv0.w, xv1.w, b0_1, b1_1);

      const float sc0 = __half2float(sh.pipe.sc[cur][2 * t][local_grp]);
      const float bi0 = __half2float(sh.pipe.bi[cur][2 * t][local_grp]);
      const float sc1 = __half2float(sh.pipe.sc[cur][2 * t + 1][local_grp]);
      const float bi1 = __half2float(sh.pipe.bi[cur][2 * t + 1][local_grp]);
      total[0] = fmaf(sc0, d0, fmaf(bi0, s0, total[0]));
      total[1] = fmaf(sc1, d1, fmaf(bi1, s0, total[1]));
      total[2] = fmaf(sc0, d2, fmaf(bi0, s1, total[2]));
      total[3] = fmaf(sc1, d3, fmaf(bi1, s1, total[3]));
    }
    __syncthreads();
  }

  sh.red[warp][g][2 * t] = total[0];
  sh.red[warp][g][2 * t + 1] = total[1];
  sh.red[warp][g + 8][2 * t] = total[2];
  sh.red[warp][g + 8][2 * t + 1] = total[3];
  __syncthreads();

  if (tid < M * n_valid) {
    const int m = tid / n_valid;
    const int n_local = tid - m * n_valid;
    float sum = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) {
      sum += sh.red[ww][m][n_local];
    }
    const nv_bfloat16 scaled = __float2bfloat16(__bfloat162float(__float2bfloat16(sum)) * 0.25f);
    if (is_inj) {
      inject_gate[static_cast<size_t>(m) * 4 + n_local] =
          __float2bfloat16(__bfloat162float(q4mhc_bf16_sigmoid(scaled)) * 2.0f);
    } else {
      down_act[static_cast<size_t>(m) * N_DOWN + row0 + n_local] = q4mhc_bf16_silu(scaled);
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 3: qwen4_hc_up_mix_mma_cuda (ALL M in 1..=128 via M_TILES in {1, 2}!)
// Fuses Q8_0 up-projection (K=320, 10 groups) + sigmoid(up)*normed + 4-stream mean:
//   mixed[m, h] = bf16(0.25f * sum_{s=0..3} bf16(sigmoid(bf16(down_act[m] . W_up[s*2560+h]))
//                                                * normed[m, s*2560+h]))
// NEVER writes [M, 10240] to DRAM!
// Optimizations:
//   - cp.async.cg streams 16 contiguous rows (5120B/warp = 10x 512B bursts!) directly
//     into SMEM, overlapped with sX load & sXSum precompute!
//   - Permuted-K uint4 sX loads + uint2 sW loads -> 2 mma.m16n8k16 per group (0 branch!).
//   - Only ~27-33 KB SMEM -> 2-3 CTAs/SM, 0 spills, saturates LPDDR5X!
// Grid: dim3(160, (M + M_BLK - 1) / M_BLK, 1), Block: dim3(32, 4, 1) -> 1 warp/stream!
// ----------------------------------------------------------------------------
template <int M_TILES>
__global__ __launch_bounds__(128, 3) void qwen4_hc_up_mix_mma_cuda(
    const nv_bfloat16* __restrict__ down_act,
    const unsigned int* __restrict__ w_up,
    const __half* __restrict__ sc_up,
    const __half* __restrict__ bi_up,
    const nv_bfloat16* __restrict__ normed,
    nv_bfloat16* __restrict__ mixed,
    int M) {
  constexpr int H = 2560;
  constexpr int W = 10240;
  constexpr int K = 320;
  constexpr int GROUPS = 10;
  constexpr int ROW_U4 = 20;
  constexpr int M_BLK = M_TILES * 8;

  const int h0 = static_cast<int>(blockIdx.x) * 16;
  const int m0 = static_cast<int>(blockIdx.y) * M_BLK;
  const int m_cnt = (M - m0 < M_BLK) ? (M - m0) : M_BLK;
  if (m_cnt <= 0) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x);
  const int s = static_cast<int>(threadIdx.y); // stream 0..3
  const int tid = s * 32 + lane;
  const int g = lane >> 2; // 0..7
  const int t = lane & 3;  // 0..3

  union SharedUpMma {
    struct {
      uint4 w[4][16 * ROW_U4];     // 20,480 B (flat contiguous per stream -> 100% cp.async!)
      __half sc[4][16 * GROUPS];   // 1,280 B
      __half bi[4][16 * GROUPS];   // 1,280 B
    } stage;
    float prod[4][M_BLK][16];      // <= 4,096 B
  };
  __shared__ __align__(16) SharedUpMma sh;
  __shared__ __align__(16) nv_bfloat16 sX[M_BLK][K]; // <= 10,240 B
  __shared__ float sXSum[M_BLK][GROUPS];             // <= 640 B

  // Issue ALL cold-DRAM W_up + scales + biases cp.async.cg FIRST!
  const size_t w_row0 = static_cast<size_t>(s) * H + h0;
  const uint4* w_base = reinterpret_cast<const uint4*>(w_up) + w_row0 * ROW_U4;
#pragma unroll
  for (int it = 0; it < 10; ++it) {
    const int idx = it * 32 + lane;
    q4mhc_cp16(&sh.stage.w[s][idx], w_base + idx);
  }
  const uint4* sc_base = reinterpret_cast<const uint4*>(sc_up + w_row0 * GROUPS);
  const uint4* bi_base = reinterpret_cast<const uint4*>(bi_up + w_row0 * GROUPS);
  if (lane < 20) {
    q4mhc_cp16(&reinterpret_cast<uint4*>(sh.stage.sc[s])[lane], sc_base + lane);
    q4mhc_cp16(&reinterpret_cast<uint4*>(sh.stage.bi[s])[lane], bi_base + lane);
  }
  q4mhc_cp_commit();

  // Overlap L2-hot down_act load & group sum while DRAM W_up cp.async is in flight!
  const int total_x_u4 = m_cnt * (K >> 3); // m_cnt * 40
  for (int idx = tid; idx < total_x_u4; idx += 128) {
    const int lm = idx / 40;
    const int c8 = idx - lm * 40;
    reinterpret_cast<uint4*>(sX[lm])[c8] =
        reinterpret_cast<const uint4*>(down_act + static_cast<size_t>(m0 + lm) * K)[c8];
  }
  __syncthreads();

  for (int idx = tid; idx < m_cnt * GROUPS; idx += 128) {
    const int lm = idx / GROUPS;
    const int grp = idx - lm * GROUPS;
    const uint4* xp = reinterpret_cast<const uint4*>(&sX[lm][grp * 32]);
    float sum = (q4mhc_sum_bf16x8(xp[0]) + q4mhc_sum_bf16x8(xp[1])) +
                (q4mhc_sum_bf16x8(xp[2]) + q4mhc_sum_bf16x8(xp[3]));
    sXSum[lm][grp] = sum;
  }

  q4mhc_cp_wait<0>();
  __syncthreads();

  float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const bool valid_b0 = (g < m_cnt);
  const bool valid_b1 = (M_TILES == 2) && (g + 8 < m_cnt);
  const uint2* sw_r0 = reinterpret_cast<const uint2*>(&sh.stage.w[s][g * ROW_U4]);
  const uint2* sw_r1 = reinterpret_cast<const uint2*>(&sh.stage.w[s][(g + 8) * ROW_U4]);
  const uint4* sx_b0 = valid_b0 ? reinterpret_cast<const uint4*>(sX[g]) : nullptr;
  const uint4* sx_b1 = valid_b1 ? reinterpret_cast<const uint4*>(sX[g + 8]) : nullptr;

#pragma unroll 2
  for (int grp = 0; grp < GROUPS; ++grp) {
    // Permuted-K uint2 W load + uint4 X load -> 0 branches, 2 MMAs per group!
    const uint2 w0 = sw_r0[grp * 4 + t];
    const uint2 w1 = sw_r1[grp * 4 + t];
    const unsigned int a0_0 = q4mhc_unpack_u8x2_bf16x2<0>(w0.x);
    const unsigned int a1_0 = q4mhc_unpack_u8x2_bf16x2<0>(w1.x);
    const unsigned int a2_0 = q4mhc_unpack_u8x2_bf16x2<1>(w0.x);
    const unsigned int a3_0 = q4mhc_unpack_u8x2_bf16x2<1>(w1.x);
    const unsigned int a0_1 = q4mhc_unpack_u8x2_bf16x2<0>(w0.y);
    const unsigned int a1_1 = q4mhc_unpack_u8x2_bf16x2<0>(w1.y);
    const unsigned int a2_1 = q4mhc_unpack_u8x2_bf16x2<1>(w0.y);
    const unsigned int a3_1 = q4mhc_unpack_u8x2_bf16x2<1>(w1.y);

    const uint4 xv0 = valid_b0 ? sx_b0[grp * 4 + t] : make_uint4(0, 0, 0, 0);
    float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q4mhc_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0_0, a1_0, a2_0, a3_0, xv0.x, xv0.y);
    q4mhc_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0_1, a1_1, a2_1, a3_1, xv0.z, xv0.w);

    const float sc0 = __half2float(sh.stage.sc[s][g * GROUPS + grp]);
    const float bi0 = __half2float(sh.stage.bi[s][g * GROUPS + grp]);
    const float sc1 = __half2float(sh.stage.sc[s][(g + 8) * GROUPS + grp]);
    const float bi1 = __half2float(sh.stage.bi[s][(g + 8) * GROUPS + grp]);

    const float xs00 = (2 * t < m_cnt) ? sXSum[2 * t][grp] : 0.0f;
    const float xs01 = (2 * t + 1 < m_cnt) ? sXSum[2 * t + 1][grp] : 0.0f;
    total0[0] = fmaf(sc0, d0[0], fmaf(bi0, xs00, total0[0]));
    total0[1] = fmaf(sc0, d0[1], fmaf(bi0, xs01, total0[1]));
    total0[2] = fmaf(sc1, d0[2], fmaf(bi1, xs00, total0[2]));
    total0[3] = fmaf(sc1, d0[3], fmaf(bi1, xs01, total0[3]));

    if constexpr (M_TILES == 2) {
      const uint4 xv1 = valid_b1 ? sx_b1[grp * 4 + t] : make_uint4(0, 0, 0, 0);
      float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      q4mhc_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0_0, a1_0, a2_0, a3_0, xv1.x, xv1.y);
      q4mhc_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0_1, a1_1, a2_1, a3_1, xv1.z, xv1.w);
      const float xs10 = (8 + 2 * t < m_cnt) ? sXSum[8 + 2 * t][grp] : 0.0f;
      const float xs11 = (8 + 2 * t + 1 < m_cnt) ? sXSum[8 + 2 * t + 1][grp] : 0.0f;
      total1[0] = fmaf(sc0, d1[0], fmaf(bi0, xs10, total1[0]));
      total1[1] = fmaf(sc0, d1[1], fmaf(bi0, xs11, total1[1]));
      total1[2] = fmaf(sc1, d1[2], fmaf(bi1, xs10, total1[2]));
      total1[3] = fmaf(sc1, d1[3], fmaf(bi1, xs11, total1[3]));
    }
  }

  auto eval_prod = [&](int lm, int lr, float val) -> float {
    if (lm >= m_cnt) return 0.0f;
    const nv_bfloat16 sig = q4mhc_bf16_sigmoid(__float2bfloat16(val));
    const nv_bfloat16 nv = normed[static_cast<size_t>(m0 + lm) * W + w_row0 + lr];
    return __bfloat162float(__float2bfloat16(__bfloat162float(sig) * __bfloat162float(nv)));
  };
  const float p00 = eval_prod(2 * t, g, total0[0]);
  const float p01 = eval_prod(2 * t + 1, g, total0[1]);
  const float p02 = eval_prod(2 * t, g + 8, total0[2]);
  const float p03 = eval_prod(2 * t + 1, g + 8, total0[3]);
  float p10 = 0.0f, p11 = 0.0f, p12 = 0.0f, p13 = 0.0f;
  if constexpr (M_TILES == 2) {
    p10 = eval_prod(8 + 2 * t, g, total1[0]);
    p11 = eval_prod(8 + 2 * t + 1, g, total1[1]);
    p12 = eval_prod(8 + 2 * t, g + 8, total1[2]);
    p13 = eval_prod(8 + 2 * t + 1, g + 8, total1[3]);
  }
  __syncthreads();

  sh.prod[s][2 * t][g] = p00;
  sh.prod[s][2 * t + 1][g] = p01;
  sh.prod[s][2 * t][g + 8] = p02;
  sh.prod[s][2 * t + 1][g + 8] = p03;
  if constexpr (M_TILES == 2) {
    sh.prod[s][8 + 2 * t][g] = p10;
    sh.prod[s][8 + 2 * t + 1][g] = p11;
    sh.prod[s][8 + 2 * t][g + 8] = p12;
    sh.prod[s][8 + 2 * t + 1][g + 8] = p13;
  }
  __syncthreads();

  for (int idx = tid; idx < m_cnt * 16; idx += 128) {
    const int lm = idx >> 4;
    const int lr = idx & 15;
    const float mean_v = 0.25f * (
        (sh.prod[0][lm][lr] + sh.prod[1][lm][lr]) +
        (sh.prod[2][lm][lr] + sh.prod[3][lm][lr]));
    mixed[static_cast<size_t>(m0 + lm) * H + h0 + lr] = __float2bfloat16(mean_v);
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 3B: qwen4_hc_up_mix_u4_m1_cuda (Ultra-high-occupancy M=1 streaming)
// Fuses Q8_0 up-projection (K=320, 20 uint4s/row) + sigmoid(up)*normed + mean.
// 1.4 KB SMEM (48 warps/SM!), 0 MMA, 100% lane utilization!
// Grid: dim3(320, 1, 1) (8 consecutive h per CTA), Block: dim3(32, 4, 1) (1 warp/stream).
// ----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128, 6) void qwen4_hc_up_mix_u4_m1_cuda(
    const nv_bfloat16* __restrict__ down_act,
    const unsigned int* __restrict__ w_up,
    const __half* __restrict__ sc_up,
    const __half* __restrict__ bi_up,
    const nv_bfloat16* __restrict__ normed,
    nv_bfloat16* __restrict__ mixed) {
  constexpr int H = 2560;
  constexpr int K = 320;
  constexpr int GROUPS = 10;
  constexpr int ROW_U4 = 20;

  const int h0 = static_cast<int>(blockIdx.x) * 8;
  const int lane = static_cast<int>(threadIdx.x);
  const int s = static_cast<int>(threadIdx.y);
  const int tid = s * 32 + lane;

  __shared__ float s_x[K];
  __shared__ float s_prod[4][8];

  if (tid < 40) {
    const uint4 xv = reinterpret_cast<const uint4*>(down_act)[tid];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      s_x[tid * 8 + j * 2] = f.x;
      s_x[tid * 8 + j * 2 + 1] = f.y;
    }
  }
  __syncthreads();

  const int h_sub = lane >> 2;  // 0..7
  const int kp = lane & 3;      // 0..3
  const size_t row = static_cast<size_t>(s) * H + h0 + h_sub;
  const uint4* w_row = reinterpret_cast<const uint4*>(w_up) + row * ROW_U4;
  const __half* sc_row = sc_up + row * GROUPS;
  const __half* bi_row = bi_up + row * GROUPS;

  float acc = 0.0f;
#pragma unroll
  for (int it = 0; it < 5; ++it) {
    const int u = it * 4 + kp;
    const uint4 wv = q4mhc_ld_cs_u4(w_row + u);
    const float* sx = s_x + u * 16;
    float d = 0.0f;
    float xs = 0.0f;
    unsigned int uu;
    uu = wv.x;
    d = fmaf(sx[0], static_cast<float>(uu & 0xFFu),
        fmaf(sx[1], static_cast<float>((uu >> 8) & 0xFFu),
        fmaf(sx[2], static_cast<float>((uu >> 16) & 0xFFu),
        fmaf(sx[3], static_cast<float>(uu >> 24), d))));
    xs += (sx[0] + sx[1]) + (sx[2] + sx[3]);
    uu = wv.y;
    d = fmaf(sx[4], static_cast<float>(uu & 0xFFu),
        fmaf(sx[5], static_cast<float>((uu >> 8) & 0xFFu),
        fmaf(sx[6], static_cast<float>((uu >> 16) & 0xFFu),
        fmaf(sx[7], static_cast<float>(uu >> 24), d))));
    xs += (sx[4] + sx[5]) + (sx[6] + sx[7]);
    uu = wv.z;
    d = fmaf(sx[8], static_cast<float>(uu & 0xFFu),
        fmaf(sx[9], static_cast<float>((uu >> 8) & 0xFFu),
        fmaf(sx[10], static_cast<float>((uu >> 16) & 0xFFu),
        fmaf(sx[11], static_cast<float>(uu >> 24), d))));
    xs += (sx[8] + sx[9]) + (sx[10] + sx[11]);
    uu = wv.w;
    d = fmaf(sx[12], static_cast<float>(uu & 0xFFu),
        fmaf(sx[13], static_cast<float>((uu >> 8) & 0xFFu),
        fmaf(sx[14], static_cast<float>((uu >> 16) & 0xFFu),
        fmaf(sx[15], static_cast<float>(uu >> 24), d))));
    xs += (sx[12] + sx[13]) + (sx[14] + sx[15]);
    const int grp = u >> 1;
    const float sc = __half2float(sc_row[grp]);
    const float bi = __half2float(bi_row[grp]);
    acc = fmaf(sc, d, fmaf(bi, xs, acc));
  }

  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);

  if (kp == 0) {
    const nv_bfloat16 sig = q4mhc_bf16_sigmoid(__float2bfloat16(acc));
    const nv_bfloat16 nv = normed[row];
    s_prod[s][h_sub] =
        __bfloat162float(__float2bfloat16(__bfloat162float(sig) * __bfloat162float(nv)));
  }
  __syncthreads();

  if (s == 0 && lane < 8) {
    const float mean_v = 0.25f * (
        (s_prod[0][lane] + s_prod[1][lane]) +
        (s_prod[2][lane] + s_prod[3][lane]));
    mixed[h0 + lane] = __float2bfloat16(mean_v);
  }
}

// ----------------------------------------------------------------------------
// Kernel 4: qwen4_router_topk_cuda (Fused FP32 softmax + top-k + normalize)
// Replaces 6 separate MLX ops per MoE layer across all 48 layers (M >= 1, E <= 512).
// Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1).
// ----------------------------------------------------------------------------
template <bool IS_FP32>
__global__ __launch_bounds__(256) void qwen4_router_topk_cuda(
    const void* __restrict__ logits,
    unsigned int* __restrict__ out_ids,
    float* __restrict__ out_scores,
    int M,
    int E,
    int top_k) {
  const int m = static_cast<int>(blockIdx.x);
  if (m >= M) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ float s_prob[512];
  __shared__ float s_wv[8];
  __shared__ int s_wi[8];
  __shared__ unsigned int s_top_ids[16];
  __shared__ float s_top_vals[16];

  auto load_logit = [&](int idx) -> float {
    if (idx >= E) return -1e30f;
    const size_t off = static_cast<size_t>(m) * E + idx;
    if constexpr (IS_FP32) {
      return reinterpret_cast<const float*>(logits)[off];
    } else {
      return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(logits)[off]);
    }
  };

  const float v0 = load_logit(tid);
  const float v1 = load_logit(tid + 256);
  float rmax = fmaxf(v0, v1);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, off));
  }
  if (lane == 0) s_wv[warp] = rmax;
  __syncthreads();
  if (warp == 0) {
    float mv = (lane < 8) ? s_wv[lane] : -1e30f;
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, off));
    }
    if (lane == 0) s_wv[0] = mv;
  }
  __syncthreads();
  const float row_max = s_wv[0];

  const float e0 = (tid < E) ? expf(v0 - row_max) : 0.0f;
  const float e1 = (tid + 256 < E) ? expf(v1 - row_max) : 0.0f;
  float rsum = q4mhc_warp_sum(e0 + e1);
  if (lane == 0) s_wv[warp] = rsum;
  __syncthreads();
  if (warp == 0) {
    float sv = (lane < 8) ? s_wv[lane] : 0.0f;
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      sv += __shfl_xor_sync(0xffffffffu, sv, off);
    }
    if (lane == 0) s_wv[0] = (sv > 0.0f) ? (1.0f / sv) : 0.0f;
  }
  __syncthreads();
  const float inv_sum = s_wv[0];

  s_prob[tid] = (tid < E) ? (e0 * inv_sum) : -1.0f;
  s_prob[tid + 256] = (tid + 256 < E) ? (e1 * inv_sum) : -1.0f;
  __syncthreads();

  for (int k = 0; k < top_k; ++k) {
    const float p0 = s_prob[tid];
    const float p1 = s_prob[tid + 256];
    float bv = (p1 > p0) ? p1 : p0;
    int bi = (p1 > p0) ? (tid + 256) : tid;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      s_wv[warp] = bv;
      s_wi[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
      float wv = (lane < 8) ? s_wv[lane] : -2.0f;
      int wi = (lane < 8) ? s_wi[lane] : E;
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, wv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, wi, off);
        if (ov > wv || (ov == wv && oi < wi)) {
          wv = ov;
          wi = oi;
        }
      }
      if (lane == 0) {
        s_top_ids[k] = static_cast<unsigned int>(wi);
        s_top_vals[k] = wv;
        s_prob[wi] = -1.0f;
      }
    }
    __syncthreads();
  }

  if (tid < top_k) {
    float denom = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      denom += s_top_vals[k];
    }
    const size_t base = static_cast<size_t>(m) * top_k + tid;
    out_ids[base] = s_top_ids[tid];
    out_scores[base] = (denom > 0.0f) ? (s_top_vals[tid] / denom) : 0.0f;
  }
}

// ----------------------------------------------------------------------------
// Kernel 6: qwen4_prefill_hc_mix_cuda (Prefill M >= 1, W = 10240, H = 2560)
// Fuses sigmoid(up) * normed + 4-stream mean(-2) without materializing [M, 10240]!
// Grid: dim3(1, M), Block: dim3(320) -> 320 threads * 8 bf16 = 2560 cols/token.
// ----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(320) void qwen4_prefill_hc_mix_cuda(
    const nv_bfloat16* __restrict__ up,
    const nv_bfloat16* __restrict__ normed,
    nv_bfloat16* __restrict__ mixed,
    int M) {
  const int m = static_cast<int>(blockIdx.y);
  const int c8 = static_cast<int>(threadIdx.x); // 0..319
  if (m >= M || c8 >= 320) return;

  const uint4* up_u4 = reinterpret_cast<const uint4*>(up + static_cast<size_t>(m) * 10240);
  const uint4* nm_u4 = reinterpret_cast<const uint4*>(normed + static_cast<size_t>(m) * 10240);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint4 uv = up_u4[s * 320 + c8];
    const uint4 nv = nm_u4[s * 320 + c8];
    const __nv_bfloat162* up2 = reinterpret_cast<const __nv_bfloat162*>(&uv);
    const __nv_bfloat162* nm2 = reinterpret_cast<const __nv_bfloat162*>(&nv);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 uf = __bfloat1622float2(up2[k]);
      const float2 nf = __bfloat1622float2(nm2[k]);
      const float sx = 1.0f / (1.0f + expf(-uf.x));
      const float sy = 1.0f / (1.0f + expf(-uf.y));
      acc[2 * k]     += __bfloat162float(__float2bfloat16(sx * nf.x));
      acc[2 * k + 1] += __bfloat162float(__float2bfloat16(sy * nf.y));
    }
  }

  uint4 out_v;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out_v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    op[k] = __floats2bfloat162_rn(acc[2 * k] * 0.25f, acc[2 * k + 1] * 0.25f);
  }
  reinterpret_cast<uint4*>(mixed + static_cast<size_t>(m) * 2560)[c8] = out_v;
}

// ----------------------------------------------------------------------------
// Kernel 7: qwen4_stream_mean_cuda (HiddenTaps stream_mean, W = 10240 -> H = 2560)
// Exact stream-order (0, 1, 2, 3) FP32 sum / 4 -> BF16 in 1 vectorized launch!
// Grid: dim3(1, M), Block: dim3(320).
// ----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(320) void qwen4_stream_mean_cuda(
    const nv_bfloat16* __restrict__ x,
    nv_bfloat16* __restrict__ out,
    int M) {
  const int m = static_cast<int>(blockIdx.y);
  const int c8 = static_cast<int>(threadIdx.x);
  if (m >= M || c8 >= 320) return;

  const uint4* x_u4 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * 10240);
  const uint4 v0 = x_u4[0 * 320 + c8];
  const uint4 v1 = x_u4[1 * 320 + c8];
  const uint4 v2 = x_u4[2 * 320 + c8];
  const uint4 v3 = x_u4[3 * 320 + c8];
  const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
  const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&v2);
  const __nv_bfloat162* p3 = reinterpret_cast<const __nv_bfloat162*>(&v3);

  uint4 out_v;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out_v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f0 = __bfloat1622float2(p0[k]);
    const float2 f1 = __bfloat1622float2(p1[k]);
    const float2 f2 = __bfloat1622float2(p2[k]);
    const float2 f3 = __bfloat1622float2(p3[k]);
    const float sx = (((f0.x + f1.x) + f2.x) + f3.x) * 0.25f;
    const float sy = (((f0.y + f1.y) + f2.y) + f3.y) * 0.25f;
    op[k] = __floats2bfloat162_rn(sx, sy);
  }
  reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560)[c8] = out_v;
}

// ----------------------------------------------------------------------------
// Kernel 8: qwen4_dequant_q8_bf16_cuda (1-pass Affine Q8_0 g32 -> BF16)
// Eliminates the intermediate FP16 allocation + astype(BF16) copy on cold prefill!
// SB_DTYPE: 0 = float16, 1 = bfloat16, 2 = float32.
// Grid: dim3((K/8 + 255) / 256, N), Block: dim3(256).
// ----------------------------------------------------------------------------
template <int SB_DTYPE>
__global__ __launch_bounds__(256) void qwen4_dequant_q8_bf16_cuda(
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    nv_bfloat16* __restrict__ out,
    int N,
    int K) {
  const int row = static_cast<int>(blockIdx.y);
  const int c8 = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
  const int k8 = K >> 3;
  if (row >= N || c8 >= k8) return;

  const int groups = K >> 5;
  const int grp = c8 >> 2;
  const size_t sb_idx = static_cast<size_t>(row) * groups + grp;
  float sc, bi;
  if constexpr (SB_DTYPE == 0) {
    sc = __half2float(static_cast<const __half*>(scales)[sb_idx]);
    bi = __half2float(static_cast<const __half*>(biases)[sb_idx]);
  } else if constexpr (SB_DTYPE == 1) {
    sc = __bfloat162float(static_cast<const nv_bfloat16*>(scales)[sb_idx]);
    bi = __bfloat162float(static_cast<const nv_bfloat16*>(biases)[sb_idx]);
  } else {
    sc = static_cast<const float*>(scales)[sb_idx];
    bi = static_cast<const float*>(biases)[sb_idx];
  }

  const uint2 w2 = reinterpret_cast<const uint2*>(
      w + static_cast<size_t>(row) * (K >> 2))[c8];
  const unsigned int u0 = w2.x;
  const unsigned int u1 = w2.y;
  uint4 out_v;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out_v);
  op[0] = __floats2bfloat162_rn(
      fmaf(sc, static_cast<float>(u0 & 0xFFu), bi),
      fmaf(sc, static_cast<float>((u0 >> 8) & 0xFFu), bi));
  op[1] = __floats2bfloat162_rn(
      fmaf(sc, static_cast<float>((u0 >> 16) & 0xFFu), bi),
      fmaf(sc, static_cast<float>(u0 >> 24), bi));
  op[2] = __floats2bfloat162_rn(
      fmaf(sc, static_cast<float>(u1 & 0xFFu), bi),
      fmaf(sc, static_cast<float>((u1 >> 8) & 0xFFu), bi));
  op[3] = __floats2bfloat162_rn(
      fmaf(sc, static_cast<float>((u1 >> 16) & 0xFFu), bi),
      fmaf(sc, static_cast<float>(u1 >> 24), bi));
  reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * K)[c8] = out_v;
}
)cuda"
