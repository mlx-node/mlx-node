R"cuda(
#include <cuda_bf16.h>

// Register-resident single-block-per-head GDN prefill recurrence with optional
// fused strided-z RMSNormGated epilogue (supports both SiLU and Qwen4 Sigmoid gates,
// BF16 or FP32 norm_weight, and BF16 or FP32 beta):
// Holds 64 FP32 state elements per thread across the entire chunk (1 float4
// state read + 1 float4 state write per chunk), software-pipelining STEPS=32
// tokens of q/k/v/g/beta into shared memory with 1-hop warp shuffle reductions.
// When FUSE_GATED_NORM=true, each token's BF16 recurrence output y_bf is staged
// in-place into vs[tt][row] (0 syncs inside the recurrence loop), then all 8
// warps process the chunk's 32 tokens in parallel using pure warp-shuffle
// RMSNorm + strided-z gating and coalesced writes to [B, T, hv * 128].
//
// Shapes (row-contiguous except z which may be strided along B/T with stride(-1)==1,
// B>=1, DK=DV=128):
//   q, k:        [B, T, hk, 128] bf16
//   v:           [B, T, hv, 128] bf16
//   g:           [B, T, hv]      f32
//   beta:        [B, T, hv]      bf16 or f32 (template BETA_BF16)
//   state_in:    [B, hv, 128, 128] f32
//   z:           [B, T, hv * 128] bf16 (used when FUSE_GATED_NORM=true)
//   norm_weight: [128] bf16 or f32 (w_is_f32 flag, used when FUSE_GATED_NORM=true)
//   y_out:       [B, T, hv * 128] (or [B, T, hv, 128]) bf16
//   state_out:   [B, hv, 128, 128] f32
// Grid: dim3(hv, B, 1), Block: dim3(256, 1, 1) -> 128 dv rows/block, 2 threads/row.
namespace gdn_pf {

constexpr int DK = 128;
constexpr int DV = 128;
constexpr int ROWS = 128;
constexpr int STEPS = 32;
constexpr int THREADS = 2 * ROWS; // 256

__device__ __forceinline__ float4 widen_bf16x4(uint2 w) {
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
  return make_float4(
      __low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

template <bool SIGMOID_GATE>
__device__ __forceinline__ nv_bfloat16 gate_norm_elem(
    float yv, float wv, float zval, float rms_factor) {
  if constexpr (SIGMOID_GATE) {
    // rms_factor is denom = sqrtf(mean_sq + eps); matches qwen4_exp exact boundary.
    const nv_bfloat16 normed = __float2bfloat16_rn((yv / rms_factor) * wv);
    const float sig_y = 1.0f / (1.0f + expf(fabsf(zval)));
    const float sig = (zval < 0.0f) ? sig_y : (1.0f - sig_y);
    return __float2bfloat16_rn(sig * __bfloat162float(normed));
  } else {
    // rms_factor is inv_rms = rsqrtf(mean_sq + eps); matches qwen3.5 SiLU boundary.
    const nv_bfloat16 sc = __float2bfloat16_rn(yv * rms_factor);
    const nv_bfloat16 normed = __float2bfloat16_rn(wv * __bfloat162float(sc));
    const float sig_y = 1.0f / (1.0f + expf(fabsf(zval)));
    const nv_bfloat16 sig =
        __float2bfloat16_rn(zval < 0.0f ? sig_y : (1.0f - sig_y));
    const nv_bfloat16 silu =
        __float2bfloat16_rn(zval * __bfloat162float(sig));
    return __float2bfloat16_rn(
        __bfloat162float(silu) * __bfloat162float(normed));
  }
}

template <bool BETA_BF16>
struct Prefetch {
  static constexpr int KQ = STEPS * (DK / 4) / THREADS; // 4
  static constexpr int VV = STEPS * ROWS / 8 / THREADS; // 2
  uint2 kr[KQ];
  uint2 qr[KQ];
  uint4 vr[VV];
  float gr;
  float br;

  __device__ __forceinline__ void fetch(
      const nv_bfloat16* __restrict__ q,
      const nv_bfloat16* __restrict__ k,
      const nv_bfloat16* __restrict__ v,
      const float* __restrict__ g,
      const void* __restrict__ beta,
      int t0, int W, int hk, int hv, int head, int key_head) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int it = 0; it < KQ; ++it) {
      const int i = tid + it * THREADS;
      const int st = i >> 5;
      const int c = i & 31;
      const int t = min(t0 + st, W - 1);
      const size_t at = ((size_t)t * hk + key_head) * DK + 4 * c;
      kr[it] = *reinterpret_cast<const uint2*>(k + at);
      qr[it] = *reinterpret_cast<const uint2*>(q + at);
    }
#pragma unroll
    for (int it = 0; it < VV; ++it) {
      const int i = tid + it * THREADS;
      const int st = i >> 4;
      const int c = i & 15;
      const int t = min(t0 + st, W - 1);
      const size_t at = ((size_t)t * hv + head) * DV + 8 * c;
      vr[it] = *reinterpret_cast<const uint4*>(v + at);
    }
    if (tid < STEPS) {
      const size_t at = (size_t)min(t0 + tid, W - 1) * hv + head;
      gr = g[at];
      if constexpr (BETA_BF16) {
        br = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(beta)[at]);
      } else {
        br = reinterpret_cast<const float*>(beta)[at];
      }
    }
  }

  __device__ __forceinline__ void store(
      float4 (*ks)[DK / 4], float4 (*qs)[DK / 4],
      float* gs, float* bs, nv_bfloat16 (*vs)[ROWS]) const {
    const int tid = threadIdx.x;
#pragma unroll
    for (int it = 0; it < KQ; ++it) {
      const int i = tid + it * THREADS;
      ks[i >> 5][i & 31] = widen_bf16x4(kr[it]);
      qs[i >> 5][i & 31] = widen_bf16x4(qr[it]);
    }
#pragma unroll
    for (int it = 0; it < VV; ++it) {
      const int i = tid + it * THREADS;
      *reinterpret_cast<uint4*>(&vs[i >> 4][8 * (i & 15)]) = vr[it];
    }
    if (tid < STEPS) {
      gs[tid] = gr;
      bs[tid] = br;
    }
  }
};

} // namespace gdn_pf

template <bool BETA_BF16, bool FUSE_GATED_NORM, bool SIGMOID_GATE = false>
__global__ __launch_bounds__(256) void mlx_gdn_prefill_chain_cuda_kernel(
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const void* __restrict__ beta,
    const float* state_in,
    const nv_bfloat16* __restrict__ z,
    const void* __restrict__ norm_weight,
    nv_bfloat16* __restrict__ y_out,
    float* state_out,
    int W, int hk, int hv, int tiled_heads,
    long long z_stride_b, long long z_stride_t, float eps, int w_is_f32 = 0) {
  using namespace gdn_pf;
  // Static SMEM: 16384 + 16384 + 128 + 128 + 8192 + 512 = 41728 bytes (<= 48 KB).
  __shared__ float4 ks[STEPS][DK / 4];
  __shared__ float4 qs[STEPS][DK / 4];
  __shared__ float gs[STEPS];
  __shared__ float bs[STEPS];
  __shared__ __align__(16) nv_bfloat16 vs[STEPS][ROWS];
  [[maybe_unused]] __shared__ __align__(16) float ws[DV];

  const int head = blockIdx.x;
  const int b = blockIdx.y;
  const int row = threadIdx.x >> 1;
  const int half = threadIdx.x & 1;
  const int key_head = tiled_heads ? (head % hk) : (head / (hv / hk));

  const size_t qk_b_off = static_cast<size_t>(b) * W * hk * DK;
  const size_t v_b_off = static_cast<size_t>(b) * W * hv * DV;
  const size_t gb_b_off = static_cast<size_t>(b) * W * hv;
  q += qk_b_off;
  k += qk_b_off;
  v += v_b_off;
  g += gb_b_off;
  if constexpr (BETA_BF16) {
    beta = reinterpret_cast<const nv_bfloat16*>(beta) + gb_b_off;
  } else {
    beta = reinterpret_cast<const float*>(beta) + gb_b_off;
  }
  y_out += v_b_off;

  if constexpr (FUSE_GATED_NORM) {
    if (threadIdx.x < DV) {
      ws[threadIdx.x] = w_is_f32
          ? reinterpret_cast<const float*>(norm_weight)[threadIdx.x]
          : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(norm_weight)[threadIdx.x]);
    }
  }

  float s[64];
  const float* s0 = state_in + (((size_t)b * hv + head) * DV + row) * DK;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
    s[4 * j + 0] = t.x;
    s[4 * j + 1] = t.y;
    s[4 * j + 2] = t.z;
    s[4 * j + 3] = t.w;
  }

  Prefetch<BETA_BF16> pf;
  pf.fetch(q, k, v, g, beta, 0, W, hk, hv, head, key_head);
  for (int t0 = 0; t0 < W; t0 += STEPS) {
    const int n = min(STEPS, W - t0);
    __syncthreads();
    pf.store(ks, qs, gs, bs, vs);
    __syncthreads();
    if (t0 + STEPS < W) {
      pf.fetch(q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head);
    }
    for (int tt = 0; tt < n; ++tt) {
      const float gt = gs[tt];
      const float bt = bs[tt];
      const float vt = __bfloat162float(vs[tt][row]);
      float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float4 kk = ks[tt][2 * j + half];
        s[4 * j + 0] *= gt;
        s[4 * j + 1] *= gt;
        s[4 * j + 2] *= gt;
        s[4 * j + 3] *= gt;
        m0 = __fmaf_rn(s[4 * j + 0], kk.x, m0);
        m1 = __fmaf_rn(s[4 * j + 1], kk.y, m1);
        m2 = __fmaf_rn(s[4 * j + 2], kk.z, m2);
        m3 = __fmaf_rn(s[4 * j + 3], kk.w, m3);
      }
      float mem = (m0 + m1) + (m2 + m3);
      mem += __shfl_xor_sync(0xffffffffu, mem, 1);
      const float delta = (vt - mem) * bt;
      float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float4 kk = ks[tt][2 * j + half];
        const float4 qq = qs[tt][2 * j + half];
        s[4 * j + 0] = __fmaf_rn(kk.x, delta, s[4 * j + 0]);
        s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
        s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
        s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
        o0 = __fmaf_rn(s[4 * j + 0], qq.x, o0);
        o1 = __fmaf_rn(s[4 * j + 1], qq.y, o1);
        o2 = __fmaf_rn(s[4 * j + 2], qq.z, o2);
        o3 = __fmaf_rn(s[4 * j + 3], qq.w, o3);
      }
      float out = (o0 + o1) + (o2 + o3);
      out += __shfl_xor_sync(0xffffffffu, out, 1);
      if (half == 0) {
        const nv_bfloat16 y_bf = __float2bfloat16_rn(out);
        if constexpr (FUSE_GATED_NORM) {
          // Reuse vs[tt][row] in place: both half=0 and half=1 in this warp
          // have already consumed vs[tt][row] before the shuffle above.
          vs[tt][row] = y_bf;
        } else {
          const size_t yat = ((size_t)(t0 + tt) * hv + head) * DV + row;
          y_out[yat] = y_bf;
        }
      }
    }
    if constexpr (FUSE_GATED_NORM) {
      __syncthreads();
      const int warp = threadIdx.x >> 5;
      const int lane = threadIdx.x & 31;
      const nv_bfloat16* const z_b = z + static_cast<long long>(b) * z_stride_b;
      const bool z_vec4 =
          (((reinterpret_cast<unsigned long long>(z_b) |
             (unsigned long long)(z_stride_t * 2)) & 7ull) == 0);
      // Issue all 4 tokens' z loads for this warp up front (4x MLP) while
      // keeping ws[4*lane..4*lane+3] in registers.
      uint2 zv[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int tt = warp + m * 8;
        if (tt < n) {
          const long long z_row =
              (long long)(t0 + tt) * z_stride_t + (long long)head * DV;
          if (z_vec4) {
            zv[m] = reinterpret_cast<const uint2*>(z_b + z_row)[lane];
          } else {
            nv_bfloat16 tmp[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              tmp[k] = z_b[z_row + 4 * lane + k];
            }
            zv[m] = *reinterpret_cast<const uint2*>(tmp);
          }
        }
      }
      const float4 wf = reinterpret_cast<const float4*>(ws)[lane];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int tt = warp + m * 8;
        if (tt < n) {
          const float4 yf = widen_bf16x4(reinterpret_cast<const uint2*>(vs[tt])[lane]);
          float sq = (yf.x * yf.x + yf.y * yf.y) + (yf.z * yf.z + yf.w * yf.w);
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, off);
          }
          const float rms_factor = SIGMOID_GATE
              ? sqrtf(sq * (1.0f / 128.0f) + eps)
              : rsqrtf(sq * (1.0f / 128.0f) + eps);
          const float4 zf = widen_bf16x4(zv[m]);
          nv_bfloat16 out4[4];
          out4[0] = gate_norm_elem<SIGMOID_GATE>(yf.x, wf.x, zf.x, rms_factor);
          out4[1] = gate_norm_elem<SIGMOID_GATE>(yf.y, wf.y, zf.y, rms_factor);
          out4[2] = gate_norm_elem<SIGMOID_GATE>(yf.z, wf.z, zf.z, rms_factor);
          out4[3] = gate_norm_elem<SIGMOID_GATE>(yf.w, wf.w, zf.w, rms_factor);
          const size_t y_row = ((size_t)(t0 + tt) * hv + head) * DV;
          reinterpret_cast<uint2*>(y_out + y_row)[lane] =
              *reinterpret_cast<const uint2*>(out4);
        }
      }
    }
  }

  float* s1 = state_out + (((size_t)b * hv + head) * DV + row) * DK;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) =
        make_float4(s[4 * j + 0], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
  }
}
)cuda"
