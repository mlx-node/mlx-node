R"cuda(
#include <cuda_bf16.h>

// Bandwidth-optimal BF16 GEMM for small M (1..=16), large K (multiple of 8), large N.
// Computes Y[M, N] = X[M, K] @ W[N, K]^T in FP32 accumulation, BF16 output.
// Grid: dim3((N + 7) / 8), Block: dim3(32, 4). Each warp computes 2 output rows of W.
template <int M_ROWS>
__global__ __launch_bounds__(128, 4) void mlx_node_bf16_gemm_few(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    nv_bfloat16* __restrict__ Y,
    int N, int K) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int n_base = (blockIdx.x * 4 + warp) * 2;
  if (n_base >= N) {
    return;
  }
  const int n0 = n_base;
  const int n1 = (n_base + 1 < N) ? (n_base + 1) : n_base;
  const bool has_n1 = (n_base + 1 < N);

  const uint4* w0 = reinterpret_cast<const uint4*>(W + (long long)n0 * K);
  const uint4* w1 = reinterpret_cast<const uint4*>(W + (long long)n1 * K);
  const int k_vec = K >> 3; // K / 8 (each uint4 holds 8 bf16 = 4 bf16x2)

  float acc0[M_ROWS];
  float acc1[M_ROWS];
#pragma unroll
  for (int m = 0; m < M_ROWS; ++m) {
    acc0[m] = 0.0f;
    acc1[m] = 0.0f;
  }

  for (int kv = lane; kv < k_vec; kv += 32) {
    const uint4 wv0 = w0[kv];
    const uint4 wv1 = has_n1 ? w1[kv] : make_uint4(0, 0, 0, 0);
    const __nv_bfloat162* wb0 = reinterpret_cast<const __nv_bfloat162*>(&wv0);
    const __nv_bfloat162* wb1 = reinterpret_cast<const __nv_bfloat162*>(&wv1);

    float wf0[8];
    float wf1[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f0 = __bfloat1622float2(wb0[i]);
      wf0[i * 2] = f0.x;
      wf0[i * 2 + 1] = f0.y;
      if (has_n1) {
        float2 f1 = __bfloat1622float2(wb1[i]);
        wf1[i * 2] = f1.x;
        wf1[i * 2 + 1] = f1.y;
      }
    }

#pragma unroll
    for (int m = 0; m < M_ROWS; ++m) {
      const uint4 xv = reinterpret_cast<const uint4*>(X + (long long)m * K)[kv];
      const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
      float s0 = 0.0f;
      float s1 = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 fx = __bfloat1622float2(xb[i]);
        s0 = fmaf(fx.x, wf0[i * 2], s0);
        s0 = fmaf(fx.y, wf0[i * 2 + 1], s0);
        if (has_n1) {
          s1 = fmaf(fx.x, wf1[i * 2], s1);
          s1 = fmaf(fx.y, wf1[i * 2 + 1], s1);
        }
      }
      acc0[m] += s0;
      if (has_n1) {
        acc1[m] += s1;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M_ROWS; ++m) {
    float v0 = acc0[m];
    float v1 = acc1[m];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v0 += __shfl_xor_sync(0xffffffffu, v0, offset);
      if (has_n1) {
        v1 += __shfl_xor_sync(0xffffffffu, v1, offset);
      }
    }
    if (lane == 0) {
      Y[(long long)m * N + n0] = __float2bfloat16(v0);
      if (has_n1) {
        Y[(long long)m * N + n1] = __float2bfloat16(v1);
      }
    }
  }
}
)cuda"
