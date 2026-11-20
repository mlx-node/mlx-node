R"cuda(
#include <cuda_bf16.h>

// Direct BF16 -> BF16 small-row (M in 2..8) kernel for affine 8-bit group_size=32
// with FP32 scales and biases (GGUF Q8_0). Vectorized uchar4/uint2 coalesced loads
// (128 bytes/warp per step) avoid BF16<->FP32 casts and qmm_naive.
template <int NUM_WARPS>
__global__ __launch_bounds__(NUM_WARPS * 32, 4) void mlx_node_affine_q8_few(
    const nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ w,
    const float* __restrict__ scales,
    const float* __restrict__ biases,
    nv_bfloat16* __restrict__ out, int M, int N, int K) {
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int groups = K >> 5;
  const unsigned char* w_row = w + (size_t)n * K;
  const float* s_row = scales + (size_t)n * groups;
  const float* b_row = biases + (size_t)n * groups;

  // 8 lanes per 32-value group (4 values per lane) -> 4 groups (128B) per warp.
  const int sub = lane & 7;
  const int g_in_warp = lane >> 3;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int g0 = warp * 4; g0 < groups; g0 += NUM_WARPS * 4) {
    const int g = g0 + g_in_warp;
    if (g < groups) {
      const int col = g * 32 + sub * 4;
      const uchar4 wv = *reinterpret_cast<const uchar4*>(w_row + col);
      const float sc = s_row[g];
      const float bi = b_row[g];
      const float wf0 = fmaf(static_cast<float>(wv.x), sc, bi);
      const float wf1 = fmaf(static_cast<float>(wv.y), sc, bi);
      const float wf2 = fmaf(static_cast<float>(wv.z), sc, bi);
      const float wf3 = fmaf(static_cast<float>(wv.w), sc, bi);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        if (m < M) {
          const uint2 xv =
              *reinterpret_cast<const uint2*>(x + (size_t)m * K + col);
          const __nv_bfloat162* xb =
              reinterpret_cast<const __nv_bfloat162*>(&xv);
          const float2 f0 = __bfloat1622float2(xb[0]);
          const float2 f1 = __bfloat1622float2(xb[1]);
          acc[m] = fmaf(
              f0.x, wf0,
              fmaf(f0.y, wf1, fmaf(f1.x, wf2, fmaf(f1.y, wf3, acc[m]))));
        }
      }
    }
  }

  __shared__ float smem[NUM_WARPS][8];
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    if (m < M) {
      float v = acc[m];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
      }
      if (lane == 0) {
        smem[warp][m] = v;
      }
    }
  }
  __syncthreads();
  if (warp == 0 && lane < M) {
    float sum = 0.0f;
#pragma unroll
    for (int ww = 0; ww < NUM_WARPS; ++ww) {
      sum += smem[ww][lane];
    }
    out[(size_t)lane * N + n] = __float2bfloat16(sum);
  }
}
)cuda"