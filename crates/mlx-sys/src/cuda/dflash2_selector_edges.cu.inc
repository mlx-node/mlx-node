R"cuda(
#include <cuda_bf16.h>

// Fused DFlash2 selector edge scoring for fixed K=16, Rank=256.
// Replaces 8 unfused CUDA kernels (3x take, concat, mul, batched matmul, add, astype)
// with a single launch. Grid: dim3(length), Block: dim3(256) (8 warps per position).
// Matches the native BF16 graph rounding (BF16 prod -> FP32 dot -> BF16 + unary -> FP32)
// when `softcap == 0.0f` (bit-identical for Qwen3.8). When `softcap > 0.0f`, keeps
// `float sum = unary + edge` in FP32 before `softcap * tanhf(sum / softcap)` (matching
// Splash `sampling.metal:1670-1675`, preserving `-INFINITY`).
extern "C" __global__ __launch_bounds__(256) void mlx_dflash2_selector_edges_cuda_kernel(
    const int* __restrict__ candidates,
    const unsigned char* __restrict__ unary_raw,
    const nv_bfloat16* __restrict__ hidden,
    const nv_bfloat16* __restrict__ predecessor,
    const nv_bfloat16* __restrict__ successor,
    const int* __restrict__ anchor,
    float* __restrict__ scores,
    int vocab,
    int unary_is_bf16,
    float softcap) {
  constexpr int K = 16;
  constexpr int Rank = 256;
  const int pos = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Stage projected hidden[pos, 0..255] and all 16 successor embeddings [16, 256] in SMEM!
  // 256*2B (hidden) + 16*256*2B (successors) = 8704 B total (tiny!).
  __shared__ nv_bfloat16 s_h[Rank];
  __shared__ nv_bfloat16 s_succ[K][Rank];

  s_h[tid] = hidden[(size_t)pos * Rank + tid];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    int cid = candidates[pos * K + j];
    if (cid < 0) cid = 0;
    if (cid >= vocab) cid = vocab - 1;
    s_succ[j][tid] = successor[(size_t)cid * Rank + tid];
  }
  __syncthreads();

  const int num_prev = (pos > 0) ? K : 1;
  for (int prev = warp; prev < num_prev; prev += 8) {
    int pid = (pos > 0) ? candidates[(pos - 1) * K + prev] : anchor[0];
    if (pid < 0) pid = 0;
    if (pid >= vocab) pid = vocab - 1;

    // 8 dims per lane across 32 lanes = 256 dims.
    float ctx[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      const nv_bfloat16 prod = __float2bfloat16_rn(
          __bfloat162float(predecessor[(size_t)pid * Rank + d]) *
          __bfloat162float(s_h[d]));
      ctx[i] = __bfloat162float(prod);
    }

#pragma unroll
    for (int j = 0; j < K; ++j) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc = fmaf(ctx[i], __bfloat162float(s_succ[j][lane + i * 32]), acc);
      }
#pragma unroll
      for (int sh = 16; sh >= 1; sh >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, sh, 32);
      }
      if (lane == 0) {
        const float u_val = unary_is_bf16
            ? __bfloat162float(reinterpret_cast<const nv_bfloat16*>(unary_raw)[pos * K + j])
            : reinterpret_cast<const float*>(unary_raw)[pos * K + j];
        const float neg_inf = -__int_as_float(0x7f800000);
        float combined;
        if (softcap > 0.0f) {
          const float sum = u_val + acc;
          combined = (sum > neg_inf) ? (softcap * tanhf(sum / softcap)) : sum;
        } else {
          const float edge_bf = __bfloat162float(__float2bfloat16_rn(acc));
          const float u_bf = unary_is_bf16
              ? u_val
              : __bfloat162float(__float2bfloat16_rn(u_val));
          combined = __bfloat162float(__float2bfloat16_rn(edge_bf + u_bf));
        }
        if (pos > 0) {
          scores[((size_t)pos * K + prev) * K + j] = combined;
        } else {
#pragma unroll
          for (int p = 0; p < K; ++p) {
            scores[(size_t)p * K + j] = combined;
          }
        }
      }
    }
  }
}
)cuda"
