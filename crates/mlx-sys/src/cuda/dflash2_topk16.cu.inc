R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ bool topk16_beats(
    float value, unsigned int token, float other, unsigned int other_token) {
  return value > other || (value == other && token < other_token);
}

__device__ __forceinline__ void topk16_insert(
    float* values, unsigned int* ids, float value, unsigned int token) {
#pragma unroll
  for (int slot = 15; slot > 0; --slot) {
    const bool here = topk16_beats(value, token, values[slot], ids[slot]);
    const bool above = topk16_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  const bool top = topk16_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

__device__ __forceinline__ void topk16_pop(float* values, unsigned int* ids) {
#pragma unroll
  for (int slot = 0; slot < 15; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[15] = -__int_as_float(0x7f800000);
  ids[15] = 0xffffffffu;
}

extern "C" __global__ __launch_bounds__(256) void mlx_dflash2_topk16_cuda_kernel(
    const unsigned char* logits, int dtype, int vocab,
    int* out_ids, float* out_values) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  float values[16];
  unsigned int ids[16];
  const float negative_inf = -__int_as_float(0x7f800000);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = negative_inf;
    ids[i] = 0xffffffffu;
  }
  for (int token = tid; token < vocab; token += 256) {
    float value;
    if (dtype == 0) {
      value = __bfloat162float(
          reinterpret_cast<const nv_bfloat16*>(logits)[row * vocab + token]);
    } else if (dtype == 1) {
      value = __half2float(
          reinterpret_cast<const __half*>(logits)[row * vocab + token]);
    } else {
      value = reinterpret_cast<const float*>(logits)[row * vocab + token];
    }
    if (topk16_beats(value, token, values[15], ids[15])) {
      topk16_insert(values, ids, value, token);
    }
  }

  __shared__ float warp_values[8];
  __shared__ unsigned int warp_ids[8];
  for (int rank = 0; rank < 16; ++rank) {
    float best = values[0];
    unsigned int best_id = ids[0];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, best, offset);
      const unsigned int other_id = __shfl_xor_sync(0xffffffffu, best_id, offset);
      if (topk16_beats(other, other_id, best, best_id)) {
        best = other;
        best_id = other_id;
      }
    }
    if (lane == 0) {
      warp_values[warp] = best;
      warp_ids[warp] = best_id;
    }
    __syncthreads();
    best = warp_values[0];
    best_id = warp_ids[0];
#pragma unroll
    for (int other = 1; other < 8; ++other) {
      if (topk16_beats(warp_values[other], warp_ids[other], best, best_id)) {
        best = warp_values[other];
        best_id = warp_ids[other];
      }
    }
    if (tid == rank) {
      out_ids[row * 16 + 15 - rank] = static_cast<int>(best_id);
      out_values[row * 16 + 15 - rank] = best;
    }
    if (values[0] == best && ids[0] == best_id) {
      topk16_pop(values, ids);
    }
    __syncthreads();
  }
}
)cuda"
