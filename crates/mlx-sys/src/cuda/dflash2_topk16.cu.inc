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

__device__ __forceinline__ void topk16_reduce_block(
    float* values, unsigned int* ids,
    int* __restrict__ out_ids, float* __restrict__ out_values) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
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
      out_ids[15 - rank] = static_cast<int>(best_id);
      out_values[15 - rank] = best;
    }
    if (values[0] == best && ids[0] == best_id) {
      topk16_pop(values, ids);
    }
    __syncthreads();
  }
}

// Single-stage fallback for small vocab (<= 8192).
extern "C" __global__ __launch_bounds__(256) void mlx_dflash2_topk16_cuda_kernel(
    const unsigned char* __restrict__ logits, int dtype, int vocab,
    long long row_stride,
    int* __restrict__ out_ids, float* __restrict__ out_values) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(row_stride);
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
          reinterpret_cast<const nv_bfloat16*>(logits)[row_off + token]);
    } else if (dtype == 1) {
      value = __half2float(
          reinterpret_cast<const __half*>(logits)[row_off + token]);
    } else {
      value = reinterpret_cast<const float*>(logits)[row_off + token];
    }
    if (topk16_beats(value, token, values[15], ids[15])) {
      topk16_insert(values, ids, value, token);
    }
  }
  topk16_reduce_block(values, ids, out_ids + row * 16, out_values + row * 16);
}

// Stage 1 (Split-V): each CTA scans one 8192-token slice with vectorized loads
// across all SMs, outputting its exact local top-16. Supports zero-copy strided rows.
template <int DTYPE>
__global__ __launch_bounds__(256, 2) void mlx_dflash2_topk16_slice(
    const unsigned char* __restrict__ logits,
    int vocab, long long row_stride, int slice_size, int num_slices,
    int* __restrict__ part_ids, float* __restrict__ part_values) {
  const int slice = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int start = slice * slice_size;
  const int end = min(start + slice_size, vocab);
  const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(row_stride);

  float values[16];
  unsigned int ids[16];
  const float negative_inf = -__int_as_float(0x7f800000);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = negative_inf;
    ids[i] = 0xffffffffu;
  }

  if constexpr (DTYPE == 0) {
    const nv_bfloat16* row_ptr =
        reinterpret_cast<const nv_bfloat16*>(logits) + row_off;
    const unsigned long long addr = reinterpret_cast<unsigned long long>(row_ptr + start);
    const bool can_vec = ((addr & 1ull) == 0);
    const int mis_bytes = static_cast<int>(addr & 15ull);
    const int prologue =
        can_vec ? ((mis_bytes == 0) ? 0 : ((16 - mis_bytes) >> 1)) : (end - start);
    const int vec_start = min(start + prologue, end);
    const int vec_end = can_vec ? (vec_start + ((end - vec_start) & ~7)) : end;
    for (int token = start + tid; token < vec_start; token += 256) {
      const float value = __bfloat162float(row_ptr[token]);
      if (topk16_beats(value, static_cast<unsigned int>(token), values[15], ids[15])) {
        topk16_insert(values, ids, value, static_cast<unsigned int>(token));
      }
    }
    for (int base = vec_start + tid * 8; base < vec_end; base += 256 * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(row_ptr + base);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 f = __bfloat1622float2(pairs[p]);
        const unsigned int t0 = static_cast<unsigned int>(base + p * 2);
        if (topk16_beats(f.x, t0, values[15], ids[15])) {
          topk16_insert(values, ids, f.x, t0);
        }
        const unsigned int t1 = t0 + 1u;
        if (topk16_beats(f.y, t1, values[15], ids[15])) {
          topk16_insert(values, ids, f.y, t1);
        }
      }
    }
    for (int token = vec_end + tid; token < end; token += 256) {
      const float value = __bfloat162float(row_ptr[token]);
      if (topk16_beats(value, static_cast<unsigned int>(token), values[15], ids[15])) {
        topk16_insert(values, ids, value, static_cast<unsigned int>(token));
      }
    }
  } else if constexpr (DTYPE == 1) {
    const __half* row_ptr =
        reinterpret_cast<const __half*>(logits) + row_off;
    const unsigned long long addr = reinterpret_cast<unsigned long long>(row_ptr + start);
    const bool can_vec = ((addr & 1ull) == 0);
    const int mis_bytes = static_cast<int>(addr & 15ull);
    const int prologue =
        can_vec ? ((mis_bytes == 0) ? 0 : ((16 - mis_bytes) >> 1)) : (end - start);
    const int vec_start = min(start + prologue, end);
    const int vec_end = can_vec ? (vec_start + ((end - vec_start) & ~7)) : end;
    for (int token = start + tid; token < vec_start; token += 256) {
      const float value = __half2float(row_ptr[token]);
      if (topk16_beats(value, static_cast<unsigned int>(token), values[15], ids[15])) {
        topk16_insert(values, ids, value, static_cast<unsigned int>(token));
      }
    }
    for (int base = vec_start + tid * 8; base < vec_end; base += 256 * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(row_ptr + base);
      const __half2* pairs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 f = __half22float2(pairs[p]);
        const unsigned int t0 = static_cast<unsigned int>(base + p * 2);
        if (topk16_beats(f.x, t0, values[15], ids[15])) {
          topk16_insert(values, ids, f.x, t0);
        }
        const unsigned int t1 = t0 + 1u;
        if (topk16_beats(f.y, t1, values[15], ids[15])) {
          topk16_insert(values, ids, f.y, t1);
        }
      }
    }
    for (int token = vec_end + tid; token < end; token += 256) {
      const float value = __half2float(row_ptr[token]);
      if (topk16_beats(value, static_cast<unsigned int>(token), values[15], ids[15])) {
        topk16_insert(values, ids, value, static_cast<unsigned int>(token));
      }
    }
  } else {
    const float* row_ptr =
        reinterpret_cast<const float*>(logits) + row_off;
    for (int token = start + tid; token < end; token += 256) {
      const float value = row_ptr[token];
      if (topk16_beats(value, static_cast<unsigned int>(token), values[15], ids[15])) {
        topk16_insert(values, ids, value, static_cast<unsigned int>(token));
      }
    }
  }

  const size_t out_off = ((size_t)row * num_slices + slice) * 16;
  topk16_reduce_block(values, ids, part_ids + out_off, part_values + out_off);
}

// Stage 2 (Merge): merges num_slices * 16 slice winners per row into exact top-16.
extern "C" __global__ __launch_bounds__(256) void mlx_dflash2_topk16_merge(
    const int* __restrict__ part_ids,
    const float* __restrict__ part_values,
    int total_candidates,
    int* __restrict__ out_ids,
    float* __restrict__ out_values) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t row_off = (size_t)row * total_candidates;
  float values[16];
  unsigned int ids[16];
  const float negative_inf = -__int_as_float(0x7f800000);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = negative_inf;
    ids[i] = 0xffffffffu;
  }
  for (int idx = tid; idx < total_candidates; idx += 256) {
    const unsigned int token = static_cast<unsigned int>(part_ids[row_off + idx]);
    const float value = part_values[row_off + idx];
    if (token != 0xffffffffu && topk16_beats(value, token, values[15], ids[15])) {
      topk16_insert(values, ids, value, token);
    }
  }
  topk16_reduce_block(values, ids, out_ids + row * 16, out_values + row * 16);
}
)cuda"
