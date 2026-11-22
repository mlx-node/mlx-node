R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Shared device body for fused QK-Norm + Non-Traditional Partial RoPE +
// In-Place KV-Cache Store.
//   - kv_pos:   cache row index written to k_cache/v_cache (offset + t)
//   - rope_pos: position fed into RoPE (offset + t, or offset + depths[t] for tree)
template <bool D256>
__device__ __forceinline__ void qk_norm_rope_kv_store_impl(
    const nv_bfloat16* __restrict__ q_in,
    const nv_bfloat16* __restrict__ k_in,
    const nv_bfloat16* __restrict__ v_in,
    const nv_bfloat16* __restrict__ q_norm_w,
    const nv_bfloat16* __restrict__ k_norm_w,
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_cache,
    nv_bfloat16* __restrict__ v_cache,
    int T, int Hq, int D,
    int rope_dims, float log2_base, float rope_scale, float eps,
    int kv_pos, int rope_pos,
    long long q_sb, long long q_st, long long q_sh,
    long long k_sb, long long k_st, long long k_sh,
    long long v_sb, long long v_st, long long v_sh,
    long long kc_sb, long long kc_sh, long long kc_st,
    long long vc_sb, long long vc_sh, long long vc_st,
    float* s_warp,
    nv_bfloat16* s_buf) {
  const int t = blockIdx.x;
  const int hy = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_q = hy < Hq;

  const nv_bfloat16* src;
  nv_bfloat16* dst;
  const nv_bfloat16* norm_w;
  const nv_bfloat16* v_src = nullptr;
  nv_bfloat16* v_dst = nullptr;

  if (is_q) {
    const int hq = hy;
    src = q_in + (long long)b * q_sb + (long long)t * q_st + (long long)hq * q_sh;
    dst = q_out + (((long long)b * Hq + hq) * T + t) * (D256 ? 256 : D);
    norm_w = q_norm_w;
  } else {
    const int hkv = hy - Hq;
    src = k_in + (long long)b * k_sb + (long long)t * k_st + (long long)hkv * k_sh;
    dst = k_cache + (long long)b * kc_sb + (long long)hkv * kc_sh +
        (long long)kv_pos * kc_st;
    norm_w = k_norm_w;
    v_src = v_in + (long long)b * v_sb + (long long)t * v_st + (long long)hkv * v_sh;
    v_dst = v_cache + (long long)b * vc_sb + (long long)hkv * vc_sh +
        (long long)kv_pos * vc_st;
  }

  const float pos_scaled = rope_scale * static_cast<float>(rope_pos);

  if constexpr (D256) {
    // Overlap V vectorized load with K RMSNorm reduction.
    uint4 v_reg;
    const bool v_vec16 =
        (!is_q) &&
        (((reinterpret_cast<unsigned long long>(v_src) |
           reinterpret_cast<unsigned long long>(v_dst)) & 15ull) == 0);
    if (v_vec16 && tid < 32) {
      v_reg = reinterpret_cast<const uint4*>(v_src)[tid];
    }

    float x0, x1;
    if ((reinterpret_cast<unsigned long long>(src) & 3ull) == 0) {
      const __nv_bfloat162 xv = reinterpret_cast<const __nv_bfloat162*>(src)[tid];
      x0 = __low2float(xv);
      x1 = __high2float(xv);
    } else {
      x0 = __bfloat162float(src[2 * tid]);
      x1 = __bfloat162float(src[2 * tid + 1]);
    }

    float sq = x0 * x0 + x1 * x1;
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, shift);
    }
    if (lane == 0) {
      s_warp[warp] = sq;
    }
    __syncthreads();

    const float sum_sq = (s_warp[0] + s_warp[1]) + (s_warp[2] + s_warp[3]);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / 256.0f) + eps);

    const __nv_bfloat162 wv = reinterpret_cast<const __nv_bfloat162*>(norm_w)[tid];
    const nv_bfloat16 sc0 = __float2bfloat16_rn(x0 * inv_rms);
    const nv_bfloat16 sc1 = __float2bfloat16_rn(x1 * inv_rms);
    const nv_bfloat16 n0 = __float2bfloat16_rn(__low2float(wv) * __bfloat162float(sc0));
    const nv_bfloat16 n1 = __float2bfloat16_rn(__high2float(wv) * __bfloat162float(sc1));

    const int d0 = 2 * tid;
    if (d0 < rope_dims) {
      s_buf[d0] = n0;
      s_buf[d0 + 1] = n1;
    } else if ((reinterpret_cast<unsigned long long>(dst) & 3ull) == 0) {
      reinterpret_cast<__nv_bfloat162*>(dst)[tid] = __halves2bfloat162(n0, n1);
    } else {
      dst[d0] = n0;
      dst[d0 + 1] = n1;
    }

    if (rope_dims > 0) {
      __syncthreads();
      const int half_rope = rope_dims >> 1;
      if (tid < half_rope) {
        const float d_frac = static_cast<float>(tid) / static_cast<float>(half_rope);
        const float inv_freq = exp2f(-d_frac * log2_base);
        const float theta = pos_scaled * inv_freq;
        const float costheta = cosf(theta);
        const float sintheta = sinf(theta);
        const float rx1_in = __bfloat162float(s_buf[tid]);
        const float rx2_in = __bfloat162float(s_buf[tid + half_rope]);
        dst[tid] = __float2bfloat16_rn(rx1_in * costheta - rx2_in * sintheta);
        dst[tid + half_rope] = __float2bfloat16_rn(rx1_in * sintheta + rx2_in * costheta);
      }
    }

    if (!is_q) {
      if (v_vec16) {
        if (tid < 32) {
          reinterpret_cast<uint4*>(v_dst)[tid] = v_reg;
        }
      } else {
        v_dst[tid] = v_src[tid];
        v_dst[tid + 128] = v_src[tid + 128];
      }
    }
  } else {
    float sq = 0.0f;
    for (int d = tid; d < D; d += 128) {
      const nv_bfloat16 xv = src[d];
      s_buf[d] = xv;
      const float xf = __bfloat162float(xv);
      sq += xf * xf;
    }
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, shift);
    }
    if (lane == 0) {
      s_warp[warp] = sq;
    }
    __syncthreads();

    const float sum_sq = (s_warp[0] + s_warp[1]) + (s_warp[2] + s_warp[3]);
    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(D) + eps);

    for (int d = tid; d < D; d += 128) {
      const float xf = __bfloat162float(s_buf[d]);
      const nv_bfloat16 sc = __float2bfloat16_rn(xf * inv_rms);
      const nv_bfloat16 n =
          __float2bfloat16_rn(__bfloat162float(norm_w[d]) * __bfloat162float(sc));
      if (d < rope_dims) {
        s_buf[d] = n;
      } else {
        dst[d] = n;
      }
    }

    if (rope_dims > 0) {
      __syncthreads();
      const int half_rope = rope_dims >> 1;
      for (int i = tid; i < half_rope; i += 128) {
        const float d_frac = static_cast<float>(i) / static_cast<float>(half_rope);
        const float inv_freq = exp2f(-d_frac * log2_base);
        const float theta = pos_scaled * inv_freq;
        const float costheta = cosf(theta);
        const float sintheta = sinf(theta);
        const float rx1_in = __bfloat162float(s_buf[i]);
        const float rx2_in = __bfloat162float(s_buf[i + half_rope]);
        dst[i] = __float2bfloat16_rn(rx1_in * costheta - rx2_in * sintheta);
        dst[i + half_rope] = __float2bfloat16_rn(rx1_in * sintheta + rx2_in * costheta);
      }
    }

    if (!is_q) {
      const bool v_vec16 =
          ((D & 7) == 0) &&
          (((reinterpret_cast<unsigned long long>(v_src) |
             reinterpret_cast<unsigned long long>(v_dst)) & 15ull) == 0);
      if (v_vec16) {
        const int nvec = D >> 3;
        for (int i = tid; i < nvec; i += 128) {
          reinterpret_cast<uint4*>(v_dst)[i] = reinterpret_cast<const uint4*>(v_src)[i];
        }
      } else {
        for (int d = tid; d < D; d += 128) {
          v_dst[d] = v_src[d];
        }
      }
    }
  }
}

// Fused QK-Norm + Non-Traditional Partial RoPE + In-Place KV-Cache Store.
// Replaces 6 dispatches (q_norm, k_norm, rope(q), rope(k), kv_store(K), kv_store(V))
// with 1 kernel launch:
//   - blockIdx.y < Hq:  Q RMSNorm + partial RoPE -> q_out[b, hq, t, d] (SDPA layout)
//   - blockIdx.y >= Hq: K RMSNorm + partial RoPE -> k_cache[b, hkv, offset+t, d]
//                       AND V copy -> v_cache[b, hkv, offset+t, d]
//
// Inputs:
//   q_in:     [B, T, Hq, D] bf16 (stride(-1) == 1)
//   k_in:     [B, T, Hkv, D] bf16 (stride(-1) == 1)
//   v_in:     [B, T, Hkv, D] bf16 (stride(-1) == 1)
//   q_norm_w: [D] bf16 (contiguous, 4-byte aligned)
//   k_norm_w: [D] bf16 (contiguous, 4-byte aligned)
// Outputs:
//   q_out:    [B, Hq, T, D] bf16 (row-contiguous)
//   k_cache:  [B, Hkv, cap, D] bf16 (mutated at offset..offset+T, stride(-1) == 1)
//   v_cache:  [B, Hkv, cap, D] bf16 (mutated at offset..offset+T, stride(-1) == 1)
// Grid: dim3(T, Hq + Hkv, B), Block: dim3(128, 1, 1)
template <bool D256>
__global__ __launch_bounds__(128) void mlx_qk_norm_rope_kv_store_cuda_kernel(
    const nv_bfloat16* __restrict__ q_in,
    const nv_bfloat16* __restrict__ k_in,
    const nv_bfloat16* __restrict__ v_in,
    const nv_bfloat16* __restrict__ q_norm_w,
    const nv_bfloat16* __restrict__ k_norm_w,
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_cache,
    nv_bfloat16* __restrict__ v_cache,
    int T, int Hq, int D,
    int rope_dims, float log2_base, float rope_scale, float eps, int offset,
    long long q_sb, long long q_st, long long q_sh,
    long long k_sb, long long k_st, long long k_sh,
    long long v_sb, long long v_st, long long v_sh,
    long long kc_sb, long long kc_sh, long long kc_st,
    long long vc_sb, long long vc_sh, long long vc_st) {
  __shared__ float s_warp[4];
  __shared__ nv_bfloat16 s_buf[D256 ? 256 : 1024];
  const int pos = offset + static_cast<int>(blockIdx.x);
  qk_norm_rope_kv_store_impl<D256>(
      q_in, k_in, v_in, q_norm_w, k_norm_w, q_out, k_cache, v_cache,
      T, Hq, D, rope_dims, log2_base, rope_scale, eps, pos, pos,
      q_sb, q_st, q_sh, k_sb, k_st, k_sh, v_sb, v_st, v_sh,
      kc_sb, kc_sh, kc_st, vc_sb, vc_sh, vc_st, s_warp, s_buf);
}

// Tree-RoPE variant of fused QK-Norm + partial RoPE + in-place KV-Cache Store.
// RoPE position for token t is `offset + depths[t]`, while KV store still
// writes slot `offset + t`.
// Grid: dim3(T, Hq + Hkv, B), Block: dim3(128, 1, 1)
template <bool D256>
__global__ __launch_bounds__(128) void mlx_qk_norm_rope_kv_store_tree_cuda_kernel(
    const nv_bfloat16* __restrict__ q_in,
    const nv_bfloat16* __restrict__ k_in,
    const nv_bfloat16* __restrict__ v_in,
    const nv_bfloat16* __restrict__ q_norm_w,
    const nv_bfloat16* __restrict__ k_norm_w,
    const int* __restrict__ depths,
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_cache,
    nv_bfloat16* __restrict__ v_cache,
    int T, int Hq, int D,
    int rope_dims, float log2_base, float rope_scale, float eps, int offset,
    long long q_sb, long long q_st, long long q_sh,
    long long k_sb, long long k_st, long long k_sh,
    long long v_sb, long long v_st, long long v_sh,
    long long kc_sb, long long kc_sh, long long kc_st,
    long long vc_sb, long long vc_sh, long long vc_st) {
  __shared__ float s_warp[4];
  __shared__ nv_bfloat16 s_buf[D256 ? 256 : 1024];
  const int t = static_cast<int>(blockIdx.x);
  const int kv_pos = offset + t;
  const int rope_pos = offset + depths[t];
  qk_norm_rope_kv_store_impl<D256>(
      q_in, k_in, v_in, q_norm_w, k_norm_w, q_out, k_cache, v_cache,
      T, Hq, D, rope_dims, log2_base, rope_scale, eps, kv_pos, rope_pos,
      q_sb, q_st, q_sh, k_sb, k_st, k_sh, v_sb, v_st, v_sh,
      kc_sb, kc_sh, kc_st, vc_sb, vc_sh, vc_st, s_warp, s_buf);
}

// Race-free in-place KV cache row compaction for tree acceptance (keep <= 16).
// Copies cache[b, h, base_offset + accepted_nodes[r], :] -> cache[b, h, base_offset + r, :]
// for both K and V in a single launch by staging all `keep` rows in registers
// before writing back (handles arbitrary permutations without hazards).
// Fast vectorized path: 16B-aligned rows with nvec = (D * itemsize) / 16 <= 256.
// Grid: dim3(2 * Hkv, B, 1), Block: dim3(nvec, 1, 1)
__global__ void mlx_kv_compact_rows_vec16_cuda_kernel(
    char* __restrict__ k_cache,
    char* __restrict__ v_cache,
    const int* __restrict__ accepted_nodes,
    int Hkv, int cap, int base_offset, int keep,
    long long kc_sb, long long kc_sh, long long kc_st,
    long long vc_sb, long long vc_sh, long long vc_st) {
  __shared__ int s_nodes[16];
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < keep; i += static_cast<int>(blockDim.x)) {
    s_nodes[i] = accepted_nodes[i];
  }
  __syncthreads();

  const int hy = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const bool is_k = hy < Hkv;
  const int h = is_k ? hy : (hy - Hkv);
  char* base_ptr = is_k
      ? (k_cache + (long long)b * kc_sb + (long long)h * kc_sh)
      : (v_cache + (long long)b * vc_sb + (long long)h * vc_sh);
  const long long st = is_k ? kc_st : vc_st;

  uint4 staged[16];
  for (int r = 0; r < keep; ++r) {
    const int src_row = base_offset + s_nodes[r];
    if (src_row >= 0 && src_row < cap) {
      staged[r] = reinterpret_cast<const uint4*>(
          base_ptr + (long long)src_row * st)[tid];
    }
  }
  __syncthreads();
  for (int r = 0; r < keep; ++r) {
    const int node = s_nodes[r];
    const int src_row = base_offset + node;
    if (node != r && src_row >= 0 && src_row < cap) {
      reinterpret_cast<uint4*>(
          base_ptr + (long long)(base_offset + r) * st)[tid] = staged[r];
    }
  }
}

// Generic scalar fallback for in-place KV cache row compaction (keep <= 16).
// Supports itemsize in {1, 2, 4}, arbitrary D, strides, and alignment.
// Grid: dim3(2 * Hkv, B, 1), Block: dim3(256, 1, 1)
__global__ __launch_bounds__(256) void mlx_kv_compact_rows_generic_cuda_kernel(
    char* __restrict__ k_cache,
    char* __restrict__ v_cache,
    const int* __restrict__ accepted_nodes,
    int Hkv, int cap, int D, int itemsize, int base_offset, int keep,
    long long kc_sb, long long kc_sh, long long kc_st, long long kc_sd,
    long long vc_sb, long long vc_sh, long long vc_st, long long vc_sd) {
  __shared__ int s_nodes[16];
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < keep) {
    s_nodes[tid] = accepted_nodes[tid];
  }
  __syncthreads();

  const int hy = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const bool is_k = hy < Hkv;
  const int h = is_k ? hy : (hy - Hkv);
  const long long st = is_k ? kc_st : vc_st;
  const long long sd = is_k ? kc_sd : vc_sd;
  const long long base_elem = is_k
      ? ((long long)b * kc_sb + (long long)h * kc_sh)
      : ((long long)b * vc_sb + (long long)h * vc_sh);
  char* raw = is_k ? k_cache : v_cache;

  for (int chunk = 0; chunk < D; chunk += 256) {
    const int d = chunk + tid;
    unsigned int staged[16];
    if (d < D) {
      for (int r = 0; r < keep; ++r) {
        const int src_row = base_offset + s_nodes[r];
        if (src_row >= 0 && src_row < cap) {
          const long long idx = base_elem + (long long)src_row * st + (long long)d * sd;
          if (itemsize == 2) {
            staged[r] = reinterpret_cast<const unsigned short*>(raw)[idx];
          } else if (itemsize == 4) {
            staged[r] = reinterpret_cast<const unsigned int*>(raw)[idx];
          } else {
            staged[r] = reinterpret_cast<const unsigned char*>(raw)[idx];
          }
        }
      }
    }
    __syncthreads();
    if (d < D) {
      for (int r = 0; r < keep; ++r) {
        const int node = s_nodes[r];
        const int src_row = base_offset + node;
        if (node != r && src_row >= 0 && src_row < cap) {
          const long long dst_idx =
              base_elem + (long long)(base_offset + r) * st + (long long)d * sd;
          if (itemsize == 2) {
            reinterpret_cast<unsigned short*>(raw)[dst_idx] =
                static_cast<unsigned short>(staged[r]);
          } else if (itemsize == 4) {
            reinterpret_cast<unsigned int*>(raw)[dst_idx] = staged[r];
          } else {
            reinterpret_cast<unsigned char*>(raw)[dst_idx] =
                static_cast<unsigned char>(staged[r]);
          }
        }
      }
    }
    __syncthreads();
  }
}
)cuda"
R"cuda(
template <typename T>
__device__ __forceinline__ float qknr_to_f32(T v);

template <>
__device__ __forceinline__ float qknr_to_f32<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float qknr_to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float qknr_to_f32<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T qknr_from_f32(float v);

template <>
__device__ __forceinline__ nv_bfloat16 qknr_from_f32<nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ __half qknr_from_f32<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ float qknr_from_f32<float>(float v) {
  return v;
}

__device__ __forceinline__ void qknr_load2(
    const nv_bfloat16* src, int tid, float& x0, float& x1) {
  if ((reinterpret_cast<unsigned long long>(src) & 3ull) == 0) {
    const __nv_bfloat162 xv = reinterpret_cast<const __nv_bfloat162*>(src)[tid];
    x0 = __low2float(xv);
    x1 = __high2float(xv);
  } else {
    x0 = __bfloat162float(src[2 * tid]);
    x1 = __bfloat162float(src[2 * tid + 1]);
  }
}

__device__ __forceinline__ void qknr_load2(
    const __half* src, int tid, float& x0, float& x1) {
  if ((reinterpret_cast<unsigned long long>(src) & 3ull) == 0) {
    const __half2 xv = reinterpret_cast<const __half2*>(src)[tid];
    x0 = __low2float(xv);
    x1 = __high2float(xv);
  } else {
    x0 = __half2float(src[2 * tid]);
    x1 = __half2float(src[2 * tid + 1]);
  }
}

__device__ __forceinline__ void qknr_load2(
    const float* src, int tid, float& x0, float& x1) {
  if ((reinterpret_cast<unsigned long long>(src) & 7ull) == 0) {
    const float2 xv = reinterpret_cast<const float2*>(src)[tid];
    x0 = xv.x;
    x1 = xv.y;
  } else {
    x0 = src[2 * tid];
    x1 = src[2 * tid + 1];
  }
}

__device__ __forceinline__ void qknr_store2(
    nv_bfloat16* dst, int tid, nv_bfloat16 n0, nv_bfloat16 n1) {
  if ((reinterpret_cast<unsigned long long>(dst) & 3ull) == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst)[tid] = __halves2bfloat162(n0, n1);
  } else {
    dst[2 * tid] = n0;
    dst[2 * tid + 1] = n1;
  }
}

__device__ __forceinline__ void qknr_store2(
    __half* dst, int tid, __half n0, __half n1) {
  if ((reinterpret_cast<unsigned long long>(dst) & 3ull) == 0) {
    reinterpret_cast<__half2*>(dst)[tid] = __halves2half2(n0, n1);
  } else {
    dst[2 * tid] = n0;
    dst[2 * tid + 1] = n1;
  }
}

__device__ __forceinline__ void qknr_store2(
    float* dst, int tid, float n0, float n1) {
  if ((reinterpret_cast<unsigned long long>(dst) & 7ull) == 0) {
    reinterpret_cast<float2*>(dst)[tid] = make_float2(n0, n1);
  } else {
    dst[2 * tid] = n0;
    dst[2 * tid + 1] = n1;
  }
}

// Standalone fused QK-Norm + Non-Traditional (half-split) RoPE, matching
// Metal's `mlx_qk_norm_rope` semantics:
//   Inputs:
//     q_in:     [B, T, Hq, D] (stride(-1) == 1)
//     k_in:     [B, T, Hk, D] (stride(-1) == 1)
//     q_norm_w: [D] contiguous RMSNorm weight (plain `w`, not `1 + w`)
//     k_norm_w: [D] contiguous RMSNorm weight
//     offsets:  [B] int32 device array (position = offsets[b] + t)
//   Outputs:
//     q_out:    [B, Hq, T, D] row-contiguous
//     k_out:    [B, Hk, T, D] row-contiguous
//   FIXED_D: 128 (draft fast path), 256 (target fast path), or 0 (generic D <= 4096)
//   Grid: dim3(T, Hq + Hk, B), Block: dim3(128, 1, 1)
template <typename T, int FIXED_D>
__global__ __launch_bounds__(128) void mlx_qk_norm_rope_cuda_kernel(
    const T* __restrict__ q_in,
    const T* __restrict__ k_in,
    const T* __restrict__ q_norm_w,
    const T* __restrict__ k_norm_w,
    const int* __restrict__ offsets,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int T_len, int Hq, int Hk, int D,
    int rope_dims, float log2_base, float rope_scale, float eps,
    long long q_sb, long long q_st, long long q_sh,
    long long k_sb, long long k_st, long long k_sh) {
  __shared__ float s_warp[4];
  __shared__ T s_buf[FIXED_D > 0 ? FIXED_D : 4096];

  const int t = static_cast<int>(blockIdx.x);
  const int hy = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_q = hy < Hq;
  const int dim = (FIXED_D > 0) ? FIXED_D : D;

  const T* src;
  T* dst;
  const T* norm_w;
  if (is_q) {
    const int hq = hy;
    src = q_in + (long long)b * q_sb + (long long)t * q_st + (long long)hq * q_sh;
    dst = q_out + (((long long)b * Hq + hq) * T_len + t) * dim;
    norm_w = q_norm_w;
  } else {
    const int hk = hy - Hq;
    src = k_in + (long long)b * k_sb + (long long)t * k_st + (long long)hk * k_sh;
    dst = k_out + (((long long)b * Hk + hk) * T_len + t) * dim;
    norm_w = k_norm_w;
  }

  const float pos_scaled = rope_scale * static_cast<float>(offsets[b] + t);

  if constexpr (FIXED_D == 128) {
    const float xf = qknr_to_f32<T>(src[tid]);
    float sq = xf * xf;
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, shift);
    }
    if (lane == 0) {
      s_warp[warp] = sq;
    }
    __syncthreads();

    const float sum_sq = (s_warp[0] + s_warp[1]) + (s_warp[2] + s_warp[3]);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / 128.0f) + eps);
    const T sc = qknr_from_f32<T>(xf * inv_rms);
    const T n = qknr_from_f32<T>(qknr_to_f32<T>(norm_w[tid]) * qknr_to_f32<T>(sc));

    if (tid < rope_dims) {
      s_buf[tid] = n;
    } else {
      dst[tid] = n;
    }

    if (rope_dims > 0) {
      __syncthreads();
      const int half_rope = rope_dims >> 1;
      if (tid < half_rope) {
        const float d_frac = static_cast<float>(tid) / static_cast<float>(half_rope);
        const float inv_freq = exp2f(-d_frac * log2_base);
        const float theta = pos_scaled * inv_freq;
        const float costheta = cosf(theta);
        const float sintheta = sinf(theta);
        const float rx1 = qknr_to_f32<T>(s_buf[tid]);
        const float rx2 = qknr_to_f32<T>(s_buf[tid + half_rope]);
        dst[tid] = qknr_from_f32<T>(rx1 * costheta - rx2 * sintheta);
        dst[tid + half_rope] = qknr_from_f32<T>(rx1 * sintheta + rx2 * costheta);
      }
    }
  } else if constexpr (FIXED_D == 256) {
    float x0, x1;
    qknr_load2(src, tid, x0, x1);
    float sq = x0 * x0 + x1 * x1;
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, shift);
    }
    if (lane == 0) {
      s_warp[warp] = sq;
    }
    __syncthreads();

    const float sum_sq = (s_warp[0] + s_warp[1]) + (s_warp[2] + s_warp[3]);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / 256.0f) + eps);

    float w0, w1;
    qknr_load2(norm_w, tid, w0, w1);
    const T sc0 = qknr_from_f32<T>(x0 * inv_rms);
    const T sc1 = qknr_from_f32<T>(x1 * inv_rms);
    const T n0 = qknr_from_f32<T>(w0 * qknr_to_f32<T>(sc0));
    const T n1 = qknr_from_f32<T>(w1 * qknr_to_f32<T>(sc1));

    const int d0 = 2 * tid;
    if (d0 < rope_dims) {
      s_buf[d0] = n0;
      s_buf[d0 + 1] = n1;
    } else {
      qknr_store2(dst, tid, n0, n1);
    }

    if (rope_dims > 0) {
      __syncthreads();
      const int half_rope = rope_dims >> 1;
      if (tid < half_rope) {
        const float d_frac = static_cast<float>(tid) / static_cast<float>(half_rope);
        const float inv_freq = exp2f(-d_frac * log2_base);
        const float theta = pos_scaled * inv_freq;
        const float costheta = cosf(theta);
        const float sintheta = sinf(theta);
        const float rx1 = qknr_to_f32<T>(s_buf[tid]);
        const float rx2 = qknr_to_f32<T>(s_buf[tid + half_rope]);
        dst[tid] = qknr_from_f32<T>(rx1 * costheta - rx2 * sintheta);
        dst[tid + half_rope] = qknr_from_f32<T>(rx1 * sintheta + rx2 * costheta);
      }
    }
  } else {
    float sq = 0.0f;
    for (int d = tid; d < D; d += 128) {
      const T xv = src[d];
      s_buf[d] = xv;
      const float xf = qknr_to_f32<T>(xv);
      sq += xf * xf;
    }
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, shift);
    }
    if (lane == 0) {
      s_warp[warp] = sq;
    }
    __syncthreads();

    const float sum_sq = (s_warp[0] + s_warp[1]) + (s_warp[2] + s_warp[3]);
    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(D) + eps);

    for (int d = tid; d < D; d += 128) {
      const float xf = qknr_to_f32<T>(s_buf[d]);
      const T sc = qknr_from_f32<T>(xf * inv_rms);
      const T n = qknr_from_f32<T>(qknr_to_f32<T>(norm_w[d]) * qknr_to_f32<T>(sc));
      if (d < rope_dims) {
        s_buf[d] = n;
      } else {
        dst[d] = n;
      }
    }

    if (rope_dims > 0) {
      __syncthreads();
      const int half_rope = rope_dims >> 1;
      for (int i = tid; i < half_rope; i += 128) {
        const float d_frac = static_cast<float>(i) / static_cast<float>(half_rope);
        const float inv_freq = exp2f(-d_frac * log2_base);
        const float theta = pos_scaled * inv_freq;
        const float costheta = cosf(theta);
        const float sintheta = sinf(theta);
        const float rx1 = qknr_to_f32<T>(s_buf[i]);
        const float rx2 = qknr_to_f32<T>(s_buf[i + half_rope]);
        dst[i] = qknr_from_f32<T>(rx1 * costheta - rx2 * sintheta);
        dst[i + half_rope] = qknr_from_f32<T>(rx1 * sintheta + rx2 * costheta);
      }
    }
  }
}
)cuda"