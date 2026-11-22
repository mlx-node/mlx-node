R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float gdn_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float gdn_sum8(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 4, 8);
  value += __shfl_xor_sync(0xffffffffu, value, 2, 8);
  value += __shfl_xor_sync(0xffffffffu, value, 1, 8);
  return value;
}

__device__ __forceinline__ nv_bfloat16 gdn_bf16_mul(nv_bfloat16 a, nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

__device__ __forceinline__ float4 gdn_widen_bf16x4(uint2 w) {
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
  return make_float4(
      __low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

__device__ __forceinline__ float4 gdn_ldcs_f4(const float* ptr) {
  return __ldcs(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ void gdn_stcs_f4(float* ptr, float4 val) {
  __stcs(reinterpret_cast<float4*>(ptr), val);
}

template <bool BF16>
__device__ __forceinline__ float gdn_load_beta(const void* ptr, long long idx) {
  if constexpr (BF16) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  } else {
    return reinterpret_cast<const float*>(ptr)[idx];
  }
}

__device__ __forceinline__ void gdn_emit_rmsnorm_gated_warp(
    const nv_bfloat16* y_row,
    const nv_bfloat16* w,
    const nv_bfloat16* z_row,
    nv_bfloat16* out_row,
    float eps,
    int lane) {
  constexpr int D = 128;
  float values[4];
  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = __bfloat162float(y_row[lane * 4 + i]);
    sq += values[i] * values[i];
  }
  const float inv = rsqrtf(gdn_warp_sum(sq) / D + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const nv_bfloat16 scaled = __float2bfloat16(values[i] * inv);
    const nv_bfloat16 normed = gdn_bf16_mul(w[lane * 4 + i], scaled);
    const nv_bfloat16 zt = z_row[lane * 4 + i];
    const nv_bfloat16 one = __float2bfloat16(1.0f);
    const nv_bfloat16 sigmoid_y = one / (one + hexp(__habs(zt)));
    const nv_bfloat16 sig = zt < __float2bfloat16(0.0f)
        ? sigmoid_y
        : one - sigmoid_y;
    const nv_bfloat16 gated = gdn_bf16_mul(sig, zt);
    out_row[lane * 4 + i] = gdn_bf16_mul(gated, normed);
  }
}

extern "C" __global__ __launch_bounds__(1024, 1) void mlx_gdn_fused_step_cuda_kernel(
    const nv_bfloat16* __restrict__ q, const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const nv_bfloat16* __restrict__ beta, const float* __restrict__ state,
    const nv_bfloat16* __restrict__ z, const nv_bfloat16* __restrict__ w,
    nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
    int T, int Hk, int Hv, long long z_stride_b,
    long long z_stride_t, float eps, int tiled_heads) {
  constexpr int D = 128;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col = warp * 4 + (lane >> 3);
  const int lane_in_col = lane & 7;
  const int hk = tiled_heads ? (h % Hk) : (h / (Hv / Hk));
  const long long state_offset = ((long long)b * Hv + h) * D * D + col * D;

  __shared__ uint2 s_k[32 * 32];
  __shared__ uint2 s_q[32 * 32];
  __shared__ nv_bfloat16 s_v[32 * D];
  __shared__ float s_g[32];
  __shared__ float s_beta[32];
  __shared__ nv_bfloat16 y_shared[32 * D];

  float4 s[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    s[r] = gdn_ldcs_f4(state + state_offset + (r * 8 + lane_in_col) * 4);
  }
  for (int idx = tid; idx < T * 32; idx += 1024) {
    const int t = idx >> 5;
    const int c4 = idx & 31;
    const long long qk_off = ((long long)b * T + t) * Hk * D + hk * D + c4 * 4;
    s_k[idx] = *reinterpret_cast<const uint2*>(k + qk_off);
    s_q[idx] = *reinterpret_cast<const uint2*>(q + qk_off);
    *reinterpret_cast<uint2*>(s_v + idx * 4) =
        *reinterpret_cast<const uint2*>(v + ((long long)b * T + t) * Hv * D + h * D + c4 * 4);
  }
  if (tid < T) {
    const long long gate_off = ((long long)b * T + tid) * Hv + h;
    s_g[tid] = g[gate_off];
    s_beta[tid] = __bfloat162float(beta[gate_off]);
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    const uint2* kt = s_k + t * 32;
    const uint2* qt = s_q + t * 32;
    float kv = 0.0f;
    float4 kv4[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
      kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
            s[r].z * kv4[r].z + s[r].w * kv4[r].w;
    }
    kv = gdn_sum8(kv);
    const float decay = s_g[t];
    const float delta = (__bfloat162float(s_v[t * D + col]) - decay * kv) * s_beta[t];
    float y = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float4 qv4 = gdn_widen_bf16x4(qt[r * 8 + lane_in_col]);
      s[r].x = decay * s[r].x + kv4[r].x * delta;
      s[r].y = decay * s[r].y + kv4[r].y * delta;
      s[r].z = decay * s[r].z + kv4[r].z * delta;
      s[r].w = decay * s[r].w + kv4[r].w * delta;
      y += s[r].x * qv4.x + s[r].y * qv4.y +
           s[r].z * qv4.z + s[r].w * qv4.w;
    }
    y = gdn_sum8(y);
    if (lane_in_col == 0) {
      y_shared[t * D + col] = __float2bfloat16(y);
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    gdn_stcs_f4(state_out + state_offset + (r * 8 + lane_in_col) * 4, s[r]);
  }
  __syncthreads();

  if (warp < T) {
    gdn_emit_rmsnorm_gated_warp(
        y_shared + warp * D, w,
        z + (long long)b * z_stride_b + (long long)warp * z_stride_t + h * D,
        out + ((long long)b * T + warp) * Hv * D + h * D, eps, lane);
  }
}

// Split-Dv (ROW_BLOCKS=4, 256 threads/CTA, 4 CTAs/head) vectorized float4 replay.
// Grid: dim3(Hv * 4, B, 1), Block: dim3(256, 1, 1).
extern "C" __global__ __launch_bounds__(256, 4) void mlx_gdn_replay_cuda_kernel(
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const nv_bfloat16* __restrict__ beta,
    const float* __restrict__ state,
    float* __restrict__ state_out,
    int accepted_steps, int window_stride, int Hk, int Hv, int tiled_heads) {
  constexpr int D = 128;
  constexpr int ROW_BLOCKS = 4;
  constexpr int ROWS_PER_BLK = D / ROW_BLOCKS; // 32
  constexpr int CHUNK = 16;
  const int rb = blockIdx.x % ROW_BLOCKS;
  const int h = blockIdx.x / ROW_BLOCKS;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int local_col = warp * 4 + (lane >> 3);
  const int col = rb * ROWS_PER_BLK + local_col;
  const int lane_in_col = lane & 7;
  const int hk = tiled_heads ? (h % Hk) : (h / (Hv / Hk));
  const long long state_offset = ((long long)b * Hv + h) * D * D + col * D;

  __shared__ uint2 s_k[CHUNK * 32];
  __shared__ nv_bfloat16 s_v[CHUNK * ROWS_PER_BLK];
  __shared__ float s_g[CHUNK];
  __shared__ float s_beta[CHUNK];

  float4 s[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    s[r] = gdn_ldcs_f4(state + state_offset + (r * 8 + lane_in_col) * 4);
  }

  for (int t0 = 0; t0 < accepted_steps; t0 += CHUNK) {
    const int n = (accepted_steps - t0 < CHUNK) ? (accepted_steps - t0) : CHUNK;
    for (int idx = tid; idx < n * 32; idx += 256) {
      const int lt = idx >> 5;
      const int c4 = idx & 31;
      const long long k_off =
          ((long long)b * window_stride + (t0 + lt)) * Hk * D + hk * D + c4 * 4;
      s_k[idx] = *reinterpret_cast<const uint2*>(k + k_off);
    }
    for (int idx = tid; idx < n * ROWS_PER_BLK; idx += 256) {
      const int lt = idx / ROWS_PER_BLK;
      const int r = idx % ROWS_PER_BLK;
      const long long v_off =
          ((long long)b * window_stride + (t0 + lt)) * Hv * D + h * D +
          rb * ROWS_PER_BLK + r;
      s_v[idx] = v[v_off];
    }
    if (tid < n) {
      const long long gate_off =
          ((long long)b * window_stride + (t0 + tid)) * Hv + h;
      s_g[tid] = g[gate_off];
      s_beta[tid] = __bfloat162float(beta[gate_off]);
    }
    __syncthreads();

    for (int lt = 0; lt < n; ++lt) {
      const uint2* kt = s_k + lt * 32;
      float kv = 0.0f;
      float4 kv4[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
        kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
              s[r].z * kv4[r].z + s[r].w * kv4[r].w;
      }
      kv = gdn_sum8(kv);
      const float decay = s_g[lt];
      const float delta =
          (__bfloat162float(s_v[lt * ROWS_PER_BLK + local_col]) - decay * kv) *
          s_beta[lt];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        s[r].x = decay * s[r].x + kv4[r].x * delta;
        s[r].y = decay * s[r].y + kv4[r].y * delta;
        s[r].z = decay * s[r].z + kv4[r].z * delta;
        s[r].w = decay * s[r].w + kv4[r].w * delta;
      }
    }
    if (t0 + CHUNK < accepted_steps) {
      __syncthreads();
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    gdn_stcs_f4(state_out + state_offset + (r * 8 + lane_in_col) * 4, s[r]);
  }
}
)cuda"
R"cuda(
// Deferred-fold GDN step with fused RMSNormGated epilogue (1 CTA/head, 512 threads,
// 2 software-pipelined 64-row batches overlapping next-batch state_base loads with
// current-batch committed_state stores):
// - Loads state_base once, folds prev_keep accepted steps -> writes out_committed_state
//   once (skipped when out_committed_state == nullptr), continues in registers for
//   T <= 16 verify steps without writing speculative state_T, then fuses
//   RMSNorm(y) * w * SiLU(z) into out [B, T, Hv * 128] in one GPU dispatch.
// Grid: dim3(Hv, B, 1), Block: dim3(512, 1, 1).
template <bool BETA_BF16, bool PREV_BETA_BF16>
__global__ __launch_bounds__(512, 2) void mlx_gdn_deferred_step_cuda_kernel(
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const void* __restrict__ beta,
    const float* __restrict__ state_base,
    const nv_bfloat16* __restrict__ prev_k,
    const nv_bfloat16* __restrict__ prev_v,
    const float* __restrict__ prev_g,
    const void* __restrict__ prev_beta,
    const nv_bfloat16* __restrict__ z,
    const nv_bfloat16* __restrict__ w,
    nv_bfloat16* __restrict__ out,
    float* __restrict__ out_committed_state,
    int T, int prev_keep, int prev_stride, int Hk, int Hv,
    long long z_stride_b, long long z_stride_t, float eps, int tiled_heads) {
  constexpr int D = 128;
  constexpr int CHUNK = 16;
  constexpr int THREADS = 512;
  constexpr int ROWS_PER_BATCH = 64;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int local_col = warp * 4 + (lane >> 3);
  const int lane_in_col = lane & 7;
  const int hk = tiled_heads ? (h % Hk) : (h / (Hv / Hk));
  const long long head_base = ((long long)b * Hv + h) * D * D;

  __shared__ uint2 s_k[CHUNK * 32];
  __shared__ uint2 s_q[CHUNK * 32];
  __shared__ uint2 s_pk[CHUNK * 32];
  __shared__ nv_bfloat16 s_v[CHUNK * D];
  __shared__ nv_bfloat16 s_pv[CHUNK * D];
  __shared__ nv_bfloat16 y_shared[CHUNK * D];
  __shared__ float s_g[CHUNK];
  __shared__ float s_beta[CHUNK];
  __shared__ float s_pg[CHUNK];
  __shared__ float s_pbeta[CHUNK];

  // Issue batch 0 state loads before SMEM barrier to hide DRAM latency.
  float4 s[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    s[r] = gdn_ldcs_f4(
        state_base + head_base + local_col * D + (r * 8 + lane_in_col) * 4);
  }

  for (int idx = tid; idx < T * 32; idx += THREADS) {
    const int t = idx >> 5;
    const int c4 = idx & 31;
    const long long qk_off = ((long long)b * T + t) * Hk * D + hk * D + c4 * 4;
    s_k[idx] = *reinterpret_cast<const uint2*>(k + qk_off);
    s_q[idx] = *reinterpret_cast<const uint2*>(q + qk_off);
    *reinterpret_cast<uint2*>(s_v + idx * 4) =
        *reinterpret_cast<const uint2*>(v + ((long long)b * T + t) * Hv * D + h * D + c4 * 4);
  }
  if (tid < T) {
    const long long go = ((long long)b * T + tid) * Hv + h;
    s_g[tid] = g[go];
    s_beta[tid] = gdn_load_beta<BETA_BF16>(beta, go);
  }

  const bool prev_fits_smem = (prev_keep > 0 && prev_keep <= CHUNK);
  if (prev_fits_smem) {
    for (int idx = tid; idx < prev_keep * 32; idx += THREADS) {
      const int lt = idx >> 5;
      const int c4 = idx & 31;
      s_pk[idx] = *reinterpret_cast<const uint2*>(
          prev_k + ((long long)b * prev_stride + lt) * Hk * D + hk * D + c4 * 4);
      *reinterpret_cast<uint2*>(s_pv + idx * 4) = *reinterpret_cast<const uint2*>(
          prev_v + ((long long)b * prev_stride + lt) * Hv * D + h * D + c4 * 4);
    }
    if (tid < prev_keep) {
      const long long pgo = ((long long)b * prev_stride + tid) * Hv + h;
      s_pg[tid] = prev_g[pgo];
      s_pbeta[tid] = gdn_load_beta<PREV_BETA_BF16>(prev_beta, pgo);
    }
  }
  __syncthreads();

  for (int batch = 0; batch < 2; ++batch) {
    const int col = batch * ROWS_PER_BATCH + local_col;
    if (prev_fits_smem) {
      for (int lt = 0; lt < prev_keep; ++lt) {
        const uint2* kt = s_pk + lt * 32;
        float kv = 0.0f;
        float4 kv4[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
          kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
                s[r].z * kv4[r].z + s[r].w * kv4[r].w;
        }
        kv = gdn_sum8(kv);
        const float decay = s_pg[lt];
        const float delta =
            (__bfloat162float(s_pv[lt * D + col]) - decay * kv) * s_pbeta[lt];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          s[r].x = decay * s[r].x + kv4[r].x * delta;
          s[r].y = decay * s[r].y + kv4[r].y * delta;
          s[r].z = decay * s[r].z + kv4[r].z * delta;
          s[r].w = decay * s[r].w + kv4[r].w * delta;
        }
      }
    } else if (prev_keep > CHUNK) {
      for (int p0 = 0; p0 < prev_keep; p0 += CHUNK) {
        const int n = (prev_keep - p0 < CHUNK) ? (prev_keep - p0) : CHUNK;
        for (int idx = tid; idx < n * 32; idx += THREADS) {
          const int lt = idx >> 5;
          const int c4 = idx & 31;
          s_pk[idx] = *reinterpret_cast<const uint2*>(
              prev_k + ((long long)b * prev_stride + (p0 + lt)) * Hk * D + hk * D + c4 * 4);
          *reinterpret_cast<uint2*>(s_pv + idx * 4) = *reinterpret_cast<const uint2*>(
              prev_v + ((long long)b * prev_stride + (p0 + lt)) * Hv * D + h * D + c4 * 4);
        }
        if (tid < n) {
          const long long pgo = ((long long)b * prev_stride + (p0 + tid)) * Hv + h;
          s_pg[tid] = prev_g[pgo];
          s_pbeta[tid] = gdn_load_beta<PREV_BETA_BF16>(prev_beta, pgo);
        }
        __syncthreads();
        for (int lt = 0; lt < n; ++lt) {
          const uint2* kt = s_pk + lt * 32;
          float kv = 0.0f;
          float4 kv4[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
            kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
                  s[r].z * kv4[r].z + s[r].w * kv4[r].w;
          }
          kv = gdn_sum8(kv);
          const float decay = s_pg[lt];
          const float delta =
              (__bfloat162float(s_pv[lt * D + col]) - decay * kv) * s_pbeta[lt];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            s[r].x = decay * s[r].x + kv4[r].x * delta;
            s[r].y = decay * s[r].y + kv4[r].y * delta;
            s[r].z = decay * s[r].z + kv4[r].z * delta;
            s[r].w = decay * s[r].w + kv4[r].w * delta;
          }
        }
        __syncthreads();
      }
    }

    float4 next_s[4];
    if (batch == 0) {
      const int ncol = ROWS_PER_BATCH + local_col;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        next_s[r] = gdn_ldcs_f4(
            state_base + head_base + ncol * D + (r * 8 + lane_in_col) * 4);
      }
    }
    if (out_committed_state != nullptr) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        gdn_stcs_f4(
            out_committed_state + head_base + col * D + (r * 8 + lane_in_col) * 4, s[r]);
      }
    }

    for (int t = 0; t < T; ++t) {
      const uint2* kt = s_k + t * 32;
      const uint2* qt = s_q + t * 32;
      float kv = 0.0f;
      float4 kv4[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
        kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
              s[r].z * kv4[r].z + s[r].w * kv4[r].w;
      }
      kv = gdn_sum8(kv);
      const float decay = s_g[t];
      const float delta = (__bfloat162float(s_v[t * D + col]) - decay * kv) * s_beta[t];
      float y = 0.0f;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float4 qv4 = gdn_widen_bf16x4(qt[r * 8 + lane_in_col]);
        s[r].x = decay * s[r].x + kv4[r].x * delta;
        s[r].y = decay * s[r].y + kv4[r].y * delta;
        s[r].z = decay * s[r].z + kv4[r].z * delta;
        s[r].w = decay * s[r].w + kv4[r].w * delta;
        y += s[r].x * qv4.x + s[r].y * qv4.y +
             s[r].z * qv4.z + s[r].w * qv4.w;
      }
      y = gdn_sum8(y);
      if (lane_in_col == 0) {
        y_shared[t * D + col] = __float2bfloat16(y);
      }
    }
    if (batch == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        s[r] = next_s[r];
      }
    }
  }
  __syncthreads();

  if (warp < T) {
    gdn_emit_rmsnorm_gated_warp(
        y_shared + warp * D, w,
        z + (long long)b * z_stride_b + (long long)warp * z_stride_t + h * D,
        out + ((long long)b * T + warp) * Hv * D + h * D, eps, lane);
  }
}
)cuda"
R"cuda(
// Zero-state-duplication speculative tree GDN (Splash gdn_decode_transform) with
// optional non-contiguous prev_indices commit + fused RMSNormGated epilogue:
// - 1 CTA/head, 512 threads (16 warps = 1 warp/node for T <= 16 query pullback
//   and 1 warp/token for RMSNormGated), 2 software-pipelined 64-row state batches.
// - When prev_indices != nullptr, folds prev_indices[0..prev_keep-1] directly
//   from prev_{k,v,g,beta} without gather kernels; when nullptr, folds 0..prev_keep-1.
// Grid: dim3(Hv, B, 1), Block: dim3(512, 1, 1).
template <bool BETA_BF16, bool PREV_BETA_BF16>
__global__ __launch_bounds__(512, 2) void mlx_gdn_tree_step_cuda_kernel(
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const void* __restrict__ beta,
    const float* __restrict__ state_base,
    const nv_bfloat16* __restrict__ prev_k,
    const nv_bfloat16* __restrict__ prev_v,
    const float* __restrict__ prev_g,
    const void* __restrict__ prev_beta,
    const int* __restrict__ prev_indices,
    const int* __restrict__ parent,
    const nv_bfloat16* __restrict__ z,
    const nv_bfloat16* __restrict__ w,
    nv_bfloat16* __restrict__ out,
    float* __restrict__ out_committed_state,
    int T, int prev_keep, int prev_stride, int Hk, int Hv,
    long long z_stride_b, long long z_stride_t, float eps, int tiled_heads) {
  constexpr int D = 128;
  constexpr int CHUNK = 16;
  constexpr int THREADS = 512;
  constexpr int ROWS_PER_BATCH = 64;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int local_col = warp * 4 + (lane >> 3);
  const int lane_in_col = lane & 7;
  const int hk = tiled_heads ? (h % Hk) : (h / (Hv / Hk));
  const long long head_base = ((long long)b * Hv + h) * D * D;

  __shared__ uint2 s_k[CHUNK * 32];
  __shared__ float4 s_z4[CHUNK * 32];
  __shared__ uint2 s_pk[CHUNK * 32];
  __shared__ float s_coeff[CHUNK * CHUNK];
  __shared__ nv_bfloat16 s_v[D * CHUNK];
  __shared__ nv_bfloat16 s_pv[CHUNK * D];
  __shared__ nv_bfloat16 y_shared[CHUNK * D];
  __shared__ float s_g[CHUNK];
  __shared__ float s_beta[CHUNK];
  __shared__ float s_pg[CHUNK];
  __shared__ float s_pbeta[CHUNK];
  __shared__ int s_parent[CHUNK];
  __shared__ int s_pidx[CHUNK];

  float4 s[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    s[r] = gdn_ldcs_f4(
        state_base + head_base + local_col * D + (r * 8 + lane_in_col) * 4);
  }

  for (int idx = tid; idx < T * 32; idx += THREADS) {
    const int t = idx >> 5;
    const int c4 = idx & 31;
    const long long qk_off = ((long long)b * T + t) * Hk * D + hk * D + c4 * 4;
    s_k[idx] = *reinterpret_cast<const uint2*>(k + qk_off);
    s_z4[idx] = gdn_widen_bf16x4(*reinterpret_cast<const uint2*>(q + qk_off));
  }
  for (int idx = tid; idx < T * D; idx += THREADS) {
    const int t = idx >> 7;
    const int r = idx & 127;
    s_v[r * CHUNK + t] = v[((long long)b * T + t) * Hv * D + h * D + r];
  }
  if (tid < T) {
    const long long go = ((long long)b * T + tid) * Hv + h;
    s_g[tid] = g[go];
    s_beta[tid] = gdn_load_beta<BETA_BF16>(beta, go);
    s_parent[tid] = parent[tid];
  }
  const bool prev_fits_smem = (prev_keep > 0 && prev_keep <= CHUNK);
  if (prev_fits_smem && tid < prev_keep) {
    s_pidx[tid] = prev_indices ? prev_indices[tid] : tid;
  }
  __syncthreads();

  // Stage prev_keep <= 16 using s_pidx (supports non-contiguous tree branch commits!).
  if (prev_fits_smem) {
    for (int idx = tid; idx < prev_keep * 32; idx += THREADS) {
      const int lt = idx >> 5;
      const int c4 = idx & 31;
      const int src_t = s_pidx[lt];
      s_pk[idx] = *reinterpret_cast<const uint2*>(
          prev_k + ((long long)b * prev_stride + src_t) * Hk * D + hk * D + c4 * 4);
      *reinterpret_cast<uint2*>(s_pv + idx * 4) = *reinterpret_cast<const uint2*>(
          prev_v + ((long long)b * prev_stride + src_t) * Hv * D + h * D + c4 * 4);
    }
    if (tid < prev_keep) {
      const int src_t = s_pidx[tid];
      const long long pgo = ((long long)b * prev_stride + src_t) * Hv + h;
      s_pg[tid] = prev_g[pgo];
      s_pbeta[tid] = gdn_load_beta<PREV_BETA_BF16>(prev_beta, pgo);
    }
  }

  // Backward query pullback across the tree (16 warps cover all T <= 16 nodes in 1 pass!).
  if (warp < T) {
    float4 zvec = s_z4[warp * 32 + lane];
    float my_coeff = 0.0f;
    int cur = warp;
    for (int step = 0; step < T && cur >= 0; ++step) {
      const float4 kv = gdn_widen_bf16x4(s_k[cur * 32 + lane]);
      float dot = kv.x * zvec.x + kv.y * zvec.y + kv.z * zvec.z + kv.w * zvec.w;
      dot = gdn_warp_sum(dot);
      const float c = s_beta[cur] * dot;
      const float d = s_g[cur];
      zvec.x = d * (zvec.x - c * kv.x);
      zvec.y = d * (zvec.y - c * kv.y);
      zvec.z = d * (zvec.z - c * kv.z);
      zvec.w = d * (zvec.w - c * kv.w);
      if (lane == cur) {
        my_coeff = c;
      }
      cur = s_parent[cur];
    }
    s_z4[warp * 32 + lane] = zvec;
    if (lane < CHUNK) {
      s_coeff[warp * CHUNK + lane] = my_coeff;
    }
  }
  __syncthreads();

  for (int batch = 0; batch < 2; ++batch) {
    const int col = batch * ROWS_PER_BATCH + local_col;
    if (prev_fits_smem) {
      for (int lt = 0; lt < prev_keep; ++lt) {
        const uint2* kt = s_pk + lt * 32;
        float kv = 0.0f;
        float4 kv4[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
          kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
                s[r].z * kv4[r].z + s[r].w * kv4[r].w;
        }
        kv = gdn_sum8(kv);
        const float decay = s_pg[lt];
        const float delta =
            (__bfloat162float(s_pv[lt * D + col]) - decay * kv) * s_pbeta[lt];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          s[r].x = decay * s[r].x + kv4[r].x * delta;
          s[r].y = decay * s[r].y + kv4[r].y * delta;
          s[r].z = decay * s[r].z + kv4[r].z * delta;
          s[r].w = decay * s[r].w + kv4[r].w * delta;
        }
      }
    } else if (prev_keep > CHUNK) {
      for (int p0 = 0; p0 < prev_keep; p0 += CHUNK) {
        const int n = (prev_keep - p0 < CHUNK) ? (prev_keep - p0) : CHUNK;
        if (tid < n) {
          s_pidx[tid] = prev_indices ? prev_indices[p0 + tid] : (p0 + tid);
        }
        __syncthreads();
        for (int idx = tid; idx < n * 32; idx += THREADS) {
          const int lt = idx >> 5;
          const int c4 = idx & 31;
          const int src_t = s_pidx[lt];
          s_pk[idx] = *reinterpret_cast<const uint2*>(
              prev_k + ((long long)b * prev_stride + src_t) * Hk * D + hk * D + c4 * 4);
          *reinterpret_cast<uint2*>(s_pv + idx * 4) = *reinterpret_cast<const uint2*>(
              prev_v + ((long long)b * prev_stride + src_t) * Hv * D + h * D + c4 * 4);
        }
        if (tid < n) {
          const int src_t = s_pidx[tid];
          const long long pgo = ((long long)b * prev_stride + src_t) * Hv + h;
          s_pg[tid] = prev_g[pgo];
          s_pbeta[tid] = gdn_load_beta<PREV_BETA_BF16>(prev_beta, pgo);
        }
        __syncthreads();
        for (int lt = 0; lt < n; ++lt) {
          const uint2* kt = s_pk + lt * 32;
          float kv = 0.0f;
          float4 kv4[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            kv4[r] = gdn_widen_bf16x4(kt[r * 8 + lane_in_col]);
            kv += s[r].x * kv4[r].x + s[r].y * kv4[r].y +
                  s[r].z * kv4[r].z + s[r].w * kv4[r].w;
          }
          kv = gdn_sum8(kv);
          const float decay = s_pg[lt];
          const float delta =
              (__bfloat162float(s_pv[lt * D + col]) - decay * kv) * s_pbeta[lt];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            s[r].x = decay * s[r].x + kv4[r].x * delta;
            s[r].y = decay * s[r].y + kv4[r].y * delta;
            s[r].z = decay * s[r].z + kv4[r].z * delta;
            s[r].w = decay * s[r].w + kv4[r].w * delta;
          }
        }
        __syncthreads();
      }
    }

    float4 next_s[4];
    if (batch == 0) {
      const int ncol = ROWS_PER_BATCH + local_col;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        next_s[r] = gdn_ldcs_f4(
            state_base + head_base + ncol * D + (r * 8 + lane_in_col) * 4);
      }
    }
    if (out_committed_state != nullptr) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        gdn_stcs_f4(
            out_committed_state + head_base + col * D + (r * 8 + lane_in_col) * 4, s[r]);
      }
    }

    const float v0 = (lane_in_col < T)
        ? __bfloat162float(s_v[col * CHUNK + lane_in_col])
        : 0.0f;
    const float v1 = (lane_in_col + 8 < T)
        ? __bfloat162float(s_v[col * CHUNK + lane_in_col + 8])
        : 0.0f;
    for (int node = 0; node < T; ++node) {
      const float4* zt = s_z4 + node * 32;
      float y = ((lane_in_col < T) ? (s_coeff[node * CHUNK + lane_in_col] * v0) : 0.0f) +
                ((lane_in_col + 8 < T) ? (s_coeff[node * CHUNK + lane_in_col + 8] * v1) : 0.0f);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float4 zv = zt[r * 8 + lane_in_col];
        y += s[r].x * zv.x + s[r].y * zv.y + s[r].z * zv.z + s[r].w * zv.w;
      }
      y = gdn_sum8(y);
      if (lane_in_col == 0) {
        y_shared[node * D + col] = __float2bfloat16(y);
      }
    }
    if (batch == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        s[r] = next_s[r];
      }
    }
  }
  __syncthreads();

  if (warp < T) {
    gdn_emit_rmsnorm_gated_warp(
        y_shared + warp * D, w,
        z + (long long)b * z_stride_b + (long long)warp * z_stride_t + h * D,
        out + ((long long)b * T + warp) * Hv * D + h * D, eps, lane);
  }
}
)cuda"
