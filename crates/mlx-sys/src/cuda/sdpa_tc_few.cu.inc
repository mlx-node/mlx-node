R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ void sdpa_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ unsigned int sdpa_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned int*>(&v);
}

// Flash-Decoding Tensor Core SDPA for D in {128, 256}, GQA folded into M = gqa * Q_len <= 64.
// MASK_MODE: 0 = causal, 1 = unmasked (full), 2 = explicit bool mask [1, 1, Q_len, K_len].
// Q: [B, Hq, Q_len, D] row-contiguous (viewed as [B, Hkv, M, D]).
// K, V: [B, Hkv, K_len, D] with arbitrary outer strides (last dim contiguous, 16B aligned).
// Grid: (num_splits, Hkv, B), Block: (32, NUM_WARPS) where NUM_WARPS = ceil(M/16) in 1..=4.
template <int D, int NUM_WARPS, int MASK_MODE>
__global__ __launch_bounds__(NUM_WARPS * 32, 2) void mlx_node_sdpa_tc_split(
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    const unsigned char* __restrict__ mask,
    float* __restrict__ partial_O,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int Hkv, int gqa, int Q_len, int K_len,
    long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss,
    float scale, int num_splits) {
  constexpr int BN = 32;
  constexpr int STRIDE = D + 8;
  constexpr int D_TILES = D / 8;
  const int split = blockIdx.x;
  const int hkv = blockIdx.y;
  const int b = blockIdx.z;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int num_threads = NUM_WARPS * 32;
  const int M = gqa * Q_len;

  const int total_tiles = (K_len + BN - 1) / BN;
  const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
  const int tile_start = split * tiles_per_split;
  const int tile_end = (tile_start + tiles_per_split < total_tiles)
      ? (tile_start + tiles_per_split) : total_tiles;

  __shared__ nv_bfloat16 sQ[NUM_WARPS * 16 * STRIDE];
  __shared__ nv_bfloat16 sKV[BN * STRIDE];

  const nv_bfloat16* Q_base = Q + ((long long)b * Hkv + hkv) * M * D;
  for (int idx = tid; idx < M * (D / 8); idx += num_threads) {
    const int r = idx / (D / 8);
    const int c8 = idx % (D / 8);
    *reinterpret_cast<uint4*>(&sQ[r * STRIDE + c8 * 8]) =
        *reinterpret_cast<const uint4*>(&Q_base[r * D + c8 * 8]);
  }
  for (int idx = M * (D / 8) + tid; idx < NUM_WARPS * 16 * (D / 8); idx += num_threads) {
    const int r = idx / (D / 8);
    const int c8 = idx % (D / 8);
    *reinterpret_cast<uint4*>(&sQ[r * STRIDE + c8 * 8]) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  const int m_base = warp * 16;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int row0 = m_base + group_id;
  const int row1 = m_base + group_id + 8;
  const int q_pos0 = (row0 < M) ? (row0 % Q_len) : 0;
  const int q_pos1 = (row1 < M) ? (row1 % Q_len) : 0;
  const int causal_limit0 = K_len - Q_len + q_pos0;
  const int causal_limit1 = K_len - Q_len + q_pos1;

  float accO[D_TILES][4];
#pragma unroll
  for (int t = 0; t < D_TILES; ++t) {
    accO[t][0] = 0.0f;
    accO[t][1] = 0.0f;
    accO[t][2] = 0.0f;
    accO[t][3] = 0.0f;
  }
  float m_row0 = -1e30f;
  float m_row1 = -1e30f;
  float l_row0 = 0.0f;
  float l_row1 = 0.0f;

  const nv_bfloat16* K_base = K + (long long)b * k_sb + (long long)hkv * k_sh;
  const nv_bfloat16* V_base = V + (long long)b * v_sb + (long long)hkv * v_sh;

  for (int tile = tile_start; tile < tile_end; ++tile) {
    const int k_start = tile * BN;
    const int k_valid = (K_len - k_start < BN) ? (K_len - k_start) : BN;

    for (int idx = tid; idx < BN * (D / 8); idx += num_threads) {
      const int n = idx / (D / 8);
      const int c8 = idx % (D / 8);
      if (n < k_valid) {
        *reinterpret_cast<uint4*>(&sKV[n * STRIDE + c8 * 8]) =
            *reinterpret_cast<const uint4*>(&K_base[(long long)(k_start + n) * k_ss + c8 * 8]);
      } else {
        *reinterpret_cast<uint4*>(&sKV[n * STRIDE + c8 * 8]) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();

    float s_acc[4][4] = {};
    for (int k = 0; k < D; k += 16) {
      const unsigned int a0 = *reinterpret_cast<const unsigned int*>(&sQ[row0 * STRIDE + k + 2 * tid_in_grp]);
      const unsigned int a1 = *reinterpret_cast<const unsigned int*>(&sQ[row1 * STRIDE + k + 2 * tid_in_grp]);
      const unsigned int a2 = *reinterpret_cast<const unsigned int*>(&sQ[row0 * STRIDE + k + 8 + 2 * tid_in_grp]);
      const unsigned int a3 = *reinterpret_cast<const unsigned int*>(&sQ[row1 * STRIDE + k + 8 + 2 * tid_in_grp]);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int kn = nt * 8 + group_id;
        const unsigned int b0 = *reinterpret_cast<const unsigned int*>(&sKV[kn * STRIDE + k + 2 * tid_in_grp]);
        const unsigned int b1 = *reinterpret_cast<const unsigned int*>(&sKV[kn * STRIDE + k + 8 + 2 * tid_in_grp]);
        sdpa_mma_m16n8k16_bf16(
            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
            a0, a1, a2, a3, b0, b1,
            s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
      }
    }

    float local_max0 = -1e30f;
    float local_max1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c0 = k_start + nt * 8 + 2 * tid_in_grp;
      const int c1 = c0 + 1;
      bool v00, v01, v10, v11;
      if constexpr (MASK_MODE == 0) {
        v00 = (c0 <= causal_limit0 && c0 < K_len);
        v01 = (c1 <= causal_limit0 && c1 < K_len);
        v10 = (c0 <= causal_limit1 && c0 < K_len);
        v11 = (c1 <= causal_limit1 && c1 < K_len);
      } else if constexpr (MASK_MODE == 1) {
        v00 = (c0 < K_len);
        v01 = (c1 < K_len);
        v10 = (c0 < K_len);
        v11 = (c1 < K_len);
      } else {
        v00 = (c0 < K_len) && mask[(long long)q_pos0 * K_len + c0];
        v01 = (c1 < K_len) && mask[(long long)q_pos0 * K_len + c1];
        v10 = (c0 < K_len) && mask[(long long)q_pos1 * K_len + c0];
        v11 = (c1 < K_len) && mask[(long long)q_pos1 * K_len + c1];
      }
      s_acc[nt][0] = v00 ? (s_acc[nt][0] * scale) : -1e30f;
      s_acc[nt][1] = v01 ? (s_acc[nt][1] * scale) : -1e30f;
      s_acc[nt][2] = v10 ? (s_acc[nt][2] * scale) : -1e30f;
      s_acc[nt][3] = v11 ? (s_acc[nt][3] * scale) : -1e30f;
      local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
      local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float alpha0 = (m_row0 > -1e20f) ? expf(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? expf(m_row1 - new_m1) : 0.0f;

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    unsigned int p_frag[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int nt0 = ks * 2;
      const int nt1 = ks * 2 + 1;
      const float p00 = (s_acc[nt0][0] > -1e20f) ? expf(s_acc[nt0][0] - new_m0) : 0.0f;
      const float p01 = (s_acc[nt0][1] > -1e20f) ? expf(s_acc[nt0][1] - new_m0) : 0.0f;
      const float p10 = (s_acc[nt0][2] > -1e20f) ? expf(s_acc[nt0][2] - new_m1) : 0.0f;
      const float p11 = (s_acc[nt0][3] > -1e20f) ? expf(s_acc[nt0][3] - new_m1) : 0.0f;
      const float p02 = (s_acc[nt1][0] > -1e20f) ? expf(s_acc[nt1][0] - new_m0) : 0.0f;
      const float p03 = (s_acc[nt1][1] > -1e20f) ? expf(s_acc[nt1][1] - new_m0) : 0.0f;
      const float p12 = (s_acc[nt1][2] > -1e20f) ? expf(s_acc[nt1][2] - new_m1) : 0.0f;
      const float p13 = (s_acc[nt1][3] > -1e20f) ? expf(s_acc[nt1][3] - new_m1) : 0.0f;
      local_sum0 += (p00 + p01) + (p02 + p03);
      local_sum1 += (p10 + p11) + (p12 + p13);
      p_frag[ks][0] = sdpa_pack_bf16x2(p00, p01);
      p_frag[ks][1] = sdpa_pack_bf16x2(p10, p11);
      p_frag[ks][2] = sdpa_pack_bf16x2(p02, p03);
      p_frag[ks][3] = sdpa_pack_bf16x2(p12, p13);
    }
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 1);
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 2);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 1);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 2);

#pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
      accO[t][0] *= alpha0;
      accO[t][1] *= alpha0;
      accO[t][2] *= alpha1;
      accO[t][3] *= alpha1;
    }
    l_row0 = l_row0 * alpha0 + local_sum0;
    l_row1 = l_row1 * alpha1 + local_sum1;
    m_row0 = new_m0;
    m_row1 = new_m1;

    __syncthreads();
    for (int idx = tid; idx < BN * (D / 8); idx += num_threads) {
      const int n = idx / (D / 8);
      const int c8 = idx % (D / 8);
      if (n < k_valid) {
        *reinterpret_cast<uint4*>(&sKV[n * STRIDE + c8 * 8]) =
            *reinterpret_cast<const uint4*>(&V_base[(long long)(k_start + n) * v_ss + c8 * 8]);
      } else {
        *reinterpret_cast<uint4*>(&sKV[n * STRIDE + c8 * 8]) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int k0 = ks * 16 + 2 * tid_in_grp;
      const int k1 = k0 + 8;
#pragma unroll
      for (int nt = 0; nt < D_TILES; ++nt) {
        const int d = nt * 8 + group_id;
        __nv_bfloat162 v0 = make_bfloat162(sKV[k0 * STRIDE + d], sKV[(k0 + 1) * STRIDE + d]);
        __nv_bfloat162 v1 = make_bfloat162(sKV[k1 * STRIDE + d], sKV[(k1 + 1) * STRIDE + d]);
        const unsigned int b0 = *reinterpret_cast<const unsigned int*>(&v0);
        const unsigned int b1 = *reinterpret_cast<const unsigned int*>(&v1);
        sdpa_mma_m16n8k16_bf16(
            accO[nt][0], accO[nt][1], accO[nt][2], accO[nt][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b0, b1,
            accO[nt][0], accO[nt][1], accO[nt][2], accO[nt][3]);
      }
    }
    __syncthreads();
  }

  const long long base_split = (((long long)b * Hkv + hkv) * num_splits + split) * M;
  if (row0 < M) {
    if (tid_in_grp == 0) {
      partial_m[base_split + row0] = m_row0;
      partial_l[base_split + row0] = l_row0;
    }
    float* dst0 = partial_O + (base_split + row0) * D;
#pragma unroll
    for (int nt = 0; nt < D_TILES; ++nt) {
      dst0[nt * 8 + 2 * tid_in_grp] = accO[nt][0];
      dst0[nt * 8 + 2 * tid_in_grp + 1] = accO[nt][1];
    }
  }
  if (row1 < M) {
    if (tid_in_grp == 0) {
      partial_m[base_split + row1] = m_row1;
      partial_l[base_split + row1] = l_row1;
    }
    float* dst1 = partial_O + (base_split + row1) * D;
#pragma unroll
    for (int nt = 0; nt < D_TILES; ++nt) {
      dst1[nt * 8 + 2 * tid_in_grp] = accO[nt][2];
      dst1[nt * 8 + 2 * tid_in_grp + 1] = accO[nt][3];
    }
  }
}

template <int D>
__global__ void mlx_node_sdpa_tc_reduce(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    nv_bfloat16* __restrict__ out,
    int Hkv, int M, int num_splits) {
  const int row = blockIdx.x;
  const int hkv = blockIdx.y;
  const int b = blockIdx.z;
  const int d = threadIdx.x;
  const long long bh = (long long)b * Hkv + hkv;

  float m_max = -1e30f;
  for (int s = 0; s < num_splits; ++s) {
    m_max = fmaxf(m_max, partial_m[(bh * num_splits + s) * M + row]);
  }
  float l_sum = 0.0f;
  float o_sum = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const long long idx = (bh * num_splits + s) * M + row;
    const float ms = partial_m[idx];
    const float ls = partial_l[idx];
    if (ls > 0.0f) {
      const float w = expf(ms - m_max);
      l_sum += ls * w;
      o_sum += partial_O[idx * D + d] * w;
    }
  }
  const float val = (l_sum > 0.0f) ? (o_sum / l_sum) : 0.0f;
  out[(bh * M + row) * D + d] = __float2bfloat16(val);
}
)cuda"
