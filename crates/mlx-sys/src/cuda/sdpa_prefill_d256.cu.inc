R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float sdpa_p256_exp2f(float x) {
  return ptx_ex2_approx_ftz(x);
}

__device__ __forceinline__ void sdpa_p256_mma(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  ptx_mma_m16n8k16_bf16_c_v(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
}

__device__ __forceinline__ void sdpa_p256_ldmatrix4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4(r0, r1, r2, r3, smem_ptr);
}

__device__ __forceinline__ void sdpa_p256_ldmatrix4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4_trans(r0, r1, r2, r3, smem_ptr);
}

__device__ __forceinline__ void sdpa_p256_cp16(void* smem_ptr, const void* glob_ptr) {
  ptx_cp_async16_cg(smem_ptr, glob_ptr);
}

__device__ __forceinline__ void sdpa_p256_cp16_zfill(
    void* smem_ptr, const void* glob_ptr, bool pred) {
  ptx_cp_async16_cg_zfill(smem_ptr, glob_ptr, pred);
}

__device__ __forceinline__ void sdpa_p256_cp_commit() {
  ptx_cp_commit();
}

__device__ __forceinline__ void sdpa_p256_cp_wait0() {
  ptx_cp_wait<0>();
}

__device__ __forceinline__ unsigned int sdpa_p256_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ int sdpa_p256_swiz256(int row, int c8) {
  return row * 256 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ int sdpa_p256_swiz128(int row, int c8) {
  return row * 128 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ void sdpa_p256_load_kv_unchecked(
    nv_bfloat16* dst, const nv_bfloat16* base, long long stride, int k_start, int tid) {
  const int r0 = tid >> 5;
  const int c8 = tid & 31;
  const nv_bfloat16* gptr = base + (long long)(k_start + r0) * stride + c8 * 8;
  const long long step = 4LL * stride;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sdpa_p256_cp16(&dst[sdpa_p256_swiz256(r0 + i * 4, c8)], gptr);
    gptr += step;
  }
  sdpa_p256_cp_commit();
}

__device__ __forceinline__ void sdpa_p256_load_kv_checked(
    nv_bfloat16* dst, const nv_bfloat16* base, long long stride, int k_start, int K_len, int tid) {
  const int r0 = tid >> 5;
  const int c8 = tid & 31;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = r0 + i * 4;
    const int gk = k_start + r;
    const bool ok = (gk < K_len);
    const nv_bfloat16* gptr = ok ? (base + (long long)gk * stride + c8 * 8) : base;
    sdpa_p256_cp16_zfill(&dst[sdpa_p256_swiz256(r, c8)], gptr, ok);
  }
  sdpa_p256_cp_commit();
}

// FlashAttention-2 BF16 causal prefill kernel for D=256 on sm_80+ / sm_121a.
// Tile: Br=64, Bc=32, 4 warps (128 threads), 48 KB static SMEM (2 CTAs/SM), zero register spills.
//   - Q[0..127] kept in registers (32 uint32/thread), Q[128..255] in sQ_hi (16 KB).
//   - sK (16 KB) and sV (16 KB) asynchronously overlapped via cp.async with QK^T and PV MMAs.
//   - 128B XOR swizzle gives bank-conflict-free cp.async and ldmatrix.x4(.trans).
// Grid:  (ceil(Q_len / 64) * gqa, Hkv, B) — reverse q_block for causal load balance,
//        consecutive gqa heads share K/V in L2.
// Block: (128, 1, 1)
// Q: [B, Hq, Q_len, 256] (last dim contiguous, outer strides 16B aligned).
// K, V: [B, Hkv, K_len, 256] (last dim contiguous, outer strides 16B aligned).
// O: [B, Hq, Q_len, 256] row-contiguous.
extern "C" __global__ __launch_bounds__(128, 2) void mlx_node_sdpa_prefill_d256(
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    nv_bfloat16* __restrict__ O,
    int Hkv, int gqa, int Q_len, int K_len,
    long long q_sb, long long q_sh, long long q_ss,
    long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss,
    float scale_log2) {
  constexpr int BR = 64;
  constexpr int BC = 32;
  constexpr int D = 256;

  const int num_q_blocks = (Q_len + BR - 1) / BR;
  const int qb_rev = blockIdx.x / gqa;
  const int gqa_idx = blockIdx.x - qb_rev * gqa;
  const int q_block = num_q_blocks - 1 - qb_rev;
  const int hkv = blockIdx.y;
  const int b = blockIdx.z;
  const int hq = hkv * gqa + gqa_idx;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int q_block_start = q_block * BR;
  const int q_block_end =
      (q_block_start + BR - 1 < Q_len) ? (q_block_start + BR - 1) : (Q_len - 1);
  const int offset = K_len - Q_len;
  const int max_k = offset + q_block_end;
  const int num_tiles = (max_k / BC) + 1;
  const int num_full_tiles = (offset + q_block_start + 1) / BC;

  __shared__ alignas(128) nv_bfloat16 sQ_hi[BR * 128]; // 16 KB (Q cols 128..255)
  __shared__ alignas(128) nv_bfloat16 sK[BC * D];      // 16 KB (initially Q cols 0..127)
  __shared__ alignas(128) nv_bfloat16 sV[BC * D];      // 16 KB

  const nv_bfloat16* Q_base = Q + (long long)b * q_sb + (long long)hq * q_sh;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = tid + i * 128;
    const int r = idx >> 4;
    const int c8 = idx & 15;
    const int gr = q_block_start + r;
    const bool ok = (gr < Q_len);
    const nv_bfloat16* row_ptr = ok ? (Q_base + (long long)gr * q_ss) : Q_base;
    sdpa_p256_cp16_zfill(&sK[sdpa_p256_swiz128(r, c8)], row_ptr + c8 * 8, ok);
    sdpa_p256_cp16_zfill(&sQ_hi[sdpa_p256_swiz128(r, c8)], row_ptr + (c8 + 16) * 8, ok);
  }
  sdpa_p256_cp_commit();
  sdpa_p256_cp_wait0();
  __syncthreads();

  unsigned int q_lo[8][4];
  const int q_ld_r = warp * 16 + (lane & 15);
  const int q_ld_c8_base = (lane >> 4) & 1;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    sdpa_p256_ldmatrix4(
        q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
        &sK[sdpa_p256_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
  }
  __syncthreads();

  const nv_bfloat16* K_base = K + (long long)b * k_sb + (long long)hkv * k_sh;
  const nv_bfloat16* V_base = V + (long long)b * v_sb + (long long)hkv * v_sh;

  if (num_full_tiles > 0) {
    sdpa_p256_load_kv_unchecked(sK, K_base, k_ss, 0, tid);
  } else {
    sdpa_p256_load_kv_checked(sK, K_base, k_ss, 0, K_len, tid);
  }

  float accO[32][4];
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    accO[t][0] = 0.0f;
    accO[t][1] = 0.0f;
    accO[t][2] = 0.0f;
    accO[t][3] = 0.0f;
  }
  float m_row0 = -1e30f;
  float m_row1 = -1e30f;
  float l_row0 = 0.0f;
  float l_row1 = 0.0f;

  const int row0_abs = q_block_start + warp * 16 + group_id;
  const int row1_abs = row0_abs + 8;
  const int clim0 = offset + ((row0_abs < Q_len) ? row0_abs : (Q_len - 1));
  const int clim1 = offset + ((row1_abs < Q_len) ? row1_abs : (Q_len - 1));

  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = (lane >> 4) & 1;

  for (int tile = 0; tile < num_tiles; ++tile) {
)cuda"
R"cuda(
    const int k_start = tile * BC;
    const bool is_masked = (tile >= num_full_tiles);

    sdpa_p256_cp_wait0();
    __syncthreads();

    if (!is_masked) {
      sdpa_p256_load_kv_unchecked(sV, V_base, v_ss, k_start, tid);
    } else {
      sdpa_p256_load_kv_checked(sV, V_base, v_ss, k_start, K_len, tid);
    }

    float s_acc[4][4] = {};
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const int c8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      sdpa_p256_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_p256_swiz256(k_ld_r0, c8)]);
      sdpa_p256_mma(
          s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      sdpa_p256_mma(
          s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
      sdpa_p256_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_p256_swiz256(k_ld_r0 + 16, c8)]);
      sdpa_p256_mma(
          s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
      sdpa_p256_mma(
          s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
    }
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      unsigned int a0, a1, a2, a3;
      sdpa_p256_ldmatrix4(
          a0, a1, a2, a3, &sQ_hi[sdpa_p256_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
      const int c8 = k_ld_c8_base + 16 + kt * 2;
      unsigned int b0, b1, b2, b3;
      sdpa_p256_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_p256_swiz256(k_ld_r0, c8)]);
      sdpa_p256_mma(
          s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
          a0, a1, a2, a3, b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      sdpa_p256_mma(
          s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
          a0, a1, a2, a3, b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
      sdpa_p256_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_p256_swiz256(k_ld_r0 + 16, c8)]);
      sdpa_p256_mma(
          s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
          a0, a1, a2, a3, b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
      sdpa_p256_mma(
          s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
          a0, a1, a2, a3, b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
    }

    float local_max0 = -1e30f;
    float local_max1 = -1e30f;
    if (!is_masked) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        s_acc[nt][0] *= scale_log2;
        s_acc[nt][1] *= scale_log2;
        s_acc[nt][2] *= scale_log2;
        s_acc[nt][3] *= scale_log2;
        local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
        local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
      }
    } else {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int c0 = k_start + nt * 8 + 2 * tid_in_grp;
        const int c1 = c0 + 1;
        s_acc[nt][0] = (c0 <= clim0) ? (s_acc[nt][0] * scale_log2) : -1e30f;
        s_acc[nt][1] = (c1 <= clim0) ? (s_acc[nt][1] * scale_log2) : -1e30f;
        s_acc[nt][2] = (c0 <= clim1) ? (s_acc[nt][2] * scale_log2) : -1e30f;
        s_acc[nt][3] = (c1 <= clim1) ? (s_acc[nt][3] * scale_log2) : -1e30f;
        local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
        local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
      }
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float alpha0 = (m_row0 > -1e20f) ? sdpa_p256_exp2f(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? sdpa_p256_exp2f(m_row1 - new_m1) : 0.0f;

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    unsigned int p_frag[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int nt0 = ks * 2;
      const int nt1 = ks * 2 + 1;
      const float p00 = sdpa_p256_exp2f(s_acc[nt0][0] - new_m0);
      const float p01 = sdpa_p256_exp2f(s_acc[nt0][1] - new_m0);
      const float p10 = sdpa_p256_exp2f(s_acc[nt0][2] - new_m1);
      const float p11 = sdpa_p256_exp2f(s_acc[nt0][3] - new_m1);
      const float p02 = sdpa_p256_exp2f(s_acc[nt1][0] - new_m0);
      const float p03 = sdpa_p256_exp2f(s_acc[nt1][1] - new_m0);
      const float p12 = sdpa_p256_exp2f(s_acc[nt1][2] - new_m1);
      const float p13 = sdpa_p256_exp2f(s_acc[nt1][3] - new_m1);
      local_sum0 += (p00 + p01) + (p02 + p03);
      local_sum1 += (p10 + p11) + (p12 + p13);
      p_frag[ks][0] = sdpa_p256_pack_bf16x2(p00, p01);
      p_frag[ks][1] = sdpa_p256_pack_bf16x2(p10, p11);
      p_frag[ks][2] = sdpa_p256_pack_bf16x2(p02, p03);
      p_frag[ks][3] = sdpa_p256_pack_bf16x2(p12, p13);
    }
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 1);
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 2);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 1);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 2);

#pragma unroll
    for (int t = 0; t < 32; ++t) {
      accO[t][0] *= alpha0;
      accO[t][1] *= alpha0;
      accO[t][2] *= alpha1;
      accO[t][3] *= alpha1;
    }
    l_row0 = l_row0 * alpha0 + local_sum0;
    l_row1 = l_row1 * alpha1 + local_sum1;
    m_row0 = new_m0;
    m_row1 = new_m1;

    sdpa_p256_cp_wait0();
    __syncthreads();

    if (tile + 1 < num_tiles) {
      if (tile + 1 < num_full_tiles) {
        sdpa_p256_load_kv_unchecked(sK, K_base, k_ss, (tile + 1) * BC, tid);
      } else {
        sdpa_p256_load_kv_checked(sK, K_base, k_ss, (tile + 1) * BC, K_len, tid);
      }
    }

#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int vr = ks * 16 + v_ld_r0;
#pragma unroll
      for (int dt = 0; dt < 16; ++dt) {
        const int c8 = v_ld_c8_base + dt * 2;
        unsigned int b0, b1, b2, b3;
        sdpa_p256_ldmatrix4_trans(b0, b1, b2, b3, &sV[sdpa_p256_swiz256(vr, c8)]);
        const int t0 = dt * 2;
        const int t1 = dt * 2 + 1;
        sdpa_p256_mma(
            accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b0, b1, accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3]);
        sdpa_p256_mma(
            accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b2, b3, accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3]);
      }
    }
  }

  const float inv_l0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
  const float inv_l1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
  nv_bfloat16* O_base = O + (((long long)b * (Hkv * gqa) + hq) * Q_len) * D;
  if (row0_abs < Q_len) {
    nv_bfloat16* dst0 = O_base + (long long)row0_abs * D;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      *reinterpret_cast<unsigned int*>(&dst0[t * 8 + 2 * tid_in_grp]) =
          sdpa_p256_pack_bf16x2(accO[t][0] * inv_l0, accO[t][1] * inv_l0);
    }
  }
  if (row1_abs < Q_len) {
    nv_bfloat16* dst1 = O_base + (long long)row1_abs * D;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      *reinterpret_cast<unsigned int*>(&dst1[t * 8 + 2 * tid_in_grp]) =
          sdpa_p256_pack_bf16x2(accO[t][2] * inv_l1, accO[t][3] * inv_l1);
    }
  }
}
)cuda"
