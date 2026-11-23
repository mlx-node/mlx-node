R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float sdpa_few_exp2f(float x) {
  return ptx_ex2_approx_ftz(x);
}

__device__ __forceinline__ void sdpa_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  ptx_mma_m16n8k16_bf16_c_v(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
}

__device__ __forceinline__ void sdpa_few_ldmatrix4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4(r0, r1, r2, r3, smem_ptr);
}

__device__ __forceinline__ void sdpa_few_ldmatrix4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4_trans(r0, r1, r2, r3, smem_ptr);
}

__device__ __forceinline__ void sdpa_few_cp16(void* smem_ptr, const void* glob_ptr) {
  ptx_cp_async16_cg(smem_ptr, glob_ptr);
}

__device__ __forceinline__ void sdpa_few_cp16_zfill(
    void* smem_ptr, const void* glob_ptr, bool pred) {
  ptx_cp_async16_cg_zfill(smem_ptr, glob_ptr, pred);
}

__device__ __forceinline__ void sdpa_few_cp_commit() {
  ptx_cp_commit();
}

__device__ __forceinline__ void sdpa_few_cp_wait0() {
  ptx_cp_wait<0>();
}

__device__ __forceinline__ unsigned int sdpa_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned int*>(&v);
}

template <int W>
__device__ __forceinline__ int sdpa_few_swiz(int row, int c8) {
  return row * W + ((c8 ^ (row & 7)) << 3);
}

template <int D, int NUM_WARPS>
__device__ __forceinline__ void sdpa_few_load_kv_unchecked(
    nv_bfloat16* dst, const nv_bfloat16* base, long long stride,
    int k_start, int tid) {
  constexpr int C8 = D / 8;
  constexpr int ROWS_STEP = (NUM_WARPS * 32) / C8;
  const int r0 = tid / C8;
  const int c8 = tid & (C8 - 1);
  const nv_bfloat16* gptr = base + (long long)(k_start + r0) * stride + c8 * 8;
  const long long step = (long long)ROWS_STEP * stride;
#pragma unroll 1
  for (int r = r0; r < 32; r += ROWS_STEP) {
    sdpa_few_cp16(&dst[sdpa_few_swiz<D>(r, c8)], gptr);
    gptr += step;
  }
  sdpa_few_cp_commit();
}

template <int D, int NUM_WARPS>
__device__ __forceinline__ void sdpa_few_load_kv_checked(
    nv_bfloat16* dst, const nv_bfloat16* base, long long stride,
    int k_start, int K_len, int tid) {
  constexpr int C8 = D / 8;
  constexpr int ROWS_STEP = (NUM_WARPS * 32) / C8;
  const int r0 = tid / C8;
  const int c8 = tid & (C8 - 1);
  long long off = (long long)(k_start + r0) * stride + c8 * 8;
  const long long step = (long long)ROWS_STEP * stride;
#pragma unroll 1
  for (int r = r0; r < 32; r += ROWS_STEP) {
    const bool ok = (k_start + r < K_len);
    const nv_bfloat16* gp = ok ? (base + off) : base;
    sdpa_few_cp16_zfill(&dst[sdpa_few_swiz<D>(r, c8)], gp, ok);
    off += step;
  }
  sdpa_few_cp_commit();
}
)cuda"
R"cuda(
// Flash-Decoding Tensor Core SDPA for D in {128, 256}, Q_len in 1..=16.
// - Q[0..127] held in registers; for D=256, Q[128..255] staged in sQ_hi (Q_HI_MODE=1,
//   NUM_WARPS<=3, 0 spills, <=45 KB SMEM) or L1 __ldg (Q_HI_MODE=0, NUM_WARPS>=4, 32.8 KB SMEM).
// - sK and sV double-buffered with cp.async.cg + 128B XOR swizzle, overlapping V fetch
//   with QK^T+softmax and next K fetch with PV MMA (only 2 __syncthreads()/tile).
// - ex2.approx.ftz.f32 softmax in log2 space, normalized BF16 partial_O output.
// - MASK_MODE: 0=causal, 1=unmasked, 2=bool mask [B_m,1,Q_m,K_len],
//              3=tree bitmask uint32[Q_len] + cache_len.
// - Grid: (num_splits, Hkv, B * num_q_tiles), Block: (32, NUM_WARPS).
template <int D, int NUM_WARPS, int MASK_MODE, int Q_HI_MODE>
__global__ __launch_bounds__(NUM_WARPS * 32, (NUM_WARPS <= 3) ? 2 : 1)
void mlx_node_sdpa_tc_split(
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    const void* __restrict__ mask_ptr,
    nv_bfloat16* __restrict__ partial_O,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int Hkv, int gqa, int Q_len, int K_len, int cache_len,
    long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss,
    float scale_log2, int num_splits, int num_q_tiles,
    long long mask_sb, long long mask_sq) {
  constexpr int BN = 32;
  constexpr int D_TILES = D / 8;
  const int split = blockIdx.x;
  const int hkv = blockIdx.y;
  const int bz = blockIdx.z;
  const int b = (num_q_tiles == 2) ? (bz >> 1) : bz;
  const int qtile = (num_q_tiles == 2) ? (bz & 1) : 0;
  const int q_tile_offset = qtile * 8;
  const int q_tile_len = (num_q_tiles == 2)
      ? ((qtile == 0) ? 8 : (Q_len - 8))
      : Q_len;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int M_total = gqa * Q_len;
  const int M_tile = gqa * q_tile_len;

  const int total_tiles = (K_len + BN - 1) / BN;
  const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
  const int tile_start = split * tiles_per_split;
  const int tile_end = (tile_start + tiles_per_split < total_tiles)
      ? (tile_start + tiles_per_split) : total_tiles;

  __shared__ __align__(128) nv_bfloat16 sK[BN * D];
  __shared__ __align__(128) nv_bfloat16 sV[BN * D];
  __shared__ __align__(128) nv_bfloat16 sQ_hi[(D == 256 && Q_HI_MODE == 1) ? (NUM_WARPS * 16 * 128) : 8];
  __shared__ unsigned int s_tree_mask[16];
  if constexpr (MASK_MODE == 3) {
    if (tid < Q_len) {
      s_tree_mask[tid] = reinterpret_cast<const unsigned int*>(mask_ptr)[tid];
    }
  }

  const int m_base = warp * 16;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int row0 = m_base + group_id;
  const int row1 = m_base + group_id + 8;
  const int head0 = (row0 < M_tile) ? (row0 / q_tile_len) : 0;
  const int head1 = (row1 < M_tile) ? (row1 / q_tile_len) : 0;
  const int q_pos0 = (row0 < M_tile) ? (q_tile_offset + (row0 % q_tile_len)) : 0;
  const int q_pos1 = (row1 < M_tile) ? (q_tile_offset + (row1 % q_tile_len)) : 0;
  const int global_row0 = head0 * Q_len + q_pos0;
  const int global_row1 = head1 * Q_len + q_pos1;

  const nv_bfloat16* Q_base = Q + ((long long)b * Hkv + hkv) * (long long)M_total * D;
  const nv_bfloat16* q_ptr0 = (row0 < M_tile) ? (Q_base + (long long)global_row0 * D) : nullptr;
  const nv_bfloat16* q_ptr1 = (row1 < M_tile) ? (Q_base + (long long)global_row1 * D) : nullptr;

  unsigned int q_lo[8][4];
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const int k = kt * 16;
    q_lo[kt][0] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 2 * tid_in_grp)) : 0u;
    q_lo[kt][2] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 8 + 2 * tid_in_grp)) : 0u;
    q_lo[kt][1] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 2 * tid_in_grp)) : 0u;
    q_lo[kt][3] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 8 + 2 * tid_in_grp)) : 0u;
  }

  if constexpr (D == 256 && Q_HI_MODE == 1) {
    constexpr int THREADS = NUM_WARPS * 32;
    for (int idx = tid; idx < NUM_WARPS * 16 * 16; idx += THREADS) {
      const int r = idx >> 4;
      const int c8 = idx & 15;
      const bool ok = (r < M_tile);
      const int gr = ok ? ((r / q_tile_len) * Q_len + q_tile_offset + (r % q_tile_len)) : 0;
      const nv_bfloat16* gp = ok ? (Q_base + (long long)gr * D + 128 + c8 * 8) : Q_base;
      sdpa_few_cp16_zfill(&sQ_hi[sdpa_few_swiz<128>(r, c8)], gp, ok);
    }
    sdpa_few_cp_commit();
  }

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

  int unmasked_k_limit = K_len;
  if constexpr (MASK_MODE == 0) {
    unmasked_k_limit = K_len - Q_len;
  } else if constexpr (MASK_MODE == 2) {
    unmasked_k_limit = 0;
  } else if constexpr (MASK_MODE == 3) {
    unmasked_k_limit = cache_len;
  }
  const int num_full_tiles = (unmasked_k_limit > 0) ? (unmasked_k_limit / BN) : 0;

  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = (lane >> 4) & 1;
  const int q_ld_r = warp * 16 + (lane & 15);
  const int q_ld_c8_base = (lane >> 4) & 1;

  if (tile_start < tile_end) {
    if (tile_start < num_full_tiles && (tile_start + 1) * BN <= K_len) {
      sdpa_few_load_kv_unchecked<D, NUM_WARPS>(sK, K_base, k_ss, tile_start * BN, tid);
    } else {
      sdpa_few_load_kv_checked<D, NUM_WARPS>(sK, K_base, k_ss, tile_start * BN, K_len, tid);
    }
  }

  unsigned int tmask0 = 0u;
  unsigned int tmask1 = 0u;
  if constexpr (MASK_MODE == 3) {
    __syncthreads();
    tmask0 = s_tree_mask[q_pos0];
    tmask1 = s_tree_mask[q_pos1];
  }
)cuda"
R"cuda(
  for (int tile = tile_start; tile < tile_end; ++tile) {
    const int k_start = tile * BN;
    const bool is_full =
        (MASK_MODE == 2) ? false : ((tile < num_full_tiles) && (k_start + BN <= K_len));

    sdpa_few_cp_wait0();
    __syncthreads();

    if (k_start + BN <= K_len) {
      sdpa_few_load_kv_unchecked<D, NUM_WARPS>(sV, V_base, v_ss, k_start, tid);
    } else {
      sdpa_few_load_kv_checked<D, NUM_WARPS>(sV, V_base, v_ss, k_start, K_len, tid);
    }

    float s_acc[4][4] = {};
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const int c8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      sdpa_few_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_few_swiz<D>(k_ld_r0, c8)]);
      sdpa_mma_m16n8k16_bf16(
          s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      sdpa_mma_m16n8k16_bf16(
          s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
      sdpa_few_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_few_swiz<D>(k_ld_r0 + 16, c8)]);
      sdpa_mma_m16n8k16_bf16(
          s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
      sdpa_mma_m16n8k16_bf16(
          s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
          q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
          b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
    }
    if constexpr (D == 256) {
      if constexpr (Q_HI_MODE == 1) {
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
          unsigned int a0, a1, a2, a3;
          sdpa_few_ldmatrix4(
              a0, a1, a2, a3, &sQ_hi[sdpa_few_swiz<128>(q_ld_r, q_ld_c8_base + kt * 2)]);
          const int c8 = k_ld_c8_base + 16 + kt * 2;
          unsigned int b0, b1, b2, b3;
          sdpa_few_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_few_swiz<D>(k_ld_r0, c8)]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
              a0, a1, a2, a3, b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
              a0, a1, a2, a3, b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
          sdpa_few_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_few_swiz<D>(k_ld_r0 + 16, c8)]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
              a0, a1, a2, a3, b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
              a0, a1, a2, a3, b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
        }
      } else {
        const nv_bfloat16* qh0 = q_ptr0 ? (q_ptr0 + 128 + 2 * tid_in_grp) : nullptr;
        const nv_bfloat16* qh1 = q_ptr1 ? (q_ptr1 + 128 + 2 * tid_in_grp) : nullptr;
#pragma unroll 2
        for (int kt = 0; kt < 8; ++kt) {
          const int k = kt * 16;
          unsigned int a0 = qh0 ? __ldg(reinterpret_cast<const unsigned int*>(qh0 + k)) : 0u;
          unsigned int a2 = qh0 ? __ldg(reinterpret_cast<const unsigned int*>(qh0 + k + 8)) : 0u;
          unsigned int a1 = qh1 ? __ldg(reinterpret_cast<const unsigned int*>(qh1 + k)) : 0u;
          unsigned int a3 = qh1 ? __ldg(reinterpret_cast<const unsigned int*>(qh1 + k + 8)) : 0u;
          const int c8 = k_ld_c8_base + 16 + kt * 2;
          unsigned int b0, b1, b2, b3;
          sdpa_few_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_few_swiz<D>(k_ld_r0, c8)]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
              a0, a1, a2, a3, b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
              a0, a1, a2, a3, b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
          sdpa_few_ldmatrix4(b0, b1, b2, b3, &sK[sdpa_few_swiz<D>(k_ld_r0 + 16, c8)]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
              a0, a1, a2, a3, b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
          sdpa_mma_m16n8k16_bf16(
              s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
              a0, a1, a2, a3, b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
        }
      }
    }
)cuda"
R"cuda(
    float local_max0 = -1e30f;
    float local_max1 = -1e30f;
    if (is_full) {
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
      if constexpr (MASK_MODE == 2) {
        const int col_base = k_start + 2 * tid_in_grp;
        const unsigned char* bmask =
            reinterpret_cast<const unsigned char*>(mask_ptr) +
            (long long)b * mask_sb;
        const unsigned char* b0 = bmask + (long long)q_pos0 * mask_sq + col_base;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int c0 = col_base + nt * 8;
          const int c1 = c0 + 1;
          const bool v00 = (c0 < K_len) && __ldg(b0 + nt * 8);
          const bool v01 = (c1 < K_len) && __ldg(b0 + nt * 8 + 1);
          s_acc[nt][0] = v00 ? (s_acc[nt][0] * scale_log2) : -1e30f;
          s_acc[nt][1] = v01 ? (s_acc[nt][1] * scale_log2) : -1e30f;
          local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
        }
        const unsigned char* b1 = bmask + (long long)q_pos1 * mask_sq + col_base;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int c0 = col_base + nt * 8;
          const int c1 = c0 + 1;
          const bool v10 = (c0 < K_len) && __ldg(b1 + nt * 8);
          const bool v11 = (c1 < K_len) && __ldg(b1 + nt * 8 + 1);
          s_acc[nt][2] = v10 ? (s_acc[nt][2] * scale_log2) : -1e30f;
          s_acc[nt][3] = v11 ? (s_acc[nt][3] * scale_log2) : -1e30f;
          local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
        }
      } else {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int c0 = k_start + nt * 8 + 2 * tid_in_grp;
          const int c1 = c0 + 1;
          bool v00, v01, v10, v11;
          if constexpr (MASK_MODE == 0) {
            const int causal_limit0 = K_len - Q_len + q_pos0;
            const int causal_limit1 = K_len - Q_len + q_pos1;
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
            auto tree_vis = [&](int c, unsigned int tm) -> bool {
              if (c < cache_len) return true;
              const int j = c - cache_len;
              return (c < K_len) && (j < Q_len) && ((tm >> j) & 1u);
            };
            v00 = tree_vis(c0, tmask0);
            v01 = tree_vis(c1, tmask0);
            v10 = tree_vis(c0, tmask1);
            v11 = tree_vis(c1, tmask1);
          }
          s_acc[nt][0] = v00 ? (s_acc[nt][0] * scale_log2) : -1e30f;
          s_acc[nt][1] = v01 ? (s_acc[nt][1] * scale_log2) : -1e30f;
          s_acc[nt][2] = v10 ? (s_acc[nt][2] * scale_log2) : -1e30f;
          s_acc[nt][3] = v11 ? (s_acc[nt][3] * scale_log2) : -1e30f;
          local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
          local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
        }
      }
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float alpha0 = (m_row0 > -1e20f) ? sdpa_few_exp2f(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? sdpa_few_exp2f(m_row1 - new_m1) : 0.0f;

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    unsigned int p_frag[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int nt0 = ks * 2;
      const int nt1 = ks * 2 + 1;
      float p00, p01, p10, p11, p02, p03, p12, p13;
      if (is_full) {
        p00 = sdpa_few_exp2f(s_acc[nt0][0] - new_m0);
        p01 = sdpa_few_exp2f(s_acc[nt0][1] - new_m0);
        p10 = sdpa_few_exp2f(s_acc[nt0][2] - new_m1);
        p11 = sdpa_few_exp2f(s_acc[nt0][3] - new_m1);
        p02 = sdpa_few_exp2f(s_acc[nt1][0] - new_m0);
        p03 = sdpa_few_exp2f(s_acc[nt1][1] - new_m0);
        p12 = sdpa_few_exp2f(s_acc[nt1][2] - new_m1);
        p13 = sdpa_few_exp2f(s_acc[nt1][3] - new_m1);
      } else {
        p00 = (s_acc[nt0][0] > -1e20f) ? sdpa_few_exp2f(s_acc[nt0][0] - new_m0) : 0.0f;
        p01 = (s_acc[nt0][1] > -1e20f) ? sdpa_few_exp2f(s_acc[nt0][1] - new_m0) : 0.0f;
        p10 = (s_acc[nt0][2] > -1e20f) ? sdpa_few_exp2f(s_acc[nt0][2] - new_m1) : 0.0f;
        p11 = (s_acc[nt0][3] > -1e20f) ? sdpa_few_exp2f(s_acc[nt0][3] - new_m1) : 0.0f;
        p02 = (s_acc[nt1][0] > -1e20f) ? sdpa_few_exp2f(s_acc[nt1][0] - new_m0) : 0.0f;
        p03 = (s_acc[nt1][1] > -1e20f) ? sdpa_few_exp2f(s_acc[nt1][1] - new_m0) : 0.0f;
        p12 = (s_acc[nt1][2] > -1e20f) ? sdpa_few_exp2f(s_acc[nt1][2] - new_m1) : 0.0f;
        p13 = (s_acc[nt1][3] > -1e20f) ? sdpa_few_exp2f(s_acc[nt1][3] - new_m1) : 0.0f;
      }
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

    sdpa_few_cp_wait0();
    __syncthreads();

    if (tile + 1 < tile_end) {
      const int next_start = (tile + 1) * BN;
      if (next_start + BN <= K_len) {
        sdpa_few_load_kv_unchecked<D, NUM_WARPS>(sK, K_base, k_ss, next_start, tid);
      } else {
        sdpa_few_load_kv_checked<D, NUM_WARPS>(sK, K_base, k_ss, next_start, K_len, tid);
      }
    }

#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int vr = ks * 16 + v_ld_r0;
#pragma unroll
      for (int dt = 0; dt < D_TILES / 2; ++dt) {
        const int c8 = v_ld_c8_base + dt * 2;
        unsigned int b0, b1, b2, b3;
        sdpa_few_ldmatrix4_trans(b0, b1, b2, b3, &sV[sdpa_few_swiz<D>(vr, c8)]);
        const int nt0 = dt * 2;
        const int nt1 = dt * 2 + 1;
        sdpa_mma_m16n8k16_bf16(
            accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b0, b1, accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3]);
        sdpa_mma_m16n8k16_bf16(
            accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b2, b3, accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3]);
      }
    }
  }

  const int ep_split = blockIdx.x;
  const int ep_hkv = blockIdx.y;
  const int ep_bz = blockIdx.z;
  const int ep_b = (num_q_tiles == 2) ? (ep_bz >> 1) : ep_bz;
  const int ep_qtile = (num_q_tiles == 2) ? (ep_bz & 1) : 0;
  const int ep_q_off = ep_qtile * 8;
  const int ep_q_len = (num_q_tiles == 2)
      ? ((ep_qtile == 0) ? 8 : (Q_len - 8))
      : Q_len;
  const int ep_M_total = gqa * Q_len;
  const int ep_M_tile = gqa * ep_q_len;
  const int ep_row0 = threadIdx.y * 16 + (threadIdx.x >> 2);
  const int ep_row1 = ep_row0 + 8;
  const int ep_tid_in_grp = threadIdx.x & 3;
  const long long base_split =
      (((long long)ep_b * Hkv + ep_hkv) * num_splits + ep_split) * ep_M_total;
  if (ep_row0 < ep_M_tile) {
    const int ep_gr0 = (ep_row0 / ep_q_len) * Q_len + ep_q_off + (ep_row0 % ep_q_len);
    if (ep_tid_in_grp == 0) {
      partial_m[base_split + ep_gr0] = m_row0;
      partial_l[base_split + ep_gr0] = l_row0;
    }
    const float inv0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
    nv_bfloat16* dst0 = partial_O + (base_split + ep_gr0) * D;
#pragma unroll
    for (int nt = 0; nt < D_TILES; ++nt) {
      *reinterpret_cast<unsigned int*>(dst0 + nt * 8 + 2 * ep_tid_in_grp) =
          sdpa_pack_bf16x2(accO[nt][0] * inv0, accO[nt][1] * inv0);
    }
  }
  if (ep_row1 < ep_M_tile) {
    const int ep_gr1 = (ep_row1 / ep_q_len) * Q_len + ep_q_off + (ep_row1 % ep_q_len);
    if (ep_tid_in_grp == 0) {
      partial_m[base_split + ep_gr1] = m_row1;
      partial_l[base_split + ep_gr1] = l_row1;
    }
    const float inv1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
    nv_bfloat16* dst1 = partial_O + (base_split + ep_gr1) * D;
#pragma unroll
    for (int nt = 0; nt < D_TILES; ++nt) {
      *reinterpret_cast<unsigned int*>(dst1 + nt * 8 + 2 * ep_tid_in_grp) =
          sdpa_pack_bf16x2(accO[nt][2] * inv1, accO[nt][3] * inv1);
    }
  }
}
)cuda"
R"cuda(
// Vectorized 1-warp-per-row BF16 partial_O reduction in log2 space.
// Grid: (M, Hkv, B), Block: (32, 1, 1).
template <int D>
__global__ void mlx_node_sdpa_tc_reduce(
    const nv_bfloat16* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    nv_bfloat16* __restrict__ out,
    int Hkv, int M, int num_splits) {
  const int row = blockIdx.x;
  const int hkv = blockIdx.y;
  const int b = blockIdx.z;
  const int lane = threadIdx.x;
  const long long bh = (long long)b * Hkv + hkv;

  float m_max = -1e30f;
  for (int s = 0; s < num_splits; ++s) {
    m_max = fmaxf(m_max, partial_m[(bh * num_splits + s) * M + row]);
  }
  constexpr int ELEMS = D / 32; // 4 for D=128, 8 for D=256
  float o_sum[ELEMS];
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    o_sum[i] = 0.0f;
  }
  float l_sum = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const long long idx = (bh * num_splits + s) * M + row;
    const float ms = partial_m[idx];
    const float ls = partial_l[idx];
    if (ls > 0.0f) {
      const float wl = ls * sdpa_few_exp2f(ms - m_max);
      l_sum += wl;
      const nv_bfloat16* p = partial_O + idx * D + lane * ELEMS;
      if constexpr (ELEMS == 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          o_sum[2 * i] += __low2float(v2[i]) * wl;
          o_sum[2 * i + 1] += __high2float(v2[i]) * wl;
        }
      } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          o_sum[2 * i] += __low2float(v2[i]) * wl;
          o_sum[2 * i + 1] += __high2float(v2[i]) * wl;
        }
      }
    }
  }
  const float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  nv_bfloat16* dst = out + (bh * M + row) * D + lane * ELEMS;
  if constexpr (ELEMS == 8) {
    uint4 packed;
    unsigned int* u = reinterpret_cast<unsigned int*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u[i] = sdpa_pack_bf16x2(o_sum[2 * i] * inv, o_sum[2 * i + 1] * inv);
    }
    *reinterpret_cast<uint4*>(dst) = packed;
  } else {
    uint2 packed;
    unsigned int* u = reinterpret_cast<unsigned int*>(&packed);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      u[i] = sdpa_pack_bf16x2(o_sum[2 * i] * inv, o_sum[2 * i + 1] * inv);
    }
    *reinterpret_cast<uint2*>(dst) = packed;
  }
}
)cuda"
