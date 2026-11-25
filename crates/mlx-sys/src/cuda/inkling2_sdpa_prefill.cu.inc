R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ float ink2_pre_exp2f(float x) {
  return ptx_ex2_approx_ftz(x);
}

__device__ __forceinline__ float ink2_pre_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ void ink2_pre_mma(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  ptx_mma_m16n8k16_bf16_c_v(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
}

__device__ __forceinline__ void ink2_pre_ldmatrix4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4(r0, r1, r2, r3, smem_ptr);
}

__device__ __forceinline__ void ink2_pre_ldmatrix4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4_trans(r0, r1, r2, r3, smem_ptr);
}

__device__ __forceinline__ unsigned int ink2_pre_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ unsigned int ink2_pre_s8x2_to_bf16x2(unsigned short packed) {
  const float f0 = static_cast<float>(static_cast<signed char>(packed & 0xFFu));
  const float f1 = static_cast<float>(static_cast<signed char>(packed >> 8));
  const __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ unsigned int ink2_pre_s8_pair_to_bf16x2(
    signed char lo, signed char hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(static_cast<float>(lo), static_cast<float>(hi));
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ int ink2_pre_swiz_bf16(int row, int c8) {
  return row * 128 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ int ink2_pre_swiz_i8(int row, int c16) {
  return row * 128 + ((c16 ^ (row & 7)) << 4);
}

__device__ __forceinline__ int ink2_pre_phys_slot(
    int k_j, int cache_len, int kv_phys_base, int write_offset, int ring_cap) {
  if (k_j < cache_len) {
    return (ring_cap > 0) ? ((kv_phys_base + k_j) % ring_cap) : k_j;
  }
  const int t_j = k_j - cache_len;
  return (ring_cap > 0) ? ((write_offset + t_j) % ring_cap) : (write_offset + t_j);
}

__device__ __forceinline__ int ink2_pre_contig_phys0(
    int k_start, int cache_len, int K_total,
    int kv_phys_base, int write_offset, int ring_cap) {
  if (k_start + 32 <= cache_len) {
    if (ring_cap == 0) return k_start;
    const int p0 = (kv_phys_base + k_start) % ring_cap;
    return (p0 + 32 <= ring_cap) ? p0 : -1;
  }
  if (k_start >= cache_len && k_start + 32 <= K_total) {
    const int t0 = k_start - cache_len;
    if (ring_cap == 0) return write_offset + t0;
    const int p0 = (write_offset + t0) % ring_cap;
    return (p0 + 32 <= ring_cap) ? p0 : -1;
  }
  return -1;
}

__device__ __forceinline__ void ink2_pre_load_kv_bf16(
    nv_bfloat16* dst, const nv_bfloat16* base, long long stride,
    int k_start, int cache_len, int K_total,
    int kv_phys_base, int write_offset, int ring_cap, int tid) {
  const int r0 = tid >> 4;
  const int c8 = tid & 15;
  const int phys0 = ink2_pre_contig_phys0(
      k_start, cache_len, K_total, kv_phys_base, write_offset, ring_cap);
  if (phys0 >= 0) {
    const nv_bfloat16* gptr = base + (long long)(phys0 + r0) * stride + c8 * 8;
    const long long step = 8LL * stride;
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
      ptx_cp_async16_cg(&dst[ink2_pre_swiz_bf16(r0 + i * 8, c8)], gptr);
      gptr += step;
    }
  } else {
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
      const int r = r0 + i * 8;
      const int k_j = k_start + r;
      const bool ok = (k_j < K_total);
      const int phys = ok
          ? ink2_pre_phys_slot(k_j, cache_len, kv_phys_base, write_offset, ring_cap)
          : 0;
      const nv_bfloat16* gptr = ok ? (base + (long long)phys * stride + c8 * 8) : base;
      ptx_cp_async16_cg_zfill(&dst[ink2_pre_swiz_bf16(r, c8)], gptr, ok);
    }
  }
  ptx_cp_commit();
}

__device__ __forceinline__ void ink2_pre_load_kv_i8(
    signed char* dst, const signed char* base, long long stride,
    int k_start, int cache_len, int K_total,
    int kv_phys_base, int write_offset, int ring_cap, int tid) {
  const int r0 = tid >> 3;
  const int c16 = tid & 7;
  const int phys0 = ink2_pre_contig_phys0(
      k_start, cache_len, K_total, kv_phys_base, write_offset, ring_cap);
  if (phys0 >= 0) {
    const signed char* gptr = base + (long long)(phys0 + r0) * stride + c16 * 16;
    const long long step = 16LL * stride;
#pragma unroll 1
    for (int i = 0; i < 2; ++i) {
      ptx_cp_async16_cg(&dst[ink2_pre_swiz_i8(r0 + i * 16, c16)], gptr);
      gptr += step;
    }
  } else {
#pragma unroll 1
    for (int i = 0; i < 2; ++i) {
      const int r = r0 + i * 16;
      const int k_j = k_start + r;
      const bool ok = (k_j < K_total);
      const int phys = ok
          ? ink2_pre_phys_slot(k_j, cache_len, kv_phys_base, write_offset, ring_cap)
          : 0;
      const signed char* gptr = ok ? (base + (long long)phys * stride + c16 * 16) : base;
      ptx_cp_async16_cg_zfill(&dst[ink2_pre_swiz_i8(r, c16)], gptr, ok);
    }
  }
  ptx_cp_commit();
}

__device__ __forceinline__ void ink2_pre_load_scales(
    float* sKScale, float* sVScale,
    const float* k_scales_base, long long ks_ss,
    const float* v_scales_base, long long vs_ss,
    int k_start, int cache_len, int K_total,
    int kv_phys_base, int write_offset, int ring_cap, int tid) {
  if (tid < 32) {
    const int k_j = k_start + tid;
    if (k_j < K_total) {
      const int phys = ink2_pre_phys_slot(
          k_j, cache_len, kv_phys_base, write_offset, ring_cap);
      sKScale[tid] = __ldg(k_scales_base + (long long)phys * ks_ss);
      sVScale[tid] = __ldg(v_scales_base + (long long)phys * vs_ss);
    } else {
      sKScale[tid] = 0.0f;
      sVScale[tid] = 0.0f;
    }
  }
}

__device__ __forceinline__ float ink2_pre_rel_dot16(
    float4 r0, float4 r1, float4 r2, float4 r3,
    const float4* __restrict__ w4_base, int dist) {
  const float4* w = w4_base + (long long)dist * 4;
  const float4 w0 = __ldg(w + 0);
  const float4 w1 = __ldg(w + 1);
  const float4 w2 = __ldg(w + 2);
  const float4 w3 = __ldg(w + 3);
  const float d0 = r0.x * w0.x + r0.y * w0.y + r0.z * w0.z + r0.w * w0.w;
  const float d1 = r1.x * w1.x + r1.y * w1.y + r1.z * w1.z + r1.w * w1.w;
  const float d2 = r2.x * w2.x + r2.y * w2.y + r2.z * w2.z + r2.w * w2.w;
  const float d3 = r3.x * w3.x + r3.y * w3.y + r3.z * w3.z + r3.w * w3.w;
  return (d0 + d1) + (d2 + d3);
}
)cuda"
R"cuda(
// Inkling-2-Mini FlashAttention-2 Prefill Kernel (BQ=64, BK=32, D=128, 4 warps = 128 thr).
// - Q[0..127] kept 100% in registers (32 uint32/thr); sR4[64][5] (5 KB, bank-conflict-free).
// - Double-buffered sK/sV via cp.async.cg (16 KB BF16 or 8 KB INT8).
// - Causal upper-triangle tile skip + SWA lower-triangle tile skip + zero-cost rel_bias skip
//   when tile min_dist >= rel_extent.
// - Supports both flat linear KV (ring_cap == 0) and circular ring KV (ring_cap > 0),
//   and both BF16 KV and INT8 KV (per-token FP32 k_scales / v_scales).
// - Fused Gate + Sink epilogue directly to [B, T, Hq * 128] (i.e. [T, 3072] for B=1) BF16.
// Grid: (ceil(T / 64) * gqa, Hkv, B) with reverse q_block for causal load balance.
// Block: (128, 1, 1).
template <bool IS_LOCAL, bool IS_INT8, bool SINK_IS_F32>
__global__ __launch_bounds__(128, 2)
void mlx_inkling2_sdpa_prefill(
    const nv_bfloat16* __restrict__ Q,
    const float* __restrict__ R,
    const void* __restrict__ K_raw,
    const void* __restrict__ V_raw,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const float* __restrict__ rel_bias_w,
    const nv_bfloat16* __restrict__ gate,
    const void* __restrict__ sink,
    nv_bfloat16* __restrict__ O,
    int Hkv, int gqa, int T, int start_pos, int cache_len,
    int kv_start_pos, int kv_phys_base, int write_offset, int ring_cap,
    int sliding_window, int rel_extent,
    long long q_sb, long long q_sh, long long q_ss,
    long long r_sb, long long r_st,
    long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss,
    long long ks_sb, long long ks_sh, long long ks_ss,
    long long vs_sb, long long vs_sh, long long vs_ss,
    long long gate_sb, long long gate_st,
    long long sink_sb, long long sink_st,
    float scale_log2, float log2e) {
  constexpr int BQ = 64;
  constexpr int BK = 32;
  constexpr int D = 128;
  constexpr int D_TILES = 16;

  const int num_q_blocks = (T + BQ - 1) / BQ;
  const int qb_rev = blockIdx.x / gqa;
  const int gqa_idx = blockIdx.x - qb_rev * gqa;
  const int q_block = num_q_blocks - 1 - qb_rev;
  const int hkv = blockIdx.y;
  const int b = blockIdx.z;
  const int Hq = Hkv * gqa;
  const int hq = hkv * gqa + gqa_idx;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int q_block_start = q_block * BQ;
  const int q_block_end = (q_block_start + BQ < T) ? (q_block_start + BQ) : T;
  const int row0_abs = q_block_start + warp * 16 + group_id;
  const int row1_abs = row0_abs + 8;
  const int r0_c = (row0_abs < T) ? row0_abs : (T - 1);
  const int r1_c = (row1_abs < T) ? row1_abs : (T - 1);
  const int q_abs0 = start_pos + r0_c;
  const int q_abs1 = start_pos + r1_c;
  const int min_q_abs = start_pos + q_block_start;
  const int max_q_abs = start_pos + (q_block_end - 1);
  const int K_total = cache_len + T;

  __shared__ __align__(128) nv_bfloat16 sK_bf[IS_INT8 ? 8 : (BK * D)];
  __shared__ __align__(128) nv_bfloat16 sV_bf[IS_INT8 ? 8 : (BK * D)];
  __shared__ __align__(128) signed char sK_i8[IS_INT8 ? (BK * D) : 16];
  __shared__ __align__(128) signed char sV_i8[IS_INT8 ? (BK * D) : 16];
  __shared__ float sKScale[IS_INT8 ? BK : 1];
  __shared__ float sVScale[IS_INT8 ? BK : 1];
  __shared__ __align__(16) float4 sR4[BQ][5];
  (void)sK_bf; (void)sV_bf; (void)sK_i8; (void)sV_i8;
  (void)sKScale; (void)sVScale;

  if (rel_extent > 0) {
    const float* R_head = R + (long long)b * r_sb + (long long)hq * 16;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + i * 128;
      const int r = idx >> 2;
      const int c4 = idx & 3;
      const int gr = q_block_start + r;
      if (gr < T) {
        sR4[r][c4] = __ldg(reinterpret_cast<const float4*>(R_head + (long long)gr * r_st + c4 * 4));
      } else {
        sR4[r][c4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
  }

  const nv_bfloat16* Q_head = Q + (long long)b * q_sb + (long long)hq * q_sh;
  const nv_bfloat16* q_ptr0 = (row0_abs < T) ? (Q_head + (long long)row0_abs * q_ss) : nullptr;
  const nv_bfloat16* q_ptr1 = (row1_abs < T) ? (Q_head + (long long)row1_abs * q_ss) : nullptr;

  unsigned int q_lo[8][4];
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const int k = kt * 16;
    q_lo[kt][0] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 2 * tid_in_grp)) : 0u;
    q_lo[kt][2] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 8 + 2 * tid_in_grp)) : 0u;
    q_lo[kt][1] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 2 * tid_in_grp)) : 0u;
    q_lo[kt][3] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 8 + 2 * tid_in_grp)) : 0u;
  }

  const int k_end_excl = cache_len + q_block_end;
  const int tile_end = (k_end_excl + BK - 1) / BK;
  int tile_first = 0;
  if constexpr (IS_LOCAL) {
    const int win_lo = min_q_abs - (sliding_window - 1);
    int k_begin = 0;
    if (cache_len == 0 || (kv_start_pos + cache_len - 1) < win_lo) {
      const int new_skip = q_block_start - (sliding_window - 1);
      k_begin = cache_len + ((new_skip > 0) ? new_skip : 0);
    } else {
      const int pref_skip = win_lo - kv_start_pos;
      k_begin = (pref_skip > 0) ? pref_skip : 0;
    }
    tile_first = k_begin / BK;
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

  const nv_bfloat16* K_bf_base = IS_INT8 ? nullptr
      : (reinterpret_cast<const nv_bfloat16*>(K_raw) + (long long)b * k_sb + (long long)hkv * k_sh);
  const nv_bfloat16* V_bf_base = IS_INT8 ? nullptr
      : (reinterpret_cast<const nv_bfloat16*>(V_raw) + (long long)b * v_sb + (long long)hkv * v_sh);
  const signed char* K_i8_base = IS_INT8
      ? (reinterpret_cast<const signed char*>(K_raw) + (long long)b * k_sb + (long long)hkv * k_sh)
      : nullptr;
  const signed char* V_i8_base = IS_INT8
      ? (reinterpret_cast<const signed char*>(V_raw) + (long long)b * v_sb + (long long)hkv * v_sh)
      : nullptr;
  const float* ks_base = IS_INT8 ? (k_scales + (long long)b * ks_sb + (long long)hkv * ks_sh) : nullptr;
  const float* vs_base = IS_INT8 ? (v_scales + (long long)b * vs_sb + (long long)hkv * vs_sh) : nullptr;
  const float4* w4_base = reinterpret_cast<const float4*>(rel_bias_w);
  (void)K_bf_base; (void)V_bf_base; (void)K_i8_base; (void)V_i8_base;
  (void)ks_base; (void)vs_base;

  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = (lane >> 4) & 1;
  (void)k_ld_r0; (void)k_ld_c8_base; (void)v_ld_r0; (void)v_ld_c8_base;

  if (tile_first < tile_end) {
    const int k_start0 = tile_first * BK;
    if constexpr (IS_INT8) {
      ink2_pre_load_kv_i8(
          sK_i8, K_i8_base, k_ss, k_start0, cache_len, K_total,
          kv_phys_base, write_offset, ring_cap, tid);
      ink2_pre_load_scales(
          sKScale, sVScale, ks_base, ks_ss, vs_base, vs_ss,
          k_start0, cache_len, K_total, kv_phys_base, write_offset, ring_cap, tid);
    } else {
      ink2_pre_load_kv_bf16(
          sK_bf, K_bf_base, k_ss, k_start0, cache_len, K_total,
          kv_phys_base, write_offset, ring_cap, tid);
    }
  }
)cuda"
R"cuda(
  for (int tile = tile_first; tile < tile_end; ++tile) {
    const int k_start = tile * BK;

    ptx_cp_wait<0>();
    __syncthreads();

    if constexpr (IS_INT8) {
      ink2_pre_load_kv_i8(
          sV_i8, V_i8_base, v_ss, k_start, cache_len, K_total,
          kv_phys_base, write_offset, ring_cap, tid);
    } else {
      ink2_pre_load_kv_bf16(
          sV_bf, V_bf_base, v_ss, k_start, cache_len, K_total,
          kv_phys_base, write_offset, ring_cap, tid);
    }

    float s_acc[4][4] = {};
    if constexpr (!IS_INT8) {
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const int c8 = k_ld_c8_base + kt * 2;
        unsigned int b0, b1, b2, b3;
        ink2_pre_ldmatrix4(b0, b1, b2, b3, &sK_bf[ink2_pre_swiz_bf16(k_ld_r0, c8)]);
        ink2_pre_mma(
            s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
            b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
        ink2_pre_mma(
            s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
            b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
        ink2_pre_ldmatrix4(b0, b1, b2, b3, &sK_bf[ink2_pre_swiz_bf16(k_ld_r0 + 16, c8)]);
        ink2_pre_mma(
            s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
            b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
        ink2_pre_mma(
            s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
            b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
      }
    } else {
      const int k_byte_lo = 2 * tid_in_grp;
      const int k_byte_hi = 8 + 2 * tid_in_grp;
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int kr = nt * 8 + group_id;
          const int base_off = ink2_pre_swiz_i8(kr, kt);
          const unsigned short p0 = *reinterpret_cast<const unsigned short*>(&sK_i8[base_off + k_byte_lo]);
          const unsigned short p1 = *reinterpret_cast<const unsigned short*>(&sK_i8[base_off + k_byte_hi]);
          const unsigned int b0 = ink2_pre_s8x2_to_bf16x2(p0);
          const unsigned int b1 = ink2_pre_s8x2_to_bf16x2(p1);
          ink2_pre_mma(
              s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3],
              q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
              b0, b1, s_acc[nt][0], s_acc[nt][1], s_acc[nt][2], s_acc[nt][3]);
        }
      }
    }

    const bool pure_prefix = (k_start + BK <= cache_len);
    const bool pure_new = (k_start >= cache_len) && (k_start + BK <= K_total);
    bool is_full = false;
    bool tile_needs_rel = (rel_extent > 0);
    int tile_min_k = 0;
    if (pure_prefix || pure_new) {
      tile_min_k = pure_prefix ? (kv_start_pos + k_start) : (start_pos + (k_start - cache_len));
      const int tile_max_k = tile_min_k + (BK - 1);
      is_full = (min_q_abs >= tile_max_k) &&
          (!IS_LOCAL || (max_q_abs - tile_min_k < sliding_window));
      tile_needs_rel = (rel_extent > 0) && ((min_q_abs - tile_max_k) < rel_extent);
    }

    const int s_row0 = warp * 16 + group_id;
    const int s_row1 = s_row0 + 8;
    float local_max0 = -1e30f;
    float local_max1 = -1e30f;
    if (is_full && !tile_needs_rel) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col0 = nt * 8 + 2 * tid_in_grp;
        const float sc0 = IS_INT8 ? (sKScale[col0] * scale_log2) : scale_log2;
        const float sc1 = IS_INT8 ? (sKScale[col0 + 1] * scale_log2) : scale_log2;
        s_acc[nt][0] *= sc0;
        s_acc[nt][1] *= sc1;
        s_acc[nt][2] *= sc0;
        s_acc[nt][3] *= sc1;
        local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
        local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
      }
    } else if (is_full) {
      const float4 r0 = sR4[s_row0][0];
      const float4 r1 = sR4[s_row0][1];
      const float4 r2 = sR4[s_row0][2];
      const float4 r3 = sR4[s_row0][3];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col0 = nt * 8 + 2 * tid_in_grp;
        const float sc0 = IS_INT8 ? (sKScale[col0] * scale_log2) : scale_log2;
        const float sc1 = IS_INT8 ? (sKScale[col0 + 1] * scale_log2) : scale_log2;
        const int d0 = q_abs0 - (tile_min_k + col0);
        const int d1 = d0 - 1;
        float v0 = s_acc[nt][0] * sc0;
        float v1 = s_acc[nt][1] * sc1;
        if ((unsigned int)d0 < (unsigned int)rel_extent) {
          v0 += ink2_pre_rel_dot16(r0, r1, r2, r3, w4_base, d0) * log2e;
        }
        if ((unsigned int)d1 < (unsigned int)rel_extent) {
          v1 += ink2_pre_rel_dot16(r0, r1, r2, r3, w4_base, d1) * log2e;
        }
        s_acc[nt][0] = v0;
        s_acc[nt][1] = v1;
        local_max0 = fmaxf(local_max0, fmaxf(v0, v1));
      }
      const float4 q1_r0 = sR4[s_row1][0];
      const float4 q1_r1 = sR4[s_row1][1];
      const float4 q1_r2 = sR4[s_row1][2];
      const float4 q1_r3 = sR4[s_row1][3];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col0 = nt * 8 + 2 * tid_in_grp;
        const float sc0 = IS_INT8 ? (sKScale[col0] * scale_log2) : scale_log2;
        const float sc1 = IS_INT8 ? (sKScale[col0 + 1] * scale_log2) : scale_log2;
        const int d0 = q_abs1 - (tile_min_k + col0);
        const int d1 = d0 - 1;
        float v0 = s_acc[nt][2] * sc0;
        float v1 = s_acc[nt][3] * sc1;
        if ((unsigned int)d0 < (unsigned int)rel_extent) {
          v0 += ink2_pre_rel_dot16(q1_r0, q1_r1, q1_r2, q1_r3, w4_base, d0) * log2e;
        }
        if ((unsigned int)d1 < (unsigned int)rel_extent) {
          v1 += ink2_pre_rel_dot16(q1_r0, q1_r1, q1_r2, q1_r3, w4_base, d1) * log2e;
        }
        s_acc[nt][2] = v0;
        s_acc[nt][3] = v1;
        local_max1 = fmaxf(local_max1, fmaxf(v0, v1));
      }
    } else {
      const float4 r0 = tile_needs_rel ? sR4[s_row0][0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 r1 = tile_needs_rel ? sR4[s_row0][1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 r2 = tile_needs_rel ? sR4[s_row0][2] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 r3 = tile_needs_rel ? sR4[s_row0][3] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col0 = nt * 8 + 2 * tid_in_grp;
        const int c0 = k_start + col0;
        const int c1 = c0 + 1;
        const float sc0 = IS_INT8 ? (sKScale[col0] * scale_log2) : scale_log2;
        const float sc1 = IS_INT8 ? (sKScale[col0 + 1] * scale_log2) : scale_log2;

        float v0 = -1e30f;
        if (c0 < K_total) {
          const int k_abs = (c0 < cache_len) ? (kv_start_pos + c0) : (start_pos + (c0 - cache_len));
          const int dist = q_abs0 - k_abs;
          if (dist >= 0 && (!IS_LOCAL || dist < sliding_window)) {
            v0 = s_acc[nt][0] * sc0;
            if (tile_needs_rel && (unsigned int)dist < (unsigned int)rel_extent) {
              v0 += ink2_pre_rel_dot16(r0, r1, r2, r3, w4_base, dist) * log2e;
            }
          }
        }
        float v1 = -1e30f;
        if (c1 < K_total) {
          const int k_abs = (c1 < cache_len) ? (kv_start_pos + c1) : (start_pos + (c1 - cache_len));
          const int dist = q_abs0 - k_abs;
          if (dist >= 0 && (!IS_LOCAL || dist < sliding_window)) {
            v1 = s_acc[nt][1] * sc1;
            if (tile_needs_rel && (unsigned int)dist < (unsigned int)rel_extent) {
              v1 += ink2_pre_rel_dot16(r0, r1, r2, r3, w4_base, dist) * log2e;
            }
          }
        }
        s_acc[nt][0] = v0;
        s_acc[nt][1] = v1;
        local_max0 = fmaxf(local_max0, fmaxf(v0, v1));
      }
      const float4 q1_r0 = tile_needs_rel ? sR4[s_row1][0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 q1_r1 = tile_needs_rel ? sR4[s_row1][1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 q1_r2 = tile_needs_rel ? sR4[s_row1][2] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 q1_r3 = tile_needs_rel ? sR4[s_row1][3] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col0 = nt * 8 + 2 * tid_in_grp;
        const int c0 = k_start + col0;
        const int c1 = c0 + 1;
        const float sc0 = IS_INT8 ? (sKScale[col0] * scale_log2) : scale_log2;
        const float sc1 = IS_INT8 ? (sKScale[col0 + 1] * scale_log2) : scale_log2;

        float v0 = -1e30f;
        if (c0 < K_total) {
          const int k_abs = (c0 < cache_len) ? (kv_start_pos + c0) : (start_pos + (c0 - cache_len));
          const int dist = q_abs1 - k_abs;
          if (dist >= 0 && (!IS_LOCAL || dist < sliding_window)) {
            v0 = s_acc[nt][2] * sc0;
            if (tile_needs_rel && (unsigned int)dist < (unsigned int)rel_extent) {
              v0 += ink2_pre_rel_dot16(q1_r0, q1_r1, q1_r2, q1_r3, w4_base, dist) * log2e;
            }
          }
        }
        float v1 = -1e30f;
        if (c1 < K_total) {
          const int k_abs = (c1 < cache_len) ? (kv_start_pos + c1) : (start_pos + (c1 - cache_len));
          const int dist = q_abs1 - k_abs;
          if (dist >= 0 && (!IS_LOCAL || dist < sliding_window)) {
            v1 = s_acc[nt][3] * sc1;
            if (tile_needs_rel && (unsigned int)dist < (unsigned int)rel_extent) {
              v1 += ink2_pre_rel_dot16(q1_r0, q1_r1, q1_r2, q1_r3, w4_base, dist) * log2e;
            }
          }
        }
        s_acc[nt][2] = v0;
        s_acc[nt][3] = v1;
        local_max1 = fmaxf(local_max1, fmaxf(v0, v1));
      }
    }
)cuda"
R"cuda(
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float alpha0 = (m_row0 > -1e20f) ? ink2_pre_exp2f(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? ink2_pre_exp2f(m_row1 - new_m1) : 0.0f;

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    unsigned int p_frag[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int nt0 = ks * 2;
      const int nt1 = ks * 2 + 1;
      float p00, p01, p10, p11, p02, p03, p12, p13;
      if (is_full) {
        p00 = ink2_pre_exp2f(s_acc[nt0][0] - new_m0);
        p01 = ink2_pre_exp2f(s_acc[nt0][1] - new_m0);
        p10 = ink2_pre_exp2f(s_acc[nt0][2] - new_m1);
        p11 = ink2_pre_exp2f(s_acc[nt0][3] - new_m1);
        p02 = ink2_pre_exp2f(s_acc[nt1][0] - new_m0);
        p03 = ink2_pre_exp2f(s_acc[nt1][1] - new_m0);
        p12 = ink2_pre_exp2f(s_acc[nt1][2] - new_m1);
        p13 = ink2_pre_exp2f(s_acc[nt1][3] - new_m1);
      } else {
        p00 = (s_acc[nt0][0] > -1e20f) ? ink2_pre_exp2f(s_acc[nt0][0] - new_m0) : 0.0f;
        p01 = (s_acc[nt0][1] > -1e20f) ? ink2_pre_exp2f(s_acc[nt0][1] - new_m0) : 0.0f;
        p10 = (s_acc[nt0][2] > -1e20f) ? ink2_pre_exp2f(s_acc[nt0][2] - new_m1) : 0.0f;
        p11 = (s_acc[nt0][3] > -1e20f) ? ink2_pre_exp2f(s_acc[nt0][3] - new_m1) : 0.0f;
        p02 = (s_acc[nt1][0] > -1e20f) ? ink2_pre_exp2f(s_acc[nt1][0] - new_m0) : 0.0f;
        p03 = (s_acc[nt1][1] > -1e20f) ? ink2_pre_exp2f(s_acc[nt1][1] - new_m0) : 0.0f;
        p12 = (s_acc[nt1][2] > -1e20f) ? ink2_pre_exp2f(s_acc[nt1][2] - new_m1) : 0.0f;
        p13 = (s_acc[nt1][3] > -1e20f) ? ink2_pre_exp2f(s_acc[nt1][3] - new_m1) : 0.0f;
      }
      local_sum0 += (p00 + p01) + (p02 + p03);
      local_sum1 += (p10 + p11) + (p12 + p13);
      if constexpr (IS_INT8) {
        const int c0 = ks * 16 + 2 * tid_in_grp;
        const float vs0 = sVScale[c0];
        const float vs1 = sVScale[c0 + 1];
        const float vs2 = sVScale[c0 + 8];
        const float vs3 = sVScale[c0 + 9];
        p_frag[ks][0] = ink2_pre_pack_bf16x2(p00 * vs0, p01 * vs1);
        p_frag[ks][1] = ink2_pre_pack_bf16x2(p10 * vs0, p11 * vs1);
        p_frag[ks][2] = ink2_pre_pack_bf16x2(p02 * vs2, p03 * vs3);
        p_frag[ks][3] = ink2_pre_pack_bf16x2(p12 * vs2, p13 * vs3);
      } else {
        p_frag[ks][0] = ink2_pre_pack_bf16x2(p00, p01);
        p_frag[ks][1] = ink2_pre_pack_bf16x2(p10, p11);
        p_frag[ks][2] = ink2_pre_pack_bf16x2(p02, p03);
        p_frag[ks][3] = ink2_pre_pack_bf16x2(p12, p13);
      }
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

    ptx_cp_wait<0>();
    __syncthreads();

    if (tile + 1 < tile_end) {
      const int next_start = (tile + 1) * BK;
      if constexpr (IS_INT8) {
        ink2_pre_load_kv_i8(
            sK_i8, K_i8_base, k_ss, next_start, cache_len, K_total,
            kv_phys_base, write_offset, ring_cap, tid);
        ink2_pre_load_scales(
            sKScale, sVScale, ks_base, ks_ss, vs_base, vs_ss,
            next_start, cache_len, K_total, kv_phys_base, write_offset, ring_cap, tid);
      } else {
        ink2_pre_load_kv_bf16(
            sK_bf, K_bf_base, k_ss, next_start, cache_len, K_total,
            kv_phys_base, write_offset, ring_cap, tid);
      }
    }

    if constexpr (!IS_INT8) {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int vr = ks * 16 + v_ld_r0;
#pragma unroll
        for (int dt = 0; dt < D_TILES / 2; ++dt) {
          const int c8 = v_ld_c8_base + dt * 2;
          unsigned int b0, b1, b2, b3;
          ink2_pre_ldmatrix4_trans(b0, b1, b2, b3, &sV_bf[ink2_pre_swiz_bf16(vr, c8)]);
          const int nt0 = dt * 2;
          const int nt1 = dt * 2 + 1;
          ink2_pre_mma(
              accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3],
              p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
              b0, b1, accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3]);
          ink2_pre_mma(
              accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3],
              p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
              b2, b3, accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3]);
        }
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int vr0 = ks * 16 + 2 * tid_in_grp;
        const int vr1 = vr0 + 1;
        const int vr8 = vr0 + 8;
        const int vr9 = vr0 + 9;
#pragma unroll
        for (int dt = 0; dt < D_TILES; ++dt) {
          const int c16 = dt >> 1;
          const int col16 = ((dt & 1) << 3) + group_id;
          const signed char v00 = sV_i8[ink2_pre_swiz_i8(vr0, c16) + col16];
          const signed char v01 = sV_i8[ink2_pre_swiz_i8(vr1, c16) + col16];
          const signed char v10 = sV_i8[ink2_pre_swiz_i8(vr8, c16) + col16];
          const signed char v11 = sV_i8[ink2_pre_swiz_i8(vr9, c16) + col16];
          const unsigned int b0 = ink2_pre_s8_pair_to_bf16x2(v00, v01);
          const unsigned int b1 = ink2_pre_s8_pair_to_bf16x2(v10, v11);
          ink2_pre_mma(
              accO[dt][0], accO[dt][1], accO[dt][2], accO[dt][3],
              p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
              b0, b1, accO[dt][0], accO[dt][1], accO[dt][2], accO[dt][3]);
        }
      }
    }
  }

  // Fused Gate + Sink epilogue directly to [B, T, Hq * 128] BF16.
  const long long HD = (long long)Hq * D;
  nv_bfloat16* O_head = O + (long long)b * T * HD + (long long)hq * D;
  const nv_bfloat16* G_head = gate + (long long)b * gate_sb + (long long)hq * D;
  if (row0_abs < T) {
    const float inv_l0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
    float sink_scale0 = 0.0f;
    if (l_row0 > 0.0f) {
      const float lse0 = (m_row0 + log2f(l_row0)) * 0.6931471805599453f;
      const long long sidx0 =
          (long long)b * sink_sb + (long long)row0_abs * sink_st + (long long)hq;
      const float sv0 = SINK_IS_F32
          ? reinterpret_cast<const float*>(sink)[sidx0]
          : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(sink)[sidx0]);
      sink_scale0 = ink2_pre_sigmoid(lse0 - sv0);
    }
    nv_bfloat16* dst0 = O_head + (long long)row0_abs * HD;
    const nv_bfloat16* g0 = G_head + (long long)row0_abs * gate_st;
#pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
      const int col = t * 8 + 2 * tid_in_grp;
      const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(&g0[col]);
      const float2 gf = __bfloat1622float2(gv);
      const float y0 =
          (__bfloat162float(__float2bfloat16(accO[t][0] * inv_l0)) * ink2_pre_sigmoid(gf.x)) * sink_scale0;
      const float y1 =
          (__bfloat162float(__float2bfloat16(accO[t][1] * inv_l0)) * ink2_pre_sigmoid(gf.y)) * sink_scale0;
      *reinterpret_cast<unsigned int*>(&dst0[col]) = ink2_pre_pack_bf16x2(y0, y1);
    }
  }
  if (row1_abs < T) {
    const float inv_l1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
    float sink_scale1 = 0.0f;
    if (l_row1 > 0.0f) {
      const float lse1 = (m_row1 + log2f(l_row1)) * 0.6931471805599453f;
      const long long sidx1 =
          (long long)b * sink_sb + (long long)row1_abs * sink_st + (long long)hq;
      const float sv1 = SINK_IS_F32
          ? reinterpret_cast<const float*>(sink)[sidx1]
          : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(sink)[sidx1]);
      sink_scale1 = ink2_pre_sigmoid(lse1 - sv1);
    }
    nv_bfloat16* dst1 = O_head + (long long)row1_abs * HD;
    const nv_bfloat16* g1 = G_head + (long long)row1_abs * gate_st;
#pragma unroll
    for (int t = 0; t < D_TILES; ++t) {
      const int col = t * 8 + 2 * tid_in_grp;
      const __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(&g1[col]);
      const float2 gf = __bfloat1622float2(gv);
      const float y0 =
          (__bfloat162float(__float2bfloat16(accO[t][2] * inv_l1)) * ink2_pre_sigmoid(gf.x)) * sink_scale1;
      const float y1 =
          (__bfloat162float(__float2bfloat16(accO[t][3] * inv_l1)) * ink2_pre_sigmoid(gf.y)) * sink_scale1;
      *reinterpret_cast<unsigned int*>(&dst1[col]) = ink2_pre_pack_bf16x2(y0, y1);
    }
  }
}
)cuda"
