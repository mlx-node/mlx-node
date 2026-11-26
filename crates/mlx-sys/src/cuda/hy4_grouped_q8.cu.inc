R"cuda(
// ============================================================================
// Hy4 CUDA Grouped SymQ8 Attention Projection Kernels (attn_k_b / attn_v_b)
//   Requires ptx_helpers.cu.inc and hy4_fp4_helpers.cu.inc prepended.
//   Input:  x      [M, n_heads, in_dim]         bfloat16 (strided view OK!)
//           w      [n_heads * out_dim, in_dim/4] uint32  (SymQ8 XOR 0x80808080)
//           scales [n_heads * out_dim, in_dim/32] float16
//           gate   [M, n_heads * out_dim] or [M, n_heads, out_dim] bf16 (optional, strided OK!)
//   Output: out    [M, n_heads, out_dim]        bfloat16
//   1. M <= 2 (or cc < 8): 8-warp 2x row-pipelined streaming ld.global.cs.v4.u32
//      with bank-conflict-free SMEM x and coalesced epilogue (half-warp <=256, full >256)
//   2. M = 3..16 (sm_80+): 2-stage cp.async.cg + BF16 Tensor Core MMA
//   3. M >= 17 (sm_80+): BM=64, BN=64, BK=64 (or BK=32) 8-warp double-buffered
//      INT8 Tensor Core MMA (m16n8k32) with on-the-fly per-32 x quant, 0 B spills!
// ============================================================================

__device__ __forceinline__ float hy4_gq8_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ unsigned int hy4_gq8_s8x2_to_bf16x2(unsigned int p16) {
  const float f0 = static_cast<float>(static_cast<signed char>(p16));
  const float f1 = static_cast<float>(static_cast<signed char>(p16 >> 8));
  const __nv_bfloat162 h = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&h);
}

__device__ __forceinline__ void hy4_gq8_unpack8_sym(
    unsigned int u0, unsigned int u1, float wf[8]) {
  const unsigned int s0 = u0 ^ 0x80808080u;
  const unsigned int s1 = u1 ^ 0x80808080u;
  wf[0] = static_cast<float>(static_cast<signed char>(s0));
  wf[1] = static_cast<float>(static_cast<signed char>(s0 >> 8));
  wf[2] = static_cast<float>(static_cast<signed char>(s0 >> 16));
  wf[3] = static_cast<float>(static_cast<signed char>(s0 >> 24));
  wf[4] = static_cast<float>(static_cast<signed char>(s1));
  wf[5] = static_cast<float>(static_cast<signed char>(s1 >> 8));
  wf[6] = static_cast<float>(static_cast<signed char>(s1 >> 16));
  wf[7] = static_cast<float>(static_cast<signed char>(s1 >> 24));
}

__device__ __forceinline__ void hy4_gq8_dot8_f2(
    const float wf[8],
    const float2& f0,
    const float2& f1,
    const float2& f2,
    const float2& f3,
    float& d) {
  d = fmaf(f0.x, wf[0], fmaf(f0.y, wf[1], d));
  d = fmaf(f1.x, wf[2], fmaf(f1.y, wf[3], d));
  d = fmaf(f2.x, wf[4], fmaf(f2.y, wf[5], d));
  d = fmaf(f3.x, wf[6], fmaf(f3.y, wf[7], d));
}

// ============================================================================
// 1. Grouped SymQ8 Streaming Few-Row Kernel (M <= 4, or cc < 8 fallback)
//    8 warps/CTA (256 thr) covering BN=32 output rows/CTA (4 rows/warp) with
//    2x software pipelining across pairs of output rows (rowA, rowB) so both
//    ptx_ld_cs_u4 loads are in flight before unpacking and x is converted once!
//    When USE_SMEM_X=true (in_dim <= 512): loads x[m_base.., h, :] into
//    bank-conflict-free s_x[2][M_CAP][32] ONCE per CTA (0 divisions, 1 wave).
//    When HAS_GATE=true: loads gate[m_base.., h, o0..o0+31] coalesced into
//    s_gate[M_CAP][32] alongside s_x (reusing the same __syncthreads()).
//    HALF_WARP=true (in_dim <= 256): 2 rows/iter (16 lanes/row, PAIRS=1).
//    HALF_WARP=false (in_dim > 256): 1 row/iter (32 lanes/row, PAIRS=2).
//    Grid: dim3((out_dim + 31) / 32, n_heads, ceil(M / M_CAP)), Block: dim3(32, 8, 1)
// ============================================================================
template <int M_CAP, bool HALF_WARP, bool USE_SMEM_X, bool HAS_GATE>
__global__ __launch_bounds__(256, (M_CAP >= 4) ? (USE_SMEM_X ? 3 : 2) : 4) void hy4_grouped_q8_few_u4(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const __half* __restrict__ scales,
    const nv_bfloat16* __restrict__ gate,
    nv_bfloat16* __restrict__ out,
    int M, int n_heads, int in_dim, int out_dim,
    long long x_m_stride_u4, long long x_h_stride_u4,
    long long gate_m_stride, long long gate_h_stride) {
  static_assert(M_CAP >= 1 && M_CAP <= 4, "hy4_grouped_q8_few_u4 requires 1 <= M_CAP <= 4");
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  [[maybe_unused]] const int tid = warp * 32 + lane;

  const int o0 = static_cast<int>(blockIdx.x) * 32;
  const int h = static_cast<int>(blockIdx.y);
  const int m_base = static_cast<int>(blockIdx.z) * M_CAP;
  const int m_cnt = min(M_CAP, M - m_base);
  if (h >= n_heads || o0 >= out_dim || m_cnt <= 0) {
    return;
  }

  const int w_u4_per_row = in_dim >> 4;   // in_dim / 16
  const int k_groups = in_dim >> 5;       // in_dim / 32
  [[maybe_unused]] const int x_u4_per_head = in_dim >> 3;  // in_dim / 8
  const size_t xm_u4 = static_cast<size_t>(x_m_stride_u4);
  const size_t xh_u4 = static_cast<size_t>(x_h_stride_u4);
  const uint4* x_u4_base = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(m_base) * xm_u4 + static_cast<size_t>(h) * xh_u4;

  // Split even/odd uint4 halves so consecutive lanes u=0..31 have 16B stride (0 bank conflicts).
  [[maybe_unused]] __shared__ alignas(16) uint4
      s_x[USE_SMEM_X ? 2 : 1][USE_SMEM_X ? M_CAP : 1][USE_SMEM_X ? 32 : 1];
  [[maybe_unused]] __shared__ nv_bfloat16 s_gate[HAS_GATE ? M_CAP : 1][HAS_GATE ? 32 : 1];
  __shared__ float s_out[M_CAP][32];
  if constexpr (USE_SMEM_X) {
    const int m_loc = tid >> 6;  // 0..3 (256 threads cover 4 x 64 uint4 in 1 wave)
    const int u_x = tid & 63;    // 0..63
    if (m_loc < m_cnt && u_x < x_u4_per_head) {
      s_x[u_x & 1][m_loc][u_x >> 1] =
          x_u4_base[static_cast<size_t>(m_loc) * xm_u4 + u_x];
    }
  }
  if constexpr (HAS_GATE) {
    const int gm = tid >> 5;  // 0..7
    const int go = tid & 31;  // 0..31
    if (gm < m_cnt && o0 + go < out_dim) {
      const size_t gate_idx =
          static_cast<size_t>(m_base + gm) * static_cast<size_t>(gate_m_stride) +
          static_cast<size_t>(h) * static_cast<size_t>(gate_h_stride) + (o0 + go);
      s_gate[gm][go] = gate[gate_idx];
    }
  }
  if constexpr (USE_SMEM_X || HAS_GATE) {
    __syncthreads();
  }

  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const size_t head_w_row0 = static_cast<size_t>(h) * out_dim;

  constexpr int LANES = HALF_WARP ? 16 : 32;
  constexpr int ROWS_PER_ITER = HALF_WARP ? 2 : 1;
  constexpr int PAIRS = HALF_WARP ? 1 : 2;  // 4 rows/warp = 1 pair (HALF_WARP) or 2 pairs
  const int sub_lane = lane & (LANES - 1);
  const int half_id = HALF_WARP ? (lane >> 4) : 0;
  const unsigned int shfl_mask =
      HALF_WARP ? (half_id ? 0xFFFF0000u : 0x0000FFFFu) : 0xFFFFFFFFu;

#pragma unroll 1
  for (int pair = 0; pair < PAIRS; ++pair) {
    const int oA_loc = warp * 4 + pair * (2 * ROWS_PER_ITER) + half_id;
    const int oB_loc = oA_loc + ROWS_PER_ITER;
    const int oA = o0 + oA_loc;
    const int oB = o0 + oB_loc;
    if (oA >= out_dim) {
      continue;
    }
    const bool validB = (oB < out_dim);

    const size_t g_rowA = head_w_row0 + oA;
    const uint4* w_rowA = w_u4 + g_rowA * w_u4_per_row;
    const __half* s_rowA = scales + g_rowA * k_groups;
    const size_t g_rowB = head_w_row0 + (validB ? oB : oA);
    const uint4* w_rowB = w_u4 + g_rowB * w_u4_per_row;
    const __half* s_rowB = scales + g_rowB * k_groups;

    float accA[M_CAP];
    float accB[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      accA[m] = 0.0f;
      accB[m] = 0.0f;
    }

    for (int u = sub_lane; u < w_u4_per_row; u += LANES) {
      // Back-to-back DRAM streaming loads for rowA and rowB before unpacking rowA.
      const uint4 wvA = ptx_ld_cs_u4(w_rowA + u);
      const float scA = __half2float(s_rowA[u >> 1]);
      uint4 wvB = make_uint4(0u, 0u, 0u, 0u);
      float scB = 0.0f;
      if (validB) {
        wvB = ptx_ld_cs_u4(w_rowB + u);
        scB = __half2float(s_rowB[u >> 1]);
      }

      float dA[M_CAP];
      float dB[M_CAP];
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        dA[m] = 0.0f;
        dB[m] = 0.0f;
      }

      float wfA[8];
      float wfB[8];
      hy4_gq8_unpack8_sym(wvA.x, wvA.y, wfA);
      if (validB) {
        hy4_gq8_unpack8_sym(wvB.x, wvB.y, wfB);
      }
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < m_cnt) {
          const uint4 x0 = USE_SMEM_X
              ? s_x[0][m][u]
              : x_u4_base[static_cast<size_t>(m) * xm_u4 + static_cast<size_t>(u) * 2];
          const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&x0);
          const float2 f0 = __bfloat1622float2(p[0]);
          const float2 f1 = __bfloat1622float2(p[1]);
          const float2 f2 = __bfloat1622float2(p[2]);
          const float2 f3 = __bfloat1622float2(p[3]);
          hy4_gq8_dot8_f2(wfA, f0, f1, f2, f3, dA[m]);
          if (validB) {
            hy4_gq8_dot8_f2(wfB, f0, f1, f2, f3, dB[m]);
          }
        }
      }

      hy4_gq8_unpack8_sym(wvA.z, wvA.w, wfA);
      if (validB) {
        hy4_gq8_unpack8_sym(wvB.z, wvB.w, wfB);
      }
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < m_cnt) {
          const uint4 x1 = USE_SMEM_X
              ? s_x[1][m][u]
              : x_u4_base[static_cast<size_t>(m) * xm_u4 + static_cast<size_t>(u) * 2 + 1];
          const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&x1);
          const float2 f0 = __bfloat1622float2(p[0]);
          const float2 f1 = __bfloat1622float2(p[1]);
          const float2 f2 = __bfloat1622float2(p[2]);
          const float2 f3 = __bfloat1622float2(p[3]);
          hy4_gq8_dot8_f2(wfA, f0, f1, f2, f3, dA[m]);
          accA[m] = fmaf(scA, dA[m], accA[m]);
          if (validB) {
            hy4_gq8_dot8_f2(wfB, f0, f1, f2, f3, dB[m]);
            accB[m] = fmaf(scB, dB[m], accB[m]);
          }
        }
      }
    }

#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < m_cnt) {
        float vA = accA[m];
        float vB = accB[m];
#pragma unroll
        for (int off = LANES / 2; off > 0; off >>= 1) {
          vA += __shfl_xor_sync(shfl_mask, vA, off, LANES);
          if (validB) {
            vB += __shfl_xor_sync(shfl_mask, vB, off, LANES);
          }
        }
        if (sub_lane == 0) {
          s_out[m][oA_loc] = vA;
          if (validB) {
            s_out[m][oB_loc] = vB;
          }
        }
      }
    }
  }

  // Coalesced 32-lane epilogue: parallel sigmoid(gate) + contiguous 64B bf16 stores.
  __syncthreads();
  const int gm = tid >> 5;  // 0..7 (m_cnt <= 4)
  const int go = tid & 31;  // 0..31
  if (gm < m_cnt && o0 + go < out_dim) {
    float v = s_out[gm][go];
    if constexpr (HAS_GATE) {
      v *= hy4_gq8_sigmoid(__bfloat162float(s_gate[gm][go]));
    }
    const size_t out_idx =
        (static_cast<size_t>(m_base + gm) * n_heads + h) * out_dim + (o0 + go);
    out[out_idx] = __float2bfloat16_rn(v);
  }
}
)cuda"
R"cuda(
// ============================================================================
// 2. Grouped SymQ8 BF16 Tensor Core MMA for M = 3..16 (sm_80+)
//    16 output rows/CTA, 4 warps/CTA (128 thr), 4 groups (128 elems) per tile.
//    Grid: dim3((out_dim + 15) / 16, n_heads, 1), Block: dim3(32, 4, 1)
// ============================================================================
template <int M_TILES, bool HAS_GATE>
__global__ __launch_bounds__(128, 4) void hy4_grouped_q8_few_mma(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const __half* __restrict__ scales,
    const nv_bfloat16* __restrict__ gate,
    nv_bfloat16* __restrict__ out,
    int M, int n_heads, int in_dim, int out_dim,
    long long x_m_stride_u4, long long x_h_stride_u4,
    long long gate_m_stride, long long gate_h_stride) {
  const int o0 = static_cast<int>(blockIdx.x) * 16;
  const int h = static_cast<int>(blockIdx.y);
  if (o0 >= out_dim || h >= n_heads) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int k_groups = in_dim >> 5;   // in_dim / 32
  const int num_tiles = k_groups >> 2; // full 4-group (128-elem) tiles
  const int tail_groups = k_groups & 3;
  const size_t w_stride_u4 = static_cast<size_t>(in_dim) >> 4;
  const size_t xm_u4 = static_cast<size_t>(x_m_stride_u4);
  const size_t xh_u4 = static_cast<size_t>(x_h_stride_u4);
  const size_t head_row0 = static_cast<size_t>(h) * out_dim;

  union SharedBuf {
    struct {
      alignas(16) uint4 w[2][16][8]; // 4,096 B (16 rows x 4 groups x 32B)
      float sc[2][16][5];            // 640 B (+1 padded)
    } pipe;
    float reduce[4][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);
  const int ld_row = tid >> 3;  // 0..15
  const int ld_col = tid & 7;   // 0..7
  const int ld_swz = ld_col ^ (ld_row & 7);
  const int ld_o = (o0 + ld_row < out_dim) ? (o0 + ld_row) : (out_dim - 1);
  const uint4* w_load_base = w_u4 + (head_row0 + ld_o) * w_stride_u4 + ld_col;

  const int sc_row = tid >> 2;  // 0..15 (for tid < 64)
  const int sc_grp = tid & 3;   // 0..3
  const int sc_o = (o0 + sc_row < out_dim) ? (o0 + sc_row) : (out_dim - 1);
  const size_t sc_base_idx = (head_row0 + sc_o) * k_groups + sc_grp;

  const bool valid_m0 = (g < M);
  const uint4* x0_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m0 ? g : 0) * xm_u4 +
      static_cast<size_t>(h) * xh_u4 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  [[maybe_unused]] const uint4* x1_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m1 ? (g + 8) : 0) * xm_u4 +
      static_cast<size_t>(h) * xh_u4 + t;

  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ g;  // (g + 8) & 7 == g

  float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (num_tiles > 0) {
    ptx_cp_async16_cg(&sh.pipe.w[0][ld_row][ld_swz], w_load_base);
    ptx_cp_commit();
    if (tid < 64) {
      sh.pipe.sc[0][sc_row][sc_grp] = __half2float(scales[sc_base_idx]);
    }
    ptx_cp_wait<0>();
    __syncthreads();

    for (int tile = 0; tile < num_tiles; ++tile) {
      const int cur = tile & 1;
      const int nxt = cur ^ 1;
      __half sc_nxt_h = __ushort_as_half(0u);
      if (tile + 1 < num_tiles) {
        const size_t nt = static_cast<size_t>(tile + 1);
        ptx_cp_async16_cg(&sh.pipe.w[nxt][ld_row][ld_swz], w_load_base + nt * 8);
        ptx_cp_commit();
        if (tid < 64) {
          sc_nxt_h = scales[sc_base_idx + nt * 4];
        }
      }

      const uint2 w0 = reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g][swz_r0])[half_u2];
      const uint2 w1 = reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g + 8][swz_r1])[half_u2];
      const size_t x_off = static_cast<size_t>(tile * 4 + warp) * 4;
      const uint4 xv0 = valid_m0 ? x0_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);
      [[maybe_unused]] const uint4 xv1 =
          (M_TILES == 2 && valid_m1) ? x1_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);

      float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      [[maybe_unused]] float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      {
        const unsigned int u0 = w0.x ^ 0x80808080u;
        const unsigned int u1 = w1.x ^ 0x80808080u;
        const unsigned int a0 = hy4_gq8_s8x2_to_bf16x2(u0);
        const unsigned int a1 = hy4_gq8_s8x2_to_bf16x2(u1);
        const unsigned int a2 = hy4_gq8_s8x2_to_bf16x2(u0 >> 16);
        const unsigned int a3 = hy4_gq8_s8x2_to_bf16x2(u1 >> 16);
        ptx_mma_m16n8k16_bf16_v(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.x, xv0.y);
        if constexpr (M_TILES == 2) {
          ptx_mma_m16n8k16_bf16_v(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.x, xv1.y);
        }
      }
      {
        const unsigned int u0 = w0.y ^ 0x80808080u;
        const unsigned int u1 = w1.y ^ 0x80808080u;
        const unsigned int a0 = hy4_gq8_s8x2_to_bf16x2(u0);
        const unsigned int a1 = hy4_gq8_s8x2_to_bf16x2(u1);
        const unsigned int a2 = hy4_gq8_s8x2_to_bf16x2(u0 >> 16);
        const unsigned int a3 = hy4_gq8_s8x2_to_bf16x2(u1 >> 16);
        ptx_mma_m16n8k16_bf16_v(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.z, xv0.w);
        if constexpr (M_TILES == 2) {
          ptx_mma_m16n8k16_bf16_v(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.z, xv1.w);
        }
      }

      const float sc0 = sh.pipe.sc[cur][g][warp];
      const float sc1 = sh.pipe.sc[cur][g + 8][warp];
      total0[0] = fmaf(sc0, d0[0], total0[0]);
      total0[1] = fmaf(sc0, d0[1], total0[1]);
      total0[2] = fmaf(sc1, d0[2], total0[2]);
      total0[3] = fmaf(sc1, d0[3], total0[3]);
      if constexpr (M_TILES == 2) {
        total1[0] = fmaf(sc0, d1[0], total1[0]);
        total1[1] = fmaf(sc0, d1[1], total1[1]);
        total1[2] = fmaf(sc1, d1[2], total1[2]);
        total1[3] = fmaf(sc1, d1[3], total1[3]);
      }

      if (tile + 1 < num_tiles) {
        if (tid < 64) {
          sh.pipe.sc[nxt][sc_row][sc_grp] = __half2float(sc_nxt_h);
        }
        ptx_cp_wait<0>();
      }
      __syncthreads();
    }
  }

  if (tail_groups > 0 && warp < tail_groups) {
    const int grp_idx = num_tiles * 4 + warp;
    const int r0 = (o0 + g < out_dim) ? (o0 + g) : (out_dim - 1);
    const int r1 = (o0 + g + 8 < out_dim) ? (o0 + g + 8) : (out_dim - 1);
    const size_t w_stride_u2 = static_cast<size_t>(in_dim) >> 3;
    const uint2* w_u2 = reinterpret_cast<const uint2*>(w);
    const uint2 w0 = w_u2[(head_row0 + r0) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];
    const uint2 w1 = w_u2[(head_row0 + r1) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];
    const __half sc0_h = scales[(head_row0 + r0) * k_groups + grp_idx];
    const __half sc1_h = scales[(head_row0 + r1) * k_groups + grp_idx];
    const size_t x_off = static_cast<size_t>(grp_idx) * 4;
    const uint4 xv0 = valid_m0 ? x0_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);
    [[maybe_unused]] const uint4 xv1 =
        (M_TILES == 2 && valid_m1) ? x1_row_u4[x_off] : make_uint4(0u, 0u, 0u, 0u);

    float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    {
      const unsigned int u0 = w0.x ^ 0x80808080u;
      const unsigned int u1 = w1.x ^ 0x80808080u;
      const unsigned int a0 = hy4_gq8_s8x2_to_bf16x2(u0);
      const unsigned int a1 = hy4_gq8_s8x2_to_bf16x2(u1);
      const unsigned int a2 = hy4_gq8_s8x2_to_bf16x2(u0 >> 16);
      const unsigned int a3 = hy4_gq8_s8x2_to_bf16x2(u1 >> 16);
      ptx_mma_m16n8k16_bf16_v(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.x, xv0.y);
      if constexpr (M_TILES == 2) {
        ptx_mma_m16n8k16_bf16_v(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.x, xv1.y);
      }
    }
    {
      const unsigned int u0 = w0.y ^ 0x80808080u;
      const unsigned int u1 = w1.y ^ 0x80808080u;
      const unsigned int a0 = hy4_gq8_s8x2_to_bf16x2(u0);
      const unsigned int a1 = hy4_gq8_s8x2_to_bf16x2(u1);
      const unsigned int a2 = hy4_gq8_s8x2_to_bf16x2(u0 >> 16);
      const unsigned int a3 = hy4_gq8_s8x2_to_bf16x2(u1 >> 16);
      ptx_mma_m16n8k16_bf16_v(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, xv0.z, xv0.w);
      if constexpr (M_TILES == 2) {
        ptx_mma_m16n8k16_bf16_v(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, xv1.z, xv1.w);
      }
    }
    const float sc0 = __half2float(sc0_h);
    const float sc1 = __half2float(sc1_h);
    total0[0] = fmaf(sc0, d0[0], total0[0]);
    total0[1] = fmaf(sc0, d0[1], total0[1]);
    total0[2] = fmaf(sc1, d0[2], total0[2]);
    total0[3] = fmaf(sc1, d0[3], total0[3]);
    if constexpr (M_TILES == 2) {
      total1[0] = fmaf(sc0, d1[0], total1[0]);
      total1[1] = fmaf(sc0, d1[1], total1[1]);
      total1[2] = fmaf(sc1, d1[2], total1[2]);
      total1[3] = fmaf(sc1, d1[3], total1[3]);
    }
  }
  __syncthreads();

  sh.reduce[warp][2 * t][g] = total0[0];
  sh.reduce[warp][2 * t + 1][g] = total0[1];
  sh.reduce[warp][2 * t][g + 8] = total0[2];
  sh.reduce[warp][2 * t + 1][g + 8] = total0[3];
  if constexpr (M_TILES == 2) {
    sh.reduce[warp][8 + 2 * t][g] = total1[0];
    sh.reduce[warp][8 + 2 * t + 1][g] = total1[1];
    sh.reduce[warp][8 + 2 * t][g + 8] = total1[2];
    sh.reduce[warp][8 + 2 * t + 1][g + 8] = total1[3];
  }
  __syncthreads();

  for (int idx = tid; idx < M * 16; idx += 128) {
    const int m = idx >> 4;
    const int o_loc = idx & 15;
    const int o = o0 + o_loc;
    if (o < out_dim) {
      float v = sh.reduce[0][m][o_loc] + sh.reduce[1][m][o_loc] +
          sh.reduce[2][m][o_loc] + sh.reduce[3][m][o_loc];
      const size_t out_idx = (static_cast<size_t>(m) * n_heads + h) * out_dim + o;
      if constexpr (HAS_GATE) {
        const size_t gate_idx =
            static_cast<size_t>(m) * static_cast<size_t>(gate_m_stride) +
            static_cast<size_t>(h) * static_cast<size_t>(gate_h_stride) + o;
        v *= hy4_gq8_sigmoid(__bfloat162float(gate[gate_idx]));
      }
      out[out_idx] = __float2bfloat16_rn(v);
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3. Grouped SymQ8 Prefill INT8 Tensor Core MMA (M >= 17, sm_80+)
//    BM=64, BN=64, BK=64 (when in_dim % 64 == 0) or BK=32 (when in_dim % 64 != 0).
//    8 warps (256 thr), 2-stage double-buffered cp.async.cg + on-the-fly per-32
//    symmetric int8 quantization of x in registers -> XOR-swizzled SMEM (0 bank
//    conflicts, 18 KB SMEM at BK=64 / 9 KB at BK=32, 0 B spills!).
//    Grid: dim3(ceil(M / 64) * ceil(out_dim / 64), n_heads, 1), Block: dim3(256, 1, 1)
// ============================================================================
template <int BK, bool HAS_GATE>
__global__ __launch_bounds__(256, 2) void hy4_grouped_q8_prefill_int8(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const __half* __restrict__ scales,
    const nv_bfloat16* __restrict__ gate,
    nv_bfloat16* __restrict__ out,
    int M, int n_heads, int in_dim, int out_dim,
    long long x_m_stride_u4, long long x_h_stride_u4,
    long long gate_m_stride, long long gate_h_stride) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int K_GRPS_STEP = BK / 32;  // 2 for BK=64, 1 for BK=32
  constexpr int U4_COLS = BK / 16;      // 4 for BK=64, 2 for BK=32 (16 int8/uint4)

  const int h = static_cast<int>(blockIdx.y);
  if (h >= n_heads) {
    return;
  }

  const int num_m_tiles = (M + BM - 1) / BM;
  const int num_n_tiles = (out_dim + BN - 1) / BN;
  constexpr int BAND_M = 4;
  const int lin = static_cast<int>(blockIdx.x);
  const int tiles_per_band = BAND_M * num_n_tiles;
  const int band = lin / tiles_per_band;
  const int in_band = lin - band * tiles_per_band;
  const int band_m0 = band * BAND_M;
  const int band_rows = min(BAND_M, num_m_tiles - band_m0);
  const int n_tile = in_band / band_rows;
  const int m_tile = band_m0 + (in_band - n_tile * band_rows);
  if (n_tile >= num_n_tiles || m_tile >= num_m_tiles) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp >> 2;  // 0..1 (32 M-rows each)
  const int wn = warp & 3;   // 0..3 (16 N-cols each)
  const int g = lane >> 2;   // 0..7
  const int t = lane & 3;    // 0..3

  const int m0 = m_tile * BM;
  const int n0 = n_tile * BN;
  const int k_groups = in_dim >> 5;
  const int k_steps = (BK == 64) ? (in_dim >> 6) : (in_dim >> 5);
  const size_t xm_u4 = static_cast<size_t>(x_m_stride_u4);
  const size_t xh_u4 = static_cast<size_t>(x_h_stride_u4);
  const size_t w_stride_u4 = static_cast<size_t>(in_dim) >> 4;
  const size_t head_row0 = static_cast<size_t>(h) * out_dim;

  struct Stage {
    alignas(16) uint4 x[BM][U4_COLS];
    alignas(16) uint4 w[BN][U4_COLS];
    alignas(16) float xsc[K_GRPS_STEP][BM];
    alignas(16) float wsc[K_GRPS_STEP][BN];
  };
  __shared__ Stage smem[2];

  const uint4* x_u4 = reinterpret_cast<const uint4*>(x) + static_cast<size_t>(h) * xh_u4;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);

  auto load_w = [&](int st, int step) {
    if constexpr (BK == 64) {
      const int r = tid >> 2;
      const int c = tid & 3;
      const int gn = n0 + r;
      uint4* dst = &smem[st].w[r][c ^ (r & 3)];
      if (gn < out_dim) {
        ptx_cp_async16_cg(
            dst, w_u4 + (head_row0 + gn) * w_stride_u4 + step * 4 + c);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
      if (tid < 128) {
        const int sr = tid >> 1;
        const int sc = tid & 1;
        const int sgn = n0 + sr;
        smem[st].wsc[sc][sr] = (sgn < out_dim)
            ? __half2float(scales[(head_row0 + sgn) * k_groups + step * 2 + sc])
            : 0.0f;
      }
    } else {
      if (tid < 128) {
        const int r = tid >> 1;
        const int c = tid & 1;
        const int gn = n0 + r;
        if (gn < out_dim) {
          ptx_cp_async16_cg(
              &smem[st].w[r][c], w_u4 + (head_row0 + gn) * w_stride_u4 + step * 2 + c);
        } else {
          smem[st].w[r][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else if (tid < 192) {
        const int sr = tid - 128;
        const int sgn = n0 + sr;
        smem[st].wsc[0][sr] = (sgn < out_dim)
            ? __half2float(scales[(head_row0 + sgn) * k_groups + step])
            : 0.0f;
      }
    }
    ptx_cp_commit();
  };

  const int xr = tid >> 2;       // 0..63
  const int xq = tid & 3;        // 0..3 (for BK=64: xh = xq >> 1, sub = xq & 1)
  const int gm_x = m0 + xr;

  auto fetch_x = [&](int step, uint4& raw0, uint4& raw1) {
    if (gm_x < M) {
      if constexpr (BK == 64) {
        const size_t base = static_cast<size_t>(gm_x) * xm_u4 + step * 8 + xq * 2;
        raw0 = x_u4[base];
        raw1 = x_u4[base + 1];
      } else {
        raw0 = x_u4[static_cast<size_t>(gm_x) * xm_u4 + step * 4 + xq];
        raw1 = make_uint4(0u, 0u, 0u, 0u);
      }
    } else {
      raw0 = make_uint4(0u, 0u, 0u, 0u);
      raw1 = make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto process_x = [&](int st, const uint4& raw0, const uint4& raw1) {
    if constexpr (BK == 64) {
      float v[16];
      float amax = 0.0f;
      if (gm_x < M) {
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&raw0);
        const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&raw1);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f0 = __bfloat1622float2(p0[k]);
          const float2 f1 = __bfloat1622float2(p1[k]);
          v[2 * k] = f0.x;
          v[2 * k + 1] = f0.y;
          v[8 + 2 * k] = f1.x;
          v[8 + 2 * k + 1] = f1.y;
          amax = fmaxf(
              amax,
              fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
        }
      } else {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          v[k] = 0.0f;
        }
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 2));
      const bool valid = (amax > 0.0f) && isfinite(amax);
      const float inv = valid ? (127.0f / amax) : 0.0f;
      unsigned int pk[4];
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const int q0 = valid ? static_cast<int>(rintf(v[w4 * 4 + 0] * inv)) : 0;
        const int q1 = valid ? static_cast<int>(rintf(v[w4 * 4 + 1] * inv)) : 0;
        const int q2 = valid ? static_cast<int>(rintf(v[w4 * 4 + 2] * inv)) : 0;
        const int q3 = valid ? static_cast<int>(rintf(v[w4 * 4 + 3] * inv)) : 0;
        pk[w4] = (static_cast<unsigned int>(q0) & 0xffu) |
            ((static_cast<unsigned int>(q1) & 0xffu) << 8) |
            ((static_cast<unsigned int>(q2) & 0xffu) << 16) |
            ((static_cast<unsigned int>(q3) & 0xffu) << 24);
      }
      smem[st].x[xr][xq ^ (xr & 3)] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
      if ((xq & 1) == 0) {
        smem[st].xsc[xq >> 1][xr] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
      }
    } else {
      float v[8];
      float amax = 0.0f;
      if (gm_x < M) {
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&raw0);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f0 = __bfloat1622float2(p0[k]);
          v[2 * k] = f0.x;
          v[2 * k + 1] = f0.y;
          amax = fmaxf(amax, fmaxf(fabsf(f0.x), fabsf(f0.y)));
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          v[k] = 0.0f;
        }
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 4));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 4));
      const bool valid = (amax > 0.0f) && isfinite(amax);
      const float inv = valid ? (127.0f / amax) : 0.0f;
      int q[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        q[k] = valid ? static_cast<int>(rintf(v[k] * inv)) : 0;
      }
      uint2 pk;
      pk.x = (static_cast<unsigned int>(q[0]) & 0xffu) |
          ((static_cast<unsigned int>(q[1]) & 0xffu) << 8) |
          ((static_cast<unsigned int>(q[2]) & 0xffu) << 16) |
          ((static_cast<unsigned int>(q[3]) & 0xffu) << 24);
      pk.y = (static_cast<unsigned int>(q[4]) & 0xffu) |
          ((static_cast<unsigned int>(q[5]) & 0xffu) << 8) |
          ((static_cast<unsigned int>(q[6]) & 0xffu) << 16) |
          ((static_cast<unsigned int>(q[7]) & 0xffu) << 24);
      reinterpret_cast<uint2*>(&smem[st].x[xr][xq >> 1])[xq & 1] = pk;
      if (xq == 0) {
        smem[st].xsc[0][xr] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
      }
    }
  };

  float acc[2][2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    }
  }
)cuda"
R"cuda(
  if (k_steps > 0) {
    load_w(0, 0);
    uint4 r0_init, r1_init;
    fetch_x(0, r0_init, r1_init);
    process_x(0, r0_init, r1_init);
  }

  const int ld_a_r = wm * 32 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int ld_a_c = lane >> 4;
  const int ld_b_r = lane & 7;
  const int ld_b_c = (lane >> 3) & 1;

#pragma unroll 1
  for (int step = 0; step < k_steps; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    uint4 r0_nxt = make_uint4(0u, 0u, 0u, 0u);
    uint4 r1_nxt = make_uint4(0u, 0u, 0u, 0u);
    if (step + 1 < k_steps) {
      load_w(nxt, step + 1);
      fetch_x(step + 1, r0_nxt, r1_nxt);
      ptx_cp_wait<1>();
    } else {
      ptx_cp_wait<0>();
    }
    __syncthreads();

#pragma unroll
    for (int hg = 0; hg < K_GRPS_STEP; ++hg) {
      const float xs00 = smem[cur].xsc[hg][wm * 32 + g];
      const float xs01 = smem[cur].xsc[hg][wm * 32 + g + 8];
      const float xs10 = smem[cur].xsc[hg][wm * 32 + 16 + g];
      const float xs11 = smem[cur].xsc[hg][wm * 32 + 16 + g + 8];

      unsigned int b0[2], b1[2];
      float ws0[2], ws1[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int nr = wn * 16 + j * 8;
        const int br = nr + ld_b_r;
        const int bc = (BK == 64) ? ((hg * 2 + ld_b_c) ^ (br & 3)) : ld_b_c;
        ptx_ldmatrix_x2(b0[j], b1[j], &smem[cur].w[br][bc]);
        b0[j] ^= 0x80808080u;
        b1[j] ^= 0x80808080u;
        const int c0 = nr + 2 * t;
        ws0[j] = smem[cur].wsc[hg][c0];
        ws1[j] = smem[cur].wsc[hg][c0 + 1];
      }

#pragma unroll
      for (int i = 0; i < 2; ++i) {
        unsigned int a0, a1, a2, a3;
        const int r = ld_a_r + i * 16;
        const int ac = (BK == 64) ? ((hg * 2 + ld_a_c) ^ (r & 3)) : ld_a_c;
        ptx_ldmatrix_x4(a0, a1, a2, a3, &smem[cur].x[r][ac]);
        int c0[4], c1[4];
        ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, b0[0], b1[0], c0[0], c0[1], c0[2], c0[3]);
        ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, b0[1], b1[1], c1[0], c1[1], c1[2], c1[3]);
        const float xs0 = (i == 0) ? xs00 : xs10;
        const float xs1 = (i == 0) ? xs01 : xs11;
        acc[i][0][0] = fmaf(xs0 * ws0[0], static_cast<float>(c0[0]), acc[i][0][0]);
        acc[i][0][1] = fmaf(xs0 * ws1[0], static_cast<float>(c0[1]), acc[i][0][1]);
        acc[i][0][2] = fmaf(xs1 * ws0[0], static_cast<float>(c0[2]), acc[i][0][2]);
        acc[i][0][3] = fmaf(xs1 * ws1[0], static_cast<float>(c0[3]), acc[i][0][3]);
        acc[i][1][0] = fmaf(xs0 * ws0[1], static_cast<float>(c1[0]), acc[i][1][0]);
        acc[i][1][1] = fmaf(xs0 * ws1[1], static_cast<float>(c1[1]), acc[i][1][1]);
        acc[i][1][2] = fmaf(xs1 * ws0[1], static_cast<float>(c1[2]), acc[i][1][2]);
        acc[i][1][3] = fmaf(xs1 * ws1[1], static_cast<float>(c1[3]), acc[i][1][3]);
      }
    }
    if (step + 1 < k_steps) {
      process_x(nxt, r0_nxt, r1_nxt);
      __syncthreads();
    }
  }

  const bool even_out = ((out_dim & 1) == 0);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r0 = m0 + wm * 32 + i * 16 + g;
    const int r1 = r0 + 8;
    const size_t base_r0 = (static_cast<size_t>(r0) * n_heads + h) * out_dim;
    const size_t base_r1 = (static_cast<size_t>(r1) * n_heads + h) * out_dim;
    [[maybe_unused]] const size_t gbase_r0 = HAS_GATE
        ? (static_cast<size_t>(r0) * static_cast<size_t>(gate_m_stride) +
           static_cast<size_t>(h) * static_cast<size_t>(gate_h_stride))
        : 0;
    [[maybe_unused]] const size_t gbase_r1 = HAS_GATE
        ? (static_cast<size_t>(r1) * static_cast<size_t>(gate_m_stride) +
           static_cast<size_t>(h) * static_cast<size_t>(gate_h_stride))
        : 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col0 = n0 + wn * 16 + j * 8 + 2 * t;
      const int col1 = col0 + 1;
      if (r0 < M) {
        float v0 = acc[i][j][0];
        float v1 = acc[i][j][1];
        if constexpr (HAS_GATE) {
          if (col0 < out_dim) v0 *= hy4_gq8_sigmoid(__bfloat162float(gate[gbase_r0 + col0]));
          if (col1 < out_dim) v1 *= hy4_gq8_sigmoid(__bfloat162float(gate[gbase_r0 + col1]));
        }
        if (even_out && col1 < out_dim) {
          *reinterpret_cast<__nv_bfloat162*>(out + base_r0 + col0) = __floats2bfloat162_rn(v0, v1);
        } else {
          if (col0 < out_dim) out[base_r0 + col0] = __float2bfloat16_rn(v0);
          if (col1 < out_dim) out[base_r0 + col1] = __float2bfloat16_rn(v1);
        }
      }
      if (r1 < M) {
        float v0 = acc[i][j][2];
        float v1 = acc[i][j][3];
        if constexpr (HAS_GATE) {
          if (col0 < out_dim) v0 *= hy4_gq8_sigmoid(__bfloat162float(gate[gbase_r1 + col0]));
          if (col1 < out_dim) v1 *= hy4_gq8_sigmoid(__bfloat162float(gate[gbase_r1 + col1]));
        }
        if (even_out && col1 < out_dim) {
          *reinterpret_cast<__nv_bfloat162*>(out + base_r1 + col0) = __floats2bfloat162_rn(v0, v1);
        } else {
          if (col0 < out_dim) out[base_r1 + col0] = __float2bfloat16_rn(v0);
          if (col1 < out_dim) out[base_r1 + col1] = __float2bfloat16_rn(v1);
        }
      }
    }
  }
}
)cuda"
