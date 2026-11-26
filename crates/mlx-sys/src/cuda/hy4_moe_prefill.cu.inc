R"cuda(
// ============================================================================
// Hy4-flash CUDA MoE Prefill Kernels (M >= 1, tuned for M = 17..4096)
//   Zipfian 2-bucket LPT hot (BM=64) / cold (BM=32) routing + 2-stage
//   double-buffered cp.async.cg/ca + BF16 Tensor Core MMA (BK=64, BN=128 or 64)
//   Modes: 0 = NVFP4 (b4/g16, FP8 E4M3 scales), 1 = SymQ8 (b8/g32, s8 = byte ^ 0x80)
//   Dual-target PTX: Windows CUDA 13.4+ HW FP4/FP8 cvt, Linux DGX Spark CUDA 13.0
//   SMEM E2M1 LUT + PTX 8.1 FP8 cvt, sm_80 bitwise SW fallback. 0 B register spills!
// ============================================================================

__device__ __forceinline__ float hy4p_read_f16_or_bf16(unsigned short raw, int is_bf16) {
  return is_bf16 ? __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(&raw))
                 : __half2float(__ushort_as_half(raw));
}

__device__ __forceinline__ void hy4p_unpack8_symq8_bf16x2(
    const uint2& w, unsigned int out[4]) {
  const unsigned int u0 = w.x ^ 0x80808080u;
  const unsigned int u1 = w.y ^ 0x80808080u;
  out[0] = hy4_pack_bf16x2(float(static_cast<signed char>(u0)),
                           float(static_cast<signed char>(u0 >> 8)));
  out[1] = hy4_pack_bf16x2(float(static_cast<signed char>(u0 >> 16)),
                           float(static_cast<signed char>(u0 >> 24)));
  out[2] = hy4_pack_bf16x2(float(static_cast<signed char>(u1)),
                           float(static_cast<signed char>(u1 >> 8)));
  out[3] = hy4_pack_bf16x2(float(static_cast<signed char>(u1 >> 16)),
                           float(static_cast<signed char>(u1 >> 24)));
}

// ---------------------------------------------------------------------------
// 1. Zipfian 2-Bucket Hot (BM=64) / Cold (BM=32) Route Sort + Tile Table
//    - Hot experts (cnt > 32) emit ceil(cnt / 64) contiguous tiles in [0, num_hot)
//    - Cold experts (1 <= cnt <= 32) emit 1 tile in [num_hot, total_tiles)
//    - num_tiles[0] = total_tiles, num_tiles[1] = num_hot_tiles
//    - Supports arbitrary 1 <= E <= 512. Grid: 1, Block: 512.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(512) void hy4_moe_prefill_route_sort(
    const unsigned int* __restrict__ topk_ids,
    int* __restrict__ sorted_pairs,
    int* __restrict__ pair_to_sorted,
    int* __restrict__ tile_expert,
    int* __restrict__ tile_begin,
    int* __restrict__ tile_end,
    int* __restrict__ num_tiles,
    int P, int E) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ int s_cnt[512];
  __shared__ int s_cursor[512];
  __shared__ int s_warp_c[16];
  __shared__ int s_warp_th[16];
  __shared__ int s_warp_tc[16];
  __shared__ int s_tot_hot;

  s_cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < P; i += 512) {
    const unsigned int e = topk_ids[i];
    if (e < static_cast<unsigned int>(E)) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();

  const int c = (tid < E) ? s_cnt[tid] : 0;
  const bool is_hot = (c > 32);
  const int nt_h = (c > 0 && is_hot) ? ((c + 63) >> 6) : 0;
  const int nt_c = (c > 0 && !is_hot) ? 1 : 0;

  int scan_c = c, scan_th = nt_h, scan_tc = nt_c;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int nc = __shfl_up_sync(0xFFFFFFFFu, scan_c, off);
    const int nh = __shfl_up_sync(0xFFFFFFFFu, scan_th, off);
    const int n_c = __shfl_up_sync(0xFFFFFFFFu, scan_tc, off);
    if (lane >= off) {
      scan_c += nc;
      scan_th += nh;
      scan_tc += n_c;
    }
  }
  if (lane == 31) {
    s_warp_c[warp] = scan_c;
    s_warp_th[warp] = scan_th;
    s_warp_tc[warp] = scan_tc;
  }
  __syncthreads();

  if (warp == 0) {
    int wc = (lane < 16) ? s_warp_c[lane] : 0;
    int wh = (lane < 16) ? s_warp_th[lane] : 0;
    int w_c = (lane < 16) ? s_warp_tc[lane] : 0;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
      const int nc = __shfl_up_sync(0xFFFFFFFFu, wc, off);
      const int nh = __shfl_up_sync(0xFFFFFFFFu, wh, off);
      const int n_c = __shfl_up_sync(0xFFFFFFFFu, w_c, off);
      if (lane >= off) {
        wc += nc;
        wh += nh;
        w_c += n_c;
      }
    }
    if (lane == 15) {
      s_tot_hot = wh;
      num_tiles[0] = wh + w_c;
      num_tiles[1] = wh;
    }
    if (lane < 16) {
      s_warp_c[lane] = wc - s_warp_c[lane];
      s_warp_th[lane] = wh - s_warp_th[lane];
      s_warp_tc[lane] = w_c - s_warp_tc[lane];
    }
  }
  __syncthreads();

  const int row_start = s_warp_c[warp] + (scan_c - c);
  s_cursor[tid] = row_start;
  if (tid < E && c > 0) {
    if (is_hot) {
      const int ts = s_warp_th[warp] + (scan_th - nt_h);
      for (int k = 0; k < nt_h; ++k) {
        const int b = row_start + k * 64;
        const int en = (b + 64 < row_start + c) ? (b + 64) : (row_start + c);
        tile_expert[ts + k] = tid;
        tile_begin[ts + k] = b;
        tile_end[ts + k] = en;
      }
    } else {
      const int ts = s_tot_hot + s_warp_tc[warp] + (scan_tc - nt_c);
      tile_expert[ts] = tid;
      tile_begin[ts] = row_start;
      tile_end[ts] = row_start + c;
    }
  }
  __syncthreads();

  for (int i = tid; i < P; i += 512) {
    const unsigned int e = topk_ids[i];
    if (e < static_cast<unsigned int>(E)) {
      const int dst = atomicAdd(&s_cursor[e], 1);
      sorted_pairs[dst] = i;
      pair_to_sorted[i] = dst;
    } else {
      pair_to_sorted[i] = 0;
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2. Prefill Fused Gate + Up + Asymmetric Clamped SwiGLU (BM=64 hot / BM=32 cold)
//    MODE: 0 = NVFP4 (b4/g16), 1 = SymQ8 (b8/g32, s8 = byte ^ 0x80)
//    BN: 128 (8 warps = 256 thr, when ff%128==0) or 64 (4 warps = 128 thr)
//    BK = 64, 2-stage double-buffered cp.async.cg/ca, 0 bank conflicts, 0 B spills!
//    Grid: dim3(ff / BN, grid_y, 1), Block: dim3(32, BN / 16, 1).
// ---------------------------------------------------------------------------
template <int MODE, int BM, int BN, bool IS_COLD>
__global__ __launch_bounds__(
    (BN / 16) * 32,
    (BN == 64) ? (BM == 64 ? 3 : 4) : ((BM == 64) ? 1 : 2))
void hy4_moe_prefill_gate_up(
    const unsigned int* __restrict__ gate_w,
    const void* __restrict__ gate_s,
    const unsigned int* __restrict__ up_w,
    const void* __restrict__ up_s,
    const nv_bfloat16* __restrict__ z,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    nv_bfloat16* __restrict__ mid,
    int lat, int ff, int top_k, float swiglu_limit, int sb_is_bf16,
    int z_row_u4) {
  const int tile_idx = IS_COLD ? (num_tiles[1] + static_cast<int>(blockIdx.y))
                               : static_cast<int>(blockIdx.y);
  const int tile_limit = IS_COLD ? num_tiles[0] : num_tiles[1];
  if (tile_idx >= tile_limit) return;

  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;

  constexpr int WARPS = BN / 16;
  constexpr int THREADS = WARPS * 32;
  constexpr int N_SLICES_MAX = BM / 8;
  constexpr int N_PASSES_MAX = BM / 16;
  constexpr int W_U4 = (MODE == 0) ? 2 : 4;
  constexpr int X_SWIZ_SHIFT = (MODE == 0) ? 1 : 2;
  constexpr int X_SWIZ_MASK = (MODE == 0) ? 3 : 1;

  const int base_row = blockIdx.x * BN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  union SharedGu {
    struct {
      alignas(16) uint4 w[2][BN][W_U4];
      alignas(16) uint4 x[2][BM][8];
      alignas(16) unsigned int sc[2][BN];
      alignas(16) int tok[BM];
    } pipe;
    alignas(16) nv_bfloat16 s_out[WARPS][16][16];
  };
  __shared__ SharedGu sh;
  __shared__ size_t s_exp_row0;
#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    for (int i = tid; i < 256; i += THREADS) {
      s_e2m1_lut[i] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(i));
    }
  } else {
    (void)s_e2m1_lut;
  }
#endif

  if (tid < BM) {
    sh.pipe.tok[tid] = (tid < cnt) ? (sorted_pairs[begin + tid] / top_k) : -1;
  }
  if (tid == 0) {
    s_exp_row0 = static_cast<size_t>(expert) * ff + base_row;
  }
  __syncthreads();

  const int n_steps = lat >> 6;
  const int row_w_u4 = (MODE == 0) ? (lat >> 5) : (lat >> 4);

  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_w = [&](int wst, const unsigned int* w_base, const void* s_base, int step) {
    const size_t er0 = s_exp_row0;
    const uint4* w_src = reinterpret_cast<const uint4*>(w_base) + er0 * row_w_u4;
    const unsigned int* s_src = reinterpret_cast<const unsigned int*>(s_base) + er0 * n_steps;
    if constexpr (MODE == 0) {
      const int r = tid >> 1;
      const int c = tid & 1;
      ptx_cp_async16_cg(
          &sh.pipe.w[wst][r][c ^ ((r >> 2) & 1)],
          w_src + static_cast<size_t>(r) * row_w_u4 + step * 2 + c);
      if (tid < BN) {
        ptx_cp_async4_ca(&sh.pipe.sc[wst][tid], s_src + static_cast<size_t>(tid) * n_steps + step);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 2; ++iter) {
        const int idx = tid + iter * THREADS;
        const int r = idx >> 2;
        const int c = idx & 3;
        ptx_cp_async16_cg(
            &sh.pipe.w[wst][r][c ^ (r & 3)],
            w_src + static_cast<size_t>(r) * row_w_u4 + step * 4 + c);
      }
      if (tid < BN) {
        ptx_cp_async4_ca(&sh.pipe.sc[wst][tid], s_src + static_cast<size_t>(tid) * n_steps + step);
      }
    }
  };

  auto load_x = [&](int xst, int step) {
    const uint4* z_u4 = reinterpret_cast<const uint4*>(z);
    const int total_x = load_x_rows * 8;
    for (int idx = tid; idx < total_x; idx += THREADS) {
      const int r = idx >> 3;
      const int c = idx & 7;
      const int tok = sh.pipe.tok[r];
      uint4* dst = &sh.pipe.x[xst][r][c ^ ((r & X_SWIZ_MASK) << X_SWIZ_SHIFT)];
      if (tok >= 0) {
        ptx_cp_async16_ca(dst, z_u4 + static_cast<size_t>(tok) * z_row_u4 + step * 8 + c);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  load_w(0, gate_w, gate_s, 0);
  load_x(0, 0);
  ptx_cp_commit();
  ptx_cp_wait<0>();
  __syncthreads();

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;
  float g_tot[N_SLICES_MAX][4];
  float u_tot[N_SLICES_MAX][4];
#pragma unroll
  for (int ms = 0; ms < N_SLICES_MAX; ++ms) {
    g_tot[ms][0] = g_tot[ms][1] = g_tot[ms][2] = g_tot[ms][3] = 0.0f;
    u_tot[ms][0] = u_tot[ms][1] = u_tot[ms][2] = u_tot[ms][3] = 0.0f;
  }

  auto compute_mat = [&](int wst, int xst, float tot[N_SLICES_MAX][4]) {
    if constexpr (MODE == 0) {
      const unsigned int sc_w0 = sh.pipe.sc[wst][r0];
      const unsigned int sc_w1 = sh.pipe.sc[wst][r1];
      const int sw0 = (r0 >> 2) & 1;
      const int sw1 = (r1 >> 2) & 1;
#pragma unroll 1
      for (int lg2 = 0; lg2 < 2; ++lg2) {
        const float2 sf0 = hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w0 >> (lg2 * 16)));
        const float2 sf1 = hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w1 >> (lg2 * 16)));
        const unsigned short* wp0 =
            reinterpret_cast<const unsigned short*>(&sh.pipe.w[wst][r0][lg2 ^ sw0]);
        const unsigned short* wp1 =
            reinterpret_cast<const unsigned short*>(&sh.pipe.w[wst][r1][lg2 ^ sw1]);
#pragma unroll 1
        for (int sub = 0; sub < 2; ++sub) {
          const int lg = lg2 * 2 + sub;
          const unsigned short w0 = wp0[sub * 4 + t];
          const unsigned short w1 = wp1[sub * 4 + t];
#if HY4_HAS_HW_FP4_CVT
          const unsigned int a0 = hy4_fast_e2m1x2_to_bf16x2(w0 & 0xFFu);
          const unsigned int a2 = hy4_fast_e2m1x2_to_bf16x2(w0 >> 8);
          const unsigned int a1 = hy4_fast_e2m1x2_to_bf16x2(w1 & 0xFFu);
          const unsigned int a3 = hy4_fast_e2m1x2_to_bf16x2(w1 >> 8);
#else
          const unsigned int a0 = s_e2m1_lut[w0 & 0xFFu];
          const unsigned int a2 = s_e2m1_lut[w0 >> 8];
          const unsigned int a1 = s_e2m1_lut[w1 & 0xFFu];
          const unsigned int a3 = s_e2m1_lut[w1 >> 8];
#endif
          const float sc0 = (sub == 0) ? sf0.x : sf0.y;
          const float sc1 = (sub == 0) ? sf1.x : sf1.y;
          const int x_col = (lg * 2 + (t >> 1)) ^ ((g & 3) << 1);
          const int x_hf = t & 1;
#pragma unroll
          for (int ms = 0; ms < N_SLICES_MAX; ++ms) {
            if (ms < n_slices) {
              const int rx = ms * 8 + g;
              const uint2 xv = reinterpret_cast<const uint2*>(&sh.pipe.x[xst][rx][x_col])[x_hf];
              float d0, d1, d2, d3;
              ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv.x, xv.y);
              tot[ms][0] = fmaf(sc0, d0, tot[ms][0]);
              tot[ms][1] = fmaf(sc0, d1, tot[ms][1]);
              tot[ms][2] = fmaf(sc1, d2, tot[ms][2]);
              tot[ms][3] = fmaf(sc1, d3, tot[ms][3]);
            }
          }
        }
      }
    } else {
      const unsigned int sc_w0 = sh.pipe.sc[wst][r0];
      const unsigned int sc_w1 = sh.pipe.sc[wst][r1];
#pragma unroll
      for (int lg = 0; lg < 2; ++lg) {
        const int col_u4 = lg * 2 + (t >> 1);
        const int hf = t & 1;
        const uint2 w0 = reinterpret_cast<const uint2*>(
            &sh.pipe.w[wst][r0][col_u4 ^ (r0 & 3)])[hf];
        const uint2 w1 = reinterpret_cast<const uint2*>(
            &sh.pipe.w[wst][r1][col_u4 ^ (r1 & 3)])[hf];
        unsigned int wb0[4], wb1[4];
        hy4p_unpack8_symq8_bf16x2(w0, wb0);
        hy4p_unpack8_symq8_bf16x2(w1, wb1);

        const float sc0 = hy4p_read_f16_or_bf16(
            static_cast<unsigned short>(sc_w0 >> (lg * 16)), sb_is_bf16);
        const float sc1 = hy4p_read_f16_or_bf16(
            static_cast<unsigned short>(sc_w1 >> (lg * 16)), sb_is_bf16);
        const int x_col = (lg * 4 + t) ^ ((g & 1) << 2);
#pragma unroll
        for (int ms = 0; ms < N_SLICES_MAX; ++ms) {
          if (ms < n_slices) {
            const int rx = ms * 8 + g;
            const uint4 xv = sh.pipe.x[xst][rx][x_col];
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv.x, xv.y);
            ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv.z, xv.w);
            tot[ms][0] = fmaf(sc0, d0, tot[ms][0]);
            tot[ms][1] = fmaf(sc0, d1, tot[ms][1]);
            tot[ms][2] = fmaf(sc1, d2, tot[ms][2]);
            tot[ms][3] = fmaf(sc1, d3, tot[ms][3]);
          }
        }
      }
    }
  };

#pragma unroll 1
  for (int step = 0; step < n_steps; ++step) {
    const int cur_x = step & 1;
    const int nxt_x = cur_x ^ 1;
    const bool has_nxt = (step + 1 < n_steps);

    load_w(1, up_w, up_s, step);
    ptx_cp_commit();
    if (has_nxt) {
      load_x(nxt_x, step + 1);
      ptx_cp_commit();
    }

    compute_mat(0, cur_x, g_tot);

    if (has_nxt) ptx_cp_wait<1>();
    else ptx_cp_wait<0>();
    __syncthreads();

    if (has_nxt) {
      load_w(0, gate_w, gate_s, step + 1);
      ptx_cp_commit();
    }

    compute_mat(1, cur_x, u_tot);

    if (has_nxt) {
      ptx_cp_wait<0>();
      __syncthreads();
    }
  }
  __syncthreads();

  const int w_col0 = base_row + warp * 16;
  const int n_passes = (cnt + 15) >> 4;
#pragma unroll
  for (int pass = 0; pass < N_PASSES_MAX; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      sh.s_out[warp][2 * t][g]         = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s0][0], u_tot[s0][0], swiglu_limit));
      sh.s_out[warp][2 * t + 1][g]     = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s0][1], u_tot[s0][1], swiglu_limit));
      sh.s_out[warp][2 * t][g + 8]     = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s0][2], u_tot[s0][2], swiglu_limit));
      sh.s_out[warp][2 * t + 1][g + 8] = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s0][3], u_tot[s0][3], swiglu_limit));
      if (s1 < n_slices) {
        sh.s_out[warp][8 + 2 * t][g]         = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s1][0], u_tot[s1][0], swiglu_limit));
        sh.s_out[warp][8 + 2 * t + 1][g]     = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s1][1], u_tot[s1][1], swiglu_limit));
        sh.s_out[warp][8 + 2 * t][g + 8]     = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s1][2], u_tot[s1][2], swiglu_limit));
        sh.s_out[warp][8 + 2 * t + 1][g + 8] = __float2bfloat16_rn(hy4_clamped_swiglu(g_tot[s1][3], u_tot[s1][3], swiglu_limit));
      }
      __syncwarp();
      const int r_in = lane >> 1;
      const int half = lane & 1;
      const int r_loc = pass * 16 + r_in;
      if (r_loc < cnt) {
        ptx_st_cs_u4(
            reinterpret_cast<uint4*>(mid + static_cast<size_t>(begin + r_loc) * ff + w_col0) + half,
            reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
      }
      __syncwarp();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3. Prefill Down Projection (BM=64 hot / BM=32 cold, BN=128 or 64, BK=64)
//    MODE: 0 = NVFP4 (b4/g16), 1 = SymQ8 (b8/g32, s8 = byte ^ 0x80)
//    2-stage double-buffered cp.async.cg/ca + BF16 TC MMA, 0 spills!
//    Grid: dim3(lat / BN, grid_y, 1), Block: dim3(32, BN / 16, 1).
// ---------------------------------------------------------------------------
template <int MODE, int BM, int BN, bool IS_COLD>
__global__ __launch_bounds__(
    (BN / 16) * 32,
    (BN == 64) ? (BM == 64 ? (MODE == 0 ? 4 : 3) : 5) : (BM == 64 ? 2 : 3))
void hy4_moe_prefill_down(
    const unsigned int* __restrict__ down_w,
    const void* __restrict__ down_s,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    nv_bfloat16* __restrict__ down_out,
    int lat, int ff, int sb_is_bf16) {
  const int tile_idx = IS_COLD ? (num_tiles[1] + static_cast<int>(blockIdx.y))
                               : static_cast<int>(blockIdx.y);
  const int tile_limit = IS_COLD ? num_tiles[0] : num_tiles[1];
  if (tile_idx >= tile_limit) return;

  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;

  constexpr int WARPS = BN / 16;
  constexpr int THREADS = WARPS * 32;
  constexpr int N_SLICES_MAX = BM / 8;
  constexpr int N_PASSES_MAX = BM / 16;
  constexpr int W_U4 = (MODE == 0) ? 2 : 4;
  constexpr int X_SWIZ_SHIFT = (MODE == 0) ? 1 : 2;
  constexpr int X_SWIZ_MASK = (MODE == 0) ? 3 : 1;

  const int base_row = blockIdx.x * BN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  struct StageDn {
    alignas(16) uint4 x[BM][8];
    alignas(16) uint4 w[BN][W_U4];
    alignas(16) unsigned int sc[BN];
  };
  union SharedDn {
    StageDn st[2];
    alignas(16) nv_bfloat16 s_out[WARPS][16][16];
  };
  __shared__ SharedDn sh;
#if !HY4_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[MODE == 0 ? 256 : 1];
  if constexpr (MODE == 0) {
    for (int i = tid; i < 256; i += THREADS) {
      s_e2m1_lut[i] = hy4_e2m1x2_to_bf16x2(static_cast<unsigned int>(i));
    }
  } else {
    (void)s_e2m1_lut;
  }
#endif

  const int n_steps = ff >> 6;
  const int row_w_u4 = (MODE == 0) ? (ff >> 5) : (ff >> 4);
  const int mid_row_u4 = ff >> 3;
  const size_t exp_row0 = static_cast<size_t>(expert) * lat + base_row;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(down_w) + exp_row0 * row_w_u4;
  const unsigned int* s_u32 = reinterpret_cast<const unsigned int*>(down_s) + exp_row0 * n_steps;
  const uint4* mid_u4 = reinterpret_cast<const uint4*>(mid);

  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_stage = [&](int s_idx, int step) {
    if constexpr (MODE == 0) {
      const int r = tid >> 1;
      const int c = tid & 1;
      ptx_cp_async16_cg(
          &sh.st[s_idx].w[r][c ^ ((r >> 2) & 1)],
          w_u4 + static_cast<size_t>(r) * row_w_u4 + step * 2 + c);
      if (tid < BN) {
        ptx_cp_async4_ca(&sh.st[s_idx].sc[tid], s_u32 + static_cast<size_t>(tid) * n_steps + step);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 2; ++iter) {
        const int idx = tid + iter * THREADS;
        const int r = idx >> 2;
        const int c = idx & 3;
        ptx_cp_async16_cg(
            &sh.st[s_idx].w[r][c ^ (r & 3)],
            w_u4 + static_cast<size_t>(r) * row_w_u4 + step * 4 + c);
      }
      if (tid < BN) {
        ptx_cp_async4_ca(&sh.st[s_idx].sc[tid], s_u32 + static_cast<size_t>(tid) * n_steps + step);
      }
    }
    const int total_x = load_x_rows * 8;
    for (int idx = tid; idx < total_x; idx += THREADS) {
      const int r = idx >> 3;
      const int c = idx & 7;
      uint4* dst = &sh.st[s_idx].x[r][c ^ ((r & X_SWIZ_MASK) << X_SWIZ_SHIFT)];
      if (r < cnt) {
        ptx_cp_async16_ca(dst, mid_u4 + static_cast<size_t>(begin + r) * mid_row_u4 + step * 8 + c);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    ptx_cp_commit();
  };

  load_stage(0, 0);

  float tot[N_SLICES_MAX][4];
#pragma unroll
  for (int ms = 0; ms < N_SLICES_MAX; ++ms) {
    tot[ms][0] = tot[ms][1] = tot[ms][2] = tot[ms][3] = 0.0f;
  }
  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;

#pragma unroll 1
  for (int step = 0; step < n_steps; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < n_steps) {
      load_stage(nxt, step + 1);
      ptx_cp_wait<1>();
    } else {
      ptx_cp_wait<0>();
    }
    __syncthreads();

    if constexpr (MODE == 0) {
      const unsigned int sc_w0 = sh.st[cur].sc[r0];
      const unsigned int sc_w1 = sh.st[cur].sc[r1];
      const int sw0 = (r0 >> 2) & 1;
      const int sw1 = (r1 >> 2) & 1;
#pragma unroll 1
      for (int lg2 = 0; lg2 < 2; ++lg2) {
        const float2 sf0 = hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w0 >> (lg2 * 16)));
        const float2 sf1 = hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(sc_w1 >> (lg2 * 16)));
        const unsigned short* wp0 = reinterpret_cast<const unsigned short*>(&sh.st[cur].w[r0][lg2 ^ sw0]);
        const unsigned short* wp1 = reinterpret_cast<const unsigned short*>(&sh.st[cur].w[r1][lg2 ^ sw1]);
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int lg = lg2 * 2 + sub;
          const unsigned short w0 = wp0[sub * 4 + t];
          const unsigned short w1 = wp1[sub * 4 + t];
#if HY4_HAS_HW_FP4_CVT
          const unsigned int a0 = hy4_fast_e2m1x2_to_bf16x2(w0 & 0xFFu);
          const unsigned int a2 = hy4_fast_e2m1x2_to_bf16x2(w0 >> 8);
          const unsigned int a1 = hy4_fast_e2m1x2_to_bf16x2(w1 & 0xFFu);
          const unsigned int a3 = hy4_fast_e2m1x2_to_bf16x2(w1 >> 8);
#else
          const unsigned int a0 = s_e2m1_lut[w0 & 0xFFu];
          const unsigned int a2 = s_e2m1_lut[w0 >> 8];
          const unsigned int a1 = s_e2m1_lut[w1 & 0xFFu];
          const unsigned int a3 = s_e2m1_lut[w1 >> 8];
#endif
          const float sc0 = (sub == 0) ? sf0.x : sf0.y;
          const float sc1 = (sub == 0) ? sf1.x : sf1.y;
          const int x_col = (lg * 2 + (t >> 1)) ^ ((g & 3) << 1);
          const int x_hf = t & 1;
#pragma unroll
          for (int ms = 0; ms < N_SLICES_MAX; ++ms) {
            if (ms < n_slices) {
              const int rx = ms * 8 + g;
              const uint2 xv = reinterpret_cast<const uint2*>(&sh.st[cur].x[rx][x_col])[x_hf];
              float d0, d1, d2, d3;
              ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv.x, xv.y);
              tot[ms][0] = fmaf(sc0, d0, tot[ms][0]);
              tot[ms][1] = fmaf(sc0, d1, tot[ms][1]);
              tot[ms][2] = fmaf(sc1, d2, tot[ms][2]);
              tot[ms][3] = fmaf(sc1, d3, tot[ms][3]);
            }
          }
        }
      }
    } else {
      const unsigned int sc_w0 = sh.st[cur].sc[r0];
      const unsigned int sc_w1 = sh.st[cur].sc[r1];
#pragma unroll
      for (int lg = 0; lg < 2; ++lg) {
        const int col_u4 = lg * 2 + (t >> 1);
        const int hf = t & 1;
        const uint2 w0 = reinterpret_cast<const uint2*>(
            &sh.st[cur].w[r0][col_u4 ^ (r0 & 3)])[hf];
        const uint2 w1 = reinterpret_cast<const uint2*>(
            &sh.st[cur].w[r1][col_u4 ^ (r1 & 3)])[hf];
        unsigned int wb0[4], wb1[4];
        hy4p_unpack8_symq8_bf16x2(w0, wb0);
        hy4p_unpack8_symq8_bf16x2(w1, wb1);

        const float sc0 = hy4p_read_f16_or_bf16(
            static_cast<unsigned short>(sc_w0 >> (lg * 16)), sb_is_bf16);
        const float sc1 = hy4p_read_f16_or_bf16(
            static_cast<unsigned short>(sc_w1 >> (lg * 16)), sb_is_bf16);
        const int x_col = (lg * 4 + t) ^ ((g & 1) << 2);
#pragma unroll
        for (int ms = 0; ms < N_SLICES_MAX; ++ms) {
          if (ms < n_slices) {
            const int rx = ms * 8 + g;
            const uint4 xv = sh.st[cur].x[rx][x_col];
            float d0, d1, d2, d3;
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv.x, xv.y);
            ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv.z, xv.w);
            tot[ms][0] = fmaf(sc0, d0, tot[ms][0]);
            tot[ms][1] = fmaf(sc0, d1, tot[ms][1]);
            tot[ms][2] = fmaf(sc1, d2, tot[ms][2]);
            tot[ms][3] = fmaf(sc1, d3, tot[ms][3]);
          }
        }
      }
    }
    __syncthreads();
  }

  const int w_col0 = base_row + warp * 16;
  const int n_passes = (cnt + 15) >> 4;
#pragma unroll
  for (int pass = 0; pass < N_PASSES_MAX; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      sh.s_out[warp][2 * t][g]         = __float2bfloat16_rn(tot[s0][0]);
      sh.s_out[warp][2 * t + 1][g]     = __float2bfloat16_rn(tot[s0][1]);
      sh.s_out[warp][2 * t][g + 8]     = __float2bfloat16_rn(tot[s0][2]);
      sh.s_out[warp][2 * t + 1][g + 8] = __float2bfloat16_rn(tot[s0][3]);
      if (s1 < n_slices) {
        sh.s_out[warp][8 + 2 * t][g]         = __float2bfloat16_rn(tot[s1][0]);
        sh.s_out[warp][8 + 2 * t + 1][g]     = __float2bfloat16_rn(tot[s1][1]);
        sh.s_out[warp][8 + 2 * t][g + 8]     = __float2bfloat16_rn(tot[s1][2]);
        sh.s_out[warp][8 + 2 * t + 1][g + 8] = __float2bfloat16_rn(tot[s1][3]);
      }
      __syncwarp();
      const int r_in = lane >> 1;
      const int half = lane & 1;
      const int r_loc = pass * 16 + r_in;
      if (r_loc < cnt) {
        ptx_st_cs_u4(
            reinterpret_cast<uint4*>(down_out + static_cast<size_t>(begin + r_loc) * lat + w_col0) + half,
            reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
      }
      __syncwarp();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 4. Prefill Deterministic Ascending-k Vectorized Weighted Combine
//    Grid: dim3(ceil((lat/8) / blockDim.x), M, 1), Block: dim3(block_x, 1, 1).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2) void hy4_moe_prefill_combine(
    const nv_bfloat16* __restrict__ down_out,
    const float* __restrict__ topk_weights,
    const int* __restrict__ pair_to_sorted,
    nv_bfloat16* __restrict__ out,
    int M, int lat, int top_k) {
  const int tok = blockIdx.y;
  if (tok >= M) return;
  const int tid = threadIdx.x;
  const int lat8 = lat >> 3;
  const int col8 = blockIdx.x * blockDim.x + tid;

  __shared__ int s_slot[16];
  __shared__ float s_w[16];
  if (tid < top_k) {
    const int p = tok * top_k + tid;
    s_slot[tid] = pair_to_sorted[p];
    s_w[tid] = topk_weights[p];
  }
  __syncthreads();

  if (col8 >= lat8) return;

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint4* dn_u4 = reinterpret_cast<const uint4*>(down_out);
  for (int k = 0; k < top_k; ++k) {
    const float w = s_w[k];
    const uint4 raw = dn_u4[static_cast<size_t>(s_slot[k]) * lat8 + col8];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      acc[2 * j] = fmaf(f.x, w, acc[2 * j]);
      acc[2 * j + 1] = fmaf(f.y, w, acc[2 * j + 1]);
    }
  }
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    op[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  }
  reinterpret_cast<uint4*>(out)[static_cast<size_t>(tok) * lat8 + col8] = ov;
}
)cuda"
