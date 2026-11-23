R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ void sym8_l_cp16(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void sym8_l_cp16_ca(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void sym8_l_cp16_l2p(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void sym8_l_st_cs_u4(uint4* glob_ptr, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
               :: "l"(glob_ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void sym8_l_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void sym8_l_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void sym8_l_ldm4(
    unsigned int& d0, unsigned int& d1, unsigned int& d2, unsigned int& d3,
    const void* smem_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}

__device__ __forceinline__ void sym8_l_mma(
    int& d0, int& d1, int& d2, int& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int BK_U4>
__device__ __forceinline__ int sym8_l_swz(int r, int c) {
  if constexpr (BK_U4 == 4) {
    return c ^ ((r >> 1) & 3);
  } else {
    return c ^ (r & 7);
  }
}

template <typename Stage, int BM, int BN, int BK_U4, int THREADS,
          bool OTF_ADDR, bool CA_A, bool L2P_B>
struct Sym8LargeLoader;

template <typename Stage, int BM, int BN, int BK_U4, int THREADS, bool CA_A, bool L2P_B>
struct Sym8LargeLoader<Stage, BM, BN, BK_U4, THREADS, false, CA_A, L2P_B> {
  static constexpr int A_ITERS = (BM * BK_U4) / THREADS;
  static constexpr int B_ITERS = (BN * BK_U4) / THREADS;
  const uint4* a_gptr[A_ITERS];
  bool a_valid[A_ITERS];
  int a_sr[A_ITERS], a_sc[A_ITERS];
  const uint4* b_gptr[B_ITERS];
  bool b_valid[B_ITERS];
  int b_sr[B_ITERS], b_sc[B_ITERS];

  __device__ __forceinline__ Sym8LargeLoader(
      const uint4* A_u4, const uint4* B_u4, size_t stride_u4,
      int m0, int n0, int M, int N, int tid) {
#pragma unroll
    for (int it = 0; it < A_ITERS; ++it) {
      const int idx = tid + it * THREADS;
      const int r = idx / BK_U4;
      const int c = idx & (BK_U4 - 1);
      const int gm = m0 + r;
      a_valid[it] = (gm < M);
      a_gptr[it] = A_u4 + static_cast<size_t>(gm) * stride_u4 + c;
      a_sr[it] = r;
      a_sc[it] = sym8_l_swz<BK_U4>(r, c);
    }
#pragma unroll
    for (int it = 0; it < B_ITERS; ++it) {
      const int idx = tid + it * THREADS;
      const int r = idx / BK_U4;
      const int c = idx & (BK_U4 - 1);
      const int gn = n0 + r;
      b_valid[it] = (gn < N);
      b_gptr[it] = B_u4 + static_cast<size_t>(gn) * stride_u4 + c;
      b_sr[it] = r;
      b_sc[it] = sym8_l_swz<BK_U4>(r, c);
    }
  }

  __device__ __forceinline__ void load(Stage* smem, int st, int step) const {
    const size_t k_off = static_cast<size_t>(step) * BK_U4;
#pragma unroll
    for (int it = 0; it < A_ITERS; ++it) {
      if (a_valid[it]) {
        if constexpr (CA_A) sym8_l_cp16_ca(&smem[st].a[a_sr[it]][a_sc[it]], a_gptr[it] + k_off);
        else sym8_l_cp16(&smem[st].a[a_sr[it]][a_sc[it]], a_gptr[it] + k_off);
      }
    }
#pragma unroll
    for (int it = 0; it < B_ITERS; ++it) {
      if (b_valid[it]) {
        if constexpr (L2P_B) sym8_l_cp16_l2p(&smem[st].b[b_sr[it]][b_sc[it]], b_gptr[it] + k_off);
        else sym8_l_cp16(&smem[st].b[b_sr[it]][b_sc[it]], b_gptr[it] + k_off);
      }
    }
    sym8_l_cp_commit();
  }
};

template <typename Stage, int BM, int BN, int BK_U4, int THREADS, bool CA_A, bool L2P_B>
struct Sym8LargeLoader<Stage, BM, BN, BK_U4, THREADS, true, CA_A, L2P_B> {
  static constexpr int A_ITERS = (BM * BK_U4) / THREADS;
  static constexpr int B_ITERS = (BN * BK_U4) / THREADS;
  static constexpr int ROWS_STEP = THREADS / BK_U4;
  const uint4* A_u4;
  const uint4* B_u4;
  size_t stride_u4;
  int m0, n0, M, N, r_base, c_base;

  __device__ __forceinline__ Sym8LargeLoader(
      const uint4* A_u4_, const uint4* B_u4_, size_t stride_u4_,
      int m0_, int n0_, int M_, int N_, int tid)
      : A_u4(A_u4_), B_u4(B_u4_), stride_u4(stride_u4_),
        m0(m0_), n0(n0_), M(M_), N(N_),
        r_base(tid / BK_U4), c_base(tid & (BK_U4 - 1)) {}

  __device__ __forceinline__ void load(Stage* smem, int st, int step) const {
    const size_t k_off = static_cast<size_t>(step) * BK_U4 + c_base;
#pragma unroll
    for (int it = 0; it < B_ITERS; ++it) {
      const int r = r_base + it * ROWS_STEP;
      const int gn = n0 + r;
      if (gn < N) {
        void* dst = &smem[st].b[r][sym8_l_swz<BK_U4>(r, c_base)];
        const void* src = B_u4 + static_cast<size_t>(gn) * stride_u4 + k_off;
        if constexpr (L2P_B) sym8_l_cp16_l2p(dst, src);
        else sym8_l_cp16(dst, src);
      }
    }
#pragma unroll
    for (int it = 0; it < A_ITERS; ++it) {
      const int r = r_base + it * ROWS_STEP;
      const int gm = m0 + r;
      if (gm < M) {
        void* dst = &smem[st].a[r][sym8_l_swz<BK_U4>(r, c_base)];
        const void* src = A_u4 + static_cast<size_t>(gm) * stride_u4 + k_off;
        if constexpr (CA_A) sym8_l_cp16_ca(dst, src);
        else sym8_l_cp16(dst, src);
      }
    }
    sym8_l_cp_commit();
  }
};
)cuda"
R"cuda(
template <int BM, int BN, int BK, int STAGES, int WM_WARPS, int WN_WARPS,
          bool USE_DYN_SMEM, int GROUP_M, bool SWIZZLE_N,
          bool CA_A, bool L2P_B, bool CS_Y, bool OTF_ADDR, bool PREF_A, bool WARP_SKIP>
__device__ __forceinline__ void sym8_imma_large_core(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ tok_scale,
    const float* __restrict__ row_scale,
    nv_bfloat16* __restrict__ Y,
    int M, int N, int K) {
  constexpr int THREADS = WM_WARPS * WN_WARPS * 32;
  constexpr int BK_U4 = BK / 16;
  constexpr int K_SUB = BK / 32;
  constexpr int WARP_M = BM / WM_WARPS;
  constexpr int WARP_N = BN / WN_WARPS;
  constexpr int MT = WARP_M / 16;
  constexpr int NT = WARP_N / 8;

  const int num_m_tiles = (M + BM - 1) / BM;
  const int num_n_tiles = (N + BN - 1) / BN;
  const int lin = static_cast<int>(blockIdx.x);
  int m_tile, n_tile;
  if constexpr (!SWIZZLE_N) {
    const int tiles_per_band = GROUP_M * num_n_tiles;
    const int band = lin / tiles_per_band;
    const int in_band = lin - band * tiles_per_band;
    const int band_m0 = band * GROUP_M;
    const int band_rows = min(GROUP_M, num_m_tiles - band_m0);
    n_tile = in_band / band_rows;
    m_tile = band_m0 + (in_band - n_tile * band_rows);
  } else {
    const int tiles_per_band = GROUP_M * num_m_tiles;
    const int band = lin / tiles_per_band;
    const int in_band = lin - band * tiles_per_band;
    const int band_n0 = band * GROUP_M;
    const int band_cols = min(GROUP_M, num_n_tiles - band_n0);
    m_tile = in_band / band_cols;
    n_tile = band_n0 + (in_band - m_tile * band_cols);
  }
  if (n_tile >= num_n_tiles || m_tile >= num_m_tiles) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WN_WARPS;
  const int wn = warp - wm * WN_WARPS;

  const int m0 = m_tile * BM;
  const int n0 = n_tile * BN;
  const int k_steps = K / BK;
  const size_t stride_u4 = static_cast<size_t>(K) >> 4;
  const bool warp_active = !WARP_SKIP || ((m0 + wm * WARP_M) < M);

  struct Stage {
    alignas(16) uint4 a[BM][BK_U4];
    alignas(16) uint4 b[BN][BK_U4];
  };
  Stage* smem = nullptr;
  if constexpr (USE_DYN_SMEM) {
    extern __shared__ unsigned char dyn_smem[];
    smem = reinterpret_cast<Stage*>(dyn_smem);
  } else {
    __shared__ Stage static_smem[STAGES];
    smem = static_smem;
  }

  Sym8LargeLoader<Stage, BM, BN, BK_U4, THREADS, OTF_ADDR, CA_A, L2P_B> loader(
      reinterpret_cast<const uint4*>(A), reinterpret_cast<const uint4*>(B),
      stride_u4, m0, n0, M, N, tid);

  int acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
      acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_steps) loader.load(smem, s, s);
  }

  const int ld_a_r0 = wm * WARP_M + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int ld_a_c = lane >> 4;
  const int ld_b_r0 = wn * WARP_N + (lane & 7) + ((lane >> 4) * 8);
  const int ld_b_c = (lane >> 3) & 1;

#pragma unroll 1
  for (int step = 0; step < k_steps; ++step) {
    const int cur = step % STAGES;
    const int nxt = step + STAGES - 1;
    if (nxt < k_steps) {
      sym8_l_cp_wait<STAGES - 2>();
    } else {
      sym8_l_cp_wait<0>();
    }
    __syncthreads();
    if (nxt < k_steps) {
      loader.load(smem, nxt % STAGES, nxt);
    }

    if (warp_active) {
#pragma unroll
      for (int ks = 0; ks < K_SUB; ++ks) {
        unsigned int bf[NT][2];
#pragma unroll
        for (int jp = 0; jp < NT / 2; ++jp) {
          const int br = ld_b_r0 + jp * 16;
          const int bc = sym8_l_swz<BK_U4>(br, ks * 2 + ld_b_c);
          sym8_l_ldm4(bf[2 * jp][0], bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1],
                      &smem[cur].b[br][bc]);
        }
        if constexpr (PREF_A) {
          unsigned int af[2][4];
          {
            const int ar = ld_a_r0;
            const int ac = sym8_l_swz<BK_U4>(ar, ks * 2 + ld_a_c);
            sym8_l_ldm4(af[0][0], af[0][1], af[0][2], af[0][3], &smem[cur].a[ar][ac]);
          }
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            if (i + 1 < MT) {
              const int ar = ld_a_r0 + (i + 1) * 16;
              const int ac = sym8_l_swz<BK_U4>(ar, ks * 2 + ld_a_c);
              sym8_l_ldm4(af[(i + 1) & 1][0], af[(i + 1) & 1][1], af[(i + 1) & 1][2], af[(i + 1) & 1][3],
                          &smem[cur].a[ar][ac]);
            }
            const int cur_a = i & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
              sym8_l_mma(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                         af[cur_a][0], af[cur_a][1], af[cur_a][2], af[cur_a][3],
                         bf[j][0], bf[j][1]);
            }
          }
        } else {
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            unsigned int a0, a1, a2, a3;
            const int ar = ld_a_r0 + i * 16;
            const int ac = sym8_l_swz<BK_U4>(ar, ks * 2 + ld_a_c);
            sym8_l_ldm4(a0, a1, a2, a3, &smem[cur].a[ar][ac]);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
              sym8_l_mma(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                         a0, a1, a2, a3, bf[j][0], bf[j][1]);
            }
          }
        }
      }
    }
  }
  sym8_l_cp_wait<0>();

  const int g = lane >> 2;
  const int t = lane & 3;
  float rs0[NT], rs1[NT];
  if (warp_active) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = n0 + wn * WARP_N + j * 8 + 2 * t;
      if (col + 1 < N) {
        float2 rs = *reinterpret_cast<const float2*>(row_scale + col);
        rs0[j] = rs.x;
        rs1[j] = rs.y;
      } else {
        rs0[j] = (col < N) ? row_scale[col] : 0.0f;
        rs1[j] = 0.0f;
      }
    }
  }

  __syncthreads();
  nv_bfloat16* s_out = reinterpret_cast<nv_bfloat16*>(smem);
  constexpr int S_STRIDE = BN + 8;
  if (warp_active) {
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int rl0 = wm * WARP_M + i * 16 + g;
      const int rl1 = rl0 + 8;
      const int r0 = m0 + rl0;
      const int r1 = m0 + rl1;
      const float s0 = (r0 < M) ? tok_scale[r0] : 0.0f;
      const float s1 = (r1 < M) ? tok_scale[r1] : 0.0f;
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int cl = wn * WARP_N + j * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(s_out + rl0 * S_STRIDE + cl) =
            __floats2bfloat162_rn(static_cast<float>(acc[i][j][0]) * s0 * rs0[j],
                                  static_cast<float>(acc[i][j][1]) * s0 * rs1[j]);
        *reinterpret_cast<__nv_bfloat162*>(s_out + rl1 * S_STRIDE + cl) =
            __floats2bfloat162_rn(static_cast<float>(acc[i][j][2]) * s1 * rs0[j],
                                  static_cast<float>(acc[i][j][3]) * s1 * rs1[j]);
      }
    }
  }
  __syncthreads();
  constexpr int BN_U4 = BN / 8;
  constexpr int OUT_ITERS = (BM * BN_U4) / THREADS;
  const size_t y_stride_u4 = static_cast<size_t>(N) >> 3;
  uint4* Y_u4 = reinterpret_cast<uint4*>(Y);
#pragma unroll
  for (int it = 0; it < OUT_ITERS; ++it) {
    const int idx = tid + it * THREADS;
    const int r = idx / BN_U4;
    const int c = idx & (BN_U4 - 1);
    const int gm = m0 + r;
    const int gn8 = (n0 >> 3) + c;
    if (gm < M && (gn8 << 3) < N) {
      const uint4 v = *reinterpret_cast<const uint4*>(s_out + r * S_STRIDE + c * 8);
      if constexpr (CS_Y) sym8_l_st_cs_u4(Y_u4 + static_cast<size_t>(gm) * y_stride_u4 + gn8, v);
      else Y_u4[static_cast<size_t>(gm) * y_stride_u4 + gn8] = v;
    }
  }
}

__global__ __launch_bounds__(256, 2) void sym8_imma_128x128_k64_s3(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ tok_scale,
    const float* __restrict__ row_scale,
    nv_bfloat16* __restrict__ Y,
    int M, int N, int K) {
  sym8_imma_large_core<128, 128, 64, 3, 2, 4, false, 16, false,
                       false, false, false, false, true, false>(
      A, B, tok_scale, row_scale, Y, M, N, K);
}

__global__ __launch_bounds__(256, 1) void sym8_imma_128x128_k128_s3(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ tok_scale,
    const float* __restrict__ row_scale,
    nv_bfloat16* __restrict__ Y,
    int M, int N, int K) {
  sym8_imma_large_core<128, 128, 128, 3, 2, 4, true, 8, true,
                       false, true, true, true, false, true>(
      A, B, tok_scale, row_scale, Y, M, N, K);
}

__global__ __launch_bounds__(256, 1) void sym8_imma_128x128_k128_s3_gm8(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ tok_scale,
    const float* __restrict__ row_scale,
    nv_bfloat16* __restrict__ Y,
    int M, int N, int K) {
  sym8_imma_large_core<128, 128, 128, 3, 2, 4, true, 8, false,
                       true, true, true, true, false, true>(
      A, B, tok_scale, row_scale, Y, M, N, K);
}
)cuda"