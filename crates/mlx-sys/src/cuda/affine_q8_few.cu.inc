R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// High-bandwidth BF16 -> BF16 small-row (M in 1..=16) kernels for affine 8-bit
// group_size=32 quantization (GGUF Q8_0) with FP32, BF16, or FP16 scales/biases,
// plus bias-free symmetric Q8_0 (`SYM=true`, `q_s8 = int8(w_byte ^ 0x80)`).
//
// Two specialized paths cover the full M in 1..=16 range near the DRAM ceiling:
// 1. M in 1..=4 (and 5..=8): `mlx_node_affine_q8_u4_1row<SB_DTYPE, M_CAP, NUM_WARPS, SYM>`
//    Single-launch, 1 row/CTA, 512B/warp coalesced uint4 cache-streaming loads.
// 2. M in 9..=16: `mlx_node_affine_q8_mma<SB_DTYPE, M_TILES, SPLIT_K, SYM>`
//    Coalesced 256B/row SMEM-staged BF16 Tensor Core MMA (m16n8k16) with
//    XOR-swizzled bank-conflict-free SMEM and deterministic Split-K (+ reduce
//    kernel `mlx_node_affine_q8_g32_reduce` when split_k > 1) to saturate all
//    SMs even at narrow N (e.g. N=96).

__device__ __forceinline__ void aq8_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 aq8_ld_cs_u4(const uint4* ptr) {
  uint4 v;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

template <int SB_DTYPE>
__device__ __forceinline__ float aq8_load_sb1(const void* ptr, size_t idx) {
  if constexpr (SB_DTYPE == 0) {
    return reinterpret_cast<const float*>(ptr)[idx];
  } else if constexpr (SB_DTYPE == 1) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  } else {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
  }
}

// Convert two 8-bit integers (0..255 when !SYM, or signed -128..127 from
// XOR-0x80 packed bytes when SYM; both exact in BF16) from `w` to bf16x2.
template <int PAIR, bool SYM>
__device__ __forceinline__ unsigned int aq8_unpack_pair(unsigned int w) {
  constexpr int shift = PAIR * 16;
  float f0, f1;
  if constexpr (SYM) {
    const unsigned int sw = w ^ 0x80808080u;
    f0 = static_cast<float>(static_cast<signed char>((sw >> shift) & 0xFFu));
    f1 = static_cast<float>(static_cast<signed char>((sw >> (shift + 8)) & 0xFFu));
  } else {
    f0 = static_cast<float>((w >> shift) & 0xFFu);
    f1 = static_cast<float>((w >> (shift + 8)) & 0xFFu);
  }
  const __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float aq8_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

template <bool SYM>
__device__ __forceinline__ void aq8_load_x_g32(
    const uint4* __restrict__ x_row_u4, size_t grp_u4, bool valid_m,
    int t, uint4& xv, float& xsum_c0, float& xsum_c1) {
  if (valid_m) {
    xv = x_row_u4[grp_u4];
  } else {
    xv = make_uint4(0, 0, 0, 0);
  }
  if constexpr (!SYM) {
    float my_sum = valid_m ? aq8_sum_bf16x8(xv) : 0.0f;
    my_sum += __shfl_xor_sync(0xffffffffu, my_sum, 1);
    my_sum += __shfl_xor_sync(0xffffffffu, my_sum, 2);
    xsum_c0 = __shfl_sync(0xffffffffu, my_sum, (2 * t) * 4);
    xsum_c1 = __shfl_sync(0xffffffffu, my_sum, (2 * t + 1) * 4);
  } else {
    (void)t;
    xsum_c0 = 0.0f;
    xsum_c1 = 0.0f;
  }
}

template <int M_TILES, bool SYM>
__device__ __forceinline__ void aq8_mma_g32(
    unsigned int r0_u0, unsigned int r0_u1,
    unsigned int r1_u0, unsigned int r1_u1,
    const uint4& x0, const uint4& x1,
    float d0[4], float d1[4]) {
  d0[0] = d0[1] = d0[2] = d0[3] = 0.0f;
  if constexpr (M_TILES == 2) {
    d1[0] = d1[1] = d1[2] = d1[3] = 0.0f;
  }
  {
    const unsigned int a0 = aq8_unpack_pair<0, SYM>(r0_u0);
    const unsigned int a1 = aq8_unpack_pair<0, SYM>(r1_u0);
    const unsigned int a2 = aq8_unpack_pair<1, SYM>(r0_u0);
    const unsigned int a3 = aq8_unpack_pair<1, SYM>(r1_u0);
    aq8_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.x, x0.y);
    if constexpr (M_TILES == 2) {
      aq8_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.x, x1.y);
    }
  }
  {
    const unsigned int a0 = aq8_unpack_pair<0, SYM>(r0_u1);
    const unsigned int a1 = aq8_unpack_pair<0, SYM>(r1_u1);
    const unsigned int a2 = aq8_unpack_pair<1, SYM>(r0_u1);
    const unsigned int a3 = aq8_unpack_pair<1, SYM>(r1_u1);
    aq8_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.z, x0.w);
    if constexpr (M_TILES == 2) {
      aq8_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.z, x1.w);
    }
  }
}

template <bool SYM>
__device__ __forceinline__ void aq8_dot4(
    unsigned int uu, __nv_bfloat162 a, __nv_bfloat162 b, float& d, float& s) {
  const float2 f0 = __bfloat1622float2(a);
  const float2 f1 = __bfloat1622float2(b);
  if constexpr (SYM) {
    const unsigned int sw = uu ^ 0x80808080u;
    d = fmaf(f0.x, static_cast<float>(static_cast<signed char>(sw & 0xFFu)),
             fmaf(f0.y, static_cast<float>(static_cast<signed char>((sw >> 8) & 0xFFu)), d));
    d = fmaf(f1.x, static_cast<float>(static_cast<signed char>((sw >> 16) & 0xFFu)),
             fmaf(f1.y, static_cast<float>(static_cast<signed char>(sw >> 24)), d));
  } else {
    d = fmaf(f0.x, static_cast<float>(uu & 0xFFu),
             fmaf(f0.y, static_cast<float>((uu >> 8) & 0xFFu), d));
    s += f0.x + f0.y;
    d = fmaf(f1.x, static_cast<float>((uu >> 16) & 0xFFu),
             fmaf(f1.y, static_cast<float>(uu >> 24), d));
    s += f1.x + f1.y;
  }
}

// Fast single-launch uint4 cache-streaming kernel for M in 1..=8 (and optional exact M=9..=16).
// Grid: dim3(N), Block: dim3(32, NUM_WARPS).
template <int SB_DTYPE, int M_CAP, int NUM_WARPS, bool SYM>
__global__ __launch_bounds__(NUM_WARPS * 32, 4) void mlx_node_affine_q8_u4_1row(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int num_u4 = K >> 4;
  const int groups = K >> 5;

  const uint4* w_row =
      reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
  const size_t sb_row = static_cast<size_t>(n) * groups;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  for (int u = warp * 32 + lane; u < num_u4; u += NUM_WARPS * 32) {
    const uint4 wv = aq8_ld_cs_u4(w_row + u);
    const int g = u >> 1;
    const float sc = aq8_load_sb1<SB_DTYPE>(scales, sb_row + g);
    [[maybe_unused]] float bi = 0.0f;
    if constexpr (!SYM) {
      bi = aq8_load_sb1<SB_DTYPE>(biases, sb_row + g);
    }

    const size_t x_off = static_cast<size_t>(u) * 2;
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < M) {
        const uint4* x_ptr = reinterpret_cast<const uint4*>(x) +
            static_cast<size_t>(m) * x_stride_u4 + x_off;
        const uint4 x0 = x_ptr[0];
        const uint4 x1 = x_ptr[1];
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
        const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
        float d = 0.0f;
        float s = 0.0f;
        aq8_dot4<SYM>(wv.x, p0[0], p0[1], d, s);
        aq8_dot4<SYM>(wv.y, p0[2], p0[3], d, s);
        aq8_dot4<SYM>(wv.z, p1[0], p1[1], d, s);
        aq8_dot4<SYM>(wv.w, p1[2], p1[3], d, s);

        if constexpr (SYM) {
          acc[m] = fmaf(sc, d, acc[m]);
        } else {
          acc[m] = fmaf(sc, d, fmaf(bi, s, acc[m]));
        }
      }
    }
  }

  __shared__ float smem[NUM_WARPS][M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      float v = acc[m];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
      }
      if (lane == 0) {
        smem[warp][m] = v;
      }
    }
  }
  __syncthreads();
  if (warp == 0 && lane < M) {
    float sum = 0.0f;
#pragma unroll
    for (int ww = 0; ww < NUM_WARPS; ++ww) {
      sum += smem[ww][lane];
    }
    out[static_cast<size_t>(lane) * N + n] = __float2bfloat16(sum);
  }
}

// Single-launch 1-warp-per-row uint4 streaming kernel for M in 1..=2 at medium N
// (e.g. LP_K/V [1024, 2304]): zero SMEM, zero __syncthreads(), 90% warp lane utilization.
// Grid: dim3((N + NUM_WARPS - 1) / NUM_WARPS), Block: dim3(32, NUM_WARPS).
template <int SB_DTYPE, int M_CAP, int NUM_WARPS, bool SYM>
__device__ __forceinline__ void aq8_u4_1warp_impl(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int n = static_cast<int>(blockIdx.x) * NUM_WARPS + warp;
  if (n >= N) {
    return;
  }
  const int num_u4 = K >> 4;
  const int groups = K >> 5;

  const uint4* w_row =
      reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
  const size_t sb_row = static_cast<size_t>(n) * groups;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  for (int u = lane; u < num_u4; u += 32) {
    const uint4 wv = aq8_ld_cs_u4(w_row + u);
    const int g = u >> 1;
    const float sc = aq8_load_sb1<SB_DTYPE>(scales, sb_row + g);
    [[maybe_unused]] float bi = 0.0f;
    if constexpr (!SYM) {
      bi = aq8_load_sb1<SB_DTYPE>(biases, sb_row + g);
    }

    const size_t x_off = static_cast<size_t>(u) * 2;
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < M) {
        const uint4* x_ptr = reinterpret_cast<const uint4*>(x) +
            static_cast<size_t>(m) * x_stride_u4 + x_off;
        const uint4 x0 = x_ptr[0];
        const uint4 x1 = x_ptr[1];
        const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
        const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
        float d = 0.0f;
        float s = 0.0f;
        aq8_dot4<SYM>(wv.x, p0[0], p0[1], d, s);
        aq8_dot4<SYM>(wv.y, p0[2], p0[3], d, s);
        aq8_dot4<SYM>(wv.z, p1[0], p1[1], d, s);
        aq8_dot4<SYM>(wv.w, p1[2], p1[3], d, s);

        if constexpr (SYM) {
          acc[m] = fmaf(sc, d, acc[m]);
        } else {
          acc[m] = fmaf(sc, d, fmaf(bi, s, acc[m]));
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      float v = acc[m];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
      }
      if (lane == 0) {
        out[static_cast<size_t>(m) * N + n] = __float2bfloat16(v);
      }
    }
  }
}

template <int SB_DTYPE, int M_CAP, int NUM_WARPS, bool SYM>
__global__ __launch_bounds__(NUM_WARPS * 32, 4) void mlx_node_affine_q8_u4_1warp(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  aq8_u4_1warp_impl<SB_DTYPE, M_CAP, NUM_WARPS, SYM>(
      x, w, scales, biases, out, M, N, K);
}

// Coalesced 256B/row SMEM-staged BF16 Tensor Core MMA kernel with Split-K.
)cuda"
R"cuda(
// Shared MMA implementation used by both single-matrix Split-K and batched pair kernels.
template <int SB_DTYPE, int M_TILES, bool SPLIT_K, bool SYM>
__device__ __forceinline__ void aq8_mma_impl(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    void* __restrict__ out,
    int M, int N, int K,
    int split_idx, int num_splits) {
  const int row0 = static_cast<int>(blockIdx.x) * 16;
  if (row0 >= N) {
    return;
  }

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int num_groups = K >> 5;
  const int num_tiles = num_groups >> 3; // full 8-group (256-weight) tiles
  const int tail_groups = num_groups & 7;
  const int tile_start = (num_tiles * split_idx) / num_splits;
  const int tile_end = (num_tiles * (split_idx + 1)) / num_splits;

  const size_t w_stride_u4 = static_cast<size_t>(K) >> 4;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  union SharedBuf {
    struct {
      uint4 w[2][16][16];    // 8192 B (XOR swizzled for 0 bank conflicts)
      float sc[2][16][9];    // 1152 B (padded +1 for 0 bank conflicts)
      float bi[SYM ? 1 : 2][SYM ? 1 : 16][SYM ? 1 : 9];
    } pipe;
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const int ld_row = tid >> 4;   // 0..15
  const int ld_col = tid & 15;   // 0..15 (256B contiguous per row)
  const int ld_swz = ld_col ^ ld_row;
  const int ld_r = (row0 + ld_row < N) ? (row0 + ld_row) : (N - 1);
  const uint4* w_load_ptr =
      reinterpret_cast<const uint4*>(w) + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;

  const int sb_is_bias = tid >> 7;        // 0 or 1
  const int sb_row = (tid >> 3) & 15;     // 0..15
  const int sb_grp = tid & 7;             // 0..7
  const int sb_r = (row0 + sb_row < N) ? (row0 + sb_row) : (N - 1);
  [[maybe_unused]] const void* sb_src = SYM ? scales : (sb_is_bias ? biases : scales);
  const size_t sb_base_idx =
      static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + sb_grp;

  const bool valid_m0 = (g < M);
  const uint4* x0_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m0 ? g : 0) * x_stride_u4 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  const uint4* x1_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m1 ? (g + 8) : 0) * x_stride_u4 + t;

  float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ (g + 8);

  const int n_tiles_mine = tile_end - tile_start;
  if (n_tiles_mine > 0) {
    sh.pipe.w[0][ld_row][ld_swz] =
        aq8_ld_cs_u4(w_load_ptr + static_cast<size_t>(tile_start) * 16);
    if constexpr (SYM) {
      if (!sb_is_bias) {
        sh.pipe.sc[0][sb_row][sb_grp] = aq8_load_sb1<SB_DTYPE>(
            scales, sb_base_idx + static_cast<size_t>(tile_start) * 8);
      }
    } else {
      const float sbv = aq8_load_sb1<SB_DTYPE>(
          sb_src, sb_base_idx + static_cast<size_t>(tile_start) * 8);
      if (sb_is_bias) {
        sh.pipe.bi[0][sb_row][sb_grp] = sbv;
      } else {
        sh.pipe.sc[0][sb_row][sb_grp] = sbv;
      }
    }
    __syncthreads();

    for (int i = 0; i < n_tiles_mine; ++i) {
      const int cur = i & 1;
      const int nxt = cur ^ 1;
      uint4 next_w;
      float next_sb = 0.0f;
      if (i + 1 < n_tiles_mine) {
        const size_t nt = static_cast<size_t>(tile_start + i + 1);
        next_w = aq8_ld_cs_u4(w_load_ptr + nt * 16);
        if constexpr (SYM) {
          if (!sb_is_bias) {
            next_sb = aq8_load_sb1<SB_DTYPE>(scales, sb_base_idx + nt * 8);
          }
        } else {
          next_sb = aq8_load_sb1<SB_DTYPE>(sb_src, sb_base_idx + nt * 8);
        }
      }

      const uint2 w0 =
          reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g][swz_r0])[half_u2];
      const uint2 w1 =
          reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g + 8][swz_r1])[half_u2];

      const int grp_idx = (tile_start + i) * 8 + warp;
      uint4 xv0, xv1;
      float xs00, xs01, xs10 = 0.0f, xs11 = 0.0f;
      aq8_load_x_g32<SYM>(
          x0_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m0, t, xv0, xs00, xs01);
      if constexpr (M_TILES == 2) {
        aq8_load_x_g32<SYM>(
            x1_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m1, t, xv1, xs10, xs11);
      }

      float d0[4], d1[4];
      aq8_mma_g32<M_TILES, SYM>(w0.x, w0.y, w1.x, w1.y, xv0, xv1, d0, d1);

      const float sc0 = sh.pipe.sc[cur][g][warp];
      const float sc1 = sh.pipe.sc[cur][g + 8][warp];
      if constexpr (SYM) {
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
      } else {
        const float bi0 = sh.pipe.bi[cur][g][warp];
        const float bi1 = sh.pipe.bi[cur][g + 8][warp];
        total0[0] = fmaf(sc0, d0[0], fmaf(bi0, xs00, total0[0]));
        total0[1] = fmaf(sc0, d0[1], fmaf(bi0, xs01, total0[1]));
        total0[2] = fmaf(sc1, d0[2], fmaf(bi1, xs00, total0[2]));
        total0[3] = fmaf(sc1, d0[3], fmaf(bi1, xs01, total0[3]));
        if constexpr (M_TILES == 2) {
          total1[0] = fmaf(sc0, d1[0], fmaf(bi0, xs10, total1[0]));
          total1[1] = fmaf(sc0, d1[1], fmaf(bi0, xs11, total1[1]));
          total1[2] = fmaf(sc1, d1[2], fmaf(bi1, xs10, total1[2]));
          total1[3] = fmaf(sc1, d1[3], fmaf(bi1, xs11, total1[3]));
        }
      }

      if (i + 1 < n_tiles_mine) {
        sh.pipe.w[nxt][ld_row][ld_swz] = next_w;
        if constexpr (SYM) {
          if (!sb_is_bias) {
            sh.pipe.sc[nxt][sb_row][sb_grp] = next_sb;
          }
        } else {
          if (sb_is_bias) {
            sh.pipe.bi[nxt][sb_row][sb_grp] = next_sb;
          } else {
            sh.pipe.sc[nxt][sb_row][sb_grp] = next_sb;
          }
        }
      }
      __syncthreads();
    }
  }

  // Handle tail groups (1..7 groups when K % 256 != 0) on the last split.
  if (tail_groups > 0 && split_idx == num_splits - 1 && warp < tail_groups) {
    const int grp_idx = num_tiles * 8 + warp;
    const int r0 = (row0 + g < N) ? (row0 + g) : (N - 1);
    const int r1 = (row0 + g + 8 < N) ? (row0 + g + 8) : (N - 1);
    const size_t w_stride_u2 = static_cast<size_t>(K) >> 3;
    const uint2 w0 = reinterpret_cast<const uint2*>(
        w)[static_cast<size_t>(r0) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];
    const uint2 w1 = reinterpret_cast<const uint2*>(
        w)[static_cast<size_t>(r1) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];

    uint4 xv0, xv1;
    float xs00, xs01, xs10 = 0.0f, xs11 = 0.0f;
    aq8_load_x_g32<SYM>(
        x0_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m0, t, xv0, xs00, xs01);
    if constexpr (M_TILES == 2) {
      aq8_load_x_g32<SYM>(
          x1_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m1, t, xv1, xs10, xs11);
    }

    float d0[4], d1[4];
    aq8_mma_g32<M_TILES, SYM>(w0.x, w0.y, w1.x, w1.y, xv0, xv1, d0, d1);

    const float sc0 =
        aq8_load_sb1<SB_DTYPE>(scales, static_cast<size_t>(r0) * num_groups + grp_idx);
    const float sc1 =
        aq8_load_sb1<SB_DTYPE>(scales, static_cast<size_t>(r1) * num_groups + grp_idx);
    if constexpr (SYM) {
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
    } else {
      const float bi0 =
          aq8_load_sb1<SB_DTYPE>(biases, static_cast<size_t>(r0) * num_groups + grp_idx);
      const float bi1 =
          aq8_load_sb1<SB_DTYPE>(biases, static_cast<size_t>(r1) * num_groups + grp_idx);
      total0[0] = fmaf(sc0, d0[0], fmaf(bi0, xs00, total0[0]));
      total0[1] = fmaf(sc0, d0[1], fmaf(bi0, xs01, total0[1]));
      total0[2] = fmaf(sc1, d0[2], fmaf(bi1, xs00, total0[2]));
      total0[3] = fmaf(sc1, d0[3], fmaf(bi1, xs01, total0[3]));
      if constexpr (M_TILES == 2) {
        total1[0] = fmaf(sc0, d1[0], fmaf(bi0, xs10, total1[0]));
        total1[1] = fmaf(sc0, d1[1], fmaf(bi0, xs11, total1[1]));
        total1[2] = fmaf(sc1, d1[2], fmaf(bi1, xs10, total1[2]));
        total1[3] = fmaf(sc1, d1[3], fmaf(bi1, xs11, total1[3]));
      }
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

  for (int idx = tid; idx < M * 16; idx += 256) {
    const int m = idx >> 4;
    const int n_local = idx & 15;
    const int n_col = row0 + n_local;
    if (n_col < N) {
      float sum = 0.0f;
#pragma unroll
      for (int ww = 0; ww < 8; ++ww) {
        sum += sh.reduce[ww][m][n_local];
      }
      if constexpr (SPLIT_K) {
        reinterpret_cast<float*>(out)[(static_cast<size_t>(split_idx) * M + m) * N + n_col] =
            sum;
      } else {
        reinterpret_cast<nv_bfloat16*>(out)[static_cast<size_t>(m) * N + n_col] =
            __float2bfloat16(sum);
      }
    }
  }
}
)cuda"
R"cuda(
// Grid: dim3((N + 15) / 16, split_k), Block: dim3(32, 8).
template <int SB_DTYPE, int M_TILES, bool SPLIT_K, bool SYM>
__global__ __launch_bounds__(256, 2) void mlx_node_affine_q8_mma(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    void* __restrict__ out,
    int M, int N, int K) {
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  aq8_mma_impl<SB_DTYPE, M_TILES, SPLIT_K, SYM>(
      x, w, scales, biases, out, M, N, K, split_idx, num_splits);
}

// Batched 2-matrix 1-warp-per-row streaming kernel for M in 1..=2.
// Grid: dim3((N + NUM_WARPS - 1) / NUM_WARPS, 2), Block: dim3(32, NUM_WARPS).
// blockIdx.y == 0 computes x0 @ W0^T -> out0; blockIdx.y == 1 computes x1 @ W1^T -> out1.
template <int SB_DTYPE, int M_CAP, int NUM_WARPS, bool SYM>
__global__ __launch_bounds__(NUM_WARPS * 32, 4) void mlx_node_affine_q8_u4_1warp_pair(
    const nv_bfloat16* __restrict__ x0,
    const unsigned int* __restrict__ w0,
    const void* __restrict__ scales0,
    const void* __restrict__ biases0,
    nv_bfloat16* __restrict__ out0,
    const nv_bfloat16* __restrict__ x1,
    const unsigned int* __restrict__ w1,
    const void* __restrict__ scales1,
    const void* __restrict__ biases1,
    nv_bfloat16* __restrict__ out1,
    int M, int N, int K) {
  const bool is1 = (blockIdx.y != 0);
  aq8_u4_1warp_impl<SB_DTYPE, M_CAP, NUM_WARPS, SYM>(
      is1 ? x1 : x0, is1 ? w1 : w0, is1 ? scales1 : scales0,
      is1 ? biases1 : biases0, is1 ? out1 : out0, M, N, K);
}

// Batched 2-matrix BF16 Tensor Core MMA kernel for M in 3..=16 (ZERO Split-K, ZERO reduce!).
// Grid: dim3((N + 15) / 16, 2), Block: dim3(32, 8).
// blockIdx.y == 0 computes x0 @ W0^T -> out0; blockIdx.y == 1 computes x1 @ W1^T -> out1.
template <int SB_DTYPE, int M_TILES, bool SYM>
__global__ __launch_bounds__(256, 3) void mlx_node_affine_q8_mma_pair(
    const nv_bfloat16* __restrict__ x0,
    const unsigned int* __restrict__ w0,
    const void* __restrict__ scales0,
    const void* __restrict__ biases0,
    void* __restrict__ out0,
    const nv_bfloat16* __restrict__ x1,
    const unsigned int* __restrict__ w1,
    const void* __restrict__ scales1,
    const void* __restrict__ biases1,
    void* __restrict__ out1,
    int M, int N, int K) {
  const bool is1 = (blockIdx.y != 0);
  aq8_mma_impl<SB_DTYPE, M_TILES, false, SYM>(
      is1 ? x1 : x0, is1 ? w1 : w0, is1 ? scales1 : scales0,
      is1 ? biases1 : biases0, is1 ? out1 : out0, M, N, K, 0, 1);
}

// Deterministic Split-K partial reduction: sums FP32 partials [split_k, M, N] -> BF16 [M, N].
extern "C" __global__ void mlx_node_affine_q8_g32_reduce(
    const float* __restrict__ partials,
    nv_bfloat16* __restrict__ out,
    int M, int N, int split_k) {
  const int total = M * N;
  const int idx =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if ((total & 3) == 0) {
    const int total4 = total >> 2;
    if (idx < total4) {
      const float4* p4 = reinterpret_cast<const float4*>(partials);
      const size_t stride4 = static_cast<size_t>(total4);
      float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (split_k == 3) {
        const float4 a = p4[idx];
        const float4 b = p4[stride4 + idx];
        const float4 c = p4[2 * stride4 + idx];
        sum.x = (a.x + b.x) + c.x;
        sum.y = (a.y + b.y) + c.y;
        sum.z = (a.z + b.z) + c.z;
        sum.w = (a.w + b.w) + c.w;
      } else {
        for (int s = 0; s < split_k; ++s) {
          const float4 v = p4[static_cast<size_t>(s) * stride4 + idx];
          sum.x += v.x;
          sum.y += v.y;
          sum.z += v.z;
          sum.w += v.w;
        }
      }
      uint2 raw;
      const __nv_bfloat162 lo = __floats2bfloat162_rn(sum.x, sum.y);
      const __nv_bfloat162 hi = __floats2bfloat162_rn(sum.z, sum.w);
      raw.x = *reinterpret_cast<const unsigned int*>(&lo);
      raw.y = *reinterpret_cast<const unsigned int*>(&hi);
      reinterpret_cast<uint2*>(out)[idx] = raw;
    }
    return;
  }
  if (idx < total) {
    float sum = 0.0f;
    const size_t stride = static_cast<size_t>(total);
    for (int s = 0; s < split_k; ++s) {
      sum += partials[static_cast<size_t>(s) * stride + idx];
    }
    out[idx] = __float2bfloat16(sum);
  }
}
)cuda"
