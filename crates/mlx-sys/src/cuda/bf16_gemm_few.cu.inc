R"cuda(
#include <cuda_bf16.h>

// High-bandwidth BF16 -> BF16 small-row (M in 1..=16) GEMM kernels.
// Computes Y[M, N] = X[M, K] @ W[N, K]^T in FP32 accumulation, BF16 output.
//
// Three specialized paths cover the full M in 1..=16 range near the DRAM ceiling:
// 1. M in 1..=3:
//    `mlx_node_bf16_gemm_few_u4_2row<M_CAP, SPLIT_K>` (+ `_pair`)
//    2-rows-per-warp, 4-warp/CTA (8 rows/CTA, 0 B SMEM) streaming uint4
//    (ld.global.cs.v4.u32) sharing L1-cached X across both rows per warp,
//    with 2x pipelined weight loads and optional Split-K for tiny N.
// 2. M in 4..=8 (and fallback for sm < 80, K < 128, or K % 16 != 0):
//    `mlx_node_bf16_gemm_few_u4<M_CAP, NUM_WARPS, SPLIT_K>` (+ `_pair`)
//    1-row-per-warp streaming uint4 (0 B SMEM, 0 barriers) with optional Split-K.
// 3. M in 9..=16 on sm_80+ with K >= 128 && K % 16 == 0:
//    `mlx_node_bf16_gemm_few_mma<2, SPLIT_K>` (+ `_pair`)
//    2-stage cp.async.cg + XOR-swizzled SMEM + m16n8k16 BF16 Tensor Cores
//    with deterministic Split-K (+ vectorized float4->bf16x4 reduce kernel).

__device__ __forceinline__ uint4 bf16f_ld_cs_u4(const uint4* ptr) {
  uint4 v;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ void bf16f_cp_async16_cg(
    void* smem_ptr, const void* glob_ptr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
#else
  (void)smem_ptr;
  (void)glob_ptr;
#endif
}

__device__ __forceinline__ void bf16f_cp_commit() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N_WAIT>
__device__ __forceinline__ void bf16f_cp_wait() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
#endif
}

__device__ __forceinline__ void bf16f_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  (void)d0;
  (void)d1;
  (void)d2;
  (void)d3;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
#endif
}

template <int M_CAP>
__device__ __forceinline__ void bf16f_u4_accum_step(
    const uint4& wv,
    const uint4* __restrict__ x_u4,
    size_t x_stride_u4,
    int kv,
    int M,
    float acc[M_CAP]) {
  const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
  const float2 wf0 = __bfloat1622float2(wb[0]);
  const float2 wf1 = __bfloat1622float2(wb[1]);
  const float2 wf2 = __bfloat1622float2(wb[2]);
  const float2 wf3 = __bfloat1622float2(wb[3]);
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      const uint4 xv = x_u4[static_cast<size_t>(m) * x_stride_u4 + kv];
      const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 xf0 = __bfloat1622float2(xb[0]);
      const float2 xf1 = __bfloat1622float2(xb[1]);
      const float2 xf2 = __bfloat1622float2(xb[2]);
      const float2 xf3 = __bfloat1622float2(xb[3]);
      acc[m] = fmaf(xf0.x, wf0.x,
               fmaf(xf0.y, wf0.y,
               fmaf(xf1.x, wf1.x,
               fmaf(xf1.y, wf1.y, acc[m]))));
      acc[m] = fmaf(xf2.x, wf2.x,
               fmaf(xf2.y, wf2.y,
               fmaf(xf3.x, wf3.x,
               fmaf(xf3.y, wf3.y, acc[m]))));
    }
  }
}

template <int M_CAP, int NUM_WARPS, bool SPLIT_K>
__device__ __forceinline__ void bf16_gemm_few_u4_impl(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K,
    int split_idx, int num_splits) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int n = static_cast<int>(blockIdx.x) * NUM_WARPS + warp;
  if (n >= N) {
    return;
  }

  const int k_vec = K >> 3;  // K / 8 (each uint4 = 8 BF16s)
  const int k_chunks = (k_vec + 31) >> 5;  // 512B (32-uint4) aligned chunks
  const int chunk_start = SPLIT_K ? ((k_chunks * split_idx) / num_splits) : 0;
  const int chunk_end = SPLIT_K ? ((k_chunks * (split_idx + 1)) / num_splits) : k_chunks;
  const int kv_start = chunk_start << 5;
  const int kv_end_raw = chunk_end << 5;
  const int kv_end = (kv_end_raw < k_vec) ? kv_end_raw : k_vec;

  const uint4* w_row =
      reinterpret_cast<const uint4*>(W + static_cast<size_t>(n) * static_cast<size_t>(K));
  const uint4* x_u4 = reinterpret_cast<const uint4*>(X);
  const size_t x_stride_u4 = static_cast<size_t>(k_vec);

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  if constexpr (M_CAP <= 3) {
    float acc1[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      acc1[m] = 0.0f;
    }
    int kv = kv_start + lane;
    for (; kv + 32 < kv_end; kv += 64) {
      const uint4 wv0 = bf16f_ld_cs_u4(w_row + kv);
      const uint4 wv1 = bf16f_ld_cs_u4(w_row + kv + 32);
      bf16f_u4_accum_step<M_CAP>(wv0, x_u4, x_stride_u4, kv, M, acc);
      bf16f_u4_accum_step<M_CAP>(wv1, x_u4, x_stride_u4, kv + 32, M, acc1);
    }
    if (kv < kv_end) {
      const uint4 wv0 = bf16f_ld_cs_u4(w_row + kv);
      bf16f_u4_accum_step<M_CAP>(wv0, x_u4, x_stride_u4, kv, M, acc);
    }
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      acc[m] += acc1[m];
    }
  } else {
    for (int kv = kv_start + lane; kv < kv_end; kv += 32) {
      const uint4 wv0 = bf16f_ld_cs_u4(w_row + kv);
      bf16f_u4_accum_step<M_CAP>(wv0, x_u4, x_stride_u4, kv, M, acc);
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
        if constexpr (SPLIT_K) {
          reinterpret_cast<float*>(Y)[(static_cast<size_t>(split_idx) * M + m) * N + n] = v;
        } else {
          reinterpret_cast<nv_bfloat16*>(Y)[static_cast<size_t>(m) * N + n] =
              __float2bfloat16_rn(v);
        }
      }
    }
  }
}

// Grid: dim3((N + NUM_WARPS - 1) / NUM_WARPS, split_k), Block: dim3(32, NUM_WARPS).
template <int M_CAP, int NUM_WARPS, bool SPLIT_K>
__global__ __launch_bounds__(NUM_WARPS * 32, (M_CAP <= 3) ? ((NUM_WARPS == 2) ? 12 : 6) : 2)
void mlx_node_bf16_gemm_few_u4(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K) {
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  bf16_gemm_few_u4_impl<M_CAP, NUM_WARPS, SPLIT_K>(
      X, W, Y, M, N, K, split_idx, num_splits);
}

// Batched 2-matrix streaming u4 kernel.
// Grid: dim3((N + NUM_WARPS - 1) / NUM_WARPS, split_k, 2), Block: dim3(32, NUM_WARPS).
template <int M_CAP, int NUM_WARPS, bool SPLIT_K>
__global__ __launch_bounds__(NUM_WARPS * 32, (M_CAP <= 3) ? ((NUM_WARPS == 2) ? 12 : 6) : 2)
void mlx_node_bf16_gemm_few_u4_pair(
    const nv_bfloat16* __restrict__ X0,
    const nv_bfloat16* __restrict__ W0,
    void* __restrict__ Y0,
    const nv_bfloat16* __restrict__ X1,
    const nv_bfloat16* __restrict__ W1,
    void* __restrict__ Y1,
    int M, int N, int K) {
  const bool is1 = (blockIdx.z != 0);
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  bf16_gemm_few_u4_impl<M_CAP, NUM_WARPS, SPLIT_K>(
      is1 ? X1 : X0, is1 ? W1 : W0, is1 ? Y1 : Y0,
      M, N, K, split_idx, num_splits);
}
)cuda"
R"cuda(
// 2-rows-per-warp streaming uint4 accumulator step for M in 1..=3.
// Shares each X[m, kv] load across both output rows (n0, n1) in the warp.
template <int M_CAP>
__device__ __forceinline__ void bf16f_u4_2row_accum_step(
    const uint4& wv0,
    const uint4& wv1,
    const uint4* __restrict__ x_u4,
    size_t x_stride_u4,
    int kv,
    int M,
    float acc0[M_CAP],
    float acc1[M_CAP]) {
  const __nv_bfloat162* w0b = reinterpret_cast<const __nv_bfloat162*>(&wv0);
  const float2 w0f0 = __bfloat1622float2(w0b[0]);
  const float2 w0f1 = __bfloat1622float2(w0b[1]);
  const float2 w0f2 = __bfloat1622float2(w0b[2]);
  const float2 w0f3 = __bfloat1622float2(w0b[3]);
  const __nv_bfloat162* w1b = reinterpret_cast<const __nv_bfloat162*>(&wv1);
  const float2 w1f0 = __bfloat1622float2(w1b[0]);
  const float2 w1f1 = __bfloat1622float2(w1b[1]);
  const float2 w1f2 = __bfloat1622float2(w1b[2]);
  const float2 w1f3 = __bfloat1622float2(w1b[3]);
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      const uint4 xv = x_u4[static_cast<size_t>(m) * x_stride_u4 + kv];
      const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 xf0 = __bfloat1622float2(xb[0]);
      const float2 xf1 = __bfloat1622float2(xb[1]);
      const float2 xf2 = __bfloat1622float2(xb[2]);
      const float2 xf3 = __bfloat1622float2(xb[3]);
      acc0[m] = fmaf(xf0.x, w0f0.x,
                fmaf(xf0.y, w0f0.y,
                fmaf(xf1.x, w0f1.x,
                fmaf(xf1.y, w0f1.y, acc0[m]))));
      acc1[m] = fmaf(xf0.x, w1f0.x,
                fmaf(xf0.y, w1f0.y,
                fmaf(xf1.x, w1f1.x,
                fmaf(xf1.y, w1f1.y, acc1[m]))));
      acc0[m] = fmaf(xf2.x, w0f2.x,
                fmaf(xf2.y, w0f2.y,
                fmaf(xf3.x, w0f3.x,
                fmaf(xf3.y, w0f3.y, acc0[m]))));
      acc1[m] = fmaf(xf2.x, w1f2.x,
                fmaf(xf2.y, w1f2.y,
                fmaf(xf3.x, w1f3.x,
                fmaf(xf3.y, w1f3.y, acc1[m]))));
    }
  }
}

template <int M_CAP, bool SPLIT_K>
__device__ __forceinline__ void bf16_gemm_few_u4_2row_impl(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K,
    int split_idx, int num_splits) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int n0 = static_cast<int>(blockIdx.x) * 8 + warp * 2;
  if (n0 >= N) {
    return;
  }
  const int n1 = n0 + 1;
  const bool has_n1 = (n1 < N);

  const int k_vec = K >> 3;  // K / 8 (each uint4 = 8 BF16s)
  const int k_chunks = (k_vec + 31) >> 5;  // 512B (32-uint4) aligned chunks
  const int chunk_start = SPLIT_K ? ((k_chunks * split_idx) / num_splits) : 0;
  const int chunk_end = SPLIT_K ? ((k_chunks * (split_idx + 1)) / num_splits) : k_chunks;
  const int kv_start = chunk_start << 5;
  const int kv_end_raw = chunk_end << 5;
  const int kv_end = (kv_end_raw < k_vec) ? kv_end_raw : k_vec;

  const uint4* w_row0 =
      reinterpret_cast<const uint4*>(W + static_cast<size_t>(n0) * static_cast<size_t>(K));
  const uint4* w_row1 = has_n1
      ? reinterpret_cast<const uint4*>(W + static_cast<size_t>(n1) * static_cast<size_t>(K))
      : w_row0;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(X);
  const size_t x_stride_u4 = static_cast<size_t>(k_vec);

  float acc0[M_CAP];
  float acc1[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc0[m] = 0.0f;
    acc1[m] = 0.0f;
  }

  int kv = kv_start + lane;
  for (; kv + 32 < kv_end; kv += 64) {
    const uint4 w0a = bf16f_ld_cs_u4(w_row0 + kv);
    const uint4 w1a = bf16f_ld_cs_u4(w_row1 + kv);
    const uint4 w0b = bf16f_ld_cs_u4(w_row0 + kv + 32);
    const uint4 w1b = bf16f_ld_cs_u4(w_row1 + kv + 32);
    bf16f_u4_2row_accum_step<M_CAP>(
        w0a, w1a, x_u4, x_stride_u4, kv, M, acc0, acc1);
    bf16f_u4_2row_accum_step<M_CAP>(
        w0b, w1b, x_u4, x_stride_u4, kv + 32, M, acc0, acc1);
  }
  if (kv < kv_end) {
    const uint4 w0a = bf16f_ld_cs_u4(w_row0 + kv);
    const uint4 w1a = bf16f_ld_cs_u4(w_row1 + kv);
    bf16f_u4_2row_accum_step<M_CAP>(
        w0a, w1a, x_u4, x_stride_u4, kv, M, acc0, acc1);
  }

#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      float v0 = acc0[m];
      float v1 = acc1[m];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        v0 += __shfl_xor_sync(0xffffffffu, v0, off);
        v1 += __shfl_xor_sync(0xffffffffu, v1, off);
      }
      if (lane == 0) {
        if constexpr (SPLIT_K) {
          float* dst = reinterpret_cast<float*>(Y);
          const size_t base = (static_cast<size_t>(split_idx) * M + m) * N;
          dst[base + n0] = v0;
          if (has_n1) {
            dst[base + n1] = v1;
          }
        } else {
          nv_bfloat16* dst = reinterpret_cast<nv_bfloat16*>(Y);
          const size_t base = static_cast<size_t>(m) * N;
          dst[base + n0] = __float2bfloat16_rn(v0);
          if (has_n1) {
            dst[base + n1] = __float2bfloat16_rn(v1);
          }
        }
      }
    }
  }
}

// 2-rows-per-warp streaming u4 kernel for M in 1..=3 (8 rows/CTA, 0 B SMEM).
// Grid: dim3((N + 7) / 8, split_k), Block: dim3(32, 4).
template <int M_CAP, bool SPLIT_K>
__global__ __launch_bounds__(128, 6) void mlx_node_bf16_gemm_few_u4_2row(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K) {
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  bf16_gemm_few_u4_2row_impl<M_CAP, SPLIT_K>(
      X, W, Y, M, N, K, split_idx, num_splits);
}

// Batched 2-matrix 2-rows-per-warp streaming u4 kernel for M in 1..=3.
// Grid: dim3((N + 7) / 8, split_k, 2), Block: dim3(32, 4).
template <int M_CAP, bool SPLIT_K>
__global__ __launch_bounds__(128, 6) void mlx_node_bf16_gemm_few_u4_2row_pair(
    const nv_bfloat16* __restrict__ X0,
    const nv_bfloat16* __restrict__ W0,
    void* __restrict__ Y0,
    const nv_bfloat16* __restrict__ X1,
    const nv_bfloat16* __restrict__ W1,
    void* __restrict__ Y1,
    int M, int N, int K) {
  const bool is1 = (blockIdx.z != 0);
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  bf16_gemm_few_u4_2row_impl<M_CAP, SPLIT_K>(
      is1 ? X1 : X0, is1 ? W1 : W0, is1 ? Y1 : Y0,
      M, N, K, split_idx, num_splits);
}
)cuda"
R"cuda(
// 2-stage cp.async.cg + XOR-swizzled SMEM + BF16 Tensor Core MMA (m16n8k16)
// for M in 4..=16, sm_80+, K >= 128 && K % 16 == 0.
// BN = 16, K_TILE = 128 (8192 B SMEM for pipe, <= 8704 B for reduce).
template <int M_TILES, bool SPLIT_K>
__device__ __forceinline__ void bf16_gemm_few_mma_impl(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K,
    int split_idx, int num_splits) {
  const int row0 = static_cast<int>(blockIdx.x) * 16;
  if (row0 >= N) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int num_tiles = K >> 7;  // full 128-BF16 tiles
  const int tail_k = K & 127;    // multiple of 16
  const int tile_start = SPLIT_K ? ((num_tiles * split_idx) / num_splits) : 0;
  const int tile_end = SPLIT_K ? ((num_tiles * (split_idx + 1)) / num_splits) : num_tiles;
  const int n_tiles_mine = tile_end - tile_start;
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 3;

  union SharedBuf {
    uint4 w[2][16][16];  // 8192 B (XOR swizzled, 0 bank conflicts)
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const int ld_row = tid >> 4;
  const int ld_col = tid & 15;
  const int ld_swz = ld_col ^ ld_row;
  const int ld_r = (row0 + ld_row < N) ? (row0 + ld_row) : (N - 1);
  const uint4* w_load_ptr =
      reinterpret_cast<const uint4*>(W) + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;

  const bool valid_m0 = (g < M);
  [[maybe_unused]] const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  const nv_bfloat16* x0_row = X + static_cast<size_t>(valid_m0 ? g : 0) * static_cast<size_t>(K);
  [[maybe_unused]] const nv_bfloat16* x1_row =
      X + static_cast<size_t>(valid_m1 ? (g + 8) : 0) * static_cast<size_t>(K);

  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ (g + 8);

  float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (n_tiles_mine > 0) {
    bf16f_cp_async16_cg(
        &sh.w[0][ld_row][ld_swz], w_load_ptr + static_cast<size_t>(tile_start) * 16);
    bf16f_cp_commit();
    bf16f_cp_wait<0>();
    __syncthreads();

    for (int i = 0; i < n_tiles_mine; ++i) {
      const int cur = i & 1;
      const int nxt = cur ^ 1;
      if (i + 1 < n_tiles_mine) {
        bf16f_cp_async16_cg(
            &sh.w[nxt][ld_row][ld_swz],
            w_load_ptr + static_cast<size_t>(tile_start + i + 1) * 16);
        bf16f_cp_commit();
      }

      const uint2 w0 =
          reinterpret_cast<const uint2*>(&sh.w[cur][g][swz_r0])[half_u2];
      const uint2 w1 =
          reinterpret_cast<const uint2*>(&sh.w[cur][g + 8][swz_r1])[half_u2];
      const int k_off = (tile_start + i) * 128 + warp * 16 + t * 4;
      const uint2 xb0 = valid_m0
          ? *reinterpret_cast<const uint2*>(x0_row + k_off)
          : make_uint2(0, 0);
      bf16f_mma_m16n8k16(
          total0[0], total0[1], total0[2], total0[3],
          w0.x, w1.x, w0.y, w1.y, xb0.x, xb0.y);
      if constexpr (M_TILES == 2) {
        const uint2 xb1 = valid_m1
            ? *reinterpret_cast<const uint2*>(x1_row + k_off)
            : make_uint2(0, 0);
        bf16f_mma_m16n8k16(
            total1[0], total1[1], total1[2], total1[3],
            w0.x, w1.x, w0.y, w1.y, xb1.x, xb1.y);
      }

      if (i + 1 < n_tiles_mine) {
        bf16f_cp_wait<0>();
      }
      __syncthreads();
    }
  }

  // Tail (16..112 BF16s when K % 128 != 0) handled on the last split.
  if (tail_k > 0 && split_idx == num_splits - 1) {
    const int k_base = num_tiles * 128 + warp * 16;
    if (k_base < K) {
      const int r0 = (row0 + g < N) ? (row0 + g) : (N - 1);
      const int r1 = (row0 + g + 8 < N) ? (row0 + g + 8) : (N - 1);
      const int k_off = k_base + t * 4;
      const uint2 w0 =
          *reinterpret_cast<const uint2*>(W + static_cast<size_t>(r0) * K + k_off);
      const uint2 w1 =
          *reinterpret_cast<const uint2*>(W + static_cast<size_t>(r1) * K + k_off);
      const uint2 xb0 = valid_m0
          ? *reinterpret_cast<const uint2*>(x0_row + k_off)
          : make_uint2(0, 0);
      bf16f_mma_m16n8k16(
          total0[0], total0[1], total0[2], total0[3],
          w0.x, w1.x, w0.y, w1.y, xb0.x, xb0.y);
      if constexpr (M_TILES == 2) {
        const uint2 xb1 = valid_m1
            ? *reinterpret_cast<const uint2*>(x1_row + k_off)
            : make_uint2(0, 0);
        bf16f_mma_m16n8k16(
            total1[0], total1[1], total1[2], total1[3],
            w0.x, w1.x, w0.y, w1.y, xb1.x, xb1.y);
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
        reinterpret_cast<float*>(Y)[(static_cast<size_t>(split_idx) * M + m) * N + n_col] =
            sum;
      } else {
        reinterpret_cast<nv_bfloat16*>(Y)[static_cast<size_t>(m) * N + n_col] =
            __float2bfloat16_rn(sum);
      }
    }
  }
}

// Grid: dim3((N + 15) / 16, split_k), Block: dim3(32, 8).
template <int M_TILES, bool SPLIT_K>
__global__ __launch_bounds__(256, M_TILES == 1 ? 4 : 3) void mlx_node_bf16_gemm_few_mma(
    const nv_bfloat16* __restrict__ X,
    const nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K) {
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  bf16_gemm_few_mma_impl<M_TILES, SPLIT_K>(
      X, W, Y, M, N, K, split_idx, num_splits);
}

// Batched 2-matrix BF16 Tensor Core MMA kernel.
// Grid: dim3((N + 15) / 16, split_k, 2), Block: dim3(32, 8).
template <int M_TILES, bool SPLIT_K>
__global__ __launch_bounds__(256, M_TILES == 1 ? 4 : 3) void mlx_node_bf16_gemm_few_mma_pair(
    const nv_bfloat16* __restrict__ X0,
    const nv_bfloat16* __restrict__ W0,
    void* __restrict__ Y0,
    const nv_bfloat16* __restrict__ X1,
    const nv_bfloat16* __restrict__ W1,
    void* __restrict__ Y1,
    int M, int N, int K) {
  const bool is1 = (blockIdx.z != 0);
  const int split_idx = SPLIT_K ? static_cast<int>(blockIdx.y) : 0;
  const int num_splits = SPLIT_K ? static_cast<int>(gridDim.y) : 1;
  bf16_gemm_few_mma_impl<M_TILES, SPLIT_K>(
      is1 ? X1 : X0, is1 ? W1 : W0, is1 ? Y1 : Y0,
      M, N, K, split_idx, num_splits);
}

__device__ __forceinline__ void bf16_gemm_few_reduce_impl(
    const float* __restrict__ partials,
    nv_bfloat16* __restrict__ out,
    int M, int N, int split_k) {
  const int total = M * N;
  const int idx =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
      static_cast<int>(threadIdx.x);
  if ((total & 3) == 0) {
    const int total4 = total >> 2;
    if (idx < total4) {
      const float4* p4 = reinterpret_cast<const float4*>(partials);
      const size_t stride4 = static_cast<size_t>(total4);
      float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (split_k == 2) {
        const float4 a = p4[idx];
        const float4 b = p4[stride4 + idx];
        sum.x = a.x + b.x;
        sum.y = a.y + b.y;
        sum.z = a.z + b.z;
        sum.w = a.w + b.w;
      } else if (split_k == 3) {
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
    out[idx] = __float2bfloat16_rn(sum);
  }
}

// Deterministic Split-K partial reduction: [split_k, M, N] FP32 -> [M, N] BF16.
extern "C" __global__ __launch_bounds__(128, 4) void mlx_node_bf16_gemm_few_reduce(
    const float* __restrict__ partials,
    nv_bfloat16* __restrict__ out,
    int M, int N, int split_k) {
  bf16_gemm_few_reduce_impl(partials, out, M, N, split_k);
}

// Batched 2-matrix Split-K partial reduction: Grid dim3((work + 127) / 128, 2).
extern "C" __global__ __launch_bounds__(128, 4) void mlx_node_bf16_gemm_few_reduce_pair(
    const float* __restrict__ partials0,
    nv_bfloat16* __restrict__ out0,
    const float* __restrict__ partials1,
    nv_bfloat16* __restrict__ out1,
    int M, int N, int split_k) {
  const bool is1 = (blockIdx.y != 0);
  bf16_gemm_few_reduce_impl(
      is1 ? partials1 : partials0, is1 ? out1 : out0, M, N, split_k);
}
)cuda"
