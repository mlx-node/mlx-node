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

__device__ __forceinline__ void aq8_mma_m16n8k16_bf16_z(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  float z = 0.0f;
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(z));
}

__device__ __forceinline__ void aq8_mma_m16n8k16_bf16_nv(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void aq8_cp_async16_cg(void* smem_ptr, const void* glob_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
#else
  *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(glob_ptr);
#endif
}

__device__ __forceinline__ void aq8_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void aq8_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

__device__ __forceinline__ unsigned int aq8_s8x2_to_bf16x2(unsigned int p16) {
  const float f0 = static_cast<float>(static_cast<signed char>(p16));
  const float f1 = static_cast<float>(static_cast<signed char>(p16 >> 8));
  const __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
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
  if constexpr (SYM) {
    {
      const unsigned int s0 = r0_u0 ^ 0x80808080u;
      const unsigned int s1 = r1_u0 ^ 0x80808080u;
      const unsigned int a0 = aq8_s8x2_to_bf16x2(s0);
      const unsigned int a1 = aq8_s8x2_to_bf16x2(s1);
      const unsigned int a2 = aq8_s8x2_to_bf16x2(s0 >> 16);
      const unsigned int a3 = aq8_s8x2_to_bf16x2(s1 >> 16);
      aq8_mma_m16n8k16_bf16_z(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.x, x0.y);
      if constexpr (M_TILES == 2) {
        aq8_mma_m16n8k16_bf16_z(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.x, x1.y);
      }
    }
    {
      const unsigned int s0 = r0_u1 ^ 0x80808080u;
      const unsigned int s1 = r1_u1 ^ 0x80808080u;
      const unsigned int a0 = aq8_s8x2_to_bf16x2(s0);
      const unsigned int a1 = aq8_s8x2_to_bf16x2(s1);
      const unsigned int a2 = aq8_s8x2_to_bf16x2(s0 >> 16);
      const unsigned int a3 = aq8_s8x2_to_bf16x2(s1 >> 16);
      aq8_mma_m16n8k16_bf16_nv(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.z, x0.w);
      if constexpr (M_TILES == 2) {
        aq8_mma_m16n8k16_bf16_nv(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.z, x1.w);
      }
    }
  } else {
    d0[0] = d0[1] = d0[2] = d0[3] = 0.0f;
    if constexpr (M_TILES == 2) {
      d1[0] = d1[1] = d1[2] = d1[3] = 0.0f;
    }
    {
      const unsigned int a0 = aq8_unpack_pair<0, false>(r0_u0);
      const unsigned int a1 = aq8_unpack_pair<0, false>(r1_u0);
      const unsigned int a2 = aq8_unpack_pair<1, false>(r0_u0);
      const unsigned int a3 = aq8_unpack_pair<1, false>(r1_u0);
      aq8_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.x, x0.y);
      if constexpr (M_TILES == 2) {
        aq8_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.x, x1.y);
      }
    }
    {
      const unsigned int a0 = aq8_unpack_pair<0, false>(r0_u1);
      const unsigned int a1 = aq8_unpack_pair<0, false>(r1_u1);
      const unsigned int a2 = aq8_unpack_pair<1, false>(r0_u1);
      const unsigned int a3 = aq8_unpack_pair<1, false>(r1_u1);
      aq8_mma_m16n8k16_bf16(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.z, x0.w);
      if constexpr (M_TILES == 2) {
        aq8_mma_m16n8k16_bf16(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.z, x1.w);
      }
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
    d = fmaf(f0.x, static_cast<float>(static_cast<signed char>(sw)),
             fmaf(f0.y, static_cast<float>(static_cast<signed char>(sw >> 8)), d));
    d = fmaf(f1.x, static_cast<float>(static_cast<signed char>(sw >> 16)),
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

)cuda"
R"cuda(
template <bool SYM>
__device__ __forceinline__ void aq8_unpack8_f32(
    unsigned int u0, unsigned int u1, float wf[8]) {
  if constexpr (SYM) {
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
  } else {
    wf[0] = static_cast<float>(u0 & 0xFFu);
    wf[1] = static_cast<float>((u0 >> 8) & 0xFFu);
    wf[2] = static_cast<float>((u0 >> 16) & 0xFFu);
    wf[3] = static_cast<float>(u0 >> 24);
    wf[4] = static_cast<float>(u1 & 0xFFu);
    wf[5] = static_cast<float>((u1 >> 8) & 0xFFu);
    wf[6] = static_cast<float>((u1 >> 16) & 0xFFu);
    wf[7] = static_cast<float>(u1 >> 24);
  }
}

template <bool SYM>
__device__ __forceinline__ void aq8_dot8_preunpacked(
    const float wf[8], const uint4& xv, float& d, float& s) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  d = fmaf(f0.x, wf[0], fmaf(f0.y, wf[1], d));
  d = fmaf(f1.x, wf[2], fmaf(f1.y, wf[3], d));
  d = fmaf(f2.x, wf[4], fmaf(f2.y, wf[5], d));
  d = fmaf(f3.x, wf[6], fmaf(f3.y, wf[7], d));
  if constexpr (!SYM) {
    s += ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
  }
}

template <int M_CAP, bool SYM>
__device__ __forceinline__ void aq8_u4_1warp_step(
    const uint4& wv,
    float sc,
    float bi,
    const uint4* __restrict__ x_u4,
    size_t x_stride_u4,
    int x_off,
    int M,
    float acc[M_CAP]) {
  if constexpr (M_CAP == 1) {
    const uint4* x_ptr = x_u4 + x_off;
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
      acc[0] = fmaf(sc, d, acc[0]);
    } else {
      acc[0] = fmaf(sc, d, fmaf(bi, s, acc[0]));
    }
  } else {
    float d[M_CAP];
    [[maybe_unused]] float s[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      d[m] = 0.0f;
      if constexpr (!SYM) {
        s[m] = 0.0f;
      }
    }
    const uint4* x_col = x_u4 + x_off;
    float wf[8];
    aq8_unpack8_f32<SYM>(wv.x, wv.y, wf);
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < M) {
        const uint4 x0 = x_col[static_cast<size_t>(m) * x_stride_u4];
        aq8_dot8_preunpacked<SYM>(wf, x0, d[m], s[m]);
      }
    }
    aq8_unpack8_f32<SYM>(wv.z, wv.w, wf);
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < M) {
        const uint4 x1 = x_col[static_cast<size_t>(m) * x_stride_u4 + 1];
        aq8_dot8_preunpacked<SYM>(wf, x1, d[m], s[m]);
        if constexpr (SYM) {
          acc[m] = fmaf(sc, d[m], acc[m]);
        } else {
          acc[m] = fmaf(sc, d[m], fmaf(bi, s[m], acc[m]));
        }
      }
    }
  }
}

// Single-launch 1-warp-per-row uint4 streaming kernel for M in 1..=4 (N >= 128, K >= 512):
// zero SMEM, zero __syncthreads(), 2x unrolled ld.global.cs.v4.u32, shared weight decode.
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
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  int u = lane;
#pragma unroll 1
  for (; u + 32 < num_u4; u += 64) {
    const uint4 wv0 = aq8_ld_cs_u4(w_row + u);
    const uint4 wv1 = aq8_ld_cs_u4(w_row + u + 32);
    const int g0 = u >> 1;
    const float sc0 = aq8_load_sb1<SB_DTYPE>(scales, sb_row + g0);
    [[maybe_unused]] float bi0 = 0.0f;
    if constexpr (!SYM) {
      bi0 = aq8_load_sb1<SB_DTYPE>(biases, sb_row + g0);
    }
    aq8_u4_1warp_step<M_CAP, SYM>(
        wv0, sc0, bi0, x_u4, x_stride_u4, u * 2, M, acc);
    const int g1 = (u + 32) >> 1;
    const float sc1 = aq8_load_sb1<SB_DTYPE>(scales, sb_row + g1);
    [[maybe_unused]] float bi1 = 0.0f;
    if constexpr (!SYM) {
      bi1 = aq8_load_sb1<SB_DTYPE>(biases, sb_row + g1);
    }
    aq8_u4_1warp_step<M_CAP, SYM>(
        wv1, sc1, bi1, x_u4, x_stride_u4, (u + 32) * 2, M, acc);
  }
  if (u < num_u4) {
    const uint4 wv0 = aq8_ld_cs_u4(w_row + u);
    const int g0 = u >> 1;
    const float sc0 = aq8_load_sb1<SB_DTYPE>(scales, sb_row + g0);
    [[maybe_unused]] float bi0 = 0.0f;
    if constexpr (!SYM) {
      bi0 = aq8_load_sb1<SB_DTYPE>(biases, sb_row + g0);
    }
    aq8_u4_1warp_step<M_CAP, SYM>(
        wv0, sc0, bi0, x_u4, x_stride_u4, u * 2, M, acc);
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
__global__ __launch_bounds__(
    NUM_WARPS * 32,
    (NUM_WARPS == 4) ? ((M_CAP >= 4) ? 6 : 8) : ((M_CAP >= 4) ? 3 : 4)) void mlx_node_affine_q8_u4_1warp(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  aq8_u4_1warp_impl<SB_DTYPE, M_CAP, NUM_WARPS, SYM>(
      x, w, scales, biases, out, M, N, K);
}

// Coalesced 256B/row cp.async.cg SMEM-staged BF16 Tensor Core MMA kernel with Split-K.
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
  const int n_tiles_mine = tile_end - tile_start;

  const size_t w_stride_u4 = static_cast<size_t>(K) >> 4;
  const size_t sc_stride_u4 = static_cast<size_t>(K) >> 8;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  constexpr bool kSymSc16 = SYM && (SB_DTYPE != 0);
  [[maybe_unused]] const bool sc_aligned16 =
      kSymSc16 && ((K & 255) == 0) && ((reinterpret_cast<unsigned long long>(scales) & 15u) == 0);

  union SharedBuf {
    struct {
      alignas(16) uint4 w[2][16][16]; // 8192 B (XOR swizzled for 0 bank conflicts)
      alignas(16) uint4 sc16[kSymSc16 ? 2 : 1][kSymSc16 ? 16 : 1]; // 512 B (f16/bf16 sym scales)
      float sc[kSymSc16 ? 1 : 2][kSymSc16 ? 1 : 16][kSymSc16 ? 1 : 9];
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

  const int sc16_r = (row0 + tid < N) ? (row0 + tid) : (N - 1);
  [[maybe_unused]] const uint4* sc16_load_ptr =
      reinterpret_cast<const uint4*>(scales) + static_cast<size_t>(sc16_r) * sc_stride_u4;

  const int sb_is_bias = tid >> 7;        // 0 or 1
  const int sb_row = (tid >> 3) & 15;     // 0..15
  const int sb_grp = tid & 7;             // 0..7
  [[maybe_unused]] const void* sb_src = SYM ? scales : (sb_is_bias ? biases : scales);

  const bool valid_m0 = (g < M);
  const uint4* x0_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m0 ? g : 0) * x_stride_u4 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  [[maybe_unused]] const uint4* x1_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m1 ? (g + 8) : 0) * x_stride_u4 + t;

  float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ (g + 8);

  const int stagger = (n_tiles_mine > 1)
      ? ((static_cast<int>(blockIdx.x) >> 2) % n_tiles_mine)
      : 0;
  auto map_tile = [&](int idx) -> int {
    int loc = idx + stagger;
    if (loc >= n_tiles_mine) {
      loc -= n_tiles_mine;
    }
    return tile_start + loc;
  };

  auto issue_stage = [&](int st, int tile_idx) {
    const size_t nt = static_cast<size_t>(tile_idx);
    aq8_cp_async16_cg(&sh.pipe.w[st][ld_row][ld_swz], w_load_ptr + nt * 16);
    if constexpr (kSymSc16) {
      if (sc_aligned16) {
        if (tid < 16) {
          aq8_cp_async16_cg(&sh.pipe.sc16[st][tid], sc16_load_ptr + nt);
        }
        aq8_cp_commit();
      } else {
        aq8_cp_commit();
        if (!sb_is_bias) {
          const int sb_r = (row0 + sb_row < N) ? (row0 + sb_row) : (N - 1);
          const size_t sb_idx =
              static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + nt * 8 + sb_grp;
          reinterpret_cast<unsigned short*>(&sh.pipe.sc16[st][sb_row])[sb_grp] =
              reinterpret_cast<const unsigned short*>(scales)[sb_idx];
        }
      }
    } else if constexpr (SYM) {
      aq8_cp_commit();
      if (!sb_is_bias) {
        const int sb_r = (row0 + sb_row < N) ? (row0 + sb_row) : (N - 1);
        const size_t sb_idx =
            static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + nt * 8 + sb_grp;
        sh.pipe.sc[st][sb_row][sb_grp] = aq8_load_sb1<SB_DTYPE>(scales, sb_idx);
      }
    } else {
      aq8_cp_commit();
      const int sb_r = (row0 + sb_row < N) ? (row0 + sb_row) : (N - 1);
      const size_t sb_idx =
          static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + nt * 8 + sb_grp;
      const float sbv = aq8_load_sb1<SB_DTYPE>(sb_src, sb_idx);
      if (sb_is_bias) {
        sh.pipe.bi[st][sb_row][sb_grp] = sbv;
      } else {
        sh.pipe.sc[st][sb_row][sb_grp] = sbv;
      }
    }
  };

  if (n_tiles_mine > 0) {
    issue_stage(0, map_tile(0));
    aq8_cp_wait<0>();
    __syncthreads();

    for (int i = 0; i < n_tiles_mine; ++i) {
      const int cur = i & 1;
      const int nxt = cur ^ 1;
      if (i + 1 < n_tiles_mine) {
        issue_stage(nxt, map_tile(i + 1));
      }

      const int tile_idx = map_tile(i);
      const uint2 w0 =
          reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g][swz_r0])[half_u2];
      const uint2 w1 =
          reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g + 8][swz_r1])[half_u2];

      const int grp_idx = tile_idx * 8 + warp;
      uint4 xv0, xv1;
      float xs00, xs01;
      [[maybe_unused]] float xs10 = 0.0f;
      [[maybe_unused]] float xs11 = 0.0f;
      aq8_load_x_g32<SYM>(
          x0_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m0, t, xv0, xs00, xs01);
      if constexpr (M_TILES == 2) {
        aq8_load_x_g32<SYM>(
            x1_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m1, t, xv1, xs10, xs11);
      }

      float d0[4], d1[4];
      aq8_mma_g32<M_TILES, SYM>(w0.x, w0.y, w1.x, w1.y, xv0, xv1, d0, d1);

      float sc0, sc1;
      if constexpr (kSymSc16 && SB_DTYPE == 2) {
        sc0 = __half2float(reinterpret_cast<const __half*>(&sh.pipe.sc16[cur][g])[warp]);
        sc1 = __half2float(reinterpret_cast<const __half*>(&sh.pipe.sc16[cur][g + 8])[warp]);
      } else if constexpr (kSymSc16 && SB_DTYPE == 1) {
        sc0 = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&sh.pipe.sc16[cur][g])[warp]);
        sc1 = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&sh.pipe.sc16[cur][g + 8])[warp]);
      } else {
        sc0 = sh.pipe.sc[cur][g][warp];
        sc1 = sh.pipe.sc[cur][g + 8][warp];
      }

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
        aq8_cp_wait<0>();
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
    float xs00, xs01;
    [[maybe_unused]] float xs10 = 0.0f;
    [[maybe_unused]] float xs11 = 0.0f;
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
