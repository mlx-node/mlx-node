R"cuda(
#include <cuda_bf16.h>

// High-bandwidth BF16 x row-major affine 4-bit (group_size=64, BF16 scales/biases)
// GEMM for small M (1..=8), K >= 1024 (K % 256 == 0), N > 0 (N % 16 == 0).
// Grid: dim3(N / 16), Block: dim3(32, 8).
// Each CTA computes 16 output rows x M batch rows via 8-warp split-K over 128-weight
// chunks (2 groups of 64). Weights stream via 128-bit ld.global.cs (cache-streaming)
// directly into registers, warp-shuffled to separate the two groups without weight
// SMEM staging, unpacked to exact BF16 integers 0..15, and accumulated with
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 before applying per-group
// scale and bias*sum(x) once per 64-element group.

__device__ __forceinline__ void aq4_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 aq4_ld_cs_u4(const uint4* ptr) {
  uint4 v;
  asm(
      "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

// Extract two consecutive 4-bit nibbles from byte BYTE_IDX of w (and w_hi = w >> 4)
// and convert to exact BF16x2 values in [0.0, 15.0] via 0x4300 (128.0) bias + hsub2.
template <int BYTE_IDX>
__device__ __forceinline__ unsigned int aq4_unpack_pair(
    unsigned int w, unsigned int w_hi) {
  constexpr unsigned int sel = BYTE_IDX | ((BYTE_IDX + 4) << 8);
  const unsigned int raw = (__byte_perm(w, w_hi, sel) & 0x000F000Fu) | 0x43004300u;
  const unsigned int magic_u = 0x43004300u;
  const __nv_bfloat162 v = __hsub2(
      *reinterpret_cast<const __nv_bfloat162*>(&raw),
      *reinterpret_cast<const __nv_bfloat162*>(&magic_u));
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float aq4_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

__device__ __forceinline__ void aq4_load_x_group64(
    const uint4* __restrict__ x_row_u4, int grp_u4, bool valid_m,
    int t, uint4& x_lo, uint4& x_hi, float& xsum_c0, float& xsum_c1) {
  if (valid_m) {
    x_lo = x_row_u4[grp_u4];
    x_hi = x_row_u4[grp_u4 + 1];
  } else {
    x_lo = make_uint4(0, 0, 0, 0);
    x_hi = make_uint4(0, 0, 0, 0);
  }
  float my_sum = valid_m ? (aq4_sum_bf16x8(x_lo) + aq4_sum_bf16x8(x_hi)) : 0.0f;
  my_sum += __shfl_xor_sync(0xffffffffu, my_sum, 1);
  my_sum += __shfl_xor_sync(0xffffffffu, my_sum, 2);
  xsum_c0 = __shfl_sync(0xffffffffu, my_sum, (2 * t) * 4);
  xsum_c1 = __shfl_sync(0xffffffffu, my_sum, (2 * t + 1) * 4);
}

__device__ __forceinline__ void aq4_mma_group64(
    unsigned int r0_w0, unsigned int r0_w1,
    unsigned int r1_w0, unsigned int r1_w1,
    const uint4& x_lo, const uint4& x_hi,
    float& dot0, float& dot1, float& dot2, float& dot3) {
  dot0 = 0.0f;
  dot1 = 0.0f;
  dot2 = 0.0f;
  dot3 = 0.0f;

  {
    const unsigned int r0_hi = r0_w0 >> 4;
    const unsigned int r1_hi = r1_w0 >> 4;
    unsigned int a0 = aq4_unpack_pair<0>(r0_w0, r0_hi);
    unsigned int a1 = aq4_unpack_pair<0>(r1_w0, r1_hi);
    unsigned int a2 = aq4_unpack_pair<1>(r0_w0, r0_hi);
    unsigned int a3 = aq4_unpack_pair<1>(r1_w0, r1_hi);
    aq4_mma_m16n8k16_bf16(dot0, dot1, dot2, dot3, a0, a1, a2, a3, x_lo.x, x_lo.y);

    a0 = aq4_unpack_pair<2>(r0_w0, r0_hi);
    a1 = aq4_unpack_pair<2>(r1_w0, r1_hi);
    a2 = aq4_unpack_pair<3>(r0_w0, r0_hi);
    a3 = aq4_unpack_pair<3>(r1_w0, r1_hi);
    aq4_mma_m16n8k16_bf16(dot0, dot1, dot2, dot3, a0, a1, a2, a3, x_lo.z, x_lo.w);
  }
  {
    const unsigned int r0_hi = r0_w1 >> 4;
    const unsigned int r1_hi = r1_w1 >> 4;
    unsigned int a0 = aq4_unpack_pair<0>(r0_w1, r0_hi);
    unsigned int a1 = aq4_unpack_pair<0>(r1_w1, r1_hi);
    unsigned int a2 = aq4_unpack_pair<1>(r0_w1, r0_hi);
    unsigned int a3 = aq4_unpack_pair<1>(r1_w1, r1_hi);
    aq4_mma_m16n8k16_bf16(dot0, dot1, dot2, dot3, a0, a1, a2, a3, x_hi.x, x_hi.y);

    a0 = aq4_unpack_pair<2>(r0_w1, r0_hi);
    a1 = aq4_unpack_pair<2>(r1_w1, r1_hi);
    a2 = aq4_unpack_pair<3>(r0_w1, r0_hi);
    a3 = aq4_unpack_pair<3>(r1_w1, r1_hi);
    aq4_mma_m16n8k16_bf16(dot0, dot1, dot2, dot3, a0, a1, a2, a3, x_hi.z, x_hi.w);
  }
}

__device__ __forceinline__ void aq4_process_chunk128(
    int ch, const uint4& wl0, const uint4& wl1, unsigned int sb_raw,
    const uint4* __restrict__ x_row_u4, bool valid_m,
    int g, int t, bool is_hi, float total[4]) {
  // Exchange half-chunks between t in {0,1} and t in {2,3} so all 4 lanes in
  // group g hold Group 0 (k=0..63) then Group 1 (k=64..127) in permutation
  // t_perm = ((t & 1) << 1) | (t >> 1).
  const unsigned int r0_sw0 = __shfl_xor_sync(0xffffffffu, is_hi ? wl0.x : wl0.z, 2);
  const unsigned int r0_sw1 = __shfl_xor_sync(0xffffffffu, is_hi ? wl0.y : wl0.w, 2);
  const unsigned int r1_sw0 = __shfl_xor_sync(0xffffffffu, is_hi ? wl1.x : wl1.z, 2);
  const unsigned int r1_sw1 = __shfl_xor_sync(0xffffffffu, is_hi ? wl1.y : wl1.w, 2);

  const unsigned int s0_raw = __shfl_sync(0xffffffffu, sb_raw, g * 4 + 0);
  const unsigned int b0_raw = __shfl_sync(0xffffffffu, sb_raw, g * 4 + 1);
  const unsigned int s1_raw = __shfl_sync(0xffffffffu, sb_raw, g * 4 + 2);
  const unsigned int b1_raw = __shfl_sync(0xffffffffu, sb_raw, g * 4 + 3);

  uint4 x_lo, x_hi;
  float xsum_c0, xsum_c1;
  float d0, d1, d2, d3;

  // Group 0
  aq4_load_x_group64(x_row_u4, ch * 16, valid_m, t, x_lo, x_hi, xsum_c0, xsum_c1);
  aq4_mma_group64(
      is_hi ? r0_sw0 : wl0.x, is_hi ? r0_sw1 : wl0.y,
      is_hi ? r1_sw0 : wl1.x, is_hi ? r1_sw1 : wl1.y,
      x_lo, x_hi, d0, d1, d2, d3);

  const float2 sc0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s0_raw));
  const float2 bi0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b0_raw));
  const float2 sc1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s1_raw));
  const float2 bi1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b1_raw));

  total[0] = fmaf(sc0.x, d0, fmaf(bi0.x, xsum_c0, total[0]));
  total[1] = fmaf(sc0.x, d1, fmaf(bi0.x, xsum_c1, total[1]));
  total[2] = fmaf(sc1.x, d2, fmaf(bi1.x, xsum_c0, total[2]));
  total[3] = fmaf(sc1.x, d3, fmaf(bi1.x, xsum_c1, total[3]));

  // Group 1
  aq4_load_x_group64(x_row_u4, ch * 16 + 8, valid_m, t, x_lo, x_hi, xsum_c0, xsum_c1);
  aq4_mma_group64(
      is_hi ? wl0.z : r0_sw0, is_hi ? wl0.w : r0_sw1,
      is_hi ? wl1.z : r1_sw0, is_hi ? wl1.w : r1_sw1,
      x_lo, x_hi, d0, d1, d2, d3);

  total[0] = fmaf(sc0.y, d0, fmaf(bi0.y, xsum_c0, total[0]));
  total[1] = fmaf(sc0.y, d1, fmaf(bi0.y, xsum_c1, total[1]));
  total[2] = fmaf(sc1.y, d2, fmaf(bi1.y, xsum_c0, total[2]));
  total[3] = fmaf(sc1.y, d3, fmaf(bi1.y, xsum_c1, total[3]));
}

extern "C" __global__ __launch_bounds__(256, 2) void mlx_node_affine_q4_g64_few(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const nv_bfloat16* __restrict__ scales,
    const nv_bfloat16* __restrict__ biases,
    nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  const int row0 = blockIdx.x * 16;
  if (row0 >= N) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int t_perm = ((t & 1) << 1) | (t >> 1);
  const bool is_hi = (t >= 2);
  const int r0 = row0 + g;
  const int r1 = row0 + g + 8;

  const size_t w_stride_u4 = static_cast<size_t>(K) >> 5;
  const size_t s_stride_u2 = static_cast<size_t>(K) >> 7;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  const uint4* w0_u4 = reinterpret_cast<const uint4*>(w) + static_cast<size_t>(r0) * w_stride_u4 + t;
  const uint4* w1_u4 = reinterpret_cast<const uint4*>(w) + static_cast<size_t>(r1) * w_stride_u4 + t;

  const unsigned int* sb_ptr =
      (t == 0) ? (reinterpret_cast<const unsigned int*>(scales) + static_cast<size_t>(r0) * s_stride_u2) :
      (t == 1) ? (reinterpret_cast<const unsigned int*>(biases) + static_cast<size_t>(r0) * s_stride_u2) :
      (t == 2) ? (reinterpret_cast<const unsigned int*>(scales) + static_cast<size_t>(r1) * s_stride_u2) :
                 (reinterpret_cast<const unsigned int*>(biases) + static_cast<size_t>(r1) * s_stride_u2);

  const bool valid_m = (g < M);
  const uint4* x_row_u4 =
      reinterpret_cast<const uint4*>(x) + static_cast<size_t>(valid_m ? g : 0) * x_stride_u4 + t_perm * 2;

  const int num_chunks = K >> 7;
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (warp < num_chunks) {
    uint4 wl0 = aq4_ld_cs_u4(w0_u4 + static_cast<size_t>(warp) * 4);
    uint4 wl1 = aq4_ld_cs_u4(w1_u4 + static_cast<size_t>(warp) * 4);
    unsigned int sb_raw = sb_ptr[warp];

    for (int ch = warp; ch < num_chunks; ch += 8) {
      uint4 n_wl0, n_wl1;
      unsigned int n_sb = 0;
      if (ch + 8 < num_chunks) {
        n_wl0 = aq4_ld_cs_u4(w0_u4 + static_cast<size_t>(ch + 8) * 4);
        n_wl1 = aq4_ld_cs_u4(w1_u4 + static_cast<size_t>(ch + 8) * 4);
        n_sb = sb_ptr[ch + 8];
      }
      aq4_process_chunk128(
          ch, wl0, wl1, sb_raw, x_row_u4, valid_m, g, t, is_hi, total);
      wl0 = n_wl0;
      wl1 = n_wl1;
      sb_raw = n_sb;
    }
  }

  __shared__ float smem[8][8][16];
  smem[warp][2 * t][g] = total[0];
  smem[warp][2 * t + 1][g] = total[1];
  smem[warp][2 * t][g + 8] = total[2];
  smem[warp][2 * t + 1][g + 8] = total[3];
  __syncthreads();

  const int tid = warp * 32 + lane;
  if (tid < M * 8) {
    const int m = tid >> 3;
    const int c2 = tid & 7;
    const int n0 = c2 * 2;
    const int n1 = n0 + 1;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) {
      sum0 += smem[ww][m][n0];
      sum1 += smem[ww][m][n1];
    }
    __nv_bfloat162* out_b2 = reinterpret_cast<__nv_bfloat162*>(out);
    out_b2[static_cast<size_t>(m) * (static_cast<size_t>(N) >> 1) +
           (static_cast<size_t>(row0) >> 1) + c2] = __floats2bfloat162_rn(sum0, sum1);
  }
}
)cuda"
