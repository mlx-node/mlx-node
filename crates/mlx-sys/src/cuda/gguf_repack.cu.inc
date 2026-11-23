R"cuda(
#include <cuda_fp16.h>

// Fast coalesced CUDA GGUF repack kernels matching gguf_repack.metal and
// gguf_cpu_repack.rs byte-for-byte for Q4_K, Q5_K (RowMajor + Tiled64),
// Q5_1, Q8_0, and Q8_0_bias.

__device__ __forceinline__ unsigned int gr_nib8(unsigned int a, unsigned int b) {
  unsigned int x = a & 0x0F0F0F0Fu;
  x = (x | (x >> 4)) & 0x00FF00FFu;
  x = (x | (x >> 8)) & 0x0000FFFFu;
  unsigned int y = b & 0x0F0F0F0Fu;
  y = (y | (y >> 4)) & 0x00FF00FFu;
  y = (y | (y >> 8)) & 0x0000FFFFu;
  return x | (y << 16);
}

__device__ __forceinline__ unsigned int gr_il2(unsigned int a, unsigned int b) {
  return (a & 0xFFu) | ((b & 0xFFu) << 8) | ((a & 0xFF00u) << 8) |
      ((b & 0xFF00u) << 16);
}

__device__ __forceinline__ uint4 gr_k4_scales(
    unsigned int q0, unsigned int q1, unsigned int q2) {
  unsigned int sl = q0 & 0x3F3F3F3Fu;
  unsigned int ml = q1 & 0x3F3F3F3Fu;
  unsigned int sh = (q2 & 0x0F0F0F0Fu) | ((q0 >> 2) & 0x30303030u);
  unsigned int mh = ((q2 >> 4) & 0x0F0F0F0Fu) | ((q1 >> 2) & 0x30303030u);
  return make_uint4(
      gr_il2(sl, ml), gr_il2(sl >> 16, ml >> 16),
      gr_il2(sh, mh), gr_il2(sh >> 16, mh >> 16));
}

__device__ __forceinline__ uint4 gr_q4k_unit_from(uint4 a, uint4 b, unsigned int u) {
  unsigned int s = (u & 1u) * 4u;
  return make_uint4(
      gr_nib8(a.x >> s, a.y >> s), gr_nib8(a.z >> s, a.w >> s),
      gr_nib8(b.x >> s, b.y >> s), gr_nib8(b.z >> s, b.w >> s));
}

__device__ __forceinline__ unsigned long long gr_pack5_8(unsigned int n, unsigned int h) {
  unsigned long long t = 0ull;
#pragma unroll
  for (unsigned int i = 0; i < 8; ++i) {
    unsigned int c = ((n >> (4u * i)) & 0xFu) | (((h >> i) & 1u) << 4);
    t |= static_cast<unsigned long long>(c) << (5u * i);
  }
  return t;
}

__device__ __forceinline__ unsigned long long gr_q5k_8codes(
    unsigned int qa, unsigned int qb, unsigned int ha, unsigned int hb, unsigned int u) {
  unsigned int s = (u & 1u) * 4u;
  unsigned int n = gr_nib8(qa >> s, qb >> s);
  unsigned int hm = (ha >> u) & 0x01010101u;
  unsigned int hn = (hb >> u) & 0x01010101u;
  unsigned int h = (hm & 1u) | ((hm >> 7) & 2u) | ((hm >> 14) & 4u) |
      ((hm >> 21) & 8u) | ((hn & 1u) << 4) | ((hn >> 3) & 0x20u) |
      ((hn >> 10) & 0x40u) | ((hn >> 17) & 0x80u);
  return gr_pack5_8(n, h);
}

__device__ __forceinline__ void gr_q5k_unit_from_b4(
    const uint4* b4, unsigned int u, unsigned int* o) {
  unsigned int hf = u >> 1;
  uint4 qs0 = b4[3 + 2 * hf];
  uint4 qs1 = b4[4 + 2 * hf];
  uint4 qh0 = b4[1];
  uint4 qh1 = b4[2];
  unsigned long long x0 = gr_q5k_8codes(qs0.x, qs0.y, qh0.x, qh0.y, u);
  unsigned long long x1 = gr_q5k_8codes(qs0.z, qs0.w, qh0.z, qh0.w, u);
  unsigned long long x2 = gr_q5k_8codes(qs1.x, qs1.y, qh1.x, qh1.y, u);
  unsigned long long x3 = gr_q5k_8codes(qs1.z, qs1.w, qh1.z, qh1.w, u);
  o[0] = static_cast<unsigned int>(x0);
  o[1] = static_cast<unsigned int>(x0 >> 32) | static_cast<unsigned int>(x1 << 8);
  o[2] = static_cast<unsigned int>(x1 >> 24) | static_cast<unsigned int>(x2 << 16);
  o[3] = static_cast<unsigned int>(x2 >> 16) | static_cast<unsigned int>(x3 << 24);
  o[4] = static_cast<unsigned int>(x3 >> 8);
}

__device__ __forceinline__ unsigned short gr_q8_0_derived_bias(unsigned short bits) {
  if ((bits & 0x7FFFu) > 0x7C00u) {
    return 0x7E00u;
  }
  __half d = __ushort_as_half(bits);
  __half b = __float2half_rn(-128.0f * __half2float(d));
  return __half_as_ushort(b);
}

// ---------------------------------------------------------------- Q4_K RowMajor
// 144-byte blocks (9 uint4). Grid: ceil(nb / 32), Block: 256 (8 warps x 4 blocks).
extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q4k(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned char* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int row0) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int b0 = (blockIdx.x * 8u + warp) * 4u;
  if (b0 >= nb) return;
  const unsigned int cnt = (nb - b0 < 4u) ? (nb - b0) : 4u;

  __shared__ uint4 s_in[8][36];
  const uint4* src = reinterpret_cast<const uint4*>(staging + src_off) +
      static_cast<unsigned long long>(b0) * 9ull;
  const unsigned int n_u4 = cnt * 9u;
  for (unsigned int i = lane; i < n_u4; i += 32u) {
    s_in[warp][i] = src[i];
  }
  __syncwarp();

  const unsigned int lb = lane >> 3;
  const unsigned int u = lane & 7u;
  if (lb < cnt) {
    const uint4* b4 = &s_in[warp][lb * 9u];
    uint4 a = b4[1u + 2u * (u >> 1)];
    uint4 c = b4[2u + 2u * (u >> 1)];
    unsigned long long gb = static_cast<unsigned long long>(row0) * kb + b0 + lb;
    reinterpret_cast<uint4*>(weight)[gb * 8ull + u] = gr_q4k_unit_from(a, c, u);
    if (u == 0u) {
      uint4 h = b4[0];
      reinterpret_cast<uint4*>(scales)[gb] = gr_k4_scales(h.y, h.z, h.w);
      reinterpret_cast<unsigned int*>(biases)[gb] = h.x;
    }
  }
}

// ---------------------------------------------------------------- Q4_K Tiled64
// Grid: ceil(nb * 8 / 256), Block: 256. Contiguous uint4 output writes.
extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q4k_t64(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned char* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int ku,
    unsigned int tile0) {
  const unsigned int g = blockIdx.x * 256u + threadIdx.x;
  if (g >= nb * 8u) return;
  const unsigned int rr = g & 63u;
  const unsigned int rest = g >> 6;
  const unsigned int U = rest % ku;
  const unsigned int T = rest / ku;
  const unsigned int n = T * 64u + rr;
  const unsigned int s = U >> 3;
  const unsigned int u = U & 7u;
  const unsigned int b = n * kb + s;
  const uint4* b4 = reinterpret_cast<const uint4*>(staging + src_off) +
      static_cast<unsigned long long>(b) * 9ull;
  uint4 a = b4[1u + 2u * (u >> 1)];
  uint4 c = b4[2u + 2u * (u >> 1)];
  const unsigned long long tg = static_cast<unsigned long long>(tile0) + T;
  reinterpret_cast<uint4*>(weight)[static_cast<unsigned long long>(tile0) * ku * 64ull + g] =
      gr_q4k_unit_from(a, c, u);
  if (u == 0u) {
    uint4 h = b4[0];
    unsigned long long o = (tg * kb + s) * 64ull + rr;
    reinterpret_cast<uint4*>(scales)[o] = gr_k4_scales(h.y, h.z, h.w);
    reinterpret_cast<unsigned int*>(biases)[o] = h.x;
  }
}

// ---------------------------------------------------------------- Q5_K RowMajor
// 176-byte blocks (11 uint4) -> 160-byte weight (10 uint4).
// Grid: ceil(nb / 32), Block: 256 (8 warps x 4 blocks).
extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q5k(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned char* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int row0) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int b0 = (blockIdx.x * 8u + warp) * 4u;
  if (b0 >= nb) return;
  const unsigned int cnt = (nb - b0 < 4u) ? (nb - b0) : 4u;

  __shared__ uint4 s_in[8][44];
  __shared__ uint4 s_out[8][40];
  const uint4* src = reinterpret_cast<const uint4*>(staging + src_off) +
      static_cast<unsigned long long>(b0) * 11ull;
  const unsigned int n_in_u4 = cnt * 11u;
  for (unsigned int i = lane; i < n_in_u4; i += 32u) {
    s_in[warp][i] = src[i];
  }
  __syncwarp();

  const unsigned int lb = lane >> 3;
  const unsigned int u = lane & 7u;
  unsigned int* out_u32 = reinterpret_cast<unsigned int*>(s_out[warp]);
  if (lb < cnt) {
    const uint4* b4 = &s_in[warp][lb * 11u];
    unsigned int o[5];
    gr_q5k_unit_from_b4(b4, u, o);
    unsigned int* dst_lane = out_u32 + lane * 5u;
#pragma unroll
    for (int k = 0; k < 5; ++k) dst_lane[k] = o[k];
    if (u == 0u) {
      uint4 h = b4[0];
      unsigned long long gb = static_cast<unsigned long long>(row0) * kb + b0 + lb;
      reinterpret_cast<uint4*>(scales)[gb] = gr_k4_scales(h.y, h.z, h.w);
      reinterpret_cast<unsigned int*>(biases)[gb] = h.x;
    }
  }
  __syncwarp();

  const unsigned long long gb0 = static_cast<unsigned long long>(row0) * kb + b0;
  uint4* w4 = reinterpret_cast<uint4*>(weight) + gb0 * 10ull;
  const unsigned int n_out_u4 = cnt * 10u;
  for (unsigned int i = lane; i < n_out_u4; i += 32u) {
    w4[i] = s_out[warp][i];
  }
}

// ---------------------------------------------------------------- Q5_K Tiled64
// Grid: ceil(nb * 8 / 256), Block: 256. Each warp coalesces 32 units = 40 uint4.
extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q5k_t64(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned char* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int ku,
    unsigned int tile0) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int g0 = (blockIdx.x * 8u + warp) * 32u;
  const unsigned int total_u = nb * 8u;
  if (g0 >= total_u) return;
  const unsigned int cnt = (total_u - g0 < 32u) ? (total_u - g0) : 32u;

  __shared__ uint4 s_out[8][40];
  unsigned int* out_u32 = reinterpret_cast<unsigned int*>(s_out[warp]);
  if (lane < cnt) {
    const unsigned int g = g0 + lane;
    const unsigned int rr = g & 63u;
    const unsigned int rest = g >> 6;
    const unsigned int U = rest % ku;
    const unsigned int T = rest / ku;
    const unsigned int n = T * 64u + rr;
    const unsigned int s = U >> 3;
    const unsigned int u = U & 7u;
    const unsigned int b = n * kb + s;
    const uint4* b4 = reinterpret_cast<const uint4*>(staging + src_off) +
        static_cast<unsigned long long>(b) * 11ull;
    unsigned int o[5];
    gr_q5k_unit_from_b4(b4, u, o);
    unsigned int* dst_lane = out_u32 + lane * 5u;
#pragma unroll
    for (int k = 0; k < 5; ++k) dst_lane[k] = o[k];
    if (u == 0u) {
      uint4 h = b4[0];
      const unsigned long long tg = static_cast<unsigned long long>(tile0) + T;
      const unsigned long long oo = (tg * kb + s) * 64ull + rr;
      reinterpret_cast<uint4*>(scales)[oo] = gr_k4_scales(h.y, h.z, h.w);
      reinterpret_cast<unsigned int*>(biases)[oo] = h.x;
    }
  }
  __syncwarp();

  const unsigned long long base_u = static_cast<unsigned long long>(tile0) * ku * 64ull + g0;
  if (cnt == 32u) {
    uint4* w4 = reinterpret_cast<uint4*>(weight + base_u * 5ull);
    for (unsigned int i = lane; i < 40u; i += 32u) {
      w4[i] = s_out[warp][i];
    }
  } else {
    unsigned int* w = weight + base_u * 5ull;
    for (unsigned int i = lane; i < cnt * 5u; i += 32u) {
      w[i] = out_u32[i];
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------- Q5_1 RowMajor
// 24-byte blocks (2-byte aligned): d @0, m @2, qh @4, qs @8 (16 B).
// Grid: ceil(nb / 256), Block: 256 (8 warps x 32 blocks).
extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q5_1(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned short* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int row0) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int b0 = (blockIdx.x * 8u + warp) * 32u;
  if (b0 >= nb) return;
  const unsigned int cnt = (nb - b0 < 32u) ? (nb - b0) : 32u;

  __shared__ uint4 s_in[8][48];   // 32 * 24 B = 768 B = 48 uint4
  __shared__ uint4 s_out[8][40];  // 32 * 20 B = 640 B = 40 uint4

  const unsigned char* src_b = staging + src_off + static_cast<unsigned long long>(b0) * 24ull;
  const unsigned long long align = reinterpret_cast<unsigned long long>(src_b);
  if ((align & 15u) == 0u && cnt == 32u) {
    const uint4* src4 = reinterpret_cast<const uint4*>(src_b);
    for (unsigned int i = lane; i < 48u; i += 32u) {
      s_in[warp][i] = src4[i];
    }
  } else if ((align & 3u) == 0u) {
    const unsigned int* src32 = reinterpret_cast<const unsigned int*>(src_b);
    unsigned int* dst32 = reinterpret_cast<unsigned int*>(s_in[warp]);
    const unsigned int n32 = cnt * 6u;
    for (unsigned int i = lane; i < n32; i += 32u) {
      dst32[i] = src32[i];
    }
  } else {
    const unsigned short* src16 = reinterpret_cast<const unsigned short*>(src_b);
    unsigned short* dst16 = reinterpret_cast<unsigned short*>(s_in[warp]);
    const unsigned int n16 = cnt * 12u;
    for (unsigned int i = lane; i < n16; i += 32u) {
      dst16[i] = src16[i];
    }
  }
  __syncwarp();

  const unsigned long long gb0 = static_cast<unsigned long long>(row0) * kb + b0;
  unsigned int* out32 = reinterpret_cast<unsigned int*>(s_out[warp]);
  if (lane < cnt) {
    const unsigned int* blk = reinterpret_cast<const unsigned int*>(s_in[warp]) + lane * 6u;
    const unsigned int dm = blk[0];
    const unsigned int qh = blk[1];
    const unsigned int q0 = blk[2];
    const unsigned int q1 = blk[3];
    const unsigned int q2 = blk[4];
    const unsigned int q3 = blk[5];
    unsigned long long x0 = gr_pack5_8(gr_nib8(q0, q1), qh);
    unsigned long long x1 = gr_pack5_8(gr_nib8(q2, q3), qh >> 8);
    unsigned long long x2 = gr_pack5_8(gr_nib8(q0 >> 4, q1 >> 4), qh >> 16);
    unsigned long long x3 = gr_pack5_8(gr_nib8(q2 >> 4, q3 >> 4), qh >> 24);
    unsigned int* w = out32 + lane * 5u;
    w[0] = static_cast<unsigned int>(x0);
    w[1] = static_cast<unsigned int>(x0 >> 32) | static_cast<unsigned int>(x1 << 8);
    w[2] = static_cast<unsigned int>(x1 >> 24) | static_cast<unsigned int>(x2 << 16);
    w[3] = static_cast<unsigned int>(x2 >> 16) | static_cast<unsigned int>(x3 << 24);
    w[4] = static_cast<unsigned int>(x3 >> 8);
    scales[gb0 + lane] = static_cast<unsigned short>(dm & 0xFFFFu);
    biases[gb0 + lane] = static_cast<unsigned short>(dm >> 16);
  }
  __syncwarp();

  unsigned int* w_dst = weight + gb0 * 5ull;
  if (cnt == 32u && (reinterpret_cast<unsigned long long>(w_dst) & 15u) == 0u) {
    uint4* w4 = reinterpret_cast<uint4*>(w_dst);
    for (unsigned int i = lane; i < 40u; i += 32u) {
      w4[i] = s_out[warp][i];
    }
  } else {
    const unsigned int n32 = cnt * 5u;
    for (unsigned int i = lane; i < n32; i += 32u) {
      w_dst[i] = out32[i];
    }
  }
}

// ---------------------------------------------------------------- Q8_0 / Q8_0_bias
// 34-byte blocks (2-byte aligned): d @0, qs @2 (32 int8).
// Grid: ceil(nb / 256), Block: 256 (8 warps x 32 blocks).
template <bool HAS_BIAS>
__device__ __forceinline__ void gr_q8_0_impl(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned short* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int row0) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int b0 = (blockIdx.x * 8u + warp) * 32u;
  if (b0 >= nb) return;
  const unsigned int cnt = (nb - b0 < 32u) ? (nb - b0) : 32u;

  __shared__ uint4 s_in[8][68];   // 32 * 34 B = 1088 B = 68 uint4
  __shared__ uint4 s_out[8][64];  // 32 * 32 B = 1024 B = 64 uint4

  const unsigned char* src_b = staging + src_off + static_cast<unsigned long long>(b0) * 34ull;
  const unsigned long long align = reinterpret_cast<unsigned long long>(src_b);
  if ((align & 15u) == 0u && cnt == 32u) {
    const uint4* src4 = reinterpret_cast<const uint4*>(src_b);
    for (unsigned int i = lane; i < 68u; i += 32u) {
      s_in[warp][i] = src4[i];
    }
  } else if ((align & 3u) == 0u && ((cnt & 1u) == 0u)) {
    const unsigned int* src32 = reinterpret_cast<const unsigned int*>(src_b);
    unsigned int* dst32 = reinterpret_cast<unsigned int*>(s_in[warp]);
    const unsigned int n32 = (cnt * 34u) >> 2;
    for (unsigned int i = lane; i < n32; i += 32u) {
      dst32[i] = src32[i];
    }
  } else {
    const unsigned short* src16 = reinterpret_cast<const unsigned short*>(src_b);
    unsigned short* dst16 = reinterpret_cast<unsigned short*>(s_in[warp]);
    const unsigned int n16 = cnt * 17u;
    for (unsigned int i = lane; i < n16; i += 32u) {
      dst16[i] = src16[i];
    }
  }
  __syncwarp();

  const unsigned long long gb0 = static_cast<unsigned long long>(row0) * kb + b0;
  if (lane < cnt) {
    const unsigned short* blk16 =
        reinterpret_cast<const unsigned short*>(s_in[warp]) + lane * 17u;
    const unsigned short d_bits = blk16[0];
    unsigned int q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      unsigned int lo = blk16[1 + 2 * k];
      unsigned int hi = blk16[2 + 2 * k];
      q[k] = (lo | (hi << 16)) ^ 0x80808080u;
    }
    s_out[warp][lane * 2u + 0u] = make_uint4(q[0], q[1], q[2], q[3]);
    s_out[warp][lane * 2u + 1u] = make_uint4(q[4], q[5], q[6], q[7]);
    scales[gb0 + lane] = d_bits;
    if constexpr (HAS_BIAS) {
      biases[gb0 + lane] = gr_q8_0_derived_bias(d_bits);
    }
  }
  __syncwarp();

  uint4* w4 = reinterpret_cast<uint4*>(weight) + gb0 * 2ull;
  const unsigned int n_out = cnt * 2u;
  for (unsigned int i = lane; i < n_out; i += 32u) {
    w4[i] = s_out[warp][i];
  }
}

extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q8_0(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned short* __restrict__ scales,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int row0) {
  gr_q8_0_impl<false>(staging, weight, scales, nullptr, src_off, nb, kb, row0);
}

extern "C" __global__ __launch_bounds__(256) void mlx_node_gguf_repack_q8_0_bias(
    const unsigned char* __restrict__ staging,
    unsigned int* __restrict__ weight,
    unsigned short* __restrict__ scales,
    unsigned short* __restrict__ biases,
    unsigned long long src_off,
    unsigned int nb,
    unsigned int kb,
    unsigned int row0) {
  gr_q8_0_impl<true>(staging, weight, scales, biases, src_off, nb, kb, row0);
}
)cuda"
