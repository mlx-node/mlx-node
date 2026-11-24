R"cuda(
// Coalesced CUDA GGUF Q6_K repack kernels (RowMajor + Tiled64) matching
// gguf_repack.metal and gguf_kquant.rs bit-for-bit.
//
// block_q6_K (210 B, 2-byte aligned):
//   ql[128] @ 0..128, qh[64] @ 128..192, scales[16] (int8) @ 192..208,
//   d (f16 raw u16) @ 208..210.

// Pack 4 bytes (each in 0..63) into a 24-bit LSB-first stream.
__device__ __forceinline__ unsigned int gr_pack6x4(unsigned int c) {
  unsigned int x = (c & 0x003F003Fu) | ((c & 0x3F003F00u) >> 2);
  return (x & 0xFFFu) | ((x >> 4) & 0xFFF000u);
}

// Read N uint32 words starting at 2-byte aligned byte offset `off` from
// 4-byte aligned base `r`. Only reads `r[i + N]` when `(off & 2u) != 0`;
// for Q6_K (210 B), ql/qh/scales end at <= B + 208 so every read stays
// strictly within [r, B + 210) and never over-reads past the band tail.
template <int N>
__device__ __forceinline__ void gr_ldu(
    const unsigned int* __restrict__ r, unsigned int off, unsigned int* o) {
  const unsigned int i = off >> 2;
  const bool odd = (off & 2u) != 0u;
  unsigned int w[N + 1];
#pragma unroll
  for (int k = 0; k < N; ++k) {
    w[k] = r[i + k];
  }
  w[N] = odd ? r[i + N] : 0u;
#pragma unroll
  for (int k = 0; k < N; ++k) {
    o[k] = odd ? ((w[k] >> 16) | (w[k + 1] << 16)) : w[k];
  }
}

__device__ __forceinline__ void gr_q6k_half_unit(
    const unsigned int* __restrict__ r,
    unsigned int q_off,
    unsigned int h_off,
    unsigned int ns,
    unsigned int hs,
    unsigned int* o) {
  unsigned int q[4], h[4], g[4];
  gr_ldu<4>(r, q_off, q);
  gr_ldu<4>(r, h_off, h);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    g[k] = gr_pack6x4(((q[k] >> ns) & 0x0F0F0F0Fu) |
                      (((h[k] >> hs) & 0x03030303u) << 4));
  }
  o[0] = g[0] | (g[1] << 24);
  o[1] = (g[1] >> 8) | (g[2] << 16);
  o[2] = (g[2] >> 16) | (g[3] << 8);
}

// Q6_K 32-value unit `u` (0..7) -> 6 packed uint32 words (192 bits).
__device__ __forceinline__ void gr_q6k_unit(
    const unsigned int* __restrict__ r,
    unsigned int B,
    unsigned int u,
    unsigned int* o) {
  const unsigned int hf = u >> 2;
  const unsigned int gi = u & 3u;
  const unsigned int q_base = B + hf * 64u + (gi & 1u) * 32u;
  const unsigned int h_base = B + 128u + hf * 32u;
  const unsigned int ns = (gi >> 1) * 4u;
  const unsigned int hs = gi * 2u;
  gr_q6k_half_unit(r, q_base, h_base, ns, hs, o);
  gr_q6k_half_unit(r, q_base + 16u, h_base + 16u, ns, hs, o + 3);
}

__device__ __forceinline__ uint4 gr_q6k_scales(
    const unsigned int* __restrict__ r, unsigned int B) {
  unsigned int s[4];
  gr_ldu<4>(r, B + 192u, s);
  return make_uint4(s[0], s[1], s[2], s[3]);
}

// ---------------------------------------------------------------- Q6_K RowMajor
// 210-byte blocks -> 192-byte weight (48 u32 = 12 uint4), 16-byte int8 scales,
// 2-byte f16 d (biases).
// Grid: ceil(nb / 32), Block: 256 (8 warps x 4 blocks, 1 lane per 32-val unit).
extern "C" __global__ __launch_bounds__(256, 2) void mlx_node_gguf_repack_q6k(
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

  __shared__ uint2 s_in[8][105];  // 4 * 210 B = 840 B = 105 uint2
  __shared__ uint4 s_out[8][48];  // 4 * 192 B = 768 B = 48 uint4

  // Exact-length coalesced staging load: never reads before src_off or past
  // src_off + nb * 210.
  const unsigned char* src_b =
      staging + src_off + static_cast<unsigned long long>(b0) * 210ull;
  const unsigned long long align = reinterpret_cast<unsigned long long>(src_b);
  if ((align & 7u) == 0u && cnt == 4u) {
    const uint2* src2 = reinterpret_cast<const uint2*>(src_b);
    for (unsigned int i = lane; i < 105u; i += 32u) {
      s_in[warp][i] = src2[i];
    }
  } else if ((align & 3u) == 0u && ((cnt & 1u) == 0u)) {
    const unsigned int* src32 = reinterpret_cast<const unsigned int*>(src_b);
    unsigned int* dst32 = reinterpret_cast<unsigned int*>(s_in[warp]);
    const unsigned int n32 = (cnt * 210u) >> 2;
    for (unsigned int i = lane; i < n32; i += 32u) {
      dst32[i] = src32[i];
    }
  } else {
    const unsigned short* src16 = reinterpret_cast<const unsigned short*>(src_b);
    unsigned short* dst16 = reinterpret_cast<unsigned short*>(s_in[warp]);
    const unsigned int n16 = cnt * 105u;
    for (unsigned int i = lane; i < n16; i += 32u) {
      dst16[i] = src16[i];
    }
  }
  __syncwarp();

  const unsigned int lb = lane >> 3;
  const unsigned int u = lane & 7u;
  unsigned int* out_u32 = reinterpret_cast<unsigned int*>(s_out[warp]);
  if (lb < cnt) {
    const unsigned int* r_smem = reinterpret_cast<const unsigned int*>(s_in[warp]);
    const unsigned int B = lb * 210u;
    unsigned int o[6];
    gr_q6k_unit(r_smem, B, u, o);
    unsigned int* dst_lane = out_u32 + lane * 6u;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      dst_lane[k] = o[k];
    }
    if (u == 0u) {
      const unsigned long long gb =
          static_cast<unsigned long long>(row0) * kb + b0 + lb;
      reinterpret_cast<uint4*>(scales)[gb] = gr_q6k_scales(r_smem, B);
      biases[gb] = reinterpret_cast<const unsigned short*>(s_in[warp])
                       [lb * 105u + 104u];
    }
  }
  __syncwarp();

  const unsigned long long gb0 = static_cast<unsigned long long>(row0) * kb + b0;
  uint4* w4 = reinterpret_cast<uint4*>(weight) + gb0 * 12ull;
  const unsigned int n_out_u4 = cnt * 12u;
  for (unsigned int i = lane; i < n_out_u4; i += 32u) {
    w4[i] = s_out[warp][i];
  }
}

// ---------------------------------------------------------------- Q6_K Tiled64
// Grid: ceil(nb * 8 / 256), Block: 256. Each warp coalesces 32 units = 48 uint4.
extern "C" __global__ __launch_bounds__(256, 2) void mlx_node_gguf_repack_q6k_t64(
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

  __shared__ uint4 s_out[8][48];  // 32 * 6 u32 = 192 u32 = 48 uint4
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
    const unsigned int* r =
        reinterpret_cast<const unsigned int*>(staging + (src_off & ~3ull));
    const unsigned int B = static_cast<unsigned int>(src_off & 3ull) + b * 210u;

    unsigned int o[6];
    gr_q6k_unit(r, B, u, o);
    unsigned int* dst_lane = out_u32 + lane * 6u;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      dst_lane[k] = o[k];
    }
    if (u == 0u) {
      const unsigned long long tg = static_cast<unsigned long long>(tile0) + T;
      const unsigned long long oo = (tg * kb + s) * 64ull + rr;
      reinterpret_cast<uint4*>(scales)[oo] = gr_q6k_scales(r, B);
      biases[oo] = reinterpret_cast<const unsigned short*>(r)[(B + 208u) >> 1];
    }
  }
  __syncwarp();

  const unsigned long long base_u =
      static_cast<unsigned long long>(tile0) * ku * 64ull + g0;
  if (cnt == 32u) {
    uint4* w4 = reinterpret_cast<uint4*>(weight + base_u * 6ull);
    for (unsigned int i = lane; i < 48u; i += 32u) {
      w4[i] = s_out[warp][i];
    }
  } else {
    unsigned int* w = weight + base_u * 6ull;
    const unsigned int n32 = cnt * 6u;
    for (unsigned int i = lane; i < n32; i += 32u) {
      w[i] = out_u32[i];
    }
  }
}
)cuda"
