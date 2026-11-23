R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Quantize BF16 activations [M, K] into K-group-major INT8 layout for KQuantPrefill
// with super-block activation group size GS = 256 (matching GGUF 256-elem super-blocks):
//   xq:       [K/64, M_pad, 64] int8   (each 128-row M-tile is 8192 contiguous bytes)
//   x_scales: [K/256, M_pad]    float  (amax / 127 per 256-elem super-block)
//   x_qsums:  [K/256, M_pad, 8] int16  (exact sum of 32 int8s per sub-block, for Q4_K/Q5_K min)
// Grid: dim3((M_pad * (K / 256) + 7) / 8), Block: dim3(256) (1 warp = 32T per 256-elem group).
extern "C" __global__ __launch_bounds__(256) void mlx_node_kquant_quantize_prefill(
    const nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ xq,
    float* __restrict__ x_scales,
    short* __restrict__ x_qsums,
    int M, int M_pad, int K) {
  const int units = K >> 8;
  const long long total_groups = (long long)M_pad * units;
  const long long group = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (group >= total_groups) {
    return;
  }
  const int unit = (int)(group / M_pad);
  const int row = (int)(group - (long long)unit * M_pad);
  const int s = lane >> 2;    // 0..7 (32-elem sub-block)
  const int sub = lane & 3;   // 0..3 (8 elems per thread)
  const size_t base_xq =
      ((size_t)(unit * 4 + (s >> 1)) * M_pad + row) * 64 + (s & 1) * 32 + sub * 8;
  const size_t qs_off = ((size_t)unit * M_pad + row) * 8 + s;
  if (row >= M) {
    *reinterpret_cast<uint2*>(xq + base_xq) = make_uint2(0u, 0u);
    if (lane == 0) {
      x_scales[group] = 0.0f;
    }
    if (sub == 0) {
      x_qsums[qs_off] = 0;
    }
    return;
  }
  const nv_bfloat16* x_ptr = x + (size_t)row * K + (size_t)unit * 256 + lane * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(x_ptr);
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float v[8];
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(pairs[k]);
    v[2 * k] = f.x;
    v[2 * k + 1] = f.y;
    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
#pragma unroll
  for (int shift = 1; shift < 32; shift <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift, 32));
  }
  const bool valid = (amax > 0.0f) && isfinite(amax);
  const float inv = valid ? (127.0f / amax) : 0.0f;
  int q[8];
  int qsum = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    q[k] = valid ? static_cast<int>(rintf(v[k] * inv)) : 0;
    qsum += q[k];
  }
#pragma unroll
  for (int shift = 1; shift < 4; shift <<= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, shift, 4);
  }
  uint2 packed;
  packed.x = static_cast<unsigned int>(q[0] & 0xff) |
      (static_cast<unsigned int>(q[1] & 0xff) << 8) |
      (static_cast<unsigned int>(q[2] & 0xff) << 16) |
      (static_cast<unsigned int>(q[3] & 0xff) << 24);
  packed.y = static_cast<unsigned int>(q[4] & 0xff) |
      (static_cast<unsigned int>(q[5] & 0xff) << 8) |
      (static_cast<unsigned int>(q[6] & 0xff) << 16) |
      (static_cast<unsigned int>(q[7] & 0xff) << 24);
  *reinterpret_cast<uint2*>(xq + base_xq) = packed;
  if (lane == 0) {
    x_scales[group] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
  }
  if (sub == 0) {
    x_qsums[qs_off] = static_cast<short>(qsum);
  }
}

__device__ __forceinline__ void kqp_cp_async16(void* smem_ptr, const void* glob_ptr) {
  ptx_cp_async16_cg(smem_ptr, glob_ptr);
}

__device__ __forceinline__ void kqp_cp_commit() {
  ptx_cp_commit();
}

template <int N_WAIT>
__device__ __forceinline__ void kqp_cp_wait() {
  ptx_cp_wait<N_WAIT>();
}

__device__ __forceinline__ void kqp_ldmatrix4(
    unsigned int& a0, unsigned int& a1, unsigned int& a2, unsigned int& a3,
    const void* smem_ptr) {
  ptx_ldmatrix_x4(a0, a1, a2, a3, smem_ptr);
}

template <int mode>
__device__ __forceinline__ unsigned int kqp_unpack_four(unsigned int u) {
  if constexpr (mode == 0) {
    return (u & 0x0000000fu) | ((u & 0x000000f0u) << 4) |
        ((u & 0x00000f00u) << 8) | ((u & 0x0000f000u) << 12);
  } else if constexpr (mode == 3 || mode == 5) {
    const unsigned int low = __byte_perm(0xbfad9881u, 0xf6eaddcfu, u);
    const unsigned int high = __byte_perm(0x26190d01u, 0x71594535u, u);
    return __byte_perm(low, high, 0x3210u | ((u & 0x8888u) >> 1));
  } else if constexpr (mode == 1) {
    return (u & 0x0000001fu) | ((u & 0x000003e0u) << 3) |
        ((u & 0x00007c00u) << 6) | ((u & 0x000f8000u) << 9);
  } else if constexpr (mode == 2) {
    const unsigned int v = (u & 0x0000003fu) | ((u & 0x00000fc0u) << 2) |
        ((u & 0x0003f000u) << 4) | ((u & 0x00fc0000u) << 6);
    const unsigned int neg = (~v) & 0x20202020u;
    return (v & 0x1f1f1f1fu) | neg | (neg << 1) | (neg << 2);
  } else {
    const unsigned int v = (u & 0x00000007u) | ((u & 0x00000038u) << 5) |
        ((u & 0x000001c0u) << 10) | ((u & 0x00000e00u) << 15);
    const unsigned int neg = (~v) & 0x04040404u;
    return (v & 0x03030303u) | (neg * 0x3fu);
  }
}

template <int mode>
__device__ __forceinline__ void kqp_load_pair(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  if constexpr (mode == 0 || mode == 3 || mode == 5) {
    const unsigned short* seg16 = reinterpret_cast<const unsigned short*>(seg);
    out0 = kqp_unpack_four<mode>(seg16[t]);
    out1 = kqp_unpack_four<mode>(seg16[4 + t]);
  } else if constexpr (mode == 1) {
    const int bit0 = t * 20;
    const int w0 = bit0 >> 5;
    const int sh0 = bit0 & 31;
    const unsigned int u0 = __funnelshift_r(seg[w0], seg[w0 + 1], sh0);
    const int bit1 = 80 + t * 20;
    const int w1 = bit1 >> 5;
    const int sh1 = bit1 & 31;
    const unsigned int u1 =
        (w1 < 4) ? __funnelshift_r(seg[w1], seg[w1 + 1], sh1) : (seg[4] >> sh1);
    out0 = kqp_unpack_four<1>(u0);
    out1 = kqp_unpack_four<1>(u1);
  }
}

template <int mode>
__device__ __forceinline__ unsigned int kqp_load_k16(
    const unsigned int* seg, int kh, int t) {
  if constexpr (mode == 2) {
    const int bit = kh * 96 + t * 24;
    const int wi = bit >> 5;
    const int sh = bit & 31;
    const unsigned int u =
        (wi < 5) ? __funnelshift_r(seg[wi], seg[wi + 1], sh) : (seg[5] >> sh);
    return kqp_unpack_four<2>(u);
  } else {
    const int bit = kh * 48 + t * 12;
    const int wi = bit >> 5;
    const int sh = bit & 31;
    const unsigned int u =
        (wi < 2) ? __funnelshift_r(seg[wi], seg[wi + 1], sh) : (seg[2] >> sh);
    return kqp_unpack_four<4>(u);
  }
}

__device__ __forceinline__ void kqp_mma_k16(
    unsigned int a0, unsigned int a1, unsigned int b,
    int& c0, int& c1, int& c2, int& c3) {
  ptx_mma_m16n8k16_s8(a0, a1, b, c0, c1, c2, c3);
}

__device__ __forceinline__ void kqp_mma(
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    int& c0, int& c1, int& c2, int& c3) {
  ptx_mma_m16n8k32_s8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
}
)cuda"
R"cuda(
// High-throughput 128(M) x 64(N) x 64(K/stage) W4A8 INT8-MMA prefill GEMM directly
// on Tiled64 GGUF K-quant weights (2 CTAs/SM, 16 warps/SM, <= 25 KB SMEM/CTA):
//   - Single Tiled64 weight tile per CTA (N is a multiple of 64)
//   - 8 warps (2 M-warps x 4 N-warps): each warp owns 64(M) x 16(N) (MT=4, NT=2)
//   - Activations as A via ldmatrix.x4 (0 bank conflicts via 4-way XOR swizzle)
//   - Weights as B unpacked once and reused across 4 M-tiles (paired 2-MMA scheduling)
//   - INT32 sub-block scaling accumulated over full K=256 super-block (1 FP32 rescale/256K)
//   - Exact Q4_K/Q5_K min dot product computed once per K=256 unit (zero m_acc registers)
//   - Modes: 0=Q4_K, 1=Q5_K, 2=Q6_K, 3=IQ4_XS, 4=Q3_K, 5=IQ4_NL.
template <int mode>
__global__ __launch_bounds__(256, 2) void mlx_node_kquant_prefill_t64(
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    const unsigned short* __restrict__ biases,
    const unsigned char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const short* __restrict__ x_qsums,
    nv_bfloat16* __restrict__ out,
    int M, int M_pad, int N, int K) {
  constexpr int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
  constexpr int TILE_V4 = 16 * bits; // 64 rows * bits * 4B / 16B (48, 64, 80, 96)
  constexpr int sc_unit_bytes = mode == 3 ? 8 : 16;
  constexpr int b_unit_entries = (mode == 0 || mode == 1) ? 2 : 1;
  constexpr bool HAS_MIN = (mode == 0 || mode == 1);
  constexpr int STAGES = 2;

  const int num_m_tiles = M_pad >> 7;
  const int num_n_tiles = N >> 6; // N % 64 == 0
  constexpr int BAND_M = 4;
  const int linear_id = blockIdx.x;
  const int tiles_per_band = BAND_M * num_n_tiles;
  const int band = linear_id / tiles_per_band;
  const int in_band = linear_id - band * tiles_per_band;
  const int band_m0 = band * BAND_M;
  const int band_rows = min(BAND_M, num_m_tiles - band_m0);
  const int n_tile = in_band / band_rows;
  const int m_tile = band_m0 + (in_band - n_tile * band_rows);
  if (n_tile >= num_n_tiles || m_tile >= num_m_tiles) {
    return;
  }

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp >> 2; // 0..1 -> 64 prompt rows (MT = 4)
  const int wn = warp & 3;  // 0..3 -> 16 output cols (NT = 2)
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  const int m0 = m_tile * 128;
  const int n0 = n_tile * 64;
  const int units = K >> 8;
  const int units64 = K >> 6;
  const int units32 = K >> 5;

  struct StageSmem {
    uint4 s_x[128][4];      // 8192 B (XOR swizzled by r & 3)
    uint4 s_w[2][TILE_V4];  // 1536..3072 B (two consecutive K=32 segments)
  };
  __shared__ StageSmem smem[STAGES];
  __shared__ unsigned char s_sc[mode == 5 ? 1 : 2][mode == 5 ? 1 : (64 * 16)];
  __shared__ float s_dw[mode == 5 ? 1 : 2][mode == 5 ? 1 : 64];
  __shared__ float s_dmw[HAS_MIN ? 2 : 1][HAS_MIN ? 64 : 1];

  auto load_stage = [&](int st, int g64) {
#pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
      const int idx = tid + iter * 256;
      const int r = idx >> 2;
      const int ch = idx & 3;
      const uint4* xp = reinterpret_cast<const uint4*>(
          xq + ((size_t)g64 * M_pad + m0 + r) * 64);
      kqp_cp_async16(&smem[st].s_x[r][ch ^ (r & 3)], xp + ch);
    }
    if (tid < TILE_V4) {
      const uint4* w0 = reinterpret_cast<const uint4*>(
          w + ((size_t)n_tile * units32 + g64 * 2) * 64 * bits);
      kqp_cp_async16(&smem[st].s_w[0][tid], w0 + tid);
    } else if (tid < 2 * TILE_V4) {
      const int loc = tid - TILE_V4;
      const uint4* w1 = reinterpret_cast<const uint4*>(
          w + ((size_t)n_tile * units32 + g64 * 2 + 1) * 64 * bits);
      kqp_cp_async16(&smem[st].s_w[1][loc], w1 + loc);
    }
    kqp_cp_commit();
  };

  auto load_unit_meta = [&](int u_buf, int unit) {
    if constexpr (mode != 5) {
      if (tid < 64) {
        const unsigned char* scp =
            scales + (((size_t)n_tile * units + unit) * 64 + tid) * sc_unit_bytes;
        if constexpr (sc_unit_bytes == 16) {
          *reinterpret_cast<uint4*>(&s_sc[u_buf][tid * 16]) =
              *reinterpret_cast<const uint4*>(scp);
        } else {
          *reinterpret_cast<uint2*>(&s_sc[u_buf][tid * 8]) =
              *reinterpret_cast<const uint2*>(scp);
        }
        const size_t bi =
            (((size_t)n_tile * units + unit) * 64 + tid) * b_unit_entries;
        s_dw[u_buf][tid] = __half2float(__ushort_as_half(biases[bi]));
        if constexpr (HAS_MIN) {
          s_dmw[u_buf][tid] = -__half2float(__ushort_as_half(biases[bi + 1]));
        }
      }
    }
  };

  float acc[4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    }
  }

  if (units64 > 0) {
    load_stage(0, 0);
  }
  load_unit_meta(0, 0);

  const int ld_r_base = wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int ld_ch_base = lane >> 4;

  for (int unit = 0; unit < units; ++unit) {
    const int ub = unit & 1;
    int w_acc[mode == 5 ? 1 : 4][mode == 5 ? 1 : 2][4];
    if constexpr (mode != 5) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          w_acc[i][j][0] = w_acc[i][j][1] = w_acc[i][j][2] = w_acc[i][j][3] = 0;
        }
      }
    }
    float ra_nl[mode == 5 ? 4 : 1][mode == 5 ? 2 : 1];
    if constexpr (mode == 5) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r0 = m0 + wm * 64 + i * 16 + g;
        ra_nl[i][0] = x_scales[(size_t)unit * M_pad + r0];
        ra_nl[i][1] = x_scales[(size_t)unit * M_pad + r0 + 8];
      }
    }

#pragma unroll 1
    for (int p = 0; p < 4; ++p) {
      const int g64 = unit * 4 + p;
      const int cur = g64 & 1;
      const int next_g64 = g64 + 1;

      kqp_cp_wait<0>();
      __syncthreads();
      if (next_g64 < units64) {
        load_stage(next_g64 & 1, next_g64);
      }
      if (p == 3 && unit + 1 < units) {
        load_unit_meta((unit + 1) & 1, unit + 1);
      }

#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s = p * 2 + h;
        unsigned int a[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = ld_r_base + i * 16;
          const int ch = h * 2 + ld_ch_base;
          kqp_ldmatrix4(
              a[i][0], a[i][1], a[i][2], a[i][3],
              &smem[cur].s_x[r][ch ^ (r & 3)]);
        }
        const unsigned int* w_smem =
            reinterpret_cast<const unsigned int*>(smem[cur].s_w[h]);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int b_col = wn * 16 + j * 8 + g;
          const unsigned int* seg = w_smem + b_col * bits;
          const int c0_loc = wn * 16 + j * 8 + 2 * t;
          const int c1_loc = c0_loc + 1;

          if constexpr (mode == 2 || mode == 4) {
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
              const int sc0 = static_cast<signed char>(s_sc[ub][c0_loc * 16 + s * 2 + kh]);
              const int sc1 = static_cast<signed char>(s_sc[ub][c1_loc * 16 + s * 2 + kh]);
              const unsigned int b = kqp_load_k16<mode>(seg, kh, t);
#pragma unroll
              for (int ib = 0; ib < 4; ib += 2) {
                int c0[4] = {0, 0, 0, 0};
                int c1[4] = {0, 0, 0, 0};
                kqp_mma_k16(a[ib][kh * 2], a[ib][kh * 2 + 1], b, c0[0], c0[1], c0[2], c0[3]);
                kqp_mma_k16(a[ib + 1][kh * 2], a[ib + 1][kh * 2 + 1], b, c1[0], c1[1], c1[2], c1[3]);
                w_acc[ib][j][0] += sc0 * c0[0];
                w_acc[ib][j][1] += sc1 * c0[1];
                w_acc[ib][j][2] += sc0 * c0[2];
                w_acc[ib][j][3] += sc1 * c0[3];
                w_acc[ib + 1][j][0] += sc0 * c1[0];
                w_acc[ib + 1][j][1] += sc1 * c1[1];
                w_acc[ib + 1][j][2] += sc0 * c1[2];
                w_acc[ib + 1][j][3] += sc1 * c1[3];
              }
            }
          } else if constexpr (mode == 5) {
            unsigned int b0, b1;
            kqp_load_pair<5>(seg, t, b0, b1);
            const int gk = unit * 8 + s;
            const size_t idx0 = ((size_t)n_tile * units32 + gk) * 64 + c0_loc;
            const size_t idx1 = idx0 + 1;
            const float eff0 = __half2float(__ushort_as_half(biases[idx0])) *
                static_cast<float>(static_cast<signed char>(scales[idx0]));
            const float eff1 = __half2float(__ushort_as_half(biases[idx1])) *
                static_cast<float>(static_cast<signed char>(scales[idx1]));
#pragma unroll
            for (int ib = 0; ib < 4; ib += 2) {
              int c0[4] = {0, 0, 0, 0};
              int c1[4] = {0, 0, 0, 0};
              kqp_mma(a[ib][0], a[ib][1], a[ib][2], a[ib][3], b0, b1, c0[0], c0[1], c0[2], c0[3]);
              kqp_mma(a[ib + 1][0], a[ib + 1][1], a[ib + 1][2], a[ib + 1][3], b0, b1, c1[0], c1[1], c1[2], c1[3]);
              acc[ib][j][0] = fmaf(eff0 * ra_nl[ib][0], (float)c0[0], acc[ib][j][0]);
              acc[ib][j][1] = fmaf(eff1 * ra_nl[ib][0], (float)c0[1], acc[ib][j][1]);
              acc[ib][j][2] = fmaf(eff0 * ra_nl[ib][1], (float)c0[2], acc[ib][j][2]);
              acc[ib][j][3] = fmaf(eff1 * ra_nl[ib][1], (float)c0[3], acc[ib][j][3]);
              acc[ib + 1][j][0] = fmaf(eff0 * ra_nl[ib + 1][0], (float)c1[0], acc[ib + 1][j][0]);
              acc[ib + 1][j][1] = fmaf(eff1 * ra_nl[ib + 1][0], (float)c1[1], acc[ib + 1][j][1]);
              acc[ib + 1][j][2] = fmaf(eff0 * ra_nl[ib + 1][1], (float)c1[2], acc[ib + 1][j][2]);
              acc[ib + 1][j][3] = fmaf(eff1 * ra_nl[ib + 1][1], (float)c1[3], acc[ib + 1][j][3]);
            }
          } else {
            unsigned int b0, b1;
            kqp_load_pair<mode>(seg, t, b0, b1);
            int sc0, sc1;
            if constexpr (mode == 3) {
              sc0 = static_cast<signed char>(s_sc[ub][c0_loc * 8 + s]);
              sc1 = static_cast<signed char>(s_sc[ub][c1_loc * 8 + s]);
            } else {
              sc0 = s_sc[ub][c0_loc * 16 + s * 2];
              sc1 = s_sc[ub][c1_loc * 16 + s * 2];
            }
#pragma unroll
            for (int ib = 0; ib < 4; ib += 2) {
              int c0[4] = {0, 0, 0, 0};
              int c1[4] = {0, 0, 0, 0};
              kqp_mma(a[ib][0], a[ib][1], a[ib][2], a[ib][3], b0, b1, c0[0], c0[1], c0[2], c0[3]);
              kqp_mma(a[ib + 1][0], a[ib + 1][1], a[ib + 1][2], a[ib + 1][3], b0, b1, c1[0], c1[1], c1[2], c1[3]);
              w_acc[ib][j][0] += sc0 * c0[0];
              w_acc[ib][j][1] += sc1 * c0[1];
              w_acc[ib][j][2] += sc0 * c0[2];
              w_acc[ib][j][3] += sc1 * c0[3];
              w_acc[ib + 1][j][0] += sc0 * c1[0];
              w_acc[ib + 1][j][1] += sc1 * c1[1];
              w_acc[ib + 1][j][2] += sc0 * c1[2];
              w_acc[ib + 1][j][3] += sc1 * c1[3];
            }
          }
        }
      }
    }

    if constexpr (mode != 5) {
      float ra[4][2];
      uint4 qs_vec[HAS_MIN ? 4 : 1][HAS_MIN ? 2 : 1];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r0 = m0 + wm * 64 + i * 16 + g;
        ra[i][0] = x_scales[(size_t)unit * M_pad + r0];
        ra[i][1] = x_scales[(size_t)unit * M_pad + r0 + 8];
        if constexpr (HAS_MIN) {
          qs_vec[i][0] = *reinterpret_cast<const uint4*>(x_qsums + ((size_t)unit * M_pad + r0) * 8);
          qs_vec[i][1] = *reinterpret_cast<const uint4*>(x_qsums + ((size_t)unit * M_pad + r0 + 8) * 8);
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int c0 = wn * 16 + j * 8 + 2 * t;
        const float dw0 = s_dw[ub][c0];
        const float dw1 = s_dw[ub][c0 + 1];
        if constexpr (HAS_MIN) {
          const float dmw0 = s_dmw[ub][c0];
          const float dmw1 = s_dmw[ub][c0 + 1];
          const unsigned char* sc0p = &s_sc[ub][c0 * 16];
          const unsigned char* sc1p = &s_sc[ub][(c0 + 1) * 16];
          int m0_arr[8], m1_arr[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            m0_arr[s] = sc0p[s * 2 + 1];
            m1_arr[s] = sc1p[s * 2 + 1];
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const short* q0 = reinterpret_cast<const short*>(&qs_vec[i][0]);
            const short* q1 = reinterpret_cast<const short*>(&qs_vec[i][1]);
            int md0 = 0, md1 = 0, md2 = 0, md3 = 0;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
              md0 += m0_arr[s] * (int)q0[s];
              md1 += m1_arr[s] * (int)q0[s];
              md2 += m0_arr[s] * (int)q1[s];
              md3 += m1_arr[s] * (int)q1[s];
            }
            acc[i][j][0] = fmaf(ra[i][0], fmaf(dw0, (float)w_acc[i][j][0], dmw0 * (float)md0), acc[i][j][0]);
            acc[i][j][1] = fmaf(ra[i][0], fmaf(dw1, (float)w_acc[i][j][1], dmw1 * (float)md1), acc[i][j][1]);
            acc[i][j][2] = fmaf(ra[i][1], fmaf(dw0, (float)w_acc[i][j][2], dmw0 * (float)md2), acc[i][j][2]);
            acc[i][j][3] = fmaf(ra[i][1], fmaf(dw1, (float)w_acc[i][j][3], dmw1 * (float)md3), acc[i][j][3]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            acc[i][j][0] = fmaf(dw0 * ra[i][0], (float)w_acc[i][j][0], acc[i][j][0]);
            acc[i][j][1] = fmaf(dw1 * ra[i][0], (float)w_acc[i][j][1], acc[i][j][1]);
            acc[i][j][2] = fmaf(dw0 * ra[i][1], (float)w_acc[i][j][2], acc[i][j][2]);
            acc[i][j][3] = fmaf(dw1 * ra[i][1], (float)w_acc[i][j][3], acc[i][j][3]);
          }
        }
      }
    }
  }
  kqp_cp_wait<0>();

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r0 = m0 + wm * 64 + i * 16 + g;
    const int r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col = n0 + wn * 16 + j * 8 + 2 * t;
      if (col < N) {
        if (r0 < M) {
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)r0 * N + col) =
              __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
        }
        if (r1 < M) {
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)r1 * N + col) =
              __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
      }
    }
  }
}
)cuda"
