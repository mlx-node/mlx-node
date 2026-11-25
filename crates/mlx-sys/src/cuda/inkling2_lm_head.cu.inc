R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Inkling-2-Mini Fused LM Head + Softcap (+ Zero-Logit-DRAM Argmax)
// Supports:
//   - SymQ8 (biases == nullptr) and AffineQ8 (biases != nullptr), group_size=32
//   - Dense BF16 (scales == nullptr)
//   - M in 1..=16, unpadded_vocab <= V_pad (skips padding rows!)
//   - OUT_MODE: 0 = FP32 [M, V], 1 = Int32 [M] argmax, 2 = BF16 [M, V]
// Requires ptx_helpers.cu.inc prepended before this file.
// ============================================================================

__device__ __forceinline__ float ink2_lm_softcap(float x, float cap) {
  if (cap <= 0.0f) {
    return x;
  }
  const float scaled = fminf(fmaxf(x / cap, -20.0f), 20.0f);
  return cap * tanhf(scaled);
}

template <int SB_DTYPE>
__device__ __forceinline__ float ink2_lm_load_sb1(const void* ptr, size_t idx) {
  if constexpr (SB_DTYPE == 0) {
    return reinterpret_cast<const float*>(ptr)[idx];
  } else if constexpr (SB_DTYPE == 1) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  } else {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
  }
}

template <int PAIR, bool SYM>
__device__ __forceinline__ unsigned int ink2_lm_unpack_pair(unsigned int w) {
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

__device__ __forceinline__ float ink2_lm_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

template <bool SYM>
__device__ __forceinline__ void ink2_lm_load_x_g32(
    const uint4* __restrict__ x_row_u4, size_t grp_u4, bool valid_m,
    int t, uint4& xv, float& xsum_c0, float& xsum_c1) {
  if (valid_m) {
    xv = x_row_u4[grp_u4];
  } else {
    xv = make_uint4(0, 0, 0, 0);
  }
  if constexpr (!SYM) {
    float my_sum = valid_m ? ink2_lm_sum_bf16x8(xv) : 0.0f;
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
__device__ __forceinline__ void ink2_lm_mma_g32(
    unsigned int r0_u0, unsigned int r0_u1,
    unsigned int r1_u0, unsigned int r1_u1,
    const uint4& x0, const uint4& x1,
    float d0[4], float d1[4]) {
  d0[0] = d0[1] = d0[2] = d0[3] = 0.0f;
  if constexpr (M_TILES == 2) {
    d1[0] = d1[1] = d1[2] = d1[3] = 0.0f;
  }
  {
    const unsigned int a0 = ink2_lm_unpack_pair<0, SYM>(r0_u0);
    const unsigned int a1 = ink2_lm_unpack_pair<0, SYM>(r1_u0);
    const unsigned int a2 = ink2_lm_unpack_pair<1, SYM>(r0_u0);
    const unsigned int a3 = ink2_lm_unpack_pair<1, SYM>(r1_u0);
    ptx_mma_m16n8k16_bf16_v(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.x, x0.y);
    if constexpr (M_TILES == 2) {
      ptx_mma_m16n8k16_bf16_v(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.x, x1.y);
    }
  }
  {
    const unsigned int a0 = ink2_lm_unpack_pair<0, SYM>(r0_u1);
    const unsigned int a1 = ink2_lm_unpack_pair<0, SYM>(r1_u1);
    const unsigned int a2 = ink2_lm_unpack_pair<1, SYM>(r0_u1);
    const unsigned int a3 = ink2_lm_unpack_pair<1, SYM>(r1_u1);
    ptx_mma_m16n8k16_bf16_v(d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, x0.z, x0.w);
    if constexpr (M_TILES == 2) {
      ptx_mma_m16n8k16_bf16_v(d1[0], d1[1], d1[2], d1[3], a0, a1, a2, a3, x1.z, x1.w);
    }
  }
}

template <bool SYM>
__device__ __forceinline__ void ink2_lm_dot4(
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

__device__ __forceinline__ void ink2_lm_argmax_update(
    float& bv, int& bi, float v, int idx) {
  if (idx >= 0 && (bi < 0 || v > bv || (v == bv && idx < bi))) {
    bv = v;
    bi = idx;
  }
}

// ============================================================================
// 1. Q8_0 Streaming uint4 Kernel for M in 1..=2 (1 warp per vocab row, 8 warps/CTA)
//    OUT_MODE 0/2: Grid dim3((V + 7) / 8), Block dim3(32, 8)
//    OUT_MODE 1:   Grid dim3(num_ctas),    Block dim3(32, 8)
// ============================================================================
template <int SB_DTYPE, int M_CAP, bool SYM, int OUT_MODE>
__global__ __launch_bounds__(256, 4) void ink2_lm_head_q8_u4(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    void* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K, float softcap) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int num_u4 = K >> 4;
  const int groups = K >> 5;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  const int n_start = static_cast<int>(blockIdx.x) * 8 + warp;
  if constexpr (OUT_MODE != 1) {
    if (n_start >= V) {
      return;
    }
  }
  const int n_step = static_cast<int>(gridDim.x) * 8;

  [[maybe_unused]] float best_v[M_CAP];
  [[maybe_unused]] int best_i[M_CAP];
  if constexpr (OUT_MODE == 1) {
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      best_v[m] = -__int_as_float(0x7f800000);
      best_i[m] = -1;
    }
  }

  for (int n = n_start; n < V; n += n_step) {
    const uint4* w_row =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n) * num_u4;
    const size_t sb_row = static_cast<size_t>(n) * groups;

    float acc[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      acc[m] = 0.0f;
    }

    for (int u = lane; u < num_u4; u += 32) {
      const uint4 wv = ptx_ld_cs_u4(w_row + u);
      const int g = u >> 1;
      const float sc = ink2_lm_load_sb1<SB_DTYPE>(scales, sb_row + g);
      [[maybe_unused]] float bi = 0.0f;
      if constexpr (!SYM) {
        bi = ink2_lm_load_sb1<SB_DTYPE>(biases, sb_row + g);
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
          ink2_lm_dot4<SYM>(wv.x, p0[0], p0[1], d, s);
          ink2_lm_dot4<SYM>(wv.y, p0[2], p0[3], d, s);
          ink2_lm_dot4<SYM>(wv.z, p1[0], p1[1], d, s);
          ink2_lm_dot4<SYM>(wv.w, p1[2], p1[3], d, s);
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
          const float sv = ink2_lm_softcap(v, softcap);
          if constexpr (OUT_MODE == 0) {
            reinterpret_cast<float*>(out)[static_cast<size_t>(m) * V + n] = sv;
          } else if constexpr (OUT_MODE == 2) {
            reinterpret_cast<nv_bfloat16*>(out)[static_cast<size_t>(m) * V + n] =
                __float2bfloat16_rn(sv);
          } else {
            ink2_lm_argmax_update(best_v[m], best_i[m], sv, n);
          }
        }
      }
    }
    if constexpr (OUT_MODE != 1) {
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    __shared__ float s_v[M_CAP][8];
    __shared__ int s_i[M_CAP][8];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        s_v[m][warp] = best_v[m];
        s_i[m][warp] = best_i[m];
      }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < M) {
          float v = (lane < 8) ? s_v[m][lane] : -__int_as_float(0x7f800000);
          int idx = (lane < 8) ? s_i[m][lane] : -1;
#pragma unroll
          for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            ink2_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            const size_t slot = static_cast<size_t>(m) * gridDim.x + blockIdx.x;
            reinterpret_cast<float*>(out)[slot] = v;
            partial_idxs[slot] = idx;
          }
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 2. Q8_0 2-Stage cp.async.cg + BF16 Tensor Core MMA for M in 3..=16
//    OUT_MODE 0/2: Grid dim3((V + 15) / 16), Block dim3(32, 8)
//    OUT_MODE 1:   Grid dim3(num_ctas),      Block dim3(32, 8)
// ============================================================================
template <int SB_DTYPE, int M_TILES, bool SYM, int OUT_MODE>
__global__ __launch_bounds__(256, 2) void ink2_lm_head_q8_mma(
    const nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ w,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    void* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K, float softcap) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int num_groups = K >> 5;
  const int num_tiles = num_groups >> 3;  // full 8-group (256-weight) tiles
  const int tail_groups = num_groups & 7;
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 4;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  union SharedBuf {
    struct {
      uint4 w[2][16][16];    // 8192 B (XOR swizzled)
      float sc[2][16][9];    // 1152 B (+1 padded)
      float bi[SYM ? 1 : 2][SYM ? 1 : 16][SYM ? 1 : 9];
    } pipe;
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const int ld_row = tid >> 4;   // 0..15
  const int ld_col = tid & 15;   // 0..15
  const int ld_swz = ld_col ^ ld_row;

  const int sb_is_bias = tid >> 7;
  const int sb_row = (tid >> 3) & 15;
  const int sb_grp = tid & 7;
  [[maybe_unused]] const void* sb_src = SYM ? scales : (sb_is_bias ? biases : scales);

  const bool valid_m0 = (g < M);
  const uint4* x0_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m0 ? g : 0) * x_stride_u4 + t;
  const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  [[maybe_unused]] const uint4* x1_row_u4 = reinterpret_cast<const uint4*>(x) +
      static_cast<size_t>(valid_m1 ? (g + 8) : 0) * x_stride_u4 + t;

  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ (g + 8);

  const int row_start = static_cast<int>(blockIdx.x) * 16;
  if constexpr (OUT_MODE != 1) {
    if (row_start >= V) {
      return;
    }
  }
  const int row_step = static_cast<int>(gridDim.x) * 16;

  [[maybe_unused]] float best_v0 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i0 = -1;
  [[maybe_unused]] float best_v1 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i1 = -1;

  for (int row0 = row_start; row0 < V; row0 += row_step) {
    const int ld_r = (row0 + ld_row < V) ? (row0 + ld_row) : (V - 1);
    const uint4* w_load_ptr =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;
    const int sb_r = (row0 + sb_row < V) ? (row0 + sb_row) : (V - 1);
    const size_t sb_base_idx =
        static_cast<size_t>(sb_r) * static_cast<size_t>(num_groups) + sb_grp;

    float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (num_tiles > 0) {
      ptx_cp_async16_cg(&sh.pipe.w[0][ld_row][ld_swz], w_load_ptr);
      ptx_cp_commit();
      if constexpr (SYM) {
        if (!sb_is_bias) {
          sh.pipe.sc[0][sb_row][sb_grp] =
              ink2_lm_load_sb1<SB_DTYPE>(scales, sb_base_idx);
        }
      } else {
        const float sbv = ink2_lm_load_sb1<SB_DTYPE>(sb_src, sb_base_idx);
        if (sb_is_bias) {
          sh.pipe.bi[0][sb_row][sb_grp] = sbv;
        } else {
          sh.pipe.sc[0][sb_row][sb_grp] = sbv;
        }
      }
      ptx_cp_wait<0>();
      __syncthreads();

      for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile & 1;
        const int nxt = cur ^ 1;
        if (tile + 1 < num_tiles) {
          const size_t nt = static_cast<size_t>(tile + 1);
          ptx_cp_async16_cg(&sh.pipe.w[nxt][ld_row][ld_swz], w_load_ptr + nt * 16);
          ptx_cp_commit();
          if constexpr (SYM) {
            if (!sb_is_bias) {
              sh.pipe.sc[nxt][sb_row][sb_grp] =
                  ink2_lm_load_sb1<SB_DTYPE>(scales, sb_base_idx + nt * 8);
            }
          } else {
            const float sbv = ink2_lm_load_sb1<SB_DTYPE>(sb_src, sb_base_idx + nt * 8);
            if (sb_is_bias) {
              sh.pipe.bi[nxt][sb_row][sb_grp] = sbv;
            } else {
              sh.pipe.sc[nxt][sb_row][sb_grp] = sbv;
            }
          }
        }

        const uint2 w0 =
            reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g][swz_r0])[half_u2];
        const uint2 w1 =
            reinterpret_cast<const uint2*>(&sh.pipe.w[cur][g + 8][swz_r1])[half_u2];

        const int grp_idx = tile * 8 + warp;
        uint4 xv0, xv1;
        float xs00, xs01;
        [[maybe_unused]] float xs10 = 0.0f, xs11 = 0.0f;
        ink2_lm_load_x_g32<SYM>(
            x0_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m0, t, xv0, xs00, xs01);
        if constexpr (M_TILES == 2) {
          ink2_lm_load_x_g32<SYM>(
              x1_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m1, t, xv1, xs10, xs11);
        }

        float d0[4], d1[4];
        ink2_lm_mma_g32<M_TILES, SYM>(w0.x, w0.y, w1.x, w1.y, xv0, xv1, d0, d1);

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

        if (tile + 1 < num_tiles) {
          ptx_cp_wait<0>();
        }
        __syncthreads();
      }
    }

    if (tail_groups > 0 && warp < tail_groups) {
      const int grp_idx = num_tiles * 8 + warp;
      const int r0 = (row0 + g < V) ? (row0 + g) : (V - 1);
      const int r1 = (row0 + g + 8 < V) ? (row0 + g + 8) : (V - 1);
      const size_t w_stride_u2 = static_cast<size_t>(K) >> 3;
      const uint2 w0 = reinterpret_cast<const uint2*>(
          w)[static_cast<size_t>(r0) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];
      const uint2 w1 = reinterpret_cast<const uint2*>(
          w)[static_cast<size_t>(r1) * w_stride_u2 + static_cast<size_t>(grp_idx) * 4 + t];

      uint4 xv0, xv1;
      float xs00, xs01;
      [[maybe_unused]] float xs10 = 0.0f, xs11 = 0.0f;
      ink2_lm_load_x_g32<SYM>(
          x0_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m0, t, xv0, xs00, xs01);
      if constexpr (M_TILES == 2) {
        ink2_lm_load_x_g32<SYM>(
            x1_row_u4, static_cast<size_t>(grp_idx) * 4, valid_m1, t, xv1, xs10, xs11);
      }

      float d0[4], d1[4];
      ink2_lm_mma_g32<M_TILES, SYM>(w0.x, w0.y, w1.x, w1.y, xv0, xv1, d0, d1);

      const float sc0 =
          ink2_lm_load_sb1<SB_DTYPE>(scales, static_cast<size_t>(r0) * num_groups + grp_idx);
      const float sc1 =
          ink2_lm_load_sb1<SB_DTYPE>(scales, static_cast<size_t>(r1) * num_groups + grp_idx);
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
            ink2_lm_load_sb1<SB_DTYPE>(biases, static_cast<size_t>(r0) * num_groups + grp_idx);
        const float bi1 =
            ink2_lm_load_sb1<SB_DTYPE>(biases, static_cast<size_t>(r1) * num_groups + grp_idx);
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

    if constexpr (OUT_MODE == 1) {
      if (warp < M) {
        float v = -__int_as_float(0x7f800000);
        int idx = -1;
        if (lane < 16 && row0 + lane < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][warp][lane];
          }
          v = ink2_lm_softcap(sum, softcap);
          idx = row0 + lane;
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, v, off);
          const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
          ink2_lm_argmax_update(v, idx, ov, oi);
        }
        if (lane == 0) {
          ink2_lm_argmax_update(best_v0, best_i0, v, idx);
        }
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < M) {
          float v = -__int_as_float(0x7f800000);
          int idx = -1;
          if (lane < 16 && row0 + lane < V) {
            float sum = 0.0f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww) {
              sum += sh.reduce[ww][warp + 8][lane];
            }
            v = ink2_lm_softcap(sum, softcap);
            idx = row0 + lane;
          }
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            ink2_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            ink2_lm_argmax_update(best_v1, best_i1, v, idx);
          }
        }
      }
      __syncthreads();
    } else {
      for (int idx = tid; idx < M * 16; idx += 256) {
        const int m = idx >> 4;
        const int n_local = idx & 15;
        const int n_col = row0 + n_local;
        if (n_col < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][m][n_local];
          }
          const float sv = ink2_lm_softcap(sum, softcap);
          if constexpr (OUT_MODE == 0) {
            reinterpret_cast<float*>(out)[static_cast<size_t>(m) * V + n_col] = sv;
          } else {
            reinterpret_cast<nv_bfloat16*>(out)[static_cast<size_t>(m) * V + n_col] =
                __float2bfloat16_rn(sv);
          }
        }
      }
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    if (lane == 0) {
      if (warp < M) {
        const size_t slot = static_cast<size_t>(warp) * gridDim.x + blockIdx.x;
        reinterpret_cast<float*>(out)[slot] = best_v0;
        partial_idxs[slot] = best_i0;
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < M) {
          const size_t slot = static_cast<size_t>(warp + 8) * gridDim.x + blockIdx.x;
          reinterpret_cast<float*>(out)[slot] = best_v1;
          partial_idxs[slot] = best_i1;
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3. BF16 Dense Streaming uint4 Kernel for M in 1..=2 (1 warp/row, 8 warps/CTA)
//    OUT_MODE 0/2: Grid dim3((V + 7) / 8), Block dim3(32, 8)
//    OUT_MODE 1:   Grid dim3(num_ctas),    Block dim3(32, 8)
// ============================================================================
template <int M_CAP, int OUT_MODE>
__global__ __launch_bounds__(256, 4) void ink2_lm_head_bf16_u4(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ w,
    void* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K, float softcap) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int k_vec = K >> 3;  // K / 8 (each uint4 = 8 BF16s)

  const int n_start = static_cast<int>(blockIdx.x) * 8 + warp;
  if constexpr (OUT_MODE != 1) {
    if (n_start >= V) {
      return;
    }
  }
  const int n_step = static_cast<int>(gridDim.x) * 8;

  [[maybe_unused]] float best_v[M_CAP];
  [[maybe_unused]] int best_i[M_CAP];
  if constexpr (OUT_MODE == 1) {
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      best_v[m] = -__int_as_float(0x7f800000);
      best_i[m] = -1;
    }
  }

  for (int n = n_start; n < V; n += n_step) {
    const uint4* w_row = reinterpret_cast<const uint4*>(
        w + static_cast<size_t>(n) * static_cast<size_t>(K));

    float acc[M_CAP];
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      acc[m] = 0.0f;
    }

    for (int kv = lane; kv < k_vec; kv += 32) {
      const uint4 wv = ptx_ld_cs_u4(w_row + kv);
      const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
      const float2 wf0 = __bfloat1622float2(wb[0]);
      const float2 wf1 = __bfloat1622float2(wb[1]);
      const float2 wf2 = __bfloat1622float2(wb[2]);
      const float2 wf3 = __bfloat1622float2(wb[3]);

#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < M) {
          const uint4 xv = reinterpret_cast<const uint4*>(
              x + static_cast<size_t>(m) * static_cast<size_t>(K))[kv];
          const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
          const float2 xf0 = __bfloat1622float2(xb[0]);
          const float2 xf1 = __bfloat1622float2(xb[1]);
          const float2 xf2 = __bfloat1622float2(xb[2]);
          const float2 xf3 = __bfloat1622float2(xb[3]);
          float s = 0.0f;
          s = fmaf(xf0.x, wf0.x, fmaf(xf0.y, wf0.y, s));
          s = fmaf(xf1.x, wf1.x, fmaf(xf1.y, wf1.y, s));
          s = fmaf(xf2.x, wf2.x, fmaf(xf2.y, wf2.y, s));
          s = fmaf(xf3.x, wf3.x, fmaf(xf3.y, wf3.y, s));
          acc[m] += s;
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
          const float sv = ink2_lm_softcap(v, softcap);
          if constexpr (OUT_MODE == 0) {
            reinterpret_cast<float*>(out)[static_cast<size_t>(m) * V + n] = sv;
          } else if constexpr (OUT_MODE == 2) {
            reinterpret_cast<nv_bfloat16*>(out)[static_cast<size_t>(m) * V + n] =
                __float2bfloat16_rn(sv);
          } else {
            ink2_lm_argmax_update(best_v[m], best_i[m], sv, n);
          }
        }
      }
    }
    if constexpr (OUT_MODE != 1) {
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    __shared__ float s_v[M_CAP][8];
    __shared__ int s_i[M_CAP][8];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        s_v[m][warp] = best_v[m];
        s_i[m][warp] = best_i[m];
      }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int m = 0; m < M_CAP; ++m) {
        if (m < M) {
          float v = (lane < 8) ? s_v[m][lane] : -__int_as_float(0x7f800000);
          int idx = (lane < 8) ? s_i[m][lane] : -1;
#pragma unroll
          for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            ink2_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            const size_t slot = static_cast<size_t>(m) * gridDim.x + blockIdx.x;
            reinterpret_cast<float*>(out)[slot] = v;
            partial_idxs[slot] = idx;
          }
        }
      }
    }
  }
}

)cuda"
R"cuda(
// ============================================================================
// 4. BF16 Dense 2-Stage cp.async.cg + Tensor Core MMA for M in 3..=16
//    K_TILE = 128 BF16s (16 uint4s/row x 16 rows = 4096 B/stage)
//    OUT_MODE 0/2: Grid dim3((V + 15) / 16), Block dim3(32, 8)
//    OUT_MODE 1:   Grid dim3(num_ctas),      Block dim3(32, 8)
// ============================================================================
template <int M_TILES, int OUT_MODE>
__global__ __launch_bounds__(256, 2) void ink2_lm_head_bf16_mma(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ w,
    void* __restrict__ out,
    int* __restrict__ partial_idxs,
    int M, int V, int K, float softcap) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int num_tiles = K >> 7;   // full 128-BF16 tiles
  const int tail_k = K & 127;     // multiple of 8
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 3;

  union SharedBuf {
    uint4 w[2][16][16];           // 8192 B (XOR swizzled)
    float reduce[8][M_TILES * 8][17];
  };
  __shared__ SharedBuf sh;

  const int ld_row = tid >> 4;
  const int ld_col = tid & 15;
  const int ld_swz = ld_col ^ ld_row;

  const bool valid_m0 = (g < M);
  [[maybe_unused]] const bool valid_m1 = (M_TILES == 2) && (g + 8 < M);
  const int col_u4 = warp * 2 + (t >> 1);
  const int half_u2 = t & 1;
  const int swz_r0 = col_u4 ^ g;
  const int swz_r1 = col_u4 ^ (g + 8);

  const int row_start = static_cast<int>(blockIdx.x) * 16;
  if constexpr (OUT_MODE != 1) {
    if (row_start >= V) {
      return;
    }
  }
  const int row_step = static_cast<int>(gridDim.x) * 16;

  [[maybe_unused]] float best_v0 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i0 = -1;
  [[maybe_unused]] float best_v1 = -__int_as_float(0x7f800000);
  [[maybe_unused]] int best_i1 = -1;

  for (int row0 = row_start; row0 < V; row0 += row_step) {
    const int ld_r = (row0 + ld_row < V) ? (row0 + ld_row) : (V - 1);
    const uint4* w_load_ptr =
        reinterpret_cast<const uint4*>(w) + static_cast<size_t>(ld_r) * w_stride_u4 + ld_col;

    float total0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float total1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (num_tiles > 0) {
      ptx_cp_async16_cg(&sh.w[0][ld_row][ld_swz], w_load_ptr);
      ptx_cp_commit();
      ptx_cp_wait<0>();
      __syncthreads();

      for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile & 1;
        const int nxt = cur ^ 1;
        if (tile + 1 < num_tiles) {
          ptx_cp_async16_cg(
              &sh.w[nxt][ld_row][ld_swz],
              w_load_ptr + static_cast<size_t>(tile + 1) * 16);
          ptx_cp_commit();
        }

        const uint2 w0 =
            reinterpret_cast<const uint2*>(&sh.w[cur][g][swz_r0])[half_u2];
        const uint2 w1 =
            reinterpret_cast<const uint2*>(&sh.w[cur][g + 8][swz_r1])[half_u2];
        const int k_off = tile * 128 + warp * 16 + t * 4;
        const uint2 xb0 = valid_m0
            ? *reinterpret_cast<const uint2*>(x + static_cast<size_t>(g) * K + k_off)
            : make_uint2(0, 0);
        ptx_mma_m16n8k16_bf16_v(
            total0[0], total0[1], total0[2], total0[3],
            w0.x, w1.x, w0.y, w1.y, xb0.x, xb0.y);
        if constexpr (M_TILES == 2) {
          const uint2 xb1 = valid_m1
              ? *reinterpret_cast<const uint2*>(x + static_cast<size_t>(g + 8) * K + k_off)
              : make_uint2(0, 0);
          ptx_mma_m16n8k16_bf16_v(
              total1[0], total1[1], total1[2], total1[3],
              w0.x, w1.x, w0.y, w1.y, xb1.x, xb1.y);
        }

        if (tile + 1 < num_tiles) {
          ptx_cp_wait<0>();
        }
        __syncthreads();
      }
    }

    if (tail_k > 0) {
      const int k_base = num_tiles * 128 + warp * 16;
      if (k_base < K) {
        const int r0 = (row0 + g < V) ? (row0 + g) : (V - 1);
        const int r1 = (row0 + g + 8 < V) ? (row0 + g + 8) : (V - 1);
        const int k_off = k_base + t * 4;
        const uint2 w0 = (k_off < K)
            ? *reinterpret_cast<const uint2*>(w + static_cast<size_t>(r0) * K + k_off)
            : make_uint2(0, 0);
        const uint2 w1 = (k_off < K)
            ? *reinterpret_cast<const uint2*>(w + static_cast<size_t>(r1) * K + k_off)
            : make_uint2(0, 0);
        const uint2 xb0 = (valid_m0 && k_off < K)
            ? *reinterpret_cast<const uint2*>(x + static_cast<size_t>(g) * K + k_off)
            : make_uint2(0, 0);
        ptx_mma_m16n8k16_bf16_v(
            total0[0], total0[1], total0[2], total0[3],
            w0.x, w1.x, w0.y, w1.y, xb0.x, xb0.y);
        if constexpr (M_TILES == 2) {
          const uint2 xb1 = (valid_m1 && k_off < K)
              ? *reinterpret_cast<const uint2*>(x + static_cast<size_t>(g + 8) * K + k_off)
              : make_uint2(0, 0);
          ptx_mma_m16n8k16_bf16_v(
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

    if constexpr (OUT_MODE == 1) {
      if (warp < M) {
        float v = -__int_as_float(0x7f800000);
        int idx = -1;
        if (lane < 16 && row0 + lane < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][warp][lane];
          }
          v = ink2_lm_softcap(sum, softcap);
          idx = row0 + lane;
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, v, off);
          const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
          ink2_lm_argmax_update(v, idx, ov, oi);
        }
        if (lane == 0) {
          ink2_lm_argmax_update(best_v0, best_i0, v, idx);
        }
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < M) {
          float v = -__int_as_float(0x7f800000);
          int idx = -1;
          if (lane < 16 && row0 + lane < V) {
            float sum = 0.0f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww) {
              sum += sh.reduce[ww][warp + 8][lane];
            }
            v = ink2_lm_softcap(sum, softcap);
            idx = row0 + lane;
          }
#pragma unroll
          for (int off = 8; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            ink2_lm_argmax_update(v, idx, ov, oi);
          }
          if (lane == 0) {
            ink2_lm_argmax_update(best_v1, best_i1, v, idx);
          }
        }
      }
      __syncthreads();
    } else {
      for (int idx = tid; idx < M * 16; idx += 256) {
        const int m = idx >> 4;
        const int n_local = idx & 15;
        const int n_col = row0 + n_local;
        if (n_col < V) {
          float sum = 0.0f;
#pragma unroll
          for (int ww = 0; ww < 8; ++ww) {
            sum += sh.reduce[ww][m][n_local];
          }
          const float sv = ink2_lm_softcap(sum, softcap);
          if constexpr (OUT_MODE == 0) {
            reinterpret_cast<float*>(out)[static_cast<size_t>(m) * V + n_col] = sv;
          } else {
            reinterpret_cast<nv_bfloat16*>(out)[static_cast<size_t>(m) * V + n_col] =
                __float2bfloat16_rn(sv);
          }
        }
      }
      break;
    }
  }

  if constexpr (OUT_MODE == 1) {
    if (lane == 0) {
      if (warp < M) {
        const size_t slot = static_cast<size_t>(warp) * gridDim.x + blockIdx.x;
        reinterpret_cast<float*>(out)[slot] = best_v0;
        partial_idxs[slot] = best_i0;
      }
      if constexpr (M_TILES == 2) {
        if (warp + 8 < M) {
          const size_t slot = static_cast<size_t>(warp + 8) * gridDim.x + blockIdx.x;
          reinterpret_cast<float*>(out)[slot] = best_v1;
          partial_idxs[slot] = best_i1;
        }
      }
    }
  }
}

// ============================================================================
// 5. Stage 2 Argmax Reduction: [M, num_ctas] -> [M] Int32 (smaller-ID tie-break)
//    Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4) void ink2_lm_head_argmax_stage2(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    int* __restrict__ out_ids,
    int M,
    int num_ctas) {
  const int m = static_cast<int>(blockIdx.x);
  if (m >= M) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const size_t base = static_cast<size_t>(m) * static_cast<size_t>(num_ctas);

  float bv = -__int_as_float(0x7f800000);
  int bi = -1;
  for (int i = tid; i < num_ctas; i += 256) {
    ink2_lm_argmax_update(bv, bi, partial_vals[base + i], partial_idxs[base + i]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
    ink2_lm_argmax_update(bv, bi, ov, oi);
  }

  __shared__ float s_v[8];
  __shared__ int s_i[8];
  if (lane == 0) {
    s_v[warp] = bv;
    s_i[warp] = bi;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < 8) ? s_v[lane] : -__int_as_float(0x7f800000);
    int idx = (lane < 8) ? s_i[lane] : -1;
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, off);
      const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
      ink2_lm_argmax_update(v, idx, ov, oi);
    }
    if (lane == 0) {
      out_ids[m] = idx;
    }
  }
}
)cuda"
