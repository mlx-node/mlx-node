R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// One eight-thread group quantizes 32 values of one activation row. The packed
// activation stores Q8 codes, then FP32 scales and Q8 sums for each 32-value
// group. Eight rows are padded so one INT8 MMA tile covers every M in 2..8.
extern "C" __global__ void mlx_node_kquant_quantize_few(
    const nv_bfloat16* x, unsigned char* act, int M, int K) {
  const int units = K / 256;
  const int groups = units * 64;
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
  const int sub = threadIdx.x % 8;
  if (group >= groups) {
    return;
  }
  const int unit = group / 64;
  const int remainder = group % 64;
  const int row = remainder / 8;
  const int s = remainder % 8;
  float values[4];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = unit * 256 + s * 32 + sub * 4 + j;
    values[j] = row < M ? __bfloat162float(x[row * K + col]) : 0.0f;
    amax = fmaxf(amax, fabsf(values[j]));
  }
#pragma unroll
  for (int shift = 1; shift < 8; shift <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift, 8));
  }
  const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
  int q[4];
  int qsum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = amax > 0.0f ? static_cast<int>(roundf(values[j] * inverse)) : 0;
    qsum += q[j];
  }
#pragma unroll
  for (int shift = 1; shift < 8; shift <<= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, shift, 8);
  }
  *reinterpret_cast<unsigned int*>(act + group * 32 + sub * 4) =
      static_cast<unsigned int>(q[0] & 0xff) |
      (static_cast<unsigned int>(q[1] & 0xff) << 8) |
      (static_cast<unsigned int>(q[2] & 0xff) << 16) |
      (static_cast<unsigned int>(q[3] & 0xff) << 24);
  float* scales = reinterpret_cast<float*>(act + units * 2048);
  float* sums = scales + groups;
  const float d = amax > 0.0f ? 1.0f / inverse : 0.0f;
  if (sub == 0) {
    scales[group] = d;
    sums[group] = d * static_cast<float>(qsum);
  }
}

template <int mode>
__device__ __forceinline__ unsigned int kqf_unpack_four(unsigned int u) {
  if constexpr (mode == 0) {
    return (u & 0x0000000fu) | ((u & 0x000000f0u) << 4) |
        ((u & 0x00000f00u) << 8) | ((u & 0x0000f000u) << 12);
  } else if constexpr (mode == 3 || mode == 5) {
    // Decode four IQ4_NL/IQ4_XS indices with byte permutes.
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
__device__ __forceinline__ void kqf_load_pair(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  if constexpr (mode == 0 || mode == 3 || mode == 5) {
    const unsigned short* seg16 = reinterpret_cast<const unsigned short*>(seg);
    out0 = kqf_unpack_four<mode>(seg16[t]);
    out1 = kqf_unpack_four<mode>(seg16[4 + t]);
  } else if constexpr (mode == 1) {
    // Q5_K: 5 uint32 words (160 bits) per 32-weight segment.
    const int bit0 = t * 20;
    const int w0 = bit0 >> 5;
    const int sh0 = bit0 & 31;
    unsigned int u0 = seg[w0];
    if (sh0 > 12) {
      u0 = __funnelshift_r(u0, seg[w0 + 1], sh0);
    } else {
      u0 >>= sh0;
    }

    const int bit1 = 80 + t * 20;
    const int w1 = bit1 >> 5;
    const int sh1 = bit1 & 31;
    unsigned int u1 = seg[w1];
    if (sh1 > 12) {
      u1 = __funnelshift_r(u1, seg[w1 + 1], sh1);
    } else {
      u1 >>= sh1;
    }
    out0 = kqf_unpack_four<1>(u0);
    out1 = kqf_unpack_four<1>(u1);
  }
}

template <int mode>
__device__ __forceinline__ unsigned int kqf_load_k16(
    const unsigned int* seg, int kh, int t) {
  if constexpr (mode == 2) {
    // Q6_K: 6 uint32 words (192 bits) per 32-weight segment.
    const int bit = kh * 96 + t * 24;
    const int wi = bit >> 5;
    const int sh = bit & 31;
    unsigned int u = seg[wi];
    if (sh > 8) {
      u = __funnelshift_r(u, seg[wi + 1], sh);
    } else {
      u >>= sh;
    }
    return kqf_unpack_four<2>(u);
  } else {
    // Q3_K: 3 uint32 words (96 bits) per 32-weight segment.
    const int bit = kh * 48 + t * 12;
    const int wi = bit >> 5;
    const int sh = bit & 31;
    unsigned int u = seg[wi];
    if (sh > 20) {
      u = __funnelshift_r(u, seg[wi + 1], sh);
    } else {
      u >>= sh;
    }
    return kqf_unpack_four<4>(u);
  }
}

__device__ __forceinline__ void kqf_mma_k16(
    const unsigned int a0, const unsigned int a1, const unsigned int b,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ void kqf_mma(
    const unsigned int a0, const unsigned int a1, const unsigned int a2,
    const unsigned int a3, const unsigned int b0, const unsigned int b1,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Warps split 256-value units. Each warp evaluates a 16-row by 8-row
// INT8 tile and applies the format's scales and optional min correction.
// Modes: 0=Q4_K, 1=Q5_K, 2=Q6_K, 3=IQ4_XS, 4=Q3_K, 5=IQ4_NL.
template <int mode, bool tiled = false, int NUM_WARPS = 4>
__global__ __launch_bounds__(NUM_WARPS * 32, 2) void mlx_node_kquant_matmul_few(
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    const unsigned short* __restrict__ biases,
    const unsigned char* __restrict__ act,
    nv_bfloat16* __restrict__ out, int M, int N, int K) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int t = lane % 4;
  const int g = lane / 4;
  const int row0 = blockIdx.x * 16;
  const int rows[2] = {min(row0 + g, N - 1), min(row0 + g + 8, N - 1)};
  const int units = K / 256;
  const int units32 = K / 32;
  constexpr int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
  constexpr int V4 = bits * 2; // 16-byte vectors per 256-weight unit (8, 10, 12, 6)
  constexpr int sc_unit_bytes = mode == 3 ? 8 : 16;
  constexpr int b_unit_entries = (mode == 0 || mode == 1) ? 2 : 1;

  union SharedBuf {
    uint4 w_tile[NUM_WARPS][16][V4];
    float partials[NUM_WARPS * 16 * 8];
  };
  __shared__ SharedBuf smem;

  const int tile_ids[2] = {rows[0] / 64, rows[1] / 64};
  const int in_tile[2] = {rows[0] % 64, rows[1] % 64};
  const float* act_scales = reinterpret_cast<const float*>(act + units * 2048);
  const float* act_sums = act_scales + units * 64;
  const int w_s_stride = tiled ? 64 * bits : bits;
  const size_t row_w_stride_u4 = (size_t)units * V4;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);

  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int unit = warp; unit < units; unit += NUM_WARPS) {
    if constexpr (!tiled) {
      // Coalesced 128-bit streaming load of 16 rows for this unit into shared memory.
      const size_t unit_off_u4 = (size_t)unit * V4;
      if constexpr (V4 == 8) {
        const int r_in = lane >> 3;
        const int c_in = lane & 7;
#pragma unroll
        for (int rb = 0; rb < 16; rb += 4) {
          const int r = rb + r_in;
          const int rr = min(row0 + r, N - 1);
          smem.w_tile[warp][r][c_in] =
              w_u4[(size_t)rr * row_w_stride_u4 + unit_off_u4 + c_in];
        }
      } else {
        constexpr int total_v4 = 16 * V4;
#pragma unroll
        for (int idx = lane; idx < total_v4; idx += 32) {
          const int r = idx / V4;
          const int c = idx % V4;
          const int rr = min(row0 + r, N - 1);
          smem.w_tile[warp][r][c] =
              w_u4[(size_t)rr * row_w_stride_u4 + unit_off_u4 + c];
        }
      }
      __syncwarp();
    }

    float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float min_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint4 sc_vec[2];
    uint2 sc_vec8[2];
    if constexpr (mode != 5) {
      const unsigned char* sc_base[2] = {
          scales + (tiled
              ? (((size_t)tile_ids[0] * units + unit) * 64 + in_tile[0]) * sc_unit_bytes
              : ((size_t)rows[0] * units + unit) * sc_unit_bytes),
          scales + (tiled
              ? (((size_t)tile_ids[1] * units + unit) * 64 + in_tile[1]) * sc_unit_bytes
              : ((size_t)rows[1] * units + unit) * sc_unit_bytes)};
      if constexpr (sc_unit_bytes == 16) {
        sc_vec[0] = *reinterpret_cast<const uint4*>(sc_base[0]);
        sc_vec[1] = *reinterpret_cast<const uint4*>(sc_base[1]);
      } else {
        sc_vec8[0] = *reinterpret_cast<const uint2*>(sc_base[0]);
        sc_vec8[1] = *reinterpret_cast<const uint2*>(sc_base[1]);
      }
    }

    const unsigned int* w_base[2] = {
        tiled
            ? (w + (((size_t)tile_ids[0] * units32 + unit * 8) * 64 + in_tile[0]) * bits)
            : reinterpret_cast<const unsigned int*>(smem.w_tile[warp][g]),
        tiled
            ? (w + (((size_t)tile_ids[1] * units32 + unit * 8) * 64 + in_tile[1]) * bits)
            : reinterpret_cast<const unsigned int*>(smem.w_tile[warp][g + 8])};

#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const unsigned int* q32 =
          reinterpret_cast<const unsigned int*>(act + (unit * 64 + g * 8 + s) * 32);
      const float ad0 = act_scales[unit * 64 + (2 * t) * 8 + s];
      const float ad1 = act_scales[unit * 64 + (2 * t + 1) * 8 + s];
      if constexpr (mode == 2 || mode == 4) {
        const unsigned int sc_w0 = (s < 2) ? sc_vec[0].x : (s < 4 ? sc_vec[0].y : (s < 6 ? sc_vec[0].z : sc_vec[0].w));
        const unsigned int sc_w1 = (s < 2) ? sc_vec[1].x : (s < 4 ? sc_vec[1].y : (s < 6 ? sc_vec[1].z : sc_vec[1].w));
        const int shift_base = (s & 1) * 16;
        const unsigned int* seg0 = w_base[0] + s * w_s_stride;
        const unsigned int* seg1 = w_base[1] + s * w_s_stride;
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          const unsigned int a0 = kqf_load_k16<mode>(seg0, kh, t);
          const unsigned int a1 = kqf_load_k16<mode>(seg1, kh, t);
          const unsigned int b = q32[kh * 4 + t];
          int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
          kqf_mma_k16(a0, a1, b, c0, c1, c2, c3);
          const int ws0 = static_cast<signed char>((sc_w0 >> (shift_base + kh * 8)) & 0xff);
          const int ws1 = static_cast<signed char>((sc_w1 >> (shift_base + kh * 8)) & 0xff);
          dot[0] += ad0 * static_cast<float>(c0 * ws0);
          dot[1] += ad1 * static_cast<float>(c1 * ws0);
          dot[2] += ad0 * static_cast<float>(c2 * ws1);
          dot[3] += ad1 * static_cast<float>(c3 * ws1);
        }
      } else {
        unsigned int a0, a1, a2, a3;
        kqf_load_pair<mode>(w_base[0] + s * w_s_stride, t, a0, a2);
        kqf_load_pair<mode>(w_base[1] + s * w_s_stride, t, a1, a3);
        const unsigned int b0 = q32[t];
        const unsigned int b1 = q32[4 + t];
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        kqf_mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
        if constexpr (mode == 5) {
          const size_t idx0 = tiled
              ? (((size_t)tile_ids[0] * units32 + unit * 8 + s) * 64 + in_tile[0])
              : ((size_t)rows[0] * units32 + unit * 8 + s);
          const size_t idx1 = tiled
              ? (((size_t)tile_ids[1] * units32 + unit * 8 + s) * 64 + in_tile[1])
              : ((size_t)rows[1] * units32 + unit * 8 + s);
          const float ds0 = __half2float(__ushort_as_half(biases[idx0])) *
              static_cast<float>(static_cast<signed char>(scales[idx0]));
          const float ds1 = __half2float(__ushort_as_half(biases[idx1])) *
              static_cast<float>(static_cast<signed char>(scales[idx1]));
          total[0] += ad0 * ds0 * static_cast<float>(c0);
          total[1] += ad1 * ds0 * static_cast<float>(c1);
          total[2] += ad0 * ds1 * static_cast<float>(c2);
          total[3] += ad1 * ds1 * static_cast<float>(c3);
        } else if constexpr (mode == 3) {
          const unsigned int sw0 = s < 4 ? sc_vec8[0].x : sc_vec8[0].y;
          const unsigned int sw1 = s < 4 ? sc_vec8[1].x : sc_vec8[1].y;
          const int sh = (s & 3) * 8;
          const int ws0 = static_cast<signed char>((sw0 >> sh) & 0xff);
          const int ws1 = static_cast<signed char>((sw1 >> sh) & 0xff);
          dot[0] += ad0 * static_cast<float>(c0 * ws0);
          dot[1] += ad1 * static_cast<float>(c1 * ws0);
          dot[2] += ad0 * static_cast<float>(c2 * ws1);
          dot[3] += ad1 * static_cast<float>(c3 * ws1);
        } else {
          const float as0 = act_sums[unit * 64 + (2 * t) * 8 + s];
          const float as1 = act_sums[unit * 64 + (2 * t + 1) * 8 + s];
          const unsigned int sw0 = (s < 2) ? sc_vec[0].x : (s < 4 ? sc_vec[0].y : (s < 6 ? sc_vec[0].z : sc_vec[0].w));
          const unsigned int sw1 = (s < 2) ? sc_vec[1].x : (s < 4 ? sc_vec[1].y : (s < 6 ? sc_vec[1].z : sc_vec[1].w));
          const int sh = (s & 1) * 16;
          const int ws0 = (sw0 >> sh) & 0xff;
          const int wm0 = (sw0 >> (sh + 8)) & 0xff;
          const int ws1 = (sw1 >> sh) & 0xff;
          const int wm1 = (sw1 >> (sh + 8)) & 0xff;
          dot[0] += ad0 * static_cast<float>(c0 * ws0);
          dot[1] += ad1 * static_cast<float>(c1 * ws0);
          dot[2] += ad0 * static_cast<float>(c2 * ws1);
          dot[3] += ad1 * static_cast<float>(c3 * ws1);
          min_sum[0] += as0 * static_cast<float>(wm0);
          min_sum[1] += as1 * static_cast<float>(wm0);
          min_sum[2] += as0 * static_cast<float>(wm1);
          min_sum[3] += as1 * static_cast<float>(wm1);
        }
      }
    }
    if constexpr (mode != 5) {
      for (int r = 0; r < 2; ++r) {
        const size_t bias_index = tiled
            ? (((size_t)tile_ids[r] * units + unit) * 64 + in_tile[r]) * b_unit_entries
            : ((size_t)rows[r] * units + unit) * b_unit_entries;
        const float d = __half2float(__ushort_as_half(biases[bias_index]));
        const float dmin = (mode == 0 || mode == 1)
            ? __half2float(__ushort_as_half(biases[bias_index + 1]))
            : 0.0f;
        total[r * 2 + 0] += d * dot[r * 2 + 0] - dmin * min_sum[r * 2 + 0];
        total[r * 2 + 1] += d * dot[r * 2 + 1] - dmin * min_sum[r * 2 + 1];
      }
    }
  }

  __syncthreads();
  const int columns[2] = {2 * t, 2 * t + 1};
  for (int r = 0; r < 2; ++r) {
    const int row = g + r * 8;
    for (int half = 0; half < 2; ++half) {
      smem.partials[(warp * 16 + row) * 8 + columns[half]] = total[r * 2 + half];
    }
  }
  __syncthreads();
  const int flat = warp * 32 + lane;
  if (flat < 128) {
    const int row = flat / 8;
    const int column = flat % 8;
    if (row0 + row < N && column < M) {
      float value = 0.0f;
#pragma unroll
      for (int ww = 0; ww < NUM_WARPS; ++ww) {
        value += smem.partials[(ww * 16 + row) * 8 + column];
      }
      out[(size_t)column * N + row0 + row] = __float2bfloat16(value);
    }
  }
}
)cuda"