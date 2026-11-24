R"cuda(
#include <cuda_fp16.h>

// Fused bit-exact FP32 K-quant (Q4_K / Q6_K, RowMajor / Tiled64) dequantize +
// optional Rademacher signs + FWHT-1024 + per-channel symmetric INT8 quantizer.
// Matches kquant.cu.inc dequantization and sym8_weight_quant.cu.inc FWHT/quant
// math bit-for-bit.

__device__ __forceinline__ void sym8_wqk_fwht256_store(
    float v[8], float* s_wht, int warp, int lane) {
  float a, b;
  a = v[0]; b = v[1]; v[0] = a + b; v[1] = a - b;
  a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b;
  a = v[4]; b = v[5]; v[4] = a + b; v[5] = a - b;
  a = v[6]; b = v[7]; v[6] = a + b; v[7] = a - b;
  a = v[0]; b = v[2]; v[0] = a + b; v[2] = a - b;
  a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b;
  a = v[4]; b = v[6]; v[4] = a + b; v[6] = a - b;
  a = v[5]; b = v[7]; v[5] = a + b; v[7] = a - b;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    a = v[e]; b = v[e + 4]; v[e] = a + b; v[e + 4] = a - b;
  }
#pragma unroll
  for (int mask = 1; mask < 32; mask <<= 1) {
    const bool hi = (lane & mask) != 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float peer = __shfl_xor_sync(0xffffffffu, v[e], mask);
      v[e] = hi ? (peer - v[e]) : (v[e] + peer);
    }
  }
  float* wptr = s_wht + warp * 260 + lane * 8;
#pragma unroll
  for (int e = 0; e < 8; ++e) wptr[e] = v[e];
  __syncthreads();
}

__device__ __forceinline__ void sym8_wqk_stage89(
    const float* s_wht, int pos, float& y0, float& y1, float& y2, float& y3) {
  constexpr float INV_SQRT_1024 = 0.03125f;
  float x0 = s_wht[0 * 260 + pos], x1 = s_wht[1 * 260 + pos];
  float x2 = s_wht[2 * 260 + pos], x3 = s_wht[3 * 260 + pos];
  float t0 = x0 + x1, t1 = x0 - x1, t2 = x2 + x3, t3 = x2 - x3;
  y0 = (t0 + t2) * INV_SQRT_1024;
  y1 = (t1 + t3) * INV_SQRT_1024;
  y2 = (t0 - t2) * INV_SQRT_1024;
  y3 = (t1 - t3) * INV_SQRT_1024;
}

__device__ __forceinline__ void sym8_wqk_reduce_scale(
    float amax, int bad, int tid, int warp, int lane,
    float* s_wmax, int* s_wbad, float* s_sw, float* s_out, int row) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  const int warp_bad = __any_sync(0xffffffffu, bad);
  if (lane == 0) {
    s_wmax[warp] = amax;
    s_wbad[warp] = warp_bad;
  }
  __syncthreads();
  if (tid < 4) {
    float m = s_wmax[tid];
    m = fmaxf(m, __shfl_xor_sync(0xfu, m, 1, 4));
    m = fmaxf(m, __shfl_xor_sync(0xfu, m, 2, 4));
    const int any_bad = __any_sync(0xfu, s_wbad[tid]);
    if (tid == 0) {
      const bool ok = !any_bad && isfinite(m);
      const float sw = ok ? (fmaxf(m, 1e-12f) / 127.0f) : 0.0f;
      *s_sw = sw;
      s_out[row] = sw;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ signed char sym8_wqk_q8_i8(float val, float sw, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val / sw), -127.0f), 127.0f)) : 0;
  return static_cast<signed char>(q);
}

__device__ __forceinline__ unsigned int sym8_wqk_q8_u8(float val, float sw, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val / sw), -127.0f), 127.0f)) : 0;
  return static_cast<unsigned int>(q) & 0xffu;
}

__device__ __forceinline__ uint2 sym8_wqk_pack8(const float v[8], float sw, bool valid) {
  uint2 pk;
  pk.x = sym8_wqk_q8_u8(v[0], sw, valid) | (sym8_wqk_q8_u8(v[1], sw, valid) << 8) |
         (sym8_wqk_q8_u8(v[2], sw, valid) << 16) | (sym8_wqk_q8_u8(v[3], sw, valid) << 24);
  pk.y = sym8_wqk_q8_u8(v[4], sw, valid) | (sym8_wqk_q8_u8(v[5], sw, valid) << 8) |
         (sym8_wqk_q8_u8(v[6], sw, valid) << 16) | (sym8_wqk_q8_u8(v[7], sw, valid) << 24);
  return pk;
}

template <int IS_Q6K, int TILED, int HAS_SIGNS>
__device__ __forceinline__ void sym8_wqk_dequant(
    const unsigned int* __restrict__ w_u32,
    const unsigned char* __restrict__ scales_u8,
    const unsigned short* __restrict__ biases_u16,
    const float4* __restrict__ signs_f4,
    int row, int sb, int lane, int K, float v[8]) {
  const int u = lane >> 2;
  const int q8 = lane & 3;
  const int num_sb = K >> 8;
  const int units32 = K >> 5;
  const int tile = row >> 6;
  const int in_tile = row & 63;

  if constexpr (IS_Q6K == 0) {
    unsigned int word;
    size_t sc_off;
    size_t bi_off;
    if constexpr (TILED == 0) {
      word = w_u32[static_cast<size_t>(row) * static_cast<size_t>(K >> 3) +
                   static_cast<size_t>(sb) * 32 + lane];
      sc_off = static_cast<size_t>(row) * static_cast<size_t>(K >> 4) +
               static_cast<size_t>(sb) * 16 + u * 2;
      bi_off = static_cast<size_t>(row) * static_cast<size_t>(K >> 7) +
               static_cast<size_t>(sb) * 2;
    } else {
      const size_t unit_base =
          (static_cast<size_t>(tile) * static_cast<size_t>(units32) +
           static_cast<size_t>(sb) * 8 + u) * 64 + in_tile;
      word = w_u32[unit_base * 4 + q8];
      const size_t comp =
          (static_cast<size_t>(tile) * static_cast<size_t>(num_sb) + sb) * 64 + in_tile;
      sc_off = comp * 16 + u * 2;
      bi_off = comp * 2;
    }
    const float scale = __fmul_rn(
        __half2float(__ushort_as_half(biases_u16[bi_off])),
        static_cast<float>(scales_u8[sc_off]));
    const float minimum = __fmul_rn(
        __half2float(__ushort_as_half(biases_u16[bi_off + 1])),
        static_cast<float>(scales_u8[sc_off + 1]));
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[e] = __fsub_rn(
          __fmul_rn(scale, static_cast<float>((word >> (4 * e)) & 0xFu)),
          minimum);
    }
  } else {
    const unsigned int* seg;
    size_t sc_off;
    size_t bi_off;
    if constexpr (TILED == 0) {
      seg = w_u32 + static_cast<size_t>(row) * (static_cast<size_t>(K >> 4) * 3) +
            static_cast<size_t>(sb) * 48 + static_cast<size_t>(u) * 6;
      sc_off = static_cast<size_t>(row) * static_cast<size_t>(K >> 4) +
               static_cast<size_t>(sb) * 16 + (lane >> 1);
      bi_off = static_cast<size_t>(row) * static_cast<size_t>(num_sb) + sb;
    } else {
      const size_t unit_base =
          (static_cast<size_t>(tile) * static_cast<size_t>(units32) +
           static_cast<size_t>(sb) * 8 + u) * 64 + in_tile;
      seg = w_u32 + unit_base * 6;
      const size_t comp =
          (static_cast<size_t>(tile) * static_cast<size_t>(num_sb) + sb) * 64 + in_tile;
      sc_off = comp * 16 + (lane >> 1);
      bi_off = comp;
    }
    const int w_idx = (q8 * 3) >> 1;
    const int sh = (q8 & 1) * 16;
    const unsigned long long bits48 =
        (static_cast<unsigned long long>(seg[w_idx]) >> sh) |
        (static_cast<unsigned long long>(seg[w_idx + 1]) << (32 - sh));
    const float scale = __fmul_rn(
        __half2float(__ushort_as_half(biases_u16[bi_off])),
        static_cast<float>(static_cast<signed char>(scales_u8[sc_off])));
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int code = static_cast<int>((bits48 >> (6 * e)) & 0x3Fu) - 32;
      v[e] = __fmul_rn(scale, static_cast<float>(code));
    }
  }

  if constexpr (HAS_SIGNS) {
    const int u_idx = sb * 32 + lane;
    const float4 s0 = signs_f4[2 * u_idx];
    const float4 s1 = signs_f4[2 * u_idx + 1];
    v[0] *= s0.x; v[1] *= s0.y; v[2] *= s0.z; v[3] *= s0.w;
    v[4] *= s1.x; v[5] *= s1.y; v[6] *= s1.z; v[7] *= s1.w;
  }
}

template <int IS_Q6K, int TILED, int HAS_SIGNS>
__device__ __forceinline__ void sym8_wqk_impl(
    const unsigned int* __restrict__ w_u32,
    const unsigned char* __restrict__ scales_u8,
    const unsigned short* __restrict__ biases_u16,
    const float* __restrict__ signs,
    signed char* __restrict__ q_out,
    float* __restrict__ s_out,
    int N, int K) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= N) return;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const float4* signs_f4 = reinterpret_cast<const float4*>(signs);
  signed char* q_row = q_out + static_cast<size_t>(row) * static_cast<size_t>(K);

  __shared__ float s_wmax[4];
  __shared__ int s_wbad[4];
  __shared__ float s_sw;
  float amax = 0.0f;
  int bad = 0;

  if constexpr (HAS_SIGNS) {
    __shared__ float s_wht[4 * 260];
    const int num_blk = K >> 10;
    for (int blk = 0; blk < num_blk; ++blk) {
      float v[8];
      sym8_wqk_dequant<IS_Q6K, TILED, 1>(
          w_u32, scales_u8, biases_u16, signs_f4, row, blk * 4 + warp, lane, K, v);
      sym8_wqk_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float y0, y1, y2, y3;
        sym8_wqk_stage89(s_wht, tid + half * 128, y0, y1, y2, y3);
        bad |= (!isfinite(y0)) | (!isfinite(y1)) | (!isfinite(y2)) | (!isfinite(y3));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
      }
      __syncthreads();
    }
    sym8_wqk_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    for (int blk = 0; blk < num_blk; ++blk) {
      float v[8];
      sym8_wqk_dequant<IS_Q6K, TILED, 1>(
          w_u32, scales_u8, biases_u16, signs_f4, row, blk * 4 + warp, lane, K, v);
      sym8_wqk_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pos = tid + half * 128;
        float y0, y1, y2, y3;
        sym8_wqk_stage89(s_wht, pos, y0, y1, y2, y3);
        signed char* dst = q_row + static_cast<size_t>(blk) * 1024 + pos;
        dst[0] = sym8_wqk_q8_i8(y0, sw, valid);
        dst[256] = sym8_wqk_q8_i8(y1, sw, valid);
        dst[512] = sym8_wqk_q8_i8(y2, sw, valid);
        dst[768] = sym8_wqk_q8_i8(y3, sw, valid);
      }
      __syncthreads();
    }
  } else {
    const int num_sb = K >> 8;
    for (int sb = warp; sb < num_sb; sb += 4) {
      float v[8];
      sym8_wqk_dequant<IS_Q6K, TILED, 0>(
          w_u32, scales_u8, biases_u16, signs_f4, row, sb, lane, K, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bad |= !isfinite(v[e]);
        amax = fmaxf(amax, fabsf(v[e]));
      }
    }
    sym8_wqk_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    uint2* q_row_u2 = reinterpret_cast<uint2*>(q_row);
    for (int sb = warp; sb < num_sb; sb += 4) {
      float v[8];
      sym8_wqk_dequant<IS_Q6K, TILED, 0>(
          w_u32, scales_u8, biases_u16, signs_f4, row, sb, lane, K, v);
      q_row_u2[sb * 32 + lane] = sym8_wqk_pack8(v, sw, valid);
    }
  }
}

template <int IS_Q6K, int TILED, int HAS_SIGNS>
__global__ __launch_bounds__(128, 2) void sym8_quantize_weight_kquant(
    const unsigned int* __restrict__ w_u32,
    const unsigned char* __restrict__ scales_u8,
    const unsigned short* __restrict__ biases_u16,
    const float* __restrict__ signs,
    signed char* __restrict__ q_out,
    float* __restrict__ s_out,
    int N, int K) {
  sym8_wqk_impl<IS_Q6K, TILED, HAS_SIGNS>(
      w_u32, scales_u8, biases_u16, signs, q_out, s_out, N, K);
}

#define SYM8_WQK_ENTRY(NAME, Q6, T64, SIG)                                  \
  extern "C" __global__ __launch_bounds__(128, 2) void NAME(                \
      const unsigned int* __restrict__ w_u32,                               \
      const unsigned char* __restrict__ scales_u8,                          \
      const unsigned short* __restrict__ biases_u16,                        \
      const float* __restrict__ signs,                                      \
      signed char* __restrict__ q_out,                                      \
      float* __restrict__ s_out,                                            \
      int N, int K) {                                                       \
    sym8_wqk_impl<Q6, T64, SIG>(                                            \
        w_u32, scales_u8, biases_u16, signs, q_out, s_out, N, K);           \
  }

SYM8_WQK_ENTRY(mlx_sym8_wq_q4k_rm_nosigns, 0, 0, 0)
SYM8_WQK_ENTRY(mlx_sym8_wq_q4k_rm_signs, 0, 0, 1)
SYM8_WQK_ENTRY(mlx_sym8_wq_q4k_t64_nosigns, 0, 1, 0)
SYM8_WQK_ENTRY(mlx_sym8_wq_q4k_t64_signs, 0, 1, 1)
SYM8_WQK_ENTRY(mlx_sym8_wq_q6k_rm_nosigns, 1, 0, 0)
SYM8_WQK_ENTRY(mlx_sym8_wq_q6k_rm_signs, 1, 0, 1)
SYM8_WQK_ENTRY(mlx_sym8_wq_q6k_t64_nosigns, 1, 1, 0)
SYM8_WQK_ENTRY(mlx_sym8_wq_q6k_t64_signs, 1, 1, 1)
#undef SYM8_WQK_ENTRY
)cuda"
