R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <int IN_DTYPE>
__device__ __forceinline__ void sym8_h_load8(
    const unsigned char* __restrict__ row_ptr, int u_idx, float v[8]) {
  if constexpr (IN_DTYPE == 0) {
    uint4 raw = reinterpret_cast<const uint4*>(row_ptr)[u_idx];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __bfloat1622float2(p[k]);
      v[2 * k] = f.x; v[2 * k + 1] = f.y;
    }
  } else if constexpr (IN_DTYPE == 1) {
    uint4 raw = reinterpret_cast<const uint4*>(row_ptr)[u_idx];
    const __half2* p = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __half22float2(p[k]);
      v[2 * k] = f.x; v[2 * k + 1] = f.y;
    }
  } else {
    const float4* fp = reinterpret_cast<const float4*>(row_ptr);
    float4 f0 = fp[2 * u_idx], f1 = fp[2 * u_idx + 1];
    v[0] = f0.x; v[1] = f0.y; v[2] = f0.z; v[3] = f0.w;
    v[4] = f1.x; v[5] = f1.y; v[6] = f1.z; v[7] = f1.w;
  }
}

template <int IN_DTYPE>
__device__ __forceinline__ float sym8_h_load1(
    const unsigned char* __restrict__ row_ptr, int c) {
  if constexpr (IN_DTYPE == 0) return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(row_ptr)[c]);
  else if constexpr (IN_DTYPE == 1) return __half2float(reinterpret_cast<const __half*>(row_ptr)[c]);
  else return reinterpret_cast<const float*>(row_ptr)[c];
}

__device__ __forceinline__ float sym8_h_silu(float g) {
  return g / (1.0f + __expf(-g));
}

__device__ __forceinline__ void sym8_h_fwht256_store(
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
  for (int e = 0; e < 4; ++e) { a = v[e]; b = v[e + 4]; v[e] = a + b; v[e + 4] = a - b; }
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

__device__ __forceinline__ void sym8_h_stage89(
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

__device__ __forceinline__ void sym8_h_reduce_scale(
    float amax, int bad, int tid, int warp, int lane,
    float* s_wmax, int* s_wbad, float* s_inv, float* tok_scale, int row) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
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
      const bool valid = !any_bad && (m > 0.0f) && isfinite(m);
      const float inv = valid ? (127.0f / m) : 0.0f;
      const bool ok = valid && isfinite(inv);
      *s_inv = ok ? inv : 0.0f;
      tok_scale[row] = ok ? (m * (1.0f / 127.0f)) : 0.0f;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ unsigned int sym8_h_q8_u8(float val, float inv, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val * inv), -127.0f), 127.0f)) : 0;
  return static_cast<unsigned int>(q) & 0xffu;
}

__device__ __forceinline__ signed char sym8_h_q8_i8(float val, float inv, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val * inv), -127.0f), 127.0f)) : 0;
  return static_cast<signed char>(q);
}

__device__ __forceinline__ uint2 sym8_h_pack8(
    float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7,
    float inv, bool valid) {
  uint2 pk;
  pk.x = sym8_h_q8_u8(a0, inv, valid) | (sym8_h_q8_u8(a1, inv, valid) << 8) |
         (sym8_h_q8_u8(a2, inv, valid) << 16) | (sym8_h_q8_u8(a3, inv, valid) << 24);
  pk.y = sym8_h_q8_u8(a4, inv, valid) | (sym8_h_q8_u8(a5, inv, valid) << 8) |
         (sym8_h_q8_u8(a6, inv, valid) << 16) | (sym8_h_q8_u8(a7, inv, valid) << 24);
  return pk;
}

template <int IN_DTYPE, bool HAS_SIGNS, bool FUSED_SWIGLU>
__device__ __forceinline__ void sym8_h_load_vec8(
    const unsigned char* __restrict__ gate_row,
    const unsigned char* __restrict__ up_row,
    const float4* __restrict__ signs_f4,
    int u_idx, float v[8]) {
  sym8_h_load8<IN_DTYPE>(gate_row, u_idx, v);
  if constexpr (FUSED_SWIGLU) {
    float u[8];
    sym8_h_load8<IN_DTYPE>(up_row, u_idx, u);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = sym8_h_silu(v[e]) * u[e];
  }
  if constexpr (HAS_SIGNS) {
    float4 s0 = signs_f4[2 * u_idx], s1 = signs_f4[2 * u_idx + 1];
    v[0] *= s0.x; v[1] *= s0.y; v[2] *= s0.z; v[3] *= s0.w;
    v[4] *= s1.x; v[5] *= s1.y; v[6] *= s1.z; v[7] *= s1.w;
  }
}

)cuda"
R"cuda(
template <int IN_DTYPE, bool HAS_SIGNS, bool FUSED_SWIGLU, bool SMEM_BF16>
__device__ __forceinline__ void sym8_h_1pass_core(
    const void* __restrict__ x, const float* __restrict__ signs,
    signed char* __restrict__ xq, float* __restrict__ tok_scale, int M, int K) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= M) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, k8 = K >> 3;
  constexpr size_t EB = (IN_DTYPE == 2) ? 4 : 2;
  const unsigned char* gate_row = reinterpret_cast<const unsigned char*>(x) +
      static_cast<size_t>(row) * static_cast<size_t>(FUSED_SWIGLU ? (2 * K) : K) * EB;
  const unsigned char* up_row = gate_row + static_cast<size_t>(K) * EB;
  const float4* signs_f4 = reinterpret_cast<const float4*>(signs);
  uint2* xq_row_u2 = reinterpret_cast<uint2*>(xq) + static_cast<size_t>(row) * k8;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* s_row_raw = HAS_SIGNS ? (smem_raw + 4 * 260 * sizeof(float)) : smem_raw;
  __shared__ float s_wmax[4];
  __shared__ int s_wbad[4];
  __shared__ float s_inv;
  float amax = 0.0f;
  int bad = 0;

  if constexpr (HAS_SIGNS) {
    float* s_wht = reinterpret_cast<float*>(smem_raw);
    const int num_blk = K >> 10;
    for (int blk = 0; blk < num_blk; ++blk) {
      float v[8];
      sym8_h_load_vec8<IN_DTYPE, true, FUSED_SWIGLU>(gate_row, up_row, signs_f4, blk * 128 + tid, v);
      sym8_h_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pos = tid + half * 128;
        float y0, y1, y2, y3;
        sym8_h_stage89(s_wht, pos, y0, y1, y2, y3);
        bad |= (!isfinite(y0)) | (!isfinite(y1)) | (!isfinite(y2)) | (!isfinite(y3));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
        if constexpr (SMEM_BF16) {
          nv_bfloat16* rbase = reinterpret_cast<nv_bfloat16*>(s_row_raw) + blk * 1024 + pos;
          rbase[0] = __float2bfloat16(y0); rbase[256] = __float2bfloat16(y1);
          rbase[512] = __float2bfloat16(y2); rbase[768] = __float2bfloat16(y3);
        } else {
          float* rbase = reinterpret_cast<float*>(s_row_raw) + blk * 1024 + pos;
          rbase[0] = y0; rbase[256] = y1; rbase[512] = y2; rbase[768] = y3;
        }
      }
      __syncthreads();
    }
  } else {
    for (int i = tid; i < k8; i += 128) {
      float v[8];
      sym8_h_load_vec8<IN_DTYPE, false, FUSED_SWIGLU>(gate_row, up_row, signs_f4, i, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bad |= !isfinite(v[e]);
        amax = fmaxf(amax, fabsf(v[e]));
      }
      if constexpr (SMEM_BF16) {
        __nv_bfloat162 b0 = __floats2bfloat162_rn(v[0], v[1]);
        __nv_bfloat162 b1 = __floats2bfloat162_rn(v[2], v[3]);
        __nv_bfloat162 b2 = __floats2bfloat162_rn(v[4], v[5]);
        __nv_bfloat162 b3 = __floats2bfloat162_rn(v[6], v[7]);
        reinterpret_cast<uint4*>(s_row_raw)[i] = make_uint4(
            *reinterpret_cast<const unsigned int*>(&b0), *reinterpret_cast<const unsigned int*>(&b1),
            *reinterpret_cast<const unsigned int*>(&b2), *reinterpret_cast<const unsigned int*>(&b3));
      } else {
        float4* s_row_f4 = reinterpret_cast<float4*>(s_row_raw);
        s_row_f4[2 * i] = make_float4(v[0], v[1], v[2], v[3]);
        s_row_f4[2 * i + 1] = make_float4(v[4], v[5], v[6], v[7]);
      }
    }
  }

  sym8_h_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_inv, tok_scale, row);
  const float inv = s_inv;
  const bool valid = (inv > 0.0f);
  if constexpr (SMEM_BF16) {
    const uint4* s_row_u4 = reinterpret_cast<const uint4*>(s_row_raw);
    for (int i = tid; i < k8; i += 128) {
      uint4 raw = s_row_u4[i];
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
      float2 f0 = __bfloat1622float2(p[0]), f1 = __bfloat1622float2(p[1]);
      float2 f2 = __bfloat1622float2(p[2]), f3 = __bfloat1622float2(p[3]);
      xq_row_u2[i] = sym8_h_pack8(f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y, inv, valid);
    }
  } else {
    const float4* s_row_f4 = reinterpret_cast<const float4*>(s_row_raw);
    for (int i = tid; i < k8; i += 128) {
      float4 f0 = s_row_f4[2 * i], f1 = s_row_f4[2 * i + 1];
      xq_row_u2[i] = sym8_h_pack8(f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w, inv, valid);
    }
  }
}

// 1. FP32 SMEM (K <= 8192, K % 8 == 0; HAS_SIGNS => K % 1024 == 0)
template <int IN_DTYPE, bool HAS_SIGNS, bool FUSED_SWIGLU>
__global__ __launch_bounds__(128, 2) void sym8_hadamard_quant_int8_fp32smem(
    const void* __restrict__ x, const float* __restrict__ signs,
    signed char* __restrict__ xq, float* __restrict__ tok_scale, int M, int K) {
  sym8_h_1pass_core<IN_DTYPE, HAS_SIGNS, FUSED_SWIGLU, false>(x, signs, xq, tok_scale, M, K);
}

// 2. BF16 SMEM (8192 < K <= 46080, K % 8 == 0; HAS_SIGNS => K % 1024 == 0)
template <int IN_DTYPE, bool HAS_SIGNS, bool FUSED_SWIGLU>
__global__ __launch_bounds__(128, 2) void sym8_hadamard_quant_int8_bf16smem(
    const void* __restrict__ x, const float* __restrict__ signs,
    signed char* __restrict__ xq, float* __restrict__ tok_scale, int M, int K) {
  sym8_h_1pass_core<IN_DTYPE, HAS_SIGNS, FUSED_SWIGLU, true>(x, signs, xq, tok_scale, M, K);
}

// 3. Tiny 4 KB static FP32 SMEM 2-pass fallback (K > 46080 or unaligned K when !HAS_SIGNS)
template <int IN_DTYPE, bool HAS_SIGNS, bool FUSED_SWIGLU>
__global__ __launch_bounds__(128, 2) void sym8_hadamard_quant_int8_2pass(
    const void* __restrict__ x, const float* __restrict__ signs,
    signed char* __restrict__ xq, float* __restrict__ tok_scale, int M, int K) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= M) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr size_t EB = (IN_DTYPE == 2) ? 4 : 2;
  const unsigned char* gate_row = reinterpret_cast<const unsigned char*>(x) +
      static_cast<size_t>(row) * static_cast<size_t>(FUSED_SWIGLU ? (2 * K) : K) * EB;
  const unsigned char* up_row = gate_row + static_cast<size_t>(K) * EB;
  signed char* xq_row = xq + static_cast<size_t>(row) * K;

  __shared__ float s_wmax[4];
  __shared__ int s_wbad[4];
  __shared__ float s_inv;
  float amax = 0.0f;
  int bad = 0;

  if constexpr (HAS_SIGNS) {
    __shared__ float s_wht[4 * 260];
    const float4* signs_f4 = reinterpret_cast<const float4*>(signs);
    const int num_blk = K >> 10;
    for (int blk = 0; blk < num_blk; ++blk) {
      float v[8];
      sym8_h_load_vec8<IN_DTYPE, true, FUSED_SWIGLU>(gate_row, up_row, signs_f4, blk * 128 + tid, v);
      sym8_h_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float y0, y1, y2, y3;
        sym8_h_stage89(s_wht, tid + half * 128, y0, y1, y2, y3);
        bad |= (!isfinite(y0)) | (!isfinite(y1)) | (!isfinite(y2)) | (!isfinite(y3));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
      }
      __syncthreads();
    }
    sym8_h_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_inv, tok_scale, row);
    const float inv = s_inv;
    const bool valid = (inv > 0.0f);
    for (int blk = 0; blk < num_blk; ++blk) {
      float v[8];
      sym8_h_load_vec8<IN_DTYPE, true, FUSED_SWIGLU>(gate_row, up_row, signs_f4, blk * 128 + tid, v);
      sym8_h_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pos = tid + half * 128;
        float y0, y1, y2, y3;
        sym8_h_stage89(s_wht, pos, y0, y1, y2, y3);
        signed char* dst = xq_row + static_cast<size_t>(blk) * 1024 + pos;
        dst[0] = sym8_h_q8_i8(y0, inv, valid); dst[256] = sym8_h_q8_i8(y1, inv, valid);
        dst[512] = sym8_h_q8_i8(y2, inv, valid); dst[768] = sym8_h_q8_i8(y3, inv, valid);
      }
      __syncthreads();
    }
  } else {
    for (int c = tid; c < K; c += 128) {
      float g = sym8_h_load1<IN_DTYPE>(gate_row, c);
      float val = FUSED_SWIGLU ? (sym8_h_silu(g) * sym8_h_load1<IN_DTYPE>(up_row, c)) : g;
      bad |= !isfinite(val);
      amax = fmaxf(amax, fabsf(val));
    }
    sym8_h_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_inv, tok_scale, row);
    const float inv = s_inv;
    const bool valid = (inv > 0.0f);
    for (int c = tid; c < K; c += 128) {
      float g = sym8_h_load1<IN_DTYPE>(gate_row, c);
      float val = FUSED_SWIGLU ? (sym8_h_silu(g) * sym8_h_load1<IN_DTYPE>(up_row, c)) : g;
      xq_row[c] = sym8_h_q8_i8(val, inv, valid);
    }
  }
}
)cuda"
