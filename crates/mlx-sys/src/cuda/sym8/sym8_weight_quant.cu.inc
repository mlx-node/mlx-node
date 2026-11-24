R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <int SB_DTYPE>
__device__ __forceinline__ float sym8_wq_load_sb(const void* __restrict__ ptr, size_t idx) {
  if constexpr (SB_DTYPE == 0) return reinterpret_cast<const float*>(ptr)[idx];
  else if constexpr (SB_DTYPE == 1) return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  else return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <int IN_DTYPE>
__device__ __forceinline__ void sym8_wq_load8_dense(
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
__device__ __forceinline__ float sym8_wq_load1_dense(
    const unsigned char* __restrict__ row_ptr, int c) {
  if constexpr (IN_DTYPE == 0) return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(row_ptr)[c]);
  else if constexpr (IN_DTYPE == 1) return __half2float(reinterpret_cast<const __half*>(row_ptr)[c]);
  else return reinterpret_cast<const float*>(row_ptr)[c];
}

__device__ __forceinline__ void sym8_wq_fwht256_store(
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

__device__ __forceinline__ void sym8_wq_stage89(
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

__device__ __forceinline__ void sym8_wq_reduce_scale(
    float amax, int bad, int tid, int warp, int lane,
    float* s_wmax, int* s_wbad, float* s_sw, float* s_out, int row) {
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
      const bool ok = !any_bad && isfinite(m);
      const float sw = ok ? (fmaxf(m, 1e-12f) / 127.0f) : 0.0f;
      *s_sw = sw;
      s_out[row] = sw;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ signed char sym8_wq_q8_i8(float val, float sw, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val / sw), -127.0f), 127.0f)) : 0;
  return static_cast<signed char>(q);
}

__device__ __forceinline__ unsigned int sym8_wq_q8_u8(float val, float sw, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val / sw), -127.0f), 127.0f)) : 0;
  return static_cast<unsigned int>(q) & 0xffu;
}

__device__ __forceinline__ uint2 sym8_wq_pack8(const float v[8], float sw, bool valid) {
  uint2 pk;
  pk.x = sym8_wq_q8_u8(v[0], sw, valid) | (sym8_wq_q8_u8(v[1], sw, valid) << 8) |
         (sym8_wq_q8_u8(v[2], sw, valid) << 16) | (sym8_wq_q8_u8(v[3], sw, valid) << 24);
  pk.y = sym8_wq_q8_u8(v[4], sw, valid) | (sym8_wq_q8_u8(v[5], sw, valid) << 8) |
         (sym8_wq_q8_u8(v[6], sw, valid) << 16) | (sym8_wq_q8_u8(v[7], sw, valid) << 24);
  return pk;
}

template <int SB_DTYPE, bool HAS_BIASES, bool HAS_SIGNS>
__device__ __forceinline__ void sym8_wq_unpack_q8_vec8(
    const uint2* __restrict__ w_row_u2, const void* __restrict__ w_scales,
    const void* __restrict__ w_biases, size_t grp_base,
    const float4* __restrict__ signs_f4, int u_idx, float v[8]) {
  const uint2 raw = w_row_u2[u_idx];
  const size_t grp = grp_base + static_cast<size_t>(u_idx >> 2);
  const float sc = sym8_wq_load_sb<SB_DTYPE>(w_scales, grp);
  const float bi = HAS_BIASES ? sym8_wq_load_sb<SB_DTYPE>(w_biases, grp) : 0.0f;
  v[0] = static_cast<float>((raw.x      ) & 0xffu) * sc + bi;
  v[1] = static_cast<float>((raw.x >>  8) & 0xffu) * sc + bi;
  v[2] = static_cast<float>((raw.x >> 16) & 0xffu) * sc + bi;
  v[3] = static_cast<float>((raw.x >> 24) & 0xffu) * sc + bi;
  v[4] = static_cast<float>((raw.y      ) & 0xffu) * sc + bi;
  v[5] = static_cast<float>((raw.y >>  8) & 0xffu) * sc + bi;
  v[6] = static_cast<float>((raw.y >> 16) & 0xffu) * sc + bi;
  v[7] = static_cast<float>((raw.y >> 24) & 0xffu) * sc + bi;
  if constexpr (HAS_SIGNS) {
    float4 s0 = signs_f4[2 * u_idx], s1 = signs_f4[2 * u_idx + 1];
    v[0] *= s0.x; v[1] *= s0.y; v[2] *= s0.z; v[3] *= s0.w;
    v[4] *= s1.x; v[5] *= s1.y; v[6] *= s1.z; v[7] *= s1.w;
  }
}

// 1. Packed affine Q8 (uint32 [N, K/4], scales/biases [N, K/32]) -> sym8 (int8 [N, K], f32 [N])
template <int SB_DTYPE, bool HAS_BIASES, bool HAS_SIGNS>
__global__ __launch_bounds__(128, 2) void sym8_quantize_weight_q8(
    const unsigned int* __restrict__ w_u32, const void* __restrict__ w_scales,
    const void* __restrict__ w_biases, const float* __restrict__ signs,
    signed char* __restrict__ q_out, float* __restrict__ s_out, int N, int K) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= N) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, k8 = K >> 3;
  const size_t grp_base = static_cast<size_t>(row) * static_cast<size_t>(K >> 5);
  const uint2* w_row_u2 = reinterpret_cast<const uint2*>(w_u32) + static_cast<size_t>(row) * k8;
  const float4* signs_f4 = reinterpret_cast<const float4*>(signs);
  signed char* q_row = q_out + static_cast<size_t>(row) * K;

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
      sym8_wq_unpack_q8_vec8<SB_DTYPE, HAS_BIASES, true>(
          w_row_u2, w_scales, w_biases, grp_base, signs_f4, blk * 128 + tid, v);
      sym8_wq_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float y0, y1, y2, y3;
        sym8_wq_stage89(s_wht, tid + half * 128, y0, y1, y2, y3);
        bad |= (!isfinite(y0)) | (!isfinite(y1)) | (!isfinite(y2)) | (!isfinite(y3));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
      }
      __syncthreads();
    }
    sym8_wq_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    for (int blk = 0; blk < num_blk; ++blk) {
      float v[8];
      sym8_wq_unpack_q8_vec8<SB_DTYPE, HAS_BIASES, true>(
          w_row_u2, w_scales, w_biases, grp_base, signs_f4, blk * 128 + tid, v);
      sym8_wq_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pos = tid + half * 128;
        float y0, y1, y2, y3;
        sym8_wq_stage89(s_wht, pos, y0, y1, y2, y3);
        signed char* dst = q_row + static_cast<size_t>(blk) * 1024 + pos;
        dst[0] = sym8_wq_q8_i8(y0, sw, valid); dst[256] = sym8_wq_q8_i8(y1, sw, valid);
        dst[512] = sym8_wq_q8_i8(y2, sw, valid); dst[768] = sym8_wq_q8_i8(y3, sw, valid);
      }
      __syncthreads();
    }
  } else {
    for (int i = tid; i < k8; i += 128) {
      float v[8];
      sym8_wq_unpack_q8_vec8<SB_DTYPE, HAS_BIASES, false>(
          w_row_u2, w_scales, w_biases, grp_base, signs_f4, i, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bad |= !isfinite(v[e]);
        amax = fmaxf(amax, fabsf(v[e]));
      }
    }
    sym8_wq_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    uint2* q_row_u2 = reinterpret_cast<uint2*>(q_row);
    for (int i = tid; i < k8; i += 128) {
      float v[8];
      sym8_wq_unpack_q8_vec8<SB_DTYPE, HAS_BIASES, false>(
          w_row_u2, w_scales, w_biases, grp_base, signs_f4, i, v);
      q_row_u2[i] = sym8_wq_pack8(v, sw, valid);
    }
  }
}

// 2. Dense BF16/FP16/FP32 weight [N, K] -> sym8 (int8 [N, K], f32 [N])
template <int IN_DTYPE, bool HAS_SIGNS>
__global__ __launch_bounds__(128, 2) void sym8_quantize_weight_dense(
    const void* __restrict__ w, const float* __restrict__ signs,
    signed char* __restrict__ q_out, float* __restrict__ s_out, int N, int K) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= N) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr size_t EB = (IN_DTYPE == 2) ? 4 : 2;
  const unsigned char* w_row = reinterpret_cast<const unsigned char*>(w) +
      static_cast<size_t>(row) * static_cast<size_t>(K) * EB;
  signed char* q_row = q_out + static_cast<size_t>(row) * K;

  __shared__ float s_wmax[4];
  __shared__ int s_wbad[4];
  __shared__ float s_sw;
  float amax = 0.0f;
  int bad = 0;

  if constexpr (HAS_SIGNS) {
    __shared__ float s_wht[4 * 260];
    const float4* signs_f4 = reinterpret_cast<const float4*>(signs);
    const int num_blk = K >> 10;
    for (int blk = 0; blk < num_blk; ++blk) {
      const int u_idx = blk * 128 + tid;
      float v[8];
      sym8_wq_load8_dense<IN_DTYPE>(w_row, u_idx, v);
      float4 s0 = signs_f4[2 * u_idx], s1 = signs_f4[2 * u_idx + 1];
      v[0] *= s0.x; v[1] *= s0.y; v[2] *= s0.z; v[3] *= s0.w;
      v[4] *= s1.x; v[5] *= s1.y; v[6] *= s1.z; v[7] *= s1.w;
      sym8_wq_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float y0, y1, y2, y3;
        sym8_wq_stage89(s_wht, tid + half * 128, y0, y1, y2, y3);
        bad |= (!isfinite(y0)) | (!isfinite(y1)) | (!isfinite(y2)) | (!isfinite(y3));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
      }
      __syncthreads();
    }
    sym8_wq_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    for (int blk = 0; blk < num_blk; ++blk) {
      const int u_idx = blk * 128 + tid;
      float v[8];
      sym8_wq_load8_dense<IN_DTYPE>(w_row, u_idx, v);
      float4 s0 = signs_f4[2 * u_idx], s1 = signs_f4[2 * u_idx + 1];
      v[0] *= s0.x; v[1] *= s0.y; v[2] *= s0.z; v[3] *= s0.w;
      v[4] *= s1.x; v[5] *= s1.y; v[6] *= s1.z; v[7] *= s1.w;
      sym8_wq_fwht256_store(v, s_wht, warp, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pos = tid + half * 128;
        float y0, y1, y2, y3;
        sym8_wq_stage89(s_wht, pos, y0, y1, y2, y3);
        signed char* dst = q_row + static_cast<size_t>(blk) * 1024 + pos;
        dst[0] = sym8_wq_q8_i8(y0, sw, valid); dst[256] = sym8_wq_q8_i8(y1, sw, valid);
        dst[512] = sym8_wq_q8_i8(y2, sw, valid); dst[768] = sym8_wq_q8_i8(y3, sw, valid);
      }
      __syncthreads();
    }
  } else if ((K & 7) == 0) {
    const int k8 = K >> 3;
    uint2* q_row_u2 = reinterpret_cast<uint2*>(q_row);
    for (int i = tid; i < k8; i += 128) {
      float v[8];
      sym8_wq_load8_dense<IN_DTYPE>(w_row, i, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bad |= !isfinite(v[e]);
        amax = fmaxf(amax, fabsf(v[e]));
      }
    }
    sym8_wq_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    for (int i = tid; i < k8; i += 128) {
      float v[8];
      sym8_wq_load8_dense<IN_DTYPE>(w_row, i, v);
      q_row_u2[i] = sym8_wq_pack8(v, sw, valid);
    }
  } else {
    for (int c = tid; c < K; c += 128) {
      float val = sym8_wq_load1_dense<IN_DTYPE>(w_row, c);
      bad |= !isfinite(val);
      amax = fmaxf(amax, fabsf(val));
    }
    sym8_wq_reduce_scale(amax, bad, tid, warp, lane, s_wmax, s_wbad, &s_sw, s_out, row);
    const float sw = s_sw;
    const bool valid = (sw > 0.0f);
    for (int c = tid; c < K; c += 128) q_row[c] = sym8_wq_q8_i8(sym8_wq_load1_dense<IN_DTYPE>(w_row, c), sw, valid);
  }
}
)cuda"
