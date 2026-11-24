R"cuda(
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned int sym8_rn_q8_u8(float val, float inv, bool valid) {
  int q = valid ? static_cast<int>(fminf(fmaxf(rintf(val * inv), -127.0f), 127.0f)) : 0;
  return static_cast<unsigned int>(q) & 0xffu;
}

__device__ __forceinline__ unsigned int sym8_rn_pack4(float4 f, float inv, bool valid) {
  return sym8_rn_q8_u8(f.x, inv, valid) | (sym8_rn_q8_u8(f.y, inv, valid) << 8) |
         (sym8_rn_q8_u8(f.z, inv, valid) << 16) | (sym8_rn_q8_u8(f.w, inv, valid) << 24);
}

__device__ __forceinline__ void sym8_rn_fwht256_inplace(
    float v[8], float* s_blk, int tid, int lane) {
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
  float4* s_f4 = reinterpret_cast<float4*>(s_blk);
  s_f4[2 * tid] = make_float4(v[0], v[1], v[2], v[3]);
  s_f4[2 * tid + 1] = make_float4(v[4], v[5], v[6], v[7]);
  __syncthreads();
}

template <bool WRITE_NORMED, bool W_IS_F32>
__device__ __forceinline__ void sym8_rn_apply_norm8(
    const float hf[8], const void* __restrict__ w_norm, uint4* __restrict__ normed_u4,
    int u_idx, float inv_rms, float v[8]) {
  __nv_bfloat162 b0, b1, b2, b3;
  if constexpr (W_IS_F32) {
    const float4* w_f4 = reinterpret_cast<const float4*>(w_norm);
    float4 w0 = w_f4[2 * u_idx];
    float4 w1 = w_f4[2 * u_idx + 1];
    b0 = __floats2bfloat162_rn((hf[0] * inv_rms) * w0.x, (hf[1] * inv_rms) * w0.y);
    b1 = __floats2bfloat162_rn((hf[2] * inv_rms) * w0.z, (hf[3] * inv_rms) * w0.w);
    b2 = __floats2bfloat162_rn((hf[4] * inv_rms) * w1.x, (hf[5] * inv_rms) * w1.y);
    b3 = __floats2bfloat162_rn((hf[6] * inv_rms) * w1.z, (hf[7] * inv_rms) * w1.w);
  } else {
    uint4 wa = reinterpret_cast<const uint4*>(w_norm)[u_idx];
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wa);
    float2 wf0 = __bfloat1622float2(wp[0]);
    float2 wf1 = __bfloat1622float2(wp[1]);
    float2 wf2 = __bfloat1622float2(wp[2]);
    float2 wf3 = __bfloat1622float2(wp[3]);
    __nv_bfloat162 sc0 = __floats2bfloat162_rn(hf[0] * inv_rms, hf[1] * inv_rms);
    __nv_bfloat162 sc1 = __floats2bfloat162_rn(hf[2] * inv_rms, hf[3] * inv_rms);
    __nv_bfloat162 sc2 = __floats2bfloat162_rn(hf[4] * inv_rms, hf[5] * inv_rms);
    __nv_bfloat162 sc3 = __floats2bfloat162_rn(hf[6] * inv_rms, hf[7] * inv_rms);
    float2 sf0 = __bfloat1622float2(sc0);
    float2 sf1 = __bfloat1622float2(sc1);
    float2 sf2 = __bfloat1622float2(sc2);
    float2 sf3 = __bfloat1622float2(sc3);
    b0 = __floats2bfloat162_rn(wf0.x * sf0.x, wf0.y * sf0.y);
    b1 = __floats2bfloat162_rn(wf1.x * sf1.x, wf1.y * sf1.y);
    b2 = __floats2bfloat162_rn(wf2.x * sf2.x, wf2.y * sf2.y);
    b3 = __floats2bfloat162_rn(wf3.x * sf3.x, wf3.y * sf3.y);
  }
  if constexpr (WRITE_NORMED) {
    normed_u4[u_idx] = make_uint4(
        *reinterpret_cast<const unsigned int*>(&b0),
        *reinterpret_cast<const unsigned int*>(&b1),
        *reinterpret_cast<const unsigned int*>(&b2),
        *reinterpret_cast<const unsigned int*>(&b3));
  }
  float2 n0 = __bfloat1622float2(b0);
  float2 n1 = __bfloat1622float2(b1);
  float2 n2 = __bfloat1622float2(b2);
  float2 n3 = __bfloat1622float2(b3);
  v[0] = n0.x; v[1] = n0.y; v[2] = n1.x; v[3] = n1.y;
  v[4] = n2.x; v[5] = n2.y; v[6] = n3.x; v[7] = n3.y;
}

template <bool HAS_DELTA, bool WRITE_NORMED, bool W_IS_F32, bool HAS_SIGNS>
__global__ __launch_bounds__(128, 4) void sym8_add_rmsnorm_hadamard_quant(
    const nv_bfloat16* __restrict__ h_in,
    const nv_bfloat16* __restrict__ delta_in,
    const void* __restrict__ w_norm,
    const float* __restrict__ signs,
    nv_bfloat16* __restrict__ h_out,
    nv_bfloat16* __restrict__ normed_out,
    signed char* __restrict__ x_i8,
    float* __restrict__ s_x,
    float eps,
    int M,
    int H) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= M) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h8 = H >> 3;
  const int h16 = H >> 4;
  const size_t row_u4 = static_cast<size_t>(row) * static_cast<size_t>(h8);

  const uint4* h_u4 = reinterpret_cast<const uint4*>(h_in) + row_u4;
  [[maybe_unused]] const uint4* d_u4 = HAS_DELTA ? (reinterpret_cast<const uint4*>(delta_in) + row_u4) : nullptr;
  [[maybe_unused]] uint4* ho_u4 = HAS_DELTA ? (reinterpret_cast<uint4*>(h_out) + row_u4) : nullptr;
  [[maybe_unused]] uint4* no_u4 = WRITE_NORMED ? (reinterpret_cast<uint4*>(normed_out) + row_u4) : nullptr;

  extern __shared__ float s_buf[];
  float4* s_buf_f4 = reinterpret_cast<float4*>(s_buf);
  __shared__ float s_red[16];
  __shared__ int s_wbad[4];
  __shared__ float s_scalar;

  // Pass 1: optional BF16 residual add (with exact BF16 rounding), store h_out & s_buf,
  // and match add_rmsnorm.cu.inc's 512-thread (16-warp) sum_sq accumulation & reduction tree.
  float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
  for (int i = tid, g = 0; i < h8; i += 128, g = (g + 1) & 3) {
    uint4 ha = h_u4[i];
    const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&ha);
    float2 f0 = __bfloat1622float2(hp[0]);
    float2 f1 = __bfloat1622float2(hp[1]);
    float2 f2 = __bfloat1622float2(hp[2]);
    float2 f3 = __bfloat1622float2(hp[3]);
    if constexpr (HAS_DELTA) {
      uint4 da = d_u4[i];
      const __nv_bfloat162* dp = reinterpret_cast<const __nv_bfloat162*>(&da);
      float2 g0 = __bfloat1622float2(dp[0]);
      float2 g1 = __bfloat1622float2(dp[1]);
      float2 g2 = __bfloat1622float2(dp[2]);
      float2 g3 = __bfloat1622float2(dp[3]);
      __nv_bfloat162 b0 = __floats2bfloat162_rn(f0.x + g0.x, f0.y + g0.y);
      __nv_bfloat162 b1 = __floats2bfloat162_rn(f1.x + g1.x, f1.y + g1.y);
      __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x + g2.x, f2.y + g2.y);
      __nv_bfloat162 b3 = __floats2bfloat162_rn(f3.x + g3.x, f3.y + g3.y);
      ho_u4[i] = make_uint4(
          *reinterpret_cast<const unsigned int*>(&b0),
          *reinterpret_cast<const unsigned int*>(&b1),
          *reinterpret_cast<const unsigned int*>(&b2),
          *reinterpret_cast<const unsigned int*>(&b3));
      f0 = __bfloat1622float2(b0);
      f1 = __bfloat1622float2(b1);
      f2 = __bfloat1622float2(b2);
      f3 = __bfloat1622float2(b3);
    }
    s_buf_f4[2 * i] = make_float4(f0.x, f0.y, f1.x, f1.y);
    s_buf_f4[2 * i + 1] = make_float4(f2.x, f2.y, f3.x, f3.y);
    float s = (g == 0) ? sq0 : (g == 1) ? sq1 : (g == 2) ? sq2 : sq3;
    s += f0.x * f0.x;
    s += f0.y * f0.y;
    s += f1.x * f1.x;
    s += f1.y * f1.y;
    s += f2.x * f2.x;
    s += f2.y * f2.y;
    s += f3.x * f3.x;
    s += f3.y * f3.y;
    if (g == 0) sq0 = s;
    else if (g == 1) sq1 = s;
    else if (g == 2) sq2 = s;
    else sq3 = s;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq0 += __shfl_xor_sync(0xffffffffu, sq0, off);
    sq1 += __shfl_xor_sync(0xffffffffu, sq1, off);
    sq2 += __shfl_xor_sync(0xffffffffu, sq2, off);
    sq3 += __shfl_xor_sync(0xffffffffu, sq3, off);
  }
  if (lane == 0) {
    s_red[0 * 4 + warp] = sq0;
    s_red[1 * 4 + warp] = sq1;
    s_red[2 * 4 + warp] = sq2;
    s_red[3 * 4 + warp] = sq3;
  }
  __syncthreads();
  if (warp == 0) {
    float total = (lane < 16) ? s_red[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      total += __shfl_xor_sync(0xffffffffu, total, off);
    }
    if (lane == 0) {
      const float denom = total / static_cast<float>(H) + eps;
      s_scalar = (denom > 0.0f) ? rsqrtf(denom) : 0.0f;
    }
  }
  __syncthreads();
  const float inv_rms = s_scalar;

  // Pass 2: RMSNorm + optional write_normed + optional signs & in-place FWHT-1024 + amax
  float amax = 0.0f;
  int bad = 0;
  if constexpr (HAS_SIGNS) {
    const float4* signs_f4 = reinterpret_cast<const float4*>(signs);
    const int num_blk = H >> 10;
    constexpr float INV_SQRT_1024 = 0.03125f;
    for (int blk = 0; blk < num_blk; ++blk) {
      const int u_idx = blk * 128 + tid;
      float4 p0 = s_buf_f4[2 * u_idx];
      float4 p1 = s_buf_f4[2 * u_idx + 1];
      float hf[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      float v[8];
      sym8_rn_apply_norm8<WRITE_NORMED, W_IS_F32>(hf, w_norm, no_u4, u_idx, inv_rms, v);
      float4 s0 = signs_f4[2 * u_idx];
      float4 s1 = signs_f4[2 * u_idx + 1];
      v[0] *= s0.x; v[1] *= s0.y; v[2] *= s0.z; v[3] *= s0.w;
      v[4] *= s1.x; v[5] *= s1.y; v[6] *= s1.z; v[7] *= s1.w;

      float* s_blk = s_buf + blk * 1024;
      sym8_rn_fwht256_inplace(v, s_blk, tid, lane);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pos = tid + half * 128;
        float x0 = s_blk[0 * 256 + pos], x1 = s_blk[1 * 256 + pos];
        float x2 = s_blk[2 * 256 + pos], x3 = s_blk[3 * 256 + pos];
        float t0 = x0 + x1, t1 = x0 - x1, t2 = x2 + x3, t3 = x2 - x3;
        float y0 = (t0 + t2) * INV_SQRT_1024;
        float y1 = (t1 + t3) * INV_SQRT_1024;
        float y2 = (t0 - t2) * INV_SQRT_1024;
        float y3 = (t1 - t3) * INV_SQRT_1024;
        bad |= (!isfinite(y0)) | (!isfinite(y1)) | (!isfinite(y2)) | (!isfinite(y3));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
        s_blk[0 * 256 + pos] = y0;
        s_blk[1 * 256 + pos] = y1;
        s_blk[2 * 256 + pos] = y2;
        s_blk[3 * 256 + pos] = y3;
      }
    }
  } else {
    for (int i = tid; i < h8; i += 128) {
      float4 p0 = s_buf_f4[2 * i];
      float4 p1 = s_buf_f4[2 * i + 1];
      float hf[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      float v[8];
      sym8_rn_apply_norm8<WRITE_NORMED, W_IS_F32>(hf, w_norm, no_u4, i, inv_rms, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bad |= !isfinite(v[e]);
        amax = fmaxf(amax, fabsf(v[e]));
      }
      s_buf_f4[2 * i] = make_float4(v[0], v[1], v[2], v[3]);
      s_buf_f4[2 * i + 1] = make_float4(v[4], v[5], v[6], v[7]);
    }
  }

  // Row amax reduction -> per-token scale (matches sym8_hadamard.cu.inc bit-for-bit)
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  const int warp_bad = __any_sync(0xffffffffu, bad);
  if (lane == 0) {
    s_red[warp] = amax;
    s_wbad[warp] = warp_bad;
  }
  __syncthreads();
  if (tid < 4) {
    float m = s_red[tid];
    m = fmaxf(m, __shfl_xor_sync(0xfu, m, 1, 4));
    m = fmaxf(m, __shfl_xor_sync(0xfu, m, 2, 4));
    const int any_bad = __any_sync(0xfu, s_wbad[tid]);
    if (tid == 0) {
      const bool valid = !any_bad && (m > 0.0f) && isfinite(m);
      const float inv = valid ? (127.0f / m) : 0.0f;
      const bool ok = valid && isfinite(inv);
      s_scalar = ok ? inv : 0.0f;
      s_x[row] = ok ? (m * (1.0f / 127.0f)) : 0.0f;
    }
  }
  __syncthreads();

  // Pass 3: INT8 quantize + vectorized uint4 (16 int8s) store
  const float inv = s_scalar;
  const bool valid = (inv > 0.0f);
  uint4* xq_u4 = reinterpret_cast<uint4*>(x_i8) + static_cast<size_t>(row) * static_cast<size_t>(h16);
  for (int i = tid; i < h16; i += 128) {
    float4 f0 = s_buf_f4[4 * i];
    float4 f1 = s_buf_f4[4 * i + 1];
    float4 f2 = s_buf_f4[4 * i + 2];
    float4 f3 = s_buf_f4[4 * i + 3];
    xq_u4[i] = make_uint4(
        sym8_rn_pack4(f0, inv, valid),
        sym8_rn_pack4(f1, inv, valid),
        sym8_rn_pack4(f2, inv, valid),
        sym8_rn_pack4(f3, inv, valid));
  }
}
)cuda"
