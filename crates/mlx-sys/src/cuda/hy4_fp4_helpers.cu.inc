R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Shared Hy4 CUDA FP4 / FP8 / Q8 / SwiGLU / BF16 helpers.
// Dual-target PTX gating:
//   - Windows RTX Spark CUDA 13.4+ (sm_121a): PTX 9.2 cvt.rn.bf16x2.{e2m1x2,e4m3x2}
//   - Linux DGX Spark CUDA 13.0   (sm_121a): PTX 8.1 cvt.rn.f16x2.e4m3x2 + SMEM E2M1 LUT
//   - sm_80 fallback: pure bitwise software decode

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define HY4_HAS_HW_FP4_CVT 1
#else
#define HY4_HAS_HW_FP4_CVT 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))
#define HY4_HAS_HW_FP8_F16_CVT 1
#else
#define HY4_HAS_HW_FP8_F16_CVT 0
#endif

__device__ __forceinline__ unsigned int hy4_pack_bf16x2(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float hy4_decode_fp8_e4m3(unsigned int bits) {
  const unsigned int v = bits & 127u;
  const unsigned int sign_bit = (bits & 128u) << 8;
  const unsigned int u = (v << 7) | (((v + 1u) >> 7) << 14) | sign_bit;
  return __half2float(__ushort_as_half(static_cast<unsigned short>(u))) * 256.0f;
}

__device__ __forceinline__ float2 hy4_decode_fp8_e4m3x2_unscaled(unsigned int pair16) {
  const unsigned int b0 = pair16 & 0xFFu;
  const unsigned int b1 = (pair16 >> 8) & 0xFFu;
  const unsigned int v0 = b0 & 127u;
  const unsigned int v1 = b1 & 127u;
  const unsigned int h0 = (v0 << 7) | (((v0 + 1u) >> 7) << 14) | ((b0 & 128u) << 8);
  const unsigned int h1 = (v1 << 7) | (((v1 + 1u) >> 7) << 14) | ((b1 & 128u) << 8);
  const unsigned int h2 = h0 | (h1 << 16);
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

__device__ __forceinline__ float2 hy4_fast_e4m3x2_to_f2(unsigned short pair16) {
#if HY4_HAS_HW_FP4_CVT
  unsigned int bf2;
  asm("{ cvt.rn.bf16x2.e4m3x2 %0, %1; }" : "=r"(bf2) : "h"(pair16));
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
#elif HY4_HAS_HW_FP8_F16_CVT
  unsigned int h2;
  asm("{ cvt.rn.f16x2.e4m3x2 %0, %1; }" : "=r"(h2) : "h"(pair16));
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
#else
  const float2 f = hy4_decode_fp8_e4m3x2_unscaled(pair16);
  return make_float2(f.x * 256.0f, f.y * 256.0f);
#endif
}

__device__ __forceinline__ float hy4_fast_e4m3_to_f(unsigned int byte8) {
  return hy4_fast_e4m3x2_to_f2(static_cast<unsigned short>(byte8 & 0xFFu)).x;
}

// Exact 0-ULP 4-bit E2M1 nibble -> BF16 bits (scale 1.0).
__device__ __forceinline__ unsigned int hy4_e2m1_to_bf16(unsigned int nib) {
  const unsigned int m = nib & 7u;
  const unsigned int s = (nib & 8u) << 12;
  const unsigned int mag = (m >= 2u) ? (0x3F00u + (m << 6)) : (m ? 0x3F00u : 0u);
  return s | mag;
}

__device__ __forceinline__ unsigned int hy4_e2m1x2_to_bf16x2(unsigned int byte8) {
  return hy4_e2m1_to_bf16(byte8 & 0xFu) | (hy4_e2m1_to_bf16((byte8 >> 4) & 0xFu) << 16);
}

__device__ __forceinline__ unsigned int hy4_fast_e2m1x2_to_bf16x2(unsigned int byte8) {
#if HY4_HAS_HW_FP4_CVT
  unsigned int out;
  unsigned short tmp = static_cast<unsigned short>(byte8);
  asm("{ .reg .b8 b0, b1; mov.b16 {b0, b1}, %1; cvt.rn.bf16x2.e2m1x2 %0, b0; }"
      : "=r"(out) : "h"(tmp));
  return out;
#else
  return hy4_e2m1x2_to_bf16x2(byte8);
#endif
}

__device__ __forceinline__ unsigned int hy4_lut_e2m1x2_to_bf16x2(
    unsigned int byte8, const unsigned int* __restrict__ lut) {
#if HY4_HAS_HW_FP4_CVT
  (void)lut;
  return hy4_fast_e2m1x2_to_bf16x2(byte8);
#else
  return lut[byte8 & 0xFFu];
#endif
}

__device__ __forceinline__ float2 hy4_e2m1x2_to_f2(unsigned int byte8) {
  unsigned int bf2 = hy4_fast_e2m1x2_to_bf16x2(byte8);
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
}

// Asymmetric clamped SwiGLU for Hy4 routed experts:
//   L > 0: silu(min(g, L)) * clamp(u, -L, +L)
//   L <= 0: silu(g) * u
__device__ __forceinline__ float hy4_clamped_swiglu(float g, float u, float limit) {
  if (limit > 0.0f) {
    if (g > limit) g = limit;
    if (u > limit) u = limit;
    if (u < -limit) u = -limit;
  }
  float sig = 1.0f / (1.0f + expf(-g));
  return (g * sig) * u;
}

__device__ __forceinline__ uint2 hy4_ld_cs_u2(const uint2* ptr) {
  uint2 v;
  asm("ld.global.cs.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(ptr));
  return v;
}

__device__ __forceinline__ float hy4_warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  }
  return v;
}

__device__ __forceinline__ float hy4_warp_max(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  }
  return v;
}
)cuda"
