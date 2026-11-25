R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Inkling-2-Mini Grouped MoE Decode/Verify CUDA Kernels (M in 1..=16)
//   E = 128, top_k = 8, H = 2304, I = 1152, route_scale = 8.0, no shared expert
//   Modes: 0=NVFP4 (b4/g16), 1=SymQ8 (b8/g32), 2=AffineQ8 (b8/g32),
//          3=Q5_K (b5/g32/QK256, gate_up only), 4=MXFP8 (b8/g32)
//   Hybrid Splash + Qwen4 design:
//     cnt <= 2: 0-barrier streaming ld.global.cs.v4.u32 GEMV (FP32 / dp4a)
//     cnt >= 3: 2-stage cp.async.cg + Tensor Core MMA (bf16 / s8)
// ============================================================================

__device__ __forceinline__ unsigned int i2_pack_bf16x2(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float i2_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

// Exact bit-for-bit FP8 E4M3FN -> FP32 (matches inkling2_moe.metal:70-75).
__device__ __forceinline__ float i2_decode_fp8_e4m3(unsigned int bits) {
  const unsigned int v = bits & 127u;
  const unsigned int sign_bit = (bits & 128u) << 8;
  const unsigned int u = (v << 7) | (((v + 1u) >> 7) << 14) | sign_bit;
  return __half2float(__ushort_as_half(static_cast<unsigned short>(u))) * 256.0f;
}

__device__ __forceinline__ float i2_decode_fp8_e4m3_x16384(unsigned int bits) {
  return i2_decode_fp8_e4m3(bits) * 16384.0f;
}

// 2x FP8 E4M3FN -> float2 unscaled by 2^-8 (caller multiplies group scale by 256.0f).
__device__ __forceinline__ float2 i2_decode_fp8_e4m3x2_unscaled(unsigned int pair16) {
  const unsigned int b0 = pair16 & 0xFFu;
  const unsigned int b1 = (pair16 >> 8) & 0xFFu;
  const unsigned int v0 = b0 & 127u;
  const unsigned int v1 = b1 & 127u;
  const unsigned int h0 = (v0 << 7) | (((v0 + 1u) >> 7) << 14) | ((b0 & 128u) << 8);
  const unsigned int h1 = (v1 << 7) | (((v1 + 1u) >> 7) << 14) | ((b1 & 128u) << 8);
  const unsigned int h2 = h0 | (h1 << 16);
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

// PTX ISA 9.2 (CUDA 13.2+) Blackwell arch-specific direct FP4/FP8 -> bf16x2 conversions.
// Earlier toolchains (e.g. CUDA 13.0 on Blackwell) fall back to PTX 8.1 cvt.rn.f16x2.e4m3x2
// for FP8 and the SMEM E2M1 LUT for NVFP4 so NVRTC never emits unsupported PTX 9.2 ops.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define I2_HAS_HW_FP4_CVT 1
#else
#define I2_HAS_HW_FP4_CVT 0
#endif

// PTX ISA 8.1 (CUDA 12.1+) Ada/Hopper/Blackwell FP8 E4M3FN -> f16x2 conversion.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))
#define I2_HAS_HW_FP8_F16_CVT 1
#else
#define I2_HAS_HW_FP8_F16_CVT 0
#endif

// Exact 0-ULP 2x FP8 E4M3FN -> bf16x2 (scale 1.0) for BF16 Tensor Cores.
// Uses 1-instr Blackwell cvt.rn.bf16x2.e4m3x2 (CUDA 13.2+), Ada/Hopper/Blackwell cvt.rn.f16x2.e4m3x2, or SW fallback.
__device__ __forceinline__ unsigned int i2_decode_fp8_e4m3x2_bf16x2(unsigned int pair16) {
#if I2_HAS_HW_FP4_CVT
  unsigned int bf2;
  const unsigned short p16 = static_cast<unsigned short>(pair16);
  asm("{ cvt.rn.bf16x2.e4m3x2 %0, %1; }" : "=r"(bf2) : "h"(p16));
  return bf2;
#elif I2_HAS_HW_FP8_F16_CVT
  unsigned int h2;
  const unsigned short p16 = static_cast<unsigned short>(pair16);
  asm("{ cvt.rn.f16x2.e4m3x2 %0, %1; }" : "=r"(h2) : "h"(p16));
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
  return i2_pack_bf16x2(f.x, f.y);
#else
  const float2 f = i2_decode_fp8_e4m3x2_unscaled(pair16);
  return i2_pack_bf16x2(f.x * 256.0f, f.y * 256.0f);
#endif
}

// Exact UE8M0 scale decode (matches inkling2_moe.metal:116-119).
__device__ __forceinline__ float i2_decode_ue8m0(unsigned int b) {
  return __uint_as_float((b == 0u) ? 0x00400000u : (b << 23));
}

// Exact 0-ULP 4-bit E2M1 nibble -> BF16 (scale 1.0, zero subnormals).
__device__ __forceinline__ unsigned int i2_e2m1_to_bf16(unsigned int nib) {
  const unsigned int m = nib & 7u;
  const unsigned int s = (nib & 8u) << 12;
  const unsigned int mag = (m >= 2u) ? (0x3F00u + (m << 6)) : (m ? 0x3F00u : 0u);
  return s | mag;
}

__device__ __forceinline__ unsigned int i2_e2m1x2_to_bf16x2(unsigned int byte8) {
  return i2_e2m1_to_bf16(byte8 & 0xFu) | (i2_e2m1_to_bf16((byte8 >> 4) & 0xFu) << 16);
}

// Bit-exact 2x E2M1 -> bf16x2 (1-instr hardware PTX on Blackwell sm_100a/120a/121a).
__device__ __forceinline__ unsigned int i2_fast_e2m1x2_to_bf16x2(unsigned int byte8) {
#if I2_HAS_HW_FP4_CVT
  unsigned int out;
  unsigned short tmp = static_cast<unsigned short>(byte8);
  asm("{ .reg .b8 b0, b1; mov.b16 {b0, b1}, %1; cvt.rn.bf16x2.e2m1x2 %0, b0; }"
      : "=r"(out) : "h"(tmp));
  return out;
#else
  return i2_e2m1x2_to_bf16x2(byte8);
#endif
}

// Bit-exact 2x FP8 E4M3FN -> float2 (1-instr hardware PTX on Ada/Hopper/Blackwell).
__device__ __forceinline__ float2 i2_fast_e4m3x2_to_f2(unsigned short pair16) {
#if I2_HAS_HW_FP4_CVT
  unsigned int bf2;
  asm("{ cvt.rn.bf16x2.e4m3x2 %0, %1; }" : "=r"(bf2) : "h"(pair16));
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf2));
#elif I2_HAS_HW_FP8_F16_CVT
  unsigned int h2;
  asm("{ cvt.rn.f16x2.e4m3x2 %0, %1; }" : "=r"(h2) : "h"(pair16));
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
#else
  const float2 f = i2_decode_fp8_e4m3x2_unscaled(pair16);
  return make_float2(f.x * 256.0f, f.y * 256.0f);
#endif
}

__device__ __forceinline__ uint2 i2_ld_cs_u2(const uint2* ptr) {
  uint2 v;
  asm("ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
      : "=r"(v.x), "=r"(v.y)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ float4 i2_ld_cs_f4(const float4* ptr) {
  float4 v;
  asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];\n"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
      : "l"(ptr));
  return v;
}

// 2x E2M1 nibbles -> float2 unscaled by 2^-14 (matches inkling2_moe.metal:20-28).
__device__ __forceinline__ float2 i2_e2m1x2_to_f2_unscaled(unsigned int byte8) {
  const unsigned int h2 =
      ((byte8 & 0x0007u) << 9)  | ((byte8 & 0x0008u) << 12) |
      ((byte8 & 0x0070u) << 21) | ((byte8 & 0x0080u) << 24);
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

__device__ __forceinline__ float i2_read_f16_or_bf16(unsigned short raw, int is_bf16) {
  return is_bf16 ? __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(&raw))
                 : __half2float(__ushort_as_half(raw));
}

__device__ __forceinline__ unsigned int i2_unpack4_q5k(unsigned int u) {
  return (u & 0x0000001fu) | ((u & 0x000003e0u) << 3) |
         ((u & 0x00007c00u) << 6) | ((u & 0x000f8000u) << 9);
}

__device__ __forceinline__ void i2_load_pair_q5k(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  const int bit0 = t * 20;
  const int w0 = bit0 >> 5;
  const int sh0 = bit0 & 31;
  unsigned int u0 = seg[w0];
  if (sh0 > 12) u0 = __funnelshift_r(u0, seg[w0 + 1], sh0);
  else u0 >>= sh0;

  const int bit1 = 80 + t * 20;
  const int w1 = bit1 >> 5;
  const int sh1 = bit1 & 31;
  unsigned int u1 = seg[w1];
  if (sh1 > 12) u1 = __funnelshift_r(u1, seg[w1 + 1], sh1);
  else u1 >>= sh1;

  out0 = i2_unpack4_q5k(u0);
  out1 = i2_unpack4_q5k(u1);
}

__device__ __forceinline__ int i2_dp4a_q5k_32(
    const unsigned int* seg, const uint4& xv0, const uint4& xv1) {
  const unsigned int w0 = seg[0];
  const unsigned int w1 = seg[1];
  const unsigned int w2 = seg[2];
  const unsigned int w3 = seg[3];
  const unsigned int w4 = seg[4];
  int d = 0;
  d = ptx_dp4a_s32(i2_unpack4_q5k(w0),                          xv0.x, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k((w0 >> 20) | (w1 << 12)),     xv0.y, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k(w1 >> 8),                     xv0.z, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k((w1 >> 28) | (w2 << 4)),      xv0.w, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k((w2 >> 16) | (w3 << 16)),     xv1.x, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k(w3 >> 4),                     xv1.y, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k((w3 >> 24) | (w4 << 8)),      xv1.z, d);
  d = ptx_dp4a_s32(i2_unpack4_q5k(w4 >> 12),                    xv1.w, d);
  return d;
}

// Exact Inkling-2-Mini SiTU(4.0, 25.0) epilogue with BF16 pre-rounding.
__device__ __forceinline__ nv_bfloat16 i2_situ_epilogue(
    float acc_g, float acc_u, float outer_scale) {
  const float g = __bfloat162float(__float2bfloat16_rn(acc_g)) * outer_scale;
  const float u = __bfloat162float(__float2bfloat16_rn(acc_u)) * outer_scale;
  const float g_sc = 4.0f * tanhf(fminf(fmaxf(g * 0.25f, -20.0f), 20.0f));
  const float u_sc = 25.0f * tanhf(fminf(fmaxf(u * 0.04f, -20.0f), 20.0f));
  const float g_sig = 1.0f / (1.0f + expf(-g));
  return __float2bfloat16_rn(g_sc * g_sig * u_sc);
}
)cuda"
R"cuda(
// Unpack 8 quantized weights in uint2 into 4 bf16x2 regs for BF16 MMA (SymQ8/AffineQ8/MXFP8).
template <int MODE>
__device__ __forceinline__ void i2_unpack8_to_bf16x2(const uint2& w, unsigned int out[4]) {
  if constexpr (MODE == 1) {
    const unsigned int u0 = w.x ^ 0x80808080u;
    const unsigned int u1 = w.y ^ 0x80808080u;
    out[0] = i2_pack_bf16x2(float(static_cast<signed char>(u0)), float(static_cast<signed char>(u0 >> 8)));
    out[1] = i2_pack_bf16x2(float(static_cast<signed char>(u0 >> 16)), float(static_cast<signed char>(u0 >> 24)));
    out[2] = i2_pack_bf16x2(float(static_cast<signed char>(u1)), float(static_cast<signed char>(u1 >> 8)));
    out[3] = i2_pack_bf16x2(float(static_cast<signed char>(u1 >> 16)), float(static_cast<signed char>(u1 >> 24)));
  } else if constexpr (MODE == 2) {
    out[0] = i2_pack_bf16x2(float(w.x & 0xFFu), float((w.x >> 8) & 0xFFu));
    out[1] = i2_pack_bf16x2(float((w.x >> 16) & 0xFFu), float(w.x >> 24));
    out[2] = i2_pack_bf16x2(float(w.y & 0xFFu), float((w.y >> 8) & 0xFFu));
    out[3] = i2_pack_bf16x2(float((w.y >> 16) & 0xFFu), float(w.y >> 24));
  } else {
    out[0] = i2_decode_fp8_e4m3x2_bf16x2(w.x);
    out[1] = i2_decode_fp8_e4m3x2_bf16x2(w.x >> 16);
    out[2] = i2_decode_fp8_e4m3x2_bf16x2(w.y);
    out[3] = i2_decode_fp8_e4m3x2_bf16x2(w.y >> 16);
  }
}

// Streaming GEMV 16-elem dot helpers (1 or 2 tokens reusing decoded weights in registers!).
__device__ __forceinline__ void i2_dot_nvfp4_16(
    unsigned int wx, unsigned int wy,
    const float* x0, const float* x1, bool has_x1, float& d0, float& d1) {
  d0 = 0.0f;
  d1 = 0.0f;
  float2 f;
  f = i2_e2m1x2_to_f2_unscaled(wx);
  d0 = fmaf(f.x, x0[0], fmaf(f.y, x0[1], d0));
  if (has_x1) d1 = fmaf(f.x, x1[0], fmaf(f.y, x1[1], d1));
  f = i2_e2m1x2_to_f2_unscaled(wx >> 8);
  d0 = fmaf(f.x, x0[2], fmaf(f.y, x0[3], d0));
  if (has_x1) d1 = fmaf(f.x, x1[2], fmaf(f.y, x1[3], d1));
  f = i2_e2m1x2_to_f2_unscaled(wx >> 16);
  d0 = fmaf(f.x, x0[4], fmaf(f.y, x0[5], d0));
  if (has_x1) d1 = fmaf(f.x, x1[4], fmaf(f.y, x1[5], d1));
  f = i2_e2m1x2_to_f2_unscaled(wx >> 24);
  d0 = fmaf(f.x, x0[6], fmaf(f.y, x0[7], d0));
  if (has_x1) d1 = fmaf(f.x, x1[6], fmaf(f.y, x1[7], d1));
  f = i2_e2m1x2_to_f2_unscaled(wy);
  d0 = fmaf(f.x, x0[8], fmaf(f.y, x0[9], d0));
  if (has_x1) d1 = fmaf(f.x, x1[8], fmaf(f.y, x1[9], d1));
  f = i2_e2m1x2_to_f2_unscaled(wy >> 8);
  d0 = fmaf(f.x, x0[10], fmaf(f.y, x0[11], d0));
  if (has_x1) d1 = fmaf(f.x, x1[10], fmaf(f.y, x1[11], d1));
  f = i2_e2m1x2_to_f2_unscaled(wy >> 16);
  d0 = fmaf(f.x, x0[12], fmaf(f.y, x0[13], d0));
  if (has_x1) d1 = fmaf(f.x, x1[12], fmaf(f.y, x1[13], d1));
  f = i2_e2m1x2_to_f2_unscaled(wy >> 24);
  d0 = fmaf(f.x, x0[14], fmaf(f.y, x0[15], d0));
  if (has_x1) d1 = fmaf(f.x, x1[14], fmaf(f.y, x1[15], d1));
}

template <int MODE>
__device__ __forceinline__ void i2_dot_8b_16(
    const uint4& wv, const float* x0, const float* x1, bool has_x1, float& d0, float& d1) {
  d0 = 0.0f;
  d1 = 0.0f;
  const unsigned int words[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (MODE == 1) {
      const unsigned int u = words[i] ^ 0x80808080u;
      const float f0 = float(static_cast<signed char>(u));
      const float f1 = float(static_cast<signed char>(u >> 8));
      const float f2 = float(static_cast<signed char>(u >> 16));
      const float f3 = float(static_cast<signed char>(u >> 24));
      d0 = fmaf(f0, x0[4 * i], fmaf(f1, x0[4 * i + 1], fmaf(f2, x0[4 * i + 2], fmaf(f3, x0[4 * i + 3], d0))));
      if (has_x1) {
        d1 = fmaf(f0, x1[4 * i], fmaf(f1, x1[4 * i + 1], fmaf(f2, x1[4 * i + 2], fmaf(f3, x1[4 * i + 3], d1))));
      }
    } else if constexpr (MODE == 2) {
      const unsigned int u = words[i];
      const float f0 = float(u & 0xFFu);
      const float f1 = float((u >> 8) & 0xFFu);
      const float f2 = float((u >> 16) & 0xFFu);
      const float f3 = float(u >> 24);
      d0 = fmaf(f0, x0[4 * i], fmaf(f1, x0[4 * i + 1], fmaf(f2, x0[4 * i + 2], fmaf(f3, x0[4 * i + 3], d0))));
      if (has_x1) {
        d1 = fmaf(f0, x1[4 * i], fmaf(f1, x1[4 * i + 1], fmaf(f2, x1[4 * i + 2], fmaf(f3, x1[4 * i + 3], d1))));
      }
    } else {
      const float2 p0 = i2_decode_fp8_e4m3x2_unscaled(words[i]);
      const float2 p1 = i2_decode_fp8_e4m3x2_unscaled(words[i] >> 16);
      d0 = fmaf(p0.x, x0[4 * i], fmaf(p0.y, x0[4 * i + 1], fmaf(p1.x, x0[4 * i + 2], fmaf(p1.y, x0[4 * i + 3], d0))));
      if (has_x1) {
        d1 = fmaf(p0.x, x1[4 * i], fmaf(p0.y, x1[4 * i + 1], fmaf(p1.x, x1[4 * i + 2], fmaf(p1.y, x1[4 * i + 3], d1))));
      }
    }
  }
}

__device__ __forceinline__ float i2_bf16_u4_to_f16(const uint4& a, const uint4& b, float out[16]) {
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f0 = __bfloat1622float2(pa[j]);
    const float2 f1 = __bfloat1622float2(pb[j]);
    out[2 * j] = f0.x;
    out[2 * j + 1] = f0.y;
    out[8 + 2 * j] = f1.x;
    out[8 + 2 * j + 1] = f1.y;
    sum += (f0.x + f0.y) + (f1.x + f1.y);
  }
  return sum;
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 1. Fused Exact-FP32 Router GEMV + Top-8 Routing + Grouping (+ Q5_K X Quant)
//    Eliminates the 48x 1-CTA route_plan launch bubble via GPU-wide atomic
//    arrival barrier (last CTA runs 3-barrier 0-stack route_plan in-kernel).
//    Grid: dim3(128, 1, 1), Block: dim3(288, 1, 1) -> 9 warps/expert,
//    576 float4 / 288 = 2 exact coalesced ld.global.cs.v4.f32 sweeps!
//    Transposed logits[128, 16] avoids cross-CTA L2 sector false sharing.
// ---------------------------------------------------------------------------
__global__ void inkling2_moe_zero_u32(unsigned int* __restrict__ ptr) {
  ptr[0] = 0u;
}

template <bool QUANT_X_Q5K>
__global__ __launch_bounds__(288, 3) void inkling2_moe_router_plan_few(
    const nv_bfloat16* __restrict__ x,
    const float* __restrict__ router_w,
    const float* __restrict__ router_bias,
    const float* __restrict__ dn_outer_scale,
    float* __restrict__ logits,
    float* __restrict__ gamma,
    int* __restrict__ unique_experts,
    int* __restrict__ bucket_offsets,
    int* __restrict__ pair_indices,
    int* __restrict__ num_unique,
    unsigned int* __restrict__ arrive_counter,
    signed char* __restrict__ xq,
    float* __restrict__ x_scales,
    float* __restrict__ x_sums,
    int M,
    float route_scale) {
  const int e = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if constexpr (QUANT_X_Q5K) {
    constexpr int GROUPS_PER_ROW = 72; // 2304 / 32
    // M * 72 * 8 = M * 576 is always a multiple of 32 (warp-uniform predicate).
    const int gtid = e * 288 + tid;
    if (gtid < M * (GROUPS_PER_ROW * 8)) {
      const int grp = gtid >> 3;
      const int sub = gtid & 7;
      const int row = grp / GROUPS_PER_ROW;
      const int s = grp - row * GROUPS_PER_ROW;
      const uint2 raw = *reinterpret_cast<const uint2*>(
          x + static_cast<size_t>(row) * 2304 + s * 32 + sub * 4);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
      const float2 f0 = __bfloat1622float2(pairs[0]);
      const float2 f1 = __bfloat1622float2(pairs[1]);
      float v[4] = {f0.x, f0.y, f1.x, f1.y};
      float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
      for (int off = 1; off < 8; off <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 8));
      }
      const bool valid = (amax > 0.0f) && isfinite(amax);
      const float inv = valid ? (127.0f / amax) : 0.0f;
      int q[4];
      int qsum = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        q[j] = valid ? static_cast<int>(rintf(v[j] * inv)) : 0;
        qsum += q[j];
      }
#pragma unroll
      for (int off = 1; off < 8; off <<= 1) {
        qsum += __shfl_xor_sync(0xffffffffu, qsum, off, 8);
      }
      *reinterpret_cast<unsigned int*>(xq + static_cast<size_t>(grp) * 32 + sub * 4) =
          (static_cast<unsigned int>(q[0]) & 0xffu) |
          ((static_cast<unsigned int>(q[1]) & 0xffu) << 8) |
          ((static_cast<unsigned int>(q[2]) & 0xffu) << 16) |
          ((static_cast<unsigned int>(q[3]) & 0xffu) << 24);
      if (sub == 0) {
        const float d = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
        x_scales[grp] = d;
        x_sums[grp] = d * static_cast<float>(qsum);
      }
    }
  }

  const uint2* x_u2 = reinterpret_cast<const uint2*>(x);
  const float4* erow = reinterpret_cast<const float4*>(router_w) + static_cast<size_t>(e) * 576;
  const float4 w0 = i2_ld_cs_f4(erow + tid);
  const float4 w1 = i2_ld_cs_f4(erow + 288 + tid);

  __shared__ float s_red[9][16];
  auto dot4 = [](const uint2& xv, const float4& wv, float a) -> float {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const float2 f0 = __bfloat1622float2(p[0]);
    const float2 f1 = __bfloat1622float2(p[1]);
    return fmaf(f0.x, wv.x, fmaf(f0.y, wv.y, fmaf(f1.x, wv.z, fmaf(f1.y, wv.w, a))));
  };

  int m = 0;
  for (; m + 3 < M; m += 4) {
    const uint2* x0 = x_u2 + static_cast<size_t>(m + 0) * 576;
    const uint2* x1 = x_u2 + static_cast<size_t>(m + 1) * 576;
    const uint2* x2 = x_u2 + static_cast<size_t>(m + 2) * 576;
    const uint2* x3 = x_u2 + static_cast<size_t>(m + 3) * 576;
    float v0 = dot4(x0[288 + tid], w1, dot4(x0[tid], w0, 0.0f));
    float v1 = dot4(x1[288 + tid], w1, dot4(x1[tid], w0, 0.0f));
    float v2 = dot4(x2[288 + tid], w1, dot4(x2[tid], w0, 0.0f));
    float v3 = dot4(x3[288 + tid], w1, dot4(x3[tid], w0, 0.0f));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v0 += __shfl_xor_sync(0xffffffffu, v0, off);
      v1 += __shfl_xor_sync(0xffffffffu, v1, off);
      v2 += __shfl_xor_sync(0xffffffffu, v2, off);
      v3 += __shfl_xor_sync(0xffffffffu, v3, off);
    }
    if (lane == 0) {
      s_red[warp][m + 0] = v0;
      s_red[warp][m + 1] = v1;
      s_red[warp][m + 2] = v2;
      s_red[warp][m + 3] = v3;
    }
  }
  for (; m < M; ++m) {
    const uint2* xm = x_u2 + static_cast<size_t>(m) * 576;
    float v = dot4(xm[288 + tid], w1, dot4(xm[tid], w0, 0.0f));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0) s_red[warp][m] = v;
  }
  __syncthreads();

  if (tid < M) {
    float tot = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 9; ++ww) tot += s_red[ww][tid];
    logits[e * 16 + tid] = tot;
  }

  // GPU-wide release-acquire arrival barrier: last of the 128 CTAs resets counter & runs route_plan.
  __syncthreads();
  __shared__ int s_is_last;
  if (tid == 0) {
    unsigned int prev = 0u;
    asm volatile("atom.acq_rel.gpu.add.u32 %0, [%1], 1;"
                 : "=r"(prev)
                 : "l"(arrive_counter)
                 : "memory");
    const int last = (prev == 127u) ? 1 : 0;
    if (last) {
      asm volatile("st.relaxed.gpu.u32 [%0], 0;" :: "l"(arrive_counter) : "memory");
    }
    s_is_last = last;
  }
  __syncthreads();
  if (!s_is_last) return;

  const int P = M * 8;
  __shared__ int s_ids[128];
  __shared__ int s_cnt[128];
  __shared__ int s_start[128];

  if (tid < 128) s_cnt[tid] = 0;

  for (int t = warp; t < M; t += 9) {
    float lp0, lp1, lp2, lp3;
    float ls0, ls1, ls2, ls3;
    int id0 = lane, id1 = lane + 32, id2 = lane + 64, id3 = lane + 96;
    auto load_e = [&](int ei, float& lp, float& ls) {
      // __ldcg bypasses stale L1 lines on reused workspace buffers -> L2 coherent.
      const float lv = __ldcg(&logits[ei * 16 + t]);
      const float p = 1.0f / (1.0f + expf(-lv));
      const float b = router_bias ? router_bias[ei] : 0.0f;
      lp = p;
      ls = p + b;
    };
    load_e(id0, lp0, ls0);
    load_e(id1, lp1, ls1);
    load_e(id2, lp2, ls2);
    load_e(id3, lp3, ls3);

    int my_id = 0;
    float my_p = 0.0f;
    float sum_p = 0.0f;

#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float best_s = -1e38f;
      int best_i = -1;
      float best_p = 0.0f;
      auto cmp = [&](int cid, float cs, float cp) {
        if (cid >= 0 && (best_i < 0 || cs > best_s || (cs == best_s && cid < best_i))) {
          best_s = cs;
          best_i = cid;
          best_p = cp;
        }
      };
      cmp(id0, ls0, lp0);
      cmp(id1, ls1, lp1);
      cmp(id2, ls2, lp2);
      cmp(id3, ls3, lp3);
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        const float os = __shfl_xor_sync(0xffffffffu, best_s, mask);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
        const float op = __shfl_xor_sync(0xffffffffu, best_p, mask);
        if (oi >= 0 && (best_i < 0 || os > best_s || (os == best_s && oi < best_i))) {
          best_s = os;
          best_i = oi;
          best_p = op;
        }
      }
      if (id0 == best_i) id0 = -1;
      if (id1 == best_i) id1 = -1;
      if (id2 == best_i) id2 = -1;
      if (id3 == best_i) id3 = -1;
      if (lane == k) {
        my_id = best_i;
        my_p = best_p;
      }
      sum_p += best_p;
    }

    if (lane < 8) {
      const float denom = sum_p + 1e-20f;
      float w = (my_p / denom) * route_scale;
      if (dn_outer_scale) w *= dn_outer_scale[my_id];
      s_ids[t * 8 + lane] = my_id;
      gamma[t * 8 + lane] = w;
    }
  }
  __syncthreads();

  if (tid < P) {
    atomicAdd(&s_cnt[s_ids[tid]], 1);
  }
  __syncthreads();

  // Single-warp prefix scan over all 128 experts (4 per lane -> 0 extra barriers).
  if (warp == 0) {
    const int e0 = lane * 4;
    const int c0 = s_cnt[e0 + 0];
    const int c1 = s_cnt[e0 + 1];
    const int c2 = s_cnt[e0 + 2];
    const int c3 = s_cnt[e0 + 3];
    const int u0 = (c0 > 0) ? 1 : 0;
    const int u1 = (c1 > 0) ? 1 : 0;
    const int u2 = (c2 > 0) ? 1 : 0;
    const int u3 = (c3 > 0) ? 1 : 0;
    const int csum = (c0 + c1) + (c2 + c3);
    const int usum = (u0 + u1) + (u2 + u3);
    int cp = csum;
    int up = usum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int nc = __shfl_up_sync(0xffffffffu, cp, off);
      const int nu = __shfl_up_sync(0xffffffffu, up, off);
      if (lane >= off) {
        cp += nc;
        up += nu;
      }
    }
    int st = cp - csum;
    int rk = up - usum;
    auto emit = [&](int ei, int ci, int ui) {
      s_start[ei] = st;
      if (ui) {
        unique_experts[rk] = ei;
        bucket_offsets[rk] = st;
      }
      st += ci;
      rk += ui;
    };
    emit(e0 + 0, c0, u0);
    emit(e0 + 1, c1, u1);
    emit(e0 + 2, c2, u2);
    emit(e0 + 3, c3, u3);
    if (lane == 31) {
      num_unique[0] = rk;
      bucket_offsets[rk] = st;
    }
  }
  __syncthreads();

  if (tid < P) {
    const int ei = s_ids[tid];
    int rank = 0;
    if (s_cnt[ei] > 1) {
      for (int j = 0; j < tid; ++j) {
        rank += (s_ids[j] == ei) ? 1 : 0;
      }
    }
    pair_indices[s_start[ei] + rank] = tid;
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2A. Hybrid Grouped Gate + Up + SiTU(4, 25) for NVFP4 / SymQ8 / AffineQ8 / MXFP8
//     MODE: 0=NVFP4, 1=SymQ8, 2=AffineQ8, 4=MXFP8
//     M_BUCKET: 0 = M<=2 (pure streaming GEMV), 1 = M<=8 (hybrid ALLOW_HI=false),
//               2 = M<=16 (hybrid ALLOW_HI=true)
//     Grid: dim3(72, M * 8), Block: dim3(32, 4) -> BN = 16, K = 2304, I = 1152
// ---------------------------------------------------------------------------
template <int MODE, bool INTERLEAVED, int M_BUCKET>
__global__ __launch_bounds__(128, M_BUCKET == 0 ? 6 : (M_BUCKET == 1 ? 4 : 3))
void inkling2_moe_gate_up_situ_few(
    const unsigned int* __restrict__ gu_w,
    const void* __restrict__ gu_scales,
    const void* __restrict__ gu_biases,
    const float* __restrict__ gu_outer_scale,
    const nv_bfloat16* __restrict__ x,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid,
    int sb_is_bf16) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  const int row0 = blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  struct StageMma {
    alignas(16) uint4 w[16][MODE == 0 ? 8 : 16];
    alignas(16) unsigned short sc[16][8]; // 16B/row: NVFP4=16 u8, Q8=8 u16, MXFP8=8 u8
    alignas(16) unsigned short bi[MODE == 2 ? 16 : 1][8];
  };
  union SharedGu {
    alignas(16) uint4 x_sh[2][2][144]; // 9,216 B (2 tokens x 2304 BF16, 0-conflict halves)
    StageMma pipe[M_BUCKET > 0 ? 2 : 1];
    struct {
      float gate[4][16][17];
      float up[4][16][17];
    } red;
  };
  __shared__ SharedGu smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    const int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p >> 3;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }
  __syncthreads();

  // ---- Fast Branch: M_BUCKET == 0 -> 0-barrier streaming ld.global.cs.v4.u32 GEMV ----
  if constexpr (M_BUCKET == 0) {
    const bool has_t1 = (cnt == 2);
    const int tok0 = s_tok[0];
    const int tok1 = has_t1 ? s_tok[1] : 0;
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
    for (int i = tid; i < 288; i += 128) {
      smem.x_sh[0][i & 1][i >> 1] = x_u4[static_cast<size_t>(tok0) * 288 + i];
      if (has_t1) {
        smem.x_sh[1][i & 1][i >> 1] = x_u4[static_cast<size_t>(tok1) * 288 + i];
      }
    }
    __syncthreads();

    const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
    const int pair0 = s_pair[0];
    const int pair1 = has_t1 ? s_pair[1] : 0;

#pragma unroll 1
    for (int r = 0; r < 4; ++r) {
      const int col = row0 + warp * 4 + r;
      const size_t g_idx = exp_base + (INTERLEAVED ? (2 * col) : col);
      const size_t u_idx_r = exp_base + (INTERLEAVED ? (2 * col + 1) : (I + col));
      float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;

      if constexpr (MODE == 0) {
        const uint4* gw = w_u4 + g_idx * 72;
        const uint4* uw = w_u4 + u_idx_r * 72;
        const unsigned short* gs = reinterpret_cast<const unsigned short*>(gu_scales) + g_idx * 72;
        const unsigned short* us = reinterpret_cast<const unsigned short*>(gu_scales) + u_idx_r * 72;
#pragma unroll 1
        for (int u = lane; u < 72; u += 32) {
          const uint4 gwv = ptx_ld_cs_u4(gw + u);
          const uint4 uwv = ptx_ld_cs_u4(uw + u);
          const unsigned short gsp = gs[u];
          const unsigned short usp = us[u];

          float xf0[16], xf1[16];
          i2_bf16_u4_to_f16(smem.x_sh[0][0][u * 2], smem.x_sh[0][1][u * 2], xf0);
          if (has_t1) i2_bf16_u4_to_f16(smem.x_sh[1][0][u * 2], smem.x_sh[1][1][u * 2], xf1);
          float d0, d1;
          const float gsl = i2_decode_fp8_e4m3_x16384(gsp & 0xFFu);
          const float usl = i2_decode_fp8_e4m3_x16384(usp & 0xFFu);
          i2_dot_nvfp4_16(gwv.x, gwv.y, xf0, xf1, has_t1, d0, d1);
          g0 = fmaf(gsl, d0, g0);
          if (has_t1) g1 = fmaf(gsl, d1, g1);
          i2_dot_nvfp4_16(uwv.x, uwv.y, xf0, xf1, has_t1, d0, d1);
          u0 = fmaf(usl, d0, u0);
          if (has_t1) u1 = fmaf(usl, d1, u1);

          i2_bf16_u4_to_f16(smem.x_sh[0][0][u * 2 + 1], smem.x_sh[0][1][u * 2 + 1], xf0);
          if (has_t1) i2_bf16_u4_to_f16(smem.x_sh[1][0][u * 2 + 1], smem.x_sh[1][1][u * 2 + 1], xf1);
          const float gsh = i2_decode_fp8_e4m3_x16384(gsp >> 8);
          const float ush = i2_decode_fp8_e4m3_x16384(usp >> 8);
          i2_dot_nvfp4_16(gwv.z, gwv.w, xf0, xf1, has_t1, d0, d1);
          g0 = fmaf(gsh, d0, g0);
          if (has_t1) g1 = fmaf(gsh, d1, g1);
          i2_dot_nvfp4_16(uwv.z, uwv.w, xf0, xf1, has_t1, d0, d1);
          u0 = fmaf(ush, d0, u0);
          if (has_t1) u1 = fmaf(ush, d1, u1);
        }
      } else {
        const uint4* gw = w_u4 + g_idx * 144;
        const uint4* uw = w_u4 + u_idx_r * 144;
#pragma unroll 1
        for (int u = lane; u < 144; u += 32) {
          const uint4 gwv = ptx_ld_cs_u4(gw + u);
          const uint4 uwv = ptx_ld_cs_u4(uw + u);
          float xf0[16], xf1[16];
          const float xs0 = i2_bf16_u4_to_f16(smem.x_sh[0][0][u], smem.x_sh[0][1][u], xf0);
          const float xs1 = has_t1 ? i2_bf16_u4_to_f16(smem.x_sh[1][0][u], smem.x_sh[1][1][u], xf1) : 0.0f;
          const int grp = u >> 1;
          float gd0, gd1, ud0, ud1;
          i2_dot_8b_16<MODE>(gwv, xf0, xf1, has_t1, gd0, gd1);
          i2_dot_8b_16<MODE>(uwv, xf0, xf1, has_t1, ud0, ud1);
          if constexpr (MODE == 1) {
            const unsigned short* gs = reinterpret_cast<const unsigned short*>(gu_scales) + g_idx * 72;
            const unsigned short* us = reinterpret_cast<const unsigned short*>(gu_scales) + u_idx_r * 72;
            const float gsc = i2_read_f16_or_bf16(gs[grp], sb_is_bf16);
            const float usc = i2_read_f16_or_bf16(us[grp], sb_is_bf16);
            g0 = fmaf(gsc, gd0, g0);
            u0 = fmaf(usc, ud0, u0);
            if (has_t1) { g1 = fmaf(gsc, gd1, g1); u1 = fmaf(usc, ud1, u1); }
          } else if constexpr (MODE == 2) {
            const unsigned short* gs = reinterpret_cast<const unsigned short*>(gu_scales) + g_idx * 72;
            const unsigned short* gb = reinterpret_cast<const unsigned short*>(gu_biases) + g_idx * 72;
            const unsigned short* us = reinterpret_cast<const unsigned short*>(gu_scales) + u_idx_r * 72;
            const unsigned short* ub = reinterpret_cast<const unsigned short*>(gu_biases) + u_idx_r * 72;
            const float gsc = i2_read_f16_or_bf16(gs[grp], sb_is_bf16);
            const float gbi = i2_read_f16_or_bf16(gb[grp], sb_is_bf16);
            const float usc = i2_read_f16_or_bf16(us[grp], sb_is_bf16);
            const float ubi = i2_read_f16_or_bf16(ub[grp], sb_is_bf16);
            g0 = fmaf(gsc, gd0, fmaf(gbi, xs0, g0));
            u0 = fmaf(usc, ud0, fmaf(ubi, xs0, u0));
            if (has_t1) {
              g1 = fmaf(gsc, gd1, fmaf(gbi, xs1, g1));
              u1 = fmaf(usc, ud1, fmaf(ubi, xs1, u1));
            }
          } else {
            const unsigned char* gs = reinterpret_cast<const unsigned char*>(gu_scales) + g_idx * 72;
            const unsigned char* us = reinterpret_cast<const unsigned char*>(gu_scales) + u_idx_r * 72;
            const float gsc = i2_decode_ue8m0(gs[grp]) * 256.0f;
            const float usc = i2_decode_ue8m0(us[grp]) * 256.0f;
            g0 = fmaf(gsc, gd0, g0);
            u0 = fmaf(usc, ud0, u0);
            if (has_t1) { g1 = fmaf(gsc, gd1, g1); u1 = fmaf(usc, ud1, u1); }
          }
        }
      }

#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        g0 += __shfl_xor_sync(0xffffffffu, g0, off);
        u0 += __shfl_xor_sync(0xffffffffu, u0, off);
        if (has_t1) {
          g1 += __shfl_xor_sync(0xffffffffu, g1, off);
          u1 += __shfl_xor_sync(0xffffffffu, u1, off);
        }
      }
      if (lane == 0) {
        mid[static_cast<size_t>(pair0) * I + col] = i2_situ_epilogue(g0, u0, outer_s);
        if (has_t1) {
          mid[static_cast<size_t>(pair1) * I + col] = i2_situ_epilogue(g1, u1, outer_s);
        }
      }
    }
    return;
  }
)cuda"
R"cuda(
  // ---- Hot Branch: cnt >= 3 (3..16) -> 2-stage cp.async.cg + mma.m16n8k16.bf16 ----
  if constexpr (M_BUCKET > 0) {
    constexpr bool ALLOW_HI = (M_BUCKET == 2);
    const int g = lane >> 2;
    const int t = lane & 3;
    const int tok_g_lo = s_tok[g];
    const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
    const bool do_hi = ALLOW_HI && (cnt > 8);
    (void)tok_g_hi;
    (void)do_hi;

    const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
    auto row_idx_of = [&](int r, bool is_up) -> size_t {
      return exp_base + (INTERLEAVED ? (2 * (row0 + r) + (is_up ? 1 : 0)) : ((is_up ? I : 0) + row0 + r));
    };

    auto load_chunk = [&](int st, int chunk, bool is_up) {
      if constexpr (MODE == 0) {
        const int r = tid >> 3;
        const int c = tid & 7;
        const size_t ri = row_idx_of(r, is_up);
        ptx_cp_async16_cg(&smem.pipe[st].w[r][c ^ (r & 7)], w_u4 + ri * 72 + chunk * 8 + c);
        if (tid < 16) {
          const size_t sr = row_idx_of(tid, is_up);
          ptx_cp_async16_cg(smem.pipe[st].sc[tid],
                            reinterpret_cast<const uint4*>(gu_scales) + sr * 9 + chunk);
        }
      } else {
        const int r = tid >> 3;
        const int c0 = (tid & 7) * 2;
        const size_t ri = row_idx_of(r, is_up);
        const uint4* wp = w_u4 + ri * 144 + chunk * 16 + c0;
        ptx_cp_async16_cg(&smem.pipe[st].w[r][c0 ^ r], wp);
        ptx_cp_async16_cg(&smem.pipe[st].w[r][(c0 + 1) ^ r], wp + 1);
        if (tid < 16) {
          const size_t sr = row_idx_of(tid, is_up);
          if constexpr (MODE == 4) {
            ptx_cp_async8_ca(smem.pipe[st].sc[tid],
                             reinterpret_cast<const unsigned char*>(gu_scales) + sr * 72 + chunk * 8);
          } else {
            ptx_cp_async16_cg(smem.pipe[st].sc[tid],
                              reinterpret_cast<const uint4*>(gu_scales) + sr * 9 + chunk);
          }
        } else if constexpr (MODE == 2) {
          if (tid < 32) {
            const size_t sr = row_idx_of(tid - 16, is_up);
            ptx_cp_async16_cg(smem.pipe[st].bi[tid - 16],
                              reinterpret_cast<const uint4*>(gu_biases) + sr * 9 + chunk);
          }
        }
      }
      ptx_cp_commit();
    };

    float g_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float g_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    load_chunk(0, 0, false);
#pragma unroll 1
    for (int step = 0; step < 18; ++step) {
      const int cur = step & 1;
      const int nxt = cur ^ 1;
      if (step + 1 < 18) {
        load_chunk(nxt, (step + 1) >> 1, ((step + 1) & 1) != 0);
        ptx_cp_wait<1>();
      } else {
        ptx_cp_wait<0>();
      }
      __syncthreads();

      const int chunk = step >> 1;
      const bool is_up = (step & 1) != 0;
      float* t_lo = is_up ? u_tot_lo : g_tot_lo;
      float* t_hi = is_up ? u_tot_hi : g_tot_hi;
      (void)t_hi;

      if constexpr (MODE == 0) {
        const unsigned char* sc_r0 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g]);
        const unsigned char* sc_r1 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g + 8]);
#pragma unroll
        for (int kg = 0; kg < 4; ++kg) {
          const int lg = warp * 4 + kg;
          const int grp16 = chunk * 16 + lg;
          const int col_u4 = lg >> 1;
          const int sub_u16 = (lg & 1) * 4 + t;
          const unsigned short w0 = reinterpret_cast<const unsigned short*>(
              &smem.pipe[cur].w[g][col_u4 ^ g])[sub_u16];
          const unsigned short w1 = reinterpret_cast<const unsigned short*>(
              &smem.pipe[cur].w[g + 8][col_u4 ^ g])[sub_u16];
          const unsigned int a0 = i2_e2m1x2_to_bf16x2(w0 & 0xFFu);
          const unsigned int a2 = i2_e2m1x2_to_bf16x2(w0 >> 8);
          const unsigned int a1 = i2_e2m1x2_to_bf16x2(w1 & 0xFFu);
          const unsigned int a3 = i2_e2m1x2_to_bf16x2(w1 >> 8);

          const uint2 xv0 = (tok_g_lo >= 0) ? reinterpret_cast<const uint2*>(
              x + static_cast<size_t>(tok_g_lo) * 2304 + grp16 * 16)[t] : make_uint2(0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
          const float sc0 = i2_decode_fp8_e4m3(sc_r0[lg]);
          const float sc1 = i2_decode_fp8_e4m3(sc_r1[lg]);
          t_lo[0] = fmaf(sc0, d0, t_lo[0]);
          t_lo[1] = fmaf(sc0, d1, t_lo[1]);
          t_lo[2] = fmaf(sc1, d2, t_lo[2]);
          t_lo[3] = fmaf(sc1, d3, t_lo[3]);
          if constexpr (ALLOW_HI) {
            if (do_hi) {
              const uint2 xv1 = (tok_g_hi >= 0) ? reinterpret_cast<const uint2*>(
                  x + static_cast<size_t>(tok_g_hi) * 2304 + grp16 * 16)[t] : make_uint2(0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
              t_hi[0] = fmaf(sc0, h0, t_hi[0]);
              t_hi[1] = fmaf(sc0, h1, t_hi[1]);
              t_hi[2] = fmaf(sc1, h2, t_hi[2]);
              t_hi[3] = fmaf(sc1, h3, t_hi[3]);
            }
          }
        }
      } else {
        const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
#pragma unroll
        for (int kg = 0; kg < 2; ++kg) {
          const int lg = warp * 2 + kg;
          const int grp32 = chunk * 8 + lg;
          const int col_u4 = lg * 2 + (t >> 1);
          const int half_u2 = t & 1;
          const uint2 w0 = reinterpret_cast<const uint2*>(
              &smem.pipe[cur].w[g][col_u4 ^ g])[half_u2];
          const uint2 w1 = reinterpret_cast<const uint2*>(
              &smem.pipe[cur].w[g + 8][col_u4 ^ (g + 8)])[half_u2];
          unsigned int wb0[4], wb1[4];
          i2_unpack8_to_bf16x2<MODE>(w0, wb0);
          i2_unpack8_to_bf16x2<MODE>(w1, wb1);

          const uint4 xv0 = (tok_g_lo >= 0)
              ? x_u4[static_cast<size_t>(tok_g_lo) * 288 + grp32 * 4 + t]
              : make_uint4(0u, 0u, 0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
          ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

          float sc0, sc1;
          if constexpr (MODE == 4) {
            const unsigned char* sc_r0 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g]);
            const unsigned char* sc_r1 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g + 8]);
            sc0 = i2_decode_ue8m0(sc_r0[lg]);
            sc1 = i2_decode_ue8m0(sc_r1[lg]);
          } else {
            sc0 = i2_read_f16_or_bf16(smem.pipe[cur].sc[g][lg], sb_is_bf16);
            sc1 = i2_read_f16_or_bf16(smem.pipe[cur].sc[g + 8][lg], sb_is_bf16);
          }

          if constexpr (MODE == 2) {
            float my_s0 = (tok_g_lo >= 0) ? i2_sum_bf16x8(xv0) : 0.0f;
            my_s0 += __shfl_xor_sync(0xffffffffu, my_s0, 1);
            my_s0 += __shfl_xor_sync(0xffffffffu, my_s0, 2);
            const float xs00 = __shfl_sync(0xffffffffu, my_s0, (2 * t) * 4);
            const float xs01 = __shfl_sync(0xffffffffu, my_s0, (2 * t + 1) * 4);
            const float bi0 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g][lg], sb_is_bf16);
            const float bi1 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g + 8][lg], sb_is_bf16);
            t_lo[0] = fmaf(sc0, d0, fmaf(bi0, xs00, t_lo[0]));
            t_lo[1] = fmaf(sc0, d1, fmaf(bi0, xs01, t_lo[1]));
            t_lo[2] = fmaf(sc1, d2, fmaf(bi1, xs00, t_lo[2]));
            t_lo[3] = fmaf(sc1, d3, fmaf(bi1, xs01, t_lo[3]));
          } else {
            t_lo[0] = fmaf(sc0, d0, t_lo[0]);
            t_lo[1] = fmaf(sc0, d1, t_lo[1]);
            t_lo[2] = fmaf(sc1, d2, t_lo[2]);
            t_lo[3] = fmaf(sc1, d3, t_lo[3]);
          }

          if constexpr (ALLOW_HI) {
            if (do_hi) {
              const uint4 xv1 = (tok_g_hi >= 0)
                  ? x_u4[static_cast<size_t>(tok_g_hi) * 288 + grp32 * 4 + t]
                  : make_uint4(0u, 0u, 0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
              ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
              if constexpr (MODE == 2) {
                float my_s1 = (tok_g_hi >= 0) ? i2_sum_bf16x8(xv1) : 0.0f;
                my_s1 += __shfl_xor_sync(0xffffffffu, my_s1, 1);
                my_s1 += __shfl_xor_sync(0xffffffffu, my_s1, 2);
                const float xs10 = __shfl_sync(0xffffffffu, my_s1, (2 * t) * 4);
                const float xs11 = __shfl_sync(0xffffffffu, my_s1, (2 * t + 1) * 4);
                const float bi0 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g][lg], sb_is_bf16);
                const float bi1 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g + 8][lg], sb_is_bf16);
                t_hi[0] = fmaf(sc0, h0, fmaf(bi0, xs10, t_hi[0]));
                t_hi[1] = fmaf(sc0, h1, fmaf(bi0, xs11, t_hi[1]));
                t_hi[2] = fmaf(sc1, h2, fmaf(bi1, xs10, t_hi[2]));
                t_hi[3] = fmaf(sc1, h3, fmaf(bi1, xs11, t_hi[3]));
              } else {
                t_hi[0] = fmaf(sc0, h0, t_hi[0]);
                t_hi[1] = fmaf(sc0, h1, t_hi[1]);
                t_hi[2] = fmaf(sc1, h2, t_hi[2]);
                t_hi[3] = fmaf(sc1, h3, t_hi[3]);
              }
            }
          }
        }
      }
      __syncthreads();
    }

    smem.red.gate[warp][2 * t][g] = g_tot_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = g_tot_lo[1];
    smem.red.gate[warp][2 * t][g + 8] = g_tot_lo[2];
    smem.red.gate[warp][2 * t + 1][g + 8] = g_tot_lo[3];
    smem.red.up[warp][2 * t][g] = u_tot_lo[0];
    smem.red.up[warp][2 * t + 1][g] = u_tot_lo[1];
    smem.red.up[warp][2 * t][g + 8] = u_tot_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = u_tot_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = g_tot_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = g_tot_hi[1];
        smem.red.gate[warp][8 + 2 * t][g + 8] = g_tot_hi[2];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = g_tot_hi[3];
        smem.red.up[warp][8 + 2 * t][g] = u_tot_hi[0];
        smem.red.up[warp][8 + 2 * t + 1][g] = u_tot_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = u_tot_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = u_tot_hi[3];
      }
    }
    __syncthreads();

    const int total_out = cnt * 16;
    for (int idx = tid; idx < total_out; idx += 128) {
      const int c = idx >> 4;
      const int r = idx & 15;
      const float gv = (smem.red.gate[0][c][r] + smem.red.gate[1][c][r]) +
                       (smem.red.gate[2][c][r] + smem.red.gate[3][c][r]);
      const float uv = (smem.red.up[0][c][r] + smem.red.up[1][c][r]) +
                       (smem.red.up[2][c][r] + smem.red.up[3][c][r]);
      mid[static_cast<size_t>(s_pair[c]) * I + row0 + r] = i2_situ_epilogue(gv, uv, outer_s);
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2A-NVFP4. Silicon-Roofline Linear-18KB Full-Row Gate + Up + SiTU(4, 25)
//           9 warps (288 thr), 4 x 4608B contiguous ld.global.cs.v4.u32 bursts
//           per pass (split: gate then up; interleaved: rows 0..15 then 16..31
//           with SMEM-store de-interleave -> 0 strided loads, 0 bank conflicts!),
//           5 barriers total (down from 36), 0 spills, 21.1 KB SMEM
//           (3 CTAs/SM = 63.4 KB <= 64 KB carveout -> full 64 KB L1 for x reuse).
//           0-ULP batch-invariant across M in 1..=16.
//           Grid: dim3(72, M * 8), Block: dim3(32, 9) -> BN = 16, K = 2304.
// ---------------------------------------------------------------------------
template <bool INTERLEAVED, bool ALLOW_HI>
__global__ __launch_bounds__(288, 3)
void inkling2_moe_gate_up_situ_nvfp4_few(
    const unsigned int* __restrict__ gu_w,
    const unsigned char* __restrict__ gu_scales,
    const float* __restrict__ gu_outer_scale,
    const nv_bfloat16* __restrict__ x,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  const int row0 = blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  union SharedNvfp4Gu {
    struct {
      alignas(16) uint4 w[16][73]; // 18,688 B (73*4 == 4 mod 32 -> 0 bank conflicts)
      alignas(16) uint4 sc[16][9]; //  2,304 B (9*4  == 4 mod 32 -> 0 bank conflicts)
    } buf;
    struct {
      float gate[9][ALLOW_HI ? 16 : 8][17];
      float up[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedNvfp4Gu smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];
#if !I2_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  if (tid < 256) s_e2m1_lut[tid] = i2_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
#endif

  if (tid < cnt) {
    const int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p >> 3;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }

  // Division-free mapping: tid in 0..287 -> (lr in 0..3, lc in 0..71); tid in 0..143 -> (sr, sc_c)
  const int r_hi = (tid >= 144) ? 2 : 0;
  const int rem = tid - r_hi * 72;
  const int r_lo = (rem >= 72) ? 1 : 0;
  const int lr = r_hi + r_lo;
  const int lc = rem - r_lo * 72;
  const int sr = (tid * 57) >> 9;
  const int sc_c = tid - sr * 9;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
  const uint4* s_u4 = reinterpret_cast<const uint4*>(gu_scales);

  auto fetch_pass = [&](int pass, uint4& v0, uint4& v1, uint4& v2, uint4& v3, uint4& sv) {
    const size_t base_r = INTERLEAVED
        ? (exp_base + 2 * row0 + (pass ? 16 : 0))
        : (exp_base + (pass ? I : 0) + row0);
    const uint4* wp = w_u4 + base_r * 72;
    const uint4* sp = s_u4 + base_r * 9;
    v0 = ptx_ld_cs_u4(wp + tid);
    v1 = ptx_ld_cs_u4(wp + 288 + tid);
    v2 = ptx_ld_cs_u4(wp + 576 + tid);
    v3 = ptx_ld_cs_u4(wp + 864 + tid);
    if (tid < 144) sv = ptx_ld_cs_u4(sp + tid);
  };
  auto store_pass = [&](const uint4& v0, const uint4& v1, const uint4& v2, const uint4& v3,
                        const uint4& sv) {
    if constexpr (!INTERLEAVED) {
      smem.buf.w[lr][lc] = v0;
      smem.buf.w[4 + lr][lc] = v1;
      smem.buf.w[8 + lr][lc] = v2;
      smem.buf.w[12 + lr][lc] = v3;
      if (tid < 144) smem.buf.sc[sr][sc_c] = sv;
    } else {
      // De-interleave on SMEM store: even rows (gate) -> [0..7], odd rows (up) -> [8..15].
      const int d_odd = (lr & 1) << 3;
      const int d_half = lr >> 1;
      smem.buf.w[d_odd + d_half][lc] = v0;
      smem.buf.w[d_odd + 2 + d_half][lc] = v1;
      smem.buf.w[d_odd + 4 + d_half][lc] = v2;
      smem.buf.w[d_odd + 6 + d_half][lc] = v3;
      if (tid < 144) smem.buf.sc[((sr & 1) << 3) | (sr >> 1)][sc_c] = sv;
    }
  };

  uint4 v0, v1, v2, v3, sv;
  fetch_pass(0, v0, v1, v2, v3, sv);
  store_pass(v0, v1, v2, v3, sv);
  __syncthreads();

  const int tok_g_lo = s_tok[g];
  const uint2* x_lo_p = (tok_g_lo >= 0)
      ? reinterpret_cast<const uint2*>(x + static_cast<size_t>(tok_g_lo) * 2304) : nullptr;
  [[maybe_unused]] const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);
  [[maybe_unused]] const uint2* x_hi_p = (ALLOW_HI && tok_g_hi >= 0)
      ? reinterpret_cast<const uint2*>(x + static_cast<size_t>(tok_g_hi) * 2304) : nullptr;

  float p0_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float p1_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float p0_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float p1_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  auto compute_all_16 = [&](float acc_lo[4], float acc_hi[4]) {
    const unsigned short* w0_s = reinterpret_cast<const unsigned short*>(&smem.buf.w[g][warp * 8]);
    const unsigned short* w1_s = reinterpret_cast<const unsigned short*>(&smem.buf.w[g + 8][warp * 8]);
    const unsigned short* sc0_s = reinterpret_cast<const unsigned short*>(&smem.buf.sc[g][warp]);
    const unsigned short* sc1_s = reinterpret_cast<const unsigned short*>(&smem.buf.sc[g + 8][warp]);
    const int grp_base = warp * 16;

#pragma unroll 2
    for (int kg2 = 0; kg2 < 8; ++kg2) {
      const float2 sf0 = i2_fast_e4m3x2_to_f2(sc0_s[kg2]);
      const float2 sf1 = i2_fast_e4m3x2_to_f2(sc1_s[kg2]);
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int kg = kg2 * 2 + sub;
        const unsigned short p0 = w0_s[kg * 4 + t];
        const unsigned short p1 = w1_s[kg * 4 + t];
#if I2_HAS_HW_FP4_CVT
        const unsigned int a0 = i2_fast_e2m1x2_to_bf16x2(p0 & 0xFFu);
        const unsigned int a2 = i2_fast_e2m1x2_to_bf16x2(p0 >> 8);
        const unsigned int a1 = i2_fast_e2m1x2_to_bf16x2(p1 & 0xFFu);
        const unsigned int a3 = i2_fast_e2m1x2_to_bf16x2(p1 >> 8);
#else
        const unsigned int a0 = s_e2m1_lut[p0 & 0xFFu];
        const unsigned int a2 = s_e2m1_lut[p0 >> 8];
        const unsigned int a1 = s_e2m1_lut[p1 & 0xFFu];
        const unsigned int a3 = s_e2m1_lut[p1 >> 8];
#endif
        const float sc0 = (sub == 0) ? sf0.x : sf0.y;
        const float sc1 = (sub == 0) ? sf1.x : sf1.y;
        const int grp = grp_base + kg;
        const uint2 xv0 = x_lo_p ? x_lo_p[grp * 4 + t] : make_uint2(0u, 0u);
        float d0, d1, d2, d3;
        ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
        acc_lo[0] = fmaf(sc0, d0, acc_lo[0]);
        acc_lo[1] = fmaf(sc0, d1, acc_lo[1]);
        acc_lo[2] = fmaf(sc1, d2, acc_lo[2]);
        acc_lo[3] = fmaf(sc1, d3, acc_lo[3]);
        if constexpr (ALLOW_HI) {
          if (do_hi) {
            const uint2 xv1 = x_hi_p ? x_hi_p[grp * 4 + t] : make_uint2(0u, 0u);
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv1.x, xv1.y);
            acc_hi[0] = fmaf(sc0, d0, acc_hi[0]);
            acc_hi[1] = fmaf(sc0, d1, acc_hi[1]);
            acc_hi[2] = fmaf(sc1, d2, acc_hi[2]);
            acc_hi[3] = fmaf(sc1, d3, acc_hi[3]);
          }
        }
      }
    }
  };

  compute_all_16(p0_lo, p0_hi);
  __syncthreads();

  fetch_pass(1, v0, v1, v2, v3, sv);
  store_pass(v0, v1, v2, v3, sv);
  __syncthreads();

  compute_all_16(p1_lo, p1_hi);
  __syncthreads();

  if constexpr (!INTERLEAVED) {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.gate[warp][2 * t][g + 8] = p0_lo[2];
    smem.red.gate[warp][2 * t + 1][g + 8] = p0_lo[3];
    smem.red.up[warp][2 * t][g] = p1_lo[0];
    smem.red.up[warp][2 * t + 1][g] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p0_hi[2];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p0_hi[3];
        smem.red.up[warp][8 + 2 * t][g] = p1_hi[0];
        smem.red.up[warp][8 + 2 * t + 1][g] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  } else {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.up[warp][2 * t][g] = p0_lo[2];
    smem.red.up[warp][2 * t + 1][g] = p0_lo[3];
    smem.red.gate[warp][2 * t][g + 8] = p1_lo[0];
    smem.red.gate[warp][2 * t + 1][g + 8] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.up[warp][8 + 2 * t][g] = p0_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g] = p0_hi[3];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p1_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  }
  __syncthreads();

  const int total_out = cnt * 16;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 4;
    const int r = idx & 15;
    float gv = 0.0f;
    float uv = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 9; ++ww) {
      gv += smem.red.gate[ww][c][r];
      uv += smem.red.up[ww][c][r];
    }
    mid[static_cast<size_t>(s_pair[c]) * I + row0 + r] = i2_situ_epilogue(gv, uv, outer_s);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2A-SYMQ8. Silicon-Roofline Linear-18KB K-Half SymQ8 Gate + Up + SiTU(4, 25)
//           9 warps (288 thr), 2 passes x 2 K-halves of 18KB W (9 cache lines/row,
//           100% coalesced ld.global.cs.v4.u32) + 2.25KB contiguous S,
//           0 bank conflicts, 0 spills, 20.9 KB SMEM (3 CTAs/SM <= 64 KB carveout
//           -> full 64 KB L1 for x reuse). Supports INTERL=false/true & F16/BF16 sc.
//           0-ULP batch-invariant across M in 1..=16.
//           Grid: dim3(72, M * 8), Block: dim3(32, 9) -> BN = 16, K = 2304.
// ---------------------------------------------------------------------------
template <bool INTERLEAVED, bool ALLOW_HI>
__global__ __launch_bounds__(288, 3)
void inkling2_moe_gate_up_situ_symq8_few(
    const unsigned int* __restrict__ gu_w,
    const unsigned short* __restrict__ gu_scales,
    const float* __restrict__ gu_outer_scale,
    const nv_bfloat16* __restrict__ x,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid,
    int sb_is_bf16) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  const int row0 = blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  union SharedSymQ8Gu {
    struct {
      alignas(16) uint4 w[16][74]; // 18,944 B (74*4 == 8 mod 32 -> 0 LDS.64 bank conflicts)
      alignas(16) uint4 sc[16][9]; //  2,304 B (9*4  == 4 mod 32 -> 0 bank conflicts)
    } buf;
    struct {
      float gate[9][ALLOW_HI ? 16 : 8][17];
      float up[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedSymQ8Gu smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    const int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p >> 3;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }

  // Division-free mapping: tid in 0..287 -> (lr in 0..3, lc in 0..71); tid in 0..143 -> (sr, sc_c)
  const int r_hi = (tid >= 144) ? 2 : 0;
  const int rem = tid - r_hi * 72;
  const int r_lo = (rem >= 72) ? 1 : 0;
  const int lr = r_hi + r_lo;
  const int lc = rem - r_lo * 72;
  const int sr = (tid * 57) >> 9;
  const int sc_c = tid - sr * 9;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
  const uint4* s_u4 = reinterpret_cast<const uint4*>(gu_scales);

  auto load_kh = [&](int pass, int kh) {
    const size_t base_r = INTERLEAVED
        ? (exp_base + 2 * row0 + (pass ? 16 : 0))
        : (exp_base + (pass ? I : 0) + row0);
    const uint4* wp = w_u4 + base_r * 144 + kh * 72 + lr * 144 + lc;
    const uint4 v0 = ptx_ld_cs_u4(wp);
    const uint4 v1 = ptx_ld_cs_u4(wp + 576);
    const uint4 v2 = ptx_ld_cs_u4(wp + 1152);
    const uint4 v3 = ptx_ld_cs_u4(wp + 1728);
    uint4 sv;
    if (kh == 0 && tid < 144) sv = ptx_ld_cs_u4(s_u4 + base_r * 9 + tid);
    if constexpr (!INTERLEAVED) {
      smem.buf.w[lr][lc] = v0;
      smem.buf.w[4 + lr][lc] = v1;
      smem.buf.w[8 + lr][lc] = v2;
      smem.buf.w[12 + lr][lc] = v3;
      if (kh == 0 && tid < 144) smem.buf.sc[sr][sc_c] = sv;
    } else {
      const int d_odd = (lr & 1) << 3;
      const int d_half = lr >> 1;
      smem.buf.w[d_odd + d_half][lc] = v0;
      smem.buf.w[d_odd + 2 + d_half][lc] = v1;
      smem.buf.w[d_odd + 4 + d_half][lc] = v2;
      smem.buf.w[d_odd + 6 + d_half][lc] = v3;
      if (kh == 0 && tid < 144) smem.buf.sc[((sr & 1) << 3) | (sr >> 1)][sc_c] = sv;
    }
  };

  load_kh(0, 0);
  __syncthreads();

  const int tok_g_lo = s_tok[g];
  const uint4* x_lo_p = (tok_g_lo >= 0)
      ? reinterpret_cast<const uint4*>(x + static_cast<size_t>(tok_g_lo) * 2304) : nullptr;
  [[maybe_unused]] const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);
  [[maybe_unused]] const uint4* x_hi_p = (ALLOW_HI && tok_g_hi >= 0)
      ? reinterpret_cast<const uint4*>(x + static_cast<size_t>(tok_g_hi) * 2304) : nullptr;

  const uint2* w_r0 = reinterpret_cast<const uint2*>(&smem.buf.w[g][warp * 8]);
  const uint2* w_r1 = reinterpret_cast<const uint2*>(&smem.buf.w[g + 8][warp * 8]);

  auto compute_kh = [&](int kh, float acc_lo[4], float acc_hi[4]) {
    const uint2 sc0_u2 = reinterpret_cast<const uint2*>(smem.buf.sc[g])[kh * 9 + warp];
    const uint2 sc1_u2 = reinterpret_cast<const uint2*>(smem.buf.sc[g + 8])[kh * 9 + warp];
    const int grp_base = kh * 36 + warp * 4;
#pragma unroll 1
    for (int kg = 0; kg < 4; ++kg) {
      const int grp32 = grp_base + kg;
      unsigned int wb0[4], wb1[4];
      i2_unpack8_to_bf16x2<1>(w_r0[kg * 4 + t], wb0);
      i2_unpack8_to_bf16x2<1>(w_r1[kg * 4 + t], wb1);

      const uint4 xv0 = x_lo_p ? x_lo_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
      float d0, d1, d2, d3;
      ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
      ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

      const unsigned short s0h = static_cast<unsigned short>(((kg < 2) ? sc0_u2.x : sc0_u2.y) >> ((kg & 1) * 16));
      const unsigned short s1h = static_cast<unsigned short>(((kg < 2) ? sc1_u2.x : sc1_u2.y) >> ((kg & 1) * 16));
      const float sc0 = i2_read_f16_or_bf16(s0h, sb_is_bf16);
      const float sc1 = i2_read_f16_or_bf16(s1h, sb_is_bf16);
      acc_lo[0] = fmaf(sc0, d0, acc_lo[0]);
      acc_lo[1] = fmaf(sc0, d1, acc_lo[1]);
      acc_lo[2] = fmaf(sc1, d2, acc_lo[2]);
      acc_lo[3] = fmaf(sc1, d3, acc_lo[3]);

      if constexpr (ALLOW_HI) {
        if (do_hi) {
          const uint4 xv1 = x_hi_p ? x_hi_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
          float h0, h1, h2, h3;
          ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
          ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
          acc_hi[0] = fmaf(sc0, h0, acc_hi[0]);
          acc_hi[1] = fmaf(sc0, h1, acc_hi[1]);
          acc_hi[2] = fmaf(sc1, h2, acc_hi[2]);
          acc_hi[3] = fmaf(sc1, h3, acc_hi[3]);
        }
      }
    }
  };

  float p0_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float p1_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float p0_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float p1_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  compute_kh(0, p0_lo, p0_hi);
  __syncthreads();
  load_kh(0, 1);
  __syncthreads();
  compute_kh(1, p0_lo, p0_hi);
  __syncthreads();

  load_kh(1, 0);
  __syncthreads();
  compute_kh(0, p1_lo, p1_hi);
  __syncthreads();
  load_kh(1, 1);
  __syncthreads();
  compute_kh(1, p1_lo, p1_hi);
  __syncthreads();

  if constexpr (!INTERLEAVED) {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.gate[warp][2 * t][g + 8] = p0_lo[2];
    smem.red.gate[warp][2 * t + 1][g + 8] = p0_lo[3];
    smem.red.up[warp][2 * t][g] = p1_lo[0];
    smem.red.up[warp][2 * t + 1][g] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p0_hi[2];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p0_hi[3];
        smem.red.up[warp][8 + 2 * t][g] = p1_hi[0];
        smem.red.up[warp][8 + 2 * t + 1][g] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  } else {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.up[warp][2 * t][g] = p0_lo[2];
    smem.red.up[warp][2 * t + 1][g] = p0_lo[3];
    smem.red.gate[warp][2 * t][g + 8] = p1_lo[0];
    smem.red.gate[warp][2 * t + 1][g + 8] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.up[warp][8 + 2 * t][g] = p0_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g] = p0_hi[3];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p1_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  }
  __syncthreads();

  const int total_out = cnt * 16;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 4;
    const int r = idx & 15;
    float gv = 0.0f;
    float uv = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 9; ++ww) {
      gv += smem.red.gate[ww][c][r];
      uv += smem.red.up[ww][c][r];
    }
    mid[static_cast<size_t>(s_pair[c]) * I + row0 + r] = i2_situ_epilogue(gv, uv, outer_s);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2A-MXFP8. Silicon-Roofline Linear-18KB K-Half MXFP8 Gate + Up + SiTU(4, 25)
//           9 warps (288 thr), 2 passes x 2 K-halves of 18KB W + 1.125KB UE8M0 S
//           (vectorized uint2 loads + 1-instr HW cvt.rn.bf16x2.e4m3x2 / f16x2!),
//           0 bank conflicts, 0 spills, 19.8 KB SMEM (3 CTAs/SM <= 64 KB carveout
//           -> full 64 KB L1 for x reuse). Supports INTERL=false and INTERL=true.
//           0-ULP batch-invariant across M in 1..=16.
//           Grid: dim3(72, M * 8), Block: dim3(32, 9) -> BN = 16, K = 2304.
// ---------------------------------------------------------------------------
template <bool INTERLEAVED, bool ALLOW_HI>
__global__ __launch_bounds__(288, 3)
void inkling2_moe_gate_up_situ_mxfp8_few(
    const unsigned int* __restrict__ gu_w,
    const unsigned char* __restrict__ gu_scales,
    const float* __restrict__ gu_outer_scale,
    const nv_bfloat16* __restrict__ x,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  const int row0 = blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  union SharedMxfp8Gu {
    struct {
      alignas(16) uint4 w[16][74]; // 18,944 B (74*4 == 8 mod 32 -> 0 LDS.64 bank conflicts)
      alignas(16) uint2 sc[16][9]; //  1,152 B (18 u32/row -> gcd(18,32)=2 -> 0 conflicts for g=0..7)
    } buf;
    struct {
      float gate[9][ALLOW_HI ? 16 : 8][17];
      float up[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedMxfp8Gu smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    const int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p >> 3;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }

  // Division-free mapping: tid in 0..287 -> (lr in 0..3, lc in 0..71); tid in 0..143 -> (sr, sc_c)
  const int r_hi = (tid >= 144) ? 2 : 0;
  const int rem = tid - r_hi * 72;
  const int r_lo = (rem >= 72) ? 1 : 0;
  const int lr = r_hi + r_lo;
  const int lc = rem - r_lo * 72;
  const int sr = (tid * 57) >> 9;
  const int sc_c = tid - sr * 9;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
  const uint2* s_u2 = reinterpret_cast<const uint2*>(gu_scales);

  auto load_kh = [&](int pass, int kh) {
    const size_t base_r = INTERLEAVED
        ? (exp_base + 2 * row0 + (pass ? 16 : 0))
        : (exp_base + (pass ? I : 0) + row0);
    const uint4* wp = w_u4 + base_r * 144 + kh * 72 + lr * 144 + lc;
    const uint4 v0 = ptx_ld_cs_u4(wp);
    const uint4 v1 = ptx_ld_cs_u4(wp + 576);
    const uint4 v2 = ptx_ld_cs_u4(wp + 1152);
    const uint4 v3 = ptx_ld_cs_u4(wp + 1728);
    uint2 sv;
    if (kh == 0 && tid < 144) sv = i2_ld_cs_u2(s_u2 + base_r * 9 + tid);
    if constexpr (!INTERLEAVED) {
      smem.buf.w[lr][lc] = v0;
      smem.buf.w[4 + lr][lc] = v1;
      smem.buf.w[8 + lr][lc] = v2;
      smem.buf.w[12 + lr][lc] = v3;
      if (kh == 0 && tid < 144) smem.buf.sc[sr][sc_c] = sv;
    } else {
      const int d_odd = (lr & 1) << 3;
      const int d_half = lr >> 1;
      smem.buf.w[d_odd + d_half][lc] = v0;
      smem.buf.w[d_odd + 2 + d_half][lc] = v1;
      smem.buf.w[d_odd + 4 + d_half][lc] = v2;
      smem.buf.w[d_odd + 6 + d_half][lc] = v3;
      if (kh == 0 && tid < 144) smem.buf.sc[((sr & 1) << 3) | (sr >> 1)][sc_c] = sv;
    }
  };

  load_kh(0, 0);
  __syncthreads();

  const int tok_g_lo = s_tok[g];
  const uint4* x_lo_p = (tok_g_lo >= 0)
      ? reinterpret_cast<const uint4*>(x + static_cast<size_t>(tok_g_lo) * 2304) : nullptr;
  [[maybe_unused]] const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);
  [[maybe_unused]] const uint4* x_hi_p = (ALLOW_HI && tok_g_hi >= 0)
      ? reinterpret_cast<const uint4*>(x + static_cast<size_t>(tok_g_hi) * 2304) : nullptr;

  const uint2* w_r0 = reinterpret_cast<const uint2*>(&smem.buf.w[g][warp * 8]);
  const uint2* w_r1 = reinterpret_cast<const uint2*>(&smem.buf.w[g + 8][warp * 8]);

  auto compute_kh = [&](int kh, float acc_lo[4], float acc_hi[4]) {
    const unsigned int sc0_u32 = reinterpret_cast<const unsigned int*>(smem.buf.sc[g])[kh * 9 + warp];
    const unsigned int sc1_u32 = reinterpret_cast<const unsigned int*>(smem.buf.sc[g + 8])[kh * 9 + warp];
    const int grp_base = kh * 36 + warp * 4;
#pragma unroll 1
    for (int kg = 0; kg < 4; ++kg) {
      const int grp32 = grp_base + kg;
      unsigned int wb0[4], wb1[4];
      i2_unpack8_to_bf16x2<4>(w_r0[kg * 4 + t], wb0);
      i2_unpack8_to_bf16x2<4>(w_r1[kg * 4 + t], wb1);

      const uint4 xv0 = x_lo_p ? x_lo_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
      float d0, d1, d2, d3;
      ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
      ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

      const float sc0 = i2_decode_ue8m0((sc0_u32 >> (kg * 8)) & 0xFFu);
      const float sc1 = i2_decode_ue8m0((sc1_u32 >> (kg * 8)) & 0xFFu);
      acc_lo[0] = fmaf(sc0, d0, acc_lo[0]);
      acc_lo[1] = fmaf(sc0, d1, acc_lo[1]);
      acc_lo[2] = fmaf(sc1, d2, acc_lo[2]);
      acc_lo[3] = fmaf(sc1, d3, acc_lo[3]);

      if constexpr (ALLOW_HI) {
        if (do_hi) {
          const uint4 xv1 = x_hi_p ? x_hi_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
          float h0, h1, h2, h3;
          ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
          ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
          acc_hi[0] = fmaf(sc0, h0, acc_hi[0]);
          acc_hi[1] = fmaf(sc0, h1, acc_hi[1]);
          acc_hi[2] = fmaf(sc1, h2, acc_hi[2]);
          acc_hi[3] = fmaf(sc1, h3, acc_hi[3]);
        }
      }
    }
  };

  float p0_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float p1_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float p0_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float p1_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  compute_kh(0, p0_lo, p0_hi);
  __syncthreads();
  load_kh(0, 1);
  __syncthreads();
  compute_kh(1, p0_lo, p0_hi);
  __syncthreads();

  load_kh(1, 0);
  __syncthreads();
  compute_kh(0, p1_lo, p1_hi);
  __syncthreads();
  load_kh(1, 1);
  __syncthreads();
  compute_kh(1, p1_lo, p1_hi);
  __syncthreads();

  if constexpr (!INTERLEAVED) {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.gate[warp][2 * t][g + 8] = p0_lo[2];
    smem.red.gate[warp][2 * t + 1][g + 8] = p0_lo[3];
    smem.red.up[warp][2 * t][g] = p1_lo[0];
    smem.red.up[warp][2 * t + 1][g] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p0_hi[2];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p0_hi[3];
        smem.red.up[warp][8 + 2 * t][g] = p1_hi[0];
        smem.red.up[warp][8 + 2 * t + 1][g] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  } else {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.up[warp][2 * t][g] = p0_lo[2];
    smem.red.up[warp][2 * t + 1][g] = p0_lo[3];
    smem.red.gate[warp][2 * t][g + 8] = p1_lo[0];
    smem.red.gate[warp][2 * t + 1][g + 8] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.up[warp][8 + 2 * t][g] = p0_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g] = p0_hi[3];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p1_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  }
  __syncthreads();

  const int total_out = cnt * 16;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 4;
    const int r = idx & 15;
    float gv = 0.0f;
    float uv = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 9; ++ww) {
      gv += smem.red.gate[ww][c][r];
      uv += smem.red.up[ww][c][r];
    }
    mid[static_cast<size_t>(s_pair[c]) * I + row0 + r] = i2_situ_epilogue(gv, uv, outer_s);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2B. Hybrid Grouped Gate + Up + SiTU(4, 25) for Q5_K (MODE = 3, K = 2304 = 9*256)
//     cnt <= 2: coalesced ld.global.cs.v4.u32 + dp4a streaming GEMV
//     cnt >= 3: 2-stage cp.async.cg + mma.m16n8k32.s8 (1 super-block/step, 4w x 2 sb)
//     Grid: dim3(72, M * 8), Block: dim3(32, 4) -> BN = 16, I = 1152
// ---------------------------------------------------------------------------
template <bool INTERLEAVED, int M_BUCKET>
__global__ __launch_bounds__(128, M_BUCKET == 0 ? 4 : (M_BUCKET == 1 ? 4 : 3))
void inkling2_moe_gate_up_situ_q5k_few(
    const unsigned int* __restrict__ gu_w,
    const unsigned char* __restrict__ gu_scales,
    const unsigned short* __restrict__ gu_biases,
    const float* __restrict__ gu_outer_scale,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const float* __restrict__ x_sums,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  constexpr int UNITS = 9;
  constexpr int GROUPS = 72;
  const int row0 = blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  struct StageQ5k {
    alignas(16) uint4 w[16][11];       // 2,816 B (10 uint4/row + 1 pad)
    alignas(16) uint4 sc[16];          //   256 B
    alignas(16) unsigned int bi[16];   //    64 B
  };
  union SharedQ5k {
    struct {
      alignas(16) uint4 xq[2][144];    // 4,608 B (2 tokens x 72 groups x 2 uint4)
      alignas(16) float xsc[2][72];    //   576 B
      alignas(16) float xsm[2][72];    //   576 B
      alignas(16) uint4 w[4][90];      // 5,760 B (4 warps x 1 row of 90 uint4)
    } gemv;                            // 11,520 B
    StageQ5k pipe[M_BUCKET > 0 ? 2 : 1];
    struct {
      float gate[4][16][17];
      float up[4][16][17];
    } red;
  };
  __shared__ SharedQ5k smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    const int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p >> 3;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }
  __syncthreads();

  if constexpr (M_BUCKET == 0) {
    const bool has_t1 = (cnt == 2);
    const int tok0 = s_tok[0];
    const int tok1 = has_t1 ? s_tok[1] : 0;
    const uint4* xq_u4 = reinterpret_cast<const uint4*>(xq);
    for (int i = tid; i < 144; i += 128) {
      smem.gemv.xq[0][i] = xq_u4[static_cast<size_t>(tok0) * 144 + i];
      if (has_t1) smem.gemv.xq[1][i] = xq_u4[static_cast<size_t>(tok1) * 144 + i];
    }
    if (tid < GROUPS) {
      smem.gemv.xsc[0][tid] = x_scales[static_cast<size_t>(tok0) * GROUPS + tid];
      smem.gemv.xsm[0][tid] = x_sums[static_cast<size_t>(tok0) * GROUPS + tid];
      if (has_t1) {
        smem.gemv.xsc[1][tid] = x_scales[static_cast<size_t>(tok1) * GROUPS + tid];
        smem.gemv.xsm[1][tid] = x_sums[static_cast<size_t>(tok1) * GROUPS + tid];
      }
    }
    __syncthreads();

    const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
    const unsigned short* sc_u16 = reinterpret_cast<const unsigned short*>(gu_scales);
    const unsigned int* bi_u32 = reinterpret_cast<const unsigned int*>(gu_biases);
    const int pair0 = s_pair[0];
    const int pair1 = has_t1 ? s_pair[1] : 0;

    auto dot_row = [&](size_t r_idx, float& acc0, float& acc1) {
      const uint4* wp = w_u4 + r_idx * 90;
      smem.gemv.w[warp][lane] = ptx_ld_cs_u4(wp + lane);
      smem.gemv.w[warp][32 + lane] = ptx_ld_cs_u4(wp + 32 + lane);
      if (lane < 26) smem.gemv.w[warp][64 + lane] = ptx_ld_cs_u4(wp + 64 + lane);
      __syncwarp();
      const unsigned int* w32 = reinterpret_cast<const unsigned int*>(smem.gemv.w[warp]);
      const unsigned short* srow = sc_u16 + r_idx * GROUPS;
      const unsigned int* brow = bi_u32 + r_idx * UNITS;
      acc0 = 0.0f;
      acc1 = 0.0f;
#pragma unroll 1
      for (int grp = lane; grp < GROUPS; grp += 32) {
        const int sb = grp >> 3;
        const unsigned short scm = srow[grp];
        const int ws = scm & 0xFF;
        const int wm = scm >> 8;
        const unsigned int bp = brow[sb];
        const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(&bp));
        const unsigned int* seg = w32 + grp * 5;
        const int d0 = i2_dp4a_q5k_32(seg, smem.gemv.xq[0][grp * 2], smem.gemv.xq[0][grp * 2 + 1]);
        acc0 = fmaf(dm.x * smem.gemv.xsc[0][grp], static_cast<float>(d0 * ws),
                    fmaf(-dm.y * smem.gemv.xsm[0][grp], static_cast<float>(wm), acc0));
        if (has_t1) {
          const int d1 = i2_dp4a_q5k_32(seg, smem.gemv.xq[1][grp * 2], smem.gemv.xq[1][grp * 2 + 1]);
          acc1 = fmaf(dm.x * smem.gemv.xsc[1][grp], static_cast<float>(d1 * ws),
                      fmaf(-dm.y * smem.gemv.xsm[1][grp], static_cast<float>(wm), acc1));
        }
      }
      __syncwarp();
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
        if (has_t1) acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
      }
    };

#pragma unroll 1
    for (int r = 0; r < 4; ++r) {
      const int col = row0 + warp * 4 + r;
      const size_t g_idx = exp_base + (INTERLEAVED ? (2 * col) : col);
      const size_t u_idx_r = exp_base + (INTERLEAVED ? (2 * col + 1) : (I + col));
      float g0, g1, u0, u1;
      dot_row(g_idx, g0, g1);
      dot_row(u_idx_r, u0, u1);
      if (lane == 0) {
        mid[static_cast<size_t>(pair0) * I + col] = i2_situ_epilogue(g0, u0, outer_s);
        if (has_t1) {
          mid[static_cast<size_t>(pair1) * I + col] = i2_situ_epilogue(g1, u1, outer_s);
        }
      }
    }
    return;
  }
)cuda"
R"cuda(
  if constexpr (M_BUCKET > 0) {
    constexpr bool ALLOW_HI = (M_BUCKET == 2);
    const int g = lane >> 2;
    const int t = lane & 3;
    const int tok_g_lo = s_tok[g];
    const int tok_c0_lo = s_tok[2 * t];
    const int tok_c1_lo = s_tok[2 * t + 1];
    const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
    const int tok_c0_hi = ALLOW_HI ? s_tok[8 + 2 * t] : -1;
    const int tok_c1_hi = ALLOW_HI ? s_tok[8 + 2 * t + 1] : -1;
    const bool do_hi = ALLOW_HI && (cnt > 8);
    (void)tok_g_hi; (void)tok_c0_hi; (void)tok_c1_hi; (void)do_hi;

    const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
    const uint4* sc_u4 = reinterpret_cast<const uint4*>(gu_scales);
    const unsigned int* bi_u32 = reinterpret_cast<const unsigned int*>(gu_biases);

    auto row_idx_of = [&](int r, bool is_up) -> size_t {
      return exp_base + (INTERLEAVED ? (2 * (row0 + r) + (is_up ? 1 : 0)) : ((is_up ? I : 0) + row0 + r));
    };

    auto load_unit = [&](int st, int unit, bool is_up) {
      for (int idx = tid; idx < 160; idx += 128) {
        const int r = idx / 10;
        const int c = idx - r * 10;
        const size_t ri = row_idx_of(r, is_up);
        ptx_cp_async16_cg(&smem.pipe[st].w[r][c], w_u4 + ri * 90 + unit * 10 + c);
      }
      if (tid < 16) {
        const size_t sr = row_idx_of(tid, is_up);
        ptx_cp_async16_cg(&smem.pipe[st].sc[tid], sc_u4 + sr * UNITS + unit);
      } else if (tid < 32) {
        const size_t sr = row_idx_of(tid - 16, is_up);
        ptx_cp_async4_ca(&smem.pipe[st].bi[tid - 16], bi_u32 + sr * UNITS + unit);
      }
      ptx_cp_commit();
    };

    float g_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float g_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    load_unit(0, 0, false);
#pragma unroll 1
    for (int step = 0; step < 18; ++step) {
      const int cur = step & 1;
      const int nxt = cur ^ 1;
      if (step + 1 < 18) {
        load_unit(nxt, (step + 1) >> 1, ((step + 1) & 1) != 0);
        ptx_cp_wait<1>();
      } else {
        ptx_cp_wait<0>();
      }
      __syncthreads();

      const int unit = step >> 1;
      const bool is_up = (step & 1) != 0;
      const unsigned int* w_r0 = reinterpret_cast<const unsigned int*>(smem.pipe[cur].w[g]);
      const unsigned int* w_r1 = reinterpret_cast<const unsigned int*>(smem.pipe[cur].w[g + 8]);
      const uint4 sc0 = smem.pipe[cur].sc[g];
      const uint4 sc1 = smem.pipe[cur].sc[g + 8];
      const unsigned int sw0 = (warp == 0) ? sc0.x : (warp == 1 ? sc0.y : (warp == 2 ? sc0.z : sc0.w));
      const unsigned int sw1 = (warp == 0) ? sc1.x : (warp == 1 ? sc1.y : (warp == 2 ? sc1.z : sc1.w));

      float dot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float min_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float dot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float min_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
      for (int ls = 0; ls < 2; ++ls) {
        const int s = warp * 2 + ls;
        const int grp_k = unit * 8 + s;
        unsigned int a0, a1, a2, a3;
        i2_load_pair_q5k(w_r0 + s * 5, t, a0, a2);
        i2_load_pair_q5k(w_r1 + s * 5, t, a1, a3);

        const int ws0 = static_cast<int>((sw0 >> (ls * 16)) & 0xFFu);
        const int ws1 = static_cast<int>((sw1 >> (ls * 16)) & 0xFFu);
        const float wm0 = static_cast<float>((sw0 >> (ls * 16 + 8)) & 0xFFu);
        const float wm1 = static_cast<float>((sw1 >> (ls * 16 + 8)) & 0xFFu);

        unsigned int b0 = 0u, b1 = 0u;
        if (tok_g_lo >= 0) {
          const unsigned int* xp = reinterpret_cast<const unsigned int*>(
              xq + (static_cast<size_t>(tok_g_lo) * GROUPS + grp_k) * 32);
          b0 = xp[t];
          b1 = xp[4 + t];
        }
        int c0, c1, c2, c3;
        ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
        const float ad0 = (tok_c0_lo >= 0) ? x_scales[tok_c0_lo * GROUPS + grp_k] : 0.0f;
        const float ad1 = (tok_c1_lo >= 0) ? x_scales[tok_c1_lo * GROUPS + grp_k] : 0.0f;
        const float as0 = (tok_c0_lo >= 0) ? x_sums[tok_c0_lo * GROUPS + grp_k] : 0.0f;
        const float as1 = (tok_c1_lo >= 0) ? x_sums[tok_c1_lo * GROUPS + grp_k] : 0.0f;
        dot_lo[0] = fmaf(ad0, static_cast<float>(c0 * ws0), dot_lo[0]);
        dot_lo[1] = fmaf(ad1, static_cast<float>(c1 * ws0), dot_lo[1]);
        dot_lo[2] = fmaf(ad0, static_cast<float>(c2 * ws1), dot_lo[2]);
        dot_lo[3] = fmaf(ad1, static_cast<float>(c3 * ws1), dot_lo[3]);
        min_lo[0] = fmaf(as0, wm0, min_lo[0]);
        min_lo[1] = fmaf(as1, wm0, min_lo[1]);
        min_lo[2] = fmaf(as0, wm1, min_lo[2]);
        min_lo[3] = fmaf(as1, wm1, min_lo[3]);

        if constexpr (ALLOW_HI) {
          if (do_hi) {
            unsigned int hb0 = 0u, hb1 = 0u;
            if (tok_g_hi >= 0) {
              const unsigned int* xp = reinterpret_cast<const unsigned int*>(
                  xq + (static_cast<size_t>(tok_g_hi) * GROUPS + grp_k) * 32);
              hb0 = xp[t];
              hb1 = xp[4 + t];
            }
            int h0, h1, h2, h3;
            ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, hb0, hb1, h0, h1, h2, h3);
            const float had0 = (tok_c0_hi >= 0) ? x_scales[tok_c0_hi * GROUPS + grp_k] : 0.0f;
            const float had1 = (tok_c1_hi >= 0) ? x_scales[tok_c1_hi * GROUPS + grp_k] : 0.0f;
            const float has0 = (tok_c0_hi >= 0) ? x_sums[tok_c0_hi * GROUPS + grp_k] : 0.0f;
            const float has1 = (tok_c1_hi >= 0) ? x_sums[tok_c1_hi * GROUPS + grp_k] : 0.0f;
            dot_hi[0] = fmaf(had0, static_cast<float>(h0 * ws0), dot_hi[0]);
            dot_hi[1] = fmaf(had1, static_cast<float>(h1 * ws0), dot_hi[1]);
            dot_hi[2] = fmaf(had0, static_cast<float>(h2 * ws1), dot_hi[2]);
            dot_hi[3] = fmaf(had1, static_cast<float>(h3 * ws1), dot_hi[3]);
            min_hi[0] = fmaf(has0, wm0, min_hi[0]);
            min_hi[1] = fmaf(has1, wm0, min_hi[1]);
            min_hi[2] = fmaf(has0, wm1, min_hi[2]);
            min_hi[3] = fmaf(has1, wm1, min_hi[3]);
          }
        }
      }

      const unsigned int b0p = smem.pipe[cur].bi[g];
      const unsigned int b1p = smem.pipe[cur].bi[g + 8];
      const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(&b0p));
      const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(&b1p));
      float* t_lo = is_up ? u_tot_lo : g_tot_lo;
      t_lo[0] += dm0.x * dot_lo[0] - dm0.y * min_lo[0];
      t_lo[1] += dm0.x * dot_lo[1] - dm0.y * min_lo[1];
      t_lo[2] += dm1.x * dot_lo[2] - dm1.y * min_lo[2];
      t_lo[3] += dm1.x * dot_lo[3] - dm1.y * min_lo[3];
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          float* t_hi = is_up ? u_tot_hi : g_tot_hi;
          t_hi[0] += dm0.x * dot_hi[0] - dm0.y * min_hi[0];
          t_hi[1] += dm0.x * dot_hi[1] - dm0.y * min_hi[1];
          t_hi[2] += dm1.x * dot_hi[2] - dm1.y * min_hi[2];
          t_hi[3] += dm1.x * dot_hi[3] - dm1.y * min_hi[3];
        }
      }
      __syncthreads();
    }

    smem.red.gate[warp][2 * t][g] = g_tot_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = g_tot_lo[1];
    smem.red.gate[warp][2 * t][g + 8] = g_tot_lo[2];
    smem.red.gate[warp][2 * t + 1][g + 8] = g_tot_lo[3];
    smem.red.up[warp][2 * t][g] = u_tot_lo[0];
    smem.red.up[warp][2 * t + 1][g] = u_tot_lo[1];
    smem.red.up[warp][2 * t][g + 8] = u_tot_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = u_tot_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = g_tot_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = g_tot_hi[1];
        smem.red.gate[warp][8 + 2 * t][g + 8] = g_tot_hi[2];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = g_tot_hi[3];
        smem.red.up[warp][8 + 2 * t][g] = u_tot_hi[0];
        smem.red.up[warp][8 + 2 * t + 1][g] = u_tot_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = u_tot_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = u_tot_hi[3];
      }
    }
    __syncthreads();

    const int total_out = cnt * 16;
    for (int idx = tid; idx < total_out; idx += 128) {
      const int c = idx >> 4;
      const int r = idx & 15;
      const float gv = (smem.red.gate[0][c][r] + smem.red.gate[1][c][r]) +
                       (smem.red.gate[2][c][r] + smem.red.gate[3][c][r]);
      const float uv = (smem.red.up[0][c][r] + smem.red.up[1][c][r]) +
                       (smem.red.up[2][c][r] + smem.red.up[3][c][r]);
      mid[static_cast<size_t>(s_pair[c]) * I + row0 + r] = i2_situ_epilogue(gv, uv, outer_s);
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2B-LIN. Silicon-Roofline Linear-Burst Q5_K Gate + Up + SiTU(4, 25)
//         9 warps (288 thr), 5 x 4608B W + 2304B S + 576B B contiguous bursts
//         per pass (ZERO strided loads for both split & interleaved!),
//         each warp owns 1 super-block (unit = warp), 5 barriers total (down
//         from 36), 0 B spills, 26.0 KB SMEM (2 CTAs/SM <= 64 KB carveout ->
//         full 64 KB L1 for xq/x_scales/x_sums). 0-ULP batch-invariant M=1..16.
//         Grid: dim3(72, M * 8), Block: dim3(32, 9) -> BN = 16, K = 2304.
// ---------------------------------------------------------------------------
template <bool INTERLEAVED, bool ALLOW_HI>
__global__ __launch_bounds__(288, 2)
void inkling2_moe_gate_up_situ_q5k_linear_few(
    const unsigned int* __restrict__ gu_w,
    const unsigned char* __restrict__ gu_scales,
    const unsigned short* __restrict__ gu_biases,
    const float* __restrict__ gu_outer_scale,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const float* __restrict__ x_sums,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int I = 1152;
  constexpr int N2 = 2304;
  constexpr int GROUPS = 72;
  const int row0 = blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t exp_base = static_cast<size_t>(expert) * N2;
  const float outer_s = gu_outer_scale ? gu_outer_scale[expert] : 1.0f;

  union SharedQ5kLin {
    struct {
      alignas(16) uint4 w[1440]; // 23,040 B (16 rows x 90 uint4)
      alignas(16) uint4 sc[144]; //  2,304 B (16 rows x 9 uint4)
      alignas(16) uint4 bi[36];  //    576 B (16 rows x 9 u32 = 36 uint4)
    } buf;
    struct {
      float gate[9][ALLOW_HI ? 16 : 8][17];
      float up[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedQ5kLin smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    const int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p >> 3;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }

  const uint4* w_u4 = reinterpret_cast<const uint4*>(gu_w);
  const uint4* s_u4 = reinterpret_cast<const uint4*>(gu_scales);
  const uint4* b_u4 = reinterpret_cast<const uint4*>(gu_biases);

  auto load_pass = [&](int pass) {
    const size_t base_r = INTERLEAVED
        ? (exp_base + 2 * row0 + (pass ? 16 : 0))
        : (exp_base + (pass ? I : 0) + row0);
    const uint4* wp = w_u4 + base_r * 90;
    const uint4* sp = s_u4 + base_r * 9;
    const uint4* bp = b_u4 + (base_r * 9) / 4;
    const uint4 v0 = ptx_ld_cs_u4(wp + tid);
    const uint4 v1 = ptx_ld_cs_u4(wp + 288 + tid);
    const uint4 v2 = ptx_ld_cs_u4(wp + 576 + tid);
    const uint4 v3 = ptx_ld_cs_u4(wp + 864 + tid);
    const uint4 v4 = ptx_ld_cs_u4(wp + 1152 + tid);
    uint4 sv, bv;
    if (tid < 144) sv = ptx_ld_cs_u4(sp + tid);
    if (tid < 36) bv = ptx_ld_cs_u4(bp + tid);
    smem.buf.w[tid] = v0;
    smem.buf.w[288 + tid] = v1;
    smem.buf.w[576 + tid] = v2;
    smem.buf.w[864 + tid] = v3;
    smem.buf.w[1152 + tid] = v4;
    if (tid < 144) smem.buf.sc[tid] = sv;
    if (tid < 36) smem.buf.bi[tid] = bv;
  };

  load_pass(0);
  __syncthreads();

  const int tok_g_lo = s_tok[g];
  const int tok_c0_lo = s_tok[2 * t];
  const int tok_c1_lo = s_tok[2 * t + 1];
  [[maybe_unused]] const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  [[maybe_unused]] const int tok_c0_hi = ALLOW_HI ? s_tok[8 + 2 * t] : -1;
  [[maybe_unused]] const int tok_c1_hi = ALLOW_HI ? s_tok[8 + 2 * t + 1] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);

  const int r0_idx = INTERLEAVED ? (2 * g) : g;
  const int r1_idx = INTERLEAVED ? (2 * g + 1) : (g + 8);
  const int unit = warp;
  const unsigned int* w_r0 =
      reinterpret_cast<const unsigned int*>(&smem.buf.w[r0_idx * 90 + unit * 10]);
  const unsigned int* w_r1 =
      reinterpret_cast<const unsigned int*>(&smem.buf.w[r1_idx * 90 + unit * 10]);
  const unsigned short* sc0_u16 =
      reinterpret_cast<const unsigned short*>(&smem.buf.sc[r0_idx * 9 + unit]);
  const unsigned short* sc1_u16 =
      reinterpret_cast<const unsigned short*>(&smem.buf.sc[r1_idx * 9 + unit]);
  const unsigned int* bi_u32 = reinterpret_cast<const unsigned int*>(smem.buf.bi);

  auto compute_unit = [&](float out_lo[4], float out_hi[4]) {
    float dot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float min_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float dot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float min_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int s = 0; s < 8; ++s) {
      const int grp_k = unit * 8 + s;
      unsigned int a0, a1, a2, a3;
      i2_load_pair_q5k(w_r0 + s * 5, t, a0, a2);
      i2_load_pair_q5k(w_r1 + s * 5, t, a1, a3);

      const unsigned short s0h = sc0_u16[s];
      const unsigned short s1h = sc1_u16[s];
      const int ws0 = static_cast<int>(s0h & 0xFFu);
      const int ws1 = static_cast<int>(s1h & 0xFFu);
      const float wm0 = static_cast<float>(s0h >> 8);
      const float wm1 = static_cast<float>(s1h >> 8);

      unsigned int b0 = 0u, b1 = 0u;
      if (tok_g_lo >= 0) {
        const unsigned int* xp = reinterpret_cast<const unsigned int*>(
            xq + (static_cast<size_t>(tok_g_lo) * GROUPS + grp_k) * 32);
        b0 = xp[t];
        b1 = xp[4 + t];
      }
      int c0, c1, c2, c3;
      ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
      const float ad0 = (tok_c0_lo >= 0) ? x_scales[tok_c0_lo * GROUPS + grp_k] : 0.0f;
      const float ad1 = (tok_c1_lo >= 0) ? x_scales[tok_c1_lo * GROUPS + grp_k] : 0.0f;
      const float as0 = (tok_c0_lo >= 0) ? x_sums[tok_c0_lo * GROUPS + grp_k] : 0.0f;
      const float as1 = (tok_c1_lo >= 0) ? x_sums[tok_c1_lo * GROUPS + grp_k] : 0.0f;
      dot_lo[0] = fmaf(ad0, static_cast<float>(c0 * ws0), dot_lo[0]);
      dot_lo[1] = fmaf(ad1, static_cast<float>(c1 * ws0), dot_lo[1]);
      dot_lo[2] = fmaf(ad0, static_cast<float>(c2 * ws1), dot_lo[2]);
      dot_lo[3] = fmaf(ad1, static_cast<float>(c3 * ws1), dot_lo[3]);
      min_lo[0] = fmaf(as0, wm0, min_lo[0]);
      min_lo[1] = fmaf(as1, wm0, min_lo[1]);
      min_lo[2] = fmaf(as0, wm1, min_lo[2]);
      min_lo[3] = fmaf(as1, wm1, min_lo[3]);

      if constexpr (ALLOW_HI) {
        if (do_hi) {
          unsigned int hb0 = 0u, hb1 = 0u;
          if (tok_g_hi >= 0) {
            const unsigned int* xp = reinterpret_cast<const unsigned int*>(
                xq + (static_cast<size_t>(tok_g_hi) * GROUPS + grp_k) * 32);
            hb0 = xp[t];
            hb1 = xp[4 + t];
          }
          int h0, h1, h2, h3;
          ptx_mma_m16n8k32_s8_z(a0, a1, a2, a3, hb0, hb1, h0, h1, h2, h3);
          const float had0 = (tok_c0_hi >= 0) ? x_scales[tok_c0_hi * GROUPS + grp_k] : 0.0f;
          const float had1 = (tok_c1_hi >= 0) ? x_scales[tok_c1_hi * GROUPS + grp_k] : 0.0f;
          const float has0 = (tok_c0_hi >= 0) ? x_sums[tok_c0_hi * GROUPS + grp_k] : 0.0f;
          const float has1 = (tok_c1_hi >= 0) ? x_sums[tok_c1_hi * GROUPS + grp_k] : 0.0f;
          dot_hi[0] = fmaf(had0, static_cast<float>(h0 * ws0), dot_hi[0]);
          dot_hi[1] = fmaf(had1, static_cast<float>(h1 * ws0), dot_hi[1]);
          dot_hi[2] = fmaf(had0, static_cast<float>(h2 * ws1), dot_hi[2]);
          dot_hi[3] = fmaf(had1, static_cast<float>(h3 * ws1), dot_hi[3]);
          min_hi[0] = fmaf(has0, wm0, min_hi[0]);
          min_hi[1] = fmaf(has1, wm0, min_hi[1]);
          min_hi[2] = fmaf(has0, wm1, min_hi[2]);
          min_hi[3] = fmaf(has1, wm1, min_hi[3]);
        }
      }
    }

    const unsigned int b0p = bi_u32[r0_idx * 9 + unit];
    const unsigned int b1p = bi_u32[r1_idx * 9 + unit];
    const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(&b0p));
    const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(&b1p));
    out_lo[0] = dm0.x * dot_lo[0] - dm0.y * min_lo[0];
    out_lo[1] = dm0.x * dot_lo[1] - dm0.y * min_lo[1];
    out_lo[2] = dm1.x * dot_lo[2] - dm1.y * min_lo[2];
    out_lo[3] = dm1.x * dot_lo[3] - dm1.y * min_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        out_hi[0] = dm0.x * dot_hi[0] - dm0.y * min_hi[0];
        out_hi[1] = dm0.x * dot_hi[1] - dm0.y * min_hi[1];
        out_hi[2] = dm1.x * dot_hi[2] - dm1.y * min_hi[2];
        out_hi[3] = dm1.x * dot_hi[3] - dm1.y * min_hi[3];
      }
    }
  };

  float p0_lo[4], p1_lo[4];
  [[maybe_unused]] float p0_hi[4], p1_hi[4];
  compute_unit(p0_lo, p0_hi);
  __syncthreads();

  load_pass(1);
  __syncthreads();

  compute_unit(p1_lo, p1_hi);
  __syncthreads();

  if constexpr (!INTERLEAVED) {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.gate[warp][2 * t][g + 8] = p0_lo[2];
    smem.red.gate[warp][2 * t + 1][g + 8] = p0_lo[3];
    smem.red.up[warp][2 * t][g] = p1_lo[0];
    smem.red.up[warp][2 * t + 1][g] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p0_hi[2];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p0_hi[3];
        smem.red.up[warp][8 + 2 * t][g] = p1_hi[0];
        smem.red.up[warp][8 + 2 * t + 1][g] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  } else {
    smem.red.gate[warp][2 * t][g] = p0_lo[0];
    smem.red.gate[warp][2 * t + 1][g] = p0_lo[1];
    smem.red.up[warp][2 * t][g] = p0_lo[2];
    smem.red.up[warp][2 * t + 1][g] = p0_lo[3];
    smem.red.gate[warp][2 * t][g + 8] = p1_lo[0];
    smem.red.gate[warp][2 * t + 1][g + 8] = p1_lo[1];
    smem.red.up[warp][2 * t][g + 8] = p1_lo[2];
    smem.red.up[warp][2 * t + 1][g + 8] = p1_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        smem.red.gate[warp][8 + 2 * t][g] = p0_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g] = p0_hi[1];
        smem.red.up[warp][8 + 2 * t][g] = p0_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g] = p0_hi[3];
        smem.red.gate[warp][8 + 2 * t][g + 8] = p1_hi[0];
        smem.red.gate[warp][8 + 2 * t + 1][g + 8] = p1_hi[1];
        smem.red.up[warp][8 + 2 * t][g + 8] = p1_hi[2];
        smem.red.up[warp][8 + 2 * t + 1][g + 8] = p1_hi[3];
      }
    }
  }
  __syncthreads();

  const int total_out = cnt * 16;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 4;
    const int r = idx & 15;
    float gv = 0.0f;
    float uv = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 9; ++ww) {
      gv += smem.red.gate[ww][c][r];
      uv += smem.red.up[ww][c][r];
    }
    mid[static_cast<size_t>(s_pair[c]) * I + row0 + r] = i2_situ_epilogue(gv, uv, outer_s);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3. Hybrid Grouped Down Projection (K = 1152, N = 2304, 64 rows/CTA)
//    MODE: 0=NVFP4, 1=SymQ8, 2=AffineQ8, 4=MXFP8
//    cnt <= 2: SMEM-staged FP32 mid[2][1152] + 0-barrier streaming uint4 GEMV
//    cnt >= 3: 4x16-row tiles x 3 K-thirds (BK=384) 2-stage cp.async.cg + bf16 MMA
//    Grid: dim3(36, M * 8), Block: dim3(32, 4).
// ---------------------------------------------------------------------------
template <int MODE, int M_BUCKET>
__global__ __launch_bounds__(128, M_BUCKET == 0 ? 6 : (M_BUCKET == 1 ? 4 : 3))
void inkling2_moe_down_few(
    const unsigned int* __restrict__ dn_w,
    const void* __restrict__ dn_scales,
    const void* __restrict__ dn_biases,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ down_pairs,
    int sb_is_bf16) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int N = 2304;
  constexpr int K = 1152;
  const int base_row = blockIdx.x * 64;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const size_t exp_base = static_cast<size_t>(expert) * N;

  struct StageDn {
    alignas(16) uint4 w[16][MODE == 0 ? 13 : 25];
    alignas(16) unsigned short sc[16][12]; // NVFP4=24 u8, Q8=12 u16, MXFP8=12 u8
    alignas(16) unsigned short bi[MODE == 2 ? 16 : 1][12];
  };
  union SharedDn {
    alignas(16) float s_x[2][K]; // 9,216 B
    StageDn pipe[M_BUCKET > 0 ? 2 : 1];
  };
  __shared__ SharedDn smem;
  __shared__ float s_red[M_BUCKET > 0 ? 4 : 1][16][17];
  __shared__ int s_pair[16];

  if (tid < cnt) s_pair[tid] = pair_indices[first + tid];
  else if (tid < 16) s_pair[tid] = -1;
  __syncthreads();

  // ---- Fast Branch: M_BUCKET == 0 -> FP32 staged mid + 0-barrier streaming uint4 GEMV ----
  if constexpr (M_BUCKET == 0) {
    const bool has_t1 = (cnt == 2);
    const int p0 = s_pair[0];
    const int p1 = has_t1 ? s_pair[1] : 0;
    const uint4* m_u4 = reinterpret_cast<const uint4*>(mid);
    for (int i = tid; i < 144; i += 128) {
      const uint4 v0 = m_u4[static_cast<size_t>(p0) * 144 + i];
      const __nv_bfloat162* q0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(q0[j]);
        smem.s_x[0][i * 8 + j * 2] = f.x;
        smem.s_x[0][i * 8 + j * 2 + 1] = f.y;
      }
      if (has_t1) {
        const uint4 v1 = m_u4[static_cast<size_t>(p1) * 144 + i];
        const __nv_bfloat162* q1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(q1[j]);
          smem.s_x[1][i * 8 + j * 2] = f.x;
          smem.s_x[1][i * 8 + j * 2 + 1] = f.y;
        }
      }
    }
    __syncthreads();

    const uint4* w_u4 = reinterpret_cast<const uint4*>(dn_w);
    const int w_row0 = base_row + warp * 16;

#pragma unroll 1
    for (int r = 0; r < 16; ++r) {
      const int row = w_row0 + r;
      const size_t r_idx = exp_base + row;
      float acc0 = 0.0f, acc1 = 0.0f;

      if constexpr (MODE == 0) {
        const uint4* wp = w_u4 + r_idx * 36;
        const unsigned short* sp = reinterpret_cast<const unsigned short*>(dn_scales) + r_idx * 36;
        auto do_u = [&](int u) {
          const uint4 wv = ptx_ld_cs_u4(wp + u);
          const unsigned short sc_pair = sp[u];
          const float sl = i2_decode_fp8_e4m3_x16384(sc_pair & 0xFFu);
          const float sh = i2_decode_fp8_e4m3_x16384(sc_pair >> 8);
          float d0, d1;
          i2_dot_nvfp4_16(wv.x, wv.y, &smem.s_x[0][u * 32], &smem.s_x[1][u * 32], has_t1, d0, d1);
          acc0 = fmaf(sl, d0, acc0);
          if (has_t1) acc1 = fmaf(sl, d1, acc1);
          i2_dot_nvfp4_16(wv.z, wv.w, &smem.s_x[0][u * 32 + 16], &smem.s_x[1][u * 32 + 16], has_t1, d0, d1);
          acc0 = fmaf(sh, d0, acc0);
          if (has_t1) acc1 = fmaf(sh, d1, acc1);
        };
        do_u(lane);
        if (lane < 4) do_u(32 + lane);
      } else {
        const uint4* wp = w_u4 + r_idx * 72;
#pragma unroll 1
        for (int u = lane; u < 72; u += 32) {
          const uint4 wv = ptx_ld_cs_u4(wp + u);
          const float* xp0 = &smem.s_x[0][u * 16];
          const float* xp1 = &smem.s_x[1][u * 16];
          float d0, d1;
          i2_dot_8b_16<MODE>(wv, xp0, xp1, has_t1, d0, d1);
          const int grp = u >> 1;
          if constexpr (MODE == 1) {
            const float sc = i2_read_f16_or_bf16(
                reinterpret_cast<const unsigned short*>(dn_scales)[r_idx * 36 + grp], sb_is_bf16);
            acc0 = fmaf(sc, d0, acc0);
            if (has_t1) acc1 = fmaf(sc, d1, acc1);
          } else if constexpr (MODE == 2) {
            const float sc = i2_read_f16_or_bf16(
                reinterpret_cast<const unsigned short*>(dn_scales)[r_idx * 36 + grp], sb_is_bf16);
            const float bi = i2_read_f16_or_bf16(
                reinterpret_cast<const unsigned short*>(dn_biases)[r_idx * 36 + grp], sb_is_bf16);
            float xs0 = 0.0f, xs1 = 0.0f;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
              xs0 += xp0[e];
              if (has_t1) xs1 += xp1[e];
            }
            acc0 = fmaf(sc, d0, fmaf(bi, xs0, acc0));
            if (has_t1) acc1 = fmaf(sc, d1, fmaf(bi, xs1, acc1));
          } else {
            const float sc = i2_decode_ue8m0(
                reinterpret_cast<const unsigned char*>(dn_scales)[r_idx * 36 + grp]) * 256.0f;
            acc0 = fmaf(sc, d0, acc0);
            if (has_t1) acc1 = fmaf(sc, d1, acc1);
          }
        }
      }

#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
        if (has_t1) acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
      }
      if (lane == 0) {
        down_pairs[static_cast<size_t>(p0) * N + row] = __float2bfloat16_rn(acc0);
        if (has_t1) {
          down_pairs[static_cast<size_t>(p1) * N + row] = __float2bfloat16_rn(acc1);
        }
      }
    }
    return;
  }
)cuda"
R"cuda(
  // ---- Hot Branch: cnt >= 3 (3..16) -> 4 tiles x 3 thirds cp.async.cg + bf16 MMA ----
  if constexpr (M_BUCKET > 0) {
    constexpr bool ALLOW_HI = (M_BUCKET == 2);
    const int g = lane >> 2;
    const int t = lane & 3;
    const int pair_g_lo = s_pair[g];
    const int pair_g_hi = ALLOW_HI ? s_pair[8 + g] : -1;
    const bool do_hi = ALLOW_HI && (cnt > 8);
    (void)pair_g_hi;
    (void)do_hi;

    const uint4* w_u4 = reinterpret_cast<const uint4*>(dn_w);
    auto load_third = [&](int st, int row0, int th) {
      const size_t r_off = exp_base + row0;
      if constexpr (MODE == 0) {
        const uint4* w_src = w_u4 + r_off * 36 + th * 12;
        for (int idx = tid; idx < 192; idx += 128) {
          const int r = idx / 12;
          const int c = idx - r * 12;
          ptx_cp_async16_cg(&smem.pipe[st].w[r][c], w_src + static_cast<size_t>(r) * 36 + c);
        }
        if (tid < 48) {
          const int r = tid / 3;
          const int c = tid - r * 3;
          const unsigned char* s_src = reinterpret_cast<const unsigned char*>(dn_scales) +
              (r_off + r) * 72 + th * 24 + c * 8;
          ptx_cp_async8_ca(reinterpret_cast<unsigned char*>(smem.pipe[st].sc[r]) + c * 8, s_src);
        }
      } else {
        const uint4* w_src = w_u4 + r_off * 72 + th * 24;
#pragma unroll
        for (int iter = 0; iter < 3; ++iter) {
          const int idx = tid + iter * 128;
          const int r = idx / 24;
          const int c = idx - r * 24;
          ptx_cp_async16_cg(&smem.pipe[st].w[r][c], w_src + static_cast<size_t>(r) * 72 + c);
        }
        if (tid < 48) {
          const int r = tid / 3;
          const int c = tid - r * 3;
          if constexpr (MODE == 4) {
            const unsigned char* s_src = reinterpret_cast<const unsigned char*>(dn_scales) +
                (r_off + r) * 36 + th * 12 + c * 4;
            ptx_cp_async4_ca(reinterpret_cast<unsigned char*>(smem.pipe[st].sc[r]) + c * 4, s_src);
          } else {
            const unsigned short* s_src = reinterpret_cast<const unsigned short*>(dn_scales) +
                (r_off + r) * 36 + th * 12 + c * 4;
            ptx_cp_async8_ca(&smem.pipe[st].sc[r][c * 4], s_src);
          }
        } else if constexpr (MODE == 2) {
          if (tid < 96) {
            const int k = tid - 48;
            const int r = k / 3;
            const int c = k - r * 3;
            const unsigned short* b_src = reinterpret_cast<const unsigned short*>(dn_biases) +
                (r_off + r) * 36 + th * 12 + c * 4;
            ptx_cp_async8_ca(&smem.pipe[st].bi[r][c * 4], b_src);
          }
        }
      }
      ptx_cp_commit();
    };

    load_third(0, base_row, 0);
    float tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int step = 0; step < 12; ++step) {
      const int cur = step & 1;
      const int nxt = cur ^ 1;
      if (step + 1 < 12) {
        const int n_ti = (step + 1) / 3;
        const int n_th = (step + 1) - n_ti * 3;
        load_third(nxt, base_row + n_ti * 16, n_th);
        ptx_cp_wait<1>();
      } else {
        ptx_cp_wait<0>();
      }
      __syncthreads();

      const int ti = step / 3;
      const int th = step - ti * 3;
      const int row0 = base_row + ti * 16;

      if constexpr (MODE == 0) {
        const unsigned short* w_r0 = reinterpret_cast<const unsigned short*>(smem.pipe[cur].w[g]);
        const unsigned short* w_r1 = reinterpret_cast<const unsigned short*>(smem.pipe[cur].w[g + 8]);
        const unsigned char* sc_r0 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g]);
        const unsigned char* sc_r1 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g + 8]);
#pragma unroll
        for (int kg = 0; kg < 6; ++kg) {
          const int lg = warp * 6 + kg;
          const int grp16 = th * 24 + lg;
          const unsigned short w0 = w_r0[lg * 4 + t];
          const unsigned short w1 = w_r1[lg * 4 + t];
          const unsigned int a0 = i2_e2m1x2_to_bf16x2(w0 & 0xFFu);
          const unsigned int a2 = i2_e2m1x2_to_bf16x2(w0 >> 8);
          const unsigned int a1 = i2_e2m1x2_to_bf16x2(w1 & 0xFFu);
          const unsigned int a3 = i2_e2m1x2_to_bf16x2(w1 >> 8);
          const uint2 xv0 = (pair_g_lo >= 0) ? reinterpret_cast<const uint2*>(
              mid + static_cast<size_t>(pair_g_lo) * K + grp16 * 16)[t] : make_uint2(0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
          const float sc0 = i2_decode_fp8_e4m3(sc_r0[lg]);
          const float sc1 = i2_decode_fp8_e4m3(sc_r1[lg]);
          tot_lo[0] = fmaf(sc0, d0, tot_lo[0]);
          tot_lo[1] = fmaf(sc0, d1, tot_lo[1]);
          tot_lo[2] = fmaf(sc1, d2, tot_lo[2]);
          tot_lo[3] = fmaf(sc1, d3, tot_lo[3]);
          if constexpr (ALLOW_HI) {
            if (do_hi) {
              const uint2 xv1 = (pair_g_hi >= 0) ? reinterpret_cast<const uint2*>(
                  mid + static_cast<size_t>(pair_g_hi) * K + grp16 * 16)[t] : make_uint2(0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, a0, a1, a2, a3, xv1.x, xv1.y);
              tot_hi[0] = fmaf(sc0, h0, tot_hi[0]);
              tot_hi[1] = fmaf(sc0, h1, tot_hi[1]);
              tot_hi[2] = fmaf(sc1, h2, tot_hi[2]);
              tot_hi[3] = fmaf(sc1, h3, tot_hi[3]);
            }
          }
        }
      } else {
        const uint2* w_r0 = reinterpret_cast<const uint2*>(smem.pipe[cur].w[g]);
        const uint2* w_r1 = reinterpret_cast<const uint2*>(smem.pipe[cur].w[g + 8]);
        const uint4* mid_u4 = reinterpret_cast<const uint4*>(mid);
#pragma unroll
        for (int kg = 0; kg < 3; ++kg) {
          const int lg = warp * 3 + kg;
          const int grp32 = th * 12 + lg;
          unsigned int wb0[4], wb1[4];
          i2_unpack8_to_bf16x2<MODE>(w_r0[lg * 4 + t], wb0);
          i2_unpack8_to_bf16x2<MODE>(w_r1[lg * 4 + t], wb1);

          const uint4 xv0 = (pair_g_lo >= 0)
              ? mid_u4[static_cast<size_t>(pair_g_lo) * 144 + grp32 * 4 + t]
              : make_uint4(0u, 0u, 0u, 0u);
          float d0, d1, d2, d3;
          ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
          ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

          float sc0, sc1;
          if constexpr (MODE == 4) {
            const unsigned char* sc_r0 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g]);
            const unsigned char* sc_r1 = reinterpret_cast<const unsigned char*>(smem.pipe[cur].sc[g + 8]);
            sc0 = i2_decode_ue8m0(sc_r0[lg]);
            sc1 = i2_decode_ue8m0(sc_r1[lg]);
          } else {
            sc0 = i2_read_f16_or_bf16(smem.pipe[cur].sc[g][lg], sb_is_bf16);
            sc1 = i2_read_f16_or_bf16(smem.pipe[cur].sc[g + 8][lg], sb_is_bf16);
          }

          if constexpr (MODE == 2) {
            float my_s0 = (pair_g_lo >= 0) ? i2_sum_bf16x8(xv0) : 0.0f;
            my_s0 += __shfl_xor_sync(0xffffffffu, my_s0, 1);
            my_s0 += __shfl_xor_sync(0xffffffffu, my_s0, 2);
            const float xs00 = __shfl_sync(0xffffffffu, my_s0, (2 * t) * 4);
            const float xs01 = __shfl_sync(0xffffffffu, my_s0, (2 * t + 1) * 4);
            const float bi0 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g][lg], sb_is_bf16);
            const float bi1 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g + 8][lg], sb_is_bf16);
            tot_lo[0] = fmaf(sc0, d0, fmaf(bi0, xs00, tot_lo[0]));
            tot_lo[1] = fmaf(sc0, d1, fmaf(bi0, xs01, tot_lo[1]));
            tot_lo[2] = fmaf(sc1, d2, fmaf(bi1, xs00, tot_lo[2]));
            tot_lo[3] = fmaf(sc1, d3, fmaf(bi1, xs01, tot_lo[3]));
          } else {
            tot_lo[0] = fmaf(sc0, d0, tot_lo[0]);
            tot_lo[1] = fmaf(sc0, d1, tot_lo[1]);
            tot_lo[2] = fmaf(sc1, d2, tot_lo[2]);
            tot_lo[3] = fmaf(sc1, d3, tot_lo[3]);
          }

          if constexpr (ALLOW_HI) {
            if (do_hi) {
              const uint4 xv1 = (pair_g_hi >= 0)
                  ? mid_u4[static_cast<size_t>(pair_g_hi) * 144 + grp32 * 4 + t]
                  : make_uint4(0u, 0u, 0u, 0u);
              float h0, h1, h2, h3;
              ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
              ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
              if constexpr (MODE == 2) {
                float my_s1 = (pair_g_hi >= 0) ? i2_sum_bf16x8(xv1) : 0.0f;
                my_s1 += __shfl_xor_sync(0xffffffffu, my_s1, 1);
                my_s1 += __shfl_xor_sync(0xffffffffu, my_s1, 2);
                const float xs10 = __shfl_sync(0xffffffffu, my_s1, (2 * t) * 4);
                const float xs11 = __shfl_sync(0xffffffffu, my_s1, (2 * t + 1) * 4);
                const float bi0 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g][lg], sb_is_bf16);
                const float bi1 = i2_read_f16_or_bf16(smem.pipe[cur].bi[g + 8][lg], sb_is_bf16);
                tot_hi[0] = fmaf(sc0, h0, fmaf(bi0, xs10, tot_hi[0]));
                tot_hi[1] = fmaf(sc0, h1, fmaf(bi0, xs11, tot_hi[1]));
                tot_hi[2] = fmaf(sc1, h2, fmaf(bi1, xs10, tot_hi[2]));
                tot_hi[3] = fmaf(sc1, h3, fmaf(bi1, xs11, tot_hi[3]));
              } else {
                tot_hi[0] = fmaf(sc0, h0, tot_hi[0]);
                tot_hi[1] = fmaf(sc0, h1, tot_hi[1]);
                tot_hi[2] = fmaf(sc1, h2, tot_hi[2]);
                tot_hi[3] = fmaf(sc1, h3, tot_hi[3]);
              }
            }
          }
        }
      }

      if (th == 2) {
        s_red[warp][2 * t][g] = tot_lo[0];
        s_red[warp][2 * t + 1][g] = tot_lo[1];
        s_red[warp][2 * t][g + 8] = tot_lo[2];
        s_red[warp][2 * t + 1][g + 8] = tot_lo[3];
        if constexpr (ALLOW_HI) {
          if (do_hi) {
            s_red[warp][8 + 2 * t][g] = tot_hi[0];
            s_red[warp][8 + 2 * t + 1][g] = tot_hi[1];
            s_red[warp][8 + 2 * t][g + 8] = tot_hi[2];
            s_red[warp][8 + 2 * t + 1][g + 8] = tot_hi[3];
          }
        }
        __syncthreads();
        const int total_out = cnt * 16;
        for (int idx = tid; idx < total_out; idx += 128) {
          const int c = idx >> 4;
          const int r = idx & 15;
          const float val = (s_red[0][c][r] + s_red[1][c][r]) +
                            (s_red[2][c][r] + s_red[3][c][r]);
          down_pairs[static_cast<size_t>(s_pair[c]) * N + row0 + r] = __float2bfloat16_rn(val);
        }
        tot_lo[0] = tot_lo[1] = tot_lo[2] = tot_lo[3] = 0.0f;
        tot_hi[0] = tot_hi[1] = tot_hi[2] = tot_hi[3] = 0.0f;
      }
      __syncthreads();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3-NVFP4. Silicon-Roofline Linear-18KB Full-Row Down Projection (K=1152, N=2304)
//          Unified BN=32 across M in 1..=16: Grid(72, P), 9w (288 thr),
//          18KB W + 2.25KB S (3 barriers total, down from 24), 0 spills,
//          21.3 KB SMEM (3 CTAs/SM = 63.9 KB <= 64 KB carveout -> full 64 KB L1).
//          Reuses mid in registers across 2x16 rows -> 0-ULP batch-invariant!
// ---------------------------------------------------------------------------
template <bool ALLOW_HI>
__global__ __launch_bounds__(288, 3)
void inkling2_moe_down_nvfp4_few(
    const unsigned int* __restrict__ dn_w,
    const unsigned char* __restrict__ dn_scales,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ down_pairs) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int N = 2304;
  constexpr int K = 1152;
  constexpr int BN = 32;
  const int row0 = blockIdx.x * BN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t base_r = static_cast<size_t>(expert) * N + row0;

  union SharedNvfp4Dn {
    struct {
      alignas(16) uint4 w[BN][37]; // 18,944 B (37*4 == 20 mod 32 -> 0 bank conflicts)
      alignas(16) uint4 sc[144];   //  2,304 B (72B/row == 18 banks/row -> 0 bank conflicts)
    } buf;
    struct {
      float t0[9][ALLOW_HI ? 16 : 8][17];
      float t1[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedNvfp4Dn smem;
  __shared__ int s_pair[16];
#if !I2_HAS_HW_FP4_CVT
  __shared__ unsigned int s_e2m1_lut[256];
  if (tid < 256) s_e2m1_lut[tid] = i2_e2m1x2_to_bf16x2(static_cast<unsigned int>(tid));
#endif

  if (tid < cnt) s_pair[tid] = pair_indices[first + tid];
  else if (tid < 16) s_pair[tid] = -1;

  // Division-free map tid in 0..287 -> (lr in 0..7, lc in 0..35)
  const int lr = (tid * 57) >> 11;
  const int lc = tid - lr * 36;

  const uint4* wp = reinterpret_cast<const uint4*>(dn_w) + base_r * 36;
  const uint4* sp = reinterpret_cast<const uint4*>(dn_scales + base_r * 72);
  const uint4 v0 = ptx_ld_cs_u4(wp + tid);
  const uint4 v1 = ptx_ld_cs_u4(wp + 288 + tid);
  const uint4 v2 = ptx_ld_cs_u4(wp + 576 + tid);
  const uint4 v3 = ptx_ld_cs_u4(wp + 864 + tid);
  uint4 sv;
  if (tid < 144) sv = ptx_ld_cs_u4(sp + tid);
  smem.buf.w[lr][lc] = v0;
  smem.buf.w[8 + lr][lc] = v1;
  smem.buf.w[16 + lr][lc] = v2;
  smem.buf.w[24 + lr][lc] = v3;
  if (tid < 144) smem.buf.sc[tid] = sv;
  __syncthreads();

  const int pair_g_lo = s_pair[g];
  const uint2* m_lo_p = (pair_g_lo >= 0)
      ? reinterpret_cast<const uint2*>(mid + static_cast<size_t>(pair_g_lo) * K) : nullptr;
  [[maybe_unused]] const int pair_g_hi = ALLOW_HI ? s_pair[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);
  [[maybe_unused]] const uint2* m_hi_p = (ALLOW_HI && pair_g_hi >= 0)
      ? reinterpret_cast<const uint2*>(mid + static_cast<size_t>(pair_g_hi) * K) : nullptr;

  float t0_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float t1_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float t0_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  [[maybe_unused]] float t1_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const unsigned short* w00 = reinterpret_cast<const unsigned short*>(&smem.buf.w[g][warp * 4]);
  const unsigned short* w01 = reinterpret_cast<const unsigned short*>(&smem.buf.w[g + 8][warp * 4]);
  const unsigned short* w10 = reinterpret_cast<const unsigned short*>(&smem.buf.w[16 + g][warp * 4]);
  const unsigned short* w11 = reinterpret_cast<const unsigned short*>(&smem.buf.w[24 + g][warp * 4]);
  const unsigned short* sc_base = reinterpret_cast<const unsigned short*>(smem.buf.sc);
  const unsigned short* s00 = sc_base + g * 36 + warp * 4;
  const unsigned short* s01 = sc_base + (g + 8) * 36 + warp * 4;
  const unsigned short* s10 = sc_base + (16 + g) * 36 + warp * 4;
  const unsigned short* s11 = sc_base + (24 + g) * 36 + warp * 4;
  const int grp_base = warp * 8;

  auto decode_w = [&](unsigned short p, unsigned int& alo, unsigned int& ahi) {
#if I2_HAS_HW_FP4_CVT
    alo = i2_fast_e2m1x2_to_bf16x2(p & 0xFFu);
    ahi = i2_fast_e2m1x2_to_bf16x2(p >> 8);
#else
    alo = s_e2m1_lut[p & 0xFFu];
    ahi = s_e2m1_lut[p >> 8];
#endif
  };

#pragma unroll 2
  for (int kg2 = 0; kg2 < 4; ++kg2) {
    const float2 sf00 = i2_fast_e4m3x2_to_f2(s00[kg2]);
    const float2 sf01 = i2_fast_e4m3x2_to_f2(s01[kg2]);
    const float2 sf10 = i2_fast_e4m3x2_to_f2(s10[kg2]);
    const float2 sf11 = i2_fast_e4m3x2_to_f2(s11[kg2]);
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int kg = kg2 * 2 + sub;
      const int grp = grp_base + kg;
      const uint2 xv0 = m_lo_p ? m_lo_p[grp * 4 + t] : make_uint2(0u, 0u);
      [[maybe_unused]] uint2 xv1 = make_uint2(0u, 0u);
      if constexpr (ALLOW_HI) {
        if (do_hi && m_hi_p) xv1 = m_hi_p[grp * 4 + t];
      }
      {
        unsigned int a0, a1, a2, a3;
        decode_w(w00[kg * 4 + t], a0, a2);
        decode_w(w01[kg * 4 + t], a1, a3);
        const float sc0 = (sub == 0) ? sf00.x : sf00.y;
        const float sc1 = (sub == 0) ? sf01.x : sf01.y;
        float d0, d1, d2, d3;
        ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
        t0_lo[0] = fmaf(sc0, d0, t0_lo[0]);
        t0_lo[1] = fmaf(sc0, d1, t0_lo[1]);
        t0_lo[2] = fmaf(sc1, d2, t0_lo[2]);
        t0_lo[3] = fmaf(sc1, d3, t0_lo[3]);
        if constexpr (ALLOW_HI) {
          if (do_hi) {
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv1.x, xv1.y);
            t0_hi[0] = fmaf(sc0, d0, t0_hi[0]);
            t0_hi[1] = fmaf(sc0, d1, t0_hi[1]);
            t0_hi[2] = fmaf(sc1, d2, t0_hi[2]);
            t0_hi[3] = fmaf(sc1, d3, t0_hi[3]);
          }
        }
      }
      {
        unsigned int a0, a1, a2, a3;
        decode_w(w10[kg * 4 + t], a0, a2);
        decode_w(w11[kg * 4 + t], a1, a3);
        const float sc0 = (sub == 0) ? sf10.x : sf10.y;
        const float sc1 = (sub == 0) ? sf11.x : sf11.y;
        float d0, d1, d2, d3;
        ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv0.x, xv0.y);
        t1_lo[0] = fmaf(sc0, d0, t1_lo[0]);
        t1_lo[1] = fmaf(sc0, d1, t1_lo[1]);
        t1_lo[2] = fmaf(sc1, d2, t1_lo[2]);
        t1_lo[3] = fmaf(sc1, d3, t1_lo[3]);
        if constexpr (ALLOW_HI) {
          if (do_hi) {
            ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, a0, a1, a2, a3, xv1.x, xv1.y);
            t1_hi[0] = fmaf(sc0, d0, t1_hi[0]);
            t1_hi[1] = fmaf(sc0, d1, t1_hi[1]);
            t1_hi[2] = fmaf(sc1, d2, t1_hi[2]);
            t1_hi[3] = fmaf(sc1, d3, t1_hi[3]);
          }
        }
      }
    }
  }
  __syncthreads();

  smem.red.t0[warp][2 * t][g] = t0_lo[0];
  smem.red.t0[warp][2 * t + 1][g] = t0_lo[1];
  smem.red.t0[warp][2 * t][g + 8] = t0_lo[2];
  smem.red.t0[warp][2 * t + 1][g + 8] = t0_lo[3];
  smem.red.t1[warp][2 * t][g] = t1_lo[0];
  smem.red.t1[warp][2 * t + 1][g] = t1_lo[1];
  smem.red.t1[warp][2 * t][g + 8] = t1_lo[2];
  smem.red.t1[warp][2 * t + 1][g + 8] = t1_lo[3];
  if constexpr (ALLOW_HI) {
    if (do_hi) {
      smem.red.t0[warp][8 + 2 * t][g] = t0_hi[0];
      smem.red.t0[warp][8 + 2 * t + 1][g] = t0_hi[1];
      smem.red.t0[warp][8 + 2 * t][g + 8] = t0_hi[2];
      smem.red.t0[warp][8 + 2 * t + 1][g + 8] = t0_hi[3];
      smem.red.t1[warp][8 + 2 * t][g] = t1_hi[0];
      smem.red.t1[warp][8 + 2 * t + 1][g] = t1_hi[1];
      smem.red.t1[warp][8 + 2 * t][g + 8] = t1_hi[2];
      smem.red.t1[warp][8 + 2 * t + 1][g + 8] = t1_hi[3];
    }
  }
  __syncthreads();

  const int total_out = cnt * BN;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 5;
    const int r = idx & 31;
    float val = 0.0f;
    if (r < 16) {
#pragma unroll
      for (int ww = 0; ww < 9; ++ww) val += smem.red.t0[ww][c][r];
    } else {
#pragma unroll
      for (int ww = 0; ww < 9; ++ww) val += smem.red.t1[ww][c][r - 16];
    }
    down_pairs[static_cast<size_t>(s_pair[c]) * N + row0 + r] = __float2bfloat16_rn(val);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3-SYMQ8. Silicon-Roofline Linear-18KB Full-Row SymQ8 Down Projection
//          Unified BN=32 across M in 1..=16: Grid(72, P), 9w (288 thr),
//          2 passes x (18KB W + 1.125KB S contiguous ld.global.cs.v4.u32),
//          5 barriers total (down from 24), 0 B spills, 20.16 KB SMEM
//          (3 CTAs/SM = 60.5 KB <= 64 KB carveout -> full 64 KB L1 for mid).
//          0-ULP batch-invariant across M in 1..=16.
// ---------------------------------------------------------------------------
template <bool ALLOW_HI>
__global__ __launch_bounds__(288, 3)
void inkling2_moe_down_symq8_few(
    const unsigned int* __restrict__ dn_w,
    const unsigned short* __restrict__ dn_scales,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ down_pairs,
    int sb_is_bf16) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int N = 2304;
  constexpr int K = 1152;
  constexpr int BN = 32;
  const int row0 = blockIdx.x * BN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t base_r = static_cast<size_t>(expert) * N + row0;

  union SharedSymQ8Dn {
    struct {
      alignas(16) uint4 w[16][74]; // 18,944 B (74*4 == 8 mod 32 -> 0 LDS.64 bank conflicts)
      alignas(16) uint4 sc[72];    //  1,152 B (36 u16/row == 18 banks/row -> 0 conflicts)
    } buf;
    struct {
      float t0[9][ALLOW_HI ? 16 : 8][17];
      float t1[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedSymQ8Dn smem;
  __shared__ int s_pair[16];

  if (tid < cnt) s_pair[tid] = pair_indices[first + tid];
  else if (tid < 16) s_pair[tid] = -1;

  // Division-free mapping: tid in 0..287 -> (lr in 0..3, lc in 0..71)
  const int r_hi = (tid >= 144) ? 2 : 0;
  const int rem = tid - r_hi * 72;
  const int r_lo = (rem >= 72) ? 1 : 0;
  const int lr = r_hi + r_lo;
  const int lc = rem - r_lo * 72;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(dn_w);
  const uint4* s_u4 = reinterpret_cast<const uint4*>(dn_scales);

  auto load_half = [&](int half) {
    const size_t r_off = base_r + (half ? 16 : 0);
    const uint4* wp = w_u4 + r_off * 72;
    const uint4* sp = s_u4 + (r_off * 36) / 8;
    const uint4 v0 = ptx_ld_cs_u4(wp + tid);
    const uint4 v1 = ptx_ld_cs_u4(wp + 288 + tid);
    const uint4 v2 = ptx_ld_cs_u4(wp + 576 + tid);
    const uint4 v3 = ptx_ld_cs_u4(wp + 864 + tid);
    uint4 sv;
    if (tid < 72) sv = ptx_ld_cs_u4(sp + tid);
    smem.buf.w[lr][lc] = v0;
    smem.buf.w[4 + lr][lc] = v1;
    smem.buf.w[8 + lr][lc] = v2;
    smem.buf.w[12 + lr][lc] = v3;
    if (tid < 72) smem.buf.sc[tid] = sv;
  };

  load_half(0);
  __syncthreads();

  const int pair_g_lo = s_pair[g];
  const uint4* m_lo_p = (pair_g_lo >= 0)
      ? reinterpret_cast<const uint4*>(mid + static_cast<size_t>(pair_g_lo) * K) : nullptr;
  [[maybe_unused]] const int pair_g_hi = ALLOW_HI ? s_pair[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);
  [[maybe_unused]] const uint4* m_hi_p = (ALLOW_HI && pair_g_hi >= 0)
      ? reinterpret_cast<const uint4*>(mid + static_cast<size_t>(pair_g_hi) * K) : nullptr;

  const uint2* w_r0 = reinterpret_cast<const uint2*>(&smem.buf.w[g][warp * 8]);
  const uint2* w_r1 = reinterpret_cast<const uint2*>(&smem.buf.w[g + 8][warp * 8]);
  const unsigned short* sc_base = reinterpret_cast<const unsigned short*>(smem.buf.sc);
  const unsigned short* sc_r0 = sc_base + g * 36 + warp * 4;
  const unsigned short* sc_r1 = sc_base + (g + 8) * 36 + warp * 4;
  const int grp_base = warp * 4;

  auto compute_half = [&](float acc_lo[4], float acc_hi[4]) {
    acc_lo[0] = acc_lo[1] = acc_lo[2] = acc_lo[3] = 0.0f;
    if constexpr (ALLOW_HI) {
      acc_hi[0] = acc_hi[1] = acc_hi[2] = acc_hi[3] = 0.0f;
    }
#pragma unroll 1
    for (int kg = 0; kg < 4; ++kg) {
      const int grp32 = grp_base + kg;
      unsigned int wb0[4], wb1[4];
      i2_unpack8_to_bf16x2<1>(w_r0[kg * 4 + t], wb0);
      i2_unpack8_to_bf16x2<1>(w_r1[kg * 4 + t], wb1);

      const uint4 xv0 = m_lo_p ? m_lo_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
      float d0, d1, d2, d3;
      ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
      ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

      const float sc0 = i2_read_f16_or_bf16(sc_r0[kg], sb_is_bf16);
      const float sc1 = i2_read_f16_or_bf16(sc_r1[kg], sb_is_bf16);
      acc_lo[0] = fmaf(sc0, d0, acc_lo[0]);
      acc_lo[1] = fmaf(sc0, d1, acc_lo[1]);
      acc_lo[2] = fmaf(sc1, d2, acc_lo[2]);
      acc_lo[3] = fmaf(sc1, d3, acc_lo[3]);

      if constexpr (ALLOW_HI) {
        if (do_hi) {
          const uint4 xv1 = m_hi_p ? m_hi_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
          float h0, h1, h2, h3;
          ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
          ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
          acc_hi[0] = fmaf(sc0, h0, acc_hi[0]);
          acc_hi[1] = fmaf(sc0, h1, acc_hi[1]);
          acc_hi[2] = fmaf(sc1, h2, acc_hi[2]);
          acc_hi[3] = fmaf(sc1, h3, acc_hi[3]);
        }
      }
    }
  };

  float t0_lo[4], t1_lo[4];
  [[maybe_unused]] float t0_hi[4], t1_hi[4];
  compute_half(t0_lo, t0_hi);
  __syncthreads();

  load_half(1);
  __syncthreads();

  compute_half(t1_lo, t1_hi);
  __syncthreads();

  smem.red.t0[warp][2 * t][g] = t0_lo[0];
  smem.red.t0[warp][2 * t + 1][g] = t0_lo[1];
  smem.red.t0[warp][2 * t][g + 8] = t0_lo[2];
  smem.red.t0[warp][2 * t + 1][g + 8] = t0_lo[3];
  smem.red.t1[warp][2 * t][g] = t1_lo[0];
  smem.red.t1[warp][2 * t + 1][g] = t1_lo[1];
  smem.red.t1[warp][2 * t][g + 8] = t1_lo[2];
  smem.red.t1[warp][2 * t + 1][g + 8] = t1_lo[3];
  if constexpr (ALLOW_HI) {
    if (do_hi) {
      smem.red.t0[warp][8 + 2 * t][g] = t0_hi[0];
      smem.red.t0[warp][8 + 2 * t + 1][g] = t0_hi[1];
      smem.red.t0[warp][8 + 2 * t][g + 8] = t0_hi[2];
      smem.red.t0[warp][8 + 2 * t + 1][g + 8] = t0_hi[3];
      smem.red.t1[warp][8 + 2 * t][g] = t1_hi[0];
      smem.red.t1[warp][8 + 2 * t + 1][g] = t1_hi[1];
      smem.red.t1[warp][8 + 2 * t][g + 8] = t1_hi[2];
      smem.red.t1[warp][8 + 2 * t + 1][g + 8] = t1_hi[3];
    }
  }
  __syncthreads();

  const int total_out = cnt * BN;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 5;
    const int r = idx & 31;
    float val = 0.0f;
    if (r < 16) {
#pragma unroll
      for (int ww = 0; ww < 9; ++ww) val += smem.red.t0[ww][c][r];
    } else {
#pragma unroll
      for (int ww = 0; ww < 9; ++ww) val += smem.red.t1[ww][c][r - 16];
    }
    down_pairs[static_cast<size_t>(s_pair[c]) * N + row0 + r] = __float2bfloat16_rn(val);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3-MXFP8. Silicon-Roofline Linear-18KB Full-Row MXFP8 Down Projection
//          Unified BN=32 across M in 1..=16: Grid(72, P), 9w (288 thr),
//          2 passes x (18KB W + 576B UE8M0 S contiguous ld.global.cs.v4.u32),
//          1-instr HW cvt.rn.bf16x2.e4m3x2 / f16x2 + 1 u32 UE8M0 LDS per row,
//          5 barriers total (down from 24), 0 B spills, 19.2 KB SMEM
//          (3 CTAs/SM = 58.9 KB <= 64 KB carveout -> full 64 KB L1 for mid).
//          0-ULP batch-invariant across M in 1..=16.
// ---------------------------------------------------------------------------
template <bool ALLOW_HI>
__global__ __launch_bounds__(288, 3)
void inkling2_moe_down_mxfp8_few(
    const unsigned int* __restrict__ dn_w,
    const unsigned char* __restrict__ dn_scales,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ down_pairs) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  constexpr int N = 2304;
  constexpr int K = 1152;
  constexpr int BN = 32;
  const int row0 = blockIdx.x * BN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t base_r = static_cast<size_t>(expert) * N + row0;

  union SharedMxfp8Dn {
    struct {
      alignas(16) uint4 w[16][74]; // 18,944 B (74*4 == 8 mod 32 -> 0 LDS.64 bank conflicts)
      alignas(16) uint4 sc[36];    //    576 B (9 u32/row -> gcd(9,32)=1 -> 0 bank conflicts)
    } buf;
    struct {
      float t0[9][ALLOW_HI ? 16 : 8][17];
      float t1[9][ALLOW_HI ? 16 : 8][17];
    } red;
  };
  __shared__ SharedMxfp8Dn smem;
  __shared__ int s_pair[16];

  if (tid < cnt) s_pair[tid] = pair_indices[first + tid];
  else if (tid < 16) s_pair[tid] = -1;

  // Division-free mapping: tid in 0..287 -> (lr in 0..3, lc in 0..71)
  const int r_hi = (tid >= 144) ? 2 : 0;
  const int rem = tid - r_hi * 72;
  const int r_lo = (rem >= 72) ? 1 : 0;
  const int lr = r_hi + r_lo;
  const int lc = rem - r_lo * 72;

  const uint4* w_u4 = reinterpret_cast<const uint4*>(dn_w);
  const uint4* s_u4 = reinterpret_cast<const uint4*>(dn_scales);

  auto load_half = [&](int half) {
    const size_t r_off = base_r + (half ? 16 : 0);
    const uint4* wp = w_u4 + r_off * 72;
    const uint4* sp = s_u4 + (r_off * 36) / 16;
    const uint4 v0 = ptx_ld_cs_u4(wp + tid);
    const uint4 v1 = ptx_ld_cs_u4(wp + 288 + tid);
    const uint4 v2 = ptx_ld_cs_u4(wp + 576 + tid);
    const uint4 v3 = ptx_ld_cs_u4(wp + 864 + tid);
    uint4 sv;
    if (tid < 36) sv = ptx_ld_cs_u4(sp + tid);
    smem.buf.w[lr][lc] = v0;
    smem.buf.w[4 + lr][lc] = v1;
    smem.buf.w[8 + lr][lc] = v2;
    smem.buf.w[12 + lr][lc] = v3;
    if (tid < 36) smem.buf.sc[tid] = sv;
  };

  load_half(0);
  __syncthreads();

  const int pair_g_lo = s_pair[g];
  const uint4* m_lo_p = (pair_g_lo >= 0)
      ? reinterpret_cast<const uint4*>(mid + static_cast<size_t>(pair_g_lo) * K) : nullptr;
  [[maybe_unused]] const int pair_g_hi = ALLOW_HI ? s_pair[8 + g] : -1;
  [[maybe_unused]] const bool do_hi = ALLOW_HI && (cnt > 8);
  [[maybe_unused]] const uint4* m_hi_p = (ALLOW_HI && pair_g_hi >= 0)
      ? reinterpret_cast<const uint4*>(mid + static_cast<size_t>(pair_g_hi) * K) : nullptr;

  const uint2* w_r0 = reinterpret_cast<const uint2*>(&smem.buf.w[g][warp * 8]);
  const uint2* w_r1 = reinterpret_cast<const uint2*>(&smem.buf.w[g + 8][warp * 8]);
  const unsigned int* sc_u32 = reinterpret_cast<const unsigned int*>(smem.buf.sc);
  const int grp_base = warp * 4;

  auto compute_half = [&](float acc_lo[4], float acc_hi[4]) {
    acc_lo[0] = acc_lo[1] = acc_lo[2] = acc_lo[3] = 0.0f;
    if constexpr (ALLOW_HI) {
      acc_hi[0] = acc_hi[1] = acc_hi[2] = acc_hi[3] = 0.0f;
    }
    const unsigned int sc0_w = sc_u32[g * 9 + warp];
    const unsigned int sc1_w = sc_u32[(g + 8) * 9 + warp];
#pragma unroll 1
    for (int kg = 0; kg < 4; ++kg) {
      const int grp32 = grp_base + kg;
      unsigned int wb0[4], wb1[4];
      i2_unpack8_to_bf16x2<4>(w_r0[kg * 4 + t], wb0);
      i2_unpack8_to_bf16x2<4>(w_r1[kg * 4 + t], wb1);

      const uint4 xv0 = m_lo_p ? m_lo_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
      float d0, d1, d2, d3;
      ptx_mma_m16n8k16_bf16_z(d0, d1, d2, d3, wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
      ptx_mma_m16n8k16_bf16(d0, d1, d2, d3, wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);

      const float sc0 = i2_decode_ue8m0((sc0_w >> (kg * 8)) & 0xFFu);
      const float sc1 = i2_decode_ue8m0((sc1_w >> (kg * 8)) & 0xFFu);
      acc_lo[0] = fmaf(sc0, d0, acc_lo[0]);
      acc_lo[1] = fmaf(sc0, d1, acc_lo[1]);
      acc_lo[2] = fmaf(sc1, d2, acc_lo[2]);
      acc_lo[3] = fmaf(sc1, d3, acc_lo[3]);

      if constexpr (ALLOW_HI) {
        if (do_hi) {
          const uint4 xv1 = m_hi_p ? m_hi_p[grp32 * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
          float h0, h1, h2, h3;
          ptx_mma_m16n8k16_bf16_z(h0, h1, h2, h3, wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
          ptx_mma_m16n8k16_bf16(h0, h1, h2, h3, wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
          acc_hi[0] = fmaf(sc0, h0, acc_hi[0]);
          acc_hi[1] = fmaf(sc0, h1, acc_hi[1]);
          acc_hi[2] = fmaf(sc1, h2, acc_hi[2]);
          acc_hi[3] = fmaf(sc1, h3, acc_hi[3]);
        }
      }
    }
  };

  float t0_lo[4], t1_lo[4];
  [[maybe_unused]] float t0_hi[4], t1_hi[4];
  compute_half(t0_lo, t0_hi);
  __syncthreads();

  load_half(1);
  __syncthreads();

  compute_half(t1_lo, t1_hi);
  __syncthreads();

  smem.red.t0[warp][2 * t][g] = t0_lo[0];
  smem.red.t0[warp][2 * t + 1][g] = t0_lo[1];
  smem.red.t0[warp][2 * t][g + 8] = t0_lo[2];
  smem.red.t0[warp][2 * t + 1][g + 8] = t0_lo[3];
  smem.red.t1[warp][2 * t][g] = t1_lo[0];
  smem.red.t1[warp][2 * t + 1][g] = t1_lo[1];
  smem.red.t1[warp][2 * t][g + 8] = t1_lo[2];
  smem.red.t1[warp][2 * t + 1][g + 8] = t1_lo[3];
  if constexpr (ALLOW_HI) {
    if (do_hi) {
      smem.red.t0[warp][8 + 2 * t][g] = t0_hi[0];
      smem.red.t0[warp][8 + 2 * t + 1][g] = t0_hi[1];
      smem.red.t0[warp][8 + 2 * t][g + 8] = t0_hi[2];
      smem.red.t0[warp][8 + 2 * t + 1][g + 8] = t0_hi[3];
      smem.red.t1[warp][8 + 2 * t][g] = t1_hi[0];
      smem.red.t1[warp][8 + 2 * t + 1][g] = t1_hi[1];
      smem.red.t1[warp][8 + 2 * t][g + 8] = t1_hi[2];
      smem.red.t1[warp][8 + 2 * t + 1][g + 8] = t1_hi[3];
    }
  }
  __syncthreads();

  const int total_out = cnt * BN;
  for (int idx = tid; idx < total_out; idx += 288) {
    const int c = idx >> 5;
    const int r = idx & 31;
    float val = 0.0f;
    if (r < 16) {
#pragma unroll
      for (int ww = 0; ww < 9; ++ww) val += smem.red.t0[ww][c][r];
    } else {
#pragma unroll
      for (int ww = 0; ww < 9; ++ww) val += smem.red.t1[ww][c][r - 16];
    }
    down_pairs[static_cast<size_t>(s_pair[c]) * N + row0 + r] = __float2bfloat16_rn(val);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 4A. Combine-only (when post_conv_w == nullptr, M in 1..=16)
//     Grid: dim3(1, M, 1), Block: 288 (288 * 8 = 2304 exact!).
// ---------------------------------------------------------------------------
template <bool EMIT_PRE_CONV>
__global__ __launch_bounds__(288, 2) void inkling2_moe_combine_only_few(
    const nv_bfloat16* __restrict__ down_pairs,
    const float* __restrict__ gamma,
    nv_bfloat16* __restrict__ out_y,
    nv_bfloat16* __restrict__ out_pre_conv,
    int M) {
  const int tok = blockIdx.y;
  const int col8 = threadIdx.x;
  if (tok >= M || col8 >= 288) return;

  __shared__ float s_g[8];
  if (col8 < 8) s_g[col8] = gamma[tok * 8 + col8];
  __syncthreads();

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint4* dp_u4 = reinterpret_cast<const uint4*>(down_pairs);
  const int bp = tok * 8;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float sc = s_g[k];
    const uint4 raw = dp_u4[static_cast<size_t>(bp + k) * 288 + col8];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      acc[2 * j] = fmaf(f.x, sc, acc[2 * j]);
      acc[2 * j + 1] = fmaf(f.y, sc, acc[2 * j + 1]);
    }
  }
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int j = 0; j < 4; ++j) op[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  reinterpret_cast<uint4*>(out_y)[static_cast<size_t>(tok) * 288 + col8] = ov;
  if constexpr (EMIT_PRE_CONV) {
    reinterpret_cast<uint4*>(out_pre_conv)[static_cast<size_t>(tok) * 288 + col8] = ov;
  }
}

// ---------------------------------------------------------------------------
// 4B. Fused Combine + Static ShortConv(k=4, linear OR tree) + Residual + Hist + Tape
//     ZERO DRAM roundtrip for moe_out! Stages hist[0..2] + moe_out[0..M-1] in SMEM!
//     Grid: dim3(144, 1, 1), Block: 128 (16 cols x 8 token workers = 2304 cols exact).
// ---------------------------------------------------------------------------
template <bool HAS_RESIDUAL, bool HAS_TAPS, bool EMIT_PRE_CONV>
__global__ __launch_bounds__(128, 4) void inkling2_moe_combine_sconv_res_few(
    const nv_bfloat16* __restrict__ down_pairs,
    const float* __restrict__ gamma,
    const float* __restrict__ post_conv_w,
    const nv_bfloat16* __restrict__ hist_in,
    const nv_bfloat16* __restrict__ residual,
    const int* __restrict__ conv_taps,
    nv_bfloat16* __restrict__ out_y,
    nv_bfloat16* __restrict__ out_hist,
    nv_bfloat16* __restrict__ out_pre_conv,
    int M,
    int update_hist) {
  constexpr int D = 2304;
  constexpr float kScale = 0.44721359549995793f; // 1 / sqrt(5)
  const int tid = threadIdx.x;
  const int lc = tid & 15;
  const int w_id = tid >> 4; // 0..7
  const int col = blockIdx.x * 16 + lc;

  __shared__ nv_bfloat16 s_full[19][16];
  __shared__ float s_gamma[128];

  if (tid < M * 8) s_gamma[tid] = gamma[tid];
  if (w_id < 3) {
    s_full[w_id][lc] = hist_in[static_cast<size_t>(w_id) * D + col];
  }
  __syncthreads();

  for (int t = w_id; t < M; t += 8) {
    const int bp = t * 8;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float dv = __bfloat162float(down_pairs[static_cast<size_t>(bp + k) * D + col]);
      sum = fmaf(dv, s_gamma[bp + k], sum);
    }
    const nv_bfloat16 moe_b = __float2bfloat16_rn(sum);
    s_full[3 + t][lc] = moe_b;
    if constexpr (EMIT_PRE_CONV) {
      out_pre_conv[static_cast<size_t>(t) * D + col] = moe_b;
    }
  }
  __syncthreads();

  const float4 wf = reinterpret_cast<const float4*>(post_conv_w)[col];
  for (int t = w_id; t < M; t += 8) {
    float x3, x2, x1, x0;
    if constexpr (HAS_TAPS) {
      const int4 taps = reinterpret_cast<const int4*>(conv_taps)[t];
      auto read_tap = [&](int idx) -> float {
        const int fidx = idx + 3;
        return (fidx >= 0 && fidx < 3 + M) ? __bfloat162float(s_full[fidx][lc]) : 0.0f;
      };
      x3 = read_tap(taps.x);
      x2 = read_tap(taps.y);
      x1 = read_tap(taps.z);
      x0 = read_tap(taps.w);
    } else {
      x3 = __bfloat162float(s_full[t + 0][lc]);
      x2 = __bfloat162float(s_full[t + 1][lc]);
      x1 = __bfloat162float(s_full[t + 2][lc]);
      x0 = __bfloat162float(s_full[t + 3][lc]);
    }
    const float conv_acc = __fadd_rn(
        __fadd_rn(__fadd_rn(__fadd_rn(x0, __fmul_rn(wf.x, x3)), __fmul_rn(wf.y, x2)),
                  __fmul_rn(wf.z, x1)),
        __fmul_rn(wf.w, x0));
    const nv_bfloat16 conv_bf = __float2bfloat16_rn(__fmul_rn(conv_acc, kScale));
    const size_t off = static_cast<size_t>(t) * D + col;
    if constexpr (HAS_RESIDUAL) {
      out_y[off] = __float2bfloat16_rn(__bfloat162float(residual[off]) + __bfloat162float(conv_bf));
    } else {
      out_y[off] = conv_bf;
    }
  }

  if (out_hist != nullptr && w_id < 3) {
    out_hist[static_cast<size_t>(w_id) * D + col] =
        s_full[(update_hist ? M : 0) + w_id][lc];
  }
}
)cuda"
