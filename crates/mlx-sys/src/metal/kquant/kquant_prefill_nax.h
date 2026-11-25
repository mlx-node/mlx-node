// Universal Tiled64 bfloat16 K-quant prefill GEMM on the Metal 4 tensor op
// (NAX): BM = 128, BN = 64, BK = 64, WM = 4, WN = 1, M-major grid,
// double-buffered FP16 B-stage (1 threadgroup barrier per BK = 64 step) with
// register prefetch overlapped with matmul2d, and coalesced TG-staged stores.
//
// Standalone pure-Metal header (zero MLX header dependencies). Included from
// kquant_prefill_nax.metal (compiled with `-std=metal4.0 -O3` for plain stores
// and `-std=metal4.0 -O3 -fno-fast-math -DKQUANT_PREFILL_EPILOGUES=1` for
// residual / SwiGLU epilogues). Serves all 17 Tiled64 formats (15 K-quant / IQ
// + a4g64 + a8g64) and the 7 epilogue modes at M >= 65.
#pragma once
#pragma clang fp reassociate(off)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshift-count-negative"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;

#include "kquant_m8_nax.h"

namespace kq_prefill {

using namespace metal;
using namespace mpp::tensor_ops;
using namespace kq_m8;

constant constexpr int KQ_EP_NONE = 0;
constant constexpr int KQ_EP_RESIDUAL = 1;
constant constexpr int KQ_EP_SWIGLU = 2;

constant constexpr ushort kPrefillBM = 128;
constant constexpr ushort kPrefillBN = 64;
constant constexpr ushort kPrefillBK = 64;
constant constexpr ushort kPrefillSGRows = 32;
constant constexpr ushort kPrefillThreads = 128;
constant constexpr uint kPrefillStage = kPrefillBN * kPrefillBK; // 4096 halfs

// Pure-Metal bfloat16 store epilogue ops matching MLX's Add / Multiply /
// Sigmoid (`unary_ops.h` + `bf16_math.h`) bit-for-bit under `-fno-fast-math`.
METAL_FUNC bfloat kq_bf16_sigmoid(bfloat x) {
  const bfloat ex = bfloat(metal::precise::exp(metal::abs(float(x))));
  const bfloat y = bfloat(1.0f) / (bfloat(1.0f) + ex);
  return (x < bfloat(0.0f)) ? y : (bfloat(1.0f) - y);
}

METAL_FUNC bfloat kq_bf16_residual(bfloat a, bfloat v) {
  return a + v;
}

METAL_FUNC bfloat kq_bf16_swiglu(bfloat a, bfloat v) {
#pragma clang fp reassociate(off)
  return (kq_bf16_sigmoid(a) * a) * v;
}

// Decode a packed pair word (code0 at bits 0..15, code1 at bits 16..31) via
// IEEE-754 FP16 mantissa bit-OR: 0x6400 is half(1024.0), whose ULP is 1.0,
// so OR-ing any unsigned code in 0..255 into bits 0..7 yields 1024 + code
// exactly. Subtracting (1024 + Zero) yields (code - Zero) exactly in half.
template <int Zero>
METAL_FUNC half2 staged_linear_pair(uint pair, float s, float m) {
  const float2 code = float2(
      as_type<half2>(pair | 0x64006400u) - half2(half(1024 + Zero)));
  if constexpr (Zero != 0) {
    return half2(code * s);
  } else {
    return half2(fma(code, float2(s), float2(m)));
  }
}

// Per-format minimal-register companion state for Tiled64 prefill.
// Pre-offsets the tile's scale/bias pointers once before the K loop so the
// hot loop carries zero unused fields (no sc2 except Q2K, no live Coef across
// matmul2d, and no per-step tile divisions).
template <Format F, int scale_shift, int group_size>
struct PrefillMetaSelect;

// Q4K / Q5K: 16 scale bytes + half2 (d, dmin) per 256-value super-block.
template <int scale_shift, int group_size>
struct PrefillMetaQ45 {
  const device uint8_t* ts;
  const device half* tb;
  uint4 sc;
  half2 d;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K) {
    const ulong base = ulong(tile) * (K / 256u) * 64u + col;
    ts = scales + base * 16u;
    tb = biases + base * 2u;
    load_sb(0u);
  }
  METAL_FUNC void load_sb(uint sb) {
    sc = *reinterpret_cast<const device uint4*>(ts + ulong(sb) * 64u * 16u);
    d = *reinterpret_cast<const device half2*>(tb + ulong(sb) * 64u * 2u);
  }
  METAL_FUNC Coef coef(uint u) const {
    const uint j = u & 7u;
    const uint sm = (sc[j >> 1] >> (16u * (j & 1u))) & 0xFFFFu;
    Coef c;
    c.s = float2(kq_shift_scale<scale_shift>(float(d.x) * float(sm & 0xFFu)));
    c.m = float2(-(float(d.y) * float(sm >> 8)));
    return c;
  }
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<Q4K, scale_shift, group_size> {
  typedef PrefillMetaQ45<scale_shift, group_size> type;
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<Q5K, scale_shift, group_size> {
  typedef PrefillMetaQ45<scale_shift, group_size> type;
};

// Q2K: 32 scale bytes (16 (sc, m) pairs) + half2 (d, dmin) per super-block.
template <int scale_shift, int group_size>
struct PrefillMetaQ2K {
  const device uint8_t* ts;
  const device half* tb;
  uint4 sc0;
  uint4 sc1;
  half2 d;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K) {
    const ulong base = ulong(tile) * (K / 256u) * 64u + col;
    ts = scales + base * 32u;
    tb = biases + base * 2u;
    load_sb(0u);
  }
  METAL_FUNC void load_sb(uint sb) {
    const device uint4* p =
        reinterpret_cast<const device uint4*>(ts + ulong(sb) * 64u * 32u);
    sc0 = p[0];
    sc1 = p[1];
    d = *reinterpret_cast<const device half2*>(tb + ulong(sb) * 64u * 2u);
  }
  METAL_FUNC Coef coef(uint u) const {
    const uint j = u & 7u;
    const uint pairs = j < 4u ? sc0[j] : sc1[j - 4u];
    const float dd = float(d.x);
    const float mm = float(d.y);
    Coef c;
    c.s = float2(
        kq_shift_scale<scale_shift>(dd * float(pairs & 0xFFu)),
        kq_shift_scale<scale_shift>(dd * float((pairs >> 16) & 0xFFu)));
    c.m = float2(
        -(mm * float((pairs >> 8) & 0xFFu)), -(mm * float(pairs >> 24)));
    return c;
  }
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<Q2K, scale_shift, group_size> {
  typedef PrefillMetaQ2K<scale_shift, group_size> type;
};

// Q6K / Q3K: 16 int8 scales + half d per super-block (Zero folded in stage32).
template <int scale_shift, int group_size>
struct PrefillMetaQ36 {
  const device uint8_t* ts;
  const device half* tb;
  uint4 sc;
  half d;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K) {
    const ulong base = ulong(tile) * (K / 256u) * 64u + col;
    ts = scales + base * 16u;
    tb = biases + base;
    load_sb(0u);
  }
  METAL_FUNC void load_sb(uint sb) {
    sc = *reinterpret_cast<const device uint4*>(ts + ulong(sb) * 64u * 16u);
    d = tb[ulong(sb) * 64u];
  }
  METAL_FUNC Coef coef(uint u) const {
    const uint j = u & 7u;
    const uint pair = (sc[j >> 1] >> (16u * (j & 1u))) & 0xFFFFu;
    const float dd = float(d);
    Coef c;
    c.s = float2(
        kq_shift_scale<scale_shift>(dd * float(as_type<char>(uchar(pair & 0xFFu)))),
        kq_shift_scale<scale_shift>(dd * float(as_type<char>(uchar(pair >> 8)))));
    c.m = float2(0.0f);
    return c;
  }
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<Q6K, scale_shift, group_size> {
  typedef PrefillMetaQ36<scale_shift, group_size> type;
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<Q3K, scale_shift, group_size> {
  typedef PrefillMetaQ36<scale_shift, group_size> type;
};

// IQ4XS / IQ3S8: 8 int8 scales + half d per super-block.
template <int scale_shift, int group_size>
struct PrefillMetaInt8Sb {
  const device uint8_t* ts;
  const device half* tb;
  uint2 sc;
  half d;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K) {
    const ulong base = ulong(tile) * (K / 256u) * 64u + col;
    ts = scales + base * 8u;
    tb = biases + base;
    load_sb(0u);
  }
  METAL_FUNC void load_sb(uint sb) {
    sc = *reinterpret_cast<const device uint2*>(ts + ulong(sb) * 64u * 8u);
    d = tb[ulong(sb) * 64u];
  }
  METAL_FUNC Coef coef(uint u) const {
    const uint j = u & 7u;
    const uint byte = (sc[j >> 2] >> (8u * (j & 3u))) & 0xFFu;
    Coef c;
    c.s = float2(kq_shift_scale<scale_shift>(
        float(d) * float(as_type<char>(uchar(byte)))));
    c.m = float2(0.0f);
    return c;
  }
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<IQ4XS, scale_shift, group_size> {
  typedef PrefillMetaInt8Sb<scale_shift, group_size> type;
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<IQ3S8, scale_shift, group_size> {
  typedef PrefillMetaInt8Sb<scale_shift, group_size> type;
};

// IQ4NL: 1 int8 scale + 1 half d per 32-value unit (no super-block sharing).
template <int scale_shift, int group_size>
struct PrefillMetaIQ4NL {
  const device uint8_t* ts;
  const device half* tb;
  uint8_t sc_byte;
  half d;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K,
      uint gi) {
    const ulong base = ulong(tile) * (K / 32u) * 64u + col;
    ts = scales + base;
    tb = biases + base;
    load_unit(gi);
  }
  METAL_FUNC void load_unit(uint u) {
    sc_byte = ts[ulong(u) * 64u];
    d = tb[ulong(u) * 64u];
  }
  METAL_FUNC Coef coef(uint) const {
    Coef c;
    c.s = float2(kq_shift_scale<scale_shift>(
        float(d) * float(as_type<char>(sc_byte))));
    c.m = float2(0.0f);
    return c;
  }
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<IQ4NL, scale_shift, group_size> {
  typedef PrefillMetaIQ4NL<scale_shift, group_size> type;
};

// A4 / A8 (MLX affine, group_size = 64 -> sr = 4 bf16 scales + 4 bf16 biases).
template <int scale_shift, int group_size>
struct PrefillMetaAffine {
  static constexpr constant uint sr = 256u / uint(group_size);
  static_assert(sr == 4u, "Tiled64 affine prefill expects group_size == 64");
  const device uint8_t* ts;
  const device half* tb;
  uint2 sc;
  uint2 bi;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K) {
    const ulong base = ulong(tile) * (K / 256u) * 64u + col;
    ts = scales + base * (2u * sr);
    tb = biases + base * sr;
    load_sb(0u);
  }
  METAL_FUNC void load_sb(uint sb) {
    sc = *reinterpret_cast<const device uint2*>(
        ts + ulong(sb) * 64u * (2u * sr));
    bi = *reinterpret_cast<const device uint2*>(tb + ulong(sb) * 64u * sr);
  }
  METAL_FUNC Coef coef(uint u) const {
    const uint e = ((u * 32u) / uint(group_size)) % sr;
    Coef c;
    c.s = float2(kq_bf16_to_float(
        uint16_t((sc[e >> 1] >> (16u * (e & 1u))) & 0xFFFFu)));
    c.m = float2(kq_bf16_to_float(
        uint16_t((bi[e >> 1] >> (16u * (e & 1u))) & 0xFFFFu)));
    return c;
  }
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<A4, scale_shift, group_size> {
  typedef PrefillMetaAffine<scale_shift, group_size> type;
};
template <int scale_shift, int group_size>
struct PrefillMetaSelect<A8, scale_shift, group_size> {
  typedef PrefillMetaAffine<scale_shift, group_size> type;
};

// 7 grid formats: half d per super-block + per-unit companion bytes into cur.sc.
template <Format F, int scale_shift, int group_size>
struct PrefillMetaGrid {
  static constexpr constant uint per = sb_scale_bytes<F>();
  static constexpr constant uint per_u = per / 8u;
  const device uint8_t* ts;
  const device half* tb;
  half d;

  METAL_FUNC void init(
      const device uint8_t* scales,
      const device half* biases,
      uint tile,
      uint col,
      uint K) {
    const ulong base = ulong(tile) * (K / 256u) * 64u + col;
    ts = scales + base * per;
    tb = biases + base;
    load_sb(0u);
  }
  METAL_FUNC void load_sb(uint sb) {
    d = tb[ulong(sb) * 64u];
  }
  METAL_FUNC uint companion(uint u) const {
    return kq_grid_companion<grid_kind<F>()>(
        ts + ulong(u >> 3u) * 64u * per + ulong(u & 7u) * per_u);
  }
  METAL_FUNC Coef coef(uint) const {
    Coef c;
    c.s = float2(float(d));
    c.m = float2(0.0f);
    return c;
  }
};

#define KQ_PREFILL_GRID_META(F)                                         \
  template <int scale_shift, int group_size>                            \
  struct PrefillMetaSelect<F, scale_shift, group_size> {                \
    typedef PrefillMetaGrid<F, scale_shift, group_size> type;           \
  };
KQ_PREFILL_GRID_META(IQ2XXS)
KQ_PREFILL_GRID_META(IQ2XS)
KQ_PREFILL_GRID_META(IQ2S)
KQ_PREFILL_GRID_META(IQ3XXS)
KQ_PREFILL_GRID_META(IQ1S)
KQ_PREFILL_GRID_META(IQ1M)
KQ_PREFILL_GRID_META(IQ3S)
#undef KQ_PREFILL_GRID_META

// One thread's 32-value weight unit dequantized into `dst` (32 halfs), each
// value rounded once from fp32.
template <Format F, int scale_shift>
METAL_FUNC void stage32_prefill(
    typename Codes<F>::W w,
    Coef c,
    threadgroup const half2* tl,
    threadgroup half* dst,
    KQGridTgTables grid) {
  if constexpr (is_grid<F>()) {
    float v[32];
    kq_grid_decode32<grid_kind<F>(), scale_shift>(w, c.s.x, v, grid);
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) =
          half4(v[8 * j], v[8 * j + 1], v[8 * j + 2], v[8 * j + 3]);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) =
          half4(v[8 * j + 4], v[8 * j + 5], v[8 * j + 6], v[8 * j + 7]);
    }
  } else if constexpr (F == IQ4XS || F == IQ4NL) {
    const float s = c.s.x;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const uchar4 b = as_type<uchar4>(w[j]);
      const half2 v0 = half2(float2(tl[b.x]) * s);
      const half2 v1 = half2(float2(tl[b.y]) * s);
      const half2 v2 = half2(float2(tl[b.z]) * s);
      const half2 v3 = half2(float2(tl[b.w]) * s);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) = half4(v0, v1);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) = half4(v2, v3);
    }
  } else if constexpr (F == Q4K || F == A4) {
    const float s = c.s.x;
    const float m = c.m.x;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const uint v = w[j];
      const half2 p04 = staged_linear_pair<0>(v & 0x000F000Fu, s, m);
      const half2 p15 = staged_linear_pair<0>((v >> 4) & 0x000F000Fu, s, m);
      const half2 p26 = staged_linear_pair<0>((v >> 8) & 0x000F000Fu, s, m);
      const half2 p37 = staged_linear_pair<0>((v >> 12) & 0x000F000Fu, s, m);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) =
          half4(p04.x, p15.x, p26.x, p37.x);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) =
          half4(p04.y, p15.y, p26.y, p37.y);
    }
  } else if constexpr (F == A8 || F == IQ3S8) {
    constexpr int kZero = F == IQ3S8 ? 128 : 0;
    const float s = c.s.x;
    const float m = c.m.x;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const uint4 h = j < 2 ? w.a : w.b;
      const uint w0 = h[2 * (j & 1)];
      const uint w1 = h[2 * (j & 1) + 1];
      const half2 p02 = staged_linear_pair<kZero>(w0 & 0x00FF00FFu, s, m);
      const half2 p13 =
          staged_linear_pair<kZero>((w0 >> 8) & 0x00FF00FFu, s, m);
      const half2 p46 = staged_linear_pair<kZero>(w1 & 0x00FF00FFu, s, m);
      const half2 p57 =
          staged_linear_pair<kZero>((w1 >> 8) & 0x00FF00FFu, s, m);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) =
          half4(p02.x, p13.x, p02.y, p13.y);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) =
          half4(p46.x, p57.x, p46.y, p57.y);
    }
  } else if constexpr (F == Q2K) {
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const float s = j < 2 ? c.s.x : c.s.y;
      const float m = j < 2 ? c.m.x : c.m.y;
      const uint v = (w[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
      const half2 p04 =
          staged_linear_pair<0>((v & 3u) | ((v << 8) & 0x00030000u), s, m);
      const half2 p15 = staged_linear_pair<0>(
          ((v >> 2) & 3u) | ((v << 6) & 0x00030000u), s, m);
      const half2 p26 = staged_linear_pair<0>(
          ((v >> 4) & 3u) | ((v << 4) & 0x00030000u), s, m);
      const half2 p37 = staged_linear_pair<0>(
          ((v >> 6) & 3u) | ((v << 2) & 0x00030000u), s, m);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) =
          half4(p04.x, p15.x, p26.x, p37.x);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) =
          half4(p04.y, p15.y, p26.y, p37.y);
    }
  } else if constexpr (F == Q3K) {
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const float s = j < 2 ? c.s.x : c.s.y;
      uint v;
      if (j == 0) {
        v = w.a.x;
      } else if (j == 1) {
        v = (w.a.x >> 24) | (w.a.y << 8);
      } else if (j == 2) {
        v = (w.a.y >> 16) | (w.b << 16);
      } else {
        v = w.b >> 8;
      }
      const half2 p04 =
          staged_linear_pair<4>((v & 7u) | ((v << 4) & 0x00070000u), s, 0.0f);
      const half2 p15 = staged_linear_pair<4>(
          ((v >> 3) & 7u) | ((v << 1) & 0x00070000u), s, 0.0f);
      const half2 p26 = staged_linear_pair<4>(
          ((v >> 6) & 7u) | ((v >> 2) & 0x00070000u), s, 0.0f);
      const half2 p37 = staged_linear_pair<4>(
          ((v >> 9) & 7u) | ((v >> 5) & 0x00070000u), s, 0.0f);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) =
          half4(p04.x, p15.x, p26.x, p37.x);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) =
          half4(p04.y, p15.y, p26.y, p37.y);
    }
  } else {
    // Q5K (Zero = 0, 32-group) and Q6K (Zero = 32, 16-group).
    constexpr int kZero = F == Q6K ? 32 : 0;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const uint2 q = bytes8<F>(w, j);
      const float s = j < 2 ? c.s.x : c.s.y;
      const float m = j < 2 ? c.m.x : c.m.y;
      const half2 p04 = staged_linear_pair<kZero>(q.x & 0x00FF00FFu, s, m);
      const half2 p15 = staged_linear_pair<kZero>(q.y & 0x00FF00FFu, s, m);
      const half2 p26 =
          staged_linear_pair<kZero>((q.x >> 8) & 0x00FF00FFu, s, m);
      const half2 p37 =
          staged_linear_pair<kZero>((q.y >> 8) & 0x00FF00FFu, s, m);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j) =
          half4(p04.x, p15.x, p26.x, p37.x);
      *reinterpret_cast<threadgroup half4*>(dst + 8 * j + 4) =
          half4(p04.y, p15.y, p26.y, p37.y);
    }
  }
}

template <
    int group_size,
    int bits,
    int super_ratio,
    bool has_min,
    int kind,
    int scale_shift,
    int epilogue>
METAL_FUNC void kquant_prefill128_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device half* biases,
    const device bfloat* x,
    device bfloat* y,
    const device bfloat* aux,
    int K,
    int N,
    int M,
    threadgroup half* stage,
    threadgroup half2* tl,
    threadgroup uint64_t* grid64,
    threadgroup uint32_t* grid32,
    uint2 group,
    uint simd_lane,
    uint simd_gid) {
  constexpr Format F = format<group_size, bits, super_ratio, has_min, kind>();
  static_assert(F != Unsupported, "unsupported K-quant prefill format");
  constexpr bool codebook = F == IQ4XS || F == IQ4NL;
  typedef typename Codes<F>::W W;

  const uint lid = simd_gid * 32u + simd_lane;
  if constexpr (codebook) {
    for (uint i = lid; i < kTableEntries; i += kPrefillThreads) {
      tl[i] = half2(float(kIQ4[i & 15]), float(kIQ4[i >> 4]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if constexpr (is_grid<F>()) {
    kq_grid_stage_table<grid_kind<F>()>(grid64, grid32, lid, kPrefillThreads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const KQGridTgTables grid{grid64, grid32};

  // M-major grid: group.x = M-tile (128 rows), group.y = N-tile (64 cols).
  // Caller guarantees M >= 65 (>= kPrefillSGRows = 32), so any partial tail
  // simdgroup shifts its 32-row read window back to [M - 32, M) for zero OOB
  // reads past `x`, and recomputes its store-row filter after the K loop.
  const uint sg_row0 = group.x * kPrefillBM + simd_gid * kPrefillSGRows;
  const bool owns_rows = sg_row0 < uint(M);
  device bfloat* x_sg = const_cast<device bfloat*>(x) +
      ulong(owns_rows ? min(sg_row0, uint(M) - uint(kPrefillSGRows)) : 0u) *
          uint(K);

  constexpr auto desc = matmul2d_descriptor(
      kPrefillSGRows,
      kPrefillBN,
      kPrefillBK,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;

  auto tA = tensor(
      x_sg, dextents<int, 2>{K, int(kPrefillSGRows)}, array<int, 2>{1, K});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
      stage,
      dextents<int, 2>{int(kPrefillBK), int(kPrefillBN)},
      array<int, 2>{1, int(kPrefillBK)});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
      stage + kPrefillStage,
      dextents<int, 2>{int(kPrefillBK), int(kPrefillBN)},
      array<int, 2>{1, int(kPrefillBK)});
  auto a_dummy = tA.template slice<kPrefillBK, kPrefillSGRows>(0, 0);
  auto b0 = bt0.template slice<kPrefillBK, kPrefillBN>(0, 0);
  auto b1 = bt1.template slice<kPrefillBK, kPrefillBN>(0, 0);

  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a_dummy),
      decltype(b0),
      float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }

  // Cooperative B-stage mapping: 128 threads = 64 cols x 2 units of 32 K.
  const uint col = lid & 63u;
  const uint gi = lid >> 6u;
  const uint num_steps = uint(K) / kPrefillBK;

  const device uint32_t* row =
      w + (ulong(group.y) * (uint(K) / 32u) * 64u + col) * uint(bits);
  constexpr uint stride = unit_stride<bits, true>();
  typename PrefillMetaSelect<F, scale_shift, group_size>::type meta;

  W cur;
  if (num_steps > 0u) {
    cur = Codes<F>::load(row, gi, stride);
    if constexpr (F == IQ4NL) {
      meta.init(scales, biases, group.y, col, uint(K), gi);
    } else {
      meta.init(scales, biases, group.y, col, uint(K));
      if constexpr (is_grid<F>()) {
        cur.sc = meta.companion(gi);
      }
    }
  }

  for (uint step = 0u; step < num_steps; ++step) {
    const uint buf = step & 1u;
    const Coef cc = meta.coef(step * 2u + gi);
    threadgroup half* dst =
        stage + buf * kPrefillStage + col * kPrefillBK + gi * 32u;
    stage32_prefill<F, scale_shift>(cur, cc, tl, dst, grid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (step + 1u < num_steps) {
      const uint u_next = (step + 1u) * 2u + gi;
      cur = Codes<F>::load(row, u_next, stride);
      if constexpr (F == IQ4NL) {
        meta.load_unit(u_next);
      } else {
        if (((step + 1u) & 3u) == 0u) {
          meta.load_sb((step + 1u) >> 2u);
        }
        if constexpr (is_grid<F>()) {
          cur.sc = meta.companion(u_next);
        }
      }
    }

    if (owns_rows) {
      auto a_slice =
          tA.template slice<kPrefillBK, kPrefillSGRows>(step * kPrefillBK, 0);
      if (buf) {
        op.run(a_slice, b1, acc);
      } else {
        op.run(a_slice, b0, acc);
      }
    }
  }

  // Stage rounded bfloat16 outputs through the 16 KiB `stage` buffer (4 SGs x
  // 32 rows x 64 cols x 2 B = 16,384 B) so the `acc` loop stays branchless
  // (no occupancy drop from per-element tail predicates) and device stores are
  // coalesced 16-byte / 8-byte transactions.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!owns_rows) {
    return;
  }

  threadgroup bfloat* sg_buf =
      reinterpret_cast<threadgroup bfloat*>(stage) +
      simd_gid * (kPrefillSGRows * kPrefillBN);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    const auto idx = acc.get_multidimensional_index(i);
    sg_buf[uint(idx[1]) * kPrefillBN + uint(idx[0])] = bfloat(float(acc[i]));
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const uint out_sg0 = group.x * kPrefillBM + simd_gid * kPrefillSGRows;
  const uint out_base = min(out_sg0, uint(M) - uint(kPrefillSGRows));
  const uint out_rstart = out_sg0 - out_base;
  const uint out_origin = group.y * kPrefillBN;
  const uint lane_r = simd_lane >> 3u;
  const uint lane_c = (simd_lane & 7u) * 8u;

  for (uint r0 = 0u; r0 < kPrefillSGRows; r0 += 4u) {
    const uint r = r0 + lane_r;
    if (r < out_rstart) {
      continue;
    }
    const ulong row_off = ulong(out_base + r) * uint(N) + out_origin + lane_c;
    if constexpr (epilogue == KQ_EP_NONE) {
      *reinterpret_cast<device uint4*>(y + row_off) =
          *reinterpret_cast<const threadgroup uint4*>(
              sg_buf + r * kPrefillBN + lane_c);
    } else {
      const threadgroup bfloat* src_row = sg_buf + r * kPrefillBN + lane_c;
      const device bfloat* aux_row = aux + row_off;
      bfloat out_vals[8];
#pragma unroll
      for (ushort e = 0; e < 8; ++e) {
        const bfloat v = src_row[e];
        const bfloat a = aux_row[e];
        if constexpr (epilogue == KQ_EP_RESIDUAL) {
          out_vals[e] = kq_bf16_residual(a, v);
        } else {
          static_assert(epilogue == KQ_EP_SWIGLU, "unknown store epilogue");
          out_vals[e] = kq_bf16_swiglu(a, v);
        }
      }
      *reinterpret_cast<device bfloat4*>(y + row_off) =
          bfloat4(out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
      *reinterpret_cast<device bfloat4*>(y + row_off + 4) =
          bfloat4(out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
    }
  }
}

} // namespace kq_prefill

#pragma clang diagnostic pop
