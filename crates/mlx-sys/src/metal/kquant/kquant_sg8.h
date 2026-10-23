// Copyright © 2023-2024 Apple Inc.

#pragma once
// Native integer-code register MMA helpers, shared without arithmetic changes.
#include <metal_simdgroup>
#include <metal_stdlib>
#include "kquant_mode.h"
using namespace metal;
namespace kq_sg8 {

enum Format : int { Q4K, Q5K, IQ4XS, Q6K, A4G64, Unsupported };

template <int group_size, int bits, int super_ratio, bool has_min, int kind>
constexpr Format format() {
  if (group_size == 64 && super_ratio == 4 && bits == 4 && !has_min &&
      kind == KQ_AFFINE) {
    return A4G64;
  }
  if (group_size == 32 && super_ratio == 8 && bits == 4) {
    if (has_min && kind == KQ_LINEAR) {
      return Q4K;
    }
    if (!has_min && kind == KQ_CODEBOOK) {
      return IQ4XS;
    }
  }
  if (group_size == 32 && super_ratio == 8 && bits == 5 && has_min &&
      kind == KQ_LINEAR) {
    return Q5K;
  }
  if (group_size == 16 && super_ratio == 16 && bits == 6 && !has_min &&
      kind == KQ_LINEAR) {
    return Q6K;
  }
  return Unsupported;
}

template <typename T>
METAL_FUNC thread vec<T, 2>& te(thread simdgroup_matrix<T, 8, 8>& m) {
  return reinterpret_cast<thread vec<T, 2>&>(m.thread_elements());
}

// c += a x b on register operands.
METAL_FUNC void mma(thread float2& c, uint a, uint b) {
  simdgroup_matrix<bfloat, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = as_type<bfloat2>(a);
  te(B) = as_type<bfloat2>(b);
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

METAL_FUNC float bf_lo(uint v) {
  return as_type<float>(v << 16);
}

METAL_FUNC float bf_hi(uint v) {
  return as_type<float>(v & 0xFFFF0000u);
}

// Sum over the 8 lanes that share fn (fm lives in lane bits 1, 2 and 4).
METAL_FUNC float fm_reduce(float v) {
  v += simd_shuffle_xor(v, ushort(2));
  v += simd_shuffle_xor(v, ushort(4));
  v += simd_shuffle_xor(v, ushort(16));
  return v;
}

// B operand pairs (x[fn][k], x[fn + 1][k]) for k = 4 fm + j, j = 0..3.
METAL_FUNC void b_pairs4(uint2 a, uint2 b, thread uint (&bq)[4]) {
  bq[0] = (a.x & 0xFFFFu) | (b.x << 16);
  bq[1] = (a.x >> 16) | (b.x & 0xFFFF0000u);
  bq[2] = (a.y & 0xFFFFu) | (b.y << 16);
  bq[3] = (a.y >> 16) | (b.y & 0xFFFF0000u);
}

constant constexpr int kIQ4[16] =
    {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// A unit is 32 consecutive k of one weight row: one group, or q6k's two
// 16-groups. A super-block is 8 units.
template <Format F>
struct Codes;

template <>
struct Codes<Q4K> {
  typedef uint W;
  static W load(const device uint32_t* row, uint u, uint c, uint stride = 1) {
    return row[u * 4 * stride + c];
  }
};

template <>
struct Codes<A4G64> {
  typedef uint W;
  static W load(const device uint32_t* row, uint u, uint c, uint stride = 1) {
    return row[u * 4 * stride + c];
  }
};

template <>
struct Codes<IQ4XS> {
  typedef uint W;
  static W load(const device uint32_t* row, uint u, uint c, uint stride = 1) {
    return row[u * 4 * stride + c];
  }
};

template <>
struct Codes<Q5K> {
  typedef uint2 W;
  // Lane c's 40 bits start at bit 8c of word c of the 5-word group.
  static W load(const device uint32_t* row, uint u, uint c, uint stride = 1) {
    return uint2(row[u * 5 * stride + c], row[u * 5 * stride + c + 1]);
  }
};

template <>
struct Codes<Q6K> {
  typedef uint4 W;
  // The first group's 24-bit chunk starts at bit 24c of the 6-word span, the
  // second's at bit 96 + 24c; .xy and .zw hold the words covering each. For
  // c = 3 the second chunk ends at bit 32 of word 5, so its high word is
  // unused and word 5 is re-read instead of reading past the span.
  static W load(const device uint32_t* row, uint u, uint c, uint stride = 1) {
    const uint wa = (24 * c) >> 5;
    const uint wb = (96 + 24 * c) >> 5;
    const device uint32_t* s = row + u * 6 * stride;
    return uint4(s[wa], s[wa + 1], s[wb], s[wb + 1 < 6 ? wb + 1 : wb]);
  }
};

// One super-block's scale data: .a = sub-scale bytes, .h = fp16 super scales.
struct SB {
  uint4 a;
  half2 h;
};

template <Format F, bool tiled = false>
METAL_FUNC SB load_sb(
    const device uint8_t* scales,
    const device half* biases,
    uint n,
    uint K,
    uint G) {
  SB s;
  const size_t block = tiled
      ? (size_t(n / 64) * (K / 256) + G) * 64 + n % 64
      : size_t(n) * (K / 256) + G;
  if constexpr (F == A4G64) {
    const uint2 scale = *reinterpret_cast<const device uint2*>(scales + block * 8);
    const uint2 bias = *reinterpret_cast<const device uint2*>(biases + block * 4);
    s.a = uint4(scale, bias);
    s.h = half2(0.0h);
  } else if constexpr (F == IQ4XS) {
    const uint2 v = *reinterpret_cast<const device uint2*>(
        scales + block * 8);
    s.a = uint4(v, 0u, 0u);
    s.h = half2(biases[block], 0.0h);
  } else if constexpr (F == Q6K) {
    s.a = *reinterpret_cast<const device uint4*>(
        scales + block * 16);
    s.h = half2(biases[block], 0.0h);
  } else {
    s.a = *reinterpret_cast<const device uint4*>(
        scales + block * 16);
    s.h = *reinterpret_cast<const device half2*>(
        biases + block * 2);
  }
  return s;
}

} // namespace kq_sg8
