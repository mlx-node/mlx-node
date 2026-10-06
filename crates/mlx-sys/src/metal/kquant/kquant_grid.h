// The grid K-quant formats (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS):
// one decode for the Metal kernels and the C++ CPU reference, so the two
// cannot drift. Compiles as Metal (constant tables, `thread` pointers) and
// as C++ (mlx_kquant.cpp); requires kquant_mode.h before it.
//
// Ported from Splash (incoai/splash, Apache-2.0; THIRD_PARTY_NOTICES):
// FmtIQ2XXS / FmtIQ2XS / FmtIQ2S / FmtIQ3XXS / FmtIQ1S / FmtIQ1M,
// quant_signs7 and quant_iq1_codes of
// runtime/metal/kernels/common/quant_formats.h and the QuantGrid arm of
// gguf_staged.h's dequant32, with the byte fetch rewritten for mlx-node's
// three-array layout. The values are llama.cpp's dequantize_row_* (MIT), bit
// for bit: the only reassociation is the exact power-of-two scale shift.
//
// Layout (gguf_kquant.rs is the on-disk contract). A unit is 32 consecutive
// values of one row; `.weight` holds `bits` uint32 words per unit (the
// LSB-first byte order of the ggml fields), `.scales` the unit's companion
// bytes (kq_scale_bytes_per_group), `.biases` one float16 d per 256-value
// super-block (IQ1_M: reassembled from the nibbles of its scale words at
// import). Per format, with ggml's field names:
//
//   IQ2_XXS  bits 1  w0 = aux32[0]: byte l = 8-bit grid index of entry l
//                    (elements 8l..8l+7)
//            sc 4 B  aux32[1]: 7-bit sign index of entry l at bits 7l,
//                    4-bit scale at bits 28..31
//            value   d * (1 + 2 sc) / 8 * +-grid[idx][j]
//   IQ2_XS   bits 2  halfword l of (w0, w1) = qs[l]: 9-bit grid index (bits
//                    0..8), 7-bit sign index (bits 9..15)
//            sc 1 B  scales[ib32]: low nibble the scale of elements 0..15,
//                    high nibble of 16..31
//            value   d * (1 + 2 sc) / 8 * +-grid[idx][j]
//   IQ2_S    bits 2  w0 = qs[4 ib32 ..]: byte l the low 8 index bits of
//                    entry l; w1 = signs[4 ib32 ..]: byte l its sign bits
//            sc 2 B  qh[ib32] (bits 2l, 2l + 1 the high index bits of entry
//                    l), scales[ib32] (two nibbles as IQ2_XS)
//            value   d * (1 + 2 sc) / 8 * +-grid[idx][j]
//   IQ3_XXS  bits 2  (w0, w1) = qs[8 ib32 ..]: byte t the 8-bit grid index of
//                    4-element entry t (elements 4t..4t+3)
//            sc 4 B  the scales_and_signs word: 7-bit sign index of elements
//                    8l..8l+7 at bits 7l, 4-bit scale at bits 28..31
//            value   d * (1 + 2 sc) / 4 * +-grid[idx][j]
//   IQ1_S    bits 1  w0 = qs[4 ib ..]: byte l the low 8 index bits of entry l
//            sc 2 B  qh[ib]: bits 3l..3l+2 the high index bits of entry l,
//                    bits 12..14 the 3-bit scale, bit 15 the delta sign
//            value   d * (2 sc + 1) * (grid[idx][j] +- 1/8), grid in -1..1
//   IQ1_M    bits 1  w0 = qs[4 ib ..] as IQ1_S
//            sc 3 B  qh[2 ib], qh[2 ib + 1]: nibble l holds the high index
//                    bits (0..2) and the delta sign (3) of entry l; then one
//                    byte: the 3-bit scale of elements 0..15 at bits 0..2
//                    and of 16..31 at bits 3..5
//            value   d * (2 sc + 1) * (grid[idx][j] +- 1/8)
//
// A chunk is 8 consecutive elements 8c..8c+7 of a unit (c = 0..3): one
// 8-element grid entry (IQ1 / IQ2) or two 4-element ones (IQ3_XXS), one sign
// byte and one scale, so kq_grid_decode8 is the natural quantum of every
// kernel and kq_grid_decode32 four of them.
#pragma once

#include "kquant_grid_tables.h"

#ifdef __METAL_VERSION__
#define KQ_THREAD thread
#define KQ_DEVICE device
#define KQ_POPCOUNT(x) popcount(x)
#else
#include <stdint.h>
#define KQ_THREAD
#define KQ_DEVICE
#define KQ_POPCOUNT(x) __builtin_popcount(x)
#endif

// uint32 words of `.weight` per unit (the mode's `bits`).
template <int kind>
constexpr int kq_grid_words() {
  return (kind == KQ_GRID_IQ2XXS || kind == KQ_GRID_IQ1S ||
          kind == KQ_GRID_IQ1M)
      ? 1
      : 2;
}

// The eight signs of a 7-bit sign index: its bits, and as bit 7 their
// parity, which keeps the count of negated elements even (llama.cpp's
// ksigns_iq2xs, Splash's quant_signs7).
inline uint32_t kq_signs7(uint32_t index) {
  return index | ((uint32_t(KQ_POPCOUNT(index)) & 1u) << 7);
}

// One unit's words and its companion bytes packed little-endian into `sc`
// (at most 4 bytes: IQ1_M's three, the sign / scale words of IQ2_XXS and
// IQ3_XXS).
struct KQGridUnit {
  uint32_t w0;
  uint32_t w1;
  uint32_t sc;
};

// A unit's companion run starts at a multiple of its byte count (unit g's
// run is at g * n in a row, and a row / super-block / expert starts at a
// multiple of 8 n), so the 2- and 4-byte runs load as one aligned halfword /
// word (one load instead of four dependent byte loads: the M = 1 kernels
// are latency-bound on this fetch); the 1- and 3-byte runs load as bytes.
template <int kind>
inline uint32_t kq_grid_companion(const KQ_DEVICE uint8_t* sc) {
  constexpr int n = kq_scale_bytes_per_group<false, kind>();
  if constexpr (n == 4) {
    return *reinterpret_cast<const KQ_DEVICE uint32_t*>(sc);
  } else if constexpr (n == 2) {
    return uint32_t(*reinterpret_cast<const KQ_DEVICE uint16_t*>(sc));
  } else {
    uint32_t v = uint32_t(sc[0]);
    if (n > 1) {
      v |= uint32_t(sc[1]) << 8;
    }
    if (n > 2) {
      v |= uint32_t(sc[2]) << 16;
    }
    return v;
  }
}

// The unit whose first word is at `w` and whose companion bytes start at
// `sc`.
template <int kind>
inline KQGridUnit kq_grid_load(
    const KQ_DEVICE uint32_t* w,
    const KQ_DEVICE uint8_t* sc) {
  KQGridUnit u;
  u.w0 = w[0];
  u.w1 = kq_grid_words<kind>() > 1 ? w[1] : 0u;
  u.sc = kq_grid_companion<kind>(sc);
  return u;
}

// Where a decode reads its grid: the constant tables above, or (Metal
// kernels that stage the table in threadgroup memory once per threadgroup,
// kquant_m8_nax.h) a threadgroup copy. `entries64` / `entries32` are the
// table lengths a kind needs.
template <int kind>
constexpr uint32_t kq_grid_entries() {
  return kind == KQ_GRID_IQ2XXS ? 256u
      : kind == KQ_GRID_IQ2XS  ? 512u
      : kind == KQ_GRID_IQ2S   ? 1024u
      : kind == KQ_GRID_IQ3XXS ? 256u
                               : 2048u; // IQ1_S / IQ1_M
}
template <int kind>
constexpr bool kq_grid_entries_are_u64() {
  return kind == KQ_GRID_IQ2XXS || kind == KQ_GRID_IQ2XS ||
      kind == KQ_GRID_IQ2S;
}

struct KQGridConstTables {
  template <int kind>
  inline uint64_t grid64(uint32_t idx) const {
    if constexpr (kind == KQ_GRID_IQ2XXS) {
      return kq_iq2xxs_grid[idx];
    } else if constexpr (kind == KQ_GRID_IQ2XS) {
      return kq_iq2xs_grid[idx];
    } else {
      return kq_iq2s_grid[idx];
    }
  }
  template <int kind>
  inline uint32_t grid32(uint32_t idx) const {
    if constexpr (kind == KQ_GRID_IQ3XXS) {
      return kq_iq3xxs_grid[idx];
    } else {
      return kq_iq1s_grid_gpu[idx];
    }
  }
};

#ifdef __METAL_VERSION__
// A kind's table copied to threadgroup memory (kq_grid_stage_table).
struct KQGridTgTables {
  threadgroup const uint64_t* t64;
  threadgroup const uint32_t* t32;
  template <int kind>
  inline uint64_t grid64(uint32_t idx) const {
    return t64[idx];
  }
  template <int kind>
  inline uint32_t grid32(uint32_t idx) const {
    return t32[idx];
  }
};

// Copies kind's table into `t64` / `t32` (whichever it uses; the other may be
// a 1-entry dummy), cooperatively over `threads` threads; the caller
// barriers.
template <int kind>
inline void kq_grid_stage_table(
    threadgroup uint64_t* t64,
    threadgroup uint32_t* t32,
    uint32_t thread_index,
    uint32_t threads) {
  constexpr uint32_t n = kq_grid_entries<kind>();
  KQGridConstTables c;
  for (uint32_t i = thread_index; i < n; i += threads) {
    if constexpr (kq_grid_entries_are_u64<kind>()) {
      t64[i] = c.grid64<kind>(i);
    } else {
      t32[i] = c.grid32<kind>(i);
    }
  }
}
#endif

// Elements 8c..8c+7 of a unit as fp32: s = d * (1 + 2 sc) * 2^shift (the
// (2 sc + 1) of IQ1 is the same number), then s * magnitude, then the sign,
// in llama.cpp's operation order. `tables` is where the grid is read from.
template <int kind, int scale_shift, typename Tables>
inline void kq_grid_decode8(
    KQGridUnit u,
    float d,
    uint32_t c,
    KQ_THREAD float* out,
    Tables tables) {
  if constexpr (kind == KQ_GRID_IQ3XXS) {
    const uint32_t word = c < 2 ? u.w0 : u.w1;
    const uint32_t hw = (word >> (16u * (c & 1u))) & 0xFFFFu;
    const uint32_t signs = kq_signs7((u.sc >> (7u * c)) & 127u);
    const float s = kq_shift_scale<scale_shift>(
        d * float(1u + 2u * (u.sc >> 28)));
    const uint32_t g1 = tables.template grid32<kind>(hw & 0xFFu);
    const uint32_t g2 = tables.template grid32<kind>(hw >> 8);
    for (uint32_t j = 0; j < 4; ++j) {
      const float v1 = s * float((g1 >> (8u * j)) & 0xFFu);
      const float v2 = s * float((g2 >> (8u * j)) & 0xFFu);
      out[j] = (signs >> j) & 1u ? -v1 : v1;
      out[4 + j] = (signs >> (4u + j)) & 1u ? -v2 : v2;
    }
  } else if constexpr (kind == KQ_GRID_IQ1S || kind == KQ_GRID_IQ1M) {
    uint32_t idx;
    uint32_t neg;
    uint32_t sc3;
    if constexpr (kind == KQ_GRID_IQ1S) {
      const uint32_t qh = u.sc & 0xFFFFu;
      idx = ((u.w0 >> (8u * c)) & 0xFFu) | (((qh >> (3u * c)) & 7u) << 8);
      neg = (qh >> 15) & 1u;
      sc3 = (qh >> 12) & 7u;
    } else {
      const uint32_t qh = u.sc & 0xFFFFu;
      idx = ((u.w0 >> (8u * c)) & 0xFFu) | (((qh >> (4u * c)) & 7u) << 8);
      neg = (qh >> (4u * c + 3u)) & 1u;
      sc3 = ((u.sc >> 16) >> (3u * (c >> 1))) & 7u;
    }
    // s = d (2 sc + 1) / 8 and the code 8 (g + 1) + (neg ? 0 : 2) - 9 =
    // 8 g -+ 1 give s * code == d (2 sc + 1) (g +- 1/8) bit for bit: the
    // scaling by 8 is exact (Splash's quant_iq1_codes, Zero 9).
    const float s = kq_shift_scale<scale_shift>(d * float(2u * sc3 + 1u));
    const uint32_t entry = tables.template grid32<kind>(idx);
    const int delta = neg ? -1 : 1;
    for (uint32_t j = 0; j < 8; ++j) {
      const uint32_t nib = (entry >> (8u * (j & 3u) + 4u * (j >> 2))) & 0xFu;
      out[j] = s * float(int(8u * nib) - 8 + delta);
    }
  } else {
    uint32_t idx;
    uint32_t signs;
    uint32_t nib;
    if constexpr (kind == KQ_GRID_IQ2XXS) {
      idx = (u.w0 >> (8u * c)) & 0xFFu;
      signs = kq_signs7((u.sc >> (7u * c)) & 127u);
      nib = u.sc >> 28;
    } else if constexpr (kind == KQ_GRID_IQ2XS) {
      const uint32_t word = c < 2 ? u.w0 : u.w1;
      const uint32_t e = (word >> (16u * (c & 1u))) & 0xFFFFu;
      idx = e & 511u;
      signs = kq_signs7(e >> 9);
      nib = (u.sc >> (4u * (c >> 1))) & 0xFu;
    } else {
      idx = ((u.w0 >> (8u * c)) & 0xFFu) | (((u.sc >> (2u * c)) & 3u) << 8);
      signs = (u.w1 >> (8u * c)) & 0xFFu;
      nib = ((u.sc >> 8) >> (4u * (c >> 1))) & 0xFu;
    }
    const float s = kq_shift_scale<scale_shift>(d * float(1u + 2u * nib));
    const uint64_t g = tables.template grid64<kind>(idx);
    for (uint32_t j = 0; j < 8; ++j) {
      const float v = s * float(uint32_t(g >> (8u * j)) & 0xFFu);
      out[j] = (signs >> j) & 1u ? -v : v;
    }
  }
}

// The same from the constant tables.
template <int kind, int scale_shift>
inline void kq_grid_decode8(
    KQGridUnit u,
    float d,
    uint32_t c,
    KQ_THREAD float* out) {
  kq_grid_decode8<kind, scale_shift>(u, d, c, out, KQGridConstTables{});
}

// All 32 values of a unit.
template <int kind, int scale_shift, typename Tables>
inline void kq_grid_decode32(
    KQGridUnit u,
    float d,
    KQ_THREAD float* out,
    Tables tables) {
  for (uint32_t c = 0; c < 4; ++c) {
    kq_grid_decode8<kind, scale_shift>(u, d, c, out + 8 * c, tables);
  }
}

template <int kind, int scale_shift>
inline void kq_grid_decode32(KQGridUnit u, float d, KQ_THREAD float* out) {
  kq_grid_decode32<kind, scale_shift>(u, d, out, KQGridConstTables{});
}

#undef KQ_POPCOUNT
#undef KQ_DEVICE
#undef KQ_THREAD
