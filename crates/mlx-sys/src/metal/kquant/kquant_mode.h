// Per-mode traits of the K-quant kernels, shared by kquant.h, kquant_nax.h
// and kquant_m8_nax.h.
//
// A kernel instantiation is identified by (group_size, bits, super_ratio,
// has_min, kind, scale_shift); the first four fix the three-array geometry
// (mlx_kquant.h), `kind` picks the value rule and `scale_shift` the
// power-of-two every decoded scale carries. The C++ side is kquant::kind /
// kquant::scale_shift / kquant::scale_bytes_per_group (mlx_kquant.h); the Rust
// side quant_dispatch::kquant_mode_params. All three must agree per mode.
//
// Also included by the C++ CPU reference (through kquant_grid.h), so the
// Metal address-space keyword is behind a macro.
#pragma once

#ifdef __METAL_VERSION__
#define KQ_MODE_CONST constant constexpr int
#else
#define KQ_MODE_CONST inline constexpr int
#endif

// How a mode's codes turn into values (kquant::Kind in mlx_kquant.h).
//   KQ_LINEAR    scale * code + bias on the packed integer code (q2k..q6k)
//   KQ_CODEBOOK  scale * table[code], the 16-entry IQ4_NL grid (iq4nl, iq4xs)
//   KQ_INT8      scale * int8: the code is the signed value offset by 128
//                (iq3s), so the affine rule applies with bias = -128 * scale
//   KQ_GRID_*    scale * signed grid magnitude: `.weight` holds the ggml grid
//                indices (and IQ2_XS's sign indices / IQ2_S's sign bytes),
//                `.scales` the rest of the group's native bytes (kquant_grid.h
//                lists each layout), `.biases` the super-block d. One kind per
//                format, since the formats share no byte layout; kq_is_grid
//                is the family test.
KQ_MODE_CONST KQ_LINEAR = 0;
KQ_MODE_CONST KQ_CODEBOOK = 1;
KQ_MODE_CONST KQ_INT8 = 2;
KQ_MODE_CONST KQ_GRID_IQ2XXS = 3;
KQ_MODE_CONST KQ_GRID_IQ2XS = 4;
KQ_MODE_CONST KQ_GRID_IQ2S = 5;
KQ_MODE_CONST KQ_GRID_IQ3XXS = 6;
KQ_MODE_CONST KQ_GRID_IQ1S = 7;
KQ_MODE_CONST KQ_GRID_IQ1M = 8;

// Whether `kind` is one of the grid formats.
template <int kind>
constexpr bool kq_is_grid() {
  return kind >= KQ_GRID_IQ2XXS && kind <= KQ_GRID_IQ1M;
}

// Whether the dot / dequantize helpers apply the IQ4_NL codebook.
template <int kind>
constexpr bool kq_codebook() {
  return kind == KQ_CODEBOOK;
}

// Whether a group's decode has an affine zero point (-(2^(bits-1)) * scale)
// to fold in; codebook and grid values carry their own sign.
template <int kind>
constexpr bool kq_affine_zero_point() {
  return kind == KQ_LINEAR || kind == KQ_INT8;
}

// Bytes of `.scales` per 32-value group (16 for q2k/q3k/q6k): an (sc, m)
// pair for the has_min modes, one sub-scale otherwise, and for the grid
// formats the group's native companion bytes (kquant_grid.h):
//   IQ2_XXS 4  the sign / scale word        IQ3_XXS 4  the sign / scale word
//   IQ2_XS  1  the two scale nibbles        IQ1_S   2  the qh halfword
//   IQ2_S   2  the qh byte, the scale byte  IQ1_M   3  qh (2), the scale byte
// The Tiled64 companion stride per super-block is
// super_ratio * kq_scale_bytes_per_group<..>().
template <bool has_min, int kind>
constexpr int kq_scale_bytes_per_group() {
  return has_min                      ? 2
      : kind == KQ_GRID_IQ2XXS        ? 4
      : kind == KQ_GRID_IQ2XS         ? 1
      : kind == KQ_GRID_IQ2S          ? 2
      : kind == KQ_GRID_IQ3XXS        ? 4
      : kind == KQ_GRID_IQ1S          ? 2
      : kind == KQ_GRID_IQ1M          ? 3
                                      : 1;
}

// 2^shift as an fp32 constant; multiplying by it is exact away from the
// subnormal range, so a shifted scale equals ggml's `d * sc * 2^shift`.
template <int shift>
constexpr float kq_scale_factor() {
  static_assert(shift > -32 && shift < 32, "scale_shift is a small exponent");
  return shift >= 0 ? float(1u << shift) : 1.0f / float(1u << -shift);
}

// Apply `scale_shift` to a decoded scale. With shift 0 this is the identity
// and compiles away, so every current mode keeps its bits.
template <int scale_shift, typename U>
inline U kq_shift_scale(U scale) {
  if constexpr (scale_shift != 0) {
    return scale * static_cast<U>(kq_scale_factor<scale_shift>());
  } else {
    return scale;
  }
}
