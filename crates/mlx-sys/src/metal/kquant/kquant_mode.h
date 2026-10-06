// Per-mode traits of the K-quant kernels, shared by kquant.h, kquant_nax.h
// and kquant_m8_nax.h.
//
// A kernel instantiation is identified by (group_size, bits, super_ratio,
// has_min, kind, scale_shift); the first four fix the three-array geometry
// (mlx_kquant.h), `kind` picks the value rule and `scale_shift` the
// power-of-two every decoded scale carries. The C++ side is kquant::kind /
// kquant::scale_shift / kquant::scale_bytes_per_group (mlx_kquant.h); the Rust
// side quant_dispatch::kquant_mode_params. All three must agree per mode.
#pragma once

// How a mode's codes turn into values (kquant::Kind in mlx_kquant.h).
//   KQ_LINEAR    scale * code + bias on the packed integer code (q2k..q6k)
//   KQ_CODEBOOK  scale * table[code], the 16-entry IQ4_NL grid (iq4nl, iq4xs)
//   KQ_INT8      scale * int8: the code is the signed value offset by 128
//                (iq3s), so the affine rule applies with bias = -128 * scale
//   KQ_GRID      scale * signed grid magnitude (IQ1/IQ2/IQ3_XXS; reserved for
//                the grid stage, no kernel decodes it yet)
constant constexpr int KQ_LINEAR = 0;
constant constexpr int KQ_CODEBOOK = 1;
constant constexpr int KQ_INT8 = 2;
constant constexpr int KQ_GRID = 3;

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

// Bytes of `.scales` per group: an (sc, m) pair for the has_min modes, one
// sub-scale otherwise. A grid mode that keeps extra per-group metadata in
// `.scales` widens this; the Tiled64 companion stride per super-block is
// super_ratio * kq_scale_bytes_per_group<..>().
template <bool has_min, int kind>
constexpr int kq_scale_bytes_per_group() {
  return has_min ? 2 : 1;
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
