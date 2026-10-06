#pragma once

// ggml K-quant and IQ formats, read-only. Each sub-block is affine
// (value = scale * code + bias), a 16-entry codebook, or (the grid formats
// IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S) a signed grid
// lookup, under a float16 super-block scale. `scales` holds the integer sub-block
// scales (int8, or interleaved uint8 (sc, m) pairs for q2k/q4k/q5k; the grid
// formats' per-unit companion bytes, uint8); `biases` holds the float16
// super-block d (and dmin for q2k/q4k/q5k) -- a scale, not a bias.
//
// Per-mode traits (super_ratio, has_sub_min, kind, scale_shift,
// scale_bytes_per_group) are the single C++ description of a mode; the Metal
// side mirrors them per kernel instantiation (metal/kquant/kquant_mode.h) and
// the Rust side in quant_dispatch::kquant_mode_params.

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <optional>
#include <string>
#include <string_view>

namespace mlx::core::kquant {

enum class Mode {
  Q6K,
  Q4K,
  Q5K,
  Q3K,
  IQ4NL,
  IQ4XS,
  // ggml IQ3_S in its packed grid form (mode string "iq3s", bits 3).
  IQ3S,
  Q2K,
  IQ2XXS,
  IQ2XS,
  IQ2S,
  IQ3XXS,
  IQ1S,
  IQ1M,
  // The legacy IQ3_S import: every value expanded to a signed 8-bit code
  // (mode string "iq3s8", bits 8). Artifacts converted before the packed
  // form carry `mode: "iq3s", bits: 8`; resolve_mode maps them here, so
  // they keep loading. Nothing produces this form any more.
  IQ3S8
};

// How a mode's codes turn into values. Mirrors KQ_LINEAR .. KQ_GRID_IQ1M in
// metal/kquant/kquant_mode.h (same values) and quant_dispatch::KQuantKind.
//   Linear    scale * code + bias on the packed integer code
//   Codebook  scale * table[code], the 16-entry IQ4_NL grid
//   Int8      scale * int8 (iq3s8: the grid value expanded to a byte)
//   Grid*     scale * signed grid magnitude, one kind per grid format since
//             they share no byte layout (metal/kquant/kquant_grid.h)
enum class Kind : int {
  Linear = 0,
  Codebook = 1,
  Int8 = 2,
  GridIQ2XXS = 3,
  GridIQ2XS = 4,
  GridIQ2S = 5,
  GridIQ3XXS = 6,
  GridIQ1S = 7,
  GridIQ1M = 8,
  GridIQ3S = 9
};

constexpr bool is_grid(Kind kind) {
  return kind >= Kind::GridIQ2XXS && kind <= Kind::GridIQ3S;
}

// In-memory layout of a 2-D K-quant weight and its companions.
//
//   RowMajor  .weight [N][K*bits/32], .scales [N][K/gs*pg], .biases
//             [N][K/(gs*sr)*pg]: the on-disk contract (gguf_kquant.rs).
//   Tiled64   the same bytes permuted so 64 consecutive rows interleave per
//             32-code unit (codes) and per 256-value super-block
//             (companions): .weight [N/64][K/32][64][bits] u32, .scales
//             [N/64][K/256][64][sr*pg], .biases [N/64][K/256][64][ps]
//             (pg = scale_bytes_per_group, ps = bias_entries_per_super_block;
//             IQ4_NL: sr = 1, so per 32-value block). A unit's bits are
//             unchanged; only unit addresses move, so a 64-column
//             threadgroup reads one contiguous 64*bits*4-byte run per step
//             instead of 64 rows K/2 bytes apart, and a row's super-block
//             scales stay one 16-byte load. The arrays keep
//             their 2-D shape; the layout rides on the mode string as the
//             "@t64" suffix (kTiledSuffix) so every consumer sees it. Only a
//             transposed matmul (x @ w.T) on a 2-D weight with N % 64 == 0
//             and K % 256 == 0 accepts it.
enum class Layout { RowMajor, Tiled64 };
constexpr int kTileRows = 64;
constexpr std::string_view kTiledSuffix = "@t64";

struct ModeLayout {
  Mode mode;
  Layout layout;
};

// The bare mode names only ("q4k"); a layout suffix is not a mode.
std::optional<Mode> parse_mode(std::string_view mode);
// The mode the caller's (mode, bits) pair names: `mode` itself, except that
// IQ3S with bits == 8 is the legacy expanded import IQ3S8 (the on-disk
// `mode: "iq3s", bits: 8` of artifacts converted before the packed form).
// Every entry point resolves through here before validating `bits`.
constexpr Mode resolve_mode(Mode mode, std::optional<int> bits) {
  return (mode == Mode::IQ3S && bits.has_value() && *bits == 8) ? Mode::IQ3S8
                                                                 : mode;
}
// "q4k" -> {Q4K, RowMajor}, "q4k@t64" -> {Q4K, Tiled64}.
std::optional<ModeLayout> parse_mode_layout(std::string_view mode);
const char *mode_name(Mode mode);
const char *layout_suffix(Layout layout);

// Groups one super-block spans; a super-block always covers
// group_size * super_ratio == 256 values (IQ4_NL: one 32-value block).
constexpr int super_ratio(Mode mode) {
  switch (mode) {
  case Mode::Q6K:
  case Mode::Q3K:
  case Mode::Q2K:
    return 16;
  case Mode::Q4K:
  case Mode::Q5K:
  case Mode::IQ4XS:
  case Mode::IQ3S:
  case Mode::IQ3S8:
  case Mode::IQ2XXS:
  case Mode::IQ2XS:
  case Mode::IQ2S:
  case Mode::IQ3XXS:
  case Mode::IQ1S:
  case Mode::IQ1M:
    return 8;
  case Mode::IQ4NL:
    return 1;
  }
  return 0;
}

// q2k/q4k/q5k interleave a minimum with every scale at both levels.
constexpr bool has_sub_min(Mode mode) {
  return mode == Mode::Q4K || mode == Mode::Q5K || mode == Mode::Q2K;
}

constexpr bool uses_iq4nl_grid(Mode mode) {
  return mode == Mode::IQ4NL || mode == Mode::IQ4XS;
}

constexpr Kind kind(Mode mode) {
  switch (mode) {
  case Mode::IQ4NL:
  case Mode::IQ4XS:
    return Kind::Codebook;
  case Mode::IQ3S8:
    return Kind::Int8;
  case Mode::Q6K:
  case Mode::Q4K:
  case Mode::Q5K:
  case Mode::Q3K:
  case Mode::Q2K:
    return Kind::Linear;
  case Mode::IQ2XXS:
    return Kind::GridIQ2XXS;
  case Mode::IQ2XS:
    return Kind::GridIQ2XS;
  case Mode::IQ2S:
    return Kind::GridIQ2S;
  case Mode::IQ3XXS:
    return Kind::GridIQ3XXS;
  case Mode::IQ1S:
    return Kind::GridIQ1S;
  case Mode::IQ1M:
    return Kind::GridIQ1M;
  case Mode::IQ3S:
    return Kind::GridIQ3S;
  }
  return Kind::Linear;
}

constexpr bool is_grid(Mode mode) { return is_grid(kind(mode)); }

// Power-of-two exponent every decoded scale is multiplied by, in fp32 and
// exactly (scale *= 2^scale_shift): ggml's `d * (0.5 + sc) * 0.25` (IQ2_*),
// `* 0.5` (IQ3_XXS) and `d * (2 sc + 1) * (g +- 1/8)` (IQ1_*) become
// `d * (1 + 2 sc) * 2^shift` times an integer, bit for bit. 0 for every
// affine / codebook / int8 mode and for IQ3_S, whose ggml scale is
// `d * (1 + 2 sc)` as written.
constexpr int scale_shift(Mode mode) {
  switch (mode) {
  case Mode::IQ2XXS:
  case Mode::IQ2XS:
  case Mode::IQ2S:
  case Mode::IQ1S:
  case Mode::IQ1M:
    return -3;
  case Mode::IQ3XXS:
    return -2;
  default:
    return 0;
  }
}

// Bytes of `.scales` per group: an (sc, m) pair, one sub-scale, or the grid
// format's companion bytes (kquant_grid.h: IQ2_XXS / IQ3_XXS the sign-scale
// word, IQ2_XS the scale nibbles, IQ2_S qh + scales, IQ1_S the qh halfword,
// IQ1_M qh + scales, IQ3_S qh + the scale nibble). The per-super-block
// companion stride is super_ratio * scale_bytes_per_group.
constexpr int scale_bytes_per_group(Mode mode) {
  switch (mode) {
  case Mode::IQ2XXS:
  case Mode::IQ3XXS:
    return 4;
  case Mode::IQ2XS:
    return 1;
  case Mode::IQ2S:
  case Mode::IQ1S:
  case Mode::IQ3S:
    return 2;
  case Mode::IQ1M:
    return 3;
  default:
    return has_sub_min(mode) ? 2 : 1;
  }
}

// Entries of `.biases` per super-block: (d, dmin) or d alone.
constexpr int bias_entries_per_super_block(Mode mode) {
  return has_sub_min(mode) ? 2 : 1;
}

// `.scales` dtype: the (sc, m) modes and the grid formats carry unsigned
// bytes, the symmetric K-quants and IQ4 / iq3s8 signed sub-scales.
inline Dtype scales_dtype(Mode mode) {
  return (has_sub_min(mode) || is_grid(mode)) ? uint8 : int8;
}

// For the grid formats `bits` is the unit's `.weight` word count (1 to 3),
// not a code width: every unit is 32 values whatever the index width.
constexpr int default_bits(Mode mode) {
  switch (mode) {
  case Mode::Q6K:
    return 6;
  case Mode::Q4K:
  case Mode::IQ4NL:
  case Mode::IQ4XS:
    return 4;
  case Mode::Q5K:
    return 5;
  case Mode::Q3K:
  case Mode::IQ3S:
    return 3;
  case Mode::Q2K:
  case Mode::IQ2XS:
  case Mode::IQ2S:
  case Mode::IQ3XXS:
    return 2;
  case Mode::IQ3S8:
    return 8;
  case Mode::IQ2XXS:
  case Mode::IQ1S:
  case Mode::IQ1M:
    return 1;
  }
  return 0;
}

constexpr int default_group_size(Mode mode) {
  return (mode == Mode::Q6K || mode == Mode::Q3K || mode == Mode::Q2K) ? 16
                                                                       : 32;
}

array quantized_matmul(const array &x, const array &w, const array &scales,
                       const std::optional<array> &biases, bool transpose,
                       std::optional<int> group_size, std::optional<int> bits,
                       Mode mode, StreamOrDevice s = {},
                       Layout layout = Layout::RowMajor);

array gather_qmm(const array &x, const array &w, const array &scales,
                 const std::optional<array> &biases,
                 std::optional<array> lhs_indices,
                 std::optional<array> rhs_indices, bool transpose,
                 std::optional<int> group_size, std::optional<int> bits,
                 Mode mode, bool sorted_indices, StreamOrDevice s = {});

array dequantize(const array &w, const array &scales,
                 const std::optional<array> &biases,
                 std::optional<int> group_size, std::optional<int> bits,
                 Mode mode, std::optional<Dtype> dtype, StreamOrDevice s = {});

// Every kernel the Metal dispatcher can request from paged_attn.metallib.
// `nax` kernels are requested only when `metal::is_nax_available()`. Defined in
// Metal builds only.
struct KernelName {
  std::string name;
  bool nax;
};
std::vector<KernelName> metal_kernel_names();

class KQuantMatmul : public UnaryPrimitive {
public:
  KQuantMatmul(Stream stream, int group_size, int bits, Mode mode,
               bool transpose, Layout layout = Layout::RowMajor)
      : UnaryPrimitive(stream), group_size_(group_size), bits_(bits),
        mode_(mode), transpose_(transpose), layout_(layout) {}

  void eval_cpu(const std::vector<array> &inputs, array &out) override;
  void eval_gpu(const std::vector<array> &inputs, array &out) override;

  DEFINE_NAME(KQuantMatmul)
  bool is_equivalent(const Primitive &other) const override;
  std::vector<Shape> output_shapes(const std::vector<array> &inputs) override;

private:
  int group_size_;
  int bits_;
  Mode mode_;
  bool transpose_;
  Layout layout_;
};

class KQuantGatherQMM : public UnaryPrimitive {
public:
  KQuantGatherQMM(Stream stream, int group_size, int bits, Mode mode,
                  bool transpose, bool left_sorted, bool right_sorted)
      : UnaryPrimitive(stream), group_size_(group_size), bits_(bits),
        mode_(mode), transpose_(transpose), left_sorted_(left_sorted),
        right_sorted_(right_sorted) {}

  void eval_cpu(const std::vector<array> &inputs, array &out) override;
  void eval_gpu(const std::vector<array> &inputs, array &out) override;

  DEFINE_NAME(KQuantGatherQMM)
  bool is_equivalent(const Primitive &other) const override;
  std::vector<Shape> output_shapes(const std::vector<array> &inputs) override;

private:
  int group_size_;
  int bits_;
  Mode mode_;
  bool transpose_;
  bool left_sorted_;
  bool right_sorted_;
};

class KQuantDequantize : public UnaryPrimitive {
public:
  KQuantDequantize(Stream stream, int group_size, int bits, Mode mode)
      : UnaryPrimitive(stream), group_size_(group_size), bits_(bits),
        mode_(mode) {}

  void eval_cpu(const std::vector<array> &inputs, array &out) override;
  void eval_gpu(const std::vector<array> &inputs, array &out) override;

  DEFINE_NAME(KQuantDequantize)
  bool is_equivalent(const Primitive &other) const override;
  std::vector<Shape> output_shapes(const std::vector<array> &inputs) override;

private:
  int group_size_;
  int bits_;
  Mode mode_;
};

} // namespace mlx::core::kquant
