#pragma once

// ggml K-quant and IQ formats, read-only. Each sub-block is affine
// (value = scale * code + bias) or a 16-entry codebook, under a float16
// super-block scale. `scales` holds the integer sub-block scales (int8, or
// interleaved uint8 (sc, m) pairs for q2k/q4k/q5k); `biases` holds the float16
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

enum class Mode { Q6K, Q4K, Q5K, Q3K, IQ4NL, IQ4XS, IQ3S, Q2K };

// How a mode's codes turn into values. Mirrors KQ_LINEAR .. KQ_GRID in
// metal/kquant/kquant_mode.h and quant_dispatch::KQuantKind.
//   Linear    scale * code + bias on the packed integer code
//   Codebook  scale * table[code], the 16-entry IQ4_NL grid
//   Int8      scale * int8 (iq3s: the grid value expanded to a byte)
//   Grid      scale * signed grid magnitude (IQ1/IQ2/IQ3_XXS; reserved)
enum class Kind : int { Linear = 0, Codebook = 1, Int8 = 2, Grid = 3 };

// In-memory layout of a 2-D K-quant weight and its companions.
//
//   RowMajor  .weight [N][K*bits/32], .scales [N][K/gs*pg], .biases
//             [N][K/(gs*sr)*pg]: the on-disk contract (gguf_kquant.rs).
//   Tiled64   the same bytes permuted so 64 consecutive rows interleave per
//             32-code unit (codes) and per 256-value super-block
//             (companions): .weight [N/64][K/32][64][bits] u32, .scales
//             [N/64][K/256][64][sr*pg], .biases [N/64][K/256][64][pg]
//             (IQ4_NL: sr = 1, so per 32-value block). A unit's bits are
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
  case Mode::IQ3S:
    return Kind::Int8;
  case Mode::Q6K:
  case Mode::Q4K:
  case Mode::Q5K:
  case Mode::Q3K:
  case Mode::Q2K:
    return Kind::Linear;
  }
  return Kind::Linear;
}

// Power-of-two exponent every decoded scale is multiplied by, in fp32 and
// exactly (scale *= 2^scale_shift). 0 for every current mode; the grid stage
// uses -3 for IQ2_* / IQ1_* and -2 for IQ3_XXS.
constexpr int scale_shift(Mode) { return 0; }

// Bytes of `.scales` per group: an (sc, m) pair or one sub-scale. The
// per-super-block companion stride is super_ratio * scale_bytes_per_group.
constexpr int scale_bytes_per_group(Mode mode) {
  return has_sub_min(mode) ? 2 : 1;
}

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
    return 3;
  case Mode::Q2K:
    return 2;
  case Mode::IQ3S:
    return 8;
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
