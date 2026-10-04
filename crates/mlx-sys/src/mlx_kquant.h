#pragma once

// ggml K-quant and IQ formats, read-only. Each sub-block is affine
// (value = scale * code + bias) or a 16-entry codebook, under a float16
// super-block scale. `scales` holds the integer sub-block scales (int8, or
// interleaved uint8 (sc, m) pairs for q4k/q5k); `biases` holds the float16
// super-block d (and dmin for q4k/q5k) -- a scale, not a bias.

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <optional>
#include <string>
#include <string_view>

namespace mlx::core::kquant {

enum class Mode { Q6K, Q4K, Q5K, Q3K, IQ4NL, IQ4XS, IQ3S };

std::optional<Mode> parse_mode(std::string_view mode);
const char *mode_name(Mode mode);

// Groups one super-block spans; a super-block always covers
// group_size * super_ratio == 256 values (IQ4_NL: one 32-value block).
constexpr int super_ratio(Mode mode) {
  switch (mode) {
  case Mode::Q6K:
  case Mode::Q3K:
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

// q4k/q5k interleave a minimum with every scale at both levels.
constexpr bool has_sub_min(Mode mode) {
  return mode == Mode::Q4K || mode == Mode::Q5K;
}

constexpr bool uses_iq4nl_grid(Mode mode) {
  return mode == Mode::IQ4NL || mode == Mode::IQ4XS;
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
  case Mode::IQ3S:
    return 8;
  }
  return 0;
}

constexpr int default_group_size(Mode mode) {
  return (mode == Mode::Q6K || mode == Mode::Q3K) ? 16 : 32;
}

array quantized_matmul(const array &x, const array &w, const array &scales,
                       const std::optional<array> &biases, bool transpose,
                       std::optional<int> group_size, std::optional<int> bits,
                       Mode mode, StreamOrDevice s = {});

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
               bool transpose)
      : UnaryPrimitive(stream), group_size_(group_size), bits_(bits),
        mode_(mode), transpose_(transpose) {}

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
