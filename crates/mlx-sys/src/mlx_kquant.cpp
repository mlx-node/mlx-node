#include "mlx_kquant.h"

#include "mlx/allocator.h"
#include "mlx/backend/common/quantized.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/transforms_impl.h"

#include <numeric>
#include <sstream>

namespace mlx::core::kquant {

std::optional<Mode> parse_mode(std::string_view mode) {
  if (mode == "q6k")
    return Mode::Q6K;
  if (mode == "q4k")
    return Mode::Q4K;
  if (mode == "q5k")
    return Mode::Q5K;
  if (mode == "q3k")
    return Mode::Q3K;
  if (mode == "iq4nl")
    return Mode::IQ4NL;
  if (mode == "iq4xs")
    return Mode::IQ4XS;
  if (mode == "iq3s")
    return Mode::IQ3S;
  return std::nullopt;
}

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Q6K:
    return "q6k";
  case Mode::Q4K:
    return "q4k";
  case Mode::Q5K:
    return "q5k";
  case Mode::Q3K:
    return "q3k";
  case Mode::IQ4NL:
    return "iq4nl";
  case Mode::IQ4XS:
    return "iq4xs";
  case Mode::IQ3S:
    return "iq3s";
  }
  throw std::invalid_argument("[kquant] Unknown quantization mode.");
}

namespace {

// Q6_K and IQ modes carry signed sub-block scales; q4k/q5k unsigned (sc, m).
void validate_mode_with_type(std::string_view tag, Mode mode,
                             const array &scales,
                             const std::optional<array> &biases,
                             std::optional<Dtype> out_type) {
  if (out_type.has_value() && !issubdtype(*out_type, floating)) {
    std::ostringstream msg;
    msg << "[" << tag << "] Only real floating types are supported but "
        << "output dtype == " << *out_type << ".";
    throw std::invalid_argument(msg.str());
  }
  auto scales_type = has_sub_min(mode) ? uint8 : int8;
  if (scales.dtype() != scales_type) {
    std::ostringstream msg;
    msg << "[" << tag << "] Scale type must be " << scales_type
        << " for quantization mode '" << mode_name(mode)
        << "' but received type " << scales.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!biases) {
    std::ostringstream msg;
    msg << "[" << tag << "] Biases must be provided for quantization mode '"
        << mode_name(mode) << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (biases->dtype() != float16) {
    std::ostringstream msg;
    msg << "[" << tag << "] Bias type must be " << float16
        << " for quantization mode '" << mode_name(mode)
        << "' but received type " << biases->dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
}

std::pair<int, int> params_from_mode(std::string_view tag, Mode mode,
                                     std::optional<int> group_size_,
                                     std::optional<int> bits_) {
  int default_gs = default_group_size(mode);
  int default_b = default_bits(mode);
  int group_size = group_size_.value_or(default_gs);
  int bits = bits_.value_or(default_b);
  if (bits <= 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] Invalid value for bits: " << bits;
    throw std::invalid_argument(msg.str());
  }
  if (group_size <= 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] Invalid value for group_size: " << group_size;
    throw std::invalid_argument(msg.str());
  }
  if (group_size != default_gs || bits != default_b) {
    std::ostringstream msg;
    msg << "[" << tag << "] " << mode_name(mode)
        << " quantization requires group size " << default_gs << " and "
        << default_b << " bits but got group size " << group_size << " and "
        << bits << " bits.";
    throw std::invalid_argument(msg.str());
  }
  return {group_size, bits};
}

void validate_quantized_input(std::string_view tag, const array &w,
                              const array &scales, const array &biases,
                              int group_size, int bits, Mode mode) {
  if (w.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[" << tag << "] The weight matrix should be uint32 "
        << "but received " << w.dtype();
    throw std::invalid_argument(msg.str());
  }
  int ratio = super_ratio(mode);
  int per_group = has_sub_min(mode) ? 2 : 1;

  auto check_batch_shape = [&](const array &a, const char *name) {
    if (a.ndim() != w.ndim() ||
        !std::equal(w.shape().begin(), w.shape().end() - 2,
                    a.shape().begin())) {
      std::ostringstream msg;
      msg << "[" << tag << "] Weight and " << name
          << " should have the same batch shape. "
          << "Received weight with shape " << w.shape() << ", " << name
          << " with " << a.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  };
  check_batch_shape(scales, "scales");

  int el_per_row = w.shape(-1) * 32 / bits;
  if (el_per_row * per_group != scales.shape(-1) * group_size) {
    std::ostringstream msg;
    msg << "[" << tag << "] The shapes of the weight and scales are "
        << "incompatible based on bits and group_size. w.shape() == "
        << w.shape() << " and scales.shape() == " << scales.shape()
        << " with group_size=" << group_size << " and bits=" << bits;
    throw std::invalid_argument(msg.str());
  }

  int super_size = group_size * ratio;
  if (el_per_row % super_size != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] " << mode_name(mode)
        << " quantization needs the last dimension to be divisible by the "
        << "super-block size " << super_size
        << " but w.shape() == " << w.shape() << " expands to " << el_per_row
        << " elements per row.";
    throw std::invalid_argument(msg.str());
  }
  check_batch_shape(biases, "biases");
  // The kernels index both companions with one flat group counter that runs
  // across rows, so a companion with fewer rows than the weight is read past
  // its end. check_batch_shape never compares the row axis.
  if (w.ndim() >= 2 &&
      (scales.shape(-2) != w.shape(-2) || biases.shape(-2) != w.shape(-2))) {
    std::ostringstream msg;
    msg << "[" << tag << "] " << mode_name(mode)
        << " quantization needs the weight, scales and biases to hold the "
        << "same number of rows. w.shape() == " << w.shape()
        << ", scales.shape() == " << scales.shape()
        << " and biases.shape() == " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (el_per_row * per_group != biases.shape(-1) * super_size) {
    std::ostringstream msg;
    msg << "[" << tag << "] The shapes of the weight and biases are "
        << "incompatible based on the " << super_size
        << " element super-block. w.shape() == " << w.shape()
        << " and biases.shape() == " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
}

std::pair<int, int> extract_matmul_dims(std::string_view tag, const array &x,
                                        const array &w, const array &scales,
                                        const array &biases, bool transpose,
                                        int group_size, int bits, Mode mode) {
  validate_quantized_input(tag, w, scales, biases, group_size, bits, mode);
  int x_inner_dims = x.shape(-1);
  int w_inner_dims = transpose ? w.shape(-1) * 32 / bits : w.shape(-2);
  int w_outer_dims = transpose ? w.shape(-2) : w.shape(-1) * 32 / bits;
  if (w_inner_dims != x_inner_dims) {
    std::ostringstream msg;
    msg << "[" << tag << "] Last dimension of first input with "
        << "shape (..., " << x_inner_dims << ") does not match "
        << "the expanded quantized matrix (" << w_inner_dims << ", "
        << w_outer_dims << ") computed from shape " << w.shape()
        << " with group_size=" << group_size << ", bits=" << bits
        << " and transpose=" << std::boolalpha << transpose;
    throw std::invalid_argument(msg.str());
  }
  return {w_inner_dims, w_outer_dims};
}

// MLX's broadcast_arrays(inputs, {-2, -1}, s), which ops.h does not export.
// Under dynamic tracing it must emit BroadcastAxes nodes even for inputs whose
// trace-time batch already matches, so a shapeless replay re-derives the batch
// shape instead of keeping the traced one.
std::vector<array> broadcast_batch(const std::vector<array> &inputs,
                                   StreamOrDevice s) {
  const std::vector<int> ignore_axes = {-2, -1};
  auto shape = BroadcastAxes::output_shape(inputs, ignore_axes);
  auto out_shape_of = [&shape](const array &in) {
    auto out_shape = shape;
    out_shape[out_shape.size() - 2] = in.shape(-2);
    out_shape[out_shape.size() - 1] = in.shape(-1);
    return out_shape;
  };
  std::vector<array> outputs;
  if (!detail::in_dynamic_tracing()) {
    for (auto &in : inputs) {
      auto out_shape = out_shape_of(in);
      if (in.shape() == out_shape) {
        outputs.push_back(in);
      } else {
        outputs.push_back(
            array(out_shape, in.dtype(),
                  std::make_shared<Broadcast>(to_stream(s), out_shape), {in}));
      }
    }
    return outputs;
  }
  std::vector<array> stop_grad_inputs;
  for (auto &in : inputs) {
    stop_grad_inputs.push_back(stop_gradient(in, s));
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::vector<array> p_inputs = {inputs[i]};
    for (size_t j = 0; j < inputs.size(); ++j) {
      if (j != i) {
        p_inputs.push_back(stop_grad_inputs[j]);
      }
    }
    outputs.push_back(
        array(out_shape_of(inputs[i]), inputs[i].dtype(),
              std::make_shared<BroadcastAxes>(to_stream(s), ignore_axes),
              std::move(p_inputs)));
  }
  return outputs;
}

array indices_or_default(std::optional<array> indices, const array &x,
                         StreamOrDevice s) {
  if (indices.has_value()) {
    return indices.value();
  }
  Shape shape(x.shape().begin(), x.shape().end() - 2);
  int64_t total = std::reduce(shape.begin(), shape.end(), int64_t{1},
                              std::multiplies<int64_t>{});
  if (total > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("[gather_qmm] Too many batch entries.");
  }
  return reshape(arange(static_cast<int>(total), uint32, s), std::move(shape),
                 s);
}

} // namespace

array quantized_matmul(const array &x, const array &w, const array &scales,
                       const std::optional<array> &biases, bool transpose,
                       std::optional<int> group_size_, std::optional<int> bits_,
                       Mode mode, StreamOrDevice s) {
  validate_mode_with_type("quantized_matmul", mode, scales, biases,
                          std::nullopt);
  auto [group_size, bits] =
      params_from_mode("quantized_matmul", mode, group_size_, bits_);
  auto [w_inner, w_outer] =
      extract_matmul_dims("quantized_matmul", x, w, scales, *biases, transpose,
                          group_size, bits, mode);
  (void)w_inner;
  auto dtype = x.dtype();
  if (!issubdtype(dtype, floating)) {
    std::ostringstream msg;
    msg << "[quantized_matmul] Only real floating types are supported but "
        << "x.dtype() == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  std::vector<array> inputs = {x, w, scales, *biases};
  if (x.ndim() > 2 && w.ndim() > 2) {
    inputs = broadcast_batch(inputs, s);
  }
  auto out_shape = inputs[0].shape();
  out_shape.back() = w_outer;
  return array(std::move(out_shape), dtype,
               std::make_shared<KQuantMatmul>(to_stream(s), group_size, bits,
                                              mode, transpose),
               std::move(inputs));
}

array gather_qmm(const array &x, const array &w, const array &scales,
                 const std::optional<array> &biases,
                 std::optional<array> lhs_indices_,
                 std::optional<array> rhs_indices_, bool transpose,
                 std::optional<int> group_size_, std::optional<int> bits_,
                 Mode mode, bool sorted_indices, StreamOrDevice s) {
  if (!lhs_indices_ && !rhs_indices_) {
    return quantized_matmul(x, w, scales, biases, transpose, group_size_, bits_,
                            mode, s);
  }
  validate_mode_with_type("gather_qmm", mode, scales, biases, std::nullopt);
  auto [group_size, bits] =
      params_from_mode("gather_qmm", mode, group_size_, bits_);
  auto [w_inner, w_outer] = extract_matmul_dims(
      "gather_qmm", x, w, scales, *biases, transpose, group_size, bits, mode);
  (void)w_inner;
  auto out_type = x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[gather_qmm] Only real floating types are supported but "
        << "x.dtype() == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  array lhs_indices = indices_or_default(lhs_indices_, x, s);
  array rhs_indices = indices_or_default(rhs_indices_, w, s);
  auto broadcast_indices = broadcast_arrays({lhs_indices, rhs_indices}, s);
  lhs_indices = broadcast_indices[0];
  rhs_indices = broadcast_indices[1];
  if (!issubdtype(lhs_indices.dtype(), integer)) {
    throw std::invalid_argument("[gather_qmm] Got lhs_indices with invalid "
                                "dtype. Indices must be integral.");
  }
  if (!issubdtype(rhs_indices.dtype(), integer)) {
    throw std::invalid_argument("[gather_qmm] Got rhs_indices with invalid "
                                "dtype. Indices must be integral.");
  }
  if (x.ndim() < 2) {
    std::ostringstream msg;
    msg << "[gather_qmm] Non-quantized input must have at least two"
        << " dimensions but got input with shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  lhs_indices = astype(lhs_indices, uint32, s);
  rhs_indices = astype(rhs_indices, uint32, s);

  auto out_shape = lhs_indices.shape();
  out_shape.push_back(x.shape(-2));
  out_shape.push_back(w_outer);
  return array(std::move(out_shape), out_type,
               std::make_shared<KQuantGatherQMM>(
                   to_stream(s), group_size, bits, mode, transpose,
                   sorted_indices && !rhs_indices_,
                   sorted_indices && !lhs_indices_),
               {astype(x, out_type, s), w, scales, *biases,
                std::move(lhs_indices), std::move(rhs_indices)});
}

array dequantize(const array &w, const array &scales,
                 const std::optional<array> &biases,
                 std::optional<int> group_size_, std::optional<int> bits_,
                 Mode mode, std::optional<Dtype> dtype, StreamOrDevice s) {
  validate_mode_with_type("dequantize", mode, scales, biases, dtype);
  auto out_type = dtype.value_or(bfloat16);
  auto [group_size, bits] =
      params_from_mode("dequantize", mode, group_size_, bits_);
  if (w.dtype() != uint32) {
    throw std::invalid_argument(
        "[dequantize] The matrix should be given as a uint32");
  }
  if (w.ndim() < 2) {
    std::ostringstream msg;
    msg << "[dequantize] The matrix to be dequantized must have at least 2 "
           "dimension but it has only "
        << w.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }
  validate_quantized_input("dequantize", w, scales, *biases, group_size, bits,
                           mode);
  auto out_shape = w.shape();
  out_shape.back() = w.shape(-1) * 32 / bits;
  return array(
      std::move(out_shape), out_type,
      std::make_shared<KQuantDequantize>(to_stream(s), group_size, bits, mode),
      {w, scales, *biases});
}

bool KQuantMatmul::is_equivalent(const Primitive &other) const {
  const auto &o = static_cast<const KQuantMatmul &>(other);
  return group_size_ == o.group_size_ && bits_ == o.bits_ && mode_ == o.mode_ &&
         transpose_ == o.transpose_;
}

std::vector<Shape>
KQuantMatmul::output_shapes(const std::vector<array> &inputs) {
  auto &w = inputs[1];
  auto out_shape = inputs[0].shape();
  out_shape.back() = transpose_ ? w.shape(-2) : w.shape(-1) * 32 / bits_;
  return {std::move(out_shape)};
}

bool KQuantGatherQMM::is_equivalent(const Primitive &other) const {
  const auto &o = static_cast<const KQuantGatherQMM &>(other);
  return group_size_ == o.group_size_ && bits_ == o.bits_ && mode_ == o.mode_ &&
         transpose_ == o.transpose_ && left_sorted_ == o.left_sorted_ &&
         right_sorted_ == o.right_sorted_;
}

std::vector<Shape>
KQuantGatherQMM::output_shapes(const std::vector<array> &inputs) {
  const auto &x = inputs[0];
  const auto &w = inputs[1];
  auto out_shape = inputs[4].shape();
  out_shape.push_back(x.shape(-2));
  out_shape.push_back(transpose_ ? w.shape(-2) : w.shape(-1) * 32 / bits_);
  return {std::move(out_shape)};
}

bool KQuantDequantize::is_equivalent(const Primitive &other) const {
  const auto &o = static_cast<const KQuantDequantize &>(other);
  return group_size_ == o.group_size_ && bits_ == o.bits_ && mode_ == o.mode_;
}

std::vector<Shape>
KQuantDequantize::output_shapes(const std::vector<array> &inputs) {
  auto out_shape = inputs[0].shape();
  out_shape.back() = inputs[0].shape(-1) * 32 / bits_;
  return {std::move(out_shape)};
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
// Both scale levels decode in float32, as ggml does: every (d, sub-scale) and
// (d, sub-scale, code) product is exact there, so q4k/q5k match llama.cpp
// bitwise and q6k up to the sign of zero (bias = -32 * scale folds the offset
// that ggml subtracts in integer). kq_qmm accumulates into T and kq_qmm_t into
// float, matching where the affine CPU kernels land.

namespace {

void validate_cpu_config(std::string_view tag, Mode mode, Dtype dtype, int bits,
                         int group_size) {
  if (dtype != float32 && dtype != float16 && dtype != bfloat16) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] Only float32, float16 and bfloat16 are supported but got "
        << dtype << ".";
    throw std::invalid_argument(msg.str());
  }
  if (bits != default_bits(mode) || group_size != default_group_size(mode)) {
    std::ostringstream msg;
    msg << "[" << tag << "] " << mode_name(mode)
        << " quantization requires group size " << default_group_size(mode)
        << " and " << default_bits(mode) << " bits but got group size "
        << group_size << " and " << bits << " bits.";
    throw std::invalid_argument(msg.str());
  }
}

array ensure_row_contiguous(const array &arr, cpu::CommandEncoder &encoder,
                            Stream s) {
  if (arr.flags().row_contiguous) {
    return arr;
  }
  auto copy = contiguous_copy_cpu(arr, s);
  encoder.add_temporary(copy);
  return copy;
}

template <typename T, int bits>
void extract_bits(const uint8_t *w_in, T *w_out) {
  static_assert(bits == 3 || bits == 5 || bits == 6);
  if (bits == 3) {
    w_out[0] = static_cast<T>(w_in[0] & 0x7);
    w_out[1] = static_cast<T>((w_in[0] & 0x38) >> 3);
    w_out[2] = static_cast<T>(((w_in[0] & 0xc0) >> 6) + ((w_in[1] & 0x1) << 2));
    w_out[3] = static_cast<T>((w_in[1] & 0xe) >> 1);
    w_out[4] = static_cast<T>((w_in[1] & 0x70) >> 4);
    w_out[5] = static_cast<T>(((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0x3) << 1));
    w_out[6] = static_cast<T>((w_in[2] & 0x1c) >> 2);
    w_out[7] = static_cast<T>((w_in[2] & 0xe0) >> 5);
  } else if (bits == 5) {
    w_out[0] = static_cast<T>(w_in[0] & 0x1f);
    w_out[1] = static_cast<T>(((w_in[0] & 0xe0) >> 5) + ((w_in[1] & 0x3) << 3));
    w_out[2] = static_cast<T>((w_in[1] & 0x7c) >> 2);
    w_out[3] = static_cast<T>(((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0xf) << 1));
    w_out[4] = static_cast<T>(((w_in[2] & 0xf0) >> 4) + ((w_in[3] & 0x1) << 4));
    w_out[5] = static_cast<T>((w_in[3] & 0x3e) >> 1);
    w_out[6] = static_cast<T>(((w_in[3] & 0xc0) >> 6) + ((w_in[4] & 0x7) << 2));
    w_out[7] = static_cast<T>((w_in[4] & 0xf8) >> 3);
  } else if (bits == 6) {
    w_out[0] = static_cast<T>(w_in[0] & 0x3f);
    w_out[1] =
        static_cast<T>(((w_in[0] >> 6) & 0x03) + ((w_in[1] & 0x0f) << 2));
    w_out[2] =
        static_cast<T>(((w_in[1] >> 4) & 0x0f) + ((w_in[2] & 0x03) << 4));
    w_out[3] = static_cast<T>((w_in[2] >> 2) & 0x3f);
  }
}

template <int bits, int super_ratio, bool has_min, bool nonlinear>
class KQScales {
public:
  static constexpr int per_group = has_min ? 2 : 1;

  KQScales(const uint8_t *scales, const float16_t *biases)
      : scales_(scales), biases_(biases) {}

  void at(size_t g, float &scale, float &bias) const {
    const float16_t *d = biases_ + (g / super_ratio) * per_group;
    const uint8_t *sc = scales_ + g * per_group;
    if constexpr (has_min) {
      scale = static_cast<float>(d[0]) * static_cast<float>(sc[0]);
      bias = -(static_cast<float>(d[1]) * static_cast<float>(sc[1]));
    } else {
      scale = static_cast<float>(d[0]) *
              static_cast<float>(static_cast<int8_t>(sc[0]));
      if constexpr (nonlinear) {
        bias = 0.0f;
      } else {
        bias = -static_cast<float>(1 << (bits - 1)) * scale;
      }
    }
  }

  void next(float &scale, float &bias) { at(g_++, scale, bias); }

private:
  const uint8_t *scales_;
  const float16_t *biases_;
  size_t g_ = 0;
};

constexpr int8_t kIQ4NLValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                     1,    13,   25,  38,  53,  69,  89,  113};

template <bool nonlinear>
inline float kq_value(float code, float scale, float bias) {
  if constexpr (nonlinear) {
    return scale * static_cast<float>(kIQ4NLValues[static_cast<int>(code)]);
  } else {
    return scale * code + bias;
  }
}

template <typename T, int bits, int group_size, int super_ratio, bool has_min,
          bool nonlinear>
void kq_qmm(T *result, const T *x, const uint32_t *w, const uint8_t *scales,
            const float16_t *biases, int M, int N, int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);
  constexpr int packs_in_group = group_size / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint8_t *w_local = (const uint8_t *)w;
    KQScales<bits, super_ratio, has_min, nonlinear> sb(scales, biases);
    std::fill(result, result + N, 0);
    for (int k = 0; k < K; k++) {
      T *result_local = result;
      float xi = static_cast<float>(*x++);
      for (int n = 0; n < N; n += group_size) {
        float scale;
        float bias;
        sb.next(scale, bias);
        for (int ng = 0; ng < packs_in_group; ng++) {
          if constexpr (bits == 3 || bits == 5 || bits == 6) {
            float wl[pack_factor];
            extract_bits<float, bits>(w_local, wl);
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              (*result_local++) +=
                  static_cast<T>(xi * kq_value<nonlinear>(wl[p], scale, bias));
            }
            w_local += bytes_per_pack;
          } else {
            uint8_t wi = *w_local++;
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              (*result_local++) += static_cast<T>(
                  xi * kq_value<nonlinear>(static_cast<float>(wi & bitmask),
                                           scale, bias));
              wi >>= bits;
            }
          }
        }
      }
    }
    result += N;
  }
}

template <typename T, int bits, int group_size, int super_ratio, bool has_min,
          bool nonlinear>
void kq_qmm_t(T *result, const T *x, const uint32_t *w, const uint8_t *scales,
              const float16_t *biases, int M, int N, int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);
  constexpr int packs_in_group = group_size / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint8_t *w_local = (const uint8_t *)w;
    KQScales<bits, super_ratio, has_min, nonlinear> sb(scales, biases);
    for (int n = 0; n < N; n++) {
      const T *x_local = x;
      float sum = 0;
      for (int k = 0; k < K; k += group_size) {
        float scale;
        float bias;
        sb.next(scale, bias);
        for (int kw = 0; kw < packs_in_group; kw++) {
          if constexpr (bits == 3 || bits == 5 || bits == 6) {
            float wl[pack_factor];
            extract_bits<float, bits>(w_local, wl);
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              sum += static_cast<float>(x_local[p]) *
                     kq_value<nonlinear>(wl[p], scale, bias);
            }
            w_local += bytes_per_pack;
            x_local += pack_factor;
          } else {
            uint8_t wi = *w_local++;
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              sum += static_cast<float>(*x_local++) *
                     kq_value<nonlinear>(static_cast<float>(wi & bitmask),
                                         scale, bias);
              wi >>= bits;
            }
          }
        }
      }
      *result = static_cast<T>(sum);
      result++;
    }
    x += K;
  }
}

template <typename T, int bits, int group_size, int super_ratio, bool has_min,
          bool nonlinear>
void kq_qmm_dispatch_transpose(T *result, const T *x, const uint32_t *w,
                               const uint8_t *scales, const float16_t *biases,
                               int M, int N, int K, bool transposed_w) {
  if (transposed_w) {
    kq_qmm_t<T, bits, group_size, super_ratio, has_min, nonlinear>(
        result, x, w, scales, biases, M, N, K);
  } else {
    kq_qmm<T, bits, group_size, super_ratio, has_min, nonlinear>(
        result, x, w, scales, biases, M, N, K);
  }
}

template <typename T>
void kq_qmm_dispatch_mode(T *result, const T *x, const uint32_t *w,
                          const uint8_t *scales, const float16_t *biases, int M,
                          int N, int K, Mode mode, bool transposed_w) {
  switch (mode) {
  case Mode::Q6K:
    kq_qmm_dispatch_transpose<T, 6, 16, 16, false, false>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  case Mode::Q4K:
    kq_qmm_dispatch_transpose<T, 4, 32, 8, true, false>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  case Mode::Q5K:
    kq_qmm_dispatch_transpose<T, 5, 32, 8, true, false>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  case Mode::Q3K:
    kq_qmm_dispatch_transpose<T, 3, 16, 16, false, false>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  case Mode::IQ4NL:
    kq_qmm_dispatch_transpose<T, 4, 32, 1, false, true>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  case Mode::IQ4XS:
    kq_qmm_dispatch_transpose<T, 4, 32, 8, false, true>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  case Mode::IQ3S:
    kq_qmm_dispatch_transpose<T, 8, 32, 8, false, false>(
        result, x, w, scales, biases, M, N, K, transposed_w);
    break;
  }
}

template <typename T>
void kq_qmm_dispatch_typed(array &out, const array &x, const array &w,
                           const array &scales, const array &biases, Mode mode,
                           bool transposed_w) {
  int K = x.shape(-1);
  int M = x.ndim() > 1 ? x.shape(-2) : 1;
  int N = out.shape(-1);
  int w_els = w.ndim() > 2 ? w.shape(-1) * w.shape(-2) : 0;
  int s_els = w.ndim() > 2 ? scales.shape(-1) * scales.shape(-2) : 0;
  int b_els = w.ndim() > 2 ? biases.shape(-1) * biases.shape(-2) : 0;
  int batch_size = x.size() / (K * M);

  auto out_ptr = out.data<T>();
  auto x_ptr = x.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<uint8_t>();
  auto biases_ptr = biases.data<float16_t>();
  for (int i = 0; i < batch_size; i++) {
    kq_qmm_dispatch_mode<T>(
        out_ptr + i * M * N,
        x_ptr + elem_to_loc(i * M * K, x.shape(), x.strides()),
        w_ptr + elem_to_loc(i * w_els, w.shape(), w.strides()),
        scales_ptr + elem_to_loc(i * s_els, scales.shape(), scales.strides()),
        biases_ptr + elem_to_loc(i * b_els, biases.shape(), biases.strides()),
        M, N, K, mode, transposed_w);
  }
}

void kq_qmm_dispatch(array &out, const array &x, const array &w,
                     const array &scales, const array &biases, Mode mode,
                     bool transposed_w) {
  switch (x.dtype()) {
  case float32:
    kq_qmm_dispatch_typed<float>(out, x, w, scales, biases, mode, transposed_w);
    break;
  case float16:
    kq_qmm_dispatch_typed<float16_t>(out, x, w, scales, biases, mode,
                                     transposed_w);
    break;
  case bfloat16:
    kq_qmm_dispatch_typed<bfloat16_t>(out, x, w, scales, biases, mode,
                                      transposed_w);
    break;
  default:
    break;
  }
}

template <typename T>
void kq_bs_qmm_dispatch_typed(array &out, const array &x, const array &w,
                              const array &scales, const array &biases,
                              const array &lhs_indices,
                              const array &rhs_indices, Mode mode,
                              bool transposed_w) {
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int w_els = w.shape(-1) * w.shape(-2);
  int s_els = scales.shape(-1) * scales.shape(-2);
  int b_els = biases.shape(-1) * biases.shape(-2);

  auto out_ptr = out.data<T>();
  auto x_ptr = x.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<uint8_t>();
  auto biases_ptr = biases.data<float16_t>();
  auto lhs_ptr = lhs_indices.data<uint32_t>();
  auto rhs_ptr = rhs_indices.data<uint32_t>();

  for (int i = 0; i < lhs_indices.size(); i++) {
    int x_idx =
        lhs_ptr[elem_to_loc(i, lhs_indices.shape(), lhs_indices.strides())];
    int w_idx =
        rhs_ptr[elem_to_loc(i, rhs_indices.shape(), rhs_indices.strides())];
    kq_qmm_dispatch_mode<T>(
        out_ptr + i * M * N,
        x_ptr + elem_to_loc(x_idx * M * K, x.shape(), x.strides()),
        w_ptr + elem_to_loc(w_idx * w_els, w.shape(), w.strides()),
        scales_ptr +
            elem_to_loc(w_idx * s_els, scales.shape(), scales.strides()),
        biases_ptr +
            elem_to_loc(w_idx * b_els, biases.shape(), biases.strides()),
        M, N, K, mode, transposed_w);
  }
}

void kq_bs_qmm_dispatch(array &out, const array &x, const array &w,
                        const array &scales, const array &biases,
                        const array &lhs_indices, const array &rhs_indices,
                        Mode mode, bool transposed_w) {
  switch (x.dtype()) {
  case float32:
    kq_bs_qmm_dispatch_typed<float>(out, x, w, scales, biases, lhs_indices,
                                    rhs_indices, mode, transposed_w);
    break;
  case float16:
    kq_bs_qmm_dispatch_typed<float16_t>(out, x, w, scales, biases, lhs_indices,
                                        rhs_indices, mode, transposed_w);
    break;
  case bfloat16:
    kq_bs_qmm_dispatch_typed<bfloat16_t>(out, x, w, scales, biases, lhs_indices,
                                         rhs_indices, mode, transposed_w);
    break;
  default:
    break;
  }
}

template <typename T, int bits, int group_size, int super_ratio, bool has_min,
          bool nonlinear>
void kq_dequantize(T *out, const uint32_t *w, const uint8_t *scales,
                   const float16_t *biases, size_t size) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);
  constexpr int packs_in_group = group_size / pack_factor;

  const uint8_t *w_local = (const uint8_t *)w;
  KQScales<bits, super_ratio, has_min, nonlinear> sb(scales, biases);
  for (size_t i = 0; i < size; i += group_size) {
    float scale;
    float bias;
    sb.next(scale, bias);
    for (int kw = 0; kw < packs_in_group; kw++) {
      if constexpr (bits == 3 || bits == 5 || bits == 6) {
        float wl[pack_factor];
        extract_bits<float, bits>(w_local, wl);
#pragma clang loop unroll(full)
        for (int p = 0; p < pack_factor; p++) {
          (*out++) = static_cast<T>(kq_value<nonlinear>(wl[p], scale, bias));
        }
        w_local += bytes_per_pack;
      } else {
        uint8_t wi = *w_local++;
#pragma clang loop unroll(full)
        for (int p = 0; p < pack_factor; p++) {
          (*out++) = static_cast<T>(kq_value<nonlinear>(
              static_cast<float>(wi & bitmask), scale, bias));
          wi >>= bits;
        }
      }
    }
  }
}

template <typename T>
void kq_dequantize_typed(array &out, const array &w, const array &scales,
                         const array &biases, Mode mode) {
  auto out_ptr = out.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<uint8_t>();
  auto biases_ptr = biases.data<float16_t>();
  size_t size = out.size();
  switch (mode) {
  case Mode::Q6K:
    kq_dequantize<T, 6, 16, 16, false, false>(out_ptr, w_ptr, scales_ptr,
                                              biases_ptr, size);
    break;
  case Mode::Q4K:
    kq_dequantize<T, 4, 32, 8, true, false>(out_ptr, w_ptr, scales_ptr,
                                            biases_ptr, size);
    break;
  case Mode::Q5K:
    kq_dequantize<T, 5, 32, 8, true, false>(out_ptr, w_ptr, scales_ptr,
                                            biases_ptr, size);
    break;
  case Mode::Q3K:
    kq_dequantize<T, 3, 16, 16, false, false>(out_ptr, w_ptr, scales_ptr,
                                              biases_ptr, size);
    break;
  case Mode::IQ4NL:
    kq_dequantize<T, 4, 32, 1, false, true>(out_ptr, w_ptr, scales_ptr,
                                            biases_ptr, size);
    break;
  case Mode::IQ4XS:
    kq_dequantize<T, 4, 32, 8, false, true>(out_ptr, w_ptr, scales_ptr,
                                            biases_ptr, size);
    break;
  case Mode::IQ3S:
    kq_dequantize<T, 8, 32, 8, false, false>(out_ptr, w_ptr, scales_ptr,
                                             biases_ptr, size);
    break;
  }
}

} // namespace

void KQuantMatmul::eval_cpu(const std::vector<array> &inputs, array &out) {
  validate_cpu_config("quantized_matmul", mode_, inputs[0].dtype(), bits_,
                      group_size_);
  auto &encoder = cpu::get_command_encoder(stream());
  auto x = ensure_row_contiguous(inputs[0], encoder, stream());
  auto w = ensure_row_contiguous(inputs[1], encoder, stream());
  auto scales = ensure_row_contiguous(inputs[2], encoder, stream());
  auto biases = ensure_row_contiguous(inputs[3], encoder, stream());
  out.set_data(allocator::malloc(out.nbytes()));
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(biases);
  encoder.set_output_array(out);
  encoder.dispatch(
      [out = array::unsafe_weak_copy(out), x = array::unsafe_weak_copy(x),
       w = array::unsafe_weak_copy(w), scales = array::unsafe_weak_copy(scales),
       biases = array::unsafe_weak_copy(biases), mode = mode_,
       transpose = transpose_]() mutable {
        kq_qmm_dispatch(out, x, w, scales, biases, mode, transpose);
      });
}

void KQuantGatherQMM::eval_cpu(const std::vector<array> &inputs, array &out) {
  validate_cpu_config("gather_qmm", mode_, inputs[0].dtype(), bits_,
                      group_size_);
  auto &lhs_indices = inputs[4];
  auto &rhs_indices = inputs[5];
  auto &encoder = cpu::get_command_encoder(stream());
  auto ensure_row_contiguous_last_dims = [s = stream(),
                                          &encoder](const array &arr) {
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (stride_0 == arr.shape(-1) && stride_1 == 1) {
      return arr;
    }
    auto copy = array(arr.shape(), arr.dtype(), nullptr, {});
    copy_cpu(arr, copy, CopyType::General, s);
    encoder.add_temporary(copy);
    return copy;
  };
  auto x = ensure_row_contiguous_last_dims(inputs[0]);
  auto w = ensure_row_contiguous_last_dims(inputs[1]);
  auto scales = ensure_row_contiguous_last_dims(inputs[2]);
  auto biases = ensure_row_contiguous_last_dims(inputs[3]);
  out.set_data(allocator::malloc(out.nbytes()));
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(biases);
  encoder.set_output_array(out);
  encoder.dispatch(
      [out = array::unsafe_weak_copy(out), x = array::unsafe_weak_copy(x),
       w = array::unsafe_weak_copy(w), scales = array::unsafe_weak_copy(scales),
       biases = array::unsafe_weak_copy(biases),
       lhs_indices = array::unsafe_weak_copy(lhs_indices),
       rhs_indices = array::unsafe_weak_copy(rhs_indices), mode = mode_,
       transpose = transpose_]() mutable {
        kq_bs_qmm_dispatch(out, x, w, scales, biases, lhs_indices, rhs_indices,
                           mode, transpose);
      });
}

void KQuantDequantize::eval_cpu(const std::vector<array> &inputs, array &out) {
  validate_cpu_config("dequantize", mode_, out.dtype(), bits_, group_size_);
  auto &encoder = cpu::get_command_encoder(stream());
  auto w = ensure_row_contiguous(inputs[0], encoder, stream());
  auto scales = ensure_row_contiguous(inputs[1], encoder, stream());
  auto biases = ensure_row_contiguous(inputs[2], encoder, stream());
  out.set_data(allocator::malloc(out.nbytes()));
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(biases);
  encoder.set_output_array(out);
  encoder.dispatch(
      [out = array::unsafe_weak_copy(out), w = array::unsafe_weak_copy(w),
       scales = array::unsafe_weak_copy(scales),
       biases = array::unsafe_weak_copy(biases), mode = mode_]() mutable {
        switch (out.dtype()) {
        case float32:
          kq_dequantize_typed<float>(out, w, scales, biases, mode);
          break;
        case float16:
          kq_dequantize_typed<float16_t>(out, w, scales, biases, mode);
          break;
        case bfloat16:
          kq_dequantize_typed<bfloat16_t>(out, w, scales, biases, mode);
          break;
        default:
          break;
        }
      });
}

} // namespace mlx::core::kquant
