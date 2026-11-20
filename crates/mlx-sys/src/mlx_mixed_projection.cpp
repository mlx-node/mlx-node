// Heterogeneous target-mixer projections, following Splash's fused segment
// dispatch. Arithmetic is the existing mlx-node K-quant M=8 NAX contract.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_kquant.h"
#include "metal/mixed_projection/abi.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#include "mlx_test_counters.h"
#endif

#include <memory>
#include <string>
#include <vector>

namespace mlx::core::mixed_projection {


namespace {

[[maybe_unused]] bool joins(unsigned format) {
  return format == kMixedA8G32 || format == kMixedIQ4XS || format == kMixedIQ4NL;
}

// Weight bits per value of a segment format; 0 for an unknown code.
int format_bits(unsigned format) {
  switch (format) {
  case kMixedQ4K:
  case kMixedIQ4XS:
  case kMixedIQ4NL:
    return 4;
  case kMixedQ5K:
    return 5;
  case kMixedQ6K:
    return 6;
  case kMixedA8G32:
    return 8;
  default:
    return 0;
  }
}

// The (.weight, .scales, .biases) planes a segment of `format` holds for an
// [N, K] projection: the K-quant contract (mlx_kquant.h) or MLX affine with
// F32 companions.
bool planes_match(unsigned format, int N, int K, const array& w, const array& sc,
                  const array& b) {
  Shape scales{N, K / 16};
  Shape biases{N, K / 256 * 2};
  Dtype scales_type = uint8;
  Dtype biases_type = float16;
  switch (format) {
  case kMixedQ6K:
    scales_type = int8;
    biases = {N, K / 256};
    break;
  case kMixedIQ4XS:
    scales = {N, K / 32};
    scales_type = int8;
    biases = {N, K / 256};
    break;
  case kMixedIQ4NL:
    scales = {N, K / 32};
    scales_type = int8;
    biases = {N, K / 32};
    break;
  case kMixedA8G32:
    scales = {N, K / 32};
    scales_type = float32;
    biases = {N, K / 32};
    biases_type = float32;
    break;
  default:
    break;
  }
  return w.shape() == Shape{N, K * format_bits(format) / 32} && w.dtype() == uint32 &&
      sc.shape() == scales && sc.dtype() == scales_type && b.shape() == biases &&
      b.dtype() == biases_type;
}

// qmv_wide's tile for M rows (mlx_affine_mixed_qmm.cpp): fewest tiles, then
// the smallest that fills them, 8 vectors only from N 2048. Bits do not
// depend on it; the standalone call's grid is kept.
unsigned qmv_wide_vectors(int M, int N) {
  const int tile_cap = N >= 2048 ? 8 : 5;
  const int n_tiles = (M + tile_cap - 1) / tile_cap;
  return unsigned((M + n_tiles - 1) / n_tiles);
}

} // namespace

const std::vector<std::string>& kernel_names() {
  static const std::vector<std::string> names{"mixed_projection_m8", "mixed_projection_m8_joined"};
  return names;
}

bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  const char* enabled = std::getenv("MLX_MIXED_PROJECTION");
  return !(enabled && std::strcmp(enabled, "0") == 0) &&
      !detail::in_grad_tracing() && default_device() == Device::gpu &&
      is_available(Device::gpu) && metal::is_nax_available();
#else
  return false;
#endif
}

bool global_split() {
  const char* value = std::getenv("MLX_MIXED_PROJECTION_GLOBAL_SPLIT");
  return value && std::strcmp(value, "1") == 0;
}

class MixedProjection : public fast::Custom {
 public:
  MixedProjection(Stream s, MixedProjectionParams p)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("mixed projection has no CPU fallback");
        }), p_(p) {}
  DEFINE_NAME(MixedProjection)
  bool is_equivalent(const Primitive& other) const override {
    return std::memcmp(&p_, &static_cast<const MixedProjection&>(other).p_, sizeof(p_)) == 0;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    std::vector<Shape> shapes;
    for (unsigned i = 0; i < p_.segments; ++i) {
      auto shape = inputs[0].shape();
      shape.back() = p_.columns[i];
      shapes.push_back(std::move(shape));
    }
    return shapes;
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("mixed projection requires Metal");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    if (inputs[0].shape(-1) != int(p_.input_size) || inputs[0].size() != 8 * size_t(p_.input_size)) {
      throw std::invalid_argument("mixed projection replay changed its eight-row contract");
    }
    // Shapeless replay substitutes operands while retaining p_ and its launch grid.
    const int K = int(p_.input_size);
    bool joined = false;
    for (unsigned i = 0; i < p_.segments; ++i) {
      if (!planes_match(p_.format[i], int(p_.columns[i]), K, inputs[1 + 3 * i],
                        inputs[2 + 3 * i], inputs[3 + 3 * i]))
        throw std::invalid_argument("mixed projection replay changed its weight-plane contract");
      joined |= joins(p_.format[i]);
    }
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    // The tensor tile and packed weight readers use 16-byte vector loads.
    // A contiguous slice may still start inside its allocation: normalize
    // that offset as well as strides before binding any operand.
    std::vector<array> a;
    a.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.flags().row_contiguous && input.offset() % 16 == 0) a.push_back(input);
      else {
        array contiguous(input.shape(), input.dtype(), nullptr, {});
        contiguous.set_data(allocator::malloc(contiguous.nbytes()));
        copy_gpu(input, contiguous, CopyType::General, s);
        enc.add_temporary(contiguous);
        a.push_back(std::move(contiguous));
      }
    }
    for (auto& output : outputs) output.set_data(allocator::malloc(output.nbytes()));
    const unsigned last = p_.segments - 1;
    const unsigned partial_count = p_.partial_offset[last] + 8 * p_.columns[last] * p_.splits[last];
    const unsigned counter_count = p_.counter_offset[last] + p_.columns[last] / 64;
    bool split = false;
    for (unsigned i = 0; i < p_.segments; ++i) split |= p_.splits[i] > 1;
    array partials = outputs[0];
    array counters = outputs[0];
    if (split) {
      partials = array({int(partial_count)}, float32, nullptr, {});
      partials.set_data(allocator::malloc(partials.nbytes()));
      enc.add_temporary(partials);
      counters = kquant::tiled_projection_counters(s, counter_count);
    }
    bridge_testing::record(joined ? "mixed_projection_joined" : "mixed_projection");
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "mixed_projection", "m8", kernel_names()[joined ? 1 : 0]));
    enc.set_input_array(a[0], 0);
    for (unsigned i = 0; i < 3; ++i) {
      unsigned j = std::min(i, last);
      for (unsigned plane = 0; plane < 3; ++plane)
        enc.set_input_array(a[1 + j * 3 + plane], 1 + i * 3 + plane);
      enc.set_output_array(outputs[j], 10 + i);
    }
    enc.set_output_array(partials, 13);
    enc.set_output_array(counters, 14);
    enc.set_bytes(p_, 15);
    enc.dispatch_threadgroups(MTL::Size(p_.group_end[last], 1, 1), MTL::Size(64, 1, 1));
#else
    throw std::runtime_error("mixed projection requires Metal");
#endif
  }
 private:
  MixedProjectionParams p_;
};

std::optional<std::vector<array>> build(std::vector<array> a, const unsigned* formats, unsigned segments) {
  if (!available()) return std::nullopt;
  if (segments < 2 || segments > 3 || a.size() != 1 + 3 * segments || a[0].ndim() < 2)
    throw std::invalid_argument("invalid mixed projection operands");
  const auto& x = a[0];
  const int K = x.shape(-1);
  if (x.dtype() != bfloat16 || K < 256 || K > (1 << 20) || K % 256 || x.size() != size_t(8) * K)
    return std::nullopt;
  MixedProjectionParams p{};
  p.input_size = K;
  p.segments = segments;
  p.spread = kquant::tiled_projection_spread(to_stream(Device::gpu));
  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes(segments, bfloat16);
  for (unsigned i = 0; i < segments; ++i) {
    const auto& w = a[1 + 3 * i];
    const auto& sc = a[2 + 3 * i];
    const auto& b = a[3 + 3 * i];
    const unsigned format = formats[i];
    if (!format_bits(format) || w.ndim() != 2)
      throw std::invalid_argument("invalid mixed projection quantization contract");
    const int N = w.shape(0);
    const bool rows = format == kMixedA8G32;
    if (N < (rows ? 1 : 64) || N > (1 << 20) || (!rows && N % 64)) return std::nullopt;
    if (!planes_match(format, N, K, w, sc, b))
      throw std::invalid_argument("malformed mixed projection weight planes");
    unsigned groups = 0;
    if (rows) {
      // The standalone call's qmv_wide grid: (8 / vectors) x (N / 8), one split.
      p.vectors[i] = qmv_wide_vectors(8, N);
      p.splits[i] = 1;
      unsigned column_blocks = (unsigned(N) + 7) / 8;
      groups = (8 + p.vectors[i] - 1) / p.vectors[i] * column_blocks;
    } else {
      // Keep the existing GEMM threshold and split policy: changing either can
      // change FP32 accumulation order even though the mathematical GEMM agrees.
      if (8 >= kquant::tiled_qmv_vector_limit(K, N)) return std::nullopt;
      p.splits[i] = kquant::tiled_projection_splits(N, K);
      groups = N / 64 * p.splits[i];
    }
    p.columns[i] = N;
    p.format[i] = format;
    p.group_end[i] = (i ? p.group_end[i - 1] : 0) + groups;
    p.partial_offset[i] = i ? p.partial_offset[i - 1] + 8 * p.columns[i - 1] * p.splits[i - 1] : 0;
    p.counter_offset[i] = i ? p.counter_offset[i - 1] + p.columns[i - 1] / 64 : 0;
    auto shape = x.shape(); shape.back() = N;
    shapes.push_back(std::move(shape));
  }
  // Opt-in source-style grid: one split count for the fused width. Preserve
  // the established per-segment policy unless explicitly requested, because
  // a different split count changes the FP32 reduction order before BF16.
  // Affine rows keep their qmv_wide grid.
  if (global_split()) {
    unsigned columns = 0;
    for (unsigned i = 0; i < segments; ++i)
      if (p.format[i] != kMixedA8G32) columns += p.columns[i];
    const unsigned splits = kquant::tiled_projection_splits(int(columns), K);
    unsigned own[3] = {};
    for (unsigned i = 0; i < segments; ++i)
      own[i] = p.group_end[i] - (i ? p.group_end[i - 1] : 0);
    unsigned groups = 0, partials = 0;
    for (unsigned i = 0; i < segments; ++i) {
      if (p.format[i] == kMixedA8G32) {
        groups += own[i];
      } else {
        p.splits[i] = splits;
        groups += p.columns[i] / 64 * splits;
      }
      p.partial_offset[i] = partials;
      partials += 8 * p.columns[i] * p.splits[i];
      p.group_end[i] = groups;
    }
  }
  return array::make_arrays(std::move(shapes), std::move(dtypes),
      std::make_shared<MixedProjection>(to_stream(Device::gpu), p), std::move(a));
}
} // namespace mlx::core::mixed_projection

extern "C" bool mlx_mixed_projection_global_split() {
  return mlx::core::mixed_projection::global_split();
}

// 1 = lazy graph constructed, 0 = unsupported and caller uses its original
// projections, -1 = invalid operands. No evaluation or cache mutation occurs.

// `formats` holds one MixedProjectionFormat code per segment
// (metal/mixed_projection/abi.h).
extern "C" int mlx_mixed_projection(mlx_array* const* inputs, size_t count,
                                     const unsigned* formats, unsigned segments,
                                     mlx_array** outputs) {
  if (outputs && segments <= 3)
    for (unsigned i = 0; i < segments; ++i) outputs[i] = nullptr;
  try {
    if (!inputs || !formats || !outputs || segments < 2 || segments > 3 || count != 1 + 3 * segments)
      throw std::invalid_argument("null or invalid mixed projection arguments");
    std::vector<mlx::core::array> a;
    a.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (!inputs[i]) throw std::invalid_argument("null mixed projection input");
      a.push_back(*reinterpret_cast<mlx::core::array*>(inputs[i]));
    }
    auto result = mlx::core::mixed_projection::build(std::move(a), formats, segments);
    if (!result) return 0;
    std::vector<std::unique_ptr<mlx::core::array>> handles;
    for (auto& output : *result) handles.push_back(std::make_unique<mlx::core::array>(std::move(output)));
    for (unsigned i = 0; i < segments; ++i)
      outputs[i] = reinterpret_cast<mlx_array*>(handles[i].release());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_mixed_projection: %s\n", e.what());
    return -1;
  }
}
