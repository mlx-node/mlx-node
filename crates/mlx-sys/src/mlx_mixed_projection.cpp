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
#endif

#include <memory>

namespace mlx::core::mixed_projection {

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
    for (unsigned i = 0; i < p_.segments; ++i) {
      const auto& w = inputs[1 + 3 * i];
      const auto& sc = inputs[2 + 3 * i];
      const auto& b = inputs[3 + 3 * i];
      const int N = int(p_.columns[i]);
      const unsigned qbits = p_.bits[i];
      if (w.shape() != Shape{N, K * int(qbits) / 32} || w.dtype() != uint32 ||
          sc.shape() != Shape{N, K / 16} || sc.dtype() != (qbits == 6 ? int8 : uint8) ||
          b.shape() != Shape{N, K / 256 * (qbits == 6 ? 1 : 2)} || b.dtype() != float16)
        throw std::invalid_argument("mixed projection replay changed its weight-plane contract");
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
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "mixed_projection", "m8", "mixed_projection_m8"));
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

std::optional<std::vector<array>> build(std::vector<array> a, const unsigned* bits, unsigned segments) {
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
    const unsigned qbits = bits[i];
    if (qbits < 4 || qbits > 6 || w.ndim() != 2)
      throw std::invalid_argument("invalid mixed projection quantization contract");
    const int N = w.shape(0);
    if (N < 64 || N > (1 << 20) || N % 64) return std::nullopt;
    if (w.shape() != Shape{N, K * int(qbits) / 32} || w.dtype() != uint32 ||
        sc.shape() != Shape{N, K / 16} || sc.dtype() != (qbits == 6 ? int8 : uint8) ||
        b.shape() != Shape{N, K / 256 * (qbits == 6 ? 1 : 2)} || b.dtype() != float16)
      throw std::invalid_argument("malformed mixed projection weight planes");
    // Keep the existing GEMM threshold and split policy: changing either can
    // change FP32 accumulation order even though the mathematical GEMM agrees.
    if (8 >= kquant::tiled_qmv_vector_limit(K, N)) return std::nullopt;
    p.columns[i] = N;
    p.bits[i] = qbits;
    p.splits[i] = kquant::tiled_projection_splits(N, K);
    p.group_end[i] = (i ? p.group_end[i - 1] : 0) + N / 64 * p.splits[i];
    p.partial_offset[i] = i ? p.partial_offset[i - 1] + 8 * p.columns[i - 1] * p.splits[i - 1] : 0;
    p.counter_offset[i] = i ? p.counter_offset[i - 1] + p.columns[i - 1] / 64 : 0;
    auto shape = x.shape(); shape.back() = N;
    shapes.push_back(std::move(shape));
  }
  // Opt-in source-style grid: one split count for the fused width. Preserve
  // the established per-segment policy unless explicitly requested, because
  // a different split count changes the FP32 reduction order before BF16.
  const char* global_split = std::getenv("MLX_MIXED_PROJECTION_GLOBAL_SPLIT");
  if (global_split && std::strcmp(global_split, "1") == 0) {
    unsigned columns = 0;
    for (unsigned i = 0; i < segments; ++i) columns += p.columns[i];
    const unsigned splits = kquant::tiled_projection_splits(int(columns), K);
    unsigned groups = 0, partials = 0;
    for (unsigned i = 0; i < segments; ++i) {
      p.splits[i] = splits;
      p.partial_offset[i] = partials;
      groups += p.columns[i] / 64 * splits;
      partials += 8 * p.columns[i] * splits;
      p.group_end[i] = groups;
    }
  }
  return array::make_arrays(std::move(shapes), std::move(dtypes),
      std::make_shared<MixedProjection>(to_stream(Device::gpu), p), std::move(a));
}
} // namespace mlx::core::mixed_projection

// 1 = lazy graph constructed, 0 = unsupported and caller uses its original
// projections, -1 = invalid operands. No evaluation or cache mutation occurs.
extern "C" int mlx_mixed_projection(mlx_array* const* inputs, size_t count,
                                     const unsigned* bits, unsigned segments,
                                     mlx_array** outputs) {
  if (outputs && segments <= 3)
    for (unsigned i = 0; i < segments; ++i) outputs[i] = nullptr;
  try {
    if (!inputs || !bits || !outputs || segments < 2 || segments > 3 || count != 1 + 3 * segments)
      throw std::invalid_argument("null or invalid mixed projection arguments");
    std::vector<mlx::core::array> a;
    a.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (!inputs[i]) throw std::invalid_argument("null mixed projection input");
      a.push_back(*reinterpret_cast<mlx::core::array*>(inputs[i]));
    }
    auto result = mlx::core::mixed_projection::build(std::move(a), bits, segments);
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
