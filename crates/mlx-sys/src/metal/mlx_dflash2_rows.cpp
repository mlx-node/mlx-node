// The target rows of a sampled DFlash2 cycle in two dispatches
// (metal/sampling/dflash2_rows.metal; structure of Splash
// runtime/ops/Sampling.cpp addSelection, Apache-2.0).
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "metal/sampling/abi.h"
#include <cmath>
#include <cstring>
#include <memory>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#endif

namespace mlx::core::dflash2_rows {
namespace {

// MLX's sampler treats a temperature at or below this as greedy
// (GREEDY_TEMPERATURE_EPS in mlx_misc_ops.cpp).
constexpr float kGreedyTemperature = 1e-6f;
// A record holds token ids as floats.
constexpr int kMaximumVocabulary = 1 << 24;
constexpr int kMaximumRows = 64;

#ifdef MLX_NODE_METAL_ENABLED
bool fits(metal::Device& device, MTL::ComputePipelineState* pipeline,
          NS::UInteger threads) {
  return pipeline->threadExecutionWidth() == 32 &&
      pipeline->maxTotalThreadsPerThreadgroup() >= threads &&
      pipeline->staticThreadgroupMemoryLength() <=
          device.mtl_device()->maxThreadgroupMemoryLength();
}
#endif

// A GPU of no Apple family (a hosted VM GPU) runs these kernels and returns
// zero records.
bool launches() {
#ifdef MLX_NODE_METAL_ENABLED
  static const bool launch = [] {
    try {
      if (!is_available(Device::gpu)) return false;
      auto& device = metal::device(Device::gpu);
      if (!device.mtl_device()->supportsFamily(MTL::GPUFamilyApple7)) return false;
      return fits(device, fast::paged::get_prebuilt_kernel(
                              device, "dflash2_rows", "mass", "dflash2_rows_mass"),
                  DFLASH2_ROWS_SHARD_THREADS) &&
          fits(device, fast::paged::get_prebuilt_kernel(
                           device, "dflash2_rows", "select", "dflash2_rows_select"),
               DFLASH2_ROWS_THREADS);
    } catch (const std::exception& e) {
      mlx_trace_native_error("dflash2_sampled_rows", e.what());
      return false;
    }
  }();
  return launch;
#else
  return false;
#endif
}

bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  return !detail::in_grad_tracing() && default_device() == Device::gpu &&
      launches();
#else
  return false;
#endif
}

// logits [R, V] f32, uniforms [R] f32, and for D > 0 drafted rows: draft ids
// [D] i32, candidate ids [D, C] i32, candidate probabilities [D, C] f32.
bool supported(const std::vector<array>& a, int drafted, int top_k) {
  if (a.size() != (drafted > 0 ? 5 : 2)) return false;
  if (top_k < 1 || top_k > int(DFLASH2_ROWS_MAX_TOP_K)) return false;
  const auto& logits = a[0];
  if (logits.ndim() != 2 || logits.dtype() != float32 || logits.shape(0) < 1 ||
      logits.shape(0) > kMaximumRows || logits.shape(1) < 1 ||
      logits.shape(1) > kMaximumVocabulary)
    return false;
  const int rows = logits.shape(0);
  if (a[1].dtype() != float32 || a[1].ndim() != 1 || a[1].shape(0) != rows) return false;
  if (drafted == 0) return true;
  if (drafted < 0 || drafted > rows) return false;
  if (a[2].dtype() != int32 || a[2].ndim() != 1 || a[2].shape(0) != drafted) return false;
  if (a[3].dtype() != int32 || a[3].ndim() != 2 || a[3].shape(0) != drafted ||
      a[3].shape(1) < 1 || a[3].shape(1) > int(DFLASH2_ROWS_MAX_CANDIDATES))
    return false;
  return a[4].dtype() == float32 && a[4].shape() == a[3].shape();
}

class Dflash2SampledRows : public fast::Custom {
 public:
  Dflash2SampledRows(Stream s, Dflash2RowsParams params)
      : Custom(s, [](std::vector<array>) -> std::vector<array> {
          throw std::runtime_error("DFlash2 sampled rows require Metal");
        }), params_(params) {}
  DEFINE_NAME(Dflash2SampledRows)
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Dflash2SampledRows&>(other);
    return std::memcmp(&params_, &rhs.params_, sizeof(params_)) == 0;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    return {{a[0].shape(0), int(DFLASH2_ROWS_RECORD)}};
  }
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DFlash2 sampled rows require Metal");
  }
  void eval_gpu(const std::vector<array>& a, std::vector<array>& outputs) override {
#ifdef MLX_NODE_METAL_ENABLED
    const int drafted = int(params_.drafted_rows);
    if (!supported(a, drafted, int(params_.top_k)) ||
        a[0].shape(0) != int(params_.rows) || a[0].shape(1) != int(params_.vocabulary) ||
        (drafted > 0 && a[3].shape(1) != int(params_.candidates)))
      throw std::invalid_argument("DFlash2 sampled rows replay changed its contract");
    auto& s = stream();
    auto& device = metal::device(s.device);
    auto& enc = metal::get_command_encoder(s);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    std::vector<array> in;
    in.reserve(a.size());
    for (const auto& input : a) {
      if (input.flags().row_contiguous) {
        in.push_back(input);
        continue;
      }
      array contiguous(input.shape(), input.dtype(), nullptr, {});
      contiguous.set_data(allocator::malloc(contiguous.nbytes()));
      copy_gpu(input, contiguous, CopyType::General, s);
      enc.add_temporary(contiguous);
      in.push_back(std::move(contiguous));
    }
    const int rows = int(params_.rows);
    array masses({rows * int(DFLASH2_ROWS_SHARDS), 3}, float32, nullptr, {});
    masses.set_data(allocator::malloc(masses.nbytes()));
    enc.add_temporary(masses);
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "dflash2_rows", "mass", "dflash2_rows_mass"));
    enc.set_input_array(in[0], 0);
    enc.set_output_array(masses, 1);
    enc.set_bytes(params_, 2);
    enc.dispatch_threadgroups(MTL::Size(rows * DFLASH2_ROWS_SHARDS, 1, 1),
                              MTL::Size(DFLASH2_ROWS_SHARD_THREADS, 1, 1));
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(
        device, "dflash2_rows", "select", "dflash2_rows_select"));
    enc.set_input_array(in[0], 0);
    enc.set_input_array(masses, 1);
    // A cycle with no drafted row reads none of the draft buffers.
    for (unsigned i = 0; i < 3; ++i)
      enc.set_input_array(drafted > 0 ? in[2 + i] : in[0], 2 + i);
    enc.set_input_array(in[1], 5);
    enc.set_output_array(out, 6);
    enc.set_bytes(params_, 7);
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(DFLASH2_ROWS_THREADS, 1, 1));
#else
    throw std::runtime_error("DFlash2 sampled rows require Metal");
#endif
  }
 private:
  Dflash2RowsParams params_;
};
}  // namespace
}  // namespace mlx::core::dflash2_rows


extern "C" bool mlx_dflash2_sampled_rows_supported() {
  return mlx::core::dflash2_rows::launches();
}

// 1: constructed; 0: not taken (no Metal, a GPU the kernels do not launch on,
// or a shape, dtype or top_k outside the kernels' contract); -1: invalid
// argument or construction error. The output is `[rows, 4]` f32, one record
// per row: the draft token's target probability, its draft probability, the
// row's drawn token (-1: none) and the count of the tokens the row keeps.
// `draft_ids`, `candidate_ids` and `candidate_probabilities` are null for a
// cycle with no drafted row.
extern "C" int mlx_dflash2_sampled_rows(
    mlx_array* logits, mlx_array* uniforms, mlx_array* draft_ids,
    mlx_array* candidate_ids, mlx_array* candidate_probabilities,
    float temperature, int top_k, float top_p, float min_p, mlx_array** output) {
  if (output) *output = nullptr;
  try {
    using namespace mlx::core;
    if (!logits || !uniforms || !output || !std::isfinite(temperature) ||
        temperature <= dflash2_rows::kGreedyTemperature)
      return -1;
    const bool drafted = draft_ids != nullptr;
    if (drafted != (candidate_ids != nullptr) ||
        drafted != (candidate_probabilities != nullptr))
      return -1;
    if (!dflash2_rows::available()) return 0;
    std::vector<array> a{*reinterpret_cast<array*>(logits),
                         *reinterpret_cast<array*>(uniforms)};
    int drafted_rows = 0;
    if (drafted) {
      a.push_back(*reinterpret_cast<array*>(draft_ids));
      a.push_back(*reinterpret_cast<array*>(candidate_ids));
      a.push_back(*reinterpret_cast<array*>(candidate_probabilities));
      drafted_rows = a[2].ndim() == 1 ? a[2].shape(0) : -1;
      if (drafted_rows == 0) return -1;
    }
    if (!dflash2_rows::supported(a, drafted_rows, top_k)) return 0;
    Dflash2RowsParams params{};
    params.vocabulary = a[0].shape(1);
    params.rows = a[0].shape(0);
    params.drafted_rows = drafted_rows;
    params.candidates = drafted ? a[3].shape(1) : 0;
    params.top_k = top_k;
    if (top_p > 0.0f && top_p < 1.0f) {
      params.filters |= DFLASH2_ROWS_TOP_P;
      params.top_p_threshold = 1.0f - (top_p - 1e-7f);
    }
    if (min_p > 0.0f) {
      params.filters |= DFLASH2_ROWS_MIN_P;
      params.min_p = min_p;
    }
    params.inverse_temperature = 1.0f / temperature;
    const Shape shape{a[0].shape(0), int(DFLASH2_ROWS_RECORD)};
    auto result = array(shape, float32,
        std::make_shared<dflash2_rows::Dflash2SampledRows>(to_stream(Device::gpu), params),
        std::move(a));
    *output = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    return 1;
  } catch (const std::exception& e) {
    mlx_trace_native_error("dflash2_sampled_rows", e.what());
    return -1;
  }
}
