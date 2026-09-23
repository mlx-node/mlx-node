#include "mlx_common.h"
#include "mlx/backend/common/segmented_sdpa_plan.h"

#ifdef MLX_NODE_METAL_ENABLED

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/sdpa_vector_plan.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

SegmentedSdpaCapabilities
segmented_sdpa_capabilities(MTL::ComputePipelineState *pipeline,
                            MTL::Device *device) {
  return {pipeline->threadExecutionWidth(),
          pipeline->maxTotalThreadsPerThreadgroup(),
          pipeline->staticThreadgroupMemoryLength(),
          device->maxThreadgroupMemoryLength()};
}

namespace {

constexpr int kHeadDimension = 256;

void validate_segmented_sdpa(const std::vector<array> &inputs) {
  if (inputs.size() != 5) {
    throw std::invalid_argument("segmented_sdpa expects five inputs");
  }
  for (const auto &input : inputs) {
    if (input.ndim() != 4 || input.dtype() != bfloat16 ||
        input.strides(-1) != 1) {
      throw std::invalid_argument("segmented_sdpa requires rank-4 BF16 inputs "
                                  "with contiguous head dimension");
    }
  }
  const auto &q = inputs[0];
  const auto &pk = inputs[1];
  const auto &pv = inputs[2];
  const auto &nk = inputs[3];
  const auto &nv = inputs[4];
  const int q_heads = q.shape(1);
  const int kv_heads = pk.shape(1);
  if (q.shape(0) != pk.shape(0) || q.shape(0) != pv.shape(0) ||
      q.shape(0) != nk.shape(0) || q.shape(0) != nv.shape(0) ||
      q.shape(2) < 1 || q.shape(2) > 8 || q.shape(3) != kHeadDimension ||
      pk.shape() != pv.shape() || nk.shape() != nv.shape() ||
      pk.shape(3) != kHeadDimension || nk.shape(3) != kHeadDimension ||
      nk.shape(1) != kv_heads || kv_heads < 1 || q_heads < 1 ||
      q_heads % kv_heads != 0 || q_heads / kv_heads > 32 ||
      nk.shape(2) < q.shape(2) || pk.shape(2) < 1 ||
      int64_t(pk.shape(2)) + nk.shape(2) > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("unsupported segmented_sdpa shape");
  }
}

std::vector<array> segmented_fallback(std::vector<array> inputs, float scale,
                                      bool causal, Stream stream) {
  auto keys = concatenate({inputs[1], inputs[3]}, 2, stream);
  auto values = concatenate({inputs[2], inputs[4]}, 2, stream);
  return {scaled_dot_product_attention(inputs[0], keys, values, scale,
                                       causal ? "causal" : "", std::nullopt,
                                       std::nullopt, stream)};
}

struct Pipelines {
  MTL::ComputePipelineState *stage1;
  MTL::ComputePipelineState *stage2;
  SegmentedSdpaLaunchPlan plan;
};

Pipelines get_pipelines(metal::Device &device, int query_length, int gqa_factor,
                        int total_length, int q_heads, int kv_heads,
                        bool causal) {
  const bool two_pass =
      sdpa_vector_uses_two_pass(device, total_length, q_heads, kv_heads);
  const int partitions =
      two_pass ? sdpa_vector_partition_count(device, total_length,
                                             gqa_factor * query_length)
               : 32;
  metal::MTLFCList constants = {{&causal, MTL::DataType::DataTypeBool, 22}};
  if (two_pass) {
    constants.emplace_back(&partitions, MTL::DataType::DataTypeInt, 26);
  }
  std::string base = two_pass
                         ? "sdpa_vector_segmented_2pass_1_bfloat16_t_256_256"
                         : "sdpa_vector_segmented_bfloat16_t_256_256";
  std::string hash =
      base + (causal ? "_c_" : "_nc_") + std::to_string(partitions);
  auto *stage1 = device.get_kernel(base, hash, constants);
  MTL::ComputePipelineState *stage2 = nullptr;
  if (two_pass) {
    stage2 = device.get_kernel("sdpa_vector_2pass_2_bfloat16_t_256");
  }
  auto c1 = segmented_sdpa_capabilities(stage1, device.mtl_device());
  std::optional<SegmentedSdpaCapabilities> c2;
  if (stage2 != nullptr) {
    c2 = segmented_sdpa_capabilities(stage2, device.mtl_device());
  }
  auto plan = plan_segmented_sdpa_launch(query_length, gqa_factor, two_pass,
                                         partitions, c1, c2 ? &*c2 : nullptr);
  return {stage1, stage2, plan};
}

class SegmentedSdpa final : public Custom {
public:
  SegmentedSdpa(Stream stream, float scale, bool causal)
      : Custom(stream,
               [scale, causal, stream](std::vector<array> inputs) {
                 return segmented_fallback(std::move(inputs), scale, causal,
                                           stream);
               }),
        scale_(scale), causal_(causal) {}

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("SegmentedSdpa CPU NYI");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    validate_segmented_sdpa(inputs);
    const auto &q = inputs[0];
    const auto &pk = inputs[1];
    const auto &pv = inputs[2];
    const auto &nk = inputs[3];
    const auto &nv = inputs[4];
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const int q_len = q.shape(2);
    const int q_heads = q.shape(1);
    const int kv_heads = pk.shape(1);
    const int gqa = q_heads / kv_heads;
    const int prefix_n = pk.shape(2);
    const int new_n = nk.shape(2);
    auto pipelines = get_pipelines(device, q_len, gqa, prefix_n + new_n,
                                   q_heads, kv_heads, causal_);
    if (!pipelines.plan.supported) {
      throw std::runtime_error("segmented SDPA pipeline capabilities changed "
                               "after graph construction");
    }

    auto &out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(pipelines.stage1);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(pk, 1);
    encoder.set_input_array(pv, 2);
    encoder.set_input_array(nk, 3);
    encoder.set_input_array(nv, 4);

    std::vector<int64_t> strides = {
        q.strides(0),  q.strides(1),  q.strides(2),  pk.strides(0),
        pk.strides(1), pk.strides(2), pv.strides(0), pv.strides(1),
        pv.strides(2), nk.strides(0), nk.strides(1), nk.strides(2),
        nv.strides(0), nv.strides(1), nv.strides(2)};

    if (!pipelines.plan.two_pass) {
      encoder.set_output_array(out, 5);
      encoder.set_bytes(gqa, 6);
      encoder.set_bytes(prefix_n, 7);
      encoder.set_bytes(new_n, 8);
      encoder.set_vector_bytes(strides, 9);
      encoder.set_bytes(scale_, 10);
      encoder.set_bytes(q_heads, 11);
      encoder.dispatch_threadgroups(
          MTL::Size(q.shape(0) * q_heads, q_len, 1),
          MTL::Size(pipelines.plan.stage1_threads, 1, 1));
      return;
    }

    const int partitions = pipelines.plan.partitions;
    Shape partial_shape = {q.shape(0), q_heads, q_len, partitions,
                           kHeadDimension};
    Shape stats_shape = {q.shape(0), q_heads, q_len, partitions};
    array partials(partial_shape, bfloat16, nullptr, {});
    array sums(stats_shape, float32, nullptr, {});
    array maxs(stats_shape, float32, nullptr, {});
    partials.set_data(allocator::malloc(partials.nbytes()));
    sums.set_data(allocator::malloc(sums.nbytes()));
    maxs.set_data(allocator::malloc(maxs.nbytes()));
    encoder.add_temporary(partials);
    encoder.add_temporary(sums);
    encoder.add_temporary(maxs);
    encoder.set_output_array(partials, 5);
    encoder.set_output_array(sums, 6);
    encoder.set_output_array(maxs, 7);
    encoder.set_bytes(prefix_n, 8);
    encoder.set_bytes(new_n, 9);
    encoder.set_vector_bytes(strides, 10);
    encoder.set_bytes(scale_, 11);
    encoder.dispatch_threadgroups(MTL::Size(kv_heads, q.shape(0), partitions),
                                  MTL::Size(32, gqa, q_len));

    encoder.set_compute_pipeline_state(pipelines.stage2);
    encoder.set_input_array(partials, 0);
    encoder.set_input_array(sums, 1);
    encoder.set_input_array(maxs, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(partitions, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(q.shape(0) * q_heads, q_len, 1),
        MTL::Size(pipelines.plan.stage2_threads, 1, 1));
  }

  std::vector<array> vjp(const std::vector<array> &, const std::vector<array> &,
                         const std::vector<int> &,
                         const std::vector<array> &) override {
    throw std::runtime_error("SegmentedSdpa is inference-only");
  }

  DEFINE_INPUT_OUTPUT_SHAPE()
  DEFINE_NAME(SegmentedSdpa)

  bool is_equivalent(const Primitive &other) const override {
    const auto &o = static_cast<const SegmentedSdpa &>(other);
    return scale_ == o.scale_ && causal_ == o.causal_;
  }

private:
  float scale_;
  bool causal_;
};

array segmented_sdpa(const array &q, const array &prefix_k,
                     const array &prefix_v, const array &new_k,
                     const array &new_v, float scale, bool causal,
                     bool require_segmented) {
  auto stream = default_stream(Device::gpu);
  std::vector<array> inputs = {q, prefix_k, prefix_v, new_k, new_v};
  validate_segmented_sdpa(inputs);
  auto fallback = [&]() {
    return segmented_fallback(inputs, scale, causal, stream)[0];
  };
  try {
    auto &device = metal::device(stream.device);
    auto pipelines =
        get_pipelines(device, q.shape(2), q.shape(1) / prefix_k.shape(1),
                      prefix_k.shape(2) + new_k.shape(2), q.shape(1),
                      prefix_k.shape(1), causal);
    if (!pipelines.plan.supported) {
      if (require_segmented) {
        throw std::runtime_error(
            "segmented SDPA was required but the launch is unsupported");
      }
      return fallback();
    }
  } catch (const std::exception &) {
    if (require_segmented) {
      throw;
    }
    return fallback();
  }
  auto primitive = std::make_shared<SegmentedSdpa>(stream, scale, causal);
  return array(q.shape(), bfloat16, primitive, std::move(inputs));
}

} // namespace
} // namespace mlx::core::fast

namespace {

mlx_array *segmented_sdpa_forward_impl(mlx_array *q, mlx_array *prefix_k,
                           mlx_array *prefix_v, mlx_array *new_k,
                           mlx_array *new_v, float scale, bool causal,
                           bool require_segmented) {
  try {
    auto result = mlx::core::fast::segmented_sdpa(
        *reinterpret_cast<array *>(q), *reinterpret_cast<array *>(prefix_k),
        *reinterpret_cast<array *>(prefix_v), *reinterpret_cast<array *>(new_k),
        *reinterpret_cast<array *>(new_v), scale, causal, require_segmented);
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_segmented_sdpa_forward: %s\n", e.what());
    return nullptr;
  }
}

} // namespace

extern "C" mlx_array *
mlx_segmented_sdpa_forward(mlx_array *q, mlx_array *prefix_k,
                           mlx_array *prefix_v, mlx_array *new_k,
                           mlx_array *new_v, float scale, bool causal) {
  return segmented_sdpa_forward_impl(q, prefix_k, prefix_v, new_k, new_v,
                                     scale, causal, false);
}

// Test-only contract: never silently qualify the concatenated fallback.
extern "C" mlx_array *
mlx_segmented_sdpa_test_forward(mlx_array *q, mlx_array *prefix_k,
                                mlx_array *prefix_v, mlx_array *new_k,
                                mlx_array *new_v, float scale, bool causal) {
  return segmented_sdpa_forward_impl(q, prefix_k, prefix_v, new_k, new_v,
                                     scale, causal, true);
}

extern "C" int mlx_segmented_sdpa_test_plan(
    int query_length, int gqa_factor, bool two_pass, int partitions,
    size_t stage1_width, size_t stage1_max_threads, size_t stage1_static_memory,
    size_t device_max_memory, size_t stage2_width, size_t stage2_max_threads,
    size_t stage2_static_memory, uint32_t *out_stage1_threads,
    uint32_t *out_stage2_threads) {
  using namespace mlx::core::fast;
  SegmentedSdpaCapabilities c1{stage1_width, stage1_max_threads,
                               stage1_static_memory, device_max_memory};
  SegmentedSdpaCapabilities c2{stage2_width, stage2_max_threads,
                               stage2_static_memory, device_max_memory};
  auto plan = plan_segmented_sdpa_launch(query_length, gqa_factor, two_pass,
                                         partitions, c1, &c2);
  if (out_stage1_threads) {
    *out_stage1_threads = plan.stage1_threads;
  }
  if (out_stage2_threads) {
    *out_stage2_threads = plan.stage2_threads;
  }
  return plan.supported ? 1 : 0;
}

extern "C" int mlx_segmented_sdpa_max_query_length(int gqa_factor) {
  if (gqa_factor < 1 || gqa_factor > 32) {
    return 0;
  }
  try {
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto &device = mlx::core::metal::device(stream.device);
    const bool causal = true;
    const int partitions = 64;
    mlx::core::metal::MTLFCList constants = {
        {&causal, MTL::DataType::DataTypeBool, 22},
        {&partitions, MTL::DataType::DataTypeInt, 26}};
    auto *pipeline = device.get_kernel(
        "sdpa_vector_segmented_2pass_1_bfloat16_t_256_256",
        "sdpa_vector_segmented_2pass_1_bfloat16_t_256_256_caps", constants);
    mlx::core::metal::MTLFCList one_pass_constants = {
        {&causal, MTL::DataType::DataTypeBool, 22}};
    auto *one_pass = device.get_kernel(
        "sdpa_vector_segmented_bfloat16_t_256_256",
        "sdpa_vector_segmented_bfloat16_t_256_256_caps", one_pass_constants);
    auto *reduction = device.get_kernel("sdpa_vector_2pass_2_bfloat16_t_256");
    const auto stage1 = mlx::core::fast::segmented_sdpa_capabilities(
        pipeline, device.mtl_device());
    const auto one_pass_caps = mlx::core::fast::segmented_sdpa_capabilities(
        one_pass, device.mtl_device());
    const auto stage2 = mlx::core::fast::segmented_sdpa_capabilities(
        reduction, device.mtl_device());
    // Query length changes the actual threadgroup shape even though Metal
    // reuses one function specialization. Walk down from the workload bound
    // and ask the same launch planner used by eval_gpu for every candidate.
    for (int query_length = 8; query_length >= 1; --query_length) {
      auto plan = mlx::core::fast::plan_segmented_sdpa_launch(
          query_length, gqa_factor, true, partitions, stage1, &stage2);
      auto short_plan = mlx::core::fast::plan_segmented_sdpa_launch(
          query_length, gqa_factor, false, 32, one_pass_caps, nullptr);
      if (plan.supported && short_plan.supported) {
        return query_length;
      }
    }
    return 0;
  } catch (const std::exception &) {
    return 0;
  }
}

#else

extern "C" mlx_array *mlx_segmented_sdpa_forward(mlx_array *, mlx_array *,
                                                 mlx_array *, mlx_array *,
                                                 mlx_array *, float, bool) {
  return nullptr;
}

extern "C" mlx_array *mlx_segmented_sdpa_test_forward(
    mlx_array *, mlx_array *, mlx_array *, mlx_array *, mlx_array *, float, bool) {
  return nullptr;
}

extern "C" int mlx_segmented_sdpa_test_plan(
    int query_length, int gqa_factor, bool two_pass, int partitions,
    size_t stage1_width, size_t stage1_max_threads,
    size_t stage1_static_memory, size_t device_max_memory, size_t stage2_width,
    size_t stage2_max_threads, size_t stage2_static_memory,
    uint32_t *out_stage1_threads, uint32_t *out_stage2_threads) {
  mlx::core::fast::SegmentedSdpaCapabilities c1{
      stage1_width, stage1_max_threads, stage1_static_memory,
      device_max_memory};
  mlx::core::fast::SegmentedSdpaCapabilities c2{
      stage2_width, stage2_max_threads, stage2_static_memory,
      device_max_memory};
  auto plan = mlx::core::fast::plan_segmented_sdpa_launch(
      query_length, gqa_factor, two_pass, partitions, c1, &c2);
  if (out_stage1_threads) {
    *out_stage1_threads = plan.stage1_threads;
  }
  if (out_stage2_threads) {
    *out_stage2_threads = plan.stage2_threads;
  }
  return plan.supported ? 1 : 0;
}

extern "C" int mlx_segmented_sdpa_max_query_length(int) { return 0; }

#endif
