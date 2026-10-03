#include "mlx_common.h"
#include "mlx_segmented_sdpa_plan.h"

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

#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "mlx_segmented_sdpa.h"
#include "mlx_test_counters.h"

namespace mlx::core::segmented_sdpa {

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
  return {fast::scaled_dot_product_attention(
      inputs[0], keys, values, scale, causal ? "causal" : "", std::nullopt,
      std::nullopt, stream)};
}

SegmentedSdpaCapabilities capabilities(MTL::ComputePipelineState *pipeline,
                                       MTL::Device *device) {
  return {pipeline->threadExecutionWidth(),
          pipeline->maxTotalThreadsPerThreadgroup(),
          pipeline->staticThreadgroupMemoryLength(),
          device->maxThreadgroupMemoryLength()};
}

const char *kSegmentedSource =
#include "metal/common/sdpa_segmented.metal.inc"
    ;

const char *kernel_name(SegmentedKernel kernel) {
  switch (kernel) {
  case SegmentedKernel::one_pass:
    return "mlx_node_sdpa_segmented_bf16_256";
  case SegmentedKernel::two_pass_1:
    return "mlx_node_sdpa_segmented_2pass_1_bf16_256";
  case SegmentedKernel::verify_two_pass_1:
    return "mlx_node_sdpa_segmented_verify_2pass_1_bf16_256";
  }
  throw std::invalid_argument("unknown segmented SDPA kernel");
}

MTL::ComputePipelineState *segmented_kernel(metal::Device &device,
                                            SegmentedKernel kernel,
                                            const std::string &hash,
                                            const metal::MTLFCList &constants) {
  if (testing::kernel_override) {
    return testing::kernel_override(device, kernel, hash, constants);
  }
  auto *lib = device.get_library("mlx_node_sdpa_segmented", [] {
    std::string source(kSegmentedSource);
    for (auto [kernel, fn] :
         {std::pair{SegmentedKernel::one_pass, "sdpa_vector_segmented"},
          std::pair{SegmentedKernel::two_pass_1,
                    "sdpa_vector_segmented_2pass_1"},
          std::pair{SegmentedKernel::verify_two_pass_1,
                    "sdpa_vector_segmented_verify_2pass_1"}}) {
      const std::string instance = std::string(fn) + "<bfloat, 256, 256>";
      source += "\ntemplate [[host_name(\"" + std::string(kernel_name(kernel)) +
                "\")]] [[kernel]] decltype(" + instance + ") " + instance +
                ";\n";
    }
    return source;
  });
  return device.get_kernel(kernel_name(kernel), lib, hash, constants);
}

// MLX's own aggregation kernel: the partials must reduce exactly as MLX's
// vector SDPA reduces them.
MTL::ComputePipelineState *reduction_kernel(metal::Device &device) {
  return device.get_kernel("sdpa_vector_2pass_2_bfloat16_t_256");
}

int64_t prefix_row_stride(const array &pk, const array &pv) {
  return std::max(pk.strides(2), pv.strides(2));
}

struct Pipelines {
  MTL::ComputePipelineState *stage1;
  MTL::ComputePipelineState *stage2;
  SegmentedSdpaLaunchPlan plan;
};

Pipelines get_pipelines(metal::Device &device, int query_length, int gqa_factor,
                        int total_length, int q_heads, int kv_heads,
                        bool causal, int64_t prefix_row_stride) {
  const char device_class = device.get_architecture().back();
  const bool two_pass =
      sdpa_vector_uses_two_pass(device_class, total_length, q_heads, kv_heads);
  const int partitions =
      two_pass ? sdpa_vector_partition_count(device_class, total_length,
                                             gqa_factor * query_length,
                                             env::get_var("MLX_SDPA_BLOCKS", 0))
               : 32;
  metal::MTLFCList constants = {{&causal, MTL::DataType::DataTypeBool, 22}};
  if (two_pass) {
    constants.emplace_back(&partitions, MTL::DataType::DataTypeInt, 26);
  }
  const auto kernel =
      two_pass ? SegmentedKernel::two_pass_1 : SegmentedKernel::one_pass;
  std::string hash = std::string(kernel_name(kernel)) +
                     (causal ? "_c_" : "_nc_") + std::to_string(partitions);
  auto *stage1 = segmented_kernel(device, kernel, hash, constants);
  MTL::ComputePipelineState *stage2 = nullptr;
  if (two_pass) {
    stage2 = reduction_kernel(device);
  }
  auto c1 = capabilities(stage1, device.mtl_device());
  std::optional<SegmentedSdpaCapabilities> c2;
  if (stage2 != nullptr) {
    c2 = capabilities(stage2, device.mtl_device());
  }
  auto plan = plan_segmented_sdpa_launch(query_length, gqa_factor, two_pass,
                                         partitions, c1, c2 ? &*c2 : nullptr);
  // The kernels advance through prefix rows with a 32-bit pointer step.
  if (int64_t{partitions} * prefix_row_stride >
      std::numeric_limits<int>::max()) {
    plan.supported = false;
  }
  return {stage1, stage2, plan};
}

} // namespace

int segmented_max_query_length(metal::Device &device, int gqa_factor) {
  if (gqa_factor < 1 || gqa_factor > 32) {
    return 0;
  }
  const bool causal = true;
  const int partitions = 64;
  metal::MTLFCList constants = {{&causal, MTL::DataType::DataTypeBool, 22},
                                {&partitions, MTL::DataType::DataTypeInt, 26}};
  auto *pipeline = segmented_kernel(
      device, SegmentedKernel::two_pass_1,
      std::string(kernel_name(SegmentedKernel::two_pass_1)) + "_caps",
      constants);
  metal::MTLFCList one_pass_constants = {
      {&causal, MTL::DataType::DataTypeBool, 22}};
  auto *one_pass = segmented_kernel(
      device, SegmentedKernel::one_pass,
      std::string(kernel_name(SegmentedKernel::one_pass)) + "_caps",
      one_pass_constants);
  auto *reduction = reduction_kernel(device);
  const auto stage1 = capabilities(pipeline, device.mtl_device());
  const auto one_pass_caps = capabilities(one_pass, device.mtl_device());
  const auto stage2 = capabilities(reduction, device.mtl_device());
  // Query length changes the actual threadgroup shape even though Metal
  // reuses one function specialization. Walk down from the workload bound
  // and ask the same launch planner used by eval_gpu for every candidate.
  for (int query_length = 8; query_length >= 1; --query_length) {
    auto plan = plan_segmented_sdpa_launch(query_length, gqa_factor, true,
                                           partitions, stage1, &stage2);
    auto short_plan = plan_segmented_sdpa_launch(
        query_length, gqa_factor, false, 32, one_pass_caps, nullptr);
    if (plan.supported && short_plan.supported) {
      return query_length;
    }
  }
  return 0;
}

namespace {

SegmentedSdpaReductionPlan reduction_plan(const Pipelines &pipelines) {
  return {pipelines.plan.two_pass, static_cast<int>(pipelines.plan.partitions)};
}

struct VerifyDispatch {
  SegmentedVerifyRoute route;
  Pipelines head;
  Pipelines tail;
  MTL::ComputePipelineState *unified;
  SegmentedSdpaLaunchPlan unified_plan;
};

// Rows [0, head_rows) see prefix + new[..head_rows]; rows [head_rows, rows)
// see prefix + all new rows: the chunks a two-call verify would dispatch.
VerifyDispatch plan_verify_dispatch(metal::Device &device, int head_rows,
                                    int rows, int gqa, int prefix_n,
                                    int q_heads, int kv_heads,
                                    int64_t prefix_row_stride) {
  const int tail_rows = rows - head_rows;
  auto head =
      get_pipelines(device, head_rows, gqa, prefix_n + head_rows, q_heads,
                    kv_heads, head_rows > 1, prefix_row_stride);
  auto tail = get_pipelines(device, tail_rows, gqa, prefix_n + rows, q_heads,
                            kv_heads, true, prefix_row_stride);
  if (!head.plan.supported || !tail.plan.supported) {
    throw std::runtime_error(
        "segmented SDPA verify chunk launch is unsupported");
  }
  const auto head_reduction = reduction_plan(head);
  const auto tail_reduction = reduction_plan(tail);
  MTL::ComputePipelineState *unified = nullptr;
  SegmentedSdpaLaunchPlan unified_plan{false, true, 0, 0, 0};
  if (head_reduction.two_pass && tail_reduction.two_pass &&
      head_reduction.partitions == tail_reduction.partitions) {
    const bool causal = true;
    const int partitions = tail_reduction.partitions;
    metal::MTLFCList constants = {{&causal, MTL::DataType::DataTypeBool, 22},
                                  {&partitions, MTL::DataType::DataTypeInt, 26},
                                  {&gqa, MTL::DataType::DataTypeInt, 27},
                                  {&rows, MTL::DataType::DataTypeInt, 28}};
    const std::string base = kernel_name(SegmentedKernel::verify_two_pass_1);
    unified =
        segmented_kernel(device, SegmentedKernel::verify_two_pass_1,
                         base + "_c_" + std::to_string(partitions) + "_g" +
                             std::to_string(gqa) + "_r" + std::to_string(rows),
                         constants);
    unified_plan = plan_segmented_verify_launch(
        rows, gqa, partitions, capabilities(unified, device.mtl_device()),
        capabilities(tail.stage2, device.mtl_device()));
  }
  const auto route = select_segmented_verify_route(
      head_rows, head_reduction, tail_reduction, unified_plan.supported);
  return {route, head, tail, unified, unified_plan};
}

std::vector<int64_t> segmented_strides(const array &q, const array &pk,
                                       const array &pv, const array &nk,
                                       const array &nv) {
  return {q.strides(0),  q.strides(1),  q.strides(2),  pk.strides(0),
          pk.strides(1), pk.strides(2), pv.strides(0), pv.strides(1),
          pv.strides(2), nk.strides(0), nk.strides(1), nk.strides(2),
          nv.strides(0), nv.strides(1), nv.strides(2)};
}

void add_reduction_temporaries(metal::CommandEncoder &encoder, int batch,
                               int q_heads, int q_rows, int partitions,
                               array &partials, array &sums, array &maxs) {
  partials = array({batch, q_heads, q_rows, partitions, kHeadDimension},
                   bfloat16, nullptr, {});
  sums = array({batch, q_heads, q_rows, partitions}, float32, nullptr, {});
  maxs = array({batch, q_heads, q_rows, partitions}, float32, nullptr, {});
  partials.set_data(allocator::malloc(partials.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  encoder.add_temporary(partials);
  encoder.add_temporary(sums);
  encoder.add_temporary(maxs);
}

void encode_reduction(metal::CommandEncoder &encoder,
                      const Pipelines &pipelines, const array &partials,
                      const array &sums, const array &maxs, array &out,
                      int batch, int q_heads, int q_rows) {
  const int partitions = pipelines.plan.partitions;
  bridge_testing::record("segmented_sdpa_2pass_2");
  encoder.set_compute_pipeline_state(pipelines.stage2);
  encoder.set_input_array(partials, 0);
  encoder.set_input_array(sums, 1);
  encoder.set_input_array(maxs, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(partitions, 4);
  encoder.dispatch_threadgroups(MTL::Size(batch * q_heads, q_rows, 1),
                                MTL::Size(pipelines.plan.stage2_threads, 1, 1));
}

// Queries [q_row0, q_row0 + q_rows) against prefix + new[..new_n]; `out`
// must already own its buffer and hold q_rows rows.
void encode_segmented_call(metal::CommandEncoder &encoder,
                           const Pipelines &pipelines, const array &q,
                           int q_row0, int q_rows, const array &pk,
                           const array &pv, const array &nk, const array &nv,
                           int new_n, float scale, array &out) {
  const int batch = q.shape(0);
  const int q_heads = q.shape(1);
  const int kv_heads = pk.shape(1);
  const int gqa = q_heads / kv_heads;
  const int prefix_n = pk.shape(2);
  encoder.set_compute_pipeline_state(pipelines.stage1);
  encoder.set_input_array(q, 0, int64_t(q_row0) * q.strides(2) * q.itemsize());
  encoder.set_input_array(pk, 1);
  encoder.set_input_array(pv, 2);
  encoder.set_input_array(nk, 3);
  encoder.set_input_array(nv, 4);
  const auto strides = segmented_strides(q, pk, pv, nk, nv);

  if (!pipelines.plan.two_pass) {
    bridge_testing::record("segmented_sdpa_one_pass");
    encoder.set_output_array(out, 5);
    encoder.set_bytes(gqa, 6);
    encoder.set_bytes(prefix_n, 7);
    encoder.set_bytes(new_n, 8);
    encoder.set_vector_bytes(strides, 9);
    encoder.set_bytes(scale, 10);
    encoder.set_bytes(q_heads, 11);
    encoder.dispatch_threadgroups(
        MTL::Size(batch * q_heads, q_rows, 1),
        MTL::Size(pipelines.plan.stage1_threads, 1, 1));
    return;
  }

  const int partitions = pipelines.plan.partitions;
  bridge_testing::record("segmented_sdpa_2pass_1");
  array partials({}, bfloat16, nullptr, {});
  array sums({}, float32, nullptr, {});
  array maxs({}, float32, nullptr, {});
  add_reduction_temporaries(encoder, batch, q_heads, q_rows, partitions,
                            partials, sums, maxs);
  encoder.set_output_array(partials, 5);
  encoder.set_output_array(sums, 6);
  encoder.set_output_array(maxs, 7);
  encoder.set_bytes(prefix_n, 8);
  encoder.set_bytes(new_n, 9);
  encoder.set_vector_bytes(strides, 10);
  encoder.set_bytes(scale, 11);
  encoder.dispatch_threadgroups(MTL::Size(kv_heads, batch, partitions),
                                MTL::Size(32, gqa, q_rows));
  encode_reduction(encoder, pipelines, partials, sums, maxs, out, batch,
                   q_heads, q_rows);
}

void encode_unified_verify(metal::CommandEncoder &encoder,
                           const VerifyDispatch &dispatch, const array &q,
                           const array &pk, const array &pv, const array &nk,
                           const array &nv, float scale, array &out) {
  const int batch = q.shape(0);
  const int q_heads = q.shape(1);
  const int rows = q.shape(2);
  const int kv_heads = pk.shape(1);
  const int prefix_n = pk.shape(2);
  const int new_n = nk.shape(2);
  const int partitions = dispatch.unified_plan.partitions;
  array partials({}, bfloat16, nullptr, {});
  array sums({}, float32, nullptr, {});
  array maxs({}, float32, nullptr, {});
  add_reduction_temporaries(encoder, batch, q_heads, rows, partitions, partials,
                            sums, maxs);
  bridge_testing::record("segmented_sdpa_verify_2pass_1");
  encoder.set_compute_pipeline_state(dispatch.unified);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(pk, 1);
  encoder.set_input_array(pv, 2);
  encoder.set_input_array(nk, 3);
  encoder.set_input_array(nv, 4);
  encoder.set_output_array(partials, 5);
  encoder.set_output_array(sums, 6);
  encoder.set_output_array(maxs, 7);
  encoder.set_bytes(prefix_n, 8);
  encoder.set_bytes(new_n, 9);
  encoder.set_vector_bytes(segmented_strides(q, pk, pv, nk, nv), 10);
  encoder.set_bytes(scale, 11);
  encoder.dispatch_threadgroups(
      MTL::Size(kv_heads, batch, partitions),
      MTL::Size(dispatch.unified_plan.stage1_threads, 1, 1));
  encode_reduction(encoder, dispatch.tail, partials, sums, maxs, out, batch,
                   q_heads, rows);
}

class SegmentedSdpa final : public fast::Custom {
public:
  SegmentedSdpa(Stream stream, float scale, bool causal, int head_rows)
      : Custom(stream,
               [scale, causal, stream](std::vector<array> inputs) {
                 return segmented_fallback(std::move(inputs), scale, causal,
                                           stream);
               }),
        scale_(scale), causal_(causal), head_rows_(head_rows) {}

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
    auto &encoder = metal::get_command_encoder(stream);
    const int q_len = q.shape(2);
    const int q_heads = q.shape(1);
    const int kv_heads = pk.shape(1);
    const int gqa = q_heads / kv_heads;
    const int prefix_n = pk.shape(2);
    const int new_n = nk.shape(2);
    auto &out = outputs[0];

    if (head_rows_ == 0) {
      bridge_testing::record("segmented_sdpa_route_single");
      auto pipelines =
          get_pipelines(device, q_len, gqa, prefix_n + new_n, q_heads, kv_heads,
                        causal_, prefix_row_stride(pk, pv));
      if (!pipelines.plan.supported) {
        throw std::runtime_error("segmented SDPA pipeline capabilities "
                                 "changed after graph construction");
      }
      out.set_data(allocator::malloc(out.nbytes()));
      encode_segmented_call(encoder, pipelines, q, 0, q_len, pk, pv, nk, nv,
                            new_n, scale_, out);
      return;
    }

    if (new_n != q_len) {
      throw std::runtime_error("segmented SDPA verify block shape changed "
                               "after graph construction");
    }
    const auto dispatch =
        plan_verify_dispatch(device, head_rows_, q_len, gqa, prefix_n, q_heads,
                             kv_heads, prefix_row_stride(pk, pv));
    switch (dispatch.route) {
    case SegmentedVerifyRoute::one_pass:
      bridge_testing::record("segmented_sdpa_route_one_pass");
      out.set_data(allocator::malloc(out.nbytes()));
      encode_segmented_call(encoder, dispatch.tail, q, 0, q_len, pk, pv, nk, nv,
                            new_n, scale_, out);
      return;
    case SegmentedVerifyRoute::unified:
      bridge_testing::record("segmented_sdpa_route_unified");
      out.set_data(allocator::malloc(out.nbytes()));
      encode_unified_verify(encoder, dispatch, q, pk, pv, nk, nv, scale_, out);
      return;
    case SegmentedVerifyRoute::split: {
      bridge_testing::record("segmented_sdpa_route_split");
      const int tail_rows = q_len - head_rows_;
      array head_out({q.shape(0), q_heads, head_rows_, kHeadDimension},
                     bfloat16, nullptr, {});
      array tail_out({q.shape(0), q_heads, tail_rows, kHeadDimension}, bfloat16,
                     nullptr, {});
      head_out.set_data(allocator::malloc(head_out.nbytes()));
      tail_out.set_data(allocator::malloc(tail_out.nbytes()));
      encoder.add_temporary(head_out);
      encoder.add_temporary(tail_out);
      encode_segmented_call(encoder, dispatch.head, q, 0, head_rows_, pk, pv,
                            nk, nv, head_rows_, scale_, head_out);
      encode_segmented_call(encoder, dispatch.tail, q, head_rows_, tail_rows,
                            pk, pv, nk, nv, new_n, scale_, tail_out);
      // Allocates `out`.
      concatenate_gpu({head_out, tail_out}, out, 2, stream);
      return;
    }
    case SegmentedVerifyRoute::single:
      break;
    }
    throw std::runtime_error("segmented SDPA verify route is invalid");
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
    return scale_ == o.scale_ && causal_ == o.causal_ &&
           head_rows_ == o.head_rows_;
  }

private:
  float scale_;
  bool causal_;
  // Rows of the leading chunk when the block is wider than one supported
  // query chunk; the dispatch is then chosen per real prefix length.
  int head_rows_;
};

} // namespace

array segmented_sdpa(const array &q, const array &prefix_k,
                     const array &prefix_v, const array &new_k,
                     const array &new_v, float scale, bool causal,
                     bool require_segmented) {
  auto stream = default_stream(Device::gpu);
  std::vector<array> inputs = {q, prefix_k, prefix_v, new_k, new_v};
  validate_segmented_sdpa(inputs);
  const int q_len = q.shape(2);
  const int q_heads = q.shape(1);
  const int kv_heads = prefix_k.shape(1);
  const int gqa = q_heads / kv_heads;
  auto fallback = [&]() {
    return segmented_fallback(inputs, scale, causal, stream)[0];
  };
  auto &device = metal::device(stream.device);
  int max_query_length = 0;
  try {
    max_query_length = segmented_max_query_length(device, gqa);
  } catch (const std::exception &) {
    if (require_segmented) {
      throw;
    }
    return fallback();
  }
  if (max_query_length >= 1 && q_len > max_query_length) {
    // No concat fallback: unfused SDPA over this block would bake the prefix
    // length into a compiled trace.
    const int head_rows = segmented_verify_head_rows(q_len, max_query_length);
    if (head_rows < 1 || !causal || new_k.shape(2) != q_len) {
      throw std::invalid_argument("unsupported segmented SDPA verify block");
    }
    plan_verify_dispatch(device, head_rows, q_len, gqa, prefix_k.shape(2),
                         q_heads, kv_heads,
                         prefix_row_stride(prefix_k, prefix_v));
    auto primitive =
        std::make_shared<SegmentedSdpa>(stream, scale, causal, head_rows);
    return array(q.shape(), bfloat16, primitive, std::move(inputs));
  }
  try {
    auto pipelines = get_pipelines(
        device, q_len, gqa, prefix_k.shape(2) + new_k.shape(2), q_heads,
        kv_heads, causal, prefix_row_stride(prefix_k, prefix_v));
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
  auto primitive = std::make_shared<SegmentedSdpa>(stream, scale, causal, 0);
  return array(q.shape(), bfloat16, primitive, std::move(inputs));
}

} // namespace mlx::core::segmented_sdpa

namespace {

mlx_array *segmented_sdpa_forward_impl(mlx_array *q, mlx_array *prefix_k,
                                       mlx_array *prefix_v, mlx_array *new_k,
                                       mlx_array *new_v, float scale,
                                       bool causal, bool require_segmented) {
  try {
    auto result = mlx::core::segmented_sdpa::segmented_sdpa(
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
  return segmented_sdpa_forward_impl(q, prefix_k, prefix_v, new_k, new_v, scale,
                                     causal, false);
}

// Test-only contract: never silently qualify the concatenated fallback.
extern "C" mlx_array *
mlx_segmented_sdpa_test_forward(mlx_array *q, mlx_array *prefix_k,
                                mlx_array *prefix_v, mlx_array *new_k,
                                mlx_array *new_v, float scale, bool causal) {
  return segmented_sdpa_forward_impl(q, prefix_k, prefix_v, new_k, new_v, scale,
                                     causal, true);
}

extern "C" int mlx_segmented_sdpa_max_query_length(int gqa_factor) {
  try {
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    return mlx::core::segmented_sdpa::segmented_max_query_length(
        mlx::core::metal::device(stream.device), gqa_factor);
  } catch (const std::exception &) {
    return 0;
  }
}

// Test-only: the verify route this device takes for a causal block of `rows`
// queries over `prefix_n` prefix rows; -1 when no segmented route exists.
extern "C" int mlx_segmented_sdpa_test_device_verify_route(
    int q_heads, int kv_heads, int rows, int prefix_n, char *out_device_class) {
  using namespace mlx::core;
  try {
    if (kv_heads < 1 || q_heads % kv_heads != 0) {
      return -1;
    }
    auto stream = default_stream(Device::gpu);
    auto &device = metal::device(stream.device);
    if (out_device_class) {
      *out_device_class = device.get_architecture().back();
    }
    const int gqa = q_heads / kv_heads;
    const int head_rows = segmented_sdpa::segmented_verify_head_rows(
        rows, segmented_sdpa::segmented_max_query_length(device, gqa));
    if (head_rows < 0) {
      return -1;
    }
    if (head_rows == 0) {
      return static_cast<int>(segmented_sdpa::SegmentedVerifyRoute::single);
    }
    // Contiguous [B, H, prefix, D] K/V rows.
    return static_cast<int>(segmented_sdpa::plan_verify_dispatch(
                                device, head_rows, rows, gqa, prefix_n, q_heads,
                                kv_heads, segmented_sdpa::kHeadDimension)
                                .route);
  } catch (const std::exception &) {
    return -1;
  }
}

#else

extern "C" mlx_array *mlx_segmented_sdpa_forward(mlx_array *, mlx_array *,
                                                 mlx_array *, mlx_array *,
                                                 mlx_array *, float, bool) {
  return nullptr;
}

extern "C" mlx_array *mlx_segmented_sdpa_test_forward(mlx_array *, mlx_array *,
                                                      mlx_array *, mlx_array *,
                                                      mlx_array *, float,
                                                      bool) {
  return nullptr;
}

extern "C" int mlx_segmented_sdpa_max_query_length(int) { return 0; }

extern "C" int mlx_segmented_sdpa_test_device_verify_route(int, int, int, int,
                                                           char *) {
  return -1;
}

#endif

extern "C" int mlx_segmented_sdpa_test_plan(
    int query_length, int gqa_factor, bool two_pass, int partitions,
    size_t stage1_width, size_t stage1_max_threads, size_t stage1_static_memory,
    size_t device_max_memory, size_t stage2_width, size_t stage2_max_threads,
    size_t stage2_static_memory, uint32_t *out_stage1_threads,
    uint32_t *out_stage2_threads) {
  using namespace mlx::core::segmented_sdpa;
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

// Test-only, platform independent: the route for a block of `rows` queries
// whose two chunks would reduce as given; -1 when two chunks cannot cover it.
extern "C" int mlx_segmented_sdpa_test_verify_route(
    int rows, int max_query_length, bool head_two_pass, int head_partitions,
    bool tail_two_pass, int tail_partitions, bool unified_supported) {
  using namespace mlx::core::segmented_sdpa;
  const int head_rows = segmented_verify_head_rows(rows, max_query_length);
  if (head_rows < 0) {
    return -1;
  }
  return static_cast<int>(select_segmented_verify_route(
      head_rows, {head_two_pass, head_partitions},
      {tail_two_pass, tail_partitions}, unified_supported));
}

extern "C" int mlx_segmented_sdpa_test_verify_plan(
    int rows, int gqa_factor, int partitions, size_t stage1_width,
    size_t stage1_max_threads, size_t stage1_static_memory,
    size_t device_max_memory, size_t stage2_width, size_t stage2_max_threads,
    size_t stage2_static_memory, uint32_t *out_stage1_threads) {
  using namespace mlx::core::segmented_sdpa;
  SegmentedSdpaCapabilities c1{stage1_width, stage1_max_threads,
                               stage1_static_memory, device_max_memory};
  SegmentedSdpaCapabilities c2{stage2_width, stage2_max_threads,
                               stage2_static_memory, device_max_memory};
  auto plan =
      plan_segmented_verify_launch(rows, gqa_factor, partitions, c1, c2);
  if (out_stage1_threads) {
    *out_stage1_threads = plan.stage1_threads;
  }
  return plan.supported ? 1 : 0;
}
