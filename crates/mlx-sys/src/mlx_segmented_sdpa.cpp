#include "mlx_common.h"
#include "mlx_segmented_sdpa_plan.h"

#include <cstdint>
#include <vector>

#ifdef MLX_NODE_METAL_ENABLED

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
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
#include "mlx_paged_metallib.h"
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
      std::nullopt, false, stream)};
}

SegmentedSdpaCapabilities capabilities(MTL::ComputePipelineState *pipeline,
                                       MTL::Device *device) {
  return {pipeline->threadExecutionWidth(),
          pipeline->maxTotalThreadsPerThreadgroup(),
          pipeline->staticThreadgroupMemoryLength(),
          device->maxThreadgroupMemoryLength()};
}

// Tile sizes plan_segmented_verify_tile_launch can return.
constexpr int kTileSizes[] = {16, 32};
// Tile sizes segmented_nax_tile_n can return, and the M values
// sdpa_segmented_nax.metal instantiates (segmented_nax_m_supported).
constexpr int kNaxTileSizes[] = {32, 64};
constexpr int kNaxMs[] = {8, 16, 24, 32, 40, 48, 56, 64};

// The function name of a kernel; the tensor-op kernel carries its template
// shape (M, tile_n) in the name, the others are specialized by function
// constants.
std::string kernel_name(SegmentedKernel kernel, int m = 0, int tile_n = 0) {
  switch (kernel) {
  case SegmentedKernel::one_pass:
    return "mlx_node_sdpa_segmented_bf16_256";
  case SegmentedKernel::two_pass_1:
    return "mlx_node_sdpa_segmented_2pass_1_bf16_256";
  case SegmentedKernel::verify_two_pass_1:
    return "mlx_node_sdpa_segmented_verify_2pass_1_bf16_256";
  case SegmentedKernel::verify_tile_two_pass_1:
    return "mlx_node_sdpa_segmented_verify_tile_2pass_1_bf16_256";
  case SegmentedKernel::verify_nax_two_pass_1:
    return "mlx_node_sdpa_segmented_verify_nax_2pass_1_bf16_256_m" +
           std::to_string(m) + "_n" + std::to_string(tile_n);
  }
  throw std::invalid_argument("unknown segmented SDPA kernel");
}

// Key count (prefix + new rows) from which a verify block takes a block
// kernel, measured once per process by calibrate_block_min_keys; 0 until
// then (the vector routes serve every block).
std::atomic<int> g_block_min_keys{0};

// MLX_SDPA_VERIFY_TILE: unset (-1) = the calibrated crossover; 0 keeps the
// vector routes (A/B kill switch); N >= 1 takes a block kernel from N keys.
// Read per call.
int block_min_keys_override() {
  return env::get_var("MLX_SDPA_VERIFY_TILE", -1);
}

int effective_block_min_keys() {
  const int override = block_min_keys_override();
  return override >= 0 ? override
                       : g_block_min_keys.load(std::memory_order_acquire);
}

bool tile_route_enabled(SegmentedTileMode mode, int total_length) {
  switch (mode) {
  case SegmentedTileMode::vector:
    return false;
  case SegmentedTileMode::tile:
  case SegmentedTileMode::nax:
    return true;
  case SegmentedTileMode::from_env:
    break;
  }
  const int min_keys = effective_block_min_keys();
  return min_keys > 0 && total_length >= min_keys;
}

// MLX_SDPA_ROUTE_LOG=1 (or MLX_METAL_COMMAND_TRACE >= 1) prints the
// calibration result once.
bool route_log_enabled() {
  static const bool enabled = env::get_var("MLX_SDPA_ROUTE_LOG", 0) == 1 ||
                              env::get_var("MLX_METAL_COMMAND_TRACE", 0) >= 1;
  return enabled;
}

// MLX_SDPA_VERIFY_NAX=0 keeps the simdgroup-matrix tile kernel on a device
// whose tensor op could serve the block (A/B kill switch). Read per call.
bool nax_route_enabled(SegmentedTileMode mode) {
  switch (mode) {
  case SegmentedTileMode::vector:
  case SegmentedTileMode::tile:
    return false;
  case SegmentedTileMode::nax:
    return true;
  case SegmentedTileMode::from_env:
    break;
  }
  return metal::is_nax_available() &&
         env::get_var("MLX_SDPA_VERIFY_NAX", 1) != 0;
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
  const auto kernel =
      two_pass ? SegmentedKernel::two_pass_1 : SegmentedKernel::one_pass;
  auto *stage1 =
      segmented_kernel(device, {kernel, causal, partitions, 0, 0, 0, 0});
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

MTL::ComputePipelineState *
segmented_kernel(metal::Device &device,
                 const SegmentedSpecialization &specialization) {
  const auto &sp = specialization;
  const std::string name = kernel_name(sp.kernel, sp.m, sp.tile_n);
  std::string hash = name;
  metal::MTLFCList constants;
  if (sp.kernel == SegmentedKernel::verify_nax_two_pass_1) {
    // The whole shape is in the function name.
    return fast::paged::get_prebuilt_kernel(device, "segmented_sdpa",
                                            "segmented SDPA", name);
  }
  if (sp.kernel == SegmentedKernel::verify_tile_two_pass_1) {
    // The tile kernel takes its shape from buffers; only the tile size is
    // compiled in (index 29).
    constants.emplace_back(&sp.tile_n, MTL::DataType::DataTypeInt, 29);
    hash += "_t" + std::to_string(sp.tile_n);
    return fast::paged::get_prebuilt_kernel(
        device, "segmented_sdpa", "segmented SDPA", name, hash, constants);
  }
  // Indices 22 and 26 are MLX's sdpa_vector.h do_causal and blocks.
  constants.emplace_back(&sp.causal, MTL::DataType::DataTypeBool, 22);
  hash += sp.causal ? "_c" : "_nc";
  if (sp.kernel != SegmentedKernel::one_pass) {
    constants.emplace_back(&sp.partitions, MTL::DataType::DataTypeInt, 26);
    hash += "_" + std::to_string(sp.partitions);
  }
  if (sp.kernel == SegmentedKernel::verify_two_pass_1) {
    constants.emplace_back(&sp.gqa, MTL::DataType::DataTypeInt, 27);
    constants.emplace_back(&sp.rows, MTL::DataType::DataTypeInt, 28);
    hash += "_g" + std::to_string(sp.gqa) + "_r" + std::to_string(sp.rows);
  }
  return fast::paged::get_prebuilt_kernel(
      device, "segmented_sdpa", "segmented SDPA", name, hash, constants);
}

std::vector<std::string> metal_kernel_names() {
  std::vector<std::string> names;
  for (auto kernel :
       {SegmentedKernel::one_pass, SegmentedKernel::two_pass_1,
        SegmentedKernel::verify_two_pass_1,
        SegmentedKernel::verify_tile_two_pass_1}) {
    names.push_back(kernel_name(kernel));
  }
  for (int m : kNaxMs) {
    for (int tile_n : kNaxTileSizes) {
      names.push_back(
          kernel_name(SegmentedKernel::verify_nax_two_pass_1, m, tile_n));
    }
  }
  return names;
}

std::vector<SegmentedSpecialization> metal_kernel_specializations() {
  // Every partition count the vector-SDPA policy returns without
  // MLX_SDPA_BLOCKS, over the boundaries of every device class.
  std::set<int> partitions;
  for (char device_class : {'s', 'd', 'g'}) {
    for (int length :
         {1, 1024, 1025, 8192, 8193, 16384, 32768, 32769, 65536, 65537}) {
      for (int simdgroups = 1; simdgroups <= 32 * 8; ++simdgroups) {
        partitions.insert(
            sdpa_vector_partition_count(device_class, length, simdgroups, 0));
      }
    }
  }
  std::vector<SegmentedSpecialization> out;
  for (bool causal : {false, true}) {
    out.push_back({SegmentedKernel::one_pass, causal, 32, 0, 0, 0, 0});
    for (int p : partitions) {
      out.push_back({SegmentedKernel::two_pass_1, causal, p, 0, 0, 0, 0});
    }
  }
  for (int p : partitions) {
    for (int gqa = 1; gqa <= 32; ++gqa) {
      for (int rows = 2; rows <= 8; ++rows) {
        out.push_back(
            {SegmentedKernel::verify_two_pass_1, true, p, gqa, rows, 0, 0});
      }
    }
  }
  for (int tile_n : kTileSizes) {
    out.push_back(
        {SegmentedKernel::verify_tile_two_pass_1, true, 0, 0, 0, tile_n, 0});
  }
  // The tensor-op pipelines build only where the op exists.
  if (metal::is_nax_available()) {
    for (int m : kNaxMs) {
      for (int tile_n : kNaxTileSizes) {
        out.push_back(
            {SegmentedKernel::verify_nax_two_pass_1, true, 0, 0, 0, tile_n, m});
      }
    }
  }
  return out;
}

int segmented_max_query_length(metal::Device &device, int gqa_factor) {
  if (gqa_factor < 1 || gqa_factor > 32) {
    return 0;
  }
  const int partitions = 64;
  auto *pipeline = segmented_kernel(
      device, {SegmentedKernel::two_pass_1, true, partitions, 0, 0, 0, 0});
  auto *one_pass = segmented_kernel(
      device, {SegmentedKernel::one_pass, true, 32, 0, 0, 0, 0});
  // Loaded here only so a metallib without the block kernels is a packaging
  // error (PrebuiltKernelMissing); the width below does not depend on them.
  segmented_kernel(device, {SegmentedKernel::verify_tile_two_pass_1, true, 0, 0,
                            0, kTileSizes[1], 0});
  if (metal::is_nax_available()) {
    segmented_kernel(device, {SegmentedKernel::verify_nax_two_pass_1, true, 0,
                              0, 0, kNaxTileSizes[1], 48});
  }
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

struct UnifiedVerifyLaunch {
  MTL::ComputePipelineState *pipeline;
  SegmentedSdpaLaunchPlan plan;
};

// Register pressure sets the verify pipeline's thread limit, so whether one
// dispatch can serve the block depends on the GPU, not only on the policy.
UnifiedVerifyLaunch unified_verify_launch(metal::Device &device, int rows,
                                          int gqa, int partitions) {
  auto *pipeline = segmented_kernel(
      device,
      {SegmentedKernel::verify_two_pass_1, true, partitions, gqa, rows, 0, 0});
  return {pipeline,
          plan_segmented_verify_launch(
              rows, gqa, partitions,
              capabilities(pipeline, device.mtl_device()),
              capabilities(reduction_kernel(device), device.mtl_device()))};
}

struct TileVerifyLaunch {
  MTL::ComputePipelineState *stage1;
  MTL::ComputePipelineState *stage2;
  SegmentedSdpaTilePlan plan;
  int partitions;
};

// The tile kernel reads K/V rows as 16-byte vectors and Q fragments as
// 4-byte pairs (buffers are page aligned; the view's byte offset and
// strides must keep 16-byte alignment).
bool tile_aligned(const array &a) {
  return a.offset() % 16 == 0 && (a.strides(0) * a.itemsize()) % 16 == 0 &&
         (a.strides(1) * a.itemsize()) % 16 == 0 &&
         (a.strides(2) * a.itemsize()) % 16 == 0;
}

// The simdgroup-matrix verify dispatch for a causal block of `rows` over
// prefix_n + new_n keys; `plan.supported` is false when the device, pipeline
// or shape rules it out. The tile size comes from the pipeline built at the
// device's tile size, so the capabilities and the function constant agree.
TileVerifyLaunch tile_verify_launch(metal::Device &device, int rows, int gqa,
                                    int total_length) {
  auto *mtl = device.mtl_device();
  auto *stage2 = reduction_kernel(device);
  const auto c2 = capabilities(stage2, mtl);
  // The tile size is a function constant, so it is chosen before the
  // pipeline exists: from the device limit alone (the kernel declares no
  // static threadgroup memory), then confirmed against the built pipeline.
  TileVerifyLaunch launch{nullptr, stage2, {false, 0, 0, 0, 0}, 0};
  if (rows < 1 || gqa < 1 || (int64_t{rows} * gqa) % 8 != 0 ||
      !mtl->supportsFamily(MTL::GPUFamilyApple7)) {
    return launch;
  }
  const int tile_n = static_cast<int>(segmented_verify_tile_n(
      c2.max_threadgroup_memory, 0, 2 * size_t(rows) * gqa / 8));
  if (tile_n == 0) {
    return launch;
  }
  launch.stage1 = segmented_kernel(
      device,
      {SegmentedKernel::verify_tile_two_pass_1, true, 0, 0, 0, tile_n, 0});
  launch.plan = plan_segmented_verify_tile_launch(
      rows, gqa, capabilities(launch.stage1, mtl), c2);
  if (launch.plan.supported && int(launch.plan.tile_n) != tile_n) {
    launch.plan.supported = false;
  }
  launch.partitions = segmented_verify_tile_partitions(
      total_length, tile_n, env::get_var("MLX_SDPA_BLOCKS", 0));
  return launch;
}

struct NaxVerifyLaunch {
  MTL::ComputePipelineState *stage1;
  MTL::ComputePipelineState *stage2;
  SegmentedSdpaNaxPlan plan;
  int partitions;
};

// The tensor-op kernel reads the M = gqa * rows queries of one KV head as
// one strided rank-2 tensor in head-major order (m = h * rows + r), which
// needs the row stride to continue across heads; every seq stride it passes
// as a tensor stride must fit an int.
bool nax_q_layout(const array &q, const array &pk, const array &pv,
                  const array &nk, const array &nv) {
  constexpr int64_t kMax = std::numeric_limits<int>::max();
  return q.strides(1) == int64_t{q.shape(2)} * q.strides(2) &&
         q.strides(2) <= kMax && pk.strides(2) <= kMax &&
         pv.strides(2) <= kMax && nk.strides(2) <= kMax &&
         nv.strides(2) <= kMax;
}

// The tensor-op verify dispatch for a causal block of `rows` over
// prefix_n + new_n keys; `plan.supported` is false when the device (no NAX),
// pipeline or shape rules it out. M and the tile size are in the function
// name, so both are chosen before the pipeline exists and confirmed against
// the built pipeline. Partitions follow the tile kernel's policy at this
// kernel's tile size.
NaxVerifyLaunch nax_verify_launch(metal::Device &device, int rows, int gqa,
                                  int total_length) {
  auto *mtl = device.mtl_device();
  auto *stage2 = reduction_kernel(device);
  const auto c2 = capabilities(stage2, mtl);
  NaxVerifyLaunch launch{nullptr, stage2, {false, 0, 0, 0, 0, 0}, 0};
  const int64_t m = int64_t{rows} * gqa;
  if (rows < 1 || gqa < 1 || !segmented_nax_m_supported(int(m)) ||
      !metal::is_nax_available()) {
    return launch;
  }
  const int tile_n = static_cast<int>(
      segmented_nax_tile_n(c2.max_threadgroup_memory, 0, uint32_t(m)));
  if (tile_n == 0) {
    return launch;
  }
  launch.stage1 = segmented_kernel(device,
                                   {SegmentedKernel::verify_nax_two_pass_1,
                                    true, 0, 0, 0, tile_n, int(m)});
  launch.plan = plan_segmented_verify_nax_launch(
      rows, gqa, capabilities(launch.stage1, mtl), c2);
  if (launch.plan.supported && int(launch.plan.tile_n) != tile_n) {
    launch.plan.supported = false;
  }
  launch.partitions = segmented_verify_tile_partitions(
      total_length, tile_n, env::get_var("MLX_SDPA_BLOCKS", 0));
  return launch;
}

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
    const auto launch =
        unified_verify_launch(device, rows, gqa, tail_reduction.partitions);
    unified = launch.pipeline;
    unified_plan = launch.plan;
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

// One block-kernel dispatch (simdgroup-matrix tile or tensor-op): the two
// kernels bind the same buffers and differ in pipeline, threads and scratch.
struct BlockVerifyDispatch {
  const char *counter;
  MTL::ComputePipelineState *stage1;
  MTL::ComputePipelineState *stage2;
  uint32_t stage1_threads;
  uint32_t stage2_threads;
  uint32_t threadgroup_bytes;
  int partitions;
};

BlockVerifyDispatch block_dispatch(const TileVerifyLaunch &launch) {
  return {"segmented_sdpa_verify_tile_2pass_1",
          launch.stage1,
          launch.stage2,
          launch.plan.stage1_threads,
          launch.plan.stage2_threads,
          launch.plan.threadgroup_bytes,
          launch.partitions};
}

BlockVerifyDispatch block_dispatch(const NaxVerifyLaunch &launch) {
  return {"segmented_sdpa_verify_nax_2pass_1",
          launch.stage1,
          launch.stage2,
          launch.plan.stage1_threads,
          launch.plan.stage2_threads,
          launch.plan.threadgroup_bytes,
          launch.partitions};
}

void encode_block_verify(metal::CommandEncoder &encoder,
                         const BlockVerifyDispatch &launch, const array &q,
                         const array &pk, const array &pv, const array &nk,
                         const array &nv, float scale, array &out) {
  const int batch = q.shape(0);
  const int q_heads = q.shape(1);
  const int rows = q.shape(2);
  const int kv_heads = pk.shape(1);
  const int gqa = q_heads / kv_heads;
  const int prefix_n = pk.shape(2);
  const int new_n = nk.shape(2);
  const int partitions = launch.partitions;
  array partials({}, bfloat16, nullptr, {});
  array sums({}, float32, nullptr, {});
  array maxs({}, float32, nullptr, {});
  add_reduction_temporaries(encoder, batch, q_heads, rows, partitions, partials,
                            sums, maxs);
  bridge_testing::record(launch.counter);
  encoder.set_compute_pipeline_state(launch.stage1);
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
  encoder.set_bytes(gqa, 12);
  encoder.set_bytes(rows, 13);
  encoder.set_bytes(partitions, 14);
  encoder.set_threadgroup_memory_length(launch.threadgroup_bytes, 0);
  encoder.dispatch_threadgroups(MTL::Size(kv_heads, batch, partitions),
                                MTL::Size(launch.stage1_threads, 1, 1));
  bridge_testing::record("segmented_sdpa_2pass_2");
  encoder.set_compute_pipeline_state(launch.stage2);
  encoder.set_input_array(partials, 0);
  encoder.set_input_array(sums, 1);
  encoder.set_input_array(maxs, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(partitions, 4);
  encoder.dispatch_threadgroups(MTL::Size(batch * q_heads, rows, 1),
                                MTL::Size(launch.stage2_threads, 1, 1));
}

// The dispatcher: encodes the verify block or chunked segmented call for
// `inputs` into `encoder`, allocating `out`. eval_gpu and the crossover
// calibration share it, so the calibration times exactly the production
// dispatch of each route.
void encode_segmented_sdpa(metal::Device &device,
                           metal::CommandEncoder &encoder, Stream stream,
                           const std::vector<array> &inputs, array &out,
                           float scale, bool causal, int head_rows,
                           SegmentedTileMode tile_mode) {
  validate_segmented_sdpa(inputs);
  const auto &q = inputs[0];
  const auto &pk = inputs[1];
  const auto &pv = inputs[2];
  const auto &nk = inputs[3];
  const auto &nv = inputs[4];
  const int q_len = q.shape(2);
  const int q_heads = q.shape(1);
  const int kv_heads = pk.shape(1);
  const int gqa = q_heads / kv_heads;
  const int prefix_n = pk.shape(2);
  const int new_n = nk.shape(2);

  // A verify block (causal, one new row per query) takes a block kernel
  // whenever this device can launch one: the tensor-op kernel where NAX
  // exists and Q is head-major contiguous, else the simdgroup-matrix tile
  // kernel. The partition count follows the real prefix, so a shapeless
  // replay stays valid as the prefix grows.
  if (causal && new_n == q_len &&
      tile_route_enabled(tile_mode, prefix_n + new_n) && tile_aligned(q) &&
      tile_aligned(pk) && tile_aligned(pv) && tile_aligned(nk) &&
      tile_aligned(nv)) {
    if (nax_route_enabled(tile_mode) && nax_q_layout(q, pk, pv, nk, nv)) {
      const auto launch =
          nax_verify_launch(device, q_len, gqa, prefix_n + new_n);
      if (launch.plan.supported) {
        bridge_testing::record("segmented_sdpa_route_nax");
        out.set_data(allocator::malloc(out.nbytes()));
        encode_block_verify(encoder, block_dispatch(launch), q, pk, pv, nk, nv,
                            scale, out);
        return;
      }
    }
    if (tile_mode != SegmentedTileMode::nax) {
      const auto launch =
          tile_verify_launch(device, q_len, gqa, prefix_n + new_n);
      if (launch.plan.supported) {
        bridge_testing::record("segmented_sdpa_route_tile");
        out.set_data(allocator::malloc(out.nbytes()));
        encode_block_verify(encoder, block_dispatch(launch), q, pk, pv, nk, nv,
                            scale, out);
        return;
      }
    }
  }
  if (tile_mode == SegmentedTileMode::tile) {
    throw std::runtime_error(
        "segmented SDPA tile route was required but is unsupported");
  }
  if (tile_mode == SegmentedTileMode::nax) {
    throw std::runtime_error(
        "segmented SDPA tensor-op route was required but is unsupported");
  }

  if (head_rows == 0) {
    bridge_testing::record("segmented_sdpa_route_single");
    auto pipelines =
        get_pipelines(device, q_len, gqa, prefix_n + new_n, q_heads, kv_heads,
                      causal, prefix_row_stride(pk, pv));
    if (!pipelines.plan.supported) {
      throw std::runtime_error("segmented SDPA pipeline capabilities "
                               "changed after graph construction");
    }
    out.set_data(allocator::malloc(out.nbytes()));
    encode_segmented_call(encoder, pipelines, q, 0, q_len, pk, pv, nk, nv,
                          new_n, scale, out);
    return;
  }

  if (new_n != q_len) {
    throw std::runtime_error("segmented SDPA verify block shape changed "
                             "after graph construction");
  }
  const auto dispatch =
      plan_verify_dispatch(device, head_rows, q_len, gqa, prefix_n, q_heads,
                           kv_heads, prefix_row_stride(pk, pv));
  switch (dispatch.route) {
  case SegmentedVerifyRoute::one_pass:
    bridge_testing::record("segmented_sdpa_route_one_pass");
    out.set_data(allocator::malloc(out.nbytes()));
    encode_segmented_call(encoder, dispatch.tail, q, 0, q_len, pk, pv, nk, nv,
                          new_n, scale, out);
    return;
  case SegmentedVerifyRoute::unified:
    bridge_testing::record("segmented_sdpa_route_unified");
    out.set_data(allocator::malloc(out.nbytes()));
    encode_unified_verify(encoder, dispatch, q, pk, pv, nk, nv, scale, out);
    return;
  case SegmentedVerifyRoute::split: {
    bridge_testing::record("segmented_sdpa_route_split");
    const int tail_rows = q_len - head_rows;
    array head_out({q.shape(0), q_heads, head_rows, kHeadDimension}, bfloat16,
                   nullptr, {});
    array tail_out({q.shape(0), q_heads, tail_rows, kHeadDimension}, bfloat16,
                   nullptr, {});
    head_out.set_data(allocator::malloc(head_out.nbytes()));
    tail_out.set_data(allocator::malloc(tail_out.nbytes()));
    encoder.add_temporary(head_out);
    encoder.add_temporary(tail_out);
    encode_segmented_call(encoder, dispatch.head, q, 0, head_rows, pk, pv, nk,
                          nv, head_rows, scale, head_out);
    encode_segmented_call(encoder, dispatch.tail, q, head_rows, tail_rows, pk,
                          pv, nk, nv, new_n, scale, tail_out);
    // Allocates `out`.
    concatenate_gpu({head_out, tail_out}, out, 2, stream);
    return;
  }
  case SegmentedVerifyRoute::single:
    break;
  }
  throw std::runtime_error("segmented SDPA verify route is invalid");
}

// Crossover calibration: the production verify shape class (B = 1, rows 8,
// 24 / 4 heads, D = 256, BF16) at each kSegmentedCalibrationKeys count, the
// vector route against the best block kernel this device launches (tensor-op
// where it exists, else the tile kernel). Each sample is one command buffer
// holding exactly one route's dispatches, timed by the GPU's own start / end
// stamps, so host scheduling is outside the measurement; the two routes
// alternate and the order flips each repetition so a clock-state change
// between samples cannot favour one of them, and the statistic is the
// minimum of kCalibrationReps.
constexpr int kCalibrationReps = 7;
constexpr int kCalibrationWarmupPairs = 6;
constexpr int kCalibrationRows = 8;
constexpr int kCalibrationQHeads = 24;
constexpr int kCalibrationKvHeads = 4;

enum class CalibrationBlockKernel : int { none = 0, tile = 1, nax = 2 };

struct BlockCalibration {
  std::once_flag once;
  int points = 0;
  int keys[kSegmentedCalibrationPoints] = {};
  double vector_seconds[kSegmentedCalibrationPoints] = {};
  double block_seconds[kSegmentedCalibrationPoints] = {};
  double elapsed_ms = 0.0;
  // Fastest host encode of one dispatch per route (diagnostic).
  double host_vector_seconds = 0.0;
  double host_block_seconds = 0.0;
  CalibrationBlockKernel block_kernel = CalibrationBlockKernel::none;
  std::string error;
  int result = 0;
};

BlockCalibration &block_calibration() {
  static auto *calibration = new BlockCalibration;
  return *calibration;
}

// GPU seconds of one route over `inputs`: the command buffer the encoder
// holds receives only this route's dispatches and is committed and awaited
// here. `host_seconds` receives the host encode time (diagnostic).
double time_segmented_route(metal::Device &device, Stream stream,
                            const std::vector<array> &inputs, int head_rows,
                            SegmentedTileMode mode, double *host_seconds) {
  auto &encoder = metal::get_command_encoder(stream);
  auto buffer = NS::RetainPtr(encoder.get_command_buffer());
  array out(inputs[0].shape(), bfloat16, nullptr, {});
  const auto host_start = std::chrono::steady_clock::now();
  encode_segmented_sdpa(device, encoder, stream, inputs, out, 0.0625f, true,
                        head_rows, mode);
  if (host_seconds) {
    *host_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - host_start)
                        .count();
  }
  encoder.synchronize();
  return buffer->GPUEndTime() - buffer->GPUStartTime();
}

// Fills `c` (throws on any failure; the caller records the fallback).
void measure_block_crossover(metal::Device &device, Stream stream,
                             BlockCalibration &c) {
  constexpr int gqa = kCalibrationQHeads / kCalibrationKvHeads;
  const int head_rows = segmented_verify_head_rows(
      kCalibrationRows, segmented_max_query_length(device, gqa));
  if (head_rows < 0) {
    throw std::runtime_error("the calibration block has no vector route");
  }
  const int max_keys =
      kSegmentedCalibrationKeys[kSegmentedCalibrationPoints - 1];
  // The block kernel the production dispatch would take at the largest
  // count; the forced modes below bypass the key threshold.
  SegmentedTileMode block_mode = SegmentedTileMode::vector;
  if (nax_route_enabled(SegmentedTileMode::from_env) &&
      nax_verify_launch(device, kCalibrationRows, gqa, max_keys)
          .plan.supported) {
    block_mode = SegmentedTileMode::nax;
    c.block_kernel = CalibrationBlockKernel::nax;
  } else if (tile_verify_launch(device, kCalibrationRows, gqa, max_keys)
                 .plan.supported) {
    block_mode = SegmentedTileMode::tile;
    c.block_kernel = CalibrationBlockKernel::tile;
  } else {
    throw std::runtime_error("this device launches no block kernel");
  }

  // Fixed keys keep the process's random stream untouched. Each smaller
  // prefix is a leading view of the largest, as a KV cache slice is.
  const Shape q_shape{1, kCalibrationQHeads, kCalibrationRows, kHeadDimension};
  const Shape kv_full{1, kCalibrationKvHeads, max_keys - kCalibrationRows,
                      kHeadDimension};
  const Shape kv_new{1, kCalibrationKvHeads, kCalibrationRows, kHeadDimension};
  array q = random::normal(q_shape, bfloat16, random::key(0x5d9a7e51), stream);
  array pk_full =
      random::normal(kv_full, bfloat16, random::key(0x5d9a7e52), stream);
  array pv_full =
      random::normal(kv_full, bfloat16, random::key(0x5d9a7e53), stream);
  array nk = random::normal(kv_new, bfloat16, random::key(0x5d9a7e54), stream);
  array nv = random::normal(kv_new, bfloat16, random::key(0x5d9a7e55), stream);
  eval({q, pk_full, pv_full, nk, nv});
  // Nothing of the caller's is in flight on this thread's queue while the
  // samples run.
  metal::get_command_encoder(stream).synchronize();

  std::vector<std::vector<array>> point_inputs;
  for (size_t i = 0; i < kSegmentedCalibrationPoints; ++i) {
    const int prefix = kSegmentedCalibrationKeys[i] - kCalibrationRows;
    array pk = slice(pk_full, {0, 0, 0, 0},
                     {1, kCalibrationKvHeads, prefix, kHeadDimension}, stream);
    array pv = slice(pv_full, {0, 0, 0, 0},
                     {1, kCalibrationKvHeads, prefix, kHeadDimension}, stream);
    eval({pk, pv});
    point_inputs.push_back({q, pk, pv, nk, nv});
  }
  double host_vector = std::numeric_limits<double>::infinity();
  double host_block = std::numeric_limits<double>::infinity();
  auto time_route = [&](size_t point, SegmentedTileMode mode) {
    double host = 0.0;
    const double gpu = time_segmented_route(device, stream, point_inputs[point],
                                            head_rows, mode, &host);
    auto &best = mode == SegmentedTileMode::vector ? host_vector : host_block;
    best = std::min(best, host);
    return gpu;
  };
  // An idle GPU starts in a low clock state and the first dispatch of each
  // pipeline pays its load: a burst over the largest block, then one
  // discarded pair per point, before any sample counts.
  for (int warm = 0; warm < kCalibrationWarmupPairs; ++warm) {
    time_route(kSegmentedCalibrationPoints - 1, SegmentedTileMode::vector);
    time_route(kSegmentedCalibrationPoints - 1, block_mode);
  }
  c.points = 0;
  for (size_t i = 0; i < kSegmentedCalibrationPoints; ++i) {
    time_route(i, SegmentedTileMode::vector);
    time_route(i, block_mode);
    double best_vector = std::numeric_limits<double>::infinity();
    double best_block = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < kCalibrationReps; ++rep) {
      if (rep % 2 == 0) {
        best_vector =
            std::min(best_vector, time_route(i, SegmentedTileMode::vector));
        best_block = std::min(best_block, time_route(i, block_mode));
      } else {
        best_block = std::min(best_block, time_route(i, block_mode));
        best_vector =
            std::min(best_vector, time_route(i, SegmentedTileMode::vector));
      }
    }
    c.keys[i] = kSegmentedCalibrationKeys[i];
    c.vector_seconds[i] = best_vector;
    c.block_seconds[i] = best_block;
    ++c.points;
  }
  c.host_vector_seconds = host_vector;
  c.host_block_seconds = host_block;
  c.result = select_segmented_block_min_keys(c.keys, c.vector_seconds,
                                             c.block_seconds, c.points);
}

void log_block_calibration(const BlockCalibration &c) {
  if (!route_log_enabled()) {
    return;
  }
  const char *kernel = c.block_kernel == CalibrationBlockKernel::nax ? "nax"
                       : c.block_kernel == CalibrationBlockKernel::tile
                           ? "tile"
                           : "none";
  std::string line = "[sdpa route] verify block kernel (" +
                     std::string(kernel) + ") from " +
                     std::to_string(c.result) + " keys; calibrated in " +
                     std::to_string(c.elapsed_ms) + " ms";
  for (int i = 0; i < c.points; ++i) {
    line += "; " + std::to_string(c.keys[i]) + ": vector " +
            std::to_string(c.vector_seconds[i] * 1e6) + " us, block " +
            std::to_string(c.block_seconds[i] * 1e6) + " us";
  }
  line += "; host encode vector " +
          std::to_string(c.host_vector_seconds * 1e6) + " us, block " +
          std::to_string(c.host_block_seconds * 1e6) + " us";
  if (!c.error.empty()) {
    line += "; fallback: " + c.error;
  }
  std::fprintf(stderr, "%s\n", line.c_str());
}

// Measures the crossover once per process on the calling thread's GPU
// stream (the graph-building thread, never inside an eval). A failed
// measurement keeps the block kernels to the ceiling count so a device the
// calibration cannot time still routes.
void calibrate_block_min_keys(metal::Device &device, Stream stream) {
  auto &c = block_calibration();
  std::call_once(c.once, [&] {
    const auto started = std::chrono::steady_clock::now();
    // The calibration dispatches must not count as the caller's.
    const bool counting = bridge_testing::counting;
    bridge_testing::counting = false;
    try {
      auto pool = metal::new_scoped_memory_pool();
      measure_block_crossover(device, stream, c);
    } catch (const std::exception &e) {
      c.error = e.what();
      c.result = kSegmentedBlockMinKeysCeiling;
    }
    bridge_testing::counting = counting;
    c.elapsed_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - started)
                       .count();
    g_block_min_keys.store(c.result, std::memory_order_release);
    log_block_calibration(c);
  });
}

class SegmentedSdpa final : public fast::Custom {
public:
  SegmentedSdpa(Stream stream, float scale, bool causal, int head_rows,
                SegmentedTileMode tile_mode)
      : Custom(stream,
               [scale, causal, stream](std::vector<array> inputs) {
                 return segmented_fallback(std::move(inputs), scale, causal,
                                           stream);
               }),
        scale_(scale), causal_(causal), head_rows_(head_rows),
        tile_mode_(tile_mode) {}

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("SegmentedSdpa CPU NYI");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    encode_segmented_sdpa(metal::device(stream.device),
                          metal::get_command_encoder(stream), stream, inputs,
                          outputs[0], scale_, causal_, head_rows_, tile_mode_);
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
           head_rows_ == o.head_rows_ && tile_mode_ == o.tile_mode_;
  }

private:
  float scale_;
  bool causal_;
  // Rows of the leading chunk when the block is wider than one supported
  // query chunk; the dispatch is then chosen per real prefix length.
  int head_rows_;
  SegmentedTileMode tile_mode_;
};

} // namespace

array segmented_sdpa(const array &q, const array &prefix_k,
                     const array &prefix_v, const array &new_k,
                     const array &new_v, float scale, bool causal,
                     bool require_segmented, SegmentedTileMode tile_mode) {
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
  if (tile_mode == SegmentedTileMode::tile ||
      tile_mode == SegmentedTileMode::nax) {
    // Forced block kernel (tests): the block must be a verify block the
    // device can serve with it; alignment is checked at eval time.
    const int total = prefix_k.shape(2) + new_k.shape(2);
    const bool verify_block = causal && new_k.shape(2) == q_len;
    if (tile_mode == SegmentedTileMode::tile &&
        (!verify_block ||
         !tile_verify_launch(device, q_len, gqa, total).plan.supported)) {
      throw std::invalid_argument(
          "segmented SDPA tile route does not support this block");
    }
    if (tile_mode == SegmentedTileMode::nax &&
        (!verify_block || !nax_q_layout(q, prefix_k, prefix_v, new_k, new_v) ||
         !nax_verify_launch(device, q_len, gqa, total).plan.supported)) {
      throw std::invalid_argument(
          "segmented SDPA tensor-op route does not support this block");
    }
    auto primitive =
        std::make_shared<SegmentedSdpa>(stream, scale, causal, 0, tile_mode);
    return array(q.shape(), bfloat16, primitive, std::move(inputs));
  }
  int max_query_length = 0;
  try {
    max_query_length = segmented_max_query_length(device, gqa);
  } catch (const fast::paged::PrebuiltKernelMissing &) {
    throw;
  } catch (const std::exception &) {
    if (require_segmented) {
      throw;
    }
    return fallback();
  }
  // A verify block may take a block kernel at eval time; the crossover it
  // compares against is measured here, on the graph-building thread, the
  // first time a process reaches one (unless MLX_SDPA_VERIFY_TILE pins it).
  if (tile_mode == SegmentedTileMode::from_env && causal &&
      new_k.shape(2) == q_len && block_min_keys_override() < 0) {
    calibrate_block_min_keys(device, stream);
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
    auto primitive = std::make_shared<SegmentedSdpa>(stream, scale, causal,
                                                     head_rows, tile_mode);
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
  } catch (const fast::paged::PrebuiltKernelMissing &) {
    throw;
  } catch (const std::exception &) {
    if (require_segmented) {
      throw;
    }
    return fallback();
  }
  auto primitive =
      std::make_shared<SegmentedSdpa>(stream, scale, causal, 0, tile_mode);
  return array(q.shape(), bfloat16, primitive, std::move(inputs));
}

} // namespace mlx::core::segmented_sdpa

namespace {

mlx_array *segmented_sdpa_forward_impl(
    mlx_array *q, mlx_array *prefix_k, mlx_array *prefix_v, mlx_array *new_k,
    mlx_array *new_v, float scale, bool causal, bool require_segmented,
    mlx::core::segmented_sdpa::SegmentedTileMode tile_mode) {
  try {
    auto result = mlx::core::segmented_sdpa::segmented_sdpa(
        *reinterpret_cast<array *>(q), *reinterpret_cast<array *>(prefix_k),
        *reinterpret_cast<array *>(prefix_v), *reinterpret_cast<array *>(new_k),
        *reinterpret_cast<array *>(new_v), scale, causal, require_segmented,
        tile_mode);
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
  return segmented_sdpa_forward_impl(
      q, prefix_k, prefix_v, new_k, new_v, scale, causal, false,
      mlx::core::segmented_sdpa::SegmentedTileMode::from_env);
}

// Test-only contract: never silently qualify the concatenated fallback, and
// stay on the vector routes (bit-identical to MLX's vector SDPA) whatever
// MLX_SDPA_VERIFY_TILE says. mlx_segmented_sdpa_test_forward_tile covers the
// tile route.
extern "C" mlx_array *
mlx_segmented_sdpa_test_forward(mlx_array *q, mlx_array *prefix_k,
                                mlx_array *prefix_v, mlx_array *new_k,
                                mlx_array *new_v, float scale, bool causal) {
  return segmented_sdpa_forward_impl(
      q, prefix_k, prefix_v, new_k, new_v, scale, causal, true,
      mlx::core::segmented_sdpa::SegmentedTileMode::vector);
}

// Test-only: the simdgroup-matrix tile route, or null (message on stderr)
// when this device or block cannot take it.
extern "C" mlx_array *
mlx_segmented_sdpa_test_forward_tile(mlx_array *q, mlx_array *prefix_k,
                                     mlx_array *prefix_v, mlx_array *new_k,
                                     mlx_array *new_v, float scale) {
  return segmented_sdpa_forward_impl(
      q, prefix_k, prefix_v, new_k, new_v, scale, true, true,
      mlx::core::segmented_sdpa::SegmentedTileMode::tile);
}

// Test-only: the tile dispatch this device plans for a causal block of `rows`
// over `total_length` keys: out[0..5] = tile keys, stage-1 threads,
// threadgroup bytes, partitions, the tile pipeline's
// maxTotalThreadsPerThreadgroup (0 when it did not build). 1 when supported,
// 0 when not, -1 on error.
extern "C" int mlx_segmented_sdpa_test_tile_plan(int q_heads, int kv_heads,
                                                 int rows, int total_length,
                                                 uint32_t *out) {
  using namespace mlx::core;
  try {
    if (out == nullptr || kv_heads < 1 || q_heads % kv_heads != 0) {
      return -1;
    }
    auto stream = default_stream(Device::gpu);
    auto &device = metal::device(stream.device);
    const auto launch = segmented_sdpa::tile_verify_launch(
        device, rows, q_heads / kv_heads, total_length);
    out[0] = launch.plan.tile_n;
    out[1] = launch.plan.stage1_threads;
    out[2] = launch.plan.threadgroup_bytes;
    out[3] = static_cast<uint32_t>(launch.partitions);
    out[4] = launch.stage1 == nullptr
                 ? 0
                 : static_cast<uint32_t>(
                       launch.stage1->maxTotalThreadsPerThreadgroup());
    return launch.plan.supported ? 1 : 0;
  } catch (const std::exception &) {
    return -1;
  }
}

// Test-only: the tensor-op (NAX) route, or null (message on stderr) when
// this device, block or Q layout cannot take it.
extern "C" mlx_array *
mlx_segmented_sdpa_test_forward_nax(mlx_array *q, mlx_array *prefix_k,
                                    mlx_array *prefix_v, mlx_array *new_k,
                                    mlx_array *new_v, float scale) {
  return segmented_sdpa_forward_impl(
      q, prefix_k, prefix_v, new_k, new_v, scale, true, true,
      mlx::core::segmented_sdpa::SegmentedTileMode::nax);
}

// Test-only: the tensor-op dispatch this device plans for a causal block of
// `rows` over `total_length` keys: out[0..6] = M, tile keys, stage-1
// threads, threadgroup bytes, partitions, the pipeline's
// maxTotalThreadsPerThreadgroup (0 when it did not build). 1 when supported,
// 0 when not (no NAX on this device included), -1 on error.
extern "C" int mlx_segmented_sdpa_test_nax_plan(int q_heads, int kv_heads,
                                                int rows, int total_length,
                                                uint32_t *out) {
  using namespace mlx::core;
  try {
    if (out == nullptr || kv_heads < 1 || q_heads % kv_heads != 0) {
      return -1;
    }
    auto stream = default_stream(Device::gpu);
    auto &device = metal::device(stream.device);
    const auto launch = segmented_sdpa::nax_verify_launch(
        device, rows, q_heads / kv_heads, total_length);
    out[0] = launch.plan.m;
    out[1] = launch.plan.tile_n;
    out[2] = launch.plan.stage1_threads;
    out[3] = launch.plan.threadgroup_bytes;
    out[4] = static_cast<uint32_t>(launch.partitions);
    out[5] = launch.stage1 == nullptr
                 ? 0
                 : static_cast<uint32_t>(
                       launch.stage1->maxTotalThreadsPerThreadgroup());
    return launch.plan.supported ? 1 : 0;
  } catch (const std::exception &) {
    return -1;
  }
}

// 0 when segmented SDPA is not supported here; -1 (message on stderr) when its
// prebuilt kernels are missing from paged_attn.metallib.
extern "C" int mlx_segmented_sdpa_max_query_length(int gqa_factor) {
  try {
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    return mlx::core::segmented_sdpa::segmented_max_query_length(
        mlx::core::metal::device(stream.device), gqa_factor);
  } catch (const mlx::core::fast::paged::PrebuiltKernelMissing &e) {
    std::fprintf(stderr, "mlx_segmented_sdpa_max_query_length: %s\n", e.what());
    return -1;
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

// Test-only: the key count (prefix + new rows) from which the production
// entry takes a block kernel: MLX_SDPA_VERIFY_TILE when set (0 = the block
// kernels are off), else the calibrated crossover (measured now if this
// process has not yet). -1 without Metal or on error.
extern "C" int mlx_segmented_sdpa_test_block_min_keys() {
  using namespace mlx::core;
  try {
    auto stream = default_stream(Device::gpu);
    auto &device = metal::device(stream.device);
    if (segmented_sdpa::block_min_keys_override() < 0) {
      segmented_sdpa::calibrate_block_min_keys(device, stream);
    }
    return segmented_sdpa::effective_block_min_keys();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_segmented_sdpa_test_block_min_keys: %s\n",
                 e.what());
    return -1;
  }
}

// Test-only: the calibration record (measured now if this process has not
// yet): per point `keys`, `vector_seconds` and `block_seconds` (up to
// `capacity` entries), the wall time of the whole calibration, the block
// kernel it timed (0 none, 1 tile, 2 tensor-op) and the selected crossover
// before any override. Returns the point count (0 when the measurement
// failed and the ceiling was recorded; its message on stderr), -1 without
// Metal or on error.
extern "C" int mlx_segmented_sdpa_test_block_calibration(
    int *keys, double *vector_seconds, double *block_seconds, size_t capacity,
    double *out_elapsed_ms, int *out_block_kernel, int *out_result) {
  using namespace mlx::core;
  try {
    auto stream = default_stream(Device::gpu);
    auto &device = metal::device(stream.device);
    segmented_sdpa::calibrate_block_min_keys(device, stream);
    const auto &c = segmented_sdpa::block_calibration();
    for (int i = 0; i < c.points && size_t(i) < capacity; ++i) {
      if (keys) {
        keys[i] = c.keys[i];
      }
      if (vector_seconds) {
        vector_seconds[i] = c.vector_seconds[i];
      }
      if (block_seconds) {
        block_seconds[i] = c.block_seconds[i];
      }
    }
    if (out_elapsed_ms) {
      *out_elapsed_ms = c.elapsed_ms;
    }
    if (out_block_kernel) {
      *out_block_kernel = static_cast<int>(c.block_kernel);
    }
    if (out_result) {
      *out_result = c.result;
    }
    if (!c.error.empty()) {
      std::fprintf(stderr, "segmented SDPA crossover calibration: %s\n",
                   c.error.c_str());
    }
    return c.points;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_segmented_sdpa_test_block_calibration: %s\n",
                 e.what());
    return -1;
  }
}

// Test-only: the raw limits of the pipelines a one-call verify dispatches,
// read from Metal without the launch planner. out[0..3] the verify kernel's
// threadExecutionWidth, maxTotalThreadsPerThreadgroup and
// staticThreadgroupMemoryLength, out[3..6] the same for MLX's reduction
// kernel, out[6] the device's maxThreadgroupMemoryLength. -1 on error.
extern "C" int mlx_segmented_sdpa_test_verify_pipeline_limits(int gqa, int rows,
                                                              int partitions,
                                                              uint64_t *out) {
  using namespace mlx::core;
  try {
    if (out == nullptr) {
      return -1;
    }
    auto stream = default_stream(Device::gpu);
    auto &device = metal::device(stream.device);
    auto *verify = segmented_sdpa::segmented_kernel(
        device, {segmented_sdpa::SegmentedKernel::verify_two_pass_1, true,
                 partitions, gqa, rows, 0, 0});
    auto *reduction = segmented_sdpa::reduction_kernel(device);
    out[0] = verify->threadExecutionWidth();
    out[1] = verify->maxTotalThreadsPerThreadgroup();
    out[2] = verify->staticThreadgroupMemoryLength();
    out[3] = reduction->threadExecutionWidth();
    out[4] = reduction->maxTotalThreadsPerThreadgroup();
    out[5] = reduction->staticThreadgroupMemoryLength();
    out[6] = device.mtl_device()->maxThreadgroupMemoryLength();
    return 0;
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

extern "C" mlx_array *
mlx_segmented_sdpa_test_forward_tile(mlx_array *, mlx_array *, mlx_array *,
                                     mlx_array *, mlx_array *, float) {
  return nullptr;
}

extern "C" int mlx_segmented_sdpa_test_tile_plan(int, int, int, int,
                                                 uint32_t *) {
  return -1;
}

extern "C" mlx_array *
mlx_segmented_sdpa_test_forward_nax(mlx_array *, mlx_array *, mlx_array *,
                                    mlx_array *, mlx_array *, float) {
  return nullptr;
}

extern "C" int mlx_segmented_sdpa_test_nax_plan(int, int, int, int,
                                                uint32_t *) {
  return -1;
}

extern "C" int mlx_segmented_sdpa_max_query_length(int) { return 0; }

extern "C" int mlx_segmented_sdpa_test_device_verify_route(int, int, int, int,
                                                           char *) {
  return -1;
}

extern "C" int mlx_segmented_sdpa_test_verify_pipeline_limits(int, int, int,
                                                              uint64_t *) {
  return -1;
}

extern "C" int mlx_segmented_sdpa_test_block_min_keys() { return -1; }

extern "C" int mlx_segmented_sdpa_test_block_calibration(int *, double *,
                                                         double *, size_t,
                                                         double *, int *,
                                                         int *) {
  return -1;
}

#endif

// Test-only, platform independent: the crossover selection over measured
// (or synthetic) per-point seconds; see select_segmented_block_min_keys.
extern "C" int mlx_segmented_sdpa_test_select_block_min_keys(
    const int *keys, const double *vector_seconds, const double *block_seconds,
    size_t count) {
  if (count > 0 && (keys == nullptr || vector_seconds == nullptr ||
                    block_seconds == nullptr)) {
    return -1;
  }
  return mlx::core::segmented_sdpa::select_segmented_block_min_keys(
      keys, vector_seconds, block_seconds, count);
}

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

// Test-only, platform independent: the tile planner over synthetic pipeline
// limits. out[0..4] = tile keys, stage-1 threads, threadgroup bytes and the
// partition count for `total_length` keys; 1 when supported, else 0.
extern "C" int mlx_segmented_sdpa_test_verify_tile_plan(
    int rows, int gqa_factor, int total_length, int blocks_override,
    size_t stage1_width, size_t stage1_max_threads, size_t stage1_static_memory,
    size_t device_max_memory, size_t stage2_width, size_t stage2_max_threads,
    size_t stage2_static_memory, uint32_t *out) {
  using namespace mlx::core::segmented_sdpa;
  SegmentedSdpaCapabilities c1{stage1_width, stage1_max_threads,
                               stage1_static_memory, device_max_memory};
  SegmentedSdpaCapabilities c2{stage2_width, stage2_max_threads,
                               stage2_static_memory, device_max_memory};
  auto plan = plan_segmented_verify_tile_launch(rows, gqa_factor, c1, c2);
  if (out) {
    out[0] = plan.tile_n;
    out[1] = plan.stage1_threads;
    out[2] = plan.threadgroup_bytes;
    out[3] = static_cast<uint32_t>(segmented_verify_tile_partitions(
        total_length, static_cast<int>(plan.tile_n), blocks_override));
  }
  return plan.supported ? 1 : 0;
}

// Test-only, platform independent: the tensor-op planner over synthetic
// pipeline limits. out[0..5] = M, tile keys, stage-1 threads, threadgroup
// bytes and the partition count for `total_length` keys; 1 when supported,
// else 0.
extern "C" int mlx_segmented_sdpa_test_verify_nax_plan(
    int rows, int gqa_factor, int total_length, int blocks_override,
    size_t stage1_width, size_t stage1_max_threads, size_t stage1_static_memory,
    size_t device_max_memory, size_t stage2_width, size_t stage2_max_threads,
    size_t stage2_static_memory, uint32_t *out) {
  using namespace mlx::core::segmented_sdpa;
  SegmentedSdpaCapabilities c1{stage1_width, stage1_max_threads,
                               stage1_static_memory, device_max_memory};
  SegmentedSdpaCapabilities c2{stage2_width, stage2_max_threads,
                               stage2_static_memory, device_max_memory};
  auto plan = plan_segmented_verify_nax_launch(rows, gqa_factor, c1, c2);
  if (out) {
    out[0] = plan.m;
    out[1] = plan.tile_n;
    out[2] = plan.stage1_threads;
    out[3] = plan.threadgroup_bytes;
    out[4] = static_cast<uint32_t>(segmented_verify_tile_partitions(
        total_length, static_cast<int>(plan.tile_n), blocks_override));
  }
  return plan.supported ? 1 : 0;
}

namespace {

struct Fnv1a {
  uint64_t state = 0xcbf29ce484222325ull;
  void add(int64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      state ^= static_cast<uint64_t>(value >> (8 * byte)) & 0xff;
      state *= 0x100000001b3ull;
    }
  }
  void add(const mlx::core::segmented_sdpa::SegmentedSdpaLaunchPlan &plan) {
    add(plan.supported);
    add(plan.two_pass);
    add(plan.partitions);
    add(plan.stage1_threads);
    add(plan.stage2_threads);
  }
};

} // namespace

// Test-only, platform independent: FNV-1a 64 digests of every planner and
// policy output over a fixed input sweep for one device class, in the order
// two-pass, partitions, head rows, launch plan, verify launch, verify route.
extern "C" int64_t mlx_segmented_sdpa_test_plan_digests(char device_class,
                                                        int blocks_override,
                                                        uint64_t *out_digests) {
  using namespace mlx::core::segmented_sdpa;
  Fnv1a digests[6];
  int64_t checked = 0;
  std::vector<int> lengths;
  for (int length = 1; length <= 140000;
       length += length < 2048 ? 1 : (length < 70000 ? 7 : 997)) {
    lengths.push_back(length);
  }
  for (int boundary = 2048; boundary <= (1 << 17); boundary *= 2) {
    for (int d = -2; d <= 2; ++d) {
      lengths.push_back(boundary + d);
    }
  }
  for (int length : lengths) {
    for (int q_heads : {1, 2, 4, 6, 8, 12, 16, 24, 32, 64}) {
      for (int kv_heads : {1, 2, 4, 8}) {
        if (q_heads % kv_heads != 0) {
          continue;
        }
        ++checked;
        digests[0].add(
            sdpa_vector_uses_two_pass(device_class, length, q_heads, kv_heads));
      }
    }
    for (int simdgroups = 1; simdgroups <= 256; ++simdgroups) {
      ++checked;
      digests[1].add(sdpa_vector_partition_count(device_class, length,
                                                 simdgroups, blocks_override));
    }
  }

  const size_t widths[] = {16, 32, 64};
  const size_t threads[] = {0, 32, 512, 960, 1023, 1024, 2048};
  const size_t memory[] = {0, 4096, 32768, 65536};
  for (int rows = -1; rows <= 10; ++rows) {
    for (int max_q = -1; max_q <= 10; ++max_q) {
      ++checked;
      digests[2].add(segmented_verify_head_rows(rows, max_q));
    }
    for (int gqa = 0; gqa <= 33; ++gqa) {
      for (int partitions : {0, 16, 31, 32, 48, 64, 128, 1024}) {
        for (size_t w : widths) {
          for (size_t t : threads) {
            for (size_t m : memory) {
              SegmentedSdpaCapabilities c1{w, t, m, 32768};
              SegmentedSdpaCapabilities c2{32, t, m, 32768};
              for (bool two_pass : {false, true}) {
                checked += 2;
                digests[3].add(plan_segmented_sdpa_launch(rows, gqa, two_pass,
                                                          partitions, c1, &c2));
                digests[3].add(plan_segmented_sdpa_launch(
                    rows, gqa, two_pass, partitions, c1, nullptr));
              }
              ++checked;
              digests[4].add(
                  plan_segmented_verify_launch(rows, gqa, partitions, c1, c2));
            }
          }
        }
      }
    }
  }
  for (int head_rows = 0; head_rows <= 8; ++head_rows) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        const int parts[] = {32, 64, 128, 256};
        for (bool unified : {false, true}) {
          ++checked;
          digests[5].add(static_cast<int>(select_segmented_verify_route(
              head_rows, {(a & 1) != 0, parts[a]}, {(b & 1) != 0, parts[b]},
              unified)));
        }
      }
    }
  }
  for (int i = 0; i < 6; ++i) {
    out_digests[i] = digests[i].state;
  }
  return checked;
}
