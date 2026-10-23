#include "mlx_common.h"

#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#endif

// DFlash2 candidate selector top-K: replaces `argpartition` (full multi-block
// merge sort over the vocab) with a sharded register top-16 scan plus a
// one-simdgroup merge. See metal/common/dflash2_topk16_*.metal.inc — the
// algorithm is adapted from Splash's draft selector (Apache-2.0).
//
// Contract: `logits` is a contiguous [rows, vocab] (or [1, rows, vocab])
// float/bf16/f16 tensor. Outputs are `ids` int32 [rows, 16] and `values`
// float32 [rows, 16], both ASCENDING by value — the same order
// `argpartition(-16).slice(-16..)` produces.

namespace {

constexpr int kTopK = 16;
constexpr int kShards = 8;
constexpr int kScanThreads = 256;
constexpr int kSimdWidth = 32;

#ifdef MLX_NODE_METAL_ENABLED
// Inspect the exact generated pipeline used by CustomKernel::eval_gpu. This
// compiles/caches shader metadata but never evaluates arrays or encodes work.
// The launch sizes are algorithm contracts, not device-performance choices.
bool selector_pipeline_supported(const array& output, int required_threads) {
  if (!output.has_primitive())
    throw std::runtime_error("selector output is missing its custom primitive");
  auto* custom =
      dynamic_cast<mlx::core::fast::CustomKernel*>(&output.primitive());
  if (!custom)
    throw std::runtime_error("selector output is not a CustomKernel");
  const auto state = custom->state();
  const auto& name = std::get<0>(state);
  const auto& source = std::get<1>(state);
  const auto& [tx, ty, tz] = std::get<3>(state);
  if (tx != required_threads || ty != 1 || tz != 1)
    throw std::runtime_error("selector threadgroup violates its algorithm contract");
  auto& device = mlx::core::metal::device(custom->stream().device);
  // CustomKernel::eval_gpu's library key; any other key compiles the source
  // a second time.
  auto* library = device.get_library(
      custom->library_name(), mlx::core::CompileOptions(std::get<10>(state)),
      [&source] { return mlx::core::metal::utils() + source; });
  auto* pipeline = device.get_kernel(name, library);
  const bool supported = pipeline->threadExecutionWidth() == kSimdWidth &&
      pipeline->maxTotalThreadsPerThreadgroup() >= required_threads &&
      pipeline->staticThreadgroupMemoryLength() <=
          device.mtl_device()->maxThreadgroupMemoryLength();
  if (!supported) {
    std::cerr << "dflash2 topk16 unsupported pipeline " << name
              << ": SIMD width=" << pipeline->threadExecutionWidth()
              << " (requires " << kSimdWidth << "), max threads="
              << pipeline->maxTotalThreadsPerThreadgroup()
              << " (requires " << required_threads << "), static memory="
              << pipeline->staticThreadgroupMemoryLength()
              << ", device memory=" << device.mtl_device()->maxThreadgroupMemoryLength()
              << std::endl;
  }
  return supported;
}

bool selector_pipelines_supported(const array& scan, const array& merge,
                                  mlx::core::Dtype dtype, int vocab) {
  auto& device = mlx::core::metal::device(scan.primitive().stream().device);
  // These private kernel names/sources/options are immutable. After validating
  // nonempty rows and vocab >= 16, generated input address spaces are always
  // device pointers; only dtype and vocab specialize their source. Row count
  // changes the grid, not source or threadgroup. Cache by actual device too,
  // avoiding source-string copies and pipeline lookup on every proposal.
  struct Entry {
    MTL::Device* device;
    mlx::core::Dtype dtype;
    int vocab;
    bool supported;
  };
  static std::mutex mutex;
  static std::vector<Entry> cache;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : cache) {
    if (entry.device == device.mtl_device() && entry.dtype == dtype &&
        entry.vocab == vocab)
      return entry.supported;
  }
  const bool supported = selector_pipeline_supported(scan, kScanThreads) &&
      selector_pipeline_supported(merge, kSimdWidth);
  cache.push_back({device.mtl_device(), dtype, vocab, supported});
  return supported;
}
#else
bool selector_pipelines_supported(const array&, const array&,
                                  mlx::core::Dtype, int) {
  return false;
}
#endif

const char* topk16_header =
#include "metal/common/topk16_helpers.metal.inc"
    ;

} // namespace

static int dflash2_topk16_impl(mlx_array* logits, mlx_array** out_ids,
                                   mlx_array** out_values) {
  if (!logits || !out_ids || !out_values)
    return 0;
  *out_ids = nullptr;
  *out_values = nullptr;
  try {
    const auto& in = *reinterpret_cast<array*>(logits);
    if (in.ndim() < 2)
      return 0;
    const int vocab = in.shape(-1);
    if (vocab < kTopK)
      return 0;
    const int rows = in.size() / vocab;
    if (rows < 1 || in.size() != rows * vocab)
      return 0;
    const auto dt = in.dtype();
    if (dt != mlx::core::bfloat16 && dt != mlx::core::float16 &&
        dt != mlx::core::float32)
      return 0;

    static auto scan = mlx::core::fast::metal_kernel(
        "dflash2_topk16_scan", {"logits"}, {"partial_ids", "partial_values"},
#include "metal/common/dflash2_topk16_scan.metal.inc"
        ,
        topk16_header);
    static auto merge = mlx::core::fast::metal_kernel(
        "dflash2_topk16_merge", {"partial_ids", "partial_values"},
        {"out_ids", "out_values"},
#include "metal/common/dflash2_topk16_merge.metal.inc"
        ,
        topk16_header);

    // `grid` is in threads: scan launches (rows × SHARDS) groups of 256,
    // merge launches one simdgroup per row.
    auto partials = scan({in},
                         {{rows * kShards * kTopK}, {rows * kShards * kTopK}},
                         {mlx::core::uint32, mlx::core::float32},
                         {kShards * kScanThreads, rows, 1},
                         {kScanThreads, 1, 1},
                         {{"T", dt}, {"VOCAB", vocab}, {"SHARDS", kShards}},
                         std::nullopt, false, mlx::core::Device::gpu);
    auto merged = merge(partials, {{rows, kTopK}, {rows, kTopK}},
                        {mlx::core::int32, mlx::core::float32},
                        {rows * kSimdWidth, 1, 1}, {kSimdWidth, 1, 1},
                        {{"SHARDS", kShards}},
                        std::nullopt, false, mlx::core::Device::gpu);
    if (!selector_pipelines_supported(partials[0], merged[0], dt, vocab))
      return 0;
    *out_ids =
        reinterpret_cast<mlx_array*>(new array(std::move(merged[0])));
    *out_values =
        reinterpret_cast<mlx_array*>(new array(std::move(merged[1])));
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "mlx_dflash2_topk16 error: " << e.what() << std::endl;
    return -1;
  }
}

extern "C" bool mlx_dflash2_topk16(mlx_array* logits, mlx_array** out_ids,
                                   mlx_array** out_values) {
  return dflash2_topk16_impl(logits, out_ids, out_values) == 1;
}

// Test-only status: distinguish unsupported/invalid input from construction
// failure so a broken fused route cannot silently pass its parity regression.
extern "C" int mlx_dflash2_topk16_test(mlx_array* logits, mlx_array** out_ids,
                                     mlx_array** out_values) {
  return dflash2_topk16_impl(logits, out_ids, out_values);
}

// Optional source-style selector edge arithmetic. All six arguments are
// explicit graph inputs, including the changing anchor. Row-contiguous copies
// are lazy and handled by CustomKernel, so sliced projected rows are accepted.
extern "C" mlx_array* mlx_dflash2_selector_edges(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor) {
#ifdef MLX_NODE_METAL_ENABLED
  using namespace mlx::core;
  if (!candidates || !unary || !hidden || !predecessor || !successor || !anchor ||
      !mlx::core::metal::is_available() || !(default_device() == Device::gpu))
    return nullptr;
  try {
    const auto& c = *reinterpret_cast<array*>(candidates);
    const auto& u = *reinterpret_cast<array*>(unary);
    const auto& h = *reinterpret_cast<array*>(hidden);
    const auto& p = *reinterpret_cast<array*>(predecessor);
    const auto& s = *reinterpret_cast<array*>(successor);
    const auto& a = *reinterpret_cast<array*>(anchor);
    if (c.dtype() != int32 || u.dtype() != float32 || h.dtype() != bfloat16 ||
        p.dtype() != bfloat16 || s.dtype() != bfloat16 || a.dtype() != int32 ||
        c.ndim() != 3 || c.shape(0) != 1 || c.shape(2) != 16 ||
        c.shape(1) < 1 || c.shape(1) > 7 || u.shape() != c.shape() ||
        h.shape() != Shape{c.shape(1), 1, 256} || p.ndim() != 2 ||
        p.shape(0) < 16 || p.shape(1) != 256 || s.shape() != p.shape() ||
        a.shape() != Shape{1})
      return nullptr;
    static auto kernel = mlx::core::fast::metal_kernel(
        "dflash2_selector_edges", {"candidates", "unary", "hidden",
                                    "predecessor", "successor", "anchor"}, {"scores"},
#include "metal/common/dflash2_selector_edges.metal.inc"
    );
    auto output = kernel({c, u, h, p, s, a}, {{c.shape(1), 16, 16}}, {float32},
                         {c.shape(1) * 256, 1, 1}, {256, 1, 1}, {},
                         std::nullopt, false, Device::gpu);
    // All supported calls have identical generated input types and rank.
    // Shape metadata is dynamic, so validate this one source once per device.
    auto* device = mlx::core::metal::device(default_stream(Device::gpu).device).mtl_device();
    static std::mutex mutex;
    static std::vector<std::pair<MTL::Device*, bool>> supported;
    bool valid = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = std::find_if(supported.begin(), supported.end(),
                            [device](const auto& item) { return item.first == device; });
      if (it == supported.end()) {
        valid = selector_pipeline_supported(output[0], 256);
        supported.emplace_back(device, valid);
      } else {
        valid = it->second;
      }
    }
    if (!valid) return nullptr;
    return reinterpret_cast<mlx_array*>(new array(std::move(output[0])));
  } catch (const std::exception& e) {
    std::cerr << "mlx_dflash2_selector_edges error: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)candidates; (void)unary; (void)hidden; (void)predecessor; (void)successor; (void)anchor;
  return nullptr;
#endif
}

// Collapse the dependent greedy predecessor walk from roughly four lazy MLX
// operations per proposal position to one Metal dispatch. The custom-kernel
// wrapper keeps inputs row-contiguous, inserting a lazy copy for transposed or
// strided views; well-formed production inputs are already contiguous.
extern "C" bool mlx_dflash2_greedy_path(mlx_array* candidates,
                                        mlx_array* scores,
                                        mlx_array** out_path) {
  if (!candidates || !scores || !out_path)
    return false;
  *out_path = nullptr;
  try {
    const auto& candidate_ids = *reinterpret_cast<array*>(candidates);
    const auto& score_table = *reinterpret_cast<array*>(scores);
    if (candidate_ids.dtype() != mlx::core::int32 ||
        score_table.dtype() != mlx::core::float32 ||
        candidate_ids.ndim() != 3 || score_table.ndim() != 3 ||
        candidate_ids.shape(0) != 1)
      return false;

    const int length = candidate_ids.shape(1);
    const int top_k = candidate_ids.shape(2);
    if (length < 1 || top_k < 1 || top_k > 64 ||
        score_table.shape(0) != length ||
        score_table.shape(1) != top_k || score_table.shape(2) != top_k)
      return false;

    static auto greedy_path = mlx::core::fast::metal_kernel(
        "dflash2_greedy_path", {"candidates", "scores"}, {"path"},
#include "metal/common/dflash2_greedy_path.metal.inc"
    );
    auto outputs = greedy_path(
        {candidate_ids, score_table}, {{length}}, {mlx::core::int32},
        {1, 1, 1}, {1, 1, 1}, {}, std::nullopt, false,
        mlx::core::Device::gpu);
    *out_path = reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_dflash2_greedy_path error: " << e.what() << std::endl;
    return false;
  }
}

extern "C" bool mlx_dflash2_greedy_path_confidence(mlx_array* candidates,
                                        mlx_array* scores,
                                        mlx_array** out_path, mlx_array** out_confidence) {
  if (!candidates || !scores || !out_path || !out_confidence)
    return false;
  *out_path = nullptr;
  *out_confidence = nullptr;
  try {
    const auto& candidate_ids = *reinterpret_cast<array*>(candidates);
    const auto& score_table = *reinterpret_cast<array*>(scores);
    if (candidate_ids.dtype() != mlx::core::int32 ||
        score_table.dtype() != mlx::core::float32 ||
        candidate_ids.ndim() != 3 || score_table.ndim() != 3 ||
        candidate_ids.shape(0) != 1)
      return false;

    const int length = candidate_ids.shape(1);
    const int top_k = candidate_ids.shape(2);
    if (length < 1 || top_k < 1 || top_k > 64 ||
        score_table.shape(0) != length ||
        score_table.shape(1) != top_k || score_table.shape(2) != top_k)
      return false;

    static auto greedy_path = mlx::core::fast::metal_kernel(
        "dflash2_greedy_path_confidence", {"candidates", "scores"}, {"path", "confidence"},
#include "metal/common/dflash2_greedy_path_confidence.metal.inc"
    );
    auto outputs = greedy_path(
        {candidate_ids, score_table}, {{length}, {length}}, {mlx::core::int32, mlx::core::float32},
        {1, 1, 1}, {1, 1, 1}, {}, std::nullopt, false,
        mlx::core::Device::gpu);
    auto path = std::make_unique<array>(std::move(outputs[0]));
    auto confidence = std::make_unique<array>(std::move(outputs[1]));
    *out_path = reinterpret_cast<mlx_array*>(path.release());
    *out_confidence = reinterpret_cast<mlx_array*>(confidence.release());
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_dflash2_greedy_path error: " << e.what() << std::endl;
    return false;
  }
}

extern "C" mlx_array* mlx_dflash2_verify_budget(mlx_array* confidence, mlx_array* parameters) {
  if (!confidence || !parameters) return nullptr;
  try {
    const auto& q = *reinterpret_cast<array*>(confidence);
    const auto& p = *reinterpret_cast<array*>(parameters);
    if (q.dtype() != mlx::core::float32 || p.dtype() != mlx::core::float32 ||
        q.ndim() != 1 || q.shape(0) != 7 || p.ndim() != 1 || p.shape(0) != 18)
      return nullptr;
    static auto kernel = mlx::core::fast::metal_kernel(
        "dflash2_verify_budget", {"confidence", "parameters"}, {"budget"},
#include "metal/common/dflash2_verify_budget.metal.inc"
    );
    auto outputs = kernel({q, p}, {{1}}, {mlx::core::int32},
                          {1, 1, 1}, {1, 1, 1}, {}, std::nullopt, false,
                          mlx::core::Device::gpu);
    return reinterpret_cast<mlx_array*>(new array(std::move(outputs[0])));
  } catch (const std::exception& e) {
    std::cerr << "mlx_dflash2_verify_budget error: " << e.what() << std::endl;
    return nullptr;
  }
}
