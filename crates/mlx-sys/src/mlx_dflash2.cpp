#include "mlx_common.h"

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

const char* topk16_header =
#include "metal/common/topk16_helpers.metal.inc"
    ;

} // namespace

extern "C" bool mlx_dflash2_topk16(mlx_array* logits, mlx_array** out_ids,
                                   mlx_array** out_values) {
  if (!logits || !out_ids || !out_values)
    return false;
  *out_ids = nullptr;
  *out_values = nullptr;
  try {
    const auto& in = *reinterpret_cast<array*>(logits);
    if (in.ndim() < 2)
      return false;
    const int vocab = in.shape(-1);
    const int rows = in.size() / vocab;
    if (rows < 1 || vocab < kTopK || in.size() != rows * vocab)
      return false;
    const auto dt = in.dtype();
    if (dt != mlx::core::bfloat16 && dt != mlx::core::float16 &&
        dt != mlx::core::float32)
      return false;

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
                        {rows * 32, 1, 1}, {32, 1, 1}, {{"SHARDS", kShards}},
                        std::nullopt, false, mlx::core::Device::gpu);
    *out_ids =
        reinterpret_cast<mlx_array*>(new array(std::move(merged[0])));
    *out_values =
        reinterpret_cast<mlx_array*>(new array(std::move(merged[1])));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_dflash2_topk16 error: " << e.what() << std::endl;
    return false;
  }
}
