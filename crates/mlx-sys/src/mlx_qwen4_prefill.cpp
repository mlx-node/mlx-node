#include "mlx_common.h"
#include <cstdlib>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"
namespace mlx::core::qwen4_preamble {
const char *nax();
const char *quantized_utils();
const char *kquant_nax();
} // namespace mlx::core::qwen4_preamble
#endif

#ifdef MLX_NODE_METAL_ENABLED
static const std::string &qwen4_prefill_header() {
  static const std::string header = [] {
    // Only the decoders/loaders are needed. Uninstantiated generic kernels
    // also require MLX's private broadcasting helpers; omit those kernels.
    std::string source = mlx::core::qwen4_preamble::kquant_nax();
    auto function = source.find("METAL_FUNC void kquant_qmm_t_nax_tgp_impl(");
    auto end = function == std::string::npos
                   ? std::string::npos
                   : source.rfind("\ntemplate <", function);
    if (end == std::string::npos)
      throw std::runtime_error("Qwen4 NAX preamble boundary changed");
    source.resize(end);
    return std::string(mlx::core::qwen4_preamble::nax()) + source;
  }();
  return header;
}
#endif

extern "C" {

// Match the ordinary affine8 NAX matrix path while avoiding global F32 copies.
// Split-K shapes keep their existing partitioning and reduction arithmetic.
mlx_array *mlx_qwen4_dense_prefill(mlx_array *input, mlx_array *weight,
                                   mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !weight || !scales || !biases)
      return nullptr;
    // Decode must return before device lookup or copies of the four handles.
    // The kernel's input vector takes ownership only for eligible prefill.
    const auto &x = *reinterpret_cast<array *>(input);
    if (x.ndim() < 2 || x.dtype() != mlx::core::bfloat16 || x.shape(-1) <= 0)
      return nullptr;
    int k = x.shape(-1);
    if (x.size() / k < 256 || x.size() / k > 65536 ||
        !mlx::core::metal::is_nax_available() || !mlx::core::env::enable_tf32())
      return nullptr;
    const auto &w = *reinterpret_cast<array *>(weight);
    const auto &s = *reinterpret_cast<array *>(scales);
    const auto &b = *reinterpret_cast<array *>(biases);
    if (w.ndim() != 2 || w.dtype() != mlx::core::uint32 || s.ndim() != 2 ||
        s.dtype() != mlx::core::float16 || b.dtype() != mlx::core::float16 ||
        s.shape() != b.shape())
      return nullptr;
    int n = w.shape(0);
    if (k % 64 || n <= 0 || n % 64)
      return nullptr;
    int rows = x.size() / k;
    if ((n < 2048 && (rows < 1024 || k < 8192)) || w.shape(1) != k / 4 ||
        s.shape(0) != n || s.shape(1) != k / 32)
      return nullptr;
    // Same selection as QuantizedMatmul::qmm_splitk; it falls through to NAX
    // only when there is one partition. Shape tails remain bounds checked.
    int partitions = std::max(1, 512 / (((n + 31) / 32) * ((rows + 31) / 32)));
    partitions = std::min(partitions, k / 32);
    while (partitions > 1 && k % (partitions * 32))
      --partitions;
    if (partitions > 1)
      return nullptr;
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_dense_prefill", {"x", "w", "scales", "biases"}, {"out"},
#include "metal/qwen4_dense_prefill.metal.inc"
        , qwen4_prefill_header());
    auto shape = x.shape();
    shape.back() = n;
    auto result = kernel({x, w, s, b}, {shape}, {mlx::core::bfloat16},
                         {32 * (n / 64), 4 * ((rows + 63) / 64), 1}, {32, 4, 1},
                         {{"C", mlx::core::float32},
                          {"T", mlx::core::bfloat16},
                          {"K", k},
                          {"N", n},
                          {"R", rows},
                          {"GS", 32},
                          {"BITS", 8}},
                         std::nullopt, false,
                         mlx::core::default_stream(mlx::core::Device::gpu));
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 compact dense prefill: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// Address sorted rows directly, preserving col_reduce_small's eight-lane
// association and the materialized product's dtype/rounding.
mlx_array *mlx_qwen4_sorted_combine(mlx_array *values, mlx_array *scores,
                                    mlx_array *inverse, int top) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!values || !scores || !inverse || top != 10)
      return nullptr;
    auto v = *reinterpret_cast<array *>(values),
         s = *reinterpret_cast<array *>(scores),
         order = *reinterpret_cast<array *>(inverse);
    if (v.ndim() != 2 || v.shape(1) < 128 || v.shape(0) % top ||
        s.size() != v.shape(0) || order.size() != v.shape(0) ||
        (v.dtype() != mlx::core::bfloat16 && v.dtype() != mlx::core::float32) ||
        (s.dtype() != mlx::core::bfloat16 && s.dtype() != mlx::core::float32) ||
        (order.dtype() != mlx::core::uint32 &&
         order.dtype() != mlx::core::int32))
      return nullptr;
    int rows = v.shape(0), h = v.shape(1), tokens = rows / top;
    auto product_type = mlx::core::promote_types(v.dtype(), s.dtype());
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_sorted_combine", {"values", "scores", "inverse"}, {"out"}, R"(
      uint i=thread_position_in_grid.x;
      if (i>=TOKENS*H) return;
      uint token=i/H, col=i%H;
      constexpr int LANES=TOP<8?TOP:8;
      P totals[LANES];
      for (int lane=0; lane<LANES; ++lane) totals[lane]=P(0);
      for (int slot=0; slot<TOP; ++slot) {
        uint row=uint(inverse[token*TOP+slot]);
        // The volatile product keeps the original multiply-round-add boundary,
        // including FP32 products, instead of allowing a fused multiply-add.
        volatile float unrounded=row<ROWS ? float(P(values[size_t(row)*H+col]))*float(P(scores[token*TOP+slot])) : 0.0f;
        P product=P(unrounded);
        totals[slot%LANES]=P(product+totals[slot%LANES]);
      }
      P total=totals[0];
      for (int lane=1; lane<LANES; ++lane) total=P(totals[lane]+total);
      out[i]=total;
    )");
    auto result = kernel({v, s, order}, {{1, tokens, h}}, {product_type},
                         {tokens * h, 1, 1}, {256, 1, 1},
                         {{"P", product_type},
                          {"TOKENS", tokens},
                          {"H", h},
                          {"ROWS", rows},
                          {"TOP", top}},
                         std::nullopt, false,
                         mlx::core::default_stream(mlx::core::Device::gpu));
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 sorted combine: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

// Sorted, router-generated expert IDs only. The table is bounded by the
// assignment count plus one partial tile per expert; no device readback.
static mlx_array *expert_tiles_impl(mlx_array *ids, int experts,
                                    int tile_rows) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!ids || experts <= 0 || experts > 1024 ||
        !mlx::core::metal::is_nax_available())
      return nullptr;
    auto selected = *reinterpret_cast<array *>(ids);
    int rows = selected.size();
    if (selected.ndim() != 1 || selected.dtype() != mlx::core::uint32 ||
        rows < 256 || rows > 65536)
      return nullptr;
    int count = (rows + tile_rows - 1) / tile_rows + experts;
    static auto kernel = mlx::core::fast::metal_kernel("qwen4_expert_tiles",
                                                       {"ids"}, {"tiles"}, R"(
      threadgroup uint starts[E], ends[E], prefix[E], scratch[E];
      uint lane = thread_index_in_threadgroup;
      for (uint e = lane; e < E; e += 256) {
        uint lo = 0, hi = R;
        while (lo < hi) { uint mid = (lo + hi) / 2; if (ids[mid] < e) lo = mid + 1; else hi = mid; }
        starts[e] = lo; hi = R;
        while (lo < hi) { uint mid = (lo + hi) / 2; if (ids[mid] <= e) lo = mid + 1; else hi = mid; }
        ends[e] = lo; prefix[e] = (lo - starts[e] + BM - 1) / BM;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = 1; stride < E; stride *= 2) {
        for (uint e = lane; e < E; e += 256) scratch[e] = prefix[e] + (e >= stride ? prefix[e-stride] : 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < E; e += 256) prefix[e] = scratch[e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint e = lane; e < E; e += 256) {
        uint tile = e ? prefix[e-1] : 0;
        for (uint row = starts[e]; row < ends[e]; row += BM, ++tile) {
          tiles[2*tile] = row; tiles[2*tile+1] = min(row+BM,ends[e]);
        }
      }
      for (uint tile = prefix[E-1]+lane; tile < NT; tile += 256) {
        tiles[2*tile] = 0; tiles[2*tile+1] = 0;
      }
    )");
    auto result = kernel(
        {selected}, {{count, 2}}, {mlx::core::uint32}, {256, 1, 1}, {256, 1, 1},
        {{"R", rows}, {"E", experts}, {"NT", count}, {"BM", tile_rows}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 expert tiles: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

mlx_array *mlx_qwen4_expert_tiles(mlx_array *ids, int experts) {
  return expert_tiles_impl(ids, experts, 64);
}

mlx_array *mlx_qwen4_expert_prefill(mlx_array *x, mlx_array *ids,
                                    mlx_array *tile_table, mlx_array *weight,
                                    mlx_array *scales, mlx_array *biases,
                                    int experts, int group, int bits,
                                    const char *mode) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!x || !ids || !tile_table || !weight || !scales || !biases || !mode ||
        experts <= 0 || !mlx::core::metal::is_nax_available())
      return nullptr;
    auto input = *reinterpret_cast<array *>(x),
         selected = *reinterpret_cast<array *>(ids);
    auto tiles = *reinterpret_cast<array *>(tile_table),
         w = *reinterpret_cast<array *>(weight);
    auto s = *reinterpret_cast<array *>(scales),
         b = *reinterpret_cast<array *>(biases);
    auto quant = mlx::core::string_to_quantization_mode(mode);
    bool affine = quant == mlx::core::QuantizationMode::Affine;
    int ratio = mlx::core::quant_super_ratio(quant);
    bool has_min = mlx::core::quant_has_sub_min(quant);
    if (input.ndim() != 3 || input.shape(1) != 1 ||
        input.dtype() != mlx::core::bfloat16 || selected.ndim() != 1 ||
        selected.dtype() != mlx::core::uint32 || tiles.ndim() != 2 ||
        tiles.shape(1) != 2 || tiles.dtype() != mlx::core::uint32 ||
        w.ndim() != 2 || w.dtype() != mlx::core::uint32 ||
        w.shape(0) % experts || s.ndim() != 2 || b.ndim() != 2 ||
        b.dtype() != mlx::core::float16)
      return nullptr;
    int rows = input.shape(0), k = input.shape(2), n = w.shape(0) / experts;
    if (rows < 256 || rows > 65536 || rows / experts < 4 ||
        selected.size() != rows || k <= 0 || k % 64 || n <= 0 || n % 64 ||
        group != 32 || tiles.shape(0) != (rows + 63) / 64 + experts)
      return nullptr;
    if (affine) {
      if ((bits != 5 && bits != 8) || s.dtype() != mlx::core::float16 ||
          s.shape() != b.shape() || s.shape(0) != w.shape(0) ||
          s.shape(1) != k / group || !mlx::core::env::enable_tf32())
        return nullptr;
    } else {
      if ((bits != 4 && bits != 5) || ratio != 8 || !has_min ||
          s.dtype() != mlx::core::uint8 || s.shape(0) != w.shape(0) ||
          s.shape(1) != 2 * k / group || b.shape(0) != w.shape(0) ||
          b.shape(1) != 2 * k / (group * ratio))
        return nullptr;
    }
    if (w.shape(1) != k * bits / 32)
      return nullptr;
    auto compute = affine ? mlx::core::float32 : input.dtype();
    const auto &header = qwen4_prefill_header();
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_expert_prefill", {"x", "ids", "tiles", "w", "scales", "biases"},
        {"out"},
#include "metal/qwen4_expert_prefill.metal.inc"
        , header);
    auto result = kernel(
        {input, selected, tiles, w, s, b}, {{rows, 1, n}}, {input.dtype()},
        {32 * (n / 64), 4 * tiles.shape(0), 1}, {32, 4, 1},
        {{"C", compute},
         {"T", input.dtype()},
         {"K", k},
         {"N", n},
         {"E", experts},
         {"R", rows},
         {"GS", group},
         {"BITS", bits},
         {"AFFINE", affine},
         {"SR", ratio},
         {"HM", has_min}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 expert prefill: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}
}

extern "C" mlx_array *mlx_qwen4_prefill_indirect(
    mlx_array *input, mlx_array *indices, mlx_array *token_rows, mlx_array *wg,
    mlx_array *sg, mlx_array *bg, mlx_array *wu, mlx_array *su, mlx_array *bu,
    mlx_array *wd, mlx_array *sd, mlx_array *bd, int experts) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!input || !indices || !token_rows || !wg || !sg || !bg || !wu || !su ||
        !bu || !wd || !sd || !bd || experts <= 0 || experts > 1024 ||
        !mlx::core::metal::is_nax_available() || !mlx::core::env::enable_tf32())
      return nullptr;
    auto x = *reinterpret_cast<array *>(input),
         ids = *reinterpret_cast<array *>(indices),
         rows = *reinterpret_cast<array *>(token_rows);
    std::vector<array> w;
    for (auto p : {wg, sg, bg, wu, su, bu, wd, sd, bd})
      w.push_back(*reinterpret_cast<array *>(p));
    if (x.ndim() != 2 || x.dtype() != mlx::core::bfloat16 || x.shape(0) < 1 ||
        ids.ndim() != 1 || ids.dtype() != mlx::core::uint32 ||
        rows.shape() != ids.shape() || rows.dtype() != mlx::core::uint32 ||
        ids.size() < 256 || ids.size() > 65536 || ids.size() / experts < 4 ||
        std::any_of(
            w.begin(), w.end(), [](const array &a) { return a.ndim() != 2; }))
      return nullptr;
    int h = x.shape(1), r = ids.size();
    if (h <= 0 || h % 256 || w[0].shape(0) % experts)
      return nullptr;
    int m = w[0].shape(0) / experts;
    if (m <= 0 || m % 64)
      return nullptr;
    int gb = w[0].shape(1) * 32 / h, db = w[6].shape(1) * 32 / m;
    if ((gb != 4 && gb != 5) || (db != 5 && db != 8) ||
        w[0].shape(1) != h * gb / 32 || w[3].shape() != w[0].shape() ||
        w[6].shape() != mlx::core::Shape{experts * h, m * db / 32})
      return nullptr;
    for (int i : {0, 3}) {
      if (w[i].dtype() != mlx::core::uint32 ||
          w[i + 1].dtype() != mlx::core::uint8 ||
          w[i + 2].dtype() != mlx::core::float16 ||
          w[i + 1].shape() != mlx::core::Shape{experts * m, h / 16} ||
          w[i + 2].shape() != mlx::core::Shape{experts * m, h / 128})
        return nullptr;
    }
    if (w[6].dtype() != mlx::core::uint32 ||
        w[7].dtype() != mlx::core::float16 ||
        w[8].dtype() != mlx::core::float16 ||
        w[7].shape() != mlx::core::Shape{experts * h, m / 32} ||
        w[8].shape() != w[7].shape())
      return nullptr;
    std::unique_ptr<array> table(
        reinterpret_cast<array *>(expert_tiles_impl(indices, experts, 32)));
    if (!table)
      return nullptr;
    static auto gate_up =
        mlx::core::fast::metal_kernel("qwen4_prefill_indirect_gate_up",
                                      {"x", "ids", "token_rows", "tiles", "wg",
                                       "sgs", "bg", "wu", "sus", "bu"},
                                      {"out"},
#include "metal/qwen4_prefill_gate_up.metal.inc"
                                      , qwen4_prefill_header());
    const char *hoist_flag = std::getenv("MLX_QWEN4_PREFILL_HOIST_ZERO");
    const int hoist_zero =
        !(hoist_flag && hoist_flag[0] == '0' && hoist_flag[1] == '\0');
    auto activated = gate_up(
        {x, ids, rows, *table, w[0], w[1], w[2], w[3], w[4], w[5]}, {{r, 1, m}},
        {x.dtype()}, {32 * (m / 64), 4 * table->shape(0), 1}, {32, 4, 1},
        {{"T", x.dtype()},
         {"S", x.shape(0)},
         {"R", r},
         {"E", experts},
         {"N", m},
         {"K", h},
         {"BITS", gb},
         {"HOIST_ZERO", hoist_zero}},
        std::nullopt, false,
        mlx::core::default_stream(mlx::core::Device::gpu))[0];
    static auto down = mlx::core::fast::metal_kernel(
        "qwen4_prefill_indirect_down",
        {"x", "ids", "tiles", "w", "scales", "biases"}, {"out"},
#include "metal/qwen4_prefill_down.metal.inc"
        , qwen4_prefill_header());
    auto out =
        down({activated, ids, *table, w[6], w[7], w[8]}, {{r, 1, h}},
             {x.dtype()}, {32 * (h / 128), 4 * table->shape(0), 1}, {32, 4, 1},
             {{"T", x.dtype()},
              {"R", r},
              {"E", experts},
              {"N", h},
              {"K", m},
              {"BITS", db}},
             std::nullopt, false,
             mlx::core::default_stream(mlx::core::Device::gpu))[0];
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 indirect prefill: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}
