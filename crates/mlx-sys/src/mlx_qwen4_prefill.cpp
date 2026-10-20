#include "mlx_common.h"
#include "mlx_kquant.h"
#include <cstdlib>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"
#include "metal/common/native_activations.h"
#include "metal/common/quantized.h"
#endif

#ifdef MLX_NODE_METAL_ENABLED
using mlx::core::metal::native_bf16_silu_table;
using mlx::core::quantized_preamble::nax_header;

template <int BM, bool ACT = false>
static std::vector<array>
qwen4_dense_prefill_graph(const std::vector<array> &a) {
  const auto &x = a[0];
  const auto &w = a[1];
  const auto &s = a[2];
  const auto &b = a[3];
  int k = x.shape(-1), n = w.shape(0), rows = x.size() / k;
  static auto kernel = mlx::core::fast::metal_kernel(
      "qwen4_dense_prefill", {"x", "w", "scales", "biases", "silu_table"},
      {"out"},
#include "metal/qwen4/dense_prefill.metal.inc"
      , nax_header());
  auto shape = x.shape();
  shape.back() = n;
  return kernel({x, w, s, b, ACT ? a[4] : x}, {shape}, {mlx::core::bfloat16},
                {32 * (n / 64), 4 * ((rows + BM - 1) / BM), 1}, {32, 4, 1},
                {{"ACT", ACT},
                 {"M_TILE", BM},
                 {"C", mlx::core::float32},
                 {"T", mlx::core::bfloat16},
                 {"K", k},
                 {"N", n},
                 {"GS", 32},
                 {"BITS", 8}},
                std::nullopt, false,
                mlx::core::default_stream(mlx::core::Device::gpu));
}

template <int BM, bool ACT = false>
static std::vector<array>
qwen4_dense_prefill_dispatch(const std::vector<array> &a) {
  static auto compiled = mlx::core::compile(qwen4_dense_prefill_graph<BM, ACT>);
  return compiled(a);
}

static std::vector<array>
qwen4_indirect_prefill_graph(const std::vector<array> &a) {
  const auto &x = a[0];
  const auto &ids = a[1];
  const auto &rows = a[2];
  const auto &table = a[3];
  std::vector<array> w(a.begin() + 4, a.end());
  int h = x.shape(1), r = ids.size(), experts = table.shape(0) - (r + 31) / 32;
  int m = w[0].shape(0) / experts, gb = w[0].shape(1) * 32 / h,
      db = w[6].shape(1) * 32 / m;
  static auto gate_up = mlx::core::fast::metal_kernel(
      "qwen4_prefill_indirect_gate_up",
      {"x", "ids", "token_rows", "tiles", "wg", "sgs", "bg", "wu", "sus", "bu"},
      {"out"},
#include "metal/common/quantized_expert_gate_up.metal.inc"
      , nax_header());
  auto activated = gate_up(
      {x, ids, rows, table, w[0], w[1], w[2], w[3], w[4], w[5]}, {{r, 1, m}},
      {x.dtype()}, {32 * (m / 64), 4 * table.shape(0), 1}, {32, 4, 1},
      {{"T", x.dtype()}, {"N", m}, {"K", h}, {"BITS", gb}}, std::nullopt,
      false, mlx::core::default_stream(mlx::core::Device::gpu))[0];
  static auto down = mlx::core::fast::metal_kernel(
      "qwen4_prefill_indirect_down",
      {"x", "ids", "tiles", "w", "scales", "biases"}, {"out"},
#include "metal/common/affine_expert_down.metal.inc"
      , nax_header());
  auto out =
      down({activated, ids, table, w[6], w[7], w[8]}, {{r, 1, h}}, {x.dtype()},
           {32 * (h / 128), 4 * table.shape(0), 1}, {32, 4, 1},
           {{"T", x.dtype()},
            {"N", h},
            {"K", m},
            {"BITS", db},
            {"C", mlx::core::float32}},
           std::nullopt, false,
           mlx::core::default_stream(mlx::core::Device::gpu))[0];
  return {out};
}

static std::vector<array> qwen4_indirect_prefill(const std::vector<array> &a) {
  static auto compiled = mlx::core::compile(qwen4_indirect_prefill_graph);
  return compiled(a);
}

#endif

extern "C" {

// Match the ordinary affine8 NAX matrix path while avoiding global F32 copies.
// Split-K shapes keep their existing partitioning and reduction arithmetic.
static mlx_array *qwen4_dense_prefill_impl(mlx_array *input, mlx_array *weight,
                                           mlx_array *scales, mlx_array *biases,
                                           bool activate) {
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
    // The reference's paired shared projection also uses plain NAX. Its
    // 640/1280 output columns have the same independent 64-column tiles.
    bool shared_shape = !activate && rows >= 512 && rows <= 1024 && k == 2560 &&
                        (n == 640 || n == 1280);
    if ((n < 2048 && (rows < 1024 || k < 8192) && !shared_shape) ||
        w.shape(1) != k / 4 || s.shape(0) != n || s.shape(1) != k / 32)
      return nullptr;
    // Same selection as QuantizedMatmul::qmm_splitk; it falls through to NAX
    // only when there is one partition. Shape tails remain bounds checked.
    int partitions = std::max(1, 512 / (((n + 31) / 32) * ((rows + 31) / 32)));
    partitions = std::min(partitions, k / 32);
    while (partitions > 1 && k % (partitions * 32))
      --partitions;
    if (partitions > 1)
      return nullptr;
    if (activate) {
      if (n != 320 || k != 10240)
        return nullptr;
      auto out = qwen4_dense_prefill_dispatch<32, true>(
          {x, w, s, b, native_bf16_silu_table()});
      return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
    }
    // The 320-column mixer projection fills the GPU only with 32-row tiles.
    const bool bm32 = n == 320 && k == 10240;
    auto result = bm32 ? qwen4_dense_prefill_dispatch<32>({x, w, s, b})
                       : qwen4_dense_prefill_dispatch<64>({x, w, s, b});
    return reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 compact dense prefill: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}

mlx_array *mlx_qwen4_dense_prefill(mlx_array *input, mlx_array *weight,
                                   mlx_array *scales, mlx_array *biases) {
  return qwen4_dense_prefill_impl(input, weight, scales, biases, false);
}
mlx_array *mlx_qwen4_prefill_mixer_act(mlx_array *input, mlx_array *weight,
                                       mlx_array *scales, mlx_array *biases) {
  return qwen4_dense_prefill_impl(input, weight, scales, biases, true);
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
      const int ROWS=values_shape[0], TOKENS=ROWS/TOP;
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
    auto result = kernel(
        {v, s, order}, {{1, tokens, h}}, {product_type}, {tokens * h, 1, 1},
        {256, 1, 1}, {{"P", product_type}, {"H", h}, {"TOP", top}},
        std::nullopt, false, mlx::core::default_stream(mlx::core::Device::gpu));
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
      const int R = ids_shape[0], NT = (R + BM - 1) / BM + E;
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
    auto result =
        kernel({selected}, {{count, 2}}, {mlx::core::uint32}, {256, 1, 1},
               {256, 1, 1}, {{"E", experts}, {"BM", tile_rows}}, std::nullopt,
               false, mlx::core::default_stream(mlx::core::Device::gpu));
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
    auto quant = mlx::core::kquant::parse_mode(mode);
    bool affine = std::string_view(mode) == "affine";
    if (!affine && !quant)
      return nullptr;
    if (quant) {
      // ("iq3s", bits 8) is the legacy expanded import (mlx_kquant.h).
      quant = mlx::core::kquant::resolve_mode(*quant, bits);
    }
    int ratio = quant ? mlx::core::kquant::super_ratio(*quant) : 0;
    bool has_min = quant && mlx::core::kquant::has_sub_min(*quant);
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
    const auto &header = nax_header();
    static auto kernel = mlx::core::fast::metal_kernel(
        "qwen4_expert_prefill", {"x", "ids", "tiles", "w", "scales", "biases"},
        {"out"},
#include "metal/common/quantized_expert_prefill.metal.inc"
        , header);
    auto result = kernel(
        {input, selected, tiles, w, s, b}, {{rows, 1, n}}, {input.dtype()},
        {32 * (n / 64), 4 * tiles.shape(0), 1}, {32, 4, 1},
        {{"C", compute},
         {"T", input.dtype()},
         {"K", k},
         {"N", n},
         {"E", experts},
         {"GS", group},
         {"BITS", bits},
         {"AFFINE", affine},
         {"SR", ratio},
         {"HM", has_min},
         {"KIND", quant ? static_cast<int>(mlx::core::kquant::kind(*quant)) : 0},
         {"SHIFT", quant ? mlx::core::kquant::scale_shift(*quant) : 0}},
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
    std::vector<array> a{x, ids, rows, *table};
    a.insert(a.end(), w.begin(), w.end());
    auto result = qwen4_indirect_prefill(a);
    auto out = std::move(result[0]);
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 indirect prefill: " << e.what() << std::endl;
  }
#endif
  return nullptr;
}
