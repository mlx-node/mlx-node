#include "mlx_common.h"
#include <cstdlib>
#include <cstring>
#include <memory>

#ifdef MLX_NODE_METAL_ENABLED
#include "metal/common/quantized.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
using mlx::core::Dtype;
namespace {
using mlx::core::quantized_preamble::nax_header;
using mlx::core::quantized_preamble::qmv_header;

enum Hy4KernelId : int {
  kKernelHcMixReduce = 0,
  kKernelHcPostIdentity = 1,
  kKernelBf16LmHead = 2,
  kKernelQ8Decode = 3,
  kKernelKqGateUp = 4,
  kKernelKqDownCombine = 5,
  kKernelKqDownWeighted = 6,
  kKernelMlaSink = 7,
  kKernelMlaSplitK = 8,
  kKernelMlaMerge = 9,
  kKernelQ8Prefill = 10,
  kKernelMlaPrefillNax = 11,
  kKernelKqPrefillIndirectGu = 12,
  kKernelKqSortedCombine = 13,
  kKernelRouterTopk = 14,
};

struct CachedPrimKey {
  int kernel_id;
  int stream_index;
  int gx;
  int gy;
  int gz;
  int tx;
  int ty;
  int tz;
  uint64_t tmpl_key;
  uint32_t const_mask;

  bool operator==(const CachedPrimKey &o) const {
    return kernel_id == o.kernel_id && stream_index == o.stream_index &&
           gx == o.gx && gy == o.gy && gz == o.gz && tx == o.tx &&
           ty == o.ty && tz == o.tz && tmpl_key == o.tmpl_key &&
           const_mask == o.const_mask;
  }
};

struct CachedPrimKeyHash {
  size_t operator()(const CachedPrimKey &k) const noexcept {
    uint64_t h = uint64_t(uint32_t(k.kernel_id)) |
                 (uint64_t(uint32_t(k.stream_index)) << 32);
    auto mix = [&h](uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(uint64_t(uint32_t(k.gx)) | (uint64_t(uint32_t(k.gy)) << 32));
    mix(uint64_t(uint32_t(k.gz)) | (uint64_t(uint32_t(k.tx)) << 32));
    mix(uint64_t(uint32_t(k.ty)) | (uint64_t(uint32_t(k.tz)) << 32));
    mix(k.tmpl_key);
    mix(uint64_t(k.const_mask));
    return size_t(h);
  }
};

// metal_kernel() switches buffer address space between `constant` (size < 8)
// and `device` (size >= 8), and scalar `&` (ndim == 0) vs pointer `*`.
uint32_t compute_const_mask(const std::vector<array> &inputs) {
  uint32_t mask = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].size() < 8) {
      mask |= (1u << i);
    }
    if (inputs[i].ndim() == 0) {
      mask |= (1u << (16u + i));
    }
  }
  return mask;
}

// Reusing CustomKernel primitives avoids metal_kernel()'s per-dispatch source
// concatenation and signature hashing.
template <typename MakeFn>
std::vector<array> dispatch_cached_kernel(
    int kernel_id, const Stream &stream, std::tuple<int, int, int> grid,
    std::tuple<int, int, int> tg, uint64_t tmpl_key,
    const std::vector<array> &inputs, const std::vector<Shape> &out_shapes,
    const std::vector<Dtype> &out_dtypes, MakeFn &&make_outputs) {
  const CachedPrimKey key{
      kernel_id,
      stream.index,
      std::get<0>(grid),
      std::get<1>(grid),
      std::get<2>(grid),
      std::get<0>(tg),
      std::get<1>(tg),
      std::get<2>(tg),
      tmpl_key,
      compute_const_mask(inputs),
  };
  static std::mutex mutex;
  static std::unordered_map<CachedPrimKey,
                            std::shared_ptr<mlx::core::Primitive>,
                            CachedPrimKeyHash>
      cache;
  std::shared_ptr<mlx::core::Primitive> prim;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = cache.find(key); it != cache.end()) {
      prim = it->second;
    }
  }
  if (prim) {
    if (out_shapes.size() == 1) {
      return {array(out_shapes[0], out_dtypes[0], std::move(prim), inputs)};
    }
    return array::make_arrays(out_shapes, out_dtypes, std::move(prim), inputs);
  }
  auto res = make_outputs();
  if (!res.empty() && res[0].has_primitive()) {
    std::lock_guard<std::mutex> lock(mutex);
    if (cache.size() < 4096) {
      cache.emplace(key, res[0].primitive_ptr());
    }
  }
  return res;
}

uint32_t float_bits(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

array cached_f32_scalar(float v) {
  const uint32_t key = float_bits(v);
  static std::mutex mutex;
  static std::unordered_map<uint32_t, array> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  array arr(&v, {1}, mlx::core::float32);
  if (cache.size() < 128) {
    cache.emplace(key, arr);
  }
  return arr;
}

struct Float3Key {
  uint32_t a;
  uint32_t b;
  uint32_t c;

  bool operator==(const Float3Key &o) const {
    return a == o.a && b == o.b && c == o.c;
  }
};

struct Float3KeyHash {
  size_t operator()(const Float3Key &k) const noexcept {
    uint64_t h = uint64_t(k.a) | (uint64_t(k.b) << 32);
    h ^= uint64_t(k.c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

array cached_i32_pair(int a, int b) {
  const uint64_t key =
      (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
  static std::mutex mutex;
  static std::unordered_map<uint64_t, array> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const int32_t vals[2] = {int32_t(a), int32_t(b)};
  array arr(vals, {2}, mlx::core::int32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_f32_pair(float a, float b) {
  const uint64_t key =
      (uint64_t(float_bits(a)) << 32) | uint64_t(float_bits(b));
  static std::mutex mutex;
  static std::unordered_map<uint64_t, array> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const float vals[2] = {a, b};
  array arr(vals, {2}, mlx::core::float32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_f32_triple(float a, float b, float c) {
  const Float3Key key{float_bits(a), float_bits(b), float_bits(c)};
  static std::mutex mutex;
  static std::unordered_map<Float3Key, array, Float3KeyHash> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  const float vals[3] = {a, b, c};
  array arr(vals, {3}, mlx::core::float32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

array cached_row_ids(int rows, int top) {
  const int n_assign = rows * top;
  if (n_assign > 512) {
    std::vector<uint32_t> rids(static_cast<size_t>(n_assign));
    for (int r = 0; r < rows; ++r) {
      for (int t = 0; t < top; ++t) {
        rids[static_cast<size_t>(r * top + t)] = static_cast<uint32_t>(r);
      }
    }
    return array(rids.data(), {n_assign}, mlx::core::uint32);
  }
  const uint64_t key =
      (uint64_t(uint32_t(rows)) << 32) | uint64_t(uint32_t(top));
  static std::mutex mutex;
  static std::unordered_map<uint64_t, array> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  std::vector<uint32_t> rids(static_cast<size_t>(n_assign));
  for (int r = 0; r < rows; ++r) {
    for (int t = 0; t < top; ++t) {
      rids[static_cast<size_t>(r * top + t)] = static_cast<uint32_t>(r);
    }
  }
  array arr(rids.data(), {n_assign}, mlx::core::uint32);
  if (cache.size() < 64) {
    cache.emplace(key, arr);
  }
  return arr;
}

const array &cached_zero_f16() {
  static const array arr = [] {
    const mlx::core::float16_t zeros[8] = {
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
        mlx::core::float16_t(0.0f), mlx::core::float16_t(0.0f),
    };
    return array(zeros, {8}, mlx::core::float16);
  }();
  return arr;
}

array ensure_contiguous_aligned(const array &a, size_t elem_align) {
  if (a.flags().row_contiguous &&
      (elem_align <= 1 || a.offset() % elem_align == 0)) {
    return a;
  }
  return copy(a);
}

const mlx::core::fast::CustomKernelFunction &hy4_hc_mix_reduce_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_hc_mix_reduce", {"x", "weight", "scale", "base", "params"},
      {"h", "post"},
#include "metal/hy4/hc_mix_reduce.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_hc_post_identity_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_hc_post_identity", {"y", "res", "post"}, {"out"},
#include "metal/hy4/hc_post_identity.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_bf16_lm_head_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_bf16_lm_head", {"x", "w"}, {"out"},
#include "metal/hy4/bf16_lm_head.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_decode_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dense_decode", {"x", "w", "scales", "biases"}, {"out"},
#include "metal/hy4/q8_dense_decode.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_prefill_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dense_prefill", {"x", "w", "scales", "biases"}, {"out"},
#include "metal/hy4/q8_dense_prefill.metal.inc"
      , nax_header());
  return kernel;
}

mlx_array *hy4_q8_prefill_impl(mlx_array *x, mlx_array *w, mlx_array *scales,
                               mlx_array *biases, bool grouped) {
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!x || !w || !scales) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    if (xa.ndim() != (grouped ? 3 : 2) || wa.ndim() != 2 || sa.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int g = grouped ? xa.shape(1) : 1;
    const int k = grouped ? xa.shape(2) : xa.shape(1);
    if (m < 1 || g < 1 || k < 64 || k % 64 != 0 || wa.shape(0) < g ||
        wa.shape(0) % g != 0 || wa.shape(1) != k / 4) {
      return nullptr;
    }
    const int n = wa.shape(0) / g;
    if (n < 1 || sa.shape() != Shape{g * n, k / 32}) {
      return nullptr;
    }
    if (size_t(g) * size_t(k) > (1ULL << 25) ||
        size_t(g) * size_t(n) > (1ULL << 25)) {
      return nullptr;
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != sa.shape()) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }

    array xc = ensure_contiguous_aligned(xa, 1);
    array wc = ensure_contiguous_aligned(wa, 4);
    array sc = ensure_contiguous_aligned(sa, 1);

    const int bm = (m <= 48 || (g == 1 && n <= 256)) ? 32 : 64;
    const bool aligned_m = (m % bm == 0);
    const bool aligned_n = (n % 64 == 0);
    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    const int grid_x = 32 * ((n + 63) / 64);
    const int grid_y = 4 * ((m + bm - 1) / bm);
    const int grid_z = g;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{grid_x, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const Dtype compute_dtype = mlx::core::float32;
    const uint64_t tmpl_key = uint64_t(bm) | (uint64_t(grouped ? 1 : 0) << 8) |
                              (uint64_t(sym_f16 ? 1 : 0) << 9) |
                              (uint64_t(aligned_n ? 1 : 0) << 10) |
                              (uint64_t(scales_f16 ? 1 : 0) << 11) |
                              (uint64_t(aligned_m ? 1 : 0) << 12);
    const std::vector<array> inputs{xc, wc, sc, bc};
    const std::vector<Shape> out_shapes{grouped ? Shape{m, g, n} : Shape{m, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    auto out = dispatch_cached_kernel(
        kKernelQ8Prefill, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_prefill_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"M_TILE", bm},
               {"GROUPED", grouped},
               {"SYM_F16", sym_f16},
               {"ALIGNED_M", aligned_m},
               {"ALIGNED_N", aligned_n},
               {"C", compute_dtype}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << (grouped ? "Hy4 q8_grouped_prefill: "
                          : "Hy4 q8_dense_prefill: ")
              << e.what() << std::endl;
    return nullptr;
  }
}

const std::string &hy4_kq_expert_header() {
  static const std::string header = qmv_header() +
#include "metal/hy4/kq_expert_decode.metal.inc"
      ;
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_gate_up_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_gate_up_swiglu",
      {"x", "row_ids", "slots", "wg", "sg", "bg", "wu", "su", "bu", "limit"},
      {"out"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_kq_gate_up_swiglu_impl<FAST, DEDUP>(
          x, row_ids, slots, wg, sg, bg, wu, su, bu, limit[0], out,
          x_shape[0], slots_shape[0], x_shape[1], wg_shape[1], wg_shape[0],
          assign, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_kq_expert_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_down_combine_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_down_combine", {"hidden", "slots", "weights", "wd", "sd", "bd"},
      {"out"},
      R"METAL(
      threadgroup float products[32 * RPS];
      hy4_kq_down_combine_impl<BITS, GS, SR, HM, FAST, RPS, KSG>(
          hidden, slots, weights, wd,
          reinterpret_cast<const device uint8_t*>(sd), bd, out, products,
          slots_shape[0], slots_shape[1], hidden_shape[1], wd_shape[1], wd_shape[0],
          threadgroup_position_in_grid.z, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_kq_expert_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_down_weighted_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_down_weighted", {"hidden", "slots", "weights", "wd", "sd", "bd"},
      {"out"},
      R"METAL(
      const uint assign = threadgroup_position_in_grid.z;
      if (assign >= uint(slots_shape[0])) return;
      hy4_kq_down_weighted_impl<BITS, GS, SR, HM, FAST, RPS>(
          hidden, slots[assign], weights[assign], wd,
          reinterpret_cast<const device uint8_t*>(sd), bd, out,
          slots_shape[0], hidden_shape[1], wd_shape[1], wd_shape[0],
          assign, threadgroup_position_in_grid.y,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      )METAL",
      hy4_kq_expert_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &
hy4_kq_prefill_indirect_gu_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_prefill_indirect_gu",
      {"x", "token_rows", "tiles", "wg", "sg", "bg", "wu", "su", "bu", "limit"},
      {"out"},
#include "metal/hy4/kq_prefill_indirect_gu.metal.inc"
      , nax_header());
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_kq_sorted_combine_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_kq_sorted_combine", {"down", "weights", "inverse"}, {"out"},
#include "metal/hy4/kq_sorted_combine.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_router_topk_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_router_topk", {"logits", "bias", "params"}, {"ids", "weights"},
#include "metal/hy4/router_topk.metal.inc"
  );
  return kernel;
}

bool validate_kq_banks(const array &wg, const array &sg, const array &bg,
                       const array &wu, const array &su, const array &bu,
                       const array &wd, const array &sd, const array &bd,
                       int k, int down_bits, int &out_s, int &out_m,
                       int &out_h) {
  if (wg.ndim() != 3 || sg.ndim() != 3 || bg.ndim() != 3 || wu.ndim() != 3 ||
      su.ndim() != 3 || bu.ndim() != 3 || wd.ndim() != 3 || sd.ndim() != 3 ||
      bd.ndim() != 3) {
    return false;
  }
  const int s = wg.shape(0);
  const int m = wg.shape(1);
  const int h = wd.shape(1);
  if (s < 1 || m < 256 || m % 256 != 0 || h < 1 || k < 256 || k % 256 != 0) {
    return false;
  }
  const Shape q4_w{s, m, k / 8};
  const Shape q4_s{s, m, k / 16};
  const Shape q4_b{s, m, k / 128};
  if (wg.dtype() != mlx::core::uint32 || wg.shape() != q4_w ||
      sg.dtype() != mlx::core::uint8 || sg.shape() != q4_s ||
      bg.dtype() != mlx::core::float16 || bg.shape() != q4_b ||
      wu.dtype() != mlx::core::uint32 || wu.shape() != q4_w ||
      su.dtype() != mlx::core::uint8 || su.shape() != q4_s ||
      bu.dtype() != mlx::core::float16 || bu.shape() != q4_b) {
    return false;
  }
  if (down_bits == 4) {
    if (wd.dtype() != mlx::core::uint32 || wd.shape() != Shape{s, h, m / 8} ||
        sd.dtype() != mlx::core::uint8 || sd.shape() != Shape{s, h, m / 16} ||
        bd.dtype() != mlx::core::float16 ||
        bd.shape() != Shape{s, h, m / 128}) {
      return false;
    }
  } else if (down_bits == 6) {
    if (wd.dtype() != mlx::core::uint32 ||
        wd.shape() != Shape{s, h, m * 6 / 32} ||
        sd.dtype() != mlx::core::int8 || sd.shape() != Shape{s, h, m / 16} ||
        bd.dtype() != mlx::core::float16 ||
        bd.shape() != Shape{s, h, m / 256}) {
      return false;
    }
  } else {
    return false;
  }
  out_s = s;
  out_m = m;
  out_h = h;
  return true;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_sink_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_sink_attention", {"q", "k0", "k1", "idx", "sinks", "scale"},
      {"out"},
#include "metal/hy4/mla_sink_attention.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_splitk_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_decode_splitk", {"q", "k0", "k1", "idx", "scale", "split_cfg"},
      {"o_part", "max_part", "sum_part"},
#include "metal/hy4/mla_decode_splitk.metal.inc"
  );
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_merge_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_decode_merge", {"o_part", "max_part", "sum_part", "sinks"},
      {"out"},
#include "metal/hy4/mla_decode_merge.metal.inc"
  );
  return kernel;
}

const std::string &hy4_mla_prefill_nax_header() {
  static const std::string header =
      std::string(mlx::core::quantized_preamble::nax()) + R"METAL(
using namespace mlx::steel;

struct Hy4MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct Hy4SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct Hy4MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct Hy4Exp2SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::exp2(x - y);
  }
};

template <typename C, bool SPLIT, int D_K, int HALF>
METAL_FUNC void hy4_load_kv_frag(
    thread metal::vec<C, 8>& dst,
    const device half* k0,
    const device half* k1,
    const int slot_a,
    const int slot_b,
    const int col_off,
    const short fn) {
  const device half* plane = (SPLIT && col_off >= HALF) ? k1 : k0;
  const int stride = SPLIT ? HALF : D_K;
  const int pcol = (SPLIT && col_off >= HALF) ? (col_off - HALF) : col_off;
  if (slot_a >= 0) {
    const half4 va = *reinterpret_cast<const device half4*>(
        plane + size_t(slot_a) * size_t(stride) + size_t(pcol + int(fn)));
    dst[0] = C(va.x);
    dst[1] = C(va.y);
    dst[2] = C(va.z);
    dst[3] = C(va.w);
  } else {
    dst[0] = C(0);
    dst[1] = C(0);
    dst[2] = C(0);
    dst[3] = C(0);
  }
  if (slot_b >= 0) {
    const half4 vb = *reinterpret_cast<const device half4*>(
        plane + size_t(slot_b) * size_t(stride) + size_t(pcol + int(fn)));
    dst[4] = C(vb.x);
    dst[5] = C(vb.y);
    dst[6] = C(vb.z);
    dst[7] = C(vb.w);
  } else {
    dst[4] = C(0);
    dst[5] = C(0);
    dst[6] = C(0);
    dst[7] = C(0);
  }
}
)METAL";
  return header;
}

const mlx::core::fast::CustomKernelFunction &hy4_mla_prefill_nax_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_mla_prefill_nax", {"q", "k0", "k1", "idx", "sinks", "scale"},
      {"out"},
#include "metal/hy4/mla_prefill_nax.metal.inc"
      , hy4_mla_prefill_nax_header());
  return kernel;
}

array hy4_mla_single_pass(const array &q, const array &k0, const array &k1,
                          const array &idx, const array &sinks,
                          const array &scale, int d_v, int d_rope, bool split,
                          int heads, int bn, const Stream &stream) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  const int tg_threads = bn * 32;
  const int head_groups = h / heads;
  const std::tuple<int, int, int> grid{tg_threads, head_groups, s};
  const std::tuple<int, int, int> tg{tg_threads, 1, 1};
  const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                            (uint64_t(split ? 1 : 0) << 32) |
                            (uint64_t(heads) << 36) | (uint64_t(bn) << 44);
  const std::vector<array> inputs{q, k0, k1, idx, sinks, scale};
  const std::vector<Shape> out_shapes{{s, h, d_v}};
  const std::vector<Dtype> out_dtypes{mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaSink, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_sink_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg,
            {{"D_V", d_v},
             {"D_ROPE", d_rope},
             {"SPLIT", split},
             {"HEADS", heads},
             {"BN", bn}},
            std::nullopt, false, stream);
      })[0];
}

std::vector<array> hy4_mla_splitk_pass1(const array &q, const array &k0,
                                        const array &k1, const array &idx,
                                        const array &scale,
                                        const array &split_cfg, int d_v,
                                        int d_rope, bool split, int heads,
                                        int bn, int n_splits,
                                        const Stream &stream) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  const int tg_threads = bn * 32;
  const int head_groups = h / heads;
  const std::tuple<int, int, int> grid{tg_threads, head_groups * n_splits, s};
  const std::tuple<int, int, int> tg{tg_threads, 1, 1};
  const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                            (uint64_t(split ? 1 : 0) << 32) |
                            (uint64_t(heads) << 36) | (uint64_t(bn) << 44);
  const std::vector<array> inputs{q, k0, k1, idx, scale, split_cfg};
  const std::vector<Shape> out_shapes{
      {s, h, n_splits, d_v}, {s, h, n_splits}, {s, h, n_splits}};
  const std::vector<Dtype> out_dtypes{
      mlx::core::float32, mlx::core::float32, mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaSplitK, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_splitk_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg,
            {{"D_V", d_v},
             {"D_ROPE", d_rope},
             {"SPLIT", split},
             {"HEADS", heads},
             {"BN", bn}},
            std::nullopt, false, stream);
      });
}

array hy4_mla_splitk_merge(const array &o_part, const array &max_part,
                           const array &sum_part, const array &sinks, int d_v,
                           const Stream &stream) {
  const int s = o_part.shape(0);
  const int h = o_part.shape(1);
  const int tg_x = d_v / 4;
  const std::tuple<int, int, int> grid{tg_x, h, s};
  const std::tuple<int, int, int> tg{tg_x, 1, 1};
  const uint64_t tmpl_key = uint64_t(d_v);
  const std::vector<array> inputs{o_part, max_part, sum_part, sinks};
  const std::vector<Shape> out_shapes{{s, h, d_v}};
  const std::vector<Dtype> out_dtypes{mlx::core::float32};
  return dispatch_cached_kernel(
      kKernelMlaMerge, stream, grid, tg, tmpl_key, inputs, out_shapes,
      out_dtypes, [&] {
        return hy4_mla_merge_kernel()(
            inputs, out_shapes, out_dtypes, grid, tg, {{"D_V", d_v}},
            std::nullopt, false, stream);
      })[0];
}

int hy4_mla_pipeline_threads(const array &output) {
  static std::mutex mutex;
  static std::unordered_map<std::string, int> cache;
  if (!output.has_primitive()) {
    return 0;
  }
  auto *custom =
      dynamic_cast<mlx::core::fast::CustomKernel *>(&output.primitive());
  if (!custom) {
    return 0;
  }
  const std::string &key = custom->library_name();
  std::lock_guard<std::mutex> lock(mutex);
  if (auto found = cache.find(key); found != cache.end()) {
    return found->second;
  }
  int threads = 0;
  try {
    const auto state = custom->state();
    const auto &name = std::get<0>(state);
    const auto &source = std::get<1>(state);
    auto &device = mlx::core::metal::device(custom->stream().device);
    auto *library = device.get_library(
        key, mlx::core::CompileOptions(std::get<10>(state)),
        [&source] { return mlx::core::metal::utils() + source; });
    auto *pipeline = device.get_kernel(name, library);
    if (pipeline->threadExecutionWidth() == 32 &&
        pipeline->staticThreadgroupMemoryLength() <=
            device.mtl_device()->maxThreadgroupMemoryLength()) {
      threads = int(pipeline->maxTotalThreadsPerThreadgroup());
    }
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_sink_attention: pipeline " << key << ": "
              << e.what() << std::endl;
    return 0;
  }
  cache.emplace(key, threads);
  return threads;
}

struct MlaConfig {
  int heads;
  int bn;
};

bool lookup_mla_config(int h, bool split, bool splitk, MlaConfig &out_cfg) {
  const uint64_t key = uint64_t(uint32_t(h)) |
                       (uint64_t(split ? 1u : 0u) << 32) |
                       (uint64_t(splitk ? 1u : 0u) << 33);
  static std::mutex mutex;
  static std::unordered_map<uint64_t, MlaConfig> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(key); it != cache.end()) {
    out_cfg = it->second;
    return true;
  }
  return false;
}

void store_mla_config(int h, bool split, bool splitk, MlaConfig cfg) {
  const uint64_t key = uint64_t(uint32_t(h)) |
                       (uint64_t(split ? 1u : 0u) << 32) |
                       (uint64_t(splitk ? 1u : 0u) << 33);
  static std::mutex mutex;
  static std::unordered_map<uint64_t, MlaConfig> cache;
  std::lock_guard<std::mutex> lock(mutex);
  cache.emplace(key, cfg);
}
} // namespace
#endif

extern "C" {

bool mlx_hy4_hc_mix_reduce(mlx_array *x, mlx_array *weight, mlx_array *scale,
                           mlx_array *base, float norm_eps, float hc_eps,
                           float magnitude, bool has_post, mlx_array **out_h,
                           mlx_array **out_post) {
  if (out_h)
    *out_h = nullptr;
  if (out_post)
    *out_post = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!x || !weight || !scale || !base || !out_h ||
        (has_post && !out_post)) {
      return false;
    }
    if (!std::isfinite(norm_eps) || !std::isfinite(hc_eps) ||
        !std::isfinite(magnitude)) {
      return false;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(weight);
    const auto &sa = *reinterpret_cast<array *>(scale);
    const auto &ba = *reinterpret_cast<array *>(base);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 1 || ba.ndim() != 1) {
      return false;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::float32 ||
        sa.dtype() != mlx::core::float32 || ba.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = xa.shape(0);
    const int hc = xa.shape(1);
    const int d = xa.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0) {
      return false;
    }
    const int n_gates = has_post ? 2 * hc : hc;
    const int n_scale = has_post ? 2 : 1;
    if (wa.shape(0) != n_gates || wa.shape(1) != hc * d ||
        sa.shape(0) != n_scale || ba.shape(0) != n_gates) {
      return false;
    }
    const bool vec4 = (d % 4 == 0);
    const size_t align = vec4 ? 4 : 1;
    array xc = ensure_contiguous_aligned(xa, align);
    array wc = ensure_contiguous_aligned(wa, align);
    array sc = ensure_contiguous_aligned(sa, 1);
    array bc = ensure_contiguous_aligned(ba, 1);
    array params = cached_f32_triple(norm_eps, hc_eps, magnitude);

    const Shape post_shape = has_post ? Shape{m, hc} : Shape{1};
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{256, m, 1};
    const std::tuple<int, int, int> tg{256, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(has_post ? 1 : 0) << 8) |
                              (uint64_t(vec4 ? 1 : 0) << 9);
    const std::vector<array> inputs{xc, wc, sc, bc, params};
    const std::vector<Shape> out_shapes{{m, d}, post_shape};
    const std::vector<Dtype> out_dtypes{mlx::core::float32, mlx::core::float32};
    auto res = dispatch_cached_kernel(
        kKernelHcMixReduce, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_hc_mix_reduce_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc}, {"HAS_POST", has_post}, {"VEC4", vec4}},
              std::nullopt, false, stream);
        });
    auto h = std::make_unique<array>(std::move(res[0]));
    if (has_post) {
      auto p = std::make_unique<array>(std::move(res[1]));
      *out_post = reinterpret_cast<mlx_array *>(p.release());
    }
    *out_h = reinterpret_cast<mlx_array *>(h.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 hc_mix_reduce: " << e.what() << std::endl;
    return false;
  }
#else
  (void)x;
  (void)weight;
  (void)scale;
  (void)base;
  (void)norm_eps;
  (void)hc_eps;
  (void)magnitude;
  (void)has_post;
  return false;
#endif
}

mlx_array *mlx_hy4_hc_post_identity(mlx_array *y, mlx_array *res,
                                    mlx_array *post) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!y || !res || !post) {
      return nullptr;
    }
    const auto &ya = *reinterpret_cast<array *>(y);
    const auto &ra = *reinterpret_cast<array *>(res);
    const auto &pa = *reinterpret_cast<array *>(post);
    if (ya.ndim() != 2 || ra.ndim() != 3 || pa.ndim() != 2) {
      return nullptr;
    }
    if (ya.dtype() != mlx::core::float32 || ra.dtype() != mlx::core::float32 ||
        pa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = ra.shape(0);
    const int hc = ra.shape(1);
    const int d = ra.shape(2);
    if (m <= 0 || hc <= 0 || hc > 16 || d <= 0 || ya.shape(0) != m ||
        ya.shape(1) != d || pa.shape(0) != m || pa.shape(1) != hc) {
      return nullptr;
    }
    const bool vec4 = (d % 4 == 0);
    const size_t align = vec4 ? 4 : 1;
    array yc = ensure_contiguous_aligned(ya, align);
    array rc = ensure_contiguous_aligned(ra, align);
    array pc = ensure_contiguous_aligned(pa, 1);

    const int cols = vec4 ? (d / 4) : d;
    const int tg_x = std::min(256, cols);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{cols, m, 1};
    const std::tuple<int, int, int> tg{tg_x, 1, 1};
    const uint64_t tmpl_key = uint64_t(hc) | (uint64_t(vec4 ? 1 : 0) << 8);
    const std::vector<array> inputs{yc, rc, pc};
    const std::vector<Shape> out_shapes{{m, hc, d}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelHcPostIdentity, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_hc_post_identity_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"HC", hc}, {"VEC4", vec4}}, std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 hc_post_identity: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)y;
  (void)res;
  (void)post;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_bf16_lm_head(mlx_array *x, mlx_array *w) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    if (xa.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 ||
        wa.dtype() != mlx::core::bfloat16) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int v = wa.shape(0);
    if (m < 1 || m > 16 || k <= 0 || k % 4 != 0 || v <= 0 || wa.shape(1) != k) {
      return nullptr;
    }
    const bool fast_k = (k % 256 == 0);
    const int r_tile = m <= 4 ? m : 4;
    constexpr int rps = 4;
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, fast_k ? 8 : 4);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{
        32, ((v + 7) / 8) * 2, (m + r_tile - 1) / r_tile};
    const std::tuple<int, int, int> tg{32, 2, 1};
    const uint64_t tmpl_key = uint64_t(fast_k ? 1 : 0) |
                              (uint64_t(r_tile) << 4) | (uint64_t(rps) << 12);
    const std::vector<array> inputs{xc, wc};
    const std::vector<Shape> out_shapes{{m, v}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelBf16LmHead, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_bf16_lm_head_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"FAST_K", fast_k}, {"R_TILE", r_tile}, {"RPS", rps}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 bf16_lm_head: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_dense_decode(mlx_array *x, mlx_array *w,
                                   mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !scales) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    if (xa.ndim() != 2 || wa.ndim() != 2 || sa.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int n = wa.shape(0);
    if (m < 1 || m > 16 || k < 32 || k % 32 != 0 || n < 1 ||
        wa.shape(1) != k / 4 || sa.shape() != Shape{n, k / 32}) {
      return nullptr;
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != Shape{n, k / 32}) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 1);
    array sc = ensure_contiguous_aligned(sa, 1);

    const int r_tile = (m <= 8) ? m : 8;
    const int grid_y = 2 * ((n + 7) / 8);
    const int grid_z = (m + r_tile - 1) / r_tile;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 2, 1};
    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    const uint64_t tmpl_key = uint64_t(r_tile) |
                              (uint64_t(sym_f16 ? 1 : 0) << 9) |
                              (uint64_t(scales_f16 ? 1 : 0) << 10);
    const std::vector<array> inputs{xc, wc, sc, bc};
    const std::vector<Shape> out_shapes{{m, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    auto out = dispatch_cached_kernel(
        kKernelQ8Decode, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_decode_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"R_TILE", r_tile}, {"GROUPED", false}, {"SYM_F16", sym_f16}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 q8_dense_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_grouped_decode(mlx_array *x, mlx_array *w,
                                     mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !w || !scales) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int g = xa.shape(1);
    const int k = xa.shape(2);
    if (m < 1 || m > 16 || g < 1 || k < 32 || k % 32 != 0 ||
        wa.shape(0) < g || wa.shape(0) % g != 0 || wa.shape(1) != k / 4) {
      return nullptr;
    }
    const int n = wa.shape(0) / g;
    if (n < 1 || sa.shape() != Shape{g * n, k / 32}) {
      return nullptr;
    }
    const bool sym_f16 = (biases == nullptr);
    array bc = cached_zero_f16();
    if (sym_f16) {
      if (sa.dtype() != mlx::core::float16) {
        return nullptr;
      }
    } else {
      const auto &ba = *reinterpret_cast<array *>(biases);
      if (ba.ndim() != 2 || ba.shape() != Shape{g * n, k / 32}) {
        return nullptr;
      }
      const bool both_f32 = (sa.dtype() == mlx::core::float32 &&
                             ba.dtype() == mlx::core::float32);
      const bool both_f16 = (sa.dtype() == mlx::core::float16 &&
                             ba.dtype() == mlx::core::float16);
      if (!both_f32 && !both_f16) {
        return nullptr;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }
    array xc = ensure_contiguous_aligned(xa, 4);
    array wc = ensure_contiguous_aligned(wa, 1);
    array sc = ensure_contiguous_aligned(sa, 1);

    const int r_tile = (m <= 8) ? m : 8;
    const int grid_y = 2 * ((n + 7) / 8);
    const int grid_z = g * ((m + r_tile - 1) / r_tile);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32, grid_y, grid_z};
    const std::tuple<int, int, int> tg{32, 2, 1};
    const bool scales_f16 = (sa.dtype() == mlx::core::float16);
    const uint64_t tmpl_key = uint64_t(r_tile) | (1ULL << 8) |
                              (uint64_t(sym_f16 ? 1 : 0) << 9) |
                              (uint64_t(scales_f16 ? 1 : 0) << 10);
    const std::vector<array> inputs{xc, wc, sc, bc};
    const std::vector<Shape> out_shapes{{m, g, n}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};

    auto out = dispatch_cached_kernel(
        kKernelQ8Decode, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_q8_decode_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"R_TILE", r_tile}, {"GROUPED", true}, {"SYM_F16", sym_f16}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 q8_grouped_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_dense_prefill(mlx_array *x, mlx_array *w,
                                    mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  return hy4_q8_prefill_impl(x, w, scales, biases, false);
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_q8_grouped_prefill(mlx_array *x, mlx_array *w,
                                      mlx_array *scales, mlx_array *biases) {
#ifdef MLX_NODE_METAL_ENABLED
  return hy4_q8_prefill_impl(x, w, scales, biases, true);
#else
  (void)x;
  (void)w;
  (void)scales;
  (void)biases;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_routed_decode(mlx_array *x, mlx_array *slots,
                                    mlx_array *weights, mlx_array *wg,
                                    mlx_array *sg, mlx_array *bg, mlx_array *wu,
                                    mlx_array *su, mlx_array *bu, mlx_array *wd,
                                    mlx_array *sd, mlx_array *bd, int down_bits,
                                    float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !slots || !weights || !wg || !sg || !bg || !wu || !su || !bu ||
        !wd || !sd || !bd || !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &sla = *reinterpret_cast<array *>(slots);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);
    const auto &wda = *reinterpret_cast<array *>(wd);
    const auto &sda = *reinterpret_cast<array *>(sd);
    const auto &bda = *reinterpret_cast<array *>(bd);

    if (xa.ndim() != 2 || sla.ndim() != 2 || wa.ndim() != 2 ||
        xa.dtype() != mlx::core::float32 || sla.dtype() != mlx::core::uint32 ||
        wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int rows = xa.shape(0);
    const int k = xa.shape(1);
    const int top = sla.shape(1);
    if (rows < 1 || top < 1 || top > 32 || sla.shape(0) != rows ||
        wa.shape() != Shape{rows, top}) {
      return nullptr;
    }
    int s = 0, m = 0, h = 0;
    if (!validate_kq_banks(wga, sga, bga, wua, sua, bua, wda, sda, bda, k,
                           down_bits, s, m, h)) {
      return nullptr;
    }
    (void)s;

    const int n_assign = rows * top;
    array row_ids = cached_row_ids(rows, top);
    array limit_arr = cached_f32_scalar(limit);

    array xc = ensure_contiguous_aligned(xa, 1);
    array slc = ensure_contiguous_aligned(sla, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array wgc = ensure_contiguous_aligned(wga, 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array wdc = ensure_contiguous_aligned(wda, 1);
    array sdc = ensure_contiguous_aligned(sda, 1);
    array bdc = ensure_contiguous_aligned(bda, 1);

    array slots_flat = reshape(slc, {n_assign});
    const bool fast_gu = (m % 8 == 0) && (k % 512 == 0);
    const bool dedup = (rows >= 2) && (n_assign <= 128);
    const uint64_t gu_tmpl =
        uint64_t(fast_gu ? 1 : 0) | (uint64_t(dedup ? 1 : 0) << 1);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> gu_grid{32, ((m + 7) / 8) * 2, n_assign};
    const std::tuple<int, int, int> gu_tg{32, 2, 1};
    const std::vector<array> gu_inputs{xc,  row_ids, slots_flat, wgc, sgc,
                                       bgc, wuc,     suc,        buc, limit_arr};
    const std::vector<Shape> gu_out_shapes{{n_assign, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto hidden = dispatch_cached_kernel(
        kKernelKqGateUp, stream, gu_grid, gu_tg, gu_tmpl, gu_inputs,
        gu_out_shapes, out_dtypes, [&] {
          return hy4_kq_gate_up_kernel()(
              gu_inputs, gu_out_shapes, out_dtypes, gu_grid, gu_tg,
              {{"FAST", fast_gu}, {"DEDUP", dedup}}, std::nullopt, false,
              stream);
        })[0];

    const bool fast_d = (h % 4 == 0) && (m % 512 == 0);
    const int gs = (down_bits == 6) ? 16 : 32;
    const int sr = (down_bits == 6) ? 16 : 8;
    const bool hm = (down_bits == 4);
    constexpr int rps = 4;
    constexpr int ksg = 8;
    const std::tuple<int, int, int> dc_grid{
        32, ((h + rps - 1) / rps) * ksg, rows};
    const std::tuple<int, int, int> dc_tg{32, ksg, 1};
    const uint64_t dc_tmpl = uint64_t(down_bits) | (uint64_t(gs) << 8) |
                             (uint64_t(sr) << 16) |
                             (uint64_t(hm ? 1 : 0) << 24) |
                             (uint64_t(fast_d ? 1 : 0) << 25) |
                             (uint64_t(rps) << 28) | (uint64_t(ksg) << 36);
    const std::vector<array> dc_inputs{hidden, slc, wc, wdc, sdc, bdc};
    const std::vector<Shape> dc_out_shapes{{rows, h}};
    auto out = dispatch_cached_kernel(
        kKernelKqDownCombine, stream, dc_grid, dc_tg, dc_tmpl, dc_inputs,
        dc_out_shapes, out_dtypes, [&] {
          return hy4_kq_down_combine_kernel()(
              dc_inputs, dc_out_shapes, out_dtypes, dc_grid, dc_tg,
              {{"BITS", down_bits},
               {"GS", gs},
               {"SR", sr},
               {"HM", hm},
               {"FAST", fast_d},
               {"RPS", rps},
               {"KSG", ksg}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_routed_decode: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)slots;
  (void)weights;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)wd;
  (void)sd;
  (void)bd;
  (void)down_bits;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_routed_subset(mlx_array *x, mlx_array *row_ids,
                                    mlx_array *slots, mlx_array *weights,
                                    mlx_array *wg, mlx_array *sg, mlx_array *bg,
                                    mlx_array *wu, mlx_array *su, mlx_array *bu,
                                    mlx_array *wd, mlx_array *sd, mlx_array *bd,
                                    int down_bits, float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!x || !row_ids || !slots || !weights || !wg || !sg || !bg || !wu ||
        !su || !bu || !wd || !sd || !bd || !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &rida = *reinterpret_cast<array *>(row_ids);
    const auto &sla = *reinterpret_cast<array *>(slots);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);
    const auto &wda = *reinterpret_cast<array *>(wd);
    const auto &sda = *reinterpret_cast<array *>(sd);
    const auto &bda = *reinterpret_cast<array *>(bd);

    if (xa.ndim() != 2 || rida.ndim() != 1 || sla.ndim() != 1 ||
        wa.ndim() != 1 || xa.dtype() != mlx::core::float32 ||
        rida.dtype() != mlx::core::uint32 || sla.dtype() != mlx::core::uint32 ||
        wa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int rows = xa.shape(0);
    const int k = xa.shape(1);
    const int n_assign = sla.shape(0);
    if (rows < 1 || n_assign < 1 || rida.shape(0) != n_assign ||
        wa.shape(0) != n_assign) {
      return nullptr;
    }
    int s = 0, m = 0, h = 0;
    if (!validate_kq_banks(wga, sga, bga, wua, sua, bua, wda, sda, bda, k,
                           down_bits, s, m, h)) {
      return nullptr;
    }
    (void)s;

    array limit_arr = cached_f32_scalar(limit);
    array xc = ensure_contiguous_aligned(xa, 1);
    array ridc = ensure_contiguous_aligned(rida, 1);
    array slc = ensure_contiguous_aligned(sla, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array wgc = ensure_contiguous_aligned(wga, 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array wdc = ensure_contiguous_aligned(wda, 1);
    array sdc = ensure_contiguous_aligned(sda, 1);
    array bdc = ensure_contiguous_aligned(bda, 1);

    const bool fast_gu = (m % 8 == 0) && (k % 512 == 0);
    const bool dedup = (rows >= 2) && (n_assign <= 128);
    const uint64_t gu_tmpl =
        uint64_t(fast_gu ? 1 : 0) | (uint64_t(dedup ? 1 : 0) << 1);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> gu_grid{32, ((m + 7) / 8) * 2, n_assign};
    const std::tuple<int, int, int> gu_tg{32, 2, 1};
    const std::vector<array> gu_inputs{xc,  ridc, slc, wgc, sgc,
                                       bgc, wuc,  suc, buc, limit_arr};
    const std::vector<Shape> gu_out_shapes{{n_assign, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto hidden = dispatch_cached_kernel(
        kKernelKqGateUp, stream, gu_grid, gu_tg, gu_tmpl, gu_inputs,
        gu_out_shapes, out_dtypes, [&] {
          return hy4_kq_gate_up_kernel()(
              gu_inputs, gu_out_shapes, out_dtypes, gu_grid, gu_tg,
              {{"FAST", fast_gu}, {"DEDUP", dedup}}, std::nullopt, false,
              stream);
        })[0];

    const bool fast_d = (h % 4 == 0) && (m % 512 == 0);
    const int gs = (down_bits == 6) ? 16 : 32;
    const int sr = (down_bits == 6) ? 16 : 8;
    const bool hm = (down_bits == 4);
    constexpr int rps = 4;
    const std::tuple<int, int, int> dw_grid{
        32, ((h + 2 * rps - 1) / (2 * rps)) * 2, n_assign};
    const std::tuple<int, int, int> dw_tg{32, 2, 1};
    const uint64_t dw_tmpl = uint64_t(down_bits) | (uint64_t(gs) << 8) |
                             (uint64_t(sr) << 16) |
                             (uint64_t(hm ? 1 : 0) << 24) |
                             (uint64_t(fast_d ? 1 : 0) << 25) |
                             (uint64_t(rps) << 28);
    const std::vector<array> dw_inputs{hidden, slc, wc, wdc, sdc, bdc};
    const std::vector<Shape> dw_out_shapes{{n_assign, h}};
    auto out = dispatch_cached_kernel(
        kKernelKqDownWeighted, stream, dw_grid, dw_tg, dw_tmpl, dw_inputs,
        dw_out_shapes, out_dtypes, [&] {
          return hy4_kq_down_weighted_kernel()(
              dw_inputs, dw_out_shapes, out_dtypes, dw_grid, dw_tg,
              {{"BITS", down_bits},
               {"GS", gs},
               {"SR", sr},
               {"HM", hm},
               {"FAST", fast_d},
               {"RPS", rps}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_routed_subset: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)row_ids;
  (void)slots;
  (void)weights;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)wd;
  (void)sd;
  (void)bd;
  (void)down_bits;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_mla_sink_attention(mlx_array *q, mlx_array *k0,
                                      mlx_array *k1, mlx_array *idx,
                                      mlx_array *sinks, int v_width,
                                      float scale) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!q || !k0 || !idx || !sinks || !std::isfinite(scale)) {
      return nullptr;
    }
    const auto &qa = *reinterpret_cast<array *>(q);
    const auto &k0a = *reinterpret_cast<array *>(k0);
    const auto &idxa = *reinterpret_cast<array *>(idx);
    const auto &sinksa = *reinterpret_cast<array *>(sinks);
    const bool split = (k1 != nullptr);

    if (qa.ndim() != 3 || k0a.ndim() != 2 || idxa.ndim() != 2 ||
        sinksa.ndim() != 1) {
      return nullptr;
    }
    if (qa.dtype() != mlx::core::float32 || k0a.dtype() != mlx::core::float16 ||
        idxa.dtype() != mlx::core::int32 ||
        sinksa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int s = qa.shape(0);
    const int h = qa.shape(1);
    const int d_k = qa.shape(2);
    const int d_v = v_width;
    const int d_rope = d_k - d_v;
    const int n_keys = k0a.shape(0);
    const int kmax = idxa.shape(1);
    if (s < 1 || h < 1 || n_keys < 1 || kmax < 1 || idxa.shape(0) != s ||
        sinksa.shape(0) != h) {
      return nullptr;
    }
    // The 32-lane split requires V=512 (16/lane), RoPE=64 (2/lane), HALF=288.
    if (d_v != 512 || d_rope != 64) {
      return nullptr;
    }
    if (split) {
      const auto &k1a = *reinterpret_cast<array *>(k1);
      if (k1a.ndim() != 2 || k1a.dtype() != mlx::core::float16 ||
          k0a.shape(1) != 288 || k1a.shape(0) != n_keys ||
          k1a.shape(1) != 288) {
        return nullptr;
      }
    } else if (k0a.shape(1) != d_k) {
      return nullptr;
    }

    array qc = ensure_contiguous_aligned(qa, 1);
    array k0c = ensure_contiguous_aligned(k0a, 4);
    array k1c = split ? ensure_contiguous_aligned(
                            *reinterpret_cast<array *>(k1), 4)
                      : cached_zero_f16();
    array idxc = ensure_contiguous_aligned(idxa, 1);
    array sinksc = ensure_contiguous_aligned(sinksa, 1);
    array scale_arr = cached_f32_scalar(scale);
    const auto stream = default_stream(Device::gpu);

    // Fixed per-split key blocks depend only on H so trailing -1 padding in
    // idx never shifts split boundaries or perturbs online softmax reduction.
    const int k_split = (h >= 64) ? 128 : 256;
    const int n_splits = std::min(8, (kmax + k_split - 1) / k_split);
    const bool use_splitk = (n_splits > 1);
    constexpr int kSplitKMaxChunk = 64;

    auto run_splitk_chunked = [&](const MlaConfig &c,
                                  std::optional<std::vector<array>> first_parts)
        -> array {
      array split_cfg = cached_i32_pair(n_splits, k_split);
      if (s <= kSplitKMaxChunk) {
        auto parts = first_parts.has_value()
                         ? std::move(*first_parts)
                         : hy4_mla_splitk_pass1(qc, k0c, k1c, idxc, scale_arr,
                                                split_cfg, d_v, d_rope, split,
                                                c.heads, c.bn, n_splits,
                                                stream);
        return hy4_mla_splitk_merge(parts[0], parts[1], parts[2], sinksc, d_v,
                                    stream);
      }
      std::vector<array> strips;
      strips.reserve(size_t((s + kSplitKMaxChunk - 1) / kSplitKMaxChunk));
      for (int row0 = 0; row0 < s; row0 += kSplitKMaxChunk) {
        const int row1 = std::min(row0 + kSplitKMaxChunk, s);
        std::vector<array> parts;
        if (row0 == 0 && first_parts.has_value()) {
          parts = std::move(*first_parts);
        } else {
          array q_sub = ensure_contiguous_aligned(
              slice(qc, {row0, 0, 0}, {row1, h, d_k}, stream), 1);
          array idx_sub = ensure_contiguous_aligned(
              slice(idxc, {row0, 0}, {row1, kmax}, stream), 1);
          parts = hy4_mla_splitk_pass1(q_sub, k0c, k1c, idx_sub, scale_arr,
                                       split_cfg, d_v, d_rope, split, c.heads,
                                       c.bn, n_splits, stream);
        }
        strips.push_back(hy4_mla_splitk_merge(parts[0], parts[1], parts[2],
                                              sinksc, d_v, stream));
      }
      return concatenate(strips, 0, stream);
    };

    MlaConfig cfg{};
    if (lookup_mla_config(h, split, use_splitk, cfg)) {
      if (!use_splitk) {
        auto result = hy4_mla_single_pass(qc, k0c, k1c, idxc, sinksc, scale_arr,
                                          d_v, d_rope, split, cfg.heads, cfg.bn,
                                          stream);
        return reinterpret_cast<mlx_array *>(new array(std::move(result)));
      }
      auto result = run_splitk_chunked(cfg, std::nullopt);
      return reinterpret_cast<mlx_array *>(new array(std::move(result)));
    }

    constexpr MlaConfig kCandidates[] = {
        {4, 8},
        {2, 16},
        {2, 8},
        {1, 16},
        {1, 8},
    };
    for (const auto &cand : kCandidates) {
      if (h % cand.heads != 0) {
        continue;
      }
      if (!use_splitk) {
        auto result = hy4_mla_single_pass(qc, k0c, k1c, idxc, sinksc, scale_arr,
                                          d_v, d_rope, split, cand.heads,
                                          cand.bn, stream);
        if (hy4_mla_pipeline_threads(result) >= cand.bn * 32) {
          store_mla_config(h, split, false, cand);
          return reinterpret_cast<mlx_array *>(new array(std::move(result)));
        }
      } else {
        array split_cfg = cached_i32_pair(n_splits, k_split);
        array q_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(qc, {0, 0, 0}, {kSplitKMaxChunk, h, d_k}, stream),
                      1)
                : qc;
        array idx_probe =
            (s > kSplitKMaxChunk)
                ? ensure_contiguous_aligned(
                      slice(idxc, {0, 0}, {kSplitKMaxChunk, kmax}, stream), 1)
                : idxc;
        auto parts = hy4_mla_splitk_pass1(q_probe, k0c, k1c, idx_probe,
                                          scale_arr, split_cfg, d_v, d_rope,
                                          split, cand.heads, cand.bn, n_splits,
                                          stream);
        if (hy4_mla_pipeline_threads(parts[0]) >= cand.bn * 32) {
          store_mla_config(h, split, true, cand);
          auto result = run_splitk_chunked(cand, std::move(parts));
          return reinterpret_cast<mlx_array *>(new array(std::move(result)));
        }
      }
    }
    return nullptr;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_sink_attention: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)q;
  (void)k0;
  (void)k1;
  (void)idx;
  (void)sinks;
  (void)v_width;
  (void)scale;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_mla_prefill_nax(mlx_array *q, mlx_array *k0, mlx_array *k1,
                                   mlx_array *idx, mlx_array *sinks,
                                   int v_width, float scale) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!q || !k0 || !idx || !sinks || !std::isfinite(scale)) {
      return nullptr;
    }
    const auto &qa = *reinterpret_cast<array *>(q);
    const auto &k0a = *reinterpret_cast<array *>(k0);
    const auto &idxa = *reinterpret_cast<array *>(idx);
    const auto &sinksa = *reinterpret_cast<array *>(sinks);
    const bool split = (k1 != nullptr);

    if (qa.ndim() != 3 || k0a.ndim() != 2 || idxa.ndim() != 2 ||
        sinksa.ndim() != 1) {
      return nullptr;
    }
    if (qa.dtype() != mlx::core::float32 || k0a.dtype() != mlx::core::float16 ||
        idxa.dtype() != mlx::core::int32 ||
        sinksa.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int s = qa.shape(0);
    const int h = qa.shape(1);
    const int d_k = qa.shape(2);
    const int d_v = v_width;
    const int d_rope = d_k - d_v;
    const int n_keys = k0a.shape(0);
    const int kmax = idxa.shape(1);
    if (s < 1 || h < 32 || h % 32 != 0 || n_keys < 1 || kmax < 1 ||
        idxa.shape(0) != s || sinksa.shape(0) != h) {
      return nullptr;
    }
    if (d_v != 512 || d_rope != 64) {
      return nullptr;
    }
    if (split) {
      const auto &k1a = *reinterpret_cast<array *>(k1);
      if (k1a.ndim() != 2 || k1a.dtype() != mlx::core::float16 ||
          k0a.shape(1) != 288 || k1a.shape(0) != n_keys ||
          k1a.shape(1) != 288) {
        return nullptr;
      }
    } else if (k0a.shape(1) != d_k) {
      return nullptr;
    }

    array qc = ensure_contiguous_aligned(qa, 1);
    array k0c = ensure_contiguous_aligned(k0a, 4);
    array k1c = split ? ensure_contiguous_aligned(
                            *reinterpret_cast<array *>(k1), 4)
                      : cached_zero_f16();
    array idxc = ensure_contiguous_aligned(idxa, 1);
    array sinksc = ensure_contiguous_aligned(sinksa, 1);
    array scale_arr = cached_f32_scalar(scale);
    const auto stream = default_stream(Device::gpu);

    constexpr int wm = 2;
    constexpr int wn = 4;
    constexpr int bh = wm * 16;
    constexpr int bk = 32;
    const Dtype compute_dtype = mlx::core::float32;
    const int head_groups = h / bh;
    const std::tuple<int, int, int> grid{32, head_groups * wm * wn, s};
    const std::tuple<int, int, int> tg{32, wm * wn, 1};
    const uint64_t tmpl_key = uint64_t(d_v) | (uint64_t(d_rope) << 16) |
                              (uint64_t(split ? 1 : 0) << 32) |
                              (uint64_t(wm) << 36) | (uint64_t(wn) << 40) |
                              (uint64_t(bk) << 44);
    const std::vector<array> inputs{qc, k0c, k1c, idxc, sinksc, scale_arr};
    const std::vector<Shape> out_shapes{{s, h, d_v}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelMlaPrefillNax, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_mla_prefill_nax_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"D_V", d_v},
               {"D_ROPE", d_rope},
               {"SPLIT", split},
               {"WM", wm},
               {"WN", wn},
               {"BK", bk},
               {"C", compute_dtype}},
              std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 mla_prefill_nax: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)q;
  (void)k0;
  (void)k1;
  (void)idx;
  (void)sinks;
  (void)v_width;
  (void)scale;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_prefill_indirect_gu(mlx_array *x, mlx_array *token_rows,
                                          mlx_array *tiles, mlx_array *wg,
                                          mlx_array *sg, mlx_array *bg,
                                          mlx_array *wu, mlx_array *su,
                                          mlx_array *bu, float limit) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    static const bool env_disabled = [] {
      const char *v = std::getenv("MLX_HY4_PREFILL_INDIRECT_GU");
      return v && std::strcmp(v, "0") == 0;
    }();
    if (env_disabled || !mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu) ||
        !mlx::core::metal::is_nax_available()) {
      return nullptr;
    }
    if (!x || !token_rows || !tiles || !wg || !sg || !bg || !wu || !su || !bu ||
        !std::isfinite(limit)) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &tra = *reinterpret_cast<array *>(token_rows);
    const auto &ta = *reinterpret_cast<array *>(tiles);
    const auto &wga = *reinterpret_cast<array *>(wg);
    const auto &sga = *reinterpret_cast<array *>(sg);
    const auto &bga = *reinterpret_cast<array *>(bg);
    const auto &wua = *reinterpret_cast<array *>(wu);
    const auto &sua = *reinterpret_cast<array *>(su);
    const auto &bua = *reinterpret_cast<array *>(bu);

    if (xa.ndim() != 2 || tra.ndim() != 1 || ta.ndim() != 2 ||
        wga.ndim() != 3 || sga.ndim() != 3 || bga.ndim() != 3 ||
        wua.ndim() != 3 || sua.ndim() != 3 || bua.ndim() != 3) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || tra.dtype() != mlx::core::uint32 ||
        ta.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int n_tok = xa.shape(0);
    const int k = xa.shape(1);
    const int r = tra.shape(0);
    const int n_tiles = ta.shape(0);
    const int s = wga.shape(0);
    const int m = wga.shape(1);
    if (n_tok < 1 || r < 1 || n_tiles < 1 || ta.shape(1) != 3 || s < 1 ||
        k < 256 || k % 256 != 0 || m < 64 || m % 64 != 0) {
      return nullptr;
    }
    const Shape q4_w{s, m, k / 8};
    const Shape q4_s{s, m, k / 16};
    const Shape q4_b{s, m, k / 128};
    if (wga.dtype() != mlx::core::uint32 || wga.shape() != q4_w ||
        sga.dtype() != mlx::core::uint8 || sga.shape() != q4_s ||
        bga.dtype() != mlx::core::float16 || bga.shape() != q4_b ||
        wua.dtype() != mlx::core::uint32 || wua.shape() != q4_w ||
        sua.dtype() != mlx::core::uint8 || sua.shape() != q4_s ||
        bua.dtype() != mlx::core::float16 || bua.shape() != q4_b) {
      return nullptr;
    }

    array xc = ensure_contiguous_aligned(xa, 4);
    array trc = ensure_contiguous_aligned(tra, 1);
    array tc = ensure_contiguous_aligned(ta, 1);
    array wgc = ensure_contiguous_aligned(wga, 1);
    array sgc = ensure_contiguous_aligned(sga, 1);
    array bgc = ensure_contiguous_aligned(bga, 1);
    array wuc = ensure_contiguous_aligned(wua, 1);
    array suc = ensure_contiguous_aligned(sua, 1);
    array buc = ensure_contiguous_aligned(bua, 1);
    array limit_arr = cached_f32_scalar(limit);

    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{32 * (m / 64), 4 * n_tiles, 1};
    const std::tuple<int, int, int> tg{32, 4, 1};
    const std::vector<array> inputs{xc,  trc, tc,  wgc, sgc,
                                    bgc, wuc, suc, buc, limit_arr};
    const std::vector<Shape> out_shapes{{r, m}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelKqPrefillIndirectGu, stream, grid, tg, 0, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_kq_prefill_indirect_gu_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg, {}, std::nullopt, false,
              stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_prefill_indirect_gu: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)x;
  (void)token_rows;
  (void)tiles;
  (void)wg;
  (void)sg;
  (void)bg;
  (void)wu;
  (void)su;
  (void)bu;
  (void)limit;
  return nullptr;
#endif
}

mlx_array *mlx_hy4_kq_sorted_combine(mlx_array *down, mlx_array *weights,
                                     mlx_array *inverse) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return nullptr;
    }
    if (!down || !weights || !inverse) {
      return nullptr;
    }
    const auto &da = *reinterpret_cast<array *>(down);
    const auto &wa = *reinterpret_cast<array *>(weights);
    const auto &ia = *reinterpret_cast<array *>(inverse);
    if (da.ndim() != 2 || wa.ndim() != 2 || ia.ndim() != 2 ||
        da.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::float32 ||
        ia.dtype() != mlx::core::uint32) {
      return nullptr;
    }
    const int r = da.shape(0);
    const int h = da.shape(1);
    const int n_tok = wa.shape(0);
    const int top = wa.shape(1);
    if (r < 1 || h < 1 || n_tok < 1 || top < 1 || top > 32 ||
        ia.shape() != Shape{n_tok, top}) {
      return nullptr;
    }

    // Only cast down/out to device float4* when size >= 8 so MLX binds both in
    // device address space rather than constant.
    const bool vec4 = (h % 4 == 0) && (da.size() >= 8) &&
                      (size_t(n_tok) * size_t(h) >= 8);
    array dc = ensure_contiguous_aligned(da, vec4 ? 4 : 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array ic = ensure_contiguous_aligned(ia, 1);

    const int cols = vec4 ? (h / 4) : h;
    const int tg_x = std::min(256, cols);
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{cols, n_tok, 1};
    const std::tuple<int, int, int> tg{tg_x, 1, 1};
    const uint64_t tmpl_key = uint64_t(top) | (uint64_t(vec4 ? 1 : 0) << 8);
    const std::vector<array> inputs{dc, wc, ic};
    const std::vector<Shape> out_shapes{{n_tok, h}};
    const std::vector<Dtype> out_dtypes{mlx::core::float32};
    auto out = dispatch_cached_kernel(
        kKernelKqSortedCombine, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_kq_sorted_combine_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"TOP", top}, {"VEC4", vec4}}, std::nullopt, false, stream);
        });
    return reinterpret_cast<mlx_array *>(new array(std::move(out[0])));
  } catch (const std::exception &e) {
    std::cerr << "Hy4 kq_sorted_combine: " << e.what() << std::endl;
    return nullptr;
  }
#else
  (void)down;
  (void)weights;
  (void)inverse;
  return nullptr;
#endif
}

bool mlx_hy4_router_topk(mlx_array *logits, mlx_array *bias, float scale,
                         float eps, int k, bool norm, mlx_array **out_ids,
                         mlx_array **out_weights) {
  if (out_ids)
    *out_ids = nullptr;
  if (out_weights)
    *out_weights = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available() ||
        !(default_device() == Device::gpu)) {
      return false;
    }
    if (!logits || !out_ids || !out_weights || !std::isfinite(scale) ||
        !std::isfinite(eps)) {
      return false;
    }
    const auto &la = *reinterpret_cast<array *>(logits);
    if (la.ndim() != 2 || la.dtype() != mlx::core::float32) {
      return false;
    }
    const int m = la.shape(0);
    const int e = la.shape(1);
    if (m < 1 || e < 1 || e > 1024 || k < 1 || k > 32 || k > e) {
      return false;
    }
    const bool has_bias = (bias != nullptr);
    array bc = cached_f32_scalar(0.0f);
    if (has_bias) {
      const auto &ba = *reinterpret_cast<array *>(bias);
      if (ba.ndim() != 1 || ba.dtype() != mlx::core::float32 ||
          ba.shape(0) != e) {
        return false;
      }
      bc = ensure_contiguous_aligned(ba, 1);
    }
    array lc = ensure_contiguous_aligned(la, 1);
    array params = cached_f32_pair(scale, eps);

    const int max_e = (e <= 512) ? 512 : 1024;
    const auto stream = default_stream(Device::gpu);
    const std::tuple<int, int, int> grid{128, m, 1};
    const std::tuple<int, int, int> tg{128, 1, 1};
    const uint64_t tmpl_key = uint64_t(k) | (uint64_t(max_e) << 8) |
                              (uint64_t(has_bias ? 1 : 0) << 20) |
                              (uint64_t(norm ? 1 : 0) << 21);
    const std::vector<array> inputs{lc, bc, params};
    const std::vector<Shape> out_shapes{{m, k}, {m, k}};
    const std::vector<Dtype> out_dtypes{mlx::core::uint32, mlx::core::float32};
    auto res = dispatch_cached_kernel(
        kKernelRouterTopk, stream, grid, tg, tmpl_key, inputs, out_shapes,
        out_dtypes, [&] {
          return hy4_router_topk_kernel()(
              inputs, out_shapes, out_dtypes, grid, tg,
              {{"K", k},
               {"MAX_E", max_e},
               {"HAS_BIAS", has_bias},
               {"NORM", norm}},
              std::nullopt, false, stream);
        });
    auto ids = std::make_unique<array>(std::move(res[0]));
    auto w = std::make_unique<array>(std::move(res[1]));
    *out_ids = reinterpret_cast<mlx_array *>(ids.release());
    *out_weights = reinterpret_cast<mlx_array *>(w.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Hy4 router_topk: " << e.what() << std::endl;
    return false;
  }
#else
  (void)logits;
  (void)bias;
  (void)scale;
  (void)eps;
  (void)k;
  (void)norm;
  return false;
#endif
}

} // extern "C"
