#include "mlx_common.h"
#include <memory>

#ifdef MLX_NODE_METAL_ENABLED
#include "metal/common/quantized.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
namespace {
using mlx::core::quantized_preamble::qmv_header;

array ensure_contiguous_aligned(const array &a, size_t elem_align) {
  if (a.flags().row_contiguous &&
      (elem_align <= 1 || a.offset() % elem_align == 0)) {
    return a;
  }
  return copy(a);
}

const mlx::core::fast::CustomKernelFunction &hy4_q8_decode_kernel() {
  static auto kernel = mlx::core::fast::metal_kernel(
      "hy4_q8_dense_decode", {"x", "w", "scales", "biases"}, {"out"},
#include "metal/hy4/q8_dense_decode.metal.inc"
      , qmv_header());
  return kernel;
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
      hy4_kq_gate_up_swiglu_impl<FAST>(
          x, row_ids[assign], slots[assign], wg, sg, bg, wu, su, bu, limit[0], out,
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

array hy4_mla_kernel_output(const array &q, const array &k0, const array &k1,
                            const array &idx, const array &sinks,
                            const array &scale, int d_v, int d_rope, bool split,
                            int heads, int bn) {
  const int s = q.shape(0);
  const int h = q.shape(1);
  const int tg_threads = bn * 32;
  const int head_groups = h / heads;
  return hy4_mla_sink_kernel()(
      {q, k0, k1, idx, sinks, scale}, {{s, h, d_v}}, {mlx::core::float32},
      {tg_threads, head_groups, s}, {tg_threads, 1, 1},
      {{"D_V", d_v},
       {"D_ROPE", d_rope},
       {"SPLIT", split},
       {"HEADS", heads},
       {"BN", bn}},
      std::nullopt, false, default_stream(Device::gpu))[0];
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
    const float p_vals[3] = {norm_eps, hc_eps, magnitude};
    array params(p_vals, {3}, mlx::core::float32);

    static auto kernel = mlx::core::fast::metal_kernel(
        "hy4_hc_mix_reduce", {"x", "weight", "scale", "base", "params"},
        {"h", "post"},
#include "metal/hy4/hc_mix_reduce.metal.inc"
    );
    const Shape post_shape = has_post ? Shape{m, hc} : Shape{1};
    auto res = kernel(
        {xc, wc, sc, bc, params}, {{m, d}, post_shape},
        {mlx::core::float32, mlx::core::float32}, {256, m, 1}, {256, 1, 1},
        {{"HC", hc}, {"HAS_POST", has_post}, {"VEC4", vec4}}, std::nullopt,
        false, default_stream(Device::gpu));
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

    static auto kernel = mlx::core::fast::metal_kernel(
        "hy4_hc_post_identity", {"y", "res", "post"}, {"out"},
#include "metal/hy4/hc_post_identity.metal.inc"
    );
    const int cols = vec4 ? (d / 4) : d;
    const int tg_x = std::min(256, cols);
    auto out = kernel({yc, rc, pc}, {{m, hc, d}}, {mlx::core::float32},
                      {cols, m, 1}, {tg_x, 1, 1},
                      {{"HC", hc}, {"VEC4", vec4}}, std::nullopt, false,
                      default_stream(Device::gpu));
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

    static auto kernel = mlx::core::fast::metal_kernel(
        "hy4_bf16_lm_head", {"x", "w"}, {"out"},
#include "metal/hy4/bf16_lm_head.metal.inc"
    );
    auto out = kernel(
        {xc, wc}, {{m, v}}, {mlx::core::float32},
        {32, ((v + 7) / 8) * 2, (m + r_tile - 1) / r_tile}, {32, 2, 1},
        {{"FAST_K", fast_k}, {"R_TILE", r_tile}, {"RPS", rps}}, std::nullopt,
        false, default_stream(Device::gpu));
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
    if (!x || !w || !scales || !biases) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    const auto &ba = *reinterpret_cast<array *>(biases);
    if (xa.ndim() != 2 || wa.ndim() != 2 || sa.ndim() != 2 || ba.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32 ||
        sa.dtype() != mlx::core::float32 || ba.dtype() != mlx::core::float32) {
      return nullptr;
    }
    const int m = xa.shape(0);
    const int k = xa.shape(1);
    const int n = wa.shape(0);
    if (m < 1 || m > 16 || k < 32 || k % 32 != 0 || n < 1 ||
        wa.shape(1) != k / 4 || sa.shape() != Shape{n, k / 32} ||
        ba.shape() != Shape{n, k / 32}) {
      return nullptr;
    }
    array xc = ensure_contiguous_aligned(xa, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array sc = ensure_contiguous_aligned(sa, 1);
    array bc = ensure_contiguous_aligned(ba, 1);

    const bool fast = (k % 256 == 0) && (n % 8 == 0);
    const int rps = (n >= 1024) ? 4 : 2;
    const int r_tile = (m <= 4) ? m : 4;
    const int grid_y = 2 * ((n + 2 * rps - 1) / (2 * rps));
    const int grid_z = (m + r_tile - 1) / r_tile;

    auto out = hy4_q8_decode_kernel()(
        {xc, wc, sc, bc}, {{m, n}}, {mlx::core::float32}, {32, grid_y, grid_z},
        {32, 2, 1},
        {{"FAST", fast},
         {"RPS", rps},
         {"R_TILE", r_tile},
         {"GROUPED", false}},
        std::nullopt, false, default_stream(Device::gpu));
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
    if (!x || !w || !scales || !biases) {
      return nullptr;
    }
    const auto &xa = *reinterpret_cast<array *>(x);
    const auto &wa = *reinterpret_cast<array *>(w);
    const auto &sa = *reinterpret_cast<array *>(scales);
    const auto &ba = *reinterpret_cast<array *>(biases);
    if (xa.ndim() != 3 || wa.ndim() != 2 || sa.ndim() != 2 || ba.ndim() != 2) {
      return nullptr;
    }
    if (xa.dtype() != mlx::core::float32 || wa.dtype() != mlx::core::uint32 ||
        sa.dtype() != mlx::core::float32 || ba.dtype() != mlx::core::float32) {
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
    if (n < 1 || sa.shape() != Shape{g * n, k / 32} ||
        ba.shape() != Shape{g * n, k / 32}) {
      return nullptr;
    }
    array xc = ensure_contiguous_aligned(xa, 1);
    array wc = ensure_contiguous_aligned(wa, 1);
    array sc = ensure_contiguous_aligned(sa, 1);
    array bc = ensure_contiguous_aligned(ba, 1);

    const bool fast = (k % 256 == 0) && (n % 8 == 0);
    const int rps = (n >= 1024) ? 4 : 2;
    const int r_tile = (m <= 4) ? m : 4;
    const int grid_y = 2 * ((n + 2 * rps - 1) / (2 * rps));
    const int grid_z = g * ((m + r_tile - 1) / r_tile);

    auto out = hy4_q8_decode_kernel()(
        {xc, wc, sc, bc}, {{m, g, n}}, {mlx::core::float32},
        {32, grid_y, grid_z}, {32, 2, 1},
        {{"FAST", fast},
         {"RPS", rps},
         {"R_TILE", r_tile},
         {"GROUPED", true}},
        std::nullopt, false, default_stream(Device::gpu));
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
    std::vector<uint32_t> rids(static_cast<size_t>(n_assign));
    for (int r = 0; r < rows; ++r) {
      for (int t = 0; t < top; ++t) {
        rids[static_cast<size_t>(r * top + t)] = static_cast<uint32_t>(r);
      }
    }
    array row_ids(rids.data(), {n_assign}, mlx::core::uint32);
    array limit_arr(&limit, {1}, mlx::core::float32);

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
    auto stream = default_stream(Device::gpu);
    auto hidden = hy4_kq_gate_up_kernel()(
        {xc, row_ids, slots_flat, wgc, sgc, bgc, wuc, suc, buc, limit_arr},
        {{n_assign, m}}, {mlx::core::float32},
        {32, ((m + 7) / 8) * 2, n_assign}, {32, 2, 1}, {{"FAST", fast_gu}},
        std::nullopt, false, stream)[0];

    const bool fast_d = (h % 4 == 0) && (m % 512 == 0);
    const int gs = (down_bits == 6) ? 16 : 32;
    const int sr = (down_bits == 6) ? 16 : 8;
    const bool hm = (down_bits == 4);
    constexpr int rps = 4;
    constexpr int ksg = 8;
    auto out = hy4_kq_down_combine_kernel()(
        {hidden, slc, wc, wdc, sdc, bdc}, {{rows, h}}, {mlx::core::float32},
        {32, ((h + rps - 1) / rps) * ksg, rows}, {32, ksg, 1},
        {{"BITS", down_bits},
         {"GS", gs},
         {"SR", sr},
         {"HM", hm},
         {"FAST", fast_d},
         {"RPS", rps},
         {"KSG", ksg}},
        std::nullopt, false, stream);
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

    array limit_arr(&limit, {1}, mlx::core::float32);
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
    auto stream = default_stream(Device::gpu);
    auto hidden = hy4_kq_gate_up_kernel()(
        {xc, ridc, slc, wgc, sgc, bgc, wuc, suc, buc, limit_arr},
        {{n_assign, m}}, {mlx::core::float32},
        {32, ((m + 7) / 8) * 2, n_assign}, {32, 2, 1}, {{"FAST", fast_gu}},
        std::nullopt, false, stream)[0];

    const bool fast_d = (h % 4 == 0) && (m % 512 == 0);
    const int gs = (down_bits == 6) ? 16 : 32;
    const int sr = (down_bits == 6) ? 16 : 8;
    const bool hm = (down_bits == 4);
    constexpr int rps = 4;
    auto out = hy4_kq_down_weighted_kernel()(
        {hidden, slc, wc, wdc, sdc, bdc}, {{n_assign, h}}, {mlx::core::float32},
        {32, ((h + 2 * rps - 1) / (2 * rps)) * 2, n_assign}, {32, 2, 1},
        {{"BITS", down_bits},
         {"GS", gs},
         {"SR", sr},
         {"HM", hm},
         {"FAST", fast_d},
         {"RPS", rps}},
        std::nullopt, false, stream);
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
    array k0c = ensure_contiguous_aligned(k0a, 1);
    const mlx::core::float16_t zero_f16(0.0f);
    array k1c = split
                    ? ensure_contiguous_aligned(
                          *reinterpret_cast<array *>(k1), 1)
                    : array(&zero_f16, {1}, mlx::core::float16);
    array idxc = ensure_contiguous_aligned(idxa, 1);
    array sinksc = ensure_contiguous_aligned(sinksa, 1);
    array scale_arr(&scale, {1}, mlx::core::float32);

    struct MlaConfig {
      int heads;
      int bn;
    };
    constexpr MlaConfig kCandidates[] = {
        {4, 8},
        {2, 16},
        {2, 8},
        {1, 16},
        {1, 8},
    };
    for (const auto &cfg : kCandidates) {
      if (h % cfg.heads != 0) {
        continue;
      }
      auto result = hy4_mla_kernel_output(qc, k0c, k1c, idxc, sinksc, scale_arr,
                                          d_v, d_rope, split, cfg.heads, cfg.bn);
      if (hy4_mla_pipeline_threads(result) >= cfg.bn * 32) {
        return reinterpret_cast<mlx_array *>(new array(std::move(result)));
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

} // extern "C"
