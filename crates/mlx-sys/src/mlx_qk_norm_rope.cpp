// Fused Q/K RMSNorm + partial RoPE for one attention layer: one
// fast::metal_kernel dispatch replaces the q_norm, k_norm, rope(q), rope(k)
// chain (and the contiguous copies those ops make for strided inputs and
// partial rotary dims). The kernel reproduces `fast::rms_norm` and
// `fast::rope` bit-for-bit (see metal/common/qk_norm_rope.metal.inc); the
// outputs are already in the `[B, H, T, D]` layout the verify attention reads.

#include "mlx_common.h"

#include <cmath>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace {

const char* kQkNormRopeBody =
#include "metal/common/qk_norm_rope.metal.inc"
    ;

mlx::core::fast::CustomKernelFunction& qk_norm_rope_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "mlx_node_qk_norm_rope",
        /* input_names  */
        {"q", "k", "wq", "wk", "offsets", "eps", "base", "scale", "heads_q",
         "heads_k", "tokens"},
        /* output_names */ {"q_out", "k_out"},
        /* source       */ kQkNormRopeBody,
        /* header       */ "",
        /* ensure_row_contiguous */ false,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

void expect(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(what);
  }
}

}  // namespace

// q `[B, T, HQ, D]`, k `[B, T, HK, D]` (any strides, contiguous last axis),
// `wq` / `wk` `[D]` RMSNorm weights, `offsets` int32 `[B]`. Writes
// `rope(rms_norm(q) * wq)` as `[B, HQ, T, D]` to `*out_q` and the same for k
// to `*out_k`; the first `rope_dims` channels rotate with `theta = base`,
// `scale`, non-traditional pairing, position `t + offsets[b]`. Returns false
// without Metal or on a contract miss (quietly) and on an MLX error (message
// on stderr); the caller keeps the four-op chain.
extern "C" bool mlx_qk_norm_rope(mlx_array* q_handle, mlx_array* k_handle,
                                 mlx_array* wq_handle, mlx_array* wk_handle,
                                 mlx_array* offsets_handle, float eps,
                                 float base, float scale, int32_t rope_dims,
                                 mlx_array** out_q, mlx_array** out_k) {
  if (out_q) *out_q = nullptr;
  if (out_k) *out_k = nullptr;
  try {
    using namespace mlx::core;
    expect(q_handle && k_handle && wq_handle && wk_handle && offsets_handle &&
               out_q && out_k,
           "mlx_qk_norm_rope: null argument");
    if (!metal::is_available() || default_device().type != Device::gpu) {
      return false;
    }
    const auto& q = *reinterpret_cast<array*>(q_handle);
    const auto& k = *reinterpret_cast<array*>(k_handle);
    const auto& wq = *reinterpret_cast<array*>(wq_handle);
    const auto& wk = *reinterpret_cast<array*>(wk_handle);
    const auto& offsets = *reinterpret_cast<array*>(offsets_handle);
    if (q.ndim() != 4 || k.ndim() != 4) {
      return false;
    }
    const int B = q.shape(0), T = q.shape(1), HQ = q.shape(2), D = q.shape(3);
    const int HK = k.shape(2);
    const auto dt = q.dtype();
    // Quiet contract misses: the caller keeps its four-op chain.
    const bool eligible =
        k.shape(0) == B && k.shape(1) == T && k.shape(3) == D &&
        D % 128 == 0 && D <= 4096 && rope_dims > 0 && rope_dims % 2 == 0 &&
        rope_dims <= D && (dt == bfloat16 || dt == float16 || dt == float32) &&
        k.dtype() == dt && wq.dtype() == dt && wk.dtype() == dt &&
        wq.ndim() == 1 && wq.shape(0) == D && wq.flags().row_contiguous &&
        wk.ndim() == 1 && wk.shape(0) == D && wk.flags().row_contiguous &&
        q.strides(3) == 1 && k.strides(3) == 1 && offsets.ndim() == 1 &&
        offsets.shape(0) == B && offsets.dtype() == int32 && B >= 1 &&
        T >= 1 && HQ >= 1 && HK >= 1;
    if (!eligible) {
      return false;
    }

    // rope.cpp: `float base = std::log2(base_)` on the float theta.
    const float log2_base = std::log2(base);
    const int lsize = D / 4;
    const int rows = B * (HQ + HK) * T;
    auto outs = qk_norm_rope_kernel()(
        {q, k, wq, wk, offsets, array(eps, float32), array(log2_base, float32),
         array(scale, float32), array(HQ, int32), array(HK, int32),
         array(T, int32)},
        /* output_shapes */ {Shape{B, HQ, T, D}, Shape{B, HK, T, D}},
        /* output_dtypes */ {dt, dt},
        /* grid        */ {lsize, rows, 1},
        /* threadgroup */ {lsize, 1, 1},
        /* template_args */ {{"T", dt}, {"D", D}, {"ROT", int(rope_dims)}},
        /* init_value */ std::nullopt,
        /* verbose */ false, default_stream(Device::gpu));
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(outs[0])));
    *out_k = reinterpret_cast<mlx_array*>(new array(std::move(outs[1])));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlx_qk_norm_rope: %s\n", e.what());
    return false;
  }
}
