// Per-row symmetric int8 KV cache rows (Splash's target KV format): one int8
// row of 256 per (batch, head, token) with one fp32 scale. The quantize
// kernel is the cache's write path (prefill chunk rows and the verify block's
// rows); the dequantize kernel serves the readers that still want BF16 rows
// (prefill chunks after the first). Both are fast::metal_kernel JIT kernels
// cached per process, with an MLX-op fallback for hosts without Metal.

#include "mlx_common.h"

#include <cstdio>
#include <mutex>
#include <optional>
#include <tuple>

namespace {

constexpr int kHeadDimension = 256;

const char *kQuantizeBody =
#include "metal/common/kv_int8_quantize_rows.metal.inc"
    ;

const char *kDequantizeBody =
#include "metal/common/kv_int8_dequantize_rows.metal.inc"
    ;

mlx::core::fast::CustomKernelFunction &quantize_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "mlx_node_kv_int8_quantize_rows",
        /* input_names  */ {"x", "rows", "heads", "tokens"},
        /* output_names */ {"q", "s"},
        /* source       */ kQuantizeBody,
        /* header       */ "",
        /* ensure_row_contiguous */ false,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

mlx::core::fast::CustomKernelFunction &dequantize_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "mlx_node_kv_int8_dequantize_rows",
        /* input_names  */ {"q", "s", "rows", "heads", "tokens"},
        /* output_names */ {"out"},
        /* source       */ kDequantizeBody,
        /* header       */ "",
        /* ensure_row_contiguous */ false,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

void validate_rows(const mlx::core::array &rows, mlx::core::Dtype dtype,
                   const char *what) {
  if (rows.ndim() != 4 || rows.shape(3) != kHeadDimension ||
      rows.dtype() != dtype || rows.strides(3) != 1) {
    throw std::invalid_argument(std::string(what) +
                                " must be rank-4 [B, H, N, 256] with a "
                                "contiguous head dimension");
  }
}

bool metal_gpu() {
  return mlx::core::metal::is_available() &&
         mlx::core::default_device() == mlx::core::Device::gpu;
}

// The MLX-op form of the kernel (hosts without Metal, and the reference the
// tests compare the kernel against): same arithmetic, same rounding.
std::pair<mlx::core::array, mlx::core::array>
quantize_rows_fallback(const mlx::core::array &x) {
  using namespace mlx::core;
  auto xf = astype(x, float32);
  auto maximum = max(abs(xf), -1, /* keepdims */ true);
  auto zero = maximum == array(0.0f);
  auto safe_max = where(zero, array(1.0f), maximum);
  auto q = clip(round(divide(multiply(xf, array(127.0f)), safe_max)),
                array(-127.0f), array(127.0f));
  q = where(zero, array(0.0f), q);
  auto s = where(zero, array(0.0f), divide(maximum, array(127.0f)));
  return {astype(q, int8), squeeze(s, -1)};
}

mlx::core::array dequantize_rows_fallback(const mlx::core::array &q,
                                          const mlx::core::array &s) {
  using namespace mlx::core;
  return astype(multiply(astype(q, float32), expand_dims(s, -1)), bfloat16);
}

std::pair<mlx::core::array, mlx::core::array>
quantize_rows(const mlx::core::array &input) {
  using namespace mlx::core;
  // The format is defined over BF16 activations; other float inputs round
  // to BF16 first (what a BF16 cache would hold after its own cast).
  array x = input.dtype() == bfloat16 ? input : astype(input, bfloat16);
  validate_rows(x, bfloat16, "kv int8 quantize input");
  if (!metal_gpu()) {
    return quantize_rows_fallback(x);
  }
  const int B = x.shape(0), H = x.shape(1), N = x.shape(2);
  const int rows = B * H * N;
  array rows_arr(rows, int32), heads_arr(H, int32), tokens_arr(N, int32);
  const int groups = (rows + 7) / 8;
  std::tuple<int, int, int> grid{256, std::max(groups, 1), 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};
  auto results = quantize_kernel()(
      {x, rows_arr, heads_arr, tokens_arr},
      /* output_shapes */ {Shape{B, H, N, kHeadDimension}, Shape{B, H, N}},
      /* output_dtypes */ {int8, float32}, grid, threadgroup,
      /* template_args */ {}, /* init_value */ std::nullopt,
      /* verbose */ false, default_stream(Device::gpu));
  return {results[0], results[1]};
}

mlx::core::array dequantize_rows(const mlx::core::array &q,
                                 const mlx::core::array &s) {
  using namespace mlx::core;
  validate_rows(q, int8, "kv int8 dequantize input");
  if (s.ndim() != 3 || s.dtype() != float32 || s.shape(0) != q.shape(0) ||
      s.shape(1) != q.shape(1) || s.shape(2) != q.shape(2)) {
    throw std::invalid_argument(
        "kv int8 scales must be float32 [B, H, N] matching their rows");
  }
  if (!metal_gpu()) {
    return dequantize_rows_fallback(q, s);
  }
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2);
  const int rows = B * H * N;
  array rows_arr(rows, int32), heads_arr(H, int32), tokens_arr(N, int32);
  const int groups = (rows + 7) / 8;
  std::tuple<int, int, int> grid{256, std::max(groups, 1), 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};
  auto results = dequantize_kernel()(
      {q, s, rows_arr, heads_arr, tokens_arr},
      /* output_shapes */ {Shape{B, H, N, kHeadDimension}},
      /* output_dtypes */ {bfloat16}, grid, threadgroup,
      /* template_args */ {}, /* init_value */ std::nullopt,
      /* verbose */ false, default_stream(Device::gpu));
  return results[0];
}

} // namespace

// BF16 `[B, H, N, 256]` rows -> int8 rows (`*out_q`) and fp32 scales
// `[B, H, N]` (`*out_s`). 0 on success; -1 (message on stderr) on error.
extern "C" int mlx_kv_int8_quantize_rows(mlx_array *x, mlx_array **out_q,
                                         mlx_array **out_s) {
  try {
    auto [q, s] = quantize_rows(*reinterpret_cast<array *>(x));
    *out_q = reinterpret_cast<mlx_array *>(new array(std::move(q)));
    *out_s = reinterpret_cast<mlx_array *>(new array(std::move(s)));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_kv_int8_quantize_rows: %s\n", e.what());
    return -1;
  }
}

// Reference form of the quantizer built from MLX ops (tests).
extern "C" int mlx_kv_int8_quantize_rows_reference(mlx_array *x,
                                                   mlx_array **out_q,
                                                   mlx_array **out_s) {
  try {
    auto [q, s] = quantize_rows_fallback(*reinterpret_cast<array *>(x));
    *out_q = reinterpret_cast<mlx_array *>(new array(std::move(q)));
    *out_s = reinterpret_cast<mlx_array *>(new array(std::move(s)));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_kv_int8_quantize_rows_reference: %s\n",
                 e.what());
    return -1;
  }
}

// int8 rows `[B, H, N, 256]` x fp32 scales `[B, H, N]` -> BF16 rows. Null
// (message on stderr) on error.
extern "C" mlx_array *mlx_kv_int8_dequantize_rows(mlx_array *q, mlx_array *s) {
  try {
    auto out = dequantize_rows(*reinterpret_cast<array *>(q),
                               *reinterpret_cast<array *>(s));
    return reinterpret_cast<mlx_array *>(new array(std::move(out)));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlx_kv_int8_dequantize_rows: %s\n", e.what());
    return nullptr;
  }
}
