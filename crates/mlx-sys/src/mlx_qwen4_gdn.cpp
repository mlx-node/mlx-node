#include "mlx_common.h"

extern "C" int mlx_gpu_architecture_gen();
extern "C" bool mlx_qwen4_gated_delta_kernel(mlx_array *, mlx_array *,
                                             mlx_array *, mlx_array *,
                                             mlx_array *, mlx_array *,
                                             mlx_array *, mlx_array **,
                                             mlx_array **);

namespace {

// The precise exponential matches the compiled pointwise path's F32 sigmoid.
// The default intrinsic differed by a few ULPs when lowered in the full kernel.
const char *complete_gdn_header = R"(
  template <typename T> METAL_FUNC T qwen4_sigmoid(T x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
)";

std::vector<array> complete_gdn_metal(const std::vector<array> &in) {
  static auto fn =
      mlx::core::fast::metal_kernel("qwen4_gdn_complete",
                                    {"qkv", "z", "a", "b", "conv", "history",
                                     "scale", "dt", "state", "norm", "eps"},
                                    {"out", "next", "next_history"},
#include "metal/qwen4_gdn_complete.metal.inc"
                                    , complete_gdn_header);
  return fn(in, {{1, 1, 6144}, in[8].shape(), in[5].shape()},
            {in[0].dtype(), mlx::core::float32, in[0].dtype()}, {32, 32, 48},
            {32, 32, 1}, {{"T", in[0].dtype()}, {"KC", in[5].shape(0) + 1}},
            std::nullopt, false,
            mlx::core::default_stream(mlx::core::Device::gpu));
}

std::vector<array> window_conv_metal(const std::vector<array> &in) {
  static auto fn = mlx::core::fast::metal_kernel(
      "qwen4_window_conv", {"x", "history", "weight"}, {"out", "next_history"},
#include "metal/qwen4_window_conv.metal.inc"
      , complete_gdn_header);
  const int tokens = in[0].shape(1), width = in[0].shape(2);
  return fn(in, {in[0].shape(), in[1].shape()}, {in[0].dtype(), in[0].dtype()},
            {width, tokens, 1}, {256, 1, 1},
            {{"T", in[0].dtype()}, {"W", width}, {"TOKENS", tokens}},
            std::nullopt, false,
            mlx::core::default_stream(mlx::core::Device::gpu));
}

// Every mutable array is an argument or result of this graph. In particular,
// compilation must not capture a layer's convolution window or recurrent state.
std::vector<array> complete_gdn(const std::vector<array> &in) {
  const auto dtype = in[0].dtype();
  const int width = in[0].shape(-1), kernel = in[5].shape(0) + 1;
  const int heads = in[8].shape(1), dim = in[8].shape(2);
  const int key_heads = (width / dim - heads) / 2;
  const int keys = key_heads * dim;
  auto window = concatenate({in[5], reshape(in[0], {1, width})}, 0);
  auto conv =
      astype(sum(astype(window, mlx::core::float32) *
                     transpose(reshape(astype(in[4], mlx::core::float32),
                                       {width, kernel})),
                 {0}),
             dtype);
  conv = conv * sigmoid(conv);
  auto normalize = [dtype, dim](const array &x) {
    auto f = astype(reshape(x, {-1, dim}), mlx::core::float32);
    return astype(f / sqrt(sum(square(f), {-1}, true) + array(1e-6f)), dtype);
  };
  auto q = normalize(slice(conv, {0}, {keys})) *
           array(float(1.0 / std::sqrt(dim)), dtype);
  auto k = normalize(slice(conv, {keys}, {2 * keys}));
  q = reshape(astype(q, mlx::core::float32), {1, 1, key_heads, dim});
  k = reshape(astype(k, mlx::core::float32), {1, 1, key_heads, dim});
  auto v = reshape(astype(slice(conv, {2 * keys}, {width}), mlx::core::float32),
                   {1, 1, heads, dim});
  auto a =
      astype(in[2], mlx::core::float32) + astype(in[7], mlx::core::float32);
  auto softplus = maximum(a, array(0.0f)) + log1p(exp(-abs(a)));
  auto decay =
      reshape(exp(softplus * astype(in[6], mlx::core::float32)), {1, 1, heads});
  auto beta =
      reshape(sigmoid(astype(in[3], mlx::core::float32)), {1, 1, heads});
  auto state = in[8];
  auto handle = [](array &x) { return reinterpret_cast<mlx_array *>(&x); };
  mlx_array *out_handle = nullptr, *state_handle = nullptr;
  if (!mlx_qwen4_gated_delta_kernel(handle(q), handle(k), handle(v),
                                    handle(decay), handle(beta), handle(state),
                                    nullptr, &out_handle, &state_handle)) {
    throw std::runtime_error("Qwen4 complete GDN recurrence failed");
  }
  std::unique_ptr<array> out(reinterpret_cast<array *>(out_handle));
  std::unique_ptr<array> next(reinterpret_cast<array *>(state_handle));
  // Keep the same BF16 boundaries as the eager path, including the cast between
  // weighted normalization and the F32 output gate. GGUF key heads are tiled by
  // modulo in the shared recurrence, not repeated consecutively as in HF.
  auto y = astype(astype(*out, dtype), mlx::core::float32);
  y = y / sqrt(mean(square(y), {-1}, true) + in[10]);
  y = astype(astype(y * astype(in[9], mlx::core::float32), dtype),
             mlx::core::float32);
  auto gated = astype(y * sigmoid(reshape(astype(in[1], mlx::core::float32),
                                          {1, 1, heads, dim})),
                      dtype);
  return {reshape(gated, {1, 1, heads * dim}), *next,
          copy(slice(window, {1, 0}, {kernel, width}))};
}
} // namespace

// Four-tap batched convolution with the same F32 products/reduction and BF16
// SiLU boundaries as conv_sequence. No global tap-window or product arrays.
extern "C" bool mlx_qwen4_window_conv(mlx_array *x, mlx_array *history,
                                      mlx_array *weight, mlx_array **out,
                                      mlx_array **next_history) {
  if (!out || !next_history)
    return false;
  *out = nullptr;
  *next_history = nullptr;
  try {
    if (!x || !history || !weight)
      return false;
    const auto &a = *reinterpret_cast<array *>(x);
    const auto &h = *reinterpret_cast<array *>(history);
    const auto &w = *reinterpret_cast<array *>(weight);
    if (a.ndim() != 3 || a.shape(0) != 1 || a.shape(1) < 1 || a.shape(2) < 1 ||
        a.dtype() != mlx::core::bfloat16 || h.shape() != Shape{3, a.shape(2)} ||
        h.dtype() != a.dtype() || w.size() != size_t(a.shape(2)) * 4 ||
        w.dtype() != mlx::core::float32)
      return false;
    auto result = window_conv_metal({a, h, w});
    auto y = std::make_unique<array>(std::move(result[0]));
    auto next = std::make_unique<array>(std::move(result[1]));
    *out = reinterpret_cast<mlx_array *>(y.release());
    *next_history = reinterpret_cast<mlx_array *>(next.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 window convolution: " << e.what() << std::endl;
    return false;
  }
}

extern "C" bool
mlx_qwen4_complete_gdn(mlx_array *qkv, mlx_array *z, mlx_array *a, mlx_array *b,
                       mlx_array *conv, mlx_array *history, mlx_array *scale,
                       mlx_array *dt, mlx_array *state, mlx_array *norm,
                       double eps, mlx_array **out, mlx_array **next,
                       mlx_array **next_history) {
  if (!out || !next || !next_history)
    return false;
  *out = nullptr;
  *next = nullptr;
  *next_history = nullptr;
  try {
    std::vector<array> in;
    for (auto *p : {qkv, z, a, b, conv, history, scale, dt, state, norm}) {
      if (!p)
        return false;
      in.push_back(*reinterpret_cast<array *>(p));
    }
    if (in[0].shape() != Shape{1, 1, 10240} ||
        in[0].dtype() != mlx::core::bfloat16 || in[1].size() != 6144 ||
        in[1].dtype() != in[0].dtype() || in[2].size() != 48 ||
        in[3].size() != 48 || in[5].ndim() != 2 || in[5].shape(0) < 1 ||
        in[5].shape(0) > 7 || in[5].shape(1) != 10240 ||
        in[5].dtype() != in[0].dtype() ||
        in[4].size() != 10240 * (in[5].shape(0) + 1) || in[6].size() != 48 ||
        in[7].size() != 48 || in[9].size() != 128 ||
        in[8].shape() != Shape{1, 48, 128, 128} ||
        in[8].dtype() != mlx::core::float32 || !std::isfinite(eps) || eps <= 0)
      return false;
    in.push_back(array(float(eps)));
    // Shape-specialized: convolution length and head geometry are part of the
    // trace. MLX fuses pointwise work while preserving the recurrent primitive.
    static auto fn = mlx::core::compile(complete_gdn);
    static const int arch = mlx_gpu_architecture_gen();
    auto setting = std::getenv("MLX_QWEN4_COMPLETE_GDN_METAL");
    auto rows = std::getenv("MLX_QWEN4_GDN_4ROWS");
    bool fused = arch >= 17 && (!setting || std::string(setting) != "0") &&
                 (!rows || std::string(rows) != "0") &&
                 std::getenv("MLX_DISABLE_E47_GDN_2VCOL") == nullptr;
    auto result = fused ? complete_gdn_metal(in) : fn(in);
    auto y = std::make_unique<array>(std::move(result[0]));
    auto s = std::make_unique<array>(std::move(result[1]));
    auto h = std::make_unique<array>(std::move(result[2]));
    *out = reinterpret_cast<mlx_array *>(y.release());
    *next = reinterpret_cast<mlx_array *>(s.release());
    *next_history = reinterpret_cast<mlx_array *>(h.release());
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 complete GDN: " << e.what() << std::endl;
    return false;
  }
}
