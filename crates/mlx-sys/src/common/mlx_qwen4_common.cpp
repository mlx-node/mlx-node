#include "mlx_common.h"

#ifdef MLX_NODE_METAL_ENABLED
extern "C" {
mlx_array *mlx_qwen4_prefill_norm(mlx_array *x, mlx_array *w, double eps);
mlx_array *mlx_qwen4_decode_norm(mlx_array *x, mlx_array *w, double eps);
}

namespace mlx_node_internal {
std::optional<mlx::core::array>
qwen4_gdn_norm_port_metal(const std::vector<mlx::core::array> &a);
}  // namespace mlx_node_internal
#endif

namespace {

// Fuse the prefill epilogue without storing intermediate F32 upcasts. Keep
// both BF16 rounding boundaries and the original compiled norm/gate algebra.
std::vector<array> qwen4_gdn_epilogue_graph(const std::vector<array> &a) {
  auto y = astype(astype(a[0], mlx::core::bfloat16), mlx::core::float32);
  auto grouped = reshape(y, {-1, 1, 128});
  static auto norm = mlx::core::compile(
      [](const std::vector<array> &v) {
        return std::vector<array>{
            v[0] / sqrt(mean(square(v[0]), {-1}, true) + v[2]) * v[1]};
      },
      true);
  auto n = astype(
      reshape(norm({grouped, reshape(a[2], {1, 128}), a[3]})[0], y.shape()),
      mlx::core::bfloat16);
  static auto gate = mlx::core::compile(
      [](const std::vector<array> &v) {
        return std::vector<array>{v[1] * sigmoid(v[0])};
      },
      true);
  return {astype(gate({astype(a[1], mlx::core::float32),
                       astype(n, mlx::core::float32)})[0],
                 mlx::core::bfloat16)};
}

}  // namespace

extern "C" {

mlx_array *mlx_qwen4_norm(mlx_array *x, mlx_array *w, int group, double eps,
                          bool centered) {
  try {
    if (!x || !w || group <= 0)
      return nullptr;
    auto input = *reinterpret_cast<array *>(x),
         weight = *reinterpret_cast<array *>(w);
    if (input.shape(-1) % group || weight.size() != input.shape(-1))
      return nullptr;
#ifdef MLX_NODE_METAL_ENABLED
    if (group == 2560 && !centered) {
      if (auto result = mlx_qwen4_prefill_norm(x, w, eps))
        return result;
      if (auto result = mlx_qwen4_decode_norm(x, w, eps))
        return result;
    }
#endif
    auto shape = input.shape();
    auto grouped = reshape(astype(input, mlx::core::float32),
                           {-1, int(weight.size()) / group, group});
    auto scale = astype(weight, mlx::core::float32);
    if (centered)
      scale = scale + array(1.0f);
    scale = reshape(scale, {int(weight.size()) / group, group});
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          auto normalized = a[0] / sqrt(mean(square(a[0]), {-1}, true) + a[2]);
          return std::vector<array>{normalized * a[1]};
        },
        true);
    auto result =
        astype(reshape(fn({grouped, scale, array(float(eps))})[0], shape),
               input.dtype());
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 normalization: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_qwen4_gdn_epilogue(mlx_array *out, mlx_array *z, mlx_array *norm,
                                  double eps) {
  try {
    if (!out || !z || !norm || !std::isfinite(eps) || eps <= 0)
      return nullptr;
    auto x = *reinterpret_cast<array *>(out), g = *reinterpret_cast<array *>(z),
         w = *reinterpret_cast<array *>(norm);
    if (x.ndim() != 4 || x.shape(0) != 1 || x.shape(1) < 1 ||
        x.shape(1) > 1024 || x.shape(3) != 128 ||
        x.dtype() != mlx::core::float32 || g.shape() != x.shape() ||
        g.dtype() != mlx::core::bfloat16 || w.size() != 128 ||
        w.dtype() != mlx::core::float32)
      return nullptr;
    std::vector<array> a{x, g, w, array(float(eps))};
#ifdef MLX_NODE_METAL_ENABLED
    if (x.shape(1) > 8) {
      if (auto y = mlx_node_internal::qwen4_gdn_norm_port_metal(a)) {
        return reinterpret_cast<mlx_array *>(new array(std::move(*y)));
      }
    }
#endif
    static auto fn = mlx::core::compile(qwen4_gdn_epilogue_graph);
    auto y = fn(a)[0];
    return reinterpret_cast<mlx_array *>(new array(std::move(y)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 GDN epilogue: " << e.what() << std::endl;
    return nullptr;
  }
}

bool mlx_qwen4_gdn_gates(mlx_array *a, mlx_array *b, mlx_array *scale,
                         mlx_array *dt, mlx_array **decay, mlx_array **beta) {
  try {
    if (!a || !b || !scale || !dt || !decay || !beta)
      return false;
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          auto z = a[0] + a[3];
          auto softplus = maximum(z, array(0.0f)) + log1p(exp(-abs(z)));
          return std::vector<array>{exp(softplus * a[2]), sigmoid(a[1])};
        },
        true);
    auto result =
        fn({*reinterpret_cast<array *>(a), *reinterpret_cast<array *>(b),
            *reinterpret_cast<array *>(scale), *reinterpret_cast<array *>(dt)});
    *decay = reinterpret_cast<mlx_array *>(new array(std::move(result[0])));
    *beta = reinterpret_cast<mlx_array *>(new array(std::move(result[1])));
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 GDN gates: " << e.what() << std::endl;
    return false;
  }
}

mlx_array *mlx_qwen4_inject(mlx_array *x, mlx_array *y, mlx_array *g) {
  try {
    if (!x || !y || !g)
      return nullptr;
    static auto fn = mlx::core::compile(
        [](const std::vector<array> &a) {
          return std::vector<array>{a[0] + a[1] * a[2]};
        },
        true);
    auto input = *reinterpret_cast<array *>(x),
         branch = *reinterpret_cast<array *>(y),
         gate = *reinterpret_cast<array *>(g);
    auto residual = reshape(input, {input.shape(0), input.shape(1),
                                    gate.shape(-1), branch.shape(-1)});
    auto result = reshape(
        fn({residual, expand_dims(branch, -2), expand_dims(gate, -1)})[0],
        input.shape());
    return reinterpret_cast<mlx_array *>(new array(std::move(result)));
  } catch (const std::exception &e) {
    std::cerr << "Qwen4 injection: " << e.what() << std::endl;
    return nullptr;
  }
}

}  // extern "C"
