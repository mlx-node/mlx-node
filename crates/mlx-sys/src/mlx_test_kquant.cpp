// TEST-ONLY hooks for the bridge K-quant ops: explicit-device entry points,
// kernel-family counters and a shapeless-compile replay. No production caller.

#include "mlx_common.h"
#include "mlx_kquant.h"

#include <limits>
#include <map>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#endif

using mlx::core::array;
namespace kquant = mlx::core::kquant;

namespace mlx::core::kquant::testing {
namespace {
thread_local std::map<std::string, uint64_t, std::less<>> family_counts;
}

void record_family(std::string_view family) {
  auto it = family_counts.find(family);
  if (it == family_counts.end()) {
    family_counts.emplace(std::string(family), 1);
  } else {
    ++it->second;
  }
}
} // namespace mlx::core::kquant::testing

namespace {
std::optional<array> opt(mlx_array *a) {
  if (!a)
    return std::nullopt;
  return *reinterpret_cast<array *>(a);
}

const array &ref(mlx_array *a) { return *reinterpret_cast<array *>(a); }

mlx::core::Device device_of(int32_t code) {
  return code == 1 ? mlx::core::Device(mlx::core::Device::gpu)
                   : mlx::core::Device(mlx::core::Device::cpu);
}

kquant::Mode mode_of(const char *mode) {
  auto parsed = kquant::parse_mode(mode ? mode : "");
  if (!parsed)
    throw std::invalid_argument("not a K-quant mode");
  return *parsed;
}

mlx_array *wrap(array a) {
  return reinterpret_cast<mlx_array *>(new array(std::move(a)));
}
} // namespace

extern "C" {

// Counts from now on are for the calling thread only; enabling resets them.
void mlx_test_kquant_counting(bool enable) {
  kquant::testing::family_counts.clear();
  kquant::testing::counting = enable;
}

uint64_t mlx_test_kquant_family_count(const char *family) {
  auto &counts = kquant::testing::family_counts;
  auto it = counts.find(std::string_view(family ? family : ""));
  return it == counts.end() ? 0 : it->second;
}

// The generation the Metal dispatcher sees (honours MLX_METAL_GPU_ARCH), or
// -1 without Metal.
int32_t mlx_test_kquant_gpu_gen() {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!mlx::core::metal::is_available())
      return -1;
    return mlx::core::metal::device(mlx::core::Device::gpu)
        .get_architecture_gen();
  } catch (...) {
    return -1;
  }
#else
  return -1;
#endif
}

mlx_array *mlx_test_kquant_quantized_matmul(mlx_array *x, mlx_array *w,
                                            mlx_array *scales,
                                            mlx_array *biases, bool transpose,
                                            int group_size, int bits,
                                            const char *mode, int32_t device) {
  try {
    return wrap(kquant::quantized_matmul(
        ref(x), ref(w), ref(scales), opt(biases), transpose, group_size, bits,
        mode_of(mode), device_of(device)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_kquant_quantized_matmul: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_test_kquant_gather_qmm(mlx_array *x, mlx_array *w,
                                      mlx_array *scales, mlx_array *biases,
                                      mlx_array *lhs_indices,
                                      mlx_array *rhs_indices, bool transpose,
                                      int group_size, int bits,
                                      const char *mode, bool sorted_indices,
                                      int32_t device) {
  try {
    return wrap(kquant::gather_qmm(ref(x), ref(w), ref(scales), opt(biases),
                                   opt(lhs_indices), opt(rhs_indices),
                                   transpose, group_size, bits, mode_of(mode),
                                   sorted_indices, device_of(device)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_kquant_gather_qmm: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array *mlx_test_kquant_dequantize(mlx_array *w, mlx_array *scales,
                                      mlx_array *biases, int group_size,
                                      int bits, int32_t out_dtype,
                                      const char *mode, int32_t device) {
  try {
    std::optional<mlx::core::Dtype> dtype = std::nullopt;
    if (out_dtype >= 0)
      dtype = to_mlx_dtype(out_dtype);
    return wrap(kquant::dequantize(ref(w), ref(scales), opt(biases), group_size,
                                   bits, mode_of(mode), dtype,
                                   device_of(device)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_kquant_dequantize: " << e.what() << std::endl;
    return nullptr;
  }
}

// Shapeless-compiles x @ w on `trace_x`, evaluates it there, then replays the
// same compiled function on `x`. Writes the replay and the eager result for x.
bool mlx_test_kquant_shapeless_replay(mlx_array *trace_x, mlx_array *x,
                                      mlx_array *w, mlx_array *scales,
                                      mlx_array *biases, bool transpose,
                                      int group_size, int bits,
                                      const char *mode, int32_t device,
                                      mlx_array **out_replay,
                                      mlx_array **out_eager) {
  try {
    auto kmode = mode_of(mode);
    auto dev = device_of(device);
    auto fn = [=](const std::vector<array> &in) {
      return std::vector<array>{kquant::quantized_matmul(
          in[0], in[1], in[2], in[3], transpose, group_size, bits, kmode, dev)};
    };
    auto compiled = mlx::core::compile(fn, /* shapeless = */ true);
    auto traced = compiled({ref(trace_x), ref(w), ref(scales), ref(biases)});
    mlx::core::eval(traced);
    auto replay = compiled({ref(x), ref(w), ref(scales), ref(biases)});
    auto eager = fn({ref(x), ref(w), ref(scales), ref(biases)});
    mlx::core::eval(replay);
    mlx::core::eval(eager);
    *out_replay = wrap(replay[0]);
    *out_eager = wrap(eager[0]);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_kquant_shapeless_replay: " << e.what() << std::endl;
    return false;
  }
}

} // extern "C"
