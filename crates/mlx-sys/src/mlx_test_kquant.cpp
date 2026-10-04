// TEST-ONLY hooks: the bridge kernel-family counter store, explicit-device
// K-quant entry points and a shapeless-compile replay. No production caller.

#include "mlx_common.h"
#include "mlx_kquant.h"
#include "mlx_test_counters.h"

#include <cstring>
#include <limits>
#include <map>
#include <string>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx_paged_metallib.h"

#include <set>
#include <sstream>
#endif

using mlx::core::array;
namespace kquant = mlx::core::kquant;

namespace mlx::core::bridge_testing {
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
} // namespace mlx::core::bridge_testing

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
  mlx::core::bridge_testing::family_counts.clear();
  mlx::core::bridge_testing::counting = enable;
}

uint64_t mlx_test_kquant_family_count(const char *family) {
  auto &counts = mlx::core::bridge_testing::family_counts;
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

// Checks paged_attn.metallib against kquant::metal_kernel_names(). `counts`
// receives {base names, NAX names, K-quant functions in the library, pipelines
// built}. `report` receives one line per missing name (one the dispatcher can
// request on this device) and per unexpected K-quant function in the library.
// `build_pipelines` also builds every required pipeline. False without Metal,
// on error, or when `report` cannot hold the text.
bool mlx_test_kquant_metallib_check(bool build_pipelines, int64_t *counts,
                                    char *report, size_t len) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!counts || !report || len == 0 || !mlx::core::metal::is_available())
      return false;
    auto &d = mlx::core::metal::device(mlx::core::Device::gpu);
    auto *lib = mlx::core::fast::paged::get_paged_attn_library(d);
    bool nax = mlx::core::metal::is_nax_available();

    std::set<std::string> in_library;
    NS::Array *functions = lib->functionNames();
    for (NS::UInteger i = 0; i < functions->count(); ++i) {
      std::string name = functions->object<NS::String>(i)->utf8String();
      auto prefix = name.substr(0, name.find('_'));
      if (kquant::parse_mode(prefix) || name.rfind("kquant_", 0) == 0) {
        in_library.insert(std::move(name));
      }
    }

    std::ostringstream out;
    std::set<std::string> known;
    int64_t base = 0, nax_names = 0, built = 0;
    for (const auto &k : kquant::metal_kernel_names()) {
      known.insert(k.name);
      (k.nax ? nax_names : base)++;
      if (k.nax && !nax)
        continue;
      if (!in_library.count(k.name)) {
        out << "missing " << k.name << "\n";
        continue;
      }
      if (!build_pipelines)
        continue;
      if (k.name.find("_gather_qmm_rhs_") != std::string::npos) {
        bool align = true;
        mlx::core::metal::MTLFCList consts = {
            {&align, MTL::DataType::DataTypeBool, 200},
            {&align, MTL::DataType::DataTypeBool, 201},
            {&align, MTL::DataType::DataTypeBool, 202},
        };
        d.get_kernel(k.name, lib, k.name + "_align_M_t_align_N_t_align_K_t",
                     consts);
      } else {
        d.get_kernel(k.name, lib);
      }
      ++built;
    }
    for (const auto &name : in_library) {
      if (!known.count(name))
        out << "unexpected " << name << "\n";
    }
    counts[0] = base;
    counts[1] = nax_names;
    counts[2] = static_cast<int64_t>(in_library.size());
    counts[3] = built;
    std::string text = out.str();
    if (text.size() + 1 > len)
      return false;
    std::memcpy(report, text.c_str(), text.size() + 1);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_kquant_metallib_check: " << e.what() << std::endl;
    return false;
  }
#else
  (void)build_pipelines;
  (void)counts;
  (void)report;
  (void)len;
  return false;
#endif
}

// The Metal architecture name: the hardware's own when `hardware`, else the
// one the dispatcher routes by (honours MLX_METAL_GPU_ARCH). False without
// Metal or when `out` is too small.
bool mlx_test_metal_architecture(bool hardware, char *out, size_t len) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!out || !mlx::core::metal::is_available())
      return false;
    auto &d = mlx::core::metal::device(mlx::core::Device::gpu);
    std::string name =
        hardware
            ? std::string(d.mtl_device()->architecture()->name()->utf8String())
            : d.get_architecture();
    if (name.size() + 1 > len)
      return false;
    std::memcpy(out, name.c_str(), name.size() + 1);
    return true;
  } catch (...) {
    return false;
  }
#else
  (void)hardware;
  (void)out;
  (void)len;
  return false;
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
