// W6.30 — MTPLX `multi3_qmv4` kernel port. See `mlx_qmv_multi3.h` for the
// envelope and contract. Implementation strategy:
//   - Metal header + source are kept byte-identical to MTPLX
//     (`verify_qmv.py::_multi3_qmv4_kernel`) so future MTPLX bumps can be
//     diff'd verbatim against this file.
//   - Per-(group_size, dtype) compiled-kernel cache mirrors MTPLX's
//     `@lru_cache(maxsize=None)`. Cache lookups are guarded by a mutex
//     because the cache is populated lazily on the hot path.
//   - The wrapper falls back to stock `quantized_matmul` outside the
//     eligibility envelope so callers can probe with malformed shapes
//     without risking invalid runtime behavior.

#include "mlx_qmv_multi3.h"

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace qmv_multi3 {

namespace {

using mlx::core::array;
using mlx::core::Dtype;

// ---------------------------------------------------------------------
// Metal header — declares the SIMD/block tiling constants and the
// `load_vector4_exact` / `qdot4_exact` helpers shared by every iteration.
// BYTE-IDENTICAL to MTPLX `verify_qmv.py:59-103`.
// ---------------------------------------------------------------------
constexpr const char* kKernelHeader = R"METAL(
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline float load_vector4_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
)METAL";

// ---------------------------------------------------------------------
// Metal source — the kernel body. BYTE-IDENTICAL to MTPLX
// `verify_qmv.py:105-183`.
// ---------------------------------------------------------------------
constexpr const char* kKernelSource = R"METAL(
        uint n_tile = threadgroup_position_in_grid.y;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;

        const device uint8_t* ws_base =
          (const device uint8_t*)w + out_row * in_vec_size_w
          + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* scales_base =
          scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* biases_base =
          biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

        const device T* x0_base = x + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x1_base = x + K + int(simd_lid) * VALUES_PER_THREAD;
        const device T* x2_base = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

        float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
        float result2[RESULTS_PER_SIMDGROUP] = {0.0f};
        float x0_thread[VALUES_PER_THREAD];
        float x1_thread[VALUES_PER_THREAD];
        float x2_thread[VALUES_PER_THREAD];

        const device uint8_t* ws = ws_base;
        const device T* sc = scales_base;
        const device T* bs = biases_base;
        const device T* x0 = x0_base;
        const device T* x1 = x1_base;
        const device T* x2 = x2_base;

        for (int k = 0; k < K; k += BLOCK_SIZE) {
          float sum0 = load_vector4_exact<T>(x0, x0_thread);
          float sum1 = load_vector4_exact<T>(x1, x1_thread);
          float sum2 = load_vector4_exact<T>(x2, x2_thread);

          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = out_row + row;
            if (n < N) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              const device T* sl = sc + row * in_vec_size_g;
              const device T* bl = bs + row * in_vec_size_g;
              float s = float(sl[0]);
              float b = float(bl[0]);
              result0[row] += qdot4_exact(wl, x0_thread, s, b, sum0);
              result1[row] += qdot4_exact(wl, x1_thread, s, b, sum1);
              result2[row] += qdot4_exact(wl, x2_thread, s, b, sum2);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          x0 += BLOCK_SIZE;
          x1 += BLOCK_SIZE;
          x2 += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          if (n < N) {
            float r0 = simd_sum(result0[row]);
            float r1 = simd_sum(result1[row]);
            float r2 = simd_sum(result2[row]);
            if (simd_lid == 0) {
              y[n] = T(r0);
              y[N + n] = T(r1);
              y[2 * N + n] = T(r2);
            }
          }
        }
)METAL";

// ---------------------------------------------------------------------
// Kernel cache — keyed by (group_size, dtype). MLX's
// `mlx::core::fast::CustomKernelFunction` is the C++ analogue of MTPLX's
// `mx.fast.metal_kernel(...)` return value; first lookup compiles, every
// subsequent lookup is O(1). Mutex protected because the cache is
// populated lazily on the hot path.
// ---------------------------------------------------------------------
struct CacheKey {
  int group_size;
  // We key on the Dtype `Val` enum's underlying integer so the hash is
  // trivial. Two different dtypes (bf16 vs fp16) produce two different
  // cached kernels because the template substitution `T` differs.
  int dtype_val;

  bool operator==(const CacheKey& other) const {
    return group_size == other.group_size && dtype_val == other.dtype_val;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const noexcept {
    // 16 bits is plenty for the group_size since we only accept {32, 64, 128}.
    return (static_cast<size_t>(k.group_size) << 8) ^ static_cast<size_t>(k.dtype_val);
  }
};

inline std::mutex& cache_mutex() {
  static std::mutex instance;
  return instance;
}

inline std::unordered_map<CacheKey, mlx::core::fast::CustomKernelFunction, CacheKeyHash>&
cache() {
  static std::unordered_map<CacheKey, mlx::core::fast::CustomKernelFunction, CacheKeyHash> instance;
  return instance;
}

// Return a tag string for the kernel name. Mirrors MTPLX's `dtype_tag` table.
inline const char* dtype_tag(Dtype dtype) {
  if (dtype == mlx::core::bfloat16) return "bf16";
  if (dtype == mlx::core::float16) return "fp16";
  return "unk";
}

// Compile (or fetch from cache) the multi3 qmv4 kernel for the given
// group_size + dtype pair. Throws via MLX if compilation fails.
mlx::core::fast::CustomKernelFunction& get_or_compile_kernel(int group_size, Dtype dtype) {
  CacheKey key{group_size, static_cast<int>(dtype.val())};
  std::lock_guard<std::mutex> lock(cache_mutex());
  auto it = cache().find(key);
  if (it != cache().end()) {
    return it->second;
  }
  std::string name = std::string("mtplx_multi3_qmv4_v2_gs")
      + std::to_string(group_size) + "_" + dtype_tag(dtype);
  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"x", "w", "scales", "biases", "K_size", "N_size"},
      {"y"},
      kKernelSource,
      kKernelHeader);
  auto [inserted, _ok] = cache().emplace(key, std::move(kernel));
  return inserted->second;
}

}  // namespace

array multi3_qmv4_matmul(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size) {
  // Helper to dispatch the stock baseline. Used both for in-envelope failures
  // (impossible by construction, but defensive) and out-of-envelope fallbacks.
  auto stock_fallback = [&]() -> array {
    return mlx::core::quantized_matmul(
        x, w, scales, std::optional<array>(biases), /*transpose=*/true,
        group_size, /*bits=*/4, /*mode=*/"affine");
  };

  // Shape gate: x must be at least 2D with x.shape[-2] == 3.
  if (x.ndim() < 2 || x.shape(-2) != 3) {
    return stock_fallback();
  }
  // Dtype gate: bf16/fp16 only, scales and biases must match.
  if (x.dtype() != mlx::core::bfloat16 && x.dtype() != mlx::core::float16) {
    return stock_fallback();
  }
  if (scales.dtype() != x.dtype() || biases.dtype() != x.dtype()) {
    return stock_fallback();
  }
  // Size gates: K = scales-last-dim * group_size; N = w.shape(0); divisibility.
  int K = x.shape(-1);
  int N = w.shape(0);
  // K must match w.shape(-1) * 8 (uint32 packed, 4 bits/value, 8 values/uint32).
  if (K != w.shape(-1) * 8) {
    return stock_fallback();
  }
  if (!is_eligible(group_size, x.dtype(), K, N)) {
    return stock_fallback();
  }
  // Leading dims must collapse to a single tile of M=3 rows.
  int batch_count = 1;
  for (int i = 0; i < x.ndim() - 2; ++i) {
    batch_count *= x.shape(i);
  }
  if (batch_count != 1) {
    return stock_fallback();
  }

  // Make the input contiguous and reshape to [3, K] to match the kernel's
  // index math. `reshape` will be a no-op when the input is already
  // contiguous and rank-2.
  auto x2 = mlx::core::reshape(x, {3, K});

  auto K_arr = mlx::core::array(K, mlx::core::int32);
  auto N_arr = mlx::core::array(N, mlx::core::int32);

  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
      {"T", x.dtype()},
      {"GS", group_size},
  };

  auto& kernel = get_or_compile_kernel(group_size, x.dtype());
  auto results = kernel(
      {x2, w, scales, biases, K_arr, N_arr},
      {mlx::core::Shape{3, N}},
      {x.dtype()},
      // Grid: (32, 2 * (N / 8), 1) — one tile per BN=8 output rows.
      std::make_tuple(32, 2 * (N / 8), 1),
      // Threadgroup: (32, 2, 1) — one simdgroup per RESULTS_PER_SIMDGROUP=4 rows.
      std::make_tuple(32, 2, 1),
      template_args,
      std::nullopt,
      /*verbose=*/false,
      mlx::core::default_stream(mlx::core::Device::gpu));

  array y = std::move(results[0]);

  // Reshape back to original leading shape, with M=3 preserved.
  if (x.ndim() == 2) {
    return y;
  }
  mlx::core::Shape out_shape;
  out_shape.reserve(x.ndim());
  for (int i = 0; i < x.ndim() - 2; ++i) {
    out_shape.push_back(x.shape(i));
  }
  out_shape.push_back(3);
  out_shape.push_back(N);
  return mlx::core::reshape(y, out_shape);
}

}  // namespace qmv_multi3

// =====================================================================
// extern "C" FFI surface for Rust-side parity tests and microbench.
// =====================================================================

extern "C" {

mlx_array* mlx_multi3_qmv4_matmul(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size) {
  if (!x || !w || !scales || !biases) return nullptr;
  try {
    auto& x_arr = *reinterpret_cast<mlx::core::array*>(x);
    auto& w_arr = *reinterpret_cast<mlx::core::array*>(w);
    auto& s_arr = *reinterpret_cast<mlx::core::array*>(scales);
    auto& b_arr = *reinterpret_cast<mlx::core::array*>(biases);
    auto out = qmv_multi3::multi3_qmv4_matmul(x_arr, w_arr, s_arr, b_arr, group_size);
    return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(out)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_multi3_qmv4_matmul error: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array* mlx_stock_qmv4_matmul(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size) {
  if (!x || !w || !scales || !biases) return nullptr;
  try {
    auto& x_arr = *reinterpret_cast<mlx::core::array*>(x);
    auto& w_arr = *reinterpret_cast<mlx::core::array*>(w);
    auto& s_arr = *reinterpret_cast<mlx::core::array*>(scales);
    auto& b_arr = *reinterpret_cast<mlx::core::array*>(biases);
    auto out = mlx::core::quantized_matmul(
        x_arr, w_arr, s_arr, std::optional<mlx::core::array>(b_arr), /*transpose=*/true,
        group_size, /*bits=*/4, /*mode=*/"affine");
    return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(out)));
  } catch (const std::exception& e) {
    std::cerr << "mlx_stock_qmv4_matmul error: " << e.what() << std::endl;
    return nullptr;
  }
}

}  // extern "C"
