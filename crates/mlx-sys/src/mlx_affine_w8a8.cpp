// Affine-group W8A8 GEMM — TILED matmul2d-per-group primitive (v2).
//
// Keeps the model's EXACT affine packed weight (no re-quant) and makes the op
// FAST by mirroring the symmetric W8A8 path's prepare/linear split:
//   * mlx_affine_w8a8_prepare        (LOAD-TIME, runs once) — unpack the affine
//     uint32 weight into the SIGNED int8 [K,N] kernel operand + keep scale +
//     precompute bias_adj = 128*scale + bias.
//   * mlx_affine_w8a8_linear_prepared (per FORWARD, timed) — per-token int8 act
//     quant + per-group act-sum S, then the TILED grouped matmul2d GEMM.
//   * mlx_affine_w8a8_linear          (thin wrapper: prepare + linear) so the
//     existing correctness/fallback tests keep one entry point.
//
// MATH (q affine UNSIGNED in [0,255]; w[n,k]=scale[n,g]*q[n,k]+bias[n,g];
// x_q per-token symmetric int8, s_x[m]=absmax/127):
//   Shift to signed so matmul2d<int8_t> accepts the weight:
//     q_s = q - 128 in [-128,127];  bias_adj = 128*scale + bias.
//     P_s[m,n,g] = sum_{k in g} x_q[m,k] * q_s[n,k]   (one matmul2d / K-group)
//     S[m,g]     = sum_{k in g} x_q[m,k]              (act group-sum, n-indep)
//     y[m,n]     = s_x[m] * sum_g ( scale[n,g]*P_s + bias_adj[n,g]*S ) -> bf16.
//   Proof: P = sum x_q*q = sum x_q*(q_s+128) = P_s + 128*S, so
//     scale*P + bias*S = scale*P_s + (128*scale+bias)*S = scale*P_s + bias_adj*S.
//
// The TILED GEMM (na_affine_w8a8_gemm.metal.inc) runs ONE
// mpp::tensor_ops::matmul2d<int8,int8->int32> per K=64 group into a COOPERATIVE
// TENSOR (in-register), folds scale*P_s + bias_adj*S into an f32 cooperative
// accumulator, then writes s_x[m]*acc -> bf16. Decisive structural finding:
// matmul2d CAN target threadgroup/cooperative destinations, so the per-group P
// stays in registers — NO device round-trips. See the kernel inc for why a
// 128x64 device/threadgroup P tile is infeasible (32 KB = the whole tg budget;
// per-group device P = K/64 round-trips).

#include "mlx_common.h"

namespace {

const char* kAffineW8a8GemmBody =
#include "metal/na_affine_w8a8_gemm.metal.inc"
    ;

const char* kAffineW8a8GemmHeader =
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace mpp::tensor_ops;\n";

// Fused per-token activation quant + per-group act-sum S (extends the symmetric
// quant with the S output the affine bias term needs).
const char* kAffineW8a8QuantSBody =
#include "metal/na_affine_w8a8_quant_s.metal.inc"
    ;

// Load-time weight unpack: affine uint32 [N,K/4] -> signed int8 [K,N].
const char* kAffineW8a8UnpackBody =
#include "metal/na_affine_w8a8_unpack.metal.inc"
    ;

// Architecture gate (cached). Mirrors na_int8_gpu_gen() in mlx_na_int8.cpp.
int affine_w8a8_gpu_gen() {
  static int gen = []() {
    try {
      auto& info = mlx::core::gpu::device_info(0);
      auto it = info.find("architecture");
      if (it == info.end()) return 0;
      auto& arch = std::get<std::string>(it->second);
      if (arch.size() < 3) return 0;
      int g = 0;
      size_t i = arch.size() - 2;
      int multiplier = 1;
      while (i > 0 && arch[i] >= '0' && arch[i] <= '9') {
        g += (arch[i] - '0') * multiplier;
        multiplier *= 10;
        i--;
      }
      return g;
    } catch (...) {
      return 0;
    }
  }();
  return gen;
}

// Cached JIT kernel for the TILED affine-group W8A8 GEMM.
mlx::core::fast::CustomKernelFunction& get_affine_w8a8_gemm_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_affine_w8a8_gemm",
        /* input_names  */
        {"x_i8", "q_s", "scl", "badj", "s_x", "S", "M", "N", "K", "G"},
        /* output_names */ {"y"},
        /* source(body) */ kAffineW8a8GemmBody,
        /* header       */ kAffineW8a8GemmHeader,
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// Cached JIT kernel for the fused activation-quant + per-group act-sum S.
mlx::core::fast::CustomKernelFunction& get_affine_w8a8_quant_s_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_affine_w8a8_quant_s",
        /* input_names  */ {"x", "M", "K", "G"},
        /* output_names */ {"x_i8", "s_x", "S"},
        /* source(body) */ kAffineW8a8QuantSBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// Cached JIT kernel for the load-time weight unpack.
mlx::core::fast::CustomKernelFunction& get_affine_w8a8_unpack_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_affine_w8a8_unpack",
        /* input_names  */ {"wq", "N", "K"},
        /* output_names */ {"q_s"},
        /* source(body) */ kAffineW8a8UnpackBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// Per-token symmetric int8 activation quant + per-group act-sum S.
//   x[M,K] bf16 -> {x_i8 int8 [M,K], s_x f32 [M,1], S int32 [M, K/G]}.
// One threadgroup per row m (grid.y = M), 256 threads/group.
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
affine_act_quant_s(const mlx::core::array& x, int group_size) {
  using namespace mlx::core;
  const int M = x.shape(0);
  const int K = x.shape(1);
  const int n_groups = K / group_size;
  array M_arr(M, int32);
  array K_arr(K, int32);
  array G_arr(group_size, int32);
  std::tuple<int, int, int> grid{256, M, 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};
  auto results = get_affine_w8a8_quant_s_kernel()(
      {x, M_arr, K_arr, G_arr},
      /* output_shapes */ {Shape{M, K}, Shape{M, 1}, Shape{M, n_groups}},
      /* output_dtypes */ {int8, float32, int32},
      grid,
      threadgroup,
      /* template_args */ {},
      /* init_value */ std::nullopt,  // every element written; no fill
      /* verbose */ false,
      default_stream(Device::gpu));
  return {results[0], results[1], results[2]};
}

// The TILED grouped GEMM core. x_i8 [M,K] int8, q_s [K,N] int8 (signed,
// pre-unpacked), scl/badj [N, K/G] f32, s_x [M] f32, S [M, K/G] int32.
// Returns bf16 [M,N]. One 128x64 output tile per threadgroup (256 threads,
// 8 simdgroups) — same grid scheme as the symmetric kernel.
mlx::core::array affine_gemm_core(const mlx::core::array& x_i8,
                                  const mlx::core::array& q_s,
                                  const mlx::core::array& scl,
                                  const mlx::core::array& badj,
                                  const mlx::core::array& s_x,
                                  const mlx::core::array& S,
                                  int group_size) {
  using namespace mlx::core;
  const int M = x_i8.shape(0);
  const int K = x_i8.shape(1);
  const int N = q_s.shape(1);  // q_s is [K,N]
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);
  array G_arr(group_size, int32);

  const int tgN = (N + 64 - 1) / 64;
  const int tgM = (M + 128 - 1) / 128;
  // dispatch_threads: 256 threads (8 simdgroups) per output tile.
  // threadgroup_position_in_grid.x = 0..tgN-1, .y = 0..tgM-1.
  std::tuple<int, int, int> grid{256 * tgN, tgM, 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};

  auto results = get_affine_w8a8_gemm_kernel()(
      {x_i8, q_s, scl, badj, s_x, S, M_arr, N_arr, K_arr, G_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {bfloat16},
      grid,
      threadgroup,
      /* template_args */ {},
      /* init_value */ std::nullopt,  // every in-bounds element written; no fill
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

}  // namespace

extern "C" {

// LOAD-TIME prepare (runs once per quantized linear). Unpacks the affine packed
// weight into the SIGNED int8 [K,N] kernel operand the tiled matmul2d wants,
// keeps the f32 scale, and precomputes bias_adj = 128*scale + bias.
//
//   packed_w: [N, K/4] uint32 affine 8-bit packed weight (model's EXACT weight)
//   scales:   [N, K/group_size] f32
//   biases:   [N, K/group_size] f32
//   group_size, bits (must be 8)
//   -> out_q_s:    opaque int8 [K,N] (q-128, kernel operand)
//      out_scale:  f32 [N, K/group_size] (scale kept)
//      out_badj:   f32 [N, K/group_size] (= 128*scale + bias)
//
// Returns false (Rust falls back) on gen<17, bits!=8, K%group_size!=0, or error.
bool mlx_affine_w8a8_prepare(mlx_array* packed_w_handle,
                             mlx_array* scales_handle,
                             mlx_array* biases_handle, int group_size, int bits,
                             mlx_array** out_q_s, mlx_array** out_scale,
                             mlx_array** out_badj) {
  using namespace mlx::core;
  if (out_q_s) *out_q_s = nullptr;
  if (out_scale) *out_scale = nullptr;
  if (out_badj) *out_badj = nullptr;
  try {
    auto& wq = *reinterpret_cast<array*>(packed_w_handle);
    auto& scl = *reinterpret_cast<array*>(scales_handle);
    auto& bia = *reinterpret_cast<array*>(biases_handle);
    if (wq.ndim() != 2 || scl.ndim() != 2 || bia.ndim() != 2) {
      std::cerr << "[mlx_affine_w8a8_prepare] expected 2D packed_w, scales, "
                   "biases\n";
      return false;
    }
    if (bits != 8) {
      std::cerr << "[mlx_affine_w8a8_prepare] unsupported bits=" << bits
                << " (only 8)\n";
      return false;
    }
    if (affine_w8a8_gpu_gen() < 17) {
      std::cerr << "[mlx_affine_w8a8_prepare] unsupported gen="
                << affine_w8a8_gpu_gen() << " (NA needs M5+)\n";
      return false;
    }
    const int N = wq.shape(0);
    const int K = wq.shape(1) * 4;  // 4 uint8 per uint32 at 8-bit
    if (group_size <= 0 || (K % group_size) != 0) {
      std::cerr << "[mlx_affine_w8a8_prepare] K % group_size != 0: K=" << K
                << " group_size=" << group_size << "\n";
      return false;
    }
    const int n_groups = K / group_size;
    if (scl.shape(0) != N || scl.shape(1) != n_groups || bia.shape(0) != N ||
        bia.shape(1) != n_groups) {
      std::cerr << "[mlx_affine_w8a8_prepare] scales/biases shape mismatch: "
                   "expected [N="
                << N << ", K/G=" << n_groups << "]\n";
      return false;
    }

    array wq_u32 = (wq.dtype() == uint32) ? wq : astype(wq, uint32);
    array scl_f32 = (scl.dtype() == float32) ? scl : astype(scl, float32);
    array bia_f32 = (bia.dtype() == float32) ? bia : astype(bia, float32);

    // Unpack uint32 [N,K/4] -> signed int8 [K,N] via the JIT kernel.
    array N_arr(N, int32);
    array K_arr(K, int32);
    std::tuple<int, int, int> grid{N, K, 1};
    std::tuple<int, int, int> threadgroup{256, 1, 1};
    auto unpacked = get_affine_w8a8_unpack_kernel()(
        {wq_u32, N_arr, K_arr},
        /* output_shapes */ {Shape{K, N}},
        /* output_dtypes */ {int8},
        grid,
        threadgroup,
        /* template_args */ {},
        /* init_value */ std::nullopt,
        /* verbose */ false,
        default_stream(Device::gpu));
    array q_s = unpacked[0];  // int8 [K,N]

    // bias_adj = 128*scale + bias  (f32 [N, K/G]).
    array badj = add(multiply(array(128.0f, float32), scl_f32), bia_f32);

    eval(q_s);
    eval(scl_f32);
    eval(badj);
    *out_q_s = reinterpret_cast<mlx_array*>(new array(std::move(q_s)));
    *out_scale = reinterpret_cast<mlx_array*>(new array(std::move(scl_f32)));
    *out_badj = reinterpret_cast<mlx_array*>(new array(std::move(badj)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_affine_w8a8_prepare] EXCEPTION: " << e.what()
              << std::endl;
    if (out_q_s) *out_q_s = nullptr;
    if (out_scale) *out_scale = nullptr;
    if (out_badj) *out_badj = nullptr;
    return false;
  }
}

// Per-FORWARD prepared linear (the TIMED hot path). Per-token int8 act quant +
// per-group act-sum S, then the TILED grouped matmul2d GEMM.
//
//   x:     [M,K] bf16 activations
//   q_s:   [K,N] int8 (signed, from mlx_affine_w8a8_prepare)
//   scale: [N, K/group_size] f32 (kept)
//   badj:  [N, K/group_size] f32 (= 128*scale + bias)
//   group_size
//   -> out: bf16 [M,N]. Output is LAZY (composes into the forward graph).
//
// Returns false (Rust falls back) on gen<17, K%group_size!=0, or error.
bool mlx_affine_w8a8_linear_prepared(mlx_array* x_handle, mlx_array* q_s_handle,
                                     mlx_array* scale_handle,
                                     mlx_array* badj_handle, int group_size,
                                     mlx_array** out) {
  using namespace mlx::core;
  if (out) *out = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& q_s = *reinterpret_cast<array*>(q_s_handle);
    auto& scale = *reinterpret_cast<array*>(scale_handle);
    auto& badj = *reinterpret_cast<array*>(badj_handle);
    if (x.ndim() != 2 || q_s.ndim() != 2 || scale.ndim() != 2 ||
        badj.ndim() != 2) {
      std::cerr << "[mlx_affine_w8a8_linear_prepared] expected 2D x, q_s, "
                   "scale, badj\n";
      return false;
    }
    const int M = x.shape(0);
    const int K = x.shape(1);
    if (q_s.shape(0) != K) {
      std::cerr << "[mlx_affine_w8a8_linear_prepared] K mismatch: x.K=" << K
                << " q_s.K=" << q_s.shape(0) << "\n";
      return false;
    }
    if (group_size <= 0 || (K % group_size) != 0) {
      std::cerr << "[mlx_affine_w8a8_linear_prepared] K % group_size != 0: K="
                << K << " group_size=" << group_size << "\n";
      return false;
    }
    if (affine_w8a8_gpu_gen() < 17) {
      std::cerr << "[mlx_affine_w8a8_linear_prepared] unsupported gen="
                << affine_w8a8_gpu_gen() << " (NA needs M5+)\n";
      return false;
    }

    array x_bf16 = (x.dtype() == bfloat16) ? x : astype(x, bfloat16);
    // Per-token int8 act quant + per-group act-sum S (one fused kernel).
    auto [x_i8, s_x, S] = affine_act_quant_s(x_bf16, group_size);
    array s_x_flat = reshape(s_x, {M});  // [M,1] -> [M]

    array scl_f32 = (scale.dtype() == float32) ? scale : astype(scale, float32);
    array badj_f32 = (badj.dtype() == float32) ? badj : astype(badj, float32);
    array q_s_i8 = (q_s.dtype() == int8) ? q_s : astype(q_s, int8);

    array y = affine_gemm_core(x_i8, q_s_i8, scl_f32, badj_f32, s_x_flat, S,
                               group_size);
    *out = reinterpret_cast<mlx_array*>(new array(std::move(y)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_affine_w8a8_linear_prepared] EXCEPTION: " << e.what()
              << std::endl;
    if (out) *out = nullptr;
    return false;
  }
}

// Thin wrapper: prepare (unpack + bias_adj) + linear_prepared, in one call.
// Keeps the existing public entry point (and the correctness/fallback tests)
// working unchanged; the microbench now calls prepare ONCE then times
// linear_prepared, so this wrapper is the convenience/parity path, not the
// timed path.
//
//   x:          [M,K] bf16 activations
//   packed_w:   [N, K/4] uint32 affine 8-bit packed weight (model's EXACT weight)
//   scales:     [N, K/group_size] f32
//   biases:     [N, K/group_size] f32
//   group_size, bits (must be 8)
//   -> out: bf16 [M,N] = x @ dequant(packed_w)^T with int8-quantized activation.
//
// Returns false (Rust falls back) on gen<17, bits!=8, K%group_size!=0, or error.
bool mlx_affine_w8a8_linear(mlx_array* x_handle, mlx_array* packed_w_handle,
                            mlx_array* scales_handle, mlx_array* biases_handle,
                            int group_size, int bits, mlx_array** out) {
  using namespace mlx::core;
  if (out) *out = nullptr;
  mlx_array* q_s = nullptr;
  mlx_array* scale = nullptr;
  mlx_array* badj = nullptr;
  bool prepared = mlx_affine_w8a8_prepare(packed_w_handle, scales_handle,
                                          biases_handle, group_size, bits, &q_s,
                                          &scale, &badj);
  if (!prepared) {
    return false;
  }
  bool ok = mlx_affine_w8a8_linear_prepared(x_handle, q_s, scale, badj,
                                            group_size, out);
  // Free the intermediate prepared handles (the wrapper owns them).
  delete reinterpret_cast<array*>(q_s);
  delete reinterpret_cast<array*>(scale);
  delete reinterpret_cast<array*>(badj);
  return ok;
}

}  // extern "C"
