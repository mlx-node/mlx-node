// NA (Neural Accelerator) int8 W8A8 prefill GEMM primitives.
//
// Three FFI ops, all gated internally on M5+ (gen>=17) and K % 16 == 0:
//   1. mlx_matmul_int8(x, w, &out_i32)        — int8 x @ w^T -> int32 [M,N]
//   2. mlx_quantize_weight_int8(w, &wkn, &sw) — per-output-channel symmetric int8,
//        weight pre-transposed to the [K,N] kernel layout at LOAD time
//   3. mlx_w8a8_linear(x, wkn, sw, &out_bf16) — per-token act quant + GEMM + rescale
//
// The int8 GEMM is the proven Stage-0 mpp::tensor_ops::matmul2d<int8,int32>
// (128x64 tile, 8 simdgroups) run through fast::metal_kernel JIT. Rust has NO
// Int8 DType, so int8 lives entirely C++-side: Stage-1 callers pass bf16/f32
// arrays holding integer values in [-127,127] which are cast to int8 here.
//
// Layout contract (see metal/na_int8_gemm.metal.inc):
//   kernel math (host view): C[m,n] = sum_k A[m,k] * B[k,n]
//   with A = [M,K] row-major, B = [K,N] row-major.
//   We want y = x @ w^T where w is [N,K] row-major (weight rows = output chans).
//   => B must be w^T, i.e. a [K,N] contiguous buffer with B[k,n] = w[n,k].
//
// STAGE 4b: that [K,N] transpose+contiguous is hoisted to LOAD time.
// `mlx_quantize_weight_int8` quantizes the [N,K] weight and stores the int8
// result ALREADY in the contiguous [K,N] layout the kernel's column-major (N,K)
// view consumes. The per-forward `int8_gemm_core` then consumes that buffer
// directly with ZERO weight reshaping. The result C[m,n] = sum_k x[m,k]*w[n,k]
// = (x @ w^T)[m,n]. The per-output-channel scale s_w[N] is unchanged — it always
// indexes the output channel N regardless of how the int8 weight is stored.

#include "mlx_common.h"

namespace {

const char* kNaInt8GemmBody =
#include "metal/na_int8_gemm.metal.inc"
    ;

const char* kNaInt8GemmHeader =
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace mpp::tensor_ops;\n";

// v1 FUSED PREFILL KERNELS.
// (2) Per-token dynamic symmetric int8 activation quant (one kernel, two strided
//     global passes over x). Replaces the ~5-7-op lazy quant chain.
const char* kNaInt8QuantBody =
#include "metal/na_int8_quant.metal.inc"
    ;
// (3) int32 -> bf16 accumulator rescale (one elementwise pass). Replaces the
//     multi-pass lazy rescale (astype/f32-mul/mul/astype-bf16).
const char* kNaInt8RescaleBody =
#include "metal/na_int8_rescale.metal.inc"
    ;
// sym8 DECODE matvec (QMV). Memory-BW-bound matvec for small M (decode): the
// prefill 128x64 GEMM wastes 127/128 rows at M=1, so decode needs a dedicated
// matvec that streams each int8 weight byte once + applies s_x[m]*s_w[n] inline.
// Reuses the shared na_int8_quant kernel for the activation int8 quant.
const char* kNaInt8QmvBody =
#include "metal/na_int8_qmv.metal.inc"
    ;
// FUSED sym8 DECODE matvec: takes bf16 x directly, folds the per-token int8
// activation quant INTO the matvec kernel (single kernel, like affine qmv).
const char* kNaInt8QmvFusedBody =
#include "metal/na_int8_qmv_fused.metal.inc"
    ;
// W8A16 sym8 DECODE matvec: bf16 x read directly, NO activation quant at all —
// single-pass f32-accumulate mixed-precision matvec. The act-quant passes of
// the fused W8A8 qmv are pure overhead at M=1 (e2e -0.50x vs affine-Q8); this
// kernel removes them AND makes decode activation-exact.
const char* kNaInt8QmvW8a16Body =
#include "metal/na_int8_qmv_w8a16.metal.inc"
    ;
// W8A16 sym8 DECODE matvec, MLX-affine-qmv style (the PRODUCTION decode
// kernel): simdgroup-per-rows on the [N,K] checkpoint orientation, wide
// contiguous per-lane weight loads, simd_sum epilogue, zero threadgroup
// memory/barriers. Fixes the 2D-block kernel's DRAM streaming deficit
// (54.7us -> ~affine-parity in the in-stream decode chain).
const char* kNaInt8QmvW8a16SgBody =
#include "metal/na_int8_qmv_w8a16_sg.metal.inc"
    ;

// MEASUREMENT ONLY (diagnostic). Identical to kNaInt8GemmBody but uses
// mode::multiply (overwrite C) instead of multiply_accumulate, so the dispatch
// does NOT require C pre-zeroed — letting us call metal_kernel with
// init_value=nullopt and skip MLX's per-call full-output fill_gpu. Used to
// isolate how much of the in-engine GEMM cost is that zero-fill pass.
const char* kNaInt8GemmBodyMul = R"(
  constexpr int TM2 = 128;
  constexpr int TN2 = 64;
  constexpr int SG2 = 8;
  constexpr auto desc = matmul2d_descriptor(
      TM2, TN2, static_cast<int>(dynamic_extent),
      false, false, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG2>> op;
  const int Mi = M;
  const int Ni = N;
  const int Ki = K;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(
      const_cast<device int8_t*>(A), dextents<int32_t, 2>(Ki, Mi));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(
      const_cast<device int8_t*>(B), dextents<int32_t, 2>(Ni, Ki));
  tensor<device int32_t, dextents<int32_t, 2>, tensor_inline> tC(
      C, dextents<int32_t, 2>(Ni, Mi));
  const uint m0 = threadgroup_position_in_grid.y * TM2;
  const uint n0 = threadgroup_position_in_grid.x * TN2;
  auto mA = tA.slice(0, m0);
  auto mB = tB.slice(n0, 0);
  auto mC = tC.slice(n0, m0);
  op.run(mA, mB, mC);
)";

// Architecture gate. M5 = gen 17 is the first arch with the Neural Accelerator
// matmul2d path; below that the JIT compile fails outright. Cached once.
int na_int8_gpu_gen() {
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

bool na_int8_supported(int K) {
  return na_int8_gpu_gen() >= 17 && (K % 16) == 0;
}

// Cached JIT GEMM kernel (single variant — fixed 128x64 tile).
mlx::core::fast::CustomKernelFunction& get_int8_gemm_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_gemm",
        /* input_names  */ {"A", "B", "M", "N", "K"},
        /* output_names */ {"C"},
        /* source(body) */ kNaInt8GemmBody,
        /* header       */ kNaInt8GemmHeader,
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// MEASUREMENT ONLY. Cached JIT kernel for the mode::multiply variant.
mlx::core::fast::CustomKernelFunction& get_int8_gemm_kernel_mul() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_gemm_mul",
        {"A", "B", "M", "N", "K"},
        {"C"},
        kNaInt8GemmBodyMul,
        kNaInt8GemmHeader,
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// v1 (2): cached JIT kernel for the fused activation-quant.
mlx::core::fast::CustomKernelFunction& get_int8_quant_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_quant",
        /* input_names  */ {"x", "M", "K"},
        /* output_names */ {"x_i8", "s_x"},
        /* source(body) */ kNaInt8QuantBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// v1 (3): cached JIT kernel for the fused int32->bf16 rescale.
mlx::core::fast::CustomKernelFunction& get_int8_rescale_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_rescale",
        /* input_names  */ {"acc", "s_x", "s_w", "M", "N"},
        /* output_names */ {"y"},
        /* source(body) */ kNaInt8RescaleBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// sym8 DECODE: cached JIT kernel for the int8 matvec (QMV). Streams the [K,N]
// int8 weight + applies the s_x[m]*s_w[n] rescale inline -> bf16 [M,N].
mlx::core::fast::CustomKernelFunction& get_int8_qmv_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_qmv",
        /* input_names  */ {"x_i8", "w_i8", "s_x", "s_w", "M", "N", "K"},
        /* output_names */ {"y"},
        /* source(body) */ kNaInt8QmvBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// FUSED sym8 DECODE matvec kernel (bf16 x in, single kernel). Folds the per-token
// int8 activation quant into the matvec — no separate na_int8_quant pass.
mlx::core::fast::CustomKernelFunction& get_int8_qmv_fused_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_qmv_fused",
        /* input_names  */ {"x", "w_i8", "s_w", "M", "N", "K"},
        /* output_names */ {"y"},
        /* source(body) */ kNaInt8QmvFusedBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// W8A16 sym8 DECODE matvec kernel (bf16 x in, NO act quant, f32 accumulate).
mlx::core::fast::CustomKernelFunction& get_int8_qmv_w8a16_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_qmv_w8a16",
        /* input_names  */ {"x", "w_i8", "s_w", "M", "N", "K"},
        /* output_names */ {"y"},
        /* source(body) */ kNaInt8QmvW8a16Body,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// W8A16 sym8 DECODE matvec kernel, simd_sum style ([N,K] checkpoint
// orientation — see metal/na_int8_qmv_w8a16_sg.metal.inc).
mlx::core::fast::CustomKernelFunction& get_int8_qmv_w8a16_sg_kernel() {
  static std::mutex mtx;
  static std::optional<mlx::core::fast::CustomKernelFunction> kernel;
  std::lock_guard<std::mutex> lock(mtx);
  if (!kernel.has_value()) {
    kernel = mlx::core::fast::metal_kernel(
        "na_int8_qmv_w8a16_sg",
        /* input_names  */ {"x", "w_i8", "s_w", "M", "N", "K"},
        /* output_names */ {"y"},
        /* source(body) */ kNaInt8QmvW8a16SgBody,
        /* header       */ "",
        /* ensure_row_contiguous */ true,
        /* atomic_outputs        */ false);
  }
  return kernel.value();
}

// v1 (2): fused per-token int8 activation quant. Input x [M,K] bf16; returns a
// pair {x_i8 int8 [M,K], s_x f32 [M,1]}. ONE kernel (two strided passes over x)
// replaces the lazy absmax/round/clip/astype chain. Output is LAZY — composes
// into the surrounding graph; the caller evals at end of forward. Bit-identical
// to the lazy chain (same f32 arithmetic; MLX `round` == metal::rint).
std::pair<mlx::core::array, mlx::core::array> int8_act_quant(
    const mlx::core::array& x) {
  using namespace mlx::core;
  const int M = x.shape(0);
  const int K = x.shape(1);
  array M_arr(M, int32);
  array K_arr(K, int32);
  // One threadgroup per row (grid.y = M), 256 threads/group (grid.x = 256).
  std::tuple<int, int, int> grid{256, M, 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};
  auto results = get_int8_quant_kernel()(
      {x, M_arr, K_arr},
      /* output_shapes */ {Shape{M, K}, Shape{M, 1}},
      /* output_dtypes */ {int8, float32},
      grid,
      threadgroup,
      /* template_args */ {},
      /* init_value */ std::nullopt,  // every element written; no fill needed
      /* verbose */ false,
      default_stream(Device::gpu));
  return {results[0], results[1]};
}

// v1 (3): fused int32->bf16 rescale. acc [M,N] int32, s_x [M,1] f32 (or [M]),
// s_w [N] f32. Returns y bf16 [M,N] = (bfloat16)((float)acc * s_x * s_w), with
// the s_x-first association of the lazy chain. Output is LAZY. Narrowed to bf16
// inside the kernel (pitfall guard).
mlx::core::array int8_rescale(const mlx::core::array& acc,
                              const mlx::core::array& s_x,
                              const mlx::core::array& s_w) {
  using namespace mlx::core;
  const int M = acc.shape(0);
  const int N = acc.shape(1);
  array M_arr(M, int32);
  array N_arr(N, int32);
  // 2D dispatch: grid.x = N (cols), grid.y = M (rows). The metal_kernel grid
  // tuple is int-typed (std::tuple<int,int,int>), so a FLAT M*N grid would
  // static_cast<int> a value that overflows negative at M*N >= 2^31 (review #7).
  // Splitting per-axis keeps each extent in the wide regime each holds in real
  // shapes (N <= ~35k, M <= tens of thousands) — neither dimension overflows int
  // and the kernel reconstructs a 64-bit flat offset = m*N + n. MLX dispatches
  // via dispatch_threads (total threads == grid), so no caller-side round-up and
  // no out-of-range threads. Element coverage and the s_x[m]/s_w[n] (m=row,
  // n=col) indexing are identical to the prior flat layout.
  std::tuple<int, int, int> grid{N, M, 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};
  auto results = get_int8_rescale_kernel()(
      {acc, s_x, s_w, M_arr, N_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {bfloat16},
      grid,
      threadgroup,
      /* template_args */ {},
      /* init_value */ std::nullopt,  // every element written; no fill needed
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// sym8 DECODE CORE. int8 matvec: x_i8 [M,K] int8 @ w_i8 [K,N] int8 with the
// s_x[m]*s_w[n] rescale folded in -> bf16 [M,N]. One thread per output column n
// (for each token row m); each thread accumulates over K in int32 then narrows
// to bf16. x[m,:] is staged into threadgroup memory in chunks so it is read once
// per group; the [K,N] weight read is coalesced (consecutive threads = consecutive
// n = contiguous bytes). Output is LAZY (composes into the forward graph). See
// metal/na_int8_qmv.metal.inc for the kernel + dispatch contract.
mlx::core::array int8_qmv_core(const mlx::core::array& x_i8,
                               const mlx::core::array& w_kn,
                               const mlx::core::array& s_x,
                               const mlx::core::array& s_w) {
  using namespace mlx::core;
  const int M = x_i8.shape(0);
  const int K = x_i8.shape(1);
  const int N = w_kn.shape(1);  // w_kn is [K,N]
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);

  // 2D-blocked GEMV: threadgroup owns BN output columns x BK k-workers
  // (BN*BK threads). BK must be a power of two (tree reduce). Default 64x16
  // (1024 threads). Tunable via env for A/B (INT8_QMV_BN / INT8_QMV_BK /
  // INT8_QMV_STAGEX).
  int BN = 32;
  int BK = 16;
  if (const char* e = std::getenv("INT8_QMV_BN")) {
    int v = std::atoi(e);
    if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256) BN = v;
  }
  if (const char* e = std::getenv("INT8_QMV_BK")) {
    int v = std::atoi(e);
    if (v == 4 || v == 8 || v == 16 || v == 32) BK = v;
  }
  // STAGE_X: stage x[m,:] into threadgroup memory (off by default — the x read
  // is a broadcast that caches well; staging adds tg-memory pressure). MAXK is
  // the compile-time tg array bound; only used when STAGE_X != 0. Round K up to
  // a few buckets so the JIT instantiates a small fixed set (cache-friendly).
  int STAGE_X = 0;
  if (const char* e = std::getenv("INT8_QMV_STAGEX")) {
    STAGE_X = (std::atoi(e) != 0) ? 1 : 0;
  }
  int MAXK = 1;
  if (STAGE_X) {
    // Smallest power-of-two-ish bucket >= K (covers all real proj K up to 17408).
    int buckets[] = {2560, 5120, 9216, 17408, 18432, 34816};
    MAXK = 34816;
    for (int b : buckets) {
      if (K <= b) {
        MAXK = b;
        break;
      }
    }
  }

  std::vector<std::pair<std::string, fast::TemplateArg>> template_args = {
      {"BN", BN},
      {"BK", BK},
      {"STAGE_X", STAGE_X},
      {"MAXK", MAXK},
  };

  // dispatch_threads: total threads = grid. Each tg = BN*BK threads laid out
  // along grid.x; one tg per BN-column block; grid.y = M (one tg-row per token).
  const int tgN = (N + BN - 1) / BN;        // number of column blocks
  const int tgThreads = BN * BK;            // threads per threadgroup
  std::tuple<int, int, int> grid{tgThreads * tgN, M, 1};
  std::tuple<int, int, int> threadgroup{tgThreads, 1, 1};
  auto results = get_int8_qmv_kernel()(
      {x_i8, w_kn, s_x, s_w, M_arr, N_arr, K_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {bfloat16},
      grid,
      threadgroup,
      template_args,
      /* init_value */ std::nullopt,  // every in-bounds element written; no fill
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// FUSED sym8 DECODE matvec: bf16 x in, single kernel (act-quant folded in).
// x_bf16 [M,K] bf16, w_kn [K,N] int8, s_w [N] f32 -> bf16 [M,N]. Same 2D-block
// geometry as int8_qmv_core (env-tunable BN/BK); MAXK is the tg x-stage bound.
mlx::core::array int8_qmv_fused_core(const mlx::core::array& x_bf16,
                                     const mlx::core::array& w_kn,
                                     const mlx::core::array& s_w) {
  using namespace mlx::core;
  const int M = x_bf16.shape(0);
  const int K = x_bf16.shape(1);
  const int N = w_kn.shape(1);  // w_kn is [K,N]
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);

  int BN = 32;
  int BK = 16;
  if (const char* e = std::getenv("INT8_QMV_BN")) {
    int v = std::atoi(e);
    if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256) BN = v;
  }
  if (const char* e = std::getenv("INT8_QMV_BK")) {
    int v = std::atoi(e);
    if (v == 4 || v == 8 || v == 16 || v == 32) BK = v;
  }
  // VEC4: each thread owns 4 columns via char4 weight reads. Needs N%4==0 and
  // BN%4==0. MEASURED to HURT the large-K (down) shape — the smaller thread count
  // it implies starves the in-kernel quant's K-stride passes — so it is OFF by
  // default; INT8_QMV_VEC4=1 opts in for A/B.
  int VEC4 = 0;
  if (const char* e = std::getenv("INT8_QMV_VEC4")) {
    if (std::atoi(e) != 0) VEC4 = 1;
  }
  if (N % 4 != 0 || BN % 4 != 0) VEC4 = 0;  // hard safety gate

  // MAXK: smallest bucket >= K (fixed small set so the JIT instantiates few).
  int MAXK;
  {
    int buckets[] = {2560, 5120, 9216, 17408, 18432, 34816};
    MAXK = 34816;
    for (int b : buckets) {
      if (K <= b) {
        MAXK = b;
        break;
      }
    }
  }
  std::vector<std::pair<std::string, fast::TemplateArg>> template_args = {
      {"BN", BN},
      {"BK", BK},
      {"VEC4", VEC4},
      {"MAXK", MAXK},
  };

  const int CC = VEC4 ? 4 : 1;
  const int tgN = (N + BN - 1) / BN;
  const int tgThreads = (BN / CC) * BK;       // NT (matches kernel)
  std::tuple<int, int, int> grid{tgThreads * tgN, M, 1};
  std::tuple<int, int, int> threadgroup{tgThreads, 1, 1};
  auto results = get_int8_qmv_fused_kernel()(
      {x_bf16, w_kn, s_w, M_arr, N_arr, K_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {bfloat16},
      grid,
      threadgroup,
      template_args,
      /* init_value */ std::nullopt,
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// W8A16 sym8 DECODE matvec: bf16 x in, single pass, NO activation quant.
// x_bf16 [M,K] bf16, w_kn [K,N] int8, s_w [N] f32 -> bf16 [M,N]. Same 2D-block
// geometry as int8_qmv_fused_core but with SEPARATE env tunables
// (INT8_QMV16_BN / INT8_QMV16_BK / INT8_QMV16_VEC4 / INT8_QMV16_STAGEX) so the
// W8A16 sweep is independent of the W8A8 one.
mlx::core::array int8_qmv_w8a16_core(const mlx::core::array& x_bf16,
                                     const mlx::core::array& w_kn,
                                     const mlx::core::array& s_w) {
  using namespace mlx::core;
  const int M = x_bf16.shape(0);
  const int K = x_bf16.shape(1);
  const int N = w_kn.shape(1);  // w_kn is [K,N]
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);

  // SHAPE-AWARE default geometry (in-stream chained decode bench + e2e paired
  // A/B, M5 Max, 2026-06-10). The decode regime streams each weight from DRAM
  // exactly once (weights don't fit cache across the 150+ calls/token), so the
  // kernel is bound by DRAM transaction width:
  //   * N % 128 == 0 (every 4B decode shape except the N=64 GDN gate projs):
  //     BN=128/BK=32/VEC4=1 (NT=1024) — each simdgroup streams 128 CONTIGUOUS
  //     weight bytes per k step (char4 x 32 column-threads), the affine-qmv
  //     access pattern. In-stream 54.2us/call vs 67.3 for BN=32 scalar
  //     (affine qmv 31.6); e2e decode A/B vs affine-Q8 0.65 -> ~0.86.
  //   * otherwise: BN=32/BK=32/VEC4=0 (NT=1024) — scalar weight reads, safe
  //     for any N (compute + epilogue both bounds-checked).
  // Env INT8_QMV16_BN/BK/VEC4 override for A/B, subject to the hard gates
  // below.
  int BN = 32;
  int BK = 32;
  int VEC4 = 0;
  if (N % 128 == 0) {
    BN = 128;
    VEC4 = 1;
  }
  if (const char* e = std::getenv("INT8_QMV16_BN")) {
    int v = std::atoi(e);
    if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256) BN = v;
  }
  if (const char* e = std::getenv("INT8_QMV16_BK")) {
    int v = std::atoi(e);
    if (v == 4 || v == 8 || v == 16 || v == 32) BK = v;
  }
  if (const char* e = std::getenv("INT8_QMV16_VEC4")) {
    VEC4 = (std::atoi(e) != 0) ? 1 : 0;
  }
  // HARD GATES (apply to defaults AND env overrides):
  // 1. The VEC4 path reads char4 at columns [ncol, ncol+3] with NO per-read
  //    bounds check (only the epilogue WRITE is guarded), so the last column
  //    block must be full: N % BN == 0 (with BN % 4 == 0 this implies the
  //    4-alignment too). N=64 GDN gate projections fall back to scalar here.
  if (VEC4 && (BN % 4 != 0 || N % BN != 0)) VEC4 = 0;
  // 2. Metal caps a threadgroup at 1024 threads. Combos over the cap (e.g.
  //    BN=64/BK=32 scalar -> NT=2048) do NOT dispatch correctly and bench
  //    impossibly fast (~3.8 TB/s in the in-stream sweep = broken, not a
  //    win). Halve BK (stays power-of-two for the tree reduce) until legal.
  while ((BN / (VEC4 ? 4 : 1)) * BK > 1024 && BK > 4) {
    BK /= 2;
  }

  // STAGE_X: stage x[m,:] into tg memory ONCE as bf16 (off by default — the x
  // read is a simdgroup-uniform broadcast that caches well; staging adds tg
  // pressure). MAXK is the compile-time tg bound; bf16 staging needs 2*MAXK
  // bytes + the 4*BK*BN reduce tile, so K above the 32KB tg budget hard-gates
  // staging OFF.
  int STAGE_X = 0;
  if (const char* e = std::getenv("INT8_QMV16_STAGEX")) {
    STAGE_X = (std::atoi(e) != 0) ? 1 : 0;
  }
  int MAXK = 1;
  if (STAGE_X) {
    int buckets[] = {2560, 5120, 9216, 17408, 18432, 34816};
    MAXK = 34816;
    for (int b : buckets) {
      if (K <= b) {
        MAXK = b;
        break;
      }
    }
    // 2*MAXK (bf16 xs) + 4*BK*BN (f32 part tile) must fit the 32KB tg budget.
    if (2 * MAXK + 4 * BK * BN > 32768) {
      STAGE_X = 0;
      MAXK = 1;
    }
  }
  std::vector<std::pair<std::string, fast::TemplateArg>> template_args = {
      {"BN", BN},
      {"BK", BK},
      {"VEC4", VEC4},
      {"STAGE_X", STAGE_X},
      {"MAXK", MAXK},
  };

  const int CC = VEC4 ? 4 : 1;
  const int tgN = (N + BN - 1) / BN;
  const int tgThreads = (BN / CC) * BK;       // NT (matches kernel)
  std::tuple<int, int, int> grid{tgThreads * tgN, M, 1};
  std::tuple<int, int, int> threadgroup{tgThreads, 1, 1};
  auto results = get_int8_qmv_w8a16_kernel()(
      {x_bf16, w_kn, s_w, M_arr, N_arr, K_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {bfloat16},
      grid,
      threadgroup,
      template_args,
      /* init_value */ std::nullopt,
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// W8A16 sym8 DECODE matvec, simd_sum style — the PRODUCTION decode core.
// x_bf16 [M,K] bf16, w_nk [N,K] int8 (the CHECKPOINT orientation — NOT the
// [K,N] GEMM operand), s_w [N] f32 -> bf16 [M,N]. Geometry (defaults from the
// in-stream chained decode bench, M5 Max 2026-06-10; env INT8_QMV16_SG_LW /
// _RPS / _NSG override for A/B, read PER CALL):
//   LW  = 16 weight bytes per lane per row per iteration (a simdgroup streams
//         512 contiguous bytes per row — the affine-qmv transaction width),
//   RPS = 4 output rows per simdgroup (x block reused across rows),
//   NSG = 2 simdgroups per threadgroup (64 threads, 8 rows per tg).
// HARD GATES: LW in {4,8,16} (must divide 16 = the host K%16 contract, so a
// lane's guarded LW-read never crosses K); RPS clamped to N (the kernel's
// N-edge clamp needs N >= RPS); 32*NSG <= 1024.
mlx::core::array int8_qmv_w8a16_sg_core(const mlx::core::array& x_bf16,
                                        const mlx::core::array& w_nk,
                                        const mlx::core::array& s_w) {
  using namespace mlx::core;
  const int M = x_bf16.shape(0);
  const int K = x_bf16.shape(1);
  const int N = w_nk.shape(0);  // w_nk is [N,K]
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);

  int LW = 16;
  int RPS = 4;
  int NSG = 2;
  if (const char* e = std::getenv("INT8_QMV16_SG_LW")) {
    int v = std::atoi(e);
    if (v == 4 || v == 8 || v == 16) LW = v;
  }
  if (const char* e = std::getenv("INT8_QMV16_SG_RPS")) {
    int v = std::atoi(e);
    if (v >= 1 && v <= 16) RPS = v;
  }
  if (const char* e = std::getenv("INT8_QMV16_SG_NSG")) {
    int v = std::atoi(e);
    if (v >= 1 && v <= 32) NSG = v;
  }
  // The kernel's N-edge handling clamps a partial simdgroup DOWN to the last
  // full RPS-row window, which requires N >= RPS.
  while (RPS > N && RPS > 1) RPS /= 2;

  std::vector<std::pair<std::string, fast::TemplateArg>> template_args = {
      {"LW", LW},
      {"RPS", RPS},
      {"NSG", NSG},
  };

  const int rows_per_tg = NSG * RPS;
  const int tgN = (N + rows_per_tg - 1) / rows_per_tg;
  const int tgThreads = 32 * NSG;
  std::tuple<int, int, int> grid{tgThreads * tgN, M, 1};
  std::tuple<int, int, int> threadgroup{tgThreads, 1, 1};
  auto results = get_int8_qmv_w8a16_sg_kernel()(
      {x_bf16, w_nk, s_w, M_arr, N_arr, K_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {bfloat16},
      grid,
      threadgroup,
      template_args,
      /* init_value */ std::nullopt,
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// v1 PRODUCTION CORE. Same dispatch as int8_gemm_core but mode::multiply +
// init_value=nullopt → MLX skips the per-call full-output zero fill. For a fresh
// GEMM (no pre-existing C to accumulate into) overwrite == accumulate-into-zero,
// bit-exact (verified by s1 + the clean cross-check). Used by both the W8A8 hot
// path and mlx_matmul_int8, plus the measurement FFIs.
mlx::core::array int8_gemm_core_nofill(const mlx::core::array& x_i8,
                                       const mlx::core::array& b_kn) {
  using namespace mlx::core;
  const int M = x_i8.shape(0);
  const int K = x_i8.shape(1);
  const int N = b_kn.shape(1);
  array A = x_i8;
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);
  const int tgN = (N + 64 - 1) / 64;
  const int tgM = (M + 128 - 1) / 128;
  std::tuple<int, int, int> grid{256 * tgN, tgM, 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};
  auto results = get_int8_gemm_kernel_mul()(
      {A, b_kn, M_arr, N_arr, K_arr},
      {Shape{M, N}},
      {int32},
      grid,
      threadgroup,
      /* template_args */ {},
      /* init_value */ std::nullopt,  // multiply mode overwrites C; no fill
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// Core int8 GEMM. `x_i8` is [M,K] int8 (row-major); `b_kn` is the PRE-TRANSPOSED
// weight: a contiguous [K,N] int8 buffer with b_kn[k,n] = w[n,k] (output channels
// = columns). Returns int32 [M,N] = x @ w^T.
//
// STAGE 4b: the caller hands the [K,N] kernel operand directly. No transpose,
// no contiguous copy, no per-call astype here — the layout work is hoisted to
// load time (see mlx_quantize_weight_int8) so the per-forward path does ZERO
// weight reshaping. The kernel's column-major (N,K) tensor view reads
// b_kn[n + k*N] = w[n,k] (see layout contract above).
mlx::core::array int8_gemm_core(const mlx::core::array& x_i8,
                                const mlx::core::array& b_kn) {
  using namespace mlx::core;
  const int M = x_i8.shape(0);
  const int K = x_i8.shape(1);
  const int N = b_kn.shape(1);  // b_kn is [K,N]

  array A = x_i8;  // already [M,K] int8, row-contiguous
  array M_arr(M, int32);
  array N_arr(N, int32);
  array K_arr(K, int32);

  const int tgN = (N + 64 - 1) / 64;
  const int tgM = (M + 128 - 1) / 128;
  std::tuple<int, int, int> grid{256 * tgN, tgM, 1};
  std::tuple<int, int, int> threadgroup{256, 1, 1};

  // The kernel's ensure_row_contiguous=true will copy a non-contiguous A/B if
  // needed; b_kn is already row-contiguous [K,N] from load time, so no copy.
  auto results = get_int8_gemm_kernel()(
      {A, b_kn, M_arr, N_arr, K_arr},
      /* output_shapes */ {Shape{M, N}},
      /* output_dtypes */ {int32},
      grid,
      threadgroup,
      /* template_args */ {},
      /* init_value (zero C for multiply_accumulate) */ 0.0f,
      /* verbose */ false,
      default_stream(Device::gpu));
  return results[0];
}

// Build the [K,N] contiguous int8 kernel operand from a [N,K] int8 weight.
// Used at LOAD time (mlx_quantize_weight_int8) and by mlx_matmul_int8 (which
// keeps a [N,K] public contract). transpose alone is lazy/strided; contiguous
// forces the row-major [K,N] copy the column-major (N,K) view consumes.
mlx::core::array int8_weight_to_kn(const mlx::core::array& w_i8) {
  using namespace mlx::core;
  return astype(contiguous(transpose(w_i8, std::vector<int>{1, 0})), int8);
}

}  // namespace

// =============================================================================
// External-linkage lazy graph builders (declared in mlx_common.h).
//
// SINGLE SOURCE OF TRUTH for the sym8 W8A8 linear math: the extern "C" FFI
// wrappers below (`mlx_w8a8_linear`, `mlx_int8_qmv` — the eager Rust path) and
// the compiled C++ forward (`linear_proj` sym8 dispatch in
// mlx_qwen35_common.h) both call these, so the two paths emit byte-identical
// graphs for the same inputs. Contract checks THROW here; the FFI wrappers
// translate to cerr + false.
// =============================================================================
namespace na_int8 {

mlx::core::array w8a8_linear_lazy(const mlx::core::array& x,
                                  const mlx::core::array& w_kn,
                                  const mlx::core::array& s_w) {
  using namespace mlx::core;
  if (x.ndim() != 2 || w_kn.ndim() != 2) {
    throw std::runtime_error("[na_int8::w8a8_linear_lazy] expected 2D x,w");
  }
  const int K = x.shape(1);
  // w_kn is [K,N] (pre-transposed): its row count is K.
  if (w_kn.shape(0) != K) {
    throw std::runtime_error(
        "[na_int8::w8a8_linear_lazy] K mismatch: x.K=" + std::to_string(K) +
        " w.K=" + std::to_string(w_kn.shape(0)) +
        " (is the [K,N] kernel operand registered, not the [N,K] checkpoint "
        "tensor?)");
  }
  if (!na_int8_supported(K)) {
    throw std::runtime_error(
        "[na_int8::w8a8_linear_lazy] unsupported (gen=" +
        std::to_string(na_int8_gpu_gen()) + ", K=" + std::to_string(K) + ")");
  }

  // v1 (2): FUSED per-token int8 activation quant in ONE kernel (replaces the
  // ~5-7-op lazy absmax/round/clip/astype chain). x must be bf16 (the MLP
  // hands bf16 activations); cast defensively so a stray f32 input still
  // works.
  array x_bf16 = (x.dtype() == bfloat16) ? x : astype(x, bfloat16);
  auto [x_i8, s_x] = int8_act_quant(x_bf16);  // x_i8 int8 [M,K], s_x f32 [M,1]

  // v1 (1): nofill (mode::multiply overwrite) GEMM — bit-exact for a fresh
  // GEMM, skips the per-call full-output zero fill. w_kn already int8 [K,N].
  array acc = int8_gemm_core_nofill(x_i8, w_kn);  // int32 [M,N]

  // v1 (3): FUSED int32->bf16 rescale in ONE elementwise pass. Narrows to
  // bf16 inside the kernel.
  return int8_rescale(acc, s_x, s_w);  // bf16 [M,N]
}

mlx::core::array qmv_lazy(const mlx::core::array& x,
                          const mlx::core::array& w_kn,
                          const mlx::core::array& s_w) {
  using namespace mlx::core;
  if (x.ndim() != 2 || w_kn.ndim() != 2) {
    throw std::runtime_error("[na_int8::qmv_lazy] expected 2D x,w");
  }
  const int K = x.shape(1);
  // w_kn is [K,N] (pre-transposed): its row count is K.
  if (w_kn.shape(0) != K) {
    throw std::runtime_error(
        "[na_int8::qmv_lazy] K mismatch: x.K=" + std::to_string(K) +
        " w.K=" + std::to_string(w_kn.shape(0)) +
        " (is the [K,N] kernel operand registered, not the [N,K] checkpoint "
        "tensor?)");
  }
  if (!na_int8_supported(K)) {
    throw std::runtime_error(
        "[na_int8::qmv_lazy] unsupported (gen=" +
        std::to_string(na_int8_gpu_gen()) + ", K=" + std::to_string(K) + ")");
  }
  array x_bf16 = (x.dtype() == bfloat16) ? x : astype(x, bfloat16);
  // s_w must be f32 [N] for the kernel; cast defensively.
  array sw_f32 = (s_w.dtype() == float32) ? s_w : astype(s_w, float32);

  // FUSED single-kernel path (act-quant folded into the matvec) is the
  // default; INT8_QMV_FUSED=0 falls back to the two-kernel path (separate
  // na_int8_quant) for A/B. The fused path removes a kernel launch + the
  // [M,K] int8 intermediate.
  bool fused = true;
  if (const char* e = std::getenv("INT8_QMV_FUSED")) {
    fused = (std::atoi(e) != 0);
  }
  if (fused) {
    return int8_qmv_fused_core(x_bf16, w_kn, sw_f32);
  }
  auto [x_i8, s_x] = int8_act_quant(x_bf16);
  return int8_qmv_core(x_i8, w_kn, s_x, sw_f32);
}

mlx::core::array qmv_w8a16_lazy(const mlx::core::array& x,
                                const mlx::core::array& w_kn,
                                const mlx::core::array& w_nk,
                                const mlx::core::array& s_w) {
  using namespace mlx::core;
  if (x.ndim() != 2 || w_kn.ndim() != 2 || w_nk.ndim() != 2) {
    throw std::runtime_error("[na_int8::qmv_w8a16_lazy] expected 2D x,w");
  }
  const int K = x.shape(1);
  // w_kn is [K,N] (pre-transposed): its row count is K.
  if (w_kn.shape(0) != K) {
    throw std::runtime_error(
        "[na_int8::qmv_w8a16_lazy] K mismatch: x.K=" + std::to_string(K) +
        " w.K=" + std::to_string(w_kn.shape(0)) +
        " (is the [K,N] kernel operand registered, not the [N,K] checkpoint "
        "tensor?)");
  }
  // w_nk is the [N,K] CHECKPOINT orientation: same shape transposed.
  if (w_nk.shape(1) != K || w_nk.shape(0) != w_kn.shape(1) ||
      w_nk.dtype() != int8) {
    throw std::runtime_error(
        "[na_int8::qmv_w8a16_lazy] w_nk mismatch: expected int8 [N=" +
        std::to_string(w_kn.shape(1)) + ",K=" + std::to_string(K) +
        "], got [" + std::to_string(w_nk.shape(0)) + "," +
        std::to_string(w_nk.shape(1)) +
        "] (the [N,K] checkpoint tensor must be plumbed alongside the [K,N] "
        "kernel operand)");
  }
  if (!na_int8_supported(K)) {
    throw std::runtime_error(
        "[na_int8::qmv_w8a16_lazy] unsupported (gen=" +
        std::to_string(na_int8_gpu_gen()) + ", K=" + std::to_string(K) + ")");
  }
  array x_bf16 = (x.dtype() == bfloat16) ? x : astype(x, bfloat16);
  // s_w must be f32 [N] for the kernel; cast defensively.
  array sw_f32 = (s_w.dtype() == float32) ? s_w : astype(s_w, float32);

  // Same-binary A/B escape hatch: INT8_QMV_W8A16=0 reroutes sym8 DECODE back
  // to the W8A8 act-quant qmv (the old path). Lives HERE — inside the single
  // shared builder — so the eager Rust forward and the compiled C++ forward
  // stay byte-identical under EITHER env value. Default ON (W8A16).
  bool w8a16 = true;
  if (const char* e = std::getenv("INT8_QMV_W8A16")) {
    w8a16 = (std::atoi(e) != 0);
  }
  if (!w8a16) {
    return qmv_lazy(x_bf16, w_kn, sw_f32);
  }
  // Kernel-variant switch: the simd_sum-style kernel on the [N,K] checkpoint
  // orientation is the default; INT8_QMV16_SG=0 falls back to the 2D-block
  // kernel on the [K,N] operand for same-binary A/B. Read PER CALL (test
  // sweeps re-dispatch without rebuilding).
  bool sg = true;
  if (const char* e = std::getenv("INT8_QMV16_SG")) {
    sg = (std::atoi(e) != 0);
  }
  if (!sg) {
    return int8_qmv_w8a16_core(x_bf16, w_kn, sw_f32);
  }
  return int8_qmv_w8a16_sg_core(x_bf16, w_nk, sw_f32);
}

}  // namespace na_int8

extern "C" {

// int8 x @ w^T -> int32 [M,N].
//
// x,w are bf16/f32 arrays holding INTEGER values in [-127,127] (Rust has no
// Int8 DType). Cast to int8 here. w is stored [N,K] (rows = output channels).
// Returns false (so Rust can fall back) when unsupported or on error.
bool mlx_matmul_int8(mlx_array* x_handle, mlx_array* w_handle,
                     mlx_array** out_i32) {
  using namespace mlx::core;
  if (out_i32) *out_i32 = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& w = *reinterpret_cast<array*>(w_handle);
    if (x.ndim() != 2 || w.ndim() != 2) {
      std::cerr << "[mlx_matmul_int8] expected 2D x,w\n";
      return false;
    }
    const int K = x.shape(1);
    if (w.shape(1) != K) {
      std::cerr << "[mlx_matmul_int8] K mismatch: x.K=" << K
                << " w.K=" << w.shape(1) << "\n";
      return false;
    }
    if (!na_int8_supported(K)) {
      std::cerr << "[mlx_matmul_int8] unsupported (gen="
                << na_int8_gpu_gen() << ", K=" << K << ")\n";
      return false;
    }
    // Cast integer-valued float arrays to int8 (truncates toward zero; callers
    // pass exact integers so no rounding ambiguity). This op keeps the [N,K]
    // public contract (S1 bit-exact test), so it builds the [K,N] kernel operand
    // here; the W8A8 hot path instead stores it pre-transposed at load time.
    array x_i8 = astype(x, int8);
    array w_i8 = astype(w, int8);
    array b_kn = int8_weight_to_kn(w_i8);
    // v1: nofill (mode::multiply overwrite) core — bit-exact for a fresh GEMM,
    // skips the per-call full-output zero fill.
    array out = int8_gemm_core_nofill(x_i8, b_kn);
    *out_i32 = reinterpret_cast<mlx_array*>(new array(std::move(out)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_matmul_int8] EXCEPTION: " << e.what() << std::endl;
    if (out_i32) *out_i32 = nullptr;
    return false;
  }
}

// Per-output-channel symmetric int8 weight quantization (load-time, runs once).
//   w_bf16: [N,K]   ->   w_kn: opaque int8 [K,N] (MxArray Rust never introspects)
//                        s_w:  f32 [N]   (s_w[n] = max_k|w[n,k]| / 127)
//
// STAGE 4b: the int8 weight is returned ALREADY transposed+contiguous in the
// [K,N] kernel layout (b_kn[k,n] = w[n,k]). The per-forward GEMM then does ZERO
// weight reshaping. The scale s_w[N] still indexes the OUTPUT channel N (rows of
// the original [N,K] weight) regardless of the stored orientation, so it stays
// correct for the GEMM accumulator rescale acc[M,N] * s_x[M] * s_w[N].
//
// Both outputs are opaque MxArray handles. The int8 weight stays int8-typed
// inside MLX; Rust holds the handle but never reads its bytes.
bool mlx_quantize_weight_int8(mlx_array* w_handle, mlx_array** out_w_i8,
                              mlx_array** out_s_w) {
  using namespace mlx::core;
  if (out_w_i8) *out_w_i8 = nullptr;
  if (out_s_w) *out_s_w = nullptr;
  try {
    auto& w = *reinterpret_cast<array*>(w_handle);
    if (w.ndim() != 2) {
      std::cerr << "[mlx_quantize_weight_int8] expected 2D [N,K] weight\n";
      return false;
    }
    array w_f32 = astype(w, float32);
    // Per-row (per-output-channel) absmax over K. keepdims for broadcast.
    array absmax = max(abs(w_f32), /* axis */ 1, /* keepdims */ true);  // [N,1]
    // Avoid divide-by-zero on all-zero rows: scale = max(absmax,eps)/127.
    array eps = array(1e-12f, float32);
    array denom = maximum(absmax, eps);                                  // [N,1]
    array s_w = divide(denom, array(127.0f, float32));                   // [N,1]
    // Quantize: round(w / s_w) clamped to [-127,127], cast to int8 [N,K].
    array q = clip(round(divide(w_f32, s_w)),
                   std::optional<array>(array(-127.0f, float32)),
                   std::optional<array>(array(127.0f, float32)));
    array w_i8 = astype(q, int8);                  // [N,K]
    // Hoist the transpose+contiguous to LOAD time → [K,N] kernel operand.
    array w_kn = int8_weight_to_kn(w_i8);          // [K,N] contiguous int8
    array s_w_flat = reshape(s_w, {w.shape(0)});   // [N]
    eval(w_kn);
    eval(s_w_flat);
    *out_w_i8 = reinterpret_cast<mlx_array*>(new array(std::move(w_kn)));
    *out_s_w = reinterpret_cast<mlx_array*>(new array(std::move(s_w_flat)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_quantize_weight_int8] EXCEPTION: " << e.what()
              << std::endl;
    if (out_w_i8) *out_w_i8 = nullptr;
    if (out_s_w) *out_s_w = nullptr;
    return false;
  }
}

// CONVERT-time sym8 quantizer: per-output-channel symmetric int8, STORABLE
// checkpoint layout.
//   w: [N,K] float (bf16/f16/f32)  ->  out_q:      int8 [N,K]  (row-major, SAME
//                                                  orientation as the source)
//                                      out_scales: f32  [N]    (s[n] = max_k|w[n,k]| / 127)
//
// Identical math to mlx_quantize_weight_int8 above but WITHOUT the [K,N]
// transpose: that function emits the runtime KERNEL operand; this one emits the
// on-disk CHECKPOINT tensors ({prefix}.weight int8 [N,K] + {prefix}.scales f32
// [N], no .biases). The loader re-derives the [K,N] kernel operand at load
// time. Dequant contract: w[n,k] ≈ scales[n] * q[n,k].
bool mlx_sym8_quantize_store(mlx_array* w_handle, mlx_array** out_q,
                             mlx_array** out_scales) {
  using namespace mlx::core;
  if (out_q) *out_q = nullptr;
  if (out_scales) *out_scales = nullptr;
  try {
    auto& w = *reinterpret_cast<array*>(w_handle);
    if (w.ndim() != 2) {
      std::cerr << "[mlx_sym8_quantize_store] expected 2D [N,K] weight\n";
      return false;
    }
    array w_f32 = astype(w, float32);
    // Per-row (per-output-channel) absmax over K. keepdims for broadcast.
    array absmax = max(abs(w_f32), /* axis */ 1, /* keepdims */ true);  // [N,1]
    // Avoid divide-by-zero on all-zero rows: scale = max(absmax,eps)/127.
    array eps = array(1e-12f, float32);
    array denom = maximum(absmax, eps);                                  // [N,1]
    array s_w = divide(denom, array(127.0f, float32));                   // [N,1]
    // Quantize: round(w / s_w) clamped to [-127,127], cast to int8 [N,K].
    array q = clip(round(divide(w_f32, s_w)),
                   std::optional<array>(array(-127.0f, float32)),
                   std::optional<array>(array(127.0f, float32)));
    array q_i8 = astype(q, int8);                  // [N,K], storage orientation
    array s_w_flat = reshape(s_w, {w.shape(0)});   // [N]
    eval(q_i8);
    eval(s_w_flat);
    *out_q = reinterpret_cast<mlx_array*>(new array(std::move(q_i8)));
    *out_scales = reinterpret_cast<mlx_array*>(new array(std::move(s_w_flat)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_sym8_quantize_store] EXCEPTION: " << e.what()
              << std::endl;
    if (out_q) *out_q = nullptr;
    if (out_scales) *out_scales = nullptr;
    return false;
  }
}

// LOAD-time sym8 kernel-operand builder: the stored checkpoint int8 [N,K]
// weight (from mlx_sym8_quantize_store) -> the contiguous [K,N] int8 kernel
// operand consumed by mlx_w8a8_linear / mlx_int8_qmv. This is EXACTLY the
// transpose+contiguous tail of mlx_quantize_weight_int8 (int8_weight_to_kn),
// minus the quantization — the checkpoint already holds the quantized values,
// so the load is requant-free and bit-exact with the convert-time quantizer.
//
// Fail-loud: rejects non-2D and non-int8 inputs (a sym8 checkpoint that
// surfaces anything else here is corrupt). Evals before returning so the
// transpose copy is materialized ONCE at load, not per forward.
bool mlx_sym8_kernel_operand(mlx_array* w_handle, mlx_array** out_w_kn) {
  using namespace mlx::core;
  if (out_w_kn) *out_w_kn = nullptr;
  try {
    auto& w = *reinterpret_cast<array*>(w_handle);
    if (w.ndim() != 2) {
      std::cerr << "[mlx_sym8_kernel_operand] expected 2D [N,K] int8 weight, "
                << "got ndim=" << w.ndim() << "\n";
      return false;
    }
    if (w.dtype() != int8) {
      std::cerr << "[mlx_sym8_kernel_operand] expected int8 weight\n";
      return false;
    }
    array w_kn = int8_weight_to_kn(w);  // [K,N] contiguous int8
    eval(w_kn);
    *out_w_kn = reinterpret_cast<mlx_array*>(new array(std::move(w_kn)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_sym8_kernel_operand] EXCEPTION: " << e.what()
              << std::endl;
    if (out_w_kn) *out_w_kn = nullptr;
    return false;
  }
}

// W8A8 linear: per-token (per-row) dynamic int8 activation quant + int8 GEMM +
// rescale. Returns bf16 [M,N] = x @ w^T (lossy by int8 quant noise).
//
//   x_bf16: [M,K] bf16 activations
//   w_kn:   [K,N] int8 weight, PRE-TRANSPOSED at load (from mlx_quantize_weight_int8)
//   s_w:    [N]   f32 per-output-channel weight scale
//
//   s_x[m]   = absmax_k|x[m,k]| / 127           (per-token act scale)
//   x_i8[m,k]= round(x[m,k]/s_x[m]) clamp [-127,127]
//   acc[m,n] = (x_i8 @ w^T)[m,n]                (int32, exact)
//   y[m,n]   = acc * s_x[m] * s_w[n]            (f32), then -> bf16
//
// STAGE 4b: the int8 weight arrives in the [K,N] kernel layout (transpose hoisted
// to load), so the hot path does ZERO weight reshaping. The output is returned
// LAZY (no per-call eval) so it composes with the downstream swiglu + down-matmul
// and MLX keeps async pipelining/fusion across the 32 layers; the caller evals at
// end of forward (mirrors mlx_gated_delta and the fused bf16 MLP path). The quant
// ops above (absmax/round/clip/astype) and the f32 rescale also stay lazy so MLX
// fuses them into the graph.
//
// CRITICAL: astype(bf16) the rescaled result BEFORE returning so a later bf16
// residual add does not get promoted to f32 by an f32 scale.
bool mlx_w8a8_linear(mlx_array* x_handle, mlx_array* w_kn_handle,
                     mlx_array* s_w_handle, mlx_array** out_bf16) {
  using namespace mlx::core;
  if (out_bf16) *out_bf16 = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& w_kn = *reinterpret_cast<array*>(w_kn_handle);
    auto& s_w = *reinterpret_cast<array*>(s_w_handle);
    // Shared builder (na_int8::w8a8_linear_lazy) — the SAME graph the compiled
    // C++ forward emits for sym8 projections. Contract violations throw and
    // surface as cerr + false below.
    array y_bf16 = na_int8::w8a8_linear_lazy(x, w_kn, s_w);  // bf16 [M,N]
    *out_bf16 = reinterpret_cast<mlx_array*>(new array(std::move(y_bf16)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_w8a8_linear] EXCEPTION: " << e.what() << std::endl;
    if (out_bf16) *out_bf16 = nullptr;
    return false;
  }
}

// sym8 DECODE matvec (QMV): per-token int8 act quant + int8 MATVEC + rescale ->
// bf16 [M,N] = x @ w^T. The small-M (decode, M=1..~16) analogue of
// mlx_w8a8_linear: reuses the SAME activation int8 quant (int8_act_quant) and the
// SAME [K,N] pre-transposed weight + per-channel s_w from mlx_quantize_weight_int8,
// but runs a dedicated BW-bound matvec (one thread per output column) instead of
// the 128x64 prefill tile (which wastes 127/128 rows at M=1). Caller routes small
// M here, large M to mlx_w8a8_linear.
//
//   x_bf16: [M,K] bf16 activations
//   w_kn:   [K,N] int8 weight, PRE-TRANSPOSED at load (from mlx_quantize_weight_int8)
//   s_w:    [N]   f32 per-output-channel weight scale
//
// Output is LAZY (no per-call eval), narrowed to bf16 inside the kernel. Returns
// false (Rust falls back to bf16) when unsupported (gen<17 or K%16!=0) or on error.
bool mlx_int8_qmv(mlx_array* x_handle, mlx_array* w_kn_handle,
                  mlx_array* s_w_handle, mlx_array** out_bf16) {
  using namespace mlx::core;
  if (out_bf16) *out_bf16 = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& w_kn = *reinterpret_cast<array*>(w_kn_handle);
    auto& s_w = *reinterpret_cast<array*>(s_w_handle);
    // Shared builder (na_int8::qmv_lazy) — the SAME graph the compiled C++
    // forward emits for sym8 decode projections (incl. the INT8_QMV_FUSED
    // env dispatch). Contract violations throw -> cerr + false below.
    array y_bf16 = na_int8::qmv_lazy(x, w_kn, s_w);
    *out_bf16 = reinterpret_cast<mlx_array*>(new array(std::move(y_bf16)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_qmv] EXCEPTION: " << e.what() << std::endl;
    if (out_bf16) *out_bf16 = nullptr;
    return false;
  }
}

// W8A16 sym8 DECODE matvec (QMV): bf16 activations read DIRECTLY (NO act
// quant) against the SAME [K,N] pre-transposed int8 weight + per-channel s_w
// as mlx_int8_qmv, f32 accumulate, y = bf16(acc * s_w[n]). Single kernel pass —
// removes the absmax + int8-staging front-end that made the W8A8 qmv ~+70us/call
// over affine qmv at M=1 — and decode becomes activation-EXACT (only weight
// quant error remains).
//
//   x_bf16: [M,K] bf16 activations
//   w_kn:   [K,N] int8 weight, PRE-TRANSPOSED at load (GEMM operand — consumed
//           by the 2D-block fallback kernel and the W8A8 A/B reroute)
//   w_nk:   [N,K] int8 CHECKPOINT weight (source orientation — consumed by the
//           default simd_sum-style kernel; buffer-shared with the params-map /
//           registry tensor, NOT an extra copy)
//   s_w:    [N]   f32 per-output-channel weight scale
//
// Output is LAZY (no per-call eval), narrowed to bf16 inside the kernel.
// Returns false when unsupported (gen<17 or K%16!=0) or on error.
bool mlx_int8_qmv_w8a16(mlx_array* x_handle, mlx_array* w_kn_handle,
                        mlx_array* w_nk_handle, mlx_array* s_w_handle,
                        mlx_array** out_bf16) {
  using namespace mlx::core;
  if (out_bf16) *out_bf16 = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& w_kn = *reinterpret_cast<array*>(w_kn_handle);
    auto& w_nk = *reinterpret_cast<array*>(w_nk_handle);
    auto& s_w = *reinterpret_cast<array*>(s_w_handle);
    // Shared builder (na_int8::qmv_w8a16_lazy) — the SAME graph the compiled
    // C++ forward emits for sym8 decode projections (incl. the INT8_QMV_W8A16
    // env dispatch). Contract violations throw -> cerr + false below.
    array y_bf16 = na_int8::qmv_w8a16_lazy(x, w_kn, w_nk, s_w);
    *out_bf16 = reinterpret_cast<mlx_array*>(new array(std::move(y_bf16)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_qmv_w8a16] EXCEPTION: " << e.what() << std::endl;
    if (out_bf16) *out_bf16 = nullptr;
    return false;
  }
}

// ============================ MEASUREMENT ONLY ============================
// DIAGNOSTIC op (profiler/test scope only — NOT a production path). Pure int8
// GEMM with a PRE-TRANSPOSED [K,N] weight, isolating the kernel from the
// per-call `int8_weight_to_kn` transpose that `mlx_matmul_int8` pays every
// iteration. This is the apples-to-apples in-engine analogue of the standalone
// Metal harness: it calls `int8_gemm_core` directly with ZERO weight reshaping,
// NO activation quant, NO rescale.
//
//   x:    [M,K] bf16/f32 holding INTEGER values in [-127,127] (cast to int8)
//   w_kn: [K,N] int8 weight, PRE-TRANSPOSED at setup (from
//         mlx_quantize_weight_int8 — already int8-typed [K,N], used directly
//         with NO cast/transpose/contiguous, exactly as the W8A8 hot path holds
//         it). Returns int32 [M,N] = x @ w^T.
bool mlx_int8_gemm_pretransposed(mlx_array* x_handle, mlx_array* w_kn_handle,
                                 mlx_array** out_i32) {
  using namespace mlx::core;
  if (out_i32) *out_i32 = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& w_kn = *reinterpret_cast<array*>(w_kn_handle);
    if (x.ndim() != 2 || w_kn.ndim() != 2) {
      std::cerr << "[mlx_int8_gemm_pretransposed] expected 2D x,w_kn\n";
      return false;
    }
    const int K = x.shape(1);
    // w_kn is [K,N] (pre-transposed): its row count is K.
    if (w_kn.shape(0) != K) {
      std::cerr << "[mlx_int8_gemm_pretransposed] K mismatch: x.K=" << K
                << " w_kn.K=" << w_kn.shape(0) << "\n";
      return false;
    }
    if (!na_int8_supported(K)) {
      std::cerr << "[mlx_int8_gemm_pretransposed] unsupported (gen="
                << na_int8_gpu_gen() << ", K=" << K << ")\n";
      return false;
    }
    array x_i8 = astype(x, int8);
    // w_kn already int8 [K,N] from quantize_weight_int8 — if a bf16/f32 handle is
    // passed instead (integer-valued), cast it; an already-int8 input is a no-op
    // astype. NO transpose, NO contiguous copy: this is the whole point.
    array w_i8 = (w_kn.dtype() == int8) ? w_kn : astype(w_kn, int8);
    array out = int8_gemm_core(x_i8, w_i8);  // int32 [M,N]
    *out_i32 = reinterpret_cast<mlx_array*>(new array(std::move(out)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_gemm_pretransposed] EXCEPTION: " << e.what()
              << std::endl;
    if (out_i32) *out_i32 = nullptr;
    return false;
  }
}

// MEASUREMENT ONLY (diagnostic). Same as mlx_int8_gemm_pretransposed but uses
// the mode::multiply kernel with init_value=nullopt → skips MLX's per-call
// full-output zero fill. Used to isolate how much of the in-engine GEMM wall
// time is that fill_gpu pass (which the multiply_accumulate production kernel
// must pay, but the standalone harness pays only ONCE outside its timed loop).
bool mlx_int8_gemm_pretransposed_nofill(mlx_array* x_handle,
                                        mlx_array* w_kn_handle,
                                        mlx_array** out_i32) {
  using namespace mlx::core;
  if (out_i32) *out_i32 = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    auto& w_kn = *reinterpret_cast<array*>(w_kn_handle);
    if (x.ndim() != 2 || w_kn.ndim() != 2) {
      std::cerr << "[mlx_int8_gemm_pretransposed_nofill] expected 2D x,w_kn\n";
      return false;
    }
    const int K = x.shape(1);
    if (w_kn.shape(0) != K) {
      std::cerr << "[mlx_int8_gemm_pretransposed_nofill] K mismatch\n";
      return false;
    }
    if (!na_int8_supported(K)) {
      std::cerr << "[mlx_int8_gemm_pretransposed_nofill] unsupported (gen="
                << na_int8_gpu_gen() << ", K=" << K << ")\n";
      return false;
    }
    array x_i8 = astype(x, int8);
    array w_i8 = (w_kn.dtype() == int8) ? w_kn : astype(w_kn, int8);
    array out = int8_gemm_core_nofill(x_i8, w_i8);
    *out_i32 = reinterpret_cast<mlx_array*>(new array(std::move(out)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_gemm_pretransposed_nofill] EXCEPTION: " << e.what()
              << std::endl;
    if (out_i32) *out_i32 = nullptr;
    return false;
  }
}

// MEASUREMENT ONLY (parity test scope). Runs the FUSED activation-quant kernel
// (v1 kernel 2). x_bf16 [M,K] -> out_i8_as_i32 = int32([M,K] int8 quant) (so
// Rust, which has no Int8 dtype, can read the bytes), out_s_x = f32 [M,1].
bool mlx_int8_act_quant_fused(mlx_array* x_handle, mlx_array** out_i8_as_i32,
                              mlx_array** out_s_x) {
  using namespace mlx::core;
  if (out_i8_as_i32) *out_i8_as_i32 = nullptr;
  if (out_s_x) *out_s_x = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    if (x.ndim() != 2) {
      std::cerr << "[mlx_int8_act_quant_fused] expected 2D x\n";
      return false;
    }
    if (na_int8_gpu_gen() < 17) {
      std::cerr << "[mlx_int8_act_quant_fused] unsupported gen\n";
      return false;
    }
    array x_bf16 = (x.dtype() == bfloat16) ? x : astype(x, bfloat16);
    auto [x_i8, s_x] = int8_act_quant(x_bf16);
    array x_i32 = astype(x_i8, int32);  // widen for Rust readback
    eval(x_i32);
    eval(s_x);
    *out_i8_as_i32 = reinterpret_cast<mlx_array*>(new array(std::move(x_i32)));
    *out_s_x = reinterpret_cast<mlx_array*>(new array(std::move(s_x)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_act_quant_fused] EXCEPTION: " << e.what() << std::endl;
    if (out_i8_as_i32) *out_i8_as_i32 = nullptr;
    if (out_s_x) *out_s_x = nullptr;
    return false;
  }
}

// MEASUREMENT ONLY (parity test scope). The LAZY activation-quant chain (the
// reference kernel 2 must match bit-for-bit). Same outputs as the fused FFI.
bool mlx_int8_act_quant_lazy(mlx_array* x_handle, mlx_array** out_i8_as_i32,
                             mlx_array** out_s_x) {
  using namespace mlx::core;
  if (out_i8_as_i32) *out_i8_as_i32 = nullptr;
  if (out_s_x) *out_s_x = nullptr;
  try {
    auto& x = *reinterpret_cast<array*>(x_handle);
    if (x.ndim() != 2) {
      std::cerr << "[mlx_int8_act_quant_lazy] expected 2D x\n";
      return false;
    }
    array x_f32 = astype(x, float32);
    array a_absmax = max(abs(x_f32), /* axis */ 1, /* keepdims */ true);  // [M,1]
    array a_denom = maximum(a_absmax, array(1e-12f, float32));            // [M,1]
    array s_x = divide(a_denom, array(127.0f, float32));                 // [M,1]
    array x_q = clip(round(divide(x_f32, s_x)),
                     std::optional<array>(array(-127.0f, float32)),
                     std::optional<array>(array(127.0f, float32)));
    array x_i8 = astype(x_q, int8);
    array x_i32 = astype(x_i8, int32);
    eval(x_i32);
    eval(s_x);
    *out_i8_as_i32 = reinterpret_cast<mlx_array*>(new array(std::move(x_i32)));
    *out_s_x = reinterpret_cast<mlx_array*>(new array(std::move(s_x)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_act_quant_lazy] EXCEPTION: " << e.what() << std::endl;
    if (out_i8_as_i32) *out_i8_as_i32 = nullptr;
    if (out_s_x) *out_s_x = nullptr;
    return false;
  }
}

// MEASUREMENT ONLY (parity test scope). Runs the FUSED rescale kernel (v1
// kernel 3). acc [M,N] int32, s_x [M,1] f32, s_w [N] f32 -> y bf16 [M,N].
bool mlx_int8_rescale_fused(mlx_array* acc_handle, mlx_array* s_x_handle,
                            mlx_array* s_w_handle, mlx_array** out_bf16) {
  using namespace mlx::core;
  if (out_bf16) *out_bf16 = nullptr;
  try {
    auto& acc = *reinterpret_cast<array*>(acc_handle);
    auto& s_x = *reinterpret_cast<array*>(s_x_handle);
    auto& s_w = *reinterpret_cast<array*>(s_w_handle);
    if (acc.ndim() != 2) {
      std::cerr << "[mlx_int8_rescale_fused] expected 2D acc\n";
      return false;
    }
    if (na_int8_gpu_gen() < 17) {
      std::cerr << "[mlx_int8_rescale_fused] unsupported gen\n";
      return false;
    }
    array acc_i32 = (acc.dtype() == int32) ? acc : astype(acc, int32);
    array sx_f32 = (s_x.dtype() == float32) ? s_x : astype(s_x, float32);
    array sw_f32 = (s_w.dtype() == float32) ? s_w : astype(s_w, float32);
    array y = int8_rescale(acc_i32, sx_f32, sw_f32);
    eval(y);
    *out_bf16 = reinterpret_cast<mlx_array*>(new array(std::move(y)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_rescale_fused] EXCEPTION: " << e.what() << std::endl;
    if (out_bf16) *out_bf16 = nullptr;
    return false;
  }
}

// MEASUREMENT ONLY (parity test scope). The LAZY rescale (the reference kernel 3
// must match to bf16 eps). Same I/O as the fused FFI.
bool mlx_int8_rescale_lazy(mlx_array* acc_handle, mlx_array* s_x_handle,
                           mlx_array* s_w_handle, mlx_array** out_bf16) {
  using namespace mlx::core;
  if (out_bf16) *out_bf16 = nullptr;
  try {
    auto& acc = *reinterpret_cast<array*>(acc_handle);
    auto& s_x = *reinterpret_cast<array*>(s_x_handle);
    auto& s_w = *reinterpret_cast<array*>(s_w_handle);
    if (acc.ndim() != 2) {
      std::cerr << "[mlx_int8_rescale_lazy] expected 2D acc\n";
      return false;
    }
    array acc_f32 = astype(acc, float32);
    array sx_f32 = (s_x.dtype() == float32) ? s_x : astype(s_x, float32);
    array sw_f32 = (s_w.dtype() == float32) ? s_w : astype(s_w, float32);
    array s_w_row = reshape(sw_f32, {1, sw_f32.shape(0)});  // [1,N]
    array y_f32 = multiply(multiply(acc_f32, sx_f32), s_w_row);
    array y_bf16 = astype(y_f32, bfloat16);
    eval(y_bf16);
    *out_bf16 = reinterpret_cast<mlx_array*>(new array(std::move(y_bf16)));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[mlx_int8_rescale_lazy] EXCEPTION: " << e.what() << std::endl;
    if (out_bf16) *out_bf16 = nullptr;
    return false;
  }
}

}  // extern "C"
