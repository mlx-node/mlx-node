// Which kernel MLX's affine `quantized_matmul` (transposed weight) takes for
// a given row count, so callers that row-merge projections can prove the
// merge is bit-exact: below the vector limit the qmv / qmv_fast / qmv_wide
// kernels compute every output element from its own row and input vector,
// independent of the output width N; at or above it `qmm_splitk` picks a
// split-K from N, which changes the summation order.

#include "mlx_common.h"

#include <cstdlib>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/common/quantized.h"
#include "mlx/backend/metal/device.h"
#endif

namespace {

#ifdef MLX_NODE_METAL_ENABLED
using namespace mlx::core;

// Mirrors `get_qmv_batch_limit` in mlx/backend/metal/quantized.cpp (an
// anonymous-namespace helper) on the same device properties.
int qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen >= 17 && arch_size != 'd') {
    if (D <= 2048 && O <= 2048) return 33;
    if (D <= 4096 && O <= 4096) return 25;
    return 13;
  }
  if (arch_gen >= 15 && arch_size != 'd') {
    if (D <= 2048 && O <= 2048) return 13;
    if (D <= 4096 && O <= 4096) return 15;
    return 13;
  }
  if (arch_size == 'd') {
    if (D <= 2048 && O <= 2048) return 32;
    if (D <= 4096 && O <= 4096) return 18;
    return 12;
  }
  if (arch_gen >= 13) {
    if (D <= 2048 && O <= 2048) return 14;
    if (D <= 4096 && O <= 4096) return 10;
    return 6;
  }
  if (D <= 2048 && O <= 2048) return 18;
  if (D <= 4096 && O <= 4096) return 12;
  return 10;
}

// Mirrors `qmv_fast_k_alignment` in quantized.cpp.
int qmv_fast_k_alignment(int bits) {
  return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;
}
#endif

}  // namespace

// Row counts `M < limit` of an affine `x @ W^T` with `W` `[N, K]` at `bits`
// take the per-row (`qmv_fast`) route on this device, honouring
// `MLX_QMM_SPLITK_MIN_M` as MLX does. Returns 0 when the shape would not
// take `qmv_fast` (N not a multiple of 8, K not aligned to the kernel's step,
// K of 64/128 which routes to qmv_quad) or without the Metal device.
extern "C" int32_t mlx_affine_qmv_fast_limit(int32_t k, int32_t n, int32_t bits) {
#ifdef MLX_NODE_METAL_ENABLED
  if (!metal::is_available() || default_device().type != Device::gpu || k <= 0 ||
      n <= 0 || bits <= 0 || k == 64 || k == 128 || n % 8 != 0 ||
      k % qmv_fast_k_alignment(bits) != 0) {
    return 0;
  }
  int limit = qmv_batch_limit(k, n, metal::device(Device::gpu));
  if (const char* e = std::getenv("MLX_QMM_SPLITK_MIN_M")) {
    int v = std::atoi(e);
    if (v > 0) {
      limit = v;
    }
  }
  return limit;
#else
  (void)k; (void)n; (void)bits;
  return 0;
#endif
}
