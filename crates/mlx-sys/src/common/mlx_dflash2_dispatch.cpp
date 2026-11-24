#include "mlx_common.h"

#ifdef MLX_NODE_METAL_ENABLED
namespace mlx_node_internal {
mlx_array* dflash2_selector_edges_metal(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor,
    float softcap);
}  // namespace mlx_node_internal
#elif defined(MLX_NODE_CUDA_ENABLED)
extern "C" mlx_array* mlx_dflash2_selector_edges_cuda(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor,
    float softcap);
#endif

// Optional source-style selector edge arithmetic. All six array arguments are
// explicit graph inputs, including the changing anchor; `softcap > 0` applies
// Splash's post-correction tanh softcap in FP32. Row-contiguous copies are lazy
// and handled by CustomKernel, so sliced projected rows are accepted.
extern "C" mlx_array* mlx_dflash2_selector_edges(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor,
    float softcap) {
#ifdef MLX_NODE_METAL_ENABLED
  return mlx_node_internal::dflash2_selector_edges_metal(
      candidates, unary, hidden, predecessor, successor, anchor, softcap);
#elif defined(MLX_NODE_CUDA_ENABLED)
  return mlx_dflash2_selector_edges_cuda(
      candidates, unary, hidden, predecessor, successor, anchor, softcap);
#else
  (void)candidates;
  (void)unary;
  (void)hidden;
  (void)predecessor;
  (void)successor;
  (void)anchor;
  (void)softcap;
  return nullptr;
#endif
}
