#include "mlx_common.h"

#ifdef MLX_NODE_METAL_ENABLED
namespace mlx_node_internal {
mlx_array* dflash2_selector_edges_metal(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor);
}  // namespace mlx_node_internal
#elif defined(MLX_NODE_CUDA_ENABLED)
extern "C" mlx_array* mlx_dflash2_selector_edges_cuda(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor);
#endif

// Optional source-style selector edge arithmetic. All six arguments are
// explicit graph inputs, including the changing anchor. Row-contiguous copies
// are lazy and handled by CustomKernel, so sliced projected rows are accepted.
extern "C" mlx_array* mlx_dflash2_selector_edges(
    mlx_array* candidates, mlx_array* unary, mlx_array* hidden,
    mlx_array* predecessor, mlx_array* successor, mlx_array* anchor) {
#ifdef MLX_NODE_METAL_ENABLED
  return mlx_node_internal::dflash2_selector_edges_metal(
      candidates, unary, hidden, predecessor, successor, anchor);
#elif defined(MLX_NODE_CUDA_ENABLED)
  return mlx_dflash2_selector_edges_cuda(
      candidates, unary, hidden, predecessor, successor, anchor);
#else
  (void)candidates;
  (void)unary;
  (void)hidden;
  (void)predecessor;
  (void)successor;
  (void)anchor;
  return nullptr;
#endif
}
