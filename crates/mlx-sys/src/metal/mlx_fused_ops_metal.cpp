#include "mlx_common.h"

namespace mlx_node_internal {

#ifdef MLX_NODE_METAL_ENABLED
std::optional<std::pair<mlx::core::array, mlx::core::array>>
fused_add_rmsnorm_metal(const mlx::core::array& x,
                        const mlx::core::array& res,
                        const mlx::core::array& w,
                        const mlx::core::array& eps,
                        int rows,
                        int axis) {
  const auto dt = x.dtype();
  static auto kernel = mlx::core::fast::metal_kernel(
      "add_rmsnorm", {"x", "res", "w", "eps"}, {"h_out", "normed"},
#include "metal/common/add_rmsnorm.metal.inc"
  );
  // Mirror normalization.cpp's threadgroup sizing so the element mapping
  // matches rms_single_row (axis <= 4096) or rms_looped (above) exactly.
  constexpr int n_reads = 4;
  const int lsize = axis <= 4096
      ? ((axis + n_reads - 1) / n_reads + 31) / 32 * 32
      : 1024;
  auto outs = kernel({x, res, w, eps}, {x.shape(), x.shape()}, {dt, dt},
                     {lsize, rows, 1}, {lsize, 1, 1},
                     {{"T", dt}, {"AXIS", axis}}, std::nullopt, false,
                     mlx::core::Device::gpu);
  return std::make_pair(std::move(outs[0]), std::move(outs[1]));
}
#else
std::optional<std::pair<mlx::core::array, mlx::core::array>>
fused_add_rmsnorm_metal(const mlx::core::array&,
                        const mlx::core::array&,
                        const mlx::core::array&,
                        const mlx::core::array&,
                        int,
                        int) {
  return std::nullopt;
}
#endif

}  // namespace mlx_node_internal
