#pragma once

// BF16 x times a transposed affine projection whose scales/biases are F32,
// with a BF16 result. Bit-identical to MLX's promoted path (cast x to F32,
// F32 quantized_matmul, cast the result to BF16) at every shape; for 2..8 rows
// where that path would run F32 qmv_wide, one kernel replaces its three
// dispatches.

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <optional>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"

#include <string>
#include <vector>
#endif

namespace mlx::core::affine_mixed {

// Nullopt unless x is BF16 [..., K], w is uint32 [N, K * bits / 32] and
// scales/biases are F32 [N, K / group_size]. Metal only, no gradients.
std::optional<array> quantized_matmul(const array &x, const array &w,
                                      const array &scales, const array &biases,
                                      int group_size, int bits,
                                      StreamOrDevice s = {});

#ifdef MLX_NODE_METAL_ENABLED
// Every qmv_wide tile width (vectors per threadgroup) eval_gpu can request,
// and the kernel name for one.
std::vector<int> qmv_wide_widths();
std::string qmv_wide_kernel_name(int vecs_per_tg);

// The prebuilt qmv_wide pipeline for `vecs_per_tg` from paged_attn.metallib;
// throws when it is missing.
MTL::ComputePipelineState *qmv_wide_kernel(metal::Device &d, int vecs_per_tg);
#endif

class AffineMixedQmm : public UnaryPrimitive {
public:
  AffineMixedQmm(Stream stream, int group_size, int bits)
      : UnaryPrimitive(stream), group_size_(group_size), bits_(bits) {}

  void eval_cpu(const std::vector<array> &inputs, array &out) override;
  void eval_gpu(const std::vector<array> &inputs, array &out) override;

  DEFINE_NAME(AffineMixedQmm)
  bool is_equivalent(const Primitive &other) const override;
  std::vector<Shape> output_shapes(const std::vector<array> &inputs) override;

private:
  int group_size_;
  int bits_;
};

} // namespace mlx::core::affine_mixed
