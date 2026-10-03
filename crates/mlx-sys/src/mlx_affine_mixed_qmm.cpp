#include "mlx_affine_mixed_qmm.h"

#include "mlx/backend/metal/metal.h"

#include <stdexcept>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx_test_counters.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#endif

namespace mlx::core::affine_mixed {

std::optional<array> quantized_matmul(const array &x, const array &w,
                                      const array &scales, const array &biases,
                                      int group_size, int bits,
                                      StreamOrDevice s) {
  auto stream = to_stream(s);
  if (stream.device.type != Device::gpu || !metal::is_available()) {
    return std::nullopt;
  }
  if (group_size <= 0 || bits <= 0 || 32 % bits != 0 || x.ndim() < 1 ||
      x.dtype() != bfloat16 || w.ndim() != 2 || w.dtype() != uint32 ||
      scales.dtype() != float32 || biases.dtype() != float32 ||
      scales.shape() != biases.shape() || scales.ndim() != 2) {
    return std::nullopt;
  }
  int k = x.shape(-1);
  int n = w.shape(0);
  if (k <= 0 || k % group_size != 0 || w.shape(1) * (32 / bits) != k ||
      scales.shape(0) != n || scales.shape(1) != k / group_size) {
    return std::nullopt;
  }
  auto shape = x.shape();
  shape.back() = n;
  return array(std::move(shape), bfloat16,
               std::make_shared<AffineMixedQmm>(stream, group_size, bits),
               {x, w, scales, biases});
}

bool AffineMixedQmm::is_equivalent(const Primitive &other) const {
  const auto &o = static_cast<const AffineMixedQmm &>(other);
  return group_size_ == o.group_size_ && bits_ == o.bits_;
}

std::vector<Shape>
AffineMixedQmm::output_shapes(const std::vector<array> &inputs) {
  auto out_shape = inputs[0].shape();
  out_shape.back() = inputs[1].shape(0);
  return {std::move(out_shape)};
}

void AffineMixedQmm::eval_cpu(const std::vector<array> &, array &) {
  throw std::runtime_error("[AffineMixedQmm] Metal only.");
}

#ifdef MLX_NODE_METAL_ENABLED

namespace {

const char *kQmvWideMixed =
#include "metal/common/affine_qmv_wide_mixed.metal.inc"
    ;

// At most 8 rows. On gen >= 15 MLX's F32 path runs qmv_wide for 2..8 rows
// on every chip (its smallest qmv batch limit there is 10, fork and upstream)
// unless MLX_QMM_SPLITK_MIN_M lowers that limit.
bool rows_take_qmv_wide(int M) {
  int limit = 9;
  if (const char *e = std::getenv("MLX_QMM_SPLITK_MIN_M")) {
    int v = std::atoi(e);
    if (v > 0) {
      limit = std::min(limit, v);
    }
  }
  return M >= 2 && M < limit;
}

void qmv_wide(const array &x, const array &w, const array &scales,
              const array &biases, array &out, int M, int N, int K,
              metal::Device &d, const Stream &s) {
  out.set_data(allocator::malloc(out.nbytes()));
  // MLX's affine qmv_wide tiling: fewest tiles, then the smallest tile that
  // fills them, up to 8 vectors only when N alone saturates the GPU.
  const int tile_cap = N >= 2048 ? 8 : 5;
  int n_tiles = (M + tile_cap - 1) / tile_cap;
  int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
  constexpr int k_lanes = 8;
  constexpr int num_simdgroups = 2;
  int rows_per_tg = (32 / k_lanes) * num_simdgroups;

  std::string kname =
      "mlx_node_affine_qmv_wide_mixed_q8g32_nv" + std::to_string(vecs_per_tg);
  bridge_testing::record("affine_mixed_qmv_wide");
  if (bridge_testing::counting) {
    bridge_testing::record("affine_mixed_qmv_wide_nv" +
                           std::to_string(vecs_per_tg));
  }
  auto *lib = d.get_library(kname, [&] {
    std::string fn =
        "affine_qmv_wide_mixed_q8g32<" + std::to_string(vecs_per_tg) + ">";
    return std::string(kQmvWideMixed) + "\ntemplate [[host_name(\"" + kname +
           "\")]] [[kernel]] decltype(" + fn + ") " + fn + ";\n";
  });
  auto *kernel = d.get_kernel(kname, lib);

  auto &enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(w, 0);
  enc.set_input_array(scales, 1);
  enc.set_input_array(biases, 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(K, 5);
  enc.set_bytes(N, 6);
  enc.set_bytes(M, 7);
  enc.dispatch_threadgroups(MTL::Size((M + vecs_per_tg - 1) / vecs_per_tg,
                                      (N + rows_per_tg - 1) / rows_per_tg, 1),
                            MTL::Size(32, num_simdgroups, 1));
}

} // namespace

// Mirrors the promoted graph: qmv_wide for 2..8 rows where MLX would run the
// F32 qmv_wide for these operands; every other shape runs that graph itself
// (cast x, MLX's F32 QuantizedMatmul, cast the result).
void AffineMixedQmm::eval_gpu(const std::vector<array> &inputs, array &out) {
  auto &s = stream();
  auto &d = metal::device(s.device);
  const array &x = inputs[0];
  const array &w = inputs[1];
  const array &scales = inputs[2];
  const array &biases = inputs[3];
  int K = x.shape(-1);
  int N = out.shape(-1);
  int M = x.size() / K;
  bool wide = group_size_ == 32 && bits_ == 8 && x.flags().row_contiguous &&
              w.flags().row_contiguous && scales.flags().row_contiguous &&
              biases.flags().row_contiguous && rows_take_qmv_wide(M) &&
              K != 64 && K != 128 && d.get_architecture_gen() >= 15;
  if (wide) {
    qmv_wide(x, w, scales, biases, out, M, N, K, d, s);
    return;
  }

  bridge_testing::record("affine_mixed_promoted");
  array x_promoted(x.shape(), float32, nullptr, {});
  copy_gpu(x, x_promoted,
           x.flags().contiguous ? CopyType::Vector : CopyType::General, s);
  array out_promoted(out.shape(), float32, nullptr, {});
  QuantizedMatmul(s, group_size_, bits_, QuantizationMode::Affine, true)
      .eval_gpu({x_promoted, w, scales, biases}, out_promoted);
  copy_gpu(out_promoted, out, CopyType::Vector, s);
  auto &enc = metal::get_command_encoder(s);
  enc.add_temporary(x_promoted);
  enc.add_temporary(out_promoted);
}

#else

void AffineMixedQmm::eval_gpu(const std::vector<array> &, array &) {
  throw std::runtime_error("[AffineMixedQmm] Metal only.");
}

#endif

} // namespace mlx::core::affine_mixed
