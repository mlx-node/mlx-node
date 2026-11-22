// TEST-ONLY hook for the mixed BF16/F32 affine op. No production caller.

#include "mlx_affine_mixed_qmm.h"
#include "mlx_common.h"

using mlx::core::array;

namespace {
const array &ref(mlx_array *a) { return *reinterpret_cast<array *>(a); }

mlx_array *wrap(array a) {
  return reinterpret_cast<mlx_array *>(new array(std::move(a)));
}
} // namespace

// Shapeless-compiles the op on `trace_x`, evaluates it there, then replays the
// compiled function on `x`. Writes the replay and the eager result for x.
extern "C" bool mlx_test_affine_mixed_shapeless_replay(
    mlx_array *trace_x, mlx_array *x, mlx_array *w, mlx_array *scales,
    mlx_array *biases, int group_size, int bits, mlx_array **out_replay,
    mlx_array **out_eager) {
  try {
    auto fn = [=](const std::vector<array> &in) {
      auto out = mlx::core::affine_mixed::quantized_matmul(
          in[0], in[1], in[2], in[3], group_size, bits, mlx::core::Device::gpu);
      if (!out)
        throw std::invalid_argument("operands rejected");
      return std::vector<array>{*out};
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
    std::cerr << "mlx_test_affine_mixed_shapeless_replay: " << e.what()
              << std::endl;
    return false;
  }
}
