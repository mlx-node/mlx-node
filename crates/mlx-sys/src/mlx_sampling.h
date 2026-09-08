#pragma once

#include "mlx_common.h"
#include <limits>

// Uniform draws include zero. A finite Gumbel value there prevents a valid
// token from tying masked (-inf) logits. All nonzero u32-derived uniforms keep
// their original value, and callers still consume exactly one MLX random key.
inline mlx::core::array mlx_categorical_with_uniforms(
    const mlx::core::array& logits, const mlx::core::array& uniforms, int axis) {
  using namespace mlx::core;
  auto u = maximum(uniforms, array(std::numeric_limits<float>::min()));
  auto gumbel = negative(log(negative(log(u))));
  return argmax(add(gumbel, logits), axis, false);
}

inline mlx::core::array mlx_categorical(const mlx::core::array& logits, int axis) {
  const int normalized_axis = axis < 0 ? axis + logits.ndim() : axis;
  if (normalized_axis < 0 || normalized_axis >= logits.ndim()) {
    throw std::invalid_argument("[categorical] Invalid axis for logits");
  }
  return mlx_categorical_with_uniforms(
      logits, mlx::core::random::uniform(logits.shape(), mlx::core::float32), normalized_axis);
}
