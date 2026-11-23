#pragma once

// =============================================================================
// Shared compiled SwiGLU activation.
//
// Used by the fused MLP and standalone activation in mlx_fused_ops.cpp.
// All functions are inline to avoid ODR violations across translation units.
// =============================================================================

#include "mlx_common.h"

#include <vector>

namespace qwen35_common {

// SwiGLU: sigmoid(gate) * gate * up — compiled for kernel fusion
inline std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
  const auto& gate = inputs[0];
  const auto& up = inputs[1];
  return {sigmoid(gate) * gate * up};
}

inline auto& compiled_swiglu() {
  static auto fn = mlx::core::compile(swiglu_impl, /*shapeless=*/true);
  return fn;
}

inline array swiglu(const array& gate, const array& up) {
  return compiled_swiglu()({gate, up})[0];
}

}  // namespace qwen35_common
