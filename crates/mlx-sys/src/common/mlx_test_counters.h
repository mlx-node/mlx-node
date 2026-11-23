#pragma once

// Test-only kernel-family counters for the bridge Metal primitives, per
// calling thread (Metal primitives are encoded on the thread that calls eval).
// Off unless a test enables them.

#include <string_view>

namespace mlx::core::bridge_testing {
inline thread_local bool counting = false;
// Test-only: keep the sorted K-quant MoE expert matmul on the simdgroup
// `gather_qmm_rhs` fallback instead of the tensor-op route (route parity).
inline thread_local bool force_gather_rhs_fallback = false;
void record_family(std::string_view family);
inline void record(std::string_view family) {
  if (counting) {
    record_family(family);
  }
}
} // namespace mlx::core::bridge_testing
