#pragma once

#include "mlx/array.h"
#include <optional>

namespace mlx::core::fast {
// Uses ordinary SIMD-group matrix operations, available before M5/NAX.
bool portable_d256_sdpa_available();
std::optional<array> portable_d256_sdpa(
    const array& q, const array& k, const array& v, float scale);
} // namespace mlx::core::fast
