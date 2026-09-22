#pragma once
#include "mlx/array.h"
#include <optional>
#include <string>

namespace mlx::core {
std::optional<array> portable_kquant_matmul(const array &x, const array &w,
                                            const array &scales,
                                            const std::optional<array> &biases,
                                            bool transpose, int group_size,
                                            int bits, const std::string &mode);
}
