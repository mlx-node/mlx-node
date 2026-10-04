#pragma once

// paged_attn.metallib: the paged-attention kernels plus the prebuilt K-quant
// kernels. Metal-only (defined in mlx_paged_dispatch.cpp).

#include <filesystem>

#include "mlx/backend/metal/device.h"

namespace mlx::core::fast::paged {

// MLX_PAGED_ATTN_METALLIB, else next to the loaded binary (or its Resources/
// or parent dir). Throws, listing the tried paths, when none exists.
std::filesystem::path paged_attn_metallib_path();

// The library, loaded once per process and cached by MLX.
MTL::Library* get_paged_attn_library(mlx::core::metal::Device& device);

} // namespace mlx::core::fast::paged
